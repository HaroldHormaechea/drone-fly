"""``RaceEnv`` — the Gymnasium start→gate→finish racing environment (AC1, AC11, AC12).

Wires the sim-agnostic adapter (:mod:`drone_fly.adapter`), the pure geometry phase machine
(:mod:`drone_fly.env.geometry`), and the pure reward (:mod:`drone_fly.env.reward`) into a
single ``gymnasium.Env`` the connectome PPO policy trains against.

Observation / action contract (locked to UC-01/UC-02)
-----------------------------------------------------
* **Observation** — ``Box`` of shape ``(OBS_DIM,)`` == ``(12,)``: relative next-waypoint
  pose ``(3)`` + attitude ``(3)`` + linear velocity ``(3)`` + angular velocity ``(3)``.
  This matches the fixed 12-d contract the connectome actor consumes.
* **Action** — ``Box(low=[0,-1,-1,-1], high=[1,1,1,1])`` == the canonical CTBR channels
  ``(THROTTLE, ROLL, PITCH, YAW)``. PPO's Gaussian head is unbounded; the adapter clips.

Termination (AC1)
-----------------
An episode ``terminated`` on a valid course completion (gate passed **and** finish
crossed) or a floor/ceiling collision; it is ``truncated`` on timeout (``max_steps``).

Determinism (AC12)
------------------
With :class:`~drone_fly.adapter.simple.SimpleDroneAdapter` the dynamics are fixed (AC11)
and noise-free, so a fixed ``seed`` yields a bit-identical episode for a given policy.
"""

from __future__ import annotations

import dataclasses
import logging

import gymnasium as gym
import numpy as np

from drone_fly.adapter import make_adapter, pybullet_available
from drone_fly.controller.encoding import ACTION_DIM, OBS_DIM
from drone_fly.env import worker_output
from drone_fly.env.config import DynamicsParams, EnvConfig
from drone_fly.env.docking import evaluate_dock, pad_under
from drone_fly.env.geometry import advance, current_target
from drone_fly.env.obstacles import (
    OBSTACLE_FEATURES_PER,
    obstacle_vision_features,
    segment_contact,
)
from drone_fly.env.randomization import sample_course, sample_dynamics
from drone_fly.env.reward import compute_reward

logger = logging.getLogger(__name__)


class RaceEnv(gym.Env):
    """Single-drone start→gate→finish racing environment.

    Parameters
    ----------
    config:
        Course / reward / episode configuration. Defaults to :class:`EnvConfig`.
    adapter:
        Backend selector passed to :func:`drone_fly.adapter.make_adapter`
        (``"auto"`` | ``"simple"`` | ``"pybullet"``).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, *, adapter: str = "auto") -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self._adapter_choice = adapter
        course = self.config.course

        # Battery drain + thrust-impact (UC-17): a single ``battery.enabled`` flag gates BOTH the
        # adapter-side physics and the env-side width-1 battery observation block (they are
        # physically coupled). Off by default → no battery param forwarded (byte-identical to
        # UC-16) and the observation width is unchanged.
        self._battery_enabled = bool(self.config.battery.enabled)
        # Integrity damage + control-authority (UC-19): a single ``damage.enabled`` flag gates BOTH
        # the adapter-side authority degradation and the env-side width-1 damage observation block
        # (physically coupled, exactly like battery). Off by default → no damage param forwarded
        # (byte-identical to UC-18) and the observation width is unchanged.
        self._damage_enabled = bool(self.config.damage.enabled)

        self.adapter = make_adapter(
            adapter,
            course.start,
            floor_z=course.floor_z,
            ceiling_z=course.ceiling_z,
            dt=self.config.episode.dt,
            battery=self.config.battery if self._battery_enabled else None,
            damage=self.config.damage if self._damage_enabled else None,
        )
        self.backend = self.adapter.backend

        # Obstacle-vision block (UC-15): schema-agnostic — the env only knows whether to append
        # the nearest-k egocentric obstacle encoding and how wide it is. Off by default → the
        # observation stays the locked 12-d contract (byte-identical to UC-01..14).
        ov = self.config.obstacle_vision
        self._obstacle_vision_enabled = bool(ov.enabled)
        self._obstacle_vision_k = int(ov.k)
        obs_dim = OBS_DIM
        if self._obstacle_vision_enabled:
            obs_dim += OBSTACLE_FEATURES_PER * self._obstacle_vision_k
        # UC-17: the width-1 battery block is appended STRICTLY AFTER the obstacle-vision block, so
        # the dim order is (vision, proprioception, obstacle_vision, battery) — matching the
        # ``battery_hunger_v3`` schema's block order.
        if self._battery_enabled:
            obs_dim += 1
        # UC-19: the width-1 damage block is appended STRICTLY AFTER the battery block, so the dim
        # order is (vision, proprioception, obstacle_vision, battery, damage) — matching the
        # ``damage_proprioception_v4`` schema's block order.
        if self._damage_enabled:
            obs_dim += 1

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

        # Indexed N-gate walk state (UC-09): how many gates have been passed (0..N) and
        # whether the finish has been crossed. Replaces the old string phase machine.
        self._gates_passed = 0
        self._done = False
        # The *active* course for the current episode (UC-08). Initialised to the fixed
        # config course; overwritten per-episode by reset() when course randomization is on.
        # Everything downstream (_observation / _dist_to_target / step) reads self._course,
        # NOT self.config.course, so a randomized course actually drives the geometry.
        self._course = self.config.course
        self._prev_pos = course.start.copy()
        self._step_count = 0
        # Effective per-episode step budget (UC-09): base max_steps + steps_per_gate*(N-1).
        # Recomputed each reset() from the active (possibly randomized) course. N=1 => 400.
        self._max_steps = self.config.episode.max_steps
        # Edge-trigger state for the obstacle penalty (UC-15 AC2/AC9): whether the drone was in
        # contact with a pillar on the *previous* step, so a penalty fires only when contact
        # begins. Reset to False every episode.
        self._prev_contact = False
        # Authoritative docked state (UC-16): the single source of truth for ``info["docked"]``.
        # LEVEL/every-step (recomputed fresh each step from the stateless dock predicate), NOT
        # edge-triggered — it stays True across every dwell step and clears the step takeoff lifts
        # the drone off the floor. No vestigial copy anywhere else in the env.
        self._docked = False

        # Grounded / no-progress early-termination (UC-25). Cache the thresholds so ``step()``
        # never re-reads the config each frame. The rule only affects the termination decision on
        # genuinely grounded/stuck episodes (see :class:`EarlyTerminationConfig`); obs/RNG are
        # untouched, so normal/crashing episodes stay byte-identical.
        et = self.config.early_termination
        self._et_enabled = bool(et.enabled)
        self._et_floor_epsilon = float(et.floor_epsilon)
        self._et_stuck_window = int(et.stuck_window)  # no-progress detector window
        self._et_grounded_window = int(et.grounded_window)  # grounded window (UC-36; shorter)
        self._et_progress_epsilon = float(et.progress_epsilon)
        self._et_rest_speed_epsilon = float(et.rest_speed_epsilon)
        # Per-episode counters / bookkeeping for the two detectors (initialised properly in
        # reset(); declared here so they exist before the first step even if reset() is skipped).
        self._grounded_counter = 0
        self._stuck_counter = 0
        self._best_dist = float("inf")
        self._prev_battery = 1.0
        self._prev_integrity = 1.0
        # UC-37: per-episode takeoff latch. Armed once the drone first leaves the floor band; gates
        # the grounded detector (which must NOT fire before takeoff, AC3) and, for an airborne-start
        # episode, is True at step 0 so behaviour stays byte-identical to UC-25/36 (AC4). Set
        # properly from the spawn state in reset(); declared here for the reset()-skipped path.
        self._took_off = False

    @property
    def obs_width(self) -> int:
        """Width of the emitted observation vector (UC-15).

        ``OBS_DIM`` (12) normally; ``OBS_DIM + OBSTACLE_FEATURES_PER * k`` when the obstacle-vision
        block is enabled. The training coordinator asserts this equals the obs schema's
        ``total_width`` (fail-loud env↔schema width coupling).
        """
        return int(self.observation_space.shape[0])

    @property
    def active_course(self):
        """The :class:`CourseConfig` in force for the current episode (UC-08 AC9).

        Equals ``config.course`` unless course randomization sampled a fresh course at the
        last ``reset()``. Read by the evaluator / recorder to stamp each episode's true
        sampled course into ``meta.course`` (so the UC-06 viewer draws the right markers).
        """
        return self._course

    # -- observation encoding -----------------------------------------------------------
    def _observation(self, state) -> np.ndarray:
        target = current_target(self._course, self._gates_passed)
        rel = target - state.position
        parts = [rel, state.attitude, state.velocity, state.angular_velocity]
        if self._obstacle_vision_enabled:
            # Append the nearest-k egocentric obstacle encoding after the base 12 dims and
            # BEFORE nan_to_num, so sanitation wraps the full concatenated observation (UC-15).
            # Uses the body/heading-frame yaw (attitude[2]) and the active course's obstacles.
            parts.append(
                obstacle_vision_features(
                    state.position,
                    state.attitude[2],
                    self._course.obstacles,
                    self._course.floor_z,
                    self._obstacle_vision_k,
                )
            )
        if self._battery_enabled:
            # UC-17 AC4: width-1 battery block, appended AFTER the obstacle-vision block (so the
            # dim order matches the schema block order) and BEFORE nan_to_num. Encoded as
            # **depletion = 1.0 - battery** (0 at full charge), so the zero-init graft input (0)
            # coincides with the trained full-charge baseline — the grafted actor's action on old
            # inputs stays bit-identical (documented in obs_schema / graft_actor).
            parts.append(np.array([1.0 - state.battery], dtype=np.float64))
        if self._damage_enabled:
            # UC-19 AC5: width-1 damage block, appended AFTER the battery block (so the dim order
            # matches the schema block order) and BEFORE nan_to_num. Encoded as **1.0 - integrity**
            # (0 at full integrity), so the zero-init graft input (0) coincides with the trained
            # pristine baseline — the grafted actor's action on old inputs stays bit-identical
            # (documented in obs_schema / graft_actor), exactly mirroring the battery encoding.
            parts.append(np.array([1.0 - state.integrity], dtype=np.float64))
        obs = np.concatenate(parts).astype(np.float32)
        # Belt-and-braces: the env never emits a non-finite observation to the policy.
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

    def _dist_to_target(self, position: np.ndarray) -> float:
        target = current_target(self._course, self._gates_passed)
        return float(np.linalg.norm(target - position))

    # -- gymnasium API ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rcfg = self.config.randomization

        # Pinned draw order: course then dynamics, each guarded so a *disabled* axis makes
        # NO np_random draw (else it would perturb the RNG stream and break UC-01..06
        # determinism / byte-identity, AC7). When enabled the sampler draws off the env's
        # seeded RNG, so a seed reproduces the same course *and* dynamics stream (AC4).
        if rcfg.enable_course:
            # UC-18: forward the battery config so an energy-constrained sampled course gets a
            # reachable recharge-pad cover (no-op when the recharge axis is off → byte-identical).
            self._course = sample_course(
                self.np_random, rcfg, self.config.course, battery=self.config.battery
            )
        else:
            self._course = self.config.course

        # UC-37 floor start (AC1): override the spawn z to rest on the floor. Applied AFTER course
        # sampling — on the fixed path it never touches the RNG, and on the sampled path the course
        # (incl. its solvability check) is already drawn, so determinism is intact (AC8). We keep
        # the sampled/fixed (x, y) and drop z to ``floor_z`` (the SimpleDroneAdapter's true ground-
        # rest height; it clamps to floor_z). NOTE: a floored start intentionally violates
        # ``is_course_solvable``'s ``floor_z + z_margin < start_z`` — takeoff is the learned
        # behaviour — so the floored course is deliberately NOT re-validated.
        if self.config.floor_start:
            sx, sy, _sz = self._course.start_position
            self._course = dataclasses.replace(
                self._course, start_position=(sx, sy, self._course.floor_z)
            )

        if rcfg.enable_dynamics:
            dynamics = sample_dynamics(self.np_random, rcfg, DynamicsParams())
        else:
            dynamics = None

        # Apply the per-episode spawn / dynamics before the adapter reset. When every axis is off
        # we skip the call entirely (not even a no-op reconfigure) so the fixed path is byte-
        # identical and never depends on the adapter implementing the hook — a scripted test-double
        # adapter without reconfigure() still works unchanged (AC7). UC-37: ``floor_start`` also
        # requires a reconfigure so the floored spawn z reaches the adapter even on the
        # randomization-off / seed-42 path; the ``hasattr`` guard stays load-bearing (the scripted
        # ``_ScriptedAdapter`` has no ``reconfigure``).
        need_reconfigure = rcfg.enable_course or rcfg.enable_dynamics or self.config.floor_start
        if need_reconfigure and hasattr(self.adapter, "reconfigure"):
            self.adapter.reconfigure(
                start=(
                    self._course.start if (rcfg.enable_course or self.config.floor_start) else None
                ),
                dynamics=dynamics,
            )

        state = self.adapter.reset(seed=seed)
        # UC-37: arm the takeoff latch from the SPAWN state. A floor start (z ≈ floor_z) is NOT
        # taken off → the grounded detector stays disarmed until the drone first lifts (AC3). An
        # airborne start (z above the floor band, e.g. the scripted/legacy z=1.0 fixtures) is
        # considered taken off at step 0 → grounded-termination arms immediately and behaves byte-
        # for-byte as before this UC (AC4). The threshold reuses ``floor_z + floor_epsilon``.
        self._took_off = bool(state.position[2] > self._course.floor_z + self._et_floor_epsilon)
        self._gates_passed = 0
        self._done = False
        self._prev_pos = state.position.copy()
        self._step_count = 0
        # Fresh episode: no prior obstacle contact (edge-trigger state, UC-15).
        self._prev_contact = False
        # Fresh episode: not docked (UC-16). Recomputed every step; reset here for the pre-first-
        # step read and so ``info["docked"]`` is well-defined before step() runs.
        self._docked = False
        # Fresh episode: reset the UC-25 early-termination bookkeeping. Counters start at 0 so
        # neither detector can fire until it accumulates a full ``stuck_window`` of qualifying
        # steps (the warm-up guard, AC5). ``_best_dist`` starts at +inf so the very first step's
        # distance is always recorded as progress (no spurious stuck increment on step 1).
        # ``_prev_battery`` / ``_prev_integrity`` seed the productive-service detector from the
        # reset state (default-safe 1.0 when battery/damage physics are off). Pure bookkeeping —
        # no observation / RNG effect.
        self._grounded_counter = 0
        self._stuck_counter = 0
        self._best_dist = float("inf")
        self._prev_battery = float(state.battery)
        self._prev_integrity = float(state.integrity)
        # Effective step budget scales with the active course's gate count (UC-09): a longer
        # course gets proportionally more time so it stays completable. N=1 => 400 exactly.
        episode = self.config.episode
        self._max_steps = episode.max_steps + episode.steps_per_gate * (self._course.num_gates - 1)
        # UC-18: grant extra budget per **rechargeable** pad so a legitimate recharge detour
        # (descend + dwell-to-full + climb-out) can still finish within the timeout — the UC-16
        # "dwell consumes the step budget" pitfall. Added ONLY when ≥1 rechargeable pad is on the
        # active course, so a no-recharge course keeps the exact UC-09 budget (byte-identical, AC5).
        num_recharge_pads = sum(1 for pad in self._course.pads if pad.rechargeable)
        if num_recharge_pads > 0:
            self._max_steps += episode.recharge_step_allowance * num_recharge_pads
        # UC-19: symmetric budget for a legitimate **repair** detour (descend + dwell-to-restore +
        # climb-out). Added ONLY when ≥1 repairable pad is on the active course, so a no-repair
        # course keeps the exact UC-09/18 budget (byte-identical, AC1). Independent of the recharge
        # allowance — a pad that both recharges and repairs adds both allowances (both detours may
        # be needed).
        num_repair_pads = sum(1 for pad in self._course.pads if pad.repairable)
        if num_repair_pads > 0:
            self._max_steps += episode.repair_step_allowance * num_repair_pads
        info = {
            "phase": self._phase_str(),
            "backend": self.backend,
            "target_gate": self._gates_passed,
        }
        return self._observation(state), info

    def _phase_str(self) -> str:
        """Human-readable phase derived from the indexed walk (back-compat ``info['phase']``).

        ``done`` once the finish is crossed; ``to_finish`` once all gates are passed but the
        finish is not yet crossed; else ``to_gate_<i>`` naming the current target gate index.
        """
        if self._done:
            return "done"
        if self._gates_passed >= self._course.num_gates:
            return "to_finish"
        return f"to_gate_{self._gates_passed}"

    def step(self, action):
        course = self._course
        dist_prev = self._dist_to_target(self._prev_pos)

        state = self.adapter.step(np.asarray(action, dtype=np.float64))
        self._step_count += 1

        # UC-37: is the drone airborne (above the floor band) this step? This gates the survival
        # reward (AC5) and latches the per-episode takeoff flag (AC3). Threshold reuses
        # ``floor_z + floor_epsilon`` — the same band the grounded detector uses.
        airborne = bool(state.position[2] > course.floor_z + self._et_floor_epsilon)
        self._took_off = self._took_off or airborne

        self._gates_passed, self._done, event = advance(
            course, self._gates_passed, self._done, self._prev_pos, state.position
        )
        completed = event == "finish"

        # Distance to the (possibly newly-advanced) target, for the progress term.
        dist_curr = self._dist_to_target(state.position)

        # Obstacle contact (UC-15 AC2): swept per-step detection over the step segment (anti-
        # tunneling), edge-triggered so a sustained overlap is penalised once. NEVER feeds
        # ``terminated`` — obstacle contact is a reward-only signal; only floor/ceiling/OOB
        # crashes end an episode, so the drone may recover aerially and still complete (AC9).
        contact = any(
            segment_contact(self._prev_pos, state.position, obstacle, course.floor_z)
            for obstacle in course.obstacles
        )
        obstacle_contact = bool(contact and not self._prev_contact)
        self._prev_contact = contact

        # Pad docking (UC-16 AC2/AC3/AC7): re-classify a floor contact as a controlled *dock*
        # (not a crash) when it lands slow, upright, and over a pad. Evaluated FRESH every step
        # (stateless) off ``self._prev_pos`` (this step's START position — updated at the very end
        # of step(), so the ``(prev_z - curr_z)/dt`` descent proxy is valid at the contact step)
        # and the adapter's contact flag. Ceiling contact never docks (floor-only). With
        # ``course.pads == ()`` the predicate short-circuits ``False`` ⇒ ``crash == state.collided``
        # and the whole termination/reward path is byte-identical to pre-UC-16 (AC8).
        docked = evaluate_dock(
            self._prev_pos,
            state.position,
            state.attitude,
            state.collided,
            course.floor_z,
            course.pads,
            self.config.episode.dt,
            self.config.dock.max_dock_descent_speed,
            self.config.dock.max_dock_tilt,
        )
        # A crash is a collision that is NOT a controlled dock. This is the only signal that
        # feeds termination and the collision penalty now (UC-16): a dock keeps the episode alive.
        crash = bool(state.collided and not docked)
        # UC-37 (AC2): before the drone has EVER taken off, a floor contact while resting in the
        # floor band is NOT a crash — a floor-start drone sitting on the ground at zero throttle
        # would otherwise insta-crash at step 1 (``SimpleDroneAdapter`` sets ``collided`` on floor
        # contact). Suppress it only pre-takeoff and only within the floor band; this is SEPARATE
        # from the UC-25/36 grounded detector, and it is placed BEFORE the early-termination block
        # below so a legitimate grounded/stuck cut can still set ``crash=True`` for a never-flyer.
        if (
            not self._took_off
            and state.collided
            and state.position[2] <= course.floor_z + self._et_floor_epsilon
        ):
            crash = False
        # Single authoritative docked state (no vestigial copy): LEVEL/every-step so it persists
        # across dwell (each docked step re-satisfies the rule) and clears on takeoff (collided
        # goes False ⇒ docked False), giving repeatable dock↔fly within one episode (AC4/AC5).
        self._docked = docked

        # UC-18 recharge (AC1/AC2): while docked on a **rechargeable** pad the battery refills.
        # Gated on THREE conditions so the pad is load-bearing but never over-reaches:
        #   * ``self._battery_enabled`` — no battery physics ⇒ nothing to recharge (byte-identical);
        #   * ``docked`` (the UC-16 stateless dock predicate: floor contact, slow, upright, over a
        #     pad) — a mere hover over a recharge pad is NOT docked, so it does NOT recharge (AC2);
        #   * ``pad is not None and pad.rechargeable`` — a plain (non-recharge) pad never refills,
        #     so docking on it drains-only exactly as UC-16/17 (AC1).
        # The adapter already applied this step's drain (thrust used the start-of-step charge), so
        # the increment nets against it: with the documented net-positive invariant
        # (recharge_rate > docked drain) a docked step is a strict gain. We overwrite ``state`` via
        # ``dataclasses.replace`` BEFORE ``_observation`` so the refill shows in the SAME step's
        # battery obs (AC6 step-by-step dwell is observable). ``pad_under`` reuses the exact UC-16
        # geometry the dock predicate already checked, so the two never disagree on which pad.
        if self._battery_enabled and docked:
            pad = pad_under(state.position, course.pads)
            if pad is not None and pad.rechargeable:
                delta = self.config.battery.recharge_rate * self.config.episode.dt
                new_battery = self.adapter.recharge(delta)
                state = dataclasses.replace(state, battery=new_battery)

        # UC-19 damage/repair (AC2/AC4), applied in the documented **damage-then-repair** order so a
        # same-step contact-on-a-pad first sheds then restores integrity, deterministically. Both
        # mutate the adapter's integrity and overwrite ``state`` via ``dataclasses.replace`` BEFORE
        # ``_observation`` so the change shows in the SAME step's damage obs dim. Neither feeds
        # ``crash`` / ``terminated`` nor the reward — integrity is a reward-neutral sensory/handicap
        # signal (AC2); only floor/ceiling crashes and a valid completion end an episode.
        if self._damage_enabled:
            # DAMAGE (AC2): reuse the UC-15 edge-triggered ``obstacle_contact`` (True only on the
            # step a contact *begins* — once per contact, never per overlapping frame, and never
            # floor/ceiling). ``obstacle_contact`` is computed above from the same signal that feeds
            # the obstacle penalty, so damage and penalty fire on exactly the same events.
            if obstacle_contact:
                new_integrity = self.adapter.damage(self.config.damage.damage_per_contact)
                state = dataclasses.replace(state, integrity=new_integrity)
            # REPAIR (AC4): mirror the UC-18 recharge-on-dock gate — restore only while the UC-16
            # stateless dock predicate holds (floor contact, slow, upright, over a pad) AND the pad
            # under the drone is ``repairable``. A mere hover over a repair pad is NOT docked ⇒ no
            # repair; docking on a non-repair pad ⇒ no repair. ``pad_under`` reuses the exact UC-16
            # geometry the dock predicate already checked, so the two never disagree on which pad.
            if docked:
                pad = pad_under(state.position, course.pads)
                if pad is not None and pad.repairable:
                    delta = self.config.damage.repair_rate * self.config.episode.dt
                    new_integrity = self.adapter.repair(delta)
                    state = dataclasses.replace(state, integrity=new_integrity)

        # UC-25 grounded / no-progress early termination. Evaluated here — AFTER dock/recharge/
        # repair are resolved (so ``self._docked`` and the post-service ``state.battery`` /
        # ``state.integrity`` are final) and BEFORE ``compute_reward``. UC-38: a fired cut no longer
        # blanket-folds into ``crash``; only the reward-side ``penalize_collision`` (computed below)
        # decides the penalty — a grounded cut pays it, a no-progress ("stuck") cut does not.
        # Nothing here touches the observation or the RNG stream: a flying / promptly-crashing
        # episode never accumulates a full window, so its per-step outcomes stay byte-identical to
        # UC-19 (AC5/AC6).
        early_termination = None
        if self._et_enabled:
            floor_z = course.floor_z
            # Productive-service flag (AC4 non-loophole): a dock exempts the episode ONLY while it
            # is *actively improving* charge or integrity — not on mere pad presence. This closes
            # the "dock once and idle forever" loophole: once battery/integrity stop rising (full
            # or nothing to fix) the dwell is no longer productive and the counters resume climbing.
            productive = self._docked and (
                (self._battery_enabled and state.battery > self._prev_battery)
                or (self._damage_enabled and state.integrity > self._prev_integrity)
            )

            # Grounded (resting) detector: airborne-frame off, sitting in the floor band at
            # near-zero speed. Docked ⇒ never grounded (a pad on the floor must not trip it). The
            # low-speed guard makes this a genuine "sitting on the ground" detector and prevents a
            # low-but-progressing real-adapter flight (real speed ≫ epsilon) from being mis-cut.
            grounded = (
                (not self._docked)
                and (floor_z <= state.position[2] <= floor_z + self._et_floor_epsilon)
                and (float(np.linalg.norm(state.velocity)) <= self._et_rest_speed_epsilon)
            )
            # UC-37 (AC3): the grounded detector ARMS only after the first takeoff. A floor-start
            # drone that has never lifted off is NOT cut by this detector (it is instead bounded by
            # the no-progress/stuck detector below and the episode timeout, AC7); once it takes off,
            # dropping back and resting fires the grounded cut exactly as UC-36. The no-progress
            # (stuck) detector is deliberately left UNGATED so a never-flyer is still bounded.
            if grounded and self._took_off:
                self._grounded_counter += 1
            else:
                self._grounded_counter = 0

            # No-progress (stuck) detector, measured against the best distance reached so far
            # (robust to hover oscillation / jitter, not just consecutive-pair deltas). On a target
            # change this step (a gate passed, or the finish crossed) the metric's reference jumps,
            # so re-baseline to the new distance and clear the counter. ``dist_curr`` already tracks
            # the finish plane once all gates are passed (geometry.current_target), so the last-leg
            # pitfall is handled for free.
            if event in ("gate", "finish"):
                self._best_dist = dist_curr
                self._stuck_counter = 0
            elif dist_curr < self._best_dist - self._et_progress_epsilon:
                self._best_dist = dist_curr
                self._stuck_counter = 0
            else:
                self._stuck_counter += 1

            # A productive service dwell stays fully alive: reset BOTH counters. When service stops
            # being productive (full / idle) the counters climb again and cut after the window.
            if productive:
                self._grounded_counter = 0
                self._stuck_counter = 0

            # Fire: either counter reaching its window cuts the episode. The grounded detector uses
            # its own short ``grounded_window`` (UC-36) — a floored drone is unambiguously dead and
            # needn't run the full ``stuck_window`` — while the no-progress detector keeps the more
            # lenient ``stuck_window``. Grounded takes priority in the reported reason if both trip
            # on the same step.
            if self._grounded_counter >= self._et_grounded_window:
                early_termination = "grounded"
            elif self._stuck_counter >= self._et_stuck_window:
                early_termination = "stuck"

        # UC-38: decouple the no-progress/timeout cut from the collision penalty. ``crash`` keeps
        # its narrow meaning — a GENUINE floor/ceiling/OOB collision (raw ``state.collided`` and
        # not a controlled dock, pre-takeoff-suppressed above) — and is the ONLY signal that eats
        # ``collision_penalty`` on its own. ``penalize_collision`` is the reward-side flag: a real
        # crash always pays the penalty (and dominates a same-step stuck cut, since ``crash`` wins
        # here regardless of ``early_termination``), and the GROUNDED cut also pays it (AC-3: a
        # post-takeoff drop back onto the floor is a failed flight). The no-progress ("stuck") cut
        # and a pure ``max_steps`` timeout are penalty-free — "stayed airborne but ran out of time /
        # stopped progressing" must not be punished like smashing into the floor, or UC-37's
        # +airborne_bonus survival gradient is swamped by the −collision_penalty terminal (the −105
        # trap). The termination reason is reported independently via ``info["early_termination"]``.
        penalize_collision = crash or (early_termination == "grounded")

        reward = compute_reward(
            dist_to_target_prev=dist_prev,
            dist_to_target_curr=dist_curr,
            event=event,
            # UC-16/UC-38: pass ``penalize_collision`` — a genuine crash OR a grounded cut earns
            # the collision_penalty; a controlled dock, a no-progress ("stuck") cut, and a pure
            # timeout do not. A raw floor/ceiling contact still eats collision_penalty as before.
            collided=penalize_collision,
            completed=completed,
            cfg=self.config.reward,
            num_gates=course.num_gates,
            obstacle_contact=obstacle_contact,
            # UC-37 (AC5): pay the survival bonus only while airborne (above the floor band); zero
            # on/at the floor, so sitting on the ground earns nothing.
            airborne=airborne,
        )

        # UC-16/UC-25/UC-38: a dock does NOT terminate. An episode ends on a valid completion, a
        # genuine (non-dock) crash, OR any early-termination cut (grounded/stuck) — control flow is
        # byte-for-byte as before UC-38 (both cuts still set ``terminated=True``); only the reward
        # magnitude and ``info["collided"]`` on a stuck cut changed (a pure timeout is truncated).
        terminated = bool(completed or crash or early_termination is not None)
        truncated = bool(not terminated and self._step_count >= self._max_steps)

        info = {
            "phase": self._phase_str(),
            "backend": self.backend,
            "event": event,
            # UC-16/UC-38: ``collided`` reports whether the collision penalty was charged — True
            # for a genuine floor/ceiling/OOB crash AND for a grounded (post-takeoff drop) cut,
            # False for a controlled dock, a no-progress ("stuck") cut, and a pure timeout. It no
            # longer falsely claims a collision for a no-progress cut (AC-4); the authoritative
            # termination reason is ``info["early_termination"]``. No downstream consumer reads
            # ``collided`` (only ``is_success``), so flipping it on stuck cuts is safe.
            "collided": bool(penalize_collision),
            # UC-16: authoritative docked flag, surfaced via ``info`` ONLY (the observation vector
            # is byte-identical — no obs-schema block, no checkpoint invalidation, AC6). LEVEL/
            # every-step so it stays True across dwell steps (AC4) and clears on takeoff (AC5).
            "docked": bool(self._docked),
            "completed": bool(completed),
            # UC-15: True only on the step an obstacle contact *begins* (edge-triggered), i.e.
            # the step the severe non-terminating obstacle penalty was applied. Additive key.
            "obstacle_contact": obstacle_contact,
            "steps": self._step_count,
            # Current target gate index (UC-09): 0..N-1 while chasing gates, clamped to N
            # once all gates are passed (targeting the finish). Additive; viewer highlights it.
            "target_gate": min(self._gates_passed, course.num_gates),
            # World-frame drone position this step (UC-05 recording draws the flight path).
            # Additive key; existing tests assert membership, so this stays back-compatible.
            "position": state.position.copy(),
            # UC-25/UC-38: reason this episode was cut early — ``"grounded"`` (rested on the floor
            # after takeoff), ``"stuck"`` (no course progress for a full window), or ``None`` (not
            # cut early — includes a pure ``max_steps`` timeout, which reports ``truncated=True``).
            # This is the AUTHORITATIVE termination-reason field: ``info["collided"]`` is True only
            # when the collision penalty was charged (genuine crash or grounded cut), NOT on a
            # stuck cut. Additive key; existing tests assert membership, so this stays safe.
            "early_termination": early_termination,
        }
        if terminated or truncated:
            info["completion_time"] = (
                self._step_count * self.config.episode.dt if completed else None
            )
            info["is_success"] = bool(completed)

        self._prev_pos = state.position.copy()
        # UC-25: remember this step's post-service battery / integrity so next step's productive-
        # service flag can detect a strict improvement (charge/integrity rising). Updated alongside
        # ``_prev_pos``; default-safe (1.0) when the respective physics axis is off.
        self._prev_battery = float(state.battery)
        self._prev_integrity = float(state.integrity)
        return self._observation(state), float(reward), terminated, truncated, info

    def close(self) -> None:
        self.adapter.close()


def make_env(config: EnvConfig | None = None, *, adapter: str = "auto") -> RaceEnv:
    """Factory for a single :class:`RaceEnv` (AC1)."""
    return RaceEnv(config, adapter=adapter)


#: Default worker count for a parallel (subproc) rollout when ``n_envs`` is not set
#: explicitly (UC-26 AC-5/AC-10). Matches the Apple M4 Pro's 8 performance cores.
DEFAULT_PARALLEL_N_ENVS = 8
#: Multiprocessing start method for :class:`SubprocVecEnv` — ``"spawn"`` is the safe choice
#: cross-platform (incl. macOS) and with native libs like pybullet (UC-26 AC-4/AC-10).
VEC_ENV_START_METHOD = "spawn"


def resolve_vec_env(
    adapter: str,
    n_envs_arg: int | None,
    cfg_n_envs: int,
) -> tuple[str, int, str]:
    """Resolve ``(resolved_adapter, resolved_n_envs, vec_backend)`` for a rollout (UC-26).

    Pure decision function (no I/O beyond the one user-facing warning) so it is independently
    unit-testable per branch (AC-1/AC-5):

    * ``resolved_adapter`` — ``adapter`` verbatim unless it is ``"auto"``, which resolves to
      ``"pybullet"`` when the sim stack imports else ``"simple"`` (mirrors
      :func:`~drone_fly.adapter.make_adapter`).
    * ``resolved_n_envs`` — the explicit ``n_envs_arg`` when given; **elif** the adapter is
      parallel-capable (pybullet) it defaults to :data:`DEFAULT_PARALLEL_N_ENVS` (8);
      **else** it falls back to ``cfg_n_envs`` (the pre-UC-26 behaviour — the load-bearing
      branch that keeps the config default working for the simple/CI path).
    * ``vec_backend`` — ``"subproc"`` only when the adapter is parallel-capable **and**
      ``resolved_n_envs > 1``; otherwise ``"dummy"`` (so ``simple``/CI and single-env runs
      stay on :class:`DummyVecEnv`, AC-1/AC-6).

    A user-facing warning fires only when more than one env was *explicitly* requested but the
    adapter is not parallel-capable (the ``simple`` adapter stays serial on ``DummyVecEnv`` —
    AC-1/AC-5 edge case).
    """
    resolved_adapter = (
        adapter if adapter != "auto" else ("pybullet" if pybullet_available() else "simple")
    )
    parallel_capable = resolved_adapter == "pybullet"

    if n_envs_arg is not None:
        resolved_n_envs = int(n_envs_arg)
    elif parallel_capable:
        resolved_n_envs = DEFAULT_PARALLEL_N_ENVS
    else:
        resolved_n_envs = int(cfg_n_envs)
    resolved_n_envs = max(1, resolved_n_envs)

    vec_backend = "subproc" if (parallel_capable and resolved_n_envs > 1) else "dummy"

    if n_envs_arg is not None and n_envs_arg > 1 and not parallel_capable:
        logger.warning(
            "n_envs=%d requested but the resolved adapter is %r (not 'pybullet'); the "
            "'simple' adapter gains nothing from subprocesses, so the rollout stays serial "
            "on DummyVecEnv. Use the pybullet adapter for real cross-core parallelism.",
            n_envs_arg,
            resolved_adapter,
        )

    return resolved_adapter, resolved_n_envs, vec_backend


def _worker_startup_error(
    exc: BaseException,
    suppressed: bool,
    worker_log_dir: str | None,
    n_envs: int,
) -> worker_output.WorkerStartupError:
    """Build a fail-loud :class:`WorkerStartupError` from an opaque SubprocVecEnv death (AC-3).

    When worker output was suppressed to per-worker files, tail those files back into the
    message so the operator sees the *real* worker traceback instead of the opaque
    ``EOFError`` / ``BrokenPipeError``. Without suppression (``--no-tui``) the worker printed its
    traceback to the console already, so we just say so.
    """
    lines = [
        f"A parallel-rollout worker failed during startup ({type(exc).__name__}: {exc}).",
    ]
    if suppressed and worker_log_dir is not None:
        paths = [worker_output.worker_log_path(worker_log_dir, i) for i in range(n_envs)]
        detail = worker_output.read_worker_errors(paths)
        if detail:
            lines.append("Real worker output (per-worker logs):\n" + detail)
        else:
            lines.append(f"Per-worker logs (if any) are under {worker_log_dir!r}.")
    else:
        lines.append(
            "The worker's own traceback was printed to the console above "
            "(no TUI output suppression was active)."
        )
    return worker_output.WorkerStartupError("\n".join(lines))


def build_vec_env(
    *,
    config: EnvConfig | None = None,
    adapter: str = "auto",
    n_envs: int = 1,
    seed: int | None = None,
    training: bool = True,
    norm_reward: bool | None = None,
    vecnormalize_path: str | None = None,
    vec_backend: str = "auto",
    suppress_worker_output: bool = False,
    worker_log_dir: str | None = None,
):
    """Build a ``VecNormalize``-wrapped vectorised env for SB3 (AC4/AC6).

    Parameters
    ----------
    training:
        ``True`` for training (VecNormalize updates its running stats and normalises
        reward); ``False`` for evaluation (frozen stats, raw reward).
    norm_reward:
        Override reward normalisation; defaults to ``training``.
    vecnormalize_path:
        If given, load saved VecNormalize stats from this path (checkpoint resume / eval)
        instead of starting fresh.
    vec_backend:
        Base vec-env backend (UC-26). ``"auto"`` (default) delegates to
        :func:`resolve_vec_env` to pick ``"subproc"`` vs ``"dummy"``; ``"dummy"`` forces the
        serial :class:`DummyVecEnv` (byte-identical to pre-UC-26); ``"subproc"`` forces a
        :class:`SubprocVecEnv` with ``start_method`` :data:`VEC_ENV_START_METHOD`. The
        explicit ``"dummy"``/``"subproc"`` values are internal test hooks and, unlike the
        auto path, forcing subproc on a non-parallel adapter does NOT warn.
    suppress_worker_output:
        Subproc-only. When ``True`` **and** ``worker_log_dir`` is set, each spawned worker
        redirects its own native ``stdout``/``stderr`` (fd 1/2) to a per-worker log FILE
        *before* building its env, so worker native spew can't corrupt the live TUI display
        (UC-26 AC-13) yet survives Windows ``spawn`` and stays inspectable (UC-32 AC-1/AC-2).
        The redirect runs ONLY inside spawned workers — never the main process — and only for
        the subproc backend.
    worker_log_dir:
        UC-32. Directory the per-worker log files are written to when
        ``suppress_worker_output`` is set (subproc backend only). ``None`` (default) means no
        file redirect at all — the pre-UC-26 leak-to-parent behaviour — so the ``suppress`` gate
        is a no-op without a target dir. The directory is pruned and recreated at build so stale
        files from a higher previous ``n_envs`` don't linger (AC-2).
    """
    import os

    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    cfg = config or EnvConfig()
    norm_reward = training if norm_reward is None else norm_reward

    if vec_backend == "auto":
        # Concrete-``n_envs`` callers pass a value, so this is the identity for n_envs; we only
        # consume the backend choice. resolved_adapter is intentionally ignored here — the
        # factory below builds against the ``adapter`` argument as given (``"auto"`` resolves
        # per-process, which is correct for spawned workers too).
        _, _, vec_backend = resolve_vec_env(adapter, n_envs, n_envs)

    # Main-process factory (dummy backend): byte-identical to pre-UC-26. Runs in THIS process.
    def _factory():
        return make_env(cfg, adapter=adapter)

    # Per-index worker factory (subproc backend): the file redirect executes only inside spawned
    # workers (never the main process — a main-process dup2 would kill logging/the TUI) and only
    # when suppress_worker_output is set AND a worker_log_dir is given. Each worker gets a DISTINCT
    # log path so tracebacks never collide (UC-32 AC-2). Otherwise it is byte-identical to the
    # dummy factory (leak-to-parent, the pre-UC-26 default).
    def _make_subproc_factory(idx: int):
        log_path = (
            worker_output.worker_log_path(worker_log_dir, idx)
            if (suppress_worker_output and worker_log_dir is not None)
            else None
        )

        def _worker_factory():
            if log_path is not None:
                worker_output.redirect_worker_fds(log_path)
            return make_env(cfg, adapter=adapter)

        return _worker_factory

    if vec_backend == "subproc":
        from stable_baselines3.common.vec_env import SubprocVecEnv

        # Prune + recreate the workers dir so stale files from a higher previous n_envs don't
        # linger and confuse a later failure diagnosis (AC-2). Only when we will actually write.
        if suppress_worker_output and worker_log_dir is not None:
            import shutil

            shutil.rmtree(worker_log_dir, ignore_errors=True)
            os.makedirs(worker_log_dir, exist_ok=True)

        factories = [_make_subproc_factory(i) for i in range(max(1, n_envs))]
        # Fail-loud (AC-3): a worker that dies during construction surfaces in SB3 as an opaque
        # EOFError / BrokenPipeError [WinError 109] at the first inter-process recv inside
        # SubprocVecEnv.__init__. Catch it and re-raise a WorkerStartupError enriched with the
        # real per-worker tracebacks (when suppression wrote them to files), or a clear pointer
        # to the console output (--no-tui, no suppression). The success path is untouched, so the
        # --no-tui path stays byte-identical (AC-10).
        try:
            venv = SubprocVecEnv(factories, start_method=VEC_ENV_START_METHOD)
        except (EOFError, OSError) as exc:
            raise _worker_startup_error(
                exc, suppress_worker_output, worker_log_dir, max(1, n_envs)
            ) from exc
    else:
        venv = DummyVecEnv([_factory for _ in range(max(1, n_envs))])
    if seed is not None:
        venv.seed(seed)

    # UC-22: wrap the training env in VecMonitor so SB3's ``ep_info_buffer`` /
    # ``ep_success_buffer`` are populated and ``rollout/{ep_rew_mean,ep_len_mean,success_rate}``
    # are collected into the existing CSV/TensorBoard outputs (the data the live TUI needs).
    # It sits INSIDE VecNormalize (wraps the raw episodes, before both the ``.load`` and fresh
    # branches) so reported episode reward/length are UN-normalised, and is gated on
    # ``training`` so eval/non-training construction stays byte-identical (no Monitor, no stat
    # change). Seeding stays on the raw env above — no RNG/obs change.
    if training:
        venv = VecMonitor(venv, info_keywords=("is_success",))

    if vecnormalize_path is not None:
        venv = VecNormalize.load(vecnormalize_path, venv)
        venv.training = training
        venv.norm_reward = norm_reward
    else:
        venv = VecNormalize(venv, training=training, norm_obs=True, norm_reward=norm_reward)
    return venv
