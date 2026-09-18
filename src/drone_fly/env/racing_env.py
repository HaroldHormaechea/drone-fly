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

from drone_fly.adapter import make_adapter
from drone_fly.controller.encoding import ACTION_DIM, OBS_DIM
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

        self.adapter = make_adapter(
            adapter,
            course.start,
            floor_z=course.floor_z,
            ceiling_z=course.ceiling_z,
            dt=self.config.episode.dt,
            battery=self.config.battery if self._battery_enabled else None,
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

        if rcfg.enable_dynamics:
            dynamics = sample_dynamics(self.np_random, rcfg, DynamicsParams())
        else:
            dynamics = None

        # Apply the per-episode spawn / dynamics before the adapter reset. When BOTH axes
        # are off we skip the call entirely (not even a no-op reconfigure) so the fixed
        # path is byte-identical and never depends on the adapter implementing the hook —
        # a scripted test-double adapter without reconfigure() still works unchanged (AC7).
        if rcfg.enable_course or rcfg.enable_dynamics:
            self.adapter.reconfigure(
                start=self._course.start if rcfg.enable_course else None,
                dynamics=dynamics,
            )

        state = self.adapter.reset(seed=seed)
        self._gates_passed = 0
        self._done = False
        self._prev_pos = state.position.copy()
        self._step_count = 0
        # Fresh episode: no prior obstacle contact (edge-trigger state, UC-15).
        self._prev_contact = False
        # Fresh episode: not docked (UC-16). Recomputed every step; reset here for the pre-first-
        # step read and so ``info["docked"]`` is well-defined before step() runs.
        self._docked = False
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

        reward = compute_reward(
            dist_to_target_prev=dist_prev,
            dist_to_target_curr=dist_curr,
            event=event,
            # UC-16: pass the CRASH flag, not the raw contact flag — a controlled dock earns the
            # neutral (no collision_penalty) reward while every other floor/ceiling contact still
            # eats collision_penalty. No new reward term (docking is unincentivised this UC).
            collided=crash,
            completed=completed,
            cfg=self.config.reward,
            num_gates=course.num_gates,
            obstacle_contact=obstacle_contact,
        )

        # UC-16: a dock does NOT terminate — only a valid completion or a (non-dock) crash does.
        terminated = bool(completed or crash)
        truncated = bool(not terminated and self._step_count >= self._max_steps)

        info = {
            "phase": self._phase_str(),
            "backend": self.backend,
            "event": event,
            # UC-16: ``collided`` now reports the CRASH (terminating floor/ceiling contact), i.e.
            # a controlled dock is NOT reported as a collision. A dock is surfaced separately via
            # ``docked`` below; no downstream consumer reads ``collided``, so this stays safe.
            "collided": bool(crash),
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
        }
        if terminated or truncated:
            info["completion_time"] = (
                self._step_count * self.config.episode.dt if completed else None
            )
            info["is_success"] = bool(completed)

        self._prev_pos = state.position.copy()
        return self._observation(state), float(reward), terminated, truncated, info

    def close(self) -> None:
        self.adapter.close()


def make_env(config: EnvConfig | None = None, *, adapter: str = "auto") -> RaceEnv:
    """Factory for a single :class:`RaceEnv` (AC1)."""
    return RaceEnv(config, adapter=adapter)


def build_vec_env(
    *,
    config: EnvConfig | None = None,
    adapter: str = "auto",
    n_envs: int = 1,
    seed: int | None = None,
    training: bool = True,
    norm_reward: bool | None = None,
    vecnormalize_path: str | None = None,
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
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cfg = config or EnvConfig()
    norm_reward = training if norm_reward is None else norm_reward

    def _factory():
        return make_env(cfg, adapter=adapter)

    venv = DummyVecEnv([_factory for _ in range(max(1, n_envs))])
    if seed is not None:
        venv.seed(seed)

    if vecnormalize_path is not None:
        venv = VecNormalize.load(vecnormalize_path, venv)
        venv.training = training
        venv.norm_reward = norm_reward
    else:
        venv = VecNormalize(venv, training=training, norm_obs=True, norm_reward=norm_reward)
    return venv
