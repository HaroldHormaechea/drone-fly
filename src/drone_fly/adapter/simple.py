"""Pure-numpy fixed-dynamics drone model — the hermetic CI / smoke / default backend.

:class:`SimpleDroneAdapter` is a deliberately small analytic point-mass + first-order
attitude model with **no** ``pybullet`` dependency. It exists so the whole default code
path — CI lint+test, ``smoke-train``, every non-``sim`` unit test — runs without the
native sim toolchain (the load-bearing CI-hermeticity constraint of UC-03).

It is emphatically **not** presented as mastery physics: the ≥80% completion bar (AC10) is
defined against real ``gym-pybullet-drones`` dynamics via
:class:`~drone_fly.adapter.pybullet_adapter.PyBulletAdapter`. This model only has to be
(a) deterministic and seedable, (b) controllable enough that a policy can in principle fly
start→gate→finish, and (c) able to detect floor / ceiling contact — enough to prove the
env + policy + PPO loop wire together and stay finite.

Dynamics (fixed; AC11 — no domain randomization)
------------------------------------------------
The canonical CTBR action drives a rigid point mass:

* Attitude follows the commanded body rates through a first-order response
  (``attitude += rate_cmd * MAX_BODY_RATE * dt``), so roll/pitch/yaw are directly
  commandable but rate-limited.
* Collective thrust ``T = throttle * MAX_THRUST`` acts along the tilted body-z axis; its
  world-frame projection gives vertical lift ``T·cos(roll)·cos(pitch)`` against gravity
  and horizontal acceleration from the tilt (``+pitch`` → ``+x`` forward, ``+roll`` →
  ``-y``). Linear drag damps velocity.

All constants are module-level and documented so the difficulty is transparent and later
tunable. There is no stochasticity, so a fixed seed yields a bit-identical trajectory
(AC12); ``seed`` is accepted for API symmetry and to seed a private RNG for any future
noise without changing the current deterministic behaviour.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from drone_fly.adapter.base import DroneAdapter, DroneState, sanitize_action
from drone_fly.controller.encoding import HOVER_THROTTLE

# --- Base dynamics constants (documented) ----------------------------------------------
# These are the *defaults*: with no reconfigure (UC-08 randomization off) the instance
# dynamics equal these exactly, so a fixed-course episode is byte-identical to UC-01..06.
GRAVITY = 9.81  # m/s^2
BASE_MASS = 1.0  # kg (point mass)
BASE_MAX_THRUST = 2.0 * BASE_MASS * GRAVITY  # N — hover is throttle≈0.5, leaving head/foot room
BASE_MAX_BODY_RATE = 4.0  # rad/s at full stick — maps normalised [-1,1] attitude command
BASE_LINEAR_DRAG = 0.15  # 1/s — velocity damping coefficient
ATTITUDE_LIMIT = np.pi / 3.0  # rad — clamp roll/pitch so tilt projection stays sane
MAX_SPEED = 25.0  # m/s — hard clamp so a divergent policy can't produce non-finite state

# Back-compat aliases (pre-UC-08 names). The instance attributes below are the live knobs.
MASS = BASE_MASS
MAX_THRUST = BASE_MAX_THRUST
MAX_BODY_RATE = BASE_MAX_BODY_RATE
LINEAR_DRAG = BASE_LINEAR_DRAG

#: Warm-up action applied while the control-latency buffer fills (exact hover at base
#: dynamics: throttle ``HOVER_THROTTLE`` -> 0.5 * 19.62 / 1.0 == 9.81 == g, level attitude).
#: The throttle channel is tied to the canonical :data:`~drone_fly.controller.encoding.
#: HOVER_THROTTLE` so it cannot silently desync from the training-time hover-bias init (UC-40).
_WARMUP_ACTION = np.array([HOVER_THROTTLE, 0.0, 0.0, 0.0], dtype=np.float64)


class SimpleDroneAdapter(DroneAdapter):
    """Deterministic point-mass drone with floor/ceiling contact, no pybullet.

    Parameters
    ----------
    start_position:
        World-frame spawn position ``[x, y, z]``.
    floor_z, ceiling_z:
        Arena vertical bounds; contact with either flags a collision and is where the env
        terminates the episode.
    dt:
        Integration timestep (seconds).
    """

    backend = "simple"

    def __init__(
        self,
        start_position: np.ndarray,
        *,
        floor_z: float,
        ceiling_z: float,
        dt: float,
        battery=None,
        damage=None,
    ) -> None:
        self._start = np.asarray(start_position, dtype=np.float64).reshape(3).copy()
        self._floor_z = float(floor_z)
        self._ceiling_z = float(ceiling_z)
        self._dt = float(dt)
        self._rng = np.random.default_rng(0)
        # Independent, reconfigurable dynamics knobs (UC-08 AC5). Defaults == the base
        # constants, so with no reconfigure the model is byte-identical to UC-01..06.
        self._mass = BASE_MASS
        self._max_thrust = BASE_MAX_THRUST
        self._drag = BASE_LINEAR_DRAG
        self._max_body_rate = BASE_MAX_BODY_RATE
        self._latency = 0  # control-latency delay in steps (0 == no buffer in the path)
        self._action_queue: deque[np.ndarray] = deque()
        self._position = self._start.copy()
        self._velocity = np.zeros(3, dtype=np.float64)
        self._attitude = np.zeros(3, dtype=np.float64)
        self._angular_velocity = np.zeros(3, dtype=np.float64)
        # Battery drain + thrust-impact (UC-17). ``battery`` is a ``BatteryConfig`` (duck-typed:
        # only its ``enabled``/``idle_rate``/``throttle_rate`` fields + ``ceiling_factor`` method
        # are read — no import, so the adapter stays below ``env.config`` in the layering) or
        # ``None``. Disabled ⇒ the step path never reads or drains ``_battery`` and thrust stays
        # exactly ``throttle * _max_thrust`` (byte-identity, AC5/AC6). ``_battery`` resets to full.
        self._battery_cfg = battery
        self._battery_enabled = battery is not None and bool(battery.enabled)
        self._battery = 1.0
        # Integrity damage + control-authority impact (UC-19). ``damage`` is a ``DamageConfig``
        # (duck-typed: only its ``enabled``/``min_authority`` fields + ``authority_factor`` method
        # are read — no import, so the adapter stays below ``env.config`` in the layering) or
        # ``None``. Disabled ⇒ the step path never reads or mutates ``_integrity`` and the body
        # rate stays exactly ``* _max_body_rate`` (byte-identity, AC3). ``_integrity`` resets to
        # full. Damage/repair are NOT applied here — the env calls :meth:`damage`/:meth:`repair`
        # from its step (mirroring the recharge idiom), so this adapter only *reads* integrity to
        # scale the body rate and *mutates* it through those two hooks.
        self._damage_cfg = damage
        self._damage_enabled = damage is not None and bool(damage.enabled)
        self._integrity = 1.0

    def reconfigure(self, *, start=None, dynamics=None) -> None:
        """Apply a new spawn and/or dynamics for the next episode (UC-08 AC5, AC7).

        Called by the env at ``reset()`` **before** :meth:`reset`. Each argument is applied
        only when not ``None`` — a call with both ``None`` (the disabled-randomization path)
        is a pure no-op, leaving the fixed dynamics and spawn untouched (byte-identity). The
        knobs are stored independently: mass and thrust do **not** recompute each other, so
        ``sample_dynamics`` scaling mass alone genuinely perturbs the trajectory.
        """
        if start is not None:
            self._start = np.asarray(start, dtype=np.float64).reshape(3).copy()
        if dynamics is not None:
            self._mass = float(dynamics.mass)
            self._max_thrust = float(dynamics.max_thrust)
            self._drag = float(dynamics.drag)
            self._max_body_rate = float(dynamics.max_body_rate)
            self._latency = int(dynamics.latency_steps)

    def recharge(self, delta: float) -> float:
        """Add ``delta`` charge to the battery, clamped at ``1.0``; return the new charge (UC-18).

        Geometry-agnostic by design: the adapter knows *how much* charge to add, never *whether*
        a recharge is warranted — the env decides that (drone docked on a ``rechargeable`` pad)
        and calls this with the per-step increment ``recharge_rate * dt``. ``delta`` is floored at
        ``0`` so a negative increment can never *drain* through this path, and the result is
        clamped at ``1.0`` (AC1 ceiling — a full battery cannot overcharge). Safe to call on the
        battery-disabled path too (``_battery`` stays a plain float), though the env only calls it
        when battery physics are enabled.
        """
        self._battery = min(1.0, self._battery + max(0.0, float(delta)))
        return self._battery

    def damage(self, amount: float) -> float:
        """Subtract ``amount`` from integrity, clamped at ``0.0``; return the new integrity (UC-19).

        Symmetric to :meth:`recharge`: geometry-agnostic — the env decides *whether* to call this
        (a UC-15 obstacle-contact edge event) and passes ``damage_per_contact``; the adapter only
        applies the decrement. ``amount`` is floored at ``0`` so a negative value can never *heal*
        through this path, and the result is clamped ``≥ 0`` (integrity cannot go negative). Safe to
        call on the damage-disabled path too (``_integrity`` stays a plain float), though the env
        only calls it when damage physics are enabled.
        """
        self._integrity = max(0.0, self._integrity - max(0.0, float(amount)))
        return self._integrity

    def repair(self, delta: float) -> float:
        """Add ``delta`` to integrity, clamped at ``1.0``; return the new integrity (UC-19).

        Symmetric to :meth:`recharge`: the env decides *whether* to call this (docked on a
        ``repairable`` pad) and passes the per-step increment ``repair_rate * dt``; the adapter
        only applies it. ``delta`` is floored at ``0`` so a negative value can never *damage*
        through this path, and the result is clamped ``≤ 1.0`` (a pristine drone cannot overheal).
        """
        self._integrity = min(1.0, self._integrity + max(0.0, float(delta)))
        return self._integrity

    def _state(self, collided: bool) -> DroneState:
        return DroneState(
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            attitude=self._attitude.copy(),
            angular_velocity=self._angular_velocity.copy(),
            collided=collided,
            # UC-17: normalized charge. Stays ``1.0`` on the disabled path (never drained), so a
            # battery-off episode reports a full charge and is byte-identical to pre-UC-17.
            battery=self._battery,
            # UC-19: normalized integrity. Stays ``1.0`` on the disabled path (never mutated), so a
            # damage-off episode reports full integrity and is byte-identical to pre-UC-19.
            integrity=self._integrity,
        )

    def reset(self, seed: int | None = None) -> DroneState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._position = self._start.copy()
        self._velocity = np.zeros(3, dtype=np.float64)
        self._attitude = np.zeros(3, dtype=np.float64)
        self._angular_velocity = np.zeros(3, dtype=np.float64)
        # UC-17: reset battery to full. Harmless when disabled (never read/drained thereafter).
        self._battery = 1.0
        # UC-19: reset integrity to full. Harmless when disabled (never read/mutated thereafter).
        self._integrity = 1.0
        # Prime the control-latency buffer with warm-up hover actions so real commands are
        # delayed by exactly ``_latency`` steps. Empty (and never touched) when latency == 0.
        self._action_queue = deque()
        if self._latency > 0:
            warm = sanitize_action(_WARMUP_ACTION)
            for _ in range(self._latency):
                self._action_queue.append(warm)
        collided = self._position[2] <= self._floor_z or self._position[2] >= self._ceiling_z
        return self._state(collided)

    def step(self, action: np.ndarray) -> DroneState:
        a = sanitize_action(action)
        if self._latency > 0:
            # FIFO control latency: enqueue the fresh command, apply the oldest pending one.
            # With latency == 0 this branch is skipped entirely (bit-identical to UC-03).
            self._action_queue.append(a)
            a = self._action_queue.popleft()
        throttle, roll_cmd, pitch_cmd, yaw_cmd = a

        # UC-19 damage control-authority impact (AC3). DISABLED: the effective body rate is EXACTLY
        # ``_max_body_rate`` — no integrity read, no new branch value — so a fixed-seed episode is
        # bit-identical to pre-UC-19. ENABLED: the effective ceiling is scaled by
        # ``authority_factor`` of the **start-of-step** integrity (integrity is mutated only by the
        # env's :meth:`damage`/:meth:`repair` hooks, which run *after* this step returns), floored
        # at ``min_authority``. At ``integrity == 1.0`` the factor is exactly 1.0, so the effective
        # rate is ``max(min_authority, _max_body_rate)`` == ``_max_body_rate`` **given the
        # documented invariant ``min_authority < _max_body_rate``** — i.e. the full-integrity
        # enabled path is byte-identical to the disabled path. ONLY ``max_body_rate`` is degraded
        # here; thrust / mass / drag below are untouched, so a damaged drone stays flyable (AC3).
        if self._damage_enabled:
            effective_max_body_rate = max(
                self._damage_cfg.min_authority,
                self._max_body_rate * self._damage_cfg.authority_factor(self._integrity),
            )
        else:
            effective_max_body_rate = self._max_body_rate
        # First-order attitude response to normalised body-rate commands.
        rates = np.array([roll_cmd, pitch_cmd, yaw_cmd], dtype=np.float64) * effective_max_body_rate
        self._angular_velocity = rates
        self._attitude = self._attitude + rates * self._dt
        # Clamp roll/pitch so the thrust projection stays physical; wrap yaw to [-pi, pi].
        self._attitude[0] = float(np.clip(self._attitude[0], -ATTITUDE_LIMIT, ATTITUDE_LIMIT))
        self._attitude[1] = float(np.clip(self._attitude[1], -ATTITUDE_LIMIT, ATTITUDE_LIMIT))
        self._attitude[2] = float((self._attitude[2] + np.pi) % (2.0 * np.pi) - np.pi)

        roll, pitch, _yaw = self._attitude
        # UC-17 battery thrust impact (AC2/AC6). DISABLED: thrust is EXACTLY ``throttle *
        # _max_thrust`` — no battery read, no ``np_random`` draw — so a fixed-seed episode is
        # bit-identical to pre-UC-17 (AC5). ENABLED: the effective ceiling is scaled by
        # ``ceiling_factor`` of the **start-of-step** charge, so ``_max_thrust`` (already
        # ``base * thrust_factor`` from UC-08) times the battery factor makes battery the THIRD
        # multiplicative factor in the documented product order (AC6). Drain is applied at the
        # END of the step (below), so this uses the charge as it was on entry.
        if self._battery_enabled:
            effective_max_thrust = self._max_thrust * self._battery_cfg.ceiling_factor(
                self._battery
            )
        else:
            effective_max_thrust = self._max_thrust
        thrust = throttle * effective_max_thrust
        # thrust_acc = throttle * max_thrust / mass -> mass is a genuine, independent knob
        # (scaling mass alone changes the trajectory; max_thrust is NOT recomputed from mass).
        thrust_acc = thrust / self._mass

        # World-frame acceleration from the tilted collective thrust, minus gravity/drag.
        acc = np.array(
            [
                thrust_acc * np.sin(pitch),  # +pitch tilts nose down/forward -> +x
                -thrust_acc * np.sin(roll),  # +roll banks right -> -y
                thrust_acc * np.cos(roll) * np.cos(pitch) - GRAVITY,
            ],
            dtype=np.float64,
        )
        acc -= self._drag * self._velocity

        self._velocity = self._velocity + acc * self._dt
        speed = float(np.linalg.norm(self._velocity))
        if speed > MAX_SPEED:  # hard clamp -> state stays finite for any policy
            self._velocity *= MAX_SPEED / speed
        self._position = self._position + self._velocity * self._dt

        collided = False
        if self._position[2] <= self._floor_z:
            self._position[2] = self._floor_z
            self._velocity[2] = 0.0
            collided = True
        elif self._position[2] >= self._ceiling_z:
            self._position[2] = self._ceiling_z
            self._velocity[2] = 0.0
            collided = True

        # UC-17 battery drain (AC1). Applied at END of step so the thrust above used the
        # start-of-step charge. Monotone non-increasing, clamped ≥ 0; the throttle-proportional
        # term makes a higher-throttle trajectory drain strictly faster over equal steps. The
        # warm-up hover actions that fill the latency buffer drain here too (single ledger).
        if self._battery_enabled:
            drain = (
                self._battery_cfg.idle_rate + self._battery_cfg.throttle_rate * float(throttle)
            ) * self._dt
            self._battery = max(0.0, self._battery - drain)

        return self._state(collided)
