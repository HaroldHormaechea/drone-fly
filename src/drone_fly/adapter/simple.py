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

import numpy as np

from drone_fly.adapter.base import DroneAdapter, DroneState, sanitize_action

# --- Fixed dynamics constants (documented; AC11) ---------------------------------------
GRAVITY = 9.81  # m/s^2
MASS = 1.0  # kg (point mass)
MAX_THRUST = 2.0 * MASS * GRAVITY  # N — hover is throttle≈0.5, leaving head/foot room
MAX_BODY_RATE = 4.0  # rad/s at full stick — maps normalised [-1,1] attitude command
ATTITUDE_LIMIT = np.pi / 3.0  # rad — clamp roll/pitch so tilt projection stays sane
LINEAR_DRAG = 0.15  # 1/s — velocity damping coefficient
MAX_SPEED = 25.0  # m/s — hard clamp so a divergent policy can't produce non-finite state


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
    ) -> None:
        self._start = np.asarray(start_position, dtype=np.float64).reshape(3).copy()
        self._floor_z = float(floor_z)
        self._ceiling_z = float(ceiling_z)
        self._dt = float(dt)
        self._rng = np.random.default_rng(0)
        self._position = self._start.copy()
        self._velocity = np.zeros(3, dtype=np.float64)
        self._attitude = np.zeros(3, dtype=np.float64)
        self._angular_velocity = np.zeros(3, dtype=np.float64)

    def _state(self, collided: bool) -> DroneState:
        return DroneState(
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            attitude=self._attitude.copy(),
            angular_velocity=self._angular_velocity.copy(),
            collided=collided,
        )

    def reset(self, seed: int | None = None) -> DroneState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._position = self._start.copy()
        self._velocity = np.zeros(3, dtype=np.float64)
        self._attitude = np.zeros(3, dtype=np.float64)
        self._angular_velocity = np.zeros(3, dtype=np.float64)
        collided = self._position[2] <= self._floor_z or self._position[2] >= self._ceiling_z
        return self._state(collided)

    def step(self, action: np.ndarray) -> DroneState:
        a = sanitize_action(action)
        throttle, roll_cmd, pitch_cmd, yaw_cmd = a

        # First-order attitude response to normalised body-rate commands.
        rates = np.array([roll_cmd, pitch_cmd, yaw_cmd], dtype=np.float64) * MAX_BODY_RATE
        self._angular_velocity = rates
        self._attitude = self._attitude + rates * self._dt
        # Clamp roll/pitch so the thrust projection stays physical; wrap yaw to [-pi, pi].
        self._attitude[0] = float(np.clip(self._attitude[0], -ATTITUDE_LIMIT, ATTITUDE_LIMIT))
        self._attitude[1] = float(np.clip(self._attitude[1], -ATTITUDE_LIMIT, ATTITUDE_LIMIT))
        self._attitude[2] = float((self._attitude[2] + np.pi) % (2.0 * np.pi) - np.pi)

        roll, pitch, _yaw = self._attitude
        thrust = throttle * MAX_THRUST
        thrust_acc = thrust / MASS

        # World-frame acceleration from the tilted collective thrust, minus gravity/drag.
        acc = np.array(
            [
                thrust_acc * np.sin(pitch),  # +pitch tilts nose down/forward -> +x
                -thrust_acc * np.sin(roll),  # +roll banks right -> -y
                thrust_acc * np.cos(roll) * np.cos(pitch) - GRAVITY,
            ],
            dtype=np.float64,
        )
        acc -= LINEAR_DRAG * self._velocity

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

        return self._state(collided)
