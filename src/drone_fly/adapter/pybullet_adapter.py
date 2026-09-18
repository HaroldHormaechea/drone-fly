"""PyBullet-backed adapter over ``gym-pybullet-drones`` (guarded, lazy import).

This is the **dev-time mastery backend**: the ≥80% completion bar (AC10) is defined
against these real quadrotor dynamics, not against
:class:`~drone_fly.adapter.simple.SimpleDroneAdapter`. It is imported lazily and guarded
exactly like the dead AxonWeave hook in :mod:`drone_fly.controller.policy`: importing this
module is always safe, but constructing :class:`PyBulletAdapter` (or calling
:func:`load_pybullet_drones`) raises an **actionable** :class:`ImportError` when the sim is
not installed — never a cryptic mid-run failure.

``gym-pybullet-drones`` is GitHub-only and its default action interface is per-motor RPMs,
so the canonical CTBR action (collective throttle + body rates) is mapped to four motor
RPMs by :func:`ctbr_to_rpm`, which is a **pure** function unit-tested without pybullet
(AC-adapter). See ``scripts/train.sh`` and the README's "Simulator bootstrap" section for
the pinned install and the tested-vs-untested boundary — the pybullet path is verified on
the owner's macOS M4, not in the hermetic Linux CI sandbox.
"""

from __future__ import annotations

import numpy as np

from drone_fly.adapter.base import DroneAdapter, DroneState, sanitize_action

#: Pinned, reproducible install ref (a fixed commit SHA, never floating ``main``). This is
#: gym-pybullet-drones v2.2.0 — the gymnasium-native line matching our gymnasium/SB3 pins.
#: ``scripts/train.sh`` installs exactly this ref. Resolved via ``git ls-remote`` (AC7).
PYBULLET_DRONES_REF = "7ebad1ecabd28a7000add2d05f888aa2e837c2cc"  # v2.2.0
PYBULLET_DRONES_GIT = f"git+https://github.com/utiasDSL/gym-pybullet-drones@{PYBULLET_DRONES_REF}"

_INSTALL_HINT = (
    "PyBullet backend unavailable. gym-pybullet-drones + pybullet are optional, GitHub-only "
    "dev-time dependencies (not installed by CI). Install them with the bootstrap script:\n"
    "    ./scripts/train.sh\n"
    f"or manually (needs a C/C++ toolchain for pybullet):\n"
    f'    uv pip install "pybullet" "{PYBULLET_DRONES_GIT}"\n'
    "Then re-run. To train without the sim, use the pure-numpy backend: adapter='simple'."
)


def ctbr_to_rpm(
    action: np.ndarray,
    *,
    hover_rpm: float,
    max_rpm: float,
    thrust_gain: float = 1.0,
    rate_gain: float = 0.15,
) -> np.ndarray:
    """Map a canonical CTBR action to four quad-X motor RPMs (pure; unit-tested).

    Layout follows ``ACTION_LAYOUT`` ``(THROTTLE, ROLL, PITCH, YAW)``. The collective
    throttle sets a base RPM around ``hover_rpm``; roll/pitch/yaw body-rate commands add a
    differential mix across the four rotors using the standard quad-X sign pattern
    (motors ordered front-right, back-right, back-left, front-left, CCW). The result is
    clipped to ``[0, max_rpm]``.

    This is intentionally a simple, documented mixing rather than a full cascaded rate
    controller: it is enough to make the canonical action steer the pybullet drone, and it
    is testable in isolation. The physical fidelity that matters for the mastery bar comes
    from pybullet's rigid-body integration, not from this mixer.
    """
    a = sanitize_action(action)
    throttle, roll, pitch, yaw = a
    base = hover_rpm + (throttle - 0.5) * 2.0 * (max_rpm - hover_rpm) * thrust_gain
    droll = roll * rate_gain * max_rpm
    dpitch = pitch * rate_gain * max_rpm
    dyaw = yaw * rate_gain * max_rpm
    # Quad-X mixing: +roll -> right rotors down/left up; +pitch -> back up/front down;
    # +yaw -> CW rotors up. Signs per the front-right, back-right, back-left, front-left order.
    rpm = np.array(
        [
            base - droll + dpitch - dyaw,  # front-right (CCW)
            base - droll - dpitch + dyaw,  # back-right (CW)
            base + droll - dpitch - dyaw,  # back-left (CCW)
            base + droll + dpitch + dyaw,  # front-left (CW)
        ],
        dtype=np.float64,
    )
    return np.clip(rpm, 0.0, max_rpm)


def load_pybullet_drones():
    """Guarded import of the pybullet sim; raise an actionable error if absent.

    Returns the ``(CtrlAviary, DroneModel, Physics)`` symbols needed by the adapter. Kept
    as a function (not a top-level import) so importing this module never requires the sim.
    """
    try:  # pragma: no cover - exercised only when the sim is installed (owner's macOS)
        import pybullet  # noqa: F401  (import-time check that the native ext loads)
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
        from gym_pybullet_drones.utils.enums import DroneModel, Physics
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_INSTALL_HINT) from exc
    return CtrlAviary, DroneModel, Physics


class PyBulletAdapter(DroneAdapter):
    """Single-drone ``gym-pybullet-drones`` backend behind the canonical CTBR interface.

    Parameters
    ----------
    start_position:
        World-frame spawn ``[x, y, z]``.
    floor_z, ceiling_z:
        Arena vertical bounds; contact flags a collision and terminates the episode.
    dt:
        Target control timestep (seconds); mapped to the aviary control frequency.
    """

    backend = "pybullet"

    def __init__(
        self,
        start_position: np.ndarray,
        *,
        floor_z: float,
        ceiling_z: float,
        dt: float,
        battery=None,
    ) -> None:  # pragma: no cover - requires the sim; verified on the owner's macOS M4
        self._start = np.asarray(start_position, dtype=np.float64).reshape(3).copy()
        self._floor_z = float(floor_z)
        self._ceiling_z = float(ceiling_z)
        self._dt = float(dt)
        # UC-17: accepted for signature parity with make_adapter / SimpleDroneAdapter but not
        # modelled here — the battery drain + thrust-impact model is asserted only on the
        # hermetic numpy backend. This backend always reports a full charge (battery=1.0 default).
        self._battery = battery
        self._CtrlAviary, self._DroneModel, self._Physics = load_pybullet_drones()
        self._env = None
        self._hover_rpm = 0.0
        self._max_rpm = 0.0
        self._pending_dynamics = None

    def reconfigure(self, *, start=None, dynamics=None) -> None:  # pragma: no cover - sim path
        """Best-effort per-episode reconfiguration on the sim backend (UC-08 AC5).

        ``start`` updates the spawn used at the next ``_build_env`` (forcing a rebuild so
        the new ``initial_xyzs`` takes effect). ``dynamics`` (mass / drag) is applied to the
        drone's rigid body after the env exists. This path is **not asserted** by the
        hermetic suite — the ``SimpleDroneAdapter`` carries the tested reconfigure contract;
        here mass/drag are applied on a best-effort basis and documented as such.
        """
        if start is not None:
            self._start = np.asarray(start, dtype=np.float64).reshape(3).copy()
            if self._env is not None:
                self._env.close()
                self._env = None  # force a rebuild with the new initial_xyzs
        if dynamics is not None:
            self._pending_dynamics = dynamics
            self._apply_dynamics()

    def _apply_dynamics(self) -> None:  # pragma: no cover - sim path
        """Apply pending mass/drag to the pybullet body, best-effort (not asserted)."""
        if self._env is None or self._pending_dynamics is None:
            return
        try:
            import pybullet as p

            body_id = self._env.DRONE_IDS[0]
            client = self._env.CLIENT
            p.changeDynamics(
                body_id,
                -1,
                mass=float(self._pending_dynamics.mass),
                linearDamping=float(self._pending_dynamics.drag),
                physicsClientId=client,
            )
        except Exception:  # noqa: BLE001 - best-effort; documented as not asserted
            pass

    def _build_env(self):  # pragma: no cover - sim path
        ctrl_freq = int(round(1.0 / self._dt))
        env = self._CtrlAviary(
            drone_model=self._DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=self._start.reshape(1, 3),
            physics=self._Physics.PYB,
            pyb_freq=max(ctrl_freq * 4, 240),
            ctrl_freq=ctrl_freq,
            gui=False,
        )
        # Hover / max RPM come from the drone model constants exposed by the aviary.
        self._hover_rpm = float(env.HOVER_RPM)
        self._max_rpm = float(env.MAX_RPM)
        return env

    def _read_state(self, collided: bool) -> DroneState:  # pragma: no cover - sim path
        raw = self._env._getDroneStateVector(0)
        position = np.asarray(raw[0:3], dtype=np.float64)
        attitude = np.asarray(raw[7:10], dtype=np.float64)  # roll, pitch, yaw
        velocity = np.asarray(raw[10:13], dtype=np.float64)
        angular_velocity = np.asarray(raw[13:16], dtype=np.float64)
        return DroneState(
            position=position,
            velocity=velocity,
            attitude=attitude,
            angular_velocity=angular_velocity,
            collided=collided,
        )

    def reset(self, seed: int | None = None) -> DroneState:  # pragma: no cover - sim path
        if self._env is None:
            self._env = self._build_env()
        self._env.reset(seed=seed)
        self._apply_dynamics()  # best-effort mass/drag on the freshly-built body (UC-08)
        return self._read_state(
            self._start[2] <= self._floor_z or self._start[2] >= self._ceiling_z
        )

    def step(self, action: np.ndarray) -> DroneState:  # pragma: no cover - sim path
        if self._env is None:
            self._env = self._build_env()
        rpm = ctbr_to_rpm(action, hover_rpm=self._hover_rpm, max_rpm=self._max_rpm)
        self._env.step(rpm.reshape(1, 4))
        state = self._read_state(False)
        z = state.position[2]
        collided = z <= self._floor_z or z >= self._ceiling_z
        if collided:
            state = DroneState(
                position=state.position,
                velocity=state.velocity,
                attitude=state.attitude,
                angular_velocity=state.angular_velocity,
                collided=True,
            )
        return state

    def close(self) -> None:  # pragma: no cover - sim path
        if self._env is not None:
            self._env.close()
            self._env = None
