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

import math
from dataclasses import dataclass

import numpy as np

from drone_fly.adapter.base import DroneAdapter, DroneState, sanitize_action
from drone_fly.adapter.simple import BASE_MASS

#: CF2X rigid-body reference constants (UC-48). Documented here so the T/W-preservation
#: logic is a **pure, hermetic** computation that CI can assert without importing pybullet.
#: ``_build_env`` reads the live ``env.HOVER_RPM`` / ``env.MAX_RPM`` / inertia off the freshly
#: built CF2X body and feeds *those* into :func:`resolve_tw_preserving_dynamics` (so the sim
#: path never depends on these literals drifting from the installed model); these constants are
#: the values that live read is expected to match, and the defaults the hermetic tests use.
CF2X_NATIVE_MASS = 0.027  # kg — the CF2X body mass its motor thrust (KF) is sized for
CF2X_HOVER_RPM = 14468.429183500699  # per-rotor RPM that hovers the native 0.027 kg body
CF2X_MAX_RPM = 21702.64377525105  # per-rotor RPM at full throttle (peak T/W ≈ 2.25 native)
CF2X_KF = 3.16e-10  # N per rpm^2 — per-rotor thrust coefficient (thrust = KF·rpm^2)
CF2X_GRAVITY = 9.8  # m/s^2 — the aviary's gravity constant


@dataclass(frozen=True)
class ResolvedPybulletDynamics:
    """Result of :func:`resolve_tw_preserving_dynamics` (pure, no pybullet).

    ``applied_mass`` is the mass to hand to ``changeDynamics``; ``hover_rpm`` / ``max_rpm``
    are the mixer RPM band the CTBR mixer should use; ``mass_ratio`` is the scale (relative
    to the native CF2X baseline) applied to mass-dependent quantities such as inertia.
    """

    applied_mass: float
    hover_rpm: float
    max_rpm: float
    mass_ratio: float


def resolve_tw_preserving_dynamics(
    sampled_mass: float,
    *,
    tw_preserving: bool = True,
    native_mass: float = CF2X_NATIVE_MASS,
    base_mass: float = BASE_MASS,
    native_hover_rpm: float = CF2X_HOVER_RPM,
    native_max_rpm: float = CF2X_MAX_RPM,
) -> ResolvedPybulletDynamics:
    """Map a sampled point-mass ``mass`` to a CF2X-consistent, T/W-preserving dynamics set.

    The sampler produces an **absolute** point-mass ``mass`` sized for
    :class:`~drone_fly.adapter.simple.SimpleDroneAdapter` (base ``1.0`` kg), whose thrust is
    ``2·m·g`` so its T/W is preserved by construction. On the CF2X body the same absolute mass
    (~1 kg) with a native-sized thrust yields T/W ≈ 0.24 → free-fall (the UC-47 root cause).

    When ``tw_preserving`` (default), the sampled mass is reinterpreted as a **CF2X-relative
    multiplier** ``mass_ratio = sampled_mass / base_mass``: the applied body mass becomes
    ``native_mass · mass_ratio`` and the mixer RPM band is scaled by ``sqrt(mass_ratio)``.
    Peak T/W = ``(max_rpm / hover_rpm)²`` is therefore **invariant** under the scale (both RPMs
    scale by the same factor), so a randomized-heavier drone keeps the native ~2.25 peak T/W —
    no mixer-*structure* change, only the mass-dependent RPM constants it is fed.

    When ``tw_preserving`` is ``False`` (opt-out / bug-lock), the sampled mass is applied
    **absolutely** with the **native, unscaled** RPM band — today's degenerate behavior, kept
    for an explicit opt-out and a regression bug-lock test. ``mass_ratio`` is ``1.0`` so inertia
    stays at the native baseline (the pre-UC-48 path never rescaled inertia).
    """
    sampled_mass = float(sampled_mass)
    if tw_preserving:
        mass_ratio = sampled_mass / base_mass
        scale = math.sqrt(mass_ratio)
        return ResolvedPybulletDynamics(
            applied_mass=native_mass * mass_ratio,
            hover_rpm=native_hover_rpm * scale,
            max_rpm=native_max_rpm * scale,
            mass_ratio=mass_ratio,
        )
    return ResolvedPybulletDynamics(
        applied_mass=sampled_mass,
        hover_rpm=native_hover_rpm,
        max_rpm=native_max_rpm,
        mass_ratio=1.0,
    )


def thrust_to_weight(
    applied_mass: float,
    rpm: float,
    *,
    kf: float = CF2X_KF,
    g: float = CF2X_GRAVITY,
) -> float:
    """Thrust-to-weight of the 4-rotor CF2X body at a per-rotor ``rpm`` (pure).

    Each of the four rotors produces ``kf·rpm²`` of vertical thrust, so the collective is
    ``4·kf·rpm²``; dividing by weight ``applied_mass·g`` gives the dimensionless T/W. Used by
    the hermetic tests to assert the preservation property without a simulator.
    """
    return 4.0 * float(kf) * float(rpm) ** 2 / (float(applied_mass) * float(g))


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
    tw_preserving:
        UC-48. When ``True`` (default), domain-randomized mass is applied so thrust-to-weight is
        preserved on the CF2X body (see :func:`resolve_tw_preserving_dynamics`); ``False`` restores
        the pre-UC-48 degenerate absolute-mass behavior for opt-out / regression testing.
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
        damage=None,
        tw_preserving: bool = True,
    ) -> None:  # pragma: no cover - requires the sim; verified on the owner's macOS M4
        self._start = np.asarray(start_position, dtype=np.float64).reshape(3).copy()
        self._floor_z = float(floor_z)
        self._ceiling_z = float(ceiling_z)
        self._dt = float(dt)
        # UC-48: when True (default) domain-randomized mass is applied T/W-preservingly on the
        # CF2X body (reinterpreted as a CF2X-relative multiplier, RPM band scaled with it) so a
        # heavier drone keeps a flyable thrust-to-weight; False restores the pre-UC-48 degenerate
        # absolute-mass behavior (~1 kg on a 0.027 kg body → T/W ≈ 0.24 free-fall). See
        # :func:`resolve_tw_preserving_dynamics`.
        self._tw_preserving = bool(tw_preserving)
        # UC-17: accepted for signature parity with make_adapter / SimpleDroneAdapter but not
        # modelled here — the battery drain + thrust-impact model is asserted only on the
        # hermetic numpy backend. This backend always reports a full charge (battery=1.0 default).
        self._battery = battery
        # UC-19: likewise accepted for signature parity but not modelled here — the integrity
        # damage + control-authority model is asserted only on the numpy backend. This backend
        # always reports full integrity (integrity=1.0 default on DroneState).
        self._damage = damage
        self._CtrlAviary, self._DroneModel, self._Physics = load_pybullet_drones()
        self._env = None
        self._hover_rpm = 0.0
        self._max_rpm = 0.0
        # Native CF2X baselines captured ONCE off the fresh body in ``_build_env`` (UC-48), so the
        # resolved RPM band / inertia are always derived from the pristine baseline — never
        # compounded across successive ``_apply_dynamics`` calls.
        self._native_hover_rpm = 0.0
        self._native_max_rpm = 0.0
        self._native_inertia: np.ndarray | None = None
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
        """Apply pending mass/drag to the pybullet body T/W-preservingly (UC-48), best-effort.

        Resolves the sampled point-mass to a CF2X-consistent ``(applied_mass, hover_rpm,
        max_rpm)`` via :func:`resolve_tw_preserving_dynamics`, always from the native baseline
        captured once in ``_build_env`` (no compounding), then applies the mass (and, best-effort,
        a mass-scaled inertia) to the rigid body while retargeting the mixer's RPM band so the
        thrust-to-weight the mixer commands tracks the applied mass. Lives here (not in
        ``reconfigure``) so both the training env and the UC-47 harness — which sets
        ``_pending_dynamics`` and calls ``reset`` directly, bypassing ``reconfigure`` — get the fix.
        The sim path is not asserted by the hermetic suite; the T/W-preservation *logic* is
        (see ``resolve_tw_preserving_dynamics`` / ``thrust_to_weight`` and their tests).
        """
        if self._env is None or self._pending_dynamics is None:
            return
        resolved = resolve_tw_preserving_dynamics(
            self._pending_dynamics.mass,
            tw_preserving=self._tw_preserving,
            native_hover_rpm=self._native_hover_rpm,
            native_max_rpm=self._native_max_rpm,
        )
        # Retarget the mixer's RPM band so commanded thrust scales with the applied mass. Derived
        # from the stored native base every call, so repeated resets never compound the scale.
        self._hover_rpm = resolved.hover_rpm
        self._max_rpm = resolved.max_rpm
        try:
            import pybullet as p

            body_id = self._env.DRONE_IDS[0]
            client = self._env.CLIENT
            kwargs = dict(
                mass=float(resolved.applied_mass),
                linearDamping=float(self._pending_dynamics.drag),
                physicsClientId=client,
            )
            # Best-effort inertia consistency: scale the fixed native diagonal by the same ratio so
            # rotational inertia does not desync from the applied mass (edge case flagged in UC-48).
            if self._native_inertia is not None:
                kwargs["localInertiaDiagonal"] = (
                    self._native_inertia * resolved.mass_ratio
                ).tolist()
            p.changeDynamics(body_id, -1, **kwargs)
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
        # Capture the native CF2X baselines ONCE on the fresh, unmodified body (UC-48). Hover / max
        # RPM come from the drone-model constants the aviary exposes; these are expected to match
        # the module CF2X_HOVER_RPM / CF2X_MAX_RPM literals. ``_apply_dynamics`` derives every
        # subsequent RPM band from these stored baselines, so nothing compounds across resets.
        self._native_hover_rpm = float(env.HOVER_RPM)
        self._native_max_rpm = float(env.MAX_RPM)
        self._hover_rpm = self._native_hover_rpm
        self._max_rpm = self._native_max_rpm
        # Capture the native local inertia diagonal so ``_apply_dynamics`` can scale it with mass
        # from a fixed baseline (best-effort; the sim path is not hermetically asserted).
        self._native_inertia = None
        try:
            import pybullet as p

            info = p.getDynamicsInfo(env.DRONE_IDS[0], -1, physicsClientId=env.CLIENT)
            self._native_inertia = np.asarray(info[2], dtype=np.float64)
        except Exception:  # noqa: BLE001 - best-effort; inertia scaling is skipped if unavailable
            self._native_inertia = None
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
