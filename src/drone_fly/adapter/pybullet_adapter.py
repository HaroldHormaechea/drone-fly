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
from collections import deque
from dataclasses import dataclass

import numpy as np

from drone_fly.adapter.base import DroneAdapter, DroneState, sanitize_action
from drone_fly.adapter.meteor75 import (
    METEOR75_ARM,
    METEOR75_MASS,
    METEOR75_MOTOR_FRACTION,
    METEOR75_TW,
    motor_position_inertia,
)
from drone_fly.adapter.rate_controller import RateController, RateControllerConfig
from drone_fly.adapter.simple import _WARMUP_ACTION, BASE_MASS

# NOTE: the UC-57 timing helpers (``pybullet_freqs`` / ``inner_dt`` / ``iteration_count``) live in
# ``drone_fly.env.timing`` and are imported LAZILY inside the sim-path methods below — never at
# module top — so importing this module (e.g. to unit-test the pure ``run_inner_control_loop`` /
# ``ctbr_to_rpm`` helpers) stays light and never drags the whole ``drone_fly.env`` package (nor
# inverts the adapter-below-env layering at import time). ``env.timing`` itself is a pure leaf.

#: CF2X rigid-body reference constants (UC-48). Documented here so the T/W-preservation
#: logic is a **pure, hermetic** computation that CI can assert without importing pybullet.
#: These remain the **physical URDF baseline** of the pybullet body (the CF2X ``KF`` is what the
#: rotor thrust is actually sized for and is never rewritten). UC-56 reparameterises the *nominal*
#: (mass / inertia / RPM band) to a Meteor75 Pro analog on top of this baseline; the CF2X literals
#: below stay as the pristine reference and the defaults the pre-UC-56 hermetic tests use.
CF2X_NATIVE_MASS = 0.027  # kg — the CF2X body mass its motor thrust (KF) is sized for
CF2X_HOVER_RPM = 14468.429183500699  # per-rotor RPM that hovers the native 0.027 kg body
CF2X_MAX_RPM = 21702.64377525105  # per-rotor RPM at full throttle (peak T/W ≈ 2.25 native)
CF2X_KF = 3.16e-10  # N per rpm^2 — per-rotor thrust coefficient (thrust = KF·rpm^2)
CF2X_GRAVITY = 9.8  # m/s^2 — the aviary's gravity constant

#: Meteor75 Pro-analog nominal RPM band (UC-56). The nominal mass / T/W / arm come from
#: :mod:`drone_fly.adapter.meteor75`; here we derive the per-rotor RPM band that reproduces them
#: on the CF2X body (KF fixed). ``METEOR75_HOVER_RPM`` is the per-rotor RPM whose collective
#: ``4·KF·rpm²`` balances the Meteor75 weight (so throttle 0.5 hovers, exactly as CF2X did);
#: ``METEOR75_MAX_RPM`` = hover · √(T/W) gives the target peak thrust-to-weight. Because the KF is
#: the CF2X value, these absolute RPMs are non-physical for a real Meteor75 ("analog"), but mass,
#: hover-at-0.5 and peak T/W are exact — which is all the dimensionless CTBR interface ever sees.
#: The resolver / summary / guard base off THESE constants, not the live CF2X ``env.HOVER_RPM``.
METEOR75_HOVER_RPM = math.sqrt(METEOR75_MASS * CF2X_GRAVITY / (4.0 * CF2X_KF))
METEOR75_MAX_RPM = METEOR75_HOVER_RPM * math.sqrt(METEOR75_TW)


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
    target_tw: float | None = None,
) -> ResolvedPybulletDynamics:
    """Map a sampled point-mass ``mass`` to a body-consistent, T/W-preserving dynamics set.

    The sampler produces an **absolute** point-mass ``mass`` sized for
    :class:`~drone_fly.adapter.simple.SimpleDroneAdapter` (base ``1.0`` kg), whose thrust is
    ``2·m·g`` so its T/W is preserved by construction. On the sim body the same absolute mass
    (~1 kg) with a native-sized thrust yields T/W ≈ 0.24 → free-fall (the UC-47 root cause).

    When ``tw_preserving`` (default), the sampled mass is reinterpreted as a **body-relative
    multiplier** ``mass_ratio = sampled_mass / base_mass``: the applied body mass becomes
    ``native_mass · mass_ratio`` and the mixer hover RPM is scaled by ``sqrt(mass_ratio)`` so
    hover stays at throttle 0.5. The peak (full-throttle) RPM is set two ways:

    * ``target_tw is None`` (default, the UC-48 CF2X path): ``max_rpm = native_max_rpm · scale``,
      so peak T/W = ``(native_max_rpm / native_hover_rpm)²`` is preserved from the native band —
      byte-identical to pre-UC-56 behaviour (keeps ``test_uc48`` green).
    * ``target_tw`` given (UC-56 wide envelope): ``max_rpm = hover_rpm · √target_tw``, so peak
      T/W = ``target_tw`` **exactly**, independent of mass. This decouples the T/W axis from the
      nominal so the whoop→5"-racer envelope can sweep T/W and mass independently while every
      sample still hovers at 0.5.

    In both cases peak T/W is **invariant under mass at a fixed target** (both hover and max RPM
    scale by the same ``√mass_ratio``), so a randomized-heavier drone stays flyable (AC5).

    When ``tw_preserving`` is ``False`` (opt-out / bug-lock), the sampled mass is applied
    **absolutely** with the **native, unscaled** RPM band — the pre-UC-48 degenerate behavior, kept
    for an explicit opt-out and a regression bug-lock test. ``mass_ratio`` is ``1.0`` so inertia
    stays at the native baseline (``target_tw`` is ignored on this path).
    """
    sampled_mass = float(sampled_mass)
    if tw_preserving:
        mass_ratio = sampled_mass / base_mass
        scale = math.sqrt(mass_ratio)
        hover_rpm = native_hover_rpm * scale
        if target_tw is None:
            max_rpm = native_max_rpm * scale
        else:
            max_rpm = hover_rpm * math.sqrt(float(target_tw))
        return ResolvedPybulletDynamics(
            applied_mass=native_mass * mass_ratio,
            hover_rpm=hover_rpm,
            max_rpm=max_rpm,
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


def mix_to_rpm(
    throttle: float,
    effort_rpy: np.ndarray,
    *,
    hover_rpm: float,
    max_rpm: float,
    thrust_gain: float = 1.0,
    rate_gain: float = 0.15,
) -> np.ndarray:
    """Pure quad-X mixer: throttle → base RPM + rpy **effort** → differential RPM (UC-55).

    Extracted from :func:`ctbr_to_rpm` so both the open-loop mixer and the inner-loop rate
    controller (:mod:`drone_fly.adapter.rate_controller`) share one **pure, stateless** mixing
    convention — the integrator lives only in the controller, keeping this helper (and
    ``ctbr_to_rpm``) stateless (preserves the purity assertions in ``test_adapter.py`` /
    ``test_uc47_thrust_pathway.py``).

    ``throttle`` (index 0 of the CTBR action) sets a base RPM around ``hover_rpm``; ``effort_rpy``
    is a length-3 ``[roll, pitch, yaw]`` control effort in ``[-1, 1]`` — the open-loop command
    itself (:func:`ctbr_to_rpm`) or the rate controller's normalized output — added as a
    differential mix across the four rotors using the standard quad-X sign pattern (motors
    ordered front-right, back-right, back-left, front-left). The result is clipped to
    ``[0, max_rpm]``.

    The throttle→base formula is unchanged byte-for-byte from the original ``ctbr_to_rpm`` so the
    analytic hover-throttle inversion in ``dynamics_summary`` stays in sync.
    """
    base = hover_rpm + (throttle - 0.5) * 2.0 * (max_rpm - hover_rpm) * thrust_gain
    roll, pitch, yaw = np.asarray(effort_rpy, dtype=np.float64).reshape(-1)
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

    This is the **open-loop feedforward** mix: it maps the *command* directly to differential
    RPM with no gyro feedback. UC-55 adds the missing inner-loop rate regulation in
    :class:`~drone_fly.adapter.rate_controller.RateController`, which the pybullet adapter calls
    before delegating the actual mixing to :func:`mix_to_rpm` (the shared pure helper this
    function now also uses). This function itself is kept pure and byte-identical so its unit
    tests and the purity assertions elsewhere continue to hold.
    """
    a = sanitize_action(action)
    return mix_to_rpm(
        a[0],
        a[1:4],
        hover_rpm=hover_rpm,
        max_rpm=max_rpm,
        thrust_gain=thrust_gain,
        rate_gain=rate_gain,
    )


def run_inner_control_loop(
    action: np.ndarray,
    *,
    iterations: int,
    inner_dt: float,
    hover_rpm: float,
    max_rpm: float,
    rate_config: RateControllerConfig,
    rate_controller: RateController,
    read_gyro,
    step_physics,
):
    """Run ``iterations`` inner control ticks for ONE policy action (UC-57 AC2, decoupled rates).

    This is the pure, **hermetically testable** core of the pybullet step: the connectome policy
    decides once (the single ``action``), then the UC-55 body-rate PID + physics advance
    ``iterations`` (== ``physics_ratio``) times at the higher inner rate. It is written against two
    injected callables so the decoupling ratio and the per-tick PID ``dt`` can be asserted with
    fakes — no simulator (the class ``step`` wires the real pybullet reads/steps; this function is
    NOT ``pragma: no cover``):

    * ``read_gyro() -> ndarray`` — the ACHIEVED body rate (rad/s, ``[roll, pitch, yaw]``) sampled
      fresh BEFORE each inner physics tick (the pybullet gyro ``raw[13:16]``). The loop closes on it
      every tick, so the PID regulates at the inner rate.
    * ``step_physics(rpm) -> (DroneState, bool)`` — advance the plant one inner tick with the mixed
      per-rotor ``rpm`` and return the new state plus whether it collided on THIS tick.

    The held CTBR command is mapped to a body-rate setpoint ONCE (it is constant across the inner
    ticks — the policy's decision rate is ``control_hz``); each tick reads the gyro, runs the PID at
    ``inner_dt``, mixes to RPM, and steps physics. **Collision is OR-latched across the ticks and
    the loop breaks on the first collided tick, returning THAT tick's state** — so the env's
    ``(prev_z − curr_z)/dt`` crash-speed proxy and the UC-16 dock / UC-53 crash classifiers stay
    meaningful (they see the contact instant, not a post-contact settled pose). Throttle (index 0)
    passes through untouched into the mixer, exactly as the single-tick UC-55 path did.

    At ``iterations == 1`` and ``inner_dt == dt`` this is byte-identical to the pre-UC-57 single
    physics tick: one gyro read, one PID update at ``dt``, one ``mix_to_rpm``, one physics step.
    """
    a = sanitize_action(action)
    # Command → body-rate setpoint ONCE per policy action (held across the inner ticks): the policy
    # decides at ``control_hz``, only the inner PID/physics run faster.
    setpoint = rate_config.command_to_setpoint(a[1:4], rate_config.max_body_rate)
    state = None
    collided = False
    for _ in range(int(iterations)):
        measured_rate = read_gyro()
        effort = rate_controller.update(setpoint, measured_rate, inner_dt)
        rpm = mix_to_rpm(a[0], effort, hover_rpm=hover_rpm, max_rpm=max_rpm)
        state, tick_collided = step_physics(rpm)
        if tick_collided:
            collided = True
            break
    return state, collided


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
    rate_controller:
        UC-55. Optional :class:`~drone_fly.adapter.rate_controller.RateControllerConfig` for the
        inner-loop body-rate PID (the missing "flight controller"). ``None`` (default) builds a
        default-config controller. The loop regulates the commanded body rate toward the achieved
        rate each :meth:`step` so command noise no longer tumbles the drone; throttle passes
        through untouched. pybullet-only — the simple backend never receives it (CI hermeticity).
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
        rate_controller: RateControllerConfig | None = None,
        physics_ratio: int = 1,
        command_latency_steps: int = 0,
    ) -> None:  # pragma: no cover - requires the sim; verified on the owner's macOS M4
        self._start = np.asarray(start_position, dtype=np.float64).reshape(3).copy()
        self._floor_z = float(floor_z)
        self._ceiling_z = float(ceiling_z)
        self._dt = float(dt)
        # UC-57: integer policy/inner-loop decoupling factor (≥ 1). The rate PID + physics run
        # ``physics_ratio`` inner ticks per policy step (at ``control_hz × physics_ratio``); the
        # CtrlAviary ``ctrl_freq`` is set to that inner rate in ``_build_env``. Default 1 ⇒ one
        # inner tick ⇒ byte-identical to the pre-UC-57 single-step path.
        self._physics_ratio = max(1, int(physics_ratio))
        # UC-57: command-latency FIFO depth in whole steps at the active rate — the env-resolved
        # standing value (``command_latency_ms`` converted; per-episode combined value re-forwarded
        # via ``reconfigure``). Branch A: the FIFO buffers the incoming CTBR action UPSTREAM of the
        # rate loop (real RC→FC ordering). Default 0 ⇒ no buffer ⇒ byte-identical.
        self._latency = int(command_latency_steps)
        self._action_queue: deque[np.ndarray] = deque()
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
        # UC-55: inner-loop body-rate PID. Owns the ONLY integrator/state in the control path
        # (the mixer stays pure/stateless). ``step`` closes the loop on the measured body rate
        # (``raw[13:16]``) so a commanded rate is regulated toward the achieved rate — the missing
        # flight controller that stops command noise from tumbling the drone. pybullet-only.
        self._rate_config = rate_controller or RateControllerConfig()
        self._rate_controller = RateController(self._rate_config)
        self._CtrlAviary, self._DroneModel, self._Physics = load_pybullet_drones()
        self._env = None
        self._hover_rpm = 0.0
        self._max_rpm = 0.0
        # Native CF2X baselines captured ONCE off the fresh body in ``_build_env`` (UC-48), kept as
        # REFERENCE / diagnostics only under UC-56 (the resolved RPM band bases off the Meteor75
        # nominal module constants, not this live read).
        self._native_hover_rpm = 0.0
        self._native_max_rpm = 0.0
        self._pending_dynamics = None

    def reconfigure(
        self, *, start=None, dynamics=None, latency_steps=None
    ) -> None:  # pragma: no cover - sim path
        """Best-effort per-episode reconfiguration on the sim backend (UC-08 AC5).

        ``start`` updates the spawn used at the next ``_build_env`` (forcing a rebuild so
        the new ``initial_xyzs`` takes effect). ``dynamics`` (mass / drag) is applied to the
        drone's rigid body after the env exists. This path is **not asserted** by the
        hermetic suite — the ``SimpleDroneAdapter`` carries the tested reconfigure contract;
        here mass/drag are applied on a best-effort basis and documented as such.

        UC-57: ``latency_steps`` is the env-resolved command-latency FIFO depth in whole steps at
        the active rate (``command_latency_ms`` + any sampled baseline latency, converted in one
        round by :func:`drone_fly.env.timing.resolve_latency_steps`). Set ONLY from this argument —
        this backend, like the numpy one, no longer reads ``dynamics.latency_steps``; ``None``
        leaves the construction-time standing value untouched.
        """
        if start is not None:
            self._start = np.asarray(start, dtype=np.float64).reshape(3).copy()
            if self._env is not None:
                self._env.close()
                self._env = None  # force a rebuild with the new initial_xyzs
        if dynamics is not None:
            self._pending_dynamics = dynamics
            self._apply_dynamics()
        if latency_steps is not None:
            self._latency = int(latency_steps)

    def _apply_dynamics(self) -> None:  # pragma: no cover - sim path
        """Apply the sampled Meteor75-envelope dynamics to the pybullet body (UC-48/UC-56).

        Reads the pybullet-only envelope axes off the pending :class:`DynamicsParams` —
        ``pybullet_mass_ratio`` (mass scale relative to the Meteor75 nominal), ``thrust_to_weight``
        (target peak T/W), ``arm_length`` (quad-X arm coordinate) — and resolves them to a
        body-consistent ``(applied_mass, hover_rpm, max_rpm)`` via
        :func:`resolve_tw_preserving_dynamics`, based off the **Meteor75 nominal module constants**
        (never the live-captured CF2X ``env.HOVER_RPM`` / ``MAX_RPM``, which are reference-only), so
        nothing compounds across resets. Rotational inertia is recomputed from the motor-position
        point-mass model at the applied mass + sampled arm length (AC3), overriding the nominal
        baked in ``_build_env``.

        Early-returns when there is no pending dynamics (randomization off) so the ``_build_env``
        Meteor75 nominal stands untouched (AC2). Lives here (not in ``reconfigure``) so both the
        training env and the UC-47 harness — which sets ``_pending_dynamics`` and calls ``reset``
        directly, bypassing ``reconfigure`` — get the fix. The sim path is not asserted by the
        hermetic suite; the T/W-preservation *logic* is (see ``resolve_tw_preserving_dynamics`` /
        ``thrust_to_weight`` and their tests).
        """
        if self._env is None or self._pending_dynamics is None:
            return
        resolved = resolve_tw_preserving_dynamics(
            self._pending_dynamics.pybullet_mass_ratio,
            tw_preserving=self._tw_preserving,
            native_mass=METEOR75_MASS,
            base_mass=1.0,  # pybullet_mass_ratio is already a ratio (× the Meteor75 nominal mass)
            native_hover_rpm=METEOR75_HOVER_RPM,
            native_max_rpm=METEOR75_MAX_RPM,
            target_tw=self._pending_dynamics.thrust_to_weight,
        )
        # Retarget the mixer's RPM band so commanded thrust tracks the applied mass + target T/W.
        # Derived from the Meteor75 nominal constants every call, so repeated resets never compound.
        self._hover_rpm = resolved.hover_rpm
        self._max_rpm = resolved.max_rpm
        try:
            import pybullet as p

            # Rotational inertia from the motor-position point-mass model at the applied mass +
            # sampled arm length (AC3), so inertia scales correctly across the whole envelope.
            inertia = motor_position_inertia(
                resolved.applied_mass,
                self._pending_dynamics.arm_length,
                METEOR75_MOTOR_FRACTION,
            )
            p.changeDynamics(
                self._env.DRONE_IDS[0],
                -1,
                mass=float(resolved.applied_mass),
                linearDamping=float(self._pending_dynamics.drag),
                localInertiaDiagonal=list(inertia),
                physicsClientId=self._env.CLIENT,
            )
        except Exception:  # noqa: BLE001 - best-effort; documented as not asserted
            pass

    def _build_env(self):  # pragma: no cover - sim path
        # UC-57: the CtrlAviary runs at the INNER rate (``control_hz × physics_ratio``) so each
        # ``env.step`` is one inner tick; the adapter runs ``physics_ratio`` such ticks per policy
        # step. ``pybullet_freqs`` sizes ctrl/pyb so pyb_freq is always an integer multiple of
        # ctrl_freq (fixes the latent ``max(ctrl_freq*4, 240)`` divisibility bug) while staying
        # ≥ 240 and ≥ 4×ctrl. At (dt=0.05, ratio=1) → ctrl 20 / pyb 240 → byte-identical to today.
        from drone_fly.env.timing import hz_from_dt, pybullet_freqs

        ctrl_freq, pyb_freq, _pyb_multiplier = pybullet_freqs(
            hz_from_dt(self._dt), self._physics_ratio
        )
        env = self._CtrlAviary(
            drone_model=self._DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=self._start.reshape(1, 3),
            physics=self._Physics.PYB,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=False,
        )
        # Capture the native CF2X baselines ONCE on the fresh, unmodified body (UC-48) for
        # REFERENCE / diagnostics ONLY. UC-56 makes the resolver + summary + guard base off the
        # Meteor75 nominal module constants instead of this live read, so the mixer band the drone
        # flies under is the Meteor75 nominal — these values are kept purely for provenance.
        self._native_hover_rpm = float(env.HOVER_RPM)
        self._native_max_rpm = float(env.MAX_RPM)
        # UC-56: bake the Meteor75 Pro-analog NOMINAL directly onto the freshly-built body (AC2) —
        # mass, point-mass inertia at the nominal arm, and the mixer RPM band. This is the plant a
        # randomization-OFF run flies. It deliberately does NOT touch ``_pending_dynamics``: when
        # randomization is ON the subsequent ``_apply_dynamics`` overrides this with the sampled
        # envelope; when it is OFF (``_pending_dynamics is None``) ``_apply_dynamics`` no-ops and
        # this nominal stands. (Baking into ``_apply_dynamics`` instead would clobber sampled
        # dynamics every episode — the round-3 challenger clobber-bug fix.)
        self._hover_rpm = METEOR75_HOVER_RPM
        self._max_rpm = METEOR75_MAX_RPM
        try:
            import pybullet as p

            nominal_inertia = motor_position_inertia(
                METEOR75_MASS, METEOR75_ARM, METEOR75_MOTOR_FRACTION
            )
            p.changeDynamics(
                env.DRONE_IDS[0],
                -1,
                mass=float(METEOR75_MASS),
                localInertiaDiagonal=list(nominal_inertia),
                physicsClientId=env.CLIENT,
            )
        except Exception:  # noqa: BLE001 - best-effort; the nominal bake is skipped if unavailable
            pass
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
        self._rate_controller.reset()  # UC-55: clear the inner-loop integrator/prev-error state
        # UC-57: prime the command-latency FIFO with warm-up hover actions so real commands are
        # delayed by exactly ``_latency`` steps (mirrors the SimpleDroneAdapter idiom, shared
        # ``_WARMUP_ACTION`` = exact hover). Empty (and never touched) when latency == 0.
        self._action_queue = deque()
        if self._latency > 0:
            warm = sanitize_action(_WARMUP_ACTION)
            for _ in range(self._latency):
                self._action_queue.append(warm)
        return self._read_state(
            self._start[2] <= self._floor_z or self._start[2] >= self._ceiling_z
        )

    def step(self, action: np.ndarray) -> DroneState:  # pragma: no cover - sim path
        if self._env is None:
            self._env = self._build_env()
        from drone_fly.env.timing import inner_dt, iteration_count

        # UC-57 Branch A: command-latency FIFO UPSTREAM of the rate loop (real RC→FC ordering) —
        # enqueue the fresh command, apply the oldest pending one. Skipped entirely (bit-identical
        # to UC-55) when latency == 0.
        a = sanitize_action(action)
        if self._latency > 0:
            self._action_queue.append(a)
            a = self._action_queue.popleft()

        # UC-57 decoupled inner loop. The policy decided once (``a``); the UC-55 rate PID + physics
        # advance ``physics_ratio`` inner ticks at ``inner_dt = dt / physics_ratio``. Each tick
        # reads the ACHIEVED body rate (``raw[13:16]``) fresh, runs the PID, mixes to RPM, steps one
        # inner physics tick. Collision is OR-latched and the loop breaks on the first collided tick
        # returning THAT tick's state (crash-speed proxy + dock/crash classifiers stay meaningful).
        # At physics_ratio == 1 and inner_dt == dt this is byte-identical to the UC-55 single tick.
        def _read_gyro() -> np.ndarray:
            return np.asarray(self._env._getDroneStateVector(0)[13:16], dtype=np.float64)

        def _step_physics(rpm: np.ndarray) -> tuple[DroneState, bool]:
            self._env.step(rpm.reshape(1, 4))
            tick_state = self._read_state(False)
            z = tick_state.position[2]
            return tick_state, bool(z <= self._floor_z or z >= self._ceiling_z)

        state, collided = run_inner_control_loop(
            a,
            iterations=iteration_count(self._physics_ratio),
            inner_dt=inner_dt(self._dt, self._physics_ratio),
            hover_rpm=self._hover_rpm,
            max_rpm=self._max_rpm,
            rate_config=self._rate_config,
            rate_controller=self._rate_controller,
            read_gyro=_read_gyro,
            step_physics=_step_physics,
        )
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
