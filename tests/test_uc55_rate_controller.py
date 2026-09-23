"""UC-55 — hermetic tests for the inner-loop body-rate controller (the missing flight controller).

These tests are fully hermetic (no pybullet, no GPU, no training — AC11): they exercise the pure
:class:`~drone_fly.adapter.rate_controller.RateController` / :func:`linear_rate_curve` against a
minimal, self-contained rotational plant, plus the pure quad-X mixer
(:func:`~drone_fly.adapter.pybullet_adapter.mix_to_rpm` / :func:`ctbr_to_rpm`) and the config
plumbing. The real per-axis pybullet polarity and closed-loop flight behaviour are the owner's
GPU/sandbox behavioral verdict (AC11), out of hermetic scope.

Hermetic rotational plant
-------------------------
The controller regulates a *rate*, and a rate is the time-integral of an angular acceleration
proportional to the applied control effort. The minimal plant that captures exactly this — and
nothing else — is a pure integrator with no aerodynamic rate damping (the property the real
open-loop mixer lacks and the tumble stems from)::

    angular_accel = effort * PLANT_GAIN        # torque / inertia
    body_rate    += angular_accel * dt         # rate integrates accel (rad/s)
    attitude     += body_rate    * dt          # attitude integrates rate (rad)

The plant sign is self-consistent with the controller sign (positive effort → positive accel →
rate rises toward a positive setpoint), so it validates the *math* of the loop — convergence,
boundedness, rate-not-angle — but deliberately NOT the real pybullet per-axis polarity (that is
inherited verbatim from the mixer convention and checked only in the owner's sandbox; see the
module docstring of ``rate_controller.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.adapter.pybullet_adapter import ctbr_to_rpm, mix_to_rpm
from drone_fly.adapter.rate_controller import (
    RateController,
    RateControllerConfig,
    linear_rate_curve,
)
from drone_fly.adapter.simple import BASE_MAX_BODY_RATE

_DT = 0.05  # the control timestep the default gains are tuned for
_PLANT_GAIN = 10.0  # torque/inertia: effort [-1,1] -> angular accel (rad/s^2)


def _simulate(
    setpoint_fn,
    *,
    steps: int,
    controller: RateController | None,
    dt: float = _DT,
    plant_gain: float = _PLANT_GAIN,
):
    """Run the rotational plant for ``steps`` steps; return (rates, attitudes) as (steps, 3) arrays.

    ``setpoint_fn(n)`` yields the length-3 body-rate setpoint (rad/s) at step ``n``. When
    ``controller`` is given the loop is CLOSED (effort = PID of the rate error); when it is ``None``
    the loop is OPEN — the setpoint's normalized command is fed straight through as effort, exactly
    as the pre-UC-55 open-loop mixer did (no gyro feedback). This is the apples-to-apples contrast
    behind the tumble finding.
    """
    rate = np.zeros(3)
    attitude = np.zeros(3)
    rates = np.zeros((steps, 3))
    attitudes = np.zeros((steps, 3))
    for n in range(steps):
        setpoint = np.asarray(setpoint_fn(n), dtype=np.float64)
        if controller is not None:
            effort = controller.update(setpoint, rate, dt)
        else:
            # Open loop: the normalized command (setpoint / max_body_rate) is dumped as raw effort.
            effort = np.clip(setpoint / BASE_MAX_BODY_RATE, -1.0, 1.0)
        rate = rate + effort * plant_gain * dt
        attitude = attitude + rate * dt
        rates[n] = rate
        attitudes[n] = attitude
    return rates, attitudes


# --------------------------------------------------------------------------- #
# AC1 — rate tracking: the measured rate converges to the commanded setpoint.
# --------------------------------------------------------------------------- #
def test_ac1_rate_converges_to_fixed_setpoint() -> None:
    """AC1: given a fixed roll/pitch/yaw setpoint, the closed-loop measured body rate converges to
    within a small tolerance of the command within a bounded time (hermetic rotational plant)."""
    controller = RateController(RateControllerConfig())
    setpoint = np.array([2.0, -1.5, 1.0])  # rad/s, within the ±BASE_MAX_BODY_RATE clamp
    rates, _ = _simulate(lambda n: setpoint, steps=300, controller=controller)

    # Converged to within a small tolerance of the command by the end of the horizon (every axis).
    assert np.allclose(rates[-1], setpoint, atol=2e-2), f"final rate {rates[-1]} != {setpoint}"
    # And converged reasonably fast — within a bounded time (well inside the horizon).
    converged_step = next(
        n for n in range(rates.shape[0]) if np.all(np.abs(rates[n] - setpoint) < 0.05)
    )
    assert converged_step < 80, f"took {converged_step} steps to converge (expected < 80)"


def test_ac1_zero_setpoint_stays_at_zero() -> None:
    """AC1 corner: a zero setpoint from rest holds zero rate — the loop injects no motion of its
    own (the byte-identity precondition for AC4)."""
    controller = RateController(RateControllerConfig())
    rates, attitudes = _simulate(lambda n: np.zeros(3), steps=50, controller=controller)
    assert np.allclose(rates, 0.0)
    assert np.allclose(attitudes, 0.0)


# --------------------------------------------------------------------------- #
# AC2 — no tumble under sustained full-range noisy commands.
# --------------------------------------------------------------------------- #
def test_ac2_no_tumble_under_noisy_commands_vs_open_loop() -> None:
    """AC2: under sustained full-range noisy rate commands the closed loop keeps the achieved rate
    within the commanded envelope and the tilt bounded well below 180°, whereas the open-loop mixer
    (no feedback) lets the rate — and hence the tilt — diverge and tumble (the pre-change baseline).
    """
    rng = np.random.default_rng(20260923)
    steps = 200  # ≈10 s at dt=0.05
    probe = 40  # ≈2 s — the short probe horizon against the pre-change ~180°/≈1 s tumble baseline
    # A shared full-range noisy command stream (normalized [-1,1] -> setpoint via the linear curve).
    noise = rng.uniform(-1.0, 1.0, size=(steps, 3))
    setpoints = np.array([linear_rate_curve(noise[n], BASE_MAX_BODY_RATE) for n in range(steps)])

    controller = RateController(RateControllerConfig())
    closed_rates, closed_att = _simulate(lambda n: setpoints[n], steps=steps, controller=controller)
    open_rates, open_att = _simulate(lambda n: setpoints[n], steps=steps, controller=None)

    closed_peak_rate = np.abs(closed_rates).max()
    open_peak_rate = np.abs(open_rates).max()
    closed_probe_tilt = np.abs(closed_att[:probe]).max()
    open_probe_tilt = np.abs(open_att[:probe]).max()

    # (1) The closed loop regulates the achieved rate to the commanded envelope (a small overshoot
    # margin over the ±BASE_MAX_BODY_RATE clamp) — it never dumps raw torque into an unbounded spin.
    assert closed_peak_rate <= BASE_MAX_BODY_RATE * 1.5, (
        f"closed-loop peak rate {closed_peak_rate:.2f} escaped the commanded envelope"
    )
    # (2) Within the probe horizon the closed loop does NOT tumble — the peak tilt stays well
    # below a 180° flip (contrast the pre-change baseline that flipped ~180° within ~1 s).
    assert closed_probe_tilt < np.pi, (
        f"closed-loop tilt {closed_probe_tilt:.2f} rad reached a flip in the probe horizon"
    )
    # (3) The open-loop mixer (no feedback) tumbles far worse: with no rate regulation the command
    # noise integrates into a much larger tilt over the same probe window...
    assert open_probe_tilt > 2.0 * closed_probe_tilt, (
        f"open-loop tilt {open_probe_tilt:.2f} not >> closed-loop {closed_probe_tilt:.2f}"
    )
    # (4) ...and its rate is less regulated and its tilt runs away over the full horizon.
    assert open_peak_rate > closed_peak_rate
    assert np.abs(open_att).max() > np.abs(closed_att).max()


# --------------------------------------------------------------------------- #
# AC3 — setpoint clamp: ±1 -> ±max_body_rate; beyond ±1 is clipped.
# --------------------------------------------------------------------------- #
def test_ac3_linear_curve_maps_and_clamps() -> None:
    """AC3: a normalized command of ±1 maps to ±max_body_rate; values beyond ±1 are clipped and
    never exceed the clamp."""
    mbr = 4.0
    # Exact endpoints.
    assert linear_rate_curve(np.array([1.0, -1.0, 0.0]), mbr) == pytest.approx([mbr, -mbr, 0.0])
    # Linear interior.
    assert linear_rate_curve(np.array([0.5, -0.25, 0.1]), mbr) == pytest.approx(
        [0.5 * mbr, -0.25 * mbr, 0.1 * mbr]
    )
    # Beyond ±1 saturates at exactly ±max_body_rate (never past the clamp).
    out = linear_rate_curve(np.array([2.0, -3.0, 1.0]), mbr)
    assert out == pytest.approx([mbr, -mbr, mbr])
    assert np.all(np.abs(out) <= mbr + 1e-12)


def test_ac3_clamp_respects_configured_max_body_rate() -> None:
    """AC3: the clamp tracks the configured ``max_body_rate`` (not a hardcoded constant)."""
    cfg = RateControllerConfig(max_body_rate=2.5)
    out = cfg.command_to_setpoint(np.array([1.0, 5.0, -5.0]), cfg.max_body_rate)
    assert out == pytest.approx([2.5, 2.5, -2.5])


# --------------------------------------------------------------------------- #
# AC4 — throttle untouched: zero command + zero measured rate is byte-identical.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("throttle", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_ac4_zero_command_is_byte_identical_to_open_loop_reference(throttle: float) -> None:
    """AC4: for a pure-throttle command (rates = 0) with zero measured rate the controller effort is
    exactly zero, so the mixer's base-only RPM is byte-identical to the pre-UC-55 open-loop
    reference ``ctbr_to_rpm([throttle, 0, 0, 0])``."""
    hover_rpm, max_rpm = 14468.429, 21702.644
    controller = RateController(RateControllerConfig())

    effort = controller.update(np.zeros(3), np.zeros(3), _DT)
    assert np.array_equal(effort, np.zeros(3)), "zero cmd + zero measured must give zero effort"

    closed = mix_to_rpm(throttle, effort, hover_rpm=hover_rpm, max_rpm=max_rpm)
    reference = ctbr_to_rpm(
        np.array([throttle, 0.0, 0.0, 0.0]), hover_rpm=hover_rpm, max_rpm=max_rpm
    )
    assert np.array_equal(closed, reference), (
        "base-only RPM must be byte-identical to the reference"
    )


# --------------------------------------------------------------------------- #
# AC6 — rate, not angle: a sustained command keeps the attitude integrating.
# --------------------------------------------------------------------------- #
def test_ac6_sustained_command_keeps_attitude_integrating() -> None:
    """AC6: a sustained nonzero rate command keeps the attitude *changing* (the loop holds the rate,
    not an angle) — verifies acro mode, not self-level. The attitude never settles; it grows roughly
    linearly at the commanded rate."""
    controller = RateController(RateControllerConfig())
    setpoint = np.array([1.2, 0.0, 0.0])  # sustained roll rate command (rad/s)
    rates, attitudes = _simulate(lambda n: setpoint, steps=200, controller=controller)

    roll = attitudes[:, 0]
    # The attitude keeps growing (no self-leveling): later > earlier, strictly and substantially.
    assert roll[-1] > roll[100] > roll[40] > 0.1
    # It tracks the commanded rate: over the settled tail, Δattitude ≈ rate * Δt.
    tail = roll[-1] - roll[-21]  # last 20 steps == 1.0 s
    assert tail == pytest.approx(setpoint[0] * (20 * _DT), rel=0.05)
    # The measured rate holds the command (not driven to zero — that would be self-leveling).
    assert rates[-1, 0] == pytest.approx(setpoint[0], abs=1e-2)


# --------------------------------------------------------------------------- #
# AC8 — config-overridable gains (dataclass defaults + YAML plumbing).
# --------------------------------------------------------------------------- #
def test_ac8_documented_defaults() -> None:
    """AC8: the PID gains have documented defaults; ``max_body_rate`` is single-sourced against
    ``BASE_MAX_BODY_RATE`` so the pybullet clamp and the simple-backend scale never diverge."""
    cfg = RateControllerConfig()
    assert (cfg.kp, cfg.ki, cfg.kd) == (0.6, 0.1, 0.02)
    assert cfg.max_body_rate == BASE_MAX_BODY_RATE
    assert cfg.integral_limit == 1.0
    assert cfg.command_to_setpoint is linear_rate_curve


def test_ac8_dataclass_override() -> None:
    """AC8: every gain is overridable via the dataclass without touching controller code."""
    cfg = RateControllerConfig(kp=1.1, ki=0.2, kd=0.05, max_body_rate=3.4, integral_limit=2.0)
    controller = RateController(cfg)
    assert controller.config.kp == 1.1
    assert controller.config.max_body_rate == 3.4
    # A higher kp closes the error faster than the default on the same fixed setpoint.
    sp = np.array([2.0, 0.0, 0.0])
    fast, _ = _simulate(lambda n: sp, steps=200, controller=RateController(cfg))
    slow, _ = _simulate(lambda n: sp, steps=200, controller=RateController(RateControllerConfig()))
    first_fast = next(n for n in range(200) if abs(fast[n, 0] - sp[0]) < 0.05)
    first_slow = next(n for n in range(200) if abs(slow[n, 0] - sp[0]) < 0.05)
    assert first_fast <= first_slow


@pytest.mark.parametrize("run_config", ["train", "evaluate"])
def test_ac8_yaml_gains_thread_into_env_config(run_config: str) -> None:
    """AC8: the ``rate_kp``/``rate_ki``/``rate_kd``/``rate_max_body_rate`` YAML keys thread — for
    BOTH train and evaluate configs — through ``_apply_rate_controller`` into a live
    ``EnvConfig.rate_controller`` (retunable without code edits)."""
    from drone_fly.cli import _apply_rate_controller
    from drone_fly.config import EvaluateRunConfig, TrainRunConfig

    keys = {"rate_kp": 0.9, "rate_ki": 0.3, "rate_kd": 0.07, "rate_max_body_rate": 3.4}
    if run_config == "train":
        cfg = TrainRunConfig.from_mapping({"name": "x", "adapter": "simple", **keys})
    else:
        cfg = EvaluateRunConfig.from_mapping(
            {"name": "x", "adapter": "simple", "checkpoint": "c.zip", **keys}
        )

    env_config = _apply_rate_controller(None, cfg)
    rc = env_config.rate_controller
    assert (rc.kp, rc.ki, rc.kd, rc.max_body_rate) == (0.9, 0.3, 0.07, 3.4)


@pytest.mark.parametrize("run_config", ["train", "evaluate"])
def test_ac8_omitting_gains_leaves_env_config_untouched(run_config: str) -> None:
    """AC8/AC9: a config that sets no rate key leaves the env config untouched (``None`` stays
    ``None``) — set-to-default == omit, and the byte-identical pre-UC-55 default holds."""
    from drone_fly.cli import _apply_rate_controller
    from drone_fly.config import EvaluateRunConfig, TrainRunConfig

    if run_config == "train":
        cfg = TrainRunConfig.from_mapping({"name": "x", "adapter": "simple"})
    else:
        cfg = EvaluateRunConfig.from_mapping(
            {"name": "x", "adapter": "simple", "checkpoint": "c.zip"}
        )
    assert cfg.rate_kp is None and cfg.rate_max_body_rate is None
    assert _apply_rate_controller(None, cfg) is None


@pytest.mark.parametrize("key", ["rate_kp", "rate_ki", "rate_kd", "rate_max_body_rate"])
def test_ac8_negative_gain_rejected(key: str) -> None:
    """AC8: gains are validated non-negative at config-load (a negative gain is meaningless)."""
    from drone_fly.config import ConfigError, TrainRunConfig

    with pytest.raises(ConfigError, match=key):
        TrainRunConfig.from_mapping({"name": "x", "adapter": "simple", key: -0.1})


def test_ac8_env_config_default_rate_controller() -> None:
    """AC8/AC9: a bare ``EnvConfig`` carries a default ``RateControllerConfig`` (append-last, so
    pre-UC-55 construction stays byte-compatible)."""
    from drone_fly.env.config import EnvConfig

    assert EnvConfig().rate_controller == RateControllerConfig()


# --------------------------------------------------------------------------- #
# AC10 — parameterizable curve hook.
# --------------------------------------------------------------------------- #
def test_ac10_curve_hook_is_swappable() -> None:
    """AC10: the command→setpoint mapping is linear by default but routed through a hook a future
    nonlinear (Betaflight/Liftoff-style) rates curve can replace without touching the policy."""
    assert RateControllerConfig().command_to_setpoint is linear_rate_curve

    def cubic_curve(norm: np.ndarray, max_body_rate: float) -> np.ndarray:
        n = np.clip(np.asarray(norm, dtype=np.float64).reshape(-1), -1.0, 1.0)
        return (n**3) * float(max_body_rate)

    cfg = RateControllerConfig(command_to_setpoint=cubic_curve)
    assert cfg.command_to_setpoint is cubic_curve
    # The swapped curve is honoured: half-stick maps to (0.5**3) of the clamp, not 0.5.
    out = cfg.command_to_setpoint(np.array([0.5, 1.0, -1.0]), cfg.max_body_rate)
    assert out == pytest.approx([0.125 * cfg.max_body_rate, cfg.max_body_rate, -cfg.max_body_rate])


# --------------------------------------------------------------------------- #
# Controller mechanics — reset, anti-windup, output clip.
# --------------------------------------------------------------------------- #
def test_reset_clears_integrator_and_prev_error() -> None:
    """``reset()`` (called from the adapter reset) clears the integrator + previous-error state so a
    new episode starts clean — two resets bracketing motion return identical first-step outputs."""
    controller = RateController(RateControllerConfig())
    sp = np.array([3.0, -2.0, 1.0])
    first = controller.update(sp, np.zeros(3), _DT).copy()
    # Drive some state in, then reset and repeat the exact first step.
    for _ in range(10):
        controller.update(sp, np.array([1.0, 1.0, 1.0]), _DT)
    controller.reset()
    again = controller.update(sp, np.zeros(3), _DT)
    assert np.array_equal(first, again)


def test_anti_windup_clamps_integrator() -> None:
    """A sustained saturating error cannot wind the integrator past ``±integral_limit`` (required so
    saturation in the noisy regime does not worsen AC2)."""
    limit = 0.5
    controller = RateController(RateControllerConfig(integral_limit=limit))
    for _ in range(500):
        controller.update(np.array([10.0, 0.0, 0.0]), np.zeros(3), _DT)
    # Access the private integrator state to prove the clamp holds (white-box guard).
    assert np.all(np.abs(controller._integral) <= limit + 1e-12)


def test_output_is_clipped_to_unit_interval() -> None:
    """A large rate error saturates the effort at exactly ``[-1, 1]`` (the mixer's effort envelope),
    in both directions."""
    controller = RateController(RateControllerConfig())
    hi = controller.update(np.array([100.0, 100.0, 100.0]), np.zeros(3), _DT)
    assert np.all(hi <= 1.0) and np.all(hi >= -1.0)
    assert hi == pytest.approx([1.0, 1.0, 1.0])
    controller.reset()
    lo = controller.update(np.array([-100.0, -100.0, -100.0]), np.zeros(3), _DT)
    assert lo == pytest.approx([-1.0, -1.0, -1.0])


def test_zero_dt_does_not_raise_or_nan() -> None:
    """A zero ``dt`` (degenerate step) yields a finite effort — the derivative guard avoids a
    divide-by-zero blowing up the loop."""
    controller = RateController(RateControllerConfig())
    out = controller.update(np.array([1.0, 0.0, 0.0]), np.zeros(3), 0.0)
    assert np.all(np.isfinite(out))
