"""UC-66 AC7 / AC8 / AC9: the fitter recovers a known synthetic plant from a recorded session.

A synthetic plant with known parameters (Betaflight rates incl. expo, first-order rate lag, pure
delays, motor lag, linear + quadratic drag, a non-linear thrust curve) is flown through the real
sequencer and recorder; ``fit_session`` reads the CSVs back and must recover every parameter
within 5 % (an absolute tolerance where the truth is 0) and every delay within one telemetry tick
(10 ms). Fitting only sees the fit passes; the held-out pass validates (AC8).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.sysid.fit import (
    ACCEL_LIMIT_G,
    RATE_LIMIT_PCT,
    _quality,
    align_clock,
    fit_session,
    load_session,
    validate,
)
from drone_fly.sysid.model import AXES, FIVE_INCH_LIKE, MICRO_LIKE, AxisParams
from drone_fly.sysid.schedule import ScheduleSettings
from drone_fly.sysid.synthetic import generate_session

REL = 0.05  # AC9: every parameter within 5 %
TICK = 0.010  # AC9: delays within one telemetry tick (100 Hz)
ZERO_ABS = {"expo": 0.05, "super_rate": 0.05}  # truths of 0: absolute tolerance

#: Micro-like plant with expo on roll / pitch so every Betaflight term is identifiable.
MICRO_EXPO = MICRO_LIKE.with_rates(
    roll=AxisParams(1.2, 0.6, 0.3, 0.04, 0.012),
    pitch=AxisParams(1.2, 0.6, 0.3, 0.04, 0.012),
)


def _fit(tmp_path_factory, params, name, **sched):
    sdir = tmp_path_factory.mktemp(name) / "session"
    result = generate_session(params, sdir, seed=1, schedule_settings=ScheduleSettings(**sched))
    assert result.completed and result.aborted == 0, result.stop_reason
    return sdir, fit_session(sdir, "synthetic-test-micro")


@pytest.fixture(scope="module")
def micro(tmp_path_factory):
    """Default schedule (amplitudes 0.2..1.0, 3 passes, last held out)."""
    sdir, fit = _fit(tmp_path_factory, MICRO_EXPO, "micro")
    return MICRO_EXPO, sdir, fit


@pytest.fixture(scope="module")
def five_inch(tmp_path_factory):
    """The 5-inch-like plant (T/W 7, faster rates); 2 passes to keep the test short."""
    sdir, fit = _fit(tmp_path_factory, FIVE_INCH_LIKE, "five", passes=2)
    return FIVE_INCH_LIKE, sdir, fit


@pytest.fixture(params=["micro", "five_inch"])
def case(request):
    return request.getfixturevalue(request.param)


def _close(got, truth, key=""):
    if truth == 0.0:
        return abs(got) <= ZERO_ABS.get(key, 1e-3)
    return abs(got - truth) <= REL * abs(truth)


# ----------------------------------------------------------------------------- AC9 recovery


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("key", ["rc_rate", "super_rate", "expo"])
def test_betaflight_rates_recovered_within_5_percent(case, axis, key):
    truth, _, fit = case
    got = getattr(fit.params.rates[axis], key)
    want = getattr(truth.rates[axis], key)
    assert _close(got, want, key), f"{axis}.{key}: fitted {got:.4f}, truth {want:.4f}"


@pytest.mark.parametrize("axis", AXES)
def test_rate_time_constant_recovered_within_5_percent(case, axis):
    truth, _, fit = case
    got, want = fit.params.rates[axis].tau_s, truth.rates[axis].tau_s
    assert _close(got, want), f"{axis}.tau_s: fitted {got:.4f}, truth {want:.4f}"


@pytest.mark.parametrize("axis", AXES)
def test_rate_delay_within_one_tick(case, axis):
    truth, _, fit = case
    assert abs(fit.params.rates[axis].delay_s - truth.rates[axis].delay_s) <= TICK


def test_latencies_within_one_tick(case):
    truth, _, fit = case
    assert abs(fit.params.input_s - truth.input_s) <= TICK
    assert abs(fit.params.control_to_gyro_s - truth.control_to_gyro_s) <= TICK
    e = fit.estimates
    assert e["latency.input_s"].value == pytest.approx(fit.params.input_s)
    assert e["latency.control_to_gyro_s"].value == pytest.approx(fit.params.control_to_gyro_s)


def test_thrust_curve_and_hover_within_5_percent(case):
    truth, _, fit = case
    assert fit.params.knots_throttle == truth.knots_throttle
    for k, (got, want) in enumerate(zip(fit.params.knots_g, truth.knots_g, strict=True)):
        assert _close(got, want), f"knot {k}: fitted {got:.3f} g, truth {want:.3f} g"
    assert _close(fit.params.hover_throttle, truth.hover_throttle)
    assert fit.estimates["hover_throttle"].value == pytest.approx(fit.params.hover_throttle)
    # the curve is monotone (invertible) and 0 g at 0
    assert fit.params.knots_g[0] == 0.0
    assert all(b > a for a, b in zip(fit.params.knots_g, fit.params.knots_g[1:], strict=False))


def test_motor_lag_within_5_percent(case):
    truth, _, fit = case
    got, want = fit.params.motor_lag_s, truth.motor_lag_s
    assert _close(got, want), f"motor_lag_s: fitted {got:.4f}, truth {want:.4f}"


def test_drag_within_5_percent(case):
    truth, _, fit = case
    assert _close(fit.params.drag_linear, truth.drag_linear), fit.params.drag_linear
    assert _close(fit.params.drag_quadratic, truth.drag_quadratic), fit.params.drag_quadratic


def test_stick_signs_measured(case):
    truth, _, fit = case
    assert fit.params.signs == truth.signs


def test_every_estimate_has_a_value_and_a_residual(case):
    _, _, fit = case
    keys = {"hover_throttle", "motor_lag_s", "drag.linear_per_s", "drag.quadratic_per_m"}
    keys |= {"latency.input_s", "latency.control_to_gyro_s"}
    keys |= {f"rates.{a}.{k}" for a in AXES for k in ("rc_rate", "super_rate", "expo", "tau_s")}
    keys |= {f"rates.{a}.delay_s" for a in AXES}
    assert keys <= set(fit.estimates)
    for k in keys:
        e = fit.estimates[k]
        assert math.isfinite(e.value), k
        assert e.residual is None or math.isfinite(e.residual), k
    assert len(fit.knot_sigma) == len(fit.params.knots_g)


# ----------------------------------------------------------------------------- AC7 / AC8


def test_held_out_validation_meets_ac8(case):
    _, _, fit = case
    v = fit.validation
    assert fit.fit_quality == "good" and "AC8 met" in fit.verdict
    for a in AXES:
        assert v["windows"][a] >= 1  # every axis validated on the held-out pass
        assert v["rate_1s_pct"][a] <= RATE_LIMIT_PCT
        assert math.isfinite(v["rate_1step_pct"][a])
    assert v["windows"]["thrust"] >= 1
    assert v["accel_1s_g"] <= ACCEL_LIMIT_G and math.isfinite(v["accel_1step_g"])


def test_fit_uses_only_the_fit_passes(micro, monkeypatch):
    """Time split: destroying the held-out pass's commands changes the validation, not the fit."""
    import drone_fly.sysid.fit as fit_mod

    _, sdir, fit = micro
    seen = []
    orig = fit_mod._experiments

    def spy(data, kinds, held_out, axis=None):
        out = orig(data, kinds, held_out, axis)
        seen.append((held_out, {e.held_out for e in out}))
        return out

    monkeypatch.setattr(fit_mod, "_experiments", spy)
    fit_mod.fit_session(sdir, "x")
    for held_out, flags in seen:
        assert flags <= {held_out}
    assert any(h for h, _ in seen) and any(not h for h, _ in seen)


def test_a_wrong_model_is_marked_poor(micro):
    truth, sdir, fit = micro
    wrong = dataclasses.replace(
        fit.params,
        knots_g=tuple(1.4 * g for g in fit.params.knots_g),
        rates={a: dataclasses.replace(r, tau_s=r.tau_s * 6, rc_rate=r.rc_rate * 0.6)
               for a, r in fit.params.rates.items()},
    )  # fmt: skip
    v = validate(load_session(sdir), wrong)
    v.pop("traces")
    assert _quality(v) == "poor"
    assert max(v["rate_1s_pct"].values()) > RATE_LIMIT_PCT and v["accel_1s_g"] > ACCEL_LIMIT_G
    poor = dataclasses.replace(fit, params=wrong, validation=v, fit_quality="poor")
    assert "AC8 NOT MET" in poor.verdict and "fit_quality: poor" in poor.verdict


@pytest.mark.parametrize(
    "v",
    [
        {"rate_1s_pct": {"roll": 16.0, "pitch": 1, "yaw": 1}, "accel_1s_g": 0.01},
        {"rate_1s_pct": {"roll": 1.0, "pitch": 1, "yaw": 1}, "accel_1s_g": 0.16},
        {"rate_1s_pct": {"roll": float("nan"), "pitch": 1, "yaw": 1}, "accel_1s_g": 0.01},
        {"rate_1s_pct": {}, "accel_1s_g": 0.01},
        {"rate_1s_pct": {"roll": 1.0, "pitch": 1, "yaw": 1}, "accel_1s_g": float("nan")},
    ],
)
def test_quality_rule_ac8(v):
    assert _quality(v) == "poor"


def test_quality_rule_good_at_the_limits():
    v = {"rate_1s_pct": {a: RATE_LIMIT_PCT for a in AXES}, "accel_1s_g": ACCEL_LIMIT_G}
    assert _quality(v) == "good"


# ----------------------------------------------------------------------------- pieces


def test_clock_alignment_follows_the_lower_envelope():
    rng = np.random.default_rng(0)
    t = np.arange(0, 20, 0.01) + 12.5
    recv = 100.0 + 1.0001 * (t - 12.5) + rng.uniform(0.0005, 0.004, t.size)
    aligned = align_clock(t, recv)
    truth = 100.0 + 1.0001 * (t - 12.5)
    assert np.max(np.abs(aligned - truth)) < 0.001  # jitter removed (to ~the envelope)


def test_reduced_schedule_still_validates(tmp_path):
    """Plan: a reduced schedule (3 amplitudes, 1 fit + 1 held-out pass) still fits and validates.
    With only 3 amplitudes expo is ill-conditioned: the fitter fixes it at 0 and says so."""
    sdir = tmp_path / "s"
    generate_session(
        MICRO_LIKE, sdir, seed=2,
        schedule_settings=ScheduleSettings(amplitudes=(0.3, 0.7, 1.0), passes=2),
    )  # fmt: skip
    fit = fit_session(sdir, "x")
    assert fit.fit_quality == "good"
    assert _close(fit.params.hover_throttle, MICRO_LIKE.hover_throttle)
    for a in AXES:
        assert abs(fit.params.rates[a].delay_s - MICRO_LIKE.rates[a].delay_s) <= TICK
        if any("expo ill-conditioned" in n and n.startswith(a) for n in fit.notes):
            assert fit.params.rates[a].expo == 0.0


def test_not_a_session_folder(tmp_path):
    with pytest.raises(HarnessConfigError, match="not a sysid session"):
        fit_session(tmp_path, "x")
