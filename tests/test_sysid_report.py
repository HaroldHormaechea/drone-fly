"""UC-66 AC2 / AC7 / AC8 / AC10: the fit is written as ``drone.yaml`` + ``sysid.md`` + PNGs.

The profile must load through ``drone_fly.drones.load`` (provenance on every measured leaf), the
default randomization is ``nominal +- max(3 sigma, 10 %)`` clamped at 0, the report holds the
fitted values with uncertainty, the held-out errors and verdict, the plots, and (micro only) the
comparison with UC64's -0.06 stick hover. A short synthetic session keeps this file fast.
"""

from __future__ import annotations

import dataclasses

import pytest

from drone_fly import drones
from drone_fly.sysid.fit import Estimate, fit_session
from drone_fly.sysid.model import AXES, MICRO_LIKE
from drone_fly.sysid.report import (
    UC64_HOVER_STICK,
    _range,
    _scale,
    profile_data,
    sysid_markdown,
    write_report,
)
from drone_fly.sysid.schedule import ScheduleSettings
from drone_fly.sysid.synthetic import generate_session

ID = "liftoff-test-micro"
CAMERA = {"hfov_deg": 120.0, "uptilt_deg": 30.0, "ref": "harness preset 'liftoff' camera"}


@pytest.fixture(scope="module")
def fit(tmp_path_factory):
    sdir = tmp_path_factory.mktemp("rep") / "20261012T100000Z"
    generate_session(
        MICRO_LIKE, sdir, seed=3,
        schedule_settings=ScheduleSettings(amplitudes=(0.3, 0.7, 1.0), passes=2),
        expected_hover=(0.40, 0.56),
    )  # fmt: skip
    return fit_session(sdir, ID)


@pytest.fixture(scope="module")
def written(fit, tmp_path_factory):
    root = tmp_path_factory.mktemp("drones")
    out = root / ID
    yaml_path, md = write_report(
        fit, out, drone_id=ID, name="Test micro",
        liftoff={"drone_name": "Test small drone", "map": "Flat Field"},
        camera=CAMERA, micro=True,
    )  # fmt: skip
    return root, out, yaml_path, md


def test_profile_loads_and_matches_the_fit(fit, written):
    root, out, yaml_path, _ = written
    assert yaml_path == out / "drone.yaml"
    prof = drones.load(ID, root)
    assert prof.source == "liftoff-sysid" and prof.fit_quality == fit.fit_quality == "good"
    assert prof.liftoff["drone_name"] == "Test small drone" and prof.liftoff["map"] == "Flat Field"
    assert prof.hover_throttle == pytest.approx(fit.params.hover_throttle, rel=1e-5)
    assert prof.thrust_curve.accel_g == pytest.approx(fit.params.knots_g, rel=1e-5)
    for a in AXES:
        r, f = prof.rates[a], fit.params.rates[a]
        assert (r.rc_rate, r.tau_s, r.delay_s) == pytest.approx(
            (f.rc_rate, f.tau_s, f.delay_s), rel=1e-5
        )
    assert prof.motor_lag_s == pytest.approx(fit.params.motor_lag_s, rel=1e-5)
    assert prof.drag.linear_per_s == pytest.approx(fit.params.drag_linear, rel=1e-5)
    assert prof.latency.control_to_gyro_s == pytest.approx(fit.params.control_to_gyro_s, rel=1e-5)
    assert prof.camera.hfov_deg == 120.0 and prof.camera.uptilt_deg == 30.0
    # the header says where it came from
    assert yaml_path.read_text(encoding="utf-8").startswith("# Drone profile liftoff-test-micro")


def test_every_measured_leaf_has_session_and_residual(fit, written):
    root = written[0]
    prof = drones.load(ID, root)
    leaves = ["hover_throttle", "thrust_curve", "motor_lag_s", "drag.linear_per_s"]
    leaves += ["drag.quadratic_per_m", "latency.input_s", "latency.control_to_gyro_s"]
    leaves += [f"rates.{a}.{k}" for a in AXES for k in ("rc_rate", "super_rate", "expo")]
    leaves += [f"rates.{a}.{k}" for a in AXES for k in ("tau_s", "delay_s")]
    for leaf in leaves:
        p = prof.provenance(leaf)
        assert p["session"] == fit.session == "20261012T100000Z", leaf
        assert "residual" in p, leaf
    assert prof.provenance("camera.hfov_deg")["ref"].startswith("harness preset")


def test_randomization_is_nominal_plus_minus_max_3sigma_10pct(fit, written):
    prof = drones.load(ID, written[0])
    rnd = prof.randomization
    e = fit.estimates
    for key, est_key in [
        ("motor_lag_s", "motor_lag_s"),
        ("drag.linear_per_s", "drag.linear_per_s"),
        ("drag.quadratic_per_m", "drag.quadratic_per_m"),
        ("latency.control_to_gyro_s", "latency.control_to_gyro_s"),
        *[(f"rates.{a}.tau_s", f"rates.{a}.tau_s") for a in AXES],
        *[(f"rates.{a}.delay_s", f"rates.{a}.delay_s") for a in AXES],
    ]:
        x = e[est_key]
        half = max(3 * (x.sigma or 0.0), 0.1 * abs(x.value))
        assert rnd[key].min == pytest.approx(max(0.0, x.value - half), rel=1e-4, abs=1e-9), key
        assert rnd[key].max == pytest.approx(x.value + half, rel=1e-4), key
    hov = e["hover_throttle"]
    half = max(3 * (hov.sigma or 0.0) / hov.value, 0.1)
    assert rnd["thrust_scale"].min == pytest.approx(1 - half, rel=1e-4)
    assert rnd["thrust_scale"].max == pytest.approx(1 + half, rel=1e-4)
    for a in AXES:
        assert rnd[f"rates.{a}.rate_scale"].min < 1.0 < rnd[f"rates.{a}.rate_scale"].max


@pytest.mark.parametrize(
    ("nominal", "sigma", "lo", "hi"),
    [
        (0.03, 0.0001, 0.027, 0.033),  # 10 % dominates
        (0.03, 0.002, 0.024, 0.036),  # 3 sigma dominates
        (0.01, 0.02, 0.0, 0.07),  # clamped at 0
        (0.0, None, 0.0, 0.0),
    ],
)
def test_range_rule(nominal, sigma, lo, hi):
    r = _range(nominal, sigma)
    assert r["min"] == pytest.approx(lo) and r["max"] == pytest.approx(hi)


def test_scale_rule():
    assert _scale(0.01) == {"min": 0.9, "max": 1.1}
    assert _scale(0.05) == pytest.approx({"min": 0.85, "max": 1.15})
    assert _scale(None) == {"min": 0.9, "max": 1.1}
    assert _scale(0.5)["min"] == 0.0


def test_markdown_sections(fit, written):
    _, out, _, md = written
    text = md.read_text(encoding="utf-8")
    for heading in (
        "# System identification: Test micro (`liftoff-test-micro`)",
        "## Fitted values",
        "### Thrust curve",
        "## Held-out validation (last pass)",
        "## Hover",
        "## Airframe check",
        "## Battery",
        "## Violations and aborted experiments",
        "## Default randomization (UC67)",
        "## Plots",
    ):
        assert heading in text, heading
    assert "| parameter | value | sigma | fit residual |" in text
    for key in fit.estimates:
        assert f"| {key} |" in text, key
    assert f"**fit_quality: {fit.fit_quality}.**" in text and "AC8 met" in text
    assert "one-step error" in text and "1 s open-loop error" in text
    for a in AXES:
        assert f"| {a} rate (% of commanded) |" in text
    assert "| vertical accel (g) |" in text
    assert "Test small drone" in text and "Flat Field" in text
    assert "ok: h0" in text  # the airframe check of the session
    assert "Voltage" in text and "Session stop reason: completed" in text


def test_micro_report_compares_with_uc64(fit, written):
    text = written[3].read_text(encoding="utf-8")
    hov = fit.params.hover_throttle
    assert "UC64 comparison" in text and f"{UC64_HOVER_STICK:+.2f}" in text
    assert f"{2 * hov - 1:+.3f}" in text


def test_non_micro_report_has_no_uc64_line(fit, written):
    data = profile_data(fit, drone_id=ID)
    assert "UC64" not in sysid_markdown(fit, data, [], micro=False)


def test_plots_are_written_next_to_the_report(written):
    _, out, _, md = written
    text = md.read_text(encoding="utf-8")
    pngs = sorted(p.name for p in out.glob("*.png"))
    expected = {f"rates_{a}.png" for a in AXES} | {"thrust_curve.png", "drag.png", "delay.png"}
    assert set(pngs) == expected
    for name in pngs:
        assert (out / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert f"![{name}]({name})" in text


def test_poor_fit_is_written_and_says_not_met(fit, tmp_path):
    v = dict(fit.validation)
    v["rate_1s_pct"] = {a: 40.0 for a in AXES}
    poor = dataclasses.replace(fit, validation=v, fit_quality="poor")
    write_report(poor, tmp_path / ID, drone_id=ID)
    prof = drones.load(ID, tmp_path)
    assert prof.fit_quality == "poor"  # the profile is still written (AC8)
    text = (tmp_path / ID / "sysid.md").read_text(encoding="utf-8")
    assert "**fit_quality: poor.**" in text and "AC8 NOT MET" in text
    assert "| roll rate (% of commanded) |" in text and "40" in text


def test_missing_sigma_still_gives_a_valid_profile(fit, tmp_path):
    est = {k: Estimate(e.value, None, e.residual) for k, e in fit.estimates.items()}
    bare = dataclasses.replace(fit, estimates=est)
    data = profile_data(bare, drone_id=ID)
    assert data["randomization"]["thrust_scale"] == {"min": 0.9, "max": 1.1}
    drones.dump(data, tmp_path / ID / "drone.yaml")
    assert drones.load(ID, tmp_path).source == "liftoff-sysid"
