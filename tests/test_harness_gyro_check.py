"""UC-62 AC5: measuring the gyro axis / sign / unit convention from the quaternion stream."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from drone_fly.harness.liftoff.gyro_check import body_rates_from_quats, check_gyro_convention
from drone_fly.harness.liftoff.telemetry import (
    GyroMap,
    LiftoffConverter,
    LiftoffCsvTelemetryReader,
    read_telemetry_csv,
)
from tests.harness_support import DATA_DIR

FIXTURE_CSV = DATA_DIR / "liftoff_telemetry.csv"


def test_fixture_gyro_matches_its_own_quaternions():
    report = LiftoffCsvTelemetryReader(FIXTURE_CSV).gyro_report()
    assert report.ok and report.n_samples == 200
    assert [a.axis for a in report.axes] == ["wx", "wy", "wz"]
    for a in report.axes:
        assert a.sign_agreement == pytest.approx(1.0)
        assert a.scale == pytest.approx(1.0, abs=1e-3)
        assert a.correlation > 0.999
    suggested = report.suggested_gyro_map()
    assert suggested == {
        "gyro_units": "deg_s",
        "gyro_map": {
            "wx": {"source": "roll", "sign": -1.0},
            "wy": {"source": "pitch", "sign": 1.0},
            "wz": {"source": "yaw", "sign": -1.0},
        },
    }
    d = report.as_dict()
    assert d["ok"] is True and d["suggested"] == suggested


def test_flipped_axis_is_detected_and_correction_suggested():
    samples = read_telemetry_csv(FIXTURE_CSV)
    wrong = GyroMap(signs=(-1.0, -1.0, -1.0))  # wy sign flipped
    report = check_gyro_convention(samples, wrong)
    assert not report.ok
    wy = report.axes[1]
    assert wy.sign_agreement < 0.05 and wy.scale == pytest.approx(-1.0, abs=1e-3)
    assert wy.suggested_source == "pitch" and wy.suggested_sign == 1.0
    assert report.axes[0].sign_agreement == pytest.approx(1.0)


def test_swapped_axes_are_detected():
    samples = read_telemetry_csv(FIXTURE_CSV)
    swapped = GyroMap(sources=("pitch", "roll", "yaw"), signs=(1.0, 1.0, -1.0))
    report = check_gyro_convention(samples, swapped)
    assert not report.ok
    assert report.axes[0].suggested_source == "roll"
    assert report.axes[1].suggested_source == "pitch"


def test_wrong_units_are_detected():
    samples = read_telemetry_csv(FIXTURE_CSV)
    report = check_gyro_convention(samples, GyroMap(units="rad_s"))
    assert not report.ok
    for a in report.axes:
        assert a.scale == pytest.approx(np.pi / 180.0, rel=1e-3)
        assert a.suggested_units == "deg_s"
    # A rad/s recording with a rad/s map is fine.
    rad = [dataclasses.replace(s, gyro=np.radians(s.gyro).astype(np.float32)) for s in samples]
    ok = check_gyro_convention(rad, GyroMap(units="rad_s"))
    assert ok.ok and all(a.suggested_units == "rad_s" for a in ok.axes)


def test_too_few_samples_is_not_ok():
    samples = read_telemetry_csv(FIXTURE_CSV)[:2]
    report = check_gyro_convention(samples, GyroMap())
    assert not report.ok and report.n_samples == 2 and report.axes == []


def test_body_rates_from_constant_yaw_rotation():
    t = np.arange(5) * 0.1
    w = 1.0  # rad/s about +z
    quats = np.array([[np.cos(w * x / 2), 0, 0, np.sin(w * x / 2)] for x in t])
    rates, valid = body_rates_from_quats(quats, t)
    assert valid.all()
    np.testing.assert_allclose(rates, [[0, 0, 1.0]] * 4, atol=1e-9)
    # A repeated timestamp is skipped, not divided by zero.
    rates2, valid2 = body_rates_from_quats(quats[:2], np.array([0.0, 0.0]))
    assert not valid2.any()


def test_body_rates_are_in_the_body_frame():
    # Pitched 90 deg nose-up about body y, then a body-x roll: the rate stays on body x.
    h = np.radians(90) / 2
    q0 = np.array([np.cos(h), 0, np.sin(h), 0])
    dt, w = 0.01, 2.0
    dq = np.array([np.cos(w * dt / 2), np.sin(w * dt / 2), 0, 0])
    from drone_fly.harness.liftoff.gyro_check import _qmul

    rates, _ = body_rates_from_quats(np.array([q0, _qmul(q0, dq)]), np.array([0.0, dt]))
    np.testing.assert_allclose(rates[0], [w, 0, 0], atol=1e-9)


def test_converter_gyro_map_flows_into_reader_report():
    reader = LiftoffCsvTelemetryReader(FIXTURE_CSV, LiftoffConverter(GyroMap(units="rad_s")))
    assert not reader.gyro_report().ok
