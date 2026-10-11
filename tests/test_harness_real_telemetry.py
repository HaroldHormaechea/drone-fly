"""UC-64 AC2: real Liftoff datagrams (committed, trimmed) pin the measured gyro convention.

``tests/data/harness/liftoff_real_datagrams.csv`` is ~4 s of raw 97-byte UDP datagrams from the
hand-flown Infinity Loop capture (see ``extract_real_datagrams.py`` for the source window). They
must pass ``gyro_check`` under the micro/Arena preset (and the 5-inch default, which received the
same fix) and fail under the UC-62 signs (wx = +roll, wy = -pitch, wz = -yaw).
"""

from __future__ import annotations

import pytest

from drone_fly.harness.config import load_config
from drone_fly.harness.liftoff.gyro_check import check_gyro_convention
from drone_fly.harness.liftoff.telemetry import GyroMap, TelemetryFormat
from drone_fly.harness.types import TelemetrySample
from tests.harness_support import DATA_DIR

REAL_CSV = DATA_DIR / "liftoff_real_datagrams.csv"
OLD_UC62_MAP = GyroMap(sources=("roll", "pitch", "yaw"), signs=(1.0, -1.0, -1.0), units="deg_s")


def _rows() -> list[tuple[int, int, bytes]]:
    out = []
    for line in REAL_CSV.read_text(encoding="ascii").splitlines()[1:]:
        t_ns, n, hexdata = line.split(",")
        out.append((int(t_ns), int(n), bytes.fromhex(hexdata)))
    return out


def _samples(telemetry_cfg: dict) -> list[TelemetrySample]:
    fmt = TelemetryFormat.from_config(telemetry_cfg)
    return [fmt.parse(data, received_at=t_ns / 1e9) for t_ns, _, data in _rows()]


@pytest.fixture(scope="module")
def preset_telemetry() -> dict:
    return load_config("liftoff", preset="liftoff_micro_arena").section("telemetry")


def test_fixture_is_a_few_seconds_of_97_byte_datagrams():
    rows = _rows()
    assert len(rows) >= 300
    assert {n for _, n, _ in rows} == {97}
    assert all(len(data) == 97 for _, _, data in rows)
    span_s = (rows[-1][0] - rows[0][0]) / 1e9
    assert 2.0 <= span_s <= 10.0


def test_rpm_count_prefix_is_parsed(preset_telemetry):
    assert preset_telemetry["rpm_count_prefix"] is True
    fmt = TelemetryFormat.from_config(preset_telemetry)
    assert fmt.size == 97
    for _, _, data in _rows()[:20]:
        assert data[20 * 4] == 4  # uint8 motor count after 20 floats, before the 4 RPM floats
        s = fmt.parse(data)
        assert s.rpm is not None and len(s.rpm) == 4 and (s.rpm > 0).all()


def test_without_count_prefix_the_rpm_block_is_misread(preset_telemetry):
    fmt = TelemetryFormat.from_config({**preset_telemetry, "rpm_count_prefix": False})
    assert fmt.size == 96
    _, _, data = _rows()[0]
    good = TelemetryFormat.from_config(preset_telemetry).parse(data)
    try:
        bad = fmt.parse(data)
    except ValueError:
        return  # rejecting the 97-byte datagram is acceptable too
    assert bad.rpm is None or list(bad.rpm) != list(good.rpm)


def test_rotation_on_all_axes(preset_telemetry):
    import numpy as np

    gyro = np.array([s.gyro for s in _samples(preset_telemetry)], np.float64)
    assert (gyro.std(axis=0) > 30.0).all()  # deg/s on pitch, roll and yaw


def test_gyro_check_passes_under_the_preset(preset_telemetry):
    gm = GyroMap.from_config(preset_telemetry)
    assert gm == GyroMap(("roll", "pitch", "yaw"), (-1.0, 1.0, -1.0), "deg_s")
    report = check_gyro_convention(_samples(preset_telemetry), gm)
    assert report.ok is True
    assert [a.axis for a in report.axes] == ["wx", "wy", "wz"]
    for a in report.axes:
        assert abs(a.correlation) >= 0.95
        assert abs(a.scale) == pytest.approx(1.0, abs=0.05)
        assert a.sign_agreement > 0.9
    assert report.suggested_gyro_map() == {
        "gyro_units": "deg_s",
        "gyro_map": {
            "wx": {"source": "roll", "sign": -1.0},
            "wy": {"source": "pitch", "sign": 1.0},
            "wz": {"source": "yaw", "sign": -1.0},
        },
    }


def test_gyro_check_fails_under_the_old_uc62_signs(preset_telemetry):
    report = check_gyro_convention(_samples(preset_telemetry), OLD_UC62_MAP)
    assert report.ok is False
    wx, wy, wz = report.axes
    # roll and pitch are inverted: anti-correlated, scale ~ -1; yaw was already right.
    for a in (wx, wy):
        assert a.correlation < -0.95 and a.scale == pytest.approx(-1.0, abs=0.05)
        assert a.sign_agreement < 0.1
    assert wz.correlation > 0.95


def test_five_inch_default_received_the_same_fix():
    telemetry = load_config("liftoff").section("telemetry")
    assert telemetry["rpm_count_prefix"] is True
    assert telemetry["gyro_units"] == "deg_s"
    assert GyroMap.from_config(telemetry) == GyroMap.from_config(
        load_config("liftoff", preset="liftoff_micro_arena").section("telemetry")
    )
    assert check_gyro_convention(_samples(telemetry), GyroMap.from_config(telemetry)).ok
