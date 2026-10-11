"""UC-62 AC5: Liftoff telemetry — UDP datagram parser, recorded-CSV reader, Unity -> contract.

Fixtures are synthetic (user-approved AC5 deviation) but use the exact recorded CSV header;
``tests/data/harness/generate_liftoff_telemetry.py`` regenerates them. No live game, and the UDP
round trip stays on 127.0.0.1.
"""

from __future__ import annotations

import csv
import socket
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.harness import registry
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.geometry import (
    UNITY_TO_CONTRACT,
    quat_to_matrix,
    unity_to_contract_position,
    unity_to_contract_quat,
)
from drone_fly.harness.liftoff.telemetry import (
    CSV_COLUMNS,
    CsvReplayStateSource,
    DiscontinuityDetector,
    GyroMap,
    LiftoffConverter,
    LiftoffCsvTelemetryReader,
    LiftoffUdpStateSource,
    TelemetryFormat,
    converter_from_config,
    read_telemetry_csv,
)
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import TelemetrySample
from tests.harness_support import DATA_DIR, RECORDED_CSV_HEADER

FIXTURE_CSV = DATA_DIR / "liftoff_telemetry.csv"


def _ctx(world_scale: float = 1.0):
    return SimpleNamespace(config=SimpleNamespace(world_scale=world_scale), contract=None)


def _rand_quat_xyzw(rng) -> np.ndarray:
    q = rng.normal(size=4)
    return (q / np.linalg.norm(q)).astype(np.float64)


def _sample(t=1.5, pos=(1, 2, 3), quat=(0, 0, 0, 1), received_at=0.0) -> TelemetrySample:
    return TelemetrySample(
        t=t,
        position=np.array(pos, np.float32),
        quat_xyzw=np.array(quat, np.float32),
        velocity=np.array([0.5, -0.25, 4.0], np.float32),
        gyro=np.array([10.0, -20.0, 30.0], np.float32),
        inputs=np.array([0.1, 0.2, 0.3, 0.4], np.float32),
        battery=np.array([0.9, 16.2], np.float32),
        rpm=np.array([1000, 2000, 3000, 4000], np.float32),
        received_at=received_at,
    )


# ----------------------------------------------------------------------------- fixture + CSV


def test_fixture_has_exact_recorded_header():
    first = FIXTURE_CSV.read_text(encoding="utf-8").splitlines()[0]
    assert first == RECORDED_CSV_HEADER
    assert ",".join(CSV_COLUMNS) == RECORDED_CSV_HEADER


def test_fixture_is_reproducible_from_its_generator(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_tele", DATA_DIR / "generate_liftoff_telemetry.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    with FIXTURE_CSV.open(newline="", encoding="utf-8") as fh:
        committed = [[float(v) for v in row] for row in list(csv.reader(fh))[1:]]
    np.testing.assert_allclose(np.array(committed), np.array(gen.rows()), rtol=1e-7, atol=1e-7)


def test_read_fixture_csv_into_typed_samples():
    samples = read_telemetry_csv(FIXTURE_CSV)
    assert len(samples) == 200
    s = samples[1]
    assert s.t == pytest.approx(0.01)
    for arr, n in (
        (s.position, 3),
        (s.quat_xyzw, 4),
        (s.velocity, 3),
        (s.gyro, 3),
        (s.inputs, 4),
        (s.battery, 2),
        (s.rpm, 4),
    ):
        assert arr.dtype == np.float32 and arr.shape == (n,)
    assert s.received_at == s.t  # no received_at column -> t
    np.testing.assert_allclose(s.rpm, [20000, 20100, 20200, 20300])


def test_csv_missing_columns_and_optional_blocks(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("t,pos_x,pos_y\n0,1,2\n", encoding="utf-8")
    with pytest.raises(HarnessConfigError, match="missing telemetry columns"):
        read_telemetry_csv(bad)
    core = tmp_path / "core.csv"
    core.write_text(
        ",".join(CSV_COLUMNS[:14]) + "\n" + ",".join(["0"] * 7 + ["1"] + ["0"] * 6) + "\n",
        encoding="utf-8",
    )
    (s,) = read_telemetry_csv(core)
    assert s.inputs is None and s.battery is None and s.rpm is None


def test_csv_reader_states_are_converted_and_flag_respawns(tmp_path):
    rows = FIXTURE_CSV.read_text(encoding="utf-8").splitlines()
    header, data = rows[0], rows[1:6]
    jumped = data[3].split(",")
    jumped[1] = str(float(jumped[1]) + 50.0)  # pos_x jump -> respawn
    data[3] = ",".join(jumped)
    p = tmp_path / "respawn.csv"
    p.write_text("\n".join([header, *data]) + "\n", encoding="utf-8")
    reader = LiftoffCsvTelemetryReader(p)
    states = reader.states()
    assert [s.discontinuity for s in states] == [False, False, False, True, True]
    assert len(list(reader)) == 5
    s0 = read_telemetry_csv(p)[0]
    np.testing.assert_allclose(states[0].pos, unity_to_contract_position(s0.position))


# ----------------------------------------------------------------------------- UDP datagram


def test_default_datagram_is_96_bytes_in_documented_order():
    fmt = TelemetryFormat()
    assert fmt.size == 96
    vals = list(range(1, 25))  # t, pos3, quat4, vel3, gyro3, inputs4, battery2, rpm4
    s = fmt.parse(struct.pack("<24f", *vals), received_at=9.0)
    assert s.t == 1.0 and s.received_at == 9.0
    np.testing.assert_array_equal(s.position, [2, 3, 4])
    np.testing.assert_array_equal(s.quat_xyzw, [5, 6, 7, 8])
    np.testing.assert_array_equal(s.velocity, [9, 10, 11])
    np.testing.assert_array_equal(s.gyro, [12, 13, 14])
    np.testing.assert_array_equal(s.inputs, [15, 16, 17, 18])
    np.testing.assert_array_equal(s.battery, [19, 20])
    np.testing.assert_array_equal(s.rpm, [21, 22, 23, 24])


def test_datagram_matching_fixture_row_round_trips():
    fmt = TelemetryFormat()
    for s in read_telemetry_csv(FIXTURE_CSV)[:20]:
        back = fmt.parse(fmt.pack(s))
        for a in ("position", "quat_xyzw", "velocity", "gyro", "inputs", "battery", "rpm"):
            np.testing.assert_array_equal(getattr(back, a), getattr(s, a), err_msg=a)
        assert back.t == pytest.approx(s.t)


def test_datagram_too_short_raises():
    with pytest.raises(ValueError, match="too short"):
        TelemetryFormat().parse(b"\x00" * 40)


def test_custom_stream_format_and_rpm_count_prefix():
    core = TelemetryFormat(blocks=("timestamp", "attitude", "position", "velocity", "gyro"))
    assert core.size == 56
    s = core.parse(struct.pack("<14f", 0.5, 0, 0, 0, 1, 7, 8, 9, 1, 2, 3, 4, 5, 6))
    np.testing.assert_array_equal(s.position, [7, 8, 9])
    np.testing.assert_array_equal(s.quat_xyzw, [0, 0, 0, 1])
    assert s.inputs is None and s.battery is None and s.rpm is None

    pref = TelemetryFormat(rpm_count_prefix=True)
    assert pref.size == 97
    sample = _sample()
    data = pref.pack(sample)
    assert data[80] == 4  # the uint8 motor count before the RPM floats
    np.testing.assert_array_equal(pref.parse(data).rpm, sample.rpm)


def test_format_from_config_and_validation():
    fmt = TelemetryFormat.from_config(
        {
            "stream_format": ["timestamp", "position", "attitude", "velocity", "gyro", "motor_rpm"],
            "block_widths": {"motor_rpm": 6},
        }
    )
    assert fmt.size == 4 * (14 + 6)
    with pytest.raises(HarnessConfigError, match="unknown telemetry blocks"):
        TelemetryFormat(blocks=("timestamp", "position", "attitude", "velocity", "gyro", "magic"))
    with pytest.raises(HarnessConfigError, match="must include"):
        TelemetryFormat(blocks=("timestamp", "position", "attitude", "velocity"))
    with pytest.raises(HarnessConfigError, match="width 3"):
        TelemetryFormat(widths=(("position", 4),))
    with pytest.raises(HarnessConfigError):
        TelemetryFormat.from_config({"block_widths": {"inputs": 0}})


# ----------------------------------------------------------------------------- Unity -> contract


def test_unity_position_mapping():
    np.testing.assert_array_equal(unity_to_contract_position([1, 2, 3]), [3, -1, 2])
    np.testing.assert_array_equal(unity_to_contract_position([0, 1, 0]), [0, 0, 1])  # up -> +Z
    np.testing.assert_array_equal(unity_to_contract_position([0, 0, 1]), [1, 0, 0])  # fwd -> +X
    np.testing.assert_array_equal(unity_to_contract_position([1, 0, 0]), [0, -1, 0])  # right -> -Y
    assert np.linalg.det(UNITY_TO_CONTRACT) == pytest.approx(-1.0)  # a reflection


def test_rotate_then_map_equals_map_then_rotate():
    """Property: M (R_u v) == R_c (M v) for random Unity attitudes and vectors."""
    rng = np.random.default_rng(5)
    m = UNITY_TO_CONTRACT.astype(np.float64)
    for _ in range(50):
        qu = _rand_quat_xyzw(rng)
        r_u = quat_to_matrix([qu[3], qu[0], qu[1], qu[2]])
        r_c = quat_to_matrix(unity_to_contract_quat(qu))
        assert np.linalg.det(r_c) == pytest.approx(1.0, abs=1e-5)
        v = rng.normal(size=3)
        np.testing.assert_allclose(m @ (r_u @ v), r_c @ (m @ v), atol=1e-5)


def test_quaternion_canonical_hemisphere_and_degenerate():
    rng = np.random.default_rng(6)
    for _ in range(20):
        q = unity_to_contract_quat(_rand_quat_xyzw(rng))
        assert q[0] >= 0 and q.dtype == np.float32
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="degenerate"):
        unity_to_contract_quat([0, 0, 0, 0])


def test_unity_yaw_right_turn_points_nose_to_contract_minus_y():
    # Unity +90 deg about +Y (up) turns the nose from +Z to +X (right) -> contract -Y.
    h = np.radians(90) / 2
    q = unity_to_contract_quat([0.0, np.sin(h), 0.0, np.cos(h)])
    np.testing.assert_allclose(quat_to_matrix(q) @ [1, 0, 0], [0, -1, 0], atol=1e-6)


def test_identity_attitude_gives_g_body_down(harness_contract):
    conv = LiftoffConverter()
    st = conv.convert(_sample(quat=(0, 0, 0, 1)))
    np.testing.assert_array_equal(st.quat, [1, 0, 0, 0])
    obs = harness_contract.make_obs(st.as_state_dict(), np.zeros(3, np.float32))
    np.testing.assert_allclose(obs[3:6], [0, 0, -1], atol=1e-7)
    # Nose pitched down 30 deg in Unity (positive rotation about +X_u = right) -> gravity in the
    # body frame gains a forward (+x) component.
    h = np.radians(30) / 2
    st2 = conv.convert(_sample(quat=(np.sin(h), 0, 0, np.cos(h))))
    g = harness_contract.make_obs(st2.as_state_dict(), np.zeros(3, np.float32))[3:6]
    assert g[0] > 0.4 and g[2] < -0.8


def test_converter_scales_lengths_only():
    raw = _sample(pos=(2, 4, 6), quat=(0.1, 0.2, 0.3, 0.927))
    a = LiftoffConverter(GyroMap(), WorldScale(1.0)).convert(raw)
    b = LiftoffConverter(GyroMap(), WorldScale(2.0)).convert(raw, discontinuity=True)
    np.testing.assert_allclose(b.pos, a.pos / 2)
    np.testing.assert_allclose(b.vel, a.vel / 2)
    np.testing.assert_array_equal(b.quat, a.quat)
    np.testing.assert_array_equal(b.omega, a.omega)
    assert b.discontinuity and not a.discontinuity and b.t == raw.t


# ----------------------------------------------------------------------------- gyro map


def test_default_gyro_map_units_axes_and_signs():
    gm = GyroMap()
    pitch, roll, yaw = 10.0, 20.0, 30.0
    out = gm.apply([pitch, roll, yaw])
    np.testing.assert_allclose(out, np.radians([roll, -pitch, -yaw]), rtol=1e-6)
    assert out.dtype == np.float32
    rad = GyroMap(units="rad_s").apply([1.0, 2.0, 3.0])
    np.testing.assert_allclose(rad, [2.0, -1.0, -3.0])


def test_gyro_map_from_config():
    gm = GyroMap.from_config(
        {
            "gyro_units": "rad_s",
            "gyro_map": {
                "wx": {"source": "pitch", "sign": -1},
                "wy": {"source": "roll", "sign": 1},
                "wz": {"source": "yaw", "sign": 1},
            },
        }
    )
    np.testing.assert_allclose(gm.apply([1.0, 2.0, 3.0]), [-1.0, 2.0, 3.0])
    assert GyroMap.from_config({}) == GyroMap()
    with pytest.raises(HarnessConfigError, match="sign"):
        GyroMap.from_config({"gyro_map": {"wx": {"source": "roll", "sign": 2}}})
    with pytest.raises(HarnessConfigError, match="source"):
        GyroMap.from_config({"gyro_map": {"wx": {"source": "throttle", "sign": 1}}})
    with pytest.raises(HarnessConfigError, match="gyro_units"):
        GyroMap.from_config({"gyro_units": "rpm"})


def test_converter_from_packaged_liftoff_config():
    from drone_fly.harness.config import load_config

    cfg = load_config("liftoff")
    conv = converter_from_config(cfg.section("telemetry"), cfg.world_scale)
    assert conv.gyro_map == GyroMap() and conv.scale == WorldScale(2.0)


# ----------------------------------------------------------------------------- discontinuities


def test_discontinuity_detector():
    det = DiscontinuityDetector(max_position_jump_m=5.0)
    assert not det.check(_sample(t=1.0))
    assert not det.check(_sample(t=1.01, pos=(1, 2, 4)))
    assert det.check(_sample(t=0.2, pos=(1, 2, 4)))  # time reset
    assert det.check(_sample(t=0.3, pos=(20, 2, 4)))  # jump
    assert not det.check(_sample(t=0.31, pos=(20, 2, 4)))
    det.reset()
    assert not det.check(_sample(t=0.0))


# ----------------------------------------------------------------------------- state sources


def _wait(cond, timeout=3.0):
    deadline = time.perf_counter() + timeout
    while not cond() and time.perf_counter() < deadline:
        time.sleep(0.005)
    return cond()


def test_udp_round_trip_on_localhost():
    fmt = TelemetryFormat()
    src = LiftoffUdpStateSource(fmt, LiftoffConverter(GyroMap(), WorldScale(2.0)), port=0)
    assert src.latest() is None and src.age(0.0) == float("inf")
    src.start()
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        port = src.bound_port
        assert port
        samples = read_telemetry_csv(FIXTURE_CSV)[:5]
        for s in samples:
            tx.sendto(fmt.pack(s), ("127.0.0.1", port))
        assert _wait(lambda: src.samples_received == 5)
        latest = src.latest()
        expect = LiftoffConverter(GyroMap(), WorldScale(2.0)).convert(samples[-1])
        np.testing.assert_allclose(latest.pos, expect.pos, atol=1e-6)
        np.testing.assert_allclose(latest.quat, expect.quat, atol=1e-6)
        np.testing.assert_allclose(latest.omega, expect.omega, atol=1e-6)
        assert 0 <= src.age(time.perf_counter()) < 2.0
        raw = src.drain_raw()
        assert len(raw) == 5 and src.drain_raw() == []

        tx.sendto(b"\x01\x02\x03", ("127.0.0.1", port))  # garbage
        assert _wait(lambda: src.parse_errors == 1)
        tx.sendto(fmt.pack(samples[0]), ("127.0.0.1", port))  # t goes backwards -> respawn
        assert _wait(lambda: src.samples_received == 6)
        assert src.discontinuity_count == 1 and src.latest().discontinuity
    finally:
        tx.close()
        src.stop()
    assert src.bound_port is None


def test_udp_source_from_config_via_registry():
    cfg = {"port": 0, "host": "127.0.0.1", "gyro_units": "rad_s", "max_position_jump_m": 3.0}
    src = registry.build_part("state_source", "liftoff-udp", cfg, _ctx(1.5))
    assert isinstance(src, LiftoffUdpStateSource)
    assert src.converter.scale == WorldScale(1.5) and src.converter.gyro_map.units == "rad_s"
    with pytest.raises(HarnessConfigError):
        registry.build_part("state_source", "liftoff-udp", {"port": -1}, _ctx())


def test_csv_replay_state_source(tmp_path):
    lines = FIXTURE_CSV.read_text(encoding="utf-8").splitlines()[:6]  # 5 rows over 40 ms
    p = tmp_path / "short.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    src = registry.build_part("state_source", "liftoff-csv", {"replay_csv": str(p)}, _ctx())
    assert isinstance(src, CsvReplayStateSource)
    src.start()
    try:
        assert _wait(lambda: src.samples_received == 5)
        assert src.latest().t == pytest.approx(0.04)
        assert len(src.drain_raw()) == 5
    finally:
        src.stop()
    with pytest.raises(HarnessConfigError, match="replay_csv"):
        registry.build_part("state_source", "liftoff-csv", {}, _ctx())
    empty = tmp_path / "empty.csv"
    empty.write_text(lines[0] + "\n", encoding="utf-8")
    with pytest.raises(HarnessConfigError, match="no rows"):
        CsvReplayStateSource(empty, LiftoffConverter())


def test_fixture_path_is_inside_tests():
    assert Path(FIXTURE_CSV).resolve().is_relative_to(Path(__file__).resolve().parent)
