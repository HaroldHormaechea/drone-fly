"""Gate detections placed at the capture pose, and Smith-predictor latency compensation.

Covers the pose history of :class:`TrackingGateEstimator` (``_pose_at`` / ``detection_from_pose``,
``gates.pose_offset_s``) and :mod:`drone_fly.harness.latency_comp` (``predict_from``,
``LatencyCompensator``, ``runner.latency_*`` config, runner and session wiring). CPU-only; the
gate tests need no OpenCV (it is imported lazily on the first frame).
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.harness import latency_comp as lc
from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.config import RunnerConfig, load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.geometry import CameraModel, quat_to_matrix
from drone_fly.harness.interfaces import Harness
from drone_fly.harness.latency_comp import LatencyCompensator, predict_from
from drone_fly.harness.liftoff import gate_cv
from drone_fly.harness.liftoff.gate_cv import POSE_STAMP_TOL_S, OpenCvGateEstimator
from drone_fly.harness.runner import HarnessRunner, RunnerSettings, SimClock
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import FlightState, TickRecord
from drone_fly.harness.yaw_align import SpawnYawAligner
from tests.harness_support import (
    FakeFreeSpaceEstimator,
    FakeGateEstimator,
    FakePolicy,
    FakeSink,
    FakeStateSource,
    ListRecorder,
    fake_contract,
)

RED = [([0, 120, 80], [10, 255, 255])]
T_CAM = np.array([0.3, -0.2, 5.0])  # gate centre in the camera frame (z forward)
N_CAM = np.array([0.0, 0.0, -1.0])


def _qz(psi: float) -> np.ndarray:
    return np.array([math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)])


def _state(pos=(0.0, 0.0, 1.0), psi=0.0, received_at=0.0, **kw) -> FlightState:
    return FlightState(
        pos=np.array(pos, np.float32),
        vel=np.array(kw.pop("vel", (0.0, 0.0, 0.0)), np.float32),
        quat=_qz(psi).astype(np.float32),
        omega=np.zeros(3, np.float32),
        received_at=received_at,
        **kw,
    )


def _est(**kw) -> OpenCvGateEstimator:
    return OpenCvGateEstimator(CameraModel(640, 480, 90.0), 1.5, 1.5, RED, WorldScale(1.0), **kw)


def _center(est: OpenCvGateEstimator, state: FlightState) -> np.ndarray:
    d = est.detection_from_pose(T_CAM, N_CAM, np.zeros((4, 2)), 0.1, state, est.camera, 1.5)
    assert d is not None
    return d.center


def _old_center(state: FlightState, camera: CameraModel) -> np.ndarray:
    """The pre-change formula: the current tick's pose."""
    r_wc = quat_to_matrix(state.quat) @ camera.body_to_camera().T
    return np.asarray(state.pos, np.float64) + r_wc @ T_CAM


def _yaw(q) -> float:
    q = np.asarray(q, np.float64)
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


A = _state((0.0, 0.0, 1.0), 0.0, received_at=1.0)
B = _state((1.0, 0.0, 1.0), math.radians(60), received_at=1.2)


def _history(est, *states):
    for s in states:
        est._record_pose(s)
    return est


# ----------------------------------------------------------------------------- pose at capture


def test_detection_uses_the_capture_pose_not_the_current_one():
    est = _history(_est(pose_offset_s=0.0), A, B)
    est._det_stamp = 1.0  # captured when the drone was at A; the tick is at B
    np.testing.assert_allclose(_center(est, B), _old_center(A, est.camera), atol=1e-6)
    d = est.detection_from_pose(T_CAM, N_CAM, np.zeros((4, 2)), 0.1, B, est.camera, 1.5)
    assert d.distance == pytest.approx(np.linalg.norm(T_CAM), abs=1e-6)  # from the capture pos
    est._det_stamp = None  # no stamp: the current pose, as before
    np.testing.assert_allclose(_center(est, B), _old_center(B, est.camera), atol=1e-6)
    assert np.linalg.norm(_old_center(A, est.camera) - _old_center(B, est.camera)) > 1.0


def test_single_entry_history_matches_the_old_behaviour():
    est = _history(_est(pose_offset_s=0.0), A)
    est._det_stamp = 0.5
    np.testing.assert_allclose(_center(est, B), _old_center(B, est.camera), atol=1e-6)
    assert est.pose_fallbacks == 0
    fresh = _est(pose_offset_s=0.0)
    fresh._det_stamp = 1.0
    np.testing.assert_allclose(_center(fresh, B), _old_center(B, fresh.camera), atol=1e-6)


def test_interpolation_midpoint_and_clamp():
    est = _history(_est(pose_offset_s=0.0), A, B)
    cur = _state((9.0, 9.0, 9.0), 2.0, received_at=1.2)
    pos, quat = est._pose_at(1.1, cur)
    np.testing.assert_allclose(pos, [0.5, 0.0, 1.0], atol=1e-6)
    assert _yaw(quat) == pytest.approx(math.radians(30), abs=1e-6)
    assert np.linalg.norm(quat) == pytest.approx(1.0)
    pos, quat = est._pose_at(0.9, cur)  # before the oldest (within tolerance): clamped
    np.testing.assert_allclose(pos, A.pos, atol=1e-6)
    assert _yaw(quat) == pytest.approx(0.0, abs=1e-6)
    pos, quat = est._pose_at(1.3, cur)  # after the newest: clamped
    np.testing.assert_allclose(pos, B.pos, atol=1e-6)
    assert _yaw(quat) == pytest.approx(math.radians(60), abs=1e-6)
    assert est.pose_fallbacks == 0


def test_slerp_takes_the_shortest_arc():
    neg = _state((1.0, 0.0, 1.0), math.radians(60), received_at=1.2)
    neg = FlightState(neg.pos, neg.vel, -neg.quat, neg.omega, received_at=1.2)  # same rotation
    est = _history(_est(pose_offset_s=0.0), A, neg)
    _, quat = est._pose_at(1.1, A)
    assert _yaw(quat) == pytest.approx(math.radians(30), abs=1e-6)


def test_three_entry_history_picks_the_right_segment():
    C = _state((1.0, 2.0, 1.0), math.radians(60), received_at=1.4)
    est = _history(_est(pose_offset_s=0.0), A, B, C)
    pos, quat = est._pose_at(1.3, A)
    np.testing.assert_allclose(pos, [1.0, 1.0, 1.0], atol=1e-6)
    assert _yaw(quat) == pytest.approx(math.radians(60), abs=1e-6)


def test_reset_clears_the_history():
    est = _history(_est(pose_offset_s=0.0), A, B)
    est.reset()
    assert len(est._poses) == 0
    pos, _ = est._pose_at(1.0, B)
    np.testing.assert_allclose(pos, B.pos)


@pytest.mark.parametrize("stamp", [1.0 - POSE_STAMP_TOL_S - 0.01, 1.2 + POSE_STAMP_TOL_S + 0.01])
def test_out_of_tolerance_stamp_falls_back_and_warns_once(caplog, stamp):
    est = _history(_est(pose_offset_s=0.0), A, B)
    cur = _state((5.0, 5.0, 5.0), 1.0, received_at=1.2)
    with caplog.at_level(logging.WARNING, logger=gate_cv.__name__):
        for _ in range(3):
            pos, quat = est._pose_at(stamp, cur)
            np.testing.assert_allclose(pos, cur.pos)
            np.testing.assert_allclose(quat, cur.quat)
    assert est.pose_fallbacks == 3
    assert caplog.text.count("outside the pose history") == 1


def test_pose_offset_shifts_the_lookup():
    est = _history(_est(pose_offset_s=-0.1), A, B)
    pos, _ = est._pose_at(1.2, B)  # looked up at 1.1
    np.testing.assert_allclose(pos, [0.5, 0.0, 1.0], atol=1e-6)
    late = _history(_est(pose_offset_s=0.1), A, B)
    pos, _ = late._pose_at(1.0, B)
    np.testing.assert_allclose(pos, [0.5, 0.0, 1.0], atol=1e-6)


@pytest.mark.parametrize("bad", [0.21, -0.21, 1.0])
def test_pose_offset_is_validated(bad):
    with pytest.raises(HarnessConfigError, match="pose_offset_s"):
        _est(pose_offset_s=bad)
    _est(pose_offset_s=0.2)
    _est(pose_offset_s=-0.2)


def test_update_records_poses_skips_duplicates_and_reports_fallbacks():
    est = _est(pose_offset_s=0.0)
    est.update(None, A, 1.0)
    est.update(None, A, 1.05)  # same received_at: not appended twice
    g = est.update(None, B, 1.2)
    assert [p[0] for p in est._poses] == [1.0, 1.2]
    assert g.info["pose_fallbacks"] == 0


def test_history_is_bounded():
    est = _est(pose_offset_s=0.0)
    for i in range(gate_cv.POSE_HISTORY + 10):
        est._record_pose(_state(received_at=float(i)))
    assert len(est._poses) == gate_cv.POSE_HISTORY


def test_config_default_pose_offset():
    cfg = load_config("liftoff")
    assert cfg.data["gates"]["pose_offset_s"] == pytest.approx(-0.025)
    assert gate_cv.DEFAULT_POSE_OFFSET_S == pytest.approx(-0.025)
    assert _est().pose_offset_s == 0.0  # constructor default unchanged


# ----------------------------------------------------------------------------- predict_from


def _toy_plant(log=None):
    """A deterministic numpy plant: a[0] accelerates up, a[1:4] set the body rates (yaw spins)."""

    def step(s, a):
        a = np.asarray(a, np.float32)
        if log is not None:
            log.append(float(a[0]))
        dt = np.float32(0.02)
        vel = np.asarray(s["vel"], np.float32) + np.array([0, 0, a[0]], np.float32) * dt
        omega = a[1:4].astype(np.float32)
        q = np.asarray(s["quat"], np.float64)
        dq = _qz(float(omega[2]) * 0.02)
        w0, x0, y0, z0 = q
        w1, x1, y1, z1 = dq
        quat = np.array([
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ], np.float32)  # fmt: skip
        return {
            "pos": np.asarray(s["pos"], np.float32) + vel * dt,
            "vel": vel,
            "quat": quat,
            "omega": omega,
        }

    return step


def _sd(vel=(0.0, 0.0, 0.0)):
    return _state(vel=vel).as_state_dict()


def _cmd(v):
    return np.array([v, 0.0, 0.0, 1.0], np.float32)


def test_no_commands_is_identity():
    s = _sd()
    out, n = predict_from(s, 1.0, 1.05, [], 0.04, 0.02, _toy_plant())
    assert out is s and n == 0


def test_zero_horizon_is_identity():
    s = _sd()
    out, n = predict_from(s, 1.0, 1.0, [(0.9, _cmd(1))], 0.0, 0.02, _toy_plant())
    assert out is s and n == 0
    out, n = predict_from(s, 1.0, 1.0, [(0.9, _cmd(1))], 0.009, 0.02, _toy_plant())
    assert out is s and n == 0


def test_horizon_steps_and_cap():
    cmds = [(0.0, _cmd(1))]
    assert predict_from(_sd(), 10.0, 10.03, cmds, 0.05, 0.02, _toy_plant())[1] == 4
    assert predict_from(_sd(), 10.0, 11.0, cmds, 0.05, 0.02, _toy_plant())[1] == 10  # 0.2 s cap
    assert (
        predict_from(_sd(), 10.0, 11.0, cmds, 0.05, 0.02, _toy_plant(), max_horizon_s=0.1)[1] == 5
    )


def test_command_selection_per_step():
    log: list[float] = []
    cmds = [(0.95, _cmd(1)), (0.99, _cmd(2)), (1.01, _cmd(3))]
    # r=1.0, D=0.04: thresholds 0.96, 0.98, 1.00, 1.02 -> 1, 1, 2, 3
    _, n = predict_from(_sd(), 1.0, 1.04, cmds, 0.04, 0.02, _toy_plant(log))
    assert n == 4 and log == [1.0, 1.0, 2.0, 3.0]


def test_pre_history_command_and_skipped_steps():
    log: list[float] = []
    cmds = [(0.99, _cmd(2))]
    _, n = predict_from(_sd(), 1.0, 1.04, cmds, 0.04, 0.02, _toy_plant(log), pre=_cmd(7))
    assert n == 4 and log == [7.0, 7.0, 2.0, 2.0]
    log.clear()
    _, n = predict_from(_sd(), 1.0, 1.04, cmds, 0.04, 0.02, _toy_plant(log))
    assert n == 2 and log == [2.0, 2.0]  # nothing flew the drone before the first command


def test_full_prediction_matches_a_manual_roll():
    plant = _toy_plant()
    s0 = _sd(vel=(1.0, 0.0, 0.0))
    out, n = predict_from(s0, 1.0, 1.02, [(0.5, _cmd(3))], 0.04, 0.02, plant)
    manual = s0
    for _ in range(3):
        manual = plant(manual, _cmd(3))
    assert n == 3
    for k in ("pos", "vel", "quat", "omega"):
        np.testing.assert_allclose(out[k], manual[k], atol=1e-6)


def test_rotation_prediction_keeps_measured_velocity():
    plant = _toy_plant()
    s0 = _sd(vel=(1.0, 0.0, 0.5))
    out, n = predict_from(s0, 1.0, 1.02, [(0.5, _cmd(3))], 0.04, 0.02, plant, predict="rotation")
    full, _ = predict_from(s0, 1.0, 1.02, [(0.5, _cmd(3))], 0.04, 0.02, plant)
    assert n == 3
    np.testing.assert_allclose(out["vel"], s0["vel"])
    np.testing.assert_allclose(out["pos"], s0["pos"] + s0["vel"] * 3 * 0.02, atol=1e-6)
    np.testing.assert_allclose(out["quat"], full["quat"])
    np.testing.assert_allclose(out["omega"], full["omega"])
    assert not np.allclose(full["vel"], s0["vel"])


# ----------------------------------------------------------------------------- LatencyCompensator


def test_compensator_validates_mode_and_predict():
    with pytest.raises(HarnessConfigError, match="latency_compensation"):
        LatencyCompensator("maybe")
    with pytest.raises(HarnessConfigError, match="latency_predict"):
        LatencyCompensator("on", predict="partial")


def test_enable_resolution(monkeypatch):
    assert LatencyCompensator("on", eligible=False).enabled
    assert not LatencyCompensator("off").enabled
    assert not LatencyCompensator("auto", eligible=True).enabled  # AUTO_LIVE is False (V1)
    assert lc.AUTO_LIVE is False
    monkeypatch.setattr(lc, "AUTO_LIVE", True)
    assert LatencyCompensator("auto", eligible=True).enabled
    assert not LatencyCompensator("auto", eligible=False).enabled


def test_resolve_v2_refuses_on_and_turns_auto_off(monkeypatch, caplog):
    with pytest.raises(HarnessConfigError, match="v1"):
        LatencyCompensator("on", plant_step=_toy_plant()).resolve(2)
    monkeypatch.setattr(lc, "AUTO_LIVE", True)
    comp = LatencyCompensator("auto", plant_step=_toy_plant())
    with caplog.at_level(logging.WARNING, logger=lc.__name__):
        assert comp.resolve(2) is False
    assert not comp.enabled and "action-spec v2" in caplog.text
    assert LatencyCompensator("on", plant_step=_toy_plant()).resolve(1) is True
    assert LatencyCompensator("on", plant_step=_toy_plant()).resolve(None) is True
    assert LatencyCompensator("off").resolve(2) is False


def test_resolve_needs_plant_step():
    with pytest.raises(HarnessConfigError, match="plant_step"):
        LatencyCompensator("on").resolve(1)
    assert LatencyCompensator("off").resolve(1) is False


def test_record_history_cap_and_pre():
    comp = LatencyCompensator("on", plant_step=_toy_plant())
    for i in range(lc.HISTORY + 2):
        comp.record(float(i), SimpleNamespace(raw=_cmd(i)))  # a command object with .raw
    assert len(comp._commands) == lc.HISTORY
    assert comp._commands[0][0] == 2.0
    np.testing.assert_array_equal(comp._pre, _cmd(1))
    comp.reset()
    assert not comp._commands and comp._pre is None
    off = LatencyCompensator("off")
    off.record(0.0, _cmd(1))
    assert not off._commands


def test_predict_returns_a_flight_state_preserving_metadata():
    comp = LatencyCompensator("on", 0.04, plant_step=_toy_plant(), predict="full")
    s = _state(received_at=1.0, t=3.5, battery=(0.8, 15.9))
    assert comp.predict(s, 1.02) is s and comp.last_steps == 0  # nothing recorded yet
    comp.record(0.5, _cmd(3))
    out = comp.predict(s, 1.02)
    assert comp.last_steps == 3 and isinstance(out, FlightState)
    assert out.battery == (0.8, 15.9) and out.t == 3.5 and out.received_at == 1.0
    assert out.vel[2] > 0 and _yaw(out.quat) > 0
    assert out.pos.dtype == np.float32
    np.testing.assert_array_equal(s.vel, 0.0)  # the measured state is untouched
    off = LatencyCompensator("off", plant_step=_toy_plant())
    assert off.predict(s, 1.02) is s


# ----------------------------------------------------------------------------- config


def _runner_dict(**kw):
    return {**load_config("sim-loopback").data["runner"], **kw}


def test_config_defaults():
    d = _runner_dict()
    for k in ("latency_compensation", "latency_delay_s", "latency_predict"):
        d.pop(k, None)
    rc = RunnerConfig.from_dict(d)
    assert (rc.latency_compensation, rc.latency_delay_s, rc.latency_predict) == (
        "auto", pytest.approx(lc.DEFAULT_DELAY_S), "rotation"
    )  # fmt: skip
    for harness in ("sim-loopback", "liftoff"):
        r = load_config(harness).runner
        assert r.latency_compensation == "auto" and r.latency_predict == "rotation"
        assert 0.0 <= r.latency_delay_s <= 0.3


@pytest.mark.parametrize(("raw", "mode"), [(False, "off"), (True, "on"), ("off", "off")])
def test_config_yaml_booleans_map_to_off_on(raw, mode):
    assert (
        RunnerConfig.from_dict(_runner_dict(latency_compensation=raw)).latency_compensation == mode
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("latency_compensation", "yes"),
        ("latency_compensation", "AUTO"),
        ("latency_delay_s", -0.01),
        ("latency_delay_s", 0.31),
        ("latency_predict", "translation"),
    ],
)
def test_config_rejects_bad_values(key, value):
    with pytest.raises(HarnessConfigError, match=f"runner.{key}"):
        RunnerConfig.from_dict(_runner_dict(**{key: value}))


# ----------------------------------------------------------------------------- runner wiring


class _SpyAssembler(ContractObservationAssembler):
    def __init__(self, contract):
        super().__init__(contract)
        self.states: list[FlightState] = []

    def assemble(self, state, gate, fs, extra_dim):
        self.states.append(state)
        return super().assemble(state, gate, fs, extra_dim)


class _SpyGate(FakeGateEstimator):
    def __init__(self):
        super().__init__()
        self.states: list[FlightState] = []

    def update(self, frame, state, now):
        self.states.append(state)
        return super().update(frame, state, now)


class _SpyAligner(SpawnYawAligner):
    def __init__(self):
        super().__init__()
        self.captured_states: list[FlightState] = []

    def capture(self, state):
        self.captured_states.append(state)
        return super().capture(state)


def _lat_runner(comp, policy=None, max_ticks=4, aligner=None):
    contract = fake_contract()
    st = FakeStateSource(_state((0.0, 0.0, 1.0), received_at=0.0))
    gate = _SpyGate()
    sink = FakeSink()
    asm = _SpyAssembler(contract)
    h = Harness(
        name="fake",
        state_source=st,
        gate_estimator=gate,
        action_sink=sink,
        free_space_estimator=FakeFreeSpaceEstimator(),
        clock=SimClock(),
    )
    rec = ListRecorder()
    r = HarnessRunner(
        h,
        policy or FakePolicy(action=(0.5, 0.0, 0.0, 1.0)),
        asm,
        contract,
        RunnerSettings(max_ticks=max_ticks),
        recorder=rec,
        yaw_aligner=aligner,
        latency_comp=comp,
    )
    return SimpleNamespace(runner=r, state=st, gate=gate, asm=asm, sink=sink, rec=rec)


def test_runner_feeds_the_forecast_to_the_assembler_only():
    comp = LatencyCompensator("on", 0.04, plant_step=_toy_plant(), predict="full")
    al = _SpyAligner()
    t = _lat_runner(comp, aligner=al)
    t.runner.run()
    measured = t.state.state
    assert all(s is measured for s in t.gate.states)  # perception: measured
    assert all(s is measured for s in al.captured_states)  # spawn-yaw capture: measured
    assert t.asm.states[0] is measured  # no command sent yet: nothing to roll
    for s in t.asm.states[1:]:
        assert s is not measured and s.vel[2] > 0 and _yaw(s.quat) > 0
    ticks = t.rec.ticks
    assert ticks[0].predicted_state is None and "predicted_state" not in ticks[0].as_dict()
    for rec in ticks[1:]:
        assert rec.predicted_state is not None and "predicted_state" in rec.as_dict()
        assert rec.state["vel"][2] == 0.0  # the recorded state stays measured
    events = [e for rec in ticks for e in rec.events if e.startswith("latency_comp:")]
    assert len(events) == 1 and events[0] == f"latency_comp:{events[0][13:-5]}steps"  # once
    assert int(events[0][13:-5]) >= 1
    assert len(comp._commands) == len(t.sink.sent)


def test_runner_without_or_disabled_compensator_is_unchanged():
    base = _lat_runner(None)
    base.runner.run()
    off = _lat_runner(LatencyCompensator("off", plant_step=_toy_plant()))
    off.runner.run()
    for a, b in zip(base.rec.ticks, off.rec.ticks, strict=True):
        np.testing.assert_array_equal(a.obs, b.obs)
        assert b.predicted_state is None
    assert all(s is off.state.state for s in off.asm.states)


def test_runner_resets_the_compensator_on_discontinuity_and_resolves_on_load():
    comp = LatencyCompensator("on", plant_step=_toy_plant())
    resets: list[int] = []
    orig = comp.reset

    def spy_reset():
        resets.append(len(comp._commands))
        orig()

    comp.reset = spy_reset  # type: ignore[method-assign]
    pol = FakePolicy(action=(0.5, 0.0, 0.0, 1.0))
    pol.contract_version = 2  # type: ignore[attr-defined]
    t = _lat_runner(comp, policy=pol)
    with pytest.raises(HarnessConfigError, match="v1"):
        t.runner.prepare(arm=False)

    class _Disc(FakeStateSource):
        def latest(self):
            self.calls = getattr(self, "calls", 0) + 1
            if self.calls == 3:
                self.discontinuity_count += 1
            return self.state

    t2 = _lat_runner(comp, max_ticks=5)
    disc = _Disc(t2.state.state)
    t2.runner.harness.state_source = disc
    resets.clear()
    t2.runner.run()
    assert len(resets) >= 2  # run start + the discontinuity
    assert resets[0] == 0 and max(resets[1:]) > 0  # the respawn dropped in-flight commands
    assert any("discontinuity:estimators_reset" in r.events for r in t2.rec.ticks)


def test_tick_record_omits_predicted_state_when_none():
    assert "predicted_state" not in TickRecord(tick=0, wall_time=0.0).as_dict()


# ----------------------------------------------------------------------------- session wiring


@pytest.fixture
def captured_comp(monkeypatch):
    from drone_fly.harness import session

    seen: list = []

    class Spy(HarnessRunner):
        def __init__(self, *a, **kw):
            seen.append(kw.get("latency_comp"))
            super().__init__(*a, **kw)

    monkeypatch.setattr(session, "HarnessRunner", Spy)
    return seen


@pytest.mark.parametrize(("mode", "enabled"), [("auto", False), ("off", False), ("on", True)])
def test_session_contract_loopback(
    harness_checkpoint_30, harness_k1_dir, tmp_path, captured_comp, mode, enabled
):
    from drone_fly.harness.session import run_harness

    session = run_harness(
        "sim-loopback",
        harness_checkpoint_30,
        30,
        config={"runner": {"latency_compensation": mode}},
        k1_path=str(harness_k1_dir),
        device="cpu",
        duration_s=0.1,
        output_root=str(tmp_path),
    )
    assert session.result.ticks > 0
    (comp,) = captured_comp
    assert isinstance(comp, LatencyCompensator) and comp.enabled is enabled
    assert comp.eligible is False  # sim clock: auto never turns on
    if enabled:  # the real contract plant_step rolled the forecast
        assert comp.plant_step is not None and comp._commands


def test_session_refuses_on_with_a_student(monkeypatch):
    from pathlib import Path

    from drone_fly.harness import student
    from drone_fly.harness.session import run_harness

    plan = SimpleNamespace(
        meta={"obs_schema": {"name": "fake", "version": 1, "dim": 30}},
        overrides={},
        path=Path("student.pt"),
    )
    monkeypatch.setattr(student, "plan_student", lambda *a, **kw: plan)
    with pytest.raises(HarnessConfigError, match="latency_compensation applies only"):
        run_harness("liftoff", "student.pt",
                    config={"runner": {"latency_compensation": "on"}}, dry_run=True)  # fmt: skip
