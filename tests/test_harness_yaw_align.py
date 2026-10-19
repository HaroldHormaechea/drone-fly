"""Spawn-yaw frame alignment for gpu_prototype contract policies (``runner.align_spawn_yaw``).

CPU-only and hermetic. Covers the :class:`SpawnYawAligner` invariance (a scene rotated by
``psi`` about +z, assembled by the real contract and aligned, equals the yaw-0 scene for the
12 / 30 / 70-dim prefixes, actions included), the byte-identical no-op cases, the pair indices
against ``EXTRA_LAYOUT``, the runner wiring (capture, re-capture on a discontinuity, the
``spawn_yaw:`` event, the reset across runs, the recorded extra), the session wiring (contract
only; an explicit ``true`` refused with a teacher / student plan), the config validation, and the
opt-in ``eyes.world_scale`` conversion.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.harness import yaw_align
from drone_fly.harness.assembler import EXTRA_LAYOUT, ContractObservationAssembler
from drone_fly.harness.config import RunnerConfig, load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.interfaces import Harness
from drone_fly.harness.runner import HarnessRunner, RunnerSettings, SimClock
from drone_fly.harness.types import FlightState, Frame, FreeSpaceEstimate, GateEstimate
from drone_fly.harness.yaw_align import (
    EXTRA_XY_PAIRS,
    OBS_XY_PAIRS,
    SpawnYawAligner,
    tilt_of,
    yaw_of,
)
from tests.harness_support import (
    FakeFreeSpaceEstimator,
    FakeGateEstimator,
    FakePolicy,
    FakeSink,
    FakeStateSource,
    ListRecorder,
    fake_contract,
)
from tests.oracle_support import age_files, write_teacher_run
from tests.test_harness_teacher_policy import (  # noqa: F401  (oracle_course is a fixture)
    oracle_course,
    run_teacher,
    spawn_csv,
    teacher_config,
)

PSIS_DEG = (-90.0, 37.0, 180.0)


# ----------------------------------------------------------------------------- scene helpers


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _qz(psi):
    return np.array([math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)])


def _rot(v, psi):
    v = np.asarray(v, np.float64).copy()
    c, s = math.cos(psi), math.sin(psi)
    v[0], v[1] = c * v[0] - s * v[1], s * v[0] + c * v[1]
    return v


#: A tilted (pitch 0.3, roll -0.2), yaw-0 base attitude: g_body is non-trivial.
_Q0 = _qmul(
    np.array([math.cos(0.15), 0.0, math.sin(0.15), 0.0]),
    np.array([math.cos(-0.1), math.sin(-0.1), 0.0, 0.0]),
)


def _scene(psi):
    """State, gate and free space of one scene, rotated as a whole by ``psi`` about +z."""
    q = _qmul(_qz(psi), _Q0)
    st = FlightState(
        pos=_rot([1.0, -0.5, 1.2], psi).astype(np.float32),
        vel=_rot([2.0, 0.7, -0.3], psi).astype(np.float32),
        quat=q.astype(np.float32),
        omega=np.array([0.4, -0.2, 0.9], np.float32),  # body frame: yaw-invariant
    )
    target = _rot([3.0, 1.0, 1.5], psi).astype(np.float32)
    gy, ny = 0.4 + psi, -0.8 + psi
    bearing = _rot([0.6, 0.8, 0.0], psi)
    curr = np.array([math.cos(gy), math.sin(gy), 0.9], np.float32)
    nxt = np.array([*bearing, math.cos(ny), math.sin(ny), 0.35, 1.0], np.float32)
    ov = np.array([*_rot([1.5, -0.4, 0.2], psi), 0.3, *_rot([-0.6, 2.0, -0.1], psi), 0.5])
    rays = np.linspace(0.1, 1.0, 40).astype(np.float32)
    fs = FreeSpaceEstimate(
        obstacle_vision=ov.astype(np.float32),
        raycast=rays,
        in_fov_mask=np.ones(40, bool),
        stamp=0.0,
    )
    return st, GateEstimate(target, curr_gate=curr, next_lookahead=nxt), fs


def _aligned(contract, psi, extra_dim):
    st, gate, fs = _scene(psi)
    obs, extra = ContractObservationAssembler(contract).assemble(st, gate, fs, extra_dim)
    al = SpawnYawAligner()
    al.capture(st)
    return al.apply(obs, extra), al


# ----------------------------------------------------------------------------- invariance


@pytest.mark.parametrize("extra_dim", [12, 30, 70])
@pytest.mark.parametrize("psi_deg", PSIS_DEG)
def test_rotated_scene_aligns_to_the_yaw0_scene(harness_contract, psi_deg, extra_dim):
    psi = math.radians(psi_deg)
    (ref_obs, ref_extra), al0 = _aligned(harness_contract, 0.0, extra_dim)
    (obs, extra), al = _aligned(harness_contract, psi, extra_dim)
    assert al0.psi0 == pytest.approx(0.0, abs=1e-6)
    assert math.remainder(al.psi0 - psi, 2 * math.pi) == pytest.approx(0.0, abs=1e-5)
    assert obs.dtype == np.float32 and extra.dtype == np.float32
    assert extra.shape == (extra_dim,)
    np.testing.assert_allclose(obs, ref_obs, atol=1e-5)
    np.testing.assert_allclose(extra, ref_extra, atol=1e-5)
    if extra_dim == 70:  # heading-relative raycast is untouched
        np.testing.assert_array_equal(extra[30:70], np.linspace(0.1, 1.0, 40).astype(np.float32))


def test_rotated_scene_is_not_aligned_without_the_fix(harness_contract):
    """Sanity: the raw assembled inputs differ between the scenes (the bug the aligner fixes)."""
    asm = ContractObservationAssembler(harness_contract)
    a = asm.assemble(*_scene(0.0), 30)[1]
    b = asm.assemble(*_scene(math.radians(-90.0)), 30)[1]
    assert np.max(np.abs(a - b)) > 0.5


@pytest.mark.parametrize("psi_deg", PSIS_DEG)
def test_stub_policy_actions_are_yaw_invariant(harness_contract, psi_deg):
    class Linear(FakePolicy):
        def act(self, obs, extra):
            w = np.arange(extra.shape[0], dtype=np.float32) / 10.0
            return np.array([extra @ w, obs[0], obs[1], extra[13]], np.float32)

    (o0, e0), _ = _aligned(harness_contract, 0.0, 30)
    (o1, e1), _ = _aligned(harness_contract, math.radians(psi_deg), 30)
    np.testing.assert_allclose(Linear().act(o1, e1), Linear().act(o0, e0), atol=1e-4)


@pytest.mark.parametrize("extra_dim", [30, 70])
def test_real_contract_policy_actions_are_yaw_invariant(
    harness_contract, harness_checkpoint_30, harness_checkpoint_70, extra_dim
):
    from drone_fly.harness.policy import ContractPolicyRunner

    ckpt = harness_checkpoint_30 if extra_dim == 30 else harness_checkpoint_70
    (o0, e0), _ = _aligned(harness_contract, 0.0, extra_dim)
    p0 = ContractPolicyRunner(harness_contract, ckpt, extra_dim)
    p0.load()
    a0 = np.asarray(p0.act(o0, e0))
    for psi_deg in PSIS_DEG:
        (o1, e1), _ = _aligned(harness_contract, math.radians(psi_deg), extra_dim)
        p1 = ContractPolicyRunner(harness_contract, ckpt, extra_dim)
        p1.load()
        np.testing.assert_allclose(np.asarray(p1.act(o1, e1)), a0, atol=1e-4, err_msg=psi_deg)


# ----------------------------------------------------------------------------- unit


def test_pair_indices_match_extra_layout():
    off, start = {}, 0
    for name, width in EXTRA_LAYOUT:
        off[name] = start
        start += width
    assert OBS_XY_PAIRS == (0, 6)
    assert EXTRA_XY_PAIRS == (0, 6, 12, 15, 18, 22, 26)
    assert off == {"obs": 0, "curr_gate": 12, "next_lookahead": 15, "obstacle_vision": 22,
                   "raycast": 30}  # fmt: skip
    # no pair touches the raycast; every pair stays inside one block
    for i in EXTRA_XY_PAIRS:
        assert i + 1 < 30
        block = [n for n, s in off.items() if s <= i][-1]
        assert off[block] <= i + 1 < off[block] + dict(EXTRA_LAYOUT)[block]


def _quat_state(quat):
    return SimpleNamespace(quat=np.asarray(quat, np.float32))


def test_yaw_and_tilt_of():
    for psi in (-math.pi / 2, 0.3, 2.5):
        assert yaw_of(_qz(psi)) == pytest.approx(psi, abs=1e-9)
        assert yaw_of(_qmul(_qz(psi), _Q0)) == pytest.approx(psi, abs=1e-9)
    assert tilt_of([1, 0, 0, 0]) == pytest.approx(0.0)
    assert tilt_of(_qz(1.0)) == pytest.approx(0.0, abs=1e-6)
    assert tilt_of([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]) == pytest.approx(90.0)
    assert tilt_of([0, 0, 0, 0]) == 0.0


@pytest.mark.parametrize("extra_dim", [12, 30, 70])
def test_psi0_zero_is_byte_identical(extra_dim):
    obs = np.arange(12, dtype=np.float32) * 0.37
    extra = np.arange(extra_dim, dtype=np.float32) * 0.11
    al = SpawnYawAligner()
    assert al.capture(_quat_state([1, 0, 0, 0])) == 0.0
    o, e = al.apply(obs, extra)
    assert o is obs and e is extra


def test_disabled_and_uncaptured_are_byte_identical():
    obs, extra = np.ones(12, np.float32), np.ones(70, np.float32)
    off = SpawnYawAligner(enabled=False)
    assert off.capture(_quat_state(_qz(1.0))) is None and off.psi0 is None
    assert all(a is b for a, b in zip(off.apply(obs, extra), (obs, extra), strict=True))
    fresh = SpawnYawAligner()
    assert not fresh.captured
    assert all(a is b for a, b in zip(fresh.apply(obs, extra), (obs, extra), strict=True))


def test_rotation_values_zero_blocks_and_untouched_entries():
    al = SpawnYawAligner()
    al.capture(_quat_state(_qz(math.pi / 2)))  # psi0 = +90: (x, y) -> (y, -x)
    obs = np.arange(1, 13, dtype=np.float32)
    extra = np.zeros(70, np.float32)
    extra[:12] = obs
    extra[30:70] = 0.5
    o, e = al.apply(obs, extra)
    np.testing.assert_allclose(o[[0, 1, 6, 7]], [2, -1, 8, -7], atol=1e-6)
    keep = [i for i in range(12) if i not in (0, 1, 6, 7)]
    np.testing.assert_array_equal(o[keep], obs[keep])
    np.testing.assert_array_equal(e[12:30], 0.0)  # zero blocks stay zero
    np.testing.assert_array_equal(e[30:70], 0.5)
    assert o is not obs and e is not extra  # copies
    assert obs[0] == 1.0 and extra[0] == 1.0  # inputs untouched


def test_capture_once_reset_and_tilt_warning(caplog):
    al = SpawnYawAligner()
    assert al.capture(_quat_state(_qz(0.5))) == pytest.approx(0.5)
    assert al.capture(_quat_state(_qz(1.5))) is None and al.psi0 == pytest.approx(0.5)
    al.reset()
    assert al.psi0 is None
    flipped = _qmul(_qz(0.2), [math.cos(1.2), math.sin(1.2), 0.0, 0.0])  # ~137 deg tilt
    with caplog.at_level(logging.WARNING, logger=yaw_align.__name__):
        assert al.capture(_quat_state(flipped)) is not None
    assert "unreliable" in caplog.text


# ----------------------------------------------------------------------------- runner


def _yaw_state(psi_deg):
    st = FakeStateSource()
    st.state = FlightState(
        pos=np.array([0.0, 0.0, 1.0], np.float32),
        vel=np.zeros(3, np.float32),
        quat=_qz(math.radians(psi_deg)).astype(np.float32),
        omega=np.zeros(3, np.float32),
    )
    return st


def _runner(state, policy, rec, aligner, max_ticks=4):
    contract = fake_contract()
    h = Harness(
        name="fake",
        state_source=state,
        gate_estimator=FakeGateEstimator(),
        action_sink=FakeSink(),
        free_space_estimator=FakeFreeSpaceEstimator(),
        clock=SimClock(),
    )
    return HarnessRunner(
        h,
        policy,
        ContractObservationAssembler(contract),
        contract,
        RunnerSettings(max_ticks=max_ticks),
        recorder=rec,
        yaw_aligner=aligner,
    )


def test_runner_captures_on_first_tick_and_records_aligned_inputs():
    state, rec, pol, al = _yaw_state(-90.0), ListRecorder(), FakePolicy(), SpawnYawAligner()
    _runner(state, pol, rec, al).run()
    assert al.psi0 == pytest.approx(-math.pi / 2, abs=1e-6)
    events = [t.events for t in rec.ticks]
    assert events[0] == ["spawn_yaw:-90.0"]
    assert sum(any(x.startswith("spawn_yaw:") for x in e) for e in events) == 1
    # FakeGateEstimator target (2, 0, 1) from pos (0, 0, 1): rel (2, 0) world -> (0, 2) aligned
    for (obs, extra), t in zip(pol.calls, rec.ticks, strict=True):
        np.testing.assert_allclose(obs[:2], [0.0, 2.0], atol=1e-6)
        np.testing.assert_array_equal(t.extra, extra)  # recorded = what the policy saw
        np.testing.assert_allclose(extra[12:14], [0.0, 1.0], atol=1e-6)  # curr gate (1, 0)
        assert t.state["pos"] is not None  # state stays raw
    np.testing.assert_allclose(rec.ticks[0].state["quat"], state.state.quat)


def test_runner_without_aligner_is_unchanged():
    state, rec, pol = _yaw_state(-90.0), ListRecorder(), FakePolicy()
    _runner(state, pol, rec, None).run()
    assert not any(x.startswith("spawn_yaw:") for t in rec.ticks for x in t.events)
    np.testing.assert_allclose(pol.calls[0][0][:2], [2.0, 0.0], atol=1e-6)


def test_runner_recaptures_after_a_discontinuity():
    state, rec, al = _yaw_state(-90.0), ListRecorder(), SpawnYawAligner()

    class Respawn(FakePolicy):
        def act(self, obs, extra):
            if len(self.calls) == 1:  # respawn facing +30 deg
                state.discontinuity_count += 1
                state.state = dataclasses.replace(state.state, quat=_qz(math.radians(30.0)).astype(np.float32))
            return super().act(obs, extra)

    pol = Respawn()
    _runner(state, pol, rec, al).run()
    events = [t.events for t in rec.ticks]
    assert events[0] == ["spawn_yaw:-90.0"]
    assert events[1] == []
    assert events[2] == ["discontinuity:estimators_reset", "spawn_yaw:30.0"]
    assert al.psi0 == pytest.approx(math.radians(30.0), abs=1e-6)
    want = _rot([2.0, 0.0, 0.0], -math.radians(30.0))[:2]
    np.testing.assert_allclose(pol.calls[2][0][:2], want, atol=1e-6)


def test_runner_resets_the_aligner_across_runs():
    state, al = _yaw_state(-90.0), SpawnYawAligner()
    rec1 = ListRecorder()
    r = _runner(state, FakePolicy(), rec1, al, max_ticks=2)
    r.run()
    state.state = dataclasses.replace(state.state, quat=_qz(math.radians(45.0)).astype(np.float32))
    rec2 = ListRecorder()
    r2 = _runner(state, FakePolicy(), rec2, al, max_ticks=2)
    r2.run()
    assert rec2.ticks[0].events == ["spawn_yaw:45.0"]
    assert al.psi0 == pytest.approx(math.radians(45.0), abs=1e-6)


def test_runner_spawn_yaw_logs_info(caplog):
    with caplog.at_level(logging.INFO, logger="drone_fly.harness.runner"):
        _runner(_yaw_state(-90.0), FakePolicy(), ListRecorder(), SpawnYawAligner()).run()
    assert "spawn yaw -90.0 deg" in caplog.text


# ----------------------------------------------------------------------------- config


def _runner_dict(**kw):
    base = load_config("sim-loopback").data["runner"]
    return {**base, **kw}


@pytest.mark.parametrize("value", ["auto", True, False])
def test_config_accepts_auto_true_false(value):
    assert RunnerConfig.from_dict(_runner_dict(align_spawn_yaw=value)).align_spawn_yaw == value


def test_config_default_is_auto():
    d = _runner_dict()
    d.pop("align_spawn_yaw", None)
    assert RunnerConfig.from_dict(d).align_spawn_yaw == "auto"
    for harness in ("sim-loopback", "liftoff"):
        cfg = load_config(harness)
        assert cfg.runner.align_spawn_yaw == "auto"
        assert RunnerSettings.from_config(cfg).align_spawn_yaw == "auto"


@pytest.mark.parametrize("value", ["yes", "on", 1, 0, None, "AUTO", "true"])
def test_config_rejects_other_values(value):
    with pytest.raises(HarnessConfigError, match="runner.align_spawn_yaw"):
        RunnerConfig.from_dict(_runner_dict(align_spawn_yaw=value))


def test_config_override_reaches_runner_settings():
    cfg = load_config("sim-loopback", {"runner": {"align_spawn_yaw": False}})
    assert RunnerSettings.from_config(cfg).align_spawn_yaw is False
    with pytest.raises(HarnessConfigError, match="align_spawn_yaw"):
        load_config("sim-loopback", {"runner": {"align_spawn_yaw": "maybe"}})


# ----------------------------------------------------------------------------- session wiring


@pytest.fixture
def captured_runner(monkeypatch):
    """Patch session.HarnessRunner to record the ``yaw_aligner`` it was built with."""
    from drone_fly.harness import session

    seen: list = []

    class Spy(HarnessRunner):
        def __init__(self, *a, **kw):
            seen.append(kw.get("yaw_aligner"))
            super().__init__(*a, **kw)

    monkeypatch.setattr(session, "HarnessRunner", Spy)
    return seen


@pytest.mark.parametrize(("value", "enabled"), [("auto", True), (True, True), (False, False)])
def test_session_contract_branch_builds_the_aligner(
    harness_checkpoint_30, harness_k1_dir, tmp_path, captured_runner, value, enabled
):
    from drone_fly.harness.session import run_harness

    session = run_harness(
        "sim-loopback",
        harness_checkpoint_30,
        30,
        config={"runner": {"align_spawn_yaw": value}},
        k1_path=str(harness_k1_dir),
        device="cpu",
        duration_s=0.1,
        output_root=str(tmp_path),
    )
    assert session.result.ticks > 0
    (al,) = captured_runner
    assert isinstance(al, SpawnYawAligner) and al.enabled is enabled
    if enabled:  # the plant spawns at yaw 0: captured, a no-op
        assert al.psi0 == pytest.approx(0.0, abs=1e-6)


def _fake_student_plan(monkeypatch):
    from drone_fly.harness import student

    plan = SimpleNamespace(
        meta={"obs_schema": {"name": "fake", "version": 1, "dim": 30}},
        overrides={},
        path=Path("student.pt"),
    )
    monkeypatch.setattr(student, "plan_student", lambda *a, **kw: plan)


def test_session_refuses_explicit_true_with_a_student(monkeypatch):
    from drone_fly.harness.session import run_harness

    _fake_student_plan(monkeypatch)
    with pytest.raises(HarnessConfigError, match="only to gpu_prototype contract checkpoints"):
        run_harness("liftoff", "student.pt", config={"runner": {"align_spawn_yaw": True}},
                    dry_run=True)  # fmt: skip


def test_session_refuses_explicit_true_with_a_teacher(
    harness_contract, harness_k1_dir, tmp_path, oracle_course
):
    run = write_teacher_run(tmp_path / "runs")
    age_files(run)
    csv_path = spawn_csv(tmp_path / "tele.csv", oracle_course.course)
    with pytest.raises(HarnessConfigError, match="only to gpu_prototype contract checkpoints"):
        run_teacher(tmp_path, harness_k1_dir,
                    teacher_config(csv_path, runner={"align_spawn_yaw": True}), run)  # fmt: skip


def test_session_teacher_auto_has_no_aligner(
    harness_contract, harness_k1_dir, tmp_path, oracle_course, captured_runner
):
    run = write_teacher_run(tmp_path / "runs")
    age_files(run)
    csv_path = spawn_csv(tmp_path / "tele.csv", oracle_course.course)
    run_teacher(tmp_path, harness_k1_dir, teacher_config(csv_path), run)
    assert captured_runner and all(a is None for a in captured_runner)


# ----------------------------------------------------------------------------- eyes world_scale


def _eyes_cfg(eyes, world_scale=None):
    data = {"parts": {"free_space_estimator": "eyes"}, "eyes": eyes}
    if world_scale is not None:
        data["scaling"] = {"world_scale": world_scale}
    return load_config("liftoff", data, preset="liftoff")


def _eyes_build(tmp_path, contract, meta_ws, eyes_ws=None, run_ws=None):
    from tests.test_harness_eyes import _contract_meta, _ctx, _write_weights

    from drone_fly.harness.liftoff.eyes import EyesFreeSpaceEstimator

    meta = _contract_meta(contract)
    meta["world_scale"] = meta_ws
    w = _write_weights(tmp_path / "eyes.pt", meta)
    eyes = {"weights": str(w), "device": "cpu"}
    if eyes_ws is not None:
        eyes["world_scale"] = eyes_ws
    cfg = _eyes_cfg(eyes, run_ws)
    return EyesFreeSpaceEstimator.from_config(cfg.section("free_space"), _ctx(cfg, contract))


def test_eyes_world_scale_null_keeps_the_run_scale_rule(tmp_path, harness_contract):
    est = _eyes_build(tmp_path, harness_contract, 4.0)
    assert est.scale_factor == 1.0
    with pytest.raises(HarnessConfigError, match=r"world_scale \(4\.0\).*\(12\.0\)"):
        _eyes_build(tmp_path, harness_contract, 4.0, run_ws=12.0)


def test_eyes_world_scale_set_checks_meta_against_it(tmp_path, harness_contract):
    est = _eyes_build(tmp_path, harness_contract, 4.0, eyes_ws=4.0, run_ws=12.0)
    assert est.scale_factor == pytest.approx(3.0)
    with pytest.raises(HarnessConfigError, match=r"world_scale \(4\.0\).*\(2\.0\)"):
        _eyes_build(tmp_path, harness_contract, 4.0, eyes_ws=2.0, run_ws=12.0)


@pytest.mark.parametrize("bad", [0, -1.0, "x"])
def test_eyes_world_scale_invalid_rejected(tmp_path, harness_contract, bad):
    with pytest.raises(HarnessConfigError, match="eyes.world_scale"):
        _eyes_build(tmp_path, harness_contract, 4.0, eyes_ws=bad)


def test_packaged_eyes_world_scale_default_is_null():
    assert load_config("liftoff").section("eyes")["world_scale"] is None


def test_free_to_run_scale_and_scale_state():
    from drone_fly.harness.liftoff.eyes import HIT_EPS, free_to_run_scale, scale_state

    f = np.array([1.0, 1.0 - HIT_EPS / 2, 0.9, 0.5, 0.0, 1.2], np.float32)
    np.testing.assert_allclose(free_to_run_scale(f, 3.0), [1, 1, 0.3, 0.5 / 3, 0, 1], atol=1e-6)
    assert free_to_run_scale(f, 1.0) is not None
    np.testing.assert_array_equal(free_to_run_scale(f, 1.0), f)
    st = FlightState(pos=np.array([1.0, 2.0, 3.0], np.float32),
                     vel=np.array([0.5, 0.0, -1.0], np.float32),
                     quat=_qz(0.4).astype(np.float32),
                     omega=np.array([0.1, 0.2, 0.3], np.float32), t=1.5)  # fmt: skip
    s3 = scale_state(st, 3.0)
    np.testing.assert_allclose(s3.pos, [3, 6, 9])
    np.testing.assert_allclose(s3.vel, [1.5, 0, -3])
    np.testing.assert_array_equal(s3.quat, st.quat)
    np.testing.assert_array_equal(s3.omega, st.omega)
    assert s3.t == 1.5 and s3.pos.dtype == np.float32
    assert scale_state(st, 1.0) is st
    np.testing.assert_allclose(st.pos, [1, 2, 3])  # input untouched


def test_eyes_process_scales_inputs_and_converts_free():
    from tests.test_harness_eyes import FakeEyes, _img

    from drone_fly.harness.liftoff.eyes import EyesFreeSpaceEstimator, obstacle_vision_from_rays

    class Recording(FakeEyes):
        def update(self, frame, st, target):
            self.states = getattr(self, "states", []) + [st]
            return super().update(frame, st, target)

    fake = Recording(k_frames=1)
    est = EyesFreeSpaceEstimator(fake, obstacle_mode="ray_clusters", scale_factor=3.0)
    st = FlightState(pos=np.array([1.0, 2.0, 1.0], np.float32),
                     vel=np.array([2.0, 0.0, 0.0], np.float32),
                     quat=np.array([1, 0, 0, 0], np.float32), omega=np.zeros(3, np.float32))  # fmt: skip
    out = est.update(Frame(_img(), 0.0, 60), st, np.array([4.0, 0.0, 1.0]), 0.0)  # free 0.6
    np.testing.assert_allclose(fake.states[-1].pos, [3, 6, 3])
    np.testing.assert_allclose(fake.states[-1].vel, [6, 0, 0])
    np.testing.assert_allclose(fake.targets[-1], [12, 0, 3])
    np.testing.assert_allclose(out.raycast, 0.2, atol=1e-6)
    assert out.info["eyes_scale_factor"] == 3.0
    want = obstacle_vision_from_rays(st.pos, np.tile([1.0, 0, 0], (40, 1)),
                                     np.full(40, 0.2, np.float32), 4.0)  # fmt: skip
    np.testing.assert_allclose(out.obstacle_vision, want, atol=1e-6)
    # a clear ray stays clear
    clear = est.update(Frame(_img(), 0.1, 100), st, np.zeros(3), 0.1)
    np.testing.assert_array_equal(clear.raycast, 1.0)


def test_eyes_process_default_factor_is_passthrough():
    from tests.test_harness_eyes import FakeEyes, _img

    from drone_fly.harness.liftoff.eyes import EyesFreeSpaceEstimator

    fake = FakeEyes(k_frames=1)
    est = EyesFreeSpaceEstimator(fake)
    out = est.update(Frame(_img(), 0.0, 60), _yaw_state(0.0).state, np.array([1.0, 2, 3]), 0.0)
    np.testing.assert_allclose(out.raycast, 0.6, atol=1e-6)
    np.testing.assert_array_equal(fake.targets[-1], [1, 2, 3])
    assert out.info["eyes_scale_factor"] == 1.0


# ----------------------------------------------------------------- UC-89 no-detection placeholder


def _neutral_from(psi, ap=0.9):
    """A no-detection scene at spawn yaw ``psi``: the UC-89 frame-neutral curr_gate [0,0,ap] and a
    hover target (rel = target - pos = 0), reusing ``_scene``'s state / free space."""
    st, _, fs = _scene(psi)
    target = np.asarray(st.pos, np.float32).copy()  # hover in place while no gate is detected
    curr = np.array([0.0, 0.0, ap], np.float32)  # UC-89: rotation-invariant placeholder
    nxt = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)  # unseen next gate
    return st, GateEstimate(target, curr_gate=curr, next_lookahead=nxt), fs


@pytest.mark.parametrize("psi_deg", PSIS_DEG)  # includes -90 and arbitrary (37, 180)
def test_uc89_neutral_placeholder_is_frame_invariant(harness_contract, psi_deg):
    """UC-89 AC1: the no-detection curr_gate [0,0,ap] + hover rel stay neutral under yaw alignment
    at a -90 deg (and arbitrary) spawn -- it can never become a phantom off-axis gate."""
    psi = math.radians(psi_deg)
    st, gate, fs = _neutral_from(psi)
    obs, extra = ContractObservationAssembler(harness_contract).assemble(st, gate, fs, 30)
    al = SpawnYawAligner()
    al.capture(st)
    o, e = al.apply(obs, extra)
    # rel (hover) stays zero; curr_gate (cos,sin) stays [0,0]; the aperture is untouched.
    np.testing.assert_allclose(o[:3], 0.0, atol=1e-6)  # obs rel
    np.testing.assert_allclose(e[:3], 0.0, atol=1e-6)  # extra rel
    np.testing.assert_allclose(e[12:14], [0.0, 0.0], atol=1e-6)  # curr_gate direction
    assert e[14] == pytest.approx(0.9)  # aperture preserved


def test_uc89_old_world_frame_placeholder_rotates_into_a_phantom_gate(harness_contract):
    """Regression guard: the OLD [1,0,ap] world-frame placeholder is NOT frame-neutral -- at a
    -90 deg spawn it rotates into a ~90 deg off-axis gate (the live-crash bug UC-89 fixes)."""
    st, _, fs = _scene(math.radians(-90.0))
    target = np.asarray(st.pos, np.float32).copy()
    bad = GateEstimate(
        target,
        curr_gate=np.array([1.0, 0.0, 0.9], np.float32),  # the pre-fix world-frame [1,0]
        next_lookahead=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32),
    )
    obs, extra = ContractObservationAssembler(harness_contract).assemble(st, bad, fs, 30)
    al = SpawnYawAligner()
    al.capture(st)
    _, e = al.apply(obs, extra)
    # the curr_gate direction is dragged off +x -> a fake side gate the policy steers at
    assert np.linalg.norm(e[12:14] - np.array([1.0, 0.0])) > 0.5
    assert abs(e[13]) == pytest.approx(1.0, abs=1e-6)  # rotated onto +/- y at -90 deg
