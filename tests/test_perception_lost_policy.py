"""UC70 ADR 0009 amendment: the tracker's lost policy (``gates.lost_policy terminate|reacquire``).

* The REAL live tracker (``TrackingGateEstimator`` fed scripted detections of the truth gate) and
  the NumPy reference (``ReferencePerception``, ``exact``) agree tick by tick under ``reacquire``:
  a stale held target re-acquires a new candidate after ``drop_after_s`` (not before), a target
  unmatched for ``lost_timeout_s`` is released (its estimate stays the target, ``target_valid``
  false) and acquired again, and ``lost`` comes only ``lost_hard_stop_s`` after the last good event
  (a match, an acquisition or a pass). Under ``terminate`` the same scripts give the legacy result.
* Config: an unknown policy is refused (training config, ``PerceptionConfig``, the live tracker);
  the student defaults to ``reacquire``, the harness YAML to ``terminate``; a student checkpoint's
  pinned policy reaches the live estimator.
* The checkpoint: the student-obs schema and the training sidecar pin the three fields; an
  export whose tracker block disagrees with this build's schema is refused at load.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drone_fly.harness.errors import HarnessConfigError, StudentSchemaError
from drone_fly.harness.geometry import CameraModel
from drone_fly.harness.liftoff.gate_cv import GateDetection, TrackingGateEstimator
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import FlightState, Frame
from drone_fly.sim.noise_model import camera_visible, gate_axes, gate_corners
from drone_fly.sim.perception import EPISODE_DRAWS, TICK_DRAWS, PerceptionConfig
from drone_fly.student.perception_ref import ReferencePerception
from tests.student_support import (
    StaticDrone,
    gate_batch,
    gate_rot,
    load_model,
    perception,
    yaw_quat,
)

A = np.array([10.0, 0.0, 0.0])
#: 4.7 m from A (outside the 2 m association gate), in view, NEARER than A (so the live
#: tracker never takes it as the next gate: parity with the reference fed no next passage)
B = np.array([6.0, 2.4, 0.0])
DT = 0.02
LOST, DROP, HARD = 0.6, 0.2, 1.5
TIMING = {"frame_period_s": DT, "frame_jitter_s": 0.0, "frame_skip_p": 0.0,
          "detection_lag_s": [0.0, 0.0]}  # fmt: skip
AWAY, FACE = math.pi, 0.0
POLICY = ("reacquire", 6.0, 0.5)  # the student's (lost_policy, lost_hard_stop_s, drop_after_s)


def _det(center: np.ndarray, pos: np.ndarray) -> GateDetection:
    return GateDetection(np.zeros((4, 2)), 0.05, np.zeros(3), np.zeros(3),
                         center.astype(np.float32), 0.0, 2.0,
                         float(np.linalg.norm(center - pos)), 0.0, "inflatable_arch")  # fmt: skip


class ScriptedTracker(TrackingGateEstimator):
    """Detects ``self.truth`` exactly when the shared visibility predicate says it is visible."""

    def __init__(self, camera, spec, **kw):
        super().__init__(camera, WorldScale(1.0), **kw)
        self.spec = spec
        self.truth = A

    def detect(self, image, state):
        q = np.asarray(state.quat, np.float64)
        pos = np.asarray(state.pos, np.float64)
        k = gate_corners(self.truth, gate_axes(0.0), (2.0, 2.0))
        if not camera_visible(self.truth[None], k[None], pos, q, self.spec)[0]:
            return []
        return [_det(self.truth, pos)]


class Blind(TrackingGateEstimator):
    """Returns ``self.dets`` (set by the test) as this frame's detections."""

    dets: list = []

    def detect(self, image, state):
        return list(self.dets)


def _state(pos, q, t):
    return FlightState(pos=pos.astype(np.float32), vel=np.zeros(3, np.float32),
                       quat=q.astype(np.float32), omega=np.zeros(3, np.float32), t=t)  # fmt: skip


def run_pair(tmp_path: Path, policy: str, script, ticks: int):
    """Drive the live tracker and the reference with ``script(t) -> (yaw, truth gate)``; assert
    tick-by-tick parity and return ``[(t, live GateEstimate)]``."""
    model = load_model(tmp_path, timing=TIMING,
                       tracker={"lost_timeout_s": LOST, "acquire_timeout_s": 0.4})  # fmt: skip
    cfg = PerceptionConfig.from_model(model, "exact", lost_policy=policy, lost_hard_stop_s=HARD,
                                      drop_after_s=DROP)  # fmt: skip
    ref = ReferencePerception(model, cfg)
    trk = ScriptedTracker(CameraModel(640, 480, 90.0), model.visibility_spec(),
                          default_aperture=2.0, assoc_radius=1.0, lost_timeout_s=LOST,
                          acquire_timeout_s=0.4, lost_policy=policy, lost_hard_stop_s=HARD,
                          drop_after_s=DROP)  # fmt: skip
    pos = np.zeros(3)
    yaw0, _ = script(0.0)
    ref.reset(0.0, pos, yaw_quat(yaw0), np.zeros(sum(w for _, w, _ in EPISODE_DRAWS)))
    trk.update(None, _state(pos, yaw_quat(yaw0), 0.0), 0.0)
    no_draws = np.zeros(sum(w for _, w, _ in TICK_DRAWS))
    out = []
    for i in range(1, ticks):
        t = round(i * DT, 6)
        yaw, truth = script(t)
        q = yaw_quat(yaw)
        trk.truth = truth
        ref.push_pose(t, pos, q)
        ref.step(t, pos, q, {"center": truth, "rot": gate_rot(0.0), "wh": (2.0, 2.0)}, None,
                 False, 1.0, no_draws)  # fmt: skip
        r = ref.estimate(1.0)
        h = trk.update(Frame(np.zeros((480, 640, 3), np.uint8), t, i), _state(pos, q, t), t)
        ctx = (policy, t, r.status, h.status)
        assert r.status == h.status, ctx
        np.testing.assert_allclose(r.target, h.target, atol=1e-5, err_msg=str(ctx))
        np.testing.assert_allclose(r.curr_gate, h.curr_gate, atol=1e-5, err_msg=str(ctx))
        np.testing.assert_allclose(r.next_lookahead, h.next_lookahead, atol=1e-5,
                                   err_msg=str(ctx))  # fmt: skip
        for key in ("matched", "target_valid", "hand_over"):
            assert r.info[key] == h.info[key], (key, ctx)
        for key in ("target_age_s", "next_age_s"):
            assert r.info[key] == pytest.approx(h.info[key], abs=1e-6), (key, ctx)
        out.append((t, h))
    return out


def at(rows, t):
    return next(h for tt, h in rows if abs(tt - t) < 1e-9)


# ----------------------------------------------------------------------------- (a) stale re-acquire


def _switch(t):
    """Facing; the truth gate is A until 0.3 s, then B (the route moved on, A never seen again)."""
    return FACE, (A if t < 0.3 else B)


def test_reacquire_replaces_a_stale_held_target_after_drop_after_s(tmp_path):
    rows = run_pair(tmp_path, "reacquire", _switch, 60)
    # A last matched at 0.28; B is detected from 0.30 but is outside the association gate
    for t in (0.30, 0.40, 0.48):
        h = at(rows, t)
        assert h.status == "holding" and not h.info["matched"], t  # held A, not yet stale
        np.testing.assert_allclose(h.target, A)
    h = at(rows, 0.50)  # 0.50 - 0.28 > drop_after_s: the acquisition rule takes B
    assert h.status == "tracking" and h.info["matched"] and h.info["target_valid"]
    np.testing.assert_allclose(h.target, B)
    assert all(h.status == "tracking" for t, h in rows if t >= 0.5)


def test_terminate_never_switches_a_held_target(tmp_path):
    rows = run_pair(tmp_path, "terminate", _switch, 80)
    for t, h in rows:
        if t >= 0.3:
            np.testing.assert_allclose(h.target, A, err_msg=str(t))  # B is never acquired
            assert not h.info["matched"]
    assert at(rows, 0.86).status == "holding"  # 0.86 - 0.28 <= lost_timeout_s
    assert at(rows, 0.90).status == "lost"  # the legacy fail-safe
    assert all(h.status == "lost" for t, h in rows if t >= 0.9)


# ----------------------------------------------------------------------------- (b) release


def _away(t):
    """Facing A until 0.3 s, away 0.3-2.2 s (past the hard stop), facing again from 2.2 s."""
    return (FACE if t < 0.3 or t >= 2.2 else AWAY), A


def test_reacquire_releases_keeps_the_estimate_then_hard_stops(tmp_path):
    rows = run_pair(tmp_path, "reacquire", _away, 130)
    last_good = 0.28
    h = at(rows, 0.86)  # held, unmatched, <= lost_timeout_s
    assert h.status == "holding" and h.info["target_valid"]
    for t in (0.90, 1.2, 1.76):  # released: the estimate is kept, target_valid false, not lost
        h = at(rows, t)
        assert h.status == "holding" and not h.info["target_valid"], t
        np.testing.assert_allclose(h.target, A, err_msg=str(t))
        assert h.info["target_age_s"] == pytest.approx(LOST)
    # the hard stop counts from the last good event, not from the release
    assert at(rows, round(last_good + HARD, 2)).status == "holding"
    assert at(rows, round(last_good + HARD + DT, 2)).status == "lost"
    for t in (1.9, 2.18):
        h = at(rows, t)
        assert h.status == "lost" and not h.info["target_valid"], t
        np.testing.assert_allclose(h.target, A)  # still the released estimate
    h = at(rows, 2.2)  # facing again: acquired, a good event, so no longer lost
    assert h.status == "tracking" and h.info["target_valid"] and h.info["matched"]


def test_terminate_keeps_the_legacy_lost_without_release(tmp_path):
    rows = run_pair(tmp_path, "terminate", _away, 130)
    assert at(rows, 0.86).status == "holding"
    for t in (0.90, 1.2, 1.9):
        h = at(rows, t)
        assert h.status == "lost" and h.info["target_valid"], t  # never released
        np.testing.assert_allclose(h.target, A)


def _late(t):
    """Away for the first 2 s, facing after: (re)acquire before the first gate is ever seen."""
    return (AWAY if t < 2.0 else FACE), A


@pytest.mark.parametrize("policy", ["terminate", "reacquire"])
def test_before_the_first_acquisition(tmp_path, policy):
    rows = run_pair(tmp_path, policy, _late, 120)
    # terminate: acquire_timeout_s (0.4); reacquire: only the hard stop from the start
    limit = 0.4 if policy == "terminate" else HARD
    assert at(rows, limit).status == "holding"
    assert at(rows, round(limit + DT, 2)).status == "lost"
    assert at(rows, 2.0).status == "tracking"
    for t in (0.2, 1.0):
        np.testing.assert_allclose(at(rows, t).target, np.zeros(3))  # hover in place


def test_terminate_is_the_legacy_tracker(tmp_path):
    """No lost_policy argument == terminate, tick by tick, on every script."""
    model = load_model(tmp_path, timing=TIMING,
                       tracker={"lost_timeout_s": LOST, "acquire_timeout_s": 0.4})  # fmt: skip
    for script in (_switch, _away, _late):
        kw = dict(default_aperture=2.0, assoc_radius=1.0, lost_timeout_s=LOST,
                  acquire_timeout_s=0.4)  # fmt: skip
        legacy = ScriptedTracker(CameraModel(640, 480, 90.0), model.visibility_spec(), **kw)
        # explicit terminate, with reacquire knobs that would change everything if they applied
        term = ScriptedTracker(CameraModel(640, 480, 90.0), model.visibility_spec(),
                               lost_policy="terminate", lost_hard_stop_s=0.01, drop_after_s=0.0,
                               **kw)  # fmt: skip
        assert legacy.lost_policy == "terminate"
        pos = np.zeros(3)
        for i in range(120):
            t = round(i * DT, 6)
            yaw, truth = script(t)
            st = _state(pos, yaw_quat(yaw), t)
            legacy.truth = term.truth = truth
            fr = None if i == 0 else Frame(np.zeros((480, 640, 3), np.uint8), t, i)
            a, b = legacy.update(fr, st, t), term.update(fr, st, t)
            assert a.status == b.status and np.array_equal(a.target, b.target), (script, t)
            assert a.info["target_valid"] == b.info["target_valid"]


def test_a_pass_is_a_good_event_under_reacquire():
    """Flying through the held gate resets the hard-stop clock (the live tracker alone)."""
    trk = Blind(CameraModel(640, 480, 90.0), WorldScale(1.0),
                                default_aperture=2.0, lost_policy="reacquire",
                                lost_hard_stop_s=1.0, drop_after_s=0.2,
                                lost_timeout_s=0.5)  # fmt: skip
    gate = np.array([2.0, 0.0, 0.0])
    dets = [_det(gate, np.zeros(3))]
    trk.dets = dets
    q = yaw_quat(0.0)
    img = np.zeros((4, 4, 3), np.uint8)
    assert trk.update(Frame(img, 0.0, 0), _state(np.zeros(3), q, 0.0), 0.0).status == "tracking"
    dets.clear()  # never seen again: flown through blind at 4 m/s
    rows = {}
    for i in range(1, 25):
        t = round(0.1 * i, 6)
        pos = np.array([min(0.4 * i, 3.0), 0.0, 0.0])
        rows[t] = trk.update(Frame(img, t, i), _state(pos, q, t), t)
    passed = [t for t, h in rows.items() if h.passed]
    assert passed == [0.3]  # 1.2 m from the gate: inside aperture / 2
    # last match 0.0 -> lost after 1.0 s without the pass; with it, only after 1.3 s
    assert rows[1.1].status == rows[1.3].status == "holding"
    assert rows[1.4].status == "lost"


# ------------------------------------------------------------------- UC70 run-2: gate memory
# These drive the batched torch twin (``GatePerception`` via ``StaticDrone``) in ``exact`` mode so
# detection is purely geometric (facing -> seen, turned away -> not) and deterministic without
# injected draws; ``reacquire`` is the student's policy, so a lost target is released and the
# memory fallback becomes observable.

A_GATE = (10.0, 0.0, 0.0)
B_GATE = (14.0, 2.0, 0.0)
FRESH = 0.3


def _memory_perc(model, *, gate_memory):
    return perception(model, 1, mode="exact", lost_policy="reacquire", lost_hard_stop_s=6.0,
                      drop_after_s=0.05, gate_memory=gate_memory, gate_memory_fresh_s=FRESH,
                      pointer_dropout_p=1.0)  # fmt: skip


def test_gate_memory_fallback_supplies_the_target_and_true_age_while_the_pointer_is_off(tmp_path):
    """UC70 run-2 (behavioural): with the pointer DROPPED for the whole episode, the student still
    has a usable target through a lost window via the world-frame gate-memory fallback. Once the
    live target is released (reacquire), memory supplies the remembered pose and the TRUE age (time
    since the last confident sighting) while fresh; with memory OFF the released estimate reports
    the saturated ``lost_timeout_s``. After ``fresh_s`` the fallback expires (age back to
    ``lost_timeout_s``). The remembered target pose is the gate in both cases."""
    model = load_model(tmp_path, timing=TIMING,
                       tracker={"lost_timeout_s": 0.1, "acquire_timeout_s": 3.0})  # fmt: skip
    gate = gate_batch(1, A_GATE)

    def fly(gate_memory):
        d = StaticDrone(_memory_perc(model, gate_memory=gate_memory), 1, DT)
        for _ in range(5):  # face & match: memory records A's confident pose; pointer dropped
            out = d.tick(gate)
        assert bool(out["matched"][0]) and float(out["pointer"][0][3]) == 0.0
        rows = []
        for _ in range(40):  # turn away: no detection -> the live target is released
            out = d.tick(gate, quat=yaw_quat(math.pi))
            rows.append((round(d.t, 3), bool(out["acquired"][0]), bool(out["target_valid"][0]),
                         out["target"][0].clone(), float(out["target_age_s"][0])))  # fmt: skip
        return rows

    off, on = fly(False), fly(True)
    A = torch.tensor(A_GATE, dtype=torch.float32)
    for (_, acq_o, _, tgt_o, _), (_, acq_n, _, tgt_n, _) in zip(off, on, strict=True):
        assert acq_o and acq_n  # both keep a target through the window
        torch.testing.assert_close(tgt_o, A, atol=1e-4, rtol=0)
        torch.testing.assert_close(tgt_n, A, atol=1e-4, rtol=0)
    # memory OFF: a released target reports the saturated lost_timeout age, never more
    assert all(age == pytest.approx(0.1) for _, _, valid, _, age in off if not valid)
    # memory ON: while fresh the age GROWS past lost_timeout (true time since the last sighting)
    grew = [age for _, _, valid, _, age in on if not valid and age > 0.12]
    assert grew and max(grew) > FRESH - 0.06
    # ...then once stale (> fresh_s since the last confident match) the fallback stops
    assert all(age == pytest.approx(0.1) for _, _, _, _, age in on[-5:])


def test_hand_over_seeds_the_gate_memory_with_the_next_slot(tmp_path):
    """UC70 run-2 (behavioural): hand-over seeding shifts the NEXT slot's memory into the target on
    a pass, so a gate that is briefly lost right after the pass stays re-targetable. Here the next
    gate B is confidently seen, then the drone passes gate A and immediately turns away — B is
    never matched as the target, so the ONLY source of its remembered pose is the seeding. The
    memory fallback must report B's pose (not the stale A), proving the seed shifted across."""
    model = load_model(tmp_path, timing=TIMING,
                       tracker={"lost_timeout_s": 0.1, "acquire_timeout_s": 3.0})  # fmt: skip
    d = StaticDrone(_memory_perc(model, gate_memory=True), 1, DT)
    gA, gB = gate_batch(1, A_GATE), gate_batch(1, B_GATE)
    for _ in range(5):  # face forward: A is the target (matched), B is the next (matched -> nm)
        out = d.tick(gA, gB)
    assert bool(out["matched"][0]) and bool(out["acquired"][0])
    # pass A AND turn away in the same tick: the hand-over makes B the target, but B is not seen
    # again -> its remembered pose can only come from the next-slot memory seeded on the pass
    out = d.tick(gB, None, crossed=torch.tensor([True]), quat=yaw_quat(math.pi))
    assert bool(out["hand_over"][0])
    B = torch.tensor(B_GATE, dtype=torch.float32)
    seen_mem = []
    for _ in range(12):  # B is released, then the (seeded) memory supplies it while fresh
        out = d.tick(gB, None, quat=yaw_quat(math.pi))
        if bool(out["acquired"][0]) and not bool(out["target_valid"][0]):
            seen_mem.append(out["target"][0].clone())
    assert seen_mem, "the memory fallback never engaged after the hand-over"
    for tgt in seen_mem:  # the remembered target is B (seeded across the pass), NOT the stale A
        torch.testing.assert_close(tgt, B, atol=1e-4, rtol=0)


# ----------------------------------------------------------------------------- config


def test_unknown_lost_policy_is_refused_everywhere(tmp_path, harness_k1_dir):
    from drone_fly.training.config import TrainConfigError
    from tests.student_support import tiny_student_config

    with pytest.raises(TrainConfigError, match="lost_policy"):
        tiny_student_config(tmp_path / "07.pt", perception={"lost_policy": "retry"})
    with pytest.raises(ValueError, match="lost_policy"):
        PerceptionConfig(mode="exact", lost_policy="retry")
    with pytest.raises(ValueError, match="lost_hard_stop_s"):
        PerceptionConfig(mode="exact", lost_hard_stop_s=0.0)
    with pytest.raises(ValueError, match="drop_after_s"):
        PerceptionConfig(mode="exact", drop_after_s=-0.1)
    with pytest.raises(HarnessConfigError, match="lost_policy"):
        Blind(CameraModel(640, 480, 90.0), default_aperture=2.0, lost_policy="Reacquire")


def test_student_defaults_to_reacquire_and_the_harness_to_terminate(tmp_path):
    from drone_fly.harness.config import load_config
    from drone_fly.harness.liftoff.gate_cv import OpenCvGateEstimator
    from drone_fly.training import distill
    from drone_fly.training.config import PerceptionSection
    from tests.student_support import tiny_student_config

    cfg = tiny_student_config(tmp_path / "07.pt")
    p = cfg.perception
    assert PerceptionSection().lost_policy == p.lost_policy == "reacquire"
    assert (p.lost_hard_stop_s, p.drop_after_s) == (6.0, 0.5)
    model = load_model(tmp_path, timing=TIMING)
    pc = distill.perception_config(cfg, model)
    assert (pc.lost_policy, pc.lost_hard_stop_s, pc.drop_after_s) == ("reacquire", 6.0, 0.5)
    cfg2 = tiny_student_config(tmp_path / "07.pt", perception={"lost_policy": "terminate",
                                                               "drop_after_s": 0.3})  # fmt: skip
    pc2 = distill.perception_config(cfg2, model)
    assert (pc2.lost_policy, pc2.drop_after_s) == ("terminate", 0.3)
    # the sim's own default stays the legacy one
    assert PerceptionConfig(mode="exact").lost_policy == "terminate"
    for preset in (None, "liftoff"):
        hc = load_config("liftoff", preset=preset)
        g = hc.section("gates")
        assert (g["lost_policy"], g["lost_hard_stop_s"], g["drop_after_s"]) == (
            "terminate", 6.0, 0.5), preset  # fmt: skip
        est = OpenCvGateEstimator.from_config(g, SimpleNamespace(config=hc))
        assert est.lost_policy == "terminate" and est.lost_hard_stop_s == 6.0
    # a missing key (an older user YAML) also builds terminate
    hc = load_config("liftoff")
    g = {k: v for k, v in hc.section("gates").items() if k not in ("lost_policy",)}
    assert OpenCvGateEstimator.from_config(g, SimpleNamespace(config=hc)).lost_policy == "terminate"
    bad = load_config("liftoff", overrides={"gates.lost_policy": "retry"})
    with pytest.raises(HarnessConfigError, match="lost_policy"):
        OpenCvGateEstimator.from_config(bad.section("gates"), SimpleNamespace(config=bad))


def test_the_learned_estimator_reads_the_policy(monkeypatch):
    from drone_fly.harness.config import load_config
    from drone_fly.harness.liftoff import gate_learned

    seen = {}

    def fake_init(self, *a, **k):
        seen.update(k)

    L = gate_learned.LearnedGateEstimator  # noqa: N806
    monkeypatch.setattr(gate_learned, "resolve_weights_path", lambda w: "weights.pt")
    monkeypatch.setattr(gate_learned, "load_weights", lambda *a, **k: (None, {}))
    # UC71: from_config checks the weights' gate-type table; the stub weights have none
    monkeypatch.setattr(gate_learned, "weights_table", lambda meta, table, path: table)
    monkeypatch.setattr(L, "__init__", fake_init)
    monkeypatch.setattr(L, "warmup", lambda self: None)
    hc = load_config("liftoff", overrides={"gates.lost_policy": "reacquire",
                                           "gates.lost_hard_stop_s": 4.0,
                                           "gates.drop_after_s": 0.25})  # fmt: skip
    L.from_config(hc.section("gates"), SimpleNamespace(config=hc))
    assert (seen["lost_policy"], seen["lost_hard_stop_s"], seen["drop_after_s"]) == (
        "reacquire", 4.0, 0.25)  # fmt: skip
    seen.clear()
    hc = load_config("liftoff")
    L.from_config(hc.section("gates"), SimpleNamespace(config=hc))
    assert seen["lost_policy"] == "terminate"


# ----------------------------------------------------------------------------- the checkpoint


def test_schema_pins_the_three_fields():
    from drone_fly.sim.student_obs import schema

    cam = {"resolution": [640, 480], "hfov_deg": 90.0, "uptilt_deg": 0.0}
    sch = schema(world_scale=10.0, lost_timeout_s=1.3, acquire_timeout_s=3.0, camera=cam,
                 lost_policy="reacquire", lost_hard_stop_s=6.0, drop_after_s=0.5)  # fmt: skip
    tr = sch["tracker"]
    assert (tr["lost_policy"], tr["lost_hard_stop_s"], tr["drop_after_s"]) == POLICY
    legacy = schema(world_scale=10.0, lost_timeout_s=1.3, acquire_timeout_s=3.0, camera=cam)
    assert legacy["tracker"]["lost_policy"] == "terminate"  # the function's legacy default


def test_the_training_sidecar_and_the_export_pin_the_policy(harness_k1_dir, tmp_path):
    from drone_fly import drones
    from drone_fly.harness.student import plan_student
    from drone_fly.sim.action import read_sidecar
    from drone_fly.training import distill
    from drone_fly.training.export import export_student
    from tests.student_support import tiny_pool, tiny_student_config, train_inputs

    inputs = train_inputs(tmp_path / "in", harness_k1_dir)
    cfg = tiny_student_config(inputs.k07, perception={"drop_after_s": 0.4})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(distill, "eval_pool", lambda *a, **k: tiny_pool(2))
        res = distill.train_student(cfg, drones.load(cfg.drone), teacher=inputs.teacher,
                                    noise=inputs.noise, run_root=tmp_path / "out",
                                    max_iterations=1, k1_path=str(inputs.k1),
                                    skip_feasibility=True, process_lister=lambda: [])  # fmt: skip
    side = read_sidecar(res.run.path / "checkpoints" / "student.pt")
    tr = side["obs_schema"]["tracker"]
    assert (tr["lost_policy"], tr["lost_hard_stop_s"], tr["drop_after_s"]) == POLICY[:2] + (0.4,)
    assert side["perception"]["lost_policy"] == "reacquire"
    assert side["perception"]["drop_after_s"] == 0.4
    out = tmp_path / "deploy" / "student.pt"
    export_student(res.run.path, out, bench=False, k1_path=harness_k1_dir)
    plan = plan_student(out)
    assert plan.overrides["gates.lost_policy"] == "reacquire"
    assert plan.overrides["gates.lost_hard_stop_s"] == 6.0
    assert plan.overrides["gates.drop_after_s"] == 0.4


def test_the_checkpoints_policy_reaches_the_live_estimator(harness_k1_dir, tmp_path):
    """plan.overrides over the harness default (terminate) build a reacquire estimator."""
    from drone_fly.harness.config import load_config
    from drone_fly.harness.liftoff.gate_cv import OpenCvGateEstimator

    overrides = {"gates.lost_policy": "reacquire", "gates.lost_hard_stop_s": 6.0,
                 "gates.drop_after_s": 0.5, "gates.lost_timeout_s": 1.3}  # fmt: skip
    hc = load_config("liftoff", overrides=overrides)
    est = OpenCvGateEstimator.from_config(hc.section("gates"), SimpleNamespace(config=hc))
    assert (est.lost_policy, est.lost_hard_stop_s, est.drop_after_s, est.lost_timeout_s) == (
        "reacquire", 6.0, 0.5, 1.3)  # fmt: skip


def _export(harness_k1_dir, tmp_path):
    from drone_fly.training.export import export_student
    from tests.student_support import write_student_run

    run = write_student_run(tmp_path, k1_path=harness_k1_dir)
    out = tmp_path / "deploy" / "student.pt"
    export_student(run, out, bench=False, k1_path=harness_k1_dir)
    return out


def _edited(src: Path, tmp_path: Path, edit) -> Path:
    from drone_fly.student.checkpoint import load_export, save_export

    state, meta = load_export(src)
    meta = copy.deepcopy(meta)
    edit(meta["obs_schema"]["tracker"])
    return save_export(tmp_path / "edited.pt", state, meta)


@pytest.mark.parametrize(
    ("name", "edit"),
    [
        ("policy reacquire", lambda tr: tr.update(lost_policy="reacquire")),
        ("hard stop 4 s", lambda tr: tr.update(lost_hard_stop_s=4.0)),
        ("drop after 0.3 s", lambda tr: tr.update(drop_after_s=0.3)),
    ],
)
def test_a_consistent_policy_loads_and_is_applied(harness_k1_dir, tmp_path, name, edit):
    from drone_fly.harness.student import plan_student

    p = _edited(_export(harness_k1_dir, tmp_path), tmp_path, edit)
    tr = {}
    edit(tr)
    plan = plan_student(p)  # the schema is rebuilt from the same tracker block: consistent
    for k, v in tr.items():
        assert plan.overrides[f"gates.{k}"] == v, name


@pytest.mark.parametrize(
    "edit",
    [
        lambda tr: tr.pop("lost_policy"),  # an export from before the policy existed
        lambda tr: tr.pop("lost_hard_stop_s"),
        lambda tr: tr.pop("drop_after_s"),
        lambda tr: tr.update(lost_policy=None),
        lambda tr: tr.update(drop_after_s="0.5"),
    ],
)
def test_a_tracker_block_unlike_this_builds_schema_is_refused(harness_k1_dir, tmp_path, edit):
    from drone_fly.harness.student import plan_student

    p = _edited(_export(harness_k1_dir, tmp_path), tmp_path, edit)
    with pytest.raises(StudentSchemaError, match="differs.*tracker"):
        plan_student(p)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("lost_policy", "retry", r"lost_policy 'retry' not in"),
        ("lost_hard_stop_s", 0.0, r"lost_hard_stop_s 0\.0 must be > 0"),
        ("lost_hard_stop_s", -1.0, r"lost_hard_stop_s -1\.0 must be > 0"),
        ("drop_after_s", -0.1, r"drop_after_s -0\.1 must be >= 0"),
    ],
)
def test_a_bad_tracker_value_in_the_checkpoint_is_refused_at_load(harness_k1_dir, tmp_path,
                                                                  field, value, match):  # fmt: skip
    """Refused by the version guard at load (plan_student), before any estimator is built."""
    from drone_fly.harness.student import plan_student

    p = _edited(_export(harness_k1_dir, tmp_path), tmp_path,
                lambda tr: tr.update({field: value}))  # fmt: skip
    with pytest.raises(StudentSchemaError, match=match):
        plan_student(p)
