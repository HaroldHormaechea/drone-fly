"""UC70 AC4 / AC7: the student report (``train eval <student run> [--baseline <mlp run>]``).

The report engine (``training.distill.evaluate_student_run``) is checked with the per-track
evaluation stubbed (known completions): the modes (student noise = headline, student exact,
teacher exact on the same tracks and seeds, the MLP baseline), the per-track teacher ratio, the
bounds and gap attribution, outcome shares incl. ``gate_lost``, the coverage strata, the targets
and the feasibility banner, in JSON and Markdown. The real per-track engine
(``evaluate_policy`` on ``StudentCourseEnv``) is run once on short hand-built courses to check its
shape. Hermetic, CPU; the student / baseline runs are tiny DAgger runs on procedural courses.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly import drones
from drone_fly.controller.student import MLPStudent
from drone_fly.sim.multi_course import OUTCOMES
from drone_fly.training import distill
from drone_fly.training.config import load_raw
from tests.oracle_support import age_files, write_teacher_run
from tests.student_support import (
    TRAIN_TIMING,
    tiny_pool,
    tiny_student_config,
    write_07,
    write_noise_model,
)

TRACKS = ("proc-a", "proc-b")
#: (kind, mode) -> per-track completions
TABLE = {
    ("teacher", "exact"): (1.0, 0.5),
    ("k1", "exact"): (0.8, 0.4),
    ("k1", "noise"): (0.6, 0.2),
    ("mlp", "exact"): (0.9, 0.5),
    ("mlp", "noise"): (0.7, 0.3),
}


def fake_evaluate_policy(policy, cfg, profile, pool, model, dev, *, mode, episodes, seed,
                         teacher=False, **kw):  # fmt: skip
    kind = "teacher" if teacher else ("mlp" if isinstance(policy, MLPStudent) else "k1")
    comp = TABLE[(kind, mode)]
    fake_evaluate_policy.calls.append((kind, mode, episodes, seed))
    per = [{"id": t, "map": "procedural", "source": "procedural", "episodes": episodes,
            "completion": c, "route_fraction": c, "stratum": ("covered", "proxy")[i],
            "outcomes": {"completed": c, "gate_lost": 1 - c}}
           for i, (t, c) in enumerate(zip(TRACKS, comp, strict=True))]  # fmt: skip
    return {"mode": mode, "teacher": teacher, "episodes": episodes, "seed": seed,
            "macro_completion": float(np.mean(comp)), "per_track": per,
            "strata": {"covered": {"tracks": 1, "completion": comp[0]},
                       "proxy": {"tracks": 1, "completion": comp[1]}},
            "outcomes": {"completed": float(np.mean(comp)),
                         "gate_lost": 1 - float(np.mean(comp))}}  # fmt: skip


@pytest.fixture(scope="module")
def runs(tmp_path_factory, harness_k1_dir):
    root = tmp_path_factory.mktemp("uc70-eval")
    k07 = write_07(root / "07" / "model_readout.pt", harness_k1_dir)
    teacher = write_teacher_run(root / "runs")
    age_files(teacher)
    noise = write_noise_model(root / "noise_model.yaml", a=0.5, timing=TRAIN_TIMING)
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(distill, "eval_pool", lambda *a, **k: tiny_pool(2))
        for actor in ("k1-student", "mlp-student"):
            cfg = tiny_student_config(k07, actor=actor)
            res = distill.train_student(cfg, drones.load(cfg.drone), teacher=teacher, noise=noise,
                                        run_root=root / "out", run_id=actor, max_iterations=1,
                                        k1_path=str(harness_k1_dir), skip_feasibility=True,
                                        process_lister=lambda: [])  # fmt: skip
            out[actor] = res.run.path
    feas = {"held_out_completion": 0.44, "infeasible": True, "banner": distill.INFEASIBLE_BANNER}
    (out["k1-student"] / "eval" / "feasibility.json").write_text(json.dumps(feas), "utf-8")
    return SimpleNamespace(k1=out["k1-student"], mlp=out["mlp-student"], k1_path=harness_k1_dir)


@pytest.fixture
def report(runs, monkeypatch):
    fake_evaluate_policy.calls = []
    monkeypatch.setattr(distill, "eval_pool", lambda *a, **k: tiny_pool(2))
    monkeypatch.setattr(distill, "evaluate_policy", fake_evaluate_policy)
    return distill.evaluate_student_run(runs.k1, baseline=runs.mlp, device="cpu",
                                        k1_path=runs.k1_path)  # fmt: skip


def test_is_student_run(runs, tmp_path):
    assert distill.is_student_run(runs.k1)
    assert distill.is_student_run(runs.k1 / "checkpoints" / "student.pt")
    assert not distill.is_student_run(tmp_path)


def test_modes_seeds_and_bounds(report):
    calls = fake_evaluate_policy.calls
    assert sorted((k, m) for k, m, _, _ in calls) == sorted(TABLE)
    assert {(e, s) for _, _, e, s in calls} == {(32, 1234)}  # same tracks, episodes and seeds
    m = report["macro"]
    assert m == pytest.approx({"student_noise": 0.4, "student_exact": 0.6, "teacher_exact": 0.75,
                               "baseline_noise": 0.5, "baseline_exact": 0.7})  # fmt: skip
    assert report["teacher_ratio"] == pytest.approx(0.4 / 0.75)
    g = report["gaps"]
    assert g["teacher_limit"] == pytest.approx(0.25)
    assert g["capacity_gap"] == pytest.approx((0.7 - 0.6) / 0.75)
    assert g["perception_gap"] == pytest.approx((0.6 - 0.4) / 0.75)
    t = report["targets"]
    assert t["completion"] == 0.70 and t["teacher_ratio"] == 0.85
    assert t["meets_completion"] is False and t["meets_teacher_ratio"] is False
    k = float(load_raw("student-liftoff")["student"]["world_scale"])  # the runs' k (ADR 0009)
    assert report["actor"] == "k1-student" and report["world_scale"] == k
    assert report["selectors"] == ["liftoff:held-out"]


def test_per_track_rows_strata_outcomes_and_banner(report):
    rows = {r["id"]: r for r in report["per_track"]}
    assert rows["proc-a"]["student_noise"] == 0.6 and rows["proc-a"]["student_exact"] == 0.8
    assert rows["proc-a"]["teacher_exact"] == 1.0 and rows["proc-a"]["ratio"] == pytest.approx(0.6)
    assert rows["proc-b"]["ratio"] == pytest.approx(0.4) and rows["proc-b"]["stratum"] == "proxy"
    assert set(report["strata"]) == {"student_noise", "student_exact", "teacher_exact",
                                     "baseline_noise", "baseline_exact"}  # fmt: skip
    assert report["gate_lost_fraction_noise"] == pytest.approx(0.6)
    assert report["banner"] == distill.INFEASIBLE_BANNER
    assert report["feasibility"]["held_out_completion"] == 0.44
    assert report["lost_timeout_s"] == pytest.approx(1.0)
    assert report["noise_model"]["detector_id"] == "synthetic-deadbeef"


def test_report_files(report, runs):
    js = runs.k1 / "eval" / "student-held-out.json"
    md = runs.k1 / "eval" / "student-held-out.md"
    assert report["written"] == str(js) and js.is_file() and md.is_file()
    assert json.loads(js.read_text("utf-8"))["macro"]["student_noise"] == pytest.approx(0.4)
    text = md.read_text("utf-8")
    assert distill.INFEASIBLE_BANNER in text
    assert "**Headline (noise): 40.0 %**" in text and "NOT met" in text
    for bound in ("teacher exact", "baseline exact", "baseline noise", "student exact",
                  "student noise"):  # fmt: skip
        assert f"| {bound} |" in text
    assert "| proc-a | covered | 60.0 % | 80.0 % | 100.0 % | 60.0 % |" in text
    assert "K1 capacity (MLP exact - K1 exact) 13.3 %" in text
    assert "gate_lost share (noise) 60.0 %" in text


def test_report_without_a_baseline(runs, monkeypatch):
    fake_evaluate_policy.calls = []
    monkeypatch.setattr(distill, "eval_pool", lambda *a, **k: tiny_pool(2))
    monkeypatch.setattr(distill, "evaluate_policy", fake_evaluate_policy)
    rep = distill.evaluate_student_run(runs.k1, device="cpu", k1_path=runs.k1_path, write=False)
    assert "baseline_noise" not in rep["macro"] and rep["gaps"]["capacity_gap"] is None
    assert "written" not in rep
    assert "K1 capacity (MLP exact - K1 exact) -" in distill.student_markdown(rep)


def test_cli_eval_routes_a_student_run(runs, monkeypatch):
    from drone_fly.training import evaluate

    seen = {}
    monkeypatch.setattr(evaluate, "cmd_teacher_eval", lambda args: seen.setdefault("a", args) and 0)
    from drone_fly.cli import main as cli

    assert cli.main(["train", "eval", str(runs.k1), "--baseline", str(runs.mlp)]) == 0
    assert seen["a"].baseline == str(runs.mlp)


# ----------------------------------------------------------------------------- the real engine


def test_evaluate_policy_engine_shape(runs, tmp_path):
    """The real per-track engine on two short hand-built courses (a covered and a proxy one)."""
    from dataclasses import replace

    from drone_fly.sim.noise_model import NoiseModel
    from drone_fly.sim.pool import PoolGroup
    from drone_fly.tracks.loader import LoadRecord, Pool
    from tests.oracle_support import make_course, opening

    s = distill.load_student_run(runs.k1, k1_path=runs.k1_path)
    arch = opening((5.0, 0.0, 1.5), 0.0)
    arch.cls = "inflatable_arch"
    courses = [make_course([arch], box=False, time_limit_s=1.0, course_id="cov"),
               make_course([opening((5.0, 0.0, 1.5), 0.0)], box=False, time_limit_s=1.0,
                           course_id="prx")]  # fmt: skip
    for c in courses:
        c.meta.update(map="SyntheticMap", source="synthetic")
    pool = Pool(courses, [PoolGroup("g", [0, 1], 1.0)], "eval", {}, LoadRecord())
    cfg = s["cfg"]
    cfg = replace(cfg, base=replace(cfg.base, device="cpu"))
    model = NoiseModel.load(runs.k1 / "inputs" / "noise_model.yaml")
    res = distill.evaluate_policy(s["actor"].eval(), cfg, drones.load(cfg.drone), pool, model,
                                  "cpu", mode="noise", episodes=2, seed=1234)  # fmt: skip
    assert res["mode"] == "noise" and res["episodes"] == 2 and res["seed"] == 1234
    tracks = {t["id"]: t for t in res["per_track"]}
    assert set(tracks) == {"cov", "prx"}
    assert tracks["cov"]["stratum"] == "covered" and tracks["prx"]["stratum"] == "proxy"
    assert all(t["episodes"] == 2 and 0.0 <= t["completion"] <= 1.0 for t in tracks.values())
    assert set(res["outcomes"]) <= set(OUTCOMES) and sum(res["outcomes"].values()) == pytest.approx(
        1.0
    )
    assert set(res["strata"]) == {"covered", "proxy"}
    # the teacher on the same pool: exact teacher observations
    teach = distill.evaluate_policy(_teacher(s), cfg,
                                    drones.load(cfg.drone), pool, model, "cpu", mode="exact",
                                    episodes=1, seed=1234, teacher=True)  # fmt: skip
    assert teach["teacher"] is True and len(teach["per_track"]) == 2


def _teacher(s):
    from drone_fly.training.evaluate import load_teacher

    return load_teacher(s["teacher"].checkpoint, "cpu")[0]
