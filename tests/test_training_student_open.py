"""UC77 AC3 / AC4 / AC5 / AC6 (d) (e): the obstacle-free student run (``student-liftoff-open``).

* (d) A tiny CPU DAgger run with ``tracks.obstacles: none`` (8 envs, 2 iterations, procedural mix,
  validation every iteration, one ``track_evals`` selector plus one that resolves to no courses):
  the training pool and every eval pool are obstacle-free, ``val_history.jsonl`` rows carry
  ``tracks`` (the empty selector skipped, N3), the sidecar records ``course_obstacles: none`` and
  ``final_report`` evaluates held-out + every track selector after the run (stubbed report,
  failures skipped). A default run keeps none of this; an interrupted run writes no final report.
* (e) The obs schema stays ``student-obs`` v3 / DIM 65 in the sidecar.
* AC4: the feasibility stage forwards and records the mode and never gates the run.
* AC5: ``evaluate_student_run`` resolves the mode from the run (or an override), records it,
  suffixes overridden report files, refuses an empty pool; ``train eval --obstacles`` is forwarded
  and the CLI prints the student-noise outcome breakdown.

CPU, hermetic: the fixture connectome, a tiny MLP teacher run and procedural courses only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import yaml

from drone_fly import drones
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.sim.action import read_sidecar
from drone_fly.tracks.catalogue import TrackDataError
from drone_fly.training import distill, evaluate
from tests.student_support import tiny_student_config, train_inputs
from tests.test_training_eval_student import fake_evaluate_policy

TRACK_SEL = "procedural:proc-liftoff"
EMPTY_SEL = "liftoff:no-such-courses"


def obstacle_pool(selectors, mode, n: int = 2):
    """The first ``n`` courses of ``selectors`` in obstacle mode ``mode`` (an eval pool)."""
    from drone_fly.sim.pool import PoolGroup
    from drone_fly.tracks.loader import build_pool

    p = build_pool([{"select": s} for s in selectors], "eval", obstacles=mode)
    return replace(p, courses=p.courses[:n], groups=[PoolGroup("tiny", list(range(n)), 1.0)])


class EvalPoolSpy:
    """Stands in for ``distill.eval_pool``: records the resolved mode, builds a tiny pool."""

    def __init__(self):
        self.calls = []

    def __call__(self, cfg, conv, selectors, model, obstacles=None):
        mode = cfg.base.tracks.obstacles if obstacles is None else obstacles
        self.calls.append((tuple(selectors), mode))
        if EMPTY_SEL in selectors:
            return SimpleNamespace(courses=[], groups=[])
        return obstacle_pool([TRACK_SEL], mode)


class BuildPoolSpy:
    def __init__(self, real):
        self.real = real
        self.calls = []

    def __call__(self, mix, purpose, **kw):
        pool = self.real(mix, purpose, **kw)
        # summarised now: training releases each course's host geometry once packed
        summary = [SimpleNamespace(id=c.id, prims=len(c.prims), harness=len(c.harness),
                                   flags=list(c.meta["flags"]), stats=dict(c.stats))
                   for c in pool.courses]  # fmt: skip
        self.calls.append((purpose, kw.get("obstacles", "all"), summary))
        return pool


def fake_report(calls, fail=(EMPTY_SEL,)):
    def fake(run_path, *, selectors, device=None, k1_path=None, **kw):
        calls.append(tuple(selectors))
        if selectors[0] in fail:
            raise HarnessConfigError(f"no evaluation courses for {list(selectors)}")
        if selectors[0] == "liftoff:broken":
            raise TrackDataError("broken track data")
        return {"obstacles": "none", "macro": {"student_noise": 0.5, "student_exact": 0.6,
                                               "teacher_exact": 0.9},
                "outcomes": {"student_noise": {"completed": 0.5, "floor": 0.5}}}  # fmt: skip

    return fake


def open_config(k07, **distill_over):
    return tiny_student_config(
        k07, tracks={"obstacles": "none"},
        distill={"val_every_iters": 1, "track_evals": [TRACK_SEL, EMPTY_SEL],
                 "final_report": True, **distill_over},
    )  # fmt: skip


def run_train(inputs, root, cfg, **kw):
    args = dict(teacher=inputs.teacher, noise=inputs.noise, run_root=root / "out",
                max_iterations=2, k1_path=str(inputs.k1), skip_feasibility=True,
                process_lister=lambda: [])  # fmt: skip
    args.update(kw)
    return distill.train_student(cfg, drones.load(cfg.drone), **args)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, harness_k1_dir):
    return train_inputs(tmp_path_factory.mktemp("uc77-train"), harness_k1_dir)


@pytest.fixture(scope="module")
def open_run(inputs, tmp_path_factory):
    from drone_fly.tracks import loader

    root = tmp_path_factory.mktemp("uc77-open")
    ev, reports = EvalPoolSpy(), []
    bp = BuildPoolSpy(loader.build_pool)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(distill, "eval_pool", ev)
        mp.setattr(loader, "build_pool", bp)
        mp.setattr(distill, "evaluate_student_run", fake_report(reports))
        cfg = open_config(inputs.k07)
        res = run_train(inputs, root, cfg)
    return SimpleNamespace(res=res, path=res.run.path, cfg=cfg, eval_calls=ev.calls,
                           pool_calls=bp.calls, reports=reports)  # fmt: skip


@pytest.fixture(scope="module")
def default_run(inputs, tmp_path_factory):
    from drone_fly.tracks import loader

    root = tmp_path_factory.mktemp("uc77-default")
    ev = EvalPoolSpy()
    bp = BuildPoolSpy(loader.build_pool)

    def no_report(*a, **k):
        raise AssertionError("final_report is off by default")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(distill, "eval_pool", ev)
        mp.setattr(loader, "build_pool", bp)
        mp.setattr(distill, "evaluate_student_run", no_report)
        cfg = tiny_student_config(inputs.k07, distill={"val_every_iters": 1})
        res = run_train(inputs, root, cfg, max_iterations=1)
    return SimpleNamespace(res=res, path=res.run.path, eval_calls=ev.calls, pool_calls=bp.calls)


def _val_rows(run):
    return [json.loads(x) for x in (run / "eval" / "val_history.jsonl").read_text().splitlines()]


# ----------------------------------------------------------------------------- the run (d)


def test_open_run_completes(open_run):
    assert open_run.res.iterations == 2 and open_run.res.steps == 2 * 4 * 8
    assert (open_run.path / "checkpoints" / "student.pt").is_file()


def test_open_run_trains_on_obstacle_free_courses(open_run):
    train_calls = [(m, p) for purpose, m, p in open_run.pool_calls if purpose == "train"]
    assert len(train_calls) == 1
    mode, courses = train_calls[0]
    assert mode == "none" and courses
    for c in courses:
        assert c.prims == 0 and c.harness == 0, c.id
        assert "obstacle-free" in c.flags and c.stats["obstacles"] == "none"
    assert any(c.stats["obstacles_removed"] for c in courses)


def test_open_run_validates_and_track_evals_obstacle_free(open_run):
    sels = {s for s, _ in open_run.eval_calls}
    assert {("liftoff:val",), (TRACK_SEL,), (EMPTY_SEL,)} <= sels
    assert {m for _, m in open_run.eval_calls} == {"none"}


def test_val_history_rows_carry_the_track_evals(open_run):
    rows = _val_rows(open_run.path)
    assert [r["it"] for r in rows] == [1, 2]
    for r in rows:
        assert set(r["tracks"]) == {TRACK_SEL}  # the empty selector is skipped (N3)
        t = r["tracks"][TRACK_SEL]
        assert 0.0 <= t["macro_completion"] <= 1.0
        assert t["outcomes"] and sum(t["outcomes"].values()) == pytest.approx(1.0, abs=1e-6)


def test_selection_stays_on_val(open_run):
    sel = json.loads((open_run.path / "eval" / "student_selection.json").read_text("utf-8"))
    assert sel["selectors"] == ["liftoff:val"]
    best = max(_val_rows(open_run.path), key=lambda r: r["macro_completion"])
    assert sel["macro_completion"] == pytest.approx(best["macro_completion"])


def test_open_sidecar_and_schema(open_run):
    from drone_fly.sim import student_obs

    side = read_sidecar(open_run.path / "checkpoints" / "student.pt")
    assert side["course_obstacles"] == "none"
    sch = side["obs_schema"]
    assert (sch["name"], sch["version"], sch["dim"]) == ("student-obs", 3, 65)
    assert side["extra_dim"] == 65 == student_obs.DIM and student_obs.VERSION == 3
    assert side["contract_version"] == 2 and side["actor"] == "k1-student"


def test_open_run_snapshot(open_run):
    snap = yaml.safe_load((open_run.path / "config.yaml").read_text("utf-8"))
    assert snap["tracks"]["obstacles"] == "none"
    assert snap["distill"]["track_evals"] == [TRACK_SEL, EMPTY_SEL]
    assert snap["distill"]["final_report"] is True


def test_final_report_after_the_run(open_run):
    assert open_run.reports == [("liftoff:held-out",), (TRACK_SEL,), (EMPTY_SEL,)]


def test_default_run_keeps_the_old_behaviour(default_run):
    side = read_sidecar(default_run.path / "checkpoints" / "student.pt")
    assert "course_obstacles" not in side
    assert all("tracks" not in r for r in _val_rows(default_run.path))
    assert {m for _, m in default_run.eval_calls} == {"all"}
    assert {m for purpose, m, _ in default_run.pool_calls if purpose == "train"} == {"all"}
    snap = yaml.safe_load((default_run.path / "config.yaml").read_text("utf-8"))
    assert "obstacles" not in snap["tracks"]
    assert "track_evals" not in snap["distill"] and "final_report" not in snap["distill"]


def test_interrupted_run_writes_no_final_report(inputs, tmp_path, monkeypatch):
    real_beta = distill.beta_at
    state = {"run": None}

    def stop_after_first(frac, d):
        if state["run"] is not None:
            (state["run"] / "STOP").write_text("")
        return real_beta(frac, d)

    class Grab(distill.StudentRun):
        def __init__(self, run, *a, **k):
            state["run"] = run.path
            super().__init__(run, *a, **k)

    reports = []
    monkeypatch.setattr(distill, "beta_at", stop_after_first)
    monkeypatch.setattr(distill, "StudentRun", Grab)
    monkeypatch.setattr(distill, "eval_pool", EvalPoolSpy())
    monkeypatch.setattr(distill, "evaluate_student_run", fake_report(reports))
    res = run_train(inputs, tmp_path, open_config(inputs.k07), max_iterations=5)
    assert res.iterations == 1 and reports == []


# ----------------------------------------------------------------------------- N3 / final_report


def test_track_eval_pools_skip_an_empty_selector(monkeypatch, caplog):
    cfg = open_config("unused.pt")
    monkeypatch.setattr(distill, "eval_pool", EvalPoolSpy())
    with caplog.at_level(logging.WARNING):
        pools = distill.track_eval_pools(cfg, None, None)
    assert set(pools) == {TRACK_SEL}
    assert f"track eval {EMPTY_SEL} has no courses: skipped" in caplog.text


def test_final_report_skips_failures_and_dedupes(monkeypatch, caplog, tmp_path):
    cfg = open_config("unused.pt", track_evals=["liftoff:held-out", TRACK_SEL, "liftoff:broken",
                                                EMPTY_SEL])  # fmt: skip
    calls = []
    monkeypatch.setattr(distill, "evaluate_student_run", fake_report(calls))
    with caplog.at_level(logging.WARNING):
        out = distill.final_report(tmp_path, cfg, device="cpu")
    assert calls == [("liftoff:held-out",), (TRACK_SEL,), ("liftoff:broken",), (EMPTY_SEL,)]
    assert set(out) == {"liftoff:held-out", TRACK_SEL}
    assert "final report liftoff:broken skipped" in caplog.text
    assert f"final report {EMPTY_SEL} skipped" in caplog.text


# ----------------------------------------------------------------------------- AC5: the report


@pytest.fixture
def stub_engine(monkeypatch):
    fake_evaluate_policy.calls = []
    ev = EvalPoolSpy()
    monkeypatch.setattr(distill, "eval_pool", ev)
    monkeypatch.setattr(distill, "evaluate_policy", fake_evaluate_policy)
    return ev


def test_report_uses_and_records_the_run_mode(open_run, inputs, stub_engine):
    res = distill.evaluate_student_run(open_run.path, selectors=("liftoff:liftoff-arena-04",),
                                       device="cpu", k1_path=inputs.k1)  # fmt: skip
    assert res["obstacles"] == "none" and stub_engine.calls == [(("liftoff:liftoff-arena-04",),
                                                                 "none")]  # fmt: skip
    js = open_run.path / "eval" / "student-liftoff-arena-04.json"
    assert res["written"] == str(js) and json.loads(js.read_text("utf-8"))["obstacles"] == "none"
    md = (open_run.path / "eval" / "student-liftoff-arena-04.md").read_text("utf-8")
    assert "course obstacles `none`" in md
    assert res["outcomes"]["student_noise"]  # the outcome breakdown


def test_report_override_is_recorded_and_suffixed(open_run, inputs, stub_engine):
    res = distill.evaluate_student_run(open_run.path, selectors=("liftoff:liftoff-arena-04",),
                                       device="cpu", k1_path=inputs.k1,
                                       obstacles="all")  # fmt: skip
    assert res["obstacles"] == "all" and stub_engine.calls[0][1] == "all"
    js = open_run.path / "eval" / "student-liftoff-arena-04-obstacles-all.json"
    assert res["written"] == str(js) and json.loads(js.read_text("utf-8"))["obstacles"] == "all"
    md = open_run.path / "eval" / "student-liftoff-arena-04-obstacles-all.md"
    assert "course obstacles `all`" in md.read_text("utf-8")


def test_report_on_a_default_run_with_none_is_suffixed(default_run, inputs, stub_engine):
    res = distill.evaluate_student_run(default_run.path, selectors=("liftoff:held-out",),
                                       device="cpu", k1_path=inputs.k1, obstacles="none",
                                       write=True)  # fmt: skip
    assert res["obstacles"] == "none"
    assert (default_run.path / "eval" / "student-held-out-obstacles-none.json").is_file()


def test_report_refuses_an_empty_pool(open_run, inputs, stub_engine):
    with pytest.raises(HarnessConfigError, match="no evaluation courses"):
        distill.evaluate_student_run(open_run.path, selectors=(EMPTY_SEL,), device="cpu",
                                     k1_path=inputs.k1, write=False)  # fmt: skip


def test_student_markdown_of_an_old_report_says_all(open_run, inputs, stub_engine):
    rep = distill.evaluate_student_run(open_run.path, selectors=("liftoff:held-out",),
                                       device="cpu", k1_path=inputs.k1, write=False)  # fmt: skip
    assert "course obstacles `none`" in distill.student_markdown(rep)
    rep.pop("obstacles")  # a pre-UC77 report carries no key
    assert "course obstacles `all`" in distill.student_markdown(rep)


@pytest.mark.parametrize(
    ("argv", "want"),
    [([], None), (["--obstacles", "all"], "all"), (["--obstacles", "none"], "none")],
)
def test_cli_student_eval_forwards_obstacles_and_prints_outcomes(open_run, monkeypatch, capsys,
                                                                 argv, want):  # fmt: skip
    from drone_fly.cli import main as cli

    seen = {}

    def fake(run, **kw):
        seen.update(kw)
        return {"obstacles": kw.get("obstacles") or "none", "teacher_ratio": 0.5,
                "macro": {"student_noise": 0.4, "student_exact": 0.6, "teacher_exact": 0.8},
                "outcomes": {"student_noise": {"completed": 0.4, "floor": 0.6}},
                "written": "x.json"}  # fmt: skip

    monkeypatch.setattr(distill, "evaluate_student_run", fake)
    assert cli.main(["train", "eval", str(open_run.path), "--tracks",
                     "liftoff:liftoff-arena-04", *argv]) == 0  # fmt: skip
    assert seen["obstacles"] == want and seen["selectors"] == ("liftoff:liftoff-arena-04",)
    out = capsys.readouterr().out
    assert f"(obstacles {want or 'none'})" in out
    assert "student noise outcomes: completed 40.0 %, floor 60.0 %" in out


# ----------------------------------------------------------------------------- AC4 feasibility


def test_feasibility_forwards_records_and_never_gates(inputs, tmp_path, monkeypatch):
    calls = []

    def fake(run_dir, selectors, **kw):
        calls.append({"selectors": selectors, **kw})
        return {"modes": {"randomized": {
            "macro_completion": 0.05, "tracks": 1, "per_track": [{"id": "t", "completion": 0.05}],
        }}}  # fmt: skip

    monkeypatch.setattr(evaluate, "evaluate_teacher", fake)
    monkeypatch.setattr(distill, "eval_pool", EvalPoolSpy())
    monkeypatch.setattr(distill, "evaluate_student_run", fake_report([]))
    res = run_train(inputs, tmp_path, open_config(inputs.k07), max_iterations=1,
                    skip_feasibility=False)  # fmt: skip
    assert res.steps > 0 and (res.run.path / "checkpoints" / "student.pt").is_file()
    assert calls and all(c["obstacles"] == "none" for c in calls)
    f = json.loads((res.run.path / "eval" / "feasibility.json").read_text("utf-8"))
    assert f["obstacles"] == "none" and f["infeasible"] is True  # logged, training continued


def test_obstacle_free_teacher_obs_give_valid_teacher_actions(inputs):
    """AC4: with the privileged obstacle block emptied the (unchanged) teacher still acts."""
    from drone_fly.sim.teacher_obs import block_slice, build_teacher_obs
    from drone_fly.training.evaluate import load_teacher
    from tests.tracks_support import make_env

    actor, side = load_teacher(inputs.teacher / "checkpoints" / "teacher.pt", "cpu")
    version = int(side["obs_schema"]["version"])
    pool = obstacle_pool([TRACK_SEL], "none", n=2)
    env = make_env(pool.courses, n_envs=4, start={"ground_fraction": 0.0})
    env.reset_all(torch.tensor([0, 1, 0, 1]))
    for _ in range(5):
        obs = build_teacher_obs(env, version)
        assert torch.isfinite(obs).all()
        assert (obs[:, 22:30] == 0).all()  # harness obstacle_vision: nothing to see
        ob = obs[:, block_slice("obstacles")].reshape(4, 8, 8)
        assert (ob[:, 1:, 7] == 0).all()  # at most the floor row
        with torch.no_grad():
            act = actor.mean_action(obs)
        assert act.shape == (4, 4) and torch.isfinite(act).all()
        env.step(act.to(env.dtype))
