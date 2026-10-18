"""UC70 AC7 (orchestrator ruling + challenger Minor 1): the NON-blocking feasibility stage.

The teacher is evaluated on held-out and val under exact observations; ``eval/feasibility.json``
is always written. Held-out completion below 0.70 / 0.85 logs a WARNING, writes the banner "AC7
absolute target infeasible by construction" and switches the operative gates, and training
CONTINUES. ``--feasibility-only`` runs the stage alone and exits 0. The teacher evaluation itself
is replaced by a stub (its engine is UC68's); the training run is the tiny CPU one.
"""

from __future__ import annotations

import json

import pytest

from drone_fly import drones
from drone_fly.training import distill, evaluate
from drone_fly.training.config import FEASIBILITY_THRESHOLD
from tests.student_support import tiny_pool, tiny_student_config, train_inputs
from tests.test_training_student import train

pytestmark = pytest.mark.usefixtures("tiny_val")


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, harness_k1_dir):
    return train_inputs(tmp_path_factory.mktemp("uc70-feas"), harness_k1_dir)


@pytest.fixture
def tiny_val(monkeypatch):
    monkeypatch.setattr(distill, "eval_pool", lambda *a, **k: tiny_pool(2))


def stub_teacher_eval(monkeypatch, held: float, val: float = 0.9):
    calls = []

    def fake(run_dir, selectors, **kw):
        calls.append({"selectors": selectors, **kw})
        macro = held if selectors == ["liftoff:held-out"] else val
        return {"modes": {"randomized": {
            "macro_completion": macro, "tracks": 2,
            "per_track": [{"id": "t1", "completion": macro}, {"id": "t2", "completion": macro}],
        }}}  # fmt: skip

    monkeypatch.setattr(evaluate, "evaluate_teacher", fake)
    return calls


def test_threshold_is_the_ratio_of_the_two_targets():
    assert FEASIBILITY_THRESHOLD == pytest.approx(0.70 / 0.85)


def test_infeasible_teacher_flags_warns_and_training_continues(inputs, tmp_path, monkeypatch,
                                                               caplog):  # fmt: skip
    calls = stub_teacher_eval(monkeypatch, held=0.44)
    with caplog.at_level("WARNING"):
        res = train(inputs, tmp_path, skip_feasibility=False, max_iterations=1)
    assert res.steps > 0 and (res.run.path / "checkpoints" / "student.pt").is_file()
    f = json.loads((res.run.path / "eval" / "feasibility.json").read_text("utf-8"))
    assert f["infeasible"] is True and f["banner"] == distill.INFEASIBLE_BANNER
    assert f["held_out_completion"] == 0.44 and f["liftoff:val"]["macro_completion"] == 0.9
    assert f["operative_gates"][0].startswith(">= 85 % of the teacher")
    assert f["mode"] == "randomized" and f["episodes"] == 32 and f["seed"] == 1234
    assert distill.INFEASIBLE_BANNER in caplog.text and "training continues" in caplog.text
    assert [c["selectors"] for c in calls] == [["liftoff:held-out"], ["liftoff:val"]]
    assert all(c["modes"] == ("randomized",) and c["write"] is False for c in calls)


def test_feasible_teacher_keeps_the_absolute_gates(inputs, tmp_path, monkeypatch):
    stub_teacher_eval(monkeypatch, held=0.95)
    res = train(inputs, tmp_path, skip_feasibility=False, max_iterations=1)
    f = json.loads((res.run.path / "eval" / "feasibility.json").read_text("utf-8"))
    assert f["infeasible"] is False and f["banner"] is None
    assert f["operative_gates"] == [">= 70 % held-out completion", ">= 85 % of the teacher's"]


def test_feasibility_only_stops_after_the_stage(inputs, tmp_path, monkeypatch):
    stub_teacher_eval(monkeypatch, held=0.3)
    res = train(inputs, tmp_path, skip_feasibility=False, feasibility_only=True)
    assert res.steps == 0 and res.iterations == 0
    assert res.checkpoint.name == "feasibility.json" and res.checkpoint.is_file()
    assert not (res.run.path / "checkpoints" / "student.pt").exists()


def test_cli_feasibility_only_exits_zero(inputs, tmp_path, monkeypatch, capsys):
    import yaml

    from drone_fly.cli import main as cli

    stub_teacher_eval(monkeypatch, held=0.3)
    cfg = tiny_student_config(inputs.k07)  # validates the overrides used below
    raw = {"extends": "student-liftoff", "n_envs": 8, "device": "cpu",
           "tracks": {"mix": [{"select": "procedural:proc-liftoff", "weight": 1.0}]},
           "student": {"warm_start": str(inputs.k07)}}  # fmt: skip
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), "utf-8")
    monkeypatch.setattr(distill, "contention_conditions", lambda *a, **k: [])
    rc = cli.main(["train", "student", "--teacher", str(inputs.teacher), "--noise",
                   str(inputs.noise), "--config", str(p), "--feasibility-only",
                   "--output-root", str(tmp_path / "out"),
                   "--k1-path", str(inputs.k1)])  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0 and "feasibility.json" in out, out
    from drone_fly.training.config import load_raw

    assert cfg.student.world_scale == float(load_raw("student-liftoff")["student"]["world_scale"])
    found = list((tmp_path / "out").rglob("feasibility.json"))
    assert len(found) == 1 and json.loads(found[0].read_text())["infeasible"] is True


def test_resume_does_not_rerun_the_stage(inputs, tmp_path, monkeypatch):
    calls = stub_teacher_eval(monkeypatch, held=0.5)
    res = train(inputs, tmp_path, skip_feasibility=False, max_iterations=1)
    n = len(calls)
    cfg = tiny_student_config(inputs.k07)
    distill.train_student(cfg, drones.load(cfg.drone), teacher=inputs.teacher, noise=inputs.noise,
                          resume=res.run.path, max_iterations=2, k1_path=str(inputs.k1),
                          process_lister=lambda: [])  # fmt: skip
    assert len(calls) == n
