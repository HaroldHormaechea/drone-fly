"""UC70 AC4 / AC6: the no-swap rule — the deployable export is always the K1 student.

``train export`` refuses an ``mlp-student`` run (``ExportError``, exit 2) unless
``--baseline-export``, which labels it ``deployable: false`` / ``role: baseline``; the harness then
refuses that export unless ``--allow-baseline``. Synthetic run folders, CPU.
"""

from __future__ import annotations

import json

import pytest

from drone_fly.harness.errors import ExportError, StudentSchemaError
from drone_fly.harness.student import plan_student
from drone_fly.student.checkpoint import load_export
from drone_fly.training.export import export_student
from tests.student_support import write_student_run


@pytest.fixture
def mlp_run(tmp_path):
    return write_student_run(tmp_path, actor="mlp-student", name="mlp-run")


def test_mlp_export_is_refused_without_the_flag(mlp_run, tmp_path):
    with pytest.raises(ExportError, match="--baseline-export"):
        export_student(mlp_run, tmp_path / "mlp.pt", bench=False)
    assert not (tmp_path / "mlp.pt").exists()


def test_cli_export_exits_2_for_the_baseline(mlp_run, tmp_path, capsys):
    from drone_fly.cli import main as cli

    out_pt = str(tmp_path / "mlp.pt")
    rc = cli.main(["train", "export", str(mlp_run), "--out", out_pt, "--no-bench"])
    assert rc == 2 and "refused" in capsys.readouterr().out
    rc = cli.main(["train", "export", str(mlp_run), "--out", out_pt, "--no-bench",
                   "--baseline-export"])  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0 and "exported baseline student" in out and "not benchmarked" in out


def test_baseline_export_is_labelled_not_deployable(mlp_run, tmp_path):
    meta = export_student(mlp_run, tmp_path / "mlp.pt", baseline_export=True, bench=False)
    assert meta["deployable"] is False and meta["role"] == "baseline" and meta["k1"] is None
    _, m = load_export(tmp_path / "mlp.pt")
    assert m["deployable"] is False and m["role"] == "baseline"


def test_harness_refuses_a_baseline_without_allow_baseline(mlp_run, tmp_path):
    export_student(mlp_run, tmp_path / "mlp.pt", baseline_export=True, bench=False)
    with pytest.raises(StudentSchemaError, match="--allow-baseline"):
        plan_student(tmp_path / "mlp.pt")
    plan = plan_student(tmp_path / "mlp.pt", allow_baseline=True)
    assert plan is not None and plan.actor == "mlp-student"


def test_k1_export_is_deployable_and_needs_no_flag(tmp_path, harness_k1_dir):
    run = write_student_run(tmp_path, k1_path=harness_k1_dir)
    meta = export_student(run, tmp_path / "k1.pt", bench=False, k1_path=harness_k1_dir)
    assert meta["deployable"] is True and meta["role"] == "deployable"
    assert plan_student(tmp_path / "k1.pt") is not None


def test_a_non_student_checkpoint_is_not_exported(tmp_path):
    run = write_student_run(tmp_path, actor="mlp-student", name="x")
    side = run / "checkpoints" / "student.contract.json"
    data = json.loads(side.read_text("utf-8"))
    data["actor"] = "mlp-teacher"
    side.write_text(json.dumps(data), "utf-8")
    with pytest.raises(ExportError, match="is not a student"):
        export_student(run, tmp_path / "t.pt", bench=False)
    side.unlink()
    with pytest.raises(ExportError, match="no .contract.json"):
        export_student(run, tmp_path / "t.pt", bench=False)


def test_bench_requires_cuda(tmp_path, harness_k1_dir, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    run = write_student_run(tmp_path, k1_path=harness_k1_dir)
    with pytest.raises(ExportError, match="needs CUDA"):
        export_student(run, tmp_path / "k1.pt", bench=True, k1_path=harness_k1_dir)
    meta = export_student(run, tmp_path / "k1.pt", bench=None, k1_path=harness_k1_dir)
    assert meta["latency"] is None  # no CUDA: not benchmarked, not an error
