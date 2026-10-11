"""UC-62 AC12: the fresh ``drone-fly`` CLI — list / run / eval and distinct exit codes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from drone_fly.cli.main import (
    EXIT_CONFIG,
    EXIT_COVERAGE,
    EXIT_DEPENDENCY,
    EXIT_FAILSAFE,
    EXIT_OK,
    build_parser,
    main,
)
from drone_fly.harness import registry
from tests.harness_support import FakeAbort

REPO = Path(__file__).resolve().parent.parent


def test_exit_codes_are_distinct():
    codes = [EXIT_OK, EXIT_CONFIG, EXIT_DEPENDENCY, EXIT_COVERAGE, EXIT_FAILSAFE]
    assert codes == [0, 2, 3, 4, 5]


def test_console_script_declared_in_pyproject():
    import tomllib

    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["drone-fly"] == "drone_fly.cli.main:main"
    extra = data["project"]["optional-dependencies"]["harness"]
    for pkg in ("opencv-python", "mss", "vgamepad", "transformers"):
        assert any(e.startswith(pkg) for e in extra), pkg
    core = " ".join(data["project"]["dependencies"])
    for pkg in ("opencv", "vgamepad", "transformers", "mss", "dxcam"):
        assert pkg not in core, f"{pkg} must stay out of the core dependencies"


def test_parser_shapes():
    p = build_parser()
    args = p.parse_args(
        [
            "harness",
            "run",
            "--harness",
            "liftoff",
            "--checkpoint",
            "m.pt",
            "--extra-dim",
            "70",
            "--config",
            "c.yaml",
            "--dry-run",
            "--duration",
            "3",
        ]
    )
    assert (args.harness, args.checkpoint, args.extra_dim, args.config) == (
        "liftoff",
        "m.pt",
        70,
        "c.yaml",
    )
    assert args.dry_run and args.duration == 3.0 and not args.no_record
    with pytest.raises(SystemExit) as ei:
        p.parse_args(
            ["harness", "run", "--harness", "x", "--checkpoint", "m.pt", "--extra-dim", "31"]
        )
    assert ei.value.code == 2
    with pytest.raises(SystemExit):
        p.parse_args(["harness", "run", "--checkpoint", "m.pt", "--extra-dim", "30"])
    ev = p.parse_args(["harness", "eval", "--session", "s", "--track", "t.xml", "--race", "r.xml"])
    assert ev.harness == "liftoff" and ev.report is None


def test_list_prints_every_registered_part(capsys):
    assert main(["harness", "list"]) == EXIT_OK
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].split() == ["kind", "name", "status", "notes"]
    for kind in registry.KINDS:
        for name in registry.names(kind):
            assert any(line.split()[:2] == [kind, name] for line in lines[1:]), (kind, name)
    loopback = next(line for line in lines if line.split()[:2] == ["harness", "sim-loopback"])
    assert loopback.split()[2] == "available"


def _run_args(ckpt, k1, out, *extra):
    return [
        "harness",
        "run",
        "--harness",
        "sim-loopback",
        "--checkpoint",
        str(ckpt),
        "--extra-dim",
        "70",
        "--k1-path",
        str(k1),
        "--device",
        "cpu",
        "--output-root",
        str(out),
        *extra,
    ]


def test_run_sim_loopback_writes_run_folder(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path, capsys
):
    code = main(_run_args(harness_checkpoint_70, harness_k1_dir, tmp_path, "--duration", "0.5"))
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "stop: completed" in out and "ticks: 25" in out
    runs = list(tmp_path.glob("sim-loopback-*"))
    assert len(runs) == 1
    run = runs[0]
    for name in ("config.yaml", "meta.json", "ticks.jsonl", "telemetry.csv", "frames.csv"):
        assert (run / name).is_file(), name
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert meta["extra_dim"] == 70 and meta["device"] == "cpu"
    assert meta["checkpoint_sha256"] and meta["stop_reason"] == "completed"
    assert meta["provided_blocks"] == [
        "curr_gate",
        "next_lookahead",
        "obstacle_vision",
        "raycast",
        "target",
    ]
    assert len((run / "ticks.jsonl").read_text(encoding="utf-8").splitlines()) == 25


def test_run_no_record_writes_nothing(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path
):
    code = main(
        _run_args(
            harness_checkpoint_70, harness_k1_dir, tmp_path, "--duration", "0.1", "--no-record"
        )
    )
    assert code == EXIT_OK and list(tmp_path.iterdir()) == []


def test_config_error_exit_code(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path, capsys
):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("runner:\n  rate_hzz: 50\n", encoding="utf-8")
    out = tmp_path / "out"
    code = main(_run_args(harness_checkpoint_70, harness_k1_dir, out, "--config", str(cfg)))
    assert code == EXIT_CONFIG
    assert "rate_hzz" in capsys.readouterr().err
    assert not out.exists()


def test_missing_checkpoint_is_config_error(harness_contract, harness_k1_dir, tmp_path, capsys):
    code = main(_run_args(tmp_path / "missing.pt", harness_k1_dir, tmp_path / "out"))
    assert code == EXIT_CONFIG and "checkpoint not found" in capsys.readouterr().err


def test_coverage_refusal_exit_code_and_nothing_written(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path, capsys
):
    cfg = tmp_path / "none.yaml"
    cfg.write_text("parts:\n  free_space_estimator: none\n", encoding="utf-8")
    out = tmp_path / "out"
    code = main(_run_args(harness_checkpoint_70, harness_k1_dir, out, "--config", str(cfg)))
    assert code == EXIT_COVERAGE
    err = capsys.readouterr().err
    assert "raycast" in err and "obstacle_vision" in err
    assert not out.exists()  # refused before the run folder is created


def test_dependency_error_exit_code(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path, capsys
):
    registry.register("free_space_estimator", "uc62-cli-missing", "no_such_backend_uc62:Estimator")
    cfg = tmp_path / "dep.yaml"
    cfg.write_text("parts:\n  free_space_estimator: uc62-cli-missing\n", encoding="utf-8")
    try:
        code = main(
            _run_args(harness_checkpoint_70, harness_k1_dir, tmp_path / "out", "--config", str(cfg))
        )
    finally:
        registry.unregister("free_space_estimator", "uc62-cli-missing")
    assert code == EXIT_DEPENDENCY
    assert "no_such_backend_uc62" in capsys.readouterr().err


def test_failsafe_stop_exit_code(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path, capsys
):
    class AbortNow(FakeAbort):
        @classmethod
        def from_config(cls, cfg, ctx):
            return cls(set_after_checks=3)

    registry.register("abort_signal", "uc62-abort", AbortNow)
    cfg = tmp_path / "abort.yaml"
    cfg.write_text("parts:\n  abort_signal: uc62-abort\n", encoding="utf-8")
    try:
        code = main(
            _run_args(harness_checkpoint_70, harness_k1_dir, tmp_path / "out", "--config", str(cfg))
        )
    finally:
        registry.unregister("abort_signal", "uc62-abort")
    assert code == EXIT_FAILSAFE
    assert "stop: user_abort" in capsys.readouterr().out
    meta = json.loads(
        next((tmp_path / "out").glob("sim-loopback-*/meta.json")).read_text(encoding="utf-8")
    )
    assert meta["stop_reason"] == "user_abort" and meta["failsafe_ok"] is True


def test_module_entry_point_runs_list():
    proc = subprocess.run(
        [sys.executable, "-m", "drone_fly.cli.main", "harness", "list"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert proc.returncode == 0, proc.stderr
    assert "sim-loopback" in proc.stdout
