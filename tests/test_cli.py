"""CLI smoke tests — argparse wiring + hermetic dispatch of the subcommands.

Thin coverage that the ``drone-fly`` CLI parses each subcommand and that ``smoke-train`` /
``evaluate`` dispatch end-to-end on the hermetic numpy backend. Real training correctness
lives in ``test_train_resume.py``; this only guards the command surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drone_fly.cli import build_parser, main
from drone_fly.train.loop import CHECKPOINT_PREFIX

FIXTURE_DIR = str(Path(__file__).parent / "fixtures")


def test_parser_requires_a_subcommand() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_train_flags() -> None:
    argv = ["train", "--resume", "x.zip", "--device", "cpu"]
    argv += ["--timesteps", "10", "--adapter", "simple"]
    args = build_parser().parse_args(argv)
    assert args.command == "train"
    assert args.resume == "x.zip"
    assert args.device == "cpu"
    assert args.timesteps == 10
    assert args.adapter == "simple"


def test_parser_evaluate_requires_checkpoint() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["evaluate"])  # --checkpoint is required


def test_smoke_train_dispatch(tmp_path, monkeypatch) -> None:
    # Run from a temp cwd so artifacts/ lands under tmp, not the repo.
    monkeypatch.chdir(tmp_path)
    rc = main(["smoke-train", "--connectome", FIXTURE_DIR, "--timesteps", "128"])
    assert rc == 0
    models = tmp_path / "artifacts" / "models"
    assert models.is_dir()
    assert any(f.name.startswith(CHECKPOINT_PREFIX) for f in models.iterdir())


def test_evaluate_dispatch(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    # First produce a checkpoint hermetically.
    assert main(["smoke-train", "--connectome", FIXTURE_DIR, "--timesteps", "128"]) == 0
    models = tmp_path / "artifacts" / "models"
    ckpt = models / f"{CHECKPOINT_PREFIX}_final.zip"
    stats = models / "vecnormalize.pkl"
    assert ckpt.is_file() and stats.is_file()

    rc = main(
        [
            "evaluate",
            "--checkpoint",
            str(ckpt),
            "--vecnormalize",
            str(stats),
            "--episodes",
            "2",
            "--adapter",
            "simple",
            "--device",
            "cpu",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "completion_rate" in out


def test_fetch_connectome_stub(capsys) -> None:
    rc = main(["fetch-connectome"])
    assert rc == 0
    assert "stub" in capsys.readouterr().out.lower()
