"""CLI smoke tests — argparse wiring + hermetic dispatch of the subcommands.

Thin coverage that the ``drone-fly`` CLI parses each subcommand and that ``smoke-train`` /
``evaluate`` dispatch end-to-end on the hermetic numpy backend. Real training correctness
lives in ``test_train_resume.py``; this only guards the command surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drone_fly.cli import build_parser, main
from drone_fly.connectome import DEFAULT_PRUNE_K, DEFAULT_PRUNE_RULE, load_connectome
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


# --------------------------------------------------------------------------- #
# UC-04 — opt-in prune flags on train / smoke-train
# --------------------------------------------------------------------------- #
def test_parser_train_prune_flags_default_off() -> None:
    """train exposes --prune (default off) and --prune-k (default DEFAULT_PRUNE_K)."""
    args = build_parser().parse_args(["train"])
    assert args.prune is False
    assert args.prune_k == DEFAULT_PRUNE_K


def test_parser_train_prune_flags_parsed() -> None:
    args = build_parser().parse_args(["train", "--prune", "--prune-k", "3"])
    assert args.prune is True
    assert args.prune_k == 3


def test_parser_smoke_train_prune_flags_parsed() -> None:
    args = build_parser().parse_args(["smoke-train", "--prune", "--prune-k", "1"])
    assert args.command == "smoke-train"
    assert args.prune is True
    assert args.prune_k == 1


# --------------------------------------------------------------------------- #
# UC-04 (AC11) — the `prune` export subcommand
# --------------------------------------------------------------------------- #
def test_parser_prune_defaults() -> None:
    """The prune subcommand parses required/optional args with documented defaults."""
    args = build_parser().parse_args(["prune", "--connectome", "in", "--out", "out"])
    assert args.command == "prune"
    assert args.connectome == "in"
    assert args.out == "out"
    assert args.prune_k == DEFAULT_PRUNE_K
    assert args.prune_rule == DEFAULT_PRUNE_RULE


def test_parser_prune_requires_connectome_and_out() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["prune", "--out", "out"])  # missing --connectome
    with pytest.raises(SystemExit):
        build_parser().parse_args(["prune", "--connectome", "in"])  # missing --out


def test_parser_prune_k_and_rule_parsed() -> None:
    args = build_parser().parse_args(
        [
            "prune",
            "--connectome",
            "in",
            "--out",
            "out",
            "--prune-k",
            "0",
            "--prune-rule",
            "path_slack",
        ]
    )
    assert args.prune_k == 0
    assert args.prune_rule == "path_slack"


def test_prune_subcommand_writes_reusable_slice(tmp_path, capsys) -> None:
    """`prune` writes .npz + _meta.csv + PRUNE_PROVENANCE.md and round-trips (AC11)."""
    out = tmp_path / "pruned"
    rc = main(["prune", "--connectome", FIXTURE_DIR, "--out", str(out), "--prune-k", "0"])
    assert rc == 0

    files = {p.name for p in out.iterdir()}
    assert "connectome_pruned.npz" in files
    assert "connectome_pruned_meta.csv" in files
    assert "PRUNE_PROVENANCE.md" in files

    # Reusable: reloads with the expected pruned scale (k=0 -> 59/751 on the fixture).
    reloaded = load_connectome(out)
    assert reloaded.neuron_count == 59
    assert reloaded.edge_count == 751

    out_text = capsys.readouterr().out
    assert "Pruned connectome written" in out_text


def test_prune_subcommand_unknown_rule_errors(tmp_path) -> None:
    """An unknown --prune-rule surfaces a clear error rather than writing garbage."""
    out = tmp_path / "pruned"
    with pytest.raises(ValueError, match="rule|Unknown"):
        main(["prune", "--connectome", FIXTURE_DIR, "--out", str(out), "--prune-rule", "bogus"])
