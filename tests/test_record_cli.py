"""AC4 — recording flags on the CLI + the evaluator's hard alignment assertion.

Recording is enabled via ``--record`` / ``--record-every N`` / ``--record-dir`` on the
``evaluate`` path (tested, deterministic primary) and on ``train`` (best-effort). Verified:

* the flags parse on both ``evaluate`` and ``train``;
* an ``evaluate --record`` run threads the flags through and writes the correct subset of
  self-contained playback files on the hermetic numpy backend;
* the evaluator re-loads the connectome (the checkpoint does not retain
  ``neuron_ids``/``superclass``/positions) and **raises** on a neuron-count mismatch
  between the re-loaded connectome and the checkpoint's actor.

Hermetic: ``adapter="simple"``, a tiny smoke-trained checkpoint, no pybullet / network.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from drone_fly.cli import build_parser
from drone_fly.evaluate.evaluator import evaluate_checkpoint
from drone_fly.train.config import TrainConfig
from drone_fly.train.loop import CHECKPOINT_PREFIX, smoke_train

FIXTURE_DIR = str(Path(__file__).parent / "fixtures")


# --- flag parsing ----------------------------------------------------------------------
def test_evaluate_parses_record_flags() -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "--checkpoint",
            "c.zip",
            "--record",
            "--record-every",
            "5",
            "--record-dir",
            "out/acts",
            "--connectome",
            FIXTURE_DIR,
        ]
    )
    assert args.record is True
    assert args.record_every == 5
    assert args.record_dir == "out/acts"
    assert args.connectome == FIXTURE_DIR


def test_train_parses_record_flags() -> None:
    args = build_parser().parse_args(["train", "--record", "--record-every", "50"])
    assert args.record is True
    assert args.record_every == 50
    assert args.record_dir is None  # defaults to artifacts/activations/


def test_record_off_by_default() -> None:
    args = build_parser().parse_args(["evaluate", "--checkpoint", "c.zip"])
    assert args.record is False
    assert args.record_every == 1


# --- checkpoint fixture ----------------------------------------------------------------
@pytest.fixture
def trained_checkpoint(connectome, tmp_path):
    """A tiny hermetic smoke-trained checkpoint on the full fixture (300 neurons)."""
    cfg = TrainConfig(
        models_dir=str(tmp_path / "models"),
        logs_dir=str(tmp_path / "logs"),
        checkpoint_freq=64,
        n_envs=1,
        n_steps=64,
        batch_size=32,
        seed=0,
    )
    smoke_train(connectome=connectome, cfg=cfg, timesteps=128)
    final = os.path.join(cfg.models_dir, f"{CHECKPOINT_PREFIX}_final.zip")
    stats = os.path.join(cfg.models_dir, cfg.vecnormalize_name)
    assert os.path.isfile(final) and os.path.isfile(stats)
    return final, stats


# --- AC4 thread-through: evaluate --record writes files --------------------------------
def test_evaluate_record_threads_through_and_writes_files(trained_checkpoint, tmp_path) -> None:
    final, stats = trained_checkpoint
    rec_dir = tmp_path / "recordings"
    evaluate_checkpoint(
        final,
        vecnormalize_path=stats,
        episodes=3,
        adapter="simple",
        device="cpu",
        record=True,
        record_every=2,
        record_dir=str(rec_dir),
        connectome_path=FIXTURE_DIR,  # unpruned -> 300, aligns with the actor
    )
    # Episodes 0 and 2 recorded (record_every=2); 1 skipped.
    names = sorted(p.name for p in rec_dir.glob("episode_*.json"))
    assert names == ["episode_0.json", "episode_2.json"]


# --- AC4 alignment assertion -----------------------------------------------------------
def test_evaluate_record_raises_on_connectome_mismatch(trained_checkpoint, tmp_path) -> None:
    """Re-loading a pruned (274-neuron) connectome against the 300-neuron actor must raise."""
    final, stats = trained_checkpoint
    with pytest.raises(ValueError, match="alignment|neuron"):
        evaluate_checkpoint(
            final,
            vecnormalize_path=stats,
            episodes=1,
            adapter="simple",
            device="cpu",
            record=True,
            record_dir=str(tmp_path / "rec"),
            connectome_path=FIXTURE_DIR,
            prune=True,  # 300 -> 274 neurons, no longer matches the checkpoint's actor
        )
