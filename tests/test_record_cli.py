"""UC-11 — config-driven recording on the ``evaluate`` / ``train`` paths.

Recording used to be enabled via ``--record`` / ``--record-every`` / ``--record-dir`` flags;
under UC-11 those live in the YAML config. This module verifies:

* an ``evaluate`` config with ``record: true`` threads the recording settings through to
  :func:`evaluate_checkpoint`, and routes recordings under ``training/<name>/recordings/`` when a
  ``name`` is set (else keeps the historical default);
* ``record_every`` omitted (config default ``None``) is coalesced to ``1`` at the dispatch
  boundary, while an explicit value passes through unchanged;
* the accepted developer deviation #1 — an ``evaluate`` config still accepts ``prune``/``prune_k``
  and routes them through to :func:`evaluate_checkpoint`;
* the evaluator still writes the correct playback files and still raises on a connectome/actor
  neuron-count mismatch (behaviour parity, AC7).

Hermetic: ``adapter="simple"``, a tiny smoke-trained checkpoint, no pybullet / network.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from drone_fly.cli import main
from drone_fly.evaluate.evaluator import evaluate_checkpoint
from drone_fly.train.config import TrainConfig
from drone_fly.train.loop import CHECKPOINT_PREFIX, smoke_train

FIXTURE_DIR = str(Path(__file__).parent / "fixtures")


def _write_config(tmp_path: Path, mapping: dict, name: str = "config.yaml") -> str:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
# Config-driven recording thread-through (evaluate)
# --------------------------------------------------------------------------- #


def test_evaluate_config_threads_record_settings(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    class _Metrics:
        def summary(self) -> str:
            return "completion_rate=0.0"

    monkeypatch.setattr(
        "drone_fly.evaluate.evaluator.evaluate_checkpoint",
        lambda checkpoint, **k: (captured.update(k, checkpoint=checkpoint), _Metrics())[1],
    )

    cfg = _write_config(
        tmp_path,
        {
            "checkpoint": "c.zip",
            "adapter": "simple",
            "record": True,
            "record_every": 5,
            "record_dir": "out/acts",
            "connectome": FIXTURE_DIR,
        },
    )
    assert main(["evaluate", "--config", cfg]) == 0
    assert captured["record"] is True
    assert captured["record_every"] == 5
    assert captured["record_dir"] == "out/acts"
    assert captured["connectome_path"] == FIXTURE_DIR


def test_evaluate_record_dir_routes_under_run_layout_when_named(tmp_path, monkeypatch) -> None:
    """name + record + no explicit record_dir -> training/<name>/recordings/ (AC2 consistency)."""
    captured: dict = {}
    monkeypatch.setattr(
        "drone_fly.evaluate.evaluator.evaluate_checkpoint",
        lambda checkpoint, **k: (captured.update(k), _StubMetrics())[1],
    )
    cfg = _write_config(
        tmp_path, {"checkpoint": "c.zip", "adapter": "simple", "record": True, "name": "myeval"}
    )
    assert main(["evaluate", "--config", cfg]) == 0
    assert captured["record_dir"] == "training/myeval/recordings"


def test_evaluate_record_dir_none_when_unnamed(tmp_path, monkeypatch) -> None:
    """No name -> record_dir stays None so the evaluator keeps its historical default (AC7)."""
    captured: dict = {}
    monkeypatch.setattr(
        "drone_fly.evaluate.evaluator.evaluate_checkpoint",
        lambda checkpoint, **k: (captured.update(k), _StubMetrics())[1],
    )
    cfg = _write_config(tmp_path, {"checkpoint": "c.zip", "adapter": "simple", "record": True})
    assert main(["evaluate", "--config", cfg]) == 0
    assert captured["record_dir"] is None


class _StubMetrics:
    def summary(self) -> str:
        return "completion_rate=0.0"


# --------------------------------------------------------------------------- #
# record_every None -> 1 coalesce at the dispatch boundary
# --------------------------------------------------------------------------- #


def test_evaluate_dispatch_coalesces_record_every_to_one(tmp_path, monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        "drone_fly.evaluate.evaluator.evaluate_checkpoint",
        lambda checkpoint, **k: (captured.update(k), _StubMetrics())[1],
    )
    cfg = _write_config(tmp_path, {"checkpoint": "c.zip", "adapter": "simple"})
    assert main(["evaluate", "--config", cfg]) == 0
    assert captured["record_every"] == 1  # coalesced from the None default


def test_train_dispatch_coalesces_record_every_to_one(tmp_path, monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr("drone_fly.train.loop.train", lambda *a, **k: captured.update(k))
    cfg = _write_config(tmp_path, {"name": "r", "adapter": "simple"})
    assert main(["train", "--config", cfg]) == 0
    assert captured["record_every"] == 1


def test_train_dispatch_forwards_explicit_record_every(tmp_path, monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr("drone_fly.train.loop.train", lambda *a, **k: captured.update(k))
    cfg = _write_config(
        tmp_path, {"name": "r", "adapter": "simple", "record": True, "record_every": 7}
    )
    assert main(["train", "--config", cfg]) == 0
    assert captured["record_every"] == 7


# --------------------------------------------------------------------------- #
# Deviation #1 — evaluate config keeps optional prune / prune_k and routes them
# --------------------------------------------------------------------------- #


def test_evaluate_config_routes_prune_keys(tmp_path, monkeypatch) -> None:
    """A pruned-checkpoint recording needs prune/prune_k to reach evaluate_checkpoint."""
    captured: dict = {}
    monkeypatch.setattr(
        "drone_fly.evaluate.evaluator.evaluate_checkpoint",
        lambda checkpoint, **k: (captured.update(k), _StubMetrics())[1],
    )
    cfg = _write_config(
        tmp_path,
        {"checkpoint": "c.zip", "adapter": "simple", "prune": True, "prune_k": 1},
    )
    assert main(["evaluate", "--config", cfg]) == 0
    assert captured["prune"] is True and captured["prune_k"] == 1


# --------------------------------------------------------------------------- #
# Behaviour parity — the evaluator still records + still asserts alignment (AC7)
# --------------------------------------------------------------------------- #


@pytest.fixture
def trained_checkpoint(connectome, tmp_path):
    """A tiny hermetic smoke-trained checkpoint on the full fixture (322 neurons)."""
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


def test_evaluate_record_writes_files(trained_checkpoint, tmp_path) -> None:
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
        connectome_path=FIXTURE_DIR,  # unpruned -> 322, aligns with the actor
    )
    names = sorted(p.name for p in rec_dir.glob("episode_*.json"))
    assert names == ["episode_0.json", "episode_2.json"]


def test_evaluate_record_raises_on_connectome_mismatch(trained_checkpoint, tmp_path) -> None:
    """Re-loading a pruned (247-neuron) connectome against the 322-neuron actor must raise."""
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
            prune=True,  # 322 -> 247 neurons, no longer matches the checkpoint's actor
        )
