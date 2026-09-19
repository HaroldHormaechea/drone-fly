"""UC-26 AC-12 — recording safety under the parallel (subproc) backend (user gate #1).

Training-time activation recording (UC-05) has a single writer: the main-process
``RecordingCallback`` + ``ActivationRecorder``, which capture env-0's slice from the batched
arrays the vec-env returns. The spawn workers only step envs — they never touch the recorder or
its output dir — so switching the rollout backend to ``SubprocVecEnv`` cannot corrupt, overwrite,
or interleave any saved episode.

This module proves that end-to-end:

* **Training recording under subproc** — a real ``train(..., record=True)`` run forced onto the
  subproc backend (2 workers) writes valid, self-contained, monotonically-indexed episode files
  whose per-frame widths are the *single-env* (env-0) slice, not a concatenated 2-env batch.
* **Eval/playback invariance** — the evaluation recording path builds a single-env stack that is
  a plain ``DummyVecEnv`` with no ``SubprocVecEnv`` anywhere, and an
  ``evaluate_checkpoint(record=True)`` recording round-trips uncorrupted.

Hermetic: ``adapter="simple"`` (no pybullet), committed fixture connectome, short episodes.
NOTE (spawn): every module-level symbol here must be import-safe — a spawned worker re-imports
this module, so nothing heavy runs at import time (only defs).
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from drone_fly.env.config import EnvConfig, EpisodeConfig
from drone_fly.env.racing_env import build_vec_env
from drone_fly.train.config import TrainConfig
from drone_fly.train.loop import CHECKPOINT_PREFIX, smoke_train

_EP_RE = re.compile(r"episode_(\d+)\.json$")


def _episode_files(record_dir: Path) -> list[Path]:
    files = [Path(p) for p in glob.glob(str(record_dir / "episode_*.json"))]
    return sorted(files, key=lambda p: int(_EP_RE.search(p.name).group(1)))


def test_training_time_recording_is_safe_under_subproc_backend(connectome, tmp_path, monkeypatch):
    """AC-12: a subproc-backed training run records valid, monotonically-numbered, single-env
    episode files — the workers never collide with the main-process recorder."""
    import drone_fly.train.loop as loop

    # Force the resolver onto the subproc backend for the hermetic `simple` adapter (2 workers).
    # Workers rebuild the env via the picklable factory closure (they never call resolve_vec_env),
    # so patching it in the parent is sufficient and does not leak into the spawned workers.
    monkeypatch.setattr(
        loop, "resolve_vec_env", lambda adapter, n_arg, cfg_n: ("simple", 2, "subproc")
    )

    cfg = TrainConfig(
        models_dir=str(tmp_path / "models"),
        logs_dir=str(tmp_path / "logs"),
        checkpoint_freq=64,
        n_envs=1,
        n_steps=32,
        batch_size=16,
        seed=0,
    )
    record_dir = tmp_path / "acts"
    ecfg = EnvConfig(episode=EpisodeConfig(max_steps=5))  # short episodes -> files appear fast

    model = loop.train(
        cfg,
        connectome=connectome,
        adapter="simple",
        device="cpu",
        tui=False,  # -> suppress_worker_output=False; no fd redirect in workers
        total_timesteps=160,
        record=True,
        record_dir=str(record_dir),
        env_config=ecfg,
    )

    # The run genuinely used the 2-worker subproc backend.
    assert model.get_env().num_envs == 2

    files = _episode_files(record_dir)
    assert files, "no episode recordings were written under the subproc backend"

    # Episode indices are contiguous and monotonically increasing from 0 (no clobber/interleave).
    indices = [int(_EP_RE.search(f.name).group(1)) for f in files]
    assert indices == list(range(len(files)))

    n_neurons = connectome.neuron_count
    for f in files:
        doc = json.loads(f.read_text())  # each file parses as valid JSON
        n_frames = doc["meta"]["n_frames"]
        assert doc["meta"]["n_neurons"] == n_neurons
        # Single-env (env-0) slice — NOT a concatenated 2-env batch: one row per frame, and each
        # activation/action/position row is one env's width, not doubled.
        assert len(doc["frames"]["activations"]) == n_frames
        assert len(doc["frames"]["actions"]) == n_frames
        assert len(doc["frames"]["drone_position"]) == n_frames
        for act in doc["frames"]["activations"]:
            assert len(act) == n_neurons
        for a in doc["frames"]["actions"]:
            assert len(a) == 4
        for p in doc["frames"]["drone_position"]:
            assert len(p) == 3


def test_eval_recording_path_is_single_env_dummy_never_subproc() -> None:
    """AC-12: the eval/playback recording path is single-env (n_envs=1) -> DummyVecEnv, so no
    SubprocVecEnv can ever appear in its stack, regardless of the auto rule."""
    venv = build_vec_env(adapter="simple", n_envs=1, training=False, seed=0)
    try:
        node = venv
        saw_dummy = False
        while hasattr(node, "venv"):
            assert not isinstance(node, SubprocVecEnv)
            node = node.venv
        assert isinstance(node, DummyVecEnv)  # the base env is the serial Dummy backend
        saw_dummy = True
        assert saw_dummy
    finally:
        venv.close()


def test_evaluate_checkpoint_recording_round_trips_uncorrupted(
    connectome, fixture_dir_copy, tmp_path
):
    """AC-12: an evaluate_checkpoint(record=True) run (always single-env) writes uncorrupted,
    valid playback files — the parallel backend never touches the tested eval recording path."""
    from drone_fly.evaluate.evaluator import evaluate_checkpoint

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
    ckpt = os.path.join(cfg.models_dir, f"{CHECKPOINT_PREFIX}_final.zip")
    stats = os.path.join(cfg.models_dir, cfg.vecnormalize_name)
    assert os.path.isfile(ckpt) and os.path.isfile(stats)

    record_dir = tmp_path / "eval_acts"
    evaluate_checkpoint(
        ckpt,
        vecnormalize_path=stats,
        episodes=2,
        seed=0,
        adapter="simple",
        device="cpu",
        record=True,
        record_every=1,
        record_dir=str(record_dir),
        # UC-27: throwaway fixture copy so the recorder's positions sidecar write lands in tmp.
        connectome_path=str(fixture_dir_copy),
    )

    files = _episode_files(record_dir)
    assert files, "evaluate_checkpoint(record=True) wrote no recordings"
    n_neurons = connectome.neuron_count
    for f in files:
        doc = json.loads(f.read_text())
        assert doc["meta"]["n_neurons"] == n_neurons
        for act in doc["frames"]["activations"]:
            assert len(act) == n_neurons
