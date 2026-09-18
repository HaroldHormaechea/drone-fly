"""AC1/AC3/AC4/AC10 — the recording rollout driver on the hermetic fixture.

:func:`drone_fly.record.rollout.record_rollout` drives a **raw** actor (no checkpoint)
through the racing env on the pure-numpy :class:`SimpleDroneAdapter`, recording every Nth
episode. Verified end-to-end on the committed fixture:

* **AC1/AC3** — a smoke-record produces one self-contained file per recorded episode with
  ``n_frames == steps``, activation width ``== n_neurons == len(neuron_ids)``, action width
  4, drone-position width 3, and activations that dequantise into ``[-1, 1]``.
* **AC4** — ``record_every=N`` records exactly the ``0, N, 2N, …`` episode subset (one
  capture per step, one file per recorded episode).
* the alignment / cadence guards raise loudly.

Hermetic: ``adapter="simple"``, short episodes, no pybullet / checkpoint / network.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.actor import ConnectomeActorNetwork
from drone_fly.env.config import EnvConfig, EpisodeConfig
from drone_fly.env.racing_env import make_env
from drone_fly.record.recorder import ActivationRecorder, dequantize_activation


def _short_env():
    """A hermetic numpy-backed racing env with short episodes for fast recording."""
    cfg = EnvConfig(episode=EpisodeConfig(max_steps=5))
    return make_env(cfg, adapter="simple")


def _actor(connectome: ConnectomeData) -> ConnectomeActorNetwork:
    torch.manual_seed(0)
    return ConnectomeActorNetwork(connectome)


# --- AC1/AC3/AC10 smoke record ---------------------------------------------------------
def test_smoke_record_one_episode(connectome: ConnectomeData, tmp_path: Path) -> None:
    actor = _actor(connectome)
    recorder = ActivationRecorder(connectome, tmp_path / "act", backend="simple", dt=0.05)
    written = record_rollout_wrapper(actor, recorder, n_episodes=1, record_every=1)

    assert len(written) == 1
    path = written[0]
    assert path.name == "episode_0.json"

    doc = json.loads(path.read_text())
    n = connectome.neuron_count
    n_frames = doc["meta"]["n_frames"]
    steps = doc["outcome"]["steps"]

    assert n_frames == steps >= 1  # one captured frame per env step
    assert doc["meta"]["n_neurons"] == n == len(doc["meta"]["neuron_ids"])
    for act in doc["frames"]["activations"]:
        assert len(act) == n  # activation width == neuron count == len(neuron_ids)
    assert len(doc["frames"]["activations"]) == n_frames
    for a in doc["frames"]["actions"]:
        assert len(a) == 4
    for p in doc["frames"]["drone_position"]:
        assert len(p) == 3

    deq = dequantize_activation(np.asarray(doc["frames"]["activations"], dtype=np.uint8))
    assert np.all(deq >= -1.0 - 1e-9) and np.all(deq <= 1.0 + 1e-9)


# --- AC4 cadence -----------------------------------------------------------------------
def test_record_every_records_correct_episode_subset(
    connectome: ConnectomeData, tmp_path: Path
) -> None:
    actor = _actor(connectome)
    recorder = ActivationRecorder(connectome, tmp_path / "act", backend="simple", dt=0.05)
    written = record_rollout_wrapper(actor, recorder, n_episodes=4, record_every=2)

    # Episodes 0 and 2 recorded; 1 and 3 skipped.
    names = sorted(p.name for p in written)
    assert names == ["episode_0.json", "episode_2.json"]
    on_disk = sorted(p.name for p in (tmp_path / "act").glob("episode_*.json"))
    assert on_disk == ["episode_0.json", "episode_2.json"]


def test_record_every_one_records_all(connectome: ConnectomeData, tmp_path: Path) -> None:
    actor = _actor(connectome)
    recorder = ActivationRecorder(connectome, tmp_path / "act", backend="simple", dt=0.05)
    written = record_rollout_wrapper(actor, recorder, n_episodes=3, record_every=1)
    assert sorted(p.name for p in written) == [
        "episode_0.json",
        "episode_1.json",
        "episode_2.json",
    ]


# --- guards ----------------------------------------------------------------------------
def test_invalid_cadence_raises(connectome: ConnectomeData, tmp_path: Path) -> None:
    from drone_fly.record.rollout import record_rollout

    actor = _actor(connectome)
    recorder = ActivationRecorder(connectome, tmp_path, backend="simple")
    env = _short_env()
    with pytest.raises(ValueError, match="record_every"):
        record_rollout(actor, env, recorder, n_episodes=1, record_every=0)


def test_recorder_actor_misalignment_raises(
    connectome: ConnectomeData, synthetic_connectome: ConnectomeData, tmp_path: Path
) -> None:
    """A recorder built from a different graph than the actor is rejected up front."""
    from drone_fly.record.rollout import record_rollout

    actor = _actor(connectome)  # 322 neurons
    recorder = ActivationRecorder(synthetic_connectome, tmp_path, backend="simple")  # 50
    env = _short_env()
    with pytest.raises(ValueError, match="aligned|neurons"):
        record_rollout(actor, env, recorder, n_episodes=1, record_every=1)


# --------------------------------------------------------------------------- #
# Recording no-clobber on resume — RecordingCallback continues the episode counter
# past any existing episode_<n>.json[.gz] in the record dir (empty dir -> 0).
# --------------------------------------------------------------------------- #
def test_highest_episode_index_empty_dir(tmp_path: Path) -> None:
    from drone_fly.train.record_callback import _highest_episode_index

    assert _highest_episode_index(tmp_path) == -1  # empty -> -1 (so next index is 0)


def test_highest_episode_index_scans_json_and_gz(tmp_path: Path) -> None:
    """The scan covers BOTH plain .json and gzipped .json.gz playback files."""
    from drone_fly.train.record_callback import _highest_episode_index

    (tmp_path / "episode_0.json").write_text("{}")
    (tmp_path / "episode_1.json").write_text("{}")
    (tmp_path / "episode_3.json.gz").write_bytes(b"")  # gz variant must count toward the max
    (tmp_path / "not_an_episode.json").write_text("{}")  # ignored (no episode_<n> pattern)
    assert _highest_episode_index(tmp_path) == 3


def _stub_callback(monkeypatch, connectome, record_dir):
    """Build a RecordingCallback with a stubbed actor so `_on_training_start` runs without a
    real PPO model (actor_from_model is patched to a no-op namespace with a settable sink)."""
    from drone_fly.record.recorder import ActivationRecorder
    from drone_fly.train import record_callback as rc_mod
    from drone_fly.train.record_callback import RecordingCallback

    recorder = ActivationRecorder(connectome, record_dir, backend="simple")
    cb = RecordingCallback(recorder, record_every=1, seed=0)
    monkeypatch.setattr(rc_mod, "actor_from_model", lambda model: types.SimpleNamespace(sink=None))
    cb.model = object()  # never dereferenced — actor_from_model is stubbed above
    return cb


def test_callback_continues_past_existing_recordings(
    connectome: ConnectomeData, tmp_path: Path, monkeypatch
) -> None:
    """A resumed run's callback starts numbering at max+1 so it never clobbers prior files."""
    rec_dir = tmp_path / "acts"
    rec_dir.mkdir()
    (rec_dir / "episode_0.json").write_text("{}")
    (rec_dir / "episode_1.json").write_text("{}")
    (rec_dir / "episode_2.json.gz").write_bytes(b"")  # max index on disk is 2

    cb = _stub_callback(monkeypatch, connectome, rec_dir)
    cb._on_training_start()
    assert cb._episode == 3  # max(0, 1, 2) + 1 — continues past the existing recordings


def test_callback_starts_at_zero_for_empty_dir(
    connectome: ConnectomeData, tmp_path: Path, monkeypatch
) -> None:
    """A fresh run (empty record dir) starts at episode 0 — fresh-run parity, no regression."""
    rec_dir = tmp_path / "acts"
    rec_dir.mkdir()

    cb = _stub_callback(monkeypatch, connectome, rec_dir)
    cb._on_training_start()
    assert cb._episode == 0  # empty dir -> -1 + 1


def record_rollout_wrapper(actor, recorder, *, n_episodes: int, record_every: int):
    """Run ``record_rollout`` with a fresh short hermetic env (helper to keep tests terse)."""
    from drone_fly.record.rollout import record_rollout

    env = _short_env()
    try:
        return record_rollout(
            actor, env, recorder, n_episodes=n_episodes, seed=0, record_every=record_every
        )
    finally:
        env.close()
