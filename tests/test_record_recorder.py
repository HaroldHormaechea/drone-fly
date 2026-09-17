"""AC3/AC10 — the activation recorder: self-contained schema + compact storage.

Covers :class:`drone_fly.record.recorder.ActivationRecorder` and its quantisation:

* **AC3** — a recorded episode is one self-contained file whose schema carries every
  documented key (ordered ``neuron_ids``, ``superclass``/``roles``, soma ``positions``,
  per-frame activations aligned to ``neuron_ids``, per-frame 4-channel action + drone
  position, and the outcome), with the activation width equal to the neuron count.
* **AC10** — activations are stored **quantised to ``uint8``** (round-trips within one
  quantisation step over ``[-1, 1]``); optional gzip round-trips; the file path + size
  are logged.
* The capture guards (no start, no sink, width mismatch) raise loudly.

Hermetic: drives the recorder directly with synthetic activations on the committed
fixture — no policy, env, checkpoint, or network.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.encoding import ACTION_DIM
from drone_fly.record.recorder import (
    ACTIVATION_OFFSET,
    ACTIVATION_SCALE,
    SCHEMA_VERSION,
    ActivationRecorder,
    dequantize_activation,
    quantize_activation,
)

_DOC_META_KEYS = {
    "neuron_ids",
    "superclass",
    "roles",
    "positions",
    "episode_index",
    "seed",
    "checkpoint",
    "backend",
    "n_frames",
    "n_neurons",
    "action_layout",
    "dt",
    "activation_scale",
    "activation_offset",
}
_DOC_POSITION_KEYS = {"source", "projection", "coords3d", "coords2d", "has_position"}
_DOC_OUTCOME_KEYS = {"completed", "completion_time", "total_reward", "steps"}


def _record_episode(
    connectome: ConnectomeData,
    out_dir: Path,
    *,
    n_frames: int = 4,
    gzip_output: bool = False,
    seed: int | None = 7,
) -> Path:
    """Drive a full episode through the recorder with deterministic synthetic frames."""
    rec = ActivationRecorder(
        connectome,
        out_dir,
        gzip_output=gzip_output,
        backend="simple",
        checkpoint="ckpt.zip",
        dt=0.05,
    )
    n = connectome.neuron_count
    rec.start_episode(episode_index=3, seed=seed)
    for f in range(n_frames):
        # Deterministic activations spanning [-1, 1] (bounded, like the tanh policy state).
        act = np.linspace(-1.0, 1.0, n, dtype=np.float32) * ((f + 1) / n_frames)
        rec.sink(act)
        rec.capture_frame(np.array([0.5, -0.2, 0.1, 0.3]), np.array([f * 1.0, 0.0, 1.0]))
    assert rec.n_frames == n_frames
    return rec.finish_episode(completed=True, completion_time=0.2, total_reward=1.5, steps=n_frames)


# --- AC10 quantisation -----------------------------------------------------------------
def test_quantize_dequantize_roundtrip_within_one_step() -> None:
    values = np.linspace(-1.0, 1.0, 513, dtype=np.float32)
    codes = quantize_activation(values)
    assert codes.dtype == np.uint8
    back = dequantize_activation(codes)
    # Lossless to 8 bits over [-1, 1] -> error is at most half a quantisation step.
    assert np.max(np.abs(back - values)) <= ACTIVATION_SCALE / 2 + 1e-9


def test_quantize_endpoints_and_clip() -> None:
    codes = quantize_activation(np.array([-1.0, 0.0, 1.0]))
    assert list(codes) == [0, 128, 255]  # offset=-1, scale=2/255 -> 0 mid-ish 255
    # Out-of-range values clamp into 0..255 rather than overflowing uint8.
    clipped = quantize_activation(np.array([-5.0, 5.0]))
    assert list(clipped) == [0, 255]


# --- AC3 self-contained schema ---------------------------------------------------------
def test_recorded_file_has_full_documented_schema(
    connectome: ConnectomeData, tmp_path: Path
) -> None:
    path = _record_episode(connectome, tmp_path / "act")
    assert path.is_file() and path.suffix == ".json"

    doc = json.loads(path.read_text())
    assert doc["schema_version"] == SCHEMA_VERSION

    meta = doc["meta"]
    assert set(meta) == _DOC_META_KEYS
    n = connectome.neuron_count
    assert meta["n_neurons"] == n
    assert meta["n_frames"] == 4
    assert len(meta["neuron_ids"]) == n
    assert len(meta["superclass"]) == n
    assert len(meta["roles"]) == n
    assert meta["action_layout"] == ["throttle", "roll", "pitch", "yaw"]
    assert meta["episode_index"] == 3
    assert meta["seed"] == 7
    assert meta["activation_scale"] == ACTIVATION_SCALE
    assert meta["activation_offset"] == ACTIVATION_OFFSET
    assert set(meta["positions"]) == _DOC_POSITION_KEYS

    frames = doc["frames"]
    assert set(frames) == {"activations", "actions", "drone_position"}
    activations = frames["activations"]
    assert len(activations) == 4
    for row in activations:
        assert len(row) == n  # aligned to neuron_ids
        assert all(0 <= v <= 255 for v in row)  # uint8 codes
    for a in frames["actions"]:
        assert len(a) == ACTION_DIM
    for p in frames["drone_position"]:
        assert len(p) == 3

    assert set(doc["outcome"]) == _DOC_OUTCOME_KEYS
    assert doc["outcome"] == {
        "completed": True,
        "completion_time": 0.2,
        "total_reward": 1.5,
        "steps": 4,
    }


def test_recorded_activations_dequantize_into_range(
    connectome: ConnectomeData, tmp_path: Path
) -> None:
    """Every stored frame dequantises back into the documented [-1, 1] band (AC10)."""
    path = _record_episode(connectome, tmp_path / "act")
    doc = json.loads(path.read_text())
    codes = np.asarray(doc["frames"]["activations"], dtype=np.uint8)
    deq = dequantize_activation(codes)
    assert deq.shape == (4, connectome.neuron_count)
    assert np.all(deq >= -1.0 - 1e-9)
    assert np.all(deq <= 1.0 + 1e-9)


# --- AC10 gzip + size logging ----------------------------------------------------------
def test_gzip_output_roundtrips(connectome: ConnectomeData, tmp_path: Path) -> None:
    path = _record_episode(connectome, tmp_path / "actgz", gzip_output=True)
    assert path.name.endswith(".json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        doc = json.load(handle)
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["meta"]["n_neurons"] == connectome.neuron_count
    assert len(doc["frames"]["activations"]) == 4


def test_finish_episode_logs_path_and_size(
    connectome: ConnectomeData, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="drone_fly.record.recorder"):
        path = _record_episode(connectome, tmp_path / "act")
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "Recorded episode" in msgs
    assert path.name in msgs
    assert "MB" in msgs  # size disclosed


# --- capture guards --------------------------------------------------------------------
def test_capture_before_start_raises(connectome: ConnectomeData, tmp_path: Path) -> None:
    rec = ActivationRecorder(connectome, tmp_path)
    rec.sink(np.zeros(connectome.neuron_count, dtype=np.float32))
    with pytest.raises(RuntimeError, match="start_episode"):
        rec.capture_frame(np.zeros(ACTION_DIM), np.zeros(3))


def test_capture_without_sink_raises(connectome: ConnectomeData, tmp_path: Path) -> None:
    rec = ActivationRecorder(connectome, tmp_path)
    rec.start_episode(0)
    with pytest.raises(RuntimeError, match="sink"):
        rec.capture_frame(np.zeros(ACTION_DIM), np.zeros(3))


def test_capture_width_mismatch_raises(connectome: ConnectomeData, tmp_path: Path) -> None:
    """The alignment guard: an activation not equal in width to n_neurons is rejected."""
    rec = ActivationRecorder(connectome, tmp_path)
    rec.start_episode(0)
    rec.sink(np.zeros(connectome.neuron_count + 1, dtype=np.float32))
    with pytest.raises(ValueError, match="misaligned|neuron count"):
        rec.capture_frame(np.zeros(ACTION_DIM), np.zeros(3))
