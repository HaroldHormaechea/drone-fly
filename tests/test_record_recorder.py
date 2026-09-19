"""AC3/AC10 — the activation recorder: self-contained schema + compact storage.

Covers :class:`drone_fly.record.recorder.ActivationRecorder` and its quantisation:

* **AC3** — a recorded episode is one self-contained file whose schema carries every
  documented key (ordered ``neuron_ids``, ``superclass``/``roles``, soma ``positions``,
  per-frame activations aligned to ``neuron_ids``, per-frame 4-channel action + drone
  position, and the outcome), with the activation width equal to the neuron count.
* **AC10** — activations are stored **quantised to ``uint8``** (round-trips within one
  quantisation step over ``[-1, 1]``); optional gzip round-trips; the file path + size
  are logged.
* **UC-06 AC8** — when a :class:`CourseConfig` is supplied, the file grows an **additive**
  ``meta.course`` block (start / gate / finish / floor / ceiling + axis conventions) read
  from the actual config; ``course=None`` omits the block entirely (back-compat).
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
from drone_fly.env.config import (
    CourseConfig,
    GateSpec,
    ObstacleSpec,
    PadSpec,
    single_gate_course,
)
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
    "modality",  # UC-28: per-neuron modality tag for the viewer overlay (AC-7)
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
# UC-28: positions gained the always-present placement / region / display3d full-coverage fields.
_DOC_POSITION_KEYS = {
    "source",
    "projection",
    "coords3d",
    "coords2d",
    "has_position",
    "placement",
    "region",
    "display3d",
}
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
    # UC-28 additive full-coverage / modality fields are per-neuron and index-aligned (AC-2/AC-7).
    assert len(meta["positions"]["display3d"]) == n
    assert len(meta["positions"]["placement"]) == n
    assert len(meta["positions"]["region"]) == n
    assert len(meta["modality"]) == n
    assert set(meta["modality"]) <= {"vision", "proprioceptive", "hunger", ""}

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


# --- UC-06 AC8: additive meta.course block ---------------------------------------------
def _record_with_course(
    connectome: ConnectomeData, out_dir: Path, course: CourseConfig | None
) -> dict:
    """Record a tiny episode with the given ``course`` and return the parsed document."""
    rec = ActivationRecorder(connectome, out_dir, backend="simple", dt=0.05, course=course)
    n = connectome.neuron_count
    rec.start_episode(episode_index=0, seed=0)
    for f in range(2):
        rec.sink(np.linspace(-1.0, 1.0, n, dtype=np.float32) * (f + 1) / 2)
        rec.capture_frame(np.array([0.1, 0.2, 0.3, 0.4]), np.array([float(f), 0.0, 1.0]))
    path = rec.finish_episode(completed=True, completion_time=0.1, total_reward=1.0, steps=2)
    return json.loads(path.read_text())


def test_meta_course_serialises_default_config(connectome: ConnectomeData, tmp_path: Path) -> None:
    """A supplied ``CourseConfig`` is serialised verbatim into ``meta.course`` (UC-09 AC7).

    UC-09: the block emits **all N gates** as a ``gates: [...]`` array (each
    ``{center, aperture, plane}``), replacing the pre-UC-09 singular ``gate`` block. Every
    value comes from the actual config (never hardcoded), and the z-up / +x-forward frame is
    stamped explicitly so the viewer never guesses.
    """
    course = CourseConfig()  # default 3-gate course; start (0,0,1); finish x=7
    doc = _record_with_course(connectome, tmp_path / "act", course)

    assert "course" in doc["meta"], "meta.course must be present when a CourseConfig is given"
    block = doc["meta"]["course"]
    assert block == {
        "start": [0.0, 0.0, 1.0],
        "gates": [
            {"center": [2.5, 0.0, 1.0], "aperture": 0.6, "plane": "yz"},
            {"center": [4.0, 0.6, 1.3], "aperture": 0.6, "plane": "yz"},
            {"center": [5.5, -0.5, 0.9], "aperture": 0.6, "plane": "yz"},
        ],
        "finish": {"x": 7.0},
        "floor_z": 0.0,
        "ceiling_z": 2.5,
        "forward_axis": "x",
        "up_axis": "z",
    }
    # The pre-UC-09 singular `gate` block is gone.
    assert "gate" not in block


def test_meta_course_reads_values_from_config_not_hardcoded(
    connectome: ConnectomeData, tmp_path: Path
) -> None:
    """A re-tuned N-gate course flows through — values read from config, not constants (AC7)."""
    course = CourseConfig(
        start_position=(1.0, 2.0, 3.0),
        gates=(
            GateSpec(center=(5.0, 0.5, 1.5), aperture=0.9),
            GateSpec(center=(7.5, -0.5, 2.0), aperture=0.7),
        ),
        finish_x=11.0,
        floor_z=0.2,
        ceiling_z=4.0,
    )
    block = _record_with_course(connectome, tmp_path / "act", course)["meta"]["course"]

    assert block["start"] == [1.0, 2.0, 3.0]
    assert len(block["gates"]) == 2
    assert block["gates"][0]["center"] == [5.0, 0.5, 1.5]
    assert block["gates"][0]["aperture"] == 0.9
    assert block["gates"][0]["plane"] == "yz"
    assert block["gates"][1]["center"] == [7.5, -0.5, 2.0]
    assert block["gates"][1]["aperture"] == 0.7
    assert block["finish"]["x"] == 11.0
    assert block["floor_z"] == 0.2
    assert block["ceiling_z"] == 4.0
    # Axis conventions are fixed by the env frame, independent of tuning.
    assert block["forward_axis"] == "x"
    assert block["up_axis"] == "z"


def test_single_gate_course_serialises_one_element_gates_array(
    connectome: ConnectomeData, tmp_path: Path
) -> None:
    """An N=1 course still serialises as a one-element ``gates`` array (UC-09 AC1/AC7)."""
    block = _record_with_course(connectome, tmp_path / "act", single_gate_course())["meta"][
        "course"
    ]
    assert len(block["gates"]) == 1
    assert block["gates"][0]["center"] == [3.0, 0.0, 1.0]
    assert block["gates"][0]["aperture"] == 0.6
    assert "gate" not in block


def test_course_none_omits_meta_course_block(connectome: ConnectomeData, tmp_path: Path) -> None:
    """``course=None`` (the default) writes NO ``meta.course`` key — back-compatible (AC8)."""
    doc = _record_with_course(connectome, tmp_path / "act", None)
    assert "course" not in doc["meta"], "meta.course must be absent when no course is supplied"
    # The rest of the documented schema is unchanged (no keys added/removed).
    assert set(doc["meta"]) == _DOC_META_KEYS


def test_build_recorder_forwards_course(connectome: ConnectomeData) -> None:
    """The evaluate plumbing exposes a ``course`` param that reaches the recorder (AC8).

    Signature-level check (constructing a full recorder needs a trained model): confirms
    ``_build_recorder`` accepts ``course`` so ``evaluate`` can pass ``ecfg.course`` through.
    """
    import inspect

    from drone_fly.evaluate.evaluator import _build_recorder

    params = inspect.signature(_build_recorder).parameters
    assert "course" in params, "_build_recorder must accept a course argument to plumb AC8"


# ===========================================================================
# UC-08 AC9 — set_course() stamps the per-episode SAMPLED course
# ===========================================================================
def test_set_course_overrides_the_static_default_in_meta(
    connectome: ConnectomeData, tmp_path: Path
) -> None:
    """set_course() makes meta.course reflect a per-episode sampled course, not the ctor default.

    The recorder is built with the fixed default course (as ``evaluate`` does), but each
    episode calls ``set_course(active_course)`` with the geometry the env actually sampled,
    so the recorded ``meta.course`` is the sampled one (UC-06 viewer draws the right markers).
    """
    rec = ActivationRecorder(
        connectome, tmp_path / "act", backend="simple", dt=0.05, course=CourseConfig()
    )
    sampled = CourseConfig(
        start_position=(0.3, -0.7, 1.2),
        gates=(
            GateSpec(center=(3.4, 0.5, 1.4), aperture=0.55),
            GateSpec(center=(5.0, -0.3, 1.1), aperture=0.45),
        ),
        finish_x=6.5,
    )
    n = connectome.neuron_count
    rec.start_episode(episode_index=0, seed=0)
    rec.set_course(sampled)  # the env's active_course for THIS episode
    for f in range(2):
        rec.sink(np.linspace(-1.0, 1.0, n, dtype=np.float32) * (f + 1) / 2)
        rec.capture_frame(np.array([0.1, 0.2, 0.3, 0.4]), np.array([float(f), 0.0, 1.0]))
    path = rec.finish_episode(completed=True, completion_time=0.1, total_reward=1.0, steps=2)

    block = json.loads(path.read_text())["meta"]["course"]
    # The SAMPLED geometry, not the default 3-gate course.
    assert block["start"] == [0.3, -0.7, 1.2]
    assert [g["center"] for g in block["gates"]] == [[3.4, 0.5, 1.4], [5.0, -0.3, 1.1]]
    assert block["gates"][0]["aperture"] == 0.55
    assert block["finish"]["x"] == 6.5


def test_set_course_is_per_episode(connectome: ConnectomeData, tmp_path: Path) -> None:
    """Each episode's meta.course reflects that episode's own set_course value (AC9)."""
    rec = ActivationRecorder(
        connectome, tmp_path / "act", backend="simple", dt=0.05, course=CourseConfig()
    )
    n = connectome.neuron_count
    courses = [
        CourseConfig(start_position=(0.1, 0.0, 1.0), gates=(GateSpec(center=(2.5, 0.0, 1.0)),)),
        CourseConfig(start_position=(-0.2, 0.4, 1.3), gates=(GateSpec(center=(3.8, 0.0, 1.0)),)),
    ]
    written = []
    for ep, course in enumerate(courses):
        rec.start_episode(episode_index=ep, seed=ep)
        rec.set_course(course)
        rec.sink(np.zeros(n, dtype=np.float32))
        rec.capture_frame(np.zeros(ACTION_DIM), np.zeros(3))
        written.append(
            rec.finish_episode(completed=False, completion_time=None, total_reward=0.0, steps=1)
        )

    b0 = json.loads(written[0].read_text())["meta"]["course"]
    b1 = json.loads(written[1].read_text())["meta"]["course"]
    assert b0["start"] == [0.1, 0.0, 1.0]
    assert b0["gates"][0]["center"][0] == 2.5
    assert b1["start"] == [-0.2, 0.4, 1.3]
    assert b1["gates"][0]["center"][0] == 3.8


def test_set_course_none_omits_meta_course(connectome: ConnectomeData, tmp_path: Path) -> None:
    """set_course(None) reverts to the back-compat 'no course block' behaviour (AC9/AC7)."""
    rec = ActivationRecorder(
        connectome, tmp_path / "act", backend="simple", dt=0.05, course=CourseConfig()
    )
    n = connectome.neuron_count
    rec.start_episode(episode_index=0, seed=0)
    rec.set_course(None)
    rec.sink(np.zeros(n, dtype=np.float32))
    rec.capture_frame(np.zeros(ACTION_DIM), np.zeros(3))
    path = rec.finish_episode(completed=False, completion_time=None, total_reward=0.0, steps=1)
    assert "course" not in json.loads(path.read_text())["meta"]


# ===========================================================================
# UC-09 AC7 — per-frame current-target-gate track (frames.target_gate)
# ===========================================================================
def test_capture_frame_records_per_frame_target_gate(
    connectome: ConnectomeData, tmp_path: Path
) -> None:
    """Passing ``target_gate`` to capture_frame serialises a ``frames.target_gate`` track (AC7)."""
    rec = ActivationRecorder(connectome, tmp_path / "act", backend="simple", dt=0.05)
    n = connectome.neuron_count
    rec.start_episode(episode_index=0, seed=0)
    target_gates = [0, 0, 1, 2, 3]
    for tg in target_gates:
        rec.sink(np.zeros(n, dtype=np.float32))
        rec.capture_frame(np.zeros(ACTION_DIM), np.array([0.0, 0.0, 1.0]), target_gate=tg)
    path = rec.finish_episode(
        completed=True, completion_time=0.2, total_reward=1.0, steps=len(target_gates)
    )

    frames = json.loads(path.read_text())["frames"]
    assert "target_gate" in frames
    assert frames["target_gate"] == target_gates
    # One target-gate entry per captured frame, aligned to the shared timeline.
    assert len(frames["target_gate"]) == len(frames["actions"])


def test_omitting_target_gate_leaves_track_absent_backcompat(
    connectome: ConnectomeData, tmp_path: Path
) -> None:
    """A legacy/single-gate recording that omits ``target_gate`` has no track (AC7 back-compat)."""
    rec = ActivationRecorder(connectome, tmp_path / "act", backend="simple", dt=0.05)
    n = connectome.neuron_count
    rec.start_episode(episode_index=0, seed=0)
    for _ in range(3):
        rec.sink(np.zeros(n, dtype=np.float32))
        rec.capture_frame(np.zeros(ACTION_DIM), np.array([0.0, 0.0, 1.0]))  # no target_gate
    path = rec.finish_episode(completed=False, completion_time=None, total_reward=0.0, steps=3)

    frames = json.loads(path.read_text())["frames"]
    assert "target_gate" not in frames
    # The rest of the frames schema is exactly the pre-UC-09 set.
    assert set(frames) == {"activations", "actions", "drone_position"}


# ===========================================================================
# UC-15 AC7 — obstacles stamped additively into meta.course
# ===========================================================================
def test_meta_course_stamps_obstacles_additively(
    connectome: ConnectomeData, tmp_path: Path
) -> None:
    """AC7: a course carrying pillars serialises an additive ``obstacles`` array read from config.

    Each pillar is ``{center: [x, y], radius, height}`` (floor-anchored). The pre-UC-15 course
    fields (start / gates / finish / floor / ceiling / axes) are unchanged — the key is purely
    additive.
    """
    course = CourseConfig(
        obstacles=(
            ObstacleSpec(center=(3.25, 1.5), radius=0.3, height=2.5),
            ObstacleSpec(center=(4.75, -1.6), radius=0.4, height=2.0),
        )
    )
    block = _record_with_course(connectome, tmp_path / "act", course)["meta"]["course"]
    assert block["obstacles"] == [
        {"center": [3.25, 1.5], "radius": 0.3, "height": 2.5},
        {"center": [4.75, -1.6], "radius": 0.4, "height": 2.0},
    ]
    # Values are read from the actual config, never hardcoded — a re-tuned pillar flows through.
    # The rest of the course block is unchanged by the additive key.
    assert block["start"] == [0.0, 0.0, 1.0]
    assert len(block["gates"]) == 3


def test_meta_course_omits_obstacles_when_course_has_none(
    connectome: ConnectomeData, tmp_path: Path
) -> None:
    """AC7 back-compat: a no-obstacle course emits NO ``obstacles`` key (viewer degrades)."""
    block = _record_with_course(connectome, tmp_path / "act", CourseConfig())["meta"]["course"]
    assert "obstacles" not in block


# ===========================================================================
# UC-21 AC1 — pads carry a display-only ``kind`` (recharge / repair / plain),
# derived from the real PadSpec flags, stamped additively + presence-guarded.
# ===========================================================================
def test_meta_course_stamps_pad_kind_from_padspec_flags(
    connectome: ConnectomeData, tmp_path: Path
) -> None:
    """AC1: each serialised pad grows a ``kind`` read from the real ``PadSpec`` flags.

    A course carrying a recharge-only pad, a repair-only pad, and a plain pad round-trips the
    correct ``kind`` (``recharge`` / ``repair`` / ``plain``) alongside the existing floor-anchored
    ``center`` / ``radius`` geometry. Values are read from the actual config, never hardcoded.
    """
    course = CourseConfig(
        pads=(
            PadSpec(center=(1.0, 0.5), radius=0.4, rechargeable=True),
            PadSpec(center=(2.0, -0.5), radius=0.3, repairable=True),
            PadSpec(center=(3.0, 0.0), radius=0.2),
        )
    )
    block = _record_with_course(connectome, tmp_path / "act", course)["meta"]["course"]
    assert block["pads"] == [
        {"center": [1.0, 0.5], "radius": 0.4, "kind": "recharge"},
        {"center": [2.0, -0.5], "radius": 0.3, "kind": "repair"},
        {"center": [3.0, 0.0], "radius": 0.2, "kind": "plain"},
    ]
    # The pre-UC-21 course fields stay unchanged by the additive key.
    assert block["start"] == [0.0, 0.0, 1.0]
    assert len(block["gates"]) == 3


def test_meta_course_pad_kind_precedence_repair_wins_over_recharge(
    connectome: ConnectomeData, tmp_path: Path
) -> None:
    """AC1 edge case: a pad flagged BOTH rechargeable and repairable serialises ``kind=="repair"``.

    The precedence is **display-only** — env behaviour is unchanged (a both-flags pad still both
    recharges and repairs); only the viewer marker resolves to a single colour, and repair wins.
    """
    course = CourseConfig(
        pads=(PadSpec(center=(1.5, 0.0), radius=0.35, rechargeable=True, repairable=True),)
    )
    block = _record_with_course(connectome, tmp_path / "act", course)["meta"]["course"]
    assert block["pads"] == [{"center": [1.5, 0.0], "radius": 0.35, "kind": "repair"}]


def test_meta_course_omits_pads_when_course_has_none(
    connectome: ConnectomeData, tmp_path: Path
) -> None:
    """AC5 back-compat: a no-pad course emits NO ``pads`` key (byte-identical to pre-UC-21)."""
    block = _record_with_course(connectome, tmp_path / "act", CourseConfig())["meta"]["course"]
    assert "pads" not in block
