"""Activation recorder: self-contained per-episode playback files (UC-05, AC1/AC3/AC10).

An :class:`ActivationRecorder` turns a rollout into one **self-contained** file per
recorded episode under ``artifacts/activations/`` that the static viewer (``viz/``) — and
any future tool — can consume without the model, the connectome, or the code that produced
it. Everything the viewer needs travels in the file: the ordered ``neuron_ids``, per-neuron
``superclass`` / ``role``, soma ``positions`` (provisioned once — see
:mod:`drone_fly.record.coordinates`), the per-frame neuron activations, the per-frame
4-channel action and drone position, and the episode outcome.

Capture is **pull-based** to make it non-invasive and double-capture-proof: the actor's
opt-in ``sink`` stashes the latest post-propagation activation
(:meth:`ActivationRecorder.sink`); the rollout then pulls exactly one frame per env step via
:meth:`capture_frame`. Even if the policy's forward runs more than once per step (e.g. an
SB3 value pass), only the last activation before the ``capture_frame`` call is kept, so a
frame always pairs the activation that produced *this* step's action with that action.

File-size management (AC10)
---------------------------
Activations are bounded to ``[-1, 1]`` by the policy's ``tanh`` output, so they are stored
**quantised to ``uint8``** over that range (``activation = value * scale + offset``, with
``scale``/``offset`` recorded in the metadata for exact dequantisation). This is ~4× smaller
than float32 and lossless to 8 bits; optional gzip compresses further. Each written file's
path and on-disk size are logged.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import numpy as np

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.encoding import ACTION_DIM, ACTION_LAYOUT
from drone_fly.env.config import CourseConfig
from drone_fly.record.coordinates import DEFAULT_PROJECTION, neuron_roles, provision_positions

logger = logging.getLogger(__name__)

#: Recording schema version. Bump on any breaking layout change so the viewer can refuse
#: incompatible files loudly rather than mis-render them.
SCHEMA_VERSION = 1

#: Default output directory for recorded episodes (relative to the working directory).
DEFAULT_RECORD_DIR = "artifacts/activations"

#: Quantisation range for the ``uint8`` activation store. The policy's per-neuron state is
#: ``tanh``-bounded to ``[-1, 1]`` (see ``controller/policy.py``), so this range never clips.
ACTIVATION_MIN = -1.0
ACTIVATION_MAX = 1.0
_QUANT_LEVELS = 255

#: ``value = code * ACTIVATION_SCALE + ACTIVATION_OFFSET`` — recorded in metadata so a
#: consumer dequantises exactly. ``code = round((value - offset) / scale)`` clamped to 0..255.
ACTIVATION_OFFSET = ACTIVATION_MIN
ACTIVATION_SCALE = (ACTIVATION_MAX - ACTIVATION_MIN) / _QUANT_LEVELS

#: Lower-case action channel layout stored in the file (viewer labels the flight traces).
ACTION_LAYOUT_LOWER = [c.lower() for c in ACTION_LAYOUT]


def quantize_activation(activation: np.ndarray) -> np.ndarray:
    """Quantise a ``[-1, 1]`` activation vector to ``uint8`` codes (0..255)."""
    codes = np.round(
        (np.asarray(activation, dtype=np.float64) - ACTIVATION_OFFSET) / ACTIVATION_SCALE
    )
    return np.clip(codes, 0, _QUANT_LEVELS).astype(np.uint8)


def dequantize_activation(codes: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize_activation` (for tests / downstream tooling)."""
    return np.asarray(codes, dtype=np.float64) * ACTIVATION_SCALE + ACTIVATION_OFFSET


class ActivationRecorder:
    """Accumulate per-frame activations for one episode at a time and serialise them.

    Parameters
    ----------
    data:
        The (pruned) connectome the policy runs over. Supplies ``neuron_ids`` /
        ``superclass`` / roles / soma positions. **Must be the same graph the checkpoint's
        actor was built from** — the caller enforces the alignment assertion
        (``len(neuron_ids) == actor.n_neurons``); :meth:`capture_frame` additionally checks
        each activation's width against ``n_neurons``.
    out_dir:
        Directory for ``episode_<n>.json`` files (created on write).
    gzip_output:
        When ``True`` write ``episode_<n>.json.gz`` (gzip) instead of plain JSON.
    backend / checkpoint / dt / seed:
        Recorded verbatim in each file's metadata for provenance/reproducibility.
    projection:
        Top-down projection plane for the stored 2-D positions (default ``"xz"``, dorsal).
    course:
        Optional course geometry (start / gate / finish / floor / ceiling from
        :mod:`drone_fly.env.config`). When given, the semantic anchors are serialised as an
        **additive** ``meta.course`` block so the viewer can place the 3-D floor and the
        start / gate / finish markers (UC-06 AC8). ``None`` omits the block entirely, keeping
        older files back-compatible; the viewer degrades gracefully when it is absent.
    """

    def __init__(
        self,
        data: ConnectomeData,
        out_dir: str | Path = DEFAULT_RECORD_DIR,
        *,
        gzip_output: bool = False,
        backend: str | None = None,
        checkpoint: str | None = None,
        dt: float | None = None,
        projection: str = DEFAULT_PROJECTION,
        course: CourseConfig | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.gzip_output = gzip_output
        self.backend = backend
        self.checkpoint = checkpoint
        self.dt = dt
        self.course = course
        self.n_neurons = data.neuron_count
        self.neuron_ids = [
            int(x) if _is_int_like(x) else str(x) for x in np.asarray(data.neuron_ids)
        ]
        self.superclass = (
            [str(s) for s in np.asarray(data.superclass).tolist()]
            if data.superclass is not None
            else ["unknown"] * self.n_neurons
        )
        self.roles = neuron_roles(data)
        self.positions = provision_positions(data, projection=projection)
        self._reset_episode_state()

    # -- episode lifecycle --------------------------------------------------------------
    def _reset_episode_state(self) -> None:
        self._episode_index: int | None = None
        self._seed: int | None = None
        self._pending: np.ndarray | None = None
        self._activations: list[list[int]] = []
        self._actions: list[list[float]] = []
        self._positions: list[list[float]] = []

    def start_episode(self, episode_index: int, seed: int | None = None) -> None:
        """Begin a fresh episode buffer (discarding any half-captured previous episode)."""
        self._reset_episode_state()
        self._episode_index = int(episode_index)
        self._seed = None if seed is None else int(seed)

    # -- capture ------------------------------------------------------------------------
    def sink(self, activation: np.ndarray) -> None:
        """Actor hook: stash the latest post-propagation activation (pull-based capture).

        Wired onto ``actor.sink``; the actor calls this with a **detached, cloned** copy of
        its ``(B, N)`` (or ``(N,)``) post-propagation state. Storing (not appending) makes
        capture double-safe: :meth:`capture_frame` reads the most recent value once per step.
        """
        self._pending = activation

    def capture_frame(self, action: np.ndarray, drone_position: np.ndarray) -> None:
        """Commit one frame: the pending activation + this step's action and drone position.

        Raises if no activation was captured since the last frame (the actor ``sink`` was not
        wired, or forward never ran) or if the activation width disagrees with ``n_neurons``
        (the alignment guard — a checkpoint/connectome mismatch).
        """
        if self._episode_index is None:
            raise RuntimeError("capture_frame() called before start_episode().")
        if self._pending is None:
            raise RuntimeError(
                "No activation captured this step: wire actor.sink = recorder.sink before "
                "the policy forward pass."
            )
        activation = np.asarray(self._pending, dtype=np.float32)
        if activation.ndim == 2:  # (B, N) — recording assumes a single env (B == 1)
            activation = activation[0]
        if activation.shape[-1] != self.n_neurons:
            raise ValueError(
                f"Activation width ({activation.shape[-1]}) != connectome neuron count "
                f"({self.n_neurons}); the checkpoint's actor and the recorder's connectome "
                f"are misaligned. Pass the same --connectome/--prune used for training."
            )
        self._activations.append(quantize_activation(activation).tolist())
        action_vec = np.asarray(action, dtype=np.float64).reshape(-1)
        self._actions.append([float(v) for v in action_vec[:ACTION_DIM]])
        pos_vec = np.asarray(drone_position, dtype=np.float64).reshape(-1)
        self._positions.append([float(v) for v in pos_vec[:3]])
        self._pending = None

    @property
    def n_frames(self) -> int:
        """Frames captured in the current episode so far."""
        return len(self._activations)

    # -- serialisation ------------------------------------------------------------------
    def finish_episode(
        self,
        *,
        completed: bool,
        completion_time: float | None,
        total_reward: float,
        steps: int,
    ) -> Path:
        """Write the current episode to ``episode_<n>.json[.gz]`` and return its path."""
        if self._episode_index is None:
            raise RuntimeError("finish_episode() called before start_episode().")

        meta = {
            "neuron_ids": self.neuron_ids,
            "superclass": self.superclass,
            "roles": self.roles,
            "positions": self.positions,
            "episode_index": self._episode_index,
            "seed": self._seed,
            "checkpoint": self.checkpoint,
            "backend": self.backend,
            "n_frames": self.n_frames,
            "n_neurons": self.n_neurons,
            "action_layout": ACTION_LAYOUT_LOWER,
            "dt": self.dt,
            "activation_scale": ACTIVATION_SCALE,
            "activation_offset": ACTIVATION_OFFSET,
        }
        if self.course is not None:
            # Additive, back-compatible block (UC-06 AC8): semantic course anchors only —
            # the viewer owns display sizing. Absent when no course is supplied.
            meta["course"] = _course_meta(self.course)

        document = {
            "schema_version": SCHEMA_VERSION,
            "meta": meta,
            "frames": {
                "activations": self._activations,
                "actions": self._actions,
                "drone_position": self._positions,
            },
            "outcome": {
                "completed": bool(completed),
                "completion_time": (None if completion_time is None else float(completion_time)),
                "total_reward": float(total_reward),
                "steps": int(steps),
            },
        }

        self.out_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".json.gz" if self.gzip_output else ".json"
        path = self.out_dir / f"episode_{self._episode_index}{suffix}"
        payload = json.dumps(document, separators=(",", ":"))
        if self.gzip_output:
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(payload)
        else:
            path.write_text(payload, encoding="utf-8")

        size_mb = path.stat().st_size / 1e6
        logger.info(
            "Recorded episode %d: %d frames x %d neurons -> %s (%.3f MB).",
            self._episode_index,
            self.n_frames,
            self.n_neurons,
            path,
            size_mb,
        )
        return path


def _course_meta(course: CourseConfig) -> dict:
    """Serialise :class:`CourseConfig` into the additive ``meta.course`` block (UC-06 AC8).

    Values are read from the actual config (never hardcoded) so a re-tuned course flows
    through to the viewer. Coordinate frame is z-up / +x-forward / right-handed — recorded
    explicitly via ``forward_axis`` / ``up_axis`` so the viewer never has to guess.
    """
    return {
        "start": [float(v) for v in course.start_position],
        "gate": {
            "center": [
                float(course.gate_x),
                float(course.gate_center_y),
                float(course.gate_center_z),
            ],
            "aperture": float(course.gate_aperture),
            "plane": "yz",
        },
        "finish": {"x": float(course.finish_x)},
        "floor_z": float(course.floor_z),
        "ceiling_z": float(course.ceiling_z),
        "forward_axis": "x",
        "up_axis": "z",
    }


def _is_int_like(value: object) -> bool:
    try:
        int(value)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False
