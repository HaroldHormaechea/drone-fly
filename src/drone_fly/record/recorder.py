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
from drone_fly.controller.modality import ModalityError, select_modality
from drone_fly.env.config import CourseConfig
from drone_fly.record.coordinates import DEFAULT_PROJECTION, neuron_roles, resolve_positions

logger = logging.getLogger(__name__)

#: Recording schema version. Bump on any breaking layout change so the viewer can refuse
#: incompatible files loudly rather than mis-render them.
SCHEMA_VERSION = 1

#: Modalities tagged per-neuron in ``meta.modality`` for the viewer's UC-28 modality overlay
#: (AC-7). This is exactly the set of UC-13 modality rules that both *exist* and *resolve* over
#: a recorded slice: ``vision`` (superclass), ``proprioceptive`` (class), and ``hunger``
#: (cell_type, approximate). ``damage`` / nociception is **deliberately absent**: MaleCNS ships
#: no nociceptive label and there is no modality rule for it, so it is documented as unavailable
#: (here, in the README, and in the viewer legend) rather than silently faked. Order is the
#: tie-break precedence when a neuron matches more than one modality (first wins).
RECORDED_MODALITIES: tuple[str, ...] = ("vision", "proprioceptive", "hunger")

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
        # UC-28: per-neuron modality tag for the viewer's modality overlay (AC-7), computed once
        # here from the connectome's biological labels. Fail-soft — an absent modality is simply
        # not tagged, never a crash (see _resolve_modalities).
        self.modality = self._resolve_modalities(data)
        # UC-27: positions are provisioned at slice/artifact time and only LOADED here (fast
        # path). If this artifact predates the sidecar (older slice), self-heal: compute once,
        # WARN, and persist it beside the connectome (best-effort — a read-only dir must not
        # crash recording) so the next run takes the fast path. ``persist_strict=False`` makes
        # a write failure non-fatal.
        positions = resolve_positions(
            data, projection=projection, persist=True, persist_strict=False
        )
        if positions is None:
            # Only the large-connectome guard returns None: recording a full-scale connectome
            # (> spectral cap) with zero real anatomy is unsupported — with no real soma to
            # anchor a schematic body it would need an infeasible dense spectral layout. Prune
            # it first, or provide anatomy. Fail clearly rather than serialise empty positions.
            raise RuntimeError(
                "Neuron positions could not be provisioned for this connectome: it exceeds the "
                "spectral-layout cap and has zero anatomy. Prune the connectome before "
                "recording, or provide anatomy via NEUPRINT_TOKEN or DRONE_FLY_SOMA_CSV."
            )
        self.positions = positions
        self._reset_episode_state()

    def _resolve_modalities(self, data: ConnectomeData) -> list[str]:
        """Return one modality tag per neuron (index-aligned; ``""`` when none) — AC-7.

        For each modality in :data:`RECORDED_MODALITIES` (in precedence order), resolve its
        neuron population via :func:`drone_fly.controller.modality.select_modality` and tag the
        matched indices — first match wins, so a neuron in two populations keeps the earlier
        one. Any :class:`ModalityError` (absent population, missing label column) is caught and
        that modality is simply skipped: recording never fails because a modality is not present
        in this slice (fail-soft). Read-only w.r.t. the connectome — no training/checkpoint side
        effects (AC-9).
        """
        tags = [""] * self.n_neurons
        for name in RECORDED_MODALITIES:
            try:
                selection = select_modality(data, name)
            except ModalityError:
                continue
            for idx in selection.indices.tolist():
                if tags[idx] == "":
                    tags[idx] = name
        return tags

    # -- episode lifecycle --------------------------------------------------------------
    def _reset_episode_state(self) -> None:
        self._episode_index: int | None = None
        self._seed: int | None = None
        self._pending: np.ndarray | None = None
        self._activations: list[list[int]] = []
        self._actions: list[list[float]] = []
        self._positions: list[list[float]] = []
        # Per-frame current-target-gate index (UC-09). Populated only when the caller passes
        # ``target_gate`` to :meth:`capture_frame`; serialised as ``frames.target_gate`` when
        # present, omitted entirely otherwise (keeps older single-gate files schema-valid).
        self._target_gates: list[int] = []
        self._has_target_gates = False

    def start_episode(self, episode_index: int, seed: int | None = None) -> None:
        """Begin a fresh episode buffer (discarding any half-captured previous episode)."""
        self._reset_episode_state()
        self._episode_index = int(episode_index)
        self._seed = None if seed is None else int(seed)

    def set_course(self, course: CourseConfig | None) -> None:
        """Set the course stamped into this episode's ``meta.course`` (UC-08 AC9).

        With domain randomization on, the caller passes the env's actual per-episode
        ``active_course`` so the recorded ``meta.course`` reflects the **sampled** geometry
        (not the static config default), and the UC-06 viewer draws the right markers.
        """
        self.course = course

    # -- capture ------------------------------------------------------------------------
    def sink(self, activation: np.ndarray) -> None:
        """Actor hook: stash the latest post-propagation activation (pull-based capture).

        Wired onto ``actor.sink``; the actor calls this with a **detached, cloned** copy of
        its ``(B, N)`` (or ``(N,)``) post-propagation state. Storing (not appending) makes
        capture double-safe: :meth:`capture_frame` reads the most recent value once per step.
        """
        self._pending = activation

    def capture_frame(
        self,
        action: np.ndarray,
        drone_position: np.ndarray,
        target_gate: int | None = None,
    ) -> None:
        """Commit one frame: the pending activation + this step's action and drone position.

        Raises if no activation was captured since the last frame (the actor ``sink`` was not
        wired, or forward never ran) or if the activation width disagrees with ``n_neurons``
        (the alignment guard — a checkpoint/connectome mismatch).

        ``target_gate`` (UC-09): the current target-gate index for this frame (``info``'s
        ``target_gate``). When supplied it is recorded per-frame and serialised as
        ``frames.target_gate`` so the viewer can highlight the gate being chased. Omit it
        (``None``) for single-gate / legacy recordings — the key is then left out entirely.
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
        if target_gate is not None:
            self._has_target_gates = True
            self._target_gates.append(int(target_gate))
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
            "modality": self.modality,
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

        frames: dict = {
            "activations": self._activations,
            "actions": self._actions,
            "drone_position": self._positions,
        }
        if self._has_target_gates:
            # Additive per-frame current-target-gate track (UC-09 AC7). Present only when the
            # caller supplied target_gate; older/single-gate recordings omit it and the viewer
            # falls back to no highlight.
            frames["target_gate"] = self._target_gates

        document = {
            "schema_version": SCHEMA_VERSION,
            "meta": meta,
            "frames": frames,
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
    """Serialise :class:`CourseConfig` into the additive ``meta.course`` block (UC-09 AC7).

    Emits **all N gates** as a ``gates: [...]`` array (each ``{center, aperture, plane}``),
    replacing the pre-UC-09 singular ``gate`` block. Values are read from the actual config
    (never hardcoded) so a re-tuned or sampled course flows through to the viewer. Coordinate
    frame is z-up / +x-forward / right-handed — recorded explicitly via ``forward_axis`` /
    ``up_axis`` so the viewer never has to guess.
    """
    meta = {
        "start": [float(v) for v in course.start_position],
        "gates": [
            {
                "center": [
                    float(gate.center[0]),
                    float(gate.center[1]),
                    float(gate.center[2]),
                ],
                "aperture": float(gate.aperture),
                "plane": "yz",
            }
            for gate in course.gates
        ],
        "finish": {"x": float(course.finish_x)},
        "floor_z": float(course.floor_z),
        "ceiling_z": float(course.ceiling_z),
        "forward_axis": "x",
        "up_axis": "z",
    }
    # UC-15: obstacle pillars stamped **additively** — the key is emitted only when the course
    # actually has obstacles, so no-obstacle runs and every pre-UC-15 recording stay byte-for-byte
    # unchanged (the viewer degrades gracefully when the field is absent). Floor-anchored: base at
    # ``floor_z``, top at ``floor_z + height``.
    obstacles = getattr(course, "obstacles", ())
    if obstacles:
        meta["obstacles"] = [
            {
                "center": [float(o.center[0]), float(o.center[1])],
                "radius": float(o.radius),
                "height": float(o.height),
            }
            for o in obstacles
        ]
    # UC-16: landing/takeoff pads stamped **additively** and presence-guarded exactly like the
    # obstacles block above — the key is emitted only when the course actually has pads, so no-pad
    # runs and every pre-UC-16 recording stay byte-for-byte unchanged (the viewer degrades
    # gracefully when the field is absent). Floor-anchored: a pad sits on ``floor_z`` with a
    # horizontal disc of ``radius`` about ``center``.
    #
    # UC-21: each pad additionally carries a display-only ``kind`` derived from the real
    # ``PadSpec.rechargeable`` / ``repairable`` flags so the viewer can colour pad kinds apart.
    # Precedence is **display-only** (repair wins over recharge for the marker) — env behaviour is
    # unchanged (a both-flags pad still recharges *and* repairs). The mapping: repairable (incl.
    # both flags) → ``"repair"``; rechargeable-only → ``"recharge"``; neither → ``"plain"``. It is
    # appended after ``radius`` within the already-presence-guarded pads block, so no-pad courses
    # and every pre-UC-21 recording stay byte-for-byte unchanged.
    pads = getattr(course, "pads", ())
    if pads:
        meta["pads"] = [
            {
                "center": [float(p.center[0]), float(p.center[1])],
                "radius": float(p.radius),
                "kind": (
                    "repair"
                    if getattr(p, "repairable", False)
                    else "recharge"
                    if getattr(p, "rechargeable", False)
                    else "plain"
                ),
            }
            for p in pads
        ]
    return meta


def _is_int_like(value: object) -> bool:
    try:
        int(value)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False
