"""Overlay wire format: runner -> overlay datagrams on ``127.0.0.1:<overlay.port>`` (UC-65).

Two datagram shapes, both one UDP datagram each (never fragmented at the application level):

* **Text** — UTF-8 JSON objects carrying ``v`` (:data:`SCHEMA_VERSION`), ``kind`` and ``run_id``:

  ``tick``    one per runner tick: the tick record (``TickRecord.as_dict()``) plus ``run_state``,
              ``run_info``, cumulative ``missed_deadlines``, publisher ``drops``, ``focus`` /
              ``focus_window`` and the tick's telemetry ``sample`` extras;
  ``meta``    static run facts (extra layout, contract constants, ray cone, age limits, camera,
              world scale, brain info), sent on the first tick and then once per second;
  ``status``  ``run_state`` while the runner waits (``waiting_for_state`` / ``waiting_for_focus``);
  ``stop``    the stop reason and fail-safe outcome (sent twice).

* **Brain** — binary: ``b"DFOB"`` magic, then a little-endian header (version, run id, tick,
  neuron count, scale, offset) and ``n`` ``uint8`` codes, one per K1 neuron, quantised with the
  recorder's scheme (``value = code * scale + offset``; see ``drone_fly.record.recorder``). K1 has
  25,627 neurons, so a frame is ~25.6 KB.

:func:`decode_datagram` never raises: garbled, truncated or foreign datagrams decode to ``None``
and are ignored by the receiver.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
DEFAULT_PORT = 9100
#: Largest datagram the publisher sends (localhost UDP allows ~65,507 bytes; keep headroom).
MAX_DATAGRAM = 60000
TEXT_KINDS = frozenset({"tick", "meta", "status", "stop"})
BRAIN_KIND = "brain"

BRAIN_MAGIC = b"DFOB"
RUN_ID_BYTES = 16  # run ids are 16 ASCII hex characters
#: magic, version, run id, tick, neuron count, scale, offset (little-endian, no padding).
BRAIN_HEADER = struct.Struct(f"<4sB{RUN_ID_BYTES}sIIff")

#: The recorder's quantisation (``drone_fly.record.recorder.ACTIVATION_SCALE`` / ``_OFFSET``),
#: mirrored for publishers built with an injected quantiser. Every brain datagram carries its own
#: scale / offset, so the receiver never assumes these.
DEFAULT_SCALE = 2.0 / 255.0
DEFAULT_OFFSET = -1.0


@dataclass(frozen=True)
class BrainFrame:
    """One decoded brain datagram: ``uint8`` codes for every neuron of one tick."""

    tick: int
    codes: np.ndarray
    scale: float
    offset: float

    @property
    def n(self) -> int:
        return int(self.codes.shape[0])

    def values(self) -> np.ndarray:
        """Dequantised activations (float32, ``[-1, 1]`` for the recorder's scheme)."""
        return self.codes.astype(np.float32) * np.float32(self.scale) + np.float32(self.offset)


@dataclass(frozen=True)
class Packet:
    """A decoded datagram. ``data`` holds the JSON object (text kinds); ``brain`` the frame."""

    kind: str
    run_id: str
    data: dict[str, Any] = field(default_factory=dict)
    brain: BrainFrame | None = None


def encode_json(kind: str, run_id: str, payload: dict[str, Any]) -> bytes:
    """``payload`` as one text datagram (``v`` / ``kind`` / ``run_id`` take precedence)."""
    if kind not in TEXT_KINDS:
        raise ValueError(f"unknown text datagram kind {kind!r}")
    body = {**payload, "v": SCHEMA_VERSION, "kind": kind, "run_id": run_id}
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def encode_brain(
    run_id: str,
    tick: int,
    codes: np.ndarray,
    scale: float = DEFAULT_SCALE,
    offset: float = DEFAULT_OFFSET,
) -> bytes:
    """One brain datagram: header + ``uint8`` codes."""
    c = np.ascontiguousarray(np.asarray(codes, dtype=np.uint8).reshape(-1))
    rid = run_id.encode("ascii")
    if len(rid) != RUN_ID_BYTES:
        raise ValueError(f"run_id must be {RUN_ID_BYTES} ASCII characters, got {run_id!r}")
    header = BRAIN_HEADER.pack(
        BRAIN_MAGIC, SCHEMA_VERSION, rid, int(tick), int(c.shape[0]), float(scale), float(offset)
    )
    return header + c.tobytes()


def _decode_brain(data: bytes) -> Packet | None:
    if len(data) < BRAIN_HEADER.size:
        return None
    magic, version, rid, tick, n, scale, offset = BRAIN_HEADER.unpack_from(data)
    if magic != BRAIN_MAGIC or version != SCHEMA_VERSION:
        return None
    if len(data) != BRAIN_HEADER.size + n:
        return None
    if not (np.isfinite(scale) and np.isfinite(offset)):
        return None
    codes = np.frombuffer(data, dtype=np.uint8, count=n, offset=BRAIN_HEADER.size).copy()
    run_id = rid.decode("ascii")
    frame = BrainFrame(tick=int(tick), codes=codes, scale=float(scale), offset=float(offset))
    return Packet(kind=BRAIN_KIND, run_id=run_id, brain=frame)


def _decode_text(data: bytes) -> Packet | None:
    obj = json.loads(data.decode("utf-8"))
    if not isinstance(obj, dict) or obj.get("v") != SCHEMA_VERSION:
        return None
    kind, run_id = obj.get("kind"), obj.get("run_id")
    if kind not in TEXT_KINDS or not isinstance(run_id, str):
        return None
    return Packet(kind=kind, run_id=run_id, data=obj)


def decode_datagram(data: bytes) -> Packet | None:
    """Decode one datagram; ``None`` for anything garbled or foreign (never raises)."""
    try:
        if data[:4] == BRAIN_MAGIC:
            return _decode_brain(bytes(data))
        return _decode_text(bytes(data))
    except Exception:  # noqa: BLE001 - a bad datagram is ignored, never fatal
        return None
