"""Load a recorded flight (manual capture or harness run folder) as frames paired with telemetry.

Two layouts are read:

* **manual capture** (``capture.py`` in the owner's reference scripts): ``telemetry_raw.csv``
  (``t_wall_ns,len,hex`` — one raw UDP datagram per row) + ``frames.csv`` (``t_wall_ns,file``)
  + ``frames/``. Both use the same wall clock; datagrams are parsed with the configured
  :class:`~drone_fly.harness.liftoff.telemetry.TelemetryFormat`.
* **harness run folder** (:mod:`~drone_fly.harness.recording`): ``telemetry.csv`` + ``frames.csv``
  (``tick,frame_index,stamp,...,file``) + ``frames/``, both on the runner's ``perf_counter``.

The telemetry stream is cut into **segments** at wall-clock gaps > ``gap_s`` (1 s), sim-time
resets and per-sample position jumps (``telemetry.max_position_jump_m``, i.e. respawns). Each frame
(shifted by ``lag_ms``) is paired with the nearest datagram within ``pair_tol_s`` (30 ms); frames
outside a segment or within ``cut_margin_s`` (0.5 s) of a cut are dropped, with the reason counted.
"""

from __future__ import annotations

import csv
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.liftoff.telemetry import (
    TelemetryFormat,
    converter_from_config,
    read_telemetry_csv,
)
from drone_fly.harness.types import FlightState, TelemetrySample


@dataclass(frozen=True)
class CaptureFrame:
    """One usable frame: image path + the paired telemetry sample and converted state."""

    index: int  # row in frames.csv
    stamp: float  # frame time (seconds, the telemetry clock)
    path: Path
    sample_index: int
    sample: TelemetrySample
    state: FlightState  # contract frame, scaled by the requested world scale
    segment: int


@dataclass
class Capture:
    root: Path
    layout: str  # "manual" | "run"
    samples: list[TelemetrySample]
    states: list[FlightState]
    stamps: np.ndarray  # receive time per sample (seconds)
    segments: list[tuple[int, int]]  # [start, end) sample index ranges
    frames: list[CaptureFrame] = field(default_factory=list)
    n_frames_total: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.root.name

    def summary(self) -> dict[str, Any]:
        return {
            "capture": str(self.root),
            "layout": self.layout,
            "samples": len(self.samples),
            "segments": [
                {
                    "start": a,
                    "end": b,
                    "t0": float(self.stamps[a]),
                    "t1": float(self.stamps[b - 1]),
                    "duration_s": float(self.stamps[b - 1] - self.stamps[a]),
                }
                for a, b in self.segments
            ],
            "frames_total": self.n_frames_total,
            "frames_paired": len(self.frames),
            "dropped": dict(self.dropped),
        }


def _read_manual(root: Path, fmt: TelemetryFormat) -> tuple[list[TelemetrySample], int]:
    samples, bad = [], 0
    with (root / "telemetry_raw.csv").open(encoding="ascii") as fh:
        next(fh, None)
        for line in fh:
            parts = line.rstrip().split(",")
            if len(parts) < 3:
                continue
            try:
                samples.append(fmt.parse(bytes.fromhex(parts[2]), received_at=int(parts[0]) / 1e9))
            except (ValueError, struct.error):
                bad += 1
    return samples, bad


def _read_frames(root: Path, layout: str) -> list[tuple[int, float, Path]]:
    path = root / "frames.csv"
    if not path.is_file():
        raise HarnessConfigError(f"{root} has no frames.csv")
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if not row.get("file"):
                continue
            if layout == "manual":
                out.append((i, int(row["t_wall_ns"]) / 1e9, root / row["file"]))
            else:
                out.append((int(row["frame_index"]), float(row["stamp"]), root / row["file"]))
    return out


def segment_samples(
    samples: list[TelemetrySample], stamps: np.ndarray, max_jump_m: float, gap_s: float = 1.0
) -> list[tuple[int, int]]:
    """``[start, end)`` ranges split at receive gaps, sim-time resets and position jumps."""
    if not samples:
        return []
    t = np.array([s.t for s in samples])
    pos = np.array([np.asarray(s.position, np.float64) for s in samples])
    jump = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    cut = np.flatnonzero((np.diff(stamps) > gap_s) | (np.diff(t) < 0) | (jump > max_jump_m)) + 1
    bounds = [0, *cut.tolist(), len(samples)]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b - a >= 2]


def load_capture(
    root: str | Path,
    telemetry_cfg: dict[str, Any],
    world_scale: float = 1.0,
    lag_ms: float = 0.0,
    *,
    pair_tol_s: float = 0.03,
    cut_margin_s: float = 0.5,
    gap_s: float = 1.0,
) -> Capture:
    """Read a capture / run folder and pair its frames with telemetry (see the module docstring)."""
    root = Path(root)
    if (root / "telemetry_raw.csv").is_file():
        layout = "manual"
        samples, bad = _read_manual(root, TelemetryFormat.from_config(telemetry_cfg))
    elif (root / "telemetry.csv").is_file():
        layout, bad = "run", 0
        samples = read_telemetry_csv(root / "telemetry.csv")
    else:
        raise HarnessConfigError(
            f"{root} is neither a manual capture (telemetry_raw.csv) nor a harness run folder"
        )
    if not samples:
        raise HarnessConfigError(f"{root} has no parseable telemetry")
    order = np.argsort([s.received_at for s in samples], kind="stable")
    samples = [samples[i] for i in order]
    stamps = np.array([s.received_at for s in samples], np.float64)
    conv = converter_from_config(telemetry_cfg, world_scale)
    states = [conv.convert(s) for s in samples]
    max_jump = float(telemetry_cfg.get("max_position_jump_m", 5.0))
    segments = segment_samples(samples, stamps, max_jump, gap_s)
    cap = Capture(root, layout, samples, states, stamps, segments)
    dropped: Counter[str] = Counter()
    if bad:
        dropped["unparseable_datagrams"] = bad
    seg_of = np.full(len(samples), -1, np.int64)
    for k, (a, b) in enumerate(segments):
        seg_of[a:b] = k
    frames = _read_frames(root, layout)
    cap.n_frames_total = len(frames)
    lag = lag_ms / 1000.0
    for idx, stamp, path in frames:
        t = stamp - lag
        j = int(np.searchsorted(stamps, t))
        cands = [i for i in (j - 1, j) if 0 <= i < len(stamps)]
        i = min(cands, key=lambda c: abs(stamps[c] - t))
        if abs(stamps[i] - t) > pair_tol_s:
            dropped["no_telemetry_within_tolerance"] += 1
            continue
        k = int(seg_of[i])
        if k < 0:
            dropped["outside_segment"] += 1
            continue
        a, b = segments[k]
        if t - stamps[a] < cut_margin_s or stamps[b - 1] - t < cut_margin_s:
            dropped["near_cut"] += 1
            continue
        if not path.is_file():
            dropped["missing_image"] += 1
            continue
        cap.frames.append(CaptureFrame(idx, stamp, path, i, samples[i], states[i], k))
    cap.dropped = dict(dropped)
    return cap


def load_frame_image(path: Path) -> np.ndarray:
    """A recorded frame (JPEG/PNG via OpenCV, or ``.npy``) as BGR uint8."""
    from drone_fly.harness.offline.evaluate import load_image

    return load_image(path)
