"""Synthetic fixtures for the UC-78 eyes tests (CPU-only, hermetic: no captures, no artifacts/).

* Geometry helpers: a contract-free Fibonacci cone, axis-angle quaternions, FlightStates.
* :func:`write_capture` writes a UC71-layout capture (``meta.json``, ``frames.csv``, tiny ``.npy``
  frames, a dummy telemetry file for the capture sha) and :func:`write_labels` its eyes labels
  (``eyes_labels.npz`` + ``eyes_labels_meta.json``) with random but well-formed arrays, so the
  dataset / train / eval / bench path runs without Liftoff, track data or OpenCV.
* :func:`fake_catalogue` is an in-memory track catalogue (map -> split).
* :func:`tiny_dataset` builds a three-map dataset (train / val / held-out) at a small input size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.types import FlightState

#: Small network input for CPU smoke runs (5 stride-2 stages still fit: 64x32 -> 2x1).
SMALL_W, SMALL_H = 64, 32
FRAME_SIZE = (48, 20)  # (w, h) of the synthetic capture frames
MAPS = {"MapA": "train", "MapB": "val", "MapC": "held-out"}


def fib_cone(n: int = 40, half_deg: float = 45.0) -> np.ndarray:
    """A contract-free Fibonacci cone ``(n, 3)`` in (right, up, forward) coordinates."""
    i = np.arange(n) + 0.5
    cos_min = np.cos(np.radians(half_deg))
    z = 1.0 - (1.0 - cos_min) * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1).astype(np.float32)


def quat_axis(axis: Any, deg: float) -> np.ndarray:
    """``[w, x, y, z]`` rotation by ``deg`` about ``axis`` (body -> world)."""
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    h = np.radians(deg) / 2.0
    return np.array([np.cos(h), *(np.sin(h) * a)])


IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def state(pos=(0, 0, 1), vel=(2, 0, 0), quat=IDENTITY, discontinuity=False) -> FlightState:
    return FlightState(
        pos=np.asarray(pos, np.float32),
        vel=np.asarray(vel, np.float32),
        quat=np.asarray(quat, np.float32),
        omega=np.zeros(3, np.float32),
        discontinuity=discontinuity,
    )


# ----------------------------------------------------------------------------- captures


def write_capture(
    root: Path,
    *,
    catalogue_id: str | None,
    map_name: str | None,
    n_frames: int = 20,
    fps: float = 10.0,
    verified: bool = True,
    uptilt_deg: float = 20.0,
    meta: bool = True,
    seed: int = 0,
) -> Path:
    """A capture folder (UC71 layout) with ``n_frames`` tiny ``.npy`` frames."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "frames").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    w, h = FRAME_SIZE
    rows = ["t_wall_ns,file"]
    for k in range(n_frames):
        name = f"frames/{k:04d}.npy"
        np.save(root / name, rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        rows.append(f"{int(round(k / fps * 1e9))},{name}")
    (root / "frames.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "telemetry_raw.csv").write_text(f"t_wall_ns,len,hex\n{seed},1,00\n", encoding="ascii")
    if meta:
        m = {
            "schema_version": 1,
            "catalogue_id": catalogue_id,
            "map": map_name,
            "camera": {"resolution": [2560, 1080], "hfov_deg": 120.0, "uptilt_deg": uptilt_deg},
            "alignment": {"verified": verified, "lag_ms": 12.0},
            "pose_source": "telemetry",
            "source": "replay",
        }
        (root / "meta.json").write_text(json.dumps(m), encoding="utf-8")
    return root


def label_settings(contract: Any, world_scale: float = 4.0, s: float = 1.0, **over: Any):
    from dataclasses import replace

    from drone_fly.eyes.labels import settings_for

    return replace(settings_for(contract, world_scale, s), **over)


def write_labels(
    labels_dir: Path,
    capture: Path,
    settings: Any,
    *,
    catalogue_id: str,
    map_name: str,
    split: str,
    fps: float = 10.0,
    uptilt_deg: float = 20.0,
    segments: np.ndarray | None = None,
    seed: int = 0,
    n_rays: int = 40,
) -> Path:
    """Well-formed eyes labels for every frame of ``capture`` (random contents)."""
    from drone_fly.eyes.capture_meta import capture_sha
    from drone_fly.eyes.labels import LABELS_FILE, META_FILE

    files = [ln.split(",")[1] for ln in (capture / "frames.csv").read_text().splitlines()[1:]]
    n = len(files)
    rng = np.random.default_rng(seed)
    free = rng.uniform(0.05, 1.0, (n, n_rays)).astype(np.float32)
    free[rng.random((n, n_rays)) < 0.5] = 1.0
    hit = free < 1.0
    rc = rng.normal(size=(n, n_rays, 3)).astype(np.float32)
    rc /= np.linalg.norm(rc, axis=2, keepdims=True)
    arr = {
        "free": free,
        "hit": hit,
        "dist_m": free * np.float32(settings.range_m),
        "hit_min_half_m": np.where(hit, rng.uniform(0.02, 2.0, (n, n_rays)), np.nan).astype(
            np.float32
        ),
        "aux_free": free.copy(),
        "aux_kind": np.where(hit, 1, 0).astype(np.int8),
        "rays_cam": rc,
        "uv": rng.uniform(-1, 1, (n, n_rays, 2)).astype(np.float32),
        "in_view": rng.random((n, n_rays)) < 0.7,
        "ego": rng.normal(size=(n, 10)).astype(np.float32),
        "warning_s": rng.uniform(0.2, 2.0, n).astype(np.float32),
        "speed_mps": rng.uniform(5, 30, n).astype(np.float32),
        "stamp": np.arange(n, dtype=np.float64) / fps,
        "segment": np.zeros(n, np.int64) if segments is None else np.asarray(segments, np.int64),
        "frame_file": np.array(files),
        "frame_index": np.arange(n, dtype=np.int64),
    }
    labels_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(labels_dir / LABELS_FILE, **arr)
    meta = {
        "format": 1,
        "capture": str(capture),
        "capture_sha256": capture_sha(capture),
        "catalogue_id": catalogue_id,
        "map": map_name,
        "catalogue_split": split,
        "camera": {"resolution": [2560, 1080], "hfov_deg": 120.0, "uptilt_deg": uptilt_deg},
        "settings": settings.as_dict(),
        "settings_sha256": settings.sha256(),
    }
    (labels_dir / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    return labels_dir


def fake_catalogue(maps: dict[str, str] = MAPS):
    from drone_fly.tracks.catalogue import Catalogue

    entries = [
        {"id": f"id-{m}", "map": m, "split": s, "source": "procedural"} for m, s in maps.items()
    ]
    return Catalogue(
        path=None, header={"maps": {m: {"split": s} for m, s in maps.items()}}, entries=entries
    )


def tiny_dataset(
    tmp: Path,
    contract: Any,
    *,
    maps: dict[str, str] = MAPS,
    n_frames: int = 20,
    seed: int = 0,
    **build: Any,
) -> dict[str, Any]:
    """One labelled capture per map, built into a dataset at :data:`SMALL_W` x :data:`SMALL_H`."""
    from drone_fly.eyes.dataset import build_dataset

    settings = label_settings(contract)
    croot, lroot = tmp / "captures", tmp / "labels"
    for k, (m, split) in enumerate(maps.items()):
        cap = write_capture(
            croot / m / f"cap{m}",
            catalogue_id=f"id-{m}",
            map_name=m,
            n_frames=n_frames,
            seed=seed + k,
        )
        write_labels(
            lroot / cap.name,
            cap,
            settings,
            catalogue_id=f"id-{m}",
            map_name=m,
            split=split,
            seed=seed + k,
        )
    kw = {
        "width": SMALL_W,
        "height": SMALL_H,
        "cache_root": tmp / "cache",
        "out": tmp / "ds",
        "labels_root": lroot,
        "catalogue": fake_catalogue(maps),
        **build,
    }
    return build_dataset(croot, **kw)


def tiny_run(tmp: Path, contract: Any, **train: Any) -> tuple[Path, Path, dict[str, Any]]:
    """``(dataset_dir, run_dir, training_log)``: :func:`tiny_dataset` + a 2-epoch CPU run."""
    from drone_fly.eyes.train import train_eyes

    tiny_dataset(tmp, contract)
    kw = {
        "out": tmp / "run",
        "epochs": 2,
        "batch": 8,
        "device": "cpu",
        "cache_root": tmp / "cache",
        **train,
    }
    return tmp / "ds", Path(kw["out"]), train_eyes(tmp / "ds", **kw)
