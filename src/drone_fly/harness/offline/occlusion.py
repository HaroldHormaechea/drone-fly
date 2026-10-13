"""Scene occlusion of labelled gates (UC71 P3 report; labels stay unflagged, see ADR 0010).

A labelled gate is **occluded** when the sight line from the drone to its aperture centre passes
through a scene collider of the map's scene cache (``tracks extract-scenes``): points every
:data:`STEP_M` along the segment, skipping :data:`NEAR_DRONE_M` at the drone end and
:data:`NEAR_GATE_M` at the gate end (the gate's own frame and its supports), are tested for
containment in the cache's boxes, capsules and spheres. Mesh and terrain colliders are not in the
cache (the extractor skips them), so the share is a **lower bound**.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

STEP_M = 0.25
NEAR_DRONE_M = 0.3
NEAR_GATE_M = 1.0


def load_scene(path: str | Path) -> Any:
    """The cache's colliders as ``ColliderArrays`` (Unity world)."""
    from drone_fly.tracks.liftoff_extract.scene import ColliderArrays

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ColliderArrays.from_colliders(data.get("colliders") or [])


def sight_points(drone: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Sample points on the drone -> target segment (both ends trimmed). Pure."""
    d = np.asarray(target, np.float64) - np.asarray(drone, np.float64)
    n = float(np.linalg.norm(d))
    a, b = NEAR_DRONE_M, n - NEAR_GATE_M
    if b <= a:
        return np.zeros((0, 3))
    t = np.arange(a, b, STEP_M)
    return np.asarray(drone, np.float64) + (t / n)[:, None] * d


def occluded(cols: Any, drone_unity: np.ndarray, gate_unity: np.ndarray) -> bool:
    from drone_fly.tracks.liftoff_extract.scene import contains_any

    pts = sight_points(drone_unity, gate_unity)
    return bool(len(pts)) and bool(contains_any(cols, pts).any())


def capture_occlusion(
    capture_dir: str | Path, labels_dir: str | Path, scene_path: str | Path, telemetry_cfg: dict
) -> dict[str, dict[str, dict[str, int]]]:
    """``{class: {bin: {"n": in-range gates, "occluded": n}}}`` for one labelled capture."""
    from drone_fly.harness.geometry import UNITY_TO_CONTRACT
    from drone_fly.harness.offline.capture import load_capture
    from drone_fly.harness.offline.gate_dataset import in_range
    from drone_fly.harness.offline.gate_diagnose import POOLED, range_bin
    from drone_fly.harness.offline.gate_train import load_labels

    m_inv = np.asarray(UNITY_TO_CONTRACT, np.float64).T  # contract -> Unity (orthonormal)
    cols = load_scene(scene_path)
    labels = load_labels(labels_dir)
    cap = load_capture(capture_dir, telemetry_cfg, 1.0, float(labels.meta.get("lag_ms", 0.0)))
    by_file = {str(f.path.relative_to(cap.root)).replace("\\", "/"): f for f in cap.frames}
    out: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"n": 0, "occluded": 0})
    )
    for r in labels.rows:
        f = by_file.get(r["file"])
        if f is None:
            continue
        drone = m_inv @ np.asarray(f.state.pos, np.float64)
        for g in r["gates"]:
            if not in_range(g):
                continue
            hit = occluded(cols, drone, m_inv @ np.asarray(g["center_world"], np.float64))
            for b in (range_bin(float(g["range_m"])), POOLED):
                if b is None:
                    continue
                cell = out[g["cls"]][b]
                cell["n"] += 1
                cell["occluded"] += int(hit)
    return {k: dict(v) for k, v in out.items()}
