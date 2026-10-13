"""Scene caches: loading, build-id handling, flags and geometric validation (UC68 D4).

A scene cache (``<scenes root>/<Map>.json``, gitignored) holds an environment's solid colliders in
Unity world space (liftoff-flight-analyzer format, see :mod:`.liftoff_extract.scene`). The scenes
root resolves as ``--scenes-root`` -> ``$DRONE_FLY_SCENES`` -> ``<trackdata root>/scenes``.

**Build ids.** A cache built from another game build than the current ``props.json`` / XMLs is a
WARNING plus the ``stale-build`` flag, never a rejection on its own: validity is decided
geometrically per map by re-running the identification score against the CURRENT XMLs with a
synthetic witness (every opening centre +-0.5 m along its normal, every spawn raised 0.5 m;
every passage centre is a gate point). It
passes with gates near >= 0.7 and witness points inside solids <= 0.5 %; only a failed check
excludes the scene (status ``invalid``).

**Statuses:** ``ok`` / ``stale-build`` (usable), ``invalid``, ``ambiguous`` (extraction could not
tell two bundles apart), ``missing``. **Flags:** ``no-terrain`` (terrain colliders but no
heightfield), ``partial`` (mesh colliders over 10 % of the primitives, not emitted),
``stale-build``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.tracks.catalogue import TrackDataError, sha256_file

log = logging.getLogger(__name__)

USABLE = ("ok", "stale-build")
STATUSES = ("ok", "stale-build", "invalid", "ambiguous", "missing")
PARTIAL_MESH_SHARE = 0.10
WITNESS_OFFSET_M = 0.5


@dataclass
class SceneData:
    environment: str
    path: Path
    sha256: str
    build_id: str | None
    bundle: str | None
    counts: dict[str, int]
    colliders: list[dict[str, Any]]
    heightfields: list[dict[str, Any]] = field(default_factory=list)

    def flags(self, current_build: str | None) -> list[str]:
        out = []
        c = self.counts
        if c.get("terrain", 0) > 0 and not self.heightfields:
            out.append("no-terrain")
        prim = c.get("box", 0) + c.get("capsule", 0) + c.get("sphere", 0) + c.get("mesh", 0)
        if prim and c.get("mesh", 0) / prim > PARTIAL_MESH_SHARE:
            out.append("partial")
        if current_build and self.build_id and str(self.build_id) != str(current_build):
            out.append("stale-build")
        return out

    def summary(self, status: str, current_build: str | None) -> dict[str, Any]:
        keys = ("box", "capsule", "sphere", "mesh", "terrain", "trigger")
        return {
            "status": status,
            "bundle": self.bundle,
            "build_id": self.build_id,
            "sha256": self.sha256,
            "counts": {k: int(self.counts.get(k, 0)) for k in keys},
            "flags": self.flags(current_build),
        }


def scene_path(scenes_root: Path, environment: str) -> Path:
    return Path(scenes_root) / f"{environment}.json"


def load_scene(path: str | Path) -> SceneData:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackDataError(f"cannot read scene cache {p}: {exc}") from exc
    if data.get("format") != 1 or not isinstance(data.get("colliders"), list):
        raise TrackDataError(f"{p}: not a format-1 scene cache")
    src = data.get("source") or {}
    return SceneData(
        environment=str(data.get("environment") or p.stem),
        path=p,
        sha256=sha256_file(p),
        build_id=None if src.get("build_id") is None else str(src.get("build_id")),
        bundle=src.get("bundle"),
        counts=dict(data.get("counts") or {}),
        colliders=data["colliders"],
        heightfields=list(data.get("heightfields") or []),
    )


def find_scene(scenes_root: Path, environment: str) -> SceneData | None:
    p = scene_path(scenes_root, environment)
    return load_scene(p) if p.is_file() else None


def synthetic_witness(
    passages: list[tuple[np.ndarray, np.ndarray, bool]], spawns: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """``(gate points, witness points)`` in UNITY world space from contract-frame passages.

    ``passages`` are ``(centre, unit normal, is_opening)`` and ``spawns`` positions, both in the
    contract frame. Every passage centre is a gate point; each opening centre +-0.5 m along its
    normal and each spawn raised 0.5 m are witness points.
    """
    from drone_fly.harness.geometry import UNITY_TO_CONTRACT

    inv = np.asarray(UNITY_TO_CONTRACT, np.float64).T  # contract -> Unity (orthogonal)
    gates, wit = [], []
    for c, n, is_opening in passages:
        gates.append(inv @ c)
        if not is_opening:
            continue
        for s in (-WITNESS_OFFSET_M, WITNESS_OFFSET_M):
            wit.append(inv @ (np.asarray(c) + s * np.asarray(n)))
    for sp in spawns:
        wit.append(inv @ (np.asarray(sp) + np.array([0.0, 0.0, WITNESS_OFFSET_M])))
    return np.asarray(gates).reshape(-1, 3), np.asarray(wit).reshape(-1, 3)


def validate_scene(
    scene: SceneData, gates_unity: np.ndarray, witness_unity: np.ndarray
) -> dict[str, Any]:
    """Re-run the identification score against the current XMLs (see the module docstring)."""
    from drone_fly.tracks.liftoff_extract import scene as ex

    cols = ex.ColliderArrays.from_colliders(scene.colliders)
    near = ex.cloud_score(cols, gates_unity)
    inside = ex.witness_inside(cols, witness_unity)
    ok = near >= ex.NEED_FRACTION and inside <= ex.MAX_PATH_INSIDE
    return {"near": round(float(near), 4), "inside": round(float(inside), 5), "valid": bool(ok)}


def scene_status(scene: SceneData | None, current_build: str | None, valid: bool | None) -> str:
    if scene is None:
        return "missing"
    if valid is False:
        return "invalid"
    if current_build and scene.build_id and str(scene.build_id) != str(current_build):
        log.warning(
            "scene %s was built from game build %s; the current data is build %s (stale-build, "
            "validated geometrically)",
            scene.environment,
            scene.build_id,
            current_build,
        )
        return "stale-build"
    return "ok"
