"""Scene-bundle discovery, collider extraction (``read_scene``) and geometric identification.

Ported from liftoff-flight-analyzer ``src/fpv_review/sources/liftoff/scene.py`` (``scene_bundles``,
``read_scene``, ``cloud_score``, ``build``) at commit d7250d5. MIT License, Copyright (c) 2026
Harold Hormaechea. drone-fly changes: identification scores every candidate bundle against every
map in ONE pass (each bundle is read once), takes a synthetic witness when no real flight is given
(every opening centre +-0.5 m along its normal, every spawn raised 0.5 m), assigns bundles to maps
uniquely and marks a map ``ambiguous`` when the best two candidates are within
:data:`AMBIGUITY_MARGIN`; terrain heightmaps are attempted (best effort).

A cache is the analyzer's format: ``{"format": 1, "environment", "source": {...}, "counts",
"skipped", "bounds", "colliders": [{"t", "p", "q", "s"|"r","h","a"}, ...]}`` in Unity world space,
plus an optional drone-fly ``"heightfields"`` list. Triggers are never emitted (not solid).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = 1
MIN_COLLIDERS = 500
NEAR_GATE_M = 25.0
NEED_FRACTION = 0.7
MAX_PATH_INSIDE = 0.005
#: Two candidates whose gate-proximity scores are this close make a map ambiguous.
AMBIGUITY_MARGIN = 0.05


def need_unitypy(unity_version: str) -> Any:
    """Import UnityPy lazily (the ``liftoff-extract`` extra) or raise an actionable error."""
    from drone_fly.tracks.catalogue import TrackDataError

    try:
        import UnityPy  # noqa: N813
    except ImportError as exc:
        raise TrackDataError(
            "extracting Liftoff scenes / props needs UnityPy (the liftoff-extract extra): "
            "uv pip install --python <venv python> UnityPy==1.23.0"
        ) from exc
    UnityPy.config.FALLBACK_UNITY_VERSION = unity_version
    return UnityPy


def scene_bundles(bundles: Path) -> list[Path]:
    """Candidate streamed-scene bundles (node table only), smallest first."""
    from drone_fly.tracks.liftoff_extract.unityfs import is_scene_bundle

    paths = sorted(Path(bundles).glob("*.bundle"), key=lambda p: p.stat().st_size)
    return [p for p in paths if is_scene_bundle(p)]


def _terrain_heightfields(env: Any, transforms: dict[Any, Any], owner_of: dict[Any, Any]) -> list:
    """TerrainData heightmaps placed by their Terrain's transform (best effort, Unity frame)."""
    from drone_fly.tracks.liftoff_extract.transforms import compose

    out = []
    cache: dict[Any, Any] = {}
    for obj in env.objects:
        if obj.type.name != "Terrain":
            continue
        try:
            terrain = obj.read()
            data = terrain.m_TerrainData.read()
            hm = data.m_Heightmap
            res = int(getattr(hm, "m_Resolution", 0) or 0)
            heights = np.asarray(hm.m_Heights, np.float64)
            sc = hm.m_Scale
            if res < 2 or heights.size != res * res:
                continue
            tid = owner_of.get(terrain.m_GameObject.path_id)
            pos = compose(transforms, tid, cache)[0] if tid is not None else (0.0, 0.0, 0.0)
            # Unity stores heights as int16 of 0..32766 -> 0..1 of the terrain height
            norm = (heights.reshape(res, res) / 32766.0).round(4)
            out.append(
                {
                    "origin": [round(float(v), 3) for v in pos],
                    "size": [float(sc.x) * (res - 1), float(sc.y), float(sc.z) * (res - 1)],
                    "resolution": res,
                    "heights": norm.tolist(),
                }
            )
        except Exception:  # best effort: a map keeps the floor plane on any failure
            continue
    return out


def read_scene(path: str | Path, unity_version: str) -> tuple[list[dict], dict[str, int], list]:
    """``(colliders, counts, heightfields)`` of one scene bundle, in Unity world space."""
    from drone_fly.tracks.liftoff_extract.transforms import compose, qrot

    unitypy = need_unitypy(unity_version)
    env = unitypy.load(str(path))
    transforms: dict[Any, dict[str, Any]] = {}
    owner_of: dict[Any, Any] = {}
    raw: list[tuple[Any, ...]] = []
    counts = {"box": 0, "capsule": 0, "sphere": 0, "mesh": 0, "terrain": 0}
    for obj in env.objects:
        kind = obj.type.name
        try:
            if kind == "Transform":
                data = obj.read()
                pos, rot, sca = data.m_LocalPosition, data.m_LocalRotation, data.m_LocalScale
                transforms[obj.path_id] = {
                    "p": (pos.x, pos.y, pos.z),
                    "q": (rot.x, rot.y, rot.z, rot.w),
                    "s": (sca.x, sca.y, sca.z),
                    "parent": data.m_Father.path_id if data.m_Father else 0,
                }
                if data.m_GameObject:
                    owner_of[data.m_GameObject.path_id] = obj.path_id
            elif kind == "BoxCollider":
                data = obj.read()
                c, s = data.m_Center, data.m_Size
                trig = bool(getattr(data, "m_IsTrigger", False))
                raw.append(
                    ("box", data.m_GameObject.path_id, (c.x, c.y, c.z), (s.x, s.y, s.z), None, trig)
                )
                counts["box"] += 1
            elif kind == "SphereCollider":
                data = obj.read()
                c = data.m_Center
                trig = bool(getattr(data, "m_IsTrigger", False))
                raw.append(
                    ("sph", data.m_GameObject.path_id, (c.x, c.y, c.z), None, data.m_Radius, trig)
                )
                counts["sphere"] += 1
            elif kind == "CapsuleCollider":
                data = obj.read()
                c = data.m_Center
                trig = bool(getattr(data, "m_IsTrigger", False))
                dims = (data.m_Radius, data.m_Height, data.m_Direction)
                raw.append(("cap", data.m_GameObject.path_id, (c.x, c.y, c.z), dims, None, trig))
                counts["capsule"] += 1
            elif kind == "MeshCollider":
                counts["mesh"] += 1
            elif kind == "TerrainCollider":
                counts["terrain"] += 1
        except Exception:  # an unreadable object is skipped
            continue
    cache: dict[Any, Any] = {}
    out: list[dict[str, Any]] = []
    triggers = 0
    for kind, go_id, centre, size, radius, trig in raw:
        if trig:
            triggers += 1  # a trigger volume is not solid
            continue
        tid = owner_of.get(go_id)
        if tid is None:
            continue
        pos, rot, scale = compose(transforms, tid, cache)
        off = qrot(rot, (centre[0] * scale[0], centre[1] * scale[1], centre[2] * scale[2]))
        world = [round(pos[i] + off[i], 3) for i in range(3)]
        if kind == "box":
            half = [round(abs(size[i] * scale[i]) / 2.0, 3) for i in range(3)]
            if max(half) >= 0.02:
                out.append({"t": "box", "p": world, "q": [round(v, 4) for v in rot], "s": half})
        elif kind == "sph":
            r = abs(radius) * max(abs(v) for v in scale)
            if r >= 0.02:
                out.append({"t": "sph", "p": world, "r": round(r, 3)})
        else:
            r, h, axis = size
            lat = [abs(scale[j]) for j in range(3) if j != int(axis)]
            r = abs(r) * max(lat)
            h = abs(h) * abs(scale[int(axis)])
            if r >= 0.02:
                out.append({"t": "cap", "p": world, "q": [round(v, 4) for v in rot],
                            "r": round(r, 3), "h": round(h, 3), "a": int(axis)})  # fmt: skip
    counts["trigger"] = triggers
    heightfields = _terrain_heightfields(env, transforms, owner_of) if counts["terrain"] else []
    return out, counts, heightfields


# ----------------------------------------------------------------------------- scoring


def _quat_matrices(q: np.ndarray) -> np.ndarray:
    """Rotation matrices of Unity ``(x, y, z, w)`` quaternions ``(N, 4) -> (N, 3, 3)``."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.maximum(x * x + y * y + z * z + w * w, 1e-12)
    s = 2.0 / n
    return np.stack(
        [
            np.stack([1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)], -1),
            np.stack([s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)], -1),
            np.stack([s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)], -1),
        ],
        axis=1,
    )


@dataclass
class ColliderArrays:
    """Scene colliders as arrays (Unity frame) for vectorised scoring."""

    t: np.ndarray  # 0 box, 1 capsule, 2 sphere
    p: np.ndarray
    rot: np.ndarray
    s: np.ndarray  # box half; capsule (r, h, axis); sphere (r, 0, 0)
    reach: np.ndarray

    @classmethod
    def from_colliders(cls, cols: Sequence[dict[str, Any]]) -> ColliderArrays:
        n = len(cols)
        t = np.zeros(n, np.int64)
        p = np.zeros((n, 3))
        q = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
        s = np.zeros((n, 3))
        for i, c in enumerate(cols):
            p[i] = c["p"]
            if c["t"] == "box":
                q[i], s[i] = c["q"], c["s"]
            elif c["t"] == "cap":
                t[i] = 1
                q[i] = c["q"]
                s[i] = (c["r"], c["h"], c["a"])
            else:
                t[i] = 2
                s[i, 0] = c["r"]
        reach = np.where(
            t == 0, np.linalg.norm(s, axis=1), np.where(t == 1, s[:, 0] + s[:, 1] / 2, s[:, 0])
        )
        return cls(t, p, _quat_matrices(q), s, reach)

    def __len__(self) -> int:
        return len(self.t)


def contains_any(cols: ColliderArrays, points: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Per point: inside any collider (exact box / capsule / sphere tests, vectorised)."""
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    out = np.zeros(len(pts), bool)
    if not len(cols):
        return out
    for a in range(0, len(pts), chunk):
        sub = pts[a : a + chunk]
        d = sub[:, None, :] - cols.p[None]
        near = (d * d).sum(-1) <= cols.reach[None] ** 2
        ii, jj = np.nonzero(near)
        if not len(ii):
            continue
        local = np.einsum("nji,nj->ni", cols.rot[jj], d[ii, jj])
        t = cols.t[jj]
        s = cols.s[jj]
        inside = np.zeros(len(ii), bool)
        b = t == 0
        inside[b] = (np.abs(local[b]) <= s[b]).all(axis=1)
        sp = t == 2
        inside[sp] = (local[sp] ** 2).sum(1) <= s[sp, 0] ** 2
        cp = np.flatnonzero(t == 1)
        if len(cp):
            ax = s[cp, 2].astype(np.int64)
            half = np.maximum(s[cp, 1] / 2.0 - s[cp, 0], 0.0)
            lc = local[cp].copy()
            along = lc[np.arange(len(cp)), ax]
            lc[np.arange(len(cp)), ax] = along - np.clip(along, -half, half)
            inside[cp] = (lc**2).sum(1) <= s[cp, 0] ** 2
        hit = np.zeros(len(sub), bool)
        np.logical_or.at(hit, ii, inside)
        out[a : a + chunk] = hit
    return out


def cloud_score(cols: ColliderArrays, gates: np.ndarray, chunk: int = 64) -> float:
    """Fraction of gate points with a collider centre within :data:`NEAR_GATE_M`."""
    gates = np.asarray(gates, np.float64).reshape(-1, 3)
    if len(cols) < MIN_COLLIDERS or not len(gates):
        return 0.0
    near = 0
    for a in range(0, len(gates), chunk):
        d = np.linalg.norm(gates[a : a + chunk, None, :] - cols.p[None], axis=2)
        near += int((d.min(axis=1) <= NEAR_GATE_M).sum())
    return near / float(len(gates))


def witness_inside(cols: ColliderArrays, witness: np.ndarray) -> float:
    witness = np.asarray(witness, np.float64).reshape(-1, 3)
    if not len(witness):
        return 0.0
    return float(contains_any(cols, witness).mean())


@dataclass
class MapCloud:
    """A map's identification fingerprint (Unity frame): gate points + free-space witness."""

    gates: np.ndarray
    witness: np.ndarray


def score(cols: ColliderArrays, cloud: MapCloud) -> tuple[float, float]:
    """``(gates near, witness inside)`` of one candidate for one map."""
    return cloud_score(cols, cloud.gates), witness_inside(cols, cloud.witness)


def assign(
    scores: dict[str, dict[str, tuple[float, float]]], margin: float = AMBIGUITY_MARGIN
) -> dict[str, dict[str, Any]]:
    """Unique bundle -> map assignment from ``scores[map][bundle] = (near, inside)``.

    A candidate passes when ``near >= NEED_FRACTION`` and ``inside <= MAX_PATH_INSIDE``. Maps
    are resolved best-score first; a bundle is used at most once. A map whose two best passing
    candidates are within ``margin`` is ``ambiguous``; one with none is ``missing``.
    """
    passing = {
        m: sorted(
            (
                (near, b)
                for b, (near, inside) in per.items()
                if near >= NEED_FRACTION and inside <= MAX_PATH_INSIDE
            ),
            reverse=True,
        )  # fmt: skip
        for m, per in scores.items()
    }
    out: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    order = sorted(passing, key=lambda m: -(passing[m][0][0] if passing[m] else -1.0))
    for m in order:
        cands = [(s, b) for s, b in passing[m] if b not in used]
        if not cands:
            out[m] = {"status": "missing"}
            continue
        if len(cands) > 1 and cands[0][0] - cands[1][0] < margin:
            out[m] = {
                "status": "ambiguous",
                "candidates": [b for _, b in cands[:3]],
                "scores": [round(s, 3) for s, _ in cands[:3]],
            }
            continue
        near, b = cands[0]
        used.add(b)
        out[m] = {"status": "ok", "bundle": b, "near": round(near, 3),
                  "inside": round(scores[m][b][1], 5)}  # fmt: skip
    return out


def write_cache(
    out_dir: Path,
    environment: str,
    bundle: Path,
    colliders: list[dict[str, Any]],
    counts: dict[str, int],
    heightfields: list,
    *,
    build_id: str | None,
    unity_version: str,
    identified_by: str,
    witness: str | None,
) -> Path:
    """Write ``<out_dir>/<environment>.json`` in the analyzer's cache format."""
    pts = np.array([c["p"] for c in colliders]) if colliders else np.zeros((1, 3))
    scene = {
        "format": FORMAT,
        "environment": environment,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "bundle": bundle.name,
            "size": bundle.stat().st_size,
            "build_id": build_id,
            "unity_version": unity_version,
            "identified_by": identified_by,
            "witness": witness,
            "extractor": "drone-fly tracks extract-scenes",
        },
        "counts": counts,
        "skipped": {k: counts[k] for k in ("mesh", "terrain") if counts.get(k)},
        "bounds": {
            "min": [round(float(v), 2) for v in pts.min(axis=0)],
            "max": [round(float(v), 2) for v in pts.max(axis=0)],
        },
        "colliders": colliders,
    }
    if heightfields:
        scene["heightfields"] = heightfields
    target = Path(out_dir) / f"{environment}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(scene, separators=(",", ":")), encoding="utf-8")
    return target
