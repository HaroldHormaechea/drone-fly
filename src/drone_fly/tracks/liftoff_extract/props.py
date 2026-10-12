"""``props.json`` (each item's colliders in its prefab frame) and the gate ``opening()``.

Ported from liftoff-flight-analyzer ``src/fpv_review/sources/liftoff/props.py``
(``prefab_colliders``, ``build``) and ``src/fpv_review/sources/liftoff_micro/props.py``
(``opening``, ``_box_corners``, ``_hull``) at commit d7250d5. MIT License, Copyright (c) 2026
Harold Hormaechea.

**opening()** reads a gate prop's TRIGGER collider (the game's own "did that pass count" surface):
a trigger thin in one horizontal axis (under :data:`THIN_SHARE` of the wider one) is an
``opening`` facing that axis (``normal`` ``x`` or ``z``) with width ``w``, height ``h``, centre
``(cu, cy)`` in (across, up) and a convex ``outline`` about it; across is local ``+x`` for a gate
facing ``z`` and local ``-z`` for one facing ``x``. A thick trigger is a ``volume``. drone-fly
adds ``cn``: the trigger's centre along the normal axis (the analyzer measured crossings from the
prefab origin and did not need it).

Building ``props.json`` needs UnityPy (imported lazily); mesh colliders are kept as triangles
(at most :data:`MAX_MESH_TRIS`) or an approximate box.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from drone_fly.tracks.liftoff_extract.transforms import compose, qmul, qrot

FORMAT = 1
#: Thinner than this share of the wider horizontal axis, a trigger is a plane you fly through.
THIN_SHARE = 0.10
MAX_MESH_TRIS = 1200
MAX_BUNDLE_MB = 250


def box_corners(c: dict[str, Any]) -> list[list[float]]:
    """A box collider's eight corners in the prefab frame."""
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                d = qrot(
                    c.get("q") or (0, 0, 0, 1), (sx * c["s"][0], sy * c["s"][1], sz * c["s"][2])
                )
                out.append([c["p"][i] + d[i] for i in range(3)])
    return out


def hull(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Convex hull of 2D points, counter-clockwise (Andrew's monotone chain)."""
    pts = sorted({(round(u, 4), round(v, 4)) for u, v in points})
    if len(pts) < 3:
        return pts

    def cross(o: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def opening(colliders: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The scoring disc of one gate prop (see the module docstring), or None without a trigger."""
    trig = [c for c in colliders if c.get("trig")]
    if not trig:
        return None
    c = trig[0]
    pts = c["v"] if c["t"] == "mesh" else box_corners(c) if c["t"] == "box" else None
    if not pts:
        return None
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    ex, ez = hi[0] - lo[0], hi[2] - lo[2]
    if min(ex, ez) > THIN_SHARE * max(ex, ez):
        return {
            "kind": "volume",
            "size": [round(ex, 3), round(hi[1] - lo[1], 3), round(ez, 3)],
            "centre": [round((hi[i] + lo[i]) / 2.0, 3) for i in range(3)],
        }
    normal = "z" if ez < ex else "x"

    def across(p: Sequence[float]) -> float:
        return p[0] if normal == "z" else -p[2]

    cu = (max(across(p) for p in pts) + min(across(p) for p in pts)) / 2.0
    cy = (hi[1] + lo[1]) / 2.0
    cn = (hi[2] + lo[2]) / 2.0 if normal == "z" else (hi[0] + lo[0]) / 2.0
    outline = hull([(across(p) - cu, p[1] - cy) for p in pts])
    w = max(u for u, _ in outline) - min(u for u, _ in outline)
    h = max(v for _, v in outline) - min(v for _, v in outline)
    return {
        "kind": "opening",
        "normal": normal,
        "w": round(w, 3),
        "h": round(h, 3),
        "cy": round(cy, 3),
        "cu": round(cu, 3),
        "cn": round(cn, 3),
        "outline": [[round(u, 3), round(v, 3)] for u, v in outline],
    }


# ----------------------------------------------------------------------------- building


def _qconj(q: Sequence[float]) -> tuple[float, float, float, float]:
    return (-q[0], -q[1], -q[2], q[3])


def cab_index(bundles: Path) -> dict[str, str]:
    """``{CAB node name: bundle filename}`` for the whole content folder (mesh lookups)."""
    from drone_fly.tracks.liftoff_extract.unityfs import node_names

    out: dict[str, str] = {}
    for path in Path(bundles).glob("*.bundle"):
        try:
            for name in node_names(path):
                out[name.lower()] = path.name
        except Exception:  # an unreadable bundle has no nodes to map
            continue
    return out


def mesh_shape(
    collider: Any, env: Any, bundles: Path | None, cabs: dict[str, str], loaded: set[str]
) -> Any:
    """A MeshCollider as ``("mesh", vertices, faces)`` or ``("box", centre, extent)``, or None."""
    ref = getattr(collider, "m_Mesh", None)
    if ref is None:
        return None
    try:
        mesh = ref.read()
    except FileNotFoundError as exc:
        name = str(exc).split()[1].lower() if len(str(exc).split()) > 1 else ""
        bundle = cabs.get(name)
        if bundles is None or bundle is None or bundle in loaded:
            return None
        loaded.add(bundle)
        try:
            env.load_file(str(Path(bundles) / bundle))
            mesh = ref.read()
        except Exception:  # the mesh lives in a bundle that cannot be read
            return None
    except Exception:  # an unreadable mesh reference
        return None
    try:
        from UnityPy.helpers.MeshHelper import MeshHandler

        handler = MeshHandler(mesh)
        handler.process()
        verts = handler.m_Vertices
        tris = handler.get_triangles()
        faces = [t for sub in tris for t in sub] if tris and isinstance(tris[0], list) else tris
        if verts and faces and len(faces) <= MAX_MESH_TRIS:
            return ("mesh", verts, faces)
    except Exception:  # fall back to the mesh bounds
        pass
    aabb = getattr(mesh, "m_LocalAABB", None)
    if aabb is None:
        return None
    c, e = aabb.m_Center, aabb.m_Extent
    return ("box", (c.x, c.y, c.z), (e.x, e.y, e.z))


def prefab_colliders(
    env: Any, want: set[str], bundles: Path | None = None, cabs: dict[str, str] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """``{name: [collider, ...]}`` for every wanted prefab root in one loaded bundle.

    Colliders are composed to world and pushed back into the prefab root's frame; when a name
    appears on several GameObjects, the one with the most collider descendants wins.
    """
    transforms: dict[Any, dict[str, Any]] = {}
    owner_of: dict[Any, Any] = {}
    name_of: dict[Any, str] = {}
    raw: list[tuple[Any, ...]] = []
    loaded: set[str] = set()
    cabs = cabs or {}
    for obj in env.objects:
        kind = obj.type.name
        try:
            if kind == "GameObject":
                name_of[obj.path_id] = obj.read().m_Name
            elif kind == "Transform":
                data = obj.read()
                pos, rot, scale = data.m_LocalPosition, data.m_LocalRotation, data.m_LocalScale
                transforms[obj.path_id] = {
                    "p": (pos.x, pos.y, pos.z),
                    "q": (rot.x, rot.y, rot.z, rot.w),
                    "s": (scale.x, scale.y, scale.z),
                    "parent": data.m_Father.path_id if data.m_Father else 0,
                }
                if data.m_GameObject:
                    owner_of[data.m_GameObject.path_id] = obj.path_id
            elif kind == "MeshCollider":
                data = obj.read()
                shape = mesh_shape(data, env, bundles, cabs, loaded)
                trig = bool(getattr(data, "m_IsTrigger", False))
                if shape and shape[0] == "mesh":
                    raw.append(("mesh", data.m_GameObject.path_id, shape[1], shape[2], None, trig))
                elif shape:
                    _, centre, ext = shape
                    full = (ext[0] * 2, ext[1] * 2, ext[2] * 2)
                    raw.append(("boxapprox", data.m_GameObject.path_id, centre, full, None, trig))
            elif kind in ("BoxCollider", "SphereCollider", "CapsuleCollider"):
                data = obj.read()
                ce = data.m_Center
                trig = bool(getattr(data, "m_IsTrigger", False))
                centre = (ce.x, ce.y, ce.z)
                if kind == "BoxCollider":
                    sz = data.m_Size
                    raw.append(
                        ("box", data.m_GameObject.path_id, centre, (sz.x, sz.y, sz.z), None, trig)
                    )
                elif kind == "SphereCollider":
                    raw.append(
                        ("sph", data.m_GameObject.path_id, centre, None, data.m_Radius, trig)
                    )
                else:
                    dims = (data.m_Radius, data.m_Height, data.m_Direction)
                    raw.append(("cap", data.m_GameObject.path_id, centre, dims, None, trig))
        except Exception:  # an unreadable object is skipped, as in the analyzer
            continue
    roots: dict[str, list[Any]] = {}
    for go_id, name in name_of.items():
        if name in want and go_id in owner_of:
            roots.setdefault(name, []).append(owner_of[go_id])
    if not roots:
        return {}
    root_ids = {tid: name for name, tids in roots.items() for tid in tids}
    cache: dict[Any, Any] = {}

    def chain(tid: Any) -> list[Any]:
        out = []
        for _ in range(128):
            if tid not in transforms:
                break
            out.append(tid)
            tid = transforms[tid]["parent"]
        return out

    per_root: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for kind, go_id, centre, size, radius, trig in raw:
        tid = owner_of.get(go_id)
        if tid is None:
            continue
        root_tid = next((a for a in chain(tid) if a in root_ids), None)
        if root_tid is None:
            continue
        wp, wq, ws = compose(transforms, tid, cache)
        rp, rq, rs = compose(transforms, root_tid, cache)
        inv = _qconj(rq)
        lp = qrot(inv, (wp[0] - rp[0], wp[1] - rp[1], wp[2] - rp[2]))
        lp = [lp[i] / (rs[i] if abs(rs[i]) > 1e-6 else 1.0) for i in range(3)]
        lq = qmul(inv, wq)
        ls = [ws[i] / (rs[i] if abs(rs[i]) > 1e-6 else 1.0) for i in range(3)]
        extra = {"trig": 1} if trig else {}
        key = (root_ids[root_tid], root_tid)
        if kind == "mesh":  # bake the triangles straight into the prefab frame
            verts = []
            for v in centre:
                rv = qrot(lq, (v[0] * ls[0], v[1] * ls[1], v[2] * ls[2]))
                verts.append([round(lp[i] + rv[i], 3) for i in range(3)])
            per_root.setdefault(key, []).append(
                {"t": "mesh", "v": verts, "f": [list(f) for f in size], **extra}
            )
            continue
        if kind == "boxapprox":
            extra["approx"] = 1
            kind = "box"
        off = qrot(lq, (centre[0] * ls[0], centre[1] * ls[1], centre[2] * ls[2]))
        p = [round(lp[i] + off[i], 3) for i in range(3)]
        if kind == "box":
            half = [round(abs(size[i] * ls[i]) / 2.0, 3) for i in range(3)]
            if max(half) >= 0.02:
                per_root.setdefault(key, []).append(
                    {"t": "box", "p": p, "q": [round(v, 4) for v in lq], "s": half, **extra}
                )
        elif kind == "sph":
            r = abs(radius) * max(abs(v) for v in ls)
            if r >= 0.02:
                per_root.setdefault(key, []).append({"t": "sph", "p": p, "r": round(r, 3), **extra})
        else:
            r, h, axis = size
            lat = [abs(ls[j]) for j in range(3) if j != int(axis)]
            r = abs(r) * max(lat)
            h = abs(h) * abs(ls[int(axis)])
            if r >= 0.02:
                per_root.setdefault(key, []).append(
                    {
                        "t": "cap",
                        "p": p,
                        "q": [round(v, 4) for v in lq],
                        "r": round(r, 3),
                        "h": round(h, 3),
                        "a": int(axis),
                        **extra,
                    }  # fmt: skip
                )
    best: dict[str, list[dict[str, Any]]] = {name: [] for name in roots}
    for (name, _), cols in per_root.items():
        if len(cols) > len(best[name]):
            best[name] = cols
    return best


def build_props(
    wanted: set[str],
    bundles: Path,
    out_path: Path,
    *,
    build_id: str | None,
    unity_version: str,
    max_mb: int = MAX_BUNDLE_MB,
    log: Any = print,
) -> dict[str, Any]:
    """Scan the non-scene bundles smallest-first and write ``props.json`` (needs UnityPy)."""
    from drone_fly.tracks.liftoff_extract.scene import need_unitypy
    from drone_fly.tracks.liftoff_extract.unityfs import is_scene_bundle

    unitypy = need_unitypy(unity_version)
    candidates = sorted(
        (
            p
            for p in Path(bundles).glob("*.bundle")
            if p.stat().st_size <= max_mb * 1024 * 1024 and not is_scene_bundle(p)
        ),
        key=lambda p: p.stat().st_size,
    )
    cabs = cab_index(Path(bundles))
    items: dict[str, dict[str, Any]] = {}
    started = time.time()
    for path in candidates:
        missing = wanted - {n for n, v in items.items() if v["colliders"]}
        if not missing:
            break
        try:
            found = prefab_colliders(unitypy.load(str(path)), missing, Path(bundles), cabs)
        except Exception:  # an unreadable bundle is skipped
            continue
        for name, cols in found.items():
            if cols or name not in items:
                items[name] = {"bundle": path.name, "colliders": cols}
    table = {
        "format": FORMAT,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": {"build_id": build_id, "unity_version": unity_version},
        "counts": {
            "wanted": len(wanted),
            "found": len(items),
            "with_shape": sum(
                1 for v in items.values() if any(not c.get("trig") for c in v["colliders"])
            ),
        },
        "items": items,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, separators=(",", ":")), encoding="utf-8")
    log(f"props: {len(items)}/{len(wanted)} items in {time.time() - started:.0f} s -> {out_path}")
    return table
