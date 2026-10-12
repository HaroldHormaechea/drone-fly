"""Quaternion maths, TRS composition, point-in-collider and the witness test.

Ported from liftoff-flight-analyzer ``src/fpv_review/common/geometry.py`` (``qmul``, ``qrot``,
``contains_point``, ``path_inside``) and ``src/fpv_review/sources/liftoff/scene.py``
(``compose``) at commit d7250d5. MIT License, Copyright (c) 2026 Harold Hormaechea.

Quaternions are Unity's ``(x, y, z, w)``; frames are Unity world space (left-handed, +Y up,
metres). A collider is the scene-cache dict: ``{"t": "box"|"sph"|"cap", "p", "q", "s" (box
half-extents), "r", "h", "a"}``. TRS composition is the analyzer's approximation (component-wise
scale, shear ignored), which is how every cache was built.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

Vec = tuple[float, float, float]
Quat = tuple[float, float, float, float]


def qmul(a: Sequence[float], b: Sequence[float]) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def qrot(q: Sequence[float], v: Sequence[float]) -> Vec:
    x, y, z, w = q
    cx = y * v[2] - z * v[1]
    cy = z * v[0] - x * v[2]
    cz = x * v[1] - y * v[0]
    dx = y * cz - z * cy
    dy = z * cx - x * cz
    dz = x * cy - y * cx
    return (v[0] + 2 * w * cx + 2 * dx, v[1] + 2 * w * cy + 2 * dy, v[2] + 2 * w * cz + 2 * dz)


def compose(
    transforms: dict[Any, dict[str, Any]], tid: Any, cache: dict[Any, Any], depth: int = 0
) -> tuple[Vec, Quat, Vec]:
    """World ``(position, rotation, scale)`` of a Transform, up the parent chain (full TRS)."""
    if tid in cache:
        return cache[tid]
    node = transforms.get(tid)
    if node is None or depth > 128:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
    parent = node["parent"]
    if not parent or parent not in transforms:
        result = (tuple(node["p"]), tuple(node["q"]), tuple(node["s"]))
    else:
        pp, pq, ps = compose(transforms, parent, cache, depth + 1)
        scaled = (node["p"][0] * ps[0], node["p"][1] * ps[1], node["p"][2] * ps[2])
        turned = qrot(pq, scaled)
        result = (
            (pp[0] + turned[0], pp[1] + turned[1], pp[2] + turned[2]),
            qmul(pq, node["q"]),
            (node["s"][0] * ps[0], node["s"][1] * ps[1], node["s"][2] * ps[2]),
        )
    cache[tid] = result
    return result  # type: ignore[return-value]


def contains_point(c: dict[str, Any], p: Sequence[float]) -> bool:
    """Is a world point inside this collider? Exact for all three primitives."""
    d = (p[0] - c["p"][0], p[1] - c["p"][1], p[2] - c["p"][2])
    if c["t"] == "sph":
        return d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= c["r"] * c["r"]
    x, y, z, w = c["q"]
    local = qrot((-x, -y, -z, w), d)
    if c["t"] == "box":
        s = c["s"]
        return abs(local[0]) <= s[0] and abs(local[1]) <= s[1] and abs(local[2]) <= s[2]
    axis = c["a"]
    half = max(c["h"] / 2.0 - c["r"], 0.0)
    clamped = max(-half, min(half, local[axis]))
    off = list(local)
    off[axis] = local[axis] - clamped
    return off[0] * off[0] + off[1] * off[1] + off[2] * off[2] <= c["r"] * c["r"]


def reach(c: dict[str, Any]) -> float:
    """A bounding radius of a collider about its centre."""
    if c["t"] == "box":
        s = c["s"]
        return (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) ** 0.5
    if c["t"] == "sph":
        return c["r"]
    return c["r"] + c["h"] / 2.0


def path_inside(colliders: Sequence[dict[str, Any]], path: Sequence[Sequence[float]]) -> float:
    """Fraction of witness points inside solid geometry (about 0 in the right scene)."""
    if not path:
        return 0.0
    quick = [(c, c["p"], reach(c) ** 2) for c in colliders]
    hit = 0
    for q in path:
        for c, centre, reach2 in quick:
            dx, dy, dz = q[0] - centre[0], q[1] - centre[1], q[2] - centre[2]
            if dx * dx + dy * dy + dz * dz <= reach2 and contains_point(c, q):
                hit += 1
                break
    return hit / float(len(path))
