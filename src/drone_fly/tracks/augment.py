"""Training-only geometric augmentation of converted courses (UC68 run 2, ADR 0007 D8).

Each Liftoff TRAIN course gets mirrored and uniformly scaled copies, built once when the pool is
built: ``T(p) = scale * F p`` about the course origin (the spawn support point), where ``F``
flips y when ``mirror`` (a reflection in the course x-z plane). Only the GEOMETRY changes; the
plant, the observation and the action are untouched, so a mirrored course is simply another
course the drone flies with the usual physics (gravity is along z, which F keeps).

* points (passage centres, primitive centres, the spawn, harness AABB centres, bounds) map
  through ``T``; sizes scale by ``scale``;
* rotation frames map to ``F R F`` (a proper rotation; exact for the mirror-symmetric primitives
  and volumes). A passage's travel frame keeps ``normal' = F normal`` and ``up' = F up`` and
  re-derives ``across'`` so it stays right-handed; under a mirror the outline's ``u`` changes sign
  (its order is reversed to stay counter-clockwise);
* the heightfield is mirrored (rows reversed) and scaled; the spawn yaw negates under a mirror;
* the time limit is re-derived from the scaled route length (the base and the cap unchanged);
* the cell table is rebuilt with the conversion's corridor / cell / capacity / range.

Val and held-out courses are never augmented (the pool's purpose must be ``train``).
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any

import numpy as np

from drone_fly.sim.pool import (
    HarnessList,
    Heightfield,
    Primitives,
    SimCourse,
    build_cell_table,
    route_polyline,
)


def _flip(mirror: bool) -> np.ndarray:
    return np.diag([1.0, -1.0 if mirror else 1.0, 1.0])


def _rot(F: np.ndarray, R: np.ndarray) -> np.ndarray:
    return F @ R @ F


def augment_course(course: SimCourse, *, mirror: bool, scale: float, conversion: Any) -> SimCourse:
    """A mirrored / scaled copy of ``course`` (see the module docstring)."""
    s = float(scale)
    if s <= 0:
        raise ValueError("augmentation scale must be > 0")
    F = _flip(mirror)

    def pt(p: Any) -> np.ndarray:
        return s * (F @ np.asarray(p, np.float64))

    passages = []
    for p in course.passages:
        n = F @ p.rot[:, 2]
        up = F @ p.rot[:, 1]
        across = np.cross(up, n)
        outline = np.asarray(p.outline, np.float64) * s
        if mirror and len(outline):
            outline = outline * np.array([-1.0, 1.0])
            outline = outline[::-1].copy()
        passages.append(
            replace(
                p,
                center=pt(p.center),
                rot=np.stack([across, up, n], axis=1),
                w=p.w * s,
                h=p.h * s,
                outline=outline,
                vol_rot=_rot(F, p.vol_rot),
                vol_half=np.asarray(p.vol_half, np.float64) * s,
            )
        )
    pr = course.prims
    prims = Primitives(
        pr.kind,
        (s * (pr.center @ F.T)).reshape(-1, 3),
        np.einsum("ij,njk,kl->nil", F, pr.rot, F).reshape(-1, 3, 3),
        pr.size * s,
        pr.source,
    )
    hl = course.harness
    harness = HarnessList((s * (hl.center @ F.T)).reshape(-1, 3), hl.half * s, hl.overflow)
    hf = course.heightfield
    if hf is not None:
        ny = hf.z.shape[0]
        y0 = -(hf.y0 + (ny - 1) * hf.cell) if mirror else hf.y0
        z = hf.z[::-1].copy() if mirror else hf.z
        hf = Heightfield(hf.x0 * s, y0 * s, hf.cell * s, z * s)
    a, b = pt(course.bounds_lo), pt(course.bounds_hi)
    spawn = pt(course.spawn_pos)
    poly = route_polyline([spawn, *[p.center for p in passages]], course.circuit)
    ws = float(conversion.world_scale)
    cells, used = build_cell_table(
        prims, poly, corridor_m=course.corridor_m, cell_m=conversion.cell_m / ws,
        capacity=conversion.cell_capacity, obs_range_m=conversion.obs_range_m / ws,
    )  # fmt: skip
    prims = prims.subset(used)
    route_m = course.stats.get("route_length_m")
    if route_m is not None:
        tl = conversion.time_base_s + float(route_m) * s / conversion.v_ref_mps
        tl = min(conversion.time_max_s, tl)
    else:
        tl = conversion.time_base_s + (course.time_limit_s - conversion.time_base_s) * s
    tag = f"{'m' if mirror else ''}s{s:g}"
    meta = {**course.meta, "id": f"{course.id}~{tag}", "base_id": course.id,
            "augment": {"mirror": bool(mirror), "scale": s}}  # fmt: skip
    return replace(
        course,
        passages=passages,
        spawn_pos=spawn,
        spawn_yaw=-course.spawn_yaw if mirror else course.spawn_yaw,
        floor_z=course.floor_z * s,
        prims=prims,
        harness=harness,
        cells=cells,
        bounds_lo=np.minimum(a, b),
        bounds_hi=np.maximum(a, b),
        time_limit_s=float(tl),
        heightfield=hf,
        meta=meta,
        stats={**course.stats, "primitives": len(prims), "cells": int(len(cells.keys))},
    )


def augment_pool(pool: Any, spec: dict[str, Any], conversion: Any) -> Any:
    """Append augmented copies of the pool's Liftoff courses to their groups (train only).

    ``spec``: ``{mirror: bool, scales: [..]}``. Every (mirror, scale) combination except the
    identity is added for each ``liftoff-xml`` course, so each base track keeps the same share of
    its group.
    """
    if pool.purpose != "train":
        raise ValueError("course augmentation is training-only")
    mirrors = [False, True] if spec.get("mirror", False) else [False]
    scales = [float(x) for x in spec.get("scales", [1.0])]
    combos = [(m, s) for m, s in itertools.product(mirrors, scales) if (m, s) != (False, 1.0)]
    if not combos:
        return pool
    courses = list(pool.courses)
    groups = []
    for g in pool.groups:
        idx = list(g.courses)
        for i in g.courses:
            c = pool.courses[i]
            if c.meta.get("source") != "liftoff-xml":
                continue
            for m, s in combos:
                idx.append(len(courses))
                courses.append(augment_course(c, mirror=m, scale=s, conversion=conversion))
        groups.append(replace(g, courses=idx))
    return replace(pool, courses=courses, groups=groups)
