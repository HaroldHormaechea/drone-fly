"""Procedural courses as :class:`~drone_fly.sim.pool.SimCourse` (UC68 AC3).

``sim.course_gen.sample_course(seed_base + i, ...)`` is used verbatim; then every position (gate
centres, the start, obstacle centres) is multiplied by a per-course ``scale`` drawn uniformly
from ``spec.scale`` with ``numpy.random.default_rng(seed)`` (deterministic), and each aperture is
mapped linearly from course_gen's ``[APERTURE_MIN, APERTURE_MAX]`` onto ``spec.aperture_m``.
Obstacle HORIZONTAL sizes are multiplied by the course's mean aperture ratio (mapped / original)
so they keep their size relative to the gates (scaling them by ``scale`` made 35-66 m blocks);
vertical sizes follow ``scale`` like the heights. Gates become square ``opening`` passages facing
their (already travel-oriented) yaw; pillars become vertical capsules (half-length = the pillar
half-height, a conservative cap) and boxes axis-aligned boxes; every obstacle is also on the
harness list (the max-half-extent rule applies). The start is on the ground: the support is the
floor plane z = 0.

The ``spec`` keys (``tracks/procedural.yaml``): ``seed_base``, ``pool_size``, ``ng: [lo, hi)``,
``laps: [lo, hi)``, ``tier`` (``tractable`` / ``full``; recorded only), ``tunnels``,
``variety``, ``no_obs``, ``scale: [lo, hi]``, ``aperture_m: [lo, hi]``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from drone_fly.sim.course_gen import APERTURE_MAX, APERTURE_MIN, sample_course
from drone_fly.sim.pool import (
    PRIM_BOX,
    PRIM_CAPSULE,
    CourseFrame,
    HarnessList,
    Passage,
    Primitives,
    SimCourse,
    build_cell_table,
    route_polyline,
)
from drone_fly.tracks.catalogue import TrackDataError
from drone_fly.tracks.liftoff import SOURCE_PROCEDURAL, ConversionConfig

SPEC_KEYS = ("seed_base", "pool_size", "ng", "laps", "tier", "tunnels", "variety", "no_obs",
             "scale", "aperture_m")  # fmt: skip
#: Seed offset of the per-course scale draw (independent of course_gen's own stream).
SCALE_SEED_OFFSET = 1_000_003


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    unknown = set(spec) - set(SPEC_KEYS)
    if unknown:
        raise TrackDataError(f"unknown procedural spec keys {sorted(unknown)}")
    out = {
        "seed_base": int(spec.get("seed_base", 0)),
        "pool_size": int(spec.get("pool_size", 256)),
        "ng": [int(x) for x in spec.get("ng", (5, 11))],
        "laps": [int(x) for x in spec.get("laps", (1, 2))],
        "tier": str(spec.get("tier", "tractable")),
        "tunnels": bool(spec.get("tunnels", False)),
        "variety": bool(spec.get("variety", False)),
        "no_obs": bool(spec.get("no_obs", False)),
        "scale": [float(x) for x in spec.get("scale", (1.0, 1.0))],
        "aperture_m": [float(x) for x in spec.get("aperture_m", (APERTURE_MIN, APERTURE_MAX))],
    }
    if (
        out["pool_size"] < 1
        or not out["ng"][0] < out["ng"][1]
        or not out["laps"][0] < out["laps"][1]
    ):
        raise TrackDataError("procedural spec needs pool_size >= 1, ng and laps as [lo, hi)")
    for k in ("scale", "aperture_m"):
        lo, hi = out[k]
        if not 0 < lo <= hi:
            raise TrackDataError(f"procedural spec {k} needs 0 < lo <= hi")
    return out


def course_scale(spec: dict[str, Any], seed: int) -> float:
    lo, hi = spec["scale"]
    return float(lo + (hi - lo) * np.random.default_rng(seed + SCALE_SEED_OFFSET).random())


def map_aperture(a: float, spec: dict[str, Any]) -> float:
    lo, hi = spec["aperture_m"]
    t = (a - APERTURE_MIN) / (APERTURE_MAX - APERTURE_MIN)
    return float(lo + (hi - lo) * min(max(t, 0.0), 1.0))


def sample_scaled(spec: dict[str, Any], seed: int) -> dict[str, Any]:
    """``sample_course(seed)`` with every length times the course scale (apertures re-mapped)."""
    c = sample_course(
        seed,
        ng_lo=spec["ng"][0],
        ng_hi=spec["ng"][1],
        laps_lo=spec["laps"][0],
        laps_hi=spec["laps"][1],
        with_tunnels=spec["tunnels"],
        with_variety=spec["variety"],
    )
    s = course_scale(spec, seed)
    gates = [
        {"center": [v * s for v in g["center"]], "yaw": g["yaw"],
         "aperture": map_aperture(g["aperture"], spec)}
        for g in c["gates"]
    ]  # fmt: skip
    ratios = [g["aperture"] / o["aperture"] for g, o in zip(gates, c["gates"], strict=True)]
    k = float(np.mean(ratios))
    obs = []
    if not spec["no_obs"]:
        for o in c["obstacles"]:
            o2 = dict(o)
            o2["center"] = [v * s for v in o["center"]]
            # horizontal sizes follow the gates (k); vertical ones the layout (s), so pillars still
            # stand on the floor and fly-over / fly-under blocks keep their gap to the gates
            if o["kind"] == "cylinder":
                o2["radius"], o2["half_h"] = o["radius"] * k, o["half_h"] * s
            else:
                hx, hy, hz = o["half"]
                o2["half"] = [hx * k, hy * k, hz * s]
            obs.append(o2)
    return {**c, "gates": gates, "obstacles": obs, "start": [v * s for v in c["start"]], "scale": s,
            "size_scale": k}  # fmt: skip


def convert_procedural(
    spec: dict[str, Any], index: int, cfg: ConversionConfig, meta: dict[str, Any] | None = None
) -> SimCourse:
    spec = validate_spec(spec)
    seed = spec["seed_base"] + index
    c = sample_scaled(spec, seed)
    ws = cfg.world_scale
    start = np.asarray(c["start"], np.float64)
    origin = np.array([start[0], start[1], 0.0])
    frame = CourseFrame(tuple(float(x) for x in origin), float(ws))  # type: ignore[arg-type]
    z = np.array([0.0, 0.0, 1.0])
    passages = []
    for g in c["gates"]:
        n = np.array([math.cos(g["yaw"]), math.sin(g["yaw"]), 0.0])
        across = np.cross(z, n)
        rot = np.stack([across, z, n], axis=1)
        a = g["aperture"] / ws / 2.0
        outline = np.array([[-a, -a], [a, -a], [a, a], [-a, a]])
        passages.append(
            Passage(
                frame.to_course_point(g["center"]),
                rot,
                "opening",
                2 * a,
                2 * a,
                outline,
                cls="procedural-square",
                item_id="procedural",
            )  # fmt: skip
        )
    parts = []
    hc, hh = [], []
    for o in c["obstacles"]:
        cen = frame.to_course_point(o["center"])
        if o["kind"] == "cylinder":
            r, hz = o["radius"] / ws, o["half_h"] / ws
            parts.append(Primitives([PRIM_CAPSULE], cen[None], np.eye(3)[None],
                                    [[r, hz, 0.0]], [SOURCE_PROCEDURAL]))  # fmt: skip
            half = np.array([r, r, hz + r])
        else:
            half = np.asarray(o["half"], np.float64) / ws
            parts.append(Primitives([PRIM_BOX], cen[None], np.eye(3)[None], half[None],
                                    [SOURCE_PROCEDURAL]))  # fmt: skip
        if half.max() <= cfg.harness_obstacle_max_half_m / ws:
            hc.append(cen)
            hh.append(half)
    prims = Primitives.concat(parts)
    spawn = frame.to_course_point(origin)
    poly = route_polyline([spawn, *[p.center for p in passages]], True)
    cells, used = build_cell_table(
        prims, poly, corridor_m=cfg.corridor_m / ws, cell_m=cfg.cell_m / ws,
        capacity=cfg.cell_capacity, obs_range_m=cfg.obs_range_m / ws,
    )  # fmt: skip
    prims = prims.subset(used)
    harness = HarnessList(np.array(hc).reshape(-1, 3), np.array(hh).reshape(-1, 3))
    pc = np.array([p.center for p in passages])
    ext = np.array([max(p.w, p.h) / 2.0 for p in passages])
    lo = np.minimum((pc - ext[:, None]).min(axis=0), spawn) - cfg.bounds_margin_m / ws
    hi = np.maximum((pc + ext[:, None]).max(axis=0), spawn) + cfg.bounds_margin_m / ws
    legs = np.linalg.norm(np.diff(np.vstack([pc, pc[:1]]), axis=0), axis=1)
    route_len = float(np.linalg.norm(pc[0] - spawn) + c["laps"] * legs.sum())
    time_limit = min(cfg.time_max_s, cfg.time_base_s + route_len * ws / cfg.v_ref_mps)
    g0 = pc[0] - spawn
    meta = {**(meta or {}), "seed": seed, "scale": c["scale"]}
    return SimCourse(
        frame=frame,
        passages=passages,
        laps=int(c["laps"]),
        circuit=True,
        spawn_pos=spawn,
        spawn_yaw=float(math.atan2(g0[1], g0[0])),
        spawn_contract=origin,
        spawn_synthetic=False,
        floor_z=0.0,
        prims=prims,
        harness=harness,
        cells=cells,
        bounds_lo=lo,
        bounds_hi=hi,
        time_limit_s=float(time_limit),
        corridor_m=cfg.corridor_m / ws,
        meta=meta,
        stats={
            "passages": len(passages),
            "laps": int(c["laps"]),
            "route_length_m": round(route_len * ws, 2),
            "primitives": len(prims),
            "harness_list": len(harness),
        },
    )
