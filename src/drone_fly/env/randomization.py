"""Pure per-episode domain-randomization samplers (UC-08 AC3/AC4, UC-09 AC5).

This module is **pure**: it imports only :mod:`numpy` and the config dataclasses — no env,
no adapter, no sim, no torch — so it is trivially unit-testable and can never pull the
heavy stack in. Both samplers draw off a caller-supplied ``numpy`` ``Generator`` (the env's
seeded ``self.np_random``), so a given seed reproduces the same course *and* dynamics
stream (AC4). The env pins the draw order (course then dynamics) and never draws for a
disabled axis, so toggling one axis never shifts the other's stream.

Course solvability (UC-09 AC5)
------------------------------
A sampled N-gate course is *flyable* iff:

* every gate centre sits **strictly inside** the vertical safety corridor
  ``(floor_z + z_margin, ceiling_z - z_margin)`` and within ``+/- lateral_bound`` in y;
* every aperture is at least ``aperture_min`` (the drone must fit through);
* the start height is strictly inside the corridor (``SimpleDroneAdapter.reset`` flags a
  step-0 collision if the spawn touches a bound);
* gate x-coordinates are **strictly increasing** (free-3D means free y/z, monotonic x);
* adjacent gates are ≥ ``min_gate_spacing`` apart in 3D (also the anti-tunneling guard);
* the first gate is ≥ ``min_start_gate_gap`` ahead of the start in x;
* ``finish_x`` is ≥ ``min_gate_finish_gap`` beyond the last gate's x;
* the **start is outside gate 0's capture sphere** (else the first gate would auto-pass).

The sampler walks x **forward from the start** in incremental deltas, so ordering, spacing,
and the finish placement are structural (correct by construction) rather than checked after
an absolute draw. Each attempt consumes a **fixed** number of draws given N, so a seed is
reproducible (AC6). Degenerate draws are rejected and resampled up to
``max_resample_attempts``; on exhaustion the sampler falls back to a deterministic,
**zero-RNG** course that is solvable-by-construction for every N in ``[1, 10]`` — so
sampling is deterministic and can never loop forever.
"""

from __future__ import annotations

import numpy as np

from drone_fly.env.config import (
    BatteryConfig,
    CourseConfig,
    DynamicsParams,
    GateSpec,
    ObstacleSpec,
    PadSpec,
    RandomizationConfig,
)
from drone_fly.env.obstacles import point_in_cylinder

#: A full battery is one unit of charge; the energy model's budget for a single flight leg
#: between recharges (UC-18). Normalized to match the adapter's ``battery ∈ [0, 1]``.
_BATTERY_BUDGET = 1.0


def is_course_solvable(
    course: CourseConfig,
    rcfg: RandomizationConfig,
    battery: BatteryConfig | None = None,
) -> bool:
    """Return ``True`` iff ``course`` is flyable under ``rcfg``'s solvability bounds (AC5).

    ``battery`` (UC-18, appended **last** so every existing caller is unaffected) enables the
    **recharge-reachability** guarantee (AC4). When it is ``None`` or disabled the recharge check
    is skipped entirely and the result is byte-identical to UC-15/16/17. When it is enabled and
    the course's full reference 3D flight path costs **more than one charge**, the course is
    solvable ONLY if it carries a reachable *covering* of recharge pads — recharge pads placed so
    that every induced sub-path (start→…→first pad, pad→…→next pad, last pad→…→finish, each with
    its descend/climb legs) fits within one charge. Otherwise the finish is unreachable and the
    course is rejected. This makes it impossible to emit an energy-constrained course without a
    completable covering; it mirrors the UC-15 obstacle-clearance guard (a conservative,
    model-level geometric guarantee).
    """
    if course.num_gates < 1:
        return False

    start = course.start_position
    start_x, start_y, start_z = float(start[0]), float(start[1]), float(start[2])

    lo = course.floor_z + rcfg.z_margin
    hi = course.ceiling_z - rcfg.z_margin

    # Start height strictly inside the corridor (spawn must not touch a bound).
    if not (lo < start_z < hi):
        return False
    if abs(start_y) > rcfg.lateral_bound:
        return False

    gates = course.gates

    # First gate far enough ahead of the start along +x.
    if gates[0].center[0] - start_x < rcfg.min_start_gate_gap:
        return False

    # Start must be OUTSIDE gate 0's capture sphere, else it auto-passes on step 0.
    if float(np.linalg.norm(course.start - gates[0].position)) <= gates[0].aperture:
        return False

    prev_x: float | None = None
    prev_center: np.ndarray | None = None
    for gate in gates:
        gx, gy, gz = float(gate.center[0]), float(gate.center[1]), float(gate.center[2])
        # Aperture must admit the drone.
        if gate.aperture < rcfg.aperture_min:
            return False
        # Gate centre strictly inside the vertical corridor and within lateral bounds.
        if not (lo < gz < hi):
            return False
        if abs(gy) > rcfg.lateral_bound:
            return False
        # Strictly-increasing x (monotonic) and 3D spacing between adjacent gates.
        if prev_x is not None:
            if gx <= prev_x:
                return False
            if float(np.linalg.norm(gate.position - prev_center)) < rcfg.min_gate_spacing:
                return False
        prev_x = gx
        prev_center = gate.position

    # Finish beyond the last gate.
    if course.finish_x - float(gates[-1].center[0]) < rcfg.min_gate_finish_gap:
        return False

    # -- obstacle evadability (UC-15 AC3, rewritten for UC-35 AC3/AC4) -------------------
    # UC-35 replaces UC-15's "polyline must stay FAR from every pillar" guard (which forbade
    # on-route pillars) with a **forced-but-evadable** model: pillars are meant to sit near the
    # nominal path so the drone must deviate, yet the course must stay feasible. A course with
    # pillars is flyable iff, for every pillar: (i) no start / gate centre / finish sits inside
    # it AND every such anchor keeps ``radius + drone_radius + evasion_margin`` horizontal
    # clearance (gate passability — the drone can still reach each waypoint); (ii) an escape lane
    # at least ``drone_radius + evasion_margin`` wide exists on the wider side of the pillar within
    # ``±lateral_bound`` (the pillar never fully blocks the corridor); and (iii) no two pillars sit
    # within ``r_i + r_j + 2·(drone_radius + evasion_margin)`` of each other (no unevadable multi-
    # pillar wall). Together these constructively guarantee ≥1 collision-free deviated path exists.
    # Checked regardless of ``enable_obstacles`` — the flag governs *sampling*, not what makes a
    # course (which already carries obstacles) solvable.
    obstacles = getattr(course, "obstacles", ())
    if obstacles:
        floor_z = course.floor_z
        last = gates[-1]
        finish_pt = np.asarray([course.finish_x, last.center[1], last.center[2]], dtype=np.float64)
        anchors = [course.start, *(g.position for g in gates), finish_pt]
        anchors_xy = [np.asarray(a, dtype=np.float64)[:2] for a in anchors]
        for i, obstacle in enumerate(obstacles):
            axis = np.asarray(
                [float(obstacle.center[0]), float(obstacle.center[1])], dtype=np.float64
            )
            radius = float(obstacle.radius)
            gate_req = radius + rcfg.drone_radius + rcfg.obstacle_evasion_margin
            # (i) no anchor inside the pillar, and every anchor keeps gate-passability clearance.
            for pt in anchors:
                if point_in_cylinder(pt, obstacle, floor_z):
                    return False
            for a_xy in anchors_xy:
                if float(np.linalg.norm(axis - a_xy)) < gate_req:
                    return False
            # (ii) an escape lane on the wider side stays within the lateral corridor.
            if not _obstacle_escape_lane_ok(axis, radius, rcfg):
                return False
            # (iii) pairwise separation — two pillars must not form an unevadable wall.
            for j in range(i + 1, len(obstacles)):
                other = obstacles[j]
                other_axis = np.asarray(
                    [float(other.center[0]), float(other.center[1])], dtype=np.float64
                )
                need = (
                    radius
                    + float(other.radius)
                    + 2.0 * (rcfg.drone_radius + rcfg.obstacle_evasion_margin)
                )
                if float(np.linalg.norm(axis - other_axis)) < need:
                    return False

    # -- pad placement guard (UC-35 AC1/AC2) --------------------------------------------
    # Service (recharge/repair) pads are placed OFF gate columns (UC-35 AC1). Defensively enforce
    # the placement invariants on any course that carries pads: every pad keeps ``R_pad`` from every
    # gate centre, stays within the lateral corridor, and its floor→gate descend column clears every
    # pillar (so a docking detour never clips an obstacle).
    pads = getattr(course, "pads", ())
    if pads:
        r_pad = rcfg.pad_min_gate_distance
        gate_centres_xy = [np.asarray(g.center, dtype=np.float64)[:2] for g in gates]
        for pad in pads:
            pad_xy = np.asarray(pad.center, dtype=np.float64)
            if abs(float(pad_xy[1])) > rcfg.lateral_bound:
                return False
            for gc in gate_centres_xy:
                if float(np.linalg.norm(pad_xy - gc)) < r_pad:
                    return False
            if not _descend_column_clear(pad_xy, obstacles, rcfg):
                return False

    # -- recharge reachability (UC-18 AC4) ----------------------------------------------
    # Only when battery physics are enabled: if the full reference 3D path costs more than one
    # charge, the course MUST carry a reachable covering of recharge pads (else the finish is
    # unreachable on a single charge with no way to refill → not solvable). Skipped entirely when
    # ``battery`` is None/disabled, so non-recharge callers are byte-identical.
    if battery is not None and battery.enabled:
        total_energy = _path_energy(_reference_path(course), rcfg, battery)
        if total_energy > _BATTERY_BUDGET:
            rechargeable = [
                pad for pad in getattr(course, "pads", ()) if getattr(pad, "rechargeable", False)
            ]
            if not rechargeable:
                return False
            if not _recharge_covering_valid(course, rcfg, battery, rechargeable):
                return False

    return True


def _floor_point(pad: PadSpec, floor_z: float) -> np.ndarray:
    """The floor-anchored 3D waypoint at ``pad``'s actual ``(x, y)``: ``(pad.x, pad.y, floor_z)``.

    UC-35: a recharge pad is placed **off** the gate columns, so the recharge detour descends to the
    pad's own ``(x, y)`` — not the gate's. Modelling the pad as this floor waypoint makes the energy
    model count the lateral detour + descend/climb legs against the pad's real position (UC-18/24
    keyed this off the gate column, which was only correct while pads sat on gates).
    """
    return np.asarray([float(pad.center[0]), float(pad.center[1]), floor_z], dtype=np.float64)


def _obstacle_escape_lane_ok(axis: np.ndarray, radius: float, rcfg: RandomizationConfig) -> bool:
    """Return ``True`` iff the drone can pass the pillar on its wider side within the corridor.

    Measures the free gap between the pillar's edge and each lateral wall (``±lateral_bound``); an
    escape lane exists iff the WIDER side leaves at least ``drone_radius + evasion_margin`` of room.
    This guarantees the pillar never fully blocks the ``±lateral_bound`` corridor (UC-35 AC4).
    """
    ay = float(axis[1])
    lane_pos = rcfg.lateral_bound - (ay + radius)  # gap to the +y wall
    lane_neg = (ay - radius) + rcfg.lateral_bound  # gap to the -y wall
    return max(lane_pos, lane_neg) >= rcfg.drone_radius + rcfg.obstacle_evasion_margin


def _reference_path(course: CourseConfig) -> list[np.ndarray]:
    """The start→gates→finish reference polyline as a list of 3D points (UC-18 energy model)."""
    gates = course.gates
    last = gates[-1]
    finish_pt = np.asarray([course.finish_x, last.center[1], last.center[2]], dtype=np.float64)
    return [course.start, *(g.position for g in gates), finish_pt]


def _path_energy(
    points: list[np.ndarray],
    rcfg: RandomizationConfig,
    battery: BatteryConfig,
) -> float:
    """Conservative modelled energy (fraction of a full charge) to fly the 3D polyline (UC-18).

    A pure, deterministic **heuristic** (dt cancels): the drone is assumed to cruise the full 3D
    path length at ``recharge_nominal_speed`` holding ``recharge_nominal_throttle``, draining at
    the battery's ``idle_rate + throttle_rate * nominal_throttle`` per second, all scaled by the
    ``recharge_energy_margin`` safety factor (>1 ⇒ over-estimate ⇒ safe direction for the AC4
    guarantee). Its ONLY guarantee is model-level reachability; AC3/AC6 "load-bearing" claims are
    discharged empirically by measured sim rollouts, never by this number.
    """
    total_len = 0.0
    for a, b in zip(points[:-1], points[1:], strict=True):
        total_len += float(np.linalg.norm(np.asarray(b, dtype=np.float64) - np.asarray(a)))
    drain_per_sec = float(battery.idle_rate) + float(battery.throttle_rate) * float(
        rcfg.recharge_nominal_throttle
    )
    nominal_speed = max(float(rcfg.recharge_nominal_speed), 1e-9)
    flight_time = total_len / nominal_speed
    return float(rcfg.recharge_energy_margin) * drain_per_sec * flight_time


def _recharge_pad_order(course: CourseConfig, rechargeable) -> list[tuple[int, PadSpec]]:
    """Order rechargeable pads along the course by their **nearest gate** (UC-35 retarget of UC-18).

    UC-18/24 associated a recharge pad with the gate it sat *under*; UC-35 places pads **off** the
    gate columns, so each rechargeable pad is instead associated with its geometrically **nearest**
    gate — the leg the recharge detour branches off. Returns ``(gate_index, pad)`` pairs sorted by
    gate index (ties broken by pad x), which fixes the order the covering check reconstructs the
    refuel legs in.
    """
    gates = course.gates
    pairs: list[tuple[int, PadSpec]] = []
    for pad in rechargeable:
        px, py = float(pad.center[0]), float(pad.center[1])
        best_idx = 0
        best_d: float | None = None
        for idx, gate in enumerate(gates):
            d = float(np.hypot(float(gate.center[0]) - px, float(gate.center[1]) - py))
            if best_d is None or d < best_d:
                best_idx, best_d = idx, d
        pairs.append((best_idx, pad))
    pairs.sort(key=lambda kp: (kp[0], float(kp[1].center[0])))
    return pairs


def _recharge_covering_valid(
    course: CourseConfig,
    rcfg: RandomizationConfig,
    battery: BatteryConfig,
    rechargeable,
) -> bool:
    """Re-verify every induced recharge sub-path fits one charge (UC-18 AC4 covering check).

    Reconstructs the flight as: ``start → gates… → floor(first recharge pad)``, then
    ``floor(pad) → gates… → floor(next recharge pad)`` for each subsequent pad, then
    ``floor(last pad) → gates… → finish``. Each pad's floor point is its **actual** ``(x, y)``
    (UC-35: pads sit off the gate columns), so the detour to the pad and back is charged against the
    pad's real position. Each leg is charged from full (start, or a fully-refilled pad) and must
    cost ≤ one charge. Uses the same :func:`_path_energy` model as the placer, so a covering the
    placer produced always re-verifies (a tiny epsilon absorbs float round-off).
    """
    gates = course.gates
    n = len(gates)
    floor_z = course.floor_z
    pad_order = _recharge_pad_order(course, rechargeable)
    last = gates[-1]
    finish_pt = np.asarray([course.finish_x, last.center[1], last.center[2]], dtype=np.float64)

    legs: list[list[np.ndarray]] = []
    charged_point = course.start
    cursor = 0  # first gate not yet consumed by a prior leg
    for k, pad in pad_order:
        floor_pt = _floor_point(pad, floor_z)
        pts = [charged_point, *(gates[j].position for j in range(cursor, k + 1))]
        pts.append(floor_pt)
        legs.append(pts)
        charged_point = floor_pt
        cursor = max(cursor, k + 1)
    # Final leg: from the last charge point through any remaining gates to the finish.
    final_pts = [charged_point, *(gates[j].position for j in range(cursor, n)), finish_pt]
    legs.append(final_pts)

    for leg in legs:
        if _path_energy(leg, rcfg, battery) > _BATTERY_BUDGET + 1e-9:
            return False
    return True


def _segment_axis_distance_2d(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """Horizontal (x, y) distance from a pillar ``axis`` to the segment ``a→b`` (UC-15 AC3)."""
    p = np.asarray(a, dtype=np.float64)[:2]
    q = np.asarray(b, dtype=np.float64)[:2]
    seg = q - p
    seg_len_sq = float(seg @ seg)
    if seg_len_sq <= 1e-18:
        return float(np.linalg.norm(axis - p))
    t = float((axis - p) @ seg / seg_len_sq)
    t = min(1.0, max(0.0, t))
    closest = p + t * seg
    return float(np.linalg.norm(axis - closest))


def _draw_num_gates(rng: np.random.Generator, rcfg: RandomizationConfig) -> int:
    """Draw the gate count uniformly from the inclusive ``num_gates_range`` (UC-09 AC5)."""
    lo, hi = rcfg.num_gates_range
    lo = max(1, int(lo))
    hi = max(lo, int(hi))
    return int(rng.integers(lo, hi + 1))


def sample_course(
    rng: np.random.Generator,
    rcfg: RandomizationConfig,
    base_course: CourseConfig,
    battery: BatteryConfig | None = None,
) -> CourseConfig:
    """Sample a solvable N-gate :class:`CourseConfig` off ``rng`` (UC-09 AC5, AC6).

    Arena bounds (``floor_z`` / ``ceiling_z``) are **not** randomized — they are copied from
    ``base_course``. Draw order is pinned per attempt: ``num_gates`` first, then the start,
    then a forward incremental-x walk over the gates (per-gate gap, y, z, aperture), then the
    finish gap, then — **only when ``rcfg.enable_obstacles``** — the obstacle draws (UC-15).
    Pinning the obstacle draws *last* means a disabled obstacle axis consumes zero RNG and so
    never perturbs the UC-08/09 course stream. RNG consumption is fixed given N (and the drawn
    obstacle count), so a seed is reproducible. Solvability is tested and failures resample up
    to ``rcfg.max_resample_attempts``; on exhaustion a deterministic zero-RNG
    :func:`_fallback_course` (solvable-by-construction, **no obstacles**) is returned.

    ``battery`` (UC-18; UC-24 single-pad model): when ``rcfg.enable_recharge`` **and**
    ``battery.enabled`` — and/or when ``rcfg.enable_repair`` — each solvable candidate is run
    through :func:`_place_service_pads` (a pure, **zero-RNG** feature-presence placer): it appends
    **exactly one** ``rechargeable`` pad and/or **exactly one** ``repairable`` pad at eligible gate
    anchors, then the result is re-verified by the (battery-aware, when recharge is active)
    :func:`is_course_solvable`. A candidate with no eligible anchor (every gate's descend column
    clips a pillar), or one whose single recharge pad cannot cover a cranked-battery-constrained
    course, is rejected and resampled. When neither service axis is active the placer is never
    called and ``is_course_solvable`` is invoked with ``battery=None``, so the RNG stream and result
    are byte-identical to UC-15/16/17 (AC5).
    """
    recharge_active = rcfg.enable_recharge and battery is not None and battery.enabled
    service_active = recharge_active or rcfg.enable_repair
    n = _draw_num_gates(rng, rcfg)

    for _ in range(max(int(rcfg.max_resample_attempts), 1)):
        start_x = float(rng.uniform(*rcfg.start_x_range))
        start_y = float(rng.uniform(*rcfg.start_y_range))
        start_z = float(rng.uniform(*rcfg.start_z_range))

        gates: list[GateSpec] = []
        x = start_x + float(rng.uniform(*rcfg.first_gate_gap_range))
        for i in range(n):
            if i > 0:
                x += float(rng.uniform(*rcfg.gate_gap_range))
            gy = float(rng.uniform(*rcfg.gate_center_y_range))
            gz = float(rng.uniform(*rcfg.gate_center_z_range))
            aperture = float(rng.uniform(*rcfg.gate_aperture_range))
            gates.append(GateSpec(center=(x, gy, gz), aperture=aperture))

        finish_x = x + float(rng.uniform(*rcfg.finish_gap_range))

        # Obstacle draws come LAST and only when enabled (pinned order → disabled axis makes no
        # draw and never perturbs the gate/dynamics stream). UC-35: pillars are drawn *between*
        # consecutive waypoints (forced-but-evadable); each drawn pillar consumes a fixed number of
        # draws whether accepted or skipped, so a seed stays reproducible and the count is an upper
        # bound. The whole candidate (gates + pillars) is then reject-resampled together.
        obstacles: tuple[ObstacleSpec, ...] = ()
        if rcfg.enable_obstacles:
            obstacles = _sample_obstacles(
                rng, rcfg, tuple(gates), (start_x, start_y, start_z), finish_x
            )

        candidate = CourseConfig(
            start_position=(start_x, start_y, start_z),
            gates=tuple(gates),
            finish_x=finish_x,
            floor_z=base_course.floor_z,
            ceiling_z=base_course.ceiling_z,
            obstacles=obstacles,
        )
        # UC-24: place the single recharge / repair feature pads on the candidate (zero-RNG, so the
        # stream is untouched). ``None`` means no eligible anchor (every gate's descend column clips
        # a pillar) → reject-resample.
        if service_active:
            placed = _place_service_pads(candidate, rcfg, battery)
            if placed is None:
                continue
            candidate = placed
        if is_course_solvable(candidate, rcfg, battery=battery if recharge_active else None):
            return candidate

    # Hard cap hit: deterministic, zero-RNG fallback (solvable-by-construction for this N).
    return _fallback_course(n, base_course, rcfg, battery=battery)


def _course_segments(
    start_xy: np.ndarray, gates: tuple[GateSpec, ...]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The waypoint→waypoint segments of the course as 2D ``(p, q)`` pairs (UC-35 AC3).

    Waypoints are the start followed by every gate centre; segments are consecutive pairs, so the
    set **includes** ``start→first-gate`` (the AC3 assumption). Used to place pillars *between*
    consecutive waypoints (inside the corridor around each segment).
    """
    waypoints = [start_xy, *(np.asarray(g.center, dtype=np.float64)[:2] for g in gates)]
    return list(zip(waypoints[:-1], waypoints[1:], strict=True))


def _obstacle_evadable(
    obstacle: ObstacleSpec,
    anchors_xy: list[np.ndarray],
    accepted: list[ObstacleSpec],
    rcfg: RandomizationConfig,
) -> bool:
    """Return ``True`` iff a candidate pillar satisfies the UC-35 AC4 evadability invariants.

    Mirrors (and is stricter than) the obstacle checks in :func:`is_course_solvable`: gate
    passability (≥ ``radius + drone_radius + evasion_margin`` from every anchor), an escape lane on
    the wider side within ``±lateral_bound``, and pairwise separation from every already-accepted
    pillar (with an extra ``min_obstacle_separation`` slack, so an accepted set always clears the
    solvability guard). Used at sampling time to accept/skip a drawn pillar with NO extra RNG.
    """
    axis = np.asarray(obstacle.center, dtype=np.float64)
    radius = float(obstacle.radius)
    gate_req = radius + rcfg.drone_radius + rcfg.obstacle_evasion_margin
    for a_xy in anchors_xy:
        if float(np.linalg.norm(axis - a_xy)) < gate_req:
            return False
    if not _obstacle_escape_lane_ok(axis, radius, rcfg):
        return False
    for other in accepted:
        other_axis = np.asarray(other.center, dtype=np.float64)
        need = (
            radius
            + float(other.radius)
            + 2.0 * (rcfg.drone_radius + rcfg.obstacle_evasion_margin)
            + rcfg.min_obstacle_separation
        )
        if float(np.linalg.norm(axis - other_axis)) < need:
            return False
    return True


def _sample_obstacles(
    rng: np.random.Generator,
    rcfg: RandomizationConfig,
    gates: tuple[GateSpec, ...],
    start: tuple[float, float, float],
    finish_x: float,
) -> tuple[ObstacleSpec, ...]:
    """Draw pillars **between consecutive waypoints**, forced-but-evadable (UC-35 AC3/AC4).

    Draw order per pillar is pinned and **fixed** (segment index, along-fraction, side, perp
    fraction, radius, height); every drawn pillar consumes the SAME number of draws whether it is
    accepted or skipped, so a seed is reproducible and toggling the axis never shifts other streams.
    The drawn count is an **upper bound**: a pillar whose position fails the evadability invariants
    (:func:`_obstacle_evadable`) is skipped (no extra RNG), so the realised count can be lower —
    even zero on pathologically short courses.

    Each pillar is anchored on an *eligible* segment (long enough to host an interior band), placed
    in that segment's interior (``obstacle_along_margin_frac`` trimmed off each end) and offset
    perpendicular by up to ``min(obstacle_corridor_half_width, radius + drone_radius)`` — close
    enough that the straight path passes within ``radius`` of the axis, so the drone is FORCED to
    evade, while the accept test guarantees an escape lane and gate passability keep it feasible.
    """
    lo, hi = rcfg.obstacle_count_range
    lo = max(0, int(lo))
    hi = max(lo, int(hi))
    count = int(rng.integers(lo, hi + 1))

    start_xy = np.asarray(start, dtype=np.float64)[:2]
    segments = _course_segments(start_xy, gates)
    # Eligible segments are long enough to host a pillar interior clear of both endpoints.
    min_len = 2.0 * rcfg.obstacle_gate_clearance
    eligible = [(p, q) for (p, q) in segments if float(np.linalg.norm(q - p)) >= min_len]
    if not eligible:
        return ()

    last = gates[-1]
    finish_xy = np.asarray([float(finish_x), float(last.center[1])], dtype=np.float64)
    anchors_xy = [
        start_xy,
        *(np.asarray(g.center, dtype=np.float64)[:2] for g in gates),
        finish_xy,
    ]

    margin = rcfg.obstacle_along_margin_frac
    accepted: list[ObstacleSpec] = []
    for _ in range(count):
        # Fixed draws per pillar (consumed regardless of the accept/skip outcome below).
        seg_i = int(rng.integers(0, len(eligible)))
        t = float(rng.uniform(margin, 1.0 - margin))
        side = 1.0 if rng.random() < 0.5 else -1.0
        offset_frac = float(rng.uniform(0.0, 1.0))
        radius = float(rng.uniform(*rcfg.obstacle_radius_range))
        height = float(rng.uniform(*rcfg.obstacle_height_range))

        p, q = eligible[seg_i]
        seg = q - p
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            continue  # degenerate segment (draws already consumed) → skip
        path_dir = seg / seg_len
        perp = np.array([-path_dir[1], path_dir[0]], dtype=np.float64)
        along = p + t * seg
        # Perpendicular offset within [0, min(corridor_half_width, radius + drone_radius)] so the
        # straight path passes within ``radius`` of the axis (forced evasion), capped to corridor.
        max_off = min(rcfg.obstacle_corridor_half_width, radius + rcfg.drone_radius)
        axis = along + side * offset_frac * max_off * perp

        candidate = ObstacleSpec(
            center=(float(axis[0]), float(axis[1])), radius=radius, height=height
        )
        if _obstacle_evadable(candidate, anchors_xy, accepted, rcfg):
            accepted.append(candidate)
        # else: skip — draws already consumed, so the count is an upper bound.
    return tuple(accepted)


def _descend_column_clear(
    point_xy,
    obstacles,
    rcfg: RandomizationConfig,
) -> bool:
    """Return ``True`` iff the floor→gate descend column at ``point_xy`` clears every pillar.

    A service pad forces the drone to descend a vertical column at the pad's ``(x, y)`` from flight
    altitude to the floor and climb back out. Since every pillar is floor-anchored, that column
    overlaps a pillar's z-band whenever it comes within ``obstacle.radius + obstacle_clearance``
    horizontally — which would make the docking detour clip the pillar. Rejecting such a point keeps
    the pad×obstacle composition safe **by construction**. UC-35 generalises this from a
    ``GateSpec`` to an arbitrary ``(x, y)`` point so it guards the pad's actual (off-gate) position.
    """
    px, py = float(point_xy[0]), float(point_xy[1])
    for obstacle in obstacles:
        ox, oy = float(obstacle.center[0]), float(obstacle.center[1])
        horiz = float(np.hypot(px - ox, py - oy))
        if horiz < float(obstacle.radius) + rcfg.obstacle_clearance:
            return False
    return True


def _offset_pad_off_gates(
    course: CourseConfig,
    rcfg: RandomizationConfig,
    anchor_idx: int,
    avoid: tuple[np.ndarray, ...] = (),
) -> np.ndarray | None:
    """Find a **zero-RNG** off-gate pad position anchored near gate ``anchor_idx`` (UC-35 AC1/AC2).

    Deterministically scans a **fixed** candidate set of offsets from the anchor gate — perp to
    the local path first (both signs), then axial, at increasing distances from ``R_pad`` — and
    returns the first ``(x, y)`` that is (a) ≥ ``R_pad`` (``pad_min_gate_distance``) from **every**
    gate centre, (b) within the lateral corridor and the course x-span, (c) descend-column-clear of
    every pillar, and (d) ≥ ``R_pad`` from every already-placed pad in ``avoid``. Because the
    candidate order is fixed and no RNG is drawn, the placer only ever *appends* a pad and never
    shifts the sampled geometry (AC5). Returns ``None`` when no candidate satisfies the constraints.
    """
    gates = course.gates
    n = len(gates)
    r_pad = rcfg.pad_min_gate_distance
    anchor_xy = np.asarray(gates[anchor_idx].center, dtype=np.float64)[:2]

    prev_pt = (
        np.asarray(gates[anchor_idx - 1].center, dtype=np.float64)[:2]
        if anchor_idx > 0
        else np.asarray(course.start, dtype=np.float64)[:2]
    )
    if anchor_idx < n - 1:
        next_pt = np.asarray(gates[anchor_idx + 1].center, dtype=np.float64)[:2]
    else:
        next_pt = np.asarray(
            [course.finish_x, float(gates[anchor_idx].center[1])], dtype=np.float64
        )
    direction = next_pt - prev_pt
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        path_dir = np.array([1.0, 0.0], dtype=np.float64)
    else:
        path_dir = direction / norm
    perp = np.array([-path_dir[1], path_dir[0]], dtype=np.float64)

    gate_centres = [np.asarray(g.center, dtype=np.float64)[:2] for g in gates]
    start_x = float(np.asarray(course.start, dtype=np.float64)[0])
    # Fixed, increasing candidate distances starting just above R_pad.
    distances = [r_pad + 0.25 * k for k in range(1, 13)]
    for offset_dir in (perp, -perp, path_dir, -path_dir):
        for dist in distances:
            cand = anchor_xy + offset_dir * dist
            if abs(float(cand[1])) > rcfg.lateral_bound:
                continue
            if not (start_x <= float(cand[0]) <= course.finish_x):
                continue
            if any(float(np.linalg.norm(cand - gc)) < r_pad for gc in gate_centres):
                continue
            if not _descend_column_clear(cand, course.obstacles, rcfg):
                continue
            if any(float(np.linalg.norm(cand - other)) < r_pad for other in avoid):
                continue
            return cand
    return None


def _eligible_gate_indices(course: CourseConfig, rcfg: RandomizationConfig) -> list[int]:
    """Ordered gate indices for which an off-gate pad position exists (UC-35 redefinition).

    A service pad is placed **off** its anchor gate (UC-35 AC1), so an anchor is eligible iff
    :func:`_offset_pad_off_gates` can find a valid off-gate position near it (≥ ``R_pad`` from every
    gate, in-corridor, descend-column-clear). On an obstacle-free course (including every
    :func:`_fallback_course`) an offset always exists, so the list is non-empty and placement never
    deadlocks.
    """
    return [
        i for i in range(len(course.gates)) if _offset_pad_off_gates(course, rcfg, i) is not None
    ]


def _energy_midpoint_gate(
    course: CourseConfig,
    rcfg: RandomizationConfig,
    battery: BatteryConfig,
    candidates: list[int],
) -> int:
    """Pick the ``candidates`` gate whose start-cumulative path energy is nearest the path midpoint.

    Deterministic and **zero-RNG**: for each candidate gate the modelled energy of
    ``start → gates[0..i]`` is compared to half the full reference-path energy; the nearest wins,
    ties broken toward the lower index (candidates are scanned in ascending order with a strict
    ``<``). Anchoring the single recharge pad near the energy midpoint gives the most useful split
    of a constrained course into two comparable-cost legs.
    """
    target = 0.5 * _path_energy(_reference_path(course), rcfg, battery)
    best = candidates[0]
    best_d: float | None = None
    for i in candidates:
        cum = _path_energy(
            [course.start, *(course.gates[j].position for j in range(i + 1))], rcfg, battery
        )
        d = abs(cum - target)
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best


def _index_midpoint_gate(candidates: list[int], num_gates: int) -> int:
    """Pick the ``candidates`` gate nearest the middle **index** of the course (UC-24, zero-RNG).

    The repair axis has no energy model (damage does not gate reachability), so a repair pad is pure
    feature placement: anchor it at the eligible gate closest to the course's middle index, ties
    broken toward the lower index (candidates scanned ascending with a strict ``<``).
    """
    target = (num_gates - 1) / 2.0
    best = candidates[0]
    best_d: float | None = None
    for i in candidates:
        d = abs(i - target)
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best


def _with_pads(course: CourseConfig, extra_pads: list[PadSpec]) -> CourseConfig:
    """Return a copy of ``course`` with ``extra_pads`` appended after its existing pads (0-RNG)."""
    return CourseConfig(
        start_position=course.start_position,
        gates=course.gates,
        finish_x=course.finish_x,
        floor_z=course.floor_z,
        ceiling_z=course.ceiling_z,
        obstacles=course.obstacles,
        pads=(*course.pads, *extra_pads),
    )


def _place_single_recharge_pad(
    course: CourseConfig,
    rcfg: RandomizationConfig,
    battery: BatteryConfig,
) -> CourseConfig | None:
    """Place **exactly one** ``rechargeable`` pad **off** an eligible gate anchor (UC-35 AC1).

    UC-24 redefinition of UC-18's placer: a single recharge pad is placed **regardless** of whether
    the course is energy-constrained (UC-18 placed pads only on over-budget courses, so under the
    shipped default battery *zero* pads were ever produced — the unreachable-placement bug this
    fixes). The anchor gate is the eligible gate nearest the path-energy midpoint
    (:func:`_energy_midpoint_gate`); UC-35 then offsets the pad **off** that gate's column via the
    zero-RNG :func:`_offset_pad_off_gates` (≥ ``R_pad`` from every gate — AC1). Anchor + offset are
    deterministic and **zero-RNG**, so placement never perturbs the seeded stream and only ever
    *appends* a pad (never shifts gate geometry). The pad radius is ``rcfg.recharge_pad_radius`` and
    it is tagged ``rechargeable=True``; existing course pads are preserved.

    Returns ``None`` when no eligible anchor admits an off-gate pad position (see
    :func:`_eligible_gate_indices`) → the caller reject-resamples.

    **One-pad narrowing of UC-18's covering guarantee (documented, load-bearing):** a single pad
    makes only courses solvable-with-one-recharge completable — not the arbitrary multi-pad covers
    UC-18 could build. Solvability is still enforced by the battery-aware :func:`is_course_solvable`
    (which accepts any pad set via :func:`_recharge_covering_valid`): under the **shipped default
    battery** a course is non-constrained, the recharge branch is skipped, and the lone pad is a
    valid bonus (always solvable). The battery-aware ``assert`` in :func:`_fallback_course` can
    therefore fire ONLY under a non-default, cranked-constrained battery whose one-charge reach is
    below a single fallback gate gap plus its vertical legs — a fail-loud precondition, not a silent
    hole (unreachable under the shipped default: ~133 m of one-charge reach vs a ~20–30 m realistic
    path).
    """
    eligible = _eligible_gate_indices(course, rcfg)
    if not eligible:
        return None
    k = _energy_midpoint_gate(course, rcfg, battery, eligible)
    pad_xy = _offset_pad_off_gates(course, rcfg, k)
    if pad_xy is None:  # defensive — k is drawn from `eligible`, so an offset always exists
        return None
    pad = PadSpec(
        center=(float(pad_xy[0]), float(pad_xy[1])),
        radius=float(rcfg.recharge_pad_radius),
        rechargeable=True,
    )
    return _with_pads(course, [pad])


def _place_single_repair_pad(
    course: CourseConfig,
    rcfg: RandomizationConfig,
) -> CourseConfig | None:
    """Place **exactly one** ``repairable`` pad at an eligible gate anchor (UC-24, zero-RNG).

    Symmetric with :func:`_place_single_recharge_pad` but with **no energy model** — damage does not
    gate whether the finish is reachable, so a repair pad is pure feature presence: a damaged drone
    simply *can* land and recover. The anchor gate is the eligible gate nearest the course's middle
    index (:func:`_index_midpoint_gate`); UC-35 then offsets the pad **off** that gate's column via
    the zero-RNG :func:`_offset_pad_off_gates` (≥ ``R_pad`` from every gate — AC1). The pad radius
    is ``rcfg.repair_pad_radius`` and it is tagged ``repairable=True``; existing course pads are
    preserved. Returns ``None`` when no eligible anchor admits an off-gate pad position → the caller
    reject-resamples.
    """
    eligible = _eligible_gate_indices(course, rcfg)
    if not eligible:
        return None
    k = _index_midpoint_gate(eligible, len(course.gates))
    pad_xy = _offset_pad_off_gates(course, rcfg, k)
    if pad_xy is None:  # defensive — k is drawn from `eligible`, so an offset always exists
        return None
    pad = PadSpec(
        center=(float(pad_xy[0]), float(pad_xy[1])),
        radius=float(rcfg.repair_pad_radius),
        repairable=True,
    )
    return _with_pads(course, [pad])


def _place_service_pads(
    course: CourseConfig,
    rcfg: RandomizationConfig,
    battery: BatteryConfig | None,
) -> CourseConfig | None:
    """Place the single recharge and/or repair feature pads on ``course`` (UC-24 coordinator).

    Dispatches by which service axes are active:

    * recharge only → :func:`_place_single_recharge_pad`;
    * repair only → :func:`_place_single_repair_pad`;
    * **both** → one ``rechargeable`` pad offset off the energy-midpoint anchor and one
      ``repairable`` pad offset off a **distinct** anchor (recharge then repair, in that order),
      each ≥ ``R_pad`` from every gate AND from each other (UC-35 AC1). When only **one** eligible
      anchor exists (e.g. a 1-gate course or the fallback), or no distinct feasible off-gate repair
      position exists, the two co-locate into a **single dual-purpose pad**
      (``rechargeable=True, repairable=True``) — a documented exception that still counts as "one of
      each" and avoids one pad shadowing the other.

    Recharge is active only when ``rcfg.enable_recharge`` **and** a battery is enabled; repair is
    active whenever ``rcfg.enable_repair``. Returns ``course`` unchanged when neither axis is
    active, and ``None`` when no eligible anchor admits an off-gate pad position → the caller
    reject-resamples. All placement is deterministic and **zero-RNG**.
    """
    recharge_active = rcfg.enable_recharge and battery is not None and battery.enabled
    repair_active = rcfg.enable_repair
    if not (recharge_active or repair_active):
        return course
    if recharge_active and not repair_active:
        return _place_single_recharge_pad(course, rcfg, battery)  # type: ignore[arg-type]
    if repair_active and not recharge_active:
        return _place_single_repair_pad(course, rcfg)

    # Both axes active: distinct off-gate anchors when possible, else a single dual-purpose pad.
    eligible = _eligible_gate_indices(course, rcfg)
    if not eligible:
        return None
    rk = _energy_midpoint_gate(course, rcfg, battery, eligible)  # type: ignore[arg-type]
    recharge_xy = _offset_pad_off_gates(course, rcfg, rk)
    if recharge_xy is None:  # defensive — rk ∈ eligible, so an offset always exists
        return None

    # Try to place the repair pad off a DISTINCT anchor, ≥ R_pad from the recharge pad too.
    repair_xy: np.ndarray | None = None
    repair_candidates = [i for i in eligible if i != rk]
    if repair_candidates:
        pk = _index_midpoint_gate(repair_candidates, len(course.gates))
        repair_xy = _offset_pad_off_gates(course, rcfg, pk, avoid=(recharge_xy,))
        if repair_xy is None:
            for i in repair_candidates:  # any other distinct anchor with a feasible offset
                repair_xy = _offset_pad_off_gates(course, rcfg, i, avoid=(recharge_xy,))
                if repair_xy is not None:
                    break

    if repair_xy is None:
        # Only one eligible anchor (e.g. a 1-gate course / the fallback), or no distinct feasible
        # repair position: co-locate into a single dual-purpose pad at the recharge position. Still
        # "one of each" (rechargeable AND repairable) and avoids one pad shadowing the other.
        dual = PadSpec(
            center=(float(recharge_xy[0]), float(recharge_xy[1])),
            radius=max(float(rcfg.recharge_pad_radius), float(rcfg.repair_pad_radius)),
            rechargeable=True,
            repairable=True,
        )
        return _with_pads(course, [dual])

    recharge_pad = PadSpec(
        center=(float(recharge_xy[0]), float(recharge_xy[1])),
        radius=float(rcfg.recharge_pad_radius),
        rechargeable=True,
    )
    repair_pad = PadSpec(
        center=(float(repair_xy[0]), float(repair_xy[1])),
        radius=float(rcfg.repair_pad_radius),
        repairable=True,
    )
    return _with_pads(course, [recharge_pad, repair_pad])


def _fallback_course(
    n: int,
    base_course: CourseConfig,
    rcfg: RandomizationConfig,
    battery: BatteryConfig | None = None,
) -> CourseConfig:
    """Deterministic, **zero-RNG** solvable N-gate course (UC-09 AC5 exhaustion fallback).

    Places all gates on the centreline (y=0) at the corridor mid-height, evenly spaced along
    +x by a safe gap, with a fixed safe aperture, finish beyond the last gate — solvable by
    construction for every N in ``[1, 10]``. Draws nothing off any RNG, so it never perturbs
    the seeded stream. Guarded by an assertion against the same solvability predicate.

    ``battery`` (UC-18; UC-24 single-pad model): when the recharge and/or repair axes are active the
    SAME zero-RNG :func:`_place_service_pads` runs on the fallback before returning, so even the
    exhaustion fallback carries its single recharge / repair feature pad(s). The fallback is
    **obstacle-free**, so every gate is an eligible anchor and the placer never returns ``None``
    (no deadlock). For N=1 with both axes active the two co-locate into a single dual-purpose pad.
    The evenly-spaced mid-altitude fallback covers for any battery whose one-charge reach ≥ a single
    gate gap plus its vertical legs; if a cranked-constrained battery makes the single recharge pad
    insufficient the battery-aware solvability assert below fires — a documented, fail-loud
    PRECONDITION (unreachable under the shipped default battery), not a silent hole.
    """
    n = max(1, int(n))
    floor_z = base_course.floor_z
    ceiling_z = base_course.ceiling_z
    mid_z = 0.5 * (floor_z + ceiling_z)

    start_x, start_y, start_z = 0.0, 0.0, mid_z

    # Safe aperture strictly above the flyable minimum, and gaps strictly above the minimums.
    aperture = rcfg.aperture_min + 0.2
    first_gap = max(rcfg.min_start_gate_gap, aperture + rcfg.z_margin) + 0.5
    gate_gap = max(rcfg.min_gate_spacing, 1.0) + 0.5

    gates: list[GateSpec] = []
    x = start_x + first_gap
    for i in range(n):
        if i > 0:
            x += gate_gap
        gates.append(GateSpec(center=(x, 0.0, mid_z), aperture=aperture))

    finish_x = x + max(rcfg.min_gate_finish_gap, 1.0) + 0.5

    course = CourseConfig(
        start_position=(start_x, start_y, start_z),
        gates=tuple(gates),
        finish_x=finish_x,
        floor_z=floor_z,
        ceiling_z=ceiling_z,
    )
    # UC-24: place the single recharge / repair feature pad(s) on the fallback too (same zero-RNG
    # placer). The fallback is obstacle-free ⇒ every gate is eligible ⇒ the placer never returns
    # ``None`` here.
    recharge_active = rcfg.enable_recharge and battery is not None and battery.enabled
    if recharge_active or rcfg.enable_repair:
        placed = _place_service_pads(course, rcfg, battery)
        if placed is not None:
            course = placed
    assert is_course_solvable(course, rcfg, battery=battery if recharge_active else None), (
        "fallback course must be solvable by construction"
    )
    return course


def sample_dynamics(
    rng: np.random.Generator,
    rcfg: RandomizationConfig,
    base: DynamicsParams,
) -> DynamicsParams:
    """Sample a :class:`DynamicsParams` by scaling ``base`` with ``rng`` draws (AC5, AC4).

    Draw order is pinned (mass, drag, thrust, rate, latency) so a seed reproduces the
    dynamics stream. Mass and thrust are **independent** knobs: the adapter computes
    ``thrust_acc = throttle * max_thrust / mass``, so scaling mass alone genuinely changes
    the trajectory (it is not cancelled by a coupled thrust rescale).
    """
    mass = base.mass * float(rng.uniform(*rcfg.mass_factor_range))
    drag = base.drag * float(rng.uniform(*rcfg.drag_factor_range))
    max_thrust = base.max_thrust * float(rng.uniform(*rcfg.thrust_factor_range))
    max_body_rate = base.max_body_rate * float(rng.uniform(*rcfg.rate_factor_range))
    lo, hi = rcfg.latency_steps_range
    latency_steps = int(rng.integers(int(lo), int(hi) + 1))
    return DynamicsParams(
        mass=mass,
        drag=drag,
        max_body_rate=max_body_rate,
        max_thrust=max_thrust,
        latency_steps=latency_steps,
    )
