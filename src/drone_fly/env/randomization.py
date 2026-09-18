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

    # -- obstacle solvability (UC-15 AC3) -----------------------------------------------
    # A course with pillars is flyable iff (a) no start / gate centre / finish sits inside a
    # pillar, and (b) the start→gates→finish polyline clears every pillar horizontally by at
    # least ``radius + obstacle_clearance`` wherever the segment's z-range overlaps the pillar
    # band. (b) constructively guarantees ≥1 collision-free path exists. Pillars are checked
    # regardless of ``enable_obstacles`` — the flag governs *sampling*, not what makes a
    # given course (which already carries obstacles) solvable.
    obstacles = getattr(course, "obstacles", ())
    if obstacles:
        floor_z = course.floor_z
        last = gates[-1]
        finish_pt = np.asarray([course.finish_x, last.center[1], last.center[2]], dtype=np.float64)
        anchors = [course.start, *(g.position for g in gates), finish_pt]
        for obstacle in obstacles:
            # (a) no anchor may lie inside the pillar.
            for pt in anchors:
                if point_in_cylinder(pt, obstacle, floor_z):
                    return False
            # (b) polyline clearance within the pillar's z-band.
            axis = np.asarray(
                [float(obstacle.center[0]), float(obstacle.center[1])], dtype=np.float64
            )
            clearance = float(obstacle.radius) + rcfg.obstacle_clearance
            top = floor_z + float(obstacle.height)
            for a, b in zip(anchors[:-1], anchors[1:], strict=True):
                z_lo = min(float(a[2]), float(b[2]))
                z_hi = max(float(a[2]), float(b[2]))
                if z_hi < floor_z or z_lo > top:
                    continue  # segment passes entirely above/below the pillar band
                if _segment_axis_distance_2d(a, b, axis) < clearance:
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


def _floor_point(gate: GateSpec, floor_z: float) -> np.ndarray:
    """The floor-anchored 3D waypoint under ``gate``: ``(gate.x, gate.y, floor_z)`` (UC-18).

    A recharge pad sits on the floor beneath a gate, so descending to it and climbing back out
    are genuine vertical legs of the flight path — modelling the pad as this floor waypoint makes
    the energy model count those legs by construction.
    """
    return np.asarray([float(gate.center[0]), float(gate.center[1]), floor_z], dtype=np.float64)


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


def _recharge_gate_indices(course: CourseConfig, rechargeable) -> list[int]:
    """Ordered gate indices that carry a rechargeable pad under their (x, y) column (UC-18).

    A gate is a recharge point iff some rechargeable pad's centre lies within that pad's own
    ``radius`` of the gate's ``(x, y)`` — exactly how :func:`_place_recharge_pads` places pads
    (at gate columns). Pads not under any gate contribute no recharge (the covering check then
    treats the intervening path as un-refilled — the safe, conservative direction).
    """
    idxs: list[int] = []
    for idx, gate in enumerate(course.gates):
        gx, gy = float(gate.center[0]), float(gate.center[1])
        for pad in rechargeable:
            px, py = float(pad.center[0]), float(pad.center[1])
            if float(np.hypot(gx - px, gy - py)) <= float(pad.radius):
                idxs.append(idx)
                break
    return idxs


def _recharge_covering_valid(
    course: CourseConfig,
    rcfg: RandomizationConfig,
    battery: BatteryConfig,
    rechargeable,
) -> bool:
    """Re-verify every induced recharge sub-path fits one charge (UC-18 AC4 covering check).

    Reconstructs the flight as: ``start → gates… → floor(first recharge gate)``, then
    ``floor(pad) → gates… → floor(next recharge gate)`` for each subsequent pad, then
    ``floor(last pad) → gates… → finish``. Each leg is charged from full (start, or a fully-
    refilled pad) and must cost ≤ one charge. Uses the same :func:`_path_energy` model as the
    placer, so a covering the placer produced always re-verifies (a tiny epsilon absorbs float
    round-off).
    """
    gates = course.gates
    n = len(gates)
    floor_z = course.floor_z
    recharge_idxs = _recharge_gate_indices(course, rechargeable)
    last = gates[-1]
    finish_pt = np.asarray([course.finish_x, last.center[1], last.center[2]], dtype=np.float64)

    legs: list[list[np.ndarray]] = []
    charged_point = course.start
    cursor = 0  # first gate not yet consumed by a prior leg
    for k in recharge_idxs:
        pts = [charged_point, *(gates[j].position for j in range(cursor, k + 1))]
        pts.append(_floor_point(gates[k], floor_z))
        legs.append(pts)
        charged_point = _floor_point(gates[k], floor_z)
        cursor = k + 1
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

    ``battery`` (UC-18): when ``rcfg.enable_recharge`` **and** ``battery.enabled``, each solvable
    candidate is run through :func:`_place_recharge_pads` (a pure, **zero-RNG** greedy cover): a
    course that fits one charge is returned unchanged (no pads); an energy-constrained one gets
    recharge pads so the finish is reachable *with* landings, then is re-verified by the
    battery-aware :func:`is_course_solvable`. A candidate whose next gate is unreachable even on a
    full charge (no valid cover) is rejected and resampled. When the recharge axis is inactive the
    placer is never called and ``is_course_solvable`` is invoked with ``battery=None``, so the RNG
    stream and result are byte-identical to UC-15/16/17 (AC5).
    """
    recharge_active = rcfg.enable_recharge and battery is not None and battery.enabled
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
        # draw and never perturbs the gate/dynamics stream). Placement is biased off the corridor
        # so the draws actually clear the solvability guard; the whole candidate (gates + pillars)
        # is then reject-resampled together.
        obstacles: tuple[ObstacleSpec, ...] = ()
        if rcfg.enable_obstacles:
            obstacles = _sample_obstacles(rng, rcfg, tuple(gates))

        candidate = CourseConfig(
            start_position=(start_x, start_y, start_z),
            gates=tuple(gates),
            finish_x=finish_x,
            floor_z=base_course.floor_z,
            ceiling_z=base_course.ceiling_z,
            obstacles=obstacles,
        )
        # UC-18: place recharge pads on an energy-constrained candidate (zero-RNG, so the stream
        # is untouched). ``None`` means no valid cover (next gate unreachable on a full charge, or
        # every reachable anchor's descend column clips a pillar) → reject-resample.
        if recharge_active:
            placed = _place_recharge_pads(candidate, rcfg, battery)
            if placed is None:
                continue
            candidate = placed
        if is_course_solvable(candidate, rcfg, battery=battery if recharge_active else None):
            return candidate

    # Hard cap hit: deterministic, zero-RNG fallback (solvable-by-construction for this N).
    return _fallback_course(n, base_course, rcfg, battery=battery)


def _sample_obstacles(
    rng: np.random.Generator,
    rcfg: RandomizationConfig,
    gates: tuple[GateSpec, ...],
) -> tuple[ObstacleSpec, ...]:
    """Draw pillars biased **off** the corridor (UC-15 AC3); reject-resampling handles misses.

    Draw order per pillar is pinned: count first, then for each pillar its anchor gate, lateral
    side, |y|-offset, radius, and height. Each pillar is placed at a random gate's ``(x, y)``
    shifted laterally by ``obstacle_lateral_offset_range`` (well outside the gate corridor), so
    most draws clear the solvability guard; the ones that don't are rejected with the whole
    course by :func:`sample_course`. Consumes a fixed number of draws given the count, so a seed
    stays reproducible.
    """
    lo, hi = rcfg.obstacle_count_range
    lo = max(0, int(lo))
    hi = max(lo, int(hi))
    count = int(rng.integers(lo, hi + 1))

    obstacles: list[ObstacleSpec] = []
    for _ in range(count):
        gate = gates[int(rng.integers(0, len(gates)))]
        gx, gy = float(gate.center[0]), float(gate.center[1])
        side = 1.0 if rng.random() < 0.5 else -1.0
        offset = float(rng.uniform(*rcfg.obstacle_lateral_offset_range))
        radius = float(rng.uniform(*rcfg.obstacle_radius_range))
        height = float(rng.uniform(*rcfg.obstacle_height_range))
        obstacles.append(
            ObstacleSpec(center=(gx, gy + side * offset), radius=radius, height=height)
        )
    return tuple(obstacles)


def _descend_column_clear(
    gate: GateSpec,
    obstacles,
    rcfg: RandomizationConfig,
) -> bool:
    """Return ``True`` iff the floor→gate descend column at ``(gate.x, gate.y)`` clears pillars.

    A recharge pad forces the drone to descend a vertical column at the gate's ``(x, y)`` from
    ``gate.z`` to the floor and climb back out. Since every pillar is floor-anchored, that column
    overlaps a pillar's z-band whenever it comes within ``obstacle.radius + obstacle_clearance``
    horizontally — which would make the recharge detour clip the pillar. Rejecting such an anchor
    keeps the recharge×obstacle composition safe **by construction** (UC-18 cross-axis decision).
    """
    for obstacle in obstacles:
        ox, oy = float(obstacle.center[0]), float(obstacle.center[1])
        horiz = float(np.hypot(float(gate.center[0]) - ox, float(gate.center[1]) - oy))
        if horiz < float(obstacle.radius) + rcfg.obstacle_clearance:
            return False
    return True


def _place_recharge_pads(
    course: CourseConfig,
    rcfg: RandomizationConfig,
    battery: BatteryConfig,
) -> CourseConfig | None:
    """Greedy, **zero-RNG** recharge-pad cover for an energy-constrained course (UC-18 AC3/AC4).

    If the full reference 3D path fits one charge, the course is returned **unchanged** (no pads —
    a non-constrained course). Otherwise a greedy interval cover walks the gates: from the current
    charged point (the start, or the last placed pad — both at full charge), it advances to the
    **furthest** gate whose cumulative sub-path energy (including the descend-to-floor at that
    gate) fits one charge, places a recharge pad there, resets the charged point to that pad, and
    repeats until the remaining path to the finish (including the climb-out from the last pad)
    fits one charge. Cumulative sub-path energy is monotonic non-decreasing in the anchor index,
    so "the furthest anchor within budget" is a valid per-leg guarantee.

    Two ways to fail (return ``None`` → the caller reject-resamples): the immediate next gate is
    unreachable even on a full charge, or every reachable anchor's descend column clips a pillar
    (cross-axis safety, :func:`_descend_column_clear`) so no pad can be placed. The pad radius is
    ``rcfg.recharge_pad_radius`` and every placed pad is ``rechargeable=True``; existing course
    pads (if any) are preserved and the recharge pads appended after them.
    """
    gates = course.gates
    n = len(gates)
    floor_z = course.floor_z
    last = gates[-1]
    finish_pt = np.asarray([course.finish_x, last.center[1], last.center[2]], dtype=np.float64)

    # Non-constrained: fits one charge as-is → no recharge pads (byte-identical to a plain course).
    if _path_energy(_reference_path(course), rcfg, battery) <= _BATTERY_BUDGET:
        return course

    placed_gate_idxs: list[int] = []
    charged_point = course.start
    cursor = 0  # first gate not yet covered by the current charge
    while True:
        # Done? remaining path (charged_point → remaining gates → finish, incl. climb-out) fits.
        remaining = [charged_point, *(gates[j].position for j in range(cursor, n)), finish_pt]
        if _path_energy(remaining, rcfg, battery) <= _BATTERY_BUDGET:
            break

        # Furthest gate k in [cursor, n-1] whose sub-path (charged → gates[cursor..k] → floor(gk))
        # fits one charge. Energy is monotonic in k, so stop at the first over-budget k.
        max_k: int | None = None
        for k in range(cursor, n):
            sub = [
                charged_point,
                *(gates[j].position for j in range(cursor, k + 1)),
                _floor_point(gates[k], floor_z),
            ]
            if _path_energy(sub, rcfg, battery) <= _BATTERY_BUDGET:
                max_k = k
            else:
                break
        if max_k is None:
            return None  # even the immediate next gate is unreachable on a full charge

        # Among the reachable anchors, pick the furthest whose descend column clears all pillars.
        chosen: int | None = None
        for k in range(max_k, cursor - 1, -1):
            if _descend_column_clear(gates[k], course.obstacles, rcfg):
                chosen = k
                break
        if chosen is None:
            return None  # no reachable anchor is obstacle-clear → cannot cover this course

        placed_gate_idxs.append(chosen)
        charged_point = _floor_point(gates[chosen], floor_z)
        cursor = chosen + 1

    if not placed_gate_idxs:
        return course

    recharge_pads = tuple(
        PadSpec(
            center=(float(gates[k].center[0]), float(gates[k].center[1])),
            radius=float(rcfg.recharge_pad_radius),
            rechargeable=True,
        )
        for k in placed_gate_idxs
    )
    return CourseConfig(
        start_position=course.start_position,
        gates=course.gates,
        finish_x=course.finish_x,
        floor_z=course.floor_z,
        ceiling_z=course.ceiling_z,
        obstacles=course.obstacles,
        pads=(*course.pads, *recharge_pads),
    )


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

    ``battery`` (UC-18): when the recharge axis is active the SAME zero-RNG
    :func:`_place_recharge_pads` runs on the fallback before returning, so even the exhaustion
    fallback honours the AC4 covering guarantee (closing the hole where a fallback could emit an
    energy-constrained pad-less course). The evenly-spaced mid-altitude fallback covers for any
    battery whose one-charge reach ≥ a single gate gap plus its vertical legs; if the placer
    cannot cover it (``None``) the battery-aware solvability assert below fires — a documented,
    fail-loud PRECONDITION, not a silent hole.
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
    # UC-18: honour the recharge covering on the fallback too (same zero-RNG placer).
    recharge_active = rcfg.enable_recharge and battery is not None and battery.enabled
    if recharge_active:
        placed = _place_recharge_pads(course, rcfg, battery)
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
