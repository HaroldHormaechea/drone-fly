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
    CourseConfig,
    DynamicsParams,
    GateSpec,
    RandomizationConfig,
)


def is_course_solvable(course: CourseConfig, rcfg: RandomizationConfig) -> bool:
    """Return ``True`` iff ``course`` is flyable under ``rcfg``'s solvability bounds (AC5)."""
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

    return True


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
) -> CourseConfig:
    """Sample a solvable N-gate :class:`CourseConfig` off ``rng`` (UC-09 AC5, AC6).

    Arena bounds (``floor_z`` / ``ceiling_z``) are **not** randomized — they are copied from
    ``base_course``. Draw order is pinned per attempt: ``num_gates`` first, then the start,
    then a forward incremental-x walk over the gates (per-gate gap, y, z, aperture), then the
    finish gap. RNG consumption is fixed given N, so a seed is reproducible. Solvability is
    tested and failures resample up to ``rcfg.max_resample_attempts``; on exhaustion a
    deterministic zero-RNG :func:`_fallback_course` (solvable-by-construction) is returned.
    """
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

        candidate = CourseConfig(
            start_position=(start_x, start_y, start_z),
            gates=tuple(gates),
            finish_x=finish_x,
            floor_z=base_course.floor_z,
            ceiling_z=base_course.ceiling_z,
        )
        if is_course_solvable(candidate, rcfg):
            return candidate

    # Hard cap hit: deterministic, zero-RNG fallback (solvable-by-construction for this N).
    return _fallback_course(n, base_course, rcfg)


def _fallback_course(
    n: int,
    base_course: CourseConfig,
    rcfg: RandomizationConfig,
) -> CourseConfig:
    """Deterministic, **zero-RNG** solvable N-gate course (UC-09 AC5 exhaustion fallback).

    Places all gates on the centreline (y=0) at the corridor mid-height, evenly spaced along
    +x by a safe gap, with a fixed safe aperture, finish beyond the last gate — solvable by
    construction for every N in ``[1, 10]``. Draws nothing off any RNG, so it never perturbs
    the seeded stream. Guarded by an assertion against the same solvability predicate.
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
    assert is_course_solvable(course, rcfg), "fallback course must be solvable by construction"
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
