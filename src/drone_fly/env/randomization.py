"""Pure per-episode domain-randomization samplers (UC-08 AC3, AC4).

This module is **pure**: it imports only :mod:`numpy` and the config dataclasses — no env,
no adapter, no sim, no torch — so it is trivially unit-testable and can never pull the
heavy stack in. Both samplers draw off a caller-supplied ``numpy`` ``Generator`` (the env's
seeded ``self.np_random``), so a given seed reproduces the same course *and* dynamics
stream (AC4). The env pins the draw order (course then dynamics) and never draws for a
disabled axis, so toggling one axis never shifts the other's stream.

Course solvability (AC3)
------------------------
A sampled course is *flyable* iff, along the +x course axis:

* the gate sits between the start and the finish with the configured minimum gaps
  (``gate_x - start_x >= min_start_gate_gap`` and ``finish_x - gate_x >= min_gate_finish_gap``);
* the gate aperture is at least ``aperture_min`` (the drone must fit through);
* the start height, the gate-centre height, and therefore the finish height (the finish
  target inherits the gate-centre z, see :func:`drone_fly.env.geometry.target_position`)
  all sit **strictly inside** ``(floor_z + z_margin, ceiling_z - z_margin)`` — start_z
  strictness is load-bearing because :meth:`SimpleDroneAdapter.reset` flags a step-0
  collision if the spawn touches a bound;
* the start-y and gate-centre-y stay within ``+/- lateral_bound``.

Degenerate draws are **rejected and resampled** up to ``max_resample_attempts``; if the cap
is hit the sampler **clamps to the base course** (the fixed, guaranteed-solvable
:class:`CourseConfig`), so sampling is deterministic and can never loop forever.
"""

from __future__ import annotations

import numpy as np

from drone_fly.env.config import CourseConfig, DynamicsParams, RandomizationConfig


def is_course_solvable(course: CourseConfig, rcfg: RandomizationConfig) -> bool:
    """Return ``True`` iff ``course`` is flyable under ``rcfg``'s solvability bounds (AC3)."""
    start = course.start_position
    start_x, start_y, start_z = float(start[0]), float(start[1]), float(start[2])

    # Gate strictly between start and finish along +x, with the configured minimum gaps.
    if course.gate_x - start_x < rcfg.min_start_gate_gap:
        return False
    if course.finish_x - course.gate_x < rcfg.min_gate_finish_gap:
        return False

    # The drone must fit through the aperture.
    if course.gate_aperture < rcfg.aperture_min:
        return False

    # All waypoint heights strictly inside the vertical safety corridor. The finish target
    # inherits the gate-centre z, so checking start_z and gate_center_z covers all three.
    lo = course.floor_z + rcfg.z_margin
    hi = course.ceiling_z - rcfg.z_margin
    if not (lo < start_z < hi):
        return False
    if not (lo < course.gate_center_z < hi):
        return False

    # Lateral bounds for the start and the gate centre.
    if abs(start_y) > rcfg.lateral_bound:
        return False
    if abs(course.gate_center_y) > rcfg.lateral_bound:
        return False

    return True


def sample_course(
    rng: np.random.Generator,
    rcfg: RandomizationConfig,
    base_course: CourseConfig,
) -> CourseConfig:
    """Sample a solvable :class:`CourseConfig` off ``rng`` (AC2, AC3, AC4).

    Arena bounds (``floor_z`` / ``ceiling_z``) are **not** randomized — they are copied from
    ``base_course``. Each attempt draws a fixed set of parameters (so the RNG consumption is
    deterministic given the number of attempts), tests solvability, and resamples on
    failure up to ``rcfg.max_resample_attempts``. On exhaustion it **clamps to
    ``base_course``** (guaranteed solvable) rather than looping forever.
    """
    for _ in range(max(int(rcfg.max_resample_attempts), 1)):
        start_x = float(rng.uniform(*rcfg.start_x_range))
        start_y = float(rng.uniform(*rcfg.start_y_range))
        start_z = float(rng.uniform(*rcfg.start_z_range))
        gate_x = float(rng.uniform(*rcfg.gate_x_range))
        gate_center_y = float(rng.uniform(*rcfg.gate_center_y_range))
        gate_center_z = float(rng.uniform(*rcfg.gate_center_z_range))
        gate_aperture = float(rng.uniform(*rcfg.gate_aperture_range))
        finish_x = float(rng.uniform(*rcfg.finish_x_range))

        candidate = CourseConfig(
            start_position=(start_x, start_y, start_z),
            gate_x=gate_x,
            gate_center_y=gate_center_y,
            gate_center_z=gate_center_z,
            gate_aperture=gate_aperture,
            finish_x=finish_x,
            floor_z=base_course.floor_z,
            ceiling_z=base_course.ceiling_z,
        )
        if is_course_solvable(candidate, rcfg):
            return candidate

    # Hard cap hit: clamp to the base course (guaranteed solvable, deterministic).
    return base_course


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
