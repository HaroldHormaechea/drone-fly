"""UC-08 AC3/AC4/AC5 — the pure domain-randomization samplers.

Covers :mod:`drone_fly.env.randomization` (``sample_course`` / ``sample_dynamics`` /
``is_course_solvable``) and the load-bearing physical consequence of the dynamics knobs:

* **AC3 (solvability + bounds)** — thousands of sampled courses are *all* flyable
  (``is_course_solvable``) and sit inside every configured bound, with the start height
  **strictly inside** ``(floor_z + z_margin, ceiling_z - z_margin)`` (the strictness is
  load-bearing: :meth:`SimpleDroneAdapter.reset` flags a step-0 collision if the spawn
  touches a bound).
* **AC3 (reject-then-clamp, no hang)** — with a pathological, empty-feasible range set the
  sampler exhausts its attempt cap and **clamps to the base course** (guaranteed solvable),
  returning promptly rather than looping forever.
* **AC4 (seeded reproducibility)** — a given seed reproduces the same *sequence* of courses
  (and dynamics); a different seed produces a different sequence.
* **AC5 (dynamics bounds + pinned order + per-knob effect)** — sampled dynamics stay inside
  the configured factor ranges, the draw order is pinned (same seed → identical params), and
  **mass alone genuinely perturbs the trajectory** (the anti-cancellation guard: the adapter
  must not recompute ``max_thrust`` from instance mass).

Pure + hermetic: drives the samplers off explicit ``numpy`` generators; the one physical
check builds two :class:`SimpleDroneAdapter` instances directly (no env / sim / network).
"""

from __future__ import annotations

import numpy as np

from drone_fly.adapter.simple import SimpleDroneAdapter
from drone_fly.env.config import CourseConfig, DynamicsParams, RandomizationConfig
from drone_fly.env.randomization import (
    is_course_solvable,
    sample_course,
    sample_dynamics,
)


# --- AC3: every sampled course is solvable and within bounds -------------------------
def test_sampled_courses_are_all_solvable_and_within_bounds() -> None:
    """2000+ draws off one RNG are each flyable and inside every configured bound (AC3)."""
    rcfg = RandomizationConfig(enable_course=True)
    base = CourseConfig()
    rng = np.random.default_rng(2024)

    lo_z = base.floor_z + rcfg.z_margin
    hi_z = base.ceiling_z - rcfg.z_margin

    n = 2500
    for _ in range(n):
        course = sample_course(rng, rcfg, base)
        # The sampler's own contract: never returns an unsolvable course.
        assert is_course_solvable(course, rcfg)

        sx, sy, sz = course.start_position
        # start_z STRICTLY inside the vertical safety corridor (load-bearing: a spawn on
        # the bound flags a step-0 collision in SimpleDroneAdapter.reset).
        assert lo_z < sz < hi_z
        # gate-centre z (and therefore the finish z, which inherits it) also strictly inside.
        assert lo_z < course.gate_center_z < hi_z

        # gate strictly between start and finish along +x, with the configured min gaps.
        assert course.gate_x - sx >= rcfg.min_start_gate_gap
        assert course.finish_x - course.gate_x >= rcfg.min_gate_finish_gap

        # the drone must fit through the aperture.
        assert course.gate_aperture >= rcfg.aperture_min

        # lateral bounds on start-y and gate-centre-y.
        assert abs(sy) <= rcfg.lateral_bound
        assert abs(course.gate_center_y) <= rcfg.lateral_bound

        # arena bounds are NOT randomized — copied verbatim from the base course.
        assert course.floor_z == base.floor_z
        assert course.ceiling_z == base.ceiling_z


def test_sampled_courses_actually_vary() -> None:
    """Sampling is not degenerate: many draws yield many distinct courses (AC2/AC3)."""
    rcfg = RandomizationConfig(enable_course=True)
    base = CourseConfig()
    rng = np.random.default_rng(0)
    starts = {tuple(np.round(sample_course(rng, rcfg, base).start_position, 6)) for _ in range(200)}
    assert len(starts) > 150  # overwhelmingly distinct spawns


# --- AC3: reject-then-clamp on an empty-feasible range set (no infinite loop) --------
def test_pathological_ranges_clamp_to_base_course_without_hanging() -> None:
    """Empty-feasible ranges exhaust the cap and clamp to the base course, promptly (AC3)."""
    # gate_x is pinned behind the start, so gate_x - start_x is always negative < the 1.0
    # minimum gap: EVERY draw is rejected, forcing the attempt-cap clamp fallback.
    rcfg = RandomizationConfig(
        enable_course=True,
        start_x_range=(0.5, 0.5),
        gate_x_range=(0.0, 0.0),
        max_resample_attempts=25,
    )
    base = CourseConfig()
    rng = np.random.default_rng(1)

    # Runs to completion (the attempt cap guarantees termination) and returns the base.
    result = sample_course(rng, rcfg, base)
    assert result == base  # frozen-dataclass equality: the exact base course
    assert is_course_solvable(result, rcfg) is True  # the fallback is guaranteed flyable


def test_is_course_solvable_flags_each_degenerate_mode() -> None:
    """The guard rejects each documented degenerate case (AC3)."""
    rcfg = RandomizationConfig()
    base = CourseConfig()
    assert is_course_solvable(base, rcfg) is True

    # gate behind / too close to the start.
    assert not is_course_solvable(CourseConfig(start_position=(2.5, 0.0, 1.0), gate_x=3.0), rcfg)
    # finish inside / too close to the gate.
    assert not is_course_solvable(CourseConfig(gate_x=3.0, finish_x=3.5), rcfg)
    # aperture smaller than the drone.
    assert not is_course_solvable(CourseConfig(gate_aperture=rcfg.aperture_min - 0.01), rcfg)
    # start height on the floor corridor edge (must be STRICTLY inside).
    assert not is_course_solvable(CourseConfig(start_position=(0.0, 0.0, rcfg.z_margin)), rcfg)
    # gate centre above the ceiling corridor.
    assert not is_course_solvable(CourseConfig(gate_center_z=2.5 - rcfg.z_margin + 0.01), rcfg)
    # start pushed outside the lateral bound.
    assert not is_course_solvable(
        CourseConfig(start_position=(0.0, rcfg.lateral_bound + 0.1, 1.0)), rcfg
    )


# --- AC4: seeded reproducibility of the course stream -------------------------------
def test_same_seed_reproduces_identical_course_sequence() -> None:
    """A given seed reproduces the same *sequence* of sampled courses (AC4)."""
    rcfg = RandomizationConfig(enable_course=True)
    base = CourseConfig()

    def sequence(seed: int, k: int = 10) -> list[CourseConfig]:
        rng = np.random.default_rng(seed)
        return [sample_course(rng, rcfg, base) for _ in range(k)]

    assert sequence(99) == sequence(99)


def test_different_seed_produces_different_course_sequence() -> None:
    rcfg = RandomizationConfig(enable_course=True)
    base = CourseConfig()

    def sequence(seed: int, k: int = 10) -> list[CourseConfig]:
        rng = np.random.default_rng(seed)
        return [sample_course(rng, rcfg, base) for _ in range(k)]

    assert sequence(1) != sequence(2)


# --- AC5: dynamics bounds, pinned draw order, per-knob effect ------------------------
def test_sampled_dynamics_within_configured_factor_ranges() -> None:
    """Every sampled DynamicsParams sits inside the configured multiplicative ranges (AC5)."""
    rcfg = RandomizationConfig(enable_dynamics=True)
    base = DynamicsParams()
    rng = np.random.default_rng(7)
    for _ in range(2000):
        d = sample_dynamics(rng, rcfg, base)
        assert (
            base.mass * rcfg.mass_factor_range[0] <= d.mass <= base.mass * rcfg.mass_factor_range[1]
        )
        assert (
            base.drag * rcfg.drag_factor_range[0] <= d.drag <= base.drag * rcfg.drag_factor_range[1]
        )
        assert (
            base.max_thrust * rcfg.thrust_factor_range[0]
            <= d.max_thrust
            <= base.max_thrust * rcfg.thrust_factor_range[1]
        )
        assert (
            base.max_body_rate * rcfg.rate_factor_range[0]
            <= d.max_body_rate
            <= base.max_body_rate * rcfg.rate_factor_range[1]
        )
        lo, hi = rcfg.latency_steps_range
        assert lo <= d.latency_steps <= hi
        assert isinstance(d.latency_steps, int)


def test_dynamics_draw_order_is_pinned_same_seed_identical() -> None:
    """The dynamics draw order is pinned: the same seed reproduces identical params (AC4)."""
    rcfg = RandomizationConfig(enable_dynamics=True)
    base = DynamicsParams()
    a = sample_dynamics(np.random.default_rng(3), rcfg, base)
    b = sample_dynamics(np.random.default_rng(3), rcfg, base)
    assert a == b
    c = sample_dynamics(np.random.default_rng(4), rcfg, base)
    assert a != c


def test_disabled_dynamics_axis_is_never_drawn_here_defaults_are_base() -> None:
    """A DynamicsParams() (the disabled-axis value) equals today's constants exactly (AC7).

    The env hands ``DynamicsParams()`` semantics (or ``None``) when dynamics is off, so the
    adapter defaults must equal the module constants for byte-identity.
    """
    from drone_fly.adapter.simple import (
        BASE_LINEAR_DRAG,
        BASE_MASS,
        BASE_MAX_BODY_RATE,
        BASE_MAX_THRUST,
    )

    d = DynamicsParams()
    assert d.mass == BASE_MASS
    assert d.drag == BASE_LINEAR_DRAG
    assert d.max_body_rate == BASE_MAX_BODY_RATE
    assert d.max_thrust == BASE_MAX_THRUST
    assert d.latency_steps == 0


def _adapter() -> SimpleDroneAdapter:
    return SimpleDroneAdapter(np.array([0.0, 0.0, 1.0]), floor_z=0.0, ceiling_z=2.5, dt=0.05)


def _rollout_positions(adapter: SimpleDroneAdapter, actions, seed: int = 5) -> np.ndarray:
    adapter.reset(seed=seed)
    return np.array([adapter.step(a).position.copy() for a in actions])


def test_mass_factor_alone_perturbs_the_trajectory() -> None:
    """Scaling mass ALONE genuinely changes the trajectory — the anti-cancellation guard (AC5).

    ``thrust_acc = throttle * max_thrust / mass``; if the adapter (wrongly) recomputed
    ``max_thrust`` from the instance mass, mass would cancel and this would fail.
    """
    rng = np.random.default_rng(0)
    actions = [rng.uniform([0.2, -0.5, -0.5, -0.5], [1.0, 0.5, 0.5, 0.5]) for _ in range(50)]

    base = DynamicsParams()  # mass 1.0
    heavy = DynamicsParams(mass=1.2)  # ONLY mass differs; max_thrust/drag/rate identical

    a, b = _adapter(), _adapter()
    a.reconfigure(dynamics=base)
    b.reconfigure(dynamics=heavy)
    pa = _rollout_positions(a, actions)
    pb = _rollout_positions(b, actions)

    assert not np.array_equal(pa, pb), "mass_factor alone must perturb the trajectory"


def test_drag_factor_alone_perturbs_the_trajectory() -> None:
    """Drag is an independent knob too: scaling it alone changes the trajectory (AC5)."""
    rng = np.random.default_rng(1)
    actions = [rng.uniform([0.2, -0.5, -0.5, -0.5], [1.0, 0.5, 0.5, 0.5]) for _ in range(50)]

    a, b = _adapter(), _adapter()
    a.reconfigure(dynamics=DynamicsParams(drag=0.15))
    b.reconfigure(dynamics=DynamicsParams(drag=0.30))
    assert not np.array_equal(_rollout_positions(a, actions), _rollout_positions(b, actions))
