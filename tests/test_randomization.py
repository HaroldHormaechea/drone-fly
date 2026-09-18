"""UC-08 + UC-09 AC5/AC6 — the pure domain-randomization samplers (N-gate courses).

Covers :mod:`drone_fly.env.randomization` (``sample_course`` / ``sample_dynamics`` /
``is_course_solvable`` / ``_fallback_course``) for the UC-09 N-gate course model:

* **UC-09 AC5 (num_gates)** — the sampler draws ``num_gates`` uniformly across the
  configured integer range (default ``[1, 10]``); the full range is exercised.
* **UC-09 AC5 (solvability + bounds)** — thousands of sampled courses are *all* flyable
  (``is_course_solvable``) for every N: every gate inside the floor/ceiling corridor and
  lateral bounds, apertures ≥ ``aperture_min``, x strictly increasing, adjacent gates
  ≥ ``min_gate_spacing`` apart, the finish beyond the last gate, and the **start outside
  gate 0's capture sphere**.
* **UC-09 AC5 (aperture range)** — sampled per-gate apertures span the configured range
  (tight and wide gates both appear), with the tightest still passing solvability.
* **UC-09 AC5/AC6 (deterministic zero-RNG fallback)** — pathological (empty-feasible)
  ranges exhaust the attempt cap and fall back to a solvable-by-construction course for
  every N∈[1,10], drawing nothing off the RNG.
* **UC-09 AC6 (reproducibility)** — a given seed reproduces the same *sequence* of courses
  (fixed draw count per attempt given N); a different seed differs.
* **UC-08 AC5 (dynamics)** — sampled dynamics stay inside the configured factor ranges, the
  draw order is pinned, and mass / drag alone genuinely perturb the trajectory.

Pure + hermetic: drives the samplers off explicit ``numpy`` generators; the one physical
check builds two :class:`SimpleDroneAdapter` instances directly (no env / sim / network).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from drone_fly.adapter.simple import SimpleDroneAdapter
from drone_fly.env.config import (
    CourseConfig,
    DynamicsParams,
    GateSpec,
    ObstacleSpec,
    RandomizationConfig,
)
from drone_fly.env.randomization import (
    _fallback_course,
    is_course_solvable,
    sample_course,
    sample_dynamics,
)


def _gate_zs(course: CourseConfig) -> list[float]:
    return [float(g.center[2]) for g in course.gates]


# =====================================================================================
# UC-09 AC5 — num_gates draw across the configured range
# =====================================================================================
def test_num_gates_spans_the_full_default_range() -> None:
    """Sampled ``num_gates`` covers the whole default [1, 10] integer range (AC5)."""
    rcfg = RandomizationConfig(enable_course=True)
    base = CourseConfig()
    rng = np.random.default_rng(2024)
    counts = {sample_course(rng, rcfg, base).num_gates for _ in range(600)}
    assert min(counts) == 1
    assert max(counts) == 10
    assert counts == set(range(1, 11)), f"missing gate counts: {set(range(1, 11)) - counts}"


def test_num_gates_respects_a_narrow_configured_range() -> None:
    """A configured [lo, hi] range bounds the drawn gate count (AC5)."""
    rcfg = RandomizationConfig(enable_course=True, num_gates_range=(2, 4))
    base = CourseConfig()
    rng = np.random.default_rng(0)
    counts = {sample_course(rng, rcfg, base).num_gates for _ in range(300)}
    assert counts == {2, 3, 4}


# =====================================================================================
# UC-09 AC5 — every sampled course is solvable and within bounds, for every N
# =====================================================================================
def test_sampled_courses_are_all_solvable_and_within_bounds() -> None:
    """2500 draws off one RNG are each flyable and inside every configured bound (AC5)."""
    rcfg = RandomizationConfig(enable_course=True)
    base = CourseConfig()
    rng = np.random.default_rng(7)

    lo_z = base.floor_z + rcfg.z_margin
    hi_z = base.ceiling_z - rcfg.z_margin

    for _ in range(2500):
        course = sample_course(rng, rcfg, base)
        # The sampler's own contract: never returns an unsolvable course.
        assert is_course_solvable(course, rcfg)

        sx, sy, sz = course.start_position
        # start height STRICTLY inside the vertical safety corridor (load-bearing: a spawn
        # on the bound flags a step-0 collision in SimpleDroneAdapter.reset).
        assert lo_z < sz < hi_z
        assert abs(sy) <= rcfg.lateral_bound

        # start must be OUTSIDE gate 0's capture sphere (else it auto-passes at step 0).
        g0 = course.gates[0]
        assert float(np.linalg.norm(course.start - g0.position)) > g0.aperture

        # per-gate bounds + strictly-increasing x + 3D spacing.
        prev_x = None
        prev_c = None
        for g in course.gates:
            gx, gy, gz = g.center
            assert lo_z < gz < hi_z
            assert abs(gy) <= rcfg.lateral_bound
            assert g.aperture >= rcfg.aperture_min
            if prev_x is not None:
                assert gx > prev_x  # strictly increasing x
                assert float(np.linalg.norm(g.position - prev_c)) >= rcfg.min_gate_spacing
            prev_x, prev_c = gx, g.position

        # first gate ahead of start, finish beyond last gate.
        assert course.gates[0].center[0] - sx >= rcfg.min_start_gate_gap
        assert course.finish_x - course.gates[-1].center[0] >= rcfg.min_gate_finish_gap

        # arena bounds are NOT randomized — copied verbatim from the base course.
        assert course.floor_z == base.floor_z
        assert course.ceiling_z == base.ceiling_z


def test_sampled_courses_actually_vary() -> None:
    """Sampling is not degenerate: many draws yield many distinct courses (AC5)."""
    rcfg = RandomizationConfig(enable_course=True)
    base = CourseConfig()
    rng = np.random.default_rng(0)
    starts = {tuple(np.round(sample_course(rng, rcfg, base).start_position, 6)) for _ in range(200)}
    assert len(starts) > 150  # overwhelmingly distinct spawns


def test_every_fixed_n_is_solvable() -> None:
    """For each N in [1, 10] pinned, every sampled course is solvable (AC5)."""
    base = CourseConfig()
    for n in range(1, 11):
        rcfg = RandomizationConfig(enable_course=True, num_gates_range=(n, n))
        rng = np.random.default_rng(100 + n)
        for _ in range(200):
            course = sample_course(rng, rcfg, base)
            assert course.num_gates == n
            assert is_course_solvable(course, rcfg)


# =====================================================================================
# UC-09 AC5 — per-gate aperture range spans tight+wide, tightest still solvable
# =====================================================================================
def test_sampled_apertures_span_the_configured_range() -> None:
    """Per-gate apertures span the configured range — tight AND wide gates both appear (AC5).

    The tightest sampled aperture still passes solvability (it is ≥ ``aperture_min``),
    so smaller apertures force accuracy without ever making a course unflyable.
    """
    rcfg = RandomizationConfig(enable_course=True)  # aperture range (0.4, 0.8), min 0.4
    base = CourseConfig()
    rng = np.random.default_rng(11)

    apertures: list[float] = []
    for _ in range(1500):
        course = sample_course(rng, rcfg, base)
        # Only collect from RNG-sampled courses (all solvable); fallback shares one aperture.
        apertures.extend(g.aperture for g in course.gates)

    lo, hi = rcfg.gate_aperture_range
    amin, amax = min(apertures), max(apertures)
    # Wide gates appear near the top of the range and tight gates near the bottom.
    assert amax > hi - 0.05, f"no wide gates sampled (max aperture {amax})"
    assert amin < lo + 0.05, f"no tight gates sampled (min aperture {amin})"
    # The tightest gate is still flyable — never below the configured minimum.
    assert amin >= rcfg.aperture_min


# =====================================================================================
# UC-09 AC5/AC6 — deterministic zero-RNG fallback on exhaustion
# =====================================================================================
def test_pathological_ranges_fall_back_deterministically_without_hanging() -> None:
    """Empty-feasible ranges exhaust the cap and return the zero-RNG fallback (AC5/AC6).

    Every gate is drawn above the ceiling corridor, so EVERY candidate is rejected and the
    sampler must return :func:`_fallback_course` — solvable-by-construction and RNG-free.
    """
    rcfg = RandomizationConfig(
        enable_course=True,
        num_gates_range=(3, 3),
        gate_center_z_range=(9.0, 9.0),  # far above the ceiling corridor → always rejected
        max_resample_attempts=25,
    )
    base = CourseConfig()
    rng = np.random.default_rng(1)

    result = sample_course(rng, rcfg, base)
    expected = _fallback_course(3, base, rcfg)
    assert result == expected  # frozen-dataclass equality: the exact fallback course
    assert is_course_solvable(result, rcfg) is True


def test_fallback_course_is_solvable_for_every_n() -> None:
    """The zero-RNG fallback is solvable-by-construction for every N∈[1, 10] (AC5)."""
    rcfg = RandomizationConfig(enable_course=True)
    base = CourseConfig()
    for n in range(1, 11):
        course = _fallback_course(n, base, rcfg)
        assert course.num_gates == n
        assert is_course_solvable(course, rcfg)


def test_fallback_course_draws_nothing_off_the_rng() -> None:
    """The fallback is deterministic and RNG-free: same inputs → identical course (AC6)."""
    rcfg = RandomizationConfig(enable_course=True)
    base = CourseConfig()
    a = _fallback_course(5, base, rcfg)
    b = _fallback_course(5, base, rcfg)
    assert a == b


# =====================================================================================
# UC-09 AC5 — is_course_solvable rejects each documented degenerate mode
# =====================================================================================
def _replace_gate(course: CourseConfig, idx: int, **kw) -> CourseConfig:
    gates = list(course.gates)
    gates[idx] = dataclasses.replace(gates[idx], **kw)
    return dataclasses.replace(course, gates=tuple(gates))


def test_is_course_solvable_flags_each_degenerate_mode() -> None:
    """The guard rejects each documented degenerate case for N-gate courses (AC5)."""
    rcfg = RandomizationConfig()
    base = CourseConfig()  # the default 3-gate course
    assert is_course_solvable(base, rcfg) is True

    # empty course (num_gates < 1).
    assert not is_course_solvable(dataclasses.replace(base, gates=()), rcfg)

    # first gate too close to / behind the start.
    near = _replace_gate(base, 0, center=(0.5, 0.0, 1.0))
    assert not is_course_solvable(near, rcfg)

    # start inside gate 0's capture sphere (would auto-pass at step 0).
    inside = dataclasses.replace(base, start_position=(2.4, 0.0, 1.0))
    assert not is_course_solvable(inside, rcfg)

    # non-monotonic x: gate 1 not strictly ahead of gate 0.
    nonmono = _replace_gate(base, 1, center=(2.0, 0.6, 1.3))
    assert not is_course_solvable(nonmono, rcfg)

    # adjacent gates closer than min_gate_spacing in 3D.
    crowded = _replace_gate(base, 1, center=(2.6, 0.0, 1.0))  # 0.1 from g0
    assert not is_course_solvable(crowded, rcfg)

    # aperture smaller than the flyable minimum.
    tiny = _replace_gate(base, 0, aperture=rcfg.aperture_min - 0.01)
    assert not is_course_solvable(tiny, rcfg)

    # a gate centre above the ceiling corridor.
    high = _replace_gate(base, 2, center=(5.5, 0.0, base.ceiling_z - rcfg.z_margin + 0.01))
    assert not is_course_solvable(high, rcfg)

    # start height on the floor corridor edge (must be STRICTLY inside).
    on_edge = dataclasses.replace(base, start_position=(0.0, 0.0, rcfg.z_margin))
    assert not is_course_solvable(on_edge, rcfg)

    # start pushed outside the lateral bound.
    wide = dataclasses.replace(base, start_position=(0.0, rcfg.lateral_bound + 0.1, 1.0))
    assert not is_course_solvable(wide, rcfg)

    # finish too close to the last gate.
    close_finish = dataclasses.replace(base, finish_x=base.gates[-1].center[0] + 0.1)
    assert not is_course_solvable(close_finish, rcfg)


def test_start_inside_gate0_sphere_is_rejected_specifically() -> None:
    """A start that already sits inside gate 0's aperture is unsolvable (AC5)."""
    rcfg = RandomizationConfig()
    # A one-gate course whose start is 0.3 from the gate centre, inside a 0.6 aperture.
    course = CourseConfig(
        start_position=(2.7, 0.0, 1.0),
        gates=(GateSpec(center=(3.0, 0.0, 1.0), aperture=0.6),),
        finish_x=6.0,
    )
    g0 = course.gates[0]
    assert float(np.linalg.norm(course.start - g0.position)) <= g0.aperture
    assert is_course_solvable(course, rcfg) is False


# =====================================================================================
# UC-09 AC6 — seeded reproducibility of the course stream
# =====================================================================================
def test_same_seed_reproduces_identical_course_sequence() -> None:
    """A given seed reproduces the same *sequence* of sampled courses (AC6)."""
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


# =====================================================================================
# UC-08 AC5 — dynamics bounds, pinned draw order, per-knob effect
# =====================================================================================
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
    """The dynamics draw order is pinned: the same seed reproduces identical params (AC6)."""
    rcfg = RandomizationConfig(enable_dynamics=True)
    base = DynamicsParams()
    a = sample_dynamics(np.random.default_rng(3), rcfg, base)
    b = sample_dynamics(np.random.default_rng(3), rcfg, base)
    assert a == b
    c = sample_dynamics(np.random.default_rng(4), rcfg, base)
    assert a != c


def test_disabled_dynamics_axis_defaults_equal_adapter_constants() -> None:
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


# =====================================================================================
# UC-15 AC3 — obstacle solvability guard: no anchor inside a pillar + polyline clearance
# =====================================================================================
def _finish_point(course: CourseConfig) -> tuple[float, float]:
    """The finish anchor's (x, y): finish_x at the last gate's lateral y (see current_target)."""
    return (course.finish_x, course.gates[-1].center[1])


def test_obstacle_on_a_gate_center_is_unsolvable() -> None:
    """A pillar swallowing a gate centre makes the course unflyable (AC3)."""
    base = CourseConfig()  # default 3-gate course
    g1 = base.gates[1]
    blocked = dataclasses.replace(
        base, obstacles=(ObstacleSpec(center=(g1.center[0], g1.center[1]), radius=0.4, height=2.5),)
    )
    assert is_course_solvable(blocked, RandomizationConfig()) is False


def test_obstacle_on_the_start_is_unsolvable() -> None:
    """A pillar on the spawn point is unflyable (AC3)."""
    base = CourseConfig()
    sx, sy, _ = base.start_position
    blocked = dataclasses.replace(
        base, obstacles=(ObstacleSpec(center=(sx, sy), radius=0.4, height=2.5),)
    )
    assert is_course_solvable(blocked, RandomizationConfig()) is False


def test_obstacle_on_the_finish_is_unsolvable() -> None:
    """A pillar on the finish anchor is unflyable (AC3)."""
    base = CourseConfig()
    blocked = dataclasses.replace(
        base, obstacles=(ObstacleSpec(center=_finish_point(base), radius=0.4, height=2.5),)
    )
    assert is_course_solvable(blocked, RandomizationConfig()) is False


def test_obstacle_on_the_polyline_fails_clearance() -> None:
    """A pillar sitting on a start→gate segment (not on an anchor) fails polyline clearance (AC3).

    The start (0,0,1)→g0 (2.5,0,1) segment runs along y=0; a pillar centred at (1.25, 0) with
    the segment inside its z-band violates the ``radius + obstacle_clearance`` margin even
    though it covers no gate/start/finish *anchor* — this is the ≥1-collision-free-path guard.
    """
    base = CourseConfig()
    on_line = dataclasses.replace(
        base, obstacles=(ObstacleSpec(center=(1.25, 0.0), radius=0.3, height=2.5),)
    )
    assert is_course_solvable(on_line, RandomizationConfig()) is False


def test_off_corridor_obstacle_is_solvable() -> None:
    """A pillar well off the corridor clears every anchor and the polyline → solvable (AC3)."""
    base = CourseConfig()
    off = dataclasses.replace(
        base, obstacles=(ObstacleSpec(center=(3.25, 1.5), radius=0.3, height=2.5),)
    )
    assert is_course_solvable(off, RandomizationConfig()) is True


def test_obstacle_below_the_polyline_z_band_does_not_block() -> None:
    """A short pillar whose band is entirely below the flight polyline never blocks it (AC3).

    The polyline flies at z≈1; a pillar of height 0.5 (band [0, 0.5]) sitting on the line is
    passed entirely above, so the segment's z-range does not overlap the band → solvable.
    """
    base = CourseConfig()
    short = dataclasses.replace(
        base, obstacles=(ObstacleSpec(center=(1.25, 0.0), radius=0.3, height=0.5),)
    )
    assert is_course_solvable(short, RandomizationConfig()) is True


# =====================================================================================
# UC-15 AC3 — obstacle SAMPLING: presence rate, solvability, obstacle-free fallback
# =====================================================================================
def test_enabled_obstacle_axis_produces_pillars_at_a_healthy_rate() -> None:
    """With the obstacle axis on, off-corridor placement clears the guard on the vast majority.

    The lateral off-corridor bias means sampled pillars almost always survive the solvability
    reject-resample, so obstacles are present at a healthy rate (not rare) (AC3).
    """
    rcfg = RandomizationConfig(enable_course=True, enable_obstacles=True)
    base = CourseConfig()
    rng = np.random.default_rng(0)
    courses = [sample_course(rng, rcfg, base) for _ in range(300)]
    with_obstacles = sum(1 for c in courses if c.obstacles)
    assert with_obstacles > 250, f"obstacles too rare when enabled: {with_obstacles}/300"
    # And every sampled obstacle course is still flyable.
    for c in courses:
        assert is_course_solvable(c, rcfg)


def test_sampled_obstacle_counts_respect_the_configured_range() -> None:
    """Drawn pillar counts stay within ``obstacle_count_range`` (AC3)."""
    rcfg = RandomizationConfig(
        enable_course=True, enable_obstacles=True, obstacle_count_range=(1, 3)
    )
    base = CourseConfig()
    rng = np.random.default_rng(1)
    counts = {len(sample_course(rng, rcfg, base).obstacles) for _ in range(300)}
    assert counts <= {1, 2, 3}
    assert max(counts) >= 1  # at least some courses carry pillars


def test_fallback_course_is_obstacle_free_even_with_obstacle_axis_on() -> None:
    """On resample exhaustion the zero-RNG fallback carries NO obstacles (AC3).

    Pathological gate ranges force every candidate to be rejected; the deterministic fallback
    is solvable-by-construction and obstacle-free, so it can never be unflyable.
    """
    rcfg = RandomizationConfig(
        enable_course=True,
        enable_obstacles=True,
        num_gates_range=(3, 3),
        gate_center_z_range=(9.0, 9.0),  # always above the ceiling corridor → always rejected
        max_resample_attempts=10,
    )
    base = CourseConfig()
    result = sample_course(np.random.default_rng(1), rcfg, base)
    assert result.obstacles == ()
    assert result == _fallback_course(3, base, rcfg)
    assert is_course_solvable(result, rcfg) is True


# =====================================================================================
# UC-15 — a disabled obstacle axis draws ZERO rng → leaves the course stream unperturbed
# =====================================================================================
def test_disabled_obstacle_axis_never_produces_obstacles() -> None:
    """With ``enable_obstacles=False`` no course ever carries pillars (default off)."""
    rcfg = RandomizationConfig(enable_course=True, enable_obstacles=False)
    base = CourseConfig()
    rng = np.random.default_rng(3)
    assert all(sample_course(rng, rcfg, base).obstacles == () for _ in range(200))


def test_disabled_obstacle_axis_leaves_the_gate_stream_unperturbed() -> None:
    """A disabled obstacle axis consumes NO rng, so obstacle *config* cannot shift the stream.

    Two configs differ only in their (unused) obstacle ranges, both with the axis OFF. Because
    a disabled axis draws nothing, the same seed yields a byte-identical course sequence — the
    UC-08/09 stream is untouched (the pinned draw-order / zero-draw guarantee).
    """
    base = CourseConfig()
    plain = RandomizationConfig(enable_course=True, enable_obstacles=False)
    fat = RandomizationConfig(
        enable_course=True,
        enable_obstacles=False,
        obstacle_count_range=(3, 3),
        obstacle_radius_range=(0.9, 0.9),
        obstacle_height_range=(2.5, 2.5),
    )

    def sequence(rcfg: RandomizationConfig, seed: int = 5, k: int = 8) -> list[CourseConfig]:
        rng = np.random.default_rng(seed)
        return [sample_course(rng, rcfg, base) for _ in range(k)]

    assert sequence(plain) == sequence(fat)
