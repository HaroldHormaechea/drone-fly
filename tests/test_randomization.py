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
    BatteryConfig,
    CourseConfig,
    DynamicsParams,
    GateSpec,
    ObstacleSpec,
    PadSpec,
    RandomizationConfig,
)
from drone_fly.env.obstacles import point_in_cylinder
from drone_fly.env.randomization import (
    _BATTERY_BUDGET,
    _course_segments,
    _eligible_gate_indices,
    _fallback_course,
    _path_energy,
    _place_service_pads,
    _place_single_recharge_pad,
    _place_single_repair_pad,
    _recharge_covering_valid,
    _reference_path,
    _segment_axis_distance_2d,
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


def test_on_corridor_obstacle_is_solvable_but_forces_evasion() -> None:
    """UC-35 AC3/AC4 contract reversal: a pillar sitting ON a start→gate segment is now SOLVABLE.

    UC-15 *forbade* on-route pillars (polyline far-clearance guard); UC-35 wants pillars near the
    nominal path so the drone is FORCED to evade, and instead enforces an *evadability* model. The
    start (0,0,1)→g0 (2.5,0,1) segment runs along y=0; a pillar centred at (1.25, 0) with clearance
    to every anchor and an escape lane is now flyable — yet the straight path passes within the
    pillar's radius of its axis, so the drone must deviate (forced evasion).
    """
    base = CourseConfig()
    on_line = dataclasses.replace(
        base, obstacles=(ObstacleSpec(center=(1.25, 0.0), radius=0.3, height=2.5),)
    )
    rcfg = RandomizationConfig()
    # Reversed contract: on-corridor-but-evadable ⇒ solvable (was False under UC-15).
    assert is_course_solvable(on_line, rcfg) is True
    # FORCED: the straight start→g0 path passes within the pillar radius of the pillar axis.
    start_xy = np.asarray(base.start, dtype=np.float64)
    g0 = base.gates[0].position
    axis = np.asarray([1.25, 0.0], dtype=np.float64)
    assert _segment_axis_distance_2d(start_xy, g0, axis) < 0.3


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
def _obstacle_maps_to_a_corridor_segment(
    course: CourseConfig, obstacle: ObstacleSpec, rcfg: RandomizationConfig, tol: float = 1e-6
) -> bool:
    """True iff ``obstacle`` sits within the perpendicular corridor of some waypoint→waypoint
    segment (incl. start→first-gate) AND the straight path passes within ``radius + drone_radius``
    of its axis (forced evasion) — the UC-35 AC3/AC4 "between-waypoints, threatens-the-route" test.
    """
    start_xy = np.asarray(course.start, dtype=np.float64)[:2]
    axis = np.asarray(obstacle.center, dtype=np.float64)
    forced_gap = float(obstacle.radius) + rcfg.drone_radius
    for p, q in _course_segments(start_xy, course.gates):
        d = _segment_axis_distance_2d(p, q, axis)
        if d <= rcfg.obstacle_corridor_half_width + tol and d <= forced_gap + tol:
            return True
    return False


def test_enabled_obstacle_axis_produces_pillars_at_a_healthy_rate() -> None:
    """UC-35 AC3/AC4: the obstacle axis places pillars BETWEEN consecutive waypoints, in-corridor.

    (a) ≥60% of sampled courses carry ≥1 pillar (a healthy rate, not rare); (b) every sampled
    obstacle course is still flyable (``is_course_solvable``); (c) every placed pillar maps to an
    inter-waypoint segment within the corridor and the straight path passes within
    ``radius + drone_radius`` of its axis (forced evasion).
    """
    rcfg = RandomizationConfig(enable_course=True, enable_obstacles=True)
    base = CourseConfig()
    rng = np.random.default_rng(0)
    courses = [sample_course(rng, rcfg, base) for _ in range(300)]
    with_obstacles = sum(1 for c in courses if c.obstacles)
    assert with_obstacles >= 180, f"healthy-rate floor not met: {with_obstacles}/300 (<60%)"
    for c in courses:
        # (b) every obstacle course is flyable.
        assert is_course_solvable(c, rcfg)
        # (c) every placed pillar threatens an inter-waypoint segment within the corridor.
        for obs in c.obstacles:
            assert _obstacle_maps_to_a_corridor_segment(c, obs, rcfg), (
                f"pillar {obs.center} does not map to a corridor segment"
            )


def test_sampled_obstacle_counts_respect_the_configured_range() -> None:
    """UC-35 AC3: ``obstacle_count_range`` is now an UPPER BOUND on the realised pillar count.

    Each drawn pillar consumes a fixed number of RNG draws but may be *skipped* if it fails the
    evadability accept test, so the realised count is ``⊆ {0, .., hi}`` — never above ``hi``, and
    possibly zero on pathologically short courses.
    """
    rcfg = RandomizationConfig(
        enable_course=True, enable_obstacles=True, obstacle_count_range=(1, 3)
    )
    base = CourseConfig()
    rng = np.random.default_rng(1)
    counts = {len(sample_course(rng, rcfg, base).obstacles) for _ in range(300)}
    assert counts <= {0, 1, 2, 3}  # upper bound: skips can drop the count, never raise it
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


# =====================================================================================
# UC-24 — full-course randomization: single recharge / repair feature pads
# =====================================================================================
# UC-24 redefines the UC-18 recharge axis. Recharge placement is now **reachable** and places
# EXACTLY ONE ``rechargeable`` pad on every course (regardless of energy-constraint) whenever the
# axis is on and a battery is enabled — under the SHIPPED DEFAULT battery UC-18 produced ZERO pads
# (the bug this fixes). A symmetric repair axis places EXACTLY ONE ``repairable`` pad (no energy
# model). Both are pure, zero-RNG feature placement at eligible gate anchors.

#: The shipped default battery: enabled with default drain rates. Under it every DEFAULT-range
#: course is energy-non-constrained, so the lone recharge pad is always a valid bonus — the AC2
#: "recharge is now reachable" case that UC-18 never produced.
_DEFAULT_BATTERY = BatteryConfig(enabled=True)

#: A cranked battery + 2-gate courses tuned so the full path often exceeds one charge yet each
#: half fits — so a SINGLE recharge pad forms a valid one-pad cover (the UC-24 narrowing of UC-18's
#: multi-pad covering guarantee). Chosen empirically: over seeds 0..59 it yields both constrained
#: and non-constrained courses, all one-pad-solvable. recharge_rate ≫ max docked drain keeps the
#: UC-17 net-positive invariant.
_TUNED_BATTERY = BatteryConfig(enabled=True, idle_rate=0.2, throttle_rate=0.1, recharge_rate=1.5)


def _rechargeable_pads(course: CourseConfig) -> list[PadSpec]:
    return [p for p in course.pads if p.rechargeable]


def _repairable_pads(course: CourseConfig) -> list[PadSpec]:
    return [p for p in course.pads if p.repairable]


def _gate_centers_xy(course: CourseConfig) -> set[tuple[float, float]]:
    return {(round(float(g.center[0]), 6), round(float(g.center[1]), 6)) for g in course.gates}


def _pad_xy(pad: PadSpec) -> tuple[float, float]:
    return (round(float(pad.center[0]), 6), round(float(pad.center[1]), 6))


def _min_pad_gate_distance(pad: PadSpec, course: CourseConfig) -> float:
    """Smallest horizontal distance from ``pad`` to any gate centre (UC-35 AC1 R_pad check)."""
    p = np.asarray(pad.center, dtype=np.float64)
    return min(
        float(np.linalg.norm(p - np.asarray(g.center, dtype=np.float64)[:2])) for g in course.gates
    )


def _pad_off_every_gate(pad: PadSpec, course: CourseConfig, rcfg: RandomizationConfig) -> bool:
    """UC-35 AC1: a placed pad keeps ≥ ``R_pad`` from EVERY gate centre and stays in-corridor."""
    if abs(float(pad.center[1])) > rcfg.lateral_bound:
        return False
    return _min_pad_gate_distance(pad, course) >= rcfg.pad_min_gate_distance - 1e-9


# --- AC2: recharge placement is reachable — exactly ONE pad under the DEFAULT battery ------
def test_enable_recharge_places_exactly_one_pad_under_default_battery() -> None:
    """AC2 (the fix): with the recharge axis on and the SHIPPED DEFAULT battery, EVERY sampled
    course carries exactly one ``rechargeable`` pad at a gate anchor and stays solvable — UC-18
    produced zero pads here (recharge was unreachable via the train path)."""
    rcfg = RandomizationConfig(enable_course=True, enable_recharge=True)
    base = CourseConfig()
    for seed in range(30):
        course = sample_course(np.random.default_rng(seed), rcfg, base, battery=_DEFAULT_BATTERY)
        pads = _rechargeable_pads(course)
        assert len(pads) == 1, f"seed {seed}: expected exactly one recharge pad, got {len(pads)}"
        # Non-constrained under the default battery ⇒ the lone pad is a valid bonus, still solvable.
        assert is_course_solvable(course, rcfg, battery=_DEFAULT_BATTERY)
        # UC-35 AC1: the pad is placed OFF every gate column (≥ R_pad from every gate centre),
        # NOT under a waypoint — and within the lateral corridor.
        assert _pad_off_every_gate(pads[0], course, rcfg)
        assert _pad_xy(pads[0]) not in _gate_centers_xy(course)
        # Recharge-only axis ⇒ no repair pad.
        assert not _repairable_pads(course)


# --- AC3: repair placement — exactly ONE repairable pad (no energy model) ------------------
def test_enable_repair_places_exactly_one_repairable_pad() -> None:
    """AC3: with the repair axis on, EVERY sampled course carries exactly one ``repairable`` pad at
    a gate anchor. Repair needs no battery (damage does not gate reachability), so it places even
    with ``battery=None``; a repair-only axis carries no recharge pad."""
    rcfg = RandomizationConfig(enable_course=True, enable_repair=True)
    base = CourseConfig()
    for seed in range(30):
        course = sample_course(np.random.default_rng(seed), rcfg, base, battery=None)
        pads = _repairable_pads(course)
        assert len(pads) == 1, f"seed {seed}: expected exactly one repair pad, got {len(pads)}"
        assert is_course_solvable(course, rcfg)
        # UC-35 AC1: repair pad is placed OFF every gate column too.
        assert _pad_off_every_gate(pads[0], course, rcfg)
        assert _pad_xy(pads[0]) not in _gate_centers_xy(course)
        assert not _rechargeable_pads(course)  # repair-only axis


# --- AC1/AC3: both axes on — one of each at DISTINCT anchors on a multi-gate course --------
def test_both_axes_place_one_recharge_and_one_repair_at_distinct_anchors() -> None:
    """AC1/AC3: with BOTH axes on and ≥2 eligible gates, a course carries exactly one recharge pad
    and one repair pad at DISTINCT gate anchors (two separate pads)."""
    rcfg = RandomizationConfig(
        enable_course=True, enable_recharge=True, enable_repair=True, num_gates_range=(4, 4)
    )
    base = CourseConfig()
    for seed in range(30):
        course = sample_course(np.random.default_rng(seed), rcfg, base, battery=_DEFAULT_BATTERY)
        recharge = _rechargeable_pads(course)
        repair = _repairable_pads(course)
        assert len(recharge) == 1 and len(repair) == 1
        # Distinct pads (no dual-purpose pad when ≥2 eligible gates), on distinct gate anchors.
        assert len(course.pads) == 2
        assert _pad_xy(recharge[0]) != _pad_xy(repair[0])
        assert not (recharge[0].rechargeable and recharge[0].repairable)
        # UC-35 AC1: both pads are placed OFF every gate column, and ≥ R_pad from each other.
        assert _pad_off_every_gate(recharge[0], course, rcfg)
        assert _pad_off_every_gate(repair[0], course, rcfg)
        sep = float(
            np.linalg.norm(
                np.asarray(recharge[0].center, dtype=np.float64)
                - np.asarray(repair[0].center, dtype=np.float64)
            )
        )
        assert sep >= rcfg.pad_min_gate_distance - 1e-9
        assert is_course_solvable(course, rcfg, battery=_DEFAULT_BATTERY)


# --- AC1/AC3 (co-location exception): both axes on a 1-gate course ⇒ ONE dual-purpose pad ---
def test_both_axes_on_one_gate_course_emit_a_single_dual_purpose_pad() -> None:
    """AC1/AC3 documented co-location exception: when only ONE eligible gate exists (a 1-gate
    course), the recharge and repair pads co-locate into a SINGLE ``rechargeable & repairable``
    pad — still 'one of each' (counts 1/1), avoiding one pad shadowing the other."""
    rcfg = RandomizationConfig(
        enable_course=True,
        enable_recharge=True,
        enable_repair=True,
        num_gates_range=(1, 1),
    )
    base = CourseConfig()
    for seed in range(20):
        course = sample_course(np.random.default_rng(seed), rcfg, base, battery=_DEFAULT_BATTERY)
        assert len(course.pads) == 1
        dual = course.pads[0]
        assert dual.rechargeable and dual.repairable
        # Still exactly one of each purpose.
        assert len(_rechargeable_pads(course)) == 1 and len(_repairable_pads(course)) == 1
        assert is_course_solvable(course, rcfg, battery=_DEFAULT_BATTERY)


# --- AC1: off ⇒ no pads at all ------------------------------------------------------------
def test_disabled_service_axes_place_no_pads() -> None:
    """AC1: with both service axes off, no course carries any pad — even with a battery enabled
    (the recharge branch requires ``enable_recharge``)."""
    rcfg = RandomizationConfig(enable_course=True, enable_recharge=False, enable_repair=False)
    base = CourseConfig()
    for seed in range(20):
        course = sample_course(np.random.default_rng(seed), rcfg, base, battery=_DEFAULT_BATTERY)
        assert course.pads == ()


def test_recharge_axis_off_when_battery_absent() -> None:
    """AC4 coherence (env level): the recharge axis is inert without an enabled battery — with
    ``battery=None`` no recharge pad is placed even if ``enable_recharge`` is on (a disabled/absent
    battery makes the pad physically meaningless). Repair, which has no battery dependency, is
    unaffected — but here it is off, so the course stays pad-free."""
    rcfg = RandomizationConfig(enable_course=True, enable_recharge=True, enable_repair=False)
    base = CourseConfig()
    for seed in range(20):
        course = sample_course(np.random.default_rng(seed), rcfg, base, battery=None)
        assert course.pads == ()


# --- AC4 coherence backstop is the CLI's job; here we verify the constrained one-pad cover --
def test_constrained_course_is_covered_by_a_single_pad_under_a_tuned_battery() -> None:
    """UC-24 one-pad narrowing of UC-18's guarantee: under a cranked battery where the full path
    often exceeds one charge, a course still carries exactly one recharge pad, is battery-solvable,
    and every constrained course's single pad forms a valid one-pad cover (each induced sub-path
    fits one charge)."""
    rcfg = RandomizationConfig(enable_course=True, enable_recharge=True, num_gates_range=(2, 2))
    base = CourseConfig()
    saw_constrained = saw_non_constrained = False
    for seed in range(60):
        course = sample_course(np.random.default_rng(seed), rcfg, base, battery=_TUNED_BATTERY)
        assert len(_rechargeable_pads(course)) == 1
        assert is_course_solvable(course, rcfg, battery=_TUNED_BATTERY)
        if _path_energy(_reference_path(course), rcfg, _TUNED_BATTERY) > _BATTERY_BUDGET:
            saw_constrained = True
            assert _recharge_covering_valid(
                course, rcfg, _TUNED_BATTERY, _rechargeable_pads(course)
            )
        else:
            saw_non_constrained = True
    assert saw_constrained, "the tuned drain must produce at least one constrained course"
    assert saw_non_constrained, "the tuned drain must also produce a non-constrained course"


# --- AC5: zero-RNG — disabled axes never perturb the gate/dynamics stream ------------------
def test_disabled_service_axes_leave_the_course_stream_byte_identical() -> None:
    """AC5/AC6: two configs differing ONLY in their (unused) recharge/repair fields, both with the
    axes OFF, yield a byte-identical course sequence — a disabled axis draws nothing."""
    base = CourseConfig()
    plain = RandomizationConfig(enable_course=True, enable_recharge=False, enable_repair=False)
    fat = RandomizationConfig(
        enable_course=True,
        enable_recharge=False,
        enable_repair=False,
        recharge_nominal_speed=0.5,
        recharge_energy_margin=9.0,
        recharge_pad_radius=1.2,
        repair_pad_radius=1.7,
    )

    def sequence(rcfg: RandomizationConfig, seed: int = 7, k: int = 8) -> list[CourseConfig]:
        rng = np.random.default_rng(seed)
        return [sample_course(rng, rcfg, base, battery=_DEFAULT_BATTERY) for _ in range(k)]

    assert sequence(plain) == sequence(fat)


def test_active_service_axes_only_append_pads_never_shift_the_geometry() -> None:
    """AC5 (zero-RNG placement): enabling recharge + repair leaves the sampled start / gates /
    finish / obstacles bit-for-bit identical to the axes-off run for the same seed — the placer
    consumes no RNG, so it can only APPEND pads, never move a gate."""
    base = CourseConfig()
    on = RandomizationConfig(
        enable_course=True, enable_recharge=True, enable_repair=True, num_gates_range=(3, 3)
    )
    off = RandomizationConfig(
        enable_course=True, enable_recharge=False, enable_repair=False, num_gates_range=(3, 3)
    )
    for seed in range(30):
        c_on = sample_course(np.random.default_rng(seed), on, base, battery=_DEFAULT_BATTERY)
        c_off = sample_course(np.random.default_rng(seed), off, base, battery=_DEFAULT_BATTERY)
        assert c_on.gates == c_off.gates
        assert c_on.start_position == c_off.start_position
        assert c_on.finish_x == c_off.finish_x
        assert c_on.obstacles == c_off.obstacles
        # The axes-off run never carries pads; the axes-on run appends exactly one of each purpose.
        assert c_off.pads == ()
        assert len(_rechargeable_pads(c_on)) == 1 and len(_repairable_pads(c_on)) == 1


def test_service_placement_is_deterministic_for_a_fixed_seed_and_battery() -> None:
    """AC5: placement is a pure function of (course, rcfg, battery) — same inputs → identical
    courses (including the placed pads), and the placer itself is deterministic (zero RNG)."""
    rcfg = RandomizationConfig(enable_course=True, enable_recharge=True, enable_repair=True)
    base = CourseConfig()
    a = sample_course(np.random.default_rng(9), rcfg, base, battery=_DEFAULT_BATTERY)
    b = sample_course(np.random.default_rng(9), rcfg, base, battery=_DEFAULT_BATTERY)
    assert a == b
    # The coordinator is deterministic on a fixed candidate too (zero RNG).
    off = RandomizationConfig(enable_course=True, enable_recharge=False, enable_repair=False)
    candidate = sample_course(np.random.default_rng(9), off, base)
    assert _place_service_pads(candidate, rcfg, _DEFAULT_BATTERY) == _place_service_pads(
        candidate, rcfg, _DEFAULT_BATTERY
    )


# --- AC1: eligibility — descend-column-clear is respected; no clear anchor ⇒ None ----------
def test_eligibility_admits_gates_whose_column_clips_a_pillar_via_offset() -> None:
    """UC-35 AC1 redefinition: eligibility is now "an OFF-gate pad position exists near the anchor",
    not "the gate column is clear". A pillar sitting on g0's column no longer disqualifies g0 — the
    pad is placed OFF the column (dodging the pillar), so BOTH gates are eligible and the placed pad
    is off every gate centre and descend-column-clear of the pillar."""
    rcfg = RandomizationConfig(enable_course=True, enable_recharge=True, enable_repair=True)
    g0 = GateSpec(center=(2.5, 0.0, 1.0), aperture=0.6)
    g1 = GateSpec(center=(4.0, 0.0, 1.3), aperture=0.6)
    # A pillar sitting on g0's (x, y) column: under UC-15 this excluded g0; UC-35 offsets the pad.
    blocker = ObstacleSpec(center=(2.5, 0.0), radius=0.6, height=2.0)
    course = CourseConfig(gates=(g0, g1), obstacles=(blocker,), finish_x=6.0)
    # Both gates now admit an off-gate pad position ⇒ both eligible ([0, 1], not [1]).
    assert _eligible_gate_indices(course, rcfg) == [0, 1]
    placed = _place_single_recharge_pad(course, rcfg, _DEFAULT_BATTERY)
    assert placed is not None
    pads = _rechargeable_pads(placed)
    assert len(pads) == 1
    # The pad is OFF every gate column (not on a gate centre) and dodges the pillar.
    assert _pad_off_every_gate(pads[0], placed, rcfg)
    assert _pad_xy(pads[0]) not in _gate_centers_xy(placed)


def test_no_clear_anchor_returns_none_for_every_placer() -> None:
    """AC1: when NO off-gate pad position exists near ANY gate, every placer returns ``None`` (→ the
    sampler reject-resamples) rather than placing an unsafe pad.

    UC-35: a single pillar on the sole gate's column no longer suffices — the pad can dodge it off
    the column. A genuine no-anchor case needs a stricter blocker: here wall pillars surround the
    lone gate on the -x/+y/-y rays (the +x ray is cut off by ``finish_x=4.0``), so the fixed
    off-gate candidate ring has no feasible, descend-column-clear position anywhere."""
    rcfg = RandomizationConfig(enable_course=True, enable_recharge=True, enable_repair=True)
    gate = GateSpec(center=(3.0, 0.0, 1.0), aperture=0.6)
    blockers = (
        ObstacleSpec(center=(3.0, 1.6), radius=0.6, height=2.5),  # +y ray
        ObstacleSpec(center=(3.0, -1.6), radius=0.6, height=2.5),  # -y ray
        ObstacleSpec(center=(1.6, 0.0), radius=0.6, height=2.5),  # -x ray (far half)
        ObstacleSpec(center=(0.4, 0.0), radius=0.6, height=2.5),  # -x ray (near half)
    )
    course = CourseConfig(gates=(gate,), obstacles=blockers, finish_x=4.0)
    assert _eligible_gate_indices(course, rcfg) == []
    assert _place_single_recharge_pad(course, rcfg, _DEFAULT_BATTERY) is None
    assert _place_single_repair_pad(course, rcfg) is None
    assert _place_service_pads(course, rcfg, _DEFAULT_BATTERY) is None


def test_service_placement_with_obstacles_never_deadlocks() -> None:
    """AC1/AC7: with the obstacle axis AND both service axes on, sampling always returns a valid
    course carrying exactly one recharge + one repair pad — the obstacle-free fallback guarantees a
    clear anchor, so placement never deadlocks the sampler."""
    rcfg = RandomizationConfig(
        enable_course=True, enable_obstacles=True, enable_recharge=True, enable_repair=True
    )
    base = CourseConfig()
    for seed in range(30):
        course = sample_course(np.random.default_rng(seed), rcfg, base, battery=_DEFAULT_BATTERY)
        assert len(_rechargeable_pads(course)) == 1
        assert len(_repairable_pads(course)) == 1
        assert is_course_solvable(course, rcfg, battery=_DEFAULT_BATTERY)


# --- AC1/AC7 (fallback): the exhaustion fallback carries its single feature pad(s) ---------
def test_fallback_under_service_axes_carries_its_single_pads() -> None:
    """AC1/AC7: the deterministic fallback (obstacle-free) carries exactly one recharge + one
    repair pad for every N, and stays solvable. For N=1 the two co-locate into one dual pad."""
    rcfg = RandomizationConfig(enable_course=True, enable_recharge=True, enable_repair=True)
    base = CourseConfig()
    for n in range(1, 11):
        fb = _fallback_course(n, base, rcfg, battery=_DEFAULT_BATTERY)
        assert len(_rechargeable_pads(fb)) == 1
        assert len(_repairable_pads(fb)) == 1
        if n == 1:
            assert len(fb.pads) == 1 and fb.pads[0].rechargeable and fb.pads[0].repairable
        assert is_course_solvable(fb, rcfg, battery=_DEFAULT_BATTERY)


def test_sample_course_exhaustion_returns_a_fallback_with_one_pad() -> None:
    """AC1/AC7: when every sampled candidate is rejected (pathological ranges), the returned
    fallback still carries its single recharge pad and is battery-solvable."""
    rcfg = RandomizationConfig(
        enable_course=True,
        enable_recharge=True,
        num_gates_range=(10, 10),
        gate_center_z_range=(9.0, 9.0),  # far above the corridor → every candidate rejected
        max_resample_attempts=20,
    )
    base = CourseConfig()
    result = sample_course(np.random.default_rng(1), rcfg, base, battery=_DEFAULT_BATTERY)
    expected = _fallback_course(10, base, rcfg, battery=_DEFAULT_BATTERY)
    assert result == expected  # the exact zero-RNG fallback
    assert len(_rechargeable_pads(result)) == 1
    assert is_course_solvable(result, rcfg, battery=_DEFAULT_BATTERY)


# =====================================================================================
# UC-35 — course randomization placement: pads OFF waypoints, obstacles BETWEEN waypoints
# =====================================================================================
# The end-to-end acceptance-criteria suite for UC-35, exercising the sampler across many seeds:
#   AC-1 no pad within R_pad of any gate; AC-2 exactly-one-each + in-bounds + toggle; AC-3 every
#   obstacle maps to an inter-waypoint corridor segment; AC-4 the min-clearance / evadability
#   invariant; AC-5 same-seed identical placement; AC-6 toggles gate each feature; AC-7 byte-
#   identity for disabled axes / non-randomized configs; plus the cranked-battery covering
#   re-derived against the OFF-gate recharge pad's actual (x, y).


def _assert_obstacle_clearance_invariant(course: CourseConfig, rcfg: RandomizationConfig) -> None:
    """AC-4: assert the full evadability invariant on every pillar of ``course`` (mirrors, and is at
    least as strict as, the obstacle branch of :func:`is_course_solvable`)."""
    obstacles = course.obstacles
    if not obstacles:
        return
    floor_z = course.floor_z
    last = course.gates[-1]
    finish_pt = np.asarray([course.finish_x, last.center[1], last.center[2]], dtype=np.float64)
    anchors = [course.start, *(g.position for g in course.gates), finish_pt]
    anchors_xy = [np.asarray(a, dtype=np.float64)[:2] for a in anchors]
    for i, obs in enumerate(obstacles):
        axis = np.asarray(obs.center, dtype=np.float64)
        radius = float(obs.radius)
        gate_req = radius + rcfg.drone_radius + rcfg.obstacle_evasion_margin
        # (i) no anchor inside the pillar, and every anchor keeps gate-passability clearance.
        for pt in anchors:
            assert not point_in_cylinder(pt, obs, floor_z), (
                f"anchor {pt} inside pillar {obs.center}"
            )
        for a_xy in anchors_xy:
            assert float(np.linalg.norm(axis - a_xy)) >= gate_req - 1e-9
        # (ii) an escape lane on the wider side within the lateral corridor.
        ay = float(axis[1])
        lane = max(rcfg.lateral_bound - (ay + radius), (ay - radius) + rcfg.lateral_bound)
        assert lane >= rcfg.drone_radius + rcfg.obstacle_evasion_margin - 1e-9
        # (iii) pairwise separation — no unevadable multi-pillar wall.
        for j in range(i + 1, len(obstacles)):
            other = obstacles[j]
            oaxis = np.asarray(other.center, dtype=np.float64)
            need = (
                radius
                + float(other.radius)
                + 2.0 * (rcfg.drone_radius + rcfg.obstacle_evasion_margin)
            )
            assert float(np.linalg.norm(axis - oaxis)) >= need - 1e-9


# --- AC-1: no pad within R_pad of any gate centre, across many seeds -----------------------
def test_uc35_ac1_no_pad_within_r_pad_of_any_gate_across_many_seeds() -> None:
    """AC-1: with recharge + repair (and obstacles) on, EVERY placed pad keeps ≥ ``R_pad`` from
    EVERY gate centre — no pad under a waypoint — across many seeds and gate counts."""
    rcfg = RandomizationConfig(
        enable_course=True, enable_obstacles=True, enable_recharge=True, enable_repair=True
    )
    base = CourseConfig()
    for seed in range(120):
        course = sample_course(np.random.default_rng(seed), rcfg, base, battery=_DEFAULT_BATTERY)
        assert course.pads, f"seed {seed}: expected service pads"
        for pad in course.pads:
            d = _min_pad_gate_distance(pad, course)
            assert d >= rcfg.pad_min_gate_distance - 1e-9, (
                f"seed {seed}: pad {pad.center} only {d:.3f} m from nearest gate (< R_pad)"
            )
        assert is_course_solvable(course, rcfg, battery=_DEFAULT_BATTERY)


# --- AC-2: exactly one of each, in bounds, and each feature is toggle-gated -----------------
def test_uc35_ac2_exactly_one_each_in_bounds_and_toggle_gated() -> None:
    """AC-2: pad placement stays valid — exactly one recharge + one repair pad, each within the
    course bounds — and the per-feature toggles still gate whether each pad is placed."""
    base = CourseConfig()
    # Both axes on (≥2 gates ⇒ distinct pads): exactly one of each, all in-bounds.
    both = RandomizationConfig(
        enable_course=True, enable_recharge=True, enable_repair=True, num_gates_range=(3, 6)
    )
    for seed in range(60):
        course = sample_course(np.random.default_rng(seed), both, base, battery=_DEFAULT_BATTERY)
        assert len(_rechargeable_pads(course)) == 1
        assert len(_repairable_pads(course)) == 1
        for pad in course.pads:
            px, py = float(pad.center[0]), float(pad.center[1])
            assert abs(py) <= both.lateral_bound + 1e-9
            assert float(course.start[0]) <= px <= course.finish_x + 1e-9
    # Toggle gating: recharge-only ⇒ no repair pad; repair-only ⇒ no recharge pad.
    rc_only = RandomizationConfig(enable_course=True, enable_recharge=True, enable_repair=False)
    rp_only = RandomizationConfig(enable_course=True, enable_recharge=False, enable_repair=True)
    for seed in range(30):
        c_rc = sample_course(np.random.default_rng(seed), rc_only, base, battery=_DEFAULT_BATTERY)
        assert len(_rechargeable_pads(c_rc)) == 1 and not _repairable_pads(c_rc)
        c_rp = sample_course(np.random.default_rng(seed), rp_only, base, battery=_DEFAULT_BATTERY)
        assert len(_repairable_pads(c_rp)) == 1 and not _rechargeable_pads(c_rp)


# --- AC-3: every obstacle maps to an inter-waypoint corridor segment, many seeds ------------
def test_uc35_ac3_every_obstacle_maps_to_a_corridor_segment_across_many_seeds() -> None:
    """AC-3: across many seeds, every generated pillar lies within the along-span AND perpendicular
    corridor of some waypoint→waypoint segment (incl. start→first-gate) — i.e. it threatens the
    route (the straight path passes within ``radius + drone_radius`` of the axis)."""
    rcfg = RandomizationConfig(enable_course=True, enable_obstacles=True)
    base = CourseConfig()
    saw_any = False
    for seed in range(150):
        course = sample_course(np.random.default_rng(seed), rcfg, base)
        for obs in course.obstacles:
            saw_any = True
            assert _obstacle_maps_to_a_corridor_segment(course, obs, rcfg), (
                f"seed {seed}: pillar {obs.center} not within any segment corridor"
            )
    assert saw_any, "expected at least some pillars to be sampled across 150 seeds"


# --- AC-4: the min-clearance / evadability invariant holds on every sampled course ----------
def test_uc35_ac4_min_clearance_invariant_holds_across_many_seeds() -> None:
    """AC-4: every sampled obstacle course is evadable — no anchor inside a pillar, every anchor
    keeps ``radius + drone_radius + evasion_margin`` clearance, an escape lane exists on the wider
    side within the corridor, and no two pillars form an unevadable wall. Verified directly (not via
    ``is_course_solvable``) across many seeds, then cross-checked against the solvability guard."""
    rcfg = RandomizationConfig(enable_course=True, enable_obstacles=True)
    base = CourseConfig()
    for seed in range(150):
        course = sample_course(np.random.default_rng(seed), rcfg, base)
        _assert_obstacle_clearance_invariant(course, rcfg)
        assert is_course_solvable(course, rcfg)


def test_uc35_ac4_multi_pillar_wall_is_unsolvable() -> None:
    """AC-4: two pillars packed closer than the pairwise-separation bound form an unevadable wall →
    the course is rejected by the clearance invariant (guards a hand-built infeasible set)."""
    base = CourseConfig()
    rcfg = RandomizationConfig()
    # Two fat pillars almost touching, straddling the corridor between start and g0.
    wall = dataclasses.replace(
        base,
        obstacles=(
            ObstacleSpec(center=(1.25, 0.2), radius=0.6, height=2.5),
            ObstacleSpec(center=(1.25, -0.2), radius=0.6, height=2.5),
        ),
    )
    assert is_course_solvable(wall, rcfg) is False


# --- AC-5: same seed → identical pad AND obstacle placement --------------------------------
def test_uc35_ac5_same_seed_reproduces_identical_placement() -> None:
    """AC-5: placement is a pure function of (seed, rcfg, battery) — the same seed reproduces the
    same course including its off-gate pads and between-waypoint pillars; other seeds differ."""
    rcfg = RandomizationConfig(
        enable_course=True, enable_obstacles=True, enable_recharge=True, enable_repair=True
    )
    base = CourseConfig()
    for seed in range(20):
        a = sample_course(np.random.default_rng(seed), rcfg, base, battery=_DEFAULT_BATTERY)
        b = sample_course(np.random.default_rng(seed), rcfg, base, battery=_DEFAULT_BATTERY)
        assert a == b  # frozen-dataclass equality: pads + obstacles + gates all identical
        assert a.pads == b.pads
        assert a.obstacles == b.obstacles
    # A different seed produces a different course stream.
    s1 = sample_course(np.random.default_rng(1), rcfg, base, battery=_DEFAULT_BATTERY)
    s2 = sample_course(np.random.default_rng(2), rcfg, base, battery=_DEFAULT_BATTERY)
    assert s1 != s2


# --- AC-6: independent toggles gate each feature -------------------------------------------
def test_uc35_ac6_toggles_independently_gate_each_feature() -> None:
    """AC-6: the obstacle / recharge / repair toggles each gate ONLY their own feature — turning one
    on never places another's pad/pillar, and all-off places nothing."""
    base = CourseConfig()
    all_off = RandomizationConfig(enable_course=True)
    for seed in range(20):
        c = sample_course(np.random.default_rng(seed), all_off, base, battery=_DEFAULT_BATTERY)
        assert c.pads == () and c.obstacles == ()
    obs_only = RandomizationConfig(enable_course=True, enable_obstacles=True)
    for seed in range(20):
        c = sample_course(np.random.default_rng(seed), obs_only, base, battery=_DEFAULT_BATTERY)
        assert c.pads == ()  # obstacle axis never places pads
    rc_only = RandomizationConfig(enable_course=True, enable_recharge=True)
    for seed in range(20):
        c = sample_course(np.random.default_rng(seed), rc_only, base, battery=_DEFAULT_BATTERY)
        assert c.obstacles == () and not _repairable_pads(c)  # recharge axis: no pillars/repair


# --- AC-7: byte-identity for disabled axes / non-randomized (UC-35 fields perturb nothing) --
def test_uc35_ac7_disabled_axes_leave_the_stream_byte_identical() -> None:
    """AC-7: two configs differing ONLY in their UC-35 placement-geometry fields, with the obstacle
    and service axes OFF, yield a byte-identical course sequence — the UC-35 fields are read only by
    the (inactive) samplers/placers, so a disabled-axis / non-randomized run is unperturbed."""
    base = CourseConfig()
    plain = RandomizationConfig(enable_course=True)
    tweaked = RandomizationConfig(
        enable_course=True,
        pad_min_gate_distance=1.7,
        drone_radius=0.25,
        obstacle_evasion_margin=0.4,
        obstacle_corridor_half_width=0.9,
        obstacle_gate_clearance=0.7,
        min_obstacle_separation=0.5,
        obstacle_along_margin_frac=0.15,
    )

    def sequence(rcfg: RandomizationConfig, seed: int = 5, k: int = 8) -> list[CourseConfig]:
        rng = np.random.default_rng(seed)
        return [sample_course(rng, rcfg, base, battery=_DEFAULT_BATTERY) for _ in range(k)]

    assert sequence(plain) == sequence(tweaked)


def test_uc35_ac7_disabled_obstacle_axis_draws_no_pillars() -> None:
    """AC-7: with the obstacle axis off, no course carries pillars regardless of UC-35 corridor
    settings — the new sampler is never entered, so its draws never perturb the gate stream."""
    base = CourseConfig()
    rcfg = RandomizationConfig(
        enable_course=True, enable_obstacles=False, obstacle_corridor_half_width=0.9
    )
    rng = np.random.default_rng(3)
    assert all(sample_course(rng, rcfg, base).obstacles == () for _ in range(200))


# --- Cranked-battery covering re-derived against the OFF-gate recharge pad ------------------
def test_uc35_cranked_battery_covering_valid_against_offset_pads() -> None:
    """UC-35 + UC-24 AC4: under a cranked battery where the full path often exceeds one charge, the
    single recharge pad is placed OFF the gate columns, yet the covering check (which now models the
    detour through the pad's ACTUAL (x, y)) still re-derives as valid for constrained courses."""
    rcfg = RandomizationConfig(enable_course=True, enable_recharge=True, num_gates_range=(2, 2))
    base = CourseConfig()
    saw_constrained = False
    for seed in range(60):
        course = sample_course(np.random.default_rng(seed), rcfg, base, battery=_TUNED_BATTERY)
        pads = _rechargeable_pads(course)
        assert len(pads) == 1
        # The recharge pad is OFF every gate column (UC-35 AC1), not under a waypoint.
        assert _pad_off_every_gate(pads[0], course, rcfg)
        assert is_course_solvable(course, rcfg, battery=_TUNED_BATTERY)
        if _path_energy(_reference_path(course), rcfg, _TUNED_BATTERY) > _BATTERY_BUDGET:
            saw_constrained = True
            # Covering re-derives against the pad's real (off-gate) position.
            assert _recharge_covering_valid(course, rcfg, _TUNED_BATTERY, pads)
    assert saw_constrained, "the tuned drain must produce at least one constrained course"
