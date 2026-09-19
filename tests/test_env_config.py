"""UC-16 — env-config dataclasses for pad docking (:mod:`drone_fly.env.config`).

Note: the sibling ``test_config.py`` covers the unrelated *run-config* module
(:mod:`drone_fly.config`, YAML loading). This file covers the **environment** config
dataclasses that UC-16 extends.

Covers:
* **AC1** — ``PadSpec`` shape/defaults; ``CourseConfig().pads == ()`` and the default course /
  ``EnvConfig()`` stay byte-identical to UC-15 (pads appended LAST, empty by default).
* **AC1 positional-ctor compat** — appending ``pads`` after ``obstacles`` (and ``dock`` after
  ``obstacle_vision``) leaves every UC-01..15 positional constructor call unshifted.
* **AC2** — ``DockConfig`` documented, conservative defaults; ``EnvConfig.dock`` present.
* factory shapes — ``default_pad_course()`` / ``single_pad_course()`` build the expected courses.
"""

from __future__ import annotations

import dataclasses

import pytest

from drone_fly.adapter.simple import BASE_MAX_BODY_RATE
from drone_fly.env.config import (
    BatteryConfig,
    CourseConfig,
    DamageConfig,
    DockConfig,
    EnvConfig,
    EpisodeConfig,
    GateSpec,
    ObstacleSpec,
    PadSpec,
    RandomizationConfig,
    default_obstacle_course,
    default_pad_course,
    default_repair_course,
    single_gate_course,
    single_pad_course,
    single_repair_pad_course,
)


# =====================================================================================
# PadSpec — floor-anchored center + horizontal radius (AC1)
# =====================================================================================
def test_pad_spec_fields() -> None:
    pad = PadSpec(center=(1.5, -2.0), radius=0.5)
    assert pad.center == (1.5, -2.0)
    assert pad.radius == 0.5


def test_pad_spec_is_frozen() -> None:
    """PadSpec mirrors ObstacleSpec/GateSpec — an immutable value object."""
    pad = PadSpec(center=(0.0, 0.0), radius=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pad.radius = 1.0  # type: ignore[misc]


def test_pad_spec_axis_xy_is_float64_ndarray() -> None:
    pad = PadSpec(center=(1.0, 2.0), radius=0.5)
    axis = pad.axis_xy
    assert axis.tolist() == [1.0, 2.0]
    assert str(axis.dtype) == "float64"


def test_pad_spec_carries_no_z() -> None:
    """A pad is floor-anchored: its center is 2-D (no z of its own)."""
    assert len(PadSpec(center=(0.0, 0.0), radius=0.5).center) == 2


# =====================================================================================
# CourseConfig.pads — empty by default, appended LAST (AC1 byte-identity)
# =====================================================================================
def test_default_course_has_zero_pads() -> None:
    """AC1: the default course has NO pads, so ``CourseConfig()`` stays UC-15-identical."""
    assert CourseConfig().pads == ()


def test_default_course_still_has_no_obstacles_and_three_gates() -> None:
    """Adding ``pads`` must not disturb the existing default course (UC-15 parity)."""
    course = CourseConfig()
    assert course.obstacles == ()
    assert course.num_gates == 3


def test_course_config_default_equality_is_stable() -> None:
    """Two default courses are equal — a frozen-dataclass proxy for byte-identity (AC1)."""
    assert CourseConfig() == CourseConfig()


def test_pads_appended_after_obstacles_positionally() -> None:
    """AC1: field order is ...obstacles, pads — pads is the LAST field.

    A pre-UC-16 positional call that ended at ``obstacles`` is unshifted, and a positional
    call may now pass ``pads`` immediately after ``obstacles``.
    """
    fields = [f.name for f in dataclasses.fields(CourseConfig)]
    assert fields[-1] == "pads"
    assert fields.index("obstacles") == fields.index("pads") - 1

    obstacles = (ObstacleSpec(center=(3.0, 1.5), radius=0.3, height=2.5),)
    pads = (PadSpec(center=(0.0, 0.0), radius=0.5),)
    # Positional: start, gates, finish_x, floor_z, ceiling_z, obstacles, pads
    course = CourseConfig((0.0, 0.0, 1.0), (), 7.0, 0.0, 2.5, obstacles, pads)
    assert course.pads == pads
    assert course.obstacles == obstacles


def test_pre_uc16_positional_call_is_unshifted() -> None:
    """A UC-15-style positional call (ending at ``obstacles``) still works and gets empty pads."""
    obstacles = (ObstacleSpec(center=(3.0, 1.5), radius=0.3, height=2.5),)
    course = CourseConfig((0.0, 0.0, 1.0), (), 7.0, 0.0, 2.5, obstacles)
    assert course.obstacles == obstacles
    assert course.pads == ()


def test_course_accepts_explicit_pads() -> None:
    pads = (PadSpec(center=(0.0, 0.0), radius=0.5), PadSpec(center=(6.5, 0.0), radius=0.4))
    assert CourseConfig(pads=pads).pads == pads


# =====================================================================================
# DockConfig — conservative, documented thresholds (AC2)
# =====================================================================================
def test_dock_config_defaults() -> None:
    dock = DockConfig()
    assert dock.max_dock_descent_speed == 0.5
    assert dock.max_dock_tilt == pytest.approx(0.2618, abs=1e-4)


def test_dock_config_is_frozen() -> None:
    dock = DockConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        dock.max_dock_descent_speed = 5.0  # type: ignore[misc]


def test_dock_descent_default_is_conservative() -> None:
    """The default descent cap is a gentle-touchdown speed (fails safe toward crash)."""
    assert 0.0 < DockConfig().max_dock_descent_speed <= 1.0


def test_dock_tilt_default_is_about_fifteen_degrees() -> None:
    import math

    assert math.degrees(DockConfig().max_dock_tilt) == pytest.approx(15.0, abs=0.5)


# =====================================================================================
# EnvConfig.dock — appended LAST with an all-default value (AC1/AC2 byte-identity)
# =====================================================================================
def test_env_config_has_default_dock() -> None:
    assert EnvConfig().dock == DockConfig()


def test_env_config_early_termination_is_last_field() -> None:
    """UC-25 (retargets the UC-19 last-field test): ``early_termination`` is appended **after**
    ``damage`` (itself after ``battery`` / ``dock`` / ``obstacle_vision``), so every UC-15..19
    positional/keyword call is unshifted and ``EnvConfig()`` stays byte-identical. ``damage`` is
    now second-to-last."""
    fields = [f.name for f in dataclasses.fields(EnvConfig)]
    assert fields[-1] == "early_termination"
    # The append-last chain is preserved: early_termination ← damage ← battery ← dock ← obstacle.
    assert fields.index("damage") == fields.index("early_termination") - 1
    assert fields.index("battery") == fields.index("damage") - 1
    assert fields.index("dock") == fields.index("battery") - 1
    assert fields.index("obstacle_vision") == fields.index("dock") - 1


def test_env_config_default_course_has_no_pads() -> None:
    """AC1/AC8: a default env carries an empty pad set (dock path never taken)."""
    assert EnvConfig().course.pads == ()


def test_env_config_default_equality_is_stable() -> None:
    assert EnvConfig() == EnvConfig()


# =====================================================================================
# default_pad_course / single_pad_course — factory shapes (AC1/AC9)
# =====================================================================================
def test_default_pad_course_has_pads_and_default_gates() -> None:
    """AC1: the manual default pad set — the standard 3-gate course plus fixed floor pads."""
    course = default_pad_course()
    assert len(course.pads) == 2
    assert all(isinstance(p, PadSpec) for p in course.pads)
    # It layers pads onto the standard default geometry (gates/obstacles unchanged).
    assert course.num_gates == 3
    assert course.obstacles == ()


def test_default_pad_course_pads_are_on_the_floor_corridor() -> None:
    """The fixed pads sit at the spawn and near the finish (documented, tunable constants)."""
    centers = {p.center for p in default_pad_course().pads}
    assert (0.0, 0.0) in centers  # start spawn
    assert (6.5, 0.0) in centers  # just past the last gate, near the finish


def test_single_pad_course_shape() -> None:
    """AC9: a minimal one-gate, one-pad fixture for hermetic docking trajectories."""
    course = single_pad_course()
    assert course.num_gates == 1
    assert len(course.pads) == 1
    assert course.pads[0].center == (0.0, 0.0)
    assert course.pads[0].radius == 0.5


def test_single_pad_course_is_overridable() -> None:
    course = single_pad_course(pad_center=(2.0, -1.0), pad_radius=0.8, floor_z=0.5)
    assert course.pads[0].center == (2.0, -1.0)
    assert course.pads[0].radius == 0.8
    assert course.floor_z == 0.5


def test_pad_courses_do_not_disturb_non_pad_factories() -> None:
    """The UC-15 obstacle course and the UC-09 single-gate course remain pad-free (parity)."""
    assert default_obstacle_course().pads == ()
    assert single_gate_course().pads == ()


def test_gate_spec_still_defaults_unchanged() -> None:
    """Sanity: adding PadSpec did not perturb the neighbouring GateSpec default."""
    assert GateSpec(center=(1.0, 2.0, 3.0)).aperture == 0.6


# =====================================================================================
# BatteryConfig — off by default, documented tunable knobs (UC-17 AC1/AC2/AC3/AC5/AC6)
# =====================================================================================
def test_battery_config_is_off_by_default() -> None:
    """AC5: battery is disabled by default so ``EnvConfig()`` is byte-identical to UC-16."""
    assert BatteryConfig().enabled is False
    assert EnvConfig().battery == BatteryConfig()
    assert EnvConfig().battery.enabled is False


def test_battery_config_default_knobs() -> None:
    """Documented tunable defaults (drain rates, knee, empty_factor)."""
    b = BatteryConfig()
    assert b.idle_rate == 0.005
    assert b.throttle_rate == 0.01
    assert b.knee == 0.2
    assert b.empty_factor == 0.3


def test_battery_config_is_frozen() -> None:
    b = BatteryConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.enabled = True  # type: ignore[misc]


def test_battery_empty_factor_below_hover_threshold() -> None:
    """AC3: base TWR is 2 → hover needs ceiling_factor >= 0.5; empty_factor < 0.5 makes an empty
    battery unable to hover (soft depletion via the crash path)."""
    assert BatteryConfig().empty_factor < 0.5


def test_ceiling_factor_full_at_and_above_knee() -> None:
    """AC2/AC6: ceiling factor is exactly 1.0 at full charge and flat 1.0 at/above the knee."""
    b = BatteryConfig()
    assert b.ceiling_factor(1.0) == 1.0
    assert b.ceiling_factor(b.knee) == 1.0
    assert b.ceiling_factor(0.5) == 1.0


def test_ceiling_factor_empty_equals_empty_factor() -> None:
    """AC2: at empty charge the ceiling factor is exactly ``empty_factor``."""
    b = BatteryConfig()
    assert b.ceiling_factor(0.0) == b.empty_factor


def test_ceiling_factor_monotone_nondecreasing() -> None:
    """AC2: ceiling factor is monotone non-decreasing in charge (lower battery ⇒ lower ceiling)."""
    b = BatteryConfig()
    charges = [i / 50.0 for i in range(51)]  # 0.0 .. 1.0
    factors = [b.ceiling_factor(c) for c in charges]
    assert all(y >= x for x, y in zip(factors, factors[1:], strict=False))
    # Strictly lower below the knee than at full.
    assert b.ceiling_factor(0.1) < b.ceiling_factor(1.0)


def test_ceiling_factor_linear_ramp_below_knee() -> None:
    """Below the knee the factor is a straight line from ``empty_factor`` (0) to 1.0 (knee)."""
    b = BatteryConfig()
    mid = b.knee / 2.0
    expected = b.empty_factor + (1.0 - b.empty_factor) * (mid / b.knee)
    assert b.ceiling_factor(mid) == pytest.approx(expected)


# =====================================================================================
# UC-18 — recharge pads: new config fields + off-by-default byte-identity (AC1/AC3/AC5)
# =====================================================================================
# --- PadSpec.rechargeable — a recharge tag, appended LAST, default False (AC1/AC5) ---------
def test_pad_spec_rechargeable_defaults_false() -> None:
    """AC1/AC5: a plain pad is not a recharge pad — the flag defaults ``False``."""
    assert PadSpec(center=(0.0, 0.0), radius=0.5).rechargeable is False


def test_pad_spec_repairable_is_the_last_field() -> None:
    """UC-19 (retargets the UC-18 last-field test): ``repairable`` is appended **after**
    ``rechargeable`` so every UC-16/17/18 positional ``PadSpec(center, radius[, rechargeable])``
    call is unshifted (byte-identity). ``rechargeable`` is now second-to-last."""
    fields = [f.name for f in dataclasses.fields(PadSpec)]
    assert fields[-1] == "repairable"
    assert fields.index("rechargeable") == fields.index("repairable") - 1
    assert fields.index("radius") == fields.index("rechargeable") - 1


def test_pad_spec_positional_ctor_is_unshifted_and_byte_identical() -> None:
    """AC5: a positional two-arg PadSpec equals the explicit non-recharge pad (bit-for-bit)."""
    assert PadSpec((1.5, -2.0), 0.5) == PadSpec(center=(1.5, -2.0), radius=0.5, rechargeable=False)


def test_pad_spec_can_be_tagged_rechargeable() -> None:
    """AC1: the flag is settable and does not perturb the geometry fields."""
    pad = PadSpec(center=(2.0, 0.0), radius=0.5, rechargeable=True)
    assert pad.rechargeable is True
    assert pad.center == (2.0, 0.0)
    assert pad.radius == 0.5


# --- BatteryConfig.recharge_rate — appended LAST, net-positive default (AC1) ---------------
def test_battery_config_recharge_rate_default() -> None:
    """AC1: the documented default recharge rate (fraction of full charge per second)."""
    assert BatteryConfig().recharge_rate == 0.5


def test_battery_config_recharge_rate_is_the_last_field() -> None:
    """UC-18: ``recharge_rate`` is appended last so ``BatteryConfig()`` — and therefore
    ``EnvConfig()`` — stays byte-identical to UC-17."""
    fields = [f.name for f in dataclasses.fields(BatteryConfig)]
    assert fields[-1] == "recharge_rate"


def test_recharge_rate_satisfies_the_net_positive_invariant() -> None:
    """AC1 (documented invariant): the default recharge rate strictly exceeds the maximum docked
    drain (``idle_rate + throttle_rate``), so a docked step nets a gain and the pad actually
    fills — otherwise it could never refill."""
    b = BatteryConfig()
    assert b.recharge_rate > b.idle_rate + b.throttle_rate


# --- EpisodeConfig.recharge_step_allowance — appended LAST, default 400 --------------------
def test_episode_config_recharge_step_allowance_default() -> None:
    assert EpisodeConfig().recharge_step_allowance == 400


def test_episode_config_repair_step_allowance_is_the_last_field() -> None:
    """UC-19 (retargets the UC-18 last-field test): ``repair_step_allowance`` is appended **after**
    ``recharge_step_allowance`` so ``EpisodeConfig()`` stays byte-identical to UC-09/18.
    ``recharge_step_allowance`` is now second-to-last."""
    fields = [f.name for f in dataclasses.fields(EpisodeConfig)]
    assert fields[-1] == "repair_step_allowance"
    assert fields.index("recharge_step_allowance") == fields.index("repair_step_allowance") - 1


# --- RandomizationConfig recharge axis — off/neutral defaults, appended LAST (AC3/AC5) -----
def test_randomization_config_recharge_axis_defaults() -> None:
    """AC5: the recharge axis is off by default with neutral energy-model constants."""
    r = RandomizationConfig()
    assert r.enable_recharge is False
    assert r.recharge_nominal_speed == 2.0
    assert r.recharge_nominal_throttle == 0.5
    assert r.recharge_pad_radius == 0.5
    assert r.recharge_energy_margin == 1.5


def test_randomization_recharge_fields_appended_in_order() -> None:
    """AC5: the five recharge fields are appended after the UC-15 obstacle fields (field-order
    compatibility with UC-15/16/17 positional/keyword construction). UC-24 appends the two repair
    fields AFTER them (see :func:`test_randomization_repair_fields_appended_last_in_order`), so the
    recharge block is no longer the tail — but it still follows ``obstacle_clearance`` in order."""
    fields = [f.name for f in dataclasses.fields(RandomizationConfig)]
    i = fields.index("enable_recharge")
    assert fields[i : i + 5] == [
        "enable_recharge",
        "recharge_nominal_speed",
        "recharge_nominal_throttle",
        "recharge_pad_radius",
        "recharge_energy_margin",
    ]
    # obstacle_clearance (the last UC-15 field) still precedes the recharge block.
    assert fields.index("obstacle_clearance") < fields.index("enable_recharge")


# --- UC-24 RandomizationConfig repair axis — off/neutral defaults, appended LAST -----------
def test_randomization_config_repair_axis_defaults() -> None:
    """UC-24 AC1/AC5: the repair axis is off by default with a neutral pad radius, so a default
    ``RandomizationConfig`` places no repair pad and stays byte-identical to UC-18."""
    r = RandomizationConfig()
    assert r.enable_repair is False
    assert r.repair_pad_radius == 0.5


def test_randomization_repair_fields_appended_last_in_order() -> None:
    """UC-24 AC1: the two repair fields (``enable_repair``, ``repair_pad_radius``) are appended
    LAST — after the whole UC-18 recharge block — so every UC-15/16/17/18 positional/keyword
    construction of ``RandomizationConfig`` is unshifted and byte-identical."""
    fields = [f.name for f in dataclasses.fields(RandomizationConfig)]
    assert fields[-2:] == ["enable_repair", "repair_pad_radius"]
    # The repair fields follow the last recharge field.
    assert fields.index("recharge_energy_margin") < fields.index("enable_repair")


def test_default_env_config_is_stable_under_uc24_repair_fields() -> None:
    """UC-24 AC6: with the two new repair fields at their defaults, ``RandomizationConfig()`` /
    ``EnvConfig()`` equality (a frozen-dataclass byte-identity proxy) still holds."""
    assert RandomizationConfig() == RandomizationConfig()
    assert EnvConfig().randomization == RandomizationConfig()


def test_energy_margin_is_conservative() -> None:
    """AC4 safety direction: the margin is > 1 so the guard over-estimates path energy (it never
    under-provisions recharge pads)."""
    assert RandomizationConfig().recharge_energy_margin > 1.0


# --- Off-by-default byte-identity of the whole config (AC5) --------------------------------
def test_default_env_config_is_stable_under_uc18_fields() -> None:
    """AC5: with every UC-18 field at its default, ``EnvConfig()`` equality (a frozen-dataclass
    proxy for byte-identity) still holds — the new fields perturb nothing by default."""
    assert EnvConfig() == EnvConfig()
    assert EnvConfig().battery == BatteryConfig()
    assert EnvConfig().episode == EpisodeConfig()
    assert EnvConfig().course.pads == ()


# =====================================================================================
# UC-19 — DamageConfig: off by default, documented tunable knobs + the load-bearing
# invariant + authority_factor contract (AC1/AC3)
# =====================================================================================
# --- PadSpec.repairable — a repair tag, appended LAST, default False, independent (AC1/AC4) -
def test_pad_spec_repairable_defaults_false() -> None:
    """AC1/AC4: a plain pad is not a repair pad — the flag defaults ``False``."""
    assert PadSpec(center=(0.0, 0.0), radius=0.5).repairable is False


def test_pad_spec_positional_ctor_unshifted_and_byte_identical_under_repairable() -> None:
    """AC1: a positional two- or three-arg PadSpec equals the explicit non-repair pad (bit-for-
    bit) — appending ``repairable`` last unshifts every UC-16/17/18 constructor call."""
    assert PadSpec((1.5, -2.0), 0.5) == PadSpec(
        center=(1.5, -2.0), radius=0.5, rechargeable=False, repairable=False
    )
    assert PadSpec((1.5, -2.0), 0.5, True) == PadSpec(
        center=(1.5, -2.0), radius=0.5, rechargeable=True, repairable=False
    )


def test_pad_spec_can_be_tagged_repairable_independently_of_rechargeable() -> None:
    """AC4: ``repairable`` is settable, does not perturb geometry, and is independent of
    ``rechargeable`` — a pad may recharge, repair, both, or neither."""
    pad = PadSpec(center=(2.0, 0.0), radius=0.5, repairable=True)
    assert pad.repairable is True
    assert pad.rechargeable is False
    assert pad.center == (2.0, 0.0) and pad.radius == 0.5
    both = PadSpec(center=(2.0, 0.0), radius=0.5, rechargeable=True, repairable=True)
    assert both.rechargeable is True and both.repairable is True


# --- DamageConfig — off by default, documented tunable knobs (AC1/AC3) ---------------------
def test_damage_config_is_off_by_default() -> None:
    """AC1: damage is disabled by default so ``EnvConfig()`` is byte-identical to UC-18."""
    assert DamageConfig().enabled is False
    assert EnvConfig().damage == DamageConfig()
    assert EnvConfig().damage.enabled is False


def test_damage_config_default_knobs() -> None:
    """Documented tunable defaults (per-contact loss, authority floor, repair rate)."""
    d = DamageConfig()
    assert d.damage_per_contact == 0.34
    assert d.min_authority == 1.0
    assert d.repair_rate == 0.5


def test_damage_config_is_frozen() -> None:
    d = DamageConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.enabled = True  # type: ignore[misc]


def test_min_authority_is_strictly_below_base_max_body_rate() -> None:
    """AC3 (the documented load-bearing invariant): ``min_authority < BASE_MAX_BODY_RATE``. This is
    what makes the ``integrity == 1.0`` **enabled** path byte-identical to the disabled path — at
    full integrity ``max(min_authority, max_body_rate * 1.0) == max_body_rate`` ONLY if the floor
    sits below the base rate. Hard-asserted per the plan (developer seed + challenger fold)."""
    assert DamageConfig().min_authority < BASE_MAX_BODY_RATE
    # And strictly positive (a floored-but-flyable handicap, never zero authority).
    assert DamageConfig().min_authority > 0.0


# --- authority_factor — linear, monotone, exactly 1.0 at full integrity (AC3) --------------
def test_authority_factor_is_exactly_one_at_full_integrity() -> None:
    """AC3: ``authority_factor(1.0) == 1.0`` **exactly** (not approx) — the invariant the
    full-integrity byte-identity relies on."""
    assert DamageConfig().authority_factor(1.0) == 1.0


def test_authority_factor_is_monotone_non_decreasing() -> None:
    """AC3: the factor is monotone non-decreasing in integrity (more damage ⇒ less authority)."""
    d = DamageConfig()
    integrities = [i / 50.0 for i in range(51)]  # 0.0 .. 1.0
    factors = [d.authority_factor(x) for x in integrities]
    assert all(y >= x for x, y in zip(factors, factors[1:], strict=False))
    # Strictly lower at low integrity than at full.
    assert d.authority_factor(0.1) < d.authority_factor(1.0)


def test_authority_factor_is_linear_identity_in_range() -> None:
    """AC3: the documented shape is linear ``f = integrity`` on ``[0, 1]``."""
    d = DamageConfig()
    for x in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert d.authority_factor(x) == pytest.approx(x)


def test_authority_factor_clamps_out_of_range_inputs() -> None:
    """AC3: defensively clamped to ``[0, 1]`` (integrity is already clamped upstream)."""
    d = DamageConfig()
    assert d.authority_factor(-0.5) == 0.0
    assert d.authority_factor(1.5) == 1.0
    assert isinstance(d.authority_factor(0.5), float)


# --- EpisodeConfig.repair_step_allowance — appended LAST, default 400 ----------------------
def test_episode_config_repair_step_allowance_default() -> None:
    """UC-19: a documented repair-detour budget, mirroring ``recharge_step_allowance``."""
    assert EpisodeConfig().repair_step_allowance == 400


# --- default_repair_course / single_repair_pad_course — factory shapes (AC7) ---------------
def test_default_repair_course_has_on_corridor_obstacles_and_one_repair_pad() -> None:
    """AC7: the damage-heavy default course — on-corridor pillars (the damage source) plus exactly
    one repairable pad mid-corridor and a plain landing pad near the finish."""
    course = default_repair_course()
    assert course.num_gates == 3  # layered onto the standard default geometry
    assert len(course.obstacles) >= 1  # on-corridor pillars shed integrity
    repair_pads = [p for p in course.pads if p.repairable]
    assert len(repair_pads) == 1
    # The mid-corridor pad is the repair pad; there is also at least one non-repair control pad.
    assert any(not p.repairable for p in course.pads)


def test_default_repair_course_obstacles_sit_on_the_corridor() -> None:
    """AC7: the pillars sit near ``y ≈ 0`` (on the forward corridor) so a straight run contacts
    several in a row — unlike the UC-15 off-corridor default obstacles."""
    course = default_repair_course()
    assert all(abs(o.center[1]) <= 0.6 for o in course.obstacles)


def test_single_repair_pad_course_shape_and_overridable() -> None:
    """AC7 fixture: a minimal one-gate, one-repair-pad course; obstacles injectable for damage."""
    course = single_repair_pad_course()
    assert course.num_gates == 1
    assert len(course.pads) == 1
    assert course.pads[0].repairable is True
    assert course.obstacles == ()  # no damage source unless injected
    # Injecting obstacles + overriding geometry is honoured.
    obst = (ObstacleSpec(center=(1.5, 0.0), radius=0.3, height=2.5),)
    c2 = single_repair_pad_course(pad_center=(2.0, -1.0), pad_radius=0.8, obstacles=obst)
    assert c2.pads[0].center == (2.0, -1.0) and c2.pads[0].radius == 0.8
    assert c2.pads[0].repairable is True
    assert c2.obstacles == obst


def test_repair_factories_do_not_disturb_non_repair_factories() -> None:
    """The UC-15/16 factories stay repair-free (parity)."""
    assert all(not p.repairable for p in default_pad_course().pads)
    assert single_gate_course().pads == ()
    assert default_obstacle_course().pads == ()


# --- Off-by-default byte-identity of the whole config under UC-19 fields (AC1/AC8) ----------
def test_default_env_config_is_stable_under_uc19_fields() -> None:
    """AC1/AC8: with every UC-19 field at its default, ``EnvConfig()`` equality still holds — the
    new ``damage`` / ``repairable`` / ``repair_step_allowance`` fields are inert by default."""
    assert EnvConfig() == EnvConfig()
    assert EnvConfig().damage == DamageConfig()
    assert EnvConfig().damage.enabled is False
    assert EnvConfig().episode.repair_step_allowance == 400
    assert EnvConfig().course.pads == ()
