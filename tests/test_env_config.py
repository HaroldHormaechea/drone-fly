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

from drone_fly.env.config import (
    BatteryConfig,
    CourseConfig,
    DockConfig,
    EnvConfig,
    GateSpec,
    ObstacleSpec,
    PadSpec,
    default_obstacle_course,
    default_pad_course,
    single_gate_course,
    single_pad_course,
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


def test_env_config_battery_is_last_field() -> None:
    """UC-17: ``battery`` is appended after ``dock`` (which is after ``obstacle_vision``), so every
    UC-15/UC-16 positional/keyword call is unshifted and ``EnvConfig()`` stays byte-identical."""
    fields = [f.name for f in dataclasses.fields(EnvConfig)]
    assert fields[-1] == "battery"
    # dock is now second-to-last, still directly after obstacle_vision.
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
