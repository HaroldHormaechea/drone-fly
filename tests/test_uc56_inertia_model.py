"""UC-56 AC3 — rotational inertia from the motor-position point-mass model.

Rotational inertia is absent from spec sheets, so UC-56 estimates it from a quad-X point-mass
model: four motor masses ``m = motor_fraction·M/4`` at ``(±a, ±a, 0)`` plus a central body mass
modelled as a point at the origin (zero contribution). The closed-form diagonal is::

    Ixx = Iyy = motor_fraction · M · a²
    Izz      = 2 · motor_fraction · M · a²

These tests pin a **known geometry → known diagonal** (AC3), the scaling across the randomization
envelope (both ``total_mass`` and ``arm_length`` are parameters), and the physical relationships
(Izz == 2·Ixx for a planar X; the perpendicular-axis theorem for the four coplanar point masses).

Hermetic: :func:`drone_fly.adapter.meteor75.motor_position_inertia` is pure arithmetic — no
pybullet, no env import.
"""

from __future__ import annotations

import pytest

from drone_fly.adapter.meteor75 import (
    METEOR75_ARM,
    METEOR75_MASS,
    METEOR75_MOTOR_FRACTION,
    motor_position_inertia,
)


# --------------------------------------------------------------------------- #
# AC3 — known geometry maps to a known diagonal
# --------------------------------------------------------------------------- #
def test_known_geometry_maps_to_known_diagonal_round_numbers() -> None:
    """Pinned closed form on friendly numbers: M=1, a=1, fraction=1 → (1, 1, 2)."""
    ixx, iyy, izz = motor_position_inertia(total_mass=1.0, arm_length=1.0, motor_fraction=1.0)
    assert ixx == pytest.approx(1.0)
    assert iyy == pytest.approx(1.0)
    assert izz == pytest.approx(2.0)


def test_diagonal_matches_closed_form_at_the_meteor75_nominal() -> None:
    """At the Meteor75 nominal (M, a, motor_fraction) the diagonal equals the documented formula."""
    ixx, iyy, izz = motor_position_inertia(
        total_mass=METEOR75_MASS,
        arm_length=METEOR75_ARM,
        motor_fraction=METEOR75_MOTOR_FRACTION,
    )
    expected_planar = METEOR75_MOTOR_FRACTION * METEOR75_MASS * METEOR75_ARM**2
    assert ixx == pytest.approx(expected_planar, rel=1e-12)
    assert iyy == pytest.approx(expected_planar, rel=1e-12)
    assert izz == pytest.approx(2.0 * expected_planar, rel=1e-12)


def test_izz_is_twice_the_planar_axes_perpendicular_axis_theorem() -> None:
    """For four coplanar (z=0) point masses, Izz == Ixx + Iyy == 2·Ixx (perpendicular-axis)."""
    ixx, iyy, izz = motor_position_inertia(total_mass=0.032, arm_length=0.0265)
    assert ixx == pytest.approx(iyy, rel=1e-12)
    assert izz == pytest.approx(ixx + iyy, rel=1e-12)
    assert izz == pytest.approx(2.0 * ixx, rel=1e-12)


def test_default_motor_fraction_is_the_module_nominal() -> None:
    """Omitting ``motor_fraction`` uses the documented whoop default (== the module constant)."""
    with_default = motor_position_inertia(total_mass=0.05, arm_length=0.03)
    with_explicit = motor_position_inertia(
        total_mass=0.05, arm_length=0.03, motor_fraction=METEOR75_MOTOR_FRACTION
    )
    assert with_default == with_explicit


# --------------------------------------------------------------------------- #
# AC3 — scaling across the randomization envelope
# --------------------------------------------------------------------------- #
def test_inertia_scales_linearly_with_mass() -> None:
    """Doubling total mass doubles every diagonal entry (mass is a linear factor)."""
    base = motor_position_inertia(total_mass=0.032, arm_length=0.0265)
    heavy = motor_position_inertia(total_mass=0.064, arm_length=0.0265)
    for b, h in zip(base, heavy, strict=True):
        assert h == pytest.approx(2.0 * b, rel=1e-12)


def test_inertia_scales_quadratically_with_arm_length() -> None:
    """Doubling the arm coordinate quadruples every diagonal entry (a² dependence)."""
    base = motor_position_inertia(total_mass=0.032, arm_length=0.0265)
    long_arm = motor_position_inertia(total_mass=0.032, arm_length=0.053)
    for b, la in zip(base, long_arm, strict=True):
        assert la == pytest.approx(4.0 * b, rel=1e-12)


def test_inertia_scales_linearly_with_motor_fraction() -> None:
    """The motor-mass share is a linear factor on the whole diagonal."""
    half = motor_position_inertia(total_mass=0.032, arm_length=0.0265, motor_fraction=0.5)
    full = motor_position_inertia(total_mass=0.032, arm_length=0.0265, motor_fraction=1.0)
    for h, f in zip(half, full, strict=True):
        assert f == pytest.approx(2.0 * h, rel=1e-12)


@pytest.mark.parametrize(
    ("mass", "arm"),
    [(0.032, 0.0265), (0.20, 0.05), (0.64, 0.078)],  # whoop → 5" racer envelope corners
)
def test_diagonal_positive_and_ordered_across_envelope(mass: float, arm: float) -> None:
    """Across the whoop→5" envelope the diagonal is strictly positive and Izz > Ixx = Iyy."""
    ixx, iyy, izz = motor_position_inertia(total_mass=mass, arm_length=arm)
    assert ixx > 0.0 and iyy > 0.0 and izz > 0.0
    assert ixx == pytest.approx(iyy, rel=1e-12)
    assert izz > ixx
