"""UC-56 AC1/AC2/AC6 — the Meteor75 Pro-analog nominal spec, determinism, and hover invariant.

The nominal is the single source of truth for the drone the acro policy trains against after UC-56
retargets the plant off the implicit CF2X reference. These tests pin:

* **AC1 (spec match)** — the nominal reproduces the sourced Meteor75 Pro-analog figures (mass,
  peak T/W, arm coordinate ≈ 26.5 mm from a 75 mm wheelbase, prop diameter 40 mm) within a
  documented tolerance, and the spec source is cited in the module docstring.
* **AC2 (deterministic nominal)** — the nominal is pure/deterministic and the ``DynamicsParams()``
  defaults (the randomization-off per-episode dynamics) equal that nominal on the pybullet axes.
* **AC6 (hover invariant)** — throttle 0.5 hovers at the nominal T/W (the shared
  :func:`drone_dynamics_summary` reports ``hover_throttle`` ≈ 0.5).

Hermetic: pure arithmetic over the ``drone_fly.adapter.meteor75`` constants + the shared summary —
**no pybullet import** anywhere (the sim body is ``# pragma: no cover``; behavioral verdict is
deferred to the owner's fresh GPU retrain, AC9).
"""

from __future__ import annotations

import math

import pytest

from drone_fly.adapter.dynamics_summary import drone_dynamics_summary
from drone_fly.adapter.meteor75 import (
    METEOR75_ARM,
    METEOR75_MASS,
    METEOR75_MOTOR_FRACTION,
    METEOR75_PROP_DIAMETER,
    METEOR75_TW,
    METEOR75_WHEELBASE,
    Meteor75Nominal,
    meteor75_nominal,
)
from drone_fly.env.config import DynamicsParams

# Sourced Meteor75 Pro-analog figures + documented tolerances (see module docstring provenance):
#   mass 30–36 g band (0.032 kg analog), whoop peak T/W ~2.0–3.0 (2.5 analog),
#   75 mm wheelbase (exact), 40 mm props (exact).
_MASS_BAND = (0.030, 0.036)  # kg — the AUW provenance band
_TW_BAND = (2.0, 3.0)  # dimensionless — the whoop peak-T/W analog band
_ARM_FROM_WHEELBASE = 0.075 / (2.0 * math.sqrt(2.0))  # ≈ 0.02652 m


# --------------------------------------------------------------------------- #
# AC1 — spec match (values within the documented tolerance)
# --------------------------------------------------------------------------- #
def test_nominal_mass_within_documented_band() -> None:
    """Nominal all-up mass sits inside the sourced 30–36 g Meteor75 Pro band."""
    assert _MASS_BAND[0] <= METEOR75_MASS <= _MASS_BAND[1]


def test_nominal_peak_tw_within_documented_band() -> None:
    """Nominal peak T/W is a documented whoop analog inside ~2.0–3.0."""
    assert _TW_BAND[0] <= METEOR75_TW <= _TW_BAND[1]


def test_nominal_arm_is_wheelbase_over_two_root_two() -> None:
    """The quad-X arm coordinate is ``wheelbase/(2·√2)`` ≈ 26.5 mm — NOT the 37.5 mm radius.

    This is the exact geometric relationship the point-mass inertia model consumes, so it is
    pinned relationally (never a magic 0.0265) and cross-checked against the 26.5 mm figure.
    """
    assert METEOR75_ARM == pytest.approx(_ARM_FROM_WHEELBASE, rel=1e-9)
    assert METEOR75_ARM == pytest.approx(0.0265, abs=5e-4)  # ≈ 26.5 mm, documented tolerance
    # Emphatically the per-axis offset, not the 37.5 mm motor radius.
    assert METEOR75_ARM < METEOR75_WHEELBASE / 2.0


def test_nominal_wheelbase_and_prop_are_exact_specs() -> None:
    """Wheelbase (75 mm) and prop diameter (40 mm) are exact catalogue specs."""
    assert METEOR75_WHEELBASE == pytest.approx(0.075, rel=1e-9)
    assert METEOR75_PROP_DIAMETER == pytest.approx(0.040, rel=1e-9)


def test_spec_source_is_cited_in_the_module() -> None:
    """AC1 requires the spec source cited in code — the module docstring names BetaFPV + a date."""
    import drone_fly.adapter.meteor75 as m

    doc = (m.__doc__ or "").lower()
    assert "betafpv" in doc
    assert "meteor75" in doc
    assert "accessed" in doc  # a dated access, so the provenance is auditable


# --------------------------------------------------------------------------- #
# AC2 — deterministic nominal (pure; DynamicsParams defaults == nominal)
# --------------------------------------------------------------------------- #
def test_meteor75_nominal_is_deterministic_and_frozen() -> None:
    """``meteor75_nominal()`` is pure — two calls return equal, immutable snapshots."""
    a = meteor75_nominal()
    b = meteor75_nominal()
    assert isinstance(a, Meteor75Nominal)
    assert a == b
    with pytest.raises((AttributeError, TypeError)):
        a.mass = 0.05  # type: ignore[misc]


def test_nominal_snapshot_matches_module_constants() -> None:
    """The nominal snapshot groups exactly the module constants (no drift between the two)."""
    n = meteor75_nominal()
    assert n.mass == METEOR75_MASS
    assert n.thrust_to_weight == METEOR75_TW
    assert n.wheelbase == METEOR75_WHEELBASE
    assert n.arm_length == METEOR75_ARM
    assert n.prop_diameter == METEOR75_PROP_DIAMETER
    assert n.motor_fraction == METEOR75_MOTOR_FRACTION


def test_dynamics_params_defaults_equal_the_meteor75_nominal() -> None:
    """AC2: the randomization-off per-episode dynamics (``DynamicsParams()``) ARE the nominal.

    The pybullet backend reads ``pybullet_mass_ratio`` / ``thrust_to_weight`` / ``arm_length``; the
    defaults must reproduce the Meteor75 nominal (ratio 1.0 → nominal mass, T/W and arm at nominal)
    so a run with randomization off flies the nominal plant.
    """
    d = DynamicsParams()
    assert d.pybullet_mass_ratio == pytest.approx(1.0)
    assert d.thrust_to_weight == pytest.approx(METEOR75_TW)
    assert d.arm_length == pytest.approx(METEOR75_ARM)


# --------------------------------------------------------------------------- #
# AC2/AC6 — the nominal plant through the shared summary (T/W + hover-at-0.5)
# --------------------------------------------------------------------------- #
def test_nominal_summary_reports_nominal_mass_and_tw() -> None:
    """Shared summary at the nominal (ratio 1.0, target_tw=None): mass 0.032 kg + T/W 2.5.

    Uses the DynamicsParams defaults as the caller would, so this is the end-to-end nominal the TUI,
    recording and CI guard all see.
    """
    d = DynamicsParams()
    s = drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=d.mass,  # ignored on the pybullet path
        max_body_rate=d.max_body_rate,
        tw_preserving=True,
        pybullet_mass_ratio=d.pybullet_mass_ratio,
        target_tw=None,  # None → the nominal Meteor75 T/W
        arm_length=d.arm_length,
    )
    assert s.applied_mass == pytest.approx(METEOR75_MASS, rel=1e-6)
    assert s.thrust_to_weight == pytest.approx(METEOR75_TW, rel=1e-6)
    assert s.arm_length == pytest.approx(METEOR75_ARM, rel=1e-6)


def test_nominal_hovers_at_half_throttle() -> None:
    """AC6: at the nominal, throttle 0.5 hovers (``hover_throttle`` ≈ 0.5 by construction)."""
    d = DynamicsParams()
    s = drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=d.mass,
        max_body_rate=d.max_body_rate,
        tw_preserving=True,
        pybullet_mass_ratio=d.pybullet_mass_ratio,
        arm_length=d.arm_length,
    )
    assert s.hover_throttle == pytest.approx(0.5, abs=1e-6)


def test_explicit_target_tw_at_nominal_matches_implicit_nominal_tw() -> None:
    """Passing ``target_tw=METEOR75_TW`` explicitly matches the ``None`` nominal path (consistency).

    ``target_tw=None`` reproduces the nominal band exactly; passing the nominal T/W explicitly must
    land on the same peak T/W, proving the two code paths agree at the nominal.
    """
    common = dict(
        backend="pybullet",
        sampled_mass=1.0,
        max_body_rate=4.0,
        tw_preserving=True,
        pybullet_mass_ratio=1.0,
    )
    implicit = drone_dynamics_summary(**common, target_tw=None)
    explicit = drone_dynamics_summary(**common, target_tw=METEOR75_TW)
    assert explicit.thrust_to_weight == pytest.approx(implicit.thrust_to_weight, rel=1e-9)
    assert explicit.thrust_to_weight == pytest.approx(METEOR75_TW, rel=1e-6)
