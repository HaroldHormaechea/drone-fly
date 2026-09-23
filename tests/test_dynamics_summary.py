"""UC-49 AC1 + AC5 — the single shared drone-dynamics summary computation.

Covers :func:`drone_fly.adapter.dynamics_summary.drone_dynamics_summary` and its
:class:`DroneDynamicsSummary` result — the anti-drift keystone every UC-49 surface (TUI,
recording ``meta.drone_dynamics``, and the CI T/W guard) reads from.

* **AC1** — field values for a known pybullet / simple case (pybullet reports the *resolved*
  applied mass off the **Meteor75 nominal**, never the raw ~1 kg sampled mass); T/W-invariance
  under ``tw_preserving=True`` across the mass-ratio range; the degenerate T/W collapse under
  ``tw_preserving=False`` (the UC-47 bug-lock).
* **AC5** — the unflyable tripwire ``logging.warning`` fires for a below-floor configuration
  (``tw_preserving=False`` + heavy mass ⇒ T/W < 1.0) and does **not** fire for the healthy default.

UC-56 retuned the pybullet path off the CF2X reference (0.027 kg / T/W 2.25) to the Meteor75
Pro-analog nominal (0.032 kg / T/W 2.5), and drives the mass axis from ``pybullet_mass_ratio``
(× the nominal mass) rather than the raw ``sampled_mass``. Nominal figures are read from the
``drone_fly.adapter.meteor75`` constants so a nominal retune is exercised automatically.

Hermetic: pure arithmetic over the UC-48/UC-56 constants — **no pybullet import** anywhere.
"""

from __future__ import annotations

import logging

import pytest

from drone_fly.adapter.dynamics_summary import (
    FLYABLE_TW_FLOOR,
    DroneDynamicsSummary,
    drone_dynamics_summary,
)
from drone_fly.adapter.meteor75 import METEOR75_ARM, METEOR75_MASS, METEOR75_TW

# The real per-episode dynamics defaults (kept in sync with drone_fly.env.config.DynamicsParams):
# full-stick body rate 4.0 rad/s. Stated as a literal so a drift is a loud, deliberate test edit.
_DEFAULT_MAX_BODY_RATE = 4.0

# UC-56: the pybullet path resolves off the **Meteor75 Pro-analog nominal** (mass 0.032 kg, peak
# T/W 2.5) with the mass axis driven by ``pybullet_mass_ratio`` (× the nominal). Read from the
# module constants (never hardcoded) so a nominal retune flows through automatically.
_NOMINAL_PEAK_TW = METEOR75_TW  # 2.5 — default (target_tw=None) peak T/W
_NOMINAL_MASS = METEOR75_MASS  # 0.032 kg — applied at mass-ratio 1.0
_CF2X_GRAVITY = 9.8  # the aviary gravity constant used on the pybullet path


# --------------------------------------------------------------------------- #
# AC1 — field values on both backends
# --------------------------------------------------------------------------- #
def test_pybullet_summary_reports_resolved_applied_mass_not_sampled() -> None:
    """pybullet path resolves off the Meteor75 nominal via the mass-ratio, not sampled mass."""
    s = drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=1.0,  # ignored on the pybullet path (mass comes from the ratio)
        max_body_rate=_DEFAULT_MAX_BODY_RATE,
        tw_preserving=True,
        pybullet_mass_ratio=1.0,
        arm_length=METEOR75_ARM,
    )
    assert isinstance(s, DroneDynamicsSummary)
    assert s.backend == "pybullet"
    # applied_mass = Meteor75 nominal 0.032 kg * mass_ratio(1.0) — NOT the raw 1.0 kg sampled mass.
    assert s.applied_mass == pytest.approx(_NOMINAL_MASS, rel=1e-6)
    assert s.applied_mass < 0.1  # emphatically not ~1 kg
    # weight = applied_mass * CF2X aviary gravity (9.8).
    assert s.weight == pytest.approx(_NOMINAL_MASS * _CF2X_GRAVITY, rel=1e-6)
    assert s.thrust_to_weight == pytest.approx(_NOMINAL_PEAK_TW, rel=1e-6)
    assert s.hover_throttle == pytest.approx(0.5, abs=1e-6)
    assert s.max_body_rate == pytest.approx(_DEFAULT_MAX_BODY_RATE)
    assert s.arm_length == pytest.approx(METEOR75_ARM, rel=1e-6)
    assert s.spawn_z is None  # not supplied


def test_pybullet_summary_arm_length_defaults_to_meteor75_nominal() -> None:
    """Omitted ``arm_length`` falls back to the Meteor75 nominal arm on the pybullet path."""
    s = drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=1.0,
        max_body_rate=_DEFAULT_MAX_BODY_RATE,
        pybullet_mass_ratio=1.0,
    )
    assert s.arm_length == pytest.approx(METEOR75_ARM, rel=1e-6)


def test_simple_summary_field_values() -> None:
    """simple backend: T/W preserved by construction (max_thrust = 2*m*g ⇒ peak T/W = 2)."""
    s = drone_dynamics_summary(
        backend="simple",
        sampled_mass=1.0,
        max_body_rate=_DEFAULT_MAX_BODY_RATE,
    )
    assert s.backend == "simple"
    assert s.applied_mass == pytest.approx(1.0)  # simple applies the sampled mass verbatim
    assert s.weight == pytest.approx(9.81, rel=1e-6)  # simple gravity 9.81
    assert s.thrust_to_weight == pytest.approx(2.0, rel=1e-6)
    assert s.hover_throttle == pytest.approx(0.5, abs=1e-6)
    assert s.max_body_rate == pytest.approx(_DEFAULT_MAX_BODY_RATE)
    assert s.arm_length is None  # arm is a pybullet-only descriptor


def test_summary_is_frozen() -> None:
    """The dataclass is immutable so a consumer can render/serialise it without defensive copies."""
    s = drone_dynamics_summary(backend="simple", sampled_mass=1.0, max_body_rate=4.0)
    with pytest.raises((AttributeError, TypeError)):
        s.applied_mass = 2.0  # type: ignore[misc]


def test_curriculum_knobs_carry_through_verbatim() -> None:
    """spawn_z is carried through unchanged for display + provenance. (UC-55 retired the
    attitude-authority knob along with the mixer it stood in for.)"""
    s = drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=1.0,
        max_body_rate=4.0,
        tw_preserving=True,
        spawn_z=1.25,
    )
    assert s.spawn_z == pytest.approx(1.25)


# --------------------------------------------------------------------------- #
# AC1 — T/W invariance (preserving) vs. degenerate collapse (opt-out)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mass_ratio", [1.0, 5.0, 20.0])
def test_tw_preserving_keeps_peak_tw_invariant_across_mass(mass_ratio: float) -> None:
    """Under ``tw_preserving=True`` the peak T/W is mass-invariant across the wide ratio range."""
    s = drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=1.0,
        max_body_rate=4.0,
        tw_preserving=True,
        pybullet_mass_ratio=mass_ratio,
    )
    assert s.thrust_to_weight == pytest.approx(_NOMINAL_PEAK_TW, abs=1e-6)
    # applied mass DOES scale with the ratio (0.032 * ratio), unlike T/W.
    assert s.applied_mass == pytest.approx(_NOMINAL_MASS * mass_ratio, rel=1e-6)


def test_tw_preserving_invariance_within_1e6_across_range_ends() -> None:
    """The two ends of the mass-ratio range agree on peak T/W to within 1e-6 (hard invariance)."""
    lo = drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=1.0,
        max_body_rate=4.0,
        tw_preserving=True,
        pybullet_mass_ratio=1.0,
    )
    hi = drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=1.0,
        max_body_rate=4.0,
        tw_preserving=True,
        pybullet_mass_ratio=20.0,
    )
    assert abs(lo.thrust_to_weight - hi.thrust_to_weight) < 1e-6


def test_tw_preserving_false_collapses_tw_with_mass() -> None:
    """Opt-out (``tw_preserving=False``) reproduces the UC-47 degenerate absolute-mass collapse.

    On this path the mass-ratio is applied as an ABSOLUTE body mass against the native (unscaled)
    Meteor75 RPM band, so a heavy value free-falls. Relational (no magic number): T/W is far below
    the flyable floor and shrinks as the absolute mass grows.
    """
    light = drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=1.0,
        max_body_rate=4.0,
        tw_preserving=False,
        pybullet_mass_ratio=1.0,
    )
    heavy = drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=1.0,
        max_body_rate=4.0,
        tw_preserving=False,
        pybullet_mass_ratio=1.2,
    )
    # Heavy absolute mass on the native (unscaled) RPM band → free-fall T/W well under 0.1.
    assert heavy.applied_mass == pytest.approx(1.2, rel=1e-6)  # absolute, unresolved
    assert heavy.thrust_to_weight < 0.1
    # T/W is inversely proportional to the absolute mass on the opt-out path.
    assert heavy.thrust_to_weight == pytest.approx(light.thrust_to_weight / 1.2, rel=1e-6)


# --------------------------------------------------------------------------- #
# AC5 — the unflyable tripwire warning
# --------------------------------------------------------------------------- #
def test_warns_when_configured_tw_below_floor(caplog: pytest.LogCaptureFixture) -> None:
    """A below-floor (unflyable) config emits a logging.warning naming the mass + T/W."""
    with caplog.at_level(logging.WARNING, logger="drone_fly.adapter.dynamics_summary"):
        s = drone_dynamics_summary(
            backend="pybullet",
            sampled_mass=1.0,
            max_body_rate=4.0,
            tw_preserving=False,
            pybullet_mass_ratio=1.2,
        )
    assert s.thrust_to_weight < FLYABLE_TW_FLOOR
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "an unflyable configuration must emit a logging.warning (AC5)"
    blob = " ".join(r.getMessage() for r in warnings).lower()
    # The message names the offending T/W and the flyable floor so the operator can diagnose it.
    assert "t/w" in blob
    assert "floor" in blob


def test_does_not_warn_for_healthy_default(caplog: pytest.LogCaptureFixture) -> None:
    """The healthy default (T/W = 2.5 ≥ floor) must NOT emit the unflyable warning."""
    with caplog.at_level(logging.WARNING, logger="drone_fly.adapter.dynamics_summary"):
        s = drone_dynamics_summary(
            backend="pybullet",
            sampled_mass=1.0,
            max_body_rate=4.0,
            tw_preserving=True,
            pybullet_mass_ratio=1.0,
        )
    assert s.thrust_to_weight >= FLYABLE_TW_FLOOR
    assert not [r for r in caplog.records if r.levelno == logging.WARNING], (
        "the healthy default must not trip the unflyable warning"
    )


def test_simple_default_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """The simple backend default (T/W = 2.0) is flyable → no warning."""
    with caplog.at_level(logging.WARNING, logger="drone_fly.adapter.dynamics_summary"):
        drone_dynamics_summary(backend="simple", sampled_mass=1.0, max_body_rate=4.0)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
