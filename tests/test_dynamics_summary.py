"""UC-49 AC1 + AC5 — the single shared drone-dynamics summary computation.

Covers :func:`drone_fly.adapter.dynamics_summary.drone_dynamics_summary` and its
:class:`DroneDynamicsSummary` result — the anti-drift keystone every UC-49 surface (TUI,
recording ``meta.drone_dynamics``, and the CI T/W guard) reads from.

* **AC1** — field values for a known ``(sampled_mass, tw_preserving)`` pair on **both** backends
  (pybullet reports the *resolved* applied mass, never the raw ~1 kg sampled mass); T/W-invariance
  under ``tw_preserving=True`` across the mass range; the degenerate T/W collapse under
  ``tw_preserving=False`` (the UC-47 bug-lock).
* **AC5** — the unflyable tripwire ``logging.warning`` fires for a below-floor configuration
  (``tw_preserving=False`` + heavy mass ⇒ T/W < 1.0) and does **not** fire for the healthy default.

Hermetic: pure arithmetic over the UC-48 constants — **no pybullet import** anywhere.
"""

from __future__ import annotations

import logging

import pytest

from drone_fly.adapter.dynamics_summary import (
    FLYABLE_TW_FLOOR,
    DroneDynamicsSummary,
    drone_dynamics_summary,
)

# The real per-episode dynamics defaults (kept in sync with drone_fly.env.config.DynamicsParams):
# base mass 1.0 kg, full-stick body rate 4.0 rad/s. Read structurally where it matters (AC4 test),
# stated as literals here so a drift in either constant is a loud, deliberate test edit.
_DEFAULT_MAX_BODY_RATE = 4.0

# Native CF2X peak T/W = (max_rpm / hover_rpm)**2 = 1.5**2 = 2.25, invariant under the UC-48 scale.
_NATIVE_PEAK_TW = 2.25


# --------------------------------------------------------------------------- #
# AC1 — field values on both backends
# --------------------------------------------------------------------------- #
def test_pybullet_summary_reports_resolved_applied_mass_not_sampled() -> None:
    """pybullet path resolves the sampled ~1 kg point-mass to the CF2X-relative applied mass."""
    s = drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=1.0,
        max_body_rate=_DEFAULT_MAX_BODY_RATE,
        tw_preserving=True,
    )
    assert isinstance(s, DroneDynamicsSummary)
    assert s.backend == "pybullet"
    # applied_mass = native 0.027 kg * mass_ratio(1.0) — NOT the raw 1.0 kg sampled mass (UC-47).
    assert s.applied_mass == pytest.approx(0.027, rel=1e-6)
    assert s.applied_mass < 0.1  # emphatically not ~1 kg
    # weight = applied_mass * CF2X gravity (9.8).
    assert s.weight == pytest.approx(0.027 * 9.8, rel=1e-6)
    assert s.thrust_to_weight == pytest.approx(_NATIVE_PEAK_TW, rel=1e-6)
    assert s.hover_throttle == pytest.approx(0.5, abs=1e-6)
    assert s.max_body_rate == pytest.approx(_DEFAULT_MAX_BODY_RATE)
    assert s.spawn_z is None  # not supplied


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
@pytest.mark.parametrize("sampled_mass", [0.8, 1.0, 1.2])
def test_tw_preserving_keeps_peak_tw_invariant_across_mass(sampled_mass: float) -> None:
    """Under ``tw_preserving=True`` the peak T/W is mass-invariant (the UC-48 fix)."""
    s = drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=sampled_mass,
        max_body_rate=4.0,
        tw_preserving=True,
    )
    assert s.thrust_to_weight == pytest.approx(_NATIVE_PEAK_TW, abs=1e-6)
    # applied mass DOES scale with the sampled multiplier (0.027 * ratio), unlike T/W.
    assert s.applied_mass == pytest.approx(0.027 * sampled_mass, rel=1e-6)


def test_tw_preserving_invariance_within_1e6_across_range_ends() -> None:
    """The two ends of the mass range agree on peak T/W to within 1e-6 (hard invariance)."""
    lo = drone_dynamics_summary(
        backend="pybullet", sampled_mass=0.8, max_body_rate=4.0, tw_preserving=True
    )
    hi = drone_dynamics_summary(
        backend="pybullet", sampled_mass=1.2, max_body_rate=4.0, tw_preserving=True
    )
    assert abs(lo.thrust_to_weight - hi.thrust_to_weight) < 1e-6


def test_tw_preserving_false_collapses_tw_with_mass() -> None:
    """Opt-out (``tw_preserving=False``) reproduces the UC-47 degenerate absolute-mass collapse."""
    s = drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=1.2,
        max_body_rate=4.0,
        tw_preserving=False,
    )
    # Heavy absolute mass on the native (unscaled) RPM band → free-fall T/W ≈ 0.05.
    assert s.applied_mass == pytest.approx(1.2, rel=1e-6)  # absolute, unresolved
    assert s.thrust_to_weight < 0.1
    assert s.thrust_to_weight == pytest.approx(0.0506, abs=5e-3)


# --------------------------------------------------------------------------- #
# AC5 — the unflyable tripwire warning
# --------------------------------------------------------------------------- #
def test_warns_when_configured_tw_below_floor(caplog: pytest.LogCaptureFixture) -> None:
    """A below-floor (unflyable) config emits a logging.warning naming the mass + T/W."""
    with caplog.at_level(logging.WARNING, logger="drone_fly.adapter.dynamics_summary"):
        s = drone_dynamics_summary(
            backend="pybullet",
            sampled_mass=1.2,
            max_body_rate=4.0,
            tw_preserving=False,
        )
    assert s.thrust_to_weight < FLYABLE_TW_FLOOR
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "an unflyable configuration must emit a logging.warning (AC5)"
    blob = " ".join(r.getMessage() for r in warnings).lower()
    # The message names the offending T/W and the flyable floor so the operator can diagnose it.
    assert "t/w" in blob
    assert "floor" in blob


def test_does_not_warn_for_healthy_default(caplog: pytest.LogCaptureFixture) -> None:
    """The healthy default (T/W = 2.25 ≥ floor) must NOT emit the unflyable warning."""
    with caplog.at_level(logging.WARNING, logger="drone_fly.adapter.dynamics_summary"):
        s = drone_dynamics_summary(
            backend="pybullet",
            sampled_mass=1.0,
            max_body_rate=4.0,
            tw_preserving=True,
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
