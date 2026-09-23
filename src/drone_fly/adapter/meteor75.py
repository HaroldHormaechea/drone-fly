"""BetaFPV Meteor75 Pro dynamics nominal + point-mass inertia model (UC-56).

This module is the **single source of truth** for the Meteor75 Pro *analog* nominal that UC-56
retargets the simulated plant to (replacing the implicit CF2X reference as the drone the acro
policy trains against). It is a **pure, hermetic leaf**: it imports only :mod:`math`, defines
frozen numeric constants + two pure functions, and pulls in nothing from :mod:`drone_fly.env`
or pybullet, so both the env config layer and the CI tests can consume it without side effects.

Nominal source & provenance (AC1)
---------------------------------
Figures are a documented **analog** of the BetaFPV Meteor75 Pro (a 75 mm-wheelbase 1S
brushless "whoop" quad), sourced from the manufacturer product page:

    BetaFPV Meteor75 Pro Brushless Whoop Quadcopter
    https://betafpv.com/products/meteor75-pro-brushless-whoop-quadcopter
    (accessed 2026-09-23)

Per-figure provenance and tolerance:

* **Mass** ``METEOR75_MASS = 0.032 kg`` — take-off weight *analog*. The product page lists the
  bare-frame weight (~20 g) plus a 1S pack; real all-up weight varies **30–36 g** with the
  chosen battery (assumed **1S 300 mAh LiHV**) and any HD/naked-cam payload. 0.032 kg sits in
  that band; documented tolerance is that band (±~4 g).
* **Peak thrust-to-weight** ``METEOR75_TW = 2.5`` — a representative whoop figure for the
  1102-class ~18000–22000 KV motors + 40 mm tri-blade props on a 1S HV pack. Thrust is not
  published as a single number (it depends on KV, pack C-rating and prop), so 2.5 is a
  documented mid-range analog (whoops span ~2.0–3.0). This is the peak T/W the mixer reproduces
  exactly at the nominal (hover stays at throttle ≈ 0.5 by construction).
* **Wheelbase** ``METEOR75_WHEELBASE = 0.075 m`` — the model's defining 75 mm diagonal
  motor-to-motor span (exact spec). The quad-X **arm coordinate** (the per-axis Cartesian
  offset of each motor from the centre, i.e. the value the point-mass inertia model needs) is
  ``a = wheelbase / (2·√2) ≈ 0.0265 m`` — NOT the 0.0375 m motor *radius*.
* **Prop diameter** ``METEOR75_PROP_DIAMETER = 0.040 m`` — 40 mm props (exact spec). Retained
  for provenance/telemetry only; the thrust model is RPM²·KF and does **not** consume prop
  geometry, so this figure is documentation, not a physics input.

Analog caveat: the underlying pybullet body keeps the CF2X per-rotor thrust coefficient
(``KF``), and only the mass / inertia / RPM band are reparameterised (no URDF swap). At the
nominal, mass, hover-at-0.5 and peak T/W are therefore **exact**; the absolute per-rotor RPM
that produces that thrust is non-physical for a real Meteor75 ("analog"), which is fine — the
policy only ever sees the dimensionless CTBR interface, never raw RPM.

Inertia model (AC3)
-------------------
Rotational inertia is absent from spec sheets, so it is estimated with a **motor-position
point-mass model**: four motor masses ``m_motor = motor_fraction·M/4`` at the quad-X arm
positions ``(±a, ±a, 0)`` plus the remaining body mass modelled as a point at the origin
(which contributes **zero** rotational inertia — a documented simplification that
under-estimates inertia, the conservative direction for a controllability margin). The
diagonal is then::

    Ixx = Iyy = motor_fraction · M · a²
    Izz      = 2 · motor_fraction · M · a²

``motor_fraction`` (the share of all-up mass concentrated at the four arm tips — motors, props,
arm ends) defaults to ``0.5``: a documented assumption for a whoop where roughly half the mass
sits in the central battery/flight-controller stack and half at the motor pods. It scales the
inertia linearly, so a later refinement is a one-constant change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Gravity used for the Meteor75 nominal hover/T-W arithmetic. Matches the pybullet aviary's
#: ``CF2X_GRAVITY`` (kept in :mod:`drone_fly.adapter.pybullet_adapter`) so hover RPM derived here
#: is consistent with the body the sim actually simulates.
METEOR75_GRAVITY = 9.8  # m/s^2

#: All-up-weight analog (kg). See module docstring for the 30–36 g provenance band.
METEOR75_MASS = 0.032

#: Peak thrust-to-weight at full throttle (dimensionless). Documented whoop analog (~2.0–3.0).
METEOR75_TW = 2.5

#: Diagonal motor-to-motor span (m) — the model's defining 75 mm wheelbase (exact spec).
METEOR75_WHEELBASE = 0.075

#: Quad-X per-axis arm coordinate (m): each motor sits at ``(±a, ±a, 0)`` with
#: ``a = wheelbase / (2·√2)`` ≈ 0.0265 m. This is the Cartesian offset the inertia model needs,
#: NOT the 0.0375 m motor radius.
METEOR75_ARM = METEOR75_WHEELBASE / (2.0 * math.sqrt(2.0))

#: Prop diameter (m) — 40 mm (exact spec). Provenance/telemetry only; not a thrust-model input.
METEOR75_PROP_DIAMETER = 0.040

#: Share of all-up mass concentrated at the four motor pods for the point-mass inertia model.
#: Documented whoop assumption (~half central stack, ~half motor pods); scales inertia linearly.
METEOR75_MOTOR_FRACTION = 0.5


@dataclass(frozen=True)
class Meteor75Nominal:
    """Immutable Meteor75 Pro-analog nominal spec (AC1/AC2).

    Groups the sourced figures so a caller (or a test) can read the whole nominal from one
    object. All fields are the module constants above; :func:`meteor75_nominal` returns the
    canonical instance.
    """

    mass: float
    thrust_to_weight: float
    wheelbase: float
    arm_length: float
    prop_diameter: float
    motor_fraction: float
    gravity: float


def meteor75_nominal() -> Meteor75Nominal:
    """Return the canonical Meteor75 Pro-analog nominal (pure; the AC1/AC2 reference)."""
    return Meteor75Nominal(
        mass=METEOR75_MASS,
        thrust_to_weight=METEOR75_TW,
        wheelbase=METEOR75_WHEELBASE,
        arm_length=METEOR75_ARM,
        prop_diameter=METEOR75_PROP_DIAMETER,
        motor_fraction=METEOR75_MOTOR_FRACTION,
        gravity=METEOR75_GRAVITY,
    )


def motor_position_inertia(
    total_mass: float,
    arm_length: float,
    motor_fraction: float = METEOR75_MOTOR_FRACTION,
) -> tuple[float, float, float]:
    """Diagonal rotational inertia ``(Ixx, Iyy, Izz)`` from a motor-position point-mass model (AC3).

    Four equal point masses ``m_motor = motor_fraction·total_mass/4`` at the quad-X arm positions
    ``(±arm_length, ±arm_length, 0)``; the remaining central mass is a point at the origin (zero
    contribution). Pure and hermetic — a known geometry maps to a known diagonal, so the AC3 test
    can pin exact values. Scales correctly across the randomization envelope because both
    ``total_mass`` and ``arm_length`` are parameters.

    Derivation (point mass at ``(±a, ±a, 0)``, four of them, each ``m = motor_fraction·M/4``):

        Ixx = Σ m·(y² + z²) = 4·m·a² = motor_fraction·M·a²
        Iyy = Σ m·(x² + z²) = 4·m·a² = motor_fraction·M·a²
        Izz = Σ m·(x² + y²) = 4·m·(a² + a²) = 2·motor_fraction·M·a²
    """
    total_mass = float(total_mass)
    arm_length = float(arm_length)
    motor_fraction = float(motor_fraction)
    arm_mass = motor_fraction * total_mass  # combined mass at the four arm tips
    ixx = arm_mass * arm_length**2
    iyy = ixx
    izz = 2.0 * arm_mass * arm_length**2
    return (ixx, iyy, izz)


def envelope_description(
    *,
    mass_ratio_range: tuple[float, float],
    tw_range: tuple[float, float],
    arm_length_range: tuple[float, float],
    nominal_mass: float = METEOR75_MASS,
) -> str:
    """Human-readable description of the drone range the randomization envelope spans (AC8).

    Takes the configured ranges (the caller passes :class:`RandomizationConfig` defaults or a
    tuned set) so the text always reflects the *effective* envelope rather than a hard-coded
    string. Converts the pybullet mass-ratio range into the absolute all-up-mass span it implies
    relative to ``nominal_mass``, framing it as the intended **whoop → 5" racer** spectrum for
    later fine-tuning.
    """
    lo_mass = nominal_mass * float(mass_ratio_range[0])
    hi_mass = nominal_mass * float(mass_ratio_range[1])
    return (
        'Dynamics randomization envelope (pybullet-only; whoop → 5" racer):\n'
        f"  mass:  {lo_mass * 1000:.0f}–{hi_mass * 1000:.0f} g "
        f"(nominal {nominal_mass * 1000:.0f} g × {mass_ratio_range[0]:g}–{mass_ratio_range[1]:g})\n"
        f"  T/W:   {tw_range[0]:g}–{tw_range[1]:g} (peak thrust-to-weight, T/W-preserving)\n"
        f"  arm:   {arm_length_range[0] * 1000:.1f}–{arm_length_range[1] * 1000:.1f} mm "
        "(quad-X per-axis motor offset)\n"
        "  NOT modelled per-scale: aerodynamic drag (single linear-damping range) and motor "
        "response lag are held constant across the mass span (documented UC-56 simplification)."
    )


__all__ = [
    "METEOR75_GRAVITY",
    "METEOR75_MASS",
    "METEOR75_TW",
    "METEOR75_WHEELBASE",
    "METEOR75_ARM",
    "METEOR75_PROP_DIAMETER",
    "METEOR75_MOTOR_FRACTION",
    "Meteor75Nominal",
    "meteor75_nominal",
    "motor_position_inertia",
    "envelope_description",
]
