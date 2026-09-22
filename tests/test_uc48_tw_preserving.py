"""UC-48 — hermetic guard for T/W-preserving pybullet dynamics randomization.

UC-47 proved the drone free-falls because domain randomization applies an *absolute*
~1 kg point mass to the CF2X body (native 0.027 kg) while leaving the mixer's RPM band
native-sized → T/W ≈ 0.24. UC-48 fixes this by reinterpreting the sampled mass as a
**CF2X-relative multiplier** and scaling the mixer's ``hover_rpm`` / ``max_rpm`` band with
``sqrt(mass_ratio)`` so peak T/W = ``(max_rpm/hover_rpm)²`` stays invariant.

These tests assert the **pure** T/W-preservation logic
(:func:`~drone_fly.adapter.pybullet_adapter.resolve_tw_preserving_dynamics` /
:func:`~drone_fly.adapter.pybullet_adapter.thrust_to_weight`) — the CI-gating surface
(AC2/AC3). They MUST NOT import pybullet: CI has no sim, and the whole point is that the
preservation property is a closed-form computation independent of the simulator. The
real-pybullet A/B confirmation (AC1/AC7) is non-gating and lives in
``docs/uc48-mass-thrust-tw-preserving.md`` + ``docs/uc48-tw-preserving-surface.json``.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from drone_fly.adapter.pybullet_adapter import (
    CF2X_HOVER_RPM,
    CF2X_MAX_RPM,
    CF2X_NATIVE_MASS,
    ResolvedPybulletDynamics,
    resolve_tw_preserving_dynamics,
    thrust_to_weight,
)
from drone_fly.adapter.simple import BASE_MASS

# The native CF2X peak T/W, computed from the module constants (NOT hard-coded): four rotors
# at CF2X_MAX_RPM against the native 0.027 kg weight. == (CF2X_MAX_RPM/CF2X_HOVER_RPM)² ≈ 2.25.
NATIVE_PEAK_TW = thrust_to_weight(CF2X_NATIVE_MASS, CF2X_MAX_RPM)
# The native hover T/W (four rotors at CF2X_HOVER_RPM against native weight) ≈ 1.0 by design.
NATIVE_HOVER_TW = thrust_to_weight(CF2X_NATIVE_MASS, CF2X_HOVER_RPM)

# The sampler's mass factor range (EnvConfig.randomization.mass_factor_range) × BASE_MASS, plus a
# deliberately heavier sweep well past 1.2× to prove the invariant holds for any randomized mass.
SAMPLER_FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2)
HEAVY_FACTORS = (1.5, 2.0, 3.0, 5.0)
ALL_FACTORS = SAMPLER_FACTORS + HEAVY_FACTORS


# --- AC2 / AC3: the T/W-preservation property (the HARD invariant) ---------------------
@pytest.mark.parametrize("factor", ALL_FACTORS)
def test_peak_tw_is_invariant_under_tw_preserving_scaling(factor: float) -> None:
    """Peak T/W is EXACTLY invariant (within 1e-6) across the whole randomized mass range.

    This is the core preservation property: scaling both hover_rpm and max_rpm by the same
    ``sqrt(mass_ratio)`` leaves ``(max_rpm/hover_rpm)²`` — and hence peak T/W — unchanged, so a
    randomized-heavier CF2X keeps its native ≈2.25 peak T/W instead of collapsing to free-fall.
    """
    sampled_mass = factor * BASE_MASS
    resolved = resolve_tw_preserving_dynamics(sampled_mass, tw_preserving=True)
    peak = thrust_to_weight(resolved.applied_mass, resolved.max_rpm)
    assert peak == pytest.approx(NATIVE_PEAK_TW, abs=1e-6)


@pytest.mark.parametrize("factor", ALL_FACTORS)
def test_peak_tw_in_sanity_band(factor: float) -> None:
    """AC1: peak T/W stays inside the healthy [1.5, 2.5] band for every randomized mass.

    Carry-forward #1: the band is AC1's [1.5, 2.5], NOT tightened to AC2's "~2.2". The native
    CF2X *computed* peak is 2.25 (AC2's ~2.2 is a measured-with-drag nominal, not a hard
    ceiling), so the hard property asserted above is invariance; this is only a sanity fence.
    """
    sampled_mass = factor * BASE_MASS
    resolved = resolve_tw_preserving_dynamics(sampled_mass, tw_preserving=True)
    peak = thrust_to_weight(resolved.applied_mass, resolved.max_rpm)
    assert 1.5 <= peak <= 2.5


@pytest.mark.parametrize("factor", ALL_FACTORS)
def test_hover_tw_is_unity_under_tw_preserving_scaling(factor: float) -> None:
    """Hover T/W ≈ 1.0 at the resolved hover_rpm for every randomized mass (AC1 hover leg).

    Hovering means the four rotors exactly cancel weight; because hover_rpm scales with
    ``sqrt(mass_ratio)`` alongside the applied mass, thrust at the resolved hover_rpm keeps
    matching weight regardless of the randomized mass.
    """
    sampled_mass = factor * BASE_MASS
    resolved = resolve_tw_preserving_dynamics(sampled_mass, tw_preserving=True)
    hover_tw = thrust_to_weight(resolved.applied_mass, resolved.hover_rpm)
    assert hover_tw == pytest.approx(1.0, abs=1e-3)
    # And it is invariant to the native hover T/W (both ≈ 1.0 by the same construction).
    assert hover_tw == pytest.approx(NATIVE_HOVER_TW, abs=1e-6)


# --- resolve_* mapping: sampled ~1 kg -> CF2X-relative applied mass --------------------
@pytest.mark.parametrize("factor", ALL_FACTORS)
def test_resolve_maps_sampled_mass_to_cf2x_relative(factor: float) -> None:
    """A sampled ~1 kg point-mass is reinterpreted as a CF2X-relative multiplier.

    ``mass_ratio = sampled_mass / BASE_MASS`` and ``applied_mass = CF2X_NATIVE_MASS · mass_ratio``
    — so the CF2X body stays at a physical ~0.027·ratio kg, never an unphysical ~1 kg quad.
    """
    sampled_mass = factor * BASE_MASS
    resolved = resolve_tw_preserving_dynamics(sampled_mass, tw_preserving=True)
    assert isinstance(resolved, ResolvedPybulletDynamics)
    expected_ratio = sampled_mass / BASE_MASS
    assert resolved.mass_ratio == pytest.approx(expected_ratio)
    assert resolved.applied_mass == pytest.approx(CF2X_NATIVE_MASS * expected_ratio)
    # RPM band scales by sqrt(ratio) off the native baseline (no compounding).
    assert resolved.hover_rpm == pytest.approx(CF2X_HOVER_RPM * expected_ratio**0.5)
    assert resolved.max_rpm == pytest.approx(CF2X_MAX_RPM * expected_ratio**0.5)


def test_resolve_at_base_mass_is_native() -> None:
    """At the base 1.0 kg sample (ratio 1.0) the applied mass is exactly native and the RPM
    band is unscaled — the fix is a no-op at the reference mass, by construction."""
    resolved = resolve_tw_preserving_dynamics(BASE_MASS, tw_preserving=True)
    assert resolved.mass_ratio == pytest.approx(1.0)
    assert resolved.applied_mass == pytest.approx(CF2X_NATIVE_MASS)
    assert resolved.hover_rpm == pytest.approx(CF2X_HOVER_RPM)
    assert resolved.max_rpm == pytest.approx(CF2X_MAX_RPM)


# --- AC-bug-lock: tw_preserving=False reproduces the UC-47 collapse (relational) -------
@pytest.mark.parametrize("sampled_mass", (0.95, 1.0, 1.064, 1.2))
def test_tw_preserving_false_reproduces_freefall(sampled_mass: float) -> None:
    """Bug-lock: with ``tw_preserving=False`` the sampled mass is applied ABSOLUTELY against the
    native, unscaled RPM band — the pre-UC-48 degenerate behavior.

    Relational (no magic number): peak T/W == ``native_peak · (native_mass / sampled_mass)`` and
    is ``< 0.1`` (unflyable free-fall) at ~1 kg sampled masses. If someone accidentally makes the
    opt-out path T/W-preserving too, this test fails.
    """
    resolved = resolve_tw_preserving_dynamics(sampled_mass, tw_preserving=False)
    # Absolute mass, native (unscaled) band, ratio pinned to 1.0 (inertia unchanged).
    assert resolved.applied_mass == pytest.approx(sampled_mass)
    assert resolved.hover_rpm == pytest.approx(CF2X_HOVER_RPM)
    assert resolved.max_rpm == pytest.approx(CF2X_MAX_RPM)
    assert resolved.mass_ratio == pytest.approx(1.0)

    peak = thrust_to_weight(resolved.applied_mass, resolved.max_rpm)
    expected = NATIVE_PEAK_TW * (CF2X_NATIVE_MASS / sampled_mass)
    assert peak == pytest.approx(expected, rel=1e-9)
    assert peak < 0.1  # unflyable — the UC-47 root cause, locked in for the opt-out path


def test_tw_preserving_false_vs_true_diverge_at_heavy_mass() -> None:
    """Direct A/B of the two modes at a heavy randomized mass: preserving stays flyable
    (~native peak) while the opt-out collapses to free-fall."""
    sampled_mass = 1.064  # the UC-47 episode_350 recorded mass
    on = resolve_tw_preserving_dynamics(sampled_mass, tw_preserving=True)
    off = resolve_tw_preserving_dynamics(sampled_mass, tw_preserving=False)
    peak_on = thrust_to_weight(on.applied_mass, on.max_rpm)
    peak_off = thrust_to_weight(off.applied_mass, off.max_rpm)
    assert peak_on == pytest.approx(NATIVE_PEAK_TW, abs=1e-6)
    assert peak_off < 0.1
    assert peak_on > 10 * peak_off


# --- AC3: hermetic import guard (no pybullet) ------------------------------------------
def test_import_does_not_pull_in_pybullet_current_session() -> None:
    """Importing the adapter module must not import pybullet in this session (it is lazy)."""
    import drone_fly.adapter.pybullet_adapter  # noqa: F401

    assert "pybullet" not in sys.modules, sorted(m for m in sys.modules if "pybullet" in m)


def test_import_does_not_pull_in_pybullet_fresh_subprocess() -> None:
    """Belt-and-braces: a fresh interpreter importing the module never loads pybullet.

    A subprocess guarantees the guard is not masked by another test in this session having
    already imported the sim; it is the real regression guard for the lazy-import design.
    """
    code = (
        "import sys\n"
        "import drone_fly.adapter.pybullet_adapter  # noqa: F401\n"
        "assert 'pybullet' not in sys.modules, "
        "sorted(m for m in sys.modules if 'pybullet' in m)\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"import pulled in pybullet or failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert "OK" in proc.stdout


# --- helper sanity: thrust_to_weight is the documented closed form ---------------------
def test_thrust_to_weight_closed_form() -> None:
    """``thrust_to_weight`` == 4·kf·rpm²/(m·g); peak/hover ratio == (max/hover)² by construction."""
    m, rpm, kf, g = 0.03, 15000.0, 3.16e-10, 9.8
    expected = 4.0 * kf * rpm**2 / (m * g)
    assert thrust_to_weight(m, rpm, kf=kf, g=g) == pytest.approx(expected)
    # Peak/hover T/W ratio is exactly (max_rpm/hover_rpm)² — the invariance lever.
    ratio = thrust_to_weight(CF2X_NATIVE_MASS, CF2X_MAX_RPM) / thrust_to_weight(
        CF2X_NATIVE_MASS, CF2X_HOVER_RPM
    )
    assert ratio == pytest.approx((CF2X_MAX_RPM / CF2X_HOVER_RPM) ** 2)


def test_numpy_import_is_available_but_pybullet_is_not() -> None:
    """The pure helpers are numpy-only — numpy is importable, pybullet is not required."""
    assert np is not None
    _ = resolve_tw_preserving_dynamics(1.0)  # smoke: callable with numpy-only deps
