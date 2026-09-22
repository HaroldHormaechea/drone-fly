"""UC-47 — hermetic regression guard for the throttle→thrust pathway harness.

These tests encode the *structural* relationships the UC-47 characterization harness
(:mod:`drone_fly.diagnostics.thrust_pathway`) measures, so the finding is regression-guarded
in CI **without pybullet** (AC7 / AC9). They assert ONLY against the pure ``ctbr_to_rpm``
mixer analysis and the ``SimpleDroneAdapter`` probes — never the ``--backend pybullet`` paths.

The real-pybullet numbers behind the verdict (AC2/AC3/AC4/AC6/AC8) are non-gating: they live
in the committed harness output ``docs/uc47-thrust-pathway-surface.json`` and the findings
note ``docs/uc47-thrust-pathway-findings.md`` (CI has no pybullet, so they are reported, not
re-derived here).

Lazy-import contract (AC1/AC7): importing the harness module must NOT pull in ``pybullet`` —
the sim toolchain is imported lazily, only inside the ``--backend pybullet`` code paths. The
guard below verifies this in a fresh subprocess so it holds regardless of what the rest of
the suite already imported.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from drone_fly.diagnostics import thrust_pathway as tp


# --- AC1 / AC7: lazy-import contract ---------------------------------------------------
def test_import_does_not_pull_in_pybullet() -> None:
    """Importing the harness must not import pybullet (verified in a clean subprocess).

    A fresh interpreter guarantees the check is not masked by another test in the session
    having already imported pybullet. This is the real regression guard for the harness's
    hermetic-safe lazy-import design.
    """
    code = (
        "import sys\n"
        "import drone_fly.diagnostics.thrust_pathway  # noqa: F401\n"
        "assert 'pybullet' not in sys.modules, sorted(m for m in sys.modules if 'pybullet' in m)\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"harness import pulled in pybullet or failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert "OK" in proc.stdout


def test_module_object_has_no_pybullet_in_current_session() -> None:
    """Belt-and-braces: after importing the module in *this* process, pybullet is absent.

    (In hermetic CI pybullet is not installed at all, so this is trivially true; on a machine
    where pybullet *is* installed it confirms the top-level import graph stays sim-free.)
    """
    assert "pybullet" not in sys.modules


# --- AC3 (central question): pure-mixer collective desaturation ------------------------
def test_saturated_rate_bleeds_collective_thrust_direction_and_magnitude() -> None:
    """throttle=1.0 + rpy=±1.0 (full authority) desaturates the collective vs rpy=0.

    Direction: Σrpm²(rpy=±1) strictly < Σrpm²(rpy=0). Magnitude: the fractional bleed lands
    in the ~0.15–0.25 band expected at rate_gain=0.15, and exactly one rotor clips at max_rpm.
    """
    m = tp.mixer_collective_metrics(1.0, 1.0, 1.0)
    # Direction — the collective bleed is real and positive.
    assert m["collective_bleed_fraction"] > 0.0
    assert m["sum_sq"] < m["sum_sq_baseline_rpy0"]
    # Magnitude band.
    assert 0.15 <= m["collective_bleed_fraction"] <= 0.25
    # One over-max rotor clips down uncompensated (the desaturation mechanism).
    assert m["n_clipped"] == 1


def test_bleed_fraction_matches_measured_value() -> None:
    """Pin the exact bleed at the canonical (throttle=1, rpy=1, auth=1) cell (~0.208)."""
    m = tp.mixer_collective_metrics(1.0, 1.0, 1.0)
    assert m["collective_bleed_fraction"] == pytest.approx(0.208125, abs=1e-4)


def test_level_command_has_zero_bleed_and_no_clip() -> None:
    """rpy=0 is the baseline: no collective loss, no clipping at hover throttle."""
    m = tp.mixer_collective_metrics(0.5, 0.0, 1.0)
    assert m["collective_bleed_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert m["n_clipped"] == 0
    assert m["sum_sq"] == pytest.approx(m["sum_sq_baseline_rpy0"])


def test_lower_authority_reduces_bleed() -> None:
    """Scaling rpy down via attitude authority monotonically reduces collective bleed.

    At the UC-46 committed authority (~0.26) the bleed is small (~5.5% at 0.25); this is the
    quantitative basis for the plan's finding that the recorded run's mixer bleed was minor.
    """
    b_full = tp.mixer_collective_metrics(1.0, 1.0, 1.0)["collective_bleed_fraction"]
    b_half = tp.mixer_collective_metrics(1.0, 1.0, 0.5)["collective_bleed_fraction"]
    b_low = tp.mixer_collective_metrics(1.0, 1.0, 0.25)["collective_bleed_fraction"]
    assert b_full > b_half > b_low > 0.0
    assert b_low == pytest.approx(0.0552, abs=5e-3)


# --- AC7: ctbr_to_rpm purity / statelessness ------------------------------------------
def test_mixer_is_pure_and_stateless() -> None:
    """Two identical calls return byte-identical RPMs — the mixer holds no motor state."""
    act = tp.make_action(0.7, 0.3)
    r1 = tp.mixer_rpms(act)
    r2 = tp.mixer_rpms(act)
    assert np.array_equal(r1, r2)
    # A different command in between must not perturb a repeat of the first.
    _ = tp.mixer_rpms(tp.make_action(0.1, -0.9))
    r3 = tp.mixer_rpms(act)
    assert np.array_equal(r1, r3)


def test_make_action_does_not_mutate_and_authority_scales_only_rpy() -> None:
    """apply_authority scales rpy channels (1..3) only; throttle (index 0) is untouched."""
    raw = tp.make_action(0.8, 1.0)
    raw_copy = raw.copy()
    scaled = tp.apply_authority(raw, 0.5)
    # Input not mutated in place.
    assert np.array_equal(raw, raw_copy)
    # Throttle preserved; rpy halved (raw rpy=1.0 clipped to box then *0.5).
    assert scaled[0] == pytest.approx(0.8)
    assert scaled[1:4] == pytest.approx(np.array([0.5, 0.5, 0.5]))


# --- AC2 (hermetic contrast): simple baseline hovers@0.5 / climbs@1.0 ------------------
def test_simple_baseline_hovers_at_half_and_climbs_at_full() -> None:
    """SimpleDroneAdapter is mass-consistent: T/W==1.0 at throttle 0.5, ==2.0 at 1.0.

    This is the hermetic contrast that localizes the pybullet free-fall to the CF2X
    mass/thrust-scale mismatch — the simple point-mass baseline is correct by construction.
    """
    sweep = tp.simple_baseline_sweep()
    by_thr = {c["throttle"]: c for c in sweep}
    assert by_thr[0.5]["tw"] == pytest.approx(1.0, abs=1e-6)
    assert by_thr[1.0]["tw"] == pytest.approx(2.0, abs=1e-6)
    # Monotonic increase across the throttle grid.
    tws = [by_thr[t]["tw"] for t in sorted(by_thr)]
    assert tws == sorted(tws)


# --- AC4 (hermetic part): convexity — bang-bang raises mean thrust proxy ---------------
def test_bang_bang_throttle_raises_mean_collective_proxy() -> None:
    """thrust ∝ RPM² is convex and throttle=0 maps to a non-zero base RPM, so bang-bang
    (0↔1) has a HIGHER mean Σrpm² than steady throttle 0.5 — i.e. oscillation does not
    explain the free-fall. (The real-pybullet magnitude is in the non-gating surface.)
    """
    p0 = tp.collective_thrust_proxy(tp.mixer_rpms(tp.make_action(0.0, 0.0)))
    p1 = tp.collective_thrust_proxy(tp.mixer_rpms(tp.make_action(1.0, 0.0)))
    p_steady = tp.collective_thrust_proxy(tp.mixer_rpms(tp.make_action(0.5, 0.0)))
    assert 0.5 * (p0 + p1) > p_steady
    # throttle=0 is NOT clipped to zero RPM (the convexity premise).
    assert p0 > 0.0


# --- AC5: secondary suspects on the simple backend ------------------------------------
def test_latency_probe_shift_equals_latency_steps() -> None:
    """action latency_steps ∈ {0,1,2} produces an exact matching step-shift in the response."""
    probes = tp.latency_probe()
    shifts = {d["latency_steps"]: d["observed_shift"] for d in probes}
    assert shifts == {0: 0, 1: 1, 2: 2}


def test_battery_full_charge_is_negligible_low_charge_collapses() -> None:
    """UC-17 battery: full charge ceiling_factor==1.0 → ~0 early thrust delta; low charge
    collapses the ceiling — so battery is NOT the free-fall cause in a fresh episode.
    """
    b = tp.battery_probe()
    assert b["ceiling_factor_full"] == pytest.approx(1.0)
    assert b["delta_full_vs_off"] == pytest.approx(0.0, abs=1e-6)
    assert b["ceiling_factor_low"] < 1.0


def test_damage_full_integrity_is_thrust_noop() -> None:
    """UC-19 damage scales max_body_rate only, never thrust; full integrity is a no-op."""
    d = tp.damage_probe()
    assert d["authority_factor_full"] == pytest.approx(1.0)
    # Damage acts on max_body_rate, explicitly never on thrust.
    assert "max_body_rate" in d["affects"]
    assert "never thrust" in d["affects"]


# --- Reference constants stay in lock-step with the adapter ----------------------------
def test_reference_constants_are_documented_values() -> None:
    """The hermetic analysis pins the CF2X constants + rate gain the pybullet run used."""
    assert tp.RATE_GAIN == 0.15
    assert tp.CF2X_MASS == pytest.approx(0.027)
    assert tp.CF2X_HOVER_RPM == pytest.approx(14468.429, abs=1e-2)
    assert tp.CF2X_MAX_RPM == pytest.approx(21702.644, abs=1e-2)
    assert tp.DEFAULT_THROTTLES == (0.0, 0.25, 0.5, 0.75, 1.0)
