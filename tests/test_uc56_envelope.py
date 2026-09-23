"""UC-56 AC4/AC5/AC8 — wide T/W-preserving randomization envelope (whoop → 5" racer).

UC-56 widens the UC-48 T/W-preserving domain randomization to span a documented whoop → 5" racer
envelope on the **pybullet backend only**: three new axes — pybullet mass-ratio, peak T/W, and arm
length — sampled independently per episode from configurable ranges. These tests pin, all reading
the **real** :class:`RandomizationConfig` ranges (never hardcoded literals, so a retune is exercised
automatically):

* **AC4** — every sampled ``DynamicsParams`` lands inside the configured envelope ranges, and the
  draws are seed-deterministic (pinned draw order, appended last).
* **AC5** — peak T/W **tracks the target** across the T/W range, and is **mass-invariant at a fixed
  T/W** across the wide mass-ratio range (the UC-48 property holds for the new nominal end-to-end,
  through the shared :func:`drone_dynamics_summary`).
* **AC8** — the drone span the envelope covers is written down (``envelope_description`` renders the
  configured ranges as an absolute mass / T/W / arm span).

Hermetic: pure arithmetic over the real config objects + the shared summary — no pybullet import.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.adapter.dynamics_summary import drone_dynamics_summary
from drone_fly.adapter.meteor75 import METEOR75_MASS, envelope_description
from drone_fly.env.config import DynamicsParams, RandomizationConfig
from drone_fly.env.randomization import sample_dynamics


def _rcfg() -> RandomizationConfig:
    """The real dynamics-enabled randomization config (envelope ranges = the shipped defaults)."""
    return RandomizationConfig(enable_dynamics=True)


# --------------------------------------------------------------------------- #
# AC4 — sampled envelope axes stay inside the configured ranges
# --------------------------------------------------------------------------- #
def test_sampled_envelope_axes_within_configured_ranges() -> None:
    """Every sampled pybullet axis sits inside its configured range, read from the real config."""
    rcfg = _rcfg()
    base = DynamicsParams()
    rng = np.random.default_rng(56)
    mr_lo, mr_hi = rcfg.pybullet_mass_ratio_range
    tw_lo, tw_hi = rcfg.tw_range
    arm_lo, arm_hi = rcfg.arm_length_range
    for _ in range(3000):
        d = sample_dynamics(rng, rcfg, base)
        assert mr_lo <= d.pybullet_mass_ratio <= mr_hi
        assert tw_lo <= d.thrust_to_weight <= tw_hi
        assert arm_lo <= d.arm_length <= arm_hi


def test_sampled_envelope_spans_a_wide_range() -> None:
    """Sanity that the envelope is genuinely WIDE (whoop → 5" racer), not a near-nominal jitter.

    The mass-ratio hi/lo ratio and the T/W span are both large (read from the real config), which is
    the UC-56 intent — a broad distribution for later fine-tuning.
    """
    rcfg = _rcfg()
    mr_lo, mr_hi = rcfg.pybullet_mass_ratio_range
    tw_lo, tw_hi = rcfg.tw_range
    assert mr_hi / mr_lo >= 5.0  # a wide mass span (whoop → 5" racer is ~20×)
    assert tw_hi - tw_lo >= 2.0  # a wide T/W span


def test_envelope_draws_are_seed_deterministic() -> None:
    """Same seed → identical envelope draws (pinned draw order, AC4)."""
    rcfg = _rcfg()
    base = DynamicsParams()
    a = sample_dynamics(np.random.default_rng(11), rcfg, base)
    b = sample_dynamics(np.random.default_rng(11), rcfg, base)
    assert a.pybullet_mass_ratio == b.pybullet_mass_ratio
    assert a.thrust_to_weight == b.thrust_to_weight
    assert a.arm_length == b.arm_length
    c = sample_dynamics(np.random.default_rng(12), rcfg, base)
    # A different seed moves at least one envelope axis (the stream is actually seeded).
    assert (a.pybullet_mass_ratio, a.thrust_to_weight, a.arm_length) != (
        c.pybullet_mass_ratio,
        c.thrust_to_weight,
        c.arm_length,
    )


def test_envelope_axes_are_absolute_not_scaled_by_base() -> None:
    """The three pybullet axes are ABSOLUTE draws from their ranges — a non-unit base does not
    rescale them (contrast the multiplicative simple-backend axes)."""
    rcfg = _rcfg()
    scaled_base = DynamicsParams(pybullet_mass_ratio=7.0, thrust_to_weight=99.0, arm_length=1.23)
    rng = np.random.default_rng(3)
    mr_lo, mr_hi = rcfg.pybullet_mass_ratio_range
    tw_lo, tw_hi = rcfg.tw_range
    arm_lo, arm_hi = rcfg.arm_length_range
    for _ in range(500):
        d = sample_dynamics(rng, rcfg, scaled_base)
        assert mr_lo <= d.pybullet_mass_ratio <= mr_hi
        assert tw_lo <= d.thrust_to_weight <= tw_hi
        assert arm_lo <= d.arm_length <= arm_hi


# --------------------------------------------------------------------------- #
# AC5 — peak T/W tracks the target and is mass-invariant at a fixed T/W
# --------------------------------------------------------------------------- #
def _peak_tw(mass_ratio: float, target_tw: float) -> float:
    """Peak T/W the shared summary resolves for a pybullet plant at (mass_ratio, target_tw)."""
    return drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=1.0,  # ignored on the pybullet path
        max_body_rate=4.0,
        tw_preserving=True,
        pybullet_mass_ratio=mass_ratio,
        target_tw=target_tw,
    ).thrust_to_weight


def test_peak_tw_tracks_target_across_the_tw_range() -> None:
    """Peak T/W equals the requested target across the configured T/W range (T/W is a real axis)."""
    tw_lo, tw_hi = _rcfg().tw_range
    for target in np.linspace(tw_lo, tw_hi, 7):
        assert _peak_tw(mass_ratio=1.0, target_tw=float(target)) == pytest.approx(
            float(target), rel=1e-6
        )


def test_peak_tw_is_mass_invariant_at_fixed_target_across_wide_mass_range() -> None:
    """AC5: at a FIXED target T/W, peak T/W is invariant as mass sweeps the whole (wide) range."""
    rcfg = _rcfg()
    mr_lo, mr_hi = rcfg.pybullet_mass_ratio_range
    target = 0.5 * (rcfg.tw_range[0] + rcfg.tw_range[1])  # a mid-range target T/W
    peaks = [
        _peak_tw(mass_ratio=float(mr), target_tw=target) for mr in np.linspace(mr_lo, mr_hi, 9)
    ]
    # Every peak equals the target and they all agree to within 1e-6 (hard mass-invariance).
    for p in peaks:
        assert p == pytest.approx(target, rel=1e-6)
    assert max(peaks) - min(peaks) < 1e-6


def test_applied_mass_scales_with_ratio_off_the_meteor75_nominal() -> None:
    """Mass DOES vary with the ratio (× the Meteor75 nominal) even though T/W stays fixed."""
    lo = drone_dynamics_summary(
        backend="pybullet", sampled_mass=1.0, max_body_rate=4.0, pybullet_mass_ratio=1.0
    )
    hi = drone_dynamics_summary(
        backend="pybullet", sampled_mass=1.0, max_body_rate=4.0, pybullet_mass_ratio=10.0
    )
    assert lo.applied_mass == pytest.approx(METEOR75_MASS, rel=1e-6)
    assert hi.applied_mass == pytest.approx(10.0 * METEOR75_MASS, rel=1e-6)


def test_every_envelope_sample_resolves_to_its_own_target_tw() -> None:
    """End-to-end: a sampled (mass_ratio, T/W) pair resolves to a peak T/W == the sampled target,
    across many draws — so a randomized-heavier drone stays flyable at its intended T/W (AC5)."""
    rcfg = _rcfg()
    base = DynamicsParams()
    rng = np.random.default_rng(99)
    for _ in range(500):
        d = sample_dynamics(rng, rcfg, base)
        peak = _peak_tw(mass_ratio=d.pybullet_mass_ratio, target_tw=d.thrust_to_weight)
        assert peak == pytest.approx(d.thrust_to_weight, rel=1e-6)
        assert peak >= 1.0  # every sampled plant can hover (T/W >= 1)


# --------------------------------------------------------------------------- #
# AC8 — the envelope span is written down
# --------------------------------------------------------------------------- #
def test_envelope_description_reflects_the_configured_ranges() -> None:
    """``envelope_description`` renders the CONFIGURED ranges (not a hardcoded string) as an
    absolute mass / T/W / arm span, so the documented drone range always matches the real config."""
    rcfg = _rcfg()
    text = envelope_description(
        mass_ratio_range=rcfg.pybullet_mass_ratio_range,
        tw_range=rcfg.tw_range,
        arm_length_range=rcfg.arm_length_range,
    )
    assert "whoop" in text.lower() and '5"' in text
    # The absolute mass span implied by the ratio range × nominal mass appears (in grams).
    lo_g = round(METEOR75_MASS * rcfg.pybullet_mass_ratio_range[0] * 1000)
    hi_g = round(METEOR75_MASS * rcfg.pybullet_mass_ratio_range[1] * 1000)
    assert f"{lo_g}" in text and f"{hi_g}" in text
    # The T/W range endpoints are named.
    assert f"{rcfg.tw_range[0]:g}" in text and f"{rcfg.tw_range[1]:g}" in text
    # The documented "not modelled" simplification (drag / motor lag) is called out.
    assert "drag" in text.lower()
