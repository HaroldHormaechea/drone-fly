"""UC-49 AC4 (+ UC-56 AC7) — end-to-end hermetic CI-gating T/W regression guard.

This is the strictly-stronger sibling of UC-48's helper-only unit test: it constructs the **real
``EnvConfig()`` default** and reads the **real UC-56 pybullet envelope** — the
``RandomizationConfig.pybullet_mass_ratio_range`` and ``.tw_range`` (× the Meteor75 nominal) seam
where the plant's mass and thrust-to-weight now live — then resolves each corner of that envelope
through the *same* shared :func:`drone_fly.adapter.dynamics_summary.drone_dynamics_summary`
computation the TUI and the recording use, on the pybullet path with ``tw_preserving=True``.

UC-56 retuned the guard to the new nominal + wide envelope: the drone mass axis is
``pybullet_mass_ratio`` (× the Meteor75 nominal 0.032 kg) and the peak T/W is an explicit target
axis (``tw_range``), so the guard now asserts (a) **peak T/W tracks the configured target** across
``tw_range`` and (b) **T/W is mass-invariant at a fixed target** across the wide mass-ratio range.
Everything is read from the real ``RandomizationConfig`` — nothing hardcoded — so a retune of the
envelope is exercised automatically. A regression that reintroduced a UC-47-style mass/thrust-scale
collapse (an absolute heavy mass on the nominal-thrust body) would drop T/W below 1.0 and fail here.

Hermetic: pure computation over the real config objects — **imports no pybullet**.
"""

from __future__ import annotations

import pytest

from drone_fly.adapter.dynamics_summary import drone_dynamics_summary
from drone_fly.adapter.meteor75 import METEOR75_MASS, METEOR75_TW
from drone_fly.env.config import DynamicsParams, EnvConfig, RandomizationConfig


def _envelope() -> RandomizationConfig:
    """The real dynamics randomization config; its envelope ranges are the shipped defaults."""
    cfg = EnvConfig()
    assert isinstance(cfg.randomization, RandomizationConfig)
    return cfg.randomization


def _peak_tw(mass_ratio: float, target_tw: float | None) -> float:
    """Peak T/W the shared summary resolves for a pybullet plant at (mass_ratio, target_tw)."""
    return drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=1.0,  # ignored on the pybullet path
        max_body_rate=DynamicsParams().max_body_rate,
        tw_preserving=True,
        pybullet_mass_ratio=mass_ratio,
        target_tw=target_tw,
    ).thrust_to_weight


# --------------------------------------------------------------------------- #
# The envelope the guard reads from the REAL config (nothing hardcoded)
# --------------------------------------------------------------------------- #
def test_envelope_ranges_are_read_from_the_real_config() -> None:
    """Sanity: the guard reads a genuinely wide envelope + the Meteor75 nominal, not stale CF2X."""
    env = _envelope()
    mr_lo, mr_hi = env.pybullet_mass_ratio_range
    tw_lo, tw_hi = env.tw_range
    # A wide envelope (whoop → 5" racer), and a nominal at the low end of the T/W range.
    assert mr_lo == pytest.approx(1.0) and mr_hi >= 5.0
    assert tw_lo <= METEOR75_TW <= tw_hi
    # DynamicsParams defaults ARE the nominal (ratio 1.0 → nominal mass, T/W at nominal).
    d = DynamicsParams()
    assert d.pybullet_mass_ratio == pytest.approx(1.0)
    assert d.thrust_to_weight == pytest.approx(METEOR75_TW)


# --------------------------------------------------------------------------- #
# AC7 — peak T/W tracks the configured target across tw_range
# --------------------------------------------------------------------------- #
def test_pybullet_peak_tw_tracks_target_across_tw_range() -> None:
    """AC7: at both ends of the real ``tw_range`` the resolved peak T/W equals the target."""
    tw_lo, tw_hi = _envelope().tw_range
    for target in (tw_lo, tw_hi):
        peak = _peak_tw(mass_ratio=1.0, target_tw=target)
        assert peak == pytest.approx(target, rel=1e-6), (
            f"peak T/W must track the configured target {target}; got {peak:.4f}"
        )


def test_nominal_default_path_matches_meteor75_tw() -> None:
    """The default (target_tw=None) path resolves to the Meteor75 nominal peak T/W (2.5)."""
    assert _peak_tw(mass_ratio=1.0, target_tw=None) == pytest.approx(METEOR75_TW, rel=1e-6)


# --------------------------------------------------------------------------- #
# AC7 — T/W is mass-invariant at a fixed target across the wide mass range
# --------------------------------------------------------------------------- #
def test_pybullet_tw_is_mass_invariant_across_the_wide_mass_range() -> None:
    """AC7 hard property: at a FIXED target T/W, peak T/W is unchanged across the whole mass-ratio
    range (the UC-48 invariance holds for the new nominal + wide envelope, within 1e-6)."""
    env = _envelope()
    mr_lo, mr_hi = env.pybullet_mass_ratio_range
    target = 0.5 * (env.tw_range[0] + env.tw_range[1])  # a mid-range target
    tw_lo = _peak_tw(mass_ratio=mr_lo, target_tw=target)
    tw_hi = _peak_tw(mass_ratio=mr_hi, target_tw=target)
    assert tw_lo == pytest.approx(target, rel=1e-6)
    assert abs(tw_lo - tw_hi) < 1e-6, (
        f"T/W must be mass-invariant at a fixed target; got {tw_lo:.6f} vs {tw_hi:.6f}"
    )


def test_pybullet_every_envelope_corner_is_flyable() -> None:
    """Across the four envelope corners (mass-ratio × T/W extremes) every plant can hover (T/W ≥ 1).

    A UC-47-style mass/thrust-scale regression (absolute heavy mass on the nominal-thrust body)
    would drop a corner below 1.0 and fail here at CI time.
    """
    env = _envelope()
    mr_lo, mr_hi = env.pybullet_mass_ratio_range
    tw_lo, tw_hi = env.tw_range
    for mass_ratio in (mr_lo, mr_hi):
        for target in (tw_lo, tw_hi):
            peak = _peak_tw(mass_ratio=mass_ratio, target_tw=target)
            assert peak == pytest.approx(target, rel=1e-6)
            assert peak >= 1.0, (
                f"corner (mass_ratio={mass_ratio}, target_tw={target}) is unflyable: T/W={peak:.4f}"
            )


def test_applied_mass_tracks_the_meteor75_nominal_times_ratio() -> None:
    """The applied body mass is the Meteor75 nominal × the mass-ratio (never the raw ~1 kg)."""
    env = _envelope()
    mr_hi = env.pybullet_mass_ratio_range[1]
    s = drone_dynamics_summary(
        backend="pybullet",
        sampled_mass=1.0,
        max_body_rate=DynamicsParams().max_body_rate,
        tw_preserving=True,
        pybullet_mass_ratio=mr_hi,
        target_tw=env.tw_range[1],
    )
    assert s.applied_mass == pytest.approx(METEOR75_MASS * mr_hi, rel=1e-6)


def test_default_env_config_has_tw_preserving_enabled() -> None:
    """The guard defends the default path — confirm the fix is actually on by default (UC-48)."""
    assert EnvConfig().pybullet_tw_preserving is True
