"""UC-49 AC4 — end-to-end hermetic CI-gating T/W regression guard.

This is the strictly-stronger sibling of UC-48's helper-only unit test: it constructs the **real
``EnvConfig()`` default** and reads the **real ``sample_dynamics`` mass range** — the
``EnvConfig → RandomizationConfig.mass_factor_range × DynamicsParams.mass`` seam where the UC-47
free-fall bug actually lived — then resolves each end of that range through the *same* shared
:func:`drone_fly.adapter.dynamics_summary.drone_dynamics_summary` computation the TUI and the
recording use, on the pybullet path with ``tw_preserving=True``.

It asserts the configured pybullet drone stays **flyable and in-band** across the entire sampled
mass range: peak T/W ∈ [1.5, 2.5], with **T/W invariance under mass** as the hard property. A
regression that reintroduced the UC-47 collapse (sampler feeding an absolute ~1 kg mass to the
native-thrust body) would drop T/W to ≈0.24 and fail here at CI time.

Hermetic: pure computation over the real config objects — **imports no pybullet**.
"""

from __future__ import annotations

import pytest

from drone_fly.adapter.dynamics_summary import drone_dynamics_summary
from drone_fly.env.config import DynamicsParams, EnvConfig, RandomizationConfig

# The flyable band the whole UC-48/49 line defends: native CF2X peak T/W = (max_rpm/hover_rpm)**2
# = 1.5**2 = 2.25, comfortably inside [1.5, 2.5].
_TW_LO, _TW_HI = 1.5, 2.5


def _sampled_mass_range() -> tuple[float, float]:
    """The real per-episode mass range: base DynamicsParams.mass × RandomizationConfig factors.

    Read from the actual config objects (never hardcoded) so a future retune of either the base
    mass or the randomization factors is exercised by this guard automatically.
    """
    cfg = EnvConfig()
    assert isinstance(cfg.randomization, RandomizationConfig)
    base_mass = DynamicsParams().mass
    lo_factor, hi_factor = cfg.randomization.mass_factor_range
    return base_mass * lo_factor, base_mass * hi_factor


def test_sampled_mass_range_is_the_documented_default() -> None:
    """Sanity: the real config yields the documented [0.8, 1.2] kg range (base 1.0 × [0.8,1.2])."""
    lo, hi = _sampled_mass_range()
    assert lo == pytest.approx(0.8, rel=1e-9)
    assert hi == pytest.approx(1.2, rel=1e-9)


def test_pybullet_tw_stays_in_band_across_sampled_mass_range() -> None:
    """AC4: every end of the real sampled mass range resolves to a flyable in-band peak T/W."""
    lo, hi = _sampled_mass_range()
    max_body_rate = DynamicsParams().max_body_rate
    for mass in (lo, hi):
        s = drone_dynamics_summary(
            backend="pybullet",
            sampled_mass=mass,
            max_body_rate=max_body_rate,
            tw_preserving=True,
        )
        assert _TW_LO <= s.thrust_to_weight <= _TW_HI, (
            f"configured pybullet drone at sampled_mass={mass} kg has peak T/W="
            f"{s.thrust_to_weight:.4f}, outside the flyable band [{_TW_LO}, {_TW_HI}] — "
            "a UC-47-style mass/thrust-scale regression"
        )


def test_pybullet_tw_is_mass_invariant_across_the_range() -> None:
    """AC4 hard property: T/W is unchanged by mass under the UC-48 fix (invariance within 1e-6)."""
    lo, hi = _sampled_mass_range()
    max_body_rate = DynamicsParams().max_body_rate
    tw_lo = drone_dynamics_summary(
        backend="pybullet", sampled_mass=lo, max_body_rate=max_body_rate, tw_preserving=True
    ).thrust_to_weight
    tw_hi = drone_dynamics_summary(
        backend="pybullet", sampled_mass=hi, max_body_rate=max_body_rate, tw_preserving=True
    ).thrust_to_weight
    assert abs(tw_lo - tw_hi) < 1e-6, (
        f"T/W must be mass-invariant under tw_preserving=True; got {tw_lo:.6f} vs {tw_hi:.6f}"
    )


def test_default_env_config_has_tw_preserving_enabled() -> None:
    """The guard defends the default path — confirm the fix is actually on by default (UC-48)."""
    assert EnvConfig().pybullet_tw_preserving is True
