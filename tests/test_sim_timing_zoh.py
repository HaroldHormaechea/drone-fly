"""UC-87 AC1 / AC10: deploy-timing — zero-order hold, the delay buffer, and the triangular
actuation-delay distribution (``drone_fly.sim.timing`` + ``drone_fly.sim.plant``).

CPU-only, hermetic, no connectome: the control step is a ZERO-ORDER HOLD (the action applied by
:class:`~drone_fly.sim.plant.Plant` is held constant across the control period while the physics
advances in finer substeps; a skipped tick holds it for two periods), and the plant's delay line
applies a pushed command exactly at ``t_push + action_delay + channel_delay``. UC-87 widens the
control period (down to ~16.7 Hz / 60 ms) and the actuation delay (0..100 ms) and adds a
``triangular`` per-episode delay distribution peaked near the measured live ~75 ms.
"""

from __future__ import annotations

import math

import pytest
import torch

from drone_fly.sim.plant import Plant, PlantConfig
from drone_fly.sim.timing import ControlTiming, TimingConfig
from tests.sim_support import all_envs, meteor75, variant


def _timing(config=None, n=20000, seed=0):
    return ControlTiming(
        config or TimingConfig(), n, torch.device("cpu"), torch.Generator().manual_seed(seed)
    )


def _plant(profile, n=1, *, substeps=2):
    cfg = PlantConfig(dtype="float64", substeps=substeps)
    p = Plant(profile, n, "cpu", 0, cfg, 0.0)
    p.reset(all_envs(n), pos=torch.tensor([0.0, 0.0, 2.0]).expand(n, 3), resample=False)
    return p


# ----------------------------------------------------------------------------- ZOH

def test_control_period_is_held_zero_order_between_policy_steps():
    """A command pushed once is held (ZOH) over the whole control period and the NEXT period too
    when no new command is pushed; integrating finer (substeps) does not change the held command."""
    prof = variant(meteor75(), randomization={})  # no DR ⇒ nominal plant
    period = 1.0 / 16.7  # ~60 ms, the UC-87 low end
    # (a) held across the step vs. a direct, un-delayed command in force
    a, b = _plant(prof, substeps=4), _plant(prof, substeps=4)
    cg = torch.full((1,), 1.6, dtype=torch.float64)
    rates = torch.tensor([[1.0, -0.5, 0.3]], dtype=torch.float64)
    a.step(cg, rates, period, 0.0)
    b.active = b._physical(cg, rates)
    b.advance(period)
    for key in ("pos", "vel", "quat", "omega", "motor_g"):
        assert torch.equal(getattr(a, key), getattr(b, key)), key
    # (b) a second period with NO new push keeps applying the same command (ZOH hold)
    before = a.active.clone()
    a.advance(period)
    assert torch.equal(a.active, before)


def test_skipped_tick_holds_the_action_for_two_periods():
    """[C3] ``ControlTiming`` returns ``dt = 2 * period`` on a skipped tick — the ZOH of a missed
    deadline, the live failure UC-87 trains against."""
    t = _timing(TimingConfig(control_hz=(20.0, 20.0), skip_prob=(0.4, 0.4)), n=4000)
    dt, _, skipped = t.tick()
    torch.testing.assert_close(dt[skipped], 2.0 * t.period_s[skipped])
    torch.testing.assert_close(dt[~skipped], t.period_s[~skipped])
    assert skipped.float().mean().item() == pytest.approx(0.4, abs=0.03)


# ----------------------------------------------------------------------------- delay buffer

def test_delay_buffer_applies_the_command_at_push_time_plus_delay():
    """The plant's delay line applies a pushed command at ``t_push + action_delay`` (meteor75 has
    zero channel delay at nominal), independent of the ZOH substructure of the control step."""
    prof = variant(meteor75(), motor_lag_s=0.01, randomization={})
    p = _plant(prof, substeps=2)
    cg = torch.full((1,), 2.0, dtype=torch.float64)
    rates = torch.zeros(1, 3, dtype=torch.float64)
    # push with the UC-87 ~75 ms actuation delay; a 60 ms period leaves it pending after one step
    p.step(cg, rates, 0.060, action_delay_s=0.075)
    assert p.motor_g[0].item() == pytest.approx(1.0, abs=1e-12)  # not yet in force
    # advance to t = 0.120 (> 0.075): the command has been applied for 0.120 - 0.075 = 0.045 s
    p.advance(0.060)
    assert p.motor_g[0].item() == pytest.approx(2.0 - math.exp(-0.045 / 0.01), abs=1e-9)


# ----------------------------------------------------------------------------- triangular delay

def test_triangular_delay_is_peaked_at_the_mode_and_within_range():
    """UC-87's ``triangular`` per-episode delay distribution concentrates the base delay near the
    measured live ~75 ms instead of spreading it uniformly over 0..100 ms."""
    cfg = TimingConfig(
        control_hz=(50.0, 50.0),
        action_delay_s=(0.0, 0.100),
        action_delay_dist="triangular",
        action_delay_mode_s=0.075,
        delay_jitter_s=0.0,
    )
    assert cfg.delay_mode_s == pytest.approx(0.075)
    t = _timing(cfg, n=40000)
    d = t.delay_s
    assert d.min() >= 0.0 and d.max() <= 0.100 + 1e-12
    # triangular mean = (lo + mode + hi) / 3 = (0 + 0.075 + 0.1)/3
    assert d.mean().item() == pytest.approx((0.0 + 0.075 + 0.100) / 3.0, abs=2e-3)
    # density is higher near the mode than near either endpoint
    near_mode = ((d > 0.065) & (d < 0.085)).float().mean().item()
    near_lo = ((d >= 0.0) & (d < 0.020)).float().mean().item()
    near_hi = ((d > 0.090) & (d <= 0.100)).float().mean().item()
    assert near_mode > near_lo and near_mode > near_hi


def test_uniform_is_the_default_delay_distribution():
    cfg = TimingConfig(action_delay_s=(0.0, 0.100), delay_jitter_s=0.0)
    assert cfg.action_delay_dist == "uniform"
    t = _timing(cfg, n=40000)
    d = t.delay_s
    assert d.mean().item() == pytest.approx(0.050, abs=2e-3)  # uniform midpoint


def test_triangular_mode_must_lie_within_the_delay_range():
    with pytest.raises(ValueError):
        TimingConfig(action_delay_s=(0.0, 0.050), action_delay_mode_s=0.075)
    with pytest.raises(ValueError):
        TimingConfig(action_delay_dist="weird")


def test_widened_ranges_from_dict_reach_the_uc87_envelope():
    cfg = TimingConfig.from_dict(
        {
            "control_hz": [16.7, 50.0],
            "action_delay_s": [0.0, 0.100],
            "action_delay_dist": "triangular",
            "action_delay_mode_s": 0.075,
        }
    )
    assert cfg.control_hz == (16.7, 50.0)
    assert cfg.max_period_s == pytest.approx(1.0 / 16.7)  # ~60 ms
    assert cfg.action_delay_s == (0.0, 0.100) and cfg.action_delay_dist == "triangular"
    assert cfg.delay_mode_s == pytest.approx(0.075)


def test_triangular_is_deterministic_for_a_seed():
    cfg = TimingConfig(action_delay_s=(0.0, 0.1), action_delay_dist="triangular")
    a = _timing(cfg, n=256, seed=7)
    b = _timing(cfg, n=256, seed=7)
    assert torch.equal(a.delay_s, b.delay_s)
