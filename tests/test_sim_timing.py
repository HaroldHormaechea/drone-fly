"""UC67 AC5: control-rate and latency randomization (``drone_fly.sim.timing``).

Per env and episode: a control period from a Hz range or set, a base action delay and a skip
probability; per tick: the delay plus jitter (clamped at 0) and a Bernoulli skipped tick whose
transition lasts two periods [C3]. ``fixed`` mode is the legacy single rate with no delay. The
defaults cover the live harness latency measured in the cited UC65/UC66 run folder.
"""

from __future__ import annotations

from importlib import resources

import pytest
import torch

from drone_fly.sim import timing as timing_mod
from drone_fly.sim.timing import LATENCY_SOURCE, TIMING_LOG_COLUMNS, ControlTiming, TimingConfig

N = 5000


def _timing(config=None, n=N, seed=0):
    return ControlTiming(config or TimingConfig(), n, torch.device("cpu"),
                         torch.Generator().manual_seed(seed))  # fmt: skip


def test_default_period_is_drawn_over_20_to_50_hz():
    t = _timing()
    hz = 1.0 / t.period_s
    assert hz.min() >= 20.0 - 1e-9 and hz.max() <= 50.0 + 1e-9
    assert hz.min() < 21.0 and hz.max() > 49.0  # the whole range is used
    assert hz.mean().item() == pytest.approx(35.0, abs=1.0)


def test_rate_set_draws_only_the_listed_rates():
    t = _timing(TimingConfig(control_hz_set=(25.0, 50.0)))
    hz = torch.round(1.0 / t.period_s)
    assert set(hz.tolist()) == {25.0, 50.0}
    assert (hz == 25.0).float().mean().item() == pytest.approx(0.5, abs=0.05)


def test_default_base_delay_is_drawn_over_0_to_60_ms():
    t = _timing()
    assert t.delay_s.min() >= 0.0 and t.delay_s.max() <= 0.060 + 1e-12
    assert t.delay_s.min() < 0.001 and t.delay_s.max() > 0.059


def test_jitter_is_per_tick_and_the_delay_is_clamped_at_zero():
    t = _timing(TimingConfig(action_delay_s=(0.0, 0.0), delay_jitter_s=0.01))
    _, d1, _ = t.tick()
    _, d2, _ = t.tick()
    assert torch.all(d1 >= 0.0) and torch.all(d2 >= 0.0)
    assert (d1 == 0.0).float().mean().item() == pytest.approx(0.5, abs=0.05)  # half clamped
    assert not torch.equal(d1, d2)  # drawn per tick
    t = _timing(TimingConfig(action_delay_s=(0.03, 0.03), delay_jitter_s=0.005))
    _, d, _ = t.tick()
    assert d.mean().item() == pytest.approx(0.03, abs=5e-4)
    assert d.std().item() == pytest.approx(0.005, rel=0.1)


def test_skipped_ticks_follow_the_skip_probability_and_double_dt():
    t = _timing(TimingConfig(skip_prob=(0.3, 0.3)))
    dt, _, skipped = t.tick()
    assert skipped.float().mean().item() == pytest.approx(0.3, abs=0.03)
    torch.testing.assert_close(dt[skipped], 2.0 * t.period_s[skipped])
    torch.testing.assert_close(dt[~skipped], t.period_s[~skipped])
    # the default draws a per-episode probability in 0..35 %
    d = _timing()
    assert d.skip_prob.min() >= 0.0 and d.skip_prob.max() <= 0.35 + 1e-12
    assert d.skip_prob.max() > 0.34


def test_no_skip_when_the_probability_is_zero():
    t = _timing(TimingConfig(skip_prob=(0.0, 0.0)))
    for _ in range(5):
        _, _, skipped = t.tick()
        assert not skipped.any()


def test_fixed_mode_is_one_rate_with_no_delay_and_no_skip():
    t = _timing(TimingConfig.fixed(50.0), n=16)
    for _ in range(3):
        dt, delay, skipped = t.tick()
        torch.testing.assert_close(dt, torch.full((16,), 0.02, dtype=torch.float64))
        assert torch.all(delay == 0.0) and not skipped.any()
    cfg = TimingConfig.fixed(50.0)
    assert cfg.max_period_s == pytest.approx(0.02) and cfg.max_delay_s == 0.0


def test_masked_resample_keeps_the_other_envs():
    t = _timing(n=8)
    before = (t.period_s.clone(), t.delay_s.clone(), t.skip_prob.clone())
    mask = torch.tensor([True] * 4 + [False] * 4)
    t.resample(mask)
    for cur, old in zip((t.period_s, t.delay_s, t.skip_prob), before, strict=True):
        assert torch.equal(cur[~mask], old[~mask])
        assert not torch.equal(cur[mask], old[mask])


def test_same_generator_seed_is_deterministic():
    a, b = _timing(seed=3, n=64), _timing(seed=3, n=64)
    for _ in range(3):
        for x, y in zip(a.tick(), b.tick(), strict=True):
            assert torch.equal(x, y)


def test_log_columns():
    t = _timing(n=4)
    cols = t.columns()
    assert cols.shape == (4, len(TIMING_LOG_COLUMNS))
    assert TIMING_LOG_COLUMNS == ("control_hz", "action_delay_s", "skip_prob")
    torch.testing.assert_close(cols[:, 0], 1.0 / t.period_s)
    torch.testing.assert_close(cols[:, 1], t.delay_s)


@pytest.mark.parametrize(
    "bad",
    [
        {"mode": "jittery"},
        {"control_hz": (50.0, 20.0)},
        {"control_hz": (0.0, 20.0)},
        {"control_hz_set": ()},
        {"control_hz_set": (25.0, -1.0)},
        {"action_delay_s": (0.05, 0.01)},
        {"delay_jitter_s": -0.001},
        {"skip_prob": (0.0, 1.0)},
        {"fixed_hz": 0.0},
    ],
)
def test_config_is_validated(bad):
    with pytest.raises(ValueError):
        TimingConfig(**bad)


def test_from_dict_and_unknown_keys():
    cfg = TimingConfig.from_dict(
        {"control_hz": [30, 40], "action_delay_s": [0.01, 0.02], "skip_prob": [0, 0.1]}
    )
    assert cfg.control_hz == (30.0, 40.0) and cfg.action_delay_s == (0.01, 0.02)
    assert cfg.max_period_s == pytest.approx(1 / 30)
    assert cfg.max_delay_s == pytest.approx(0.02 + 5 * 0.005)
    with pytest.raises(ValueError, match="unknown timing keys"):
        TimingConfig.from_dict({"hz": 50})


def test_defaults_cover_the_measured_live_latency_and_cite_it():
    """Live run (07 checkpoint, preset liftoff): 50 Hz target / 36.9 Hz effective, 35 % missed
    deadlines, observation -> action ~28 ms p50 / 43 ms p95 (plan Analysis 7)."""
    d = TimingConfig()
    lo, hi = d.control_hz
    assert lo <= 36.9 <= hi and lo <= 50.0 <= hi
    assert d.action_delay_s[0] <= 0.028 and 0.043 <= d.action_delay_s[1]
    # 41 / 116 = 35.3 %: the approved plan rounds it to a 0.35 upper bound
    assert d.skip_prob[0] <= 0.35 <= d.skip_prob[1]
    assert 41 / 116 - d.skip_prob[1] < 0.005
    assert "liftoff-20261011T233116352187Z/latency_summary.json" in LATENCY_SOURCE
    assert LATENCY_SOURCE in (timing_mod.__doc__ or "")
    text = (
        resources.files("drone_fly.training")
        .joinpath("configs", "proc-course-dr.yaml")
        .read_text("utf-8")
    )
    assert LATENCY_SOURCE in text  # the training config comments cite the run folder
