"""UC-87 AC6 / AC10: the pure-CPU smoothness metrics (``drone_fly.sim.smooth_metrics``).

Exercised on SYNTHETIC raw-action command traces (no env, no torch, no RNG), so the numbers are
checked against closed-form expectations. These metric functions are the single source of truth
for the AC7 bars; the empirical numbers themselves are produced post-merge on the GPU box (D6).
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.sim import smooth_metrics as sm
from drone_fly.sim.smooth_metrics import (
    FLIP_RATE,
    MAX_BODY_RATE,
    SAT_RATE,
    aggregate_metrics,
    body_rates_from_raw,
    hf_power_share,
    mean_abs_delta,
    saturation_share,
    sign_flips_per_s,
    smoothness_metrics,
)


def test_body_rates_decode_uses_the_v2_tanh_times_max_rate_convention():
    raw = np.array([[0.5, 0.0, 1.0, -1.0]])  # a0 (collective) is ignored
    rates = body_rates_from_raw(raw)
    assert rates.shape == (1, 3)
    np.testing.assert_allclose(
        rates[0], [0.0, np.tanh(1.0) * MAX_BODY_RATE, np.tanh(-1.0) * MAX_BODY_RATE], atol=1e-12
    )


def test_mean_abs_delta_matches_hand_computation():
    # three ticks, one axis of interest; collective (col 0) must be ignored
    rates = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 3.0, 0.0]])
    # |Δ| over axes & ticks = (|1|+|0|+|0| + |2|+|0|+|0|) / (2 ticks * 3 axes) = 3/6
    assert mean_abs_delta(rates) == pytest.approx(0.5)


def test_mean_abs_delta_zero_for_a_constant_trace():
    rates = np.full((10, 3), 2.0)
    assert mean_abs_delta(rates) == 0.0


def test_mean_abs_delta_needs_two_samples():
    assert mean_abs_delta(np.zeros((1, 3))) == 0.0


def test_saturation_share_counts_samples_at_or_above_the_threshold():
    rates = np.array([[3.8, 0.0, -3.9], [3.7, 4.0, 0.0]])  # 3 of 6 at |rate| >= 3.8
    assert saturation_share(rates) == pytest.approx(3 / 6)
    assert SAT_RATE == 3.8


def test_saturation_share_empty_trace():
    assert saturation_share(np.zeros((0, 3))) == 0.0


def test_sign_flips_per_s_counts_genuine_reversals_above_flip_rate():
    dt = 0.1  # 10 Hz
    # axis 0: +, -, + → two reversals, both above FLIP_RATE; axis 1/2 constant
    rates = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    flips = sign_flips_per_s(rates, dt)
    duration = (3 - 1) * dt  # 0.2 s
    assert flips[0] == pytest.approx(2 / duration)
    assert flips[1] == 0.0 and flips[2] == 0.0


def test_sign_flips_ignores_tiny_reversals_around_zero():
    dt = 0.1
    small = FLIP_RATE / 2.0
    rates = np.array([[small, 0, 0], [-small, 0, 0], [small, 0, 0]])  # below FLIP_RATE ⇒ ignored
    assert np.all(sign_flips_per_s(rates, dt) == 0.0)


def test_hf_power_share_high_for_a_fast_oscillation_low_for_a_slow_ramp():
    dt = 0.02  # 50 Hz sampling (Nyquist 25 Hz), cutoff 5 Hz
    t = np.arange(400) * dt
    fast = np.zeros((t.size, 3))
    fast[:, 2] = np.sin(2 * np.pi * 12.0 * t)  # 12 Hz > 5 Hz cutoff
    slow = np.zeros((t.size, 3))
    slow[:, 2] = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz < 5 Hz cutoff
    assert hf_power_share(fast, dt) > 0.9
    assert hf_power_share(slow, dt) < 0.1


def test_smoothness_metrics_bundle_keys_and_yaw_axis():
    dt = 0.1
    # yaw is axis 2 (col 3 of raw); make it reverse to exercise yaw_flips_per_s
    raw = np.array([[0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, -2.0], [0.0, 0.0, 0.0, 2.0]])
    m = smoothness_metrics(raw, dt)
    assert set(m) >= {
        "ticks",
        "mean_abs_delta",
        "saturation_share",
        "sign_flips_per_s",
        "yaw_flips_per_s",
        "hf_power_share",
    }
    assert m["ticks"] == 3
    assert len(m["sign_flips_per_s"]) == 3
    # yaw reverses twice over 0.2 s
    assert m["yaw_flips_per_s"] == pytest.approx(2 / 0.2)
    assert m["sign_flips_per_s"][2] == pytest.approx(m["yaw_flips_per_s"])


def test_aggregate_averages_over_episodes_and_skips_short_traces():
    dt = 0.1
    good = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])  # 2 ticks
    short = np.zeros((1, 4))  # skipped (< 2 ticks)
    agg = aggregate_metrics([good, short, good], dt)
    assert agg["episodes"] == 2  # the 1-tick trace is dropped
    single = smoothness_metrics(good, dt)
    assert agg["mean_abs_delta"] == pytest.approx(single["mean_abs_delta"])


def test_aggregate_empty_is_all_zero():
    agg = aggregate_metrics([], 0.1)
    assert agg["episodes"] == 0
    assert agg["mean_abs_delta"] == 0.0 and agg["yaw_flips_per_s"] == 0.0
    assert agg["sign_flips_per_s"] == [0.0, 0.0, 0.0]


def test_module_has_no_torch_or_rng_dependency():
    """AC10: the metrics are stateless NumPy — no torch import, no RNG."""
    src = (sm.__file__ or "").replace(".pyc", ".py")
    text = open(src, encoding="utf-8").read()
    assert "import torch" not in text
    assert "random" not in text.lower() or "rng" not in text.lower()
