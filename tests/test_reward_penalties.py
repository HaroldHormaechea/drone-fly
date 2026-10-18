"""UC-87 AC3: the action-rate + saturation reward penalties (``CourseEnv.step_command``).

The reward gains ``-(λ·‖tanh(a_t)−tanh(a_{t-1})‖² + w·Σ relu(|tanh(a_i)|−thr))``, rate-scaled like
the other dense terms (``dt / reward_reference_dt_s``), accumulated per episode and logged in
``EPISODE_TAIL`` as ``action_rate`` / ``saturation``. CPU-hermetic: two otherwise-identical envs
(same seed ⇒ identical dynamics) are stepped with the same action, so the reward difference isolates
the penalty exactly.
"""

from __future__ import annotations

import torch

from drone_fly.sim.action import ActionSpec
from drone_fly.sim.course import (
    EPISODE_COLUMNS,
    EPISODE_TAIL,
    CourseConfig,
    CourseEnv,
    StartConfig,
)
from drone_fly.sim.plant import PlantConfig
from drone_fly.sim.timing import TimingConfig
from tests.sim_support import meteor75

V2 = ActionSpec.v2(1.0, 4.0)
REF_DT = 0.02
THR = 0.9


def _env(*, lam=0.0, w=0.0, hz=50.0, n=4, seed=0, log=None, episode_s=28.0):
    cfg = CourseConfig(
        reward_reference_dt_s=REF_DT,
        no_obs=True,
        action_rate_lambda=lam,
        saturation_weight=w,
        saturation_threshold=THR,
        pool_size=32,
        episode_s=episode_s,
        ceiling_z=50.0,
        start=StartConfig(mode="mixed", ground_fraction=0.0, air_z_m=(3.0, 3.0)),
    )
    return CourseEnv(
        meteor75(), n, "cpu", seed, cfg, PlantConfig(), TimingConfig.fixed(hz), 0.0,
        action_spec=V2, log_capacity=log,
    )  # fmt: skip


def _expected(a_now, a_prev, lam, w, rscale):
    # compute in the env's working dtype (float32) so tolerances match the env's arithmetic
    ta, tp = torch.tanh(a_now.float()), torch.tanh(a_prev.float())
    arate = ((ta - tp) ** 2).sum(-1)
    sat = torch.relu(ta[:, 1:4].abs() - THR).sum(-1)
    return (lam * arate + w * sat) * rscale, arate, sat


def test_tail_schema_carries_the_penalty_columns():
    assert EPISODE_TAIL[-2:] == ("action_rate", "saturation")
    assert "action_rate" in EPISODE_COLUMNS and "saturation" in EPISODE_COLUMNS


def test_penalty_subtracts_exactly_the_action_rate_and_saturation_terms():
    lam, w = 0.1, 0.05
    base, pen = _env(), _env(lam=lam, w=w)
    a = torch.zeros(4, 4, dtype=torch.float64)
    a[:, 1:] = 2.0  # tanh(2)=0.964 > 0.9 ⇒ saturated on all three rate axes
    _, r_base, d0, _ = base.step(a.clone())
    _, r_pen, d1, _ = pen.step(a.clone())
    assert not d0.any() and not d1.any()
    # a_prev is the zero-seed at step 1; rscale = dt/ref_dt = 1 at 50 Hz
    exp, arate, sat = _expected(a, torch.zeros_like(a), lam, w, 1.0)
    assert arate[0].item() > 0 and sat[0].item() > 0  # the action truly triggers both terms
    torch.testing.assert_close(r_base - r_pen, exp)


def test_penalties_off_leave_the_reward_and_accumulators_untouched():
    env = _env()  # lam = w = 0
    a = torch.zeros(4, 4, dtype=torch.float64)
    a[:, 1:] = 2.0
    env.step(a)
    assert torch.count_nonzero(env.ep_arate) == 0 and torch.count_nonzero(env.ep_sat) == 0


def test_accumulators_sum_over_the_episode():
    lam, w = 0.1, 0.05
    env = _env(lam=lam, w=w)
    a1 = torch.zeros(4, 4, dtype=torch.float64)
    a1[:, 1:] = 1.0
    env.step(a1.clone())
    _, arate1, sat1 = _expected(a1, torch.zeros_like(a1), lam, w, 1.0)
    torch.testing.assert_close(env.ep_arate, arate1)
    torch.testing.assert_close(env.ep_sat, sat1)
    a2 = torch.zeros(4, 4, dtype=torch.float64)
    a2[:, 1:] = 2.0
    env.step(a2.clone())
    _, arate2, sat2 = _expected(a2, a1, lam, w, 1.0)  # prev_action is now a1
    torch.testing.assert_close(env.ep_arate, arate1 + arate2)
    torch.testing.assert_close(env.ep_sat, sat1 + sat2)


def test_penalty_is_rate_scaled_by_dt_over_reference():
    lam, w = 0.1, 0.05
    a = torch.zeros(4, 4, dtype=torch.float64)
    a[:, 1:] = 2.0
    # 50 Hz (dt = 0.02 = ref ⇒ rscale 1) vs 25 Hz (dt = 0.04 ⇒ rscale 2)
    fast_base, fast_pen = _env(hz=50.0), _env(lam=lam, w=w, hz=50.0)
    slow_base, slow_pen = _env(hz=25.0), _env(lam=lam, w=w, hz=25.0)
    _, rb_f, _, _ = fast_base.step(a.clone())
    _, rp_f, _, _ = fast_pen.step(a.clone())
    _, rb_s, _, _ = slow_base.step(a.clone())
    _, rp_s, _, _ = slow_pen.step(a.clone())
    pen_fast = rb_f - rp_f
    pen_slow = rb_s - rp_s
    torch.testing.assert_close(pen_slow, 2.0 * pen_fast, rtol=1e-6, atol=1e-9)


def test_penalty_scale_anneals_only_the_action_rate_lambda():
    """UC-87 redesign: CourseEnv.penalty_scale multiplies ONLY action_rate_lambda (λ mild→target);
    saturation_weight is intentionally un-annealed."""
    lam, w = 0.1, 0.05
    a = torch.zeros(4, 4, dtype=torch.float64)
    a[:, 1:] = 2.0
    # full penalty (scale 1) vs. half-annealed λ (scale 0.5) vs. baseline (no penalties)
    base = _env()
    full = _env(lam=lam, w=w)
    half = _env(lam=lam, w=w)
    half.penalty_scale = 0.5
    _, r_base, _, _ = base.step(a.clone())
    _, r_full, _, _ = full.step(a.clone())
    _, r_half, _, _ = half.step(a.clone())
    _, arate, sat = _expected(a, torch.zeros_like(a), lam, w, 1.0)
    # half: λ scaled by 0.5, saturation weight UNCHANGED
    exp_half = (0.5 * lam * arate + w * sat)
    torch.testing.assert_close(r_base - r_half, exp_half)
    # the difference between full and half isolates exactly half the action-rate term (not sat)
    torch.testing.assert_close((r_full - r_half) * -1.0, 0.5 * lam * arate - 0.0 * sat)


def test_penalty_scale_zero_disables_only_the_action_rate_term():
    lam, w = 0.1, 0.05
    a = torch.zeros(4, 4, dtype=torch.float64)
    a[:, 1:] = 2.0
    base = _env()
    env = _env(lam=lam, w=w)
    env.penalty_scale = 0.0  # λ fully off; saturation guard-rail still active
    _, r_base, _, _ = base.step(a.clone())
    _, r_pen, _, _ = env.step(a.clone())
    _, _arate, sat = _expected(a, torch.zeros_like(a), lam, w, 1.0)
    torch.testing.assert_close(r_base - r_pen, w * sat)  # only the saturation term remains


def test_penalty_scale_default_is_full():
    assert _env(lam=0.1).penalty_scale == 1.0


def test_penalties_are_logged_in_the_episode_tail():
    lam, w = 0.1, 0.05
    # a short episode_s so a timeout row is logged within a few steps
    env = _env(lam=lam, w=w, log=16, episode_s=0.04)
    a = torch.zeros(4, 4, dtype=torch.float64)
    a[:, 1:] = 1.5
    for _ in range(4):
        env.step(a.clone())
    rows = env.log.drain()
    assert rows.shape[0] >= 1
    ar = EPISODE_COLUMNS.index("action_rate")
    sa = EPISODE_COLUMNS.index("saturation")
    assert (rows[:, ar] > 0).all() and (rows[:, sa] > 0).all()
