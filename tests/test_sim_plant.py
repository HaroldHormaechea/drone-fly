"""UC67 AC2 / AC5 / AC9: the profile-driven plant (``drone_fly.sim.plant.Plant``).

Determinism per seed, the delay line (zero delay is a no-op, a step command lands exactly after
its delay, per-env jittered delays, monotonic effect times [C2]), a skipped tick holds the action
for two periods [C3], the nominal-envelope clamp times the per-env scales, the drag sign, the
stick (replay) path through the Betaflight curve and the thrust curve, and the common-mode
``control_to_gyro_s`` offset applied to all four channels with the nominal never added [C1].

Analytic checks use ``float64`` and the default ``exact`` integrator, whose first-order responses
are exact zero-order-hold solutions with one switch at the delayed command's effect time.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from drone_fly.sim.plant import DEFAULT_SIGNS, Plant, PlantConfig, betaflight_rate_rad, interp
from drone_fly.sysid.model import betaflight_rate_rad as sysid_betaflight
from tests.sim_support import AXES, all_envs, meteor75, micro, rng_range, variant

F64 = PlantConfig(dtype="float64")


def _plant(profile, n=1, *, seed=0, scale=0.0, config=F64):
    p = Plant(profile, n, "cpu", seed, config, scale)
    p.reset(all_envs(n), pos=torch.tensor([0.0, 0.0, 2.0]).expand(n, 3), resample=False)
    return p


def _cmd(n, g, rates=(0.0, 0.0, 0.0), dtype=torch.float64):
    return (
        torch.full((n,), float(g), dtype=dtype),
        torch.tensor(rates, dtype=dtype).expand(n, 3).clone(),
    )


# ----------------------------------------------------------------------------- determinism


def _rollout(seed: int, steps: int = 30) -> tuple[Plant, list[torch.Tensor]]:
    p = Plant(micro(), 16, "cpu", seed, PlantConfig(), 1.0)
    p.reset(all_envs(16), pos=torch.tensor([0.0, 0.0, 2.0]).expand(16, 3))
    g = torch.Generator().manual_seed(123)  # identical actions for every seed
    out = []
    for _ in range(steps):
        cg = 1.0 + 0.3 * torch.randn(16, generator=g)
        rates = 2.0 * torch.randn(16, 3, generator=g)
        p.step(cg, rates, 0.02, 0.01)
        out.append(torch.cat([p.pos, p.vel, p.quat, p.omega], 1).clone())
    return p, out


def test_same_seed_is_bit_identical_and_another_seed_differs():
    a, ta = _rollout(3)
    b, tb = _rollout(3)
    c, _ = _rollout(4)
    assert torch.equal(a.params.table, b.params.table)
    for x, y in zip(ta, tb, strict=True):
        assert torch.equal(x, y)
    assert not torch.equal(a.params.table, c.params.table)


def test_plant_never_touches_the_global_rng():
    torch.manual_seed(5)
    before = torch.get_rng_state()
    _rollout(7, steps=3)
    assert torch.equal(torch.get_rng_state(), before)


# ----------------------------------------------------------------------------- delay line


def test_zero_delay_line_is_a_no_op():
    """With no delay, pushing through the line equals setting the command in force directly."""
    prof = variant(meteor75(), randomization={})
    a, b = _plant(prof, 3), _plant(prof, 3)
    cg, rates = _cmd(3, 1.7, (1.0, -2.0, 0.5))
    for _ in range(5):
        a.step(cg, rates, 0.02, 0.0)
        b.active = b._physical(cg, rates)
        b.advance(0.02)
    for key in ("pos", "vel", "quat", "omega", "motor_g"):
        assert torch.equal(getattr(a, key), getattr(b, key)), key
    # meteor75 applies thrust instantly; the rate loop is the exact first-order response
    assert a.motor_g[0].item() == pytest.approx(1.7, abs=1e-12)
    tau = prof.rates["roll"].tau_s
    assert a.omega[0, 0].item() == pytest.approx(1.0 - math.exp(-0.1 / tau), abs=1e-12)


def _delay_profile():
    # throttle: input 25 ms; roll: + 10 ms of axis delay; first-order lags of 10 / 20 ms
    return variant(
        meteor75(),
        input_s=0.025,
        motor_lag_s=0.01,
        rates={"roll": {"delay_s": 0.01, "tau_s": 0.02}},
        randomization={},
    )


def test_step_command_lands_exactly_after_its_per_channel_delay():
    p = _plant(_delay_profile())
    p.reset(all_envs(1), pos=torch.tensor([[0.0, 0.0, 2.0]]), motor_g=1.0, resample=False)
    cg, rates = _cmd(1, 2.0, (1.0, 0.0, 0.0))
    p.push(p._physical(cg, rates), 0.0)
    for k in range(1, 9):
        p.advance(0.01)
        t = 0.01 * k
        motor = 1.0 if t <= 0.025 else 2.0 - math.exp(-(t - 0.025) / 0.01)
        roll = 0.0 if t <= 0.035 else 1.0 - math.exp(-(t - 0.035) / 0.02)
        assert p.motor_g[0].item() == pytest.approx(motor, abs=1e-12), t
        assert p.omega[0, 0].item() == pytest.approx(roll, abs=1e-12), t
        assert p.omega[0, 1].item() == 0.0 and p.omega[0, 2].item() == 0.0


def test_action_delay_adds_to_the_channel_delay():
    p = _plant(_delay_profile())
    p.reset(all_envs(1), pos=torch.tensor([[0.0, 0.0, 2.0]]), motor_g=1.0, resample=False)
    cg, rates = _cmd(1, 2.0)
    p.step(cg, rates, 0.04, action_delay_s=0.02)  # throttle lands at 0.045
    assert p.motor_g[0].item() == pytest.approx(1.0, abs=1e-12)
    p.advance(0.01)
    assert p.motor_g[0].item() == pytest.approx(2.0 - math.exp(-0.005 / 0.01), abs=1e-12)


def test_per_env_jittered_delays_switch_each_env_at_its_own_time():
    prof = variant(meteor75(), motor_lag_s=0.01, randomization={})
    p = _plant(prof, 3)
    p.reset(all_envs(3), pos=torch.tensor([0.0, 0.0, 2.0]).expand(3, 3), motor_g=1.0)
    delays = torch.tensor([0.0, 0.013, 0.031], dtype=torch.float64)
    cg, rates = _cmd(3, 2.0)
    p.step(cg, rates, 0.04, delays)
    expect = [2.0 - math.exp(-(0.04 - d) / 0.01) for d in delays.tolist()]
    np.testing.assert_allclose(p.motor_g.numpy(), expect, rtol=0, atol=1e-12)


def test_effect_times_are_monotonic_in_push_order():
    """[C2] A newer command with a shorter (jittered) delay never lets an older one override it."""
    prof = variant(meteor75(), randomization={})  # instant thrust: the motor shows the command
    p = _plant(prof)
    p.reset(all_envs(1), pos=torch.tensor([[0.0, 0.0, 2.0]]), motor_g=1.0, resample=False)
    seen = []
    cg_a, r = _cmd(1, 2.0)
    p.step(cg_a, r, 0.01, action_delay_s=0.05)  # A at t=0, lands at 0.05
    seen.append(p.motor_g.item())
    cg_b, _ = _cmd(1, 0.5)
    p.step(cg_b, r, 0.01, action_delay_s=0.0)  # B at t=0.01: would land at 0.01, pushed to 0.05
    seen.append(p.motor_g.item())
    for _ in range(6):
        p.advance(0.01)
        seen.append(p.motor_g.item())
    assert 2.0 not in seen  # A is never in force: B (newer) takes over at A's effect time
    np.testing.assert_allclose(seen[:5], 1.0, atol=1e-12)  # nothing before t = 0.05
    np.testing.assert_allclose(seen[5:], 0.5, atol=1e-12)
    assert p.last_eff[0, 0].item() == pytest.approx(0.05)


def test_a_skipped_tick_holds_the_action_for_two_periods():
    """[C3] dt x2 inside one transition, with the action pushed once."""
    prof = variant(meteor75(), motor_lag_s=0.03, randomization={})
    p = _plant(prof, 2)
    p.reset(all_envs(2), pos=torch.tensor([0.0, 0.0, 2.0]).expand(2, 3), motor_g=1.0)
    period = 0.025
    dt = torch.tensor([period, 2 * period], dtype=torch.float64)
    cg, rates = _cmd(2, 1.5)
    p.step(cg, rates, dt)
    assert p.t.tolist() == pytest.approx([period, 2 * period])
    expect = [1.5 - 0.5 * math.exp(-h / 0.03) for h in dt.tolist()]
    np.testing.assert_allclose(p.motor_g.numpy(), expect, rtol=0, atol=1e-12)
    assert p._pushes == 1


# ----------------------------------------------------------------------------- clamps / scales


def test_command_is_clamped_to_the_nominal_envelope_then_scaled_per_env():
    prof = variant(
        meteor75(),
        randomization={
            "thrust_scale": rng_range(0.7, 1.3),
            "rates.roll.rate_scale": rng_range(0.5, 1.5),
        },
    )
    n = 64
    p = Plant(prof, n, "cpu", 1, F64, 1.0)
    full = p.full_stick_rate[0].item()
    assert full == pytest.approx(4.0, rel=1e-9)  # meteor75: MAX_BODY_RATE
    phys = p._physical(*_cmd(n, 100.0, (100.0, -100.0, 2.0)))
    ts, rs = p.params.thrust_scale, p.params.rate_scale
    assert ts.std() > 0 and rs[:, 0].std() > 0
    torch.testing.assert_close(phys[:, 0], prof.max_g * ts)
    torch.testing.assert_close(phys[:, 1], full * rs[:, 0])
    torch.testing.assert_close(phys[:, 2], torch.full((n,), -full, dtype=torch.float64))
    torch.testing.assert_close(phys[:, 3], torch.full((n,), 2.0, dtype=torch.float64))
    assert torch.all(p._physical(*_cmd(n, -3.0))[:, 0] == 0.0)  # no negative thrust


def test_thrust_curve_interp_matches_numpy():
    prof = micro()
    p = Plant(prof, 1, "cpu", 0, F64, 0.0)
    x = torch.linspace(-0.2, 1.2, 57, dtype=torch.float64)
    expect = np.interp(x.numpy(), prof.thrust_curve.throttle, prof.thrust_curve.accel_g)
    np.testing.assert_allclose(p.thrust_g(x).numpy(), expect, rtol=0, atol=1e-12)
    xp = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    fp = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float64)
    assert interp(torch.tensor([0.75], dtype=torch.float64), xp, fp).item() == pytest.approx(2.5)


# ----------------------------------------------------------------------------- drag


@pytest.mark.parametrize("vx", [2.0, -2.0])
def test_drag_opposes_the_velocity(vx):
    prof = variant(meteor75(), drag=(0.5, 0.2), randomization={})
    p = _plant(prof)
    p.reset(
        all_envs(1),
        pos=torch.tensor([[0.0, 0.0, 2.0]]),
        vel=torch.tensor([[vx, 0.0, 0.0]]),
        motor_g=1.0,
        resample=False,
    )
    p.step(*_cmd(1, 1.0), 0.02)
    dv = -(0.5 * vx + 0.2 * abs(vx) * vx) * 0.02  # body drag on the old velocity, level
    assert p.vel[0, 0].item() == pytest.approx(vx + dv, abs=1e-12)
    assert abs(p.vel[0, 0].item()) < abs(vx)
    assert p.vel[0, 2].item() == pytest.approx(0.0, abs=1e-12)  # 1 g holds altitude


def test_per_body_axis_drag_acts_on_its_own_axis_only():
    prof = variant(meteor75(), drag=((0.5, 0.0, 0.0), (0.0, 0.0, 0.0)), randomization={})
    p = _plant(prof)
    p.reset(
        all_envs(1),
        pos=torch.tensor([[0.0, 0.0, 2.0]]),
        vel=torch.tensor([[2.0, 2.0, 0.0]]),
        motor_g=1.0,
        resample=False,
    )
    p.step(*_cmd(1, 1.0), 0.02)
    assert p.vel[0, 0].item() == pytest.approx(2.0 - 0.5 * 2.0 * 0.02, abs=1e-12)
    assert p.vel[0, 1].item() == pytest.approx(2.0, abs=1e-12)


# ----------------------------------------------------------------------------- stick path


def test_stick_rates_are_the_betaflight_curve_times_the_signs():
    prof = variant(
        micro(), rates={"roll": {"expo": 0.3}, "yaw": {"rc_rate": 2.4}}, randomization={}
    )
    p = Plant(prof, 1, "cpu", 0, F64, 0.0)
    sticks = np.linspace(-1.0, 1.0, 21)
    got = p.stick_rates(torch.as_tensor(np.stack([sticks] * 3, 1)))
    for i, a in enumerate(AXES):
        r = prof.rates[a]
        expect = DEFAULT_SIGNS[i] * sysid_betaflight(sticks, r.rc_rate, r.super_rate, r.expo)
        np.testing.assert_allclose(got[:, i].numpy(), expect, rtol=1e-12, atol=1e-12)
    # the torch mirror itself, on the incremental (rc_rate > 2) branch
    one = betaflight_rate_rad(
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(2.4, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
    )
    assert one.item() == pytest.approx(float(sysid_betaflight(0.5, 2.4, 0.0, 0.0)), rel=1e-12)


def test_step_sticks_drives_the_thrust_curve_and_the_rate_curve():
    prof = micro()
    p = _plant(prof)
    sticks = torch.tensor([[0.3, -0.2, 0.5]], dtype=torch.float64)
    for _ in range(10):  # 100 ms: past every channel delay of the micro (< 15 ms)
        p.step_sticks(torch.tensor([0.6], dtype=torch.float64), sticks, 0.01)
    assert p.active[0, 0].item() == pytest.approx(
        float(np.interp(0.6, prof.thrust_curve.throttle, prof.thrust_curve.accel_g)), rel=1e-12
    )
    torch.testing.assert_close(p.active[:, 1:], p.stick_rates(sticks))
    assert p.active[0, 3].item() < 0  # yaw stick right -> clockwise (negative wz)


# ----------------------------------------------------------------------------- [C1] delays


def test_common_mode_offset_hits_all_four_channels_on_meteor75():
    prof = meteor75()  # c2g randomized 0..40 ms, no input / axis delay
    p = Plant(prof, 256, "cpu", 2, F64, 1.0)
    off = p.params.c2g_offset_s
    assert off.min() >= 0.0 and off.max() <= 0.04 and off.std() > 0.005
    for ch in range(4):
        torch.testing.assert_close(p.channel_delay[:, ch], off)


def test_micro_channel_delays_stack_input_axis_delay_and_the_offset():
    prof = micro()
    p = Plant(prof, 128, "cpu", 2, F64, 1.0)
    prm = p.params
    nominal_c2g = prof.latency.control_to_gyro_s
    lo = prof.randomization["latency.control_to_gyro_s"].min - nominal_c2g
    hi = prof.randomization["latency.control_to_gyro_s"].max - nominal_c2g
    assert prm.c2g_offset_s.min() >= lo - 1e-12 and prm.c2g_offset_s.max() <= hi + 1e-12
    inp = prof.latency.input_s
    torch.testing.assert_close(p.channel_delay[:, 0], inp + prm.c2g_offset_s)
    for i in range(3):
        torch.testing.assert_close(
            p.channel_delay[:, 1 + i], inp + prm.delay_s[:, i] + prm.c2g_offset_s
        )


def test_nominal_control_to_gyro_is_never_added_on_its_own():
    prof = micro()
    p = Plant(prof, 4, "cpu", 0, F64, 0.0)
    inp = prof.latency.input_s
    expect = [inp] + [inp + prof.rates[a].delay_s for a in AXES]
    for row in p.channel_delay.tolist():
        assert row == pytest.approx(expect, abs=1e-12)
    assert prof.latency.control_to_gyro_s > inp + 1e-3  # it would have shown up


# ----------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    "bad",
    [{"integrator": "rk4"}, {"dtype": "float16"}, {"substeps": 0}, {"delay_slots": 1}],
)
def test_plant_config_is_validated(bad):
    with pytest.raises(ValueError):
        PlantConfig(**bad)
    with pytest.raises(ValueError, match="unknown plant keys"):
        PlantConfig.from_dict({"substep": 2})


def test_substeps_split_the_step():
    prof = variant(meteor75(), motor_lag_s=0.02, randomization={})
    p = _plant(prof, config=PlantConfig(dtype="float64", substeps=4))
    p.reset(all_envs(1), pos=torch.tensor([[0.0, 0.0, 2.0]]), motor_g=1.0, resample=False)
    p.step(*_cmd(1, 2.0), 0.02)
    # the exact first-order response does not depend on the substep count
    assert p.motor_g[0].item() == pytest.approx(2.0 - math.exp(-1.0), abs=1e-12)
    assert p.t[0].item() == pytest.approx(0.02)
