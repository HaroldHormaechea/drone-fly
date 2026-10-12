"""UC67 AC7: ground starts at rest with an arm delay, and the resting-contact model [C9].

A ground start spawns level on the floor at the rest height; for ``arm_delay_s`` the plant gets
idle commands whatever the policy outputs (the policy still observes), and the drone must not
move. Afterwards it stays down under a < 1 g command and lifts off under a > 1 g one. Commands
given while it rests (rates, sub-hover thrust) do not wind up into a jump or a tilt at lift-off.
Floor contact is non-terminal until the first lift-off; the mixed start draws ground vs air at
the configured fraction.
"""

from __future__ import annotations

import pytest
import torch

from drone_fly.sim.action import ActionSpec
from drone_fly.sim.course import (
    EPISODE_COLUMNS,
    FLOOR_KILL_Z,
    START_AIR,
    START_GROUND,
    CourseConfig,
    CourseEnv,
    StartConfig,
)
from drone_fly.sim.plant import Plant, PlantConfig
from tests.sim_support import all_envs, meteor75, micro, variant

HZ = 50.0
DT = 1.0 / HZ


def _env(profile=None, *, ground_fraction=1.0, arm_delay_s=0.5, n=8, seed=0, log=None, **kw):
    from drone_fly.sim.timing import TimingConfig

    cfg = CourseConfig(
        reward_reference_dt_s=0.02,
        no_obs=True,
        pool_size=32,
        start=StartConfig(
            mode="mixed", ground_fraction=ground_fraction, arm_delay_s=arm_delay_s, **kw
        ),
    )
    return CourseEnv(
        profile or meteor75(),
        n,
        "cpu",
        seed,
        cfg,
        PlantConfig(),
        TimingConfig.fixed(HZ),
        0.0,
        action_spec=ActionSpec.v2(1.0, 4.0),
        log_capacity=log,
    )


def _hold(env, g, rates=(0.0, 0.0, 0.0), steps=1):
    n = env.n
    out = None
    for _ in range(steps):
        out = env.step_command(torch.full((n,), float(g)), torch.tensor(rates).expand(n, 3))
        assert not out[2].any(), "no episode may end while it rests / lifts off"
    return out


def test_ground_start_spawns_at_rest_on_the_floor():
    env = _env()
    assert torch.all(env.start_mode == START_GROUND)
    rest = env.plant.rest_z
    assert rest > FLOOR_KILL_Z  # [C9] the rest height is above the floor-kill threshold
    torch.testing.assert_close(env.pos[:, 2], torch.full((env.n,), rest))
    assert torch.all(env.vel == 0) and torch.all(env.omega == 0)
    torch.testing.assert_close(env.quat, torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(env.n, 4))
    assert torch.all(env.plant.motor_g == 0)  # motors idle
    assert torch.all(env.plant.grounded) and not env.plant.lifted.any()


def test_no_motion_during_the_arm_delay_even_under_a_2g_command():
    env = _env(arm_delay_s=0.5)
    start = env.pos.clone()
    steps = int(round(0.5 * HZ))
    for _ in range(steps):
        obs, _, done, _ = _hold(env, 2.0, rates=(3.0, -3.0, 2.0))
        assert torch.equal(env.pos, start) and torch.all(env.vel == 0)
        assert torch.all(env.omega == 0) and torch.all(env.plant.motor_g == 0)
        assert obs.shape == (env.n, 12)  # the policy still observes
    assert not env.plant.lifted.any()
    # first step after the delay: the command reaches the motors and it lifts off
    _hold(env, 2.0)
    assert torch.all(env.pos[:, 2] > start[:, 2]) and torch.all(env.vel[:, 2] > 0)
    assert env.plant.lifted.all() and not env.plant.grounded.any()


def test_stays_down_under_a_sub_hover_command():
    env = _env(arm_delay_s=0.0)
    start = env.pos.clone()
    for _ in range(int(2.0 * HZ)):  # 2 s, well past the 0.4 s floor grace
        _hold(env, 0.8)
        assert torch.equal(env.pos, start) and torch.all(env.vel == 0)
    assert torch.all(env.plant.grounded) and not env.plant.lifted.any()
    # the motor state is capped at the lift threshold (1 g level), not wound up
    assert torch.all(env.plant.motor_g <= 1.0 + 1e-6)


def test_lifts_off_under_a_super_hover_command_after_the_delay():
    env = _env(arm_delay_s=0.5)
    _hold(env, 1.3, steps=int(0.5 * HZ))
    _hold(env, 1.3, steps=int(0.5 * HZ))
    rest = env.plant.rest_z
    # 0.5 s at 0.3 g net (meteor75: instant thrust, no drag): z = rest + 0.5 * 0.3 g * t^2
    expect = rest + 0.5 * 0.3 * 9.81 * 0.5**2
    torch.testing.assert_close(env.pos[:, 2], torch.full((env.n,), expect), atol=0.02, rtol=0)
    assert env.plant.lifted.all()


@pytest.mark.parametrize("profile_fn", [meteor75, micro])
def test_no_rate_wind_up_at_lift_off(profile_fn):
    """[C9] Rate commands given on the ground leave no stored body rate or tilt at lift-off.

    meteor75 has no delay and a slow (55 ms) rate response: a filter wound up on the ground would
    still carry most of it through the lift-off step. The micro's commands are delayed (< 15 ms),
    so two grounded steps with centred sticks let the rate commands still in flight land first.
    """
    env = _env(profile_fn(), arm_delay_s=0.0)
    _hold(env, 0.6, rates=(4.0, -4.0, 3.0), steps=int(1.0 * HZ))  # rests, sticks hard over
    assert torch.all(env.omega == 0)
    torch.testing.assert_close(env.quat, torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(env.n, 4))
    if profile_fn is micro:
        _hold(env, 0.6, steps=2)
        assert torch.all(env.plant.grounded)
    _hold(env, 1.5, steps=10)  # 200 ms of a level climb
    assert env.plant.lifted.all()
    assert env.omega.abs().max() < 1e-4
    tilt = torch.rad2deg(2 * torch.acos(env.quat[:, 0].clamp(-1, 1)))
    assert tilt.max() < 0.05


def test_no_thrust_wind_up_at_lift_off():
    """A long sub-hover hold ends at the lift threshold, so lift-off is no jump."""
    prof = variant(meteor75(), motor_lag_s=0.05, randomization={})
    p = Plant(prof, 2, "cpu", 0, PlantConfig(dtype="float64"), 0.0)
    rest = p.rest_z
    p.reset(
        all_envs(2),
        pos=torch.tensor([0.0, 0.0, rest]).expand(2, 3),
        motor_g=0.0,
        grounded=True,
    )
    cg = torch.tensor([0.99, 0.0], dtype=torch.float64)  # env 1 never commanded thrust
    for _ in range(50):
        p.step(cg, torch.zeros(2, 3, dtype=torch.float64), DT)
    assert p.grounded.all() and p.motor_g[0].item() == pytest.approx(0.99, abs=1e-6)
    climb = torch.full((2,), 1.05, dtype=torch.float64)
    for _ in range(10):
        p.step(climb, torch.zeros(2, 3, dtype=torch.float64), DT)
    # env 0 lifts off gently: its motor state was 0.99 g, never above the 1 g threshold
    assert p.lifted[0] and p.vel[0, 2].item() < 0.05 * 9.81 * 0.2 + 1e-6
    assert p.pos[0, 2].item() >= rest


def test_tilted_rest_caps_the_motor_at_the_tilted_lift_threshold():
    prof = variant(meteor75(), motor_lag_s=0.05, randomization={})
    p = Plant(prof, 1, "cpu", 0, PlantConfig(dtype="float64"), 0.0)
    ang = 0.3
    q = torch.tensor([[torch.cos(torch.tensor(ang / 2)), torch.sin(torch.tensor(ang / 2)), 0, 0]])
    p.reset(all_envs(1), pos=torch.tensor([[0.0, 0.0, p.rest_z]]), quat=q, grounded=True)
    for _ in range(100):
        p.step(torch.tensor([1.02], dtype=torch.float64), torch.zeros(1, 3), DT)
    # 1.02 g tilted by 0.3 rad cannot lift (vertical component < 1 g): it stays, capped
    assert p.grounded.all()
    cap = 1.0 / torch.cos(torch.tensor(ang, dtype=torch.float64)).item()
    assert p.motor_g[0].item() <= cap + 1e-9


def test_floor_contact_is_not_terminal_before_lift_off_but_is_after():
    env = _env(arm_delay_s=0.0, n=4)
    _hold(env, 0.0, steps=int(1.0 * HZ))  # resting 1 s: past the time-based floor grace
    assert torch.all(env.steps == int(1.0 * HZ))
    _hold(env, 1.6, steps=10)  # lift off and climb
    assert env.plant.lifted.all()
    done = None
    for _ in range(60):  # cut the motors: it falls back and crashes on the floor
        _, _, done, _ = env.step_command(torch.zeros(4), torch.zeros(4, 3))
        if done.any():
            break
    assert done is not None and done.all() and env.cause_floor.all()


def test_first_step_never_terminates():
    env = _env(arm_delay_s=0.5, n=16)
    _, _, done, _ = env.step_command(torch.zeros(16), torch.zeros(16, 3))
    assert not done.any()


@pytest.mark.parametrize("fraction", [0.0, 0.3, 1.0])
def test_mixed_start_draws_ground_vs_air_at_the_fraction(fraction):
    env = _env(ground_fraction=fraction, n=2000, air_z_m=(0.8, 1.5))
    ground = env.start_mode == START_GROUND
    air = env.start_mode == START_AIR
    assert torch.all(ground | air)
    assert ground.float().mean().item() == pytest.approx(fraction, abs=0.04)
    if air.any():
        z = env.pos[air, 2]
        assert z.min() >= 0.8 - 1e-6 and z.max() <= 1.5 + 1e-6
        assert torch.all(env.plant.motor_g[air] == 1.0)  # air start at hover thrust
        assert not env.plant.grounded[air].any()
    if ground.any():
        assert torch.all(env.plant.motor_g[ground] == 0.0)
        torch.testing.assert_close(
            env.arm_until[ground], torch.full((int(ground.sum()),), 0.5, dtype=torch.float64)
        )


def test_episode_log_records_the_start_mode():
    env = _env(ground_fraction=0.5, n=64, log=256, arm_delay_s=0.0)
    modes = env.start_mode.clone()
    env.cfg = CourseConfig(**{**env.cfg.__dict__, "episode_s": 0.04})
    for _ in range(2):
        env.step_command(torch.zeros(64), torch.zeros(64, 3))
    rows = env.log.drain()
    col = EPISODE_COLUMNS.index
    assert len(rows) == 64
    got = {int(r[col("env")]): int(r[col("start_mode")]) for r in rows}
    assert got == {i: int(m) for i, m in enumerate(modes.tolist())}


def test_legacy_start_mode_is_the_prototype_spawn():
    cfg = StartConfig(mode="legacy")
    assert cfg.mode == "legacy"
    with pytest.raises(ValueError):
        StartConfig(mode="ground")
