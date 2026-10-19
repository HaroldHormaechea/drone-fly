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


# ----------------------------------------------------------------- UC-89 grounded-launch alignment


def _gstart_env(
    *,
    frac=0.0,
    pre_acquire_s=0.0,
    spawn_distance_m=None,
    no_obs=True,
    ground_fraction=1.0,
    n=32,
    seed=0,
):
    """A CourseEnv exposing the two UC-89 default-off knobs (``ground_gate_ahead_frac`` /
    ``pre_acquire_s``). Mirrors ``_env`` but routes the knobs to the CourseConfig."""
    from drone_fly.sim.timing import TimingConfig

    cfg = CourseConfig(
        reward_reference_dt_s=0.02,
        no_obs=no_obs,
        pool_size=32,
        ground_gate_ahead_frac=frac,
        pre_acquire_s=pre_acquire_s,
        spawn_distance_m=spawn_distance_m,
        start=StartConfig(mode="mixed", ground_fraction=ground_fraction, arm_delay_s=0.5),
    )
    return CourseEnv(
        meteor75(),
        n,
        "cpu",
        seed,
        cfg,
        PlantConfig(),
        TimingConfig.fixed(HZ),
        0.0,
        action_spec=ActionSpec.v2(1.0, 4.0),
    )


def test_ground_gate_ahead_places_gate0_on_plus_x():
    """UC-89 AC3: with the knob on, a grounded start presents gate-0 dead-ahead (+x) at the
    configured distance (live ~38 m / world_scale 12 = 3.2 training-scale)."""
    env = _gstart_env(frac=1.0, spawn_distance_m=(3.2, 3.2), ground_fraction=1.0, n=64)
    assert torch.all(env.start_mode == START_GROUND)
    assert torch.all(env._gstart_active)  # frac=1 & every start grounded -> all rotated
    rel = env.obs()[:, :3]  # [rel_x, rel_y, rel_z] to the current (gate-0) target
    torch.testing.assert_close(rel[:, 0], torch.full((env.n,), 3.2), atol=2e-3, rtol=0)
    torch.testing.assert_close(rel[:, 1], torch.zeros(env.n), atol=2e-3, rtol=0)
    assert torch.all(rel[:, 0] > 0)  # dead ahead, not off to the side


def test_ground_gate_ahead_default_off_is_byte_identical():
    """UC-89 AC6: ``ground_gate_ahead_frac=0`` draws no extra RNG and leaves spawn + obs/extra
    byte-identical to an env built without the knob."""
    from drone_fly.sim.timing import TimingConfig

    off = _gstart_env(frac=0.0, ground_fraction=1.0, n=16, seed=3)
    base_cfg = CourseConfig(
        reward_reference_dt_s=0.02,
        no_obs=True,
        pool_size=32,
        start=StartConfig(mode="mixed", ground_fraction=1.0, arm_delay_s=0.5),
    )
    base = CourseEnv(
        meteor75(), 16, "cpu", 3, base_cfg, PlantConfig(),
        TimingConfig.fixed(HZ), 0.0, action_spec=ActionSpec.v2(1.0, 4.0),
    )  # fmt: skip
    assert off._gstart_on is False and off._preacq_on is False
    torch.testing.assert_close(off.pos, base.pos, atol=0, rtol=0)
    torch.testing.assert_close(off.extra(), base.extra(), atol=0, rtol=0)
    torch.testing.assert_close(off.obs(), base.obs(), atol=0, rtol=0)
    for _ in range(5):  # byte-identical dynamics too
        o1, r1, d1, _ = off.step_command(torch.ones(16), torch.zeros(16, 3))
        o2, r2, d2, _ = base.step_command(torch.ones(16), torch.zeros(16, 3))
        torch.testing.assert_close(o1, o2, atol=0, rtol=0)
        torch.testing.assert_close(r1, r2, atol=0, rtol=0)
        assert torch.equal(d1, d2)


def test_ground_gate_rotated_frame_guard_no_box_course():
    """UC-89 frame guard (no-box course, exact): the per-env scene rotation is applied at EVERY
    gate read-point, so each gate's world centre is the natural centre rigidly rotated about the
    spawn by the stored angle, with z untouched. Gate-0 lands on +x from the spawn.

    Scoped to a no-box course (``no_obs=True``): box obstacles are stored as axis-aligned AABBs
    with no orientation, so the scene rotation moves their centres but leaves their faces
    axis-aligned (analyst-accepted representational limit). Gates/cylinders rotate exactly; the
    exact-equality guard therefore runs where only gate geometry is in play.
    """
    env = _gstart_env(frac=1.0, ground_fraction=1.0, no_obs=True, n=32, seed=1)
    assert torch.all(env._gstart_active)
    c, s = torch.cos(env._gstart_theta), torch.sin(env._gstart_theta)
    cx, cy = env._gstart_cxy[:, 0], env._gstart_cxy[:, 1]
    ng = int(env._ng().min().item())
    for k in range(min(ng, 4)):
        raw = env.P["cen"][env.cid, k]  # unrotated world centre of gate k
        got = env._gate(torch.full_like(env.tgt, k))[0]
        dx, dy = raw[:, 0] - cx, raw[:, 1] - cy
        ex, ey = cx + c * dx - s * dy, cy + s * dx + c * dy
        torch.testing.assert_close(got[:, 0], ex, atol=1e-4, rtol=0)
        torch.testing.assert_close(got[:, 1], ey, atol=1e-4, rtol=0)
        torch.testing.assert_close(got[:, 2], raw[:, 2], atol=0, rtol=0)  # z not rotated
    # gate-0 is dead-ahead (+x) from the spawn (pos == rotation centre at spawn)
    g0 = env._gate(torch.zeros_like(env.tgt))[0]
    assert torch.all(g0[:, 0] - cx > 0)
    torch.testing.assert_close(g0[:, 1] - cy, torch.zeros(env.n), atol=1e-4, rtol=0)
    # no env spuriously leaves the arena / crashes on the first grounded steps (frame-consistent
    # bounds + clearance): a read left in the old frame would desync these.
    for _ in range(5):
        _, _, done, _ = env.step_command(torch.ones(env.n), torch.zeros(env.n, 3))
        assert not done.any()


def test_ground_gate_ahead_box_course_sanity():
    """UC-89 box-course SANITY (not exact-equality): on a box-containing ground course the knob
    still presents gate-0 on +x and rotates obstacle CENTRES with the scene; default-off stays
    byte-identical. Box faces stay axis-aligned (AABB schema) -- exact equality is NOT asserted."""
    from drone_fly.sim.timing import TimingConfig

    env = _gstart_env(
        frac=1.0, ground_fraction=1.0, no_obs=False, spawn_distance_m=(3.2, 3.2), n=32, seed=2
    )
    assert torch.all(env._gstart_active)
    rel = env.obs()[:, :3]
    assert torch.all(rel[:, 0] > 0)  # gate-0 dead-ahead even with obstacles present
    torch.testing.assert_close(rel[:, 1], torch.zeros(env.n), atol=2e-3, rtol=0)
    # obstacle centres actually rotate with the scene (the _ocen choke-point fires).
    if env._omask().any():
        env._gstart_on = False
        raw_oc = env._ocen().clone()
        env._gstart_on = True
        rot_oc = env._ocen()
        assert not torch.allclose(rot_oc, raw_oc)  # centres moved under the rotation
    # default-off on a box course is byte-identical.
    base_cfg = CourseConfig(
        reward_reference_dt_s=0.02, no_obs=False, pool_size=32,
        spawn_distance_m=(3.2, 3.2),
        start=StartConfig(mode="mixed", ground_fraction=1.0, arm_delay_s=0.5),
    )  # fmt: skip
    base = CourseEnv(
        meteor75(), 32, "cpu", 2, base_cfg, PlantConfig(),
        TimingConfig.fixed(HZ), 0.0, action_spec=ActionSpec.v2(1.0, 4.0),
    )  # fmt: skip
    off = _gstart_env(
        frac=0.0, ground_fraction=1.0, no_obs=False, spawn_distance_m=(3.2, 3.2), n=32, seed=2
    )
    torch.testing.assert_close(off.pos, base.pos, atol=0, rtol=0)
    torch.testing.assert_close(off.extra(), base.extra(), atol=0, rtol=0)


def test_ground_gate_ahead_command_ramps_from_zero_across_arm_to_launch():
    """UC-89 AC3/AC5 + AC6 no-regression: with the knob on, a grounded env still honours the
    arm->idle->launch sequence -- motors idle (0 g) through the arm delay, then the command
    ramps up from 0 and it lifts off toward the dead-ahead gate."""
    env = _gstart_env(frac=1.0, spawn_distance_m=(3.2, 3.2), ground_fraction=1.0, n=16)
    assert torch.all(env.plant.motor_g == 0.0)  # grounded, motors idle at spawn
    start_z = env.pos[:, 2].clone()
    steps = int(round(0.5 * HZ))  # the 0.5 s arm delay
    for _ in range(steps):
        env.step_command(torch.full((env.n,), 2.0), torch.zeros(env.n, 3))
        assert torch.all(env.plant.motor_g == 0.0)  # idle: command does not reach the motors
        torch.testing.assert_close(env.pos[:, 2], start_z, atol=0, rtol=0)
    assert not env.plant.lifted.any()
    # after the delay the command reaches the motors (ramping up from 0) and it climbs
    for _ in range(5):
        env.step_command(torch.full((env.n,), 2.0), torch.zeros(env.n, 3))
    assert torch.all(env.plant.motor_g > 0.0) and env.plant.lifted.all()
    assert torch.all(env.pos[:, 2] > start_z)
