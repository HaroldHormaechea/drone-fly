"""UC67 AC1 / AC3 / AC5 / AC9: the procedural course env (``drone_fly.sim.course.CourseEnv``).

* ``sim.course_gen`` is a verbatim port: identical courses for the same seed and tier.
* ``CourseEnv`` on meteor75 (randomization off, fixed 50 Hz, legacy start, ``euler-legacy``)
  reproduces ``gpu_prototype``'s ``BatchedProcCourse`` from identical injected state and course
  ids over a short horizon with no respawn: obs, extra, reward and done. The ``CourseConfig`` is
  built from the flag globals the fixture-imported ``gates_proc_env`` actually has [C5].
* Rate invariance: dense rewards scale with ``dt / reward_reference_dt_s``, the episode length
  and the floor-kill grace are times, a skipped tick is one transition of two periods [C3].
* The per-episode log carries every dynamics / timing sample, the start mode and the outcome.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

from drone_fly.sim.action import ActionSpec
from drone_fly.sim.course import (
    EPISODE_COLUMNS,
    OUTCOMES,
    TRACTABLE_KW,
    CourseConfig,
    CourseEnv,
    StartConfig,
    build_pool,
)
from drone_fly.sim.course_gen import sample_course
from drone_fly.sim.plant import PlantConfig
from drone_fly.sim.randomization import LOG_COLUMNS
from drone_fly.sim.timing import TimingConfig
from tests.sim_support import meteor75

V2 = ActionSpec.v2(1.0, 4.0)
REF_DT = 0.02


@pytest.fixture(scope="module")
def gpe(harness_contract):
    """The fixture-imported ``gates_proc_env``; its PROC_* flags were read from the shell [C5]."""
    set_flags = sorted(k for k in os.environ if k.startswith("PROC_"))
    if set_flags:
        pytest.skip(
            f"PROC_* environment variables are set in this shell ({', '.join(set_flags)}); "
            "gpu_prototype.gates_proc_env read them at import. Unset them to run the "
            "CourseEnv-vs-prototype comparison."
        )
    return sys.modules["gates_proc_env"]


def _small_env(*, timing=None, start=None, n=8, seed=0, scale=0.0, log=None, **course_kw):
    cfg = CourseConfig(
        reward_reference_dt_s=REF_DT,
        no_obs=True,
        pool_size=32,
        start=start or StartConfig(mode="mixed", ground_fraction=0.0, air_z_m=(1.2, 1.2)),
        **course_kw,
    )
    return CourseEnv(
        meteor75(),
        n,
        "cpu",
        seed,
        cfg,
        PlantConfig(),
        timing or TimingConfig.fixed(50.0),
        scale,
        action_spec=V2,
        log_capacity=log,
    )


# ----------------------------------------------------------------------------- course_gen


def _assert_same(a, b, path="course"):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=path)
    else:
        assert a == b, path


@pytest.mark.parametrize(
    "kw",
    [TRACTABLE_KW, {}, {"with_tunnels": True}, {"with_variety": True, "laps_lo": 2, "laps_hi": 5}],
    ids=["tractable", "full", "tunnels", "variety-multilap"],
)
def test_course_gen_is_a_verbatim_port(harness_contract, kw):
    proto = harness_contract.import_reference("course_gen")
    for seed in (0, 1, 7, 123, 1023):
        _assert_same(sample_course(seed, **kw), proto.sample_course(seed, **kw))


def test_pool_matches_the_prototype_pool(gpe):
    cfg = CourseConfig.from_flags(vars(gpe), reward_reference_dt_s=REF_DT)
    gpe.BatchedProcCourse(2)  # builds the module-global pool
    ours = build_pool(1024, 0, cfg.no_obs, **cfg.pool_kwargs)
    for key, val in gpe._POOL.items():
        np.testing.assert_array_equal(
            np.asarray(val.cpu() if hasattr(val, "cpu") else val), ours[key], err_msg=key
        )


# ----------------------------------------------------------------------------- vs prototype


def test_course_config_from_flags_reads_the_module_globals(gpe):
    cfg = CourseConfig.from_flags(vars(gpe), reward_reference_dt_s=REF_DT)
    assert cfg.no_obs == gpe.NO_OBS and cfg.raycast == gpe.RAYCAST
    assert cfg.shaping == gpe.SHAPING and cfg.gap == gpe.GAP and cfg.floor == gpe.FLOOR
    assert cfg.obs_terminal == gpe.OBS_TERMINAL
    assert cfg.pool_kwargs == gpe.POOL_KW
    assert cfg.extra_dim == gpe.EXTRA_DIM


def test_course_env_matches_batched_proc_course(gpe):
    n, horizon = 48, 15  # < the 20-step floor grace: no respawn in the window
    cfg = CourseConfig.from_flags(
        vars(gpe), reward_reference_dt_s=REF_DT, start=StartConfig(mode="legacy")
    )
    torch.manual_seed(0)
    proto = gpe.BatchedProcCourse(n)
    env = CourseEnv(
        meteor75(),
        n,
        "cpu",
        0,
        cfg,
        PlantConfig(integrator="euler-legacy", substeps=1),
        TimingConfig.fixed(50.0),
        0.0,
        action_spec=ActionSpec(),  # the prototype's (v1) action semantics
    )
    env.reset_all(proto.cid.clone())
    assert torch.equal(env.cid, proto.cid)
    for key in ("pos", "vel", "quat", "omega"):
        setattr(proto, key, getattr(env, key).clone())
    proto.tgt.zero_()
    proto.lap.zero_()
    proto.t.zero_()
    proto.prev_clear = env.prev_clear.clone()
    proto.prev_floor = env.prev_floor.clone()

    def same_extra(a, b, k):
        # an occluded next gate gets a noisy bearing from each env's own RNG: compare the rest
        occluded = (a[:, 21] == 0) | (b[:, 21] == 0)
        mask = torch.ones_like(a, dtype=torch.bool)
        mask[occluded, 15:18] = False
        torch.testing.assert_close(a[mask], b[mask], atol=1e-5, rtol=0, msg=f"extra step {k}")
        assert torch.equal(a[:, 21], b[:, 21]), k

    same_extra(env.extra(), proto.extra(), -1)
    g = torch.Generator().manual_seed(3)
    for k in range(horizon):
        raw = (torch.rand(n, 4, generator=g) * 2.0 - 1.0) * 1.2
        o1, r1, d1, t1 = env.step(raw.clone())
        o2, r2, d2, t2 = proto.step(raw.clone())
        assert torch.equal(d1, d2) and not d1.any(), k
        torch.testing.assert_close(o1, o2, atol=1e-5, rtol=0, msg=f"obs step {k}")
        torch.testing.assert_close(r1, r2, atol=1e-4, rtol=0, msg=f"reward step {k}")
        torch.testing.assert_close(t1, t2, atol=1e-4, rtol=0, msg=f"tilt step {k}")
        assert torch.equal(env.tgt, proto.tgt) and torch.equal(env.lap, proto.lap), k
        same_extra(env.extra(), proto.extra(), k)


# ----------------------------------------------------------------------------- rate invariance


@pytest.mark.parametrize("hz", [50.0, 25.0, 40.0])
def test_dense_rewards_scale_with_dt(hz):
    """A level 1 g hover does not move: the reward is the dense upright + low-spin terms only."""
    env = _small_env(timing=TimingConfig.fixed(hz))
    n = env.n
    _, rew, done, _ = env.step_command(torch.ones(n), torch.zeros(n, 3))
    assert not done.any() and torch.all(env.tgt == 0)
    expect = (0.3 + 0.2) * (1.0 / hz) / REF_DT
    torch.testing.assert_close(rew, torch.full((n,), expect), atol=1e-5, rtol=0)


def test_a_skipped_tick_is_one_transition_of_two_periods():
    env = _small_env(timing=TimingConfig(control_hz=(40.0, 40.0), action_delay_s=(0.0, 0.0)))
    env.timing.skip_prob.fill_(1.0)  # every tick misses its deadline
    n = env.n
    _, rew, _, _ = env.step_command(torch.ones(n), torch.zeros(n, 3))
    assert env.last_skipped.all()
    torch.testing.assert_close(env.last_dt, torch.full((n,), 0.05, dtype=torch.float64))
    torch.testing.assert_close(env.plant.t, torch.full((n,), 0.05, dtype=torch.float64))
    assert torch.all(env.steps == 1)  # no extra policy step
    torch.testing.assert_close(rew, torch.full((n,), 0.5 * 0.05 / REF_DT), atol=1e-5, rtol=0)


@pytest.mark.parametrize("hz,steps", [(50.0, 5), (25.0, 3)])
def test_episode_length_is_a_time(hz, steps):
    env = _small_env(timing=TimingConfig.fixed(hz), episode_s=0.1, log=64)
    n = env.n
    for k in range(1, steps + 1):
        _, _, done, _ = env.step_command(torch.ones(n), torch.zeros(n, 3))
        assert bool(done.all()) == (k == steps), k
    rows = env.log.drain()
    col = EPISODE_COLUMNS.index
    assert len(rows) == n
    assert np.all(rows[:, col("outcome")] == OUTCOMES.index("timeout"))
    assert np.all(rows[:, col("steps")] == steps)
    np.testing.assert_allclose(rows[:, col("length_s")], steps / hz, rtol=1e-5)


@pytest.mark.parametrize("hz", [50.0, 25.0])
def test_floor_kill_grace_is_a_time(hz):
    env = _small_env(timing=TimingConfig.fixed(hz), start=StartConfig(mode="legacy"))
    n = env.n
    pos = env.pos.clone()
    pos[:, 2] = 0.12  # the prototype's on-the-ground spawn height
    env.pos = pos
    below_at = None
    for k in range(1, 41):
        _, _, done, _ = env.step_command(torch.zeros(n), torch.zeros(n, 3))  # motors off
        if done.any():
            break
        if below_at is None and bool((env.pos[:, 2] < 0.08).all()):
            below_at = k  # under the floor-kill height, but inside the grace
    assert below_at is not None and below_at / hz <= 0.4
    assert done.all() and env.cause_floor.all()
    # the crash lands on the first step whose time passes the 0.4 s grace (20 steps at 50 Hz)
    assert k == int(np.floor(0.4 * hz + 1e-9)) + 1
    assert k / hz > 0.4 and (k - 1) / hz <= 0.4 + 1e-9


# ----------------------------------------------------------------------------- episode log


def test_episode_log_rows_carry_the_samples_and_the_outcome():
    timing = TimingConfig(control_hz=(25.0, 50.0), action_delay_s=(0.0, 0.03))
    env = _small_env(timing=timing, episode_s=0.2, log=64, scale=1.0, n=16)
    params = env.plant.params.table.clone()
    timing_cols = env.timing.columns().clone()
    modes = env.start_mode.clone()
    for _ in range(30):
        _, _, done, _ = env.step_command(torch.ones(16), torch.zeros(16, 3))
    rows = env.log.drain()
    col = EPISODE_COLUMNS.index
    first = {}
    for r in rows:  # the first finished episode of every env is the one sampled above
        first.setdefault(int(r[col("env")]), r)
    assert sorted(first) == list(range(16))
    for i, r in first.items():
        np.testing.assert_allclose(
            r[[col(c) for c in LOG_COLUMNS]], params[i].numpy(), rtol=1e-5, atol=1e-7
        )
        np.testing.assert_allclose(r[col("control_hz")], timing_cols[i, 0].item(), rtol=1e-5)
        np.testing.assert_allclose(r[col("action_delay_s")], timing_cols[i, 1].item(), atol=1e-7)
        assert r[col("start_mode")] == modes[i].item()
        assert r[col("outcome")] == OUTCOMES.index("timeout") and r[col("completed")] == 0
        assert r[col("length_s")] >= 0.2 - 1e-6
    # every env was resampled at its reset: the table changed
    assert not torch.equal(params, env.plant.params.table)


def test_episode_log_capacity_drops_and_counts():
    env = _small_env(episode_s=0.02, log=4, n=8)
    env.step_command(torch.ones(8), torch.zeros(8, 3))
    rows = env.log.drain()
    assert len(rows) == 4 and env.log.dropped == 4
    assert len(env.log.drain()) == 0


def test_reset_all_on_given_course_ids():
    env = _small_env(n=4)
    env.reset_all(torch.tensor([3, 1, 4, 1]))
    assert env.cid.tolist() == [3, 1, 4, 1]
    torch.testing.assert_close(env.pos[:, :2], env.P["start"][env.cid][:, :2], atol=0.3, rtol=0)


# ----------------------------------------------------------------------------- config


def test_reward_reference_dt_has_no_default():
    """[C4]: no sim default for reward_reference_dt_s."""
    with pytest.raises(TypeError):
        CourseConfig()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="reward_reference_dt_s"):
        CourseConfig.from_dict({"tier": "tractable"})
    with pytest.raises(ValueError, match="positive"):
        CourseConfig(reward_reference_dt_s=0.0)
    with pytest.raises(ValueError, match="CourseConfig"):
        CourseEnv(meteor75(), 2, "cpu", 0, None, action_spec=V2)


@pytest.mark.parametrize(
    "bad",
    [
        {"tier": "nightmare"},
        {"start": {"mode": "teleport"}},
        {"start": {"ground_fraction": 1.5}},
        {"start": {"air_z_m": [2.0, 1.0]}},
        {"start": {"arm_delay_s": -0.1}},
        {"pool": 3},
    ],
)
def test_course_config_is_validated(bad):
    with pytest.raises(ValueError):
        CourseConfig.from_dict({"reward_reference_dt_s": REF_DT, **bad})


def test_extra_dim_follows_raycast():
    assert CourseConfig(reward_reference_dt_s=REF_DT).extra_dim == 30
    assert CourseConfig(reward_reference_dt_s=REF_DT, raycast=True).extra_dim == 70
    env = _small_env(n=2, raycast=True)
    assert env.extra().shape == (2, 70)
