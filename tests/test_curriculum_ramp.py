"""UC-87 redesign: the staged-curriculum ramp machinery — the ``_ramp_between`` schedule fn, the
new ``CurriculumConfig`` knobs, the fresh-train config loading/building, the ``CourseEnv`` deploy-DR
ramp (timing + plant), and the train-loop wiring that drives the three scales from the schedule.

All CPU-hermetic: pure-fn checks, config loads, and a tiny 1-iteration train on the fixture
connectome (no GPU, no real budget).
"""

from __future__ import annotations

import json

import pytest
import torch

from drone_fly import drones
from drone_fly.sim.action import ActionSpec
from drone_fly.sim.course import CourseConfig, CourseEnv, StartConfig
from drone_fly.sim.plant import PlantConfig
from drone_fly.sim.timing import TimingConfig
from drone_fly.training.config import TrainConfigError, load_train_config
from drone_fly.training.ppo import _ramp_between
from tests.sim_support import meteor75

V2 = ActionSpec.v2(1.0, 4.0)
FAST_TAU = (0.005, 0.083)
PROFILE_MIN = 0.038461538461538464


# ----------------------------------------------------------------------------- _ramp_between

def test_ramp_between_is_zero_before_start_linear_within_one_after_end():
    pair = (0.3, 0.6)
    assert _ramp_between(0.0, pair) == 0.0
    assert _ramp_between(0.3, pair) == 0.0  # at start
    assert _ramp_between(0.45, pair) == pytest.approx(0.5)  # midpoint
    assert _ramp_between(0.6, pair) == pytest.approx(1.0)  # at end
    assert _ramp_between(0.9, pair) == 1.0  # past end, clamped


def test_ramp_between_none_is_full_scale():
    # None ⇒ ramp off ⇒ pinned at 1.0 (the pre-redesign "full from step 0" behaviour)
    assert _ramp_between(0.0, None) == 1.0
    assert _ramp_between(1.0, None) == 1.0


def test_ramp_between_degenerate_pair_is_a_step():
    assert _ramp_between(0.4, (0.5, 0.5)) == 0.0
    assert _ramp_between(0.5, (0.5, 0.5)) == 1.0


def test_penalty_scale_formula_blends_init_to_one():
    # the ppo wiring: penalty_scale = penalty_init_scale + ramp·(1 − penalty_init_scale)
    init = 0.1
    for frac, want in [(0.0, 0.1), (0.5, 0.1), (0.65, 0.55), (0.8, 1.0), (1.0, 1.0)]:
        ramp = _ramp_between(frac, (0.5, 0.8))
        assert init + ramp * (1.0 - init) == pytest.approx(want)


# ----------------------------------------------------------------------------- CurriculumConfig

def test_curriculum_config_validates_fraction_pairs():
    from drone_fly.training.config import CurriculumConfig

    assert CurriculumConfig(timing_dr_frac=(0.3, 0.6)).timing_dr_frac == (0.3, 0.6)
    assert CurriculumConfig().timing_dr_frac is None  # default off
    assert CurriculumConfig().penalty_init_scale == 1.0
    for bad in [(0.6, 0.3), (-0.1, 0.5), (0.5, 1.2), (0.5,)]:
        with pytest.raises(TrainConfigError):
            CurriculumConfig(input_dr_frac=bad)
    with pytest.raises(TrainConfigError):
        CurriculumConfig(penalty_init_scale=1.5)


# ------------------------------------------------------------------- CourseEnv deploy ramp

def _deploy_env(n=4096, scale_cfg=1.0):
    cfg = CourseConfig(
        reward_reference_dt_s=0.02, no_obs=True, pool_size=32, episode_s=28.0, ceiling_z=50.0,
        start=StartConfig(mode="mixed", ground_fraction=0.0, air_z_m=(3.0, 3.0)),
    )  # fmt: skip
    ov = {f"rates.{a}.tau_s": FAST_TAU for a in ("roll", "pitch", "yaw")}
    plant = PlantConfig(range_overrides=ov)
    timing = TimingConfig(control_hz=(16.7, 50.0), action_delay_s=(0.0, 0.1), skip_prob=(0.0, 0.35))
    return CourseEnv(meteor75(), n, "cpu", 0, cfg, plant, timing, scale_cfg, action_spec=V2)


def test_deploy_dr_scale_zero_gives_clean_timing_and_nominal_plant():
    env = _deploy_env()
    env.deploy_dr_scale = 0.0
    env.reset_all()
    # clean fixed-rate timing (period == 1/fixed_hz), zero delay, zero skip
    torch.testing.assert_close(
        env.timing.period_s, torch.full_like(env.timing.period_s, 1.0 / 50.0)
    )
    assert torch.all(env.timing.delay_s == 0.0) and torch.all(env.timing.skip_prob == 0.0)
    # nominal plant: τ collapses to the profile nominal (no override span)
    nominal = meteor75().rates["roll"].tau_s
    torch.testing.assert_close(
        env.plant.params.tau_s, torch.full_like(env.plant.params.tau_s, nominal), rtol=0, atol=1e-6
    )


def test_deploy_dr_scale_one_opens_the_full_timing_and_tau_ranges():
    env = _deploy_env()
    env.deploy_dr_scale = 1.0
    env.reset_all()
    hz = 1.0 / env.timing.period_s
    assert hz.min() >= 16.7 - 1e-6 and hz.min() < 18.0  # the wide control range is in use
    assert env.timing.delay_s.max() > 0.09
    assert env.plant.params.tau_s.min().item() < PROFILE_MIN  # the fast-loop override span opens


def test_deploy_dr_scale_default_is_full():
    env = _deploy_env()
    assert env.deploy_dr_scale == 1.0 and env.input_dr_scale == 1.0 and env.penalty_scale == 1.0


# ----------------------------------------------------------------------------- fresh-train config

def test_fresh_config_loads_as_a_staged_fresh_train():
    cfg = load_train_config("uc87-deploy-robust")
    assert cfg.warm_start.checkpoint is None  # FRESH — no warm start
    assert cfg.course.spawn_yaw is False and cfg.course.spawn_distance_m is None  # aligned spawn
    assert cfg.course.memory is True and cfg.actor.extra_dim == 98 == cfg.course.extra_dim
    cur = cfg.curriculum
    assert cur.timing_dr_frac == (0.3, 0.6)
    assert cur.input_dr_frac == (0.6, 0.85)
    assert cur.penalty_frac == (0.5, 0.8) and cur.penalty_init_scale == 0.1
    # base-stage completion recipe restored (fresh run has no prior stage)
    assert cur.difficulty_frac is not None and cur.obs_count_frac is not None


def test_warmstart_config_is_the_proven_run5_recipe():
    # The warm-start config was PROMOTED to the shipped run-5 winner (the 2026-10-18/19 campaign).
    # It warm-starts model 11 and keeps difficulty FULL from step 0 (no destabilising ramp — null
    # fracs), then gently ramps ONLY the deployment/perception DR + the action-rate penalty on top,
    # earlier and over a shorter window than the fresh recipe. So it is NO LONGER identical to the
    # fresh curriculum — it has its own staggered schedule.
    ws = load_train_config("uc87-deploy-robust-warmstart")
    ck = ws.warm_start.checkpoint
    assert ck is not None and "11-count-curriculum" in ck
    assert ws.warm_start.src_extra_dim == 70  # zero-init the 28 memory columns
    cur = ws.curriculum
    # difficulty / obstacles held FULL from step 0 (null = no ramp — model 11 already solves them)
    assert cur.difficulty_frac is None
    assert cur.obs_pos_frac is None and cur.obs_count_frac is None
    # the proven run-5 staggered DR + penalty schedule (earlier/shorter than the fresh recipe)
    assert cur.timing_dr_frac == (0.15, 0.45)
    assert cur.input_dr_frac == (0.45, 0.75)
    assert cur.penalty_frac == (0.25, 0.65) and cur.penalty_init_scale == 0.1
    assert ws.course.spawn_yaw is False and ws.actor.extra_dim == 98


def test_fresh_config_builds_an_env():
    from drone_fly.training.ppo import build_env

    cfg = load_train_config("uc87-deploy-robust")
    env = build_env(cfg, drones.load(cfg.drone), torch.device("cpu"), n_envs=4, episode_log=False)
    assert env.extra_dim == 98
    assert env.deploy_dr_scale == 1.0  # the loop sets it per-iteration; the env default is full


# ----------------------------------------------------------------------------- train-loop wiring

@pytest.fixture(scope="module")
def fixture_k1(harness_contract, harness_k1_dir):
    from drone_fly.connectome import load_connectome

    return load_connectome(harness_k1_dir)


def test_train_loop_drives_the_three_scales_from_the_schedule(fixture_k1, tmp_path):
    from drone_fly.training.ppo import train

    # tiny 1-iteration fresh train on the fixture connectome; frac ≈ 0 at iteration 0
    cfg = load_train_config(
        "uc87-deploy-robust",
        {
            "n_envs": 4,
            "budget": 32,
            "device": "cpu",
            "course": {"pool_size": 16, "episode_s": 0.1},
            "ppo": {"roll": 8, "epochs": 1, "minibatches": 2},
            "log_every_iters": 1,
        },
    )
    res = train(
        cfg, drones.load(cfg.drone), run_root=tmp_path, connectome=fixture_k1,
        device="cpu", max_iterations=1,
    )  # fmt: skip
    rows = [
        json.loads(line)
        for line in (res.run.path / "logs" / "metrics.jsonl").read_text("utf-8").splitlines()
    ]
    assert rows, "no metrics row logged"
    row0 = rows[0]
    # the three staged scales are logged (AC3/AC6 report visibility)
    assert {"deploy_dr_scale", "input_dr_scale", "penalty_scale"} <= set(row0)
    # at frac ≈ 0: timing/input ramps haven't started (start 0.3/0.6) ⇒ 0; penalty at its init 0.1
    assert row0["deploy_dr_scale"] == 0.0
    assert row0["input_dr_scale"] == 0.0
    assert row0["penalty_scale"] == pytest.approx(0.1)
