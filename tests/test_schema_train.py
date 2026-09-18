"""UC-13 (AC4) — the migrated block schema trains finitely on the committed fixture.

AC4 re-binds today's flight observation into biologically-mapped blocks (vision +
proprioception) and requires a **retrain** under the new schema (the old checkpoint is not
carried over). This is the trainability sanity check: a few-step ``smoke_train`` with the
migrated schema wired through must complete a finite PPO update on the pure-numpy backend —
proving the schema-mode actor + env + PPO loop + checkpointing wire together end-to-end.

Hermetic: ``adapter="simple"`` (no pybullet), tiny step budget, runs on the 322-neuron
fixture with no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.obs_schema import MIGRATED_SCHEMA_V1


def test_migrated_schema_smoke_trains_finitely(connectome: ConnectomeData, tmp_path) -> None:
    from drone_fly.train.config import TrainConfig
    from drone_fly.train.loop import smoke_train

    cfg = TrainConfig(
        models_dir=str(tmp_path / "models"),
        logs_dir=str(tmp_path / "logs"),
        checkpoint_freq=64,
        n_envs=1,
        n_steps=64,
        batch_size=32,
        seed=0,
    )
    model = smoke_train(
        connectome=connectome,
        cfg=cfg,
        timesteps=128,
        obs_schema=MIGRATED_SCHEMA_V1,
    )
    # A finite, completed update: the run reached the requested step budget...
    assert model.num_timesteps == 128
    # ...and the schema-mode actor is the one that trained (block projections, not legacy).
    actor = model.policy.features_extractor.actor
    assert actor.obs_schema == MIGRATED_SCHEMA_V1
    assert actor.sensory_mode == "schema"
    # Parameters stayed finite (no NaN/Inf blow-up during the update) — trainability (AC4).
    for p in actor.parameters():
        assert np.isfinite(p.detach().cpu().numpy()).all()


def test_obstacle_vision_v2_smoke_trains_finitely_on_the_obstacle_course(
    connectome: ConnectomeData, tmp_path
) -> None:
    """UC-15 AC8: a smoke-train under ``obstacle_vision_v2`` on ``default_obstacle_course``
    completes a finite PPO update end-to-end (hermetic, numpy backend).

    Proves the whole obstacle path wires together: the 24-d obstacle-vision env (widened by the
    training coordinator's name-based ``k`` reconciliation + the fail-loud
    ``env.obs_width == total_width`` assertion), the v2 schema-mode actor, the fixed obstacle
    course, PPO, and checkpointing — one completed update with no NaN/Inf blow-up.
    """
    from drone_fly.controller.obs_schema import OBSTACLE_VISION_V2
    from drone_fly.env.config import EnvConfig, default_obstacle_course
    from drone_fly.train.config import TrainConfig
    from drone_fly.train.loop import smoke_train

    cfg = TrainConfig(
        models_dir=str(tmp_path / "models"),
        logs_dir=str(tmp_path / "logs"),
        checkpoint_freq=64,
        n_envs=1,
        n_steps=64,
        batch_size=32,
        seed=0,
    )
    model = smoke_train(
        connectome=connectome,
        cfg=cfg,
        timesteps=128,
        obs_schema=OBSTACLE_VISION_V2,
        env_config=EnvConfig(course=default_obstacle_course()),
    )
    # A finite, completed update against the 24-d obstacle-vision observation.
    assert model.num_timesteps == 128
    actor = model.policy.features_extractor.actor
    assert actor.obs_schema == OBSTACLE_VISION_V2
    assert actor.sensory_mode == "schema"
    # The env genuinely emitted the wider observation the v2 schema demands.
    assert model.env.observation_space.shape == (OBSTACLE_VISION_V2.total_width,) == (24,)
    for p in actor.parameters():
        assert np.isfinite(p.detach().cpu().numpy()).all()


def test_battery_hunger_v3_smoke_trains_finitely_on_the_obstacle_course(
    connectome: ConnectomeData, tmp_path
) -> None:
    """UC-17 AC7: a smoke-train under ``battery_hunger_v3`` completes a finite PPO update
    end-to-end (hermetic, numpy backend).

    Proves the whole battery path wires together: ``_reconcile_battery`` enables battery physics +
    the width-1 battery obs dim (coupled), the hunger population resolves on the regenerated
    fixture, the 25-d env passes the ``env.obs_width == total_width`` assertion, the v3 schema-mode
    actor + PPO + checkpointing run, and no NaN/Inf blow-up occurs.
    """
    from drone_fly.controller.obs_schema import BATTERY_HUNGER_V3
    from drone_fly.env.config import EnvConfig, default_obstacle_course
    from drone_fly.train.config import TrainConfig
    from drone_fly.train.loop import smoke_train

    cfg = TrainConfig(
        models_dir=str(tmp_path / "models"),
        logs_dir=str(tmp_path / "logs"),
        checkpoint_freq=64,
        n_envs=1,
        n_steps=64,
        batch_size=32,
        seed=0,
    )
    model = smoke_train(
        connectome=connectome,
        cfg=cfg,
        timesteps=128,
        obs_schema=BATTERY_HUNGER_V3,
        env_config=EnvConfig(course=default_obstacle_course()),
    )
    assert model.num_timesteps == 128
    actor = model.policy.features_extractor.actor
    assert actor.obs_schema == BATTERY_HUNGER_V3
    assert actor.sensory_mode == "schema"
    # The env genuinely emitted the 25-d battery-hunger observation.
    assert model.env.observation_space.shape == (BATTERY_HUNGER_V3.total_width,) == (25,)
    for p in actor.parameters():
        assert np.isfinite(p.detach().cpu().numpy()).all()


def test_damage_proprioception_v4_smoke_trains_finitely_on_the_repair_course(
    connectome: ConnectomeData, tmp_path
) -> None:
    """UC-19 AC8: a smoke-train under ``damage_proprioception_v4`` on ``default_repair_course()``
    completes a finite PPO update end-to-end (hermetic, numpy backend).

    Proves the whole damage path wires together: the ``_reconcile_damage(_reconcile_battery(...))``
    chain forces BOTH damage and battery physics + their width-1 obs dims on (v4 carries both
    blocks), the proprioceptive population resolves, the 26-d env passes the ``env.obs_width ==
    total_width`` assertion, the v4 schema-mode actor (two proprioceptive blocks) + PPO +
    checkpointing run on the damage-heavy repair course, and no NaN/Inf blow-up occurs.
    """
    from drone_fly.controller.obs_schema import DAMAGE_PROPRIOCEPTION_V4
    from drone_fly.env.config import EnvConfig, default_repair_course
    from drone_fly.train.config import TrainConfig
    from drone_fly.train.loop import smoke_train

    cfg = TrainConfig(
        models_dir=str(tmp_path / "models"),
        logs_dir=str(tmp_path / "logs"),
        checkpoint_freq=64,
        n_envs=1,
        n_steps=64,
        batch_size=32,
        seed=0,
    )
    model = smoke_train(
        connectome=connectome,
        cfg=cfg,
        timesteps=128,
        obs_schema=DAMAGE_PROPRIOCEPTION_V4,
        env_config=EnvConfig(course=default_repair_course()),
    )
    assert model.num_timesteps == 128
    actor = model.policy.features_extractor.actor
    assert actor.obs_schema == DAMAGE_PROPRIOCEPTION_V4
    assert actor.sensory_mode == "schema"
    # The env genuinely emitted the 26-d damage-proprioception observation (obs_width == 26).
    assert model.env.observation_space.shape == (DAMAGE_PROPRIOCEPTION_V4.total_width,) == (26,)
    for p in actor.parameters():
        assert np.isfinite(p.detach().cpu().numpy()).all()


def test_default_course_energy_budget_keeps_battery_above_knee() -> None:
    """UC-17 AC7 (controller-free energy budget): the default course is completable WITHOUT
    recharge — even the worst case (full throttle every step for the entire step budget) leaves
    the battery above the ``knee``, so the thrust ceiling never sags into the crash regime.

    Deterministic and controller-free: it drives a battery-enabled adapter with the max-drain
    action (throttle=1.0) for the full default step budget and checks the residual charge. No
    policy, no RNG — a hard bound on the drain schedule."""
    from drone_fly.adapter.simple import SimpleDroneAdapter
    from drone_fly.env.config import BatteryConfig, EnvConfig

    env_cfg = EnvConfig()  # default 3-gate course
    ep = env_cfg.episode
    # Env truncation budget: max_steps + steps_per_gate * (num_gates - 1) (see EpisodeConfig).
    budget = ep.max_steps + ep.steps_per_gate * (env_cfg.course.num_gates - 1)
    assert budget == 800  # default 3-gate course

    battery_cfg = BatteryConfig(enabled=True)
    ad = SimpleDroneAdapter(
        np.array([0.0, 0.0, 1.0]), floor_z=0.0, ceiling_z=2.5, dt=ep.dt, battery=battery_cfg
    )
    ad.reset(seed=0)
    st = None
    for _ in range(budget):
        st = ad.step(np.array([1.0, 0.0, 0.0, 0.0]))  # worst-case: full throttle every step
    assert st.battery > battery_cfg.knee, (
        f"full-throttle {budget}-step budget drained to {st.battery} (knee={battery_cfg.knee})"
    )

    # Analytic worst-case bound cross-check: end charge == 1 - (idle+throttle)*dt*budget.
    worst_drop = (battery_cfg.idle_rate + battery_cfg.throttle_rate * 1.0) * ep.dt * budget
    assert st.battery == pytest.approx(1.0 - worst_drop)
    assert (1.0 - worst_drop) > battery_cfg.knee  # the bound itself clears the knee
