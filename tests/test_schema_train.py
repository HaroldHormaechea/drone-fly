"""UC-13 (AC4) — the migrated block schema trains finitely on the committed fixture.

AC4 re-binds today's flight observation into biologically-mapped blocks (vision +
proprioception) and requires a **retrain** under the new schema (the old checkpoint is not
carried over). This is the trainability sanity check: a few-step ``smoke_train`` with the
migrated schema wired through must complete a finite PPO update on the pure-numpy backend —
proving the schema-mode actor + env + PPO loop + checkpointing wire together end-to-end.

Hermetic: ``adapter="simple"`` (no pybullet), tiny step budget, runs on the 300-neuron
fixture with no network.
"""

from __future__ import annotations

import numpy as np

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
