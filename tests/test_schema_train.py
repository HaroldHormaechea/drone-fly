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
