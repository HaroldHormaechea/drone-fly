"""AC4/AC6/AC9 — real PPO train → checkpoint → resume integration (no mocks).

This is a genuine end-to-end exercise of the training infrastructure on the hermetic
``SimpleDroneAdapter`` backend and the committed connectome fixture — the real
:class:`ConnectomeFeaturesExtractor` + SB3 ``PPO``, no stubbed policy. It proves:

* **AC9** — the env + connectome policy + PPO loop + checkpointing wire together and stay
  finite on a handful of steps, hermetically (``--extra dev``, no pybullet).
* **AC4** — periodic checkpoints and a learning curve (CSV + TensorBoard) are written.
* **AC6** — resuming from a checkpoint continues the step counter
  (``reset_num_timesteps=False``) rather than restarting, and the saved ``VecNormalize``
  stats round-trip through pickle (compared element-wise with ``assert_allclose``, not
  bit-exact ``==``, to survive the float representation of the pickle round-trip).

Kept tiny (a couple of PPO iterations) so it runs in a second or two on CI.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from drone_fly.env.racing_env import build_vec_env
from drone_fly.train.config import TrainConfig
from drone_fly.train.loop import (
    CHECKPOINT_PREFIX,
    _resolve_resume,
    find_latest_checkpoint,
    smoke_train,
    train,
)


@pytest.fixture
def tiny_cfg(tmp_path) -> TrainConfig:
    """A minimal, fast TrainConfig writing artifacts under a temp dir."""
    return TrainConfig(
        models_dir=str(tmp_path / "models"),
        logs_dir=str(tmp_path / "logs"),
        checkpoint_freq=64,
        n_envs=1,
        n_steps=64,
        batch_size=32,
        seed=0,
    )


def test_smoke_train_writes_checkpoints_and_learning_curve(connectome, tiny_cfg) -> None:
    model = smoke_train(connectome=connectome, cfg=tiny_cfg, timesteps=128)

    # AC9: the loop ran and the step counter is finite and advanced.
    assert model.num_timesteps == 128

    # AC4: a checkpoint .zip exists...
    ckpt = find_latest_checkpoint(tiny_cfg.models_dir)
    assert ckpt is not None
    assert os.path.isfile(ckpt)
    models = os.listdir(tiny_cfg.models_dir)
    assert any(f.startswith(CHECKPOINT_PREFIX) and f.endswith("_steps.zip") for f in models)
    # ...VecNormalize stats were saved alongside...
    assert tiny_cfg.vecnormalize_name in models

    # AC4: ...and both a CSV and a TensorBoard learning curve were emitted.
    logs = os.listdir(tiny_cfg.logs_dir)
    assert "progress.csv" in logs
    assert any(f.startswith("events.out.tfevents") for f in logs), "no TensorBoard event file"


def test_resume_advances_step_counter_not_restart(connectome, tiny_cfg) -> None:
    # 1) Train a little and checkpoint.
    m1 = smoke_train(connectome=connectome, cfg=tiny_cfg, timesteps=128)
    steps0 = m1.num_timesteps
    assert steps0 == 128
    ckpt = find_latest_checkpoint(tiny_cfg.models_dir)
    assert ckpt is not None

    # 2) Resume from the checkpoint for a few more steps.
    m2 = train(
        tiny_cfg,
        connectome=connectome,
        adapter="simple",
        device="cpu",
        resume=ckpt,
        total_timesteps=64,
    )

    # AC6: the counter continued from the checkpoint (reset_num_timesteps=False), it did
    # not restart from zero.
    assert m2.num_timesteps > steps0
    assert m2.num_timesteps == pytest.approx(steps0 + 64, abs=tiny_cfg.n_steps)


def test_resume_reloads_vecnormalize_stats_via_pickle_roundtrip(connectome, tiny_cfg) -> None:
    m1 = smoke_train(connectome=connectome, cfg=tiny_cfg, timesteps=128)

    # Capture the running normalisation stats before they hit disk.
    vn = m1.get_vec_normalize_env()
    assert vn is not None
    presave_mean = vn.obs_rms.mean.copy()
    presave_var = vn.obs_rms.var.copy()
    assert np.isfinite(presave_mean).all()
    assert np.isfinite(presave_var).all()

    # Reload the saved VecNormalize (eval mode) and compare element-wise. Use
    # assert_allclose — NOT ==: the pickle round-trip is not guaranteed bit-identical.
    stats_path = os.path.join(tiny_cfg.models_dir, tiny_cfg.vecnormalize_name)
    assert os.path.isfile(stats_path)
    venv = build_vec_env(
        adapter="simple",
        n_envs=1,
        training=False,
        norm_reward=False,
        vecnormalize_path=stats_path,
    )
    try:
        np.testing.assert_allclose(venv.obs_rms.mean, presave_mean, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(venv.obs_rms.var, presave_var, rtol=1e-6, atol=1e-8)
        assert np.isfinite(venv.obs_rms.mean).all()
        assert np.isfinite(venv.obs_rms.var).all()
        # Eval-mode VecNormalize must be frozen and not normalise reward.
        assert venv.training is False
        assert venv.norm_reward is False
    finally:
        venv.close()


def test_find_latest_checkpoint_picks_highest_step_count(connectome, tiny_cfg) -> None:
    # Two checkpoints (64 and 128 steps) are written; the resolver must pick the furthest.
    smoke_train(connectome=connectome, cfg=tiny_cfg, timesteps=128)
    ckpt = find_latest_checkpoint(tiny_cfg.models_dir)
    assert ckpt is not None
    assert ckpt.endswith("_128_steps.zip")


def test_find_latest_checkpoint_none_when_empty(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert find_latest_checkpoint(str(empty)) is None


# --------------------------------------------------------------------------- #
# _resolve_resume — --resume ("latest" sentinel / directory / explicit .zip / None)
# --------------------------------------------------------------------------- #
def _seed_fake_checkpoints(directory, steps) -> None:
    """Touch empty ``ppo_racer_<N>_steps.zip`` files so newest-selection can be tested
    without a real (slow) training run — resolution keys purely off the filename."""
    directory.mkdir(parents=True, exist_ok=True)
    for n in steps:
        (directory / f"{CHECKPOINT_PREFIX}_{n}_steps.zip").write_bytes(b"")


def test_resolve_resume_none_is_none(tmp_path) -> None:
    """No --resume (None) resolves to None (a fresh run), models_dir irrelevant."""
    assert _resolve_resume(None, str(tmp_path)) is None


def test_resolve_resume_latest_picks_newest_in_models_dir(tmp_path) -> None:
    """The "latest" sentinel resolves to the newest checkpoint in models_dir."""
    models = tmp_path / "models"
    _seed_fake_checkpoints(models, [64, 256, 128])
    resolved = _resolve_resume("latest", str(models))
    assert resolved.endswith(f"{CHECKPOINT_PREFIX}_256_steps.zip")


def test_resolve_resume_directory_picks_newest_in_that_dir(tmp_path) -> None:
    """A directory value resolves to the newest checkpoint inside *that* directory
    (not models_dir)."""
    models = tmp_path / "models"
    _seed_fake_checkpoints(models, [10])  # a decoy in models_dir that must NOT be picked
    other = tmp_path / "elsewhere"
    _seed_fake_checkpoints(other, [64, 512])
    resolved = _resolve_resume(str(other), str(models))
    assert resolved.endswith(f"{CHECKPOINT_PREFIX}_512_steps.zip")
    assert str(other) in resolved


def test_resolve_resume_explicit_zip_passthrough_unchanged(tmp_path) -> None:
    """An explicit .zip path is returned byte-identical — never validated or rewritten."""
    explicit = str(tmp_path / "some" / "ckpt.zip")  # need not even exist
    assert _resolve_resume(explicit, str(tmp_path / "models")) == explicit


def test_resolve_resume_empty_dir_raises_naming_dir(tmp_path) -> None:
    """A directory with no checkpoints raises FileNotFoundError naming the searched dir
    (loud hard error — never a silent fresh start)."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=str(empty)):
        _resolve_resume(str(empty), str(tmp_path / "models"))


def test_resolve_resume_latest_no_checkpoints_raises_naming_models_dir(tmp_path) -> None:
    """The "latest" sentinel with no checkpoints in models_dir raises, naming models_dir."""
    models = tmp_path / "models"
    models.mkdir()
    with pytest.raises(FileNotFoundError, match=str(models)):
        _resolve_resume("latest", str(models))


def test_train_resume_latest_continues_counter(connectome, tiny_cfg) -> None:
    """End-to-end: `train(..., resume="latest")` resolves the newest checkpoint in
    cfg.models_dir and continues the step counter rather than restarting."""
    m1 = smoke_train(connectome=connectome, cfg=tiny_cfg, timesteps=128)
    steps0 = m1.num_timesteps
    assert steps0 == 128
    assert find_latest_checkpoint(tiny_cfg.models_dir) is not None

    m2 = train(
        tiny_cfg,
        connectome=connectome,
        adapter="simple",
        device="cpu",
        resume="latest",
        total_timesteps=64,
    )
    assert m2.num_timesteps > steps0


def test_train_resume_directory_continues_counter(connectome, tiny_cfg) -> None:
    """End-to-end: `train(..., resume=<models_dir>)` resolves the newest checkpoint in that
    directory and continues the step counter."""
    m1 = smoke_train(connectome=connectome, cfg=tiny_cfg, timesteps=128)
    steps0 = m1.num_timesteps
    assert steps0 == 128

    m2 = train(
        tiny_cfg,
        connectome=connectome,
        adapter="simple",
        device="cpu",
        resume=tiny_cfg.models_dir,  # a directory -> newest checkpoint within it
        total_timesteps=64,
    )
    assert m2.num_timesteps > steps0


def test_train_resume_latest_no_checkpoint_raises(connectome, tiny_cfg) -> None:
    """`train(..., resume="latest")` with no checkpoint yet is a loud FileNotFoundError,
    never a silent train-from-scratch."""
    with pytest.raises(FileNotFoundError):
        train(
            tiny_cfg,
            connectome=connectome,
            adapter="simple",
            device="cpu",
            resume="latest",
            total_timesteps=64,
        )


# --------------------------------------------------------------------------- #
# --n-envs — loop-level override of cfg.n_envs at the build_vec_env call site
# --------------------------------------------------------------------------- #
class _StopBeforeTrain(Exception):
    """Sentinel to abort train() right after build_vec_env, keeping the test hermetic/fast."""


def test_train_n_envs_overrides_cfg_at_build_vec_env(connectome, tiny_cfg, monkeypatch) -> None:
    """Passing n_envs=N to train() reaches build_vec_env(n_envs=N), overriding cfg.n_envs."""
    assert tiny_cfg.n_envs == 1  # the override value below must differ to be meaningful
    captured: dict = {}

    def fake_build_vec_env(*args, **kwargs):
        captured["n_envs"] = kwargs.get("n_envs")
        raise _StopBeforeTrain

    monkeypatch.setattr("drone_fly.train.loop.build_vec_env", fake_build_vec_env)

    with pytest.raises(_StopBeforeTrain):
        train(
            tiny_cfg,
            connectome=connectome,
            adapter="simple",
            device="cpu",
            n_envs=3,
            total_timesteps=64,
        )
    assert captured["n_envs"] == 3


def test_train_n_envs_defaults_to_cfg_when_omitted(connectome, tmp_path, monkeypatch) -> None:
    """Omitting n_envs falls back to cfg.n_envs (not a hardcoded default)."""
    cfg = TrainConfig(
        models_dir=str(tmp_path / "models"),
        logs_dir=str(tmp_path / "logs"),
        checkpoint_freq=64,
        n_envs=2,  # distinct from 1 so a fallback bug (hardcoded 1) would be caught
        n_steps=64,
        batch_size=32,
        seed=0,
    )
    captured: dict = {}

    def fake_build_vec_env(*args, **kwargs):
        captured["n_envs"] = kwargs.get("n_envs")
        raise _StopBeforeTrain

    monkeypatch.setattr("drone_fly.train.loop.build_vec_env", fake_build_vec_env)

    with pytest.raises(_StopBeforeTrain):
        train(cfg, connectome=connectome, adapter="simple", device="cpu", total_timesteps=64)
    assert captured["n_envs"] == 2


def test_train_n_envs_save_freq_uses_resolved_count(connectome, tiny_cfg, monkeypatch) -> None:
    """The checkpoint save_freq divisor uses the resolved (overridden) n_envs, not cfg.n_envs.

    checkpoint_freq=64, n_envs override=4 -> save_freq = max(64 // 4, 1) = 16. We stop the run
    right after the CheckpointCallback is constructed and inspect its save_freq.
    """
    import stable_baselines3.common.callbacks as sb3_cb

    captured: dict = {}
    RealCB = sb3_cb.CheckpointCallback

    class SpyCheckpointCallback(RealCB):
        def __init__(self, *args, **kwargs):
            captured["save_freq"] = kwargs.get("save_freq", args[0] if args else None)
            raise _StopBeforeTrain

    monkeypatch.setattr(
        "stable_baselines3.common.callbacks.CheckpointCallback", SpyCheckpointCallback
    )

    with pytest.raises(_StopBeforeTrain):
        train(
            tiny_cfg,
            connectome=connectome,
            adapter="simple",
            device="cpu",
            n_envs=4,
            total_timesteps=64,
        )
    # tiny_cfg.checkpoint_freq == 64; resolved n_envs == 4 -> 64 // 4 == 16
    assert captured["save_freq"] == 16


def test_resume_smoke_tolerates_changed_n_envs(connectome, tiny_cfg) -> None:
    """Optional end-to-end: a checkpoint trained at n_envs=1 resumes fine at --n-envs 2."""
    m1 = smoke_train(connectome=connectome, cfg=tiny_cfg, timesteps=128)
    steps0 = m1.num_timesteps
    assert steps0 == 128
    ckpt = find_latest_checkpoint(tiny_cfg.models_dir)
    assert ckpt is not None

    m2 = train(
        tiny_cfg,
        connectome=connectome,
        adapter="simple",
        device="cpu",
        resume=ckpt,
        total_timesteps=64,
        n_envs=2,
    )
    # Resume continued (did not restart) even though the parallel env count changed.
    assert m2.num_timesteps > steps0
    assert m2.get_env().num_envs == 2
