"""PPO training loop wiring the connectome policy to the racing env (AC4, AC6, AC9).

Responsibilities:

* Build an SB3 ``PPO`` whose feature backbone is UC-02's
  :class:`~drone_fly.controller.sb3.ConnectomeFeaturesExtractor`, wired via
  ``policy_kwargs`` with ``net_arch=dict(pi=[], vf=[...])`` so the 4-channel connectome
  features stay **load-bearing** (an empty policy head means the connectome output *is* the
  policy latent; SB3 still appends a final linear action net + ``log_std``).
* Periodic checkpointing with SB3 ``CheckpointCallback(save_vecnormalize=True)`` and a
  TensorBoard **and** CSV learning curve (AC4).
* Checkpoint/resume (AC6): ``PPO.load(ckpt, env=venv)`` + ``VecNormalize.load(stats, venv)``
  + ``set_env`` → ``learn(reset_num_timesteps=False)`` so ``num_timesteps`` continues from
  the checkpoint rather than restarting.
* A tiny ``smoke_train`` (AC9) on the pure-numpy backend for CI.

The connectome may be passed in (tests inject the committed fixture for a fully hermetic
run) or loaded from disk via :func:`drone_fly.connectome.load_connectome`.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from pathlib import Path

from drone_fly.connectome import load_connectome
from drone_fly.connectome.loader import ConnectomeData
from drone_fly.connectome.prune import DEFAULT_PRUNE_K, prune_to_subcircuit
from drone_fly.controller.sb3 import ConnectomeFeaturesExtractor
from drone_fly.env.config import EnvConfig
from drone_fly.env.racing_env import build_vec_env
from drone_fly.train.config import TrainConfig
from drone_fly.train.device import resolve_device

logger = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "ppo_racer"


def build_policy_kwargs(connectome: ConnectomeData, cfg: TrainConfig) -> dict:
    """Assemble ``policy_kwargs`` keeping the connectome features load-bearing (AC4).

    ``net_arch=dict(pi=[], vf=cfg.vf_arch)``: the policy head is empty so PPO's action
    distribution is a linear map straight off the 4-channel connectome features; the value
    head is a small MLP on the same features.
    """
    return {
        "features_extractor_class": ConnectomeFeaturesExtractor,
        "features_extractor_kwargs": {"data": connectome},
        "net_arch": {"pi": [], "vf": list(cfg.vf_arch)},
    }


def _make_logger(logs_dir: str):
    """SB3 logger writing stdout + TensorBoard + CSV (AC4 learning curve)."""
    from stable_baselines3.common.logger import configure

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    return configure(logs_dir, ["stdout", "csv", "tensorboard"])


def find_latest_checkpoint(models_dir: str) -> str | None:
    """Return the newest ``*_steps.zip`` checkpoint in ``models_dir`` (AC7 idempotent resume).

    Selects by the step count embedded in the filename, so ``train.sh`` re-runs resume from
    the furthest-along checkpoint rather than restarting.
    """
    matches = glob.glob(os.path.join(models_dir, f"{CHECKPOINT_PREFIX}_*_steps.zip"))
    if not matches:
        return None

    def _steps(path: str) -> int:
        m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(matches, key=_steps)


def _infer_vecnormalize_path(model_path: str, models_dir: str, cfg: TrainConfig) -> str | None:
    """Best-effort locate the VecNormalize stats saved alongside ``model_path``.

    Tries the ``CheckpointCallback`` naming (``..._vecnormalize_<steps>_steps.pkl``), then
    the canonical ``<models_dir>/<vecnormalize_name>``. Returns ``None`` if neither exists;
    the caller then starts fresh normalisation stats (documented best-effort — AC12).
    """
    m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(model_path))
    if m:
        cand = os.path.join(models_dir, f"{CHECKPOINT_PREFIX}_vecnormalize_{m.group(1)}_steps.pkl")
        if os.path.isfile(cand):
            return cand
    canonical = os.path.join(models_dir, cfg.vecnormalize_name)
    return canonical if os.path.isfile(canonical) else None


def train(
    cfg: TrainConfig | None = None,
    *,
    connectome: ConnectomeData | None = None,
    connectome_path: str | None = None,
    env_config: EnvConfig | None = None,
    adapter: str = "auto",
    device: str | None = None,
    resume: str | None = None,
    total_timesteps: int | None = None,
    prune: bool = False,
    prune_k: int = DEFAULT_PRUNE_K,
):
    """Run (or resume) PPO training; return the trained model.

    Parameters
    ----------
    cfg:
        Training config; defaults to :class:`TrainConfig`.
    connectome / connectome_path:
        Inject a loaded connectome (hermetic tests) or a path to load from; if both are
        ``None`` the default :func:`load_connectome` resolution is used.
    adapter:
        Backend selector (``"auto"`` picks pybullet if importable, else the numpy model).
    device:
        Explicit torch device override, or ``None`` for the AC8 auto-policy.
    resume:
        Path to a checkpoint ``.zip`` to continue from (``reset_num_timesteps=False``).
    total_timesteps:
        Override ``cfg.total_timesteps`` for this call.
    prune:
        Opt-in (UC-04): reduce the loaded connectome to its directed sensory→motor
        subcircuit via :func:`~drone_fly.connectome.prune.prune_to_subcircuit` before the
        policy is built. Default ``False`` leaves UC-01/02/03 behaviour byte-identical.
        A no-op (skipped with a warning) when ``resume`` is set, since the checkpoint
        already carries its own (possibly pruned) graph.
    prune_k:
        Corridor slack passed to the pruner when ``prune`` is set (default
        :data:`~drone_fly.connectome.prune.DEFAULT_PRUNE_K`).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    cfg = cfg or TrainConfig()
    steps = total_timesteps if total_timesteps is not None else cfg.total_timesteps
    resolved_device = resolve_device(device)

    if connectome is None:
        connectome = load_connectome(connectome_path)

    if prune:
        if resume is not None:
            logger.warning(
                "Ignoring --prune on a resume run: the checkpoint %s already carries its own "
                "(possibly pruned) connectome graph, so re-pruning here would desync it.",
                resume,
            )
        else:
            before = (connectome.neuron_count, connectome.edge_count)
            connectome = prune_to_subcircuit(connectome, k=prune_k)
            logger.info(
                "Applied subcircuit pruning (k=%d): %d -> %d neurons, %d -> %d edges.",
                prune_k,
                before[0],
                connectome.neuron_count,
                before[1],
                connectome.edge_count,
            )

    logger.info(
        "Training on connectome '%s' (%d neurons, %d edges); device=%s; adapter=%s",
        getattr(connectome, "source", "?"),
        connectome.neuron_count,
        connectome.edge_count,
        resolved_device,
        adapter,
    )

    Path(cfg.models_dir).mkdir(parents=True, exist_ok=True)

    resuming = resume is not None
    stats_path = None
    if resuming:
        stats_path = _infer_vecnormalize_path(resume, cfg.models_dir, cfg)
        if stats_path is None:
            logger.warning(
                "No VecNormalize stats found for resume checkpoint %s; starting fresh "
                "normalisation stats (best-effort; see AC12).",
                resume,
            )

    venv = build_vec_env(
        config=env_config,
        adapter=adapter,
        n_envs=cfg.n_envs,
        seed=cfg.seed,
        training=True,
        vecnormalize_path=stats_path,
    )

    if resuming:
        logger.info("Resuming from checkpoint %s (reset_num_timesteps=False).", resume)
        model = PPO.load(resume, env=venv, device=resolved_device)
        model.set_env(venv)
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            seed=cfg.seed,
            device=resolved_device,
            policy_kwargs=build_policy_kwargs(connectome, cfg),
        )

    model.set_logger(_make_logger(cfg.logs_dir))

    checkpoint_cb = CheckpointCallback(
        save_freq=max(cfg.checkpoint_freq // max(cfg.n_envs, 1), 1),
        save_path=cfg.models_dir,
        name_prefix=CHECKPOINT_PREFIX,
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=steps,
        reset_num_timesteps=not resuming,
        callback=checkpoint_cb,
        progress_bar=False,
    )

    # Persist a stable final model + canonical VecNormalize stats for eval/resume.
    final_model = os.path.join(cfg.models_dir, f"{CHECKPOINT_PREFIX}_final.zip")
    model.save(final_model)
    venv.save(os.path.join(cfg.models_dir, cfg.vecnormalize_name))
    logger.info("Saved final model to %s and VecNormalize stats.", final_model)

    return model


def smoke_train(
    *,
    connectome: ConnectomeData | None = None,
    connectome_path: str | None = None,
    cfg: TrainConfig | None = None,
    timesteps: int | None = None,
    prune: bool = False,
    prune_k: int = DEFAULT_PRUNE_K,
):
    """A few-step training run on the pure-numpy backend (AC9, CI).

    Forces ``adapter="simple"`` (no pybullet) and a tiny step budget, proving the env +
    connectome policy + PPO loop + checkpointing wire together and stay finite. ``prune`` /
    ``prune_k`` (UC-04) are threaded through so the pruned subcircuit can be smoke-tested
    end-to-end.
    """
    cfg = cfg or TrainConfig()
    steps = timesteps if timesteps is not None else cfg.smoke_timesteps
    # Keep the on-policy rollout short so a smoke run actually completes an update.
    cfg.n_steps = min(cfg.n_steps, max(steps // max(cfg.n_envs, 1), 16))
    cfg.batch_size = min(cfg.batch_size, cfg.n_steps)
    return train(
        cfg,
        connectome=connectome,
        connectome_path=connectome_path,
        adapter="simple",
        device="cpu",
        total_timesteps=steps,
        prune=prune,
        prune_k=prune_k,
    )
