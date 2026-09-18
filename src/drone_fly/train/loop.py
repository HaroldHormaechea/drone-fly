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


def build_policy_kwargs(connectome: ConnectomeData, cfg: TrainConfig, obs_schema=None) -> dict:
    """Assemble ``policy_kwargs`` keeping the connectome features load-bearing (AC4).

    ``net_arch=dict(pi=[], vf=cfg.vf_arch)``: the policy head is empty so PPO's action
    distribution is a linear map straight off the 4-channel connectome features; the value
    head is a small MLP on the same features.

    ``obs_schema`` (UC-13): when ``None`` (default) the ``features_extractor_kwargs`` are
    byte-identical to UC-01..12 (legacy single-projection actor). When an
    :class:`~drone_fly.controller.obs_schema.ObsSchema` is supplied it is added to the kwargs,
    so it is pickled into the checkpoint and ``PPO.load`` rebuilds the actor under the same
    block schema (AC3).
    """
    features_extractor_kwargs: dict = {"data": connectome}
    if obs_schema is not None:
        features_extractor_kwargs["obs_schema"] = obs_schema
    return {
        "features_extractor_class": ConnectomeFeaturesExtractor,
        "features_extractor_kwargs": features_extractor_kwargs,
        "net_arch": {"pi": [], "vf": list(cfg.vf_arch)},
    }


def _make_logger(logs_dir: str):
    """SB3 logger writing stdout + TensorBoard + CSV (AC4 learning curve)."""
    from stable_baselines3.common.logger import configure

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    return configure(logs_dir, ["stdout", "csv", "tensorboard"])


def _reconcile_obstacle_vision(env_config, obs_schema):
    """Make the env's obstacle-vision block agree with ``obs_schema`` (UC-15 width coupling).

    The obs schema is **authoritative** for the observation width. If it carries an
    ``obstacle_vision`` block, the env must emit that block: this returns an ``EnvConfig`` whose
    :class:`~drone_fly.env.config.ObstacleVisionConfig` is enabled with ``k`` derived
    **name-based** — ``block.width // OBSTACLE_FEATURES_PER`` — never by arithmetic on the
    schema's ``total_width`` (which mixes in the unrelated base blocks). When course
    randomization is already on, obstacle randomization is switched on too, so a randomized
    obstacle-vision run actually sees pillars. When ``obs_schema`` is ``None`` or has no such
    block, ``env_config`` is returned unchanged (byte-identical to pre-UC-15).
    """
    if obs_schema is None:
        return env_config
    block = next((b for b in obs_schema.blocks if b.name == "obstacle_vision"), None)
    if block is None:
        return env_config

    from dataclasses import replace

    from drone_fly.env.config import EnvConfig, ObstacleVisionConfig
    from drone_fly.env.obstacles import OBSTACLE_FEATURES_PER

    k = int(block.width) // OBSTACLE_FEATURES_PER
    base = env_config or EnvConfig()
    updates = {"obstacle_vision": ObstacleVisionConfig(enabled=True, k=k)}
    if base.randomization.enable_course and not base.randomization.enable_obstacles:
        updates["randomization"] = replace(base.randomization, enable_obstacles=True)
    return replace(base, **updates)


def _reconcile_battery(env_config, obs_schema):
    """Make the env's battery block agree with ``obs_schema`` (UC-17 width coupling).

    Mirrors :func:`_reconcile_obstacle_vision`: the obs schema is **authoritative** for the
    observation width. If it carries a ``battery`` block, the env must emit it — this returns an
    ``EnvConfig`` whose :class:`~drone_fly.env.config.BatteryConfig` is enabled (which turns on
    BOTH the adapter-side battery physics and the env-side width-1 battery observation dim; they
    are physically coupled). Any battery knobs already set on ``env_config`` (drain rates, knee,
    empty_factor) are preserved — only ``enabled`` is forced on. When ``obs_schema`` is ``None`` or
    has no ``battery`` block, ``env_config`` is returned unchanged (byte-identical to pre-UC-17).
    """
    if obs_schema is None:
        return env_config
    block = next((b for b in obs_schema.blocks if b.name == "battery"), None)
    if block is None:
        return env_config

    from dataclasses import replace

    from drone_fly.env.config import EnvConfig

    base = env_config or EnvConfig()
    return replace(base, battery=replace(base.battery, enabled=True))


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


def _resolve_resume(resume: str | None, models_dir: str) -> str | None:
    """Resolve the ``--resume`` argument to a concrete checkpoint ``.zip`` path (or ``None``).

    Resolution order (the ``"latest"`` sentinel is checked *before* the directory test):

    * ``None`` → ``None`` (fresh run, no resume).
    * ``"latest"`` sentinel (bare ``--resume``) → newest checkpoint in ``models_dir``.
    * an existing directory → newest checkpoint in that directory.
    * anything else → returned unchanged (an explicit ``.zip`` path; byte-identical to before).

    For the ``"latest"``/directory forms, if no ``*_steps.zip`` checkpoint is found we raise
    :class:`FileNotFoundError` naming the searched directory. This is a deliberate hard error:
    silently starting from scratch would be the exact data-loss (an unnoticed training rewind)
    this resolution is meant to prevent. An explicit ``.zip`` path is never validated here — it
    flows through untouched so existing behaviour is unchanged.
    """
    if resume is None:
        return None
    if resume == "latest":
        latest = find_latest_checkpoint(models_dir)
        if latest is None:
            raise FileNotFoundError(
                f"--resume: no {CHECKPOINT_PREFIX}_*_steps.zip checkpoint found in {models_dir!r}."
            )
        return latest
    if os.path.isdir(resume):
        latest = find_latest_checkpoint(resume)
        if latest is None:
            raise FileNotFoundError(
                f"--resume: no {CHECKPOINT_PREFIX}_*_steps.zip checkpoint found in {resume!r}."
            )
        return latest
    return resume


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
    n_envs: int | None = None,
    prune: bool = False,
    prune_k: int = DEFAULT_PRUNE_K,
    record: bool = False,
    record_every: int = 1,
    record_dir: str | None = None,
    obs_schema=None,
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
        Checkpoint to continue from (``reset_num_timesteps=False``). Accepts an explicit
        ``.zip`` path (byte-identical to before), a directory (resume its newest
        ``*_steps.zip``), or the ``"latest"`` sentinel (newest in ``cfg.models_dir``);
        resolved via :func:`_resolve_resume`. ``None`` starts fresh.
    total_timesteps:
        Override ``cfg.total_timesteps`` for this call.
    n_envs:
        Override ``cfg.n_envs`` (parallel rollout envs) for this call; ``None`` leaves the
        config value untouched (byte-identical to the unflagged run).
    prune:
        Opt-in (UC-04): reduce the loaded connectome to its directed sensory→motor
        subcircuit via :func:`~drone_fly.connectome.prune.prune_to_subcircuit` before the
        policy is built. Default ``False`` leaves UC-01/02/03 behaviour byte-identical.
        A no-op (skipped with a warning) when ``resume`` is set, since the checkpoint
        already carries its own (possibly pruned) graph.
    prune_k:
        Corridor slack passed to the pruner when ``prune`` is set (default
        :data:`~drone_fly.connectome.prune.DEFAULT_PRUNE_K`).
    obs_schema:
        Optional :class:`~drone_fly.controller.obs_schema.ObsSchema` (UC-13). ``None`` (default)
        trains the legacy single-projection actor exactly (AC7 parity). A schema re-binds the
        observation into biologically-mapped blocks; it is pickled into the checkpoint so a
        resumed / reloaded run reconstructs the same block layout. Ignored on a ``resume`` run
        (the checkpoint already carries its own schema).
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    cfg = cfg or TrainConfig()
    steps = total_timesteps if total_timesteps is not None else cfg.total_timesteps
    resolved_n_envs = n_envs if n_envs is not None else cfg.n_envs
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

    # Resolve --resume ("latest" sentinel / directory / explicit .zip) to a concrete path
    # before any resume logic runs, so a directory or bare --resume auto-selects the newest
    # checkpoint instead of erroring, while an explicit .zip stays byte-identical.
    resume = _resolve_resume(resume, cfg.models_dir)

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

    # UC-15/UC-17: the obs schema is authoritative for the observation width. Reconcile BOTH the
    # obstacle-vision block (widen the env + enable obstacle randomization when course
    # randomization is on) and the battery block (enable battery physics + the width-1 battery
    # obs dim). ``battery_hunger_v3`` carries both blocks, so both run; each is a no-op when its
    # block is absent (byte-identical to before). The generic width assertion below then validates
    # ``env.obs_width == obs_schema.total_width`` for the fully-reconciled config.
    env_config = _reconcile_battery(_reconcile_obstacle_vision(env_config, obs_schema), obs_schema)

    venv = build_vec_env(
        config=env_config,
        adapter=adapter,
        n_envs=resolved_n_envs,
        seed=cfg.seed,
        training=True,
        vecnormalize_path=stats_path,
    )

    # UC-15 fail-loud env↔schema width coupling: the env's observation width MUST equal the
    # schema's total width, else the schema-mode actor's block scatter would silently mismatch
    # the observation. Names both widths so a drift is diagnosable at once.
    if obs_schema is not None:
        env_obs_width = int(venv.get_attr("obs_width")[0])
        if env_obs_width != obs_schema.total_width:
            raise ValueError(
                f"env.obs_width ({env_obs_width}) != obs_schema.total_width "
                f"({obs_schema.total_width}); the obstacle-vision env block and the obs schema "
                f"are out of sync (UC-15 width coupling)."
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
            policy_kwargs=build_policy_kwargs(connectome, cfg, obs_schema=obs_schema),
        )

    model.set_logger(_make_logger(cfg.logs_dir))

    checkpoint_cb = CheckpointCallback(
        save_freq=max(cfg.checkpoint_freq // max(resolved_n_envs, 1), 1),
        save_path=cfg.models_dir,
        name_prefix=CHECKPOINT_PREFIX,
        save_vecnormalize=True,
    )

    callbacks: list = [checkpoint_cb]
    if record:
        # UC-05 best-effort training-time capture (documented; eval is the tested primary).
        # Records env-0's every-Nth episode during on-policy rollout collection. Guarded so
        # a recording error can never crash a training run.
        from drone_fly.record.recorder import ActivationRecorder
        from drone_fly.train.record_callback import RecordingCallback

        recorder = ActivationRecorder(
            connectome,
            record_dir or "artifacts/activations",
            backend=venv.get_attr("backend")[0],
            checkpoint="(training)",
            dt=(env_config or EnvConfig()).episode.dt,
            course=(env_config or EnvConfig()).course,
        )
        callbacks.append(RecordingCallback(recorder, record_every=record_every, seed=cfg.seed))
        logger.info(
            "Training-time activation recording enabled (every %d episodes, best-effort).",
            record_every,
        )

    model.learn(
        total_timesteps=steps,
        reset_num_timesteps=not resuming,
        callback=callbacks if len(callbacks) > 1 else checkpoint_cb,
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
    obs_schema=None,
    env_config: EnvConfig | None = None,
):
    """A few-step training run on the pure-numpy backend (AC9, CI).

    Forces ``adapter="simple"`` (no pybullet) and a tiny step budget, proving the env +
    connectome policy + PPO loop + checkpointing wire together and stay finite. ``prune`` /
    ``prune_k`` (UC-04) are threaded through so the pruned subcircuit can be smoke-tested
    end-to-end. ``obs_schema`` (UC-13) is likewise threaded through so the migrated block schema
    can be smoke-trained on the fixture — a finite completed update proves trainability (AC4).
    ``env_config`` (UC-15) lets a smoke run target a specific course — e.g.
    ``EnvConfig(course=default_obstacle_course())`` under ``obstacle_vision_v2`` — so the
    obstacle env + schema/graft path get an end-to-end finite-update proof (AC8). ``None`` keeps
    the fixed default course (byte-identical to prior smoke runs).
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
        env_config=env_config,
        adapter="simple",
        device="cpu",
        total_timesteps=steps,
        prune=prune,
        prune_k=prune_k,
        obs_schema=obs_schema,
    )
