"""End-to-end post-training activation-pruning workflow (UC-07 AC1–AC10).

Orchestrates the train→prune→fine-tune pipeline on a *trained* connectome policy:

1. **Measure** per-neuron importance over a representative episode set (completion-before).
2. **Select** the retained neurons (endpoints + above-threshold interneurons) and pin the
   sensory/motor sub-populations by neuron identity.
3. **Slice** the connectome to the retained neurons (UC-04's ``slice_connectome`` primitive).
4. **Connectivity-check** the pruned graph: every pinned motor must stay forward-reachable from
   the pinned sensory set (warn on partial, hard error on zero).
5. **Transfer** the trained weights exactly onto a fresh pruned+pinned PPO policy.
6. **Fine-tune** briefly (opt-in) to recover completion; measure completion after prune and after
   fine-tune.
7. **Save** the fine-tuned pruned checkpoint, the pruned connectome slice (round-trips through
   ``load_connectome``), a provenance note, and a Markdown report.

The step is **opt-in and off by default** (a new CLI subcommand); omitting it leaves UC-01..06
behaviour byte-identical. The orchestrator is the sole writer of the report/provenance artifacts.

The whole result is only meaningful relative to the measurement distribution (the use case's
central caveat): when neither randomization axis is enabled the produced circuit is
**course-specific** and the workflow says so loudly, in the logs, the report, and the slice
provenance — it is a demonstration of *this* run's used sub-network, not a general fly circuit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drone_fly.connectome import load_connectome, save_connectome, slice_connectome
from drone_fly.connectome.prune import DEFAULT_PRUNE_K, prune_to_subcircuit
from drone_fly.prune_trained.importance import (
    ABSOLUTE_MODE,
    DEFAULT_THRESHOLD,
    reachable_motors,
    select_kept_neurons,
)
from drone_fly.prune_trained.measure import (
    DEFAULT_EPS,
    DEFAULT_METRIC,
    measure_importance_checkpoint,
)
from drone_fly.prune_trained.transfer import (
    assert_pins_subset_of_kept,
    remap_indices,
    transfer_actor_weights,
    transfer_policy_heads,
)

logger = logging.getLogger(__name__)

#: Output artifact names written under ``out_dir``.
PRUNED_MODEL_NAME = "pruned_model.zip"
PRUNED_VECNORMALIZE_NAME = "vecnormalize.pkl"
PRUNED_CONNECTOME_STEM = "connectome_pruned"
REPORT_NAME = "PRUNE_TRAINED_REPORT.md"
PROVENANCE_NAME = "PRUNE_TRAINED_PROVENANCE.md"

#: Default number of episodes to measure importance / completion over.
DEFAULT_EPISODES = 10

#: Default fine-tune continue-training budget (env steps). ``0`` skips fine-tune entirely.
DEFAULT_FINETUNE_STEPS = 0


@dataclass
class PruneTrainedReport:
    """Structured result of a :func:`prune_trained` run (also serialised to Markdown)."""

    # scale
    neurons_before: int
    neurons_after: int
    edges_before: int
    edges_after: int
    fraction_neurons_removed: float
    fraction_edges_removed: float
    # metric provenance
    metric: str
    threshold: float
    threshold_mode: str
    cut_value: float
    eps: float
    episodes: int
    frames: int
    # metric summary
    metric_min: float
    metric_median: float
    metric_max: float
    kept_metric_min: float
    dropped_metric_max: float
    # endpoint / interneuron breakdown
    n_sensory: int
    n_motor: int
    n_interneurons_before: int
    n_interneurons_kept: int
    n_interneurons_dropped: int
    # completion
    completion_before: float
    completion_after_prune: float
    completion_after_finetune: float
    finetune_steps: int
    # connectivity + distribution
    disconnected_motors: list[int]
    n_motors_reachable: int
    course_specific: bool
    randomized: bool
    sensory_mode: str
    motor_mode: str
    # provenance
    source: str
    checkpoint: str
    # outputs
    out_dir: str
    report_path: str = ""
    connectome_npz: str = ""
    model_path: str = ""
    warnings: list[str] = field(default_factory=list)


def _randomization_flags(env_config) -> tuple[bool, bool]:
    """Return ``(randomized, course_specific)`` for the measurement env config."""
    from drone_fly.env.config import EnvConfig

    rcfg = (env_config or EnvConfig()).randomization
    randomized = bool(rcfg.enable_course or rcfg.enable_dynamics)
    return randomized, not randomized


def _evaluate_completion(
    model,
    *,
    n_episodes: int,
    seed: int,
    env_config,
    adapter: str,
    vecnormalize_path: str | None,
) -> float:
    """Run ``n_episodes`` deterministic episodes and return the completion rate."""
    from drone_fly.env.racing_env import build_vec_env

    venv = build_vec_env(
        config=env_config,
        adapter=adapter,
        n_envs=1,
        seed=seed,
        training=False,
        norm_reward=False,
        vecnormalize_path=vecnormalize_path,
    )
    completed = 0
    try:
        for _ep in range(n_episodes):
            obs = venv.reset()
            done = False
            info: dict = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, dones, infos = venv.step(action)
                done = bool(dones[0])
                info = infos[0]
            if info.get("completed"):
                completed += 1
    finally:
        venv.close()
    return completed / n_episodes if n_episodes else 0.0


def _build_pruned_ppo(
    orig_model,
    pruned,
    pinned_sensory: np.ndarray,
    pinned_motor: np.ndarray,
    *,
    env_config,
    adapter: str,
    device: str | None,
    seed: int,
    finetune_steps: int,
    vecnormalize_path: str | None,
):
    """Construct a fresh PPO on the pruned graph with pinned sub-populations.

    Reuses the original policy's ``net_arch`` (so the downstream heads transfer 1:1) and injects
    the pinned indices through ``features_extractor_kwargs`` (so they pickle into the checkpoint).
    """
    from stable_baselines3 import PPO

    from drone_fly.controller.sb3 import ConnectomeFeaturesExtractor
    from drone_fly.env.racing_env import build_vec_env
    from drone_fly.train.config import TrainConfig

    cfg = TrainConfig()
    orig_pk = dict(getattr(orig_model, "policy_kwargs", {}) or {})
    net_arch = orig_pk.get("net_arch", {"pi": [], "vf": list(cfg.vf_arch)})

    policy_kwargs = {
        "features_extractor_class": ConnectomeFeaturesExtractor,
        "features_extractor_kwargs": {
            "data": pruned,
            "sensory_index": np.asarray(pinned_sensory, dtype=np.int64),
            "motor_index": np.asarray(pinned_motor, dtype=np.int64),
        },
        "net_arch": net_arch,
    }

    # Keep the rollout short enough that even a tiny fine-tune budget completes an update.
    if finetune_steps > 0:
        n_steps = max(16, min(cfg.n_steps, finetune_steps))
    else:
        n_steps = 64
    batch_size = min(cfg.batch_size, n_steps)

    venv = build_vec_env(
        config=env_config,
        adapter=adapter,
        n_envs=1,
        seed=seed,
        training=True,
        vecnormalize_path=vecnormalize_path,
    )
    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=cfg.learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        seed=seed,
        device=device or "cpu",
        policy_kwargs=policy_kwargs,
    )
    return model


def prune_trained(
    *,
    checkpoint: str,
    out_dir: str,
    connectome_path: str | None = None,
    vecnormalize_path: str | None = None,
    prune: bool = False,
    prune_k: int = DEFAULT_PRUNE_K,
    metric: str = DEFAULT_METRIC,
    threshold: float = DEFAULT_THRESHOLD,
    threshold_mode: str = ABSOLUTE_MODE,
    episodes: int = DEFAULT_EPISODES,
    eps: float = DEFAULT_EPS,
    finetune_steps: int = DEFAULT_FINETUNE_STEPS,
    env_config=None,
    adapter: str = "simple",
    device: str | None = None,
    seed: int = 0,
) -> PruneTrainedReport:
    """Run the full activation-pruning workflow; return a :class:`PruneTrainedReport`.

    Parameters mirror the ``drone-fly prune-trained`` CLI. ``connectome_path`` + ``prune`` /
    ``prune_k`` reconstruct the exact graph the checkpoint was trained on (composing UC-04's
    structural prune when the checkpoint was trained with ``--prune``). The input checkpoint file
    and input connectome are never mutated (AC7).
    """
    from stable_baselines3 import PPO

    from drone_fly.controller.sb3 import actor_from_model

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. Reconstruct the exact training graph -------------------------------------------
    base = load_connectome(connectome_path)
    if prune:
        base = prune_to_subcircuit(base, k=prune_k)
    neurons_before = base.neuron_count
    edges_before = base.edge_count

    randomized, course_specific = _randomization_flags(env_config)

    # --- 2. Load the trained policy + align --------------------------------------------------
    from drone_fly.env.racing_env import build_vec_env

    load_env = build_vec_env(
        config=env_config,
        adapter=adapter,
        n_envs=1,
        seed=seed,
        training=False,
        norm_reward=False,
        vecnormalize_path=vecnormalize_path,
    )
    orig_model = PPO.load(checkpoint, env=load_env, device=device or "cpu")
    orig_actor = actor_from_model(orig_model)
    if base.neuron_count != orig_actor.n_neurons:
        load_env.close()
        raise ValueError(
            f"Alignment failure: the reconstructed connectome has {base.neuron_count} neurons but "
            f"the checkpoint's actor has {orig_actor.n_neurons}. Pass the SAME "
            f"--connectome/--prune/--prune-k used to train this checkpoint."
        )

    # --- 3. Measure importance (completion-before) -------------------------------------------
    imp = measure_importance_checkpoint(
        orig_model,
        base,
        n_episodes=episodes,
        seed=seed,
        metric=metric,
        eps=eps,
        env_config=env_config,
        adapter=adapter,
        vecnormalize_path=vecnormalize_path,
    )
    completion_before = imp.completion_rate

    # --- 4. Select retained neurons + pin sub-populations ------------------------------------
    kept, sel = select_kept_neurons(
        base, imp.values, threshold=threshold, threshold_mode=threshold_mode
    )

    orig_sensory = orig_actor.sensory_index.detach().cpu().numpy()
    orig_motor = orig_actor.motor_index.detach().cpu().numpy()
    assert_pins_subset_of_kept(orig_sensory, orig_motor, kept)
    pinned_sensory = remap_indices(orig_sensory, kept)
    pinned_motor = remap_indices(orig_motor, kept)

    # --- 5. Slice the connectome -------------------------------------------------------------
    slice_source = (
        f"{base.source} [activation-pruned metric={metric} thr={threshold} "
        f"mode={threshold_mode} episodes={episodes}]"
    )
    pruned = slice_connectome(base, kept, source=slice_source)
    if pruned.edge_count == 0:
        load_env.close()
        raise ValueError(
            "Activation pruning produced a graph with no edges (degenerate); the threshold is too "
            "aggressive. Lower --threshold or measure over more/representative episodes."
        )

    # --- 6. Post-prune connectivity check ----------------------------------------------------
    reached, disconnected = reachable_motors(pruned, pinned_sensory, pinned_motor)
    if len(reached) == 0:
        load_env.close()
        raise ValueError(
            "Post-prune connectivity failure: ZERO pinned motor neurons are forward-reachable from "
            "the pinned sensory neurons on the pruned graph — the circuit cannot fly and fine-tune "
            "cannot recover it. Lower --threshold to retain more interneurons."
        )
    run_warnings: list[str] = []
    if disconnected:
        msg = (
            f"{len(disconnected)}/{len(pinned_motor)} pinned motor neuron(s) became unreachable "
            f"from the sensory set after pruning (indices {disconnected[:8]}); the pruned circuit "
            f"is partially disconnected — completion-after is the real check."
        )
        logger.warning("Post-prune connectivity: %s", msg)
        run_warnings.append(msg)

    # --- 7. Build pruned+pinned PPO and transfer weights exactly -----------------------------
    load_env.close()
    pruned_model = _build_pruned_ppo(
        orig_model,
        pruned,
        pinned_sensory,
        pinned_motor,
        env_config=env_config,
        adapter=adapter,
        device=device,
        seed=seed,
        finetune_steps=finetune_steps,
        vecnormalize_path=vecnormalize_path,
    )
    pruned_actor = actor_from_model(pruned_model)
    transfer_actor_weights(orig_actor, pruned_actor, kept)
    transfer_policy_heads(orig_model.policy, pruned_model.policy)

    # --- 8. Completion after prune (before fine-tune) ----------------------------------------
    completion_after_prune = _evaluate_completion(
        pruned_model,
        n_episodes=episodes,
        seed=seed,
        env_config=env_config,
        adapter=adapter,
        vecnormalize_path=vecnormalize_path,
    )

    # --- 9. Fine-tune (opt-in) ---------------------------------------------------------------
    if finetune_steps > 0:
        logger.info("Fine-tuning the pruned policy for %d steps (continue-pass).", finetune_steps)
        pruned_model.learn(
            total_timesteps=finetune_steps,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        completion_after_finetune = _evaluate_completion(
            pruned_model,
            n_episodes=episodes,
            seed=seed,
            env_config=env_config,
            adapter=adapter,
            vecnormalize_path=vecnormalize_path,
        )
    else:
        logger.info("Fine-tune skipped (finetune_steps=0); after-finetune == after-prune.")
        completion_after_finetune = completion_after_prune

    if course_specific:
        warn = (
            "COURSE-SPECIFIC RESULT: importance was measured with NO domain randomization, so this "
            "circuit reflects one fixed course only — it is a demonstration of this run's used "
            "sub-network, NOT a general minimal fly flight circuit. Re-run with --randomize for a "
            "robust result."
        )
        logger.warning(warn)
        run_warnings.append(warn)

    # --- 10. Save artifacts ------------------------------------------------------------------
    model_path = out / PRUNED_MODEL_NAME
    pruned_model.save(str(model_path))
    vecnorm_env = pruned_model.get_vec_normalize_env()
    if vecnorm_env is not None:
        vecnorm_env.save(str(out / PRUNED_VECNORMALIZE_NAME))
    npz_path, _meta_path = save_connectome(pruned, out, stem=PRUNED_CONNECTOME_STEM)

    # UC-27 (AC-3/AC-10): provision positions for the activation-pruned subcircuit, with its own
    # node-set-keyed sidecars (``<stem>_positions.csv`` + ``<stem>_soma.csv``) beside this
    # artifact — keyed off the explicit npz path, never the pruned data.source. Real anatomy is
    # sourced tokenlessly from the SOURCE connectome's meta ``somaLocation`` (``source_data=base``,
    # the connectome activation-pruning ran over — its meta carries the soma column; the subcircuit
    # artifact's own meta does not), subset to this subcircuit's bodyids.
    from drone_fly.record.coordinates import DEFAULT_PROJECTION, resolve_positions

    resolve_positions(
        pruned,
        projection=DEFAULT_PROJECTION,
        artifact_npz=npz_path,
        source_data=base,
        persist=True,
    )

    # metric summary over all neurons + kept/dropped split
    values = np.asarray(imp.values, dtype=np.float64)
    kept_mask = np.zeros(base.neuron_count, dtype=bool)
    kept_mask[kept] = True
    dropped_vals = values[~kept_mask]
    report = PruneTrainedReport(
        neurons_before=neurons_before,
        neurons_after=pruned.neuron_count,
        edges_before=edges_before,
        edges_after=pruned.edge_count,
        fraction_neurons_removed=1.0 - pruned.neuron_count / neurons_before,
        fraction_edges_removed=1.0 - pruned.edge_count / max(edges_before, 1),
        metric=metric,
        threshold=float(threshold),
        threshold_mode=threshold_mode,
        cut_value=float(sel["cut_value"]),
        eps=float(eps),
        episodes=episodes,
        frames=imp.frames,
        metric_min=float(values.min()),
        metric_median=float(np.median(values)),
        metric_max=float(values.max()),
        kept_metric_min=float(values[kept_mask].min()),
        dropped_metric_max=float(dropped_vals.max()) if dropped_vals.size else float("nan"),
        n_sensory=sel["n_sensory"],
        n_motor=sel["n_motor"],
        n_interneurons_before=sel["n_interneurons_before"],
        n_interneurons_kept=sel["n_interneurons_kept"],
        n_interneurons_dropped=sel["n_interneurons_dropped"],
        completion_before=completion_before,
        completion_after_prune=completion_after_prune,
        completion_after_finetune=completion_after_finetune,
        finetune_steps=finetune_steps,
        disconnected_motors=disconnected,
        n_motors_reachable=len(reached),
        course_specific=course_specific,
        randomized=randomized,
        sensory_mode=pruned_actor.sensory_mode,
        motor_mode=pruned_actor.motor_mode,
        source=base.source,
        checkpoint=checkpoint,
        out_dir=str(out),
        connectome_npz=str(npz_path),
        model_path=str(model_path),
        warnings=run_warnings,
    )
    report_path = out / REPORT_NAME
    report_path.write_text(_render_report(report), encoding="utf-8")
    report.report_path = str(report_path)
    (out / PROVENANCE_NAME).write_text(_render_provenance(report), encoding="utf-8")

    logger.info(
        "prune-trained: %d -> %d neurons, %d -> %d edges; completion before/after-prune/"
        "after-finetune = %.0f%% / %.0f%% / %.0f%%. Report: %s",
        report.neurons_before,
        report.neurons_after,
        report.edges_before,
        report.edges_after,
        100 * completion_before,
        100 * completion_after_prune,
        100 * completion_after_finetune,
        report_path,
    )
    return report


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _render_report(r: PruneTrainedReport) -> str:
    """Render the human-facing Markdown run report (AC6/AC9)."""
    lines = [
        "# Post-training activation-pruning report (UC-07)",
        "",
        f"- **Source connectome:** {r.source}",
        f"- **Input checkpoint:** {r.checkpoint}",
        f"- **Importance metric:** `{r.metric}` (secondary stat: `active_fraction`)",
        f"- **Threshold:** {r.threshold} (mode `{r.threshold_mode}`, resolved cut = "
        f"{r.cut_value:g}); activity eps = {r.eps:g}",
        f"- **Episodes measured:** {r.episodes} ({r.frames} frames)",
        f"- **Sub-population selection:** sensory `{r.sensory_mode}` / motor `{r.motor_mode}` "
        "(pinned by neuron identity)",
        "",
        "## Reduction",
        "",
        f"- **Neurons:** {r.neurons_before} -> {r.neurons_after} "
        f"({_pct(r.fraction_neurons_removed)} removed)",
        f"- **Edges:** {r.edges_before} -> {r.edges_after} "
        f"({_pct(r.fraction_edges_removed)} removed)",
        f"- **Endpoints retained:** {r.n_sensory} sensory + {r.n_motor} motor (all kept)",
        f"- **Interneurons:** {r.n_interneurons_before} -> {r.n_interneurons_kept} "
        f"({r.n_interneurons_dropped} pruned)",
        "",
        "## Importance summary",
        "",
        f"- **All neurons ({r.metric}):** min {r.metric_min:g} / median {r.metric_median:g} / "
        f"max {r.metric_max:g}",
        f"- **Kept min:** {r.kept_metric_min:g}  **Dropped max:** {r.dropped_metric_max:g}",
        "",
        "## Completion rate (honest before / after / after-finetune)",
        "",
        f"- **Before pruning:** {_pct(r.completion_before)}",
        f"- **Immediately after pruning:** {_pct(r.completion_after_prune)}",
        f"- **After fine-tune ({r.finetune_steps} steps):** {_pct(r.completion_after_finetune)}",
        "",
        "## Connectivity",
        "",
        f"- **Pinned motors forward-reachable from sensory:** {r.n_motors_reachable}/"
        f"{r.n_motors_reachable + len(r.disconnected_motors)}",
        f"- **Disconnected motors:** {r.disconnected_motors if r.disconnected_motors else 'none'}",
        "",
        "## Distribution",
        "",
        f"- **Randomized measurement:** {r.randomized}",
        f"- **Course-specific:** {r.course_specific}",
    ]
    if r.course_specific:
        lines += [
            "",
            "> **WARNING — course-specific circuit.** Importance was measured with no domain "
            "randomization, so this slice reflects a single fixed course. It is a demonstration of "
            "the technique and of *this* run's used sub-network — NOT a general minimal fly flight "
            "circuit. A robust result requires measuring over a varied distribution "
            "(`--randomize` / `--randomize-dynamics`).",
        ]
    if r.warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {w}" for w in r.warnings]
    lines += [
        "",
        "## Outputs",
        "",
        f"- Fine-tuned pruned checkpoint: `{PRUNED_MODEL_NAME}`",
        f"- Pruned connectome slice: `{PRUNED_CONNECTOME_STEM}.npz` (+ `_meta.csv`; round-trips "
        "through `load_connectome`, viewer-loadable)",
        f"- Provenance: `{PROVENANCE_NAME}`",
        "",
        "> Completion numbers above are measured on the configured backend/episodes; real "
        "mastery numbers are a dev-time step on the owner's machine with the full trained policy "
        "and a real fine-tune (documented boundary, mirroring UC-03/UC-04).",
        "",
    ]
    return "\n".join(lines)


def _render_provenance(r: PruneTrainedReport) -> str:
    """Render the slice provenance note stored beside the pruned connectome."""
    course = (
        "course-specific (no randomization — NOT a general circuit)"
        if r.course_specific
        else "randomized distribution"
    )
    return (
        "# Activation-pruned connectome provenance (UC-07)\n\n"
        f"- **Source:** {r.source}\n"
        f"- **Input checkpoint:** {r.checkpoint}\n"
        f"- **Metric:** {r.metric} (threshold {r.threshold}, mode {r.threshold_mode}, "
        f"cut {r.cut_value:g})\n"
        f"- **Episodes:** {r.episodes} ({r.frames} frames), distribution: {course}\n"
        f"- **Scale:** {r.neurons_before} -> {r.neurons_after} neurons, "
        f"{r.edges_before} -> {r.edges_after} edges\n"
        f"- **Endpoints retained:** {r.n_sensory} sensory + {r.n_motor} motor\n"
        f"- **Completion before/after-prune/after-finetune:** {_pct(r.completion_before)} / "
        f"{_pct(r.completion_after_prune)} / {_pct(r.completion_after_finetune)}\n"
        f"- **Disconnected motors:** {r.disconnected_motors if r.disconnected_motors else 'none'}\n"
        "\nReuse the slice with `drone-fly evaluate --connectome "
        f"{r.out_dir} ...` (round-trips through load_connectome; viewer-loadable).\n"
    )
