"""CLI stage: thin command-line entry points.

Orchestrates the pipeline stages: fetch-connectome (stub), train, evaluate, prune,
prune-trained, smoke-train. Kept thin — argument parsing and wiring only; the real logic
lives in the stage subpackages (:mod:`drone_fly.train`, :mod:`drone_fly.evaluate`,
:mod:`drone_fly.prune_trained`).

Config-driven surface (UC-11)
-----------------------------
``train`` / ``evaluate`` / ``prune`` / ``prune-trained`` each take a single ``--config
<path.yaml>`` and load **all** their settings from it (the previous per-setting flags are
gone). Schemas + defaults live in :mod:`drone_fly.config`; a bad config raises
:class:`~drone_fly.config.ConfigError`, which :func:`main` turns into a one-line message and
exit code 2 (no stack trace). ``smoke-train`` and ``fetch-connectome`` keep their small
historical flag surface — they are CI/dev helpers, not the four settings-heavy commands.

Per-run output layout: a ``train`` config's required ``name`` routes all of that run's
outputs under ``training/<name>/{checkpoints,logs,recordings}/`` (see
:mod:`drone_fly.config`), so differently-named runs never collide.

The ``clean`` command wipes those training *outputs* to start from scratch. Like every other
command it operates **relative to the current working directory** (its target roots are the
same CWD-relative locations the other stages write to), so run it from the project root. It
is a **dry-run by default** — it lists what would be removed and deletes nothing — which also
protects against an accidental wrong-CWD invocation; an explicit ``--yes``/``--force`` is
required to delete (and ``--dry-run`` always wins over them). It takes no YAML config.
"""

from __future__ import annotations

import argparse
import logging
import sys

from drone_fly.connectome.prune import DEFAULT_PRUNE_K

#: Report filename hint printed by the CLI (kept in sync with workflow.REPORT_NAME).
REPORT_NAME_HINT = "PRUNE_TRAINED_REPORT.md"


def _add_prune_args(p: argparse.ArgumentParser) -> None:
    """Add the opt-in UC-04 subcircuit-pruning flags (still used by ``smoke-train``)."""
    p.add_argument(
        "--prune",
        action="store_true",
        help="Prune the connectome to its directed sensory->motor subcircuit before "
        "building the policy (UC-04). Off by default (UC-01/02/03 behaviour unchanged).",
    )
    p.add_argument(
        "--prune-k",
        type=int,
        default=DEFAULT_PRUNE_K,
        help=f"Path-slack corridor width for --prune (default {DEFAULT_PRUNE_K}; 0 = tight "
        "shortest-path corridor, larger = richer neighbourhood).",
    )


def _env_config(randomize: bool, randomize_dynamics: bool):
    """Build an :class:`EnvConfig` from the randomization settings, or ``None`` if both off.

    Returning ``None`` when neither axis is requested keeps train / evaluate byte-identical
    to UC-03 (the callees treat ``env_config=None`` as the fixed default).
    """
    if not (randomize or randomize_dynamics):
        return None
    from drone_fly.env.config import EnvConfig, RandomizationConfig

    return EnvConfig(
        randomization=RandomizationConfig(
            enable_course=randomize,
            enable_dynamics=randomize_dynamics,
        )
    )


def _warn_record_every_without_record(record_every: int | None, record: bool) -> None:
    """Warn when ``record_every`` is set without ``record`` (a silent no-op otherwise).

    ``record_every`` only takes effect when recording is enabled, so setting it alone is a
    user mistake worth surfacing. Detection relies on the config default being ``None`` (see
    :mod:`drone_fly.config`), so "explicitly set" is distinguishable from "left at default".
    """
    if record_every is not None and not record:
        logging.getLogger("drone_fly.cli").warning(
            "record_every was set without record; it has no effect. Set 'record: true' to "
            "enable activation recording."
        )


def _resolve_config_resume(resume: str | None, checkpoints_dir: str) -> str | None:
    """Translate a config ``resume`` value into what :func:`drone_fly.train.loop.train` expects.

    * ``None`` (omitted / ``null``) → ``None`` (fresh run).
    * ``"latest"`` → passed through unchanged; ``train`` resolves the newest checkpoint under
      ``checkpoints_dir`` and hard-errors if none exists (PR #12 behaviour).
    * ``"auto"`` → newest checkpoint if one exists, else ``None`` (fresh) — no error. This is
      the idempotent-bootstrap value ``scripts/train.sh`` relies on.
    * anything else → an explicit ``.zip`` path, passed through unchanged.

    ``train``/``_resolve_resume``/``find_latest_checkpoint`` are left untouched; the only new
    value (``"auto"``) is resolved here so the newest-else-fresh case never hard-errors.
    """
    if resume is None:
        return None
    if resume == "auto":
        from drone_fly.train.loop import find_latest_checkpoint

        return find_latest_checkpoint(checkpoints_dir)
    return resume


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drone-fly", description="Connectome-seeded drone racing."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train the PPO racing policy from a YAML config.")
    train_p.add_argument(
        "--config", required=True, help="Path to the training YAML config (see README)."
    )

    smoke_p = sub.add_parser("smoke-train", help="A few-step CI/correctness run (numpy backend).")
    smoke_p.add_argument("--timesteps", type=int, default=None, help="Override smoke timesteps.")
    smoke_p.add_argument("--connectome", default=None, help="Path to the cached connectome.")
    _add_prune_args(smoke_p)

    eval_p = sub.add_parser("evaluate", help="Evaluate a checkpoint from a YAML config.")
    eval_p.add_argument(
        "--config", required=True, help="Path to the evaluate YAML config (see README)."
    )

    prune_p = sub.add_parser(
        "prune",
        help="Prune a connectome to its sensory->motor subcircuit and write it to disk for reuse.",
    )
    prune_p.add_argument(
        "--config", required=True, help="Path to the prune YAML config (see README)."
    )

    pt_p = sub.add_parser(
        "prune-trained",
        help="Post-training activation prune: measure per-neuron usage on a trained policy, "
        "remove dead interneurons, fine-tune, and save the minimal circuit (UC-07).",
    )
    pt_p.add_argument(
        "--config", required=True, help="Path to the prune-trained YAML config (see README)."
    )

    clean_p = sub.add_parser(
        "clean",
        help="Wipe training outputs (checkpoints, logs, activations, training/<name>/) so a "
        "run can start from scratch. Dry-run by default; pass --yes to actually delete.",
    )
    clean_p.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete the training outputs (without this, or without --force, the "
        "command only lists what would be removed).",
    )
    clean_p.add_argument(
        "--force",
        action="store_true",
        help="Alias for --yes; either flag triggers deletion.",
    )
    clean_p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be removed and delete nothing. Wins over --yes/--force.",
    )
    clean_p.add_argument(
        "--include-prunes",
        action="store_true",
        help="Also remove the prepared prune slices (pruned* under artifacts/ and data/). "
        "Without this, prune slices and all other inputs are preserved.",
    )

    sub.add_parser("fetch-connectome", help="[stub] Provision connectome data (see README).")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    from drone_fly.config import ConfigError

    try:
        if args.command == "train":
            return _run_train(args.config)

        if args.command == "smoke-train":
            from drone_fly.train.loop import smoke_train

            smoke_train(
                connectome_path=args.connectome,
                timesteps=args.timesteps,
                prune=args.prune,
                prune_k=args.prune_k,
            )
            return 0

        if args.command == "evaluate":
            return _run_evaluate(args.config)

        if args.command == "prune":
            return _run_prune_export(args.config)

        if args.command == "prune-trained":
            return _run_prune_trained(args.config)

        if args.command == "clean":
            return _run_clean(args)

        if args.command == "fetch-connectome":
            print(
                "fetch-connectome is a stub. Provision the cached connectome under "
                "data/connectome/ (or set DRONE_FLY_CONNECTOME_DIR). See the README "
                "'Provisioning connectome data' section."
            )
            return 0
    except ConfigError as e:
        # Clean one-line message, no stack trace (AC6).
        logging.getLogger("drone_fly.cli").error("%s", e)
        return 2

    return 1  # pragma: no cover - argparse requires a subcommand


def _run_train(config_path: str) -> int:
    """Load a train config, build the ``training/<name>/`` layout, and dispatch to ``train``."""
    from drone_fly.config import TrainRunConfig, load_yaml, run_layout
    from drone_fly.controller.obs_schema import resolve_schema
    from drone_fly.train.config import TrainConfig
    from drone_fly.train.loop import train

    cfg = TrainRunConfig.from_mapping(load_yaml(config_path))
    layout = run_layout(cfg.name)

    _warn_record_every_without_record(cfg.record_every, cfg.record)
    record_every = cfg.record_every if cfg.record_every is not None else 1
    record_dir = cfg.record_dir if cfg.record_dir is not None else layout.recordings
    resume = _resolve_config_resume(cfg.resume, layout.checkpoints)
    # UC-13: None (default) -> legacy single-projection run (AC7 parity); a named schema opts in.
    obs_schema = resolve_schema(cfg.schema)

    # Route this run's checkpoints + logs under training/<name>/; every other TrainConfig
    # default is unchanged, so smoke-train (which never comes through here) stays identical.
    train_cfg = TrainConfig(models_dir=layout.checkpoints, logs_dir=layout.logs)

    train(
        train_cfg,
        connectome_path=cfg.connectome,
        env_config=_env_config(cfg.randomize, cfg.randomize_dynamics),
        adapter=cfg.adapter,
        device=cfg.device,
        resume=resume,
        total_timesteps=cfg.timesteps,
        n_envs=cfg.n_envs,
        prune=cfg.prune,
        prune_k=cfg.prune_k,
        record=cfg.record,
        record_every=record_every,
        record_dir=record_dir,
        obs_schema=obs_schema,
    )
    return 0


def _run_evaluate(config_path: str) -> int:
    """Load an evaluate config and dispatch to :func:`evaluate_checkpoint`."""
    from drone_fly.config import EvaluateRunConfig, load_yaml, run_layout
    from drone_fly.evaluate.evaluator import evaluate_checkpoint

    cfg = EvaluateRunConfig.from_mapping(load_yaml(config_path))

    _warn_record_every_without_record(cfg.record_every, cfg.record)
    record_every = cfg.record_every if cfg.record_every is not None else 1
    if cfg.record_dir is not None:
        record_dir: str | None = cfg.record_dir
    elif cfg.name and cfg.record:
        record_dir = run_layout(cfg.name).recordings
    else:
        record_dir = None  # evaluator keeps its historical artifacts/activations default

    metrics = evaluate_checkpoint(
        cfg.checkpoint,
        vecnormalize_path=cfg.vecnormalize,
        episodes=cfg.episodes,
        seed=cfg.seed,
        adapter=cfg.adapter,
        env_config=_env_config(cfg.randomize, cfg.randomize_dynamics),
        device=cfg.device,
        record=cfg.record,
        record_every=record_every,
        record_dir=record_dir,
        connectome_path=cfg.connectome,
        prune=cfg.prune,
        prune_k=cfg.prune_k,
    )
    print(metrics.summary())
    return 0


def _run_prune_export(config_path: str) -> int:
    """Load a connectome, prune it, and write the reusable pruned slice to ``out`` (AC11).

    Writes ``<out>/connectome_pruned.npz`` + ``connectome_pruned_meta.csv`` (round-trips
    through :func:`~drone_fly.connectome.load_connectome`) plus a ``PRUNE_PROVENANCE.md``
    recording source, rule, k, and the input→pruned counts.
    """
    from pathlib import Path

    from drone_fly.config import PruneRunConfig, load_yaml
    from drone_fly.connectome import load_connectome, prune_to_subcircuit, save_connectome

    cfg = PruneRunConfig.from_mapping(load_yaml(config_path))

    logger = logging.getLogger("drone_fly.cli.prune")
    data = load_connectome(cfg.connectome)
    before = (data.neuron_count, data.edge_count)
    pruned = prune_to_subcircuit(data, k=cfg.prune_k, rule=cfg.prune_rule)
    npz_path, meta_path = save_connectome(pruned, cfg.out)

    prov_path = Path(cfg.out) / "PRUNE_PROVENANCE.md"
    prov_path.write_text(
        "# Pruned connectome provenance\n\n"
        f"- **Source:** {data.source}\n"
        f"- **Rule:** {cfg.prune_rule}\n"
        f"- **k:** {cfg.prune_k}\n"
        f"- **Input scale:** {before[0]} neurons, {before[1]} edges\n"
        f"- **Pruned scale:** {pruned.neuron_count} neurons, {pruned.edge_count} edges\n"
        f"- **Matrix:** {npz_path.name}\n"
        f"- **Meta:** {meta_path.name}\n\n"
        "Reuse with `drone-fly train --config <cfg>` where the config sets `connectome: "
        f"{cfg.out}` (no prune — the slice is already pruned).\n"
    )

    logger.info(
        "Wrote pruned connectome to %s: %d -> %d neurons, %d -> %d edges (rule=%s, k=%d).",
        cfg.out,
        before[0],
        pruned.neuron_count,
        before[1],
        pruned.edge_count,
        cfg.prune_rule,
        cfg.prune_k,
    )
    print(
        f"Pruned connectome written to {cfg.out}/ "
        f"({before[0]}->{pruned.neuron_count} neurons, {before[1]}->{pruned.edge_count} edges). "
        f"Reuse via a train config with `connectome: {cfg.out}`."
    )
    return 0


def _run_prune_trained(config_path: str) -> int:
    """Drive the UC-07 post-training activation-pruning workflow from a YAML config."""
    from drone_fly.config import PruneTrainedRunConfig, load_yaml
    from drone_fly.prune_trained.workflow import prune_trained

    cfg = PruneTrainedRunConfig.from_mapping(load_yaml(config_path))

    report = prune_trained(
        checkpoint=cfg.checkpoint,
        out_dir=cfg.out,
        connectome_path=cfg.connectome,
        vecnormalize_path=cfg.vecnormalize,
        prune=cfg.prune,
        prune_k=cfg.prune_k,
        metric=cfg.metric,
        threshold=cfg.threshold,
        threshold_mode=cfg.threshold_mode,
        episodes=cfg.episodes,
        eps=cfg.eps,
        finetune_steps=cfg.finetune_steps,
        env_config=_env_config(cfg.randomize, cfg.randomize_dynamics),
        adapter=cfg.adapter,
        device=cfg.device,
        seed=cfg.seed,
    )
    print(
        f"prune-trained: {report.neurons_before}->{report.neurons_after} neurons, "
        f"{report.edges_before}->{report.edges_after} edges; completion "
        f"before/after-prune/after-finetune = "
        f"{report.completion_before:.0%}/{report.completion_after_prune:.0%}/"
        f"{report.completion_after_finetune:.0%}. "
        f"Artifacts in {report.out_dir}/ (see {REPORT_NAME_HINT})."
    )
    if report.course_specific:
        print(
            "WARNING: course-specific circuit (no domain randomization) — not a general minimal "
            "fly flight circuit. Re-run with randomize enabled for a robust result."
        )
    return 0


def _run_clean(args: argparse.Namespace) -> int:
    """Wipe training outputs relative to the CWD (dry-run unless --yes/--force) (UC-10).

    Effective delete = ``(--yes or --force) and not --dry-run`` — ``--dry-run`` always wins
    (AC8). Prints every target path, then a one-line summary, and always returns 0 (there is
    no config to fail on and a missing output tree is not an error — AC1/AC2/AC7).
    """
    from pathlib import Path

    from drone_fly.clean import clean

    delete = (args.yes or args.force) and not args.dry_run
    report = clean(Path.cwd(), delete=delete, include_prunes=args.include_prunes)

    verb = "would remove" if report.dry_run else "removed"
    for rel in report.relative_paths():
        print(f"{verb}: {rel}")
    if report.dry_run and report.paths:
        print("Dry-run: nothing was deleted. Re-run with --yes (or --force) to delete.")
    print(report.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
