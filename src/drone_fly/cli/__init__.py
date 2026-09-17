"""CLI stage: thin command-line entry points.

Orchestrates the pipeline stages: fetch-connectome (stub), train, evaluate, smoke-train.
Kept thin — argument parsing and wiring only; the real logic lives in the stage
subpackages (:mod:`drone_fly.train`, :mod:`drone_fly.evaluate`).

Subcommands
-----------
* ``train`` — full PPO training; ``--resume``, ``--device``, ``--timesteps``, ``--n-envs``,
  ``--adapter``.
* ``evaluate`` — load a checkpoint and report completion rate + mean time over N episodes.
* ``smoke-train`` — a few-step CI/correctness run on the pure-numpy backend (no pybullet).
* ``fetch-connectome`` — documented stub (connectome provisioning is UC-01/owner territory).
"""

from __future__ import annotations

import argparse
import logging
import sys

from drone_fly.connectome.prune import DEFAULT_PRUNE_K, DEFAULT_PRUNE_RULE


def _add_prune_args(p: argparse.ArgumentParser) -> None:
    """Add the opt-in UC-04 subcircuit-pruning flags (shared by train / smoke-train)."""
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


def _add_record_args(p: argparse.ArgumentParser) -> None:
    """Add the opt-in UC-05 activation-recording flags (shared by evaluate / train)."""
    p.add_argument(
        "--record",
        action="store_true",
        help="Record per-frame neuron activations + actions + drone path for playback in "
        "viz/viewer.html (UC-05). Off by default (eval/train numerics unchanged).",
    )
    p.add_argument(
        "--record-every",
        type=int,
        default=1,
        help="Record every Nth episode (default 1 = every episode).",
    )
    p.add_argument(
        "--record-dir",
        default=None,
        help="Directory for episode_<n>.json playback files (default artifacts/activations/).",
    )


def _add_randomize_args(p: argparse.ArgumentParser) -> None:
    """Add the opt-in UC-08 domain-randomization flags (shared by train / evaluate).

    Both default off, so an unflagged invocation is byte-identical to UC-03. The two axes
    are independent: either may be enabled with the other off.
    """
    p.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize the COURSE (start / gate / finish) per episode within configured, "
        "solvable ranges (UC-08). Off by default; the anti-memorization axis.",
    )
    p.add_argument(
        "--randomize-dynamics",
        action="store_true",
        help="Randomize DYNAMICS (mass / drag / thrust / body-rate / control latency) per "
        "episode (UC-08). Off by default; the robustness / sim-to-sim axis, independent of "
        "--randomize.",
    )


def _build_env_config(args: argparse.Namespace):
    """Build an :class:`EnvConfig` from the randomization flags, or ``None`` if both off.

    Returning ``None`` when neither axis is requested keeps train / evaluate byte-identical
    to UC-03 (the callees treat ``env_config=None`` as the fixed default).
    """
    enable_course = bool(getattr(args, "randomize", False))
    enable_dynamics = bool(getattr(args, "randomize_dynamics", False))
    if not (enable_course or enable_dynamics):
        return None
    from drone_fly.env.config import EnvConfig, RandomizationConfig

    return EnvConfig(
        randomization=RandomizationConfig(
            enable_course=enable_course,
            enable_dynamics=enable_dynamics,
        )
    )


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--resume", default=None, help="Checkpoint .zip to continue from.")
    p.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Explicit torch device; omit for the Apple-Silicon-aware auto-policy.",
    )
    p.add_argument("--timesteps", type=int, default=None, help="Override total timesteps.")
    p.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Override TrainConfig.n_envs (default 1): number of parallel rollout envs. "
        "Raise it to use more CPU cores / increase throughput. Omit for unchanged behaviour.",
    )
    p.add_argument(
        "--adapter",
        default="auto",
        choices=["auto", "simple", "pybullet"],
        help="Sim backend: auto (pybullet if installed), simple (numpy), or pybullet.",
    )
    p.add_argument(
        "--connectome",
        default=None,
        help="Path to the cached connectome dir/.npz (else DRONE_FLY_CONNECTOME_DIR/default).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drone-fly", description="Connectome-seeded drone racing."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train the PPO racing policy.")
    _add_train_args(train_p)
    _add_prune_args(train_p)
    _add_record_args(train_p)
    _add_randomize_args(train_p)

    smoke_p = sub.add_parser("smoke-train", help="A few-step CI/correctness run (numpy backend).")
    smoke_p.add_argument("--timesteps", type=int, default=None, help="Override smoke timesteps.")
    smoke_p.add_argument("--connectome", default=None, help="Path to the cached connectome.")
    _add_prune_args(smoke_p)

    eval_p = sub.add_parser("evaluate", help="Evaluate a checkpoint over N episodes.")
    eval_p.add_argument("--checkpoint", required=True, help="Checkpoint .zip to evaluate.")
    eval_p.add_argument("--vecnormalize", default=None, help="VecNormalize stats .pkl.")
    eval_p.add_argument("--episodes", type=int, default=None, help="Number of eval episodes.")
    eval_p.add_argument("--seed", type=int, default=0, help="Evaluation seed.")
    eval_p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    eval_p.add_argument("--adapter", default="auto", choices=["auto", "simple", "pybullet"])
    eval_p.add_argument(
        "--connectome",
        default=None,
        help="Connectome dir/.npz to re-load for --record (MUST match the one used to train "
        "this checkpoint; the checkpoint does not retain neuron_ids/superclass/positions).",
    )
    _add_prune_args(eval_p)
    _add_record_args(eval_p)
    _add_randomize_args(eval_p)

    prune_p = sub.add_parser(
        "prune",
        help="Prune a connectome to its sensory->motor subcircuit and write it to disk for reuse.",
    )
    prune_p.add_argument(
        "--connectome",
        required=True,
        help="Path to the input connectome dir/.npz to prune (e.g. the full matrix dir).",
    )
    prune_p.add_argument(
        "--out",
        required=True,
        help="Output directory for the pruned connectome (.npz + _meta.csv + provenance note). "
        "Reuse it later via `train --connectome <out>` (no --prune, no recompute).",
    )
    prune_p.add_argument(
        "--prune-k",
        type=int,
        default=DEFAULT_PRUNE_K,
        help=f"Path-slack corridor width (default {DEFAULT_PRUNE_K}; 0 = tight shortest-path).",
    )
    prune_p.add_argument(
        "--prune-rule",
        default=DEFAULT_PRUNE_RULE,
        help=f"Pruning rule (default {DEFAULT_PRUNE_RULE!r}).",
    )

    _add_prune_trained_parser(sub)

    sub.add_parser("fetch-connectome", help="[stub] Provision connectome data (see README).")
    return parser


def _add_prune_trained_parser(sub) -> None:
    """Add the opt-in UC-07 post-training activation-pruning subcommand.

    Off by default (a distinct subcommand); omitting it leaves UC-01..06 byte-identical. The
    ``--prune``/``--prune-k`` flags rebuild the SAME base graph the checkpoint was trained on
    (composing UC-04 then UC-07).
    """
    from drone_fly.prune_trained.importance import (
        ABSOLUTE_MODE,
        DEFAULT_THRESHOLD,
        THRESHOLD_MODES,
    )
    from drone_fly.prune_trained.measure import DEFAULT_EPS, DEFAULT_METRIC, METRICS
    from drone_fly.prune_trained.workflow import DEFAULT_EPISODES, DEFAULT_FINETUNE_STEPS

    p = sub.add_parser(
        "prune-trained",
        help="Post-training activation prune: measure per-neuron usage on a trained policy, "
        "remove dead interneurons, fine-tune, and save the minimal circuit (UC-07).",
    )
    p.add_argument("--checkpoint", required=True, help="Trained checkpoint .zip to prune.")
    p.add_argument(
        "--connectome",
        default=None,
        help="Connectome dir/.npz the checkpoint was trained on (MUST match; the checkpoint does "
        "not retain neuron_ids/superclass/adjacency). Defaults to DRONE_FLY_CONNECTOME_DIR.",
    )
    p.add_argument(
        "--vecnormalize",
        default=None,
        help="VecNormalize stats .pkl used at training time (carried into measurement + "
        "fine-tune; warn + fresh stats if absent).",
    )
    p.add_argument("--out", required=True, help="Output directory for the pruned artifacts.")
    _add_prune_args(p)  # --prune / --prune-k rebuild the base (UC-04) graph the ckpt trained on
    p.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=list(METRICS),
        help=f"Per-neuron importance metric (default {DEFAULT_METRIC!r}).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Interneurons at/below this importance are pruned (default {DEFAULT_THRESHOLD}; "
        "absolute mode is calibrated against the tanh [-1,1] activation range).",
    )
    p.add_argument(
        "--threshold-mode",
        default=ABSOLUTE_MODE,
        choices=list(THRESHOLD_MODES),
        help=f"Interpret --threshold as an absolute cut or percentile (default {ABSOLUTE_MODE!r}).",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Episodes to measure importance / completion over (default {DEFAULT_EPISODES}).",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=DEFAULT_EPS,
        help=f"Activity epsilon for active_fraction (default {DEFAULT_EPS:g}).",
    )
    p.add_argument(
        "--finetune-steps",
        type=int,
        default=DEFAULT_FINETUNE_STEPS,
        help=f"Short PPO continue-training budget after pruning (default {DEFAULT_FINETUNE_STEPS} "
        "= skip; a real recovery run is a dev-time step on the owner's machine).",
    )
    p.add_argument("--adapter", default="simple", choices=["auto", "simple", "pybullet"])
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=0, help="Measurement / fine-tune seed.")
    _add_randomize_args(p)  # measurement distribution (drives the course-specific warning)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    if args.command == "train":
        if args.n_envs is not None and args.n_envs < 1:
            build_parser().error("--n-envs must be >= 1")
        from drone_fly.train.loop import train

        train(
            connectome_path=args.connectome,
            env_config=_build_env_config(args),
            adapter=args.adapter,
            device=args.device,
            resume=args.resume,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            prune=args.prune,
            prune_k=args.prune_k,
            record=args.record,
            record_every=args.record_every,
            record_dir=args.record_dir,
        )
        return 0

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
        from drone_fly.evaluate.evaluator import evaluate_checkpoint

        metrics = evaluate_checkpoint(
            args.checkpoint,
            vecnormalize_path=args.vecnormalize,
            episodes=args.episodes,
            seed=args.seed,
            adapter=args.adapter,
            env_config=_build_env_config(args),
            device=args.device,
            record=args.record,
            record_every=args.record_every,
            record_dir=args.record_dir,
            connectome_path=args.connectome,
            prune=args.prune,
            prune_k=args.prune_k,
        )
        print(metrics.summary())
        return 0

    if args.command == "prune":
        return _run_prune_export(args)

    if args.command == "prune-trained":
        return _run_prune_trained(args)

    if args.command == "fetch-connectome":
        print(
            "fetch-connectome is a stub. Provision the cached connectome under "
            "data/connectome/ (or set DRONE_FLY_CONNECTOME_DIR). See the README "
            "'Provisioning connectome data' section."
        )
        return 0

    return 1  # pragma: no cover - argparse requires a subcommand


def _run_prune_export(args: argparse.Namespace) -> int:
    """Load a connectome, prune it, and write the reusable pruned slice to ``--out`` (AC11).

    Writes ``<out>/connectome_pruned.npz`` + ``connectome_pruned_meta.csv`` (round-trips
    through :func:`~drone_fly.connectome.load_connectome`) plus a ``PRUNE_PROVENANCE.md``
    recording source, rule, k, and the input→pruned counts.
    """
    from pathlib import Path

    from drone_fly.connectome import load_connectome, prune_to_subcircuit, save_connectome

    logger = logging.getLogger("drone_fly.cli.prune")
    data = load_connectome(args.connectome)
    before = (data.neuron_count, data.edge_count)
    pruned = prune_to_subcircuit(data, k=args.prune_k, rule=args.prune_rule)
    npz_path, meta_path = save_connectome(pruned, args.out)

    prov_path = Path(args.out) / "PRUNE_PROVENANCE.md"
    prov_path.write_text(
        "# Pruned connectome provenance\n\n"
        f"- **Source:** {data.source}\n"
        f"- **Rule:** {args.prune_rule}\n"
        f"- **k:** {args.prune_k}\n"
        f"- **Input scale:** {before[0]} neurons, {before[1]} edges\n"
        f"- **Pruned scale:** {pruned.neuron_count} neurons, {pruned.edge_count} edges\n"
        f"- **Matrix:** {npz_path.name}\n"
        f"- **Meta:** {meta_path.name}\n\n"
        "Reuse with `drone-fly train --connectome "
        f"{args.out}` (no --prune — the slice is already pruned).\n"
    )

    logger.info(
        "Wrote pruned connectome to %s: %d -> %d neurons, %d -> %d edges (rule=%s, k=%d).",
        args.out,
        before[0],
        pruned.neuron_count,
        before[1],
        pruned.edge_count,
        args.prune_rule,
        args.prune_k,
    )
    print(
        f"Pruned connectome written to {args.out}/ "
        f"({before[0]}->{pruned.neuron_count} neurons, {before[1]}->{pruned.edge_count} edges). "
        f"Reuse via: drone-fly train --connectome {args.out}"
    )
    return 0


def _run_prune_trained(args: argparse.Namespace) -> int:
    """Drive the UC-07 post-training activation-pruning workflow from CLI args."""
    from drone_fly.prune_trained.workflow import prune_trained

    report = prune_trained(
        checkpoint=args.checkpoint,
        out_dir=args.out,
        connectome_path=args.connectome,
        vecnormalize_path=args.vecnormalize,
        prune=args.prune,
        prune_k=args.prune_k,
        metric=args.metric,
        threshold=args.threshold,
        threshold_mode=args.threshold_mode,
        episodes=args.episodes,
        eps=args.eps,
        finetune_steps=args.finetune_steps,
        env_config=_build_env_config(args),
        adapter=args.adapter,
        device=args.device,
        seed=args.seed,
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
            "fly flight circuit. Re-run with --randomize for a robust result."
        )
    return 0


#: Report filename hint printed by the CLI (kept in sync with workflow.REPORT_NAME).
REPORT_NAME_HINT = "PRUNE_TRAINED_REPORT.md"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
