"""CLI stage: thin command-line entry points.

Orchestrates the pipeline stages: fetch-connectome (stub), train, evaluate, smoke-train.
Kept thin — argument parsing and wiring only; the real logic lives in the stage
subpackages (:mod:`drone_fly.train`, :mod:`drone_fly.evaluate`).

Subcommands
-----------
* ``train`` — full PPO training; ``--resume``, ``--device``, ``--timesteps``, ``--adapter``.
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

    sub.add_parser("fetch-connectome", help="[stub] Provision connectome data (see README).")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    if args.command == "train":
        from drone_fly.train.loop import train

        train(
            connectome_path=args.connectome,
            adapter=args.adapter,
            device=args.device,
            resume=args.resume,
            total_timesteps=args.timesteps,
            prune=args.prune,
            prune_k=args.prune_k,
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
            device=args.device,
        )
        print(metrics.summary())
        return 0

    if args.command == "prune":
        return _run_prune_export(args)

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


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
