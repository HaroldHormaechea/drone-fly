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

    smoke_p = sub.add_parser("smoke-train", help="A few-step CI/correctness run (numpy backend).")
    smoke_p.add_argument("--timesteps", type=int, default=None, help="Override smoke timesteps.")
    smoke_p.add_argument("--connectome", default=None, help="Path to the cached connectome.")

    eval_p = sub.add_parser("evaluate", help="Evaluate a checkpoint over N episodes.")
    eval_p.add_argument("--checkpoint", required=True, help="Checkpoint .zip to evaluate.")
    eval_p.add_argument("--vecnormalize", default=None, help="VecNormalize stats .pkl.")
    eval_p.add_argument("--episodes", type=int, default=None, help="Number of eval episodes.")
    eval_p.add_argument("--seed", type=int, default=0, help="Evaluation seed.")
    eval_p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    eval_p.add_argument("--adapter", default="auto", choices=["auto", "simple", "pybullet"])

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
        )
        return 0

    if args.command == "smoke-train":
        from drone_fly.train.loop import smoke_train

        smoke_train(connectome_path=args.connectome, timesteps=args.timesteps)
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

    if args.command == "fetch-connectome":
        print(
            "fetch-connectome is a stub. Provision the cached connectome under "
            "data/connectome/ (or set DRONE_FLY_CONNECTOME_DIR). See the README "
            "'Provisioning connectome data' section."
        )
        return 0

    return 1  # pragma: no cover - argparse requires a subcommand


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
