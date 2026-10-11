"""Generic ``train_aug`` launcher for the desktop app's ``train_aug`` recipe (UC-63).

The named ``gpu_prototype/`` launchers hardcode every knob except ``<save> [budget]``; this script
is the app's way to run ``reservoir_aug.train_aug`` with user-chosen knobs. It is pure argument
forwarding — no training, env, or reward logic (AC10):

    python app/reservoir_launch.py --save-path X [--budget Y] [--N 512] [--warm-from ckpt.pt] ...

Only knobs given on the command line are forwarded, so every other knob keeps ``train_aug``'s own
default. When ``--warm-from`` is set and ``--warm-src-extra-dim`` is not, the runtime ``EXTRA_DIM``
is used (same-architecture warm start). The parsing/resolving half (:func:`parse_args`,
:func:`resolve_kwargs`) never imports torch; only :func:`main` imports ``gpu_prototype``.

Run it with the same ``PROC_*`` env flags as any launcher (``gates_proc_env`` reads them at import
time) and with ``gpu_prototype/`` as the CWD; :class:`app.runs.RunRegistry` does both.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

#: Forwarded ``train_aug`` knobs and their argument types. ``app.runs`` builds ``--<knob>`` options
#: (underscores → dashes) for exactly these names.
KNOB_TYPES: dict[str, type] = {
    "N": int,
    "budget": float,
    "roll": int,
    "epochs": int,
    "logstd_init": float,
    "logstd_final": float,
    "anneal_start": float,
    "count_curr_frac": float,
    "pos_curr_frac": float,
    "warm_from": str,
    "warm_src_extra_dim": int,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reservoir_launch",
        description="Run gpu_prototype reservoir_aug.train_aug on BatchedProcCourse.",
    )
    parser.add_argument("--save-path", dest="save_path", required=True, help="Checkpoint (.pt).")
    for knob, kind in KNOB_TYPES.items():
        parser.add_argument("--" + knob.replace("_", "-"), dest=knob, type=kind, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_kwargs(args: argparse.Namespace, extra_dim: int | None) -> dict[str, Any]:
    """Return the ``train_aug`` keyword arguments: ``save_path`` plus only the knobs that were set.

    ``warm_src_extra_dim`` is forwarded only when ``warm_from`` is set; if it was left blank it
    resolves to ``extra_dim`` (the runtime ``EXTRA_DIM``).
    """
    kwargs: dict[str, Any] = {"save_path": args.save_path}
    for knob in KNOB_TYPES:
        value = getattr(args, knob, None)
        if value is not None:
            kwargs[knob] = value
    if kwargs.get("warm_from") is None:
        kwargs.pop("warm_src_extra_dim", None)
    elif kwargs.get("warm_src_extra_dim") is None:
        kwargs["warm_src_extra_dim"] = extra_dim
    return kwargs


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs torch + a GPU env
    args = parse_args(argv)
    gpu_prototype = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpu_prototype"
    )
    sys.path.insert(0, gpu_prototype)
    from gates_proc_env import EXTRA_DIM, BatchedProcCourse
    from reservoir_aug import train_aug

    kwargs = resolve_kwargs(args, EXTRA_DIM)
    print(f"[app] reservoir_launch train_aug kwargs: {kwargs}", flush=True)
    train_aug(BatchedProcCourse, EXTRA_DIM, **kwargs)
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
