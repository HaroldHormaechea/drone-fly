"""``drone-fly perception ...`` (UC-86): argparse only; the work loads inside the commands."""

from __future__ import annotations

import argparse
import json
from typing import Any


def cmd_eval(args: argparse.Namespace) -> int:
    from drone_fly.harness.offline.perception_eval import run_perception_eval

    rep = run_perception_eval(
        args.provider, args.dataset, config=args.config, preset=args.preset,
        harness=args.harness, split=args.split, device=args.device, max_frames=args.max_frames,
        out=args.out, tol_m=args.tol_m, gpu_prototype_dir=args.gpu_prototype_dir,
        trackdata_root=args.trackdata_root, scenes_root=args.scenes_root,
    )  # fmt: skip
    print(json.dumps({"aggregate": rep.get("aggregate", {"frames": 0}),
                      "latency_ms": rep["latency_ms"],
                      "skipped_frames": rep["skipped_frames"]}, indent=2))  # fmt: skip
    print(f"report: {rep['out_dir']}")
    return 0


def add_parser(sub: Any) -> None:
    perc = sub.add_parser("perception", help="pluggable perception providers (UC-86)")
    ps = perc.add_subparsers(dest="perception_command", required=True)
    p = ps.add_parser(
        "eval", help="score an obstacle-vision provider on an eyes dataset (UC-78 metrics)"
    )
    p.add_argument(
        "--provider", required=True, help="registered name, module.path:Class or entry point"
    )
    p.add_argument("--dataset", required=True, help="eyes dataset folder (drone-fly eyes dataset)")
    p.add_argument("--config", help="YAML overlaid on the preset (e.g. eyes.weights)")
    p.add_argument("--preset", default="liftoff", help="harness preset (camera, telemetry)")
    p.add_argument("--harness", default="liftoff", help="harness whose config to use")
    p.add_argument("--split", default="held-out", choices=("held-out", "val"))
    p.add_argument("--device", help="eyes/depth device override (auto | cpu | cuda)")
    p.add_argument("--max-frames", type=int, help="score at most N rows, evenly spread")
    p.add_argument("--tol-m", type=float, help="ray tolerance in training metres (default 0.5)")
    p.add_argument("--out", help="default <data root>/perception-eval/<provider>-<UTC stamp>")
    p.add_argument("--trackdata-root", help="Liftoff track XMLs (the label route targets)")
    p.add_argument("--scenes-root")
    p.add_argument("--gpu-prototype-dir")
    p.set_defaults(func=cmd_eval)
