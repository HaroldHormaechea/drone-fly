"""``drone-fly perception eval`` (UC-86): the command body.

The parser is defined in :mod:`drone_fly.cli.main`, so building the CLI (``harness list``...)
never imports this offline package; the work loads inside the command.
"""

from __future__ import annotations

import argparse
import json


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
