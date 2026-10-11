"""``drone-fly`` command line (argparse; imports no heavy backend at startup).

    drone-fly harness list
    drone-fly harness run --harness <name> --checkpoint <model_readout.pt> --extra-dim {12,30,70}
                          [--config <yaml>] [--k1-path] [--gpu-prototype-dir] [--device]
                          [--duration] [--dry-run] [--output-root] [--no-record]
    drone-fly harness eval --session <run dir> --track <xml> --race <xml> --config <yaml>
                          [--report <json>]

Exit codes: 0 ok, 2 config/usage error, 3 missing optional dependency, 4 perception coverage
refusal, 5 fail-safe stop.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from drone_fly.harness.errors import (
    HarnessConfigError,
    HarnessDependencyError,
    PerceptionCoverageError,
)

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_DEPENDENCY = 3
EXIT_COVERAGE = 4
EXIT_FAILSAFE = 5


def _cmd_list(args: argparse.Namespace) -> int:
    from drone_fly.harness import registry

    rows = [(kind, name) for kind in registry.KINDS for name in registry.names(kind)]
    kw = max([len("kind"), *(len(k) for k, _ in rows)]) + 2
    nw = max([len("name"), *(len(n) for _, n in rows)]) + 2
    print(f"{'kind':<{kw}}{'name':<{nw}}{'status':<12}notes")
    for kind, name in rows:
        e = registry.entry(kind, name)
        p = registry.probe(kind, name)
        status = "available" if p["available"] else "missing"
        notes = e.description
        if p["missing"]:
            notes = f"needs {', '.join(p['missing'])} (uv sync --extra harness)"
        print(f"{kind:<{kw}}{name:<{nw}}{status:<12}{notes}")
    return EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    from drone_fly.harness.session import run_harness

    session = run_harness(
        args.harness,
        args.checkpoint,
        args.extra_dim,
        config=args.config,
        k1_path=args.k1_path,
        gpu_prototype_dir=args.gpu_prototype_dir,
        device=args.device,
        duration_s=args.duration,
        dry_run=args.dry_run,
        output_root=args.output_root,
        record=False if args.no_record else None,
    )
    r = session.result
    print(
        f"stop: {r.reason}  ticks: {r.ticks}  missed deadlines: {r.missed_deadlines}"
        + (f"  error: {r.error}" if r.error else "")
    )
    if session.run_dir is not None:
        print(f"recorded: {session.run_dir}")
    return EXIT_FAILSAFE if r.is_failsafe_stop else EXIT_OK


def _cmd_eval(args: argparse.Namespace) -> int:
    from drone_fly.harness.offline.evaluate import run_offline_eval

    report = run_offline_eval(
        session=args.session,
        track=args.track,
        race=args.race,
        config=args.config,
        harness=args.harness,
        report_path=args.report,
        gpu_prototype_dir=args.gpu_prototype_dir,
    )
    print(report.summary())
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drone-fly", description="drone-fly tools")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    harness = sub.add_parser("harness", help="run a trained model against a live flight system")
    hsub = harness.add_subparsers(dest="harness_command", required=True)

    p_list = hsub.add_parser("list", help="list registered harnesses/parts and their availability")
    p_list.set_defaults(func=_cmd_list)

    p_run = hsub.add_parser("run", help="run a harness")
    p_run.add_argument("--harness", required=True, help="registered harness name")
    p_run.add_argument("--checkpoint", required=True, help="stripped model_readout.pt")
    p_run.add_argument(
        "--extra-dim", type=int, required=True, choices=(12, 30, 70), help="model extra width"
    )
    p_run.add_argument("--config", help="YAML overlaid on the harness's packaged default")
    p_run.add_argument("--k1-path", help="K1 connectome dir (default: artifacts/pruned/k1)")
    p_run.add_argument("--gpu-prototype-dir", help="dir holding gpu_prototype/inference.py")
    p_run.add_argument("--device", choices=("auto", "cpu", "cuda"))
    p_run.add_argument("--duration", type=float, help="stop after this many seconds")
    p_run.add_argument(
        "--dry-run", action="store_true", help="record commands but do not actuate (no sink)"
    )
    p_run.add_argument("--output-root", help="recording root (default: artifacts/harness)")
    p_run.add_argument("--no-record", action="store_true", help="disable the run recording")
    p_run.set_defaults(func=_cmd_run)

    p_eval = hsub.add_parser("eval", help="score vision estimators on a recorded session")
    p_eval.add_argument("--session", required=True, help="recorded run folder")
    p_eval.add_argument("--track", required=True, help="Liftoff track XML (placed items)")
    p_eval.add_argument("--race", required=True, help="Liftoff race XML (gate order)")
    p_eval.add_argument("--config", help="YAML overlaid on the harness default")
    p_eval.add_argument("--harness", default="liftoff", help="harness whose config to use")
    p_eval.add_argument("--report", help="eval_report.json path (default: <session>/)")
    p_eval.add_argument("--gpu-prototype-dir", help="dir holding gpu_prototype/inference.py")
    p_eval.set_defaults(func=_cmd_eval)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except PerceptionCoverageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_COVERAGE
    except HarnessDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DEPENDENCY
    except HarnessConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
