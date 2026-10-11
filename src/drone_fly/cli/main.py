"""``drone-fly`` command line (argparse; imports no heavy backend at startup).

    drone-fly harness list
    drone-fly harness run --harness <name> --checkpoint <model_readout.pt> --extra-dim {12,30,70}
                          [--preset <name>] [--config <yaml>] [--k1-path] [--gpu-prototype-dir]
                          [--device] [--duration] [--dry-run] [--output-root] [--no-record]
                          [--overlay [--no-overlay-launch]] [--recordable]
    drone-fly harness eval --session <run dir> --track <xml> --race <xml> --config <yaml>
                          [--report <json>]
    drone-fly harness eval --capture <dir> --labels <dir> [--preset] [--gate-weights] [--device]
    drone-fly harness label --capture <dir> --track <xml> [--race] [--props] [--preset] [--out]
                          [--preview N] [--lag-ms]
    drone-fly harness train-gates --labels <dir>... [--out] [--preset] [--epochs] [--backbone]
                          [--holdout-frac 0.2 | --holdout t0:t1]
    drone-fly harness depth-fit --capture <dir> --track <xml> [--race] [--props] [--preset]
                          [--every N] [--report <dir>]
    drone-fly harness pad-check [--preset] [--config] [--magnitude] [--pulse-s] [--min-alt]
                          [--max-alt] [--out]
    drone-fly harness latency-report <run dir> [--recompute]
    drone-fly harness overlay [--port N] [--monitor M] [--layout default|mask_confined]
                          [--replay <run dir>] [--fps F] [--k1-path] [--preset] [--config]
                          [--harness]

Exit codes: 0 ok, 2 config/usage error, 3 missing optional dependency, 4 perception coverage
refusal, 5 fail-safe stop.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

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
        preset=args.preset,
        overlay=True if (args.overlay or args.overlay_launch or args.recordable) else None,
        overlay_launch=(args.overlay or args.overlay_launch or args.recordable)
        and not args.no_overlay_launch,
        recordable=True if args.recordable else None,
    )
    r = session.result
    print(
        f"stop: {r.reason}  ticks: {r.ticks}  missed deadlines: {r.missed_deadlines}"
        + (f"  error: {r.error}" if r.error else "")
    )
    if session.run_dir is not None:
        print(f"recorded: {session.run_dir}")
    _hold_until_ctrl_c(session.overlay_process)
    return EXIT_FAILSAFE if r.is_failsafe_stop else EXIT_OK


PAD_KEPT_MESSAGE = (
    "virtual pad still connected (neutral, disarmed): Liftoff re-arms and spins the motors up "
    "when it disconnects, so reset or pause the drone in Liftoff before pressing Ctrl+C"
)


def _hold_until_ctrl_c(overlay_process: object = None) -> None:
    """After a run: keep the overlay (result) and/or the virtual pad alive until Ctrl+C.

    The pad is kept because Liftoff re-arms when the controller disconnects
    (``controls.keep_pad_connected``); the overlay keeps showing the stop reason.
    """
    from drone_fly.harness.liftoff.gamepad import kept_pads, release_kept_pads
    from drone_fly.harness.session import stop_overlay

    overlay_alive = overlay_process is not None and overlay_process.poll() is None  # type: ignore[attr-defined]
    pads = bool(kept_pads())
    if not overlay_alive and not pads:
        return
    if not sys.stdin or not sys.stdin.isatty():  # scripted / test use: nobody to press Ctrl+C
        if pads:
            print(PAD_KEPT_MESSAGE.split(":")[0] + ": released (non-interactive)")
        stop_overlay(overlay_process)
        release_kept_pads()
        return
    if overlay_alive:
        print("overlay still open with the result")
    if pads:
        print(PAD_KEPT_MESSAGE)
    print("press Ctrl+C to close everything and exit")
    try:
        while True:
            if overlay_alive and overlay_process.poll() is not None and not pads:  # type: ignore[attr-defined]
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_overlay(overlay_process)
        release_kept_pads()


def _cmd_eval(args: argparse.Namespace) -> int:
    if args.capture is not None:
        from drone_fly.harness.offline.evaluate import capture_eval_summary, run_capture_eval

        if args.labels is None:
            raise HarnessConfigError("eval --capture needs --labels (from `harness label`)")
        rep = run_capture_eval(
            args.capture,
            args.labels,
            config=args.config,
            preset=args.preset,
            harness=args.harness,
            gate_weights=args.gate_weights,
            device=args.device,
            report_path=args.report,
            all_frames=args.all_frames,
        )
        print(capture_eval_summary(rep))
        return EXIT_OK
    from drone_fly.harness.offline.evaluate import run_offline_eval

    if not (args.session and args.track):
        raise HarnessConfigError(
            "eval needs --session + --track (+ --race), or --capture + --labels"
        )
    report = run_offline_eval(
        session=args.session,
        track=args.track,
        race=args.race,
        config=args.config,
        harness=args.harness,
        report_path=args.report,
        gpu_prototype_dir=args.gpu_prototype_dir,
        preset=args.preset,
    )
    print(report.summary())
    return EXIT_OK


def _cmd_label(args: argparse.Namespace) -> int:
    from drone_fly.harness.offline.autolabel import run_autolabel

    meta = run_autolabel(
        args.capture,
        args.track,
        args.race,
        args.props,
        config=args.config,
        preset=args.preset,
        harness=args.harness,
        out=args.out,
        preview=args.preview,
        preview_dir=args.preview_dir,
        lag_ms=args.lag_ms,
    )
    print(json.dumps({"counts": meta["counts"], "drop_reasons": meta["drop_reasons"]}, indent=2))
    for w in meta["props_warnings"]:
        print(f"warning: {w}")
    print(f"labels: {meta['out_dir']}")
    if meta["previews"]:
        print(f"previews: {Path(meta['previews'][0]).parent}")
    return EXIT_OK


def _cmd_train_gates(args: argparse.Namespace) -> int:
    from drone_fly.harness.offline.gate_train import train_gates

    meta = train_gates(
        args.labels,
        out=args.out,
        config=args.config,
        preset=args.preset,
        harness=args.harness,
        epochs=args.epochs,
        backbone=args.backbone,
        batch_size=args.batch_size,
        lr=args.lr,
        holdout_frac=args.holdout_frac,
        holdout=args.holdout,
        val_frac=args.val_frac,
        guard_s=args.guard_s,
        select=args.select,
        patience=None if args.patience <= 0 else args.patience,
        device=args.device,
    )
    val = meta["chosen_val_loss"]
    print(f"weights: {meta['weights']}")
    print(f"training log: {meta['training_log']}")
    print(
        f"kept epoch {meta['chosen_epoch']} of {meta['epochs_run']} ({meta['select']}): train loss "
        f"{meta['chosen_train_loss']:.4f}, validation loss "
        + ("-" if val is None else f"{val:.4f}")
    )
    return EXIT_OK


def _cmd_depth_fit(args: argparse.Namespace) -> int:
    from drone_fly.harness.offline.depth_fit import run_depth_fit

    rep = run_depth_fit(
        args.capture,
        args.track,
        args.race,
        args.props,
        config=args.config,
        preset=args.preset,
        harness=args.harness,
        every=args.every,
        fit_range=_float_pair(args.fit_range, "--fit-range"),
        report_dir=args.report,
        lag_ms=args.lag_ms,
        device=args.device,
    )
    fit, rays = rep["fit"], rep["raycast"]
    for v in fit["variants"]:
        bins = ", ".join(
            f"{k} {b['median_abs_rel_error']:.0%}" for k, b in v["bins"].items() if b.get("n")
        )
        rc = v["raycast"]
        print(
            f"variant {v['name']}: near score {v['near_score']:.1%} ({bins}); rays: median "
            f"{rc['ray_error_m_median']:.2f} m, within 0.5 m {rc['ray_within_0_5m']:.0%}"
        )
    print(f"chosen: {fit['chosen']}")
    print(
        f"pairs {fit['pairs']} on {fit['frames']} frames: median |rel err| "
        f"{fit['before']['median_abs_rel_error']:.1%} -> "
        f"{fit['after']['median_abs_rel_error']:.1%}, "
        f"RMSE {fit['before']['rmse_m']:.2f} -> {fit['after']['rmse_m']:.2f} m"
    )
    print(
        f"raycast: median ray error {rays['ray_error_m_median']:.3f} m (in-FOV "
        f"{rays['ray_error_m_median_in_fov']:.3f}), within 0.5 m {rays['ray_within_0_5m']:.1%} "
        f"(in-FOV {rays['ray_within_0_5m_in_fov']:.1%}), coverage {rays['coverage_mean']:.1%}"
    )
    ref = rays["all_clear_reference"]
    print(
        f"reference all-clear scan: within 0.5 m {ref['ray_within_0_5m']:.0%} (truth rays that "
        f"hit something: {rays['truth_ray_hit_share']:.0%})"
    )
    for w in rep["props_warnings"]:
        print(f"warning: {w}")
    print(rep["yaml_snippet"])
    print(f"report: {rep['report_path']}")
    return EXIT_OK


def _cmd_pad_server(args: argparse.Namespace) -> int:
    from drone_fly.harness.liftoff.pad_server import run_pad_server

    return run_pad_server(args.port, args.timeout)


def _cmd_pad_check(args: argparse.Namespace) -> int:
    from drone_fly.harness.liftoff.pad_check import PadCheckSettings, run_pad_check

    result = run_pad_check(
        config=args.config,
        preset=args.preset,
        harness=args.harness,
        settings=PadCheckSettings(
            magnitude=args.magnitude,
            pulse_s=args.pulse_s,
            min_alt=args.min_alt,
            max_alt=args.max_alt,
        ),
        out=args.out,
    )
    print(f"stop: {result.stop_reason}" + (f"  error: {result.error}" if result.error else ""))
    for axis, a in result.axes.items():
        print(
            f"{axis}: sign agreement {a['sign_agreement']:.0%}, magnitude ratio "
            f"{a['magnitude_ratio']:.2f}, stick_sign {a['current_sign']} -> {a['suggested_sign']}"
        )
    print(f"suggested controls.stick_signs: {result.suggested_stick_signs}")
    print(f"report: {result.report_path}")
    _hold_until_ctrl_c()
    return EXIT_OK if result.completed else EXIT_FAILSAFE


def _cmd_latency_report(args: argparse.Namespace) -> int:
    from drone_fly.harness.recording import latency_markdown, load_latency_summary

    print(latency_markdown(load_latency_summary(args.run, recompute=args.recompute)))
    return EXIT_OK


def _cmd_overlay(args: argparse.Namespace) -> int:
    from drone_fly.overlay.app import run_overlay

    return run_overlay(
        harness=args.harness,
        preset=args.preset,
        config=args.config,
        port=args.port,
        monitor=args.monitor,
        layout=args.layout,
        replay=args.replay,
        fps=args.fps,
        k1_path=args.k1_path,
        recordable=True if args.recordable else None,
        ready_file=args.ready_file,
    )


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
    p_run.add_argument("--preset", help="packaged preset overlaid before --config")
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
    p_run.add_argument(
        "--overlay",
        action="store_true",
        help="publish to the live overlay and start the overlay window (unless one is already "
        "running; see --no-overlay-launch)",
    )
    p_run.add_argument(
        "--no-overlay-launch",
        action="store_true",
        help="with --overlay: only publish, don't start the overlay window",
    )
    p_run.add_argument(
        "--overlay-launch",
        action="store_true",
        help="same as --overlay (kept for compatibility)",
    )
    p_run.add_argument(
        "--recordable",
        action="store_true",
        help="overlay visible to screen recorders; mask its panels out of the capture "
        "(implies --overlay; overlay.recordable)",
    )
    p_run.set_defaults(func=_cmd_run)

    p_eval = hsub.add_parser(
        "eval", help="score vision estimators on a recorded session or a labelled capture"
    )
    p_eval.add_argument("--session", help="recorded run folder (with --track / --race)")
    p_eval.add_argument("--track", help="Liftoff track XML (placed items)")
    p_eval.add_argument("--race", help="Liftoff race XML (gate order)")
    p_eval.add_argument("--capture", help="capture folder to score the learned gate detector on")
    p_eval.add_argument("--labels", help="label folder of --capture (from `harness label`)")
    p_eval.add_argument("--gate-weights", help="gate-net weights (default: gates.learned.weights)")
    p_eval.add_argument("--device", choices=("auto", "cpu", "cuda"), help="gate-net device")
    p_eval.add_argument(
        "--all-frames", action="store_true", help="score every frame, not only the held-out split"
    )
    _common(p_eval)
    p_eval.add_argument(
        "--report", help="eval_report.json path (default: <session>/ or artifacts/...)"
    )
    p_eval.add_argument("--gpu-prototype-dir", help="dir holding gpu_prototype/inference.py")
    p_eval.set_defaults(func=_cmd_eval)

    p_label = hsub.add_parser("label", help="auto-label gates in a capture from track XML + pose")
    p_label.add_argument("--capture", required=True, help="capture or harness run folder")
    p_label.add_argument("--track", required=True, help="Liftoff track XML")
    p_label.add_argument("--race", help="Liftoff race XML (recorded in the metadata)")
    p_label.add_argument("--props", help="props.json: cross-check gate_types against colliders")
    _common(p_label)
    p_label.add_argument("--out", help="label folder (default artifacts/harness/liftoff/labels/)")
    p_label.add_argument("--preview", type=int, default=0, help="write N overlay previews")
    p_label.add_argument("--preview-dir", help="preview root (default .../reports/label-preview)")
    p_label.add_argument("--lag-ms", type=float, default=0.0, help="frame -> telemetry lag")
    p_label.set_defaults(func=_cmd_label)

    p_train = hsub.add_parser("train-gates", help="train the learned gate detector on labels")
    p_train.add_argument("--labels", nargs="+", required=True, help="label folder(s)")
    p_train.add_argument("--out", help="weights path (default .../models/liftoff-gates.pt)")
    _common(p_train)
    p_train.add_argument("--epochs", type=int, default=60)
    p_train.add_argument("--backbone", choices=("resnet18-hf", "tiny-cnn"), default="resnet18-hf")
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=1e-3)
    split = p_train.add_mutually_exclusive_group()
    split.add_argument(
        "--holdout-frac",
        type=float,
        default=0.2,
        help="last fraction held out (0 with --val-frac 0 --select last: deploy model)",
    )
    split.add_argument("--holdout", help="held-out window t0:t1 (s from the first frame)")
    p_train.add_argument(
        "--val-frac", type=float, default=0.1, help="validation slice before the held-out one"
    )
    p_train.add_argument("--guard-s", type=float, default=2.0, help="gap between time slices")
    p_train.add_argument(
        "--select", choices=("best-val", "last"), default="best-val", help="epoch to keep"
    )
    p_train.add_argument(
        "--patience", type=int, default=10, help="early-stop patience in epochs (0 = off)"
    )
    p_train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p_train.set_defaults(func=_cmd_train_gates)

    p_depth = hsub.add_parser("depth-fit", help="fit depth calibration + score the 40-ray scan")
    p_depth.add_argument("--capture", required=True, help="capture or harness run folder")
    p_depth.add_argument("--track", required=True, help="Liftoff track XML")
    p_depth.add_argument("--race", help="Liftoff race XML (gate order for the ray target)")
    p_depth.add_argument("--props", help="props.json: cross-check item_table sizes")
    _common(p_depth)
    p_depth.add_argument("--every", type=int, default=5, help="use every Nth paired frame")
    p_depth.add_argument(
        "--fit-range", default="2:20", help="range (m) the calibration is chosen for, lo:hi"
    )
    p_depth.add_argument("--report", help="report folder (default .../reports/depth-fit/)")
    p_depth.add_argument("--lag-ms", type=float, default=0.0, help="frame -> telemetry lag")
    p_depth.add_argument("--device", choices=("auto", "cpu", "cuda"), help="depth model device")
    p_depth.set_defaults(func=_cmd_depth_fit)

    p_srv = hsub.add_parser(
        "pad", help="hold one virtual Xbox pad for the whole session (runs drive it)"
    )
    p_srv.add_argument(
        "--port", type=int, default=9200, help="localhost UDP port (controls.pad_port)"
    )
    p_srv.add_argument(
        "--timeout", type=float, default=0.5, help="neutral after this long without commands (s)"
    )
    p_srv.set_defaults(func=_cmd_pad_server)

    p_pad = hsub.add_parser("pad-check", help="live stick-direction check (Liftoff focused)")
    _common(p_pad)
    p_pad.add_argument("--magnitude", type=float, default=0.15, help="pulse stick deflection")
    p_pad.add_argument("--pulse-s", type=float, default=0.2, help="pulse length (s)")
    p_pad.add_argument("--min-alt", type=float, default=0.5, help="abort below this height (m)")
    p_pad.add_argument("--max-alt", type=float, default=6.0, help="abort above this height (m)")
    p_pad.add_argument("--out", help="report folder (default .../reports/pad-check/<stamp>)")
    p_pad.set_defaults(func=_cmd_pad_check)

    p_lat = hsub.add_parser("latency-report", help="per-stage latency table of a run folder")
    p_lat.add_argument("run", help="harness run folder")
    p_lat.add_argument(
        "--recompute", action="store_true", help="recompute from ticks.jsonl (ignore the summary)"
    )
    p_lat.set_defaults(func=_cmd_latency_report)

    p_ovl = hsub.add_parser(
        "overlay",
        help="live brain + debug overlay window over the game (needs `uv sync --extra overlay`)",
    )
    p_ovl.add_argument("--port", type=int, help="UDP port to listen on (default overlay.port)")
    p_ovl.add_argument(
        "--monitor", type=int, help="mss monitor index to cover (default camera.monitor; 0 = all)"
    )
    p_ovl.add_argument(
        "--layout", choices=("default", "mask_confined"), help="panel layout (overlay.layout)"
    )
    p_ovl.add_argument("--replay", help="drive the overlay from a recorded harness run folder")
    p_ovl.add_argument("--fps", type=float, help="redraw rate 1-20 (overlay.fps)")
    p_ovl.add_argument("--k1-path", help="K1 connectome dir for the brain panel")
    p_ovl.add_argument(
        "--recordable",
        action="store_true",
        help="let screen recorders (OBS, Game Bar) capture the overlay; run the harness with "
        "--recordable too so it masks the panels (overlay.recordable)",
    )
    p_ovl.add_argument(
        "--ready-file",
        help=argparse.SUPPRESS,  # created once the window is up (used by `harness run --overlay`)
    )
    _common(p_ovl)
    p_ovl.set_defaults(func=_cmd_overlay)
    return parser


def _float_pair(text: str, flag: str) -> tuple[float, float]:
    try:
        lo, hi = (float(v) for v in text.split(":"))
    except ValueError as exc:
        raise HarnessConfigError(f"{flag} must be lo:hi, got {text!r}") from exc
    if hi <= lo:
        raise HarnessConfigError(f"{flag} needs hi > lo, got {text!r}")
    return lo, hi


def _common(p: argparse.ArgumentParser) -> None:
    """``--preset`` / ``--config`` / ``--harness`` shared by the offline subcommands."""
    p.add_argument("--preset", help="packaged preset overlaid before --config")
    p.add_argument("--config", help="YAML overlaid on the harness default")
    p.add_argument("--harness", default="liftoff", help="harness whose config to use")


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
