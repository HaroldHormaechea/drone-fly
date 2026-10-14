"""``drone-fly`` command line (argparse; imports no heavy backend at startup).

    drone-fly harness list
    drone-fly harness run --harness <name> --checkpoint <model_readout.pt> --extra-dim {12,30,70}
                          [--preset <name>] [--config <yaml>] [--k1-path] [--gpu-prototype-dir]
                          [--device] [--duration] [--dry-run] [--output-root] [--no-record]
                          [--overlay [--no-overlay-launch]] [--recordable]
    drone-fly harness run --harness liftoff --teacher <run|teacher.pt> --track <id>
                          [--perception oracle|detector] [--trackdata-root] [--scenes-root]
                          [--allow-concurrent-training] [--session <dir> | --session-stamp S]
    drone-fly harness run --harness liftoff --checkpoint <student.pt> [--preset liftoff]
                          [--allow-baseline] [--allow-slow-cpu] [--overlay] [--dry-run]
    drone-fly harness oracle-report --session <dir> [--sim-episodes 32] [--no-sim]
    drone-fly harness eval --session <run dir> --track <xml> --race <xml> --config <yaml>
                          [--report <json>]
    drone-fly harness eval --capture <dir> --labels <dir> [--preset] [--gate-weights] [--device]
    drone-fly harness eval --detector <weights|dir> --split held-out [--captures ...]
                          [--compare <weights>] [--protected-capture <dir>] [--device]
    drone-fly harness label --capture <dir> --track <xml> [--race] [--props] [--preset] [--out]
                          [--preview N] [--lag-ms]
    drone-fly harness captures register <dir> --track <catalogue-id> [--trackdata-root] [--force]
    drone-fly harness captures record --track <catalogue-id> [--max-s N] [--fps 10] [--port]
    drone-fly harness captures replay --recording <name> [--max-s N] [--fps 10]
                          [--strategy auto|A|B|C] [--allow-pending-camera]
                          --browser-pos ENV,RACE,ROW
    drone-fly harness captures inventory [--dir <Recordings>] [--json out] [--trackdata-root]
                          [--no-pose]
    drone-fly harness captures verify <dir> [--search-lag] [--search-uptilt] [--trackdata-root]
                          [--no-write] [--calibrate]
    drone-fly harness gates train (--captures <dir|glob|name>... | --all-captures) [--split-source]
                          [--detector-id] [--epochs] [--backbone] [--input-width] [--frame-stride]
                          [--max-frames-per-capture] [--select val-detection|val-loss|last]
                          [--val-every 2] [--protect-holdout-of <weights>] [--deploy]
    drone-fly harness gates dataset-report (--captures ... | --all-captures) [--json out]
    drone-fly harness gates draft-types --props <props.json> --items <itemID>...
    drone-fly harness gates diagnose --weights <weights|dir> --captures <dir>... [--all-frames]
                          [--json out] [--device]
    drone-fly harness train-gates --labels <dir>... [--out] [--preset] [--epochs] [--backbone]
                          [--holdout-frac 0.2 | --holdout t0:t1]
    drone-fly harness depth-fit --capture <dir> --track <xml> [--race] [--props] [--preset]
                          [--every N] [--report <dir>]
    drone-fly harness pad-check [--preset] [--config] [--magnitude] [--pulse-s] [--min-alt]
                          [--max-alt] [--out]
    drone-fly harness ui launch|last-session|reset|retry|save-recording --name N|
                          select-drone --name N [--slot S] [--level L]|quit|shot [--out]
    drone-fly harness latency-report <run dir> [--recompute]
    drone-fly harness overlay [--port N] [--monitor M] [--layout default|mask_confined]
                          [--replay <run dir>] [--fps F] [--k1-path] [--preset] [--config]
                          [--harness]
    drone-fly sysid liftoff --drone <id> [--session NAME] [--resume] [--select [--level L]]
                          [--liftoff-name]
                          [--map] [--seed] [--passes] [--hold-alt] [--home-forward]
                          [--ceiling] [--radius]
                          [--expect-hover lo:hi] [--preset liftoff] [--config]
                          [--plant liftoff|synthetic|synthetic-5inch]
    drone-fly sysid fit <session dir> --drone <id> [--out <dir>] [--force] [--name] [--map]
                          [--camera-from-preset <preset>] [--micro]
    drone-fly train run <config> [--k1-path] [--device] [--n-envs] [--budget] [--seed]
                          [--max-iterations] [--output-root]
    drone-fly train eval <checkpoint> --config <config> [--episodes] [--seed] [--legacy-v1]
                          [--compare-prototype] [--k1-path] [--device] [--json]
    drone-fly train bench [--config] [--n-envs] [--steps] [--substeps ...] [--no-prototype]
    drone-fly train teacher --drone <id> --tracks <selector> [--procedural F] [--config]
                          [--resume <run>] [--budget] [--n-envs] [--device] [--max-iterations]
    drone-fly train eval <run|teacher.pt> --tracks <selector> [--episodes] [--mode] [--seed]
    drone-fly train detector-noise --detector <weights> --captures <dir>... [--labels <dir>...]
                          [--live-runs <run>...] [--preset] [--device] [--out] [--latency-only]
    drone-fly train student --teacher <run> --noise <noise_model.yaml> [--drone] [--tracks S]
                          [--procedural F] [--actor k1-student|mlp-student] [--config]
                          [--world-scale K] [--residual] [--budget] [--n-envs] [--resume <run>]
                          [--ppo-steps N] [--feasibility-only] [--allow-gpu-contention]
                          [--k1-path] [--device] | --scale-report [--trackdata-root] [--out]
    drone-fly train export <student run> [--checkpoint val|final|it<N>] --out <path>
                          [--baseline-export] [--bench|--no-bench]
    drone-fly train eval <student run> [--tracks liftoff:held-out] [--baseline <mlp run>]
    drone-fly tracks list [--split S] [--source S] | check [--root] [--scenes-root]
                          [--require-scenes] | scan --source liftoff [--root] [--scenes-root]
                          [--seed] | stats --split S | extract-scenes [--map M] [--bundle M=B]
                          [--witness M=F] [--force] | extract-props [--force]
    drone-fly sim replay-sysid <session dir> --drone <id> [--write-report] [--gate|--no-gate]

Exit codes: 0 ok, 2 config/usage error (also: anti-cheat software running), 3 missing optional
dependency, 4 perception coverage refusal, 5 fail-safe stop (sysid: fail-safe, crash or drone
mismatch; the session is resumable), 6 start misaligned (UC69 oracle: the drone is not at the
track's start; it was never armed).
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
#: UC69: the telemetry spawn does not match the track's start (the drone was never armed).
EXIT_START_MISALIGNED = 6


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
    from drone_fly.harness.errors import StartMisalignedError
    from drone_fly.harness.session import run_harness

    if args.teacher is not None and (args.checkpoint is not None or args.extra_dim is not None):
        raise HarnessConfigError("--teacher is exclusive with --checkpoint / --extra-dim")
    student = False
    if args.teacher is None and args.checkpoint is not None:
        from drone_fly.student.checkpoint import is_student_checkpoint

        student = is_student_checkpoint(args.checkpoint)
        if student and args.extra_dim is not None:
            raise HarnessConfigError(
                "--extra-dim is not allowed with a student checkpoint (the export records it)"
            )
    if args.teacher is None and not student and (args.checkpoint is None or args.extra_dim is None):
        raise HarnessConfigError(
            "pass --checkpoint and --extra-dim, a student export (--checkpoint), or --teacher"
        )
    teacher_kw = {}
    if args.teacher is not None:
        teacher_kw = {
            "teacher": args.teacher,
            "track": args.track,
            "perception": args.perception,
            "trackdata_root": args.trackdata_root,
            "scenes_root": args.scenes_root,
            "allow_concurrent_training": args.allow_concurrent_training,
            "session_dir": args.session,
            "session_stamp": args.session_stamp,
        }
    elif args.perception == "oracle":
        raise HarnessConfigError("--perception oracle needs --teacher")
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
        save_recording=args.save_recording,
        allow_baseline=args.allow_baseline,
        allow_slow_cpu=args.allow_slow_cpu,
        **teacher_kw,
    )
    r = session.result
    print(
        f"stop: {r.reason}  ticks: {r.ticks}  missed deadlines: {r.missed_deadlines}"
        + (f"  error: {r.error}" if r.error else "")
    )
    if session.run_dir is not None:
        print(f"recorded: {session.run_dir}")
    if session.session_dir is not None:
        print(f"session: {session.session_dir}")
    _hold_until_ctrl_c(session.overlay_process)
    if r.reason == StartMisalignedError.REASON:
        return EXIT_START_MISALIGNED
    return EXIT_FAILSAFE if r.is_failsafe_stop else EXIT_OK


def _cmd_oracle_report(args: argparse.Namespace) -> int:
    from drone_fly.harness.offline.oracle_report import write_report

    out = write_report(args.session, sim_episodes=args.sim_episodes, sim=not args.no_sim)
    print(out["verdict"]["text"])
    print(f"report: {out['written']['md']}")
    return EXIT_OK


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
    if args.detector is not None:
        from drone_fly.harness.offline.evaluate import heldout_eval_summary, run_heldout_eval

        rep = run_heldout_eval(
            args.detector,
            captures=args.captures,
            split=args.split,
            compare=args.compare,
            protected_capture=args.protected_capture,
            config=args.config,
            preset=args.preset or "liftoff",
            device=None if args.device in (None, "auto") else args.device,
            split_source=args.split_source,
            report_path=args.report,
        )
        print(heldout_eval_summary(rep))
        print(f"report: {rep['report_path']}")
        return EXIT_OK
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


def _cmd_captures_register(args: argparse.Namespace) -> int:
    from drone_fly.harness.offline.capture_index import capture_sha256, register

    meta = register(
        args.dir,
        args.track,
        config=args.config,
        preset=args.preset or "liftoff",
        trackdata_root=args.trackdata_root,
        source=args.source,
        force=args.force,
    )
    c = meta["completeness"]
    print(f"registered {meta['name']}: {meta['catalogue_id']} ({meta['map']}, {meta['split']})")
    print(f"completeness: {c['status']} ({c['source']}: {c.get('passed')}/{c.get('required')})")
    print(f"capture sha256: {capture_sha256(args.dir)}")
    print("alignment: unverified (run `drone-fly harness captures verify`)")
    return EXIT_OK


def _cmd_captures_record(args: argparse.Namespace) -> int:
    from drone_fly.harness.liftoff.recorder import record_manual

    out, res = record_manual(
        args.track,
        max_s=args.max_s,
        fps=args.fps,
        port=args.port,
        trackdata_root=args.trackdata_root,
        preset=args.preset or "liftoff",
    )
    print(json.dumps(res["stats"], indent=2))
    print(f"capture: {out}")
    print(f"next: drone-fly harness captures verify {out} --search-lag")
    return EXIT_OK


def _browser_pos(
    text: str | None,
) -> tuple[tuple[int | None, int | None, int], tuple[int | None, int | None]] | None:
    """``ENV[/N],RACE[/N],ROW`` (0-based; ``-`` keeps that filter; ``/N`` = the dropdown's entry
    count) -> ``((env, race, row), (env count, race count))``."""
    if not text:
        return None
    parts = text.split(",")
    if len(parts) != 3:
        raise HarnessConfigError(f"--browser-pos must be ENV,RACE,ROW, got {text!r}")
    try:
        idx, counts = [], []
        for v in parts[:2]:
            i, _, n = v.strip().partition("/")
            idx.append(None if i == "-" else int(i))
            counts.append(int(n) if n else None)
        return (idx[0], idx[1], int(parts[2])), (counts[0], counts[1])
    except ValueError as exc:
        raise HarnessConfigError(f"--browser-pos must be integers or '-', got {text!r}") from exc


def _browser_kwargs(text: str | None) -> dict[str, object]:
    parsed = _browser_pos(text)
    return {} if parsed is None else {"browser_pos": parsed[0], "dropdown_counts": parsed[1]}


def _cmd_captures_replay(args: argparse.Namespace) -> int:
    from drone_fly.harness.liftoff.recorder import record_replay

    out, res = record_replay(
        args.recording,
        max_s=args.max_s,
        fps=args.fps,
        port=args.port,
        strategy=args.strategy,
        recordings_dir=args.dir,
        trackdata_root=args.trackdata_root,
        preset=args.preset or "liftoff",
        allow_pending_camera=args.allow_pending_camera,
        **_browser_kwargs(args.browser_pos),
    )
    print(json.dumps(res, indent=2))
    print(f"capture: {out}")
    return EXIT_OK if res["alignment"]["verified"] else EXIT_CONFIG


def _cmd_captures_inventory(args: argparse.Namespace) -> int:
    from drone_fly.harness.offline.recordings import inventory, inventory_table

    rows = inventory(
        args.dir,
        trackdata_root=args.trackdata_root,
        pose_completeness=not args.no_pose,
    )
    print(inventory_table(rows))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps([r.as_dict() for r in rows], indent=2),
                                   encoding="utf-8")  # fmt: skip
        print(f"inventory: {args.json}")
    return EXIT_OK


def _cmd_captures_verify(args: argparse.Namespace) -> int:
    from drone_fly.harness.offline import align

    kw = {"config": args.config, "preset": args.preset or "liftoff",
          "trackdata_root": args.trackdata_root, "max_frames": args.max_frames}  # fmt: skip
    if args.calibrate:
        rep = align.calibration_report(args.dir, **kw)
        print(json.dumps(rep, indent=2))
        return EXIT_OK
    res = align.verify_capture(
        args.dir,
        write=not args.no_write,
        search_lag=args.search_lag,
        search_uptilt=args.search_uptilt,
        **kw,
    )
    print(json.dumps(res.alignment_block(), indent=2))
    print("verified" if res.verified else "NOT verified: " + "; ".join(res.reasons))
    return EXIT_OK if res.verified else EXIT_CONFIG


def _gate_captures(args: argparse.Namespace) -> list[Path]:
    from drone_fly.harness.offline.capture_index import iter_capture_dirs, resolve_captures

    if args.all_captures:
        return iter_capture_dirs(args.captures_root)
    if not args.captures:
        raise HarnessConfigError("pass --captures <dir|glob|name>... or --all-captures")
    return resolve_captures(args.captures, args.captures_root)


def _trainable_captures(args: argparse.Namespace) -> list[Path]:
    """``--all-captures`` leaves out held-out-map captures (by name they still raise)."""
    caps = _gate_captures(args)
    if not args.all_captures:
        return caps
    from drone_fly.harness.offline.capture_index import CaptureMetaError, link_entry, read_meta
    from drone_fly.tracks import catalogue as cat

    catalogue = cat.load_catalogue(args.split_source) if args.split_source else cat.load_named(
        "liftoff")  # fmt: skip
    out = []
    for d in caps:
        try:
            if link_entry(read_meta(d), catalogue)["split"] == "held-out":
                logging.getLogger(__name__).info("held-out map: %s left out", d.name)
                continue
        except CaptureMetaError:
            pass  # assemble records the refusal
        out.append(d)
    return out


def _floats(text: str, flag: str) -> tuple[float, ...]:
    try:
        return tuple(float(v) for v in text.split(","))
    except ValueError as exc:
        raise HarnessConfigError(f"{flag} must be comma-separated numbers, got {text!r}") from exc


def _cmd_gates_train(args: argparse.Namespace) -> int:
    from drone_fly.harness.offline.gate_train import RollAug, train_multi

    meta = train_multi(
        _trainable_captures(args),
        trackdata_root=args.trackdata_root,
        split_source=args.split_source,
        detector_id=args.detector_id,
        config=args.config,
        preset=args.preset or "liftoff",
        epochs=args.epochs,
        backbone=args.backbone,
        batch_size=args.batch_size,
        lr=args.lr,
        input_width=args.input_width,
        frame_stride=args.frame_stride,
        max_frames_per_capture=args.max_frames_per_capture,
        select=args.select,
        val_every=args.val_every,
        patience=args.patience,
        protect_holdout_of=None if args.protect_holdout_of == "none" else args.protect_holdout_of,
        deploy=args.deploy,
        device=args.device,
        roll=RollAug(
            p=args.aug_roll_p,
            edges=_floats(args.aug_roll_edges, "--aug-roll-edges"),
            weights=_floats(args.aug_roll_weights, "--aug-roll-weights"),
            fallback_weights=_floats(args.aug_roll_fallback_weights, "--aug-roll-fallback-weights"),
            shift=args.aug_shift,
            fill=args.aug_roll_fill,
            keep_sign_p=args.aug_roll_keep_sign_p,
        ).validate(),
        rotated_val=_floats(args.rotated_val, "--rotated-val") if args.rotated_val else (),
        live_roll_val=args.live_roll_val,
        resume=args.resume,
    )
    print(f"weights: {meta['weights']} ({meta['purpose']}, epoch {meta['chosen_epoch']})")
    if meta.get("chosen_val_detection"):
        print(f"val detection: {meta['chosen_val_detection']}")
    print(f"data manifest: {meta['data_manifest']}")
    return EXIT_OK


def _peak_rss_mb() -> float | None:
    """This process's peak resident memory (MB), or None when the platform can't say."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t)
                for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                          "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                          "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")
            ]  # fmt: skip

        c = _Counters()
        c.cb = ctypes.sizeof(c)
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return c.PeakWorkingSetSize / 2**20
    try:
        import resource
    except ImportError:
        return None
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 2**20 if sys.platform == "darwin" else kb / 1024


def _cmd_gates_dataset_report(args: argparse.Namespace) -> int:
    from drone_fly.harness.config import load_config
    from drone_fly.harness.gate_types import GateTypeTable
    from drone_fly.harness.offline import gate_dataset as gd
    from drone_fly.harness.offline.capture_index import CaptureMetaError, link_entry, read_meta
    from drone_fly.harness.offline.gate_train import load_labels
    from drone_fly.tracks import catalogue as cat

    cfg = load_config("liftoff", args.config, preset=args.preset or "liftoff")
    catalogue = cat.load_catalogue(args.split_source) if args.split_source else cat.load_named(
        "liftoff")  # fmt: skip
    usable, heldout_rows = [], {}
    for d in _gate_captures(args):
        try:
            meta = read_meta(d)
            split = link_entry(meta, catalogue)["split"]
        except CaptureMetaError:
            usable.append(d)  # assemble records the refusal
            continue
        if split == "held-out":
            lab = gd.labels_dir_for(d, meta)
            ok_cam = gd.same_camera(meta, float(cfg.section("camera")["uptilt_deg"]))
            if lab is not None and meta["alignment"].get("verified") and ok_cam:
                heldout_rows.setdefault(meta["map"], []).extend(load_labels(lab).rows)
            continue
        usable.append(d)
    protect = None if args.protect_holdout_of == "none" else args.protect_holdout_of
    if protect == "release:v1":
        from drone_fly.harness.liftoff.gate_learned import configured_weights

        protect = configured_weights({"release": "v1"})
    ds = gd.assemble(
        usable,
        catalogue=catalogue,
        table=GateTypeTable.from_config(cfg.get("gate_types")),
        not_gates=cfg.not_gates,
        trackdata_root=args.trackdata_root,
        protect=gd.read_weights_meta(protect),
        frame_stride=args.frame_stride,
        max_frames_per_capture=args.max_frames_per_capture,
        telemetry_cfg=cfg.section("telemetry"),
        preset_uptilt_deg=float(cfg.section("camera")["uptilt_deg"]),
    )
    rep = gd.dataset_report(ds, heldout_rows)
    print(gd.report_summary(rep))
    if args.build_cache:
        lw = args.input_width or int(
            (cfg.section("gates").get("learned") or {}).get("input_width", 768)
        )
        _, idx = gd.build_frame_cache(ds, lw, cfg.section("camera"))
        print(f"frame cache: {gd.cache_dir(ds, lw)} ({idx['shape'][0]} frames)")
        peak = _peak_rss_mb()
        if peak is not None:
            print(f"peak RAM: {peak:.0f} MB")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"report: {args.json}")
    return EXIT_OK if rep["g1"]["go"] else EXIT_COVERAGE


def _cmd_gates_diagnose(args: argparse.Namespace) -> int:
    from drone_fly.harness.offline.gate_diagnose import diagnose, diagnose_table

    rep = diagnose(
        args.weights,
        args.captures,
        config=args.config,
        preset=args.preset or "liftoff",
        device=None if args.device in (None, "auto") else args.device,
        all_frames=args.all_frames,
    )
    print(diagnose_table(rep))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"report: {args.json}")
    return EXIT_OK


def _cmd_gates_draft_types(args: argparse.Namespace) -> int:
    import yaml

    from drone_fly.harness.offline.autolabel import draft_gate_types, load_props

    drafts = draft_gate_types(load_props(args.props), args.items)
    print("# CANDIDATES ONLY: check every class with label previews before training on it.")
    print(yaml.safe_dump({"gate_types": drafts}, sort_keys=False, default_flow_style=None))
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


def _cmd_ui(args: argparse.Namespace) -> int:
    from drone_fly.harness.liftoff import ui_driver

    if args.action == "shot":
        print(f"saved {ui_driver.shot(args.out)} (half size: x2 = screen pixels)")
    elif args.action == "select-drone":
        if not args.name:
            raise HarnessConfigError("ui select-drone needs --name (the exact Liftoff drone name)")
        slot = ui_driver.select_drone(args.name, args.slot, level=args.level)
        print(f"select-drone: {args.name!r} (slot {slot})")
    elif args.action == "save-recording":
        if not args.name:
            raise HarnessConfigError("ui save-recording needs --name")
        print(f"save-recording: {ui_driver.save_recording(args.name)}")
    elif args.action == "retry":
        ui_driver.retry(args.level or "infinity-loop")
        print("retry: ok")
    else:
        ui_driver.ACTIONS[args.action]()
        print(f"{args.action}: ok")
    return EXIT_OK


def _cmd_sysid_liftoff(args: argparse.Namespace) -> int:
    from drone_fly.sysid.session import run_sysid_liftoff

    band = _float_pair(args.expect_hover, "--expect-hover") if args.expect_hover else None
    result = run_sysid_liftoff(
        args.drone,
        session=args.session,
        resume=args.resume,
        select=args.select,
        liftoff_name=args.liftoff_name,
        seed=args.seed,
        passes=args.passes,
        hold_alt=args.hold_alt,
        home_forward=args.home_forward,
        ceiling=args.ceiling,
        radius=args.radius,
        expect_hover=band,
        preset=args.preset,
        config=args.config,
        plant=args.plant,
        map_name=args.map,
        level=args.level,
    )
    print(
        f"stop: {result.stop_reason}  done: {result.done}  aborted: {result.aborted}"
        + (f"  h0: {result.h0:.3f}" if result.h0 is not None else "")
        + (f"  error: {result.error}" if result.error else "")
    )
    print(f"session: {result.session_dir}")
    if result.resumable:
        print(f"resume with: --session {Path(result.session_dir).name} --resume")
    if result.stop_reason == "anticheat":
        return EXIT_CONFIG
    return EXIT_OK if result.completed else EXIT_FAILSAFE


def _cmd_sysid_fit(args: argparse.Namespace) -> int:
    from drone_fly.drones.store import drones_root
    from drone_fly.sysid.fit import fit_session
    from drone_fly.sysid.report import write_report

    out = Path(args.out) if args.out else drones_root() / args.drone
    if (out / "drone.yaml").is_file() and not args.force:
        raise HarnessConfigError(f"{out / 'drone.yaml'} exists: pass --force to overwrite it")
    fit = fit_session(args.session_dir, args.drone)
    liftoff = {"drone_name": fit.meta.get("liftoff_name") or args.drone}
    for key, value in (
        ("map", args.map or fit.meta.get("map")),
        ("preset", fit.meta.get("preset")),
    ):
        if value:
            liftoff[key] = value
    camera = None
    if args.camera_from_preset:
        from drone_fly.harness.config import load_config

        cam = load_config("liftoff", None, preset=args.camera_from_preset).section("camera")
        camera = {
            "hfov_deg": float(cam["hfov_deg"]),
            "uptilt_deg": float(cam["uptilt_deg"]),
            "ref": f"harness preset {args.camera_from_preset!r} camera (verified in UC-64)",
        }
    yaml_path, md = write_report(
        fit,
        out,
        drone_id=args.drone,
        name=args.name,
        liftoff=liftoff,
        camera=camera,
        micro=args.micro,
    )
    print(f"fit_quality: {fit.fit_quality}. {fit.verdict}")
    for note in fit.notes:
        if note.startswith("WARNING"):
            print(note)
    print(f"profile: {yaml_path}")
    print(f"report: {md}")
    return EXIT_OK


def _cmd_sim_replay(args: argparse.Namespace) -> int:
    from drone_fly import drones
    from drone_fly.sim.replay import replay_session, write_report_section

    profile = drones.load(args.drone)
    result = replay_session(args.session_dir, profile, gated=args.gate)
    print(result.verdict)
    for h in (1.0, 2.0):
        print(
            f"  {h:g} s: median attitude {result.median('att_deg', h):.3f} deg, velocity "
            f"{result.median('vel_m_s', h):.3f} m/s over {result.counts(h)} windows"
        )
    if args.write_report:
        md = drones.profile_path(args.drone).with_name("sysid.md")
        print(f"report: {write_report_section(md, result)}")
    return 1 if result.gated and not result.passed else EXIT_OK


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
    p_run.add_argument("--checkpoint", help="stripped model_readout.pt (or --teacher)")
    p_run.add_argument(
        "--extra-dim", type=int, choices=(12, 30, 70), help="model extra width (with --checkpoint)"
    )
    p_run.add_argument(
        "--teacher",
        help="UC69: a UC68 teacher run folder or .pt (exclusive with --checkpoint/--extra-dim)",
    )
    p_run.add_argument("--track", help="UC69: catalogue track id, e.g. liftoff-arena-04")
    p_run.add_argument(
        "--allow-baseline",
        action="store_true",
        help="UC70: fly a deployable: false (baseline) student export",
    )
    p_run.add_argument(
        "--allow-slow-cpu",
        action="store_true",
        help="UC70: run a K1 student on the CPU (~220 ms per tick; debugging only)",
    )
    p_run.add_argument(
        "--perception",
        choices=("oracle", "detector"),
        help="UC69 with --teacher: oracle (exact gates from the track; default) or detector",
    )
    p_run.add_argument("--trackdata-root", help="track XMLs + props.json ($DRONE_FLY_TRACKDATA)")
    p_run.add_argument("--scenes-root", help="scene caches ($DRONE_FLY_SCENES)")
    p_run.add_argument(
        "--allow-concurrent-training",
        action="store_true",
        help="fly while the teacher's run still trains (labelled; never counts toward AC9)",
    )
    p_run.add_argument(
        "--session",
        help="UC69 session folder (default artifacts/eval/<drone>/<run>/oracle/<track>/sessions/"
        "<stamp>-<sha8>)",
    )
    p_run.add_argument(
        "--session-stamp", help="UC69: the <stamp> of the default session folder (default: now)"
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
        "--save-recording",
        metavar="NAME",
        help="Liftoff: right after the stop, save the in-game recording of this attempt as NAME",
    )
    p_run.add_argument(
        "--recordable",
        action="store_true",
        help="overlay visible to screen recorders; mask its panels out of the capture "
        "(implies --overlay; overlay.recordable)",
    )
    p_run.set_defaults(func=_cmd_run)

    p_orep = hsub.add_parser(
        "oracle-report", help="UC69: score an oracle teacher session and compare it with the sim"
    )
    p_orep.add_argument("--session", required=True, help="session folder (with attempts/)")
    p_orep.add_argument(
        "--sim-episodes", type=int, default=32, help="sim episodes per mode (default 32)"
    )
    p_orep.add_argument("--no-sim", action="store_true", help="skip the sim evaluation")
    p_orep.set_defaults(func=_cmd_oracle_report)

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
    p_eval.add_argument("--detector", help="UC71: weights file or models/<detector-id>/ folder")
    p_eval.add_argument("--split", choices=("held-out", "val"), default="held-out",
                        help="maps to score (catalogue split) with --detector")  # fmt: skip
    p_eval.add_argument("--captures", nargs="+", help="with --detector: these captures only")
    p_eval.add_argument("--compare", help="with --detector: the weights to compare against (v1)")
    p_eval.add_argument("--protected-capture", help="the Infinity Loop reference capture folder")
    p_eval.add_argument("--split-source", help="track catalogue (default tracks/liftoff.yaml)")
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

    p_caps = hsub.add_parser("captures", help="multi-map gate captures (UC71)")
    csub = p_caps.add_subparsers(dest="captures_command", required=True)
    p_reg = csub.add_parser("register", help="write meta.json for an existing capture folder")
    p_reg.add_argument("dir", help="capture folder (frames.csv + telemetry)")
    p_reg.add_argument("--track", required=True, help="tracks/liftoff.yaml catalogue id")
    p_reg.add_argument("--trackdata-root", help="trackdata root (completeness from telemetry)")
    p_reg.add_argument("--source", choices=("manual",), default="manual")
    p_reg.add_argument("--force", action="store_true", help="rewrite an existing meta.json")
    _common(p_reg)
    p_reg.set_defaults(func=_cmd_captures_register)
    p_rec = csub.add_parser("record", help="record a hand-flown capture of a catalogue track")
    p_rec.add_argument("--track", required=True, help="tracks/liftoff.yaml catalogue id")
    p_rec.add_argument("--max-s", type=float, help="stop after N seconds (default: Ctrl+C)")
    p_rec.add_argument("--fps", type=float, default=10.0)
    p_rec.add_argument("--port", type=int, default=9001, help="Liftoff telemetry UDP port")
    p_rec.add_argument("--trackdata-root", help="trackdata root (completeness)")
    _common(p_rec)
    p_rec.set_defaults(func=_cmd_captures_record)
    p_rep = csub.add_parser("replay", help="play a stored recording in Liftoff and capture it")
    p_rep.add_argument("--recording", required=True, help="recording name (file stem) or path")
    p_rep.add_argument("--max-s", type=float, help="cap the capture length")
    p_rep.add_argument("--fps", type=float, default=10.0)
    p_rep.add_argument("--port", type=int, default=9001)
    p_rep.add_argument("--strategy", choices=("auto", "A", "B", "C"), default="auto")
    p_rep.add_argument("--browser-pos", help="ENV[/N],RACE[/N],ROW: replay-browser dropdown / "
                       "list indices, 0-based ('-' keeps a filter; /N = entries in that "
                       "dropdown, needed for its last 5)")  # fmt: skip
    p_rep.add_argument("--dir", help="Recordings folder (default %%USERPROFILE%%\\AppData\\...)")
    p_rep.add_argument("--trackdata-root", help="trackdata root (verify)")
    p_rep.add_argument("--allow-pending-camera", action="store_true",
                       help="try a recording made with another drone/camera")  # fmt: skip
    _common(p_rep)
    p_rep.set_defaults(func=_cmd_captures_replay)
    p_inv = csub.add_parser("inventory", help="list Liftoff's stored recordings (read-only)")
    p_inv.add_argument("--dir", help="Recordings folder (default %%USERPROFILE%%\\AppData\\...)")
    p_inv.add_argument("--json", help="write the rows as JSON here")
    p_inv.add_argument("--trackdata-root", help="trackdata root (pose completeness)")
    p_inv.add_argument("--no-pose", action="store_true", help="skip pose-based completeness")
    p_inv.set_defaults(func=_cmd_captures_inventory)
    p_ver = csub.add_parser("verify", help="projection alignment: lag, camera, verified flag")
    p_ver.add_argument("dir", help="capture folder with meta.json")
    p_ver.add_argument("--search-lag", action="store_true", help="search the time offset +-1.5 s")
    p_ver.add_argument("--search-uptilt", action="store_true", help="search uptilt +-5 deg")
    p_ver.add_argument("--trackdata-root", help="trackdata root (the capture's track XML)")
    p_ver.add_argument("--max-frames", type=int, default=120, help="frames sampled")
    p_ver.add_argument("--no-write", action="store_true", help="do not update meta.json")
    p_ver.add_argument("--calibrate", action="store_true",
                       help="print margins of the best vs deliberately wrong poses")  # fmt: skip
    _common(p_ver)
    p_ver.set_defaults(func=_cmd_captures_verify)

    p_gates = hsub.add_parser("gates", help="multi-map gate detector: train, report, types")
    gsub = p_gates.add_subparsers(dest="gates_command", required=True)

    def _gate_inputs(p: argparse.ArgumentParser) -> None:
        p.add_argument("--captures", nargs="+", help="capture folders, globs or names")
        p.add_argument("--all-captures", action="store_true", help="every capture found")
        p.add_argument("--captures-root", help="default artifacts/harness/captures")
        p.add_argument("--split-source", help="track catalogue (default tracks/liftoff.yaml)")
        p.add_argument("--trackdata-root", help="trackdata root (track XMLs)")
        p.add_argument("--frame-stride", type=int, default=1)
        p.add_argument("--max-frames-per-capture", type=int)
        p.add_argument("--protect-holdout-of", default="release:v1",
                       help="weights whose held-out windows stay out ('none' = off)")  # fmt: skip
        _common(p)

    p_gt = gsub.add_parser("train", help="train on train-map captures, select on val maps")
    _gate_inputs(p_gt)
    p_gt.add_argument("--detector-id", help="default liftoff-gates-v2-<YYYYMMDD-HHMM>")
    p_gt.add_argument("--epochs", type=int, default=40)
    p_gt.add_argument("--backbone", choices=("resnet18-hf", "tiny-cnn"), default="resnet18-hf")
    p_gt.add_argument("--batch-size", type=int, default=8)
    p_gt.add_argument("--lr", type=float, default=1e-3)
    p_gt.add_argument("--input-width", type=int)
    p_gt.add_argument("--select", choices=("val-detection", "val-loss", "last"),
                      default="val-detection")  # fmt: skip
    p_gt.add_argument("--val-every", type=int, default=2, help="val decode every N epochs")
    p_gt.add_argument("--patience", type=int, help="early stop after N epochs without gain")
    p_gt.add_argument("--deploy", action="store_true",
                      help="retrain on train + val maps (select last) -> *-deploy.pt")  # fmt: skip
    p_gt.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p_gt.add_argument("--resume", action="store_true",
                      help="continue <detector-id>/resume.pt after the last finished epoch "
                           "(same captures and flags)")  # fmt: skip
    p_gt.add_argument("--aug-roll-p", type=float, default=0.0,
                      help="share of images rolled (0 = off, as R1)")  # fmt: skip
    p_gt.add_argument("--aug-roll-edges", default="0,15,35,60,90,135",
                      help="target |roll| buckets, deg, max 135 (live mix)")  # fmt: skip
    p_gt.add_argument("--aug-roll-weights", default="0.15,0.35,0.40,0.07,0.03",
                      help="target weight per bucket (live flight, ADR 0010)")  # fmt: skip
    p_gt.add_argument("--aug-roll-fallback-weights", default="0.10,0.25,0.35,0.25,0.05",
                      help="raw-angle weights for frames with no known roll")  # fmt: skip
    p_gt.add_argument("--aug-roll-keep-sign-p", type=float, default=0.8,
                      help="chance a target keeps the frame's own roll sign")  # fmt: skip
    p_gt.add_argument("--live-roll-val", action="store_true",
                      help="also report val re-rolled to the live mix")  # fmt: skip
    p_gt.add_argument("--aug-roll-fill", choices=("mean", "reflect"), default="mean",
                      help="uncovered corners: ImageNet mean colour or reflection")  # fmt: skip
    p_gt.add_argument("--rotated-val", help="report val detection on frames rolled by these "
                      "angles, e.g. 0,30,60,90,135 (not used for selection)")  # fmt: skip
    p_gt.add_argument("--aug-shift", type=float, default=0.0,
                      help="random translation, fraction of the width (<= 0.05)")  # fmt: skip
    p_gt.set_defaults(func=_cmd_gates_train)
    p_gr = gsub.add_parser("dataset-report", help="in-range gate-frame counts and the G1 gate")
    _gate_inputs(p_gr)
    p_gr.add_argument("--json", help="write the report JSON here")
    p_gr.add_argument(
        "--build-cache",
        action="store_true",
        help="also build the training frame cache (CPU) for `gates train`",
    )
    p_gr.add_argument("--input-width", type=int, help="network input width of the cache")
    p_gr.set_defaults(func=_cmd_gates_dataset_report)
    p_gd = gsub.add_parser("draft-types", help="candidate gate_types entries from props.json")
    p_gd.add_argument("--props", required=True, help="props.json (liftoff-flight-analyzer)")
    p_gd.add_argument("--items", nargs="+", required=True, help="itemIDs")
    p_gd.set_defaults(func=_cmd_gates_draft_types)
    p_gx = gsub.add_parser("diagnose", help="attribute detector misses per range bin")
    p_gx.add_argument("--weights", required=True, help="weights file or models/<detector-id>/")
    p_gx.add_argument("--captures", nargs="+", required=True, help="labelled capture folders")
    p_gx.add_argument("--all-frames", action="store_true",
                      help="also score frames the weights trained on")  # fmt: skip
    p_gx.add_argument("--json", help="write the full report JSON here")
    p_gx.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    _common(p_gx)
    p_gx.set_defaults(func=_cmd_gates_diagnose)

    p_train = hsub.add_parser(
        "train-gates", help="train the learned gate detector on labels (alias, single-map UC64)"
    )
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

    p_ui = hsub.add_parser(
        "ui",
        help="drive Liftoff's menus: launch | last-session | reset | retry (after a finished "
        "race) | save-recording | select-drone | quit | shot",
    )
    p_ui.add_argument(
        "action",
        choices=(
            "launch",
            "last-session",
            "reset",
            "retry",
            "save-recording",
            "select-drone",
            "quit",
            "shot",
        ),
    )
    p_ui.add_argument("--out", default="liftoff-screen.png", help="shot: PNG path (half size)")
    p_ui.add_argument(
        "--name",
        help="save-recording: recording name (no extension); select-drone: exact drone name",
    )
    p_ui.add_argument("--slot", type=int, help="select-drone: list row (default: DRONE_SLOTS)")
    p_ui.add_argument(
        "--level",
        help="select-drone: also load this level; retry: the level whose spawn to verify "
        "(default infinity-loop; ui_driver.QUICK_SETUP_LEVELS)",
    )
    p_ui.set_defaults(func=_cmd_ui)

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

    sysid = sub.add_parser("sysid", help="identify a drone's dynamics (drones/<id>/drone.yaml)")
    ssub = sysid.add_subparsers(dest="sysid_command", required=True)
    p_sl = ssub.add_parser("liftoff", help="fly the seeded manoeuvre schedule and record it")
    p_sl.add_argument("--drone", required=True, help="profile id, e.g. liftoff-gremlin-micro")
    p_sl.add_argument("--session", help="session name (default: UTC stamp)")
    p_sl.add_argument(
        "--resume", action="store_true", help="continue --session from its next experiment"
    )
    p_sl.add_argument(
        "--select", action="store_true", help="select --liftoff-name in Liftoff's menus first"
    )
    p_sl.add_argument("--liftoff-name", help="exact Liftoff drone name (recorded; --select)")
    p_sl.add_argument("--map", help="Liftoff map / level flown (recorded)")
    p_sl.add_argument(
        "--level", help="with --select: load this level (ui_driver.QUICK_SETUP_LEVELS)"
    )
    p_sl.add_argument("--seed", type=int, default=0, help="schedule seed")
    p_sl.add_argument("--passes", type=int, default=3, help="passes (>= 2; the last is held out)")
    p_sl.add_argument("--hold-alt", type=float, default=5.0, help="hover height (m)")
    p_sl.add_argument(
        "--home-forward",
        type=float,
        default=0.0,
        help="fly this far forward from the spawn before the experiments (m; geofence centre)",
    )
    p_sl.add_argument("--ceiling", type=float, default=12.0, help="soft ceiling (m above ground)")
    p_sl.add_argument("--radius", type=float, default=30.0, help="geofence radius (m)")
    p_sl.add_argument("--expect-hover", help="airframe guard: hover throttle band lo:hi")
    p_sl.add_argument("--preset", default="liftoff", help="harness preset (telemetry, pad)")
    p_sl.add_argument("--config", help="YAML overlaid on the harness default")
    p_sl.add_argument(
        "--plant",
        choices=("liftoff", "synthetic", "synthetic-5inch"),
        default="liftoff",
        help="synthetic*: a known-parameter simulated plant (dry run, no game)",
    )
    p_sl.set_defaults(func=_cmd_sysid_liftoff)
    p_sf = ssub.add_parser("fit", help="fit a session and write drones/<id>/ (yaml + report)")
    p_sf.add_argument("session_dir", help="artifacts/sysid/<id>/<session>")
    p_sf.add_argument("--drone", required=True, help="profile id (output folder name)")
    p_sf.add_argument("--out", help="output folder (default drones/<id>)")
    p_sf.add_argument("--force", action="store_true", help="overwrite an existing drone.yaml")
    p_sf.add_argument("--name", help="human-readable profile name")
    p_sf.add_argument("--map", help="map flown (default: from the session)")
    p_sf.add_argument(
        "--camera-from-preset", help="copy camera hfov/uptilt from this harness preset"
    )
    p_sf.add_argument(
        "--micro", action="store_true", help="compare the hover with UC64's -0.06 stick"
    )
    p_sf.set_defaults(func=_cmd_sysid_fit)

    from drone_fly.training.cli import add_parser as add_train_parser  # light: no torch import

    add_train_parser(sub)

    from drone_fly.tracks.cli import add_parser as add_tracks_parser  # light: argparse only

    add_tracks_parser(sub)

    sim = sub.add_parser("sim", help="the batched simulator (drone_fly.sim)")
    simsub = sim.add_subparsers(dest="sim_command", required=True)
    p_rp = simsub.add_parser(
        "replay-sysid", help="replay a sysid session's held-out sticks through the plant (AC8)"
    )
    p_rp.add_argument("session_dir", help="artifacts/sysid/<id>/<session>")
    p_rp.add_argument("--drone", required=True, help="profile id (drones/<id>/drone.yaml)")
    p_rp.add_argument(
        "--write-report", action="store_true", help="write the section into drones/<id>/sysid.md"
    )
    p_rp.add_argument(
        "--gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="apply the AC8 gate (default: only when the profile's fit_quality is good)",
    )
    p_rp.set_defaults(func=_cmd_sim_replay)
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
