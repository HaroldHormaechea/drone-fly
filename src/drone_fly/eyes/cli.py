"""``drone-fly eyes fov-report | label | dataset | train | eval | bench`` (UC-78).

* ``fov-report [--capture DIR ...] [--pitch-min 0 --pitch-max 60 --pitch-step 5] [--band 10:40]
  [--fpa 15] [--out]``: AC1, which rays the camera sees.
* ``label --capture DIR [--catalogue-id ID --lag-ms MS] [--trackdata-root] [--scenes-root]
  [--world-scale W] [--ray-range-scale S] [--pair-tol-s 0.03] [--preview N] [--labels-root]
  [--force]``: AC2, exact ray labels.
* ``dataset [--captures-root] [--labels-root] [--label-missing] [--out] [--input-size 192x80]
  [--k 3] [--stride-s 0.2] [--history-tol-s] [--uptilt-tol-deg 2] [--build-cache] [--cache-root]
  [--disk-margin-gb 1]``: AC3, the map-split dataset.
* ``train --dataset D [--out] [--epochs 40] [--batch 64] [--lr 1e-3] [--weight-decay 1e-4]
  [--w-hit 4] [--tol-m 0.5] [--device] [--select best-val|last] [--patience 8] [--seed 78]
  [--num-workers 0] [--resume] [--max-samples N]``: AC4.
* ``eval --weights W --dataset D [--split held-out|val] [--tol-m] [--thin-m 0.2] [--large-m 1.0]
  [--device] [--out]``: AC6.
* ``bench --weights W [--device] [--iters 500] [--warmup 20] [--with-gate-detector]
  [--gate-weights] [--budget-ms] [--out]``: AC7.

Every subcommand takes ``--preset`` (default ``liftoff``), ``--config`` and ``--harness``. Light at
import (argparse only); the work modules load inside the commands.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from drone_fly.harness.errors import HarnessConfigError


def _pair(text: str, flag: str, sep: str = ":") -> tuple[float, float]:
    try:
        a, b = (float(v) for v in text.split(sep))
    except ValueError as exc:
        raise HarnessConfigError(f"{flag} must be a{sep}b, got {text!r}") from exc
    return a, b


def cmd_fov(args: argparse.Namespace) -> int:
    from drone_fly.eyes.fov import run_fov_report

    rep = run_fov_report(
        args.preset, args.config, args.harness, out=args.out, captures=args.capture,
        labels_root=args.labels_root, pitch=(args.pitch_min, args.pitch_max, args.pitch_step),
        band=_pair(args.band, "--band"), fpa=args.fpa, gpu_prototype_dir=args.gpu_prototype_dir,
    )  # fmt: skip
    print(json.dumps(rep["band_summary"], indent=2))
    print(f"report: {rep['out_dir']}")
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    from drone_fly.eyes.labels import run_label

    meta = run_label(
        args.capture, preset=args.preset, config=args.config, harness=args.harness,
        catalogue_id=args.catalogue_id, lag_ms=args.lag_ms, trackdata_root=args.trackdata_root,
        scenes_root=args.scenes_root, world_scale=args.world_scale,
        ray_range_scale=args.ray_range_scale, pair_tol_s=args.pair_tol_s, preview=args.preview,
        labels_root=args.labels_root, force=args.force, gpu_prototype_dir=args.gpu_prototype_dir,
    )  # fmt: skip
    if meta.get("skipped"):
        print(f"skipped {meta['capture']}: {meta['skipped']}")
        return 0
    if meta.get("up_to_date"):
        print(f"up to date (use --force to relabel): {meta['out_dir']}")
        return 0
    keys = ("counts", "drop_reasons", "alignment_s", "hit_kinds", "speed_mps", "warning_s")
    print(json.dumps({k: meta[k] for k in keys}, indent=2))
    print(f"labels: {meta['out_dir']}")
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    from drone_fly.eyes.dataset import build_dataset

    w, h = (int(v) for v in _pair(args.input_size, "--input-size", "x"))
    label_kw = {"trackdata_root": args.trackdata_root, "scenes_root": args.scenes_root}
    m = build_dataset(
        args.captures_root, labels_root=args.labels_root, out=args.out,
        label_missing=args.label_missing, width=w, height=h, k_frames=args.k,
        stride_s=args.stride_s, hist_tol_s=args.history_tol_s, uptilt_tol_deg=args.uptilt_tol_deg,
        build_cache=args.build_cache, cache_root=args.cache_root,
        disk_margin_gb=args.disk_margin_gb, preset=args.preset, config=args.config,
        harness=args.harness, label_kwargs=label_kw, hit_frac=args.hit_frac,
        mining_seed=args.mining_seed,
    )  # fmt: skip
    print(json.dumps({"id": m["id"], "split_maps": m["split_maps"],
                      "samples": {s: v["samples"] for s, v in m["splits"].items()},
                      "mining": m.get("mining"),
                      "drop_reasons": m["drop_reasons"]}, indent=2))  # fmt: skip
    print(f"dataset: {m['out_dir']}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from drone_fly.eyes.train import train_eyes

    r = train_eyes(
        args.dataset, out=args.out, epochs=args.epochs, batch=args.batch, lr=args.lr,
        weight_decay=args.weight_decay, w_hit=args.w_hit, tol_m=args.tol_m, device=args.device,
        select=args.select, patience=args.patience, seed=args.seed, num_workers=args.num_workers,
        resume=args.resume, max_samples=args.max_samples, cache_root=args.cache_root,
        gpu_prototype_dir=args.gpu_prototype_dir, hit_head=args.hit_head, focal=args.focal,
        focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma, w_focal=args.w_focal,
        hit_sampler=args.hit_sampler, hit_sample_w=args.hit_sample_w,
    )  # fmt: skip
    last = r["log"][-1]
    print(f"{r['status']}: {len(r['log'])} epochs, best epoch {r['best_epoch']}, last train "
          f"{last['train_loss']:.4f} val {last['val_loss']}")  # fmt: skip
    print(f"run: {r['run_dir']}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from drone_fly.eyes.evaluate import run_eval

    rep = run_eval(
        args.weights, args.dataset, split=args.split, tol_m=args.tol_m, thin_m=args.thin_m,
        large_m=args.large_m, device=args.device, out=args.out, cache_root=args.cache_root,
    )  # fmt: skip
    print(json.dumps(rep.get("aggregate", {"frames": 0}), indent=2))
    print(f"report: {rep['out_dir']}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from drone_fly.eyes.evaluate import run_compare

    cmp = run_compare(
        args.v1, args.v2, min_rel_reduction=args.min_rel_reduction,
        max_fa_clear=args.max_fa_clear, out=args.out,
    )  # fmt: skip
    print(json.dumps({k: cmp[k] for k in (
        "v1_nearest_quarter_in_view_miss", "v2_nearest_quarter_in_view_miss", "relative_reduction",
        "v2_false_alarm_rate_clear_all", "miss_passed", "fa_passed", "passed", "verdict",
    )}, indent=2))  # fmt: skip
    print(f"compare: {cmp['out_dir']}")
    return 0


def cmd_noise_model(args: argparse.Namespace) -> int:
    from drone_fly.eyes.ray_noise_model import run_noise_model

    res = run_noise_model(
        args.preset, args.config, args.harness, from_eval=args.from_eval, out=args.out
    )  # fmt: skip
    m = res["model"]
    print(json.dumps({"source": m["source"], "miss_prob": m["miss_prob"],
                      "false_alarm_clear": m["false_alarm_clear"],
                      "sigma_m": m["sigma_m"]}, indent=2))  # fmt: skip
    print(f"ray-noise model: {res['out']}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from drone_fly.eyes.bench import run_bench

    rep = run_bench(
        args.weights, preset=args.preset, config=args.config, harness=args.harness,
        device=args.device, iters=args.iters, warmup=args.warmup,
        with_gate_detector=args.with_gate_detector, gate_weights=args.gate_weights,
        budget_ms=args.budget_ms, out=args.out,
    )  # fmt: skip
    print(json.dumps({"eyes_ms": rep["eyes_ms"], "within_budget": rep["eyes_p95_within_budget"],
                      "detector": rep["detector"]}, indent=2))  # fmt: skip
    print(f"report: {rep['out_dir']}")
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", default="liftoff", help="harness preset (camera, scales, masks)")
    p.add_argument("--config", help="YAML overlaid on the preset")
    p.add_argument("--harness", default="liftoff", help="harness whose config to use")


def add_parser(sub: Any) -> None:
    eyes = sub.add_parser("eyes", help="vision eyes: camera estimator of the 40 rays (UC-78)")
    es = eyes.add_subparsers(dest="eyes_command", required=True)

    p = es.add_parser("fov-report", help="which rays the camera sees (AC1)")
    _common(p)
    p.add_argument("--capture", action="append", help="add empirical fractions from its labels")
    p.add_argument("--labels-root", help="eyes labels root (default <data root>/labels/eyes)")
    p.add_argument("--pitch-min", type=float, default=0.0)
    p.add_argument("--pitch-max", type=float, default=60.0)
    p.add_argument("--pitch-step", type=float, default=5.0)
    p.add_argument("--band", default="10:40", help="racing pitch band lo:hi (deg)")
    p.add_argument("--fpa", type=float, default=15.0, help="+- flight-path angle (deg)")
    p.add_argument("--out", help="default <data root>/reports/eyes/fov")
    p.add_argument("--gpu-prototype-dir")
    p.set_defaults(func=cmd_fov)

    p = es.add_parser("label", help="exact 40-ray labels for a capture (AC2)")
    _common(p)
    p.add_argument("--capture", required=True)
    p.add_argument("--catalogue-id", help="legacy capture without meta.json")
    p.add_argument("--lag-ms", type=float, help="legacy capture: frame-telemetry lag")
    p.add_argument("--trackdata-root")
    p.add_argument("--scenes-root")
    p.add_argument("--world-scale", type=float, help="default: the preset's scaling.world_scale")
    p.add_argument("--ray-range-scale", type=float, help="default: depth.ray_range_scale")
    p.add_argument("--pair-tol-s", type=float, default=0.03)
    p.add_argument("--preview", type=int, default=0, help="overlay N frames (max 20)")
    p.add_argument("--labels-root", help="default <data root>/labels/eyes")
    p.add_argument("--force", action="store_true", help="relabel even when up to date")
    p.add_argument("--gpu-prototype-dir")
    p.set_defaults(func=cmd_label)

    p = es.add_parser("dataset", help="map-split dataset over a captures root (AC3)")
    _common(p)
    p.add_argument("--captures-root", help="default <data root>/captures")
    p.add_argument("--labels-root")
    p.add_argument("--label-missing", action="store_true", help="label missing / stale captures")
    p.add_argument("--trackdata-root", help="for --label-missing")
    p.add_argument("--scenes-root", help="for --label-missing")
    p.add_argument("--out", help="default <data root>/datasets/eyes/<id>")
    p.add_argument("--input-size", default="192x80")
    p.add_argument("--k", type=int, default=3, help="frames per sample")
    p.add_argument("--stride-s", type=float, default=0.2, help="history stride (s)")
    p.add_argument("--history-tol-s", type=float, help="default: 50 %% of the stride")
    p.add_argument("--uptilt-tol-deg", type=float, default=2.0)
    p.add_argument("--build-cache", action="store_true", help="also build the frame cache now")
    p.add_argument("--cache-root", help="default <data root>/cache/eyes")
    p.add_argument("--disk-margin-gb", type=float, default=1.0)
    # UC-83 AC2: opt-in hit-biased frame filter on the train split (default off => v1 dataset).
    p.add_argument("--hit-frac", type=float,
                   help="subsample clear train frames to reach ~this hit-bearing-frame fraction")  # fmt: skip
    p.add_argument("--mining-seed", type=int, default=0, help="seed for the hit-biased filter")
    p.set_defaults(func=cmd_dataset)

    p = es.add_parser("train", help="train the estimator (AC4)")
    _common(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", help="run folder (default <models root>/eyes/<stamp>-<dataset id>)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--w-hit", type=float, default=4.0)
    p.add_argument("--tol-m", type=float, default=0.5, help="confidence tolerance (training m)")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--select", default="best-val", choices=("best-val", "last"))
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=78)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue --out from resume.pt")
    p.add_argument("--max-samples", type=int, help="cap samples per split (smoke runs)")
    p.add_argument("--cache-root")
    p.add_argument("--gpu-prototype-dir")
    # UC-83 v2 levers (all default-off => byte-identical to v1).
    p.add_argument("--hit-head", action="store_true",
                   help="build the opt-in hit/no-hit classification head (eyes v2)")  # fmt: skip
    p.add_argument("--focal", action="store_true",
                   help="add the focal term on the hit head (needs --hit-head)")  # fmt: skip
    p.add_argument("--focal-alpha", type=float, default=0.25, help="focal class weight on hits")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="focal easy-ray down-weighting")
    p.add_argument("--w-focal", type=float, default=1.0, help="weight of the focal term in the loss")
    p.add_argument("--hit-sampler", action="store_true",
                   help="oversample hit-bearing frames each epoch (train-time; v2 primary lever)")  # fmt: skip
    p.add_argument("--hit-sample-w", type=float, default=8.0,
                   help="relative sampling weight of a hit-bearing frame when --hit-sampler")  # fmt: skip
    p.set_defaults(func=cmd_train)

    p = es.add_parser("eval", help="accuracy report on held-out / val maps (AC6)")
    _common(p)
    p.add_argument("--weights", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="held-out", choices=("held-out", "val"))
    p.add_argument("--tol-m", type=float, help="default: the weights meta's")
    p.add_argument("--thin-m", type=float, default=0.2)
    p.add_argument("--large-m", type=float, default=1.0)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--out", help="default: the weights' folder")
    p.add_argument("--cache-root")
    p.set_defaults(func=cmd_eval)

    p = es.add_parser("compare", help="v1-vs-v2 ray-accuracy verdict from two eval reports (AC4)")
    _common(p)
    p.add_argument("--v1", required=True, help="v1-baseline eval_<split>.json")
    p.add_argument("--v2", required=True, help="v2 eval_<split>.json (same held-out split)")
    p.add_argument("--min-rel-reduction", type=float, default=0.30,
                   help="min relative reduction in nearest-quarter in-view miss to pass")  # fmt: skip
    p.add_argument("--max-fa-clear", type=float, default=0.10,
                   help="max v2 clear false-alarm rate to pass")  # fmt: skip
    p.add_argument("--out", help="default: the v2 report's folder")
    p.set_defaults(func=cmd_compare)

    p = es.add_parser("noise-model", help="build the obstacle-ray noise model (UC-82 AC1)")
    _common(p)
    p.add_argument(
        "--from-eval", help="derive miss / FA from an eyes eval report (eval_held-out.json)"
    )
    p.add_argument("--out", help="output YAML (default <repo>/artifacts/eyes/ray-noise.yaml)")
    p.set_defaults(func=cmd_noise_model)

    p = es.add_parser("bench", help="estimator latency (AC7)")
    _common(p)
    p.add_argument("--weights", required=True)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--with-gate-detector", action="store_true")
    p.add_argument("--gate-weights", help="default: the preset's gates.learned.weights")
    p.add_argument("--budget-ms", type=float, help="default: 1000 / runner.rate_hz")
    p.add_argument("--out", help="default: the weights' folder")
    p.set_defaults(func=cmd_bench)
