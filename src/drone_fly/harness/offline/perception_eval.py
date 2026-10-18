"""Offline evaluation of any obstacle-vision provider on an eyes dataset (UC-86 AC8).

``drone-fly perception eval --provider <name|module:Class> --dataset <eyes dataset>`` builds the
provider like a live session would (``build_part("free_space_estimator", ...)``, synchronous mode:
no frame source, so it processes each frame handed to ``update()``), feeds it the dataset's
full-resolution capture frames + paired telemetry in time order, and scores its ``raycast``
against the eyes labels with the UC-78 metrics (:func:`drone_fly.eyes.evaluate.evaluate_arrays`)
plus ``update()`` latency on the current device. The report has the ``eyes eval`` shape, so
``drone-fly eyes compare`` compares two providers (e.g. ``depth`` vs ``eyes``) like-for-like.

Frames: every scored row gets a warm-up window of the preceding frames of its segment
(``warmup_s``, enough history for the eyes net); the provider is reset at each capture start and
whenever the fed frames are not contiguous (a new segment or a stride gap). A scored row whose
estimate is not valid yet (stamp ``-inf`` / not the row's frame) is counted in
``skipped_frames`` and left out of the metrics.

The gate target each frame is fed is the label route's passage centre (track geometry, the same
target the labels used). This is offline-only: a live session stays screen-only.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError

EVAL_SPLITS = ("val", "held-out")
DEFAULT_WARMUP_S = 0.6


def _slug(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", ref.split(":")[-1] if ":" in ref else ref).strip("-")


def default_out_dir(provider: str) -> Path:
    from drone_fly import paths

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return paths.data_root() / "perception-eval" / f"{_slug(provider)}-{stamp}"


def build_provider(ref: str, cfg: Any, contract: Any) -> Any:
    """The provider ``ref`` from ``cfg`` (free_space section + part_options), synchronous mode."""
    from drone_fly.harness.interfaces import BuildContext
    from drone_fly.harness.perception import check_spec
    from drone_fly.harness.registry import build_part

    section = dict(cfg.get("free_space") or {})
    section.update((cfg.get("part_options") or {}).get("free_space_estimator") or {})
    ctx = BuildContext(config=cfg, contract=contract, resources={}, purpose="eval")
    provider = build_part("free_space_estimator", ref, section, ctx)
    check_spec(provider, "free_space_estimator", ref)
    if "raycast" not in set(getattr(provider, "provides", ()) or ()):
        raise HarnessConfigError(
            f"perception eval scores raycast; provider {ref!r} provides "
            f"{sorted(getattr(provider, 'provides', ()) or ())}"
        )
    return provider


def label_course_target_fn(world_scale: float, **course_kw: Any) -> Callable[..., np.ndarray]:
    """Default target: the label route's passage centre, in the state frame (training scale)."""
    from drone_fly.eyes.labels import load_label_course

    cache: dict[str, Any] = {}

    def target(entry: dict[str, Any], labels: dict[str, np.ndarray], row: int, state: Any):
        cid = entry["catalogue_id"]
        if cid not in cache:
            course, _, _ = load_label_course(cid, world_scale, **course_kw)
            cache[cid] = (course, course.frame.origin / world_scale)
        course, origin_c = cache[cid]
        c = course.passages[int(labels["target_index"][row])].center
        return (np.asarray(c, np.float64) + origin_c).astype(np.float32)

    return target


def _check_leakage(provider: Any, man: dict[str, Any], split: str) -> None:
    meta = getattr(provider, "meta", None)
    if not isinstance(meta, dict):
        return
    trained = set((meta.get("split_maps") or {}).get("train") or [])
    overlap = trained & set(man["split_maps"].get(split) or [])
    if overlap:
        raise HarnessConfigError(
            f"perception eval refuses maps the provider trained on: {sorted(overlap)}"
        )


def _latency_ms(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "mean": None}
    ms = np.asarray(values, np.float64) * 1000.0
    return {
        "n": int(ms.size),
        "p50": float(np.percentile(ms, 50)),
        "p95": float(np.percentile(ms, 95)),
        "p99": float(np.percentile(ms, 99)),
        "mean": float(ms.mean()),
    }


def _pick_rows(n: int, max_frames: int | None) -> np.ndarray:
    if max_frames is None or max_frames >= n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, int(max_frames)).round().astype(np.int64))


def run_perception_eval(
    provider: str,
    dataset: str | Path,
    *,
    config: str | Path | dict[str, Any] | None = None,
    preset: str | None = "liftoff",
    harness: str = "liftoff",
    split: str = "held-out",
    device: str | None = None,
    max_frames: int | None = None,
    out: str | Path | None = None,
    tol_m: float | None = None,
    target_fn: Callable[..., np.ndarray] | None = None,
    gpu_prototype_dir: str | None = None,
    warmup_s: float = DEFAULT_WARMUP_S,
    frame_loader: Callable[[Path], np.ndarray] | None = None,
    trackdata_root: str | Path | None = None,
    scenes_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the evaluation and write ``perception_eval_<split>.json`` / ``.md``; see the module doc.

    ``target_fn(entry, labels, row, state)`` and ``frame_loader(path)`` are injectable (tests).
    """
    from drone_fly.eyes.dataset import SAMPLES_FILE, SPLITS, load_manifest
    from drone_fly.eyes.evaluate import eval_markdown, evaluate_arrays
    from drone_fly.eyes.fov import offline_contract
    from drone_fly.eyes.labels import LABELS_FILE, read_labels_meta
    from drone_fly.harness.config import load_config
    from drone_fly.harness.offline.capture import load_capture, load_frame_image
    from drone_fly.harness.perception import describe
    from drone_fly.harness.types import Frame

    if split == "train":
        raise HarnessConfigError(
            "perception eval refuses the train split (evaluate val or held-out)"
        )
    if split not in EVAL_SPLITS:
        raise HarnessConfigError(f"--split must be val or held-out, got {split!r}")
    man = load_manifest(dataset)
    ls = man.get("label_settings") or {}
    W = float(ls.get("world_scale", 1.0))
    rt = float(ls.get("ray_range", 4.0)) * float(ls.get("ray_range_scale", 1.0))
    tol = float(tol_m if tol_m is not None else 0.5)
    overrides: dict[str, Any] = {}
    cfg = load_config(harness, config, None, preset=preset)
    if device is not None:
        for key in ("eyes", "depth"):
            if isinstance(cfg.get(key), dict) and "device" in cfg.get(key):
                overrides[f"{key}.device"] = device
        cfg = load_config(harness, config, overrides, preset=preset)
    contract = offline_contract(gpu_prototype_dir)
    prov = build_provider(provider, cfg, contract)
    _check_leakage(prov, man, split)
    target_fn = target_fn or label_course_target_fn(
        W, trackdata_root=trackdata_root, scenes_root=scenes_root
    )
    load_image = frame_loader or load_frame_image

    with np.load(Path(dataset) / SAMPLES_FILE, allow_pickle=False) as z:
        sel = np.flatnonzero(z["split"] == SPLITS.index(split))
        tr = z["split"] == SPLITS.index("train")
        train_mean = z["free"][tr].mean(axis=0) if tr.any() else None
        data = {k: z[k][sel] for k in ("capture", "row", "free", "in_view", "hit_min_half_m",
                                       "warning_s")}  # fmt: skip
    order = np.lexsort((data["row"], data["capture"]))
    data = {k: v[order] for k, v in data.items()}
    picked = _pick_rows(len(data["row"]), max_frames)
    caps = man["captures"]
    preds: list[np.ndarray] = []
    scored: list[int] = []
    latency: list[float] = []
    skipped = 0
    telemetry = cfg.section("telemetry")
    prov.start()
    try:
        for ci in np.unique(data["capture"][picked]):
            entry = caps[int(ci)]
            rows_here = picked[data["capture"][picked] == ci]
            labels_dir = Path(entry["labels_dir"])
            lmeta = read_labels_meta(labels_dir) or {}
            with np.load(labels_dir / LABELS_FILE, allow_pickle=False) as lz:
                labels = {k: lz[k] for k in ("frame_file", "stamp", "segment", "target_index",
                                              "frame_index")}  # fmt: skip
            cap = load_capture(entry["capture"], telemetry, W, float(lmeta.get("lag_ms", 0.0)))
            root = Path(entry["capture"])
            by_file = {f.path.relative_to(root).as_posix(): f for f in cap.frames}
            stamps, segs = labels["stamp"], labels["segment"]
            prov.reset()
            last_fed: int | None = None
            for k in rows_here:
                r = int(data["row"][k])
                start = r
                while (
                    start > 0
                    and segs[start - 1] == segs[r]
                    and (stamps[r] - stamps[start - 1] <= warmup_s)
                ):
                    start -= 1
                if last_fed is not None and segs[last_fed] == segs[r] and last_fed >= start - 1:
                    start = last_fed + 1
                elif last_fed is not None:
                    prov.reset()
                est = None
                for j in range(start, r + 1):
                    cf = by_file.get(str(labels["frame_file"][j]))
                    if cf is None:
                        continue
                    frame = Frame(
                        load_image(cf.path), float(stamps[j]), int(labels["frame_index"][j])
                    )
                    target = target_fn(entry, labels, j, cf.state)
                    t0 = time.perf_counter()
                    e = prov.update(frame, cf.state, target, float(stamps[j]))
                    if j == r:
                        latency.append(time.perf_counter() - t0)
                        est = e
                    last_fed = j
                if (
                    est is None
                    or not math.isfinite(float(est.stamp))
                    or (float(est.stamp) != float(stamps[r]))
                ):
                    skipped += 1
                    continue
                preds.append(np.asarray(est.raycast, np.float32).reshape(-1))
                scored.append(int(k))
    finally:
        prov.stop()
    idx = np.asarray(scored, np.int64)
    maps = np.array([caps[int(c)].get("map", "") for c in data["capture"][idx]])
    if len(idx):
        body = evaluate_arrays(
            np.stack(preds), None, data["free"][idx], data["in_view"][idx],
            range_train=rt, world_scale=W, tol_m=tol,
            hit_min_half_m=data["hit_min_half_m"][idx], warning_s=data["warning_s"][idx],
            maps=maps, train_mean=train_mean,
        )  # fmt: skip
    else:
        body = {"frames": 0, "note": f"no scorable frames in the {split} split"}
    rep = {
        "format": 1,
        "provider": provider,
        "provider_describe": describe(prov),
        "device": device or "config",
        "dataset": man["id"],
        "split": split,
        "maps": man["split_maps"].get(split) or [],
        "tol_m": tol,
        "range_train_m": rt,
        "range_liftoff_m": rt * W,
        "world_scale": W,
        "targets_note": "perception eval: a measured provider baseline on intact eyes labels",
        "split_rows": int(len(data["row"])),
        "picked_rows": int(len(picked)),
        "skipped_frames": int(skipped),
        "latency_ms": _latency_ms(latency),
        **body,
    }
    out_dir = Path(out) if out is not None else default_out_dir(provider)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"perception_eval_{split}.json").write_text(json.dumps(rep, indent=2), "utf-8")
    (out_dir / f"perception_eval_{split}.md").write_text(perception_markdown(rep, eval_markdown),
                                                         "utf-8")  # fmt: skip
    rep["out_dir"] = str(out_dir)
    return rep


def perception_markdown(rep: dict[str, Any], eval_markdown: Callable[[dict], str]) -> str:
    lat = rep["latency_ms"]

    def f(v: Any) -> str:
        return "-" if v is None else f"{v:.1f}"

    head = eval_markdown(rep).replace(
        "# Eyes accuracy:", f"# Perception eval `{rep['provider']}`:", 1
    )
    return "\n".join([
        head.rstrip(),
        "",
        "## Latency (update wall time, this device)",
        "",
        "| n | p50 ms | p95 ms | p99 ms | mean ms |",
        "|---:|---:|---:|---:|---:|",
        f"| {lat['n']} | {f(lat['p50'])} | {f(lat['p95'])} | {f(lat['p99'])} | {f(lat['mean'])} |",
        "",
        f"Rows: {rep['picked_rows']} of {rep['split_rows']} picked, {rep['skipped_frames']} "
        "skipped (no valid estimate yet).",
        "",
    ])  # fmt: skip
