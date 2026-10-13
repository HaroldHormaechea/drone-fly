"""Per-gate scoring records and miss attribution for the learned gate detector (UC71 AC7).

One engine serves the multi-map evaluation (``harness eval --split held-out``) and
``harness gates diagnose``: :func:`frame_records` scores one labelled frame and returns, per scored
gate (4-25 m, centre in the image and unmasked, as in :func:`~.evaluate.score_capture_frame`):

* ``detected`` + position / relative / yaw errors (the eval matching rule, world frame);
* the ADR 0003 root-cause metrics on **matched-by-centre** detections: the raw network detection of
  the same class whose image centre lies within :func:`centre_radius_px` of the gate's labelled
  centre, with at least ``min_keypoints`` usable keypoints. ``range_err_rel`` is the PnP range minus
  the true range over the true range (signed: + = placed too far), ``spread_ratio`` the RMS spread
  of the predicted keypoints about their centroid over the labelled ones (same indices, visible in
  both; v1: 0.72);
* for a miss, its cause (:data:`CAUSES`), first that applies:

  - ``no_centre_peak`` - no raw detection centred near the gate;
  - ``class_confusion`` - the nearest one has another class;
  - ``few_keypoints`` - same class but fewer than ``min_keypoints`` usable keypoints;
  - ``pnp_range`` - enough keypoints, but the PnP pose misses the eval tolerance (see the
    signed range error and spread ratio);
  - ``occluded`` - the label flags the gate as occluded by the scene (only when a scene cache
    produced that flag; reporting only, never counted as a detector miss in the targets).

Range bins: 4-8, 8-15, 15-25 m, plus pooled 4-25 m.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError

RANGE_BINS = ((4.0, 8.0), (8.0, 15.0), (15.0, 25.0))
POOLED = "4-25"
CAUSES = ("no_centre_peak", "class_confusion", "few_keypoints", "pnp_range", "occluded")
MIN_CENTRE_RADIUS_PX = 12.0


def range_bin(r: float) -> str | None:
    for lo, hi in RANGE_BINS:
        if lo <= r <= hi:
            return f"{lo:g}-{hi:g}"
    return None


def centre_radius_px(gate: dict[str, Any]) -> float:
    """Centre-match radius: a quarter of the labelled keypoints' extent, at least 12 px."""
    kp = np.asarray(gate["keypoints"], np.float64)
    front = np.asarray(gate.get("in_front", [True] * len(kp)), bool)
    kp = kp[front] if front.any() else kp
    ext = float(np.linalg.norm(kp.max(axis=0) - kp.min(axis=0))) if len(kp) else 0.0
    return max(MIN_CENTRE_RADIUS_PX, 0.25 * ext)


def _spread(points: np.ndarray) -> float:
    if len(points) < 2:
        return float("nan")
    c = points.mean(axis=0)
    return float(np.sqrt(((points - c) ** 2).sum(axis=1).mean()))


def _usable(raw: dict[str, Any], k: int, w: int, h: int, masks: Any) -> np.ndarray:
    kp = raw["keypoints"][:k]
    keep = (
        (raw["visibility"][:k] >= 0.5)
        & (kp[:, 0] >= 0)
        & (kp[:, 0] <= w - 1)
        & (kp[:, 1] >= 0)
        & (kp[:, 1] <= h - 1)
    )
    if masks:
        keep &= ~masks.contains(kp, w, h)
    return keep


def frame_records(
    est: Any, image: np.ndarray, stamp: float, index: int, state: Any, row: dict[str, Any]
) -> tuple[list[dict[str, Any]], int, float | None]:
    """Score one labelled frame: ``(per-gate records, false detections, inference s)``.

    ``est`` is a synchronous :class:`~drone_fly.harness.liftoff.gate_learned.LearnedGateEstimator`
    at world scale 1 (Liftoff metres); its raw detections are read back from its frame cache.
    """
    from drone_fly.harness.offline.evaluate import score_capture_frame
    from drone_fly.harness.scale import WorldScale
    from drone_fly.harness.types import Frame

    if getattr(est, "async_inference", False):
        raise HarnessConfigError("scoring needs a synchronous estimator (async_inference: false)")
    dets = est.detect_frame(Frame(image, stamp, index), state)
    raw = list(getattr(est, "_cache", []) or [])
    res = score_capture_frame(dets, row["gates"], WorldScale(1.0))
    h, w = image.shape[:2]
    by_centre = {tuple(np.round(d.center_px, 3)): d for d in dets if d.pixel_center is not None}
    out = []
    for e in res["scored"]:
        g = row["gates"][e["gate_index"]]
        rng = float(g["range_m"])
        rec = {
            **e,
            "bin": range_bin(rng),
            "range_err_rel": None,
            "spread_ratio": None,
            "cause": None,
        }
        c = np.asarray(g["center_px"], np.float64)
        rad = centre_radius_px(g)
        near = []
        for r in raw:
            dist = float(np.linalg.norm(np.asarray(r["center"]) - c))
            if dist <= rad:
                near.append((dist, r))
        near.sort(key=lambda t: t[0])
        same = [r for _, r in near if est.table.types[r["cls"]].cls == g["cls"]]
        k = len(g["keypoints"])
        matched = None
        for r in same:
            keep = _usable(r, k, w, h, est.masks)
            if int(keep.sum()) >= est.min_keypoints:
                matched = (r, keep)
                break
        if matched is not None:
            r, keep = matched
            det = by_centre.get(tuple(np.round(np.asarray(r["center"], np.float64), 3)))
            if det is not None:
                rec["range_err_rel"] = (float(np.linalg.norm(det.t_cam)) - rng) / max(rng, 1e-6)
            vis = np.asarray(g["visible"], bool)[:k] & keep
            if int(vis.sum()) >= 2:
                pred = r["keypoints"][:k][vis]
                true = np.asarray(g["keypoints"], np.float64)[:k][vis]
                st = _spread(true)
                if st > 1e-6:
                    rec["spread_ratio"] = _spread(pred) / st
        if not e["detected"]:
            if g.get("occluded"):
                rec["cause"] = "occluded"
            elif not near:
                rec["cause"] = "no_centre_peak"
            elif not same:
                rec["cause"] = "class_confusion"
            elif matched is None:
                rec["cause"] = "few_keypoints"
            else:
                rec["cause"] = "pnp_range"
        out.append(rec)
    return out, len(res["false_detections"]), getattr(est, "inference_s", None)


def _median(v: list[float]) -> float | None:
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Detection rate and the error medians of a set of gate records."""
    n = len(records)
    hits = [r for r in records if r["detected"]]
    misses = Counter(r["cause"] for r in records if not r["detected"])
    return {
        "n": n,
        "detected": len(hits),
        "detection_rate": len(hits) / n if n else None,
        "position_error_m_median": _median([r["position_error_m"] for r in hits]),
        "position_error_rel_median": _median([r["position_error_rel"] for r in hits]),
        "yaw_error_deg_median": _median([r["yaw_error_deg"] for r in hits]),
        "range_err_rel_median": _median([r["range_err_rel"] for r in records]),
        "spread_ratio_median": _median([r["spread_ratio"] for r in records]),
        "misses": {c: misses.get(c, 0) for c in CAUSES},
    }


def binned(records: list[dict[str, Any]]) -> dict[str, Any]:
    """:func:`summarise` per range bin plus the pooled 4-25 m entry."""
    out = {POOLED: summarise(records)}
    for lo, hi in RANGE_BINS:
        key = f"{lo:g}-{hi:g}"
        out[key] = summarise([r for r in records if r["bin"] == key])
    return out


def by_class(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[r["cls"]].append(r)
    return {cls: binned(rs) for cls, rs in sorted(groups.items())}


# ----------------------------------------------------------------------------- estimators + data


def resolve_detector(path: str | Path) -> Path:
    """``--detector``: a weights file, or a ``models/<detector-id>/`` folder (its eval weights)."""
    p = Path(path)
    if p.is_dir():
        cands = sorted(q for q in p.glob("*.pt") if not q.stem.endswith("-deploy"))
        if len(cands) != 1:
            raise HarnessConfigError(
                f"{p} must hold exactly one non-deploy weights file, found "
                f"{[q.name for q in cands]}"
            )
        return cands[0]
    from drone_fly.harness.liftoff.gate_learned import resolve_weights_path

    return resolve_weights_path(p)


def build_estimator(
    weights: str | Path,
    *,
    config: Any = None,
    preset: str | None = "liftoff",
    device: str | None = None,
) -> Any:
    """A synchronous learned estimator at world scale 1 for ``weights``."""
    from drone_fly.harness.config import load_config
    from drone_fly.harness.interfaces import BuildContext
    from drone_fly.harness.registry import build_part

    over = {
        "gates.learned.weights": str(Path(weights).resolve()),
        "gates.learned.async_inference": False,
        "scaling.world_scale": 1.0,
    }
    if device is not None:
        over["gates.learned.device"] = device
    cfg = load_config("liftoff", config, over, preset=preset)
    est = build_part(
        "gate_estimator",
        "learned",
        cfg.section("gates"),
        BuildContext(config=cfg, contract=None, resources={}),
    )
    return est, cfg


def capture_frames(
    capture_dir: Path, cfg: Any, window: dict[str, Any] | None = None
) -> tuple[Any, list[tuple[dict[str, Any], Any]], dict[str, Any]]:
    """``(camera config, [(label row, capture frame)], labels meta)`` of a labelled capture.

    ``window`` limits the rows to a held-out stamp window. The camera comes from ``meta.json``.
    """
    from drone_fly.harness.offline.capture import load_capture
    from drone_fly.harness.offline.capture_index import META_FILE, camera_config, read_meta
    from drone_fly.harness.offline.gate_dataset import labels_dir_for
    from drone_fly.harness.offline.gate_train import is_held_out, load_labels

    meta = read_meta(capture_dir) if (capture_dir / META_FILE).is_file() else None
    lab_dir = labels_dir_for(capture_dir, meta)
    if lab_dir is None:
        raise HarnessConfigError(f"{capture_dir} has no labels (run `harness label`)")
    labels = load_labels(lab_dir)
    cap = load_capture(
        capture_dir, cfg.section("telemetry"), 1.0, float(labels.meta.get("lag_ms", 0.0))
    )
    by_file = {str(f.path.relative_to(cap.root)).replace("\\", "/"): f for f in cap.frames}
    pairs = []
    for r in labels.rows:
        if window is not None and not is_held_out(r["stamp"], window):
            continue
        f = by_file.get(r["file"])
        if f is not None:
            pairs.append((r, f))
    return camera_config(cfg.section("camera"), meta), pairs, labels.meta


def score_capture(est: Any, cam_cfg: dict[str, Any], pairs: list) -> dict[str, Any]:
    """Every gate record, false detections per frame and latencies of one capture."""
    from drone_fly.harness.liftoff.gate_cv import camera_from_config
    from drone_fly.harness.liftoff.masks import FrameMasks
    from drone_fly.harness.offline.evaluate import load_image

    est.camera = camera_from_config(cam_cfg)
    est.masks = FrameMasks.from_config(cam_cfg)
    records, false, lat = [], [], []
    for row, f in pairs:
        recs, n_false, inf = frame_records(est, load_image(f.path), f.stamp, f.index, f.state, row)
        records.extend(recs)
        false.append(n_false)
        if inf is not None:
            lat.append(float(inf))
    return {"records": records, "false": false, "latency_s": lat, "frames": len(pairs)}


def diagnose(
    weights: str | Path,
    captures: list[str | Path],
    *,
    config: Any = None,
    preset: str | None = "liftoff",
    device: str | None = None,
    all_frames: bool = False,
) -> dict[str, Any]:
    """``harness gates diagnose``: per class x range bin, misses by cause + ADR 0003 metrics.

    A capture the weights trained on is scored on its held-out window only (none -> refused,
    unless ``all_frames``); a capture they never saw, on every frame.
    """
    from drone_fly.harness.offline.detector_noise import NoiseLeakageError, heldout_window

    wpath = resolve_detector(weights)
    est, cfg = build_estimator(wpath, config=config, preset=preset, device=device)
    per_capture, records = {}, []
    for c in captures:
        d = Path(c)
        window = None
        if not all_frames:
            try:
                window = heldout_window(est.meta, d.name)
            except NoiseLeakageError as exc:
                raise HarnessConfigError(str(exc)) from exc
        cam_cfg, pairs, _ = capture_frames(d, cfg, window)
        res = score_capture(est, cam_cfg, pairs)
        records.extend(res["records"])
        per_capture[d.name] = {
            "frames": res["frames"],
            "window": window,
            "classes": by_class(res["records"]),
        }
    return {
        "weights": str(wpath),
        "captures": per_capture,
        "classes": by_class(records),
        "causes": list(CAUSES),
    }


def diagnose_table(rep: dict[str, Any]) -> str:
    """Markdown table: class x bin -> n, detection, misses by cause, range bias, spread ratio."""
    head = (
        "| class | range m | n | detected | "
        + " | ".join(CAUSES)
        + " | range err (signed, median) | spread ratio (median) |"
    )
    lines = [head, "|" + "---|" * (6 + len(CAUSES))]
    for cls, bins in rep["classes"].items():
        for b in ("4-8", "8-15", "15-25", POOLED):
            s = bins[b]
            if not s["n"]:
                continue
            rate = f"{100 * s['detection_rate']:.1f} %"
            causes = " | ".join(str(s["misses"][c]) for c in CAUSES)
            rb = s["range_err_rel_median"]
            sr = s["spread_ratio_median"]
            lines.append(
                f"| {cls} | {b} | {s['n']} | {rate} | {causes} | "
                f"{'-' if rb is None else f'{100 * rb:+.1f} %'} | "
                f"{'-' if sr is None else f'{sr:.2f}'} |"
            )
    return "\n".join(lines)
