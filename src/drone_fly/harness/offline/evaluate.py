"""Offline evaluation of the vision estimators against track/race-XML ground truth.

Replays a recorded run folder (``frames.csv`` + ``frames/`` + ``telemetry.csv``): each sampled
frame is paired with the nearest telemetry sample (by receive time; both use the runner's
``perf_counter`` clock), and the estimators are re-run **synchronously** on it. Ground truth:

* gates — the XML course with the env's pass rule run over the whole telemetry stream gives the
  true current gate at every instant;
* free space — the contract's own ``raycast()`` on the XML obstacles (and the
  ``obstacle_vision`` block from :mod:`course_senses`).

Reported: gate position error (training and target metres), yaw error (deg), detection rate;
free-space per-ray MAE (all rays and in-FOV only), a near-weighted error (weight
``1 - ground truth``, i.e. obstacles matter more than clear sky), ``obstacle_vision`` error;
the gyro-convention check; and a suggested depth-calibration scale fit. Writes
``eval_report.json`` (default: inside the session folder) and returns an :class:`EvalReport`.

**Capture mode** (:func:`run_capture_eval`, ``drone-fly harness eval --capture --labels``) scores
the learned gate detector per frame against auto-labels (:mod:`.autolabel`), on the held-out
frames recorded in the weights metadata only: per gate type, for labelled gates 4-25 m ahead with
the centre in the image, detection rate, median position error (m and % of range), median yaw
error (180 deg folded), false detections per frame, and CUDA-synchronised inference latency.
Targets (>= 80 % detection, median error <= 15 % of range) are printed, not asserted.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.config import HarnessConfig, load_config
from drone_fly.harness.course import Course
from drone_fly.harness.course_senses import GateProgress, obstacle_vision_block
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.interfaces import BuildContext, FreeSpaceEstimator, GateEstimator
from drone_fly.harness.liftoff.gyro_check import check_gyro_convention
from drone_fly.harness.liftoff.telemetry import (
    DiscontinuityDetector,
    converter_from_config,
    read_telemetry_csv,
)
from drone_fly.harness.offline.track_xml import load_track_course
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import FlightState, Frame


@dataclass
class EvalReport:
    gate: dict[str, Any] = field(default_factory=dict)
    free_space: dict[str, Any] | None = None
    gyro: dict[str, Any] | None = None
    depth_scale_fit: float | None = None
    n_frames: int = 0
    report_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_frames": self.n_frames,
            "gate": self.gate,
            "free_space": self.free_space,
            "gyro": self.gyro,
            "suggested_depth_scale_multiplier": self.depth_scale_fit,
        }

    def summary(self) -> str:
        g = self.gate
        lines = [f"frames evaluated: {self.n_frames}"]
        if g.get("n_matched"):
            lines.append(
                f"gate: detection rate {g['detection_rate']:.1%}, position error mean "
                f"{g['position_error_m_mean']:.3f} m (target scale), yaw error mean "
                f"{g['yaw_error_deg_mean']:.1f} deg"
            )
        else:
            lines.append(f"gate: detection rate {g.get('detection_rate', 0.0):.1%}")
        if self.free_space:
            f = self.free_space
            lines.append(
                f"free space: ray MAE {f['ray_mae']:.3f} (in-FOV {f['ray_mae_in_fov']:.3f}), "
                f"near-weighted {f['ray_near_weighted_error']:.3f}, coverage "
                f"{f['coverage_mean']:.1%}, obstacle_vision MAE {f['obstacle_vision_mae']:.3f}"
            )
        if self.depth_scale_fit is not None:
            lines.append(f"suggested depth calibration scale x{self.depth_scale_fit:.3f}")
        if self.gyro is not None:
            lines.append(f"gyro convention ok: {self.gyro['ok']}")
        if self.report_path:
            lines.append(f"report: {self.report_path}")
        return "\n".join(lines)


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _ray_metre_stats(errors: list[np.ndarray], suffix: str) -> dict[str, float]:
    """Median per-ray error (training m) and the share of rays within 0.5 m."""
    if not errors:
        return {
            f"ray_error_m_median{suffix}": float("nan"),
            f"ray_within_0_5m{suffix}": float("nan"),
        }
    e = np.concatenate([np.asarray(x, np.float64).reshape(-1) for x in errors])
    return {
        f"ray_error_m_median{suffix}": float(np.median(e)),
        f"ray_within_0_5m{suffix}": float(np.mean(e <= 0.5)),
    }


def load_session_frames(session: Path) -> list[tuple[int, float, Path]]:
    """``(frame_index, stamp, image path)`` for every frame saved in a run folder."""
    path = session / "frames.csv"
    if not path.is_file():
        raise HarnessConfigError(f"{session} has no frames.csv (is it a harness run folder?)")
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("file"):
                out.append((int(row["frame_index"]), float(row["stamp"]), session / row["file"]))
    return out


def load_image(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    from drone_fly.harness.errors import lazy_import

    cv2 = lazy_import("cv2", package="opencv-python", purpose="reading recorded frames")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise HarnessConfigError(f"cannot read frame {path}")
    return img


def ground_truth_targets(states: Sequence[FlightState], course: Course) -> list[int]:
    """True current-gate index at every state (env pass rule; resets on discontinuities)."""
    prog = GateProgress(course)
    out = []
    for s in states:
        if s.discontinuity:
            prog.reset()
        prog.advance_if_passed(s.pos)
        out.append(prog.tgt)
    return out


def evaluate_frames(
    frames: Sequence[tuple[Frame, FlightState, int]],
    course: Course,
    contract: Any,
    scale: WorldScale,
    gate_estimator: GateEstimator | None,
    free_space_estimator: FreeSpaceEstimator | None,
) -> EvalReport:
    """Score estimators on ``(frame, state, true gate index)`` triples (synchronous replay)."""
    report = EvalReport(n_frames=len(frames))
    pos_err, yaw_err, matched = [], [], 0
    ray_abs, ray_abs_fov, near_w, cov, ov_abs = [], [], [], [], []
    ray_m, ray_m_fov = [], []  # per-ray absolute error in training metres
    fit_num = fit_den = 0.0
    obstacles = course.raycast_obstacles()
    r_range = float(contract.RAY_RANGE)
    for frame, state, tgt in frames:
        gate_true = course.gates[tgt]
        if gate_estimator is not None:
            est = gate_estimator.update(frame, state, frame.stamp)
            seen = bool(est.info.get("matched", est.status == "tracking"))
            if seen:
                matched += 1
                pos_err.append(float(np.linalg.norm(est.target - gate_true.center)))
                if est.curr_gate is not None:
                    yaw = math.degrees(math.atan2(est.curr_gate[1], est.curr_gate[0]))
                    yaw_err.append(abs(_wrap_deg(yaw - math.degrees(gate_true.yaw))))
        if free_space_estimator is not None:
            fs = free_space_estimator.update(frame, state, gate_true.center, frame.stamp)
            truth = np.asarray(contract.raycast(state.as_state_dict(), obstacles, gate_true.center))
            if fs.raycast is not None:
                err = np.abs(np.asarray(fs.raycast) - truth)
                ray_abs.append(err.mean())
                ray_m.append(err * r_range)
                mask = fs.in_fov_mask if fs.in_fov_mask is not None else np.ones(len(err), bool)
                if mask.any():
                    ray_abs_fov.append(err[mask].mean())
                    ray_m_fov.append(err[mask] * r_range)
                    hit = mask & (truth < 1.0) & (np.asarray(fs.raycast) < 1.0)
                    fit_num += float(np.sum(truth[hit] * fs.raycast[hit]))
                    fit_den += float(np.sum(fs.raycast[hit] ** 2))
                w = 1.0 - truth
                near_w.append(float((w * err).sum() / max(w.sum(), 1e-9)) if w.sum() > 0 else 0.0)
                cov.append(float(mask.mean()))
            if fs.obstacle_vision is not None:
                ov_true = obstacle_vision_block(state.pos, course.obstacles)
                ov_abs.append(float(np.abs(np.asarray(fs.obstacle_vision) - ov_true).mean()))
    n = max(len(frames), 1)
    report.gate = {
        "n_matched": matched,
        "detection_rate": matched / n,
        "position_error_training_m_mean": float(np.mean(pos_err)) if pos_err else None,
        "position_error_m_mean": scale.to_target_length(float(np.mean(pos_err)))
        if pos_err
        else None,
        "position_error_m_median": scale.to_target_length(float(np.median(pos_err)))
        if pos_err
        else None,
        "yaw_error_deg_mean": float(np.mean(yaw_err)) if yaw_err else None,
        "yaw_error_deg_median": float(np.median(yaw_err)) if yaw_err else None,
    }
    if free_space_estimator is not None:
        report.free_space = {
            "ray_mae": float(np.mean(ray_abs)) if ray_abs else float("nan"),
            "ray_mae_in_fov": float(np.mean(ray_abs_fov)) if ray_abs_fov else float("nan"),
            "ray_near_weighted_error": float(np.mean(near_w)) if near_w else float("nan"),
            "coverage_mean": float(np.mean(cov)) if cov else 0.0,
            "obstacle_vision_mae": float(np.mean(ov_abs)) if ov_abs else float("nan"),
            "ray_range_m_training": r_range,
            **_ray_metre_stats(ray_m, ""),
            **_ray_metre_stats(ray_m_fov, "_in_fov"),
        }
        # truth ~= k * estimate on in-FOV hits -> multiply the depth calibration scale by k.
        report.depth_scale_fit = fit_num / fit_den if fit_den > 0 else None
    return report


def run_offline_eval(
    session: str | Path,
    track: str | Path,
    race: str | Path | None,
    config: str | Path | dict[str, Any] | None = None,
    harness: str = "liftoff",
    report_path: str | Path | None = None,
    gpu_prototype_dir: str | None = None,
    *,
    contract: Any = None,
    gate_estimator: GateEstimator | None = None,
    free_space_estimator: FreeSpaceEstimator | None = None,
    depth_model: Any = None,
    evaluate_free_space: bool | None = None,
    preset: str | None = None,
) -> EvalReport:
    """Evaluate a recorded session; see the module docstring. Estimators may be injected."""
    session = Path(session)
    cfg: HarnessConfig = load_config(harness, config, preset=preset)
    scale = WorldScale(cfg.world_scale)
    if contract is None:
        from drone_fly.harness.contract import load_contract

        contract = load_contract(gpu_prototype_dir or cfg.contract.gpu_prototype_dir, None, "cpu")
    off = cfg.section("offline")
    course = load_track_course(
        track,
        race,
        off.get("item_table") or [],
        scale,
        race_id_tag=str(off.get("race_id_tag", "instanceID")),
        laps=off.get("laps"),
        gate_types=cfg.get("gate_types"),
    )

    tele_cfg = cfg.section("telemetry")
    samples = read_telemetry_csv(session / "telemetry.csv")
    if not samples:
        raise HarnessConfigError(f"{session}/telemetry.csv has no samples")
    conv = converter_from_config(tele_cfg, cfg.world_scale)
    det = DiscontinuityDetector(float(tele_cfg.get("max_position_jump_m", 5.0)))
    states = [conv.convert(s, discontinuity=det.check(s)) for s in samples]
    truth_idx = ground_truth_targets(states, course)
    stamps = np.array([s.received_at for s in samples])

    ctx = BuildContext(config=cfg, contract=contract, resources={})
    if depth_model is not None:
        ctx.resources["depth_model"] = depth_model
    if gate_estimator is None:
        from drone_fly.harness.registry import build_part

        gate_estimator = build_part(
            "gate_estimator", cfg.parts.gate_estimator, cfg.section("gates"), ctx
        )
    if evaluate_free_space is None:
        evaluate_free_space = (
            free_space_estimator is not None or cfg.parts.free_space_estimator == "depth"
        )
    if free_space_estimator is None and evaluate_free_space:
        from drone_fly.harness.registry import build_part

        free_space_estimator = build_part(
            "free_space_estimator", "depth", cfg.section("free_space"), ctx
        )

    triples = []
    for idx, stamp, path in load_session_frames(session):
        j = int(np.argmin(np.abs(stamps - stamp)))
        triples.append((Frame(load_image(path), stamp, idx), states[j], truth_idx[j]))
    report = evaluate_frames(
        triples,
        course,
        contract,
        scale,
        gate_estimator,
        free_space_estimator if evaluate_free_space else None,
    )
    report.gyro = check_gyro_convention(samples, conv.gyro_map).as_dict()
    out = Path(report_path) if report_path else session / "eval_report.json"
    out.write_text(json.dumps(report.as_dict(), indent=2, default=float), encoding="utf-8")
    report.report_path = str(out)
    return report


# ----------------------------------------------------------------------------- capture mode

EVAL_RANGE_M = (4.0, 25.0)
TARGET_DETECTION = 0.80
TARGET_REL_ERROR = 0.15
THRESHOLD_NOTE = (
    "Targets are provisional and reported, not asserted: for the UC-64 proof of concept the owner "
    "pre-accepted that a miss does not block (more tracks will be captured later)."
)
SPLIT_CAVEAT = (
    "Held-out frames come from a time split of the training capture(s). With a single capture the "
    "held-out laps revisit viewpoints seen in training, so these scores are optimistic; more "
    "captures (other tracks) give an honest estimate."
)


def _fold_yaw_deg(a: float, b: float) -> float:
    """Yaw error between two gate normals, ignoring the normal's sign (folded to [0, 90])."""
    d = abs(_wrap_deg(a - b))
    return min(d, 180.0 - d)


def score_capture_frame(
    detections: list[Any], gates: list[dict[str, Any]], scale: WorldScale
) -> dict[str, Any]:
    """Match one frame's detections to its labelled gates (Liftoff metres).

    Scored gates: true range in :data:`EVAL_RANGE_M`, centre in the image and outside the masks.
    A gate is detected by a same-class detection within ``max(1 m, 0.25 x range)``, greedily by
    distance. A detection matching no labelled gate (any range) is a false detection.
    """
    det_world = [scale.to_target_vector(d.center).astype(np.float64) for d in detections]
    pairs = []
    for gi, g in enumerate(gates):
        c = np.asarray(g["center_world"], np.float64)
        tol = max(1.0, 0.25 * float(g["range_m"]))
        for di, d in enumerate(detections):
            if d.cls != g["cls"]:
                continue
            dist = float(np.linalg.norm(det_world[di] - c))
            if dist <= tol:
                pairs.append((dist, gi, di))
    pairs.sort()
    used_g: dict[int, int] = {}
    used_d: set[int] = set()
    for _, gi, di in pairs:
        if gi in used_g or di in used_d:
            continue
        used_g[gi] = di
        used_d.add(di)
    scored = []
    for gi, g in enumerate(gates):
        in_range = EVAL_RANGE_M[0] <= g["range_m"] <= EVAL_RANGE_M[1]
        if not (in_range and g["center_in_image"] and not g.get("center_masked")):
            continue
        entry: dict[str, Any] = {
            "gate_index": gi, "cls": g["cls"], "range_m": float(g["range_m"]), "detected": False
        }  # fmt: skip
        if gi in used_g:
            d = detections[used_g[gi]]
            err = float(np.linalg.norm(det_world[used_g[gi]] - np.asarray(g["center_world"])))
            entry.update(
                detected=True,
                position_error_m=err,
                position_error_rel=err / max(float(g["range_m"]), 1e-6),
                yaw_error_deg=_fold_yaw_deg(math.degrees(d.yaw), float(g["normal_yaw_deg"])),
            )
        scored.append(entry)
    false = [detections[i].cls for i in range(len(detections)) if i not in used_d]
    return {"scored": scored, "false_detections": false}


def run_capture_eval(
    capture: str | Path,
    labels: str | Path,
    *,
    config: str | Path | dict[str, Any] | None = None,
    preset: str | None = None,
    harness: str = "liftoff",
    gate_weights: str | Path | None = None,
    device: str | None = None,
    report_path: str | Path | None = None,
    all_frames: bool = False,
    estimator: Any = None,
) -> dict[str, Any]:
    """Score the learned gate detector on a capture's held-out labelled frames (see module)."""
    from drone_fly.harness.config import find_repo_root
    from drone_fly.harness.offline.capture import load_capture
    from drone_fly.harness.offline.gate_train import check_disjoint, is_held_out, load_labels

    overrides: dict[str, Any] = {}
    if gate_weights is not None:
        overrides["gates.learned.weights"] = str(Path(gate_weights).resolve())
    if device is not None:
        overrides["gates.learned.device"] = device
    cfg = load_config(harness, config, overrides, preset=preset)
    scale = WorldScale(cfg.world_scale)
    lab = load_labels(labels)
    if estimator is None:
        from drone_fly.harness.registry import build_part

        ctx = BuildContext(config=cfg, contract=None, resources={})
        estimator = build_part("gate_estimator", "learned", cfg.section("gates"), ctx)
    meta = getattr(estimator, "meta", {}) or {}
    split = (meta.get("splits") or {}).get(lab.name)
    window = None if split is None else split["holdout"]
    if meta.get("purpose") == "deploy" and not all_frames:
        raise HarnessConfigError(
            "these gate weights are a deploy model (trained on every frame, no held-out split); "
            "they cannot be scored. Evaluate weights trained with a held-out split."
        )
    if window is None and not all_frames:
        raise HarnessConfigError(
            f"the gate weights record no train/val/held-out split for capture {lab.name!r}; pass "
            "--all-frames to score every frame (includes training frames)"
        )
    if split is not None and not all_frames:
        check_disjoint(split)  # held-out frames were never trained on nor used for selection
    cap = load_capture(
        capture, cfg.section("telemetry"), cfg.world_scale, float(lab.meta.get("lag_ms", 0.0))
    )
    by_file = {str(f.path.relative_to(cap.root)).replace("\\", "/"): f for f in cap.frames}
    rows = [r for r in lab.rows if all_frames or is_held_out(r["stamp"], window)]
    per_cls: dict[str, dict[str, list]] = {}
    false_per_frame, latencies, n_frames = [], [], 0
    for r in rows:
        f = by_file.get(r["file"])
        if f is None:
            continue
        frame = Frame(load_image(f.path), f.stamp, f.index)
        dets = estimator.detect_frame(frame, f.state)
        if getattr(estimator, "inference_s", None) is not None:
            latencies.append(float(estimator.inference_s))
        res = score_capture_frame(dets, r["gates"], scale)
        n_frames += 1
        false_per_frame.append(len(res["false_detections"]))
        for e in res["scored"]:
            d = per_cls.setdefault(e["cls"], {"det": [], "err": [], "rel": [], "yaw": []})
            d["det"].append(e["detected"])
            if e["detected"]:
                d["err"].append(e["position_error_m"])
                d["rel"].append(e["position_error_rel"])
                d["yaw"].append(e["yaw_error_deg"])
    lat = np.asarray(latencies, np.float64) * 1000.0
    cuda = str(getattr(estimator, "device", "cpu")).startswith("cuda")
    types = {}
    for cls, d in sorted(per_cls.items()):
        n = len(d["det"])
        rate = float(np.mean(d["det"])) if n else 0.0
        med_rel = float(np.median(d["rel"])) if d["rel"] else None
        types[cls] = {
            "n": n,
            "detected": int(np.sum(d["det"])),
            "detection_rate": rate,
            "position_error_m_median": float(np.median(d["err"])) if d["err"] else None,
            "position_error_rel_median": med_rel,
            "yaw_error_deg_median": float(np.median(d["yaw"])) if d["yaw"] else None,
            "meets_detection_target": rate >= TARGET_DETECTION,
            "meets_error_target": med_rel is not None and med_rel <= TARGET_REL_ERROR,
        }
    report = {
        "capture": str(Path(capture).resolve()),
        "labels": str(Path(labels).resolve()),
        "weights": str(getattr(estimator, "weights_path", None)),
        "split": "all frames" if all_frames else window,
        "frames_scored": n_frames,
        "range_m": list(EVAL_RANGE_M),
        "targets": {"detection_rate": TARGET_DETECTION, "position_error_rel": TARGET_REL_ERROR},
        "per_type": types,
        "false_detections_per_frame": float(np.mean(false_per_frame)) if false_per_frame else 0.0,
        "inference_ms": {
            "device": str(getattr(estimator, "device", "cpu")),
            "cuda_synchronised": cuda,
            "representative": cuda,
            "p50": float(np.percentile(lat, 50)) if lat.size else None,
            "p95": float(np.percentile(lat, 95)) if lat.size else None,
            "n": int(lat.size),
        },
        "caveat": SPLIT_CAVEAT,
        "threshold_note": THRESHOLD_NOTE,
    }
    if report_path is None:
        root = find_repo_root()
        report_path = (
            (root if root is not None else Path.cwd())
            / "artifacts"
            / "harness"
            / "liftoff"
            / "reports"
            / "gate-eval"
            / "eval_report.json"
        )
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(out)
    return report


def capture_eval_summary(report: dict[str, Any]) -> str:
    """Human-readable lines for a :func:`run_capture_eval` report (targets are not asserted)."""
    lines = [
        f"frames scored: {report['frames_scored']} (gates {report['range_m'][0]:g}-"
        f"{report['range_m'][1]:g} m ahead, centre in image)"
    ]
    t = report["targets"]
    for cls, d in report["per_type"].items():
        err = d["position_error_m_median"]
        rel = d["position_error_rel_median"]
        yaw = d["yaw_error_deg_median"]
        lines.append(
            f"{cls}: n {d['n']}, detection {d['detection_rate']:.1%} (target "
            f">= {t['detection_rate']:.0%}: "
            f"{'met' if d['meets_detection_target'] else 'NOT met'}), "
            f"median position error {'-' if err is None else f'{err:.2f} m'} "
            f"({'-' if rel is None else f'{rel:.1%}'} of range; target <= "
            f"{t['position_error_rel']:.0%}: {'met' if d['meets_error_target'] else 'NOT met'}), "
            f"median yaw error {'-' if yaw is None else f'{yaw:.1f} deg'}"
        )
    lat = report["inference_ms"]
    note = "" if lat["representative"] else " (CPU: not representative of the GPU budget)"
    if lat["p50"] is not None:
        lines.append(
            f"inference p50 {lat['p50']:.1f} ms / p95 {lat['p95']:.1f} ms on {lat['device']}{note}"
        )
    lines.append(f"false detections per frame: {report['false_detections_per_frame']:.2f}")
    lines.append(report["caveat"])
    lines.append(report["threshold_note"])
    lines.append(f"report: {report.get('report_path')}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- UC71: held-out maps

#: AC7 targets on held-out maps (the single-capture UC64 eval keeps 80 % / 15 %).
HELDOUT_TARGET_DETECTION = 0.70
HELDOUT_TARGET_REL_ERROR = 0.20
JUDGE_MIN_N = 30
COMPARE_MAX_DROP = 0.05
D_U1_RULE = (
    "D-U1 (proposed, pending the user's acceptance): judged per gate type pooled over held-out "
    "maps at 4-25 m; a type is judged only if it was seen in training and has n >= 30; types only "
    "on held-out maps are reported as 'unseen type', types with n < 30 as 'insufficient n'."
)


def trained_capture_names(meta: dict[str, Any]) -> set[str]:
    """Captures the weights trained OR selected on (``captures`` + ``selection_captures``)."""
    names = {str(c.get("name")) for c in meta.get("captures") or []}
    names |= {str(c.get("name")) for c in meta.get("selection_captures") or []}
    return names


def seen_types(meta: dict[str, Any]) -> set[str]:
    """Gate types present in the training data (``trained_types``; v1: its classes)."""
    if meta.get("trained_types") is not None:
        return {k for k, n in meta["trained_types"].items() if n}
    return set(meta.get("classes") or [])


def _latency(lat_s: list[float], device: str) -> dict[str, Any]:
    lat = np.asarray(lat_s, np.float64) * 1000.0
    cuda = str(device).startswith("cuda")
    return {
        "device": str(device),
        "cuda_synchronised": cuda,
        "flag": None if cuda else "measured on the CPU: not representative of the live budget",
        "p50": float(np.percentile(lat, 50)) if lat.size else None,
        "p95": float(np.percentile(lat, 95)) if lat.size else None,
        "n": int(lat.size),
    }


def run_heldout_eval(
    detector: str | Path,
    *,
    captures: list[str | Path] | None = None,
    split: str = "held-out",
    compare: str | Path | None = None,
    protected_capture: str | Path | None = None,
    config: str | Path | dict[str, Any] | None = None,
    preset: str | None = "liftoff",
    device: str | None = None,
    split_source: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """``harness eval --detector <weights|dir> --split held-out [--captures] [--compare <v1>]``.

    Scores every frame of the ``split``-map captures (from the catalogue; default every
    discovered capture on such a map) and refuses any capture the weights list as trained or
    selected on. Per map x type and per type pooled over maps, with range bins and the ADR 0003
    metrics; types are judged under :data:`D_U1_RULE`. ``compare`` scores both detectors on the
    protected Infinity Loop window (``protected_capture``, default the v1 reference capture) and
    passes when the new arch detection at 4-25 m is >= the other's - 5 pp.
    """
    from drone_fly.harness.config import find_repo_root
    from drone_fly.harness.offline import gate_diagnose as gdg
    from drone_fly.harness.offline.capture_index import (
        iter_capture_dirs,
        link_entry,
        read_meta,
        resolve_captures,
    )
    from drone_fly.tracks import catalogue as cat

    if split not in ("held-out", "val"):
        raise HarnessConfigError(f"--split must be held-out or val, got {split!r}")
    catalogue = cat.load_catalogue(split_source) if split_source else cat.load_named("liftoff")
    wpath = gdg.resolve_detector(detector)
    est, cfg = gdg.build_estimator(wpath, config=config, preset=preset, device=device)
    meta = est.meta or {}
    trained = trained_capture_names(meta)
    dirs = resolve_captures([str(c) for c in captures]) if captures else iter_capture_dirs()
    chosen = []
    for d in dirs:
        m = read_meta(d)
        if link_entry(m, catalogue)["split"] != split:
            if captures:
                raise HarnessConfigError(f"capture {d.name} is not on a {split} map ({m['map']})")
            continue
        if d.name in trained:
            raise HarnessConfigError(
                f"the weights trained or selected on capture {d.name}: it cannot be scored"
            )
        chosen.append((d, m))
    if not chosen:
        raise HarnessConfigError(f"no labelled capture on a {split} map to score")
    by_map: dict[str, list] = {}
    false, lat, frames, per_capture = [], [], 0, {}
    for d, m in chosen:
        cam_cfg, pairs, _ = gdg.capture_frames(d, cfg)
        res = gdg.score_capture(est, cam_cfg, pairs)
        by_map.setdefault(m["map"], []).extend(res["records"])
        false += res["false"]
        lat += res["latency_s"]
        frames += res["frames"]
        per_capture[d.name] = {"map": m["map"], "frames": res["frames"]}
    pooled = [r for rs in by_map.values() for r in rs]
    per_type = gdg.by_class(pooled)
    seen = seen_types(meta)
    judged = {}
    for cls, bins in per_type.items():
        s = bins[gdg.POOLED]
        if cls not in seen:
            judged[cls] = {"stratum": "unseen type", "judged": False}
        elif s["n"] < JUDGE_MIN_N:
            judged[cls] = {"stratum": "insufficient n", "judged": False}
        else:
            ok_det = s["detection_rate"] >= HELDOUT_TARGET_DETECTION
            rel = s["position_error_rel_median"]
            ok_err = rel is not None and rel <= HELDOUT_TARGET_REL_ERROR
            judged[cls] = {
                "stratum": "judged",
                "judged": True,
                "meets_detection_target": ok_det,
                "meets_error_target": ok_err,
                "pass": ok_det and ok_err,
            }
    report: dict[str, Any] = {
        "weights": str(wpath),
        "split": split,
        "captures": per_capture,
        "frames_scored": frames,
        "range_bins_m": [list(b) for b in gdg.RANGE_BINS],
        "targets": {
            "detection_rate": HELDOUT_TARGET_DETECTION,
            "position_error_rel": HELDOUT_TARGET_REL_ERROR,
            "judge_min_n": JUDGE_MIN_N,
        },
        "rule": D_U1_RULE,
        "per_map_type": {mp: gdg.by_class(rs) for mp, rs in sorted(by_map.items())},
        "per_type": per_type,
        "judgement": judged,
        "pass": bool(judged)
        and all(j["pass"] for j in judged.values() if j["judged"])
        and any(j["judged"] for j in judged.values()),
        "false_detections_per_frame": float(np.mean(false)) if false else 0.0,
        "inference_ms": _latency(lat, getattr(est, "device", "cpu")),
    }
    if compare is not None:
        report["compare"] = compare_protected(
            wpath, compare, protected_capture, config=config, preset=preset, device=device
        )
    if report_path is None:
        root = find_repo_root()
        report_path = (
            (root if root is not None else Path.cwd())
            / "artifacts"
            / "harness"
            / "liftoff"
            / "reports"
            / "gate-eval"
            / f"heldout_{wpath.stem}.json"
        )
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(out)
    return report


def compare_protected(
    new: str | Path,
    old: str | Path,
    protected_capture: str | Path | None = None,
    *,
    config: Any = None,
    preset: str | None = "liftoff",
    device: str | None = None,
    cls: str = "inflatable_arch",
) -> dict[str, Any]:
    """Both detectors on ``old``'s protected window of the Infinity Loop reference capture."""
    from drone_fly.harness.offline import gate_diagnose as gdg
    from drone_fly.harness.offline.capture_index import captures_root
    from drone_fly.harness.offline.detector_noise import heldout_window

    cap = (
        Path(protected_capture)
        if protected_capture
        else (captures_root() / "2026-10-11-infinity-manual")
    )
    old_w = gdg.resolve_detector(old)
    out: dict[str, Any] = {"capture": cap.name, "class": cls}
    window = None
    for name, path in (("old", old_w), ("new", gdg.resolve_detector(new))):
        est, cfg = gdg.build_estimator(path, config=config, preset=preset, device=device)
        if name == "old":
            window = heldout_window(est.meta, cap.name)
            if window is None:
                raise HarnessConfigError(f"{old_w} records no held-out window for {cap.name}")
        else:
            w_new = heldout_window(est.meta, cap.name)  # raises if the new weights trained on it
            if w_new is not None and w_new != window:
                raise HarnessConfigError("the new weights hold out a different window")
        cam_cfg, pairs, _ = gdg.capture_frames(cap, cfg, window)
        res = gdg.score_capture(est, cam_cfg, pairs)
        out[name] = {
            "weights": str(path),
            "frames": res["frames"],
            "bins": gdg.binned([r for r in res["records"] if r["cls"] == cls]),
        }
    out["window"] = window
    a = out["new"]["bins"][gdg.POOLED]["detection_rate"] or 0.0
    b = out["old"]["bins"][gdg.POOLED]["detection_rate"] or 0.0
    out["delta_pp"] = 100.0 * (a - b)
    out["pass"] = a >= b - COMPARE_MAX_DROP
    out["near_bin_tracked"] = {
        "new": out["new"]["bins"]["4-8"]["detection_rate"],
        "old": out["old"]["bins"]["4-8"]["detection_rate"],
    }
    return out


def heldout_eval_summary(rep: dict[str, Any]) -> str:
    lines = [f"{rep['split']} eval of {rep['weights']}: {rep['frames_scored']} frames"]
    for cls, bins in rep["per_type"].items():
        s = bins["4-25"]
        j = rep["judgement"][cls]
        rate = "-" if s["detection_rate"] is None else f"{100 * s['detection_rate']:.1f} %"
        rel = s["position_error_rel_median"]
        lines.append(
            f"  {cls:28s} n {s['n']:5d}  det {rate:>7s}  rel err "
            f"{'-' if rel is None else f'{100 * rel:.1f} %'}  [{j['stratum']}]"
            + (f" {'PASS' if j['pass'] else 'FAIL'}" if j["judged"] else "")
        )
    lines.append(f"  false detections / frame {rep['false_detections_per_frame']:.2f}")
    if "compare" in rep:
        c = rep["compare"]
        lines.append(
            f"  Infinity Loop window: new vs old {c['delta_pp']:+.1f} pp -> "
            f"{'PASS' if c['pass'] else 'FAIL'} (4-8 m: {c['near_bin_tracked']})"
        )
    lines.append("overall: " + ("PASS" if rep["pass"] else "FAIL / not judged"))
    return "\n".join(lines)
