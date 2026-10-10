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
                mask = fs.in_fov_mask if fs.in_fov_mask is not None else np.ones(len(err), bool)
                if mask.any():
                    ray_abs_fov.append(err[mask].mean())
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
) -> EvalReport:
    """Evaluate a recorded session; see the module docstring. Estimators may be injected."""
    session = Path(session)
    cfg: HarnessConfig = load_config(harness, config)
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
