"""Gate back-projection evaluation on recorded live runs (UC-91): gate-centre error vs range.

``drone-fly harness gates backprojection-eval --runs <run dir>... --track <xml>`` measures how far
the learned estimator's world gate centres land from the track's true gate positions, binned by
true range, for each ``gates.learned.range_mode`` (:mod:`~drone_fly.harness.liftoff.gate_range`):

* **Truth.** The catalogue course of the track XML (:func:`~.detector_noise.load_catalogue_course`)
  gives the route passages; their centres in the contract frame (Liftoff metres) are
  ``frame.to_contract_point(centre)``, de-duplicated (a circuit repeats passages). A detection is
  matched to the passage whose bearing from the drone is closest to the estimate's
  (:func:`match_by_bearing`); it is **ambiguous** (left out of the error statistics, counted per
  bin) when another distinct passage lies within the tolerance or within ``ambiguity_margin_deg``
  of the best, and **unmatched** when the best is beyond the tolerance (a false positive or a wild
  PnP bearing). Errors (estimate - truth, metres): world ``z``, the line-of-sight radial /
  tangential-horizontal / tangential-vertical components (``sim.noise_model.los_frame``) and 3-D.
  Range bins: ``detector_noise.RANGE_EDGES`` plus 40-60 m and > 60 m.
* **Pass A** (primary; no GPU). Every logged detection with its full keypoint set
  (``ticks.jsonl`` ``gate.info.detections_px``) is re-solved with
  :func:`~drone_fly.harness.liftoff.gate_cv.solve_planar_pnp` against the class's object points
  and placed in the world with the telemetry pose (``telemetry.csv``: linear position, slerped
  attitude) at the detection's source-frame stamp + ``gates.pose_offset_s`` — the live estimator's
  rule. The source frame is the tick's ``frame_index - detection_lag_frames``; its stamp is the
  ``gate.stamp`` of a tick whose newest frame it was. Detections are de-duplicated by
  ``(source frame, order)``; stamps outside the telemetry span are skipped and counted; partial
  detections (fewer logged points than the class has: keypoint identity unknown) are counted as
  not re-solvable. Each mode keeps or drops every record; a size-range column (evaluator only)
  compares :func:`~drone_fly.harness.liftoff.gate_range.size_range` with the truth.
* **Pass B** (secondary). A replay :class:`~drone_fly.harness.liftoff.gate_cv.TrackingGateEstimator`
  is driven tick by tick with the logged state, stamps and the run's tracker config, its
  ``detect_frame`` returning the Pass-A detections of that tick placed through the tracker's own
  pose history (``_pose_at``, so ``pose_offset_s`` means what it means live). Both modes see
  identical inputs; partial detections are missing from both. **Fidelity**: the replayed ``pnp``
  target against the logged ``gate.target`` (share within 0.5 m, median deviation); low fidelity
  marks Pass B indicative only.
* **Pass C** (optional ``--with-frames``). The detector re-runs on the saved frames (every
  ``recording.frame_every_n`` ticks), placed with the frame's own stamp; thin bins are flagged.
* **Logged target.** The logged ``gate.target`` (while a gate is held) against the truth: the live
  gate error the policy saw.
* **Tune / held-out split.** ``--tune-runs`` picks the ``range_check`` knobs on those runs only
  (:func:`tune_range_check`); ``--runs`` are reported with them. A run in both sets is an error.
  :func:`rule_verdict` applies the pre-registered AC4 rule to the held-out Pass A.

**OneDrive hygiene.** Run folders may sit in a synced cloud folder, where listing or walking a
tree hydrates files and pops notifications. This tool takes explicit run folders and opens only
``config.yaml``, ``ticks.jsonl``, ``telemetry.csv`` and ``frames.csv`` in each — plus, with
``--with-frames``, the ``frames/<file>`` paths ``frames.csv`` lists. It never globs, rglobs,
lists or walks a run folder.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.geometry import CameraModel, quat_to_matrix, unity_to_contract_position
from drone_fly.harness.liftoff.gate_range import (
    RANGE_MODES,
    REJECT_REASONS,
    RangeCheck,
    check_detection,
    evaluate_detection,
    hull_area,
)

#: Range-bin edges (m): the detector-noise edges plus 40-60 m and beyond 60 m.
RANGE_BIN_EDGES = (0.0, 4.0, 8.0, 12.0, 18.0, 25.0, 40.0, 60.0, math.inf)
MATCH_TOL_DEG = 5.0
AMBIGUITY_MARGIN_DEG = 3.0
FIDELITY_TOL_M = 0.5
FIDELITY_MIN_SHARE = 0.8  # below this the Pass B comparison is indicative only
THIN_BIN_N = 15
FAR_M = 40.0
# The pre-registered AC4 rule (held-out runs, Pass A; plan 91 section 5).
RULE_MIN_N = 15
RULE_NOISE_DROP = 0.30
RULE_RETENTION = 0.90
RULE_MEDIAN_Z_M = 2.0
#: Physics starting points of ``range_check`` (the :class:`RangeCheck` defaults; see its module).
PHYSICS_START = RangeCheck()
#: ``range_check`` grid searched on the tune runs.
TUNE_GRID = {
    "size_ratio_min": (0.35, 0.40, 0.47, 0.55, 0.62, 0.70),
    "size_ratio_max": (1.15, 1.25, 1.33, 1.5, 1.75, 2.0),
    "max_normal_tilt_deg": (15.0, 20.0, 30.0, 45.0, None),
}


# ----------------------------------------------------------------------------- run I/O


@dataclass
class RunData:
    path: Path
    name: str
    cfg: dict[str, Any]
    ticks: list[dict[str, Any]]
    tel_t: np.ndarray  # (N,) received_at
    tel_pos: np.ndarray  # (N, 3) contract metres
    tel_quat: np.ndarray  # (N, 4) contract [w, x, y, z]
    frames: list[dict[str, Any]] = field(default_factory=list)

    @property
    def world_scale(self) -> float:
        return float((self.cfg.get("scaling") or {}).get("world_scale", 1.0))


def _telemetry(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = csv.DictReader(fh)
        need = ("received_at", "pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w")
        for row in rows:
            for k in need:
                cols.setdefault(k, []).append(float(row[k]))
    if not cols:
        raise HarnessConfigError(f"{path} holds no telemetry rows")
    t = np.asarray(cols["received_at"], np.float64)
    pu = np.stack([cols["pos_x"], cols["pos_y"], cols["pos_z"]], axis=1)
    pos = np.array([unity_to_contract_position(p) for p in pu], np.float64)
    # unity_to_contract_quat, vectorised: vector part through the frame change, negated
    m = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]], np.float64)
    vec = -(np.stack([cols["quat_x"], cols["quat_y"], cols["quat_z"]], axis=1) @ m.T)
    q = np.concatenate([np.asarray(cols["quat_w"])[:, None], vec], axis=1)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    order = np.argsort(t, kind="stable")
    return t[order], pos[order], q[order]


def load_run(path: str | Path) -> RunData:
    """Read one recorded run: the four files only (see the module docstring)."""
    import yaml

    d = Path(path)
    files = {k: d / k for k in ("config.yaml", "ticks.jsonl", "telemetry.csv", "frames.csv")}
    for k in ("config.yaml", "ticks.jsonl", "telemetry.csv"):
        if not files[k].is_file():
            raise HarnessConfigError(f"run folder {d} has no {k}")
    cfg = yaml.safe_load(files["config.yaml"].read_text(encoding="utf-8")) or {}
    ticks = [
        json.loads(line)
        for line in files["ticks.jsonl"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    t, pos, quat = _telemetry(files["telemetry.csv"])
    frames: list[dict[str, Any]] = []
    if files["frames.csv"].is_file():
        with files["frames.csv"].open(newline="", encoding="utf-8") as fh:
            frames = [dict(r) for r in csv.DictReader(fh)]
    return RunData(d, d.name, cfg, ticks, t, pos, quat, frames)


class TelemetryPose:
    """Pose at time ``t`` from the telemetry (linear position, slerp attitude); ``None`` outside."""

    def __init__(self, t: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> None:
        self.t, self.pos, self.quat = t, pos, quat

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray] | None:
        from drone_fly.harness.liftoff.gate_cv import _slerp

        if len(self.t) < 2 or t < self.t[0] or t > self.t[-1]:
            return None
        j = int(np.searchsorted(self.t, t, side="right"))
        j = min(max(j, 1), len(self.t) - 1)
        t0, t1 = self.t[j - 1], self.t[j]
        w = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        p = self.pos[j - 1] + w * (self.pos[j] - self.pos[j - 1])
        return p, _slerp(self.quat[j - 1], self.quat[j], w)


# ----------------------------------------------------------------------------- truth


@dataclass(frozen=True)
class Truth:
    centers: np.ndarray  # (G, 3) contract metres
    normals: np.ndarray  # (G, 3)
    item_ids: tuple[str, ...]
    track: str


def truth_passages(track_xml: str | Path, trackdata_root: str | Path | None = None) -> Truth:
    """De-duplicated route passages of the track's catalogue course, contract metres."""
    from drone_fly.harness.offline.detector_noise import load_catalogue_course

    course = load_catalogue_course(str(track_xml), trackdata_root)
    centers, normals, ids = [], [], []
    for p in course.passages:
        c = np.asarray(course.frame.to_contract_point(p.center), np.float64)
        if any(np.linalg.norm(c - q) < 0.1 for q in centers):
            continue
        centers.append(c)
        normals.append(np.asarray(p.normal, np.float64))
        ids.append(str(p.item_id))
    return Truth(np.array(centers), np.array(normals), tuple(ids), Path(str(track_xml)).name)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    c = np.sum(a * b, axis=-1) / np.maximum(na * nb, 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def match_by_bearing(
    pos: np.ndarray,
    est: np.ndarray,
    centers: np.ndarray,
    tol_deg: float = MATCH_TOL_DEG,
    margin_deg: float = AMBIGUITY_MARGIN_DEG,
) -> tuple[int | None, float, bool]:
    """``(best, angle_deg, ambiguous)`` for an estimate seen from ``pos``.

    ``best`` is ``None`` when no passage lies within ``tol_deg`` of the estimate's bearing
    (unmatched). ``ambiguous`` when another passage is within ``tol_deg`` or within
    ``margin_deg`` of the best angle.
    """
    pos = np.asarray(pos, np.float64)
    ang = _angle_deg(np.asarray(est, np.float64) - pos, np.asarray(centers, np.float64) - pos)
    order = np.argsort(ang)
    best = int(order[0])
    a0 = float(ang[best])
    if a0 > tol_deg:
        return None, a0, False
    ambiguous = len(order) > 1 and (
        float(ang[order[1]]) <= tol_deg or float(ang[order[1]]) <= a0 + margin_deg
    )
    return best, a0, bool(ambiguous)


def error_components(pos: np.ndarray, est: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Estimate - truth in metres: world z, LOS radial / tangential (h, v), 3-D; true range.

    ``elev_err_deg`` / ``azim_err_deg`` are the tangential errors as angles seen from the drone
    (``atan(tangential / range)``): a camera uptilt error shows up as a constant ``elev_err_deg``
    at every range, an hfov (focal) error as a radial error proportional to range.
    """
    from drone_fly.sim.noise_model import los_frame

    pos, est, truth = (np.asarray(v, np.float64) for v in (pos, est, truth))
    err = est - truth
    rel = truth - pos
    rad, th, tv = los_frame(rel) @ err
    return {
        "range_m": float(np.linalg.norm(rel)),
        "z_m": float(err[2]),
        "radial_m": float(rad),
        "tan_h_m": float(th),
        "tan_v_m": float(tv),
        "err3d_m": float(np.linalg.norm(err)),
        "elev_err_deg": float(np.degrees(np.arctan2(tv, np.linalg.norm(rel)))),
        "azim_err_deg": float(np.degrees(np.arctan2(th, np.linalg.norm(rel)))),
    }


def bin_label(r: float, edges: tuple[float, ...] = RANGE_BIN_EDGES) -> str:
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if lo <= r < hi:
            return f">{lo:g}" if math.isinf(hi) else f"{lo:g}-{hi:g}"
    return f">{edges[-2]:g}"


BIN_LABELS = tuple(
    bin_label(0.5 * (a + b) if math.isfinite(b) else a + 1.0)
    for a, b in zip(RANGE_BIN_EDGES[:-1], RANGE_BIN_EDGES[1:], strict=True)
)


# ----------------------------------------------------------------------------- camera + table


def run_camera(cfg: dict[str, Any], width: int, height: int) -> CameraModel:
    from drone_fly.harness.liftoff.gate_cv import camera_from_config

    cam = camera_from_config(cfg.get("camera") or {})
    return cam if (cam.width, cam.height) == (width, height) else cam.scaled_to(width, height)


def run_table(cfg: dict[str, Any]) -> Any:
    from drone_fly.harness.gate_types import GateTypeTable, resolve_gate_types

    return GateTypeTable.from_config(resolve_gate_types(cfg.get("gate_types")))


def run_masks(cfg: dict[str, Any]) -> Any:
    from drone_fly.harness.liftoff.masks import FrameMasks

    return FrameMasks.from_config(cfg.get("camera") or {})


def pose_offset(cfg: dict[str, Any]) -> float:
    from drone_fly.harness.liftoff.gate_cv import DEFAULT_POSE_OFFSET_S

    return float((cfg.get("gates") or {}).get("pose_offset_s", DEFAULT_POSE_OFFSET_S))


# ----------------------------------------------------------------------------- Pass A


def frame_stamps(ticks: list[dict[str, Any]]) -> dict[int, float]:
    """Frame index -> capture stamp, from every tick whose newest frame it was."""
    out: dict[int, float] = {}
    for t in ticks:
        g = t.get("gate") or {}
        fi = (g.get("info") or {}).get("frame_index")
        if fi is not None and g.get("stamp") is not None:
            out.setdefault(int(fi), float(g["stamp"]))
    return out


def source_frame(info: dict[str, Any]) -> int | None:
    fi = info.get("frame_index")
    if fi is None:
        return None
    return int(fi) - int(info.get("detection_lag_frames") or 0)


def logged_detections(run: RunData) -> tuple[list[dict[str, Any]], Counter]:
    """Unique logged detections ``(source frame, order)`` with their source stamp."""
    stamps = frame_stamps(run.ticks)
    seen: set[tuple[int, int]] = set()
    out: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for t in run.ticks:
        info = ((t.get("gate") or {}).get("info")) or {}
        dets = info.get("detections_px") or []
        si = source_frame(info)
        if not dets or si is None:
            continue
        for k, d in enumerate(dets):
            key = (si, k)
            if key in seen:
                continue
            seen.add(key)
            counts["unique"] += 1
            if si not in stamps:
                counts["no_source_stamp"] += 1
                continue
            out.append(
                {
                    "src_index": si,
                    "order": k,
                    "src_stamp": stamps[si],
                    "tick": int(t["tick"]),
                    "cls": d.get("cls"),
                    "points": np.asarray(d["points"], np.float64).reshape(-1, 2),
                    "pixel_center": d.get("center"),
                    "frame_size": tuple(int(v) for v in info.get("frame_size") or (0, 0)),
                }
            )
    return out, counts


def solve_detection(
    points: np.ndarray,
    obj: np.ndarray,
    full_obj: np.ndarray,
    camera: CameraModel,
    pose: tuple[np.ndarray, np.ndarray],
    check: RangeCheck,
    masks: Any,
) -> dict[str, Any] | None:
    """PnP + world placement (metres) + the range-check inputs; ``None`` if unsolvable."""
    from drone_fly.harness.errors import lazy_import
    from drone_fly.harness.liftoff.gate_cv import solve_planar_pnp

    cv2 = lazy_import("cv2", package="opencv-python", purpose="PnP re-solve")
    try:
        t_cam, n_cam = solve_planar_pnp(obj, points, camera)
    except (ValueError, cv2.error):
        return None
    if not (np.isfinite(t_cam).all() and np.isfinite(n_cam).all()) or t_cam[2] <= 0:
        return None
    pos, quat = pose
    r_wc = quat_to_matrix(quat) @ camera.body_to_camera().T
    center = pos + r_wc @ t_cam
    n_world = r_wc @ n_cam
    v = evaluate_detection(t_cam, n_world, points, obj, full_obj, camera, check, masks)
    return {
        "t_cam": t_cam,
        "n_cam": n_cam,
        "n_world": n_world,
        "pos": pos,
        "est": center,
        "r_pnp": v["r_pnp"],
        "r_size": v["r_size"],
        "r_size_raw": v["r_size_raw"],
        "size_invalid": v["size_invalid"],
        "tilt_deg": v["tilt_deg"],
    }


def annotate(rec: dict[str, Any], truth: Truth) -> dict[str, Any]:
    """Match + error components (in place)."""
    best, ang, amb = match_by_bearing(rec["pos"], rec["est"], truth.centers)
    rec["match"], rec["match_angle_deg"], rec["ambiguous"] = best, ang, amb
    if best is None:
        rec["category"] = "unmatched"
        rec["bin"] = bin_label(rec["r_pnp"])
        return rec
    rec["category"] = "ambiguous" if amb else "matched"
    rec.update(error_components(rec["pos"], rec["est"], truth.centers[best]))
    rec["bin"] = bin_label(rec["range_m"])
    if rec.get("r_size_raw") is not None and np.isfinite(rec["r_size_raw"]):
        rec["size_err_m"] = float(rec["r_size_raw"] - rec["range_m"])
        rec["pnp_range_err_m"] = float(rec["r_pnp"] - rec["range_m"])
    return rec


def pass_a(run: RunData, truth: Truth, check: RangeCheck) -> tuple[list[dict[str, Any]], dict]:
    """Re-solve every logged full-keypoint detection (see the module docstring)."""
    table = run_table(run.cfg)
    masks = run_masks(run.cfg)
    poses = TelemetryPose(run.tel_t, run.tel_pos, run.tel_quat)
    off = pose_offset(run.cfg)
    dets, counts = logged_detections(run)
    recs = []
    for d in dets:
        gtype = table[d["cls"]]
        if len(d["points"]) != gtype.n_keypoints:
            counts["partial_not_resolvable"] += 1
            continue
        pose = poses.at(d["src_stamp"] + off)
        if pose is None:
            counts["outside_telemetry"] += 1
            continue
        w, h = d["frame_size"]
        camera = run_camera(run.cfg, w, h)
        obj = gtype.object_points()
        s = solve_detection(d["points"], obj, obj, camera, pose, check, masks)
        if s is None:
            counts["pnp_failed"] += 1
            continue
        counts["resolved"] += 1
        rec = {
            "run": run.name,
            "pass": "A",
            "src_index": d["src_index"],
            "order": d["order"],
            "stamp": d["src_stamp"],
            "cls": d["cls"],
            "n_points": len(d["points"]),
            **s,
        }
        recs.append(annotate(rec, truth))
    return recs, dict(counts)


def accepted(rec: dict[str, Any], mode: str, check: RangeCheck) -> tuple[bool, str | None]:
    """Does ``mode`` keep this record (``pnp`` keeps all)?"""
    if mode == "pnp":
        return True, None
    return check_detection(rec["r_pnp"], rec["r_size"], rec["n_world"], check)


# ----------------------------------------------------------------------------- Pass B


def _replay_class() -> type:
    from drone_fly.harness.liftoff.gate_cv import TrackingGateEstimator

    class ReplayTracker(TrackingGateEstimator):
        """The live tracker fed with pre-solved detections per tick (Pass B)."""

        def __init__(
            self, camera: CameraModel, scale: Any, *, mode: str, check: RangeCheck, **tracker: Any
        ) -> None:
            super().__init__(camera, scale, **tracker)
            self.mode = mode
            self.check = check
            self.tick_dets: list[dict[str, Any]] = []
            self.tick_stamp: float | None = None

        def detect(self, image: np.ndarray, state: Any) -> list[Any]:  # pragma: no cover
            return []

        def detect_frame(self, frame: Any, state: Any) -> list[Any]:
            self._det_stamp = self.tick_stamp
            out = []
            for d in self.tick_dets:
                det = self.detection_from_pose(
                    d["t_cam"],
                    d["n_cam"],
                    d["points"],
                    d["area_frac"],
                    state,
                    d["camera"],
                    d["aperture_m"],
                    cls=d["cls"],
                    pixel_center=d["pixel_center"],
                )
                if det is None:
                    continue
                if self.mode == "pnp_checked":
                    ok, _ = check_detection(d["r_pnp"], d["r_size"], det.n_world, self.check)
                    if not ok:
                        continue
                out.append(det)
            return out

    return ReplayTracker


def tracker_kwargs(cfg: dict[str, Any], table: Any, scale: Any) -> dict[str, Any]:
    from drone_fly.harness.liftoff.gate_cv import DEFAULT_POSE_OFFSET_S

    g = cfg.get("gates") or {}
    return {
        "default_aperture": scale.length(table.types[0].min_aperture),
        "assoc_radius": float(g.get("assoc_radius", 1.0)),
        "lost_timeout_s": float(g.get("lost_timeout_s", 1.0)),
        "acquire_timeout_s": float(g.get("acquire_timeout_s", 3.0)),
        "pass_area_frac": float(g.get("pass_area_frac", 0.2)),
        "gates_per_lap": None if g.get("gates_per_lap") is None else int(g["gates_per_lap"]),
        "lost_policy": str(g.get("lost_policy", "terminate")),
        "lost_hard_stop_s": float(g.get("lost_hard_stop_s", 6.0)),
        "drop_after_s": float(g.get("drop_after_s", 0.5)),
        "pose_offset_s": float(g.get("pose_offset_s", DEFAULT_POSE_OFFSET_S)),
    }


def tick_inputs(run: RunData, check: RangeCheck) -> list[dict[str, Any]]:
    """Per tick: state, ``now``, source stamp and the camera-frame solved full detections."""
    from drone_fly.harness.types import FlightState

    table = run_table(run.cfg)
    masks = run_masks(run.cfg)
    stamps = frame_stamps(run.ticks)
    cache: dict[tuple[int, int], dict[str, Any] | None] = {}
    eye = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))  # PnP + checks are pose-free here
    out = []
    for t in run.ticks:
        st = t.get("state")
        g = t.get("gate")
        if st is None or g is None:
            continue
        info = g.get("info") or {}
        ages = t.get("ages_s") or {}
        sim = float(t.get("sim_time") or 0.0)
        state = FlightState(
            pos=np.asarray(st["pos"], np.float64),
            vel=np.asarray(st["vel"], np.float64),
            quat=np.asarray(st["quat"], np.float64),
            omega=np.asarray(st["omega"], np.float64),
            received_at=sim - float(ages.get("telemetry", 0.0)),
        )
        fi = info.get("frame_index")
        has_frame = fi is not None
        now = float(g["stamp"]) + float(ages.get("frame", 0.0)) if has_frame else sim
        si = source_frame(info)
        dets = []
        for k, d in enumerate(info.get("detections_px") or []):
            key = (int(si), k) if si is not None else (-1, k)
            if key not in cache:
                cache[key] = None
                gtype = table[d["cls"]]
                pts = np.asarray(d["points"], np.float64).reshape(-1, 2)
                w, h = (int(v) for v in info.get("frame_size") or (0, 0))
                if len(pts) == gtype.n_keypoints and w and h:
                    cam = run_camera(run.cfg, w, h)
                    obj = gtype.object_points()
                    s = solve_detection(pts, obj, obj, cam, eye, check, masks)
                    if s is not None:
                        cache[key] = {
                            "t_cam": s["t_cam"],
                            "n_cam": s["n_cam"],
                            "points": pts,
                            "area_frac": hull_area(pts) / float(w * h),
                            "camera": cam,
                            "aperture_m": gtype.min_aperture,
                            "cls": gtype.cls,
                            "pixel_center": d.get("center"),
                            "r_pnp": s["r_pnp"],
                            "r_size": s["r_size"],
                        }
            if cache[key] is not None:
                dets.append(cache[key])
        out.append(
            {
                "tick": int(t["tick"]),
                "state": state,
                "now": now,
                "has_frame": has_frame,
                "frame_size": tuple(int(v) for v in info.get("frame_size") or (1, 1)),
                "src_index": si,
                "src_stamp": stamps.get(si) if si is not None else None,
                "dets": dets,
                "reset": "discontinuity:estimators_reset" in (t.get("events") or []),
                "logged_target": np.asarray(g["target"], np.float64),
                "logged_valid": bool(info.get("target_valid", g.get("status") == "tracking")),
            }
        )
    return out


def pass_b(
    run: RunData, inputs: list[dict[str, Any]], truth: Truth, mode: str, check: RangeCheck
) -> list[dict[str, Any]]:
    """Replay the tracker over ``inputs`` in ``mode``; per-tick target records (metres)."""
    from drone_fly.harness.scale import WorldScale
    from drone_fly.harness.types import Frame

    table = run_table(run.cfg)
    scale = WorldScale(run.world_scale)
    first = next((x for x in inputs if x["has_frame"]), None)
    w, h = first["frame_size"] if first else (2560, 1080)
    cam = run_camera(run.cfg, w, h)
    est = _replay_class()(
        cam, scale, mode=mode, check=check, **tracker_kwargs(run.cfg, table, scale)
    )
    ws = run.world_scale
    out = []
    for x in inputs:
        if x["reset"]:
            est.reset()
        est.tick_dets = x["dets"]
        est.tick_stamp = x["src_stamp"]
        frame = None
        if x["has_frame"]:
            fw, fh = x["frame_size"]
            img = np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (fh, fw, 3))
            frame = Frame(
                image=img, stamp=float(x["src_stamp"] or x["now"]), index=int(x["src_index"] or 0)
            )
        ge = est.update(frame, x["state"], x["now"])
        valid = bool(ge.info.get("target_valid"))
        rec = {
            "run": run.name,
            "pass": "B",
            "mode": mode,
            "tick": x["tick"],
            "valid": valid,
            "target": np.asarray(ge.target, np.float64) * ws,
            "logged_target": x["logged_target"] * ws,
            "logged_valid": x["logged_valid"],
            "pos": np.asarray(x["state"].pos, np.float64) * ws,
        }
        out.append(rec)
    return out


def target_records(
    rows: list[dict[str, Any]], truth: Truth, key: str, valid_key: str
) -> list[dict[str, Any]]:
    """Per-tick held targets vs truth (``matched`` / ``ambiguous`` / ``unmatched``)."""
    out = []
    for r in rows:
        if not r[valid_key]:
            continue
        est = r[key]
        best, ang, amb = match_by_bearing(r["pos"], est, truth.centers)
        rec = {
            "run": r["run"],
            "tick": r["tick"],
            "pos": r["pos"],
            "est": est,
            "match": best,
            "ambiguous": amb,
            "match_angle_deg": ang,
        }
        if best is None:
            rec["category"] = "unmatched"
            rec["bin"] = bin_label(float(np.linalg.norm(est - r["pos"])))
        else:
            rec["category"] = "ambiguous" if amb else "matched"
            rec.update(error_components(r["pos"], est, truth.centers[best]))
            rec["bin"] = bin_label(rec["range_m"])
        out.append(rec)
    return out


def fidelity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Replayed ``pnp`` target vs the logged target (metres)."""
    both = [r for r in rows if r["valid"] and r["logged_valid"]]
    agree = sum(r["valid"] == r["logged_valid"] for r in rows)
    dev = np.array([float(np.linalg.norm(r["target"] - r["logged_target"])) for r in both])
    share = float(np.mean(dev <= FIDELITY_TOL_M)) if len(dev) else 0.0
    return {
        "ticks": len(rows),
        "held_both": len(both),
        "held_agreement": agree / max(len(rows), 1),
        "share_within_m": share,
        "tol_m": FIDELITY_TOL_M,
        "median_dev_m": float(np.median(dev)) if len(dev) else None,
        "p90_dev_m": float(np.percentile(dev, 90)) if len(dev) else None,
        "indicative_only": share < FIDELITY_MIN_SHARE,
    }


# ----------------------------------------------------------------------------- Pass C


def pass_c(
    run: RunData, truth: Truth, check: RangeCheck, device: str | None = None
) -> tuple[list[dict[str, Any]], dict]:
    """Re-run the detector on the run's saved frames (``frames.csv`` paths only)."""
    from drone_fly.harness.offline.evaluate import load_image
    from drone_fly.harness.offline.gate_diagnose import build_estimator

    lc = (run.cfg.get("gates") or {}).get("learned") or {}
    from drone_fly.harness.liftoff.gate_learned import configured_weights, resolve_weights_path

    weights = resolve_weights_path(configured_weights(lc), fetch=False)
    est, _ = build_estimator(
        weights, config=str(run.path / "config.yaml"), preset=None, device=device
    )
    table = est.table
    masks = run_masks(run.cfg)
    poses = TelemetryPose(run.tel_t, run.tel_pos, run.tel_quat)
    off = pose_offset(run.cfg)
    counts: Counter = Counter()
    recs = []
    for row in run.frames:
        path = run.path / row["file"]
        if not path.is_file():
            counts["missing_frame"] += 1
            continue
        image = load_image(path)
        h, w = image.shape[:2]
        stamp = float(row["stamp"])
        pose = poses.at(stamp + off)
        if pose is None:
            counts["outside_telemetry"] += 1
            continue
        counts["frames"] += 1
        camera = run_camera(run.cfg, w, h)
        raw = est.infer(masks.apply(image) if masks else image)
        for k, d in enumerate(raw):
            gtype = table.types[d["cls"]]
            n = gtype.n_keypoints
            kp = d["keypoints"][:n]
            keep = (
                (d["visibility"][:n] >= 0.5)
                & (kp[:, 0] >= 0)
                & (kp[:, 0] <= w - 1)
                & (kp[:, 1] >= 0)
                & (kp[:, 1] <= h - 1)
            )
            if masks:
                keep &= ~masks.contains(kp, w, h)
            if int(keep.sum()) < est.min_keypoints:
                counts["few_keypoints"] += 1
                continue
            if masks and masks.contains(np.asarray(d["center"])[None], w, h)[0]:
                counts["centre_masked"] += 1
                continue
            obj = gtype.object_points()
            s = solve_detection(kp[keep], obj[keep], obj, camera, pose, check, masks)
            if s is None:
                counts["pnp_failed"] += 1
                continue
            counts["solved"] += 1
            rec = {
                "run": run.name,
                "pass": "C",
                "src_index": int(row["tick"]),
                "order": k,
                "stamp": stamp,
                "cls": gtype.cls,
                "n_points": int(keep.sum()),
                **s,
            }
            recs.append(annotate(rec, truth))
    return recs, dict(counts)


# ----------------------------------------------------------------------------- statistics


def _err_stats(recs: list[dict[str, Any]]) -> dict[str, Any]:
    if not recs:
        return {"n": 0}
    z = np.array([r["z_m"] for r in recs])
    rad = np.array([r["radial_m"] for r in recs])
    th = np.array([r["tan_h_m"] for r in recs])
    tv = np.array([r["tan_v_m"] for r in recs])
    e3 = np.array([r["err3d_m"] for r in recs])
    el = np.array([r["elev_err_deg"] for r in recs])
    out = {
        "n": len(recs),
        "z_median": float(np.median(z)),
        "z_mean": float(np.mean(z)),
        "z_sd": float(np.std(z, ddof=1)) if len(z) > 1 else 0.0,
        "z_p90_abs": float(np.percentile(np.abs(z), 90)),
        "radial_median": float(np.median(rad)),
        "radial_sd": float(np.std(rad, ddof=1)) if len(rad) > 1 else 0.0,
        "tan_h_median": float(np.median(th)),
        "tan_h_sd": float(np.std(th, ddof=1)) if len(th) > 1 else 0.0,
        "tan_v_median": float(np.median(tv)),
        "err3d_median": float(np.median(e3)),
        "err3d_p90": float(np.percentile(e3, 90)),
        "elev_err_deg_median": float(np.median(el)),
        "radial_rel_median": float(np.median(rad / np.array([r["range_m"] for r in recs]))),
    }
    se = [r["size_err_m"] for r in recs if r.get("size_err_m") is not None]
    if se:
        pe = [r["pnp_range_err_m"] for r in recs if r.get("size_err_m") is not None]
        out["size_range_err_median"] = float(np.median(se))
        out["size_range_err_sd"] = float(np.std(se, ddof=1)) if len(se) > 1 else 0.0
        out["pnp_range_err_median"] = float(np.median(pe))
        out["pnp_range_err_sd"] = float(np.std(pe, ddof=1)) if len(pe) > 1 else 0.0
    return out


def binned_stats(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Error statistics of matched (unambiguous) records per range bin, plus ``<40`` / ``>40``."""
    m = [r for r in recs if r["category"] == "matched"]
    out = {b: _err_stats([r for r in m if r["bin"] == b]) for b in BIN_LABELS}
    out["<40"] = _err_stats([r for r in m if r["range_m"] < FAR_M])
    out[">40"] = _err_stats([r for r in m if r["range_m"] >= FAR_M])
    for k, v in out.items():
        v["thin"] = v["n"] < THIN_BIN_N
        if k in BIN_LABELS:
            v["ambiguous"] = sum(1 for r in recs if r["category"] == "ambiguous" and r["bin"] == k)
    return out


def mode_records(recs: list[dict[str, Any]], mode: str, check: RangeCheck) -> list[dict]:
    return [r for r in recs if accepted(r, mode, check)[0]]


def retention(recs: list[dict[str, Any]], mode: str, check: RangeCheck) -> dict[str, Any]:
    """Per bin: kept / rejected per category, and rejections per reason."""
    out: dict[str, Any] = {}
    for b in (*BIN_LABELS, "<40", ">40", "all"):
        sel = [r for r in recs if _in_bin(r, b)]
        row: dict[str, Any] = {}
        reasons: Counter = Counter()
        for cat in ("matched", "ambiguous", "unmatched"):
            kept = rej = 0
            for r in sel:
                if r["category"] != cat:
                    continue
                ok, why = accepted(r, mode, check)
                kept += ok
                rej += not ok
                if why:
                    reasons[why] += 1
            row[cat] = {
                "kept": kept,
                "rejected": rej,
                "kept_share": kept / (kept + rej) if kept + rej else None,
            }
        row["reasons"] = {k: reasons.get(k, 0) for k in REJECT_REASONS}
        out[b] = row
    return out


def _in_bin(r: dict[str, Any], b: str) -> bool:
    if b == "all":
        return True
    if b in ("<40", ">40"):
        rng = r.get("range_m", r.get("r_pnp", 0.0))
        return (rng < FAR_M) == (b == "<40")
    return r["bin"] == b


def _drop(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before <= 0:
        return None
    return 1.0 - after / before


def rule_verdict(recs: list[dict[str, Any]], check: RangeCheck) -> dict[str, Any]:
    """The pre-registered AC4 rule on Pass-A records (see :data:`RULE_NOISE_DROP` & co.).

    Beyond 40 m (matched, unambiguous; pooled): ``pnp_checked`` lowers both the p90 of ``|z|`` and
    the ``z`` sd by at least 30 % vs ``pnp``, keeping at least 90 % of matched detections in every
    bin beyond 40 m with ``n >= 15``. Within 40 m: ``|median z| <= 2 m`` and retention >= 90 % in
    every bin with ``n >= 15``.
    """
    base = binned_stats(mode_records(recs, "pnp", check))
    new = binned_stats(mode_records(recs, "pnp_checked", check))
    ret = retention(recs, "pnp_checked", check)
    reasons = []
    far_b, far_n = base[">40"], new[">40"]
    p90_drop = _drop(far_b.get("z_p90_abs"), far_n.get("z_p90_abs"))
    sd_drop = _drop(far_b.get("z_sd"), far_n.get("z_sd"))
    if far_b["n"] < RULE_MIN_N:
        reasons.append(f"beyond 40 m: only {far_b['n']} matched detections (< {RULE_MIN_N})")
    if p90_drop is None or p90_drop < RULE_NOISE_DROP:
        reasons.append(f"beyond 40 m: p90 |z| drop {_pct(p90_drop)} < {RULE_NOISE_DROP:.0%}")
    if sd_drop is None or sd_drop < RULE_NOISE_DROP:
        reasons.append(f"beyond 40 m: z sd drop {_pct(sd_drop)} < {RULE_NOISE_DROP:.0%}")
    bins = []
    for b in BIN_LABELS:
        n = base[b]["n"]
        share = ret[b]["matched"]["kept_share"]
        lo = _bin_lo(b)
        row = {"bin": b, "n": n, "kept_share": share, "median_z_new": new[b].get("z_median")}
        if n >= RULE_MIN_N:
            if share is not None and share < RULE_RETENTION:
                reasons.append(f"{b} m: retention {share:.0%} < {RULE_RETENTION:.0%}")
            if lo < FAR_M and new[b]["n"] and abs(new[b]["z_median"]) > RULE_MEDIAN_Z_M:
                reasons.append(
                    f"{b} m: |median z| {abs(new[b]['z_median']):.2f} > {RULE_MEDIAN_Z_M} m"
                )
        bins.append(row)
    return {
        "pass": not reasons,
        "reasons": reasons,
        "far": {
            "p90_abs_z_pnp": far_b.get("z_p90_abs"),
            "p90_abs_z_checked": far_n.get("z_p90_abs"),
            "z_sd_pnp": far_b.get("z_sd"),
            "z_sd_checked": far_n.get("z_sd"),
            "p90_drop": p90_drop,
            "sd_drop": sd_drop,
            "n_pnp": far_b["n"],
            "n_checked": far_n["n"],
        },
        "bins": bins,
    }


def _bin_lo(b: str) -> float:
    return float(b.lstrip(">").split("-")[0])


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def tune_range_check(recs: list[dict[str, Any]], start: RangeCheck = PHYSICS_START) -> dict:
    """Pick ``range_check`` on the tune runs' Pass-A records (never the held-out ones).

    Selection (pre-registered): among grid cells whose tune result passes :func:`rule_verdict`,
    the highest minimum retention (ties: fewest knob changes from ``start``); if none passes,
    among cells whose retention holds (>= 90 % in every bin with n >= 15) the largest
    ``min(p90 drop, sd drop)`` beyond 40 m; if none, ``start`` unchanged.
    """
    rows = []
    for lo in TUNE_GRID["size_ratio_min"]:
        for hi in TUNE_GRID["size_ratio_max"]:
            for tilt in TUNE_GRID["max_normal_tilt_deg"]:
                chk = replace(start, size_ratio_min=lo, size_ratio_max=hi, max_normal_tilt_deg=tilt)
                v = rule_verdict(recs, chk)
                shares = [
                    b["kept_share"]
                    for b in v["bins"]
                    if b["n"] >= RULE_MIN_N and b["kept_share"] is not None
                ]
                min_ret = min(shares) if shares else None
                drops = [d for d in (v["far"]["p90_drop"], v["far"]["sd_drop"]) if d is not None]
                changes = sum(getattr(chk, k) != getattr(start, k) for k in TUNE_GRID)
                rows.append(
                    {
                        "check": chk,
                        "pass": v["pass"],
                        "min_retention": min_ret,
                        "noise_drop": min(drops) if len(drops) == 2 else None,
                        "p90_drop": v["far"]["p90_drop"],
                        "sd_drop": v["far"]["sd_drop"],
                        "changes": changes,
                    }
                )
    passing = [r for r in rows if r["pass"]]
    if passing:
        best = max(passing, key=lambda r: (r["min_retention"] or 0.0, -r["changes"]))
        how = "passes the rule on the tune runs (highest retention)"
    else:
        holding = [
            r
            for r in rows
            if r["min_retention"] is not None
            and r["min_retention"] >= RULE_RETENTION
            and r["noise_drop"] is not None
        ]
        if holding:
            best = max(holding, key=lambda r: (r["noise_drop"], -r["changes"]))
            how = "no cell passes on the tune runs; largest noise drop with retention held"
        else:
            best = {"check": start}
            how = "no cell holds retention on the tune runs; physics starting points kept"
    scored = [r for r in rows if r["noise_drop"] is not None]
    top = max(scored, key=lambda r: r["noise_drop"]) if scored else None
    return {
        "chosen": best["check"],
        "how": how,
        "best_any": None
        if top is None
        else {
            "noise_drop": top["noise_drop"],
            "min_retention": top["min_retention"],
            **top["check"].as_dict(),
        },
        "grid": [
            {**{k: v for k, v in r.items() if k != "check"}, **r["check"].as_dict()} for r in rows
        ],
    }


# ----------------------------------------------------------------------------- driver


def _plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, RangeCheck):
        return x.as_dict()
    return x


def evaluate_runs(
    runs: list[RunData],
    truth: Truth,
    check: RangeCheck,
    *,
    modes: tuple[str, ...],
    with_frames: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Passes A/B (and C), logged targets and statistics for one set of runs."""
    a_recs: list[dict[str, Any]] = []
    a_counts: dict[str, dict] = {}
    b_rows: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}
    c_recs: list[dict[str, Any]] = []
    c_counts: dict[str, dict] = {}
    fid: dict[str, Any] = {}
    for run in runs:
        recs, counts = pass_a(run, truth, check)
        a_recs += recs
        a_counts[run.name] = counts
        inputs = tick_inputs(run, check)
        for m in modes:
            rows = pass_b(run, inputs, truth, m, check)
            b_rows[m] += rows
            if m == "pnp":
                fid[run.name] = fidelity(rows)
        if with_frames:
            recs_c, counts_c = pass_c(run, truth, check, device)
            c_recs += recs_c
            c_counts[run.name] = counts_c
    out: dict[str, Any] = {
        "runs": [r.name for r in runs],
        "pass_a": {
            "counts": a_counts,
            "modes": {m: binned_stats(mode_records(a_recs, m, check)) for m in modes},
            "retention": {m: retention(a_recs, m, check) for m in modes},
            "categories": dict(Counter(r["category"] for r in a_recs)),
        },
        "pass_b": {
            "fidelity": fid,
            "fidelity_pooled": fidelity([r for m in ("pnp",) if m in b_rows for r in b_rows[m]]),
            "modes": {
                m: binned_stats(target_records(b_rows[m], truth, "target", "valid")) for m in modes
            },
        },
        "logged_target": binned_stats(
            target_records(b_rows[modes[0]], truth, "logged_target", "logged_valid")
        ),
    }
    if "pnp" in modes and "pnp_checked" in modes:
        out["rule"] = rule_verdict(a_recs, check)
        bb, bn = out["pass_b"]["modes"]["pnp"][">40"], out["pass_b"]["modes"]["pnp_checked"][">40"]
        out["pass_b"]["far_drop"] = {
            "p90": _drop(bb.get("z_p90_abs"), bn.get("z_p90_abs")),
            "sd": _drop(bb.get("z_sd"), bn.get("z_sd")),
        }
    if with_frames:
        out["pass_c"] = {
            "counts": c_counts,
            "modes": {m: binned_stats(mode_records(c_recs, m, check)) for m in modes},
            "retention": {m: retention(c_recs, m, check) for m in modes},
        }
    out["_records_a"] = a_recs
    return out


def run_backprojection_eval(
    runs: list[str | Path],
    track: str | Path,
    *,
    trackdata_root: str | Path | None = None,
    tune_runs: list[str | Path] | None = None,
    modes: tuple[str, ...] = RANGE_MODES,
    with_frames: bool = False,
    device: str | None = None,
    out_dir: str | Path | None = None,
    check: RangeCheck | None = None,
    invocation: str | None = None,
) -> dict[str, Any]:
    """The whole evaluation; writes ``report.md`` + ``report.json`` under ``out_dir/<run-id>/``."""
    for m in modes:
        if m not in RANGE_MODES:
            raise HarnessConfigError(f"unknown range mode {m!r}; known: {list(RANGE_MODES)}")
    held = [Path(p) for p in runs]
    tune = [Path(p) for p in tune_runs or []]
    both = {p.resolve() for p in held} & {p.resolve() for p in tune}
    if both:
        raise HarnessConfigError(
            f"run(s) in both the tune and held-out sets: {sorted(str(p) for p in both)}"
        )
    truth = truth_passages(track, trackdata_root)
    start = check or PHYSICS_START
    report: dict[str, Any] = {
        "invocation": invocation,
        "track": truth.track,
        "truth_passages": truth.centers,
        "modes": list(modes),
        "physics_start": start,
    }
    if tune:
        tune_data = [load_run(p) for p in tune]
        t0 = evaluate_runs(tune_data, truth, start, modes=modes)
        tuned = tune_range_check(t0.pop("_records_a"), start)
        chosen = tuned["chosen"]
        report["tuning"] = {
            "how": tuned["how"],
            "chosen": chosen,
            "best_any": tuned["best_any"],
            "grid": tuned["grid"],
        }
        report["tune"] = evaluate_runs(
            tune_data, truth, chosen, modes=modes, with_frames=with_frames, device=device
        )
        report["tune"].pop("_records_a")
    else:
        chosen = start
    report["range_check"] = chosen
    held_data = [load_run(p) for p in held]
    report["held_out"] = evaluate_runs(
        held_data, truth, chosen, modes=modes, with_frames=with_frames, device=device
    )
    report["held_out"].pop("_records_a")
    if out_dir is not None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        d = Path(out_dir) / stamp
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.json").write_text(json.dumps(_plain(report), indent=2), encoding="utf-8")
        (d / "report.md").write_text(report_markdown(report), encoding="utf-8")
        report["out_dir"] = str(d)
    return report


# ----------------------------------------------------------------------------- report


def _f(v: Any, fmt: str = "+.2f") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return format(v, fmt)


def stats_table(stats: dict[str, dict[str, Any]], *, size: bool = False) -> list[str]:
    head = (
        "| range m | n | amb | z med | z mean | z sd | p90 abs z | rad med | rad sd | 3-D med "
        "| elev deg med |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    if size:
        head += " PnP-range err med | size-range err med |"
        sep += "---|---|"
    L = [head, sep]
    for b in (*BIN_LABELS, "<40", ">40"):
        s = stats.get(b) or {"n": 0}
        n = s["n"]
        row = (
            f"| {b} | {n}{' (thin)' if s.get('thin') and n else ''} | {s.get('ambiguous', '')}"
            f" | {_f(s.get('z_median'))} | {_f(s.get('z_mean'))} | {_f(s.get('z_sd'), '.2f')}"
            f" | {_f(s.get('z_p90_abs'), '.2f')} | {_f(s.get('radial_median'))}"
            f" | {_f(s.get('radial_sd'), '.2f')} | {_f(s.get('err3d_median'), '.2f')}"
            f" | {_f(s.get('elev_err_deg_median'))} |"
        )
        if size:
            row += f" {_f(s.get('pnp_range_err_median'))} | {_f(s.get('size_range_err_median'))} |"
        L.append(row)
    return L


def retention_table(ret: dict[str, Any]) -> list[str]:
    L = [
        "| range m | matched kept / rej | ambiguous kept / rej | unmatched kept / rej "
        "| size_ratio | normal_tilt |",
        "|---|---|---|---|---|---|",
    ]
    for b in (*BIN_LABELS, "<40", ">40", "all"):
        r = ret[b]
        cells = []
        for cat in ("matched", "ambiguous", "unmatched"):
            c = r[cat]
            share = "" if c["kept_share"] is None else f" ({c['kept_share']:.0%})"
            cells.append(f"{c['kept']} / {c['rejected']}{share}")
        L.append(
            f"| {b} | {' | '.join(cells)} | {r['reasons']['size_ratio']} "
            f"| {r['reasons']['normal_tilt']} |"
        )
    return L


def _set_section(name: str, res: dict[str, Any]) -> list[str]:
    L = [f"## {name}: {', '.join(res['runs'])}", ""]
    L += ["### Pass A (re-solved logged full-keypoint detections; primary)", ""]
    L.append(
        "Counts per run: "
        + "; ".join(
            f"{k}: " + ", ".join(f"{a} {b}" for a, b in sorted(v.items()))
            for k, v in res["pass_a"]["counts"].items()
        )
    )
    L.append("")
    L.append(
        "Categories: "
        + ", ".join(f"{k} {v}" for k, v in sorted(res["pass_a"]["categories"].items()))
    )
    for m, st in res["pass_a"]["modes"].items():
        L += ["", f"**{m}** (matched, unambiguous; metres, estimate - truth)", ""]
        L += stats_table(st, size=True)
    for m, ret in res["pass_a"]["retention"].items():
        if m == "pnp":
            continue
        L += ["", f"Retention, {m} (unmatched/ambiguous binned by PnP range)", ""]
        L += retention_table(ret)
    if "rule" in res:
        v = res["rule"]
        far = v["far"]
        L += ["", f"**AC4 rule: {'PASS' if v['pass'] else 'FAIL'}**", ""]
        L.append(
            f"- beyond 40 m: p90 |z| {_f(far['p90_abs_z_pnp'], '.2f')} -> "
            f"{_f(far['p90_abs_z_checked'], '.2f')} m (drop {_pct(far['p90_drop'])}); z sd "
            f"{_f(far['z_sd_pnp'], '.2f')} -> {_f(far['z_sd_checked'], '.2f')} m (drop "
            f"{_pct(far['sd_drop'])}); n {far['n_pnp']} -> {far['n_checked']}"
        )
        L += [f"- {r}" for r in v["reasons"]]
    L += ["", "### Pass B (tracker replay; secondary)", ""]
    fp = res["pass_b"]["fidelity_pooled"]
    L.append(
        f"Fidelity (replayed pnp target vs logged gate.target, ticks both hold a gate): "
        f"{fp['share_within_m']:.0%} within {fp['tol_m']} m, median {_f(fp['median_dev_m'], '.2f')}"
        f" m, p90 {_f(fp['p90_dev_m'], '.2f')} m over {fp['held_both']} ticks; held-state "
        f"agreement {fp['held_agreement']:.0%}."
        + (" **Indicative only.**" if fp["indicative_only"] else "")
    )
    for k, f in res["pass_b"]["fidelity"].items():
        L.append(
            f"- {k}: {f['share_within_m']:.0%} within, median "
            f"{_f(f['median_dev_m'], '.2f')} m ({f['held_both']} ticks)"
        )
    if "far_drop" in res["pass_b"]:
        fd = res["pass_b"]["far_drop"]
        L.append(f"- beyond 40 m: p90 |z| drop {_pct(fd['p90'])}, z sd drop {_pct(fd['sd'])}")
    for m, st in res["pass_b"]["modes"].items():
        L += ["", f"**{m}** (tracked target per tick, matched)", ""]
        L += stats_table(st)
    L += ["", "### Logged target (the live gate.target the policy saw, per tick)", ""]
    L += stats_table(res["logged_target"])
    if "pass_c" in res:
        L += ["", "### Pass C (detector re-run on the saved frames)", ""]
        L.append(
            "Counts per run: "
            + "; ".join(
                f"{k}: " + ", ".join(f"{a} {b}" for a, b in sorted(v.items()))
                for k, v in res["pass_c"]["counts"].items()
            )
        )
        for m, st in res["pass_c"]["modes"].items():
            L += ["", f"**{m}**", ""]
            L += stats_table(st, size=True)
        for m, ret in res["pass_c"]["retention"].items():
            if m != "pnp":
                L += ["", f"Retention, {m}", ""]
                L += retention_table(ret)
    return L + [""]


def report_markdown(rep: dict[str, Any]) -> str:
    L = ["# Gate back-projection evaluation (UC-91)", ""]
    if rep.get("invocation"):
        L += ["```", rep["invocation"], "```", ""]
    L.append(
        f"Track: {rep['track']}; {len(rep['truth_passages'])} distinct passages (truth). "
        f"Match: bearing within {MATCH_TOL_DEG} deg, ambiguity margin "
        f"{AMBIGUITY_MARGIN_DEG} deg."
    )
    L.append("")
    ps, ch = rep["physics_start"], rep["range_check"]
    L += ["## range_check", "", "| knob | physics start | final |", "|---|---|---|"]
    for k, v in ps.as_dict().items():
        L.append(f"| {k} | {v} | {getattr(ch, k)} |")
    if "tuning" in rep:
        L += ["", f"Tuned on the tune runs only: {rep['tuning']['how']}."]
        top = rep["tuning"].get("best_any")
        if top:
            L.append(
                f"Largest beyond-40 m noise drop anywhere on the grid (retention ignored): "
                f"{_pct(top['noise_drop'])} at min retention {_pct(top['min_retention'])} "
                f"(size_ratio {top['size_ratio_min']}-{top['size_ratio_max']}, tilt "
                f"{top['max_normal_tilt_deg']})."
            )
    L.append("")
    if "tune" in rep:
        L += _set_section("Tune runs", rep["tune"])
    L += _set_section("Held-out runs", rep["held_out"])
    return "\n".join(L) + "\n"


def summary(rep: dict[str, Any]) -> str:
    res = rep["held_out"]
    L = [f"held-out runs: {', '.join(res['runs'])}"]
    if "rule" in res:
        v = res["rule"]
        L.append(f"AC4 rule (held-out, Pass A): {'PASS' if v['pass'] else 'FAIL'}")
        L += [f"  - {r}" for r in v["reasons"]]
    L.append(f"range_check: {rep['range_check'].as_dict()}")
    fp = res["pass_b"]["fidelity_pooled"]
    L.append(
        f"Pass B fidelity: {fp['share_within_m']:.0%} within {fp['tol_m']} m, median "
        f"{_f(fp['median_dev_m'], '.2f')} m"
    )
    if rep.get("out_dir"):
        L.append(f"report: {rep['out_dir']}")
    return "\n".join(L)
