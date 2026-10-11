"""Fit the depth calibration on a recorded flight and score the reconstructed 40-ray scan.

``drone-fly harness depth-fit --capture <dir> --track <xml> [--race <xml>] [--props <json>]
[--preset ...] [--every N] [--report <dir>]``

1. **Ground-truth points** per frame (contract frame, Liftoff metres), from the track XML:
   gate-type tube centre lines (``gate_types[].outline``, densified) moved ``tube_radius`` toward
   the camera; grids on the camera-facing faces of ``offline.item_table`` boxes (barriers); points
   along ``cylinder`` axes (flags) offset by their radius toward the camera. Kept when in front,
   in the image, outside ``camera.masks`` and 2-40 m away; a coarse z-buffer (16 px cells) drops
   points well behind the nearest one in their cell.
2. **Fit**: the depth model's raw patch median at each point vs. the true camera ``z`` (or
   ``1 / z`` with ``depth.inverse``), robustly (RANSAC, then Huber IRLS):
   ``z = scale * raw + shift``. Reported: pairs, median absolute relative error and RMSE before
   (the configured calibration) and after, per range bin.
3. **Raycast scoring**: the synchronous
   :class:`~drone_fly.harness.liftoff.depth.MonocularDepthFreeSpaceEstimator`, with the fitted
   calibration, against the contract ``raycast()`` on the XML obstacles at training scale: median
   ray error, share of rays within 0.5 m (all rays and in-FOV), coverage. Arches are gates, not
   raycast obstacles, so ray truth ignores their legs.
4. Writes ``depth_fit_report.json`` (default ``artifacts/harness/liftoff/reports/depth-fit/``) and
   returns the report, including the ``depth.calibration`` YAML snippet.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.gate_types import GateTypeTable
from drone_fly.harness.geometry import CameraModel
from drone_fly.harness.liftoff.gate_cv import camera_from_config
from drone_fly.harness.liftoff.masks import FrameMasks
from drone_fly.harness.offline.autolabel import (
    _M,
    artifacts_root,
    check_item_table_against_props,
    load_props,
    place_gate,
    project_points,
)
from drone_fly.harness.offline.capture import load_capture
from drone_fly.harness.offline.track_xml import load_track_course, parse_track_items
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import FlightState, Frame

RANGE_M = (2.0, 40.0)
RANGE_BINS = ((2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0))
ZBUF_CELL_PX = 16
RAW_DOWNSAMPLE = 4


# ----------------------------------------------------------------------------- truth points


def _densify(poly: np.ndarray, step: float) -> np.ndarray:
    out = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:], strict=True):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.asarray(out)


def build_truth_geometry(
    items: list[Any], table: GateTypeTable, item_table: list[dict[str, Any]], step: float = 0.15
) -> dict[str, Any]:
    """Static ground-truth primitives in the contract frame (Liftoff metres)."""
    tubes, tube_r = [], []
    faces, face_n = [], []
    axes, axis_r = [], []
    for it in items:
        gtype = table.classify(it.item_id)
        if gtype is not None:
            g = place_gate(it, gtype)
            line = _densify(gtype.outline_points(), step)
            pts = g.local_to_world(gtype.local_points(line))
            tubes.append(pts)
            tube_r.append(np.full(len(pts), gtype.tube_radius * float(np.max(it.scale))))
            continue
        entry = next((e for e in item_table if re.search(e.get("match", "$^"), it.item_id)), None)
        if entry is None:
            continue
        rot_c = _M @ it.rotation  # Unity-local -> contract
        if entry.get("kind") == "box":
            size = np.asarray(entry["size"], np.float64) * it.scale
            off = np.asarray(entry.get("center_offset", [0.0, size[1] / 2.0, 0.0]), np.float64)
            center = _M @ (it.position + it.rotation @ off)
            half = size / 2.0
            grid = np.linspace(-1.0, 1.0, 5)
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    others = [a for a in range(3) if a != axis]
                    uu, vv = np.meshgrid(grid, grid)
                    local = np.zeros((uu.size, 3))
                    local[:, axis] = sign * half[axis]
                    local[:, others[0]] = uu.ravel() * half[others[0]]
                    local[:, others[1]] = vv.ravel() * half[others[1]]
                    normal = np.zeros(3)
                    normal[axis] = sign
                    faces.append(local @ rot_c.T + center)
                    face_n.append(np.tile(rot_c @ normal, (len(local), 1)))
        elif entry.get("kind") == "cylinder":
            r = float(entry["radius"]) * max(it.scale[0], it.scale[2])
            hgt = float(entry["height"]) * it.scale[1]
            off = np.asarray(entry.get("center_offset", [0.0, hgt / 2.0, 0.0]), np.float64)
            base = it.position + it.rotation @ (off - np.array([0.0, hgt / 2.0, 0.0]))
            ys = np.arange(0.1, hgt, 0.2)
            pts = np.array([_M @ (base + np.array([0.0, y, 0.0])) for y in ys])
            axes.append(pts)
            axis_r.append(np.full(len(pts), r))
    cat = lambda xs, w: np.concatenate(xs) if xs else np.zeros((0, w))  # noqa: E731
    return {
        "tube": cat(tubes, 3),
        "tube_r": cat(tube_r, 1).reshape(-1) if tube_r else np.zeros(0),
        "face": cat(faces, 3),
        "face_n": cat(face_n, 3),
        "axis": cat(axes, 3),
        "axis_r": cat(axis_r, 1).reshape(-1) if axis_r else np.zeros(0),
    }


def truth_points(geom: dict[str, Any], cam_pos: np.ndarray) -> np.ndarray:
    """Surface points visible-side for a camera position (contract frame, Liftoff metres)."""
    cam = np.asarray(cam_pos, np.float64)
    out = []
    if len(geom["tube"]):
        d = cam - geom["tube"]
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        out.append(geom["tube"] + d * geom["tube_r"][:, None])
    if len(geom["face"]):
        facing = np.einsum("ij,ij->i", geom["face_n"], cam - geom["face"]) > 0
        out.append(geom["face"][facing])
    if len(geom["axis"]):
        d = cam - geom["axis"]
        d[:, 2] = 0.0
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        out.append(geom["axis"] + d * geom["axis_r"][:, None])
    return np.concatenate(out) if out else np.zeros((0, 3))


def visible_truth(
    points_w: np.ndarray,
    camera: CameraModel,
    masks: FrameMasks,
    pos: np.ndarray,
    quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``(pixels, camera z)`` of points in front, in image, unmasked, 2-40 m, z-buffered."""
    if len(points_w) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    px, front, p_cam = project_points(camera, pos, quat, points_w)
    rng = np.linalg.norm(p_cam, axis=1)
    keep = camera.in_image(px, front) & (rng >= RANGE_M[0]) & (rng <= RANGE_M[1])
    if masks:
        keep &= ~masks.contains(px, camera.width, camera.height)
    px, z = px[keep], p_cam[keep, 2]
    if len(z) == 0:
        return px, z
    cell = (px // ZBUF_CELL_PX).astype(np.int64)
    key = cell[:, 1] * 100000 + cell[:, 0]
    zmin: dict[int, float] = {}
    for k, zz in zip(key, z, strict=True):
        zmin[int(k)] = min(zmin.get(int(k), np.inf), float(zz))
    nearest = np.array([zmin[int(k)] for k in key])
    ok = z <= nearest + np.maximum(0.5, 0.05 * nearest)
    return px[ok], z[ok]


def patch_median(raw: np.ndarray, px: np.ndarray, half: int = 3) -> np.ndarray:
    h, w = raw.shape[:2]
    out = np.empty(len(px))
    for i, (u, v) in enumerate(np.round(px).astype(int)):
        out[i] = float(
            np.median(
                raw[
                    max(v - half, 0) : min(v + half + 1, h), max(u - half, 0) : min(u + half + 1, w)
                ]
            )
        )
    return out


# ----------------------------------------------------------------------------- fit


def robust_affine_fit(
    x: np.ndarray, y: np.ndarray, *, iters: int = 300, rel_tol: float = 0.15, seed: int = 0
) -> tuple[float, float, int]:
    """``y ~= a * x + b``: RANSAC (relative residual), then Huber IRLS -> ``(a, b, inliers)``."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 2:
        raise HarnessConfigError("depth-fit needs at least 2 ground-truth pairs")
    rng = np.random.default_rng(seed)
    best, best_n = (1.0, 0.0), -1
    for _ in range(iters):
        i, j = rng.choice(len(x), 2, replace=False)
        if abs(x[i] - x[j]) < 1e-9:
            continue
        a = (y[i] - y[j]) / (x[i] - x[j])
        b = y[i] - a * x[i]
        n = int(np.sum(np.abs(a * x + b - y) <= rel_tol * np.abs(y)))
        if n > best_n:
            best, best_n = (a, b), n
    a, b = best
    inl = np.abs(a * x + b - y) <= rel_tol * np.abs(y)
    if inl.sum() >= 2:
        a, b = np.polyfit(x[inl], y[inl], 1)
    for _ in range(20):  # Huber IRLS on all pairs
        r = a * x + b - y
        mad = float(np.median(np.abs(r - np.median(r)))) * 1.4826 + 1e-9
        delta = 1.345 * mad
        wts = np.where(np.abs(r) <= delta, 1.0, delta / np.maximum(np.abs(r), 1e-12))
        sw = np.sqrt(wts)
        A = np.stack([x * sw, sw], axis=1)
        a, b = np.linalg.lstsq(A, y * sw, rcond=None)[0]
    inl = np.abs(a * x + b - y) <= rel_tol * np.abs(y)
    return float(a), float(b), int(inl.sum())


def depth_errors(pred: np.ndarray, z: np.ndarray, rng: np.ndarray) -> dict[str, Any]:
    rel = np.abs(pred - z) / z
    out = {
        "median_abs_rel_error": float(np.median(rel)),
        "rmse_m": float(np.sqrt(np.mean((pred - z) ** 2))),
        "bins": {},
    }
    for lo, hi in RANGE_BINS:
        sel = (rng >= lo) & (rng < hi)
        out["bins"][f"{lo:g}-{hi:g} m"] = (
            {"n": int(sel.sum()), "median_abs_rel_error": float(np.median(rel[sel]))}
            if sel.any()
            else {"n": 0}
        )
    return out


# ----------------------------------------------------------------------------- run


def run_depth_fit(
    capture: str | Path,
    track: str | Path,
    race: str | Path | None = None,
    props: str | Path | None = None,
    *,
    config: str | Path | dict[str, Any] | None = None,
    preset: str | None = None,
    harness: str = "liftoff",
    every: int = 5,
    report_dir: str | Path | None = None,
    lag_ms: float = 0.0,
    device: str | None = None,
    depth_model: Any = None,
    contract: Any = None,
) -> dict[str, Any]:
    """See the module docstring. ``depth_model`` / ``contract`` may be injected (tests)."""
    from drone_fly.harness.interfaces import BuildContext
    from drone_fly.harness.liftoff.depth import (
        DepthCalibration,
        MonocularDepthFreeSpaceEstimator,
        depth_model_from_config,
    )
    from drone_fly.harness.offline.evaluate import (
        _ray_metre_stats,
        ground_truth_targets,
        load_image,
    )

    overrides = {} if device is None else {"depth.device": device}
    cfg = load_config(harness, config, overrides, preset=preset)
    cam_cfg = cfg.section("camera")
    if cam_cfg.get("resolution") is None:
        raise HarnessConfigError("depth-fit needs camera.resolution (the capture's pixel grid)")
    camera = camera_from_config(cam_cfg)
    masks = FrameMasks.from_config(cam_cfg)
    table = GateTypeTable.from_config(cfg.get("gate_types"))
    off = cfg.section("offline")
    item_table = off.get("item_table") or []
    dcfg = cfg.section("depth")
    inverse = bool(dcfg.get("inverse", False))
    calib_cfg = dcfg.get("calibration") or {}
    before = DepthCalibration(
        float(calib_cfg.get("scale", 1.0)),
        float(calib_cfg.get("shift", 0.0)),
        inverse,
        float(calib_cfg.get("min_depth_m", 0.05)),
    )
    if every < 1:
        raise HarnessConfigError("--every must be >= 1")

    items = parse_track_items(track)
    geom = build_truth_geometry(items, table, item_table)
    props_warnings = []
    if props is not None:
        props_warnings = check_item_table_against_props(
            item_table, {it.item_id for it in items}, load_props(props)
        )
    cap = load_capture(capture, cfg.section("telemetry"), 1.0, lag_ms)
    frames = cap.frames[::every]
    if depth_model is None:
        depth_model = depth_model_from_config(dcfg)

    # -- pass 1: model on every sampled frame, truth pairs at full resolution --
    raws, xs, zs, rngs = [], [], [], []
    t0 = time.perf_counter()
    for f in frames:
        img = load_image(f.path)
        raw = np.asarray(depth_model.predict(img), np.float32)
        pos, quat = f.state.pos, f.state.quat
        px, z = visible_truth(truth_points(geom, pos), camera, masks, pos, quat)
        if len(z):
            xs.append(patch_median(raw, px))
            zs.append(z)
            pr = np.linalg.norm(camera.pixel_rays(px), axis=1) * z  # range along the pixel ray
            rngs.append(pr)
        raws.append(raw[::RAW_DOWNSAMPLE, ::RAW_DOWNSAMPLE].astype(np.float16))
    infer_s = time.perf_counter() - t0
    if not xs:
        raise HarnessConfigError("no ground-truth points were visible in the sampled frames")
    x, z, rng = np.concatenate(xs), np.concatenate(zs), np.concatenate(rngs)
    y = 1.0 / z if inverse else z
    a, b, n_inl = robust_affine_fit(x, y)
    after = DepthCalibration(a, b, inverse, before.min_depth_m)
    fit = {
        "pairs": int(len(z)),
        "inliers_15pct": n_inl,
        "frames": len(frames),
        "scale": a,
        "shift": b,
        "inverse": inverse,
        "before": {
            "scale": before.scale,
            "shift": before.shift,
            **depth_errors(before.apply(x), z, rng),
        },
        "after": depth_errors(after.apply(x), z, rng),
    }

    # -- pass 2: synchronous depth estimator with the fitted calibration vs contract raycast --
    if contract is None:
        from drone_fly.harness.contract import load_contract

        contract = load_contract(cfg.contract.gpu_prototype_dir, None, "cpu")
    scale = WorldScale(cfg.world_scale)
    course = load_track_course(
        track,
        race,
        item_table,
        scale,
        race_id_tag=str(off.get("race_id_tag", "instanceID")),
        laps=off.get("laps"),
        gate_types=table,
    )
    obstacles = course.raycast_obstacles()
    ctx = BuildContext(config=cfg, contract=contract, resources={"depth_model": depth_model})
    est = MonocularDepthFreeSpaceEstimator.from_config(cfg.section("free_space"), ctx)
    est.calibration = after
    est.frame_source = None

    def scaled(s: FlightState) -> FlightState:
        return FlightState(
            pos=scale.vector(s.pos), vel=scale.vector(s.vel), quat=s.quat, omega=s.omega, t=s.t
        )

    states_t = [scaled(s) for s in cap.states]
    truth_idx = ground_truth_targets(states_t, course)
    err_all, err_fov, cov = [], [], []
    for f, raw in zip(frames, raws, strict=True):
        st = states_t[f.sample_index]
        target = course.gates[truth_idx[f.sample_index]].center
        est.process_raw(
            raw.astype(np.float32), Frame(np.zeros((1, 1, 3), np.uint8), f.stamp, f.index), st
        )
        fs = est.update(None, st, target, f.stamp)
        truth = np.asarray(contract.raycast(st.as_state_dict(), obstacles, target), np.float64)
        e = np.abs(np.asarray(fs.raycast, np.float64) - truth) * float(contract.RAY_RANGE)
        err_all.append(e)
        if fs.in_fov_mask is not None and fs.in_fov_mask.any():
            err_fov.append(e[fs.in_fov_mask])
        cov.append(float(np.mean(fs.in_fov_mask)) if fs.in_fov_mask is not None else 0.0)
    rays = {
        **_ray_metre_stats(err_all, ""),
        **_ray_metre_stats(err_fov, "_in_fov"),
        "coverage_mean": float(np.mean(cov)) if cov else 0.0,
        "ray_range_m_training": float(contract.RAY_RANGE),
        "world_scale": cfg.world_scale,
        "frames": len(frames),
        "note": "arches are gates, not raycast obstacles: ray truth ignores the arch legs",
    }
    snippet = (
        "depth:\n  calibration:\n"
        f"    scale: {a:.4f}\n    shift: {b:.4f}\n"
        f"    # fitted by depth-fit on {cap.name}: {len(z)} pairs, median |rel err| "
        f"{fit['after']['median_abs_rel_error']:.1%}"
    )
    report = {
        "capture": str(Path(capture).resolve()),
        "track": str(Path(track).resolve()),
        "race": None if race is None else str(Path(race).resolve()),
        "preset": preset,
        "every": every,
        "lag_ms": lag_ms,
        "model": {"backend": dcfg.get("backend"), "model_id": dcfg.get("model_id")},
        "inference_s_total": infer_s,
        "fit": fit,
        "raycast": rays,
        "props_warnings": props_warnings,
        "yaml_snippet": snippet,
    }
    out_dir = Path(report_dir) if report_dir else artifacts_root() / "reports" / "depth-fit"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "depth_fit_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(path)
    return report
