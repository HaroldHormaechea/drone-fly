"""Measure the learned gate detector's error statistics and fit a noise model (UC70 D1).

``drone-fly train detector-noise --detector <weights> --captures <dir>... [--labels <dir>...]
[--live-runs <run>...] [--preset liftoff] [--device auto|cpu|cuda] [--out <dir>] [--latency-only]``

**Held-out only.** Per capture: when the weights' metadata records a split for it, only frames
inside its held-out window are used (the split is checked with
:func:`~drone_fly.harness.offline.gate_train.check_disjoint`); a capture the weights never
trained on is used whole. Deploy weights (trained on every frame), or a split without a held-out
window, raise :class:`~drone_fly.harness.errors.NoiseLeakageError`. There is no in-sample mode.

**Records.** Per frame the real estimator's ``detect_frame`` runs with the frame's telemetry
state (synchronously, Liftoff metres). Detections are matched to the frame's auto-labels by
class within ``max(1 m, 0.25 x range)``, greedily by distance. Every labelled gate that passes the
shared visibility predicate (:func:`drone_fly.sim.noise_model.visibility` — the simulator applies
the same one) becomes a record: camera-frame range / azimuth / elevation, the image-edge distance
of its centre, the detected flag and, when detected, the position error in the line-of-sight
frame (radial, tangential-h, tangential-v; :func:`~drone_fly.sim.noise_model.los_frame`), the
signed yaw error folded to +-90 deg and whether the class was right. Unmatched detections are
false positives. Per-instance sequences give the temporal statistics.

**Fit** (:func:`fit_noise_model`, pure): binned tables (range x |azimuth| x elevation per class,
bins with fewer than ``n_min`` samples pooled bearing first, then range; flagged), a parametric
fallback (logistic detection probability over range and edge distance, radial bias linear in
range, sigmas proportional to range, constant yaw sigma — maximum likelihood), the two-state
dropout (mean miss-burst seconds per range bin), the AR(1) error correlation time, the
false-positive rate + (range, azimuth) histogram + next-gate mis-association rate, class-nominal
size with a confusion matrix, inference latency and (``--live-runs``) the live frame timing.

**Outputs** go next to the weights, ``<weights dir>/<stem>-<sha256[:8]>/``: ``noise_model.yaml``
(re-running refits in place and keeps the previous file as ``noise_model.r<N>.yaml``),
``noise_report.md`` / ``.json`` and ``plots/*.png``. Weights are never moved or modified.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError, NoiseLeakageError

log = logging.getLogger(__name__)

RANGE_EDGES = (0.0, 4.0, 8.0, 12.0, 18.0, 25.0, 40.0)
AZIMUTH_EDGES = (0.0, 20.0, 40.0, 60.0)
ELEVATION_EDGES = (-90.0, -12.0, 12.0, 90.0)
FP_AZIMUTH_EDGES = (-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0)
N_MIN = 15
LOW_N_DETECTED = 200
#: Records belong to one temporal sequence while consecutive frames are this many periods apart.
SEQ_GAP_PERIODS = 3.0
#: ``gates.lost_timeout_s`` recommendation: the miss-burst p99 below this range, clamped.
LOST_TIMEOUT_RANGE_M = 25.0
LOST_TIMEOUT_CLAMP_S = (1.0, 3.0)
#: Window of the rolling effective control rate (``control_hz`` lower bound).
ROLLING_TICKS = 8
ERR_NAMES = ("radial_m", "tangential_h_m", "tangential_v_m", "yaw_deg")


@dataclass(frozen=True)
class NoiseFitConfig:
    range_edges: tuple[float, ...] = RANGE_EDGES  # max_range_m is appended
    azimuth_edges: tuple[float, ...] = AZIMUTH_EDGES
    elevation_edges: tuple[float, ...] = ELEVATION_EDGES
    fp_azimuth_edges: tuple[float, ...] = FP_AZIMUTH_EDGES
    n_min: int = N_MIN
    max_range_m: float = 60.0
    low_n_detected: int = LOW_N_DETECTED
    bootstrap: int = 1000
    seed: int = 0

    @property
    def full_range_edges(self) -> list[float]:
        e = [x for x in self.range_edges if x < self.max_range_m]
        return [*e, float(self.max_range_m)]


@dataclass
class Records:
    """Everything :func:`fit_noise_model` needs (plain lists; JSON-able)."""

    gates: list[dict[str, Any]] = field(default_factory=list)
    false_positives: list[dict[str, Any]] = field(default_factory=list)
    frames: list[dict[str, Any]] = field(default_factory=list)
    visibility: Counter = field(default_factory=Counter)
    latency_s: list[float] = field(default_factory=list)


# ----------------------------------------------------------------------------- helpers


def detector_id(weights: Path, sha: str) -> str:
    return f"{weights.stem}-{sha[:8]}"


def map_name(track: str | None) -> str:
    """``LiftoffArenaTrack04_0001.xml`` -> ``LiftoffArena`` (the stem before ``Track``)."""
    if not track:
        return "unknown"
    stem = Path(str(track)).stem
    m = re.match(r"(.+?)Track", stem)
    return m.group(1) if m else stem


def heldout_window(meta: dict[str, Any], capture_name: str) -> dict[str, Any] | None:
    """The held-out stamp window to use for ``capture_name`` (``None`` = every frame).

    Raises :class:`NoiseLeakageError` for deploy weights trained on the capture, or a split with
    no held-out window.
    """
    from drone_fly.harness.offline.gate_train import check_disjoint

    trained = {str(c.get("name")) for c in meta.get("captures") or []}
    split = (meta.get("splits") or {}).get(capture_name)
    if capture_name not in trained and split is None:
        return None  # never trained on this capture: every frame is held out
    if meta.get("purpose") == "deploy" or split is None or not split.get("holdout"):
        raise NoiseLeakageError(
            f"the detector weights were trained on capture {capture_name!r} without a held-out "
            "window (a deploy model or a split with no holdout); a noise model must be fitted on "
            "held-out frames only. Use weights trained with a held-out split, or a capture they "
            "never saw."
        )
    try:
        check_disjoint(split)
    except HarnessConfigError as exc:
        raise NoiseLeakageError(f"capture {capture_name!r}: {exc}") from exc
    return dict(split["holdout"])


def match_detections(
    det_centers: list[np.ndarray], det_cls: list[str | None], gates: list[dict[str, Any]]
) -> tuple[dict[int, int], set[int]]:
    """Greedy same-class matching within ``max(1 m, 0.25 x range)``: ``{gate: det}``, used dets."""
    pairs = []
    for gi, g in enumerate(gates):
        c = np.asarray(g["center_world"], np.float64)
        tol = max(1.0, 0.25 * float(g["range_m"]))
        for di, dc in enumerate(det_centers):
            if det_cls[di] != g["cls"]:
                continue
            dist = float(np.linalg.norm(dc - c))
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
    return used_g, used_d


def _gate_geometry(g: dict[str, Any], aperture: dict[str, tuple[float, float]]) -> tuple:
    from drone_fly.sim.noise_model import gate_axes, gate_corners

    c = np.asarray(g["center_world"], np.float64)
    axes = gate_axes(math.radians(float(g["normal_yaw_deg"])))
    return c, gate_corners(c, axes, aperture[g["cls"]])


def frame_records(
    *,
    capture: str,
    row: dict[str, Any],
    dets: list[Any],
    pos: np.ndarray,
    quat: np.ndarray,
    spec: Any,
    aperture: dict[str, tuple[float, float]],
    out: Records,
) -> None:
    """Append one frame's gate / false-positive / frame records to ``out`` (Liftoff metres)."""
    from drone_fly.sim.noise_model import (
        VIS_NAMES,
        VIS_OK,
        bearing_cam,
        edge_distance_norm,
        fold_yaw_signed,
        los_frame,
        project,
        to_camera,
        visibility,
    )

    gates = [g for g in row["gates"] if g["cls"] in aperture]
    det_c = [np.asarray(d.center, np.float64) for d in dets]
    det_cls = [d.cls for d in dets]
    matched, used = match_detections(det_c, det_cls, gates)
    cam = spec.camera
    if gates:
        geo = [_gate_geometry(g, aperture) for g in gates]
        centers = np.stack([c for c, _ in geo])
        corners = np.stack([k for _, k in geo])
        codes = visibility(centers, corners, pos, quat, spec)
        c_cam = to_camera(centers, pos, quat, cam)
        rng, az, el = bearing_cam(c_cam)
        u, v = project(c_cam, cam)
        edge = edge_distance_norm(u, v, cam.width, cam.height)
    else:
        codes = np.zeros(0, np.int64)
    for gi, g in enumerate(gates):
        out.visibility[VIS_NAMES[int(codes[gi])]] += 1
        if codes[gi] != VIS_OK:
            continue
        rec: dict[str, Any] = {
            "capture": capture,
            "frame": int(row["frame"]),
            "stamp": float(row["stamp"]),
            "instance": str(g.get("instance")),
            "cls": g["cls"],
            "range_m": float(rng[gi]),
            "azimuth_deg": float(az[gi]),
            "elevation_deg": float(el[gi]),
            "edge": float(edge[gi]),
            "detected": gi in matched,
            "err": None,
            "class_ok": None,
        }
        if gi in matched:
            d = dets[matched[gi]]
            truth = centers[gi]
            e_w = det_c[matched[gi]] - truth
            e_los = los_frame(truth - pos) @ e_w
            dyaw = fold_yaw_signed(float(d.yaw) - math.radians(float(g["normal_yaw_deg"])))
            rec["err"] = [float(e_los[0]), float(e_los[1]), float(e_los[2]),
                          float(math.degrees(dyaw))]  # fmt: skip
            rec["class_ok"] = True
        else:
            tol = max(1.0, 0.25 * float(g["range_m"]))
            other = [
                di for di in range(len(dets))
                if di not in used and det_cls[di] != g["cls"]
                and float(np.linalg.norm(det_c[di] - centers[gi])) <= tol
            ]  # fmt: skip
            rec["class_ok"] = False if other else None
            if other:
                rec["confused_with"] = det_cls[other[0]]
        out.gates.append(rec)
    fps = [di for di in range(len(dets)) if di not in used]
    for di in fps:
        t = np.asarray(dets[di].t_cam, np.float64)
        r, a, e = bearing_cam(t)
        out.false_positives.append({
            "capture": capture, "frame": int(row["frame"]), "stamp": float(row["stamp"]),
            "cls": det_cls[di], "range_m": float(r[0]), "azimuth_deg": float(a[0]),
            "elevation_deg": float(e[0]),
        })  # fmt: skip
    # the tracker's next-gate rule on this frame: nearest true detection = current, then the
    # farther detection closest to it; a false positive chosen there is a mis-association
    nxt = None
    true_idx = sorted(matched.values(), key=lambda di: float(np.linalg.norm(det_c[di] - pos)))
    if true_idx:
        cur = det_c[true_idx[0]]
        cur_d = float(np.linalg.norm(cur - pos))
        farther = [di for di in range(len(dets)) if di != true_idx[0]
                   and float(np.linalg.norm(det_c[di] - pos)) > cur_d]  # fmt: skip
        if farther:
            nxt = min(farther, key=lambda di: float(np.linalg.norm(det_c[di] - cur)))
    out.frames.append({
        "capture": capture, "frame": int(row["frame"]), "stamp": float(row["stamp"]),
        "n_detections": len(dets), "n_false": len(fps),
        "next_chosen": nxt is not None, "next_is_false": nxt is not None and nxt in fps,
    })  # fmt: skip


# ----------------------------------------------------------------------------- fitting


def fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3, iters: int = 50) -> np.ndarray:
    """Maximum-likelihood logistic regression (IRLS, a small ridge keeps separable data finite)."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    w = np.zeros(X.shape[1])
    if len(y):
        p0 = np.clip(y.mean(), 1e-3, 1 - 1e-3)
        w[0] = math.log(p0 / (1 - p0))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        g = X.T @ (y - p) - ridge * w
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def fit_parametric(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """The parametric fallback for one class (see the module docstring); Liftoff m / degrees."""
    r = np.array([x["range_m"] for x in recs], np.float64)
    edge = np.array([x["edge"] for x in recs], np.float64)
    y = np.array([x["detected"] for x in recs], np.float64)
    w = fit_logistic(np.stack([np.ones_like(r), r, edge], 1), y) if len(r) else np.zeros(3)
    det = [x for x in recs if x["detected"]]
    out: dict[str, Any] = {
        "p_det": {
            "form": "sigmoid(a + b * range_m + c * edge)",
            "a": float(w[0]),
            "b": float(w[1]),
            "c": float(w[2]),
        },  # fmt: skip
        "n": len(recs),
        "n_detected": len(det),
    }
    if det:
        rd = np.array([x["range_m"] for x in det], np.float64)
        e = np.array([x["err"] for x in det], np.float64)
        c_r = float(np.sum(e[:, 0] * rd) / max(np.sum(rd * rd), 1e-9))
        res = e[:, :3].copy()
        res[:, 0] -= c_r * rd
        s = np.sqrt(np.mean((res / rd[:, None]) ** 2, axis=0))
        yaw_b = float(np.mean(e[:, 3]))
        yaw_s = float(np.sqrt(np.mean((e[:, 3] - yaw_b) ** 2))) if len(e) > 1 else 10.0
    else:  # nothing detected: errors unknown; keep wide, flagged values
        c_r, s, yaw_b, yaw_s = 0.0, np.array([0.1, 0.1, 0.1]), 0.0, 15.0
        out["errors_unmeasured"] = True
    out.update(
        radial_bias_per_m=c_r,
        sigma_per_m=[float(v) for v in s],
        yaw_bias_deg=yaw_b,
        yaw_sigma_deg=max(yaw_s, 0.5),
    )
    return out


def parametric_sigma(par: dict[str, Any], rng: np.ndarray) -> np.ndarray:
    s = np.asarray(par["sigma_per_m"], np.float64)
    out = np.empty((len(rng), 4))
    out[:, :3] = np.maximum(rng[:, None] * s, 1e-6)
    out[:, 3] = max(float(par["yaw_sigma_deg"]), 1e-6)
    return out


def parametric_bias(par: dict[str, Any], rng: np.ndarray) -> np.ndarray:
    out = np.zeros((len(rng), 4))
    out[:, 0] = float(par["radial_bias_per_m"]) * rng
    out[:, 3] = float(par.get("yaw_bias_deg", 0.0))
    return out


def _bin_index(values: np.ndarray, edges: list[float] | tuple[float, ...]) -> np.ndarray:
    return np.clip(np.searchsorted(np.asarray(edges), values, "right") - 1, 0, len(edges) - 2)


def _stats(sel: list[dict[str, Any]], par: dict[str, Any], r_mid: float) -> dict[str, Any]:
    det = [x for x in sel if x["detected"]]
    n, nd = len(sel), len(det)
    out: dict[str, Any] = {"n": n, "n_detected": nd,
                           "p_det": float(nd / n) if n else None}  # fmt: skip
    if nd >= 2:
        e = np.array([x["err"] for x in det], np.float64)
        out["bias"] = [float(v) for v in e.mean(0)]
        out["cov"] = np.cov(e.T, ddof=1).tolist()
        out["error_source"] = "measured"
    else:
        rr = np.array([r_mid])
        out["bias"] = [float(v) for v in parametric_bias(par, rr)[0]]
        out["cov"] = np.diag(parametric_sigma(par, rr)[0] ** 2).tolist()
        out["error_source"] = "parametric (fewer than 2 detections)"
    wrong = sum(1 for x in sel if x.get("class_ok") is False)
    out["misclassification_rate"] = float(wrong / n) if n else None
    return out


def _pool(
    recs: list[dict[str, Any]],
    key_bins: np.ndarray,
    target: tuple,
    n_of: Any,
    n_min: int,
    n_range: int,
) -> tuple[list[int], list[str]]:
    """Indices pooled for fine bin ``target = (r, a, e)``: the bin itself, else all bearings of
    its range bin, else neighbouring range bins (nearest first) until ``n_of >= n_min``."""
    r, a, e = target
    same = [i for i in range(len(recs)) if tuple(key_bins[i]) == target]
    if n_of(same) >= n_min:
        return same, [f"r{r}a{a}e{e}"]
    rows = [i for i in range(len(recs)) if key_bins[i][0] == r]
    names = [f"r{r}"]
    if n_of(rows) >= n_min:
        return rows, names
    for step in range(1, n_range):
        for rr in (r - step, r + step):
            if 0 <= rr < n_range:
                rows += [i for i in range(len(recs)) if key_bins[i][0] == rr]
                names.append(f"r{rr}")
        if n_of(rows) >= n_min:
            break
    return rows, names


def fit_bins(recs: list[dict[str, Any]], par: dict[str, Any], cfg: NoiseFitConfig) -> dict:
    """Binned tables (fine range x |azimuth| x elevation + the range-marginal ``range_table``)."""
    redges = cfg.full_range_edges
    nr, na, ne = len(redges) - 1, len(cfg.azimuth_edges) - 1, len(cfg.elevation_edges) - 1
    if recs:
        rng = np.array([x["range_m"] for x in recs])
        kb = np.stack([
            _bin_index(rng, redges),
            _bin_index(np.abs([x["azimuth_deg"] for x in recs]), cfg.azimuth_edges),
            _bin_index(np.array([x["elevation_deg"] for x in recs]), cfg.elevation_edges),
        ], 1)  # fmt: skip
        r_lo, r_hi = float(rng.min()), float(rng.max())
    else:
        kb = np.zeros((0, 3), np.int64)
        r_lo, r_hi = math.inf, -math.inf

    def n_all(idx: list[int]) -> int:
        return len(idx)

    def n_det(idx: list[int]) -> int:
        return sum(1 for i in idx if recs[i]["detected"])

    def cell(target: tuple, lo: float, hi: float) -> dict[str, Any]:
        mid = 0.5 * (lo + min(hi, cfg.max_range_m))
        idx_p, from_p = _pool(recs, kb, target, n_all, cfg.n_min, nr)
        idx_e, from_e = _pool(recs, kb, target, n_det, cfg.n_min, nr)
        sp = _stats([recs[i] for i in idx_p], par, mid)
        se = _stats([recs[i] for i in idx_e], par, mid)
        own = sum(1 for i in range(len(recs)) if tuple(kb[i]) == target)
        return {
            "n_own": own,
            "n": sp["n"],
            "p_det": sp["p_det"],
            "misclassification_rate": sp["misclassification_rate"],
            "pooled_from": from_p,
            "n_detected": se["n_detected"],
            "bias": se["bias"],
            "cov": se["cov"],
            "error_source": se["error_source"],
            "error_pooled_from": from_e,
            "pooled": len(from_p) > 1
            or from_p[0] != f"r{target[0]}a{target[1]}e{target[2]}"
            or len(from_e) > 1
            or from_e[0] != f"r{target[0]}a{target[1]}e{target[2]}",
            "extrapolated": hi < r_lo or lo > r_hi,
        }

    cells = []
    for r in range(nr):
        for a in range(na):
            for e in range(ne):
                c = cell((r, a, e), redges[r], redges[r + 1])
                c.update(range_bin=r, azimuth_bin=a, elevation_bin=e)
                cells.append(c)
    # range-marginal table (what the binned simulator path uses): pool across bearing first
    table = []
    for r in range(nr):
        rows = [i for i in range(len(recs)) if kb[i][0] == r]
        names = [f"r{r}"]
        rows_d = list(rows)
        names_d = list(names)
        for step in range(1, nr):
            if len(rows) >= cfg.n_min:
                break
            for rr in (r - step, r + step):
                if 0 <= rr < nr:
                    rows += [i for i in range(len(recs)) if kb[i][0] == rr]
                    names.append(f"r{rr}")
        for step in range(1, nr):
            if n_det(rows_d) >= cfg.n_min:
                break
            for rr in (r - step, r + step):
                if 0 <= rr < nr:
                    rows_d += [i for i in range(len(recs)) if kb[i][0] == rr]
                    names_d.append(f"r{rr}")
        mid = 0.5 * (redges[r] + redges[r + 1])
        sp = _stats([recs[i] for i in rows], par, mid)
        se = _stats([recs[i] for i in rows_d], par, mid)
        table.append({
            "range_m": [redges[r], redges[r + 1]],
            "n_own": sum(1 for i in range(len(recs)) if kb[i][0] == r),
            "n": sp["n"], "p_det": sp["p_det"] if sp["p_det"] is not None else 0.0,
            "pooled_from": names, "n_detected": se["n_detected"], "bias": se["bias"],
            "cov": se["cov"], "error_source": se["error_source"], "error_pooled_from": names_d,
            "pooled": len(names) > 1 or len(names_d) > 1,
            "extrapolated": redges[r + 1] < r_lo or redges[r] > r_hi,
        })  # fmt: skip
    return {
        "range_edges": redges,
        "azimuth_edges_abs_deg": list(cfg.azimuth_edges),
        "elevation_edges_deg": list(cfg.elevation_edges),
        "n_min": cfg.n_min,
        "error_units": list(ERR_NAMES),
        "cells": cells,
        "range_table": table,
    }


def _sequences(recs: list[dict[str, Any]], frames: list[dict[str, Any]]) -> tuple[list, float]:
    """Per (capture, instance) ordered visible-record runs, split at gaps; + the frame period."""
    by_cap: dict[str, list[float]] = defaultdict(list)
    for f in frames:
        by_cap[f["capture"]].append(f["stamp"])
    periods = [float(np.median(np.diff(sorted(s)))) for s in by_cap.values() if len(s) > 1]
    period = float(np.median(periods)) if periods else 0.1
    frame_order = {c: {s: i for i, s in enumerate(sorted(v))} for c, v in by_cap.items()}
    seqs = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for x in recs:
        groups[(x["capture"], x["instance"])].append(x)
    for (cap, _), xs in groups.items():
        xs.sort(key=lambda x: x["stamp"])
        order = frame_order.get(cap, {})
        run = [xs[0]]
        for prev, cur in zip(xs, xs[1:], strict=False):
            consecutive = order.get(cur["stamp"], -9) - order.get(prev["stamp"], -99) == 1
            if consecutive and cur["stamp"] - prev["stamp"] <= SEQ_GAP_PERIODS * period:
                run.append(cur)
            else:
                seqs.append(run)
                run = [cur]
        seqs.append(run)
    return seqs, period


def fit_temporal(
    recs: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    par: dict[str, Any],
    cfg: NoiseFitConfig,
    lost_timeout_s: float,
) -> dict[str, Any]:
    """Markov dropout, miss-burst lengths (s) per range bin, AR(1) correlation time."""
    seqs, period = _sequences(recs, frames)
    bursts: list[tuple[float, float, bool]] = []  # (seconds, range at start, censored)
    trans = Counter()
    pairs: list[tuple[np.ndarray, np.ndarray, float]] = []
    for run in seqs:
        i = 0
        while i < len(run):
            if not run[i]["detected"]:
                j = i
                while j < len(run) and not run[j]["detected"]:
                    j += 1
                censored = j == len(run)  # the run ended (out of view) before a detection
                bursts.append(((j - i) * period, run[i]["range_m"], censored))
                i = j
            else:
                i += 1
        for a, b in zip(run, run[1:], strict=False):
            trans[(a["detected"], b["detected"])] += 1
            if a["detected"] and b["detected"]:
                ra = np.array([a["range_m"]])
                rb = np.array([b["range_m"]])
                za = (np.asarray(a["err"]) - parametric_bias(par, ra)[0]) / parametric_sigma(
                    par, ra)[0]  # fmt: skip
                zb = (np.asarray(b["err"]) - parametric_bias(par, rb)[0]) / parametric_sigma(
                    par, rb)[0]  # fmt: skip
                pairs.append((za, zb, b["stamp"] - a["stamp"]))
    redges = cfg.full_range_edges
    b_s = np.array([b[0] for b in bursts]) if bursts else np.zeros(0)
    b_r = np.array([b[1] for b in bursts]) if bursts else np.zeros(0)
    mean_all = float(b_s.mean()) if len(b_s) else period
    by_range = []
    for r in range(len(redges) - 1):
        m = (b_r >= redges[r]) & (b_r < redges[r + 1])
        by_range.append(float(b_s[m].mean()) if m.any() else None)
    vals = [v if v is not None else mean_all for v in by_range]
    near = b_s[b_r < LOST_TIMEOUT_RANGE_M] if len(b_s) else b_s
    p99 = float(np.percentile(near, 99)) if len(near) else None
    rho, tau = None, 0.0
    if pairs:
        za = np.stack([p[0] for p in pairs])
        zb = np.stack([p[1] for p in pairs])
        dt = float(np.median([p[2] for p in pairs]))
        rho = float(np.sum(za * zb) / max(math.sqrt(np.sum(za * za) * np.sum(zb * zb)), 1e-12))
        if rho <= 0:
            tau = 0.0
        elif rho >= 0.999:
            tau = 10.0
        else:
            tau = float(-dt / math.log(rho))
    n_det_prev = trans[(True, True)] + trans[(True, False)]
    n_miss_prev = trans[(False, False)] + trans[(False, True)]
    rec_timeout = None
    if p99 is not None:
        rec_timeout = float(np.clip(p99, *LOST_TIMEOUT_CLAMP_S))
    return {
        "frame_period_s": period,
        "markov": {
            "p_miss_given_det": float(trans[(True, False)] / n_det_prev) if n_det_prev else None,
            "p_miss_given_miss": float(trans[(False, False)] / n_miss_prev)
            if n_miss_prev
            else None,
            "transitions": {f"{int(a)}->{int(b)}": int(v) for (a, b), v in trans.items()},
        },
        "miss_burst_s_mean": mean_all,
        "miss_burst_s_by_range": {"edges": redges, "values": vals, "measured": by_range},
        "miss_burst_s_p99_below_25m": p99,
        "miss_bursts": {
            "n": len(bursts),
            "censored": int(sum(b[2] for b in bursts)),
            "seconds": [float(v) for v in b_s],
        },  # fmt: skip
        "ar1_rho": rho,
        "ar1_tau_s": tau,
        "ar1_pairs": len(pairs),
        "lost_timeout_s_current": lost_timeout_s,
        "lost_timeout_s_recommended": rec_timeout,
    }


def fit_false_positives(
    fps: list[dict[str, Any]], frames: list[dict[str, Any]], cfg: NoiseFitConfig
) -> dict[str, Any]:
    redges = cfg.full_range_edges
    aedges = list(cfg.fp_azimuth_edges)
    counts = np.zeros((len(redges) - 1, len(aedges) - 1))
    for f in fps:
        if f["range_m"] > redges[-1] or abs(f["azimuth_deg"]) > aedges[-1]:
            continue
        counts[_bin_index(np.array([f["range_m"]]), redges)[0],
               _bin_index(np.array([f["azimuth_deg"]]), aedges)[0]] += 1  # fmt: skip
    chosen = sum(1 for f in frames if f["next_chosen"])
    wrong = sum(1 for f in frames if f["next_is_false"])
    return {
        "rate_per_frame": float(len(fps) / len(frames)) if frames else 0.0,
        "n": len(fps),
        "frames": len(frames),
        "hist": {"range_edges": redges, "azimuth_edges": aedges, "counts": counts.tolist()},
        "next_misassoc_rate": float(wrong / chosen) if chosen else 0.0,
        "next_chosen_frames": chosen,
    }


def fit_noise_model(
    records: Records,
    cfg: NoiseFitConfig,
    *,
    classes: dict[str, tuple[float, float]],
    lost_timeout_s: float = 1.0,
) -> dict[str, Any]:
    """Pure fit: records -> the ``classes`` / ``false_positives`` / ``size`` model sections."""
    out_classes: dict[str, Any] = {}
    for cls, ap in classes.items():
        recs = [x for x in records.gates if x["cls"] == cls]
        par = fit_parametric(recs)
        bins = fit_bins(recs, par, cfg)
        temporal = fit_temporal(recs, records.frames, par, cfg, lost_timeout_s)
        relevant = [t for t in bins["range_table"] if t["range_m"][0] < 40.0]
        pooled = any(t["pooled"] or t["extrapolated"] for t in relevant)
        model = "parametric" if pooled or not recs else "binned"
        reason = (
            "a simulator-relevant range bin (< 40 m) was pooled below n_min or extrapolated"
            if pooled
            else ("no records" if not recs else "every simulator-relevant bin has n >= n_min")
        )
        out_classes[cls] = {
            "model": model,
            "model_reason": reason,
            "aperture_m": [float(ap[0]), float(ap[1])],
            "n": len(recs),
            "n_detected": sum(1 for x in recs if x["detected"]),
            "bins": bins,
            "parametric": par,
            "temporal": temporal,
        }
    confusion: dict[str, dict[str, int]] = {}
    for x in records.gates:
        row = confusion.setdefault(x["cls"], Counter())
        if x["detected"]:
            row[x["cls"]] += 1
        elif x.get("class_ok") is False:
            row[str(x.get("confused_with"))] += 1
        else:
            row["missed"] += 1
    return {
        "classes": out_classes,
        "false_positives": fit_false_positives(records.false_positives, records.frames, cfg),
        "size": {
            "mode": "class-nominal",
            "error_m": 0.0,
            "confusion": {k: dict(v) for k, v in confusion.items()},
        },
    }


# ----------------------------------------------------------------------------- live timing


def _pct(x: list[float] | np.ndarray, qs: tuple[float, ...] = (5, 50, 95)) -> dict[str, Any]:
    a = np.asarray(x, np.float64)
    if not a.size:
        return {f"p{int(q)}": None for q in qs} | {"n": 0}
    return {f"p{int(q)}": float(np.percentile(a, q)) for q in qs} | {"n": int(a.size)}


def live_timing(runs: list[str | Path]) -> dict[str, Any] | None:
    """Frame / tick timing of live ``harness run`` folders (learned gate estimator only)."""
    periods, ages, frame_dt, gaps, lags, delays, missed = [], [], [], [], [], [], []
    rolling, on_time, run_hz = [], [], []
    used, skipped = [], []
    for run in runs:
        d = Path(run)
        meta_p, ticks_p = d / "meta.json", d / "ticks.jsonl"
        if not ticks_p.is_file():
            skipped.append(f"{d.name}: no ticks.jsonl")
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
        parts = (meta.get("harness_info") or {}).get("parts") or {}
        if parts.get("gate_estimator") not in (None, "learned"):
            skipped.append(f"{d.name}: gate_estimator {parts.get('gate_estimator')}")
            continue
        ticks = [json.loads(line) for line in ticks_p.read_text(encoding="utf-8").splitlines()
                 if line.strip()]  # fmt: skip
        if len(ticks) < 3:
            skipped.append(f"{d.name}: fewer than 3 ticks")
            continue
        used.append(d.name)
        wall = np.array([t["wall_time"] for t in ticks], np.float64)
        dw = np.diff(wall)
        periods += list(dw)
        run_hz.append(float(len(dw) / max(wall[-1] - wall[0], 1e-9)))
        if len(dw) >= ROLLING_TICKS:
            rolling += list(1.0 / np.convolve(dw, np.ones(ROLLING_TICKS) / ROLLING_TICKS, "valid"))
        flags = [bool(t.get("missed_deadline")) for t in ticks]
        missed += flags
        on_time += [1.0 / p for p, m in zip(dw, flags[1:], strict=True) if not m and p > 0]
        last_idx, last_t = None, None
        for t in ticks:
            info = (t.get("gate") or {}).get("info") or {}
            a = (t.get("ages_s") or {}).get("frame")
            lag = info.get("detection_lag_frames")
            idx = info.get("frame_index")
            if a is not None:
                ages.append(float(a))
            lat = (t.get("latency_s") or {}).get("total")
            if lat is not None:
                delays.append(float(lat))
            if idx is None:
                continue
            if last_idx is not None and idx != last_idx and idx > last_idx:
                gaps.append(int(idx - last_idx))
                frame_dt.append((t["wall_time"] - last_t) / (idx - last_idx))
            if idx != last_idx:
                last_idx, last_t = idx, t["wall_time"]
            if lag is not None:
                lags.append((float(a or 0.0), int(lag)))
    if not used:
        return None
    period = float(np.median(frame_dt)) if frame_dt else None
    jitter = float(np.std(frame_dt)) if frame_dt else None
    # age of the frame behind the detections = newest frame age + lag frames x period
    lag_s = [age + n * (period or 0.0) for age, n in lags] or list(ages)
    g = np.asarray(gaps, np.float64)
    lo = float(np.percentile(rolling, 5)) if rolling else float(min(run_hz))
    hi = float(np.percentile(on_time, 95)) if on_time else float(max(run_hz))
    out = {
        "runs": used,
        "skipped": skipped,
        "tick_period_s": _pct(periods),
        "control_hz": [lo, max(hi, lo)],
        "control_hz_rule": f"[p5 of the {ROLLING_TICKS}-tick rolling effective rate, p95 of the "
        "on-deadline tick rate]: the simulator's nominal rate range; missed deadlines are its "
        "skipped ticks",
        "effective_hz_per_run": run_hz,
        "missed_deadline_rate": float(np.mean(missed)) if missed else None,
        "frame_age_s": _pct(ages),
        "frame_period_s": period,
        "frame_jitter_s": jitter,
        "frame_skip_p": float(1.0 - len(g) / g.sum()) if g.size else 0.0,
        "detection_lag_s": [float(np.percentile(lag_s, 5)), float(np.percentile(lag_s, 95))],
        "detection_lag_s_pct": _pct(lag_s),
        "obs_action_delay_s": [float(np.percentile(delays, 5)), float(np.percentile(delays, 95))]
        if delays
        else None,
    }
    return out


# ----------------------------------------------------------------------------- outputs


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_model(out_dir: Path, model: dict[str, Any]) -> Path:
    """Write ``noise_model.yaml``; an existing one is kept as ``noise_model.r<N>.yaml``."""
    import yaml

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "noise_model.yaml"
    history: list[dict[str, Any]] = []
    revision = 1
    if path.is_file():
        try:
            old = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            old = {}
        prev = int(old.get("revision") or 1)
        shutil.copy2(path, out_dir / f"noise_model.r{prev}.yaml")
        history = list(old.get("history") or [])
        revision = prev + 1
    history.append({
        "revision": revision, "fitted_utc": model["fitted_utc"],
        "captures": [c["name"] for c in model["captures"]],
        "n_detected": {k: v["n_detected"] for k, v in model["classes"].items()},
    })  # fmt: skip
    model["revision"] = revision
    model["history"] = history
    path.write_text(yaml.safe_dump(_plain(model), sort_keys=False, width=100), encoding="utf-8")
    return path


def _plain(x: Any) -> Any:
    """YAML-safe plain types (numpy scalars / tuples / inf)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else (1e9 if v > 0 else -1e9)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    return x


def bootstrap_ci(values: np.ndarray, stat: Any, n: int, seed: int) -> tuple[float, float] | None:
    v = np.asarray(values)
    if v.size < 2:
        return None
    rng = np.random.default_rng(seed)
    s = [stat(v[rng.integers(0, v.size, v.size)]) for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def report_tables(model: dict[str, Any], records: Records, cfg: NoiseFitConfig) -> dict:
    """Per-range-bin n, p_det (+ 95 % CI), median errors (+ CI); JSON-able."""
    out = {}
    for cls, c in model["classes"].items():
        recs = [x for x in records.gates if x["cls"] == cls]
        rows = []
        edges = c["bins"]["range_edges"]
        for r in range(len(edges) - 1):
            sel = [x for x in recs if edges[r] <= x["range_m"] < edges[r + 1]]
            det = np.array([x["detected"] for x in sel], np.float64)
            errs = np.array([x["err"] for x in sel if x["detected"]], np.float64).reshape(-1, 4)
            row: dict[str, Any] = {
                "range_m": [edges[r], edges[r + 1]],
                "n": len(sel),
                "n_detected": int(det.sum()),
                "p_det": float(det.mean()) if len(det) else None,
                "p_det_ci95": bootstrap_ci(det, np.mean, cfg.bootstrap, cfg.seed),
            }
            for k, name in enumerate(ERR_NAMES):
                col = np.abs(errs[:, k]) if len(errs) else np.zeros(0)
                row[f"median_abs_{name}"] = float(np.median(col)) if len(col) else None
                row[f"median_abs_{name}_ci95"] = bootstrap_ci(col, np.median, cfg.bootstrap,
                                                              cfg.seed)  # fmt: skip
            rows.append(row)
        out[cls] = rows
    return out


def write_plots(out_dir: Path, model: dict[str, Any], records: Records) -> list[str]:
    """``plots/*.png`` (matplotlib, Agg); returns the written file names."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is a dev dependency
        log.warning("matplotlib not installed: no plots")
        return []
    from drone_fly.sim.noise_model import sigmoid

    pdir = out_dir / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    written = []
    for cls, c in model["classes"].items():
        recs = [x for x in records.gates if x["cls"] == cls]
        rng = np.array([x["range_m"] for x in recs])
        det = np.array([x["detected"] for x in recs], float)
        par = c["parametric"]
        edges = c["bins"]["range_edges"]
        fig, ax = plt.subplots(figsize=(6, 4))
        mids, ps = [], []
        for r in range(len(edges) - 1):
            m = (rng >= edges[r]) & (rng < edges[r + 1])
            if m.any():
                mids.append(0.5 * (edges[r] + edges[r + 1]))
                ps.append(det[m].mean())
        ax.plot(mids, ps, "o", label="measured (per range bin)")
        rr = np.linspace(0, edges[-1], 200)
        pp = par["p_det"]
        ax.plot(rr, sigmoid(pp["a"] + pp["b"] * rr + pp["c"] * 0.5), label="parametric (edge 0.5)")
        ax.set(xlabel="range (m)", ylabel="P(detect)", title=f"{cls}: detection vs range",
               ylim=(0, 1))  # fmt: skip
        ax.legend()
        fig.tight_layout()
        fig.savefig(pdir / f"{cls}_p_det.png", dpi=90)
        plt.close(fig)
        written.append(f"{cls}_p_det.png")
        d = [x for x in recs if x["detected"]]
        if d:
            rd = np.array([x["range_m"] for x in d])
            e = np.array([x["err"] for x in d])
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            for k, lab in enumerate(ERR_NAMES[:3]):
                axs[0].plot(rd, e[:, k], ".", label=lab)
            for k, s in enumerate(par["sigma_per_m"]):
                axs[0].plot(rr, rr * s, "--", lw=0.8, color=f"C{k}")
            axs[0].set(xlabel="range (m)", ylabel="error (m)", title="position error (LOS frame)")
            axs[0].legend()
            axs[1].plot(rd, e[:, 3], ".")
            axs[1].axhline(par["yaw_sigma_deg"], ls="--", lw=0.8)
            axs[1].axhline(-par["yaw_sigma_deg"], ls="--", lw=0.8)
            axs[1].set(xlabel="range (m)", ylabel="yaw error (deg)", title="yaw error (folded)")
            fig.tight_layout()
            fig.savefig(pdir / f"{cls}_errors.png", dpi=90)
            plt.close(fig)
            written.append(f"{cls}_errors.png")
        t = c["temporal"]
        fig, ax = plt.subplots(figsize=(6, 4))
        secs = t["miss_bursts"]["seconds"]
        if secs:
            ax.hist(secs, bins=30)
        ax.axvline(t["lost_timeout_s_current"], color="r", label="lost_timeout_s (current)")
        if t["lost_timeout_s_recommended"] is not None:
            ax.axvline(t["lost_timeout_s_recommended"], color="g", ls="--",
                       label="recommended (p99 < 25 m, clamped)")  # fmt: skip
        ax.set(xlabel="miss burst (s)", ylabel="count", title=f"{cls}: miss bursts")
        ax.legend()
        fig.tight_layout()
        fig.savefig(pdir / f"{cls}_bursts.png", dpi=90)
        plt.close(fig)
        written.append(f"{cls}_bursts.png")
    fp = model["false_positives"]["hist"]
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(np.asarray(fp["counts"]), origin="lower", aspect="auto")
    ax.set(xlabel="azimuth bin", ylabel="range bin", title="false positives (count)")
    ax.set_xticks(range(len(fp["azimuth_edges"]) - 1))
    ax.set_xticklabels([f"{a:g}" for a in fp["azimuth_edges"][:-1]])
    ax.set_yticks(range(len(fp["range_edges"]) - 1))
    ax.set_yticklabels([f"{r:g}" for r in fp["range_edges"][:-1]])
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(pdir / "false_positives.png", dpi=90)
    plt.close(fig)
    written.append("false_positives.png")
    if records.latency_s:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.asarray(records.latency_s) * 1000.0, bins=30)
        ax.set(xlabel="inference (ms)", ylabel="frames",
               title=f"latency ({model['latency']['device']})")  # fmt: skip
        fig.tight_layout()
        fig.savefig(pdir / "latency.png", dpi=90)
        plt.close(fig)
        written.append("latency.png")
    return written


def _fmt(v: Any, f: str = ".3f") -> str:
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x, f) for x in v) + "]"
    return format(v, f)


def write_report(
    out_dir: Path,
    model: dict[str, Any],
    records: Records,
    cfg: NoiseFitConfig,
    plots: list[str],
    replay: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    tables = report_tables(model, records, cfg)
    rep = {
        "detector": model["detector"],
        "captures": model["captures"],
        "coverage": model["coverage"],
        "visibility_counts": dict(records.visibility),
        "frames": len(records.frames),
        "per_class": {
            cls: {
                "model": c["model"],
                "model_reason": c["model_reason"],
                "n": c["n"],
                "n_detected": c["n_detected"],
                "parametric": c["parametric"],
                "range_table": tables[cls],
                "pooled_range_bins": [
                    t["range_m"] for t in c["bins"]["range_table"] if t["pooled"]
                ],
                "extrapolated_range_bins": [
                    t["range_m"] for t in c["bins"]["range_table"] if t["extrapolated"]
                ],
                "temporal": {k: v for k, v in c["temporal"].items() if k != "miss_bursts"}
                | {"miss_bursts_n": c["temporal"]["miss_bursts"]["n"]},
                "low_n": c["n_detected"] < cfg.low_n_detected,
            }
            for cls, c in model["classes"].items()
        },  # fmt: skip
        "false_positives": model["false_positives"],
        "size": model["size"],
        "latency": model["latency"],
        "timing": model["timing"],
        "replay": replay,
        "plots": plots,
    }
    jpath = out_dir / "noise_report.json"
    jpath.write_text(json.dumps(_plain(rep), indent=2), encoding="utf-8")
    L = [f"# Detector noise report — `{model['detector']['id']}`", ""]
    if any(v["low_n"] for v in rep["per_class"].values()):
        L += [f"> **LOW-n, provisional**: fewer than {cfg.low_n_detected} detected samples. "
              "Refit when more held-out captures exist (UC71).", ""]  # fmt: skip
    det = model["detector"]
    L += [
        f"- Detector: `{det['name']}` sha256 `{det['sha256'][:16]}…` (purpose: "
        f"{det.get('purpose') or 'evaluate'}), git `{det.get('git_head')}`",
        f"- Fitted {model['fitted_utc']} (revision {model.get('revision')}), git "
        f"`{model.get('git_head')}`",
        f"- Maps covered: {', '.join(model['coverage']['maps'])} (classes: "
        f"{', '.join(model['coverage']['classes'])})",
        "- Captures (held out only):",
    ]
    for c in model["captures"]:
        w = c.get("window")
        win = "every frame (never trained on)" if not w else f"{w['t_start']:.3f}–{w['t_end']:.3f}"
        L.append(f"  - `{c['name']}` ({c['map']}): {c['frames_used']} frames, window {win}")
    vc = records.visibility
    L += ["", "## Visibility (labelled gates per frame)", "",
          "| outcome | count |", "|---|---|"]  # fmt: skip
    for k in (
        "visible",
        "behind",
        "beyond_range",
        "centre_out_of_image",
        "outline_below_frac",
        "centre_masked",
    ):
        L.append(f"| {k} | {vc.get(k, 0)} |")
    L += ["", "Records are the `visible` gates only (the shared predicate: centre in front, "
          f"centre in the image, >= {model['camera']['min_in_frame_frac']:.0%} of the aperture "
          "corners in the image, centre outside the HUD masks, range <= "
          f"{model['camera']['max_range_m']:g} m).", ""]  # fmt: skip
    for cls, c in model["classes"].items():
        par = c["parametric"]
        pp = par["p_det"]
        L += [
            f"## Class `{cls}`", "",
            f"- Records: {c['n']}, detected {c['n_detected']}; simulator model: "
            f"**{c['model']}** ({c['model_reason']})",
            f"- Parametric: P(det) = sigmoid({pp['a']:.3f} + {pp['b']:.4f}·range + "
            f"{pp['c']:.3f}·edge); radial bias {par['radial_bias_per_m']:.4f}·range; σ = "
            f"{_fmt(par['sigma_per_m'], '.4f')}·range (radial, tan-h, tan-v); yaw bias "
            f"{par['yaw_bias_deg']:.2f}°, σ {par['yaw_sigma_deg']:.2f}°",
            "",
            "| range (m) | n | det | P(det) [95 % CI] | med abs radial (m) | med abs tan-h (m) "
            "| med abs yaw (°) | pooled | extrapolated |",
            "|---|---|---|---|---|---|---|---|---|",
        ]  # fmt: skip
        for row, t in zip(tables[cls], c["bins"]["range_table"], strict=True):
            ci = row["p_det_ci95"]
            L.append(
                f"| {row['range_m'][0]:g}–{row['range_m'][1]:g} | {row['n']} | "
                f"{row['n_detected']} | {_fmt(row['p_det'])} {_fmt(ci, '.2f') if ci else ''} | "
                f"{_fmt(row['median_abs_radial_m'], '.2f')} | "
                f"{_fmt(row['median_abs_tangential_h_m'], '.2f')} | "
                f"{_fmt(row['median_abs_yaw_deg'], '.1f')} | "
                f"{'+'.join(t['pooled_from']) if t['pooled'] else 'no'} | "
                f"{'yes' if t['extrapolated'] else 'no'} |"
            )
        t = c["temporal"]
        L += [
            "",
            f"- Temporal: capture frame period {t['frame_period_s'] * 1000:.0f} ms; "
            f"P(miss|det) {_fmt(t['markov']['p_miss_given_det'])}, P(miss|miss) "
            f"{_fmt(t['markov']['p_miss_given_miss'])}; mean miss burst "
            f"{t['miss_burst_s_mean']:.2f} s over {t['miss_bursts']['n']} bursts "
            f"({t['miss_bursts']['censored']} censored by leaving the view); AR(1) ρ "
            f"{_fmt(t['ar1_rho'])} over {t['ar1_pairs']} pairs → τ {t['ar1_tau_s']:.3f} s",
            f"- Miss bursts vs `lost_timeout_s` {t['lost_timeout_s_current']:g} s: p99 below 25 m "
            f"= {_fmt(t['miss_burst_s_p99_below_25m'], '.2f')} s → recommended "
            f"{_fmt(t['lost_timeout_s_recommended'], '.2f')} s (clamped to "
            f"{LOST_TIMEOUT_CLAMP_S[0]:g}–{LOST_TIMEOUT_CLAMP_S[1]:g} s)",
            "",
        ]  # fmt: skip
    fp = model["false_positives"]
    lat = model["latency"]
    lat_note = (
        "" if lat["representative"]
        else " — NOT representative (rerun `--latency-only --device cuda`)"
    )  # fmt: skip
    L += [
        "## False positives", "",
        f"- {fp['rate_per_frame']:.2f} per frame ({fp['n']} over {fp['frames']} frames); next-gate "
        f"mis-association rate {fp['next_misassoc_rate']:.2f} over {fp['next_chosen_frames']} "
        "frames where the tracker's next-gate rule picks a detection",
        "",
        "## Size", "",
        f"- `{model['size']['mode']}` (error 0 m); confusion {model['size']['confusion']}",
        "",
        "## Latency", "",
        f"- {lat['device']}: p50 {_fmt(lat['p50_ms'], '.1f')} ms, p95 "
        f"{_fmt(lat['p95_ms'], '.1f')} ms (n {lat['n']}){lat_note}",
        "",
        "## Live timing", "",
    ]  # fmt: skip
    tm = model["timing"]
    if tm:
        L += [
            f"- Runs: {', '.join(tm['runs'])}",
            f"- Control rate p5–p95: {tm['control_hz'][0]:.1f}–{tm['control_hz'][1]:.1f} Hz; "
            f"missed deadlines {_fmt(tm['missed_deadline_rate'], '.1%')}",
            f"- Frame period {_fmt(tm['frame_period_s'], '.4f')} s (jitter σ "
            f"{_fmt(tm['frame_jitter_s'], '.4f')} s), skip p {tm['frame_skip_p']:.2f}; frame age "
            f"p50 {_fmt(tm['frame_age_s']['p50'], '.3f')} s; detection lag p5–p95 "
            f"{_fmt(tm['detection_lag_s'], '.3f')} s; obs→action "
            f"{_fmt(tm['obs_action_delay_s'])} s",
        ]  # fmt: skip
    else:
        L.append("- none (`--live-runs` not given): the simulator uses its default timing")
    L += ["", "## Validation replay", ""]
    if replay:
        L += [f"- {k}: {v}" for k, v in replay.get("summary", {}).items()]
    else:
        L.append("- not run")
    L += ["", "## Plots", ""] + [f"![{p}](plots/{p})" for p in plots]
    mpath = out_dir / "noise_report.md"
    mpath.write_text("\n".join(L) + "\n", encoding="utf-8")
    return mpath, jpath


# ----------------------------------------------------------------------------- validation replay

#: Reference-perception seeds averaged in the replay.
REPLAY_SEEDS = 20


def load_catalogue_course(track_file: str | None, trackdata_root: str | Path | None) -> Any:
    """The catalogue course whose ``track`` file is ``track_file`` (no scene needed)."""
    from drone_fly.tracks import catalogue as cat
    from drone_fly.tracks.catalogue import TrackDataError
    from drone_fly.tracks.loader import load_course

    if not track_file:
        raise TrackDataError("the labels name no track file")
    try:
        c = cat.load_named("liftoff")
    except TrackDataError:
        import drone_fly

        src = Path(drone_fly.__file__).resolve().parents[2]
        c = cat.load_catalogue(src / "tracks" / "liftoff.yaml")
    name = Path(str(track_file)).name
    entry = next((e for e in c.entries if e.get("track") == name), None)
    if entry is None:
        raise TrackDataError(f"no catalogue entry with track {name}")
    root = Path(trackdata_root) if trackdata_root else Path(str(track_file)).parent
    return load_course(entry, catalogue=c, root=root, scenes_root=None)[0]


def truth_progress(cap: Any, course: Any) -> np.ndarray:
    """Route progress ``k`` per telemetry sample (``RouteProgress`` + ``gate_crossed``, slack 0;
    restarted at every capture segment, i.e. after respawns)."""
    from drone_fly.sim.pool import RouteProgress, gate_crossed

    pos = np.array([course.frame.to_course_point(s.pos) for s in cap.states], np.float64)
    ks = np.zeros(len(pos), np.int64)
    for a, b in cap.segments:
        prog = RouteProgress(course.n_passages, course.laps, course.circuit)
        for i in range(a + 1, b):
            p = course.passages[prog.target]
            if not prog.finished and gate_crossed(pos[i - 1], pos[i], p, 0.0):
                prog.advance()
            ks[i] = prog.k
    return ks


def _passage_dict(p: Any) -> dict[str, Any]:
    return {
        "center": np.asarray(p.center, np.float64),
        "rot": np.asarray(p.rot, np.float64),
        "wh": (float(p.w), float(p.h)),
    }


def _hold_runs(statuses: list[str], stamps: list[float]) -> tuple[list[float], int]:
    """Durations of consecutive ``holding`` runs and the number of entries into ``lost``."""
    holds, lost, start = [], 0, None
    for i, st in enumerate(statuses):
        if st == "holding" and start is None:
            start = stamps[i]
        if st != "holding" and start is not None:
            holds.append(stamps[i] - start)
            start = None
        if st == "lost" and (i == 0 or statuses[i - 1] != "lost"):
            lost += 1
    if start is not None:
        holds.append(stamps[-1] - start)
    return holds, lost


def _summ(x: list[float]) -> dict[str, Any]:
    a = np.asarray(x, np.float64)
    return {
        "n": int(a.size),
        "median": float(np.median(a)) if a.size else None,
        "p90": float(np.percentile(a, 90)) if a.size else None,
    }


#: Below this many frames a proportion uses the conservative (upper) 95 % Wilson bound.
ACQ_MIN_N = 30


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def fit_acquisition(seq: list[tuple[Any, list[Any]]], k_at: list[int], course: Any) -> dict:
    """What the tracker's acquisition rule (``min distance * (1 + off_axis)``) picks on each
    held-out frame: P(a wrong candidate — a false positive or another gate — is present) and
    P(the rule picks a wrong one | the truth target is detected too). Small samples use the
    upper (conservative) 95 % Wilson bound."""
    n = course.n_passages
    frames = present = both = wrong_pick = 0
    for (f, dets), k in zip(seq, k_at, strict=True):
        frames += 1
        if not dets:
            continue
        truth = course.frame.to_contract_point(course.passages[k % n].center)
        pos = np.asarray(f.state.pos, np.float64)
        tol = max(1.0, 0.25 * float(np.linalg.norm(truth - pos)))
        is_t = [float(np.linalg.norm(np.asarray(d.center, np.float64) - truth)) <= tol
                for d in dets]  # fmt: skip
        has_wrong = not all(is_t)
        present += int(has_wrong)
        if has_wrong and any(is_t):
            both += 1
            pick = min(range(len(dets)), key=lambda i: dets[i].distance * (1 + dets[i].off_axis))
            wrong_pick += int(not is_t[pick])

    def est(k: int, m: int) -> dict[str, Any]:
        lo, hi = wilson(k, m)
        p = k / m if m else None
        small = m < ACQ_MIN_N
        rule = "upper 95 % Wilson bound (n < 30)" if small else "point estimate"
        used = hi if small or p is None else p
        return {"k": k, "n": m, "p": p, "ci95": [lo, hi], "used": used, "rule": rule}

    pres = est(present, frames)
    pick = est(wrong_pick, both)
    return {
        "p_wrong_present": pres,
        "p_pick_wrong": pick,
        "used": {
            "p_wrong_present": float(pres["used"]),
            "p_pick_wrong": float(pick["used"]),
            "source": f"validation replay of {frames} held-out frames ("
            f"{pres['rule']} / {pick['rule']})",
        },
    }


def validation_replay(
    seq: list[tuple[Any, list[Any]]],
    ks: np.ndarray,
    course: Any,
    model: dict[str, Any],
    tracker: dict[str, Any],
    camera: Any,
    seeds: int = REPLAY_SEEDS,
) -> dict[str, Any]:
    """The real ``TrackingGateEstimator`` over the held-out frames (their recorded detections) vs
    the reference perception on the same truth trajectory (see the module docstring)."""
    from drone_fly.harness.liftoff.gate_cv import TrackingGateEstimator
    from drone_fly.harness.scale import WorldScale
    from drone_fly.harness.types import Frame
    from drone_fly.sim.noise_model import NoiseModel
    from drone_fly.sim.perception import (
        EPISODE_DRAWS,
        STATUS_NAMES,
        TICK_DRAWS,
        PerceptionConfig,
        draw_width,
    )
    from drone_fly.student.perception_ref import ReferencePerception, sample

    cache = {i: dets for i, (_, dets) in enumerate(seq)}

    class _Replay(TrackingGateEstimator):
        def detect(self, image: np.ndarray, state: Any) -> list[Any]:  # pragma: no cover
            return []

        def detect_frame(self, frame: Any, state: Any) -> list[Any]:
            return cache[frame.index]

    k_at = [int(ks[f.sample_index]) for f, _ in seq]
    acq = fit_acquisition(seq, k_at, course)
    model["acquisition"] = acq
    nm = NoiseModel(model)
    ap = nm.class_noise().aperture_m
    tr = _Replay(
        camera,
        WorldScale(1.0),
        default_aperture=float(min(ap)),
        assoc_radius=float(tracker["assoc_radius"]),
        lost_timeout_s=float(tracker["lost_timeout_s"]),
        acquire_timeout_s=float(tracker["acquire_timeout_s"]),
        pass_area_frac=float(tracker.get("pass_area_frac", 0.2)),
    )
    n = course.n_passages
    total = course.total_crossings
    stamps = [f.stamp for f, _ in seq]
    truth_cross = sum(max(0, b - a) for a, b in zip(k_at, k_at[1:], strict=False))
    blank = np.zeros((1, 1, 3), np.uint8)
    real_status, real_err, real_pass = [], [], 0
    for i, ((f, _), k) in enumerate(zip(seq, k_at, strict=True)):
        est = tr.update(Frame(blank, f.stamp, i), f.state, f.stamp)
        real_status.append(est.status)
        real_pass += int(est.passed)
        if tr.current is not None:
            truth = course.frame.to_contract_point(course.passages[k % n].center)
            real_err.append(float(np.linalg.norm(np.asarray(est.target, np.float64) - truth)))
    holds_r, lost_r = _hold_runs(real_status, stamps)
    period = float(np.median(np.diff(stamps))) if len(stamps) > 1 else 0.1
    cfg = PerceptionConfig.from_model(
        nm,
        "noise",
        detection_lag_s=(0.0, 0.0),
        frame_period_s=period,
        frame_jitter_s=0.0,
        frame_skip_p=0.0,
        lost_timeout_s=float(tracker["lost_timeout_s"]),
        world_scale_k=1.0,
    )
    scale = course.frame.world_scale
    dur_min = max((stamps[-1] - stamps[0]) / 60.0, 1e-9)

    def run_reference(model_: Any) -> dict[str, Any]:
        ref_err, ref_holds, ref_lost, ref_track = [], [], [], []
        for s in range(seeds):
            rng = np.random.default_rng(1000 + s)
            ref = ReferencePerception(model_, cfg)
            f0 = seq[0][0]
            ref.reset(f0.stamp, course.frame.to_course_point(f0.state.pos), f0.state.quat,
                      np.zeros(draw_width(EPISODE_DRAWS)))  # fmt: skip
            sts = []
            for i, ((f, _), k) in enumerate(zip(seq, k_at, strict=True)):
                pc = course.frame.to_course_point(f.state.pos)
                ref.push_pose(f.stamp, pc, f.state.quat)
                nxt = course.passages[(k + 1) % n] if k + 1 < total else None
                crossed = i > 0 and k > k_at[i - 1]
                ref.step(
                    f.stamp, pc, f.state.quat, _passage_dict(course.passages[k % n]),
                    None if nxt is None else _passage_dict(nxt), crossed, scale,
                    sample(rng, TICK_DRAWS),
                )  # fmt: skip
                out = ref.outputs()
                sts.append(STATUS_NAMES[out["status"]])
                if out["acquired"]:
                    dd = np.linalg.norm(out["target"] - course.passages[k % n].center) * scale
                    ref_err.append(float(dd))
            h, lo = _hold_runs(sts, stamps)
            ref_holds += h
            ref_lost.append(lo)
            ref_track.append(float(np.mean([x == "tracking" for x in sts])))
        return {
            "seeds": seeds,
            "target_error_m": _summ(ref_err),
            "tracking_share": float(np.mean(ref_track)),
            "hold_s": _summ(ref_holds),
            "lost_events_mean": float(np.mean(ref_lost)),
            "lost_events_per_min": float(np.mean(ref_lost)) / dur_min,
            "passes": truth_cross,
        }

    ref_sum = run_reference(nm)
    unfitted = {k: v for k, v in model.items() if k != "acquisition"}
    ref_default = run_reference(NoiseModel(unfitted))
    real = {
        "target_error_m": _summ(real_err),
        "tracking_share": float(np.mean([x == "tracking" for x in real_status])),
        "lost_share": float(np.mean([x == "lost" for x in real_status])),
        "hold_s": _summ(holds_r),
        "lost_events": lost_r,
        "lost_events_per_min": lost_r / dur_min,
        "tracker_passes": real_pass,
    }

    re, fe = real["target_error_m"]["median"], ref_sum["target_error_m"]["median"]
    rh, fh = real["hold_s"]["median"], ref_sum["hold_s"]["median"]
    summary = {
        "frames": len(seq),
        "truth crossings in the window": truth_cross,
        "real tracker passes (own pass rule)": real_pass,
        "target error median real / reference (m)": f"{_fmt(re, '.2f')} / {_fmt(fe, '.2f')}",
        "tracking share real / reference": (
            f"{real['tracking_share']:.2f} / {ref_sum['tracking_share']:.2f}"
        ),
        "hold median real / reference (s)": f"{_fmt(rh, '.2f')} / {_fmt(fh, '.2f')}",
        "lost events per min real / reference": (
            f"{real['lost_events_per_min']:.2f} / {ref_sum['lost_events_per_min']:.2f}"
        ),
        "note": "reference: truth-driven hand-over, synchronous frames (no latency), "
        f"{seeds} seeds; real: recorded detections, the tracker's own association + pass rule",
    }
    summary["acquisition fit (used)"] = (
        f"P(wrong candidate) {acq['used']['p_wrong_present']:.2f}, P(pick wrong | target "
        f"detected) {acq['used']['p_pick_wrong']:.2f} ({acq['used']['source']})"
    )
    de, dt_ = ref_default["target_error_m"]["median"], ref_default["tracking_share"]
    summary["reference without the acquisition fit (0.5 pick, Poisson presence)"] = (
        f"target error median {_fmt(de, '.2f')} m, tracking share {dt_:.2f}, lost events per "
        f"min {ref_default['lost_events_per_min']:.2f}"
    )
    return {"real": real, "reference": ref_sum, "reference_unfitted": ref_default,
            "summary": summary, "acquisition": acq}  # fmt: skip


# ----------------------------------------------------------------------------- driver


def default_labels_dir(weights: Path, capture: Path) -> Path:
    """``<weights>/../../labels/<capture>``, else the repo's ``artifacts/.../labels``."""
    cand = weights.parent.parent / "labels" / capture.name
    if cand.is_dir():
        return cand
    from drone_fly.harness.offline.autolabel import artifacts_root

    return artifacts_root() / "labels" / capture.name


def _camera_block(cfg: Any, lab_meta: dict[str, Any], fit: NoiseFitConfig) -> dict[str, Any]:
    """The model's camera block from the preset, verified against every label set's camera."""
    from drone_fly.sim.noise_model import normalise_masks

    cam = cfg.section("camera")
    res = [int(v) for v in cam.get("resolution") or (1920, 1080)]
    for key, want in (("resolution", res), ("hfov_deg", float(cam.get("hfov_deg"))),
                      ("uptilt_deg", float(cam.get("uptilt_deg", 0.0)))):  # fmt: skip
        got = (lab_meta.get("camera") or {}).get(key)
        if got is None:
            continue
        same = [int(v) for v in got] == want if key == "resolution" else abs(float(got) - want)\
            < 1e-6  # fmt: skip
        if not same:
            raise HarnessConfigError(
                f"camera.{key} differs between the preset ({want}) and the labels ({got}); "
                "relabel the capture or pick the matching preset"
            )
    masks = normalise_masks(cam.get("masks") or [], res)
    lab_masks = normalise_masks(lab_meta.get("masks") or [], res)
    if lab_meta.get("masks") is not None and lab_masks != masks:
        raise HarnessConfigError("camera.masks differ between the preset and the labels")
    return {
        "resolution": res,
        "hfov_deg": float(cam.get("hfov_deg")),
        "uptilt_deg": float(cam.get("uptilt_deg", 0.0)),
        "masks_norm": masks,
        "min_in_frame_frac": 0.5,
        "max_range_m": float(fit.max_range_m),
    }


def build_estimator(cfg: Any) -> Any:
    from drone_fly.harness.interfaces import BuildContext
    from drone_fly.harness.registry import build_part

    ctx = BuildContext(config=cfg, contract=None, resources={})
    return build_part("gate_estimator", "learned", cfg.section("gates"), ctx)


def measure_latency(estimator: Any, frames: list[Any], n: int = 200) -> list[float]:
    """CUDA-synchronised inference seconds over up to ``n`` frames (the network only)."""
    from drone_fly.harness.offline.evaluate import load_image

    out = []
    for f in frames[:n]:
        estimator.infer(estimator.masks.apply(load_image(f.path)) if estimator.masks
                        else load_image(f.path))  # fmt: skip
        out.append(float(estimator.inference_s))
    return out


def run_detector_noise(
    detector: str | Path,
    captures: list[str | Path],
    *,
    labels: list[str | Path] | None = None,
    live_runs: list[str | Path] | None = None,
    preset: str | None = "liftoff",
    harness: str = "liftoff",
    config: Any = None,
    device: str = "auto",
    out: str | Path | None = None,
    latency_only: bool = False,
    fit: NoiseFitConfig | None = None,
    estimator: Any = None,
    max_frames: int | None = None,
    replay: bool = True,
    trackdata_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the measurement and write the model + report (see the module docstring)."""
    import torch

    from drone_fly.harness.config import find_repo_root, load_config
    from drone_fly.harness.gate_types import GateTypeTable
    from drone_fly.harness.liftoff.gate_learned import resolve_weights_path
    from drone_fly.harness.offline.capture import load_capture
    from drone_fly.harness.offline.evaluate import load_image
    from drone_fly.harness.offline.gate_train import is_held_out, load_labels
    from drone_fly.harness.recording import git_head
    from drone_fly.harness.types import Frame
    from drone_fly.sim.noise_model import VisibilitySpec, sha256_file

    fit = fit or NoiseFitConfig()
    weights = resolve_weights_path(str(Path(detector).resolve()))
    sha = sha256_file(weights)
    did = detector_id(weights, sha)
    out_dir = Path(out) if out else weights.parent / did
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    overrides = {
        "gates.learned.weights": str(weights),
        "gates.learned.device": device,
        "gates.learned.async_inference": False,
        "scaling.world_scale": 1.0,  # Liftoff metres throughout
    }
    cfg = load_config(harness, config, overrides, preset=preset)
    if estimator is None:
        estimator = build_estimator(cfg)
    meta = getattr(estimator, "meta", {}) or {}
    table = GateTypeTable.from_config(cfg.get("gate_types"))
    aperture = {t.cls: (float(t.aperture[0]), float(t.aperture[1])) for t in table.types}
    gcfg = cfg.section("gates")
    lost_timeout = float(gcfg.get("lost_timeout_s", 1.0))
    labels = list(labels or [])
    if labels and len(labels) != len(captures):
        raise HarnessConfigError("--labels must give one folder per --captures entry")
    cuda = str(device).startswith("cuda")

    if latency_only:
        import yaml

        path = out_dir / "noise_model.yaml"
        if not path.is_file():
            raise HarnessConfigError(f"--latency-only needs an existing {path}")
        model = yaml.safe_load(path.read_text(encoding="utf-8"))
        cap = load_capture(captures[0], cfg.section("telemetry"), 1.0)
        lat = measure_latency(estimator, cap.frames)
        model["latency"] = _latency_block(lat, device, cuda)
        model.setdefault("history", []).append(
            {"revision": model.get("revision"), "latency_updated_utc": _now(), "device": device}
        )
        path.write_text(yaml.safe_dump(_plain(model), sort_keys=False, width=100),
                        encoding="utf-8")  # fmt: skip
        return {"model_path": str(path), "latency": model["latency"], "out_dir": str(out_dir)}

    records = Records()
    cap_meta = []
    replay_input = []
    camera_block = None
    for i, cap_dir in enumerate(captures):
        cap_dir = Path(cap_dir)
        lab = load_labels(labels[i] if labels else default_labels_dir(weights, cap_dir))
        window = heldout_window(meta, lab.name)
        block = _camera_block(cfg, lab.meta, fit)
        if camera_block is not None and block != camera_block:
            raise HarnessConfigError("captures disagree on the camera block")
        camera_block = block
        spec = VisibilitySpec.from_camera_block(camera_block)
        cap = load_capture(cap_dir, cfg.section("telemetry"), 1.0,
                           float(lab.meta.get("lag_ms", 0.0)))  # fmt: skip
        by_file = {str(f.path.relative_to(cap.root)).replace("\\", "/"): f for f in cap.frames}
        rows = [r for r in lab.rows if window is None or is_held_out(r["stamp"], window)]
        if max_frames is not None:
            rows = rows[:max_frames]
        used = 0
        seq = []
        for r in rows:
            f = by_file.get(r["file"])
            if f is None:
                continue
            frame = Frame(load_image(f.path), f.stamp, f.index)
            dets = estimator.detect_frame(frame, f.state)
            if getattr(estimator, "inference_s", None) is not None:
                records.latency_s.append(float(estimator.inference_s))
            frame_records(
                capture=lab.name, row=r, dets=dets,
                pos=np.asarray(f.state.pos, np.float64), quat=np.asarray(f.state.quat, np.float64),
                spec=spec, aperture=aperture, out=records,
            )  # fmt: skip
            seq.append((f, dets))
            used += 1
        seq.sort(key=lambda x: x[0].stamp)
        replay_input.append((cap, seq, lab.meta.get("track")))
        cap_meta.append({
            "name": lab.name, "path": str(cap_dir.resolve()), "labels": str(lab.labels_dir),
            "map": map_name(lab.meta.get("track")), "track": lab.meta.get("track"),
            "frames_used": used, "window": window, "held_out": True,
            "split_source": "weights meta holdout" if window else "never trained on",
        })  # fmt: skip
    assert camera_block is not None
    model: dict[str, Any] = {
        "schema": "detector-noise",
        "version": 1,
        "revision": None,
        "fitted_utc": _now(),
        "git_head": git_head(find_repo_root()),
        "detector": {
            "id": did,
            "name": weights.name,
            "path": str(weights),
            "sha256": sha,
            "git_head": meta.get("git_head"),
            "purpose": meta.get("purpose", "evaluate"),
            "classes": list(meta.get("classes") or aperture),
        },  # fmt: skip
        "captures": cap_meta,
        "coverage": {
            "maps": sorted({c["map"] for c in cap_meta}),
            "classes": sorted(aperture),
        },
        "camera": camera_block,
        "tracker": {
            "assoc_radius": float(gcfg.get("assoc_radius", 1.0)),
            "lost_timeout_s": lost_timeout,
            "acquire_timeout_s": float(gcfg.get("acquire_timeout_s", 3.0)),
            "score_threshold": float(getattr(estimator, "score_threshold", 0.3)),
            "pass_area_frac": float(gcfg.get("pass_area_frac", 0.2)),
        },
    }
    model.update(fit_noise_model(records, fit, classes=aperture, lost_timeout_s=lost_timeout))
    model["latency"] = _latency_block(records.latency_s, device, cuda)
    model["timing"] = live_timing(live_runs) if live_runs else None
    rec_to = [c["temporal"]["lost_timeout_s_recommended"] for c in model["classes"].values()]
    model["recommended"] = {
        "lost_timeout_s": max([v for v in rec_to if v is not None], default=None),
        "rule": f"miss-burst p99 below {LOST_TIMEOUT_RANGE_M:g} m, clamped to "
        f"[{LOST_TIMEOUT_CLAMP_S[0]:g}, {LOST_TIMEOUT_CLAMP_S[1]:g}] s",
    }
    replay_res = None
    if replay and replay_input and replay_input[0][1]:
        cap0, seq0, track0 = replay_input[0]
        try:
            from drone_fly.harness.liftoff.gate_cv import camera_from_config

            course = load_catalogue_course(track0, trackdata_root)
            ks = truth_progress(cap0, course)
            cam = camera_from_config(cfg.section("camera"))
            replay_res = validation_replay(seq0, ks, course, model, model["tracker"], cam)
        except Exception as exc:  # noqa: BLE001 - the replay is diagnostic; report why it failed
            log.warning("validation replay skipped: %s", exc)
            replay_res = {"summary": {"skipped": str(exc)}}
    model_path = write_model(out_dir, model)
    plots = write_plots(out_dir, model, records)
    md, js = write_report(out_dir, model, records, fit, plots, replay_res)
    (out_dir / "records.json").write_text(
        json.dumps({"gates": records.gates, "false_positives": records.false_positives,
                    "frames": records.frames}, default=float),
        encoding="utf-8",
    )  # fmt: skip
    return {
        "out_dir": str(out_dir),
        "model_path": str(model_path),
        "report": str(md),
        "report_json": str(js),
        "detector_id": did,
        "model": model,
        "records": records,
        "replay": replay_res,
    }


def _latency_block(lat: list[float], device: str, cuda: bool) -> dict[str, Any]:
    a = np.asarray(lat, np.float64) * 1000.0
    return {
        "device": str(device),
        "cuda_synchronised": bool(cuda),
        "representative": bool(cuda),
        "p50_ms": float(np.percentile(a, 50)) if a.size else None,
        "p95_ms": float(np.percentile(a, 95)) if a.size else None,
        "n": int(a.size),
    }


def summary(res: dict[str, Any]) -> str:
    m = res["model"]
    lines = [f"noise model: {res['model_path']} (revision {m['revision']})"]
    for cls, c in m["classes"].items():
        par = c["parametric"]
        lines.append(
            f"{cls}: {c['n']} visible records, {c['n_detected']} detected "
            f"({c['n_detected'] / max(c['n'], 1):.1%}); model {c['model']}; radial sigma "
            f"{par['sigma_per_m'][0]:.3f}*range, yaw sigma {par['yaw_sigma_deg']:.1f} deg; "
            f"mean miss burst {c['temporal']['miss_burst_s_mean']:.2f} s, tau "
            f"{c['temporal']['ar1_tau_s']:.3f} s"
        )
    fp = m["false_positives"]
    lines.append(f"false positives {fp['rate_per_frame']:.2f}/frame; next mis-association "
                 f"{fp['next_misassoc_rate']:.2f}")  # fmt: skip
    lat = m["latency"]
    lines.append(f"latency {lat['device']} p50 {_fmt(lat['p50_ms'], '.1f')} ms"
                 f"{'' if lat['representative'] else ' (not representative)'}")  # fmt: skip
    lines.append(f"recommended lost_timeout_s: {_fmt(m['recommended']['lost_timeout_s'], '.2f')}")
    lines.append(f"report: {res['report']}")
    return "\n".join(lines)
