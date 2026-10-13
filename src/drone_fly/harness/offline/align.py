"""Projection-alignment score (UC71): does the recorded pose put the gates where the image has them?

For a set of sampled frames, every gate of the track (the gate-type table's classes) is projected
through the pose at ``frame stamp + offset`` and the capture's camera. Its visible edges -- the
outline polyline, offset by +-``tube_radius`` in the gate plane for tube gates -- are sampled
densely, and the **score** is the mean Sobel gradient component ACROSS the projected edge
(``|g . n|``, per-frame mean-normalised, downscaled image) along those points. The **baseline**
is the same score with the pose displaced by seeded random offsets (2-4 m, 5-15 deg yaw): a wrong
pose still lands on some edges, so the ratio ``margin = best / baseline`` is what tells a correct
pose from a wrong one.

A grid search covers the time offset (default +-1.5 s in 20 ms steps) and optionally the camera
uptilt (+-5 deg in 1 deg steps). ``verified`` requires all of:

* ``margin >= MARGIN_MIN`` (1.25, calibrated 2026-10-13 on the reference capture: the correct pose
  scores 1.40, a 3 deg uptilt error 1.09-1.15, a 0.5 s clock error 1.03-1.05; ADR 0010);
* the best offset is not on the edge of the searched range (an edge optimum means the true
  offset may lie outside it);
* at least ``MIN_FRAMES`` (50) sampled frames with a gate in view;
* **the configured camera is where the score peaks** (CP6b): at the best offset the score is
  re-evaluated over uptilt +-``CAM_UPTILT_RANGE`` (1 deg steps) x hfov +-``CAM_HFOV_RANGE``
  (5 deg steps) around the capture's camera; a best cell more than one step away on either axis
  means the camera is wrong ("camera off: best uptilt 12 vs meta 20"). It runs only when the
  margin passes (below it the score carries no camera signal). With ``--search-uptilt`` the
  camera under test has the searched uptilt, which ``verify_capture`` then writes to
  ``meta.camera`` (``source: align``). The absolute margin alone
  does not reject a wrong camera: the arch legs are vertical, so a pitch or scale error keeps
  many edge samples on them (QA, synthetic: a 10 deg hfov / uptilt error still scored 1.4-1.6).

Uses: time alignment (mandatory for a recording-XML pose), lag estimation (``lag_ms = -offset``,
replacing the hand-set ``--lag-ms``), and verifying the FPV camera and its parameters (a chase cam
or a wrong FOV / uptilt fails the margin or the camera check). CPU + OpenCV only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError, lazy_import
from drone_fly.harness.geometry import CameraModel

MARGIN_MIN = 1.25  # calibrated 2026-10-13 on the reference capture (ADR 0010)
MIN_FRAMES = 50
CAM_UPTILT_RANGE, CAM_UPTILT_STEP = 6.0, 1.0
CAM_HFOV_RANGE, CAM_HFOV_STEP = 10.0, 5.0
MAX_RANGE_M = 25.0
WORK_WIDTH = 640  # gradient images are computed at this width
SAMPLE_STEP_M = 0.1  # outline sampling step (gate-local metres)
BASELINE_DRAWS = 8


@dataclass
class AlignResult:
    offset_s: float
    uptilt_deg: float
    score: float
    baseline: float
    margin: float
    frames: int
    on_edge: bool
    verified: bool
    reasons: list[str] = field(default_factory=list)
    offsets_s: list[float] = field(default_factory=list)
    uptilts_deg: list[float] = field(default_factory=list)
    grid: list[list[float]] = field(default_factory=list)  # [uptilt][offset] scores
    camera: dict[str, Any] | None = None  # the camera check (see camera_check)

    @property
    def lag_ms(self) -> float:
        return -1000.0 * self.offset_s

    def alignment_block(self) -> dict[str, Any]:
        """The ``meta.json`` ``alignment`` block."""
        return {
            "verified": self.verified,
            "margin": round(self.margin, 4),
            "offset_s": round(self.offset_s, 4),
            "lag_ms": round(self.lag_ms, 1),
            "uptilt_deg": self.uptilt_deg,
            "score": round(self.score, 5),
            "baseline": round(self.baseline, 5),
            "frames": self.frames,
            "on_edge": self.on_edge,
            "reasons": list(self.reasons),
            "camera_check": None
            if self.camera is None
            else {
                k: self.camera[k]
                for k in (
                    "ok",
                    "best_uptilt_deg",
                    "best_hfov_deg",
                    "neighbours",
                    "meta_uptilt_deg",
                    "meta_hfov_deg",
                )
            },  # fmt: skip
            "method": f"edge-gradient margin >= {MARGIN_MIN}, >= {MIN_FRAMES} frames, no edge, "
            "camera at the score peak",
        }

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lag_ms"] = self.lag_ms
        return d


# ----------------------------------------------------------------------------- geometry


def edge_points_local(gtype: Any, step: float = SAMPLE_STEP_M) -> tuple[np.ndarray, np.ndarray]:
    """Unity-local ``(N, 3)`` points on a gate type's visible edges and their unit tangents."""
    line = np.asarray(gtype.outline_points(), np.float64)
    r = float(gtype.tube_radius)
    pts, tans = [], []
    for a, b in zip(line[:-1], line[1:], strict=True):
        d = b - a
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            continue
        k = max(3, int(math.ceil(n / step)) + 1)
        seg = a + np.linspace(0.0, 1.0, k)[1:-1, None] * d  # corners are ill-defined edges
        perp = np.array([-d[1], d[0]]) / n
        for o in (r, -r) if r > 0 else (0.0,):
            pts.append(seg + o * perp)
            tans.append(np.repeat((d / n)[None], len(seg), axis=0))
    p2 = np.concatenate(pts) if pts else np.zeros((0, 2))
    t2 = np.concatenate(tans) if tans else np.zeros((0, 2))
    z = np.zeros((len(p2), 1))
    return np.concatenate([p2, z], axis=1), np.concatenate([t2, z], axis=1)


TANGENT_M = 0.05


@dataclass(frozen=True)
class GateEdges:
    centre: np.ndarray  # aperture centre, contract world
    points: np.ndarray  # (N, 3) edge points, contract world
    ahead: np.ndarray  # (N, 3) the same points moved TANGENT_M along the edge


def gate_edge_world(gates: list[Any]) -> list[GateEdges]:
    """Per placed gate: its centre, edge points and tangent-shifted edge points (world)."""
    out = []
    for g in gates:
        p, t = edge_points_local(g.gtype)
        out.append(GateEdges(g.center_w, g.local_to_world(p), g.local_to_world(p + TANGENT_M * t)))
    return out


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q / (np.linalg.norm(q) + 1e-12))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, np.float64)
    return q / np.linalg.norm(q)


class PoseTrack:
    """Pose at an arbitrary time from capture samples (linear position, nlerp attitude)."""

    def __init__(self, stamps: np.ndarray, pos: np.ndarray, quat: np.ndarray, segments: Any):
        self.t = np.asarray(stamps, np.float64)
        self.pos = np.asarray(pos, np.float64)
        q = np.asarray(quat, np.float64).copy()
        for i in range(1, len(q)):  # hemisphere-continuous for nlerp
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
        self.quat = q
        self.seg = np.full(len(self.t), -1, np.int64)
        for k, (a, b) in enumerate(segments):
            self.seg[a:b] = k

    @classmethod
    def from_capture(cls, cap: Any) -> PoseTrack:
        return cls(
            cap.stamps,
            np.array([s.pos for s in cap.states]),
            np.array([s.quat for s in cap.states]),
            cap.segments,
        )

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray] | None:
        j = int(np.searchsorted(self.t, t))
        if j <= 0 or j >= len(self.t):
            return None
        a, b = j - 1, j
        if self.seg[a] < 0 or self.seg[a] != self.seg[b]:
            return None
        u = (t - self.t[a]) / max(self.t[b] - self.t[a], 1e-9)
        p = self.pos[a] + u * (self.pos[b] - self.pos[a])
        q = self.quat[a] + u * (self.quat[b] - self.quat[a])
        return p, q / np.linalg.norm(q)


# ----------------------------------------------------------------------------- images


def gradient_image(image: np.ndarray, width: int = WORK_WIDTH) -> np.ndarray:
    """Downscaled Sobel gradient ``(H, W, 2)`` (gx, gy), normalised by its mean magnitude."""
    cv2 = lazy_import("cv2", package="opencv-python", purpose="projection alignment")
    h, w = image.shape[:2]
    scale = width / float(w)
    small = cv2.resize(image, (width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    gray = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return np.stack([gx, gy], axis=-1) / max(float(mag.mean()), 1e-6)


@dataclass
class AlignFrame:
    stamp: float
    grad: np.ndarray
    scale: float  # work pixels per frame pixel


# ----------------------------------------------------------------------------- scoring


def frame_score(
    fr: AlignFrame,
    camera: CameraModel,
    masks: Any,
    edges: list[GateEdges],
    pos: np.ndarray,
    quat: np.ndarray,
) -> tuple[float, int]:
    """``(sum, count)`` of the image gradient ACROSS each projected edge (``|g . n|``)."""
    R = camera.world_to_camera(quat)
    gh, gw = fr.grad.shape[:2]
    total, n = 0.0, 0
    for e in edges:
        if float(np.linalg.norm(e.centre - pos)) > MAX_RANGE_M:
            continue
        px, front = camera.project((e.points - pos) @ R.T)
        ok = camera.in_image(px, front)
        if masks:
            ok &= ~masks.contains(px, camera.width, camera.height)
        if not ok.any():
            continue
        ax, _ = camera.project((e.ahead[ok] - pos) @ R.T)
        tan = ax - px[ok]
        nrm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        q = px[ok] * fr.scale
        xi = np.clip(q[:, 0].astype(np.int64), 0, gw - 1)
        yi = np.clip(q[:, 1].astype(np.int64), 0, gh - 1)
        total += float(np.abs((fr.grad[yi, xi] * nrm).sum(axis=1)).sum())
        n += int(ok.sum())
    return total, n


def score_pose_set(
    frames: list[AlignFrame],
    poses: list[tuple[np.ndarray, np.ndarray] | None],
    camera: CameraModel,
    masks: Any,
    edges: list[GateEdges],
) -> tuple[float, int]:
    """Mean of per-frame mean edge gradients over frames with a gate in view; ``(score, n)``."""
    vals = []
    for fr, pose in zip(frames, poses, strict=True):
        if pose is None:
            continue
        s, n = frame_score(fr, camera, masks, edges, pose[0], pose[1])
        if n >= 20:
            vals.append(s / n)
    return (float(np.mean(vals)) if vals else 0.0), len(vals)


def _yaw(q: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0, 0, 1.0]])
    return _matrix_to_quat_wxyz(Rz @ _quat_wxyz_to_matrix(q))


def baseline_score(
    frames: list[AlignFrame],
    poses: list[tuple[np.ndarray, np.ndarray] | None],
    camera: CameraModel,
    masks: Any,
    edges: list[GateEdges],
    seed: int = 0,
    draws: int = BASELINE_DRAWS,
) -> float:
    """Mean score at seeded random pose displacements (2-4 m, 5-15 deg yaw, random signs)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        moved = []
        for pose in poses:
            if pose is None:
                moved.append(None)
                continue
            d = rng.normal(size=3)
            d *= rng.uniform(2.0, 4.0) / (np.linalg.norm(d) + 1e-12)
            yaw = rng.uniform(5.0, 15.0) * rng.choice([-1.0, 1.0])
            moved.append((pose[0] + d, _yaw(pose[1], yaw)))
        s, n = score_pose_set(frames, moved, camera, masks, edges)
        if n:
            vals.append(s)
    return float(np.mean(vals)) if vals else 0.0


def search(
    frames: list[AlignFrame],
    track: PoseTrack,
    camera: CameraModel,
    masks: Any,
    edges: list[GateEdges],
    *,
    offsets_s: np.ndarray,
    uptilts_deg: np.ndarray,
    seed: int = 0,
) -> AlignResult:
    """Grid search (uptilt x offset), baseline at the best cell, and the verdict."""
    grid = np.zeros((len(uptilts_deg), len(offsets_s)))
    counts = np.zeros_like(grid, dtype=np.int64)
    cams = [
        CameraModel(camera.width, camera.height, camera.hfov_deg, float(u)) for u in uptilts_deg
    ]
    for j, off in enumerate(offsets_s):
        poses = [track.at(fr.stamp + float(off)) for fr in frames]
        for i, cam in enumerate(cams):
            grid[i, j], counts[i, j] = score_pose_set(frames, poses, cam, masks, edges)
    i, j = np.unravel_index(int(np.argmax(grid)), grid.shape)
    best_off, best_up = float(offsets_s[j]), float(uptilts_deg[i])
    poses = [track.at(fr.stamp + best_off) for fr in frames]
    base = baseline_score(frames, poses, cams[i], masks, edges, seed)
    margin = float(grid[i, j] / base) if base > 0 else 0.0
    on_edge = (len(offsets_s) > 1 and j in (0, len(offsets_s) - 1)) or (
        len(uptilts_deg) > 1 and i in (0, len(uptilts_deg) - 1)
    )
    reasons = []
    if margin < MARGIN_MIN:
        reasons.append(f"margin {margin:.2f} < {MARGIN_MIN}")
    if on_edge:
        reasons.append("best cell on the edge of the search range")
    if counts[i, j] < MIN_FRAMES:
        reasons.append(f"{int(counts[i, j])} frames with a gate in view < {MIN_FRAMES}")
    return AlignResult(
        offset_s=best_off,
        uptilt_deg=best_up,
        score=float(grid[i, j]),
        baseline=base,
        margin=margin,
        frames=int(counts[i, j]),
        on_edge=bool(on_edge),
        verified=not reasons,
        reasons=reasons,
        offsets_s=[float(v) for v in offsets_s],
        uptilts_deg=[float(v) for v in uptilts_deg],
        grid=np.round(grid, 5).tolist(),
    )


def camera_check(
    frames: list[AlignFrame],
    track: PoseTrack,
    camera: CameraModel,
    masks: Any,
    edges: list[GateEdges],
    offset_s: float,
) -> dict[str, Any]:
    """Re-score at ``offset_s`` over uptilt x hfov around ``camera``; ``ok`` when the peak is
    within one step of it on both axes (see the module docstring)."""
    poses = [track.at(fr.stamp + offset_s) for fr in frames]
    ups = camera.uptilt_deg + _grid(CAM_UPTILT_RANGE, CAM_UPTILT_STEP, True)
    hfs = camera.hfov_deg + _grid(CAM_HFOV_RANGE, CAM_HFOV_STEP, True)
    hfs = hfs[(hfs > 1.0) & (hfs < 179.0)]
    grid = np.zeros((len(ups), len(hfs)))
    for i, u in enumerate(ups):
        for j, h in enumerate(hfs):
            cam = CameraModel(camera.width, camera.height, float(h), float(u))
            grid[i, j] = score_pose_set(frames, poses, cam, masks, edges)[0]
    i, j = np.unravel_index(int(np.argmax(grid)), grid.shape)
    bu, bh = float(ups[i]), float(hfs[j])
    reasons = []
    if abs(bu - camera.uptilt_deg) > CAM_UPTILT_STEP + 1e-9:
        reasons.append(f"camera off: best uptilt {bu:g} vs meta {camera.uptilt_deg:g}")
    if abs(bh - camera.hfov_deg) > CAM_HFOV_STEP + 1e-9:
        reasons.append(f"camera off: best hfov {bh:g} vs meta {camera.hfov_deg:g}")
    # evidence: each axis neighbour's score relative to the camera under test (the grid centre)
    ci, cj = (
        int(np.argmin(np.abs(ups - camera.uptilt_deg))),
        int(np.argmin(np.abs(hfs - camera.hfov_deg))),
    )
    centre = float(grid[ci, cj])
    neighbours = {}
    for d in (-6, -3, 3, 6):
        k = ci + int(round(d / CAM_UPTILT_STEP))
        if 0 <= k < len(ups):
            neighbours[f"uptilt{d:+d}"] = round(float(grid[k, cj]) / max(centre, 1e-12), 4)
    for d in (-10, -5, 5, 10):
        k = cj + int(round(d / CAM_HFOV_STEP))
        if 0 <= k < len(hfs):
            neighbours[f"hfov{d:+d}"] = round(float(grid[ci, k]) / max(centre, 1e-12), 4)
    return {
        "ok": not reasons,
        "reasons": reasons,
        "neighbours": neighbours,
        "best_uptilt_deg": bu,
        "best_hfov_deg": bh,
        "meta_uptilt_deg": camera.uptilt_deg,
        "meta_hfov_deg": camera.hfov_deg,
        "uptilts_deg": [float(v) for v in ups],
        "hfovs_deg": [float(v) for v in hfs],
        "grid": np.round(grid, 5).tolist(),
    }


# ----------------------------------------------------------------------------- capture level


def _grid(half: float, step: float, enabled: bool, centre: float = 0.0) -> np.ndarray:
    if not enabled:
        return np.array([centre])
    n = int(round(half / step))
    return centre + step * np.arange(-n, n + 1)


def load_align_frames(
    cap: Any, track: PoseTrack, edges: list[Any], camera: CameraModel, masks: Any,
    max_frames: int, offset_s: float = 0.0,
) -> list[AlignFrame]:  # fmt: skip
    """Up to ``max_frames`` evenly spaced frames with a gate in view at ``offset_s``."""
    from drone_fly.harness.offline.evaluate import load_image

    cand = []
    for f in cap.frames:  # paired, static-trimmed (a parked drone says nothing about the lag)
        stamp, path = f.stamp, f.path
        pose = track.at(stamp + offset_s)
        if pose is None or not path.is_file():
            continue
        R = camera.world_to_camera(pose[1])
        for e in edges:
            if float(np.linalg.norm(e.centre - pose[0])) > MAX_RANGE_M:
                continue
            px, front = camera.project(((e.centre - pose[0]) @ R.T)[None])
            if camera.in_image(px, front)[0]:
                cand.append((stamp, path))
                break
    if not cand:
        return []
    pick = np.unique(np.linspace(0, len(cand) - 1, min(max_frames, len(cand))).astype(int))
    out = []
    for k in pick:
        stamp, path = cand[k]
        img = load_image(path)
        g = gradient_image(img)
        out.append(AlignFrame(stamp, g, g.shape[1] / float(img.shape[1])))
    return out


def align_capture(
    capture_dir: str | Path,
    *,
    config: Any = None,
    preset: str | None = "liftoff",
    trackdata_root: str | Path | None = None,
    track_xml: str | Path | None = None,
    search_lag: bool = True,
    search_uptilt: bool = False,
    lag_range_s: float = 1.5,
    lag_step_s: float = 0.02,
    uptilt_range_deg: float = 5.0,
    uptilt_step_deg: float = 1.0,
    max_frames: int = 120,
    seed: int = 0,
    uptilt_bias_deg: float = 0.0,
    offset_bias_s: float = 0.0,
    check_camera: bool = True,
) -> AlignResult:
    """Score a capture's alignment (see the module docstring).

    The capture's ``meta.json`` gives the track and camera when present; ``track_xml`` overrides
    the track (e.g. an unregistered capture). ``uptilt_bias_deg`` / ``offset_bias_s`` shift the
    evaluated camera / clock (threshold calibration: score deliberately wrong poses).
    ``check_camera`` runs :func:`camera_check` at the best offset (part of the verdict).
    """
    from drone_fly.harness.config import load_config
    from drone_fly.harness.gate_types import GateTypeTable
    from drone_fly.harness.liftoff.gate_cv import camera_from_config
    from drone_fly.harness.liftoff.masks import FrameMasks
    from drone_fly.harness.offline.autolabel import placed_gates
    from drone_fly.harness.offline.capture import load_capture
    from drone_fly.harness.offline.capture_index import META_FILE, camera_config, read_meta
    from drone_fly.harness.offline.track_xml import parse_track_items

    d = Path(capture_dir)
    cfg = load_config("liftoff", config, preset=preset)
    meta = read_meta(d) if (d / META_FILE).is_file() else None
    if track_xml is None:
        if meta is None:
            raise HarnessConfigError(f"{d} has no meta.json: pass the track XML")
        from drone_fly.tracks import catalogue as cat

        root = cat.resolve_root(trackdata_root, cat.load_named("liftoff").header)
        track_xml = root / meta["track"]["file"]
    cam_cfg = camera_config(cfg.section("camera"), meta)
    camera = camera_from_config(cam_cfg)
    camera = CameraModel(camera.width, camera.height, camera.hfov_deg,
                         camera.uptilt_deg + uptilt_bias_deg)  # fmt: skip
    masks = FrameMasks.from_config(cam_cfg)
    table = GateTypeTable.from_config(cfg.get("gate_types"))
    gates = placed_gates(parse_track_items(track_xml), table)
    if not gates:
        raise HarnessConfigError(f"no gate of the gate-type table in {track_xml}")
    edges = gate_edge_world(gates)
    cap = load_capture(d, cfg.section("telemetry"), 1.0, static_trim=True)
    track = PoseTrack.from_capture(cap)
    centre_off = float((meta or {}).get("time", {}).get("offset_s", 0.0)) + offset_bias_s
    frames = load_align_frames(cap, track, edges, camera, masks, max_frames, centre_off)
    offsets = _grid(lag_range_s, lag_step_s, search_lag, centre_off)
    uptilts = camera.uptilt_deg + _grid(uptilt_range_deg, uptilt_step_deg, search_uptilt)
    res = search(frames, track, camera, masks, edges, offsets_s=offsets, uptilts_deg=uptilts,
                 seed=seed)  # fmt: skip
    if check_camera and frames and res.margin >= MARGIN_MIN:
        # the camera under test: the capture's, with the searched uptilt when --search-uptilt
        # measured it (verify_capture then records that uptilt in meta.camera)
        tested = CameraModel(camera.width, camera.height, camera.hfov_deg, res.uptilt_deg)
        res.camera = camera_check(frames, track, tested, masks, edges, res.offset_s)
        res.reasons += res.camera["reasons"]
        res.verified = not res.reasons
    return res


def verify_capture(capture_dir: str | Path, *, write: bool = True, **kw: Any) -> AlignResult:
    """:func:`align_capture` and record the result in the capture's ``meta.json``."""
    from drone_fly.harness.offline.capture_index import read_meta, write_meta

    res = align_capture(capture_dir, **kw)
    if write:
        meta = read_meta(capture_dir)
        meta["alignment"] = res.alignment_block()
        if kw.get("search_uptilt"):
            meta["alignment"]["uptilt_searched"] = True
            if res.verified and res.uptilt_deg != meta["camera"]["uptilt_deg"]:
                meta["camera"]["uptilt_deg"] = res.uptilt_deg  # measured; was the preset's
                meta["camera"]["source"] = "align"
        write_meta(capture_dir, meta)
    return res


def calibration_report(
    capture_dir: str | Path, *, uptilt_error_deg: float = 3.0, lag_error_s: float = 0.5, **kw: Any
) -> dict[str, Any]:
    """Margins of the best pose vs deliberately wrong ones (the ``MARGIN_MIN`` evidence).

    Runs the lag search, then re-scores (no search) at the best offset with the uptilt off by
    +-``uptilt_error_deg`` and at the best uptilt with the clock off by +-``lag_error_s``. A usable
    threshold sits between the correct margin and the wrong ones.
    """
    best = align_capture(capture_dir, search_lag=True, **kw)
    rows = {"correct": best.margin}
    for name, du, dt in (
        (f"uptilt+{uptilt_error_deg:g}", uptilt_error_deg, 0.0),
        (f"uptilt-{uptilt_error_deg:g}", -uptilt_error_deg, 0.0),
        (f"lag+{lag_error_s:g}", 0.0, lag_error_s),
        (f"lag-{lag_error_s:g}", 0.0, -lag_error_s),
    ):
        r = align_capture(capture_dir, search_lag=False, uptilt_bias_deg=du,
                          offset_bias_s=best.offset_s + dt, check_camera=False, **kw)  # fmt: skip
        rows[name] = r.margin
    return {"best_offset_s": best.offset_s, "frames": best.frames, "margins": rows,
            "margin_min": MARGIN_MIN}  # fmt: skip
