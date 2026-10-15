"""AC2: exact 40-ray labels for captured frames (``drone-fly eyes label``; offline).

For every frame of a capture with a valid telemetry pose:

1. **Pose.** :func:`~drone_fly.harness.offline.capture.load_capture` with the preset's telemetry
   config and ``world_scale = W`` (``scaling.world_scale``, 4.0 for ``liftoff``), i.e. the SAME
   converter the live state source uses, so each :class:`~drone_fly.harness.types.FlightState` is
   what the estimator gets live (training scale).
2. **Course.** The capture's catalogue entry (``meta.json`` ``catalogue_id``; its map must match)
   through :func:`drone_fly.tracks.loader.load_course` with ``ConversionConfig.world_scale == W``
   (any other value is refused), so the course frame is ``(p - origin) / W`` -- a pure
   translation of the FlightState: the raycast position is ``state.pos - origin / W``. Never
   apply ``to_course_point`` to a FlightState (it is already divided by ``W``).
3. **Target.** :class:`~drone_fly.sim.pool.RouteProgress` + ``gate_crossed`` over every telemetry
   sample of each segment (a new segment restarts the route); it orients the cone below 0.3 m/s.
4. **Label = the training raycast.** ``contract.raycast`` against the harness obstacle list
   (``to_harness_course(course).raycast_obstacles()``, the list the UC69 oracle uses) -- obstacles
   only, never gates or the floor, exactly as the 70 model's training sim. ``ray_range_scale``
   ``s`` is honoured by dividing every length by ``s`` before the cast (range ``RAY_RANGE * s``
   training m; ``RAY_RANGE * s * W`` Liftoff m). ``free`` in ``[0, 1]`` (1 = clear),
   ``dist_m`` in Liftoff metres, ``hit`` and the hit obstacle's index / minimum half-extent.
5. **Diagnostic channel** (AC2's hit kind): the same rays against the FULL geometry -- oriented
   boxes (slab), capsules, the obstacle AABBs, the floor plane / heightfield (march) -- give
   ``aux_free`` and ``aux_kind`` (:data:`AUX_KINDS`). An item primitive whose centre lies within a
   passage's outer half-size + 1 m (Liftoff) of the passage centre counts as ``gate``.
   ``aux_*`` is never the training target; it shows what the label leaves out.
6. **Rays through the camera + ego-motion** from the FlightState via the deployable functions
   (:func:`~drone_fly.eyes.camera_rays.ray_view`, :func:`~drone_fly.eyes.ego.ego_vector` with the
   stride-1 history frame of :func:`~drone_fly.eyes.history.select_history`).

Per frame it also stores the cone's forward source, the frame-telemetry pair ``|dt|``, the speed
(Liftoff m/s) and the warning time ``range_m / speed`` (the time the sensing range buys at that
speed; UC-73 AC10's range-vs-speed evidence).

Captures are skipped with a reason (``no-meta``, ``map-mismatch``, ``no-geometry``,
``unconvertible: ...``, ``no-telemetry``). Writes ``eyes_labels.npz`` + ``eyes_labels_meta.json``
(default ``<data root>/labels/eyes/<capture>/``) and, with ``--preview N``, up to 20 overlays.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.eyes.camera_rays import FWD_SOURCES, camera_meta, cone_namespace, ray_view
from drone_fly.eyes.capture_meta import CaptureMeta, capture_sha, read_capture_meta
from drone_fly.eyes.ego import EGO_DIM, ego_vector
from drone_fly.eyes.fov import array_sha, contract_sha, offline_contract, preset_view
from drone_fly.eyes.history import HIST_TOL_S, select_history
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.geometry import CameraModel, ray_directions
from drone_fly.harness.types import FlightState

log = logging.getLogger(__name__)

LABELS_FILE = "eyes_labels.npz"
META_FILE = "eyes_labels_meta.json"
LABEL_VERSION = 1
HIT_RULE = (
    "label = contract.raycast against the harness obstacle list (placed non-gate items, AABBs): "
    "obstacles only, as the 70-input model's training sim; AC2 hit kind = aux_kind (full geometry)"
)
AUX_KINDS = ("none", "obstacle", "gate", "scene", "item", "floor")
AUX_NONE, AUX_OBSTACLE, AUX_GATE, AUX_SCENE, AUX_ITEM, AUX_FLOOR = range(6)
#: An item primitive within a passage's outer half-size + this (Liftoff m) is a ``gate``.
AUX_GATE_MARGIN_M = 1.0
EGO_DT_S = 0.1
PAIR_TOL_S = 0.03
MAX_PREVIEW = 20
_EPS = 1e-3


@dataclass(frozen=True)
class LabelSettings:
    """Everything that changes a label; its sha must agree across a dataset's captures."""

    world_scale: float
    ray_range_scale: float
    ray_range: float
    n_rays: int
    cone_sha256: str
    contract_sha256: str
    pair_tol_s: float = PAIR_TOL_S
    ego_dt_s: float = EGO_DT_S
    hist_tol_s: float = HIST_TOL_S
    aux_gate_margin_m: float = AUX_GATE_MARGIN_M
    hit_rule: str = HIT_RULE
    version: int = LABEL_VERSION

    @property
    def range_train(self) -> float:
        """The scaled sensing range in training metres (``RAY_RANGE * s``)."""
        return float(self.ray_range) * float(self.ray_range_scale)

    @property
    def range_m(self) -> float:
        """The scaled sensing range in Liftoff metres (``RAY_RANGE * s * W``)."""
        return self.range_train * float(self.world_scale)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def settings_for(contract: Any, world_scale: float, ray_range_scale: float) -> LabelSettings:
    return LabelSettings(
        world_scale=float(world_scale),
        ray_range_scale=float(ray_range_scale),
        ray_range=float(contract.RAY_RANGE),
        n_rays=int(contract.N_RAYS),
        cone_sha256=array_sha(contract.CONE),
        contract_sha256=contract_sha(contract),
    )


# ----------------------------------------------------------------------------- ray casts


def scale_obstacles(obstacles: list[dict[str, Any]], s: float) -> list[dict[str, Any]]:
    """Raycast dicts with every length divided by ``s`` (the ``ray_range_scale`` trick)."""
    return [
        {"center": np.asarray(o["center"], np.float32) / np.float32(s),
         "half_extents": np.asarray(o["half_extents"], np.float32) / np.float32(s),
         "type": o.get("type", "cylinder")}
        for o in obstacles
    ]  # fmt: skip


def _aabb_distance(p: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.maximum(np.abs(p[None] - center) - half, 0.0), axis=1)


def ray_aabb(o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """``(R, P)`` hit distances (``inf`` = miss) of rays ``o + t d`` against AABBs (contract rule:
    the entry distance, or the exit distance from inside)."""
    d = np.asarray(d, np.float64)[:, None, :]
    ds = np.where(np.abs(d) < 1e-6, 1e-6, d)
    t1 = (lo[None] - o) / ds
    t2 = (hi[None] - o) / ds
    tmn = np.minimum(t1, t2).max(-1)
    tmx = np.maximum(t1, t2).min(-1)
    t = np.where(tmn > _EPS, tmn, tmx)
    ok = (tmx >= np.maximum(tmn, 0.0)) & (tmx > _EPS)
    return np.where(ok, t, np.inf)


def ray_obb(o: np.ndarray, d: np.ndarray, c: np.ndarray, rot: np.ndarray, half: np.ndarray):
    """``(R, P)`` hit distances against oriented boxes (``rot`` columns = local axes)."""
    ol = np.einsum("pji,pj->pi", rot, o[None] - c)  # (P, 3)
    dl = np.einsum("pji,rj->rpi", rot, np.asarray(d, np.float64))  # (R, P, 3)
    ds = np.where(np.abs(dl) < 1e-9, 1e-9, dl)
    t1 = (-half[None] - ol[None]) / ds
    t2 = (half[None] - ol[None]) / ds
    tmn = np.minimum(t1, t2).max(-1)
    tmx = np.maximum(t1, t2).min(-1)
    t = np.where(tmn > _EPS, tmn, tmx)
    ok = (tmx >= np.maximum(tmn, 0.0)) & (tmx > _EPS)
    return np.where(ok, t, np.inf)


def ray_capsule(o: np.ndarray, d: np.ndarray, c: np.ndarray, axis: np.ndarray, half_len, radius):
    """``(R, P)`` entry distances against capsules (axis unit vectors, half length, radius)."""
    d = np.asarray(d, np.float64)
    hl = np.asarray(half_len, np.float64)[None]
    r2 = np.asarray(radius, np.float64)[None] ** 2
    oc = o[None] - c  # (P, 3)
    da = d @ axis.T  # (R, P)
    oca = np.einsum("pi,pi->p", oc, axis)[None]  # (1, P)
    dd = d[:, None, :] - da[..., None] * axis[None]
    od = oc - oca.T * axis  # (P, 3)
    qa = (dd * dd).sum(-1)
    qb = 2 * (dd * od[None]).sum(-1)
    qc = (od * od).sum(-1)[None] - r2
    disc = qb * qb - 4 * qa * qc
    safe = np.where(qa > 1e-12, 2 * qa, 1.0)
    t = (-qb - np.sqrt(np.maximum(disc, 0.0))) / safe
    axial = oca + t * da
    best = np.where((disc >= 0) & (qa > 1e-12) & (t > _EPS) & (np.abs(axial) <= hl), t, np.inf)
    for sign in (-1.0, 1.0):
        s = c + axis * (sign * hl.T)  # (P, 3) end-sphere centres
        os_ = o[None] - s
        b = d @ os_.T  # (R, P)
        cc = (os_ * os_).sum(-1)[None] - r2
        dsc = b * b - cc
        ts = -b - np.sqrt(np.maximum(dsc, 0.0))
        best = np.minimum(best, np.where((dsc >= 0) & (ts > _EPS), ts, np.inf))
    return best


@dataclass
class AuxGeometry:
    """The full course geometry for the diagnostic channel (course frame, training scale)."""

    box_c: np.ndarray
    box_rot: np.ndarray
    box_half: np.ndarray
    box_kind: np.ndarray
    cap_c: np.ndarray
    cap_axis: np.ndarray
    cap_half: np.ndarray
    cap_r: np.ndarray
    cap_kind: np.ndarray
    obs_lo: np.ndarray
    obs_hi: np.ndarray
    floor_z: float
    heightfield: Any = None
    course: Any = None

    @classmethod
    def from_course(
        cls, course: Any, obstacles: list[dict[str, Any]], gate_margin_m: float, world_scale: float
    ) -> AuxGeometry:
        from drone_fly.sim.pool import PRIM_BOX
        from drone_fly.tracks.liftoff import SOURCE_SCENE

        p = course.prims
        kind = np.where(p.source == SOURCE_SCENE, AUX_SCENE, AUX_ITEM).astype(np.int8)
        if len(p) and course.passages:
            pc = np.array([q.center for q in course.passages])
            reach = np.array([max(q.w, q.h) / 2.0 for q in course.passages])
            reach = reach + gate_margin_m / float(world_scale)
            dist = np.linalg.norm(p.center[:, None, :] - pc[None], axis=2)
            near = (dist <= reach[None]).any(axis=1)
            kind = np.where((kind == AUX_ITEM) & near, AUX_GATE, kind).astype(np.int8)
        box = p.kind == PRIM_BOX
        cap = ~box
        oc = np.array([o["center"] for o in obstacles], np.float64).reshape(-1, 3)
        oh = np.array([o["half_extents"] for o in obstacles], np.float64).reshape(-1, 3)
        oh = np.where(np.array([o.get("type") == "cylinder" for o in obstacles])[:, None],
                      oh[:, [0, 0, 2]], oh) if len(obstacles) else oh  # fmt: skip
        return cls(
            p.center[box], p.rot[box], p.size[box], kind[box],
            p.center[cap], p.rot[cap][:, :, 2] if cap.any() else np.zeros((0, 3)),
            p.size[cap][:, 1], p.size[cap][:, 0], kind[cap],
            oc - oh, oc + oh, float(course.floor_z), course.heightfield, course,
        )  # fmt: skip

    def cast(self, o: np.ndarray, rays: np.ndarray, rng: float) -> tuple[np.ndarray, np.ndarray]:
        """``(t (R,), kind (R,))``: the nearest surface within ``rng`` along each ray."""
        o = np.asarray(o, np.float64)
        n = len(rays)
        best = np.full(n, np.inf)
        kind = np.zeros(n, np.int8)

        def take(t: np.ndarray, k: np.ndarray) -> None:
            if t.size == 0:
                return
            j = np.argmin(t, axis=1)
            tm = t[np.arange(n), j]
            better = tm < best
            best[better] = tm[better]
            kind[better] = k[j[better]]

        if len(self.box_c):
            near = np.linalg.norm(self.box_c - o, axis=1) <= rng + np.linalg.norm(
                self.box_half, axis=1
            )
            if near.any():
                take(ray_obb(o, rays, self.box_c[near], self.box_rot[near], self.box_half[near]),
                     self.box_kind[near])  # fmt: skip
        if len(self.cap_c):
            near = np.linalg.norm(self.cap_c - o, axis=1) <= rng + self.cap_half + self.cap_r
            if near.any():
                t = ray_capsule(o, rays, self.cap_c[near], self.cap_axis[near],
                                self.cap_half[near], self.cap_r[near])  # fmt: skip
                take(t, self.cap_kind[near])
        if len(self.obs_lo):
            take(ray_aabb(o, rays, self.obs_lo, self.obs_hi),
                 np.full(len(self.obs_lo), AUX_OBSTACLE, np.int8))  # fmt: skip
        tf = self.floor_hit(o, rays, rng)
        better = tf < best
        best[better] = tf[better]
        kind[better] = AUX_FLOOR
        kind[best > rng] = AUX_NONE
        return np.minimum(best, rng), kind

    def floor_hit(self, o: np.ndarray, rays: np.ndarray, rng: float, steps: int = 128):
        """Floor plane (analytic) or heightfield (march + linear refinement) hit distances."""
        dz = rays[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(dz < -1e-9, (self.floor_z - o[2]) / dz, np.inf)
        t = np.where(t > _EPS, t, np.inf)
        if self.heightfield is None or self.course is None:
            return t
        from drone_fly.sim.pool import floor_height

        ts = np.linspace(0.0, rng, steps + 1)
        pts = o[None, None] + ts[None, :, None] * rays[:, None, :]  # (R, S, 3)
        gap = pts[..., 2] - floor_height(self.course, pts[..., 0], pts[..., 1])
        below = gap < 0
        out = np.full(len(rays), np.inf)
        for r in np.flatnonzero(below.any(axis=1)):
            k = int(np.argmax(below[r]))
            if k == 0:
                out[r] = _EPS
                continue
            g0, g1 = gap[r, k - 1], gap[r, k]
            out[r] = ts[k - 1] + (ts[k] - ts[k - 1]) * g0 / max(g0 - g1, 1e-12)
        return np.minimum(out, t)


# ----------------------------------------------------------------------------- core


def route_targets(states: list[FlightState], segments: list[tuple[int, int]], course: Any,
                  origin_c: np.ndarray) -> np.ndarray:  # fmt: skip
    """Per telemetry sample, the index of the target passage (route restarts per segment)."""
    from drone_fly.sim.pool import RouteProgress, gate_crossed

    out = np.full(len(states), -1, np.int64)
    for a, b in segments:
        prog = RouteProgress(course.n_passages, course.laps, course.circuit)
        prev = None
        for i in range(a, b):
            pos = np.asarray(states[i].pos, np.float64) - origin_c
            if prev is not None and not prog.finished:
                if gate_crossed(prev, pos, course.passages[prog.target], 0.0):
                    prog.advance()
            prev = pos
            out[i] = prog.k % course.n_passages
    return out


def compute_labels(
    states: list[FlightState],
    stamps: np.ndarray,
    segments: np.ndarray,
    targets_c: np.ndarray | None,
    origin_c: np.ndarray,
    obstacles: list[dict[str, Any]],
    contract: Any,
    settings: LabelSettings,
    camera: CameraModel,
    masks: list[list[float]] = (),
    aux: AuxGeometry | None = None,
) -> dict[str, np.ndarray]:
    """Label arrays for frames (see the module docstring); ``targets_c`` rows in the course frame.

    ``states`` are the frames' FlightStates (contract frame, training scale), ``origin_c`` the
    course origin divided by ``W`` (so ``course point = state.pos - origin_c``) and ``obstacles``
    the harness raycast dicts in the course frame.
    """
    n, nr = len(states), int(settings.n_rays)
    s, W = float(settings.ray_range_scale), float(settings.world_scale)
    rng_t = settings.range_train
    cone = cone_namespace(contract.CONE)
    obs_s = scale_obstacles(obstacles, s)
    oc = np.array([o["center"] for o in obstacles], np.float64).reshape(-1, 3)
    oh = np.array([o["half_extents"] for o in obstacles], np.float64).reshape(-1, 3)
    o_min_half = oh.min(axis=1) if len(oh) else np.zeros(0)
    origin_c = np.asarray(origin_c, np.float64)
    out = {
        "free": np.ones((n, nr), np.float32),
        "dist_m": np.full((n, nr), settings.range_m, np.float32),
        "hit": np.zeros((n, nr), bool),
        "hit_obstacle": np.full((n, nr), -1, np.int32),
        "hit_min_half_m": np.full((n, nr), np.nan, np.float32),
        "aux_free": np.ones((n, nr), np.float32),
        "aux_kind": np.zeros((n, nr), np.int8),
        "rays_world": np.zeros((n, nr, 3), np.float32),
        "rays_cam": np.zeros((n, nr, 3), np.float32),
        "uv": np.zeros((n, nr, 2), np.float32),
        "in_view": np.zeros((n, nr), bool),
        "ego": np.zeros((n, EGO_DIM), np.float32),
        "ego_history": np.zeros(n, bool),
        "fwd_source": np.zeros(n, np.int8),
        "pos_c": np.zeros((n, 3), np.float32),
        "vel": np.zeros((n, 3), np.float32),
        "quat": np.zeros((n, 4), np.float32),
        "speed_mps": np.zeros(n, np.float32),
        "warning_s": np.full(n, np.inf, np.float32),
    }
    for i, st in enumerate(states):
        pos_c = np.asarray(st.pos, np.float64) - origin_c  # NOT to_course_point (already / W)
        vel = np.asarray(st.vel, np.float32)
        tgt_c = None if targets_c is None or not np.all(np.isfinite(targets_c[i])) else targets_c[i]
        # the training raycast (lengths / s)
        sd = {"pos": (pos_c / s).astype(np.float32), "vel": vel}
        tgt_s = None if tgt_c is None else (np.asarray(tgt_c) / s).astype(np.float32)
        free = np.asarray(contract.raycast(sd, obs_s, tgt_s), np.float32)
        hit = free < 1.0
        out["free"][i] = free
        out["hit"][i] = hit
        out["dist_m"][i] = free * settings.range_m
        if hit.any() and len(obs_s):
            cand = np.flatnonzero(_aabb_distance(pos_c, oc, oh) <= rng_t + 1e-6)
            per = np.stack([np.asarray(contract.raycast(sd, [obs_s[j]], tgt_s)) for j in cand])
            j = cand[np.argmin(per, axis=0)]
            out["hit_obstacle"][i] = np.where(hit, j, -1)
            out["hit_min_half_m"][i] = np.where(hit, o_min_half[j] * W, np.nan)
        # rays through the camera, from the contract-frame state (as the live estimator)
        tgt_w = None if tgt_c is None else (np.asarray(tgt_c) + origin_c).astype(np.float32)
        view = ray_view(st.pos, st.vel, st.quat, tgt_w, cone, camera, masks)
        out["rays_world"][i] = view.rays_world
        out["rays_cam"][i] = view.rays_cam
        out["uv"][i] = view.uv
        out["in_view"][i] = view.in_view
        out["fwd_source"][i] = view.fwd_source
        if aux is not None:
            rays = ray_directions(pos_c.astype(np.float32), vel, tgt_c, cone).astype(np.float64)
            t, k = aux.cast(pos_c, rays, rng_t)
            out["aux_free"][i] = (t / rng_t).astype(np.float32)
            out["aux_kind"][i] = k
        h = select_history(stamps, i, 2, settings.ego_dt_s, settings.hist_tol_s, segments)
        if h:
            j = h[0]
            out["ego"][i] = ego_vector(
                st.quat, st.vel, states[j].quat, float(stamps[i] - stamps[j])
            )
            out["ego_history"][i] = True
        else:
            out["ego"][i] = ego_vector(st.quat, st.vel)
        sp = float(np.linalg.norm(vel)) * W
        out["speed_mps"][i] = sp
        out["warning_s"][i] = settings.range_m / sp if sp > 1e-6 else np.inf
        out["pos_c"][i] = pos_c
        out["vel"][i] = vel
        out["quat"][i] = st.quat
    return out


# ----------------------------------------------------------------------------- course


def load_label_course(
    catalogue_id: str,
    world_scale: float,
    *,
    catalogue: Any = None,
    trackdata_root: str | Path | None = None,
    scenes_root: str | Path | None = None,
    conversion: dict[str, Any] | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[Any, dict[str, Any], Any]:
    """``(course, entry, ConversionConfig)`` of a catalogue entry at ``world_scale``."""
    from drone_fly.tracks import catalogue as cat
    from drone_fly.tracks.liftoff import ConversionConfig
    from drone_fly.tracks.loader import load_course

    c = catalogue or cat.load_named("liftoff")
    entry = c.by_id(catalogue_id)
    d = dict(conversion or {})
    if "world_scale" in d and float(d["world_scale"]) != float(world_scale):
        raise HarnessConfigError(
            f"tracks conversion world_scale {d['world_scale']} != the preset's {world_scale}: "
            "labels need the course frame at the live state's scale"
        )
    d["world_scale"] = float(world_scale)
    conv = ConversionConfig.from_dict(d)
    if conv.world_scale != float(world_scale):
        raise HarnessConfigError("conversion world_scale differs from the preset's")
    root = cat.resolve_root(trackdata_root, c.header) if entry["source"] == "liftoff-xml" else None
    sr = cat.resolve_scenes_root(scenes_root, c.header, root) if root is not None else None
    courses = load_course(entry, catalogue=c, root=root, scenes_root=sr, conversion=conv,
                          cache_dir=Path(cache_dir) if cache_dir else None)  # fmt: skip
    return courses[0], entry, conv


# ----------------------------------------------------------------------------- run


def default_out(capture_name: str) -> Path:
    from drone_fly import paths

    return paths.labels_root() / "eyes" / capture_name


def _skip(meta: CaptureMeta | None, capture: Path, reason: str) -> dict[str, Any]:
    log.warning("skipping %s: %s", capture, reason)
    return {"capture": str(capture), "skipped": reason,
            "catalogue_id": None if meta is None else meta.catalogue_id}  # fmt: skip


def _pct(a: np.ndarray) -> dict[str, float | None]:
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"p05": None, "p50": None, "p95": None, "max": None}
    return {"p05": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}  # fmt: skip


def run_label(
    capture: str | Path,
    *,
    preset: str | None = "liftoff",
    config: Any = None,
    harness: str = "liftoff",
    out: str | Path | None = None,
    catalogue_id: str | None = None,
    lag_ms: float | None = None,
    trackdata_root: str | Path | None = None,
    scenes_root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    preview: int = 0,
    gpu_prototype_dir: str | Path | None = None,
    aux: bool = True,
) -> dict[str, Any]:
    """Label one capture; returns its meta, or ``{"skipped": reason}``."""
    from drone_fly.harness.offline.capture import load_capture
    from drone_fly.tracks.catalogue import TrackDataError
    from drone_fly.tracks.teacher_obs_ref import to_harness_course

    capture = Path(capture)
    cmeta = read_capture_meta(capture, catalogue_id=catalogue_id, lag_ms=lag_ms)
    if cmeta is None:
        return _skip(None, capture, "no-meta")
    view = preset_view(preset, config, harness)
    contract = offline_contract(gpu_prototype_dir)
    settings = settings_for(contract, view.world_scale, view.ray_range_scale)
    try:
        course, entry, conv = load_label_course(
            cmeta.catalogue_id, view.world_scale, trackdata_root=trackdata_root,
            scenes_root=scenes_root, conversion=(view.config.get("tracks") or {}).get("conversion"),
            cache_dir=cache_dir,
        )  # fmt: skip
    except TrackDataError as exc:
        msg = str(exc)
        if "no catalogue entry" in msg or "is missing" in msg:
            return _skip(cmeta, capture, f"no-geometry: {msg}")
        return _skip(cmeta, capture, f"unconvertible: {msg}")
    if cmeta.map is not None and entry["map"] != cmeta.map:
        return _skip(cmeta, capture, f"map-mismatch: meta {cmeta.map} vs catalogue {entry['map']}")
    try:
        cap = load_capture(capture, view.telemetry, view.world_scale, cmeta.lag_ms,
                           pair_tol_s=settings.pair_tol_s)  # fmt: skip
    except HarnessConfigError as exc:
        return _skip(cmeta, capture, f"no-telemetry: {exc}")
    if not cap.frames:
        return _skip(cmeta, capture, f"no-telemetry: no frame paired ({cap.dropped})")
    camera = view.camera
    if cmeta.camera:
        from drone_fly.eyes.camera_rays import camera_from_meta

        camera = camera_from_meta({**camera_meta(view.camera), **cmeta.camera})
    origin_c = course.frame.origin / float(view.world_scale)
    obstacles = to_harness_course(course).raycast_obstacles()
    tgt_idx = route_targets(cap.states, cap.segments, course, origin_c)
    frames = cap.frames
    targets = np.array([course.passages[tgt_idx[f.sample_index]].center for f in frames])
    stamps = np.array([f.stamp for f in frames], np.float64)
    segs = np.array([f.segment for f in frames], np.int64)
    geom = (AuxGeometry.from_course(course, obstacles, settings.aux_gate_margin_m,
                                    view.world_scale) if aux else None)  # fmt: skip
    arr = compute_labels([f.state for f in frames], stamps, segs, targets, origin_c, obstacles,
                         contract, settings, camera, view.masks, geom)  # fmt: skip
    lag = cmeta.lag_ms / 1000.0
    pair = np.array([abs(f.sample.received_at - (f.stamp - lag)) for f in frames])
    arr.update({
        "frame_index": np.array([f.index for f in frames], np.int64),
        "stamp": stamps,
        "frame_file": np.array([f.path.relative_to(capture).as_posix() for f in frames]),
        "segment": segs,
        "sample_index": np.array([f.sample_index for f in frames], np.int64),
        "target_index": np.array([tgt_idx[f.sample_index] for f in frames], np.int64),
        "pair_dt_s": pair.astype(np.float32),
    })  # fmt: skip
    out_dir = Path(out) if out is not None else default_out(cap.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (LABELS_FILE + ".tmp.npz")
    np.savez_compressed(tmp, **arr)
    tmp.replace(out_dir / LABELS_FILE)
    hit = arr["hit"]
    kinds = arr["aux_kind"]
    meta = {
        "format": LABEL_VERSION,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "capture": str(capture.resolve()),
        "capture_name": cap.name,
        "capture_sha256": capture_sha(capture),
        "capture_meta": cmeta.as_dict(),
        "catalogue_id": entry["id"],
        "map": entry["map"],
        "catalogue_split": entry["split"],
        "conversion_fingerprint": conv.fingerprint(),
        "course_flags": list(course.meta.get("flags") or []),
        "preset": view.as_dict(),
        "camera": camera_meta(camera),
        "settings": settings.as_dict(),
        "settings_sha256": settings.sha256(),
        "lag_ms": cmeta.lag_ms,
        "hit_rule": HIT_RULE,
        "aux_kinds": list(AUX_KINDS),
        "world_scale": {"preset": view.world_scale, "conversion": conv.world_scale},
        "range_m": settings.range_m,
        "range_train_m": settings.range_train,
        "n_obstacles": len(obstacles),
        "counts": {
            "frames_total": cap.n_frames_total,
            "frames_labelled": len(frames),
            "frames_with_hit": int(hit.any(axis=1).sum()),
            "rays_hit": int(hit.sum()),
            "rays_in_view": int(arr["in_view"].sum()),
            "ego_history": int(arr["ego_history"].sum()),
            "fwd_source": {
                name: int((arr["fwd_source"] == k).sum()) for k, name in enumerate(FWD_SOURCES)
            },  # fmt: skip
        },
        "drop_reasons": dict(cap.dropped),
        "alignment_s": _pct(pair),
        "hit_kinds": {
            "label": {"obstacle": int(hit.sum()), "none": int((~hit).sum())},
            "aux": {name: int((kinds == k).sum()) for k, name in enumerate(AUX_KINDS)},
            "aux_where_label_clear": {
                name: int(((kinds == k) & ~hit).sum()) for k, name in enumerate(AUX_KINDS)
            },  # fmt: skip
        },
        "speed_mps": _pct(arr["speed_mps"]),
        "warning_s": {
            **_pct(arr["warning_s"]),
            "share_lt_0_3": float(np.mean(arr["warning_s"] < 0.3)),
            "share_lt_0_5": float(np.mean(arr["warning_s"] < 0.5)),
        },  # fmt: skip
    }
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["previews"] = write_previews(capture, arr, camera, view.masks, out_dir, preview)
    meta["out_dir"] = str(out_dir)
    return meta


def write_previews(capture: Path, arr: dict[str, np.ndarray], camera: CameraModel,
                   masks: list[list[float]], out_dir: Path, n: int) -> list[str]:  # fmt: skip
    """Up to :data:`MAX_PREVIEW` frames with the in-view rays drawn (green clear -> red near)."""
    n = min(int(n), MAX_PREVIEW)
    if n <= 0 or not len(arr["free"]):
        return []
    from drone_fly.harness.errors import lazy_import
    from drone_fly.harness.offline.evaluate import load_image

    cv2 = lazy_import("cv2", package="opencv-python", purpose="eyes label previews")
    pdir = out_dir / "previews"
    pdir.mkdir(parents=True, exist_ok=True)
    rows = np.unique(np.linspace(0, len(arr["free"]) - 1, n).astype(int))
    paths = []
    for i in rows:
        img = load_image(capture / str(arr["frame_file"][i]))
        h, w = img.shape[:2]
        for x0, y0, x1, y1 in masks:
            cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)),
                          (128, 128, 128), 2)  # fmt: skip
        for r in np.flatnonzero(arr["in_view"][i]):
            u = (arr["uv"][i, r, 0] + 1) / 2 * (w - 1)
            v = (arr["uv"][i, r, 1] + 1) / 2 * (h - 1)
            f = float(arr["free"][i, r])
            col = (0, int(255 * f), int(255 * (1 - f)))
            cv2.circle(img, (int(u), int(v)), 14, col, -1 if arr["hit"][i, r] else 3)
            kind = AUX_KINDS[int(arr["aux_kind"][i, r])]
            if kind != "none":
                cv2.putText(img, kind[:2], (int(u) + 16, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)  # fmt: skip
        name = pdir / f"{Path(str(arr['frame_file'][i])).stem}.jpg"
        cv2.imwrite(str(name), img)
        paths.append(str(name))
    return paths
