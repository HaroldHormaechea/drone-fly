"""Vision gate estimators: the shared tracker + the OpenCV (HSV quad) detector.

:class:`TrackingGateEstimator` is the base every image-based gate estimator shares; a subclass
only implements :meth:`~TrackingGateEstimator.detect` (image -> :class:`GateDetection` list).
:class:`OpenCvGateEstimator` is the first one; the learned keypoint detector
(:mod:`~drone_fly.harness.liftoff.gate_learned`) is the second.

OpenCV detection, per frame:

1. HSV threshold (one or more configured ranges, e.g. two for red's hue wrap) + morphology
   -> external contours -> convex 4-corner quads (area / convexity filters).
2. ``solvePnP`` (IPPE) of each quad against the **nominal outer gate rectangle**
   (``gates.nominal_width_m`` x ``gates.nominal_height_m``, target metres), intrinsics from the
   configured horizontal FOV.

Shared by every subclass (the tracker):

3. camera -> body (uptilt) -> world (live attitude + position): world gate centre, and the
   gate yaw from the plane normal, oriented along the direction of travel (away from the drone).
   Lengths go through :class:`WorldScale` (training scale). Each detection carries its own
   aperture (per gate class) and class name.
4. Masks (``camera.masks``): the image is masked before :meth:`detect`, and detections whose
   centre pixel lies in a mask are dropped.
5. Tracking: the *current* gate is the detection associated with the tracked gate — same class
   preferred — (else, on acquisition, the most central / nearest candidate); the *next* gate is
   the nearest other candidate farther away than the current one. A pass — the env's proximity
   rule (``dist < aperture/2``), a signed plane crossing within the aperture, or the gate's image
   area peaking above ``pass_area_frac`` and then vanishing while close — advances the gate.
6. Lost: the last world estimate is held (``rel`` recomputed from the live position) for
   ``lost_timeout_s``; after that the status is ``"lost"`` and the runner fails safe. Before the
   first detection the target is the current position (hover) for ``acquire_timeout_s``.
7. Encodings via :mod:`~drone_fly.harness.course_senses`; an unseen next gate uses the
   occluded form (``vis = 0``, bearing to its last-known position, else to the current gate).
8. UC70 ages: each tracked gate keeps the capture stamp of the frame behind its last match
   (``detect_frame`` records it; the async learned estimator substitutes its older frame's stamp).
   ``info`` carries ``target_age_s`` / ``next_age_s`` (now - that stamp; ``lost_timeout_s`` when
   nothing is held), ``target_valid`` (a current gate is held) and ``hand_over`` (passed this
   tick). No behaviour changes.
9. UC70 lost policy (``gates.lost_policy``; the simulator's ``sim.perception`` mirrors it):
   ``terminate`` (default, steps 6 as above) or ``reacquire``: (a) a held gate unmatched for more
   than ``drop_after_s`` is replaced by the acquisition rule whenever detections exist; (b) after
   ``lost_timeout_s`` unmatched it is RELEASED (its last estimate stays the target, ``target_valid``
   false) and the tracker acquires again; the status is ``lost`` (the runner's fail-safe) only when
   nothing good — a match, an acquisition or a pass — happened for ``lost_hard_stop_s``.
10. Pose at capture: ``update`` keeps a short history of ``(received_at, pos, quat)`` and step 3
    uses the drone pose **at the detections' capture stamp** (plus ``gates.pose_offset_s``),
    interpolated (linear position, slerp attitude), instead of the current tick's pose — the async
    learned detector's results are several frames old, and pairing them with the current attitude
    swings the world estimate while the drone rotates. With fewer than two history stamps, no
    stamp, or a stamp more than ``POSE_STAMP_TOL_S`` outside the history (clock mismatch; warned
    once, counted in ``pose_fallbacks``) the current pose is used. Tracking and the pass rule stay
    on the current pose.

OpenCV is imported lazily (on the first frame / detection call).
"""

from __future__ import annotations

import collections
import logging
from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from drone_fly.harness.config import require_number, require_vector
from drone_fly.harness.course import Gate
from drone_fly.harness.course_senses import (
    curr_gate_block,
    lookahead_unseen,
    lookahead_visible,
)
from drone_fly.harness.errors import HarnessConfigError, lazy_import
from drone_fly.harness.geometry import CameraModel, quat_to_matrix
from drone_fly.harness.interfaces import GateEstimator
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import FlightState, Frame, GateEstimate

log = logging.getLogger(__name__)

_F = np.float32
#: Default ``gates.pose_offset_s``: the V2 fit on the 2026-10-18 runs (-25..-50 ms in all three).
DEFAULT_POSE_OFFSET_S = -0.025
#: A capture stamp farther than this outside the pose history is a clock mismatch: current pose.
POSE_STAMP_TOL_S = 0.25
POSE_HISTORY = 64


def _cv2() -> Any:
    return lazy_import("cv2", package="opencv-python", purpose="the OpenCV gate estimator")


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 pixel points as top-left, top-right, bottom-right, bottom-left."""
    p = np.asarray(pts, np.float64).reshape(4, 2)
    s = p.sum(axis=1)
    d = p[:, 1] - p[:, 0]
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]])


def camera_from_config(cam: dict[str, Any]) -> CameraModel:
    """:class:`CameraModel` from the ``camera`` section (1920x1080 when no resolution is set).

    ``camera.principal_point: [cx, cy]`` (pixels at ``resolution``, inside the image) overrides
    the image-centre principal point (UC-91); unset keeps the centre.
    """
    res = cam.get("resolution")
    width, height = (
        (1920, 1080)
        if res is None
        else (int(v) for v in require_vector(res, "camera.resolution", 2))
    )
    pp = cam.get("principal_point")
    if pp is not None:
        cx, cy = require_vector(pp, "camera.principal_point", 2)
        if not (0.0 <= cx <= width and 0.0 <= cy <= height):
            raise HarnessConfigError(
                f"camera.principal_point {[cx, cy]} must lie inside the {width}x{height} image"
            )
        pp = (cx, cy)
    return CameraModel(
        width,
        height,
        float(
            require_number(cam.get("hfov_deg"), "camera.hfov_deg", min_value=1.0, max_value=179.0)
        ),
        float(
            require_number(
                cam.get("uptilt_deg", 0.0), "camera.uptilt_deg", min_value=-90.0, max_value=90.0
            )
        ),
        principal_point=pp,
    )


def solve_planar_pnp(
    object_points: np.ndarray, image_points: np.ndarray, camera: CameraModel
) -> tuple[np.ndarray, np.ndarray]:
    """IPPE PnP of >= 4 coplanar object points (``z = 0``) -> (centre, normal) in the camera frame.

    The object frame origin is the gate centre; the returned normal is the object ``+z`` axis.
    """
    cv2 = _cv2()
    obj = np.asarray(object_points, np.float64).reshape(-1, 3)
    img = np.asarray(image_points, np.float64).reshape(-1, 1, 2)
    if len(obj) < 4 or len(obj) != len(img):
        raise ValueError("planar PnP needs >= 4 matching points")
    ok, rvec, tvec = cv2.solvePnP(obj, img, camera.intrinsics(), None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("solvePnP failed")
    rot, _ = cv2.Rodrigues(rvec)
    return tvec.reshape(3), rot[:, 2]


@dataclass(frozen=True)
class GateDetection:
    corners: np.ndarray  # (N, 2) image points used for PnP (TL, TR, BR, BL for quads)
    area_frac: float  # image area of the detection / image area
    t_cam: np.ndarray  # gate centre in the camera frame (target metres)
    n_cam: np.ndarray  # gate plane normal in the camera frame
    center: np.ndarray  # world centre (training scale)
    yaw: float  # world yaw of the travel-direction normal
    aperture: float  # training scale
    distance: float  # training scale, drone -> centre
    off_axis: float  # radians between the optical axis and the centre
    cls: str | None = None  # gate class (gate-type table); None for single-style detectors
    pixel_center: np.ndarray | None = None  # gate centre in the image (pixels)
    n_world: np.ndarray | None = None  # gate plane normal in the world (camera normal, rotated)

    @property
    def center_px(self) -> np.ndarray:
        if self.pixel_center is not None:
            return np.asarray(self.pixel_center, np.float64)
        return np.asarray(self.corners, np.float64).reshape(-1, 2).mean(axis=0)


@dataclass
class _Tracked:
    center: np.ndarray
    yaw: float
    aperture: float
    last_seen: float
    area_frac: float = 0.0
    prev_signed: float | None = None
    cls: str | None = None
    src_stamp: float = 0.0  # capture stamp of the frame behind the last match (UC70)

    @classmethod
    def of(cls, d: GateDetection, now: float, src_stamp: float | None = None) -> _Tracked:
        return cls(d.center.copy(), d.yaw, d.aperture, now, d.area_frac, cls=d.cls,
                   src_stamp=now if src_stamp is None else float(src_stamp))  # fmt: skip

    @property
    def gate(self) -> Gate:
        return Gate(self.center, self.yaw, self.aperture)


class TrackingGateEstimator(GateEstimator):
    """Detections -> current / next gate tracking (see the module docstring, steps 3-7)."""

    provides = frozenset({"target", "curr_gate", "next_lookahead"})
    provenance = "screen"

    def __init__(
        self,
        camera: CameraModel,
        scale: WorldScale | None = None,
        *,
        default_aperture: float,
        masks: Any = None,
        pointer_reader: Any = None,
        assoc_radius: float = 1.0,
        lost_timeout_s: float = 1.0,
        acquire_timeout_s: float = 3.0,
        pass_area_frac: float = 0.2,
        gates_per_lap: int | None = None,
        lost_policy: str = "terminate",
        lost_hard_stop_s: float = 6.0,
        drop_after_s: float = 0.5,
        pose_offset_s: float = 0.0,
    ) -> None:
        if lost_policy not in ("terminate", "reacquire"):
            raise HarnessConfigError(
                f"gates.lost_policy must be terminate or reacquire, got {lost_policy!r}"
            )
        self.lost_policy = lost_policy
        self.lost_hard_stop_s = float(lost_hard_stop_s)
        self.drop_after_s = float(drop_after_s)
        self.pose_offset_s = float(require_number(
            pose_offset_s, "gates.pose_offset_s", min_value=-0.2, max_value=0.2
        ))  # fmt: skip
        self.pose_fallbacks = 0  # capture stamps outside the pose history (clock mismatch)
        self._pose_warned = False
        self.camera = camera
        self.scale = scale or WorldScale(1.0)
        self.default_aperture = float(default_aperture)  # training scale; acquire fallback
        self.masks = masks
        #: Optional UC74 HUD pointer reader (live only). When set, ``update`` populates
        #: ``info["pointer"]`` from the RAW frame over its own ROI; the gate detector masks its own
        #: copy, so the two never collide. ``None`` leaves ``info["pointer"]`` unset (sim / legacy).
        self.pointer_reader = pointer_reader
        self.assoc_radius = assoc_radius
        self.lost_timeout_s = lost_timeout_s
        self.acquire_timeout_s = acquire_timeout_s
        self.pass_area_frac = pass_area_frac
        self.gates_per_lap = gates_per_lap
        self.reset()

    # ------------------------------------------------------------------ detection hooks

    @abstractmethod
    def detect(self, image: np.ndarray, state: FlightState) -> list[GateDetection]:
        """All gates visible in ``image`` (already masked) as world-frame detections."""

    def detect_frame(self, frame: Frame, state: FlightState) -> list[GateDetection]:
        """Mask the frame, detect, and drop detections centred in a mask."""
        self._det_stamp = frame.stamp  # an async detector replaces it with its source frame's
        image = frame.image
        if self.masks:
            image = self.masks.apply(image)
        dets = self.detect(image, state)
        if self.masks and dets:
            h, w = frame.image.shape[:2]
            inside = self.masks.contains(np.array([d.center_px for d in dets]), w, h)
            dets = [d for d, m in zip(dets, inside, strict=True) if not m]
        return dets

    def extra_info(self) -> dict[str, Any]:
        """Extra fields for ``GateEstimate.info`` (e.g. inference time)."""
        return {}

    def _camera_for(self, image: np.ndarray) -> CameraModel:
        """Intrinsics for the actual capture size (same FOV / uptilt if the size differs)."""
        h, w = image.shape[:2]
        if (w, h) != (self.camera.width, self.camera.height):
            self.camera = self.camera.scaled_to(w, h)
        return self.camera

    def detection_from_pose(
        self,
        t_cam: np.ndarray,
        n_cam: np.ndarray,
        image_points: np.ndarray,
        area_frac: float,
        state: FlightState,
        camera: CameraModel,
        aperture_m: float,
        cls: str | None = None,
        pixel_center: np.ndarray | None = None,
    ) -> GateDetection | None:
        """Camera-frame gate pose -> world :class:`GateDetection`.

        ``None`` behind the camera, or for a non-finite PnP solution (UC-91: a degenerate 4-point
        IPPE solve returned NaN, which ``t_cam[2] <= 0`` let through to the tracker).
        """
        if not (np.isfinite(t_cam).all() and np.isfinite(n_cam).all()) or t_cam[2] <= 0:
            return None
        pos, quat = self._pose_at(self._det_stamp, state)  # the pose the frame was captured at
        r_wc = quat_to_matrix(quat) @ camera.body_to_camera().T  # camera -> world
        center = pos + r_wc @ (np.asarray(t_cam, np.float64) / self.scale.factor)
        d = center - pos
        n_world = r_wc @ np.asarray(n_cam, np.float64)
        nh = n_world[:2]
        if np.linalg.norm(nh) < 1e-6:
            nh = d[:2]
        if np.dot(nh, d[:2]) < 0:
            nh = -nh
        return GateDetection(
            corners=np.asarray(image_points, np.float64),
            area_frac=float(area_frac),
            t_cam=np.asarray(t_cam, np.float64),
            n_cam=np.asarray(n_cam, np.float64),
            center=center.astype(_F),
            yaw=float(np.arctan2(nh[1], nh[0])),
            aperture=self.scale.length(aperture_m),
            distance=float(np.linalg.norm(d)),
            off_axis=float(np.arctan2(np.linalg.norm(t_cam[:2]), t_cam[2])),
            cls=cls,
            pixel_center=None if pixel_center is None else np.asarray(pixel_center, np.float64),
            n_world=n_world,
        )

    # ------------------------------------------------------------------ pose history

    def _record_pose(self, state: FlightState) -> None:
        stamp = float(state.received_at)
        if self._poses and self._poses[-1][0] == stamp:
            return
        self._poses.append(
            (stamp, np.asarray(state.pos, np.float64), np.asarray(state.quat, np.float64))
        )

    def _pose_at(self, stamp: float | None, state: FlightState) -> tuple[np.ndarray, np.ndarray]:
        """``(pos, quat)`` at ``stamp + pose_offset_s`` from the history (else the current pose)."""
        current = (np.asarray(state.pos, np.float64), np.asarray(state.quat, np.float64))
        hist = self._poses
        if stamp is None or len(hist) < 2:
            return current
        t = float(stamp) + self.pose_offset_s
        oldest, newest = hist[0][0], hist[-1][0]
        if t < oldest - POSE_STAMP_TOL_S or t > newest + POSE_STAMP_TOL_S:
            self.pose_fallbacks += 1
            if not self._pose_warned:
                self._pose_warned = True
                log.warning(
                    "gate capture stamp %.3f is outside the pose history [%.3f, %.3f]: frame and "
                    "telemetry clocks disagree; using the current pose", t, oldest, newest,
                )  # fmt: skip
            return current
        t = min(max(t, oldest), newest)
        i = 1
        while i < len(hist) - 1 and hist[i][0] < t:
            i += 1
        (t0, p0, q0), (t1, p1, q1) = hist[i - 1], hist[i]
        w = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return p0 + w * (p1 - p0), _slerp(q0, q1, w)

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        self._poses: collections.deque[tuple[float, np.ndarray, np.ndarray]] = collections.deque(
            maxlen=POSE_HISTORY
        )
        self.current: _Tracked | None = None
        self.next_known: _Tracked | None = None
        self.gate_index = 0
        self._t0: float | None = None
        self._last_event: float | None = None  # last sighting or pass
        self._det_stamp: float | None = None  # capture stamp behind the latest detections
        self._released: _Tracked | None = None  # reacquire: the last released target
        self._good: float | None = None  # reacquire: the last match / acquisition / pass

    @property
    def lap(self) -> int:
        return 0 if not self.gates_per_lap else self.gate_index // self.gates_per_lap

    # ------------------------------------------------------------------ tracking

    def _passed(self, cur: _Tracked, pos: np.ndarray, seen: bool) -> bool:
        rel = pos - cur.center
        dist = float(np.linalg.norm(rel))
        if dist < cur.aperture / 2.0:  # the env's pass rule
            return True
        n = np.array([np.cos(cur.yaw), np.sin(cur.yaw), 0.0])
        s = float(np.dot(rel, n))
        lateral = float(np.linalg.norm(rel - s * n))
        crossed = cur.prev_signed is not None and cur.prev_signed < 0 <= s
        cur.prev_signed = s
        if crossed and lateral < cur.aperture:
            return True
        return (not seen) and cur.area_frac >= self.pass_area_frac and dist < 1.5 * cur.aperture

    def _associate(self, dets: list[GateDetection]) -> GateDetection | None:
        """The tracked gate's detection: within the association radius, same class first."""
        cur = self.current
        assert cur is not None
        limit = self.assoc_radius * max(1.0, cur.aperture)
        ranked = sorted(
            dets, key=lambda d: (d.cls != cur.cls, float(np.linalg.norm(d.center - cur.center)))
        )
        for d in ranked:
            if np.linalg.norm(d.center - cur.center) <= limit:
                return d
        return None

    def update(self, frame: Frame | None, state: FlightState, now: float) -> GateEstimate:
        if self._t0 is None:
            self._t0 = now
        pos = np.asarray(state.pos, np.float64)
        self._record_pose(state)
        self._det_stamp = None
        dets = [] if frame is None else self.detect_frame(frame, state)
        src = self._det_stamp

        reacq = self.lost_policy == "reacquire"
        if self._good is None:
            self._good = now
        match: GateDetection | None = None
        acquired = False
        if self.current is not None and dets:
            match = self._associate(dets)
            if (match is None and reacq
                    and now - self.current.last_seen > self.drop_after_s):  # fmt: skip
                # (a) a stale held gate is replaced by the acquisition rule
                match = min(dets, key=lambda d: d.distance * (1.0 + d.off_axis))
                self.current = _Tracked.of(match, now, src)
                acquired = True
        elif self.current is None and dets:
            match = min(dets, key=lambda d: d.distance * (1.0 + d.off_axis))
            self.current = _Tracked.of(match, now, src)
            acquired = True
        if match is not None and self.current is not None:
            prev = None if acquired and reacq else self.current.prev_signed
            self.current = _Tracked.of(match, now, src)
            self.current.prev_signed = prev
            self._last_event = now
            if reacq:
                self._good = now
                self._released = None

        next_det: GateDetection | None = None
        if self.current is not None:
            others = [d for d in dets if d is not match]
            ref = self.current
            farther = [d for d in others if d.distance > float(np.linalg.norm(ref.center - pos))]
            if farther:
                next_det = min(farther, key=lambda d: float(np.linalg.norm(d.center - ref.center)))
                self.next_known = _Tracked.of(next_det, now, src)

        passed = False
        if self.current is not None and self._passed(self.current, pos, match is not None):
            passed = True
            self.gate_index += 1
            self.current = None
            if self.next_known is not None:
                self.current = replace(self.next_known, last_seen=now, prev_signed=None)
            self.next_known = None
            next_det = None
            self._last_event = now
            if reacq:
                self._good = now
                self._released = None

        status: str
        if reacq:
            if self.current is not None and now - self.current.last_seen > self.lost_timeout_s:
                self._released = self.current  # (b) release; keep its estimate as the target
                self.current = None
            status = "tracking" if match is not None and not passed else "holding"
            if now - self._good > self.lost_hard_stop_s:
                status = "lost"
            held = self.current if self.current is not None else self._released
            if held is not None:
                target = held.center.astype(_F)
                curr = curr_gate_block(held.gate)
            else:
                target = pos.astype(_F)
                # UC-89: frame-neutral no-gate placeholder. A world-frame [1,0] (yaw 0)
                # rotates under the spawn-yaw aligner into a fake ~90-deg-off gate; [0,0] is
                # rotation-invariant so yaw alignment maps it to "no usable gate". target=pos
                # already gives rel=0 (hover). Only the genuine no-gate branches change.
                curr = np.array([0.0, 0.0, self.default_aperture], _F)
        elif self.current is not None:
            age = now - self.current.last_seen
            status = "tracking" if match is not None and not passed else "holding"
            if age > self.lost_timeout_s:
                status = "lost"
            target = self.current.center.astype(_F)
            curr = curr_gate_block(self.current.gate)
        else:
            since = now - (self._last_event if self._last_event is not None else self._t0)
            limit = self.acquire_timeout_s if self._last_event is None else self.lost_timeout_s
            status = "holding" if since <= limit else "lost"
            target = pos.astype(_F)  # hover in place while (re)acquiring
            # UC-89: frame-neutral no-gate placeholder (see the reacq branch above).
            curr = np.array([0.0, 0.0, self.default_aperture], _F)

        if next_det is not None:
            nxt = lookahead_visible(pos, next_det.center, next_det.yaw)
        elif self.next_known is not None:
            nxt = lookahead_unseen(pos, self.next_known.center)
        elif self.current is not None:
            nxt = lookahead_unseen(pos, self.current.center)
        elif reacq and self._released is not None:
            nxt = lookahead_unseen(pos, self._released.center)
        else:
            nxt = lookahead_unseen(pos, pos + np.array([1.0, 0.0, 0.0]))

        info: dict[str, Any] = {"detections": len(dets), "matched": match is not None}
        lt = float(self.lost_timeout_s)
        info["target_age_s"] = lt if self.current is None else float(now - self.current.src_stamp)
        info["next_age_s"] = (
            lt if self.next_known is None else float(now - self.next_known.src_stamp)
        )
        info["target_valid"] = self.current is not None
        info["hand_over"] = passed
        info["pose_fallbacks"] = self.pose_fallbacks
        if self.current is not None and self.current.cls is not None:
            info["gate_class"] = self.current.cls
        if frame is not None:
            # UC-65 live overlay: what was detected, in **frame pixel** coordinates.
            h, w = frame.image.shape[:2]
            info["detections_px"] = [
                {
                    "cls": d.cls,
                    "points": np.asarray(d.corners, np.float64).reshape(-1, 2).tolist(),
                    "center": d.center_px.tolist(),
                    "matched": d is match,
                }
                for d in dets
            ]
            info["frame_size"] = [int(w), int(h)]
            info["frame_index"] = int(frame.index)
        info.update(self.extra_info())
        if frame is not None and self.pointer_reader is not None:
            # UC74: the live HUD next-gate pointer (body unit bearing 3 + availability bit 1),
            # read from the RAW frame. zeros + bit 0 when absent/ambiguous (the degraded block).
            info["pointer"] = self.pointer_reader.read(frame.image)
        return GateEstimate(
            target=target,
            curr_gate=curr,
            next_lookahead=nxt,
            gate_index=self.gate_index,
            lap=self.lap,
            status=status,  # type: ignore[arg-type]
            stamp=frame.stamp if frame is not None else now,
            passed=passed,
            info=info,
        )


def _slerp(q0: np.ndarray, q1: np.ndarray, w: float) -> np.ndarray:
    """Normalized shortest-arc slerp between unit quaternions ``[w, x, y, z]``."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + w * (q1 - q0)
    else:
        theta = np.arccos(dot)
        q = (np.sin((1.0 - w) * theta) * q0 + np.sin(w * theta) * q1) / np.sin(theta)
    return q / np.linalg.norm(q)


class OpenCvGateEstimator(TrackingGateEstimator):
    """HSV quads + PnP against the nominal outer rectangle (single gate style)."""

    def __init__(
        self,
        camera: CameraModel,
        nominal_width_m: float,
        nominal_height_m: float,
        hsv_ranges: list[tuple[list[float], list[float]]],
        scale: WorldScale | None = None,
        *,
        min_area_frac: float = 0.0005,
        approx_eps: float = 0.04,
        morph_kernel: int = 5,
        assoc_radius: float = 1.0,
        lost_timeout_s: float = 1.0,
        acquire_timeout_s: float = 3.0,
        pass_area_frac: float = 0.2,
        gates_per_lap: int | None = None,
        masks: Any = None,
        pointer_reader: Any = None,
        lost_policy: str = "terminate",
        lost_hard_stop_s: float = 6.0,
        drop_after_s: float = 0.5,
        pose_offset_s: float = 0.0,
    ) -> None:
        if nominal_width_m <= 0 or nominal_height_m <= 0:
            raise HarnessConfigError("gates.nominal_width_m / nominal_height_m must be > 0")
        if not hsv_ranges:
            raise HarnessConfigError("gates.hsv_ranges needs at least one [low, high] pair")
        self.w = float(nominal_width_m)
        self.h = float(nominal_height_m)
        self.hsv_ranges = [
            (np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in hsv_ranges
        ]
        self.min_area_frac = min_area_frac
        self.approx_eps = approx_eps
        self.morph_kernel = morph_kernel
        scale = scale or WorldScale(1.0)
        super().__init__(
            camera,
            scale,
            default_aperture=scale.length(min(self.w, self.h)),
            masks=masks,
            pointer_reader=pointer_reader,
            assoc_radius=assoc_radius,
            lost_timeout_s=lost_timeout_s,
            acquire_timeout_s=acquire_timeout_s,
            pass_area_frac=pass_area_frac,
            gates_per_lap=gates_per_lap,
            lost_policy=lost_policy,
            lost_hard_stop_s=lost_hard_stop_s,
            drop_after_s=drop_after_s,
            pose_offset_s=pose_offset_s,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> OpenCvGateEstimator:
        from drone_fly.harness.liftoff.masks import FrameMasks
        from drone_fly.harness.liftoff.pointer_cv import pointer_reader_from_config

        cam = ctx.config.section("camera")
        camera = camera_from_config(cam)
        masks = FrameMasks.from_config(cam)
        ranges = []
        for i, pair in enumerate(cfg.get("hsv_ranges") or []):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise HarnessConfigError(f"gates.hsv_ranges[{i}] must be [[h,s,v], [h,s,v]]")
            ranges.append(
                (
                    require_vector(pair[0], f"gates.hsv_ranges[{i}][0]", 3),
                    require_vector(pair[1], f"gates.hsv_ranges[{i}][1]", 3),
                )
            )
        gpl = cfg.get("gates_per_lap")
        num = lambda k, d, **kw: float(require_number(cfg.get(k, d), f"gates.{k}", **kw))  # noqa: E731
        return cls(
            camera,
            num("nominal_width_m", 1.5, min_value=0.0, min_exclusive=True),
            num("nominal_height_m", 1.5, min_value=0.0, min_exclusive=True),
            ranges,
            WorldScale(ctx.config.world_scale),
            min_area_frac=num("min_area_frac", 0.0005, min_value=0.0),
            approx_eps=num("approx_eps", 0.04, min_value=0.0, min_exclusive=True),
            morph_kernel=int(num("morph_kernel", 5, min_value=1.0)),
            assoc_radius=num("assoc_radius", 1.0, min_value=0.0, min_exclusive=True),
            lost_timeout_s=num("lost_timeout_s", 1.0, min_value=0.0, min_exclusive=True),
            acquire_timeout_s=num("acquire_timeout_s", 3.0, min_value=0.0),
            pass_area_frac=num("pass_area_frac", 0.2, min_value=0.0, max_value=1.0),
            gates_per_lap=None if gpl is None else int(gpl),
            lost_policy=str(cfg.get("lost_policy", "terminate")),
            lost_hard_stop_s=num("lost_hard_stop_s", 6.0, min_value=0.0, min_exclusive=True),
            drop_after_s=num("drop_after_s", 0.5, min_value=0.0),
            pose_offset_s=num(
                "pose_offset_s", DEFAULT_POSE_OFFSET_S, min_value=-0.2, max_value=0.2
            ),
            masks=masks,
            pointer_reader=pointer_reader_from_config(camera, cam, masks),
        )

    # ------------------------------------------------------------------ detection

    def detect_quads(self, image: np.ndarray) -> list[np.ndarray]:
        """HSV threshold -> convex 4-corner quads (TL, TR, BR, BL), largest first."""
        cv2 = _cv2()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in self.hsv_ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        k = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_area_frac * mask.shape[0] * mask.shape[1]
        quads: list[tuple[float, np.ndarray]] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < min_area:
                continue
            hull = cv2.convexHull(c)
            approx = cv2.approxPolyDP(hull, self.approx_eps * cv2.arcLength(hull, True), True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quads.append((area, order_corners(approx.reshape(4, 2))))
        quads.sort(key=lambda x: -x[0])
        return [q for _, q in quads]

    def solve_pose(self, corners: np.ndarray, camera: CameraModel) -> tuple[np.ndarray, np.ndarray]:
        """PnP of a quad against the nominal rectangle -> (centre, normal) in the camera frame."""
        w, h = self.w / 2.0, self.h / 2.0
        obj = np.array([[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0]], np.float64)
        return solve_planar_pnp(obj, corners, camera)

    def detect(self, image: np.ndarray, state: FlightState) -> list[GateDetection]:
        camera = self._camera_for(image)
        img_area = float(image.shape[0] * image.shape[1])
        cv2 = _cv2()
        out = []
        for q in self.detect_quads(image):
            try:
                t_cam, n_cam = self.solve_pose(q, camera)
            except (ValueError, cv2.error):
                continue
            det = self.detection_from_pose(
                t_cam,
                n_cam,
                q,
                float(cv2.contourArea(q.astype(np.float32))) / img_area,
                state,
                camera,
                min(self.w, self.h),
            )
            if det is not None:
                out.append(det)
        return out
