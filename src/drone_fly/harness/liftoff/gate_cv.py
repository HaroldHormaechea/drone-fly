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

OpenCV is imported lazily (on the first frame / detection call).
"""

from __future__ import annotations

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

_F = np.float32


def _cv2() -> Any:
    return lazy_import("cv2", package="opencv-python", purpose="the OpenCV gate estimator")


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 pixel points as top-left, top-right, bottom-right, bottom-left."""
    p = np.asarray(pts, np.float64).reshape(4, 2)
    s = p.sum(axis=1)
    d = p[:, 1] - p[:, 0]
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]])


def camera_from_config(cam: dict[str, Any]) -> CameraModel:
    """:class:`CameraModel` from the ``camera`` section (1920x1080 when no resolution is set)."""
    res = cam.get("resolution")
    width, height = (
        (1920, 1080)
        if res is None
        else (int(v) for v in require_vector(res, "camera.resolution", 2))
    )
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

    def __init__(
        self,
        camera: CameraModel,
        scale: WorldScale | None = None,
        *,
        default_aperture: float,
        masks: Any = None,
        assoc_radius: float = 1.0,
        lost_timeout_s: float = 1.0,
        acquire_timeout_s: float = 3.0,
        pass_area_frac: float = 0.2,
        gates_per_lap: int | None = None,
    ) -> None:
        self.camera = camera
        self.scale = scale or WorldScale(1.0)
        self.default_aperture = float(default_aperture)  # training scale; acquire fallback
        self.masks = masks
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
        """Camera-frame gate pose -> world :class:`GateDetection` (``None`` behind the camera)."""
        if t_cam[2] <= 0:
            return None
        r_wc = quat_to_matrix(state.quat) @ camera.body_to_camera().T  # camera -> world
        pos = np.asarray(state.pos, np.float64)
        center = pos + r_wc @ (np.asarray(t_cam, np.float64) / self.scale.factor)
        d = center - pos
        nh = (r_wc @ np.asarray(n_cam, np.float64))[:2]
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
        )

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        self.current: _Tracked | None = None
        self.next_known: _Tracked | None = None
        self.gate_index = 0
        self._t0: float | None = None
        self._last_event: float | None = None  # last sighting or pass
        self._det_stamp: float | None = None  # capture stamp behind the latest detections

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
        self._det_stamp = None
        dets = [] if frame is None else self.detect_frame(frame, state)
        src = self._det_stamp

        match: GateDetection | None = None
        if self.current is not None and dets:
            match = self._associate(dets)
        elif self.current is None and dets:
            match = min(dets, key=lambda d: d.distance * (1.0 + d.off_axis))
            self.current = _Tracked.of(match, now, src)
        if match is not None and self.current is not None:
            prev = self.current.prev_signed
            self.current = _Tracked.of(match, now, src)
            self.current.prev_signed = prev
            self._last_event = now

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

        status: str
        if self.current is not None:
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
            curr = np.array([1.0, 0.0, self.default_aperture], _F)

        if next_det is not None:
            nxt = lookahead_visible(pos, next_det.center, next_det.yaw)
        elif self.next_known is not None:
            nxt = lookahead_unseen(pos, self.next_known.center)
        elif self.current is not None:
            nxt = lookahead_unseen(pos, self.current.center)
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
            assoc_radius=assoc_radius,
            lost_timeout_s=lost_timeout_s,
            acquire_timeout_s=acquire_timeout_s,
            pass_area_frac=pass_area_frac,
            gates_per_lap=gates_per_lap,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> OpenCvGateEstimator:
        from drone_fly.harness.liftoff.masks import FrameMasks

        cam = ctx.config.section("camera")
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
            camera_from_config(cam),
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
            masks=FrameMasks.from_config(cam),
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
