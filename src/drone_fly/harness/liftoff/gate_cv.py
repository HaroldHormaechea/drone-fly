"""OpenCV gate estimator (first :class:`GateEstimator` implementation for vision targets).

Pipeline per frame:

1. HSV threshold (one or more configured ranges, e.g. two for red's hue wrap) + morphology
   -> external contours -> convex 4-corner quads (area / convexity filters).
2. ``solvePnP`` (IPPE) of each quad against the **nominal outer gate rectangle**
   (``gates.nominal_width_m`` x ``gates.nominal_height_m``, target metres), intrinsics from the
   configured horizontal FOV.
3. camera -> body (uptilt) -> world (live attitude + position): world gate centre, and the
   gate yaw from the plane normal, oriented along the direction of travel (away from the drone).
   Lengths go through :class:`WorldScale` (training scale).
4. Tracking: the *current* gate is the detection associated with the tracked gate (else, on
   acquisition, the most central / nearest candidate); the *next* gate is the nearest other
   candidate farther away than the current one. A pass — the env's proximity rule
   (``dist < aperture/2``), a signed plane crossing within the aperture, or the gate's image
   area peaking above ``pass_area_frac`` and then vanishing while close — advances the gate.
5. Lost: the last world estimate is held (``rel`` recomputed from the live position) for
   ``lost_timeout_s``; after that the status is ``"lost"`` and the runner fails safe. Before the
   first detection the target is the current position (hover) for ``acquire_timeout_s``.
6. Encodings via :mod:`~drone_fly.harness.course_senses`; an unseen next gate uses the
   occluded form (``vis = 0``, bearing to its last-known position, else to the current gate).

OpenCV is imported lazily (on the first frame / detection call).
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class GateDetection:
    corners: np.ndarray  # (4, 2) TL, TR, BR, BL pixels
    area_frac: float  # quad area / image area
    t_cam: np.ndarray  # gate centre in the camera frame (target metres)
    n_cam: np.ndarray  # gate plane normal in the camera frame
    center: np.ndarray  # world centre (training scale)
    yaw: float  # world yaw of the travel-direction normal
    aperture: float  # training scale
    distance: float  # training scale, drone -> centre
    off_axis: float  # radians between the optical axis and the centre


@dataclass
class _Tracked:
    center: np.ndarray
    yaw: float
    aperture: float
    last_seen: float
    area_frac: float = 0.0
    prev_signed: float | None = None

    @classmethod
    def of(cls, d: GateDetection, now: float) -> _Tracked:
        return cls(d.center.copy(), d.yaw, d.aperture, now, d.area_frac)

    @property
    def gate(self) -> Gate:
        return Gate(self.center, self.yaw, self.aperture)


class OpenCvGateEstimator(GateEstimator):
    provides = frozenset({"target", "curr_gate", "next_lookahead"})

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
    ) -> None:
        if nominal_width_m <= 0 or nominal_height_m <= 0:
            raise HarnessConfigError("gates.nominal_width_m / nominal_height_m must be > 0")
        if not hsv_ranges:
            raise HarnessConfigError("gates.hsv_ranges needs at least one [low, high] pair")
        self.camera = camera
        self.w = float(nominal_width_m)
        self.h = float(nominal_height_m)
        self.hsv_ranges = [
            (np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in hsv_ranges
        ]
        self.scale = scale or WorldScale(1.0)
        self.min_area_frac = min_area_frac
        self.approx_eps = approx_eps
        self.morph_kernel = morph_kernel
        self.assoc_radius = assoc_radius
        self.lost_timeout_s = lost_timeout_s
        self.acquire_timeout_s = acquire_timeout_s
        self.pass_area_frac = pass_area_frac
        self.gates_per_lap = gates_per_lap
        self.reset()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> OpenCvGateEstimator:
        cam = ctx.config.section("camera")
        res = cam.get("resolution")
        width, height = (
            (1920, 1080)
            if res is None
            else (int(v) for v in require_vector(res, "camera.resolution", 2))
        )
        camera = CameraModel(
            width,
            height,
            float(
                require_number(
                    cam.get("hfov_deg"), "camera.hfov_deg", min_value=1.0, max_value=179.0
                )
            ),
            float(
                require_number(
                    cam.get("uptilt_deg", 0.0), "camera.uptilt_deg", min_value=-90.0, max_value=90.0
                )
            ),
        )
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
        )

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        self.current: _Tracked | None = None
        self.next_known: _Tracked | None = None
        self.gate_index = 0
        self._t0: float | None = None
        self._last_event: float | None = None  # last sighting or pass

    @property
    def lap(self) -> int:
        return 0 if not self.gates_per_lap else self.gate_index // self.gates_per_lap

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
        cv2 = _cv2()
        w, h = self.w / 2.0, self.h / 2.0
        obj = np.array([[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0]], np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj,
            np.asarray(corners, np.float64).reshape(4, 1, 2),
            camera.intrinsics(),
            None,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok:
            raise ValueError("solvePnP failed")
        rot, _ = cv2.Rodrigues(rvec)
        return tvec.reshape(3), rot[:, 2]

    def _camera_for(self, image: np.ndarray) -> CameraModel:
        """Intrinsics for the actual capture size (same FOV / uptilt if the size differs)."""
        h, w = image.shape[:2]
        if (w, h) != (self.camera.width, self.camera.height):
            self.camera = self.camera.scaled_to(w, h)
        return self.camera

    def detect(self, image: np.ndarray, state: FlightState) -> list[GateDetection]:
        camera = self._camera_for(image)
        r_wc = quat_to_matrix(state.quat) @ camera.body_to_camera().T  # camera -> world
        pos = np.asarray(state.pos, np.float64)
        img_area = float(image.shape[0] * image.shape[1])
        cv2 = _cv2()
        out = []
        for q in self.detect_quads(image):
            try:
                t_cam, n_cam = self.solve_pose(q, camera)
            except (ValueError, cv2.error):
                continue
            if t_cam[2] <= 0:
                continue
            center = pos + r_wc @ (t_cam / self.scale.factor)
            d = center - pos
            nh = (r_wc @ n_cam)[:2]
            if np.linalg.norm(nh) < 1e-6:
                nh = d[:2]
            if np.dot(nh, d[:2]) < 0:
                nh = -nh
            out.append(
                GateDetection(
                    corners=q,
                    area_frac=float(cv2.contourArea(q.astype(np.float32))) / img_area,
                    t_cam=t_cam,
                    n_cam=n_cam,
                    center=center.astype(_F),
                    yaw=float(np.arctan2(nh[1], nh[0])),
                    aperture=self.scale.length(min(self.w, self.h)),
                    distance=float(np.linalg.norm(d)),
                    off_axis=float(np.arctan2(np.linalg.norm(t_cam[:2]), t_cam[2])),
                )
            )
        return out

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

    def update(self, frame: Frame | None, state: FlightState, now: float) -> GateEstimate:
        if self._t0 is None:
            self._t0 = now
        pos = np.asarray(state.pos, np.float64)
        dets = [] if frame is None else self.detect(frame.image, state)

        match: GateDetection | None = None
        if self.current is not None and dets:
            cand = min(dets, key=lambda d: float(np.linalg.norm(d.center - self.current.center)))
            if np.linalg.norm(cand.center - self.current.center) <= self.assoc_radius * max(
                1.0, self.current.aperture
            ):
                match = cand
        elif self.current is None and dets:
            match = min(dets, key=lambda d: d.distance * (1.0 + d.off_axis))
            self.current = _Tracked.of(match, now)
        if match is not None and self.current is not None:
            prev = self.current.prev_signed
            self.current = _Tracked.of(match, now)
            self.current.prev_signed = prev
            self._last_event = now

        next_det: GateDetection | None = None
        if self.current is not None:
            others = [d for d in dets if d is not match]
            ref = self.current
            farther = [d for d in others if d.distance > float(np.linalg.norm(ref.center - pos))]
            if farther:
                next_det = min(farther, key=lambda d: float(np.linalg.norm(d.center - ref.center)))
                self.next_known = _Tracked.of(next_det, now)

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
            curr = np.array([1.0, 0.0, self.scale.length(min(self.w, self.h))], _F)

        if next_det is not None:
            nxt = lookahead_visible(pos, next_det.center, next_det.yaw)
        elif self.next_known is not None:
            nxt = lookahead_unseen(pos, self.next_known.center)
        elif self.current is not None:
            nxt = lookahead_unseen(pos, self.current.center)
        else:
            nxt = lookahead_unseen(pos, pos + np.array([1.0, 0.0, 0.0]))

        return GateEstimate(
            target=target,
            curr_gate=curr,
            next_lookahead=nxt,
            gate_index=self.gate_index,
            lap=self.lap,
            status=status,  # type: ignore[arg-type]
            stamp=frame.stamp if frame is not None else now,
            passed=passed,
            info={"detections": len(dets), "matched": match is not None},
        )
