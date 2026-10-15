"""Live HUD next-gate pointer reader (classic OpenCV, no ML) — UC74.

Liftoff renders an on-screen *next-gate pointer* (a small fixed HUD icon that points toward the
gate to fly through next). The UC70 run-2 student already consumes a ``("pointer", 4)`` observation
tail — a body-frame unit bearing (3) + an availability bit (1) — but live it is always ``None`` ->
zeros / bit 0, because nothing populates ``est.info["pointer"]`` (the learned tracker leaves it
unset). This module fills that gap with **classic image processing only** (HSV colour threshold +
contour-shape analysis, optional multi-scale template match) — zero ML, no torch, no model load.

It deliberately mirrors how gates become model inputs (:mod:`.gate_cv`): detect the icon in the
image, convert the pixel to a camera-frame ray via the configured FOV / principal point
(:class:`~drone_fly.harness.geometry.CameraModel`), and rotate it into the FLU body frame via the
camera uptilt. For a camera-locked HUD icon drawn at the gate's true projection, the body bearing
needs **only the uptilt model, not the live attitude** (attitude cancels): ``v_body = R_cb^T
v_cam`` where ``R_cb = body_to_camera()``. The result is the exact ``[cos(el)cos(az),
cos(el)sin(az), sin(el)]`` unit vector the sim contract emits (``perception.py`` /
``perception_ref.py``), so the already-trained student consumes it unchanged (no retrain, no
obs-schema bump — schema ``VERSION`` stays 3, ``DIM`` stays 65).

Masks / ROI (AC7). The pointer lives inside the HUD region the gate detector *masks out*. The
reader reads the **unmasked** frame over a dedicated pointer ROI, while the gate detector keeps
masking that region on its own copy, so the two consumers never fight. The ROI is resolved by a
**hybrid policy** (:class:`PointerRoi`): an explicit ``camera.pointer_roi`` rect wins; otherwise it
is derived from the HUD ``camera.masks`` geometry; both scale with capture resolution.

Off-axis trust guard (UC75). UC74's real-frame captures showed the HUD pointer is a faithful
projection only within ``|az| ~20-25 deg`` and clamps toward the screen edge beyond ~30-40 deg
(error 15-75 deg). ``camera.pointer.max_bearing_az_deg`` (default ``None`` = off) marks a detection
whose body-frame ``|az| = |atan2(by, bx)|`` exceeds it as :data:`UNAVAILABLE`, so the live reader
and the run2b noise model agree about where the pointer is trusted. A guarded block is identical
to "no icon" downstream; the only pointer false-positive gate is sim-side and already masked by
availability, so the guard composes with it consistently.

OpenCV is imported lazily on first use (``_cv2()``), matching the gate estimator's pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.config import require_number, require_vector
from drone_fly.harness.errors import HarnessConfigError, lazy_import
from drone_fly.harness.geometry import CameraModel

_F = np.float32

#: The degraded pointer block: zero bearing + availability bit 0 (pointer absent / ambiguous).
UNAVAILABLE = np.zeros(4, _F)


def _cv2() -> Any:
    return lazy_import("cv2", package="opencv-python", purpose="the HUD pointer reader")


@dataclass(frozen=True)
class PointerRoi:
    """A rectangular pointer ROI defined at a reference resolution (hybrid-resolved).

    ``rect`` is ``(x, y, w, h)`` in pixels at ``resolution`` (the grid it was authored on, the same
    convention as :class:`~drone_fly.harness.liftoff.masks.MaskRegion`). :meth:`rect_px` scales it
    to an actual frame size.
    """

    rect: tuple[float, float, float, float]
    resolution: tuple[int, int]

    @classmethod
    def from_config(cls, camera_cfg: dict[str, Any], masks: Any = None) -> PointerRoi | None:
        """Hybrid ROI: explicit ``camera.pointer_roi`` wins, else derive from ``camera.masks``.

        Returns ``None`` when neither an override nor masks are configured — the reader then scans
        the whole frame.
        """
        res = camera_cfg.get("resolution")
        raw = camera_cfg.get("pointer_roi")
        if raw is not None:
            x, y, w, h = require_vector(raw, "camera.pointer_roi", 4)
            if w <= 0 or h <= 0:
                raise HarnessConfigError("camera.pointer_roi width/height must be > 0")
            if res is None:
                raise HarnessConfigError("camera.pointer_roi needs camera.resolution (its grid)")
            rx, ry = require_vector(res, "camera.resolution", 2)
            return cls((x, y, w, h), (int(rx), int(ry)))
        if masks:  # derive from HUD mask geometry
            regions = masks.regions
            name = camera_cfg.get("pointer_roi_mask")
            if name is not None:
                regions = [r for r in regions if r.name == name]
                if not regions:
                    raise HarnessConfigError(
                        f"camera.pointer_roi_mask {name!r} matches no camera.masks region"
                    )
            if masks.resolution is None:
                return None
            x0 = min(r.rect[0] for r in regions)
            y0 = min(r.rect[1] for r in regions)
            x1 = max(r.rect[0] + r.rect[2] for r in regions)
            y1 = max(r.rect[1] + r.rect[3] for r in regions)
            return cls((x0, y0, x1 - x0, y1 - y0), masks.resolution)
        return None

    def rect_px(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Integer ``(x0, y0, x1, y1)`` clamped to a ``width x height`` frame."""
        sx, sy = width / self.resolution[0], height / self.resolution[1]
        x, y, w, h = self.rect
        x0 = max(int(np.floor(x * sx)), 0)
        y0 = max(int(np.floor(y * sy)), 0)
        x1 = min(int(np.ceil((x + w) * sx)), width)
        y1 = min(int(np.ceil((y + h) * sy)), height)
        return x0, y0, x1, y1


class HudPointerReader:
    """Detect the fixed HUD pointer icon and convert it to the sim pointer contract.

    :meth:`read` takes the **raw** (unmasked) BGR frame and returns a ``float32[4]`` =
    ``[bx, by, bz, avail]`` — the body-frame unit bearing (3) + availability bit (1) — exactly the
    shape / dtype / frame convention ``est.info["pointer"]`` expects. When the icon is absent or
    ambiguous the bearing is zeroed and the bit is 0 (:data:`UNAVAILABLE`), never a stale guess.

    No ML: HSV colour threshold + contour-shape (area / circularity / solidity) with an optional
    multi-scale greyscale template fallback. No torch, no learned weights, no gate-net import.
    """

    def __init__(
        self,
        camera: CameraModel,
        roi: PointerRoi | None,
        *,
        hsv_ranges: list[tuple[np.ndarray, np.ndarray]] | None = None,
        min_area_frac: float = 2e-4,
        max_area_frac: float = 0.05,
        min_circularity: float = 0.6,
        min_solidity: float = 0.8,
        morph_kernel: int = 3,
        ambiguity_ratio: float = 0.8,
        center_exclude_radius_frac: float = 0.0,
        template: str | Path | None = None,
        template_threshold: float = 0.7,
        template_scales: tuple[float, ...] = (0.8, 1.0, 1.25),
        max_bearing_az_deg: float | None = None,
    ) -> None:
        self.camera = camera
        self.roi = roi
        self.hsv_ranges = list(hsv_ranges or [])
        self.min_area_frac = float(min_area_frac)
        self.max_area_frac = float(max_area_frac)
        self.min_circularity = float(min_circularity)
        self.min_solidity = float(min_solidity)
        self.morph_kernel = int(morph_kernel)
        self.ambiguity_ratio = float(ambiguity_ratio)
        #: Drop candidates within this fraction of ``min(frame_w, frame_h)`` of the principal
        #: point. 0 disables it. Suppresses a fixed centre crosshair / aiming reticle (e.g.
        #: Liftoff's ACRO reticle) that is otherwise an identical-looking white-ring confuser.
        self.center_exclude_radius_frac = float(center_exclude_radius_frac)
        self.template_path = None if template is None else Path(template)
        self.template_threshold = float(template_threshold)
        self.template_scales = tuple(float(s) for s in template_scales)
        self._template: np.ndarray | None = None
        #: Off-axis trust guard (UC75): a detection whose body-frame ``|az|`` (deg) is strictly
        #: greater than this reads :data:`UNAVAILABLE`. ``None`` disables it (pre-UC75 behaviour).
        self.max_bearing_az_deg = None if max_bearing_az_deg is None else float(max_bearing_az_deg)

    @classmethod
    def from_config(
        cls, camera: CameraModel, camera_cfg: dict[str, Any], masks: Any = None
    ) -> HudPointerReader:
        """Build from the ``camera.pointer`` block (see :func:`pointer_reader_from_config`)."""
        pc = camera_cfg.get("pointer") or {}
        ranges = []
        for i, pair in enumerate(pc.get("hsv_ranges") or []):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise HarnessConfigError(
                    f"camera.pointer.hsv_ranges[{i}] must be [[h,s,v],[h,s,v]]"
                )
            lo = require_vector(pair[0], f"camera.pointer.hsv_ranges[{i}][0]", 3)
            hi = require_vector(pair[1], f"camera.pointer.hsv_ranges[{i}][1]", 3)
            ranges.append((np.array(lo, np.uint8), np.array(hi, np.uint8)))
        num = lambda k, d, **kw: float(  # noqa: E731
            require_number(pc.get(k, d), f"camera.pointer.{k}", **kw)
        )
        scales = pc.get("template_scales") or (0.8, 1.0, 1.25)
        return cls(
            camera,
            PointerRoi.from_config(camera_cfg, masks),
            hsv_ranges=ranges,
            min_area_frac=num("min_area_frac", 2e-4, min_value=0.0),
            max_area_frac=num("max_area_frac", 0.05, min_value=0.0, min_exclusive=True),
            min_circularity=num("min_circularity", 0.6, min_value=0.0, max_value=1.0),
            min_solidity=num("min_solidity", 0.8, min_value=0.0, max_value=1.0),
            morph_kernel=int(num("morph_kernel", 3, min_value=1.0)),
            ambiguity_ratio=num("ambiguity_ratio", 0.8, min_value=0.0, max_value=1.0),
            center_exclude_radius_frac=num(
                "center_exclude_radius_frac", 0.0, min_value=0.0, max_value=1.0
            ),
            template=pc.get("template"),
            template_threshold=num("template_threshold", 0.7, min_value=0.0, max_value=1.0),
            template_scales=tuple(float(s) for s in scales),
            # Not via `num`: it float()s the result, and null (= guard off) must stay None.
            max_bearing_az_deg=require_number(
                pc.get("max_bearing_az_deg"),
                "camera.pointer.max_bearing_az_deg",
                min_value=0.0,
                min_exclusive=True,
                max_value=180.0,
                allow_none=True,
            ),
        )

    # ------------------------------------------------------------------ geometry

    def _camera_for(self, width: int, height: int) -> CameraModel:
        """Intrinsics for the actual capture size (same FOV / uptilt if it differs)."""
        if (width, height) != (self.camera.width, self.camera.height):
            self.camera = self.camera.scaled_to(width, height)
        return self.camera

    def body_bearing(self, pixel: Any, width: int, height: int) -> np.ndarray:
        """Icon pixel ``(x, y)`` -> body-frame **unit** bearing (``float32[3]``, FLU).

        ``v_cam = pixel_rays(pixel)`` (camera frame), ``v_body = body_to_camera()^T v_cam``. The
        live attitude is intentionally absent: a camera-locked HUD icon is fixed in the camera
        frame, so attitude cancels. A principal-point icon maps to the optical axis
        ``[cos(uptilt), 0, sin(uptilt)]``.
        """
        cam = self._camera_for(int(width), int(height))
        ray = cam.pixel_rays(np.asarray(pixel, np.float64).reshape(1, 2))[0]
        body = cam.body_to_camera().T @ ray
        n = float(np.linalg.norm(body))
        if n < 1e-9:
            return np.zeros(3, _F)
        return (body / n).astype(_F)

    def resolve_roi(self, width: int, height: int) -> tuple[int, int, int, int]:
        """The pixel ROI ``(x0, y0, x1, y1)`` for a frame (whole frame when no ROI set)."""
        if self.roi is None:
            return 0, 0, int(width), int(height)
        x0, y0, x1, y1 = self.roi.rect_px(int(width), int(height))
        if x1 <= x0 or y1 <= y0:
            return 0, 0, int(width), int(height)
        return x0, y0, x1, y1

    # ------------------------------------------------------------------ detection

    def _candidates_hsv(self, roi_img: np.ndarray) -> list[tuple[float, float, float]]:
        """``(score, cx, cy)`` icon candidates in ROI-local pixels via HSV + contour shape."""
        if not self.hsv_ranges:
            return []
        cv2 = _cv2()
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in self.hsv_ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        k = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = float(roi_img.shape[0] * roi_img.shape[1])
        lo_area = self.min_area_frac * roi_area
        hi_area = self.max_area_frac * roi_area
        out: list[tuple[float, float, float]] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < lo_area or area > hi_area:
                continue
            perim = float(cv2.arcLength(c, True))
            if perim <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perim * perim)
            if circularity < self.min_circularity:
                continue
            hull_area = float(cv2.contourArea(cv2.convexHull(c)))
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < self.min_solidity:
                continue
            m = cv2.moments(c)
            if m["m00"] <= 0:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            out.append((area, float(cx), float(cy)))
        return out

    def _candidates_template(self, roi_img: np.ndarray) -> list[tuple[float, float, float]]:
        """``(score, cx, cy)`` from a multi-scale greyscale template match (fallback only)."""
        if self.template_path is None:
            return []
        cv2 = _cv2()
        if self._template is None:
            tmpl = cv2.imread(str(self.template_path), cv2.IMREAD_GRAYSCALE)
            if tmpl is None:
                raise HarnessConfigError(
                    f"camera.pointer.template not readable: {self.template_path}"
                )
            self._template = tmpl
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        best: tuple[float, float, float] | None = None
        for s in self.template_scales:
            th = max(int(round(self._template.shape[0] * s)), 1)
            tw = max(int(round(self._template.shape[1] * s)), 1)
            if th > gray.shape[0] or tw > gray.shape[1]:
                continue
            resized = cv2.resize(self._template, (tw, th), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= self.template_threshold and (best is None or max_val > best[0]):
                best = (float(max_val), max_loc[0] + tw / 2.0, max_loc[1] + th / 2.0)
        return [] if best is None else [best]

    def detect_icon(self, image: np.ndarray) -> np.ndarray | None:
        """Full-frame pixel ``(x, y)`` of the icon, or ``None`` (absent / ambiguous).

        Reads the raw frame over the resolved ROI. HSV + shape first; the template match is a
        fallback when HSV finds nothing. A fixed centre reticle (see
        ``center_exclude_radius_frac``) is dropped *before* the uniqueness test, so it neither wins
        nor spuriously triggers the ambiguity reject. Ambiguity reject: if the second-best
        remaining candidate's score is within ``ambiguity_ratio`` of the best, it is unavailable.
        """
        h, w = image.shape[:2]
        x0, y0, x1, y1 = self.resolve_roi(w, h)
        roi_img = image[y0:y1, x0:x1]
        if roi_img.size == 0:
            return None
        cands = self._candidates_hsv(roi_img)
        if not cands:
            cands = self._candidates_template(roi_img)
        # ROI-local -> full-frame pixels, then drop anything inside the centre-exclusion disk.
        full = [(score, cx + x0, cy + y0) for score, cx, cy in cands]
        if self.center_exclude_radius_frac > 0 and full:
            cam = self._camera_for(w, h)
            r2 = (self.center_exclude_radius_frac * min(w, h)) ** 2
            full = [c for c in full if (c[1] - cam.cx) ** 2 + (c[2] - cam.cy) ** 2 >= r2]
        if not full:
            return None
        full.sort(key=lambda t: -t[0])
        top = full[0][0]
        if len(full) >= 2 and top > 0 and full[1][0] / top >= self.ambiguity_ratio:
            return None  # best match is not unique enough -> unavailable
        _, cx, cy = full[0]
        return np.array([cx, cy], np.float64)

    def read(self, image: np.ndarray) -> np.ndarray:
        """Raw BGR frame -> ``float32[4]`` pointer block (:data:`UNAVAILABLE` when no icon).

        Also :data:`UNAVAILABLE` when the off-axis guard is set and the bearing's ``|az|`` is
        strictly greater than ``max_bearing_az_deg`` (exactly equal stays available).
        """
        px = self.detect_icon(image)
        if px is None:
            return UNAVAILABLE.copy()
        h, w = image.shape[:2]
        bearing = self.body_bearing(px, w, h)
        if not np.any(bearing):
            return UNAVAILABLE.copy()
        if self.max_bearing_az_deg is not None:
            az_deg = np.degrees(abs(np.arctan2(float(bearing[1]), float(bearing[0]))))
            if az_deg > self.max_bearing_az_deg:
                return UNAVAILABLE.copy()  # off-axis: the HUD pointer clamps here (UC74)
        return np.concatenate([bearing, [1.0]]).astype(_F)


def pointer_reader_from_config(
    camera: CameraModel, camera_cfg: dict[str, Any], masks: Any = None
) -> HudPointerReader | None:
    """A :class:`HudPointerReader` when ``camera.pointer`` is configured + enabled, else ``None``.

    Returning ``None`` keeps legacy configs (and the sim, which sources the pointer from truth)
    unchanged: with no reader the gate estimator leaves ``info["pointer"]`` unset, so the assembler
    falls back to zeros / bit 0 exactly as before.
    """
    pc = camera_cfg.get("pointer")
    if not isinstance(pc, dict) or not pc:
        return None
    if not pc.get("enabled", True):
        return None
    return HudPointerReader.from_config(camera, camera_cfg, masks)
