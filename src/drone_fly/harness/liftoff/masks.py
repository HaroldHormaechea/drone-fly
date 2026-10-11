"""Screen-region masks: HUD, stick widgets, lap / ghost list (``camera.masks``).

Each region is ``{name, rect: [x, y, w, h]}`` in pixels at ``camera.resolution`` (required when
masks are configured) and is scaled to the actual frame size. Masks are layout-specific (one set
per resolution + HUD configuration). Consumers:

* gate estimators — the image is masked before detection, and a detection whose centre lies in a
  mask is dropped (:class:`~drone_fly.harness.liftoff.gate_cv.TrackingGateEstimator`);
* the depth estimator — masked pixels become ``+inf`` after calibration, so no ray hits them and
  they form no obstacle cluster;
* the auto-labeller — keypoints inside a mask are not visible, and training images are masked the
  same way.

Pure NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drone_fly.harness.config import require_vector
from drone_fly.harness.errors import HarnessConfigError


@dataclass(frozen=True)
class MaskRegion:
    name: str
    rect: tuple[float, float, float, float]  # x, y, w, h at the reference resolution


class FrameMasks:
    """A set of rectangular masks defined at a reference resolution."""

    def __init__(
        self, regions: list[MaskRegion] | tuple[MaskRegion, ...] = (), resolution=None
    ) -> None:
        self.regions = tuple(regions)
        if self.regions and resolution is None:
            raise HarnessConfigError("camera.masks needs camera.resolution (the masks' pixel grid)")
        self.resolution = None if resolution is None else (int(resolution[0]), int(resolution[1]))

    @classmethod
    def from_config(cls, camera_cfg: dict[str, Any]) -> FrameMasks:
        raw = camera_cfg.get("masks") or []
        if not isinstance(raw, list):
            raise HarnessConfigError("camera.masks must be a list of {name, rect: [x, y, w, h]}")
        regions = []
        for i, m in enumerate(raw):
            if not isinstance(m, dict) or "rect" not in m:
                raise HarnessConfigError(f"camera.masks[{i}] must be {{name, rect: [x, y, w, h]}}")
            x, y, w, h = require_vector(m["rect"], f"camera.masks[{i}].rect", 4)
            if w <= 0 or h <= 0:
                raise HarnessConfigError(f"camera.masks[{i}].rect width/height must be > 0")
            regions.append(MaskRegion(str(m.get("name") or f"mask{i}"), (x, y, w, h)))
        res = camera_cfg.get("resolution")
        resolution = None if res is None else require_vector(res, "camera.resolution", 2)
        return cls(regions, resolution)

    def __bool__(self) -> bool:
        return bool(self.regions)

    def __len__(self) -> int:
        return len(self.regions)

    def rects(self, width: int, height: int) -> np.ndarray:
        """Integer ``(N, 4)`` ``x0, y0, x1, y1`` rectangles scaled to a ``width x height`` frame."""
        if not self.regions:
            return np.zeros((0, 4), np.int64)
        assert self.resolution is not None
        sx, sy = width / self.resolution[0], height / self.resolution[1]
        out = []
        for r in self.regions:
            x, y, w, h = r.rect
            x0, y0 = int(np.floor(x * sx)), int(np.floor(y * sy))
            x1, y1 = int(np.ceil((x + w) * sx)), int(np.ceil((y + h) * sy))
            out.append([max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)])
        return np.asarray(out, np.int64)

    def valid_mask(self, width: int, height: int) -> np.ndarray:
        """``height x width`` bool map, True where pixels are usable (not masked)."""
        ok = np.ones((height, width), bool)
        for x0, y0, x1, y1 in self.rects(width, height):
            ok[y0:y1, x0:x1] = False
        return ok

    def apply(self, image: np.ndarray, fill: int | float = 0) -> np.ndarray:
        """A copy of ``image`` with every masked region set to ``fill`` (unchanged if no masks)."""
        if not self.regions:
            return image
        h, w = image.shape[:2]
        out = image.copy()
        for x0, y0, x1, y1 in self.rects(w, h):
            out[y0:y1, x0:x1] = fill
        return out

    def contains(self, points: np.ndarray, width: int, height: int) -> np.ndarray:
        """``(N,)`` bool: which pixel points ``(N, 2)`` fall inside a mask."""
        p = np.asarray(points, np.float64).reshape(-1, 2)
        inside = np.zeros(len(p), bool)
        for x0, y0, x1, y1 in self.rects(width, height):
            inside |= (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
        return inside
