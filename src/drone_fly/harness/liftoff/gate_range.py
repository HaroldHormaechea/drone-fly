"""Size-based gate range and PnP plausibility checks (UC-91, ``gates.learned.range_mode``).

The learned estimator already gets range from the gate's known geometry: IPPE PnP of the kept
keypoints against the class's object points (:func:`~drone_fly.harness.liftoff.gate_cv.
solve_planar_pnp`). This module adds an independent cross-check for ``range_mode: pnp_checked``:

* **Size range** (:func:`size_range`): the area of the kept keypoints' convex hull on the gate
  (metres², from the gate-type table) against the solid angle they subtend. The image points are
  turned into camera rays and projected onto the plane perpendicular to their mean ray at unit
  distance, so lens position (the wide 120 deg FOV stretches off-axis pixels by up to
  ``1 / cos^3``) and the principal point cancel; ``r = sqrt(A_obj / A_tangent)``. A gate seen at
  an angle ``phi`` to its normal shrinks by ``cos(phi)``, so ``r_size = r / sqrt(cos(phi))`` is
  never below the true range: the PnP / size ratio is ``sqrt(cos(phi))`` for a correct pose (0.71
  at 60 deg) — the band is asymmetric.
* **Validity** (:func:`size_range_valid`): size range needs the whole kept hull on the gate. It is
  invalid when a kept keypoint lies within ``edge_margin_px`` of the image border or a mask (a
  truncated box), when the kept hull covers less than ``min_hull_frac`` of the class's full
  hull, or with fewer than 4 points.
* **Check** (:func:`check_detection`): (a) ``size_ratio_min <= r_pnp / r_size <=
  size_ratio_max`` when the size range is valid; (b) the world gate-plane normal tilts at most
  ``max_normal_tilt_deg`` out of the horizontal (gates stand upright; the IPPE planar-ambiguity
  flip and wild solutions tilt it). With an invalid size range only (b) applies — the documented
  truncated-box fallback. ``max_normal_tilt_deg: null`` disables (b).

Pure NumPy: microseconds per detection, no measurable cost at the 26-31 Hz live loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from drone_fly.harness.config import require_choice, require_number
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.geometry import CameraModel

#: ``gates.learned.range_mode``: ``pnp`` (PnP range, every solved detection kept; the default)
#: or ``pnp_checked`` (PnP range, detections failing :func:`check_detection` dropped).
RANGE_MODES = ("pnp", "pnp_checked")
#: Why :func:`check_detection` rejects a detection.
REJECT_REASONS = ("size_ratio", "normal_tilt")


@dataclass(frozen=True)
class RangeCheck:
    """``gates.learned.range_check`` (see the module docstring)."""

    size_ratio_min: float = 0.47
    size_ratio_max: float = 1.33
    max_normal_tilt_deg: float | None = 30.0
    edge_margin_px: float = 8.0
    min_hull_frac: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.size_ratio_min < self.size_ratio_max:
            raise HarnessConfigError(
                "gates.learned.range_check needs 0 < size_ratio_min < size_ratio_max, got "
                f"{self.size_ratio_min} / {self.size_ratio_max}"
            )

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> RangeCheck:
        c = cfg or {}
        unknown = sorted(set(c) - set(cls.__dataclass_fields__))
        if unknown:
            raise HarnessConfigError(f"unknown gates.learned.range_check keys {unknown}")
        d = cls()
        k = "gates.learned.range_check"

        def num(key: str, **kw: Any) -> float:
            return float(require_number(c.get(key, getattr(d, key)), f"{k}.{key}", **kw))

        tilt = c.get("max_normal_tilt_deg", d.max_normal_tilt_deg)
        return cls(
            size_ratio_min=num("size_ratio_min", min_value=0.0, min_exclusive=True),
            size_ratio_max=num("size_ratio_max", min_value=0.0, min_exclusive=True),
            max_normal_tilt_deg=None
            if tilt is None
            else float(
                require_number(tilt, f"{k}.max_normal_tilt_deg", min_value=0.0, max_value=90.0)
            ),
            edge_margin_px=num("edge_margin_px", min_value=0.0),
            min_hull_frac=num("min_hull_frac", min_value=0.0, max_value=1.0),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def range_mode_from_config(value: Any) -> str:
    return require_choice(value, "gates.learned.range_mode", RANGE_MODES)


# ----------------------------------------------------------------------------- geometry


def hull_area(points: np.ndarray) -> float:
    """Area of the convex hull of ``(N, 2)`` points (monotone chain + shoelace; 0 for N < 3)."""
    p = np.unique(np.asarray(points, np.float64).reshape(-1, 2), axis=0)
    if len(p) < 3:
        return 0.0
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts: np.ndarray) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for q in pts:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) > 0:
                    break
                out.pop()
            out.append(q)
        return out

    lower, upper = half(p), half(p[::-1])
    h = np.array(lower[:-1] + upper[:-1])
    x, y = h[:, 0], h[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def tangent_points(image_points: np.ndarray, camera: CameraModel) -> np.ndarray:
    """Pixels -> ``(N, 2)`` points on the plane perpendicular to their mean ray at distance 1."""
    rays = camera.pixel_rays(image_points)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    c = rays.mean(axis=0)
    c /= np.linalg.norm(c)
    p = rays / (rays @ c)[:, None]  # on the plane c . x = 1
    e1 = np.cross(c, [0.0, 1.0, 0.0])
    if np.linalg.norm(e1) < 1e-9:
        e1 = np.cross(c, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    return np.stack([p @ e1, p @ e2], axis=1)


def size_range(object_points: np.ndarray, image_points: np.ndarray, camera: CameraModel) -> float:
    """Range (object units) from the kept points' hull area vs the solid angle they subtend.

    ``inf`` when the image hull is degenerate. Exact for a gate facing the line of sight; an
    oblique gate reads farther by ``1 / sqrt(cos(phi))``.
    """
    a_obj = hull_area(np.asarray(object_points, np.float64).reshape(-1, 3)[:, :2])
    a_img = hull_area(tangent_points(image_points, camera))
    if a_img <= 0.0 or a_obj <= 0.0:
        return float("inf")
    return float(np.sqrt(a_obj / a_img))


def size_range_valid(
    image_points: np.ndarray,
    object_points: np.ndarray,
    full_object_points: np.ndarray,
    width: int,
    height: int,
    check: RangeCheck,
    masks: Any = None,
) -> tuple[bool, str | None]:
    """``(valid, why_not)``: can :func:`size_range` be trusted for these kept keypoints?"""
    img = np.asarray(image_points, np.float64).reshape(-1, 2)
    if len(img) < 4:
        return False, "few_points"
    m = float(check.edge_margin_px)
    u, v = img[:, 0], img[:, 1]
    if ((u < m) | (u > width - 1 - m) | (v < m) | (v > height - 1 - m)).any():
        return False, "image_edge"
    if masks:
        near = np.zeros(len(img), bool)
        for dx, dy in ((0, 0), (-m, 0), (m, 0), (0, -m), (0, m)):
            near |= masks.contains(img + np.array([dx, dy]), width, height)
        if near.any():
            return False, "mask_edge"
    full = hull_area(np.asarray(full_object_points, np.float64).reshape(-1, 3)[:, :2])
    kept = hull_area(np.asarray(object_points, np.float64).reshape(-1, 3)[:, :2])
    if full <= 0.0 or kept < check.min_hull_frac * full:
        return False, "small_hull"
    return True, None


def normal_tilt_deg(n_world: np.ndarray) -> float:
    """Angle (deg) of a gate-plane normal out of the horizontal plane (0 = upright gate)."""
    n = np.asarray(n_world, np.float64).reshape(3)
    return float(np.degrees(np.arcsin(min(1.0, abs(n[2]) / max(np.linalg.norm(n), 1e-12)))))


def check_detection(
    r_pnp: float, r_size: float | None, n_world: np.ndarray, check: RangeCheck
) -> tuple[bool, str | None]:
    """``(accepted, reason)``; ``r_size`` is ``None`` when the size range is invalid."""
    if r_size is not None and np.isfinite(r_size) and r_size > 0:
        ratio = float(r_pnp) / float(r_size)
        if not check.size_ratio_min <= ratio <= check.size_ratio_max:
            return False, "size_ratio"
    if check.max_normal_tilt_deg is not None and normal_tilt_deg(n_world) > (
        check.max_normal_tilt_deg
    ):
        return False, "normal_tilt"
    return True, None


def evaluate_detection(
    t_cam: np.ndarray,
    n_world: np.ndarray,
    image_points: np.ndarray,
    object_points: np.ndarray,
    full_object_points: np.ndarray,
    camera: CameraModel,
    check: RangeCheck,
    masks: Any = None,
) -> dict[str, Any]:
    """Everything :func:`check_detection` decides on, for one solved detection.

    ``{"accepted", "reason", "r_pnp", "r_size" (None when invalid), "r_size_raw",
    "size_invalid" (why, or None), "tilt_deg"}``.
    """
    r_pnp = float(np.linalg.norm(np.asarray(t_cam, np.float64)))
    valid, why = size_range_valid(
        image_points,
        object_points,
        full_object_points,
        camera.width,
        camera.height,
        check,
        masks,
    )
    raw = size_range(object_points, image_points, camera) if len(image_points) >= 3 else None
    r_size = raw if valid else None
    ok, reason = check_detection(r_pnp, r_size, n_world, check)
    return {
        "accepted": ok,
        "reason": reason,
        "r_pnp": r_pnp,
        "r_size": r_size,
        "r_size_raw": raw,
        "size_invalid": why,
        "tilt_deg": normal_tilt_deg(n_world),
    }
