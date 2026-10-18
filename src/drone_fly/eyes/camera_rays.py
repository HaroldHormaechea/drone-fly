"""The 40 training rays seen through the FPV camera (deployable, NumPy only).

* **World basis**: :func:`drone_fly.harness.geometry.ray_directions`, the ``inference.raycast``
  cone basis (velocity heading above 0.3 m/s, else the target direction, else +x; roll from world
  up). The canonical cone comes from a frozen namespace (:func:`cone_namespace`) built from the
  weights meta, so inference never loads the contract.
* **Camera**: :class:`~drone_fly.harness.geometry.CameraModel` (OpenCV axes, pitched up by
  ``uptilt``). Per ray: the unit camera-frame direction, ``(u, v)`` in ``[-1, 1]``
  (``grid_sample`` with ``align_corners=True``: -1 / +1 are the first / last pixel centres) and
  ``in_view`` = in front of the camera, inside the image and outside every HUD mask.
* **Masks** are normalised ``[x0, y0, x1, y1]`` fractions of the frame (:func:`normalise_masks`),
  so they apply at any resolution.

Every quantity is scale-free (directions only), so the training-scale world of the
:class:`~drone_fly.harness.types.FlightState` is used as is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from drone_fly.harness.geometry import CameraModel, ray_directions

_F = np.float32

#: ``inference.raycast``'s heading switch: the velocity above this speed (training m/s).
HEADING_MIN_SPEED = 0.3
#: Where the cone's forward axis came from (labels record it per frame).
FWD_VELOCITY, FWD_TARGET, FWD_DEFAULT = 0, 1, 2
FWD_SOURCES = ("velocity", "target", "default")


def cone_namespace(cone: Any) -> SimpleNamespace:
    """A frozen ``contract``-like namespace carrying only ``CONE`` (for ``ray_directions``)."""
    c = np.array(cone, dtype=_F).reshape(-1, 3)
    c.setflags(write=False)
    return SimpleNamespace(CONE=c)


def fwd_source(vel: Any, target: Any | None) -> int:
    """Which branch of the cone basis applies (same thresholds as ``ray_directions``)."""
    sp = float(np.linalg.norm(np.asarray(vel, _F)))
    if sp > HEADING_MIN_SPEED:
        return FWD_VELOCITY
    return FWD_TARGET if target is not None else FWD_DEFAULT


def normalise_masks(rects_px: Sequence[Sequence[float]], resolution: Sequence[float]) -> list:
    """``camera.masks`` pixel rects ``[x, y, w, h]`` at ``resolution`` -> ``[x0, y0, x1, y1]``."""
    w, h = float(resolution[0]), float(resolution[1])
    out = []
    for x, y, rw, rh in rects_px:
        out.append(
            [float(x) / w, float(y) / h, (float(x) + float(rw)) / w, (float(y) + float(rh)) / h]
        )
    return out


def masks_from_camera_config(cam: dict[str, Any]) -> list:
    """Normalised masks of a ``camera`` config section (``[]`` when none are configured)."""
    raw = cam.get("masks") or []
    if not raw:
        return []
    res = cam.get("resolution")
    if res is None:
        raise ValueError("camera.masks needs camera.resolution (the masks' pixel grid)")
    return normalise_masks([m["rect"] for m in raw], res)


def in_masks(
    pixels: np.ndarray, width: int, height: int, masks: Sequence[Sequence[float]]
) -> np.ndarray:
    """``(N,)`` bool: pixel points ``(N, 2)`` inside a normalised mask rectangle."""
    p = np.asarray(pixels, np.float64).reshape(-1, 2)
    inside = np.zeros(len(p), bool)
    for x0, y0, x1, y1 in masks:
        inside |= (
            (p[:, 0] >= x0 * width)
            & (p[:, 0] < x1 * width)
            & (p[:, 1] >= y0 * height)
            & (p[:, 1] < y1 * height)
        )
    return inside


@dataclass(frozen=True)
class RayView:
    """The rays of one pose: world directions, camera directions, image coords, visibility."""

    rays_world: np.ndarray  # (N, 3) unit, world frame
    rays_cam: np.ndarray  # (N, 3) unit, OpenCV camera frame
    uv: np.ndarray  # (N, 2) in [-1, 1] (clipped; meaningful only where in_view)
    in_view: np.ndarray  # (N,) bool
    fwd_source: int


def project_rays(
    rays_world: np.ndarray,
    quat: Any,
    camera: CameraModel,
    masks: Sequence[Sequence[float]] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rays_cam, uv, in_view)`` of world directions for a drone attitude (body -> world)."""
    r_cw = camera.world_to_camera(np.asarray(quat, np.float64))
    d = np.asarray(rays_world, np.float64) @ r_cw.T
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    px, front = camera.project(d)
    view = camera.in_image(px, front)
    if masks:
        view &= ~in_masks(px, camera.width, camera.height, masks)
    u = 2.0 * px[:, 0] / max(camera.width - 1, 1) - 1.0
    v = 2.0 * px[:, 1] / max(camera.height - 1, 1) - 1.0
    uv = np.clip(np.stack([u, v], axis=1), -1.0, 1.0)
    uv[~front] = 0.0
    return d.astype(_F), uv.astype(_F), view


def ray_view(
    pos: Any,
    vel: Any,
    quat: Any,
    target: Any | None,
    cone: SimpleNamespace,
    camera: CameraModel,
    masks: Sequence[Sequence[float]] = (),
) -> RayView:
    """The 40 training rays of a pose, through the camera (see the module docstring)."""
    rays = ray_directions(np.asarray(pos, _F), np.asarray(vel, _F), target, cone)
    rays_cam, uv, view = project_rays(rays, quat, camera, masks)
    return RayView(rays.astype(_F), rays_cam, uv, view, fwd_source(vel, target))


def camera_from_meta(cam: dict[str, Any]) -> CameraModel:
    """The :class:`CameraModel` of a weights / labels meta ``camera`` block."""
    w, h = (int(v) for v in cam["resolution"])
    return CameraModel(w, h, float(cam["hfov_deg"]), float(cam.get("uptilt_deg", 0.0)))


def camera_meta(camera: CameraModel) -> dict[str, Any]:
    return {
        "resolution": [int(camera.width), int(camera.height)],
        "hfov_deg": float(camera.hfov_deg),
        "uptilt_deg": float(camera.uptilt_deg),
    }
