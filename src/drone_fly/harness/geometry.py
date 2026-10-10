"""Frames, rays and camera geometry.

Frames
------
* **Contract world**: right-handed, Z-up, metres (training scale after :mod:`scale`).
* **Body**: FLU — x forward, y left, z up. Quaternions ``[w, x, y, z]`` rotate body -> world.
* **Camera**: OpenCV — x right, y down, z along the optical axis. The FPV camera is pitched up
  by ``uptilt`` about the body y axis.
* **Unity** (Liftoff): left-handed, Y-up, Z-forward, X-right. Mapped by
  :func:`unity_to_contract_position` (X = z_u, Y = -x_u, Z = y_u).

Rays
----
:func:`ray_directions` reproduces the ~6-line cone basis of ``inference.raycast`` (velocity
heading above 0.3 m/s, else the target direction, else +x; ``right = fwd x up``) so a vision
estimator can sample along the *same* 40 rays the model was trained on. A test pins it
against ``raycast()``; ground-truth casting always calls ``contract.raycast()`` itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_F = np.float32

# Unity (x right, y up, z forward; left-handed) -> contract (X fwd, Y left, Z up).
# det = -1 (a reflection), which is why quaternion vector parts are negated after mapping.
UNITY_TO_CONTRACT = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]], dtype=_F)


# ----------------------------------------------------------------------------- rays


def ray_directions(
    pos: np.ndarray, vel: np.ndarray, target: np.ndarray | None, contract: Any
) -> np.ndarray:
    """World-frame unit ray directions ``(N_RAYS, 3)`` — the ``inference.raycast`` cone basis."""
    pos = np.asarray(pos, _F)
    v = np.asarray(vel, _F)
    sp = float(np.linalg.norm(v))
    if sp > 0.3:
        fwd = v / (sp + 1e-6)
    elif target is not None:
        t = np.asarray(target, _F) - pos
        fwd = t / (np.linalg.norm(t) + 1e-6)
    else:
        fwd = np.array([1.0, 0.0, 0.0], _F)
    wup = np.array([0.0, 0.0, 1.0], _F)
    right = np.cross(fwd, wup)
    rn = float(np.linalg.norm(right))
    right = np.array([1.0, 0.0, 0.0], _F) if rn < 1e-3 else right / (rn + 1e-6)
    up = np.cross(right, fwd)
    cone = np.asarray(contract.CONE, _F)
    return (cone[:, 0:1] * right + cone[:, 1:2] * up + cone[:, 2:3] * fwd).astype(_F)


# ----------------------------------------------------------------------------- quaternions


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit quaternion ``[w, x, y, z]`` (body -> world)."""
    w, x, y, z = (float(c) for c in np.asarray(q, np.float64))
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_from_vector(v: np.ndarray) -> float:
    return float(np.arctan2(v[1], v[0]))


# ----------------------------------------------------------------------------- Unity frames


def unity_to_contract_position(p_unity: np.ndarray) -> np.ndarray:
    """Unity position/velocity (x right, y up, z fwd) -> contract (X = z_u, Y = -x_u, Z = y_u)."""
    return (UNITY_TO_CONTRACT @ np.asarray(p_unity, _F).reshape(3)).astype(_F)


def unity_to_contract_quat(q_xyzw: np.ndarray) -> np.ndarray:
    """Unity quaternion ``[x, y, z, w]`` -> contract ``[w, x, y, z]``.

    The frame change ``M`` is a reflection (det -1). For a rotation ``R_u`` the contract rotation
    is ``M R_u M^T``; on the quaternion this maps the vector part through ``M`` and negates it
    (``v' = det(M) * M v``), leaving ``w`` unchanged. The result is renormalised.
    """
    x, y, z, w = (float(c) for c in np.asarray(q_xyzw, np.float64).reshape(4))
    vec = -(UNITY_TO_CONTRACT.astype(np.float64) @ np.array([x, y, z]))
    q = np.array([w, *vec], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-9:
        raise ValueError("degenerate quaternion")
    q /= n
    if q[0] < 0:  # canonical hemisphere (same rotation)
        q = -q
    return q.astype(_F)


# ----------------------------------------------------------------------------- camera


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera from horizontal FOV + resolution, pitched up by ``uptilt_deg``.

    Square pixels; principal point at the image centre. ``vfov`` follows from the aspect.
    """

    width: int
    height: int
    hfov_deg: float
    uptilt_deg: float = 0.0

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / np.tan(np.radians(self.hfov_deg) / 2.0)

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan((self.height / 2.0) / self.fy)))

    def intrinsics(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], np.float64)

    def body_to_camera(self) -> np.ndarray:
        """Rotation ``R_cb`` with ``v_cam = R_cb @ v_body`` (rows = camera axes in body FLU)."""
        t = np.radians(self.uptilt_deg)
        z = np.array([np.cos(t), 0.0, np.sin(t)])  # optical axis
        x = np.array([0.0, -1.0, 0.0])  # image right = body right
        y = np.cross(z, x)  # image down
        return np.stack([x, y, z])

    def world_to_camera(self, quat: np.ndarray) -> np.ndarray:
        """Rotation ``R_cw`` for a drone attitude ``quat`` (body -> world)."""
        return self.body_to_camera() @ quat_to_matrix(quat).T

    def project(self, points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Camera-frame points ``(N, 3)`` -> pixels ``(N, 2)`` and an in-front mask."""
        p = np.asarray(points_cam, np.float64).reshape(-1, 3)
        z = p[:, 2]
        front = z > 1e-6
        zs = np.where(front, z, 1.0)
        u = self.fx * p[:, 0] / zs + self.cx
        v = self.fy * p[:, 1] / zs + self.cy
        return np.stack([u, v], axis=1), front

    def in_image(self, pixels: np.ndarray, front: np.ndarray) -> np.ndarray:
        u, v = pixels[:, 0], pixels[:, 1]
        return front & (u >= 0) & (u <= self.width - 1) & (v >= 0) & (v <= self.height - 1)

    def pixel_rays(self, pixels: np.ndarray) -> np.ndarray:
        """Pixels ``(N, 2)`` -> camera-frame direction vectors with z = 1."""
        px = np.asarray(pixels, np.float64).reshape(-1, 2)
        return np.stack(
            [(px[:, 0] - self.cx) / self.fx, (px[:, 1] - self.cy) / self.fy, np.ones(len(px))],
            axis=1,
        )

    def scaled_to(self, width: int, height: int) -> CameraModel:
        """Same FOV/uptilt for a different capture resolution."""
        return CameraModel(width, height, self.hfov_deg, self.uptilt_deg)
