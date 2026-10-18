"""A teaching-example perception plugin: "dark patch = obstacle" rays (UC-86 AC9).

**Never fly with this.** It exists to show the whole plugin path with nothing but NumPy:

1. *implement* -- subclass :class:`~drone_fly.harness.interfaces.FreeSpaceEstimator`;
2. *declare the contract* -- ``provides``, ``provenance = "screen"`` (it reads only pixels), the
   stamp rule (``frame.stamp`` of the newest frame used, ``-inf`` before the first frame);
3. *select it from config* --
   ``parts: {free_space_estimator: drone_fly.harness.examples.brightness:BrightnessRaysProvider}``
   (an import reference: nothing is added to ``builtins.py``);
4. *pass the conformance kit* -- :func:`drone_fly.harness.perception_kit.check_provider`;
5. *evaluate it* -- ``drone-fly perception eval --provider
   drone_fly.harness.examples.brightness:BrightnessRaysProvider --dataset ... --split held-out``.

Each of the 40 contract rays (the ``inference.raycast`` cone around the flight direction) is
projected into the frame with the camera model; a ray whose pixel patch has a mean grey level
below ``threshold`` reads ``blocked_free`` (default 0.5), else 1.0 (clear). Rays outside the image
read 1.0 with ``in_fov_mask`` False. ``obstacle_vision`` is zeros. Synchronous: ``update()``
processes the frame it is handed (it starts no thread).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from drone_fly.harness.config import require_number
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.geometry import CameraModel, ray_directions
from drone_fly.harness.interfaces import FreeSpaceEstimator
from drone_fly.harness.types import FlightState, Frame, FreeSpaceEstimate


class BrightnessRaysProvider(FreeSpaceEstimator):
    """See the module docstring (teaching example, never for flight)."""

    provides = frozenset({"raycast", "obstacle_vision"})
    provenance = "screen"

    def __init__(
        self,
        camera: CameraModel,
        cone: np.ndarray,
        *,
        threshold: float = 60.0,
        blocked_free: float = 0.5,
        patch_px: int = 4,
    ) -> None:
        self.camera = camera
        self._cone = SimpleNamespace(CONE=np.asarray(cone, np.float32))
        self.n_rays = int(self._cone.CONE.shape[0])
        self.threshold = float(threshold)
        self.blocked_free = float(blocked_free)
        self.patch_px = int(patch_px)
        self._held: FreeSpaceEstimate | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> BrightnessRaysProvider:
        from drone_fly.harness.liftoff.gate_cv import camera_from_config

        if ctx.contract is None:
            raise HarnessConfigError("BrightnessRaysProvider needs the inference contract (CONE)")
        camera = ctx.config.get("camera")
        if not isinstance(camera, dict):
            raise HarnessConfigError("BrightnessRaysProvider needs a 'camera' config section")
        opts = cfg.get("brightness") or {}
        return cls(
            camera_from_config(camera),
            np.asarray(ctx.contract.CONE, np.float32),
            threshold=float(require_number(opts.get("threshold", 60.0), "brightness.threshold")),
            blocked_free=float(
                require_number(
                    opts.get("blocked_free", 0.5), "brightness.blocked_free",
                    min_value=0.0, max_value=1.0,
                )
            ),
        )  # fmt: skip

    def reset(self) -> None:
        self._held = None  # no estimate until the next frame

    def update(
        self, frame: Frame | None, state: FlightState, target: np.ndarray, now: float
    ) -> FreeSpaceEstimate:
        if frame is None or (self._held is not None and self._held.stamp == frame.stamp):
            if self._held is not None:
                return self._held  # a gap holds the last estimate (and its stamp)
            return FreeSpaceEstimate(
                obstacle_vision=np.zeros(8, np.float32),
                raycast=np.ones(self.n_rays, np.float32),
                in_fov_mask=np.zeros(self.n_rays, bool),
                stamp=-np.inf,
            )
        image = frame.image
        h, w = image.shape[:2]
        cam = self.camera
        if (w, h) != (cam.width, cam.height):
            cam = self.camera = cam.scaled_to(w, h)
        rays_w = ray_directions(state.pos, state.vel, target, self._cone)
        dirs = rays_w.astype(np.float64) @ cam.world_to_camera(state.quat).T
        px, front = cam.project(dirs)
        in_fov = cam.in_image(px, front)
        grey = image.mean(axis=2) if image.ndim == 3 else image
        rays = np.ones(self.n_rays, np.float32)
        k = self.patch_px
        for i in np.flatnonzero(in_fov):
            u, v = int(round(px[i, 0])), int(round(px[i, 1]))
            patch = grey[max(v - k, 0) : min(v + k + 1, h), max(u - k, 0) : min(u + k + 1, w)]
            if patch.size and float(patch.mean()) < self.threshold:
                rays[i] = self.blocked_free
        self._held = FreeSpaceEstimate(
            obstacle_vision=np.zeros(8, np.float32),
            raycast=rays,
            in_fov_mask=np.asarray(in_fov, bool),
            stamp=float(frame.stamp),
            info={"frame_index": int(frame.index)},
        )
        return self._held
