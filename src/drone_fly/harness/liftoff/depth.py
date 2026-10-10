"""Monocular-depth free-space estimator (first raycast-capable :class:`FreeSpaceEstimator`).

Reconstructs the contract's 40-ray scan (and ``obstacle_vision``) from a single camera:

1. A pluggable :class:`DepthModel` turns the latest frame into a depth map on its **own worker
   thread** (latest frame wins); the map is calibrated (affine ``scale * raw + shift``; with
   ``inverse: true`` the affine result is inverse depth) and published with its capture time
   and the drone pose at capture.
2. ``update()`` (synchronous, non-blocking) builds the ray fan with
   :func:`~drone_fly.harness.geometry.ray_directions` — the contract cone, velocity-aligned with
   the target-direction fallback below 0.3 m/s — and projects each ray world -> body -> camera ->
   pixel using the capture-time attitude.
3. In-front, in-image rays take a robust patch minimum (low percentile) of the depth map; the
   camera z-depth becomes range along the ray (``z / cos``), goes through :class:`WorldScale`, is
   clamped at ``RAY_RANGE * ray_range_scale`` and normalised to [0, 1] (1 = clear), as in
   ``inference.raycast``. Hits below ``ground.height_m`` above ``ground.z_m`` are ground, not
   obstacles (training rays never hit the floor), and read as clear.
4. Rays outside the image use the configured fill policy: ``clear`` | ``nearest_in_fov``
   (default: the value of the angularly nearest measured ray) | ``hold_last`` (the ray's last
   measured value, up to ``hold_max_age_s``, else clear). The in-FOV mask and coverage fraction
   are recorded.
5. ``obstacle_vision``: near-depth connected components (``depth_clusters``) back-projected to
   a centre + half-extent proxy, nearest two — or ``zeros``.

The first backend is Depth-Anything-V2 (metric) via ``transformers``, imported lazily when the
estimator starts. The runner fails safe when the estimate is older than ``max_estimate_age_s``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from drone_fly.harness.config import (
    require_bool,
    require_choice,
    require_int,
    require_number,
    require_vector,
)
from drone_fly.harness.course_senses import obstacle_vision_block
from drone_fly.harness.errors import HarnessConfigError, lazy_import
from drone_fly.harness.geometry import CameraModel, quat_to_matrix, ray_directions
from drone_fly.harness.interfaces import DepthModel, FreeSpaceEstimator
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import FlightState, Frame, FreeSpaceEstimate

log = logging.getLogger(__name__)

FILL_POLICIES = ("clear", "nearest_in_fov", "hold_last")
OBSTACLE_MODES = ("depth_clusters", "zeros")
DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


# ----------------------------------------------------------------------------- depth models


class DepthAnythingV2Model(DepthModel):
    """Depth-Anything-V2 (metric) through ``transformers`` (lazy import, lazy weights)."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        inverse: bool = False,
        **model_kwargs: Any,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.inverse = inverse
        self.model_kwargs = model_kwargs
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        tf = lazy_import("transformers", purpose="the monocular depth model")
        torch = lazy_import("torch", purpose="the monocular depth model")
        lazy_import("PIL.Image", package="pillow", purpose="the monocular depth model")
        dev = self.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = tf.AutoImageProcessor.from_pretrained(self.model_id)
        self._model = tf.AutoModelForDepthEstimation.from_pretrained(
            self.model_id, **self.model_kwargs
        ).to(dev)
        self._model.eval()
        self._torch = torch
        self.device = dev

    def predict(self, image: np.ndarray) -> np.ndarray:
        self.load()
        from PIL import Image

        torch = self._torch
        rgb = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]))
        inputs = self._processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pred = self._model(**inputs).predicted_depth  # (1, h, w)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=image.shape[:2], mode="bicubic", align_corners=False
            )
        return pred[0, 0].float().cpu().numpy()


def depth_model_from_config(cfg: dict[str, Any]) -> DepthModel:
    backend = require_choice(
        cfg.get("backend", "depth-anything-v2"), "depth.backend", ("depth-anything-v2",)
    )
    if backend == "depth-anything-v2":
        return DepthAnythingV2Model(
            model_id=str(cfg.get("model_id") or DEFAULT_MODEL_ID),
            device=require_choice(
                cfg.get("device", "auto"), "depth.device", ("auto", "cpu", "cuda")
            ),
            inverse=require_bool(cfg.get("inverse", False), "depth.inverse"),
            **(cfg.get("model_kwargs") or {}),
        )
    raise HarnessConfigError(f"unknown depth backend {backend!r}")  # pragma: no cover


# ----------------------------------------------------------------------------- calibration


@dataclass(frozen=True)
class DepthCalibration:
    """``metres = scale * raw + shift``; with ``inverse`` the affine result is inverse depth."""

    scale: float = 1.0
    shift: float = 0.0
    inverse: bool = False
    min_depth_m: float = 0.05

    def apply(self, raw: np.ndarray) -> np.ndarray:
        v = self.scale * np.asarray(raw, np.float32) + self.shift
        if self.inverse:
            v = 1.0 / np.maximum(v, 1e-6)
        return np.maximum(v, self.min_depth_m).astype(np.float32)


@dataclass(frozen=True)
class DepthResult:
    depth_m: np.ndarray  # H x W calibrated depth (target metres, camera z)
    stamp: float  # frame capture time
    quat: np.ndarray  # attitude at capture
    pos: np.ndarray  # position at capture (training scale)
    frame_index: int


# ----------------------------------------------------------------------------- estimator


class MonocularDepthFreeSpaceEstimator(FreeSpaceEstimator):
    provides = frozenset({"obstacle_vision", "raycast"})

    def __init__(
        self,
        model: DepthModel,
        camera: CameraModel,
        contract: Any,
        *,
        calibration: DepthCalibration | None = None,
        scale: WorldScale | None = None,
        patch_px: int = 3,
        patch_percentile: float = 10.0,
        ray_range_scale: float = 1.0,
        fill_policy: str = "nearest_in_fov",
        hold_max_age_s: float = 0.5,
        obstacle_mode: str = "depth_clusters",
        near_m: float = 3.0,
        cluster_stride: int = 8,
        min_cluster_cells: int = 4,
        ground_z_m: float | None = 0.0,
        ground_height_m: float = 0.15,
        frame_source: Any = None,
        state_source: Any = None,
        warmup_timeout_s: float = 60.0,
        clock: Any = time.perf_counter,
    ) -> None:
        require_choice(fill_policy, "free_space.fill_policy", FILL_POLICIES)
        require_choice(obstacle_mode, "free_space.obstacle_vision", OBSTACLE_MODES)
        self.model = model
        self.camera = camera
        self.contract = contract
        self.calibration = calibration or DepthCalibration(inverse=getattr(model, "inverse", False))
        self.scale = scale or WorldScale(1.0)
        self.patch_px = patch_px
        self.patch_percentile = patch_percentile
        self.ray_range_scale = ray_range_scale
        self.fill_policy = fill_policy
        self.hold_max_age_s = hold_max_age_s
        self.obstacle_mode = obstacle_mode
        self.near_m = near_m
        self.cluster_stride = cluster_stride
        self.min_cluster_cells = min_cluster_cells
        self.ground_z_m = ground_z_m
        self.ground_height_m = ground_height_m
        self.frame_source = frame_source
        self.state_source = state_source
        self.warmup_timeout_s = warmup_timeout_s
        self.clock = clock
        self._lock = threading.Lock()
        self._result: DepthResult | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self.reset()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> MonocularDepthFreeSpaceEstimator:
        conf = ctx.config
        dcfg = conf.section("depth")
        cam = conf.section("camera")
        res = cam.get("resolution")
        w, h = (
            (1920, 1080)
            if res is None
            else (int(v) for v in require_vector(res, "camera.resolution", 2))
        )
        camera = CameraModel(
            w,
            h,
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
        model = ctx.resources.get("depth_model") or depth_model_from_config(dcfg)
        calib = dcfg.get("calibration") or {}
        ground = dcfg.get("ground") or {}
        num = lambda d, k, default, **kw: float(require_number(d.get(k, default), k, **kw))  # noqa: E731
        return cls(
            model,
            camera,
            ctx.contract,
            calibration=DepthCalibration(
                scale=num(calib, "scale", 1.0),
                shift=num(calib, "shift", 0.0),
                inverse=require_bool(dcfg.get("inverse", False), "depth.inverse"),
                min_depth_m=num(calib, "min_depth_m", 0.05, min_value=0.0),
            ),
            scale=WorldScale(conf.world_scale),
            patch_px=require_int(dcfg.get("patch_px", 3), "depth.patch_px", min_value=0),
            patch_percentile=num(dcfg, "patch_percentile", 10.0, min_value=0.0, max_value=100.0),
            ray_range_scale=num(dcfg, "ray_range_scale", 1.0, min_value=0.0, min_exclusive=True),
            fill_policy=require_choice(
                dcfg.get("fill_policy", "nearest_in_fov"), "depth.fill_policy", FILL_POLICIES
            ),
            hold_max_age_s=num(dcfg, "hold_max_age_s", 0.5, min_value=0.0),
            obstacle_mode=require_choice(
                dcfg.get("obstacle_vision", "depth_clusters"),
                "depth.obstacle_vision",
                OBSTACLE_MODES,
            ),
            near_m=num(dcfg, "near_m", 3.0, min_value=0.0, min_exclusive=True),
            cluster_stride=require_int(
                dcfg.get("cluster_stride", 8), "depth.cluster_stride", min_value=1
            ),
            min_cluster_cells=require_int(
                dcfg.get("min_cluster_cells", 4), "depth.min_cluster_cells", min_value=1
            ),
            ground_z_m=None if ground.get("z_m") is None else num(ground, "z_m", 0.0),
            ground_height_m=num(ground, "height_m", 0.15, min_value=0.0),
            frame_source=ctx.resources.get("frame_source"),
            state_source=ctx.resources.get("state_source"),
            warmup_timeout_s=num(dcfg, "warmup_timeout_s", 60.0, min_value=0.0),
        )

    # ------------------------------------------------------------------ lifecycle

    def reset(self) -> None:
        n = int(self.contract.N_RAYS) if self.contract is not None else 40
        self._last_values = np.ones(n, np.float32)
        self._last_seen = np.full(n, -np.inf)

    def start(self) -> None:
        """Load the model and, with a frame source, start the worker and wait for a first map."""
        load = getattr(self.model, "load", None)
        if callable(load):
            load()
        if self.frame_source is None or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="depth-worker", daemon=True)
        self._thread.start()
        deadline = self.clock() + self.warmup_timeout_s
        while self.latest_result() is None and self.clock() < deadline:
            if self._worker_error is not None:
                raise self._worker_error
            time.sleep(0.01)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _worker(self) -> None:
        last_index = None
        while not self._stop.is_set():
            frame = self.frame_source.latest()
            state = self.state_source.latest() if self.state_source is not None else None
            if frame is None or state is None or frame.index == last_index:
                time.sleep(0.002)
                continue
            last_index = frame.index
            try:
                self.process(frame, state)
            except BaseException as exc:  # noqa: BLE001 - surfaced via age / start()
                self._worker_error = exc
                log.exception("depth worker failed")
                return

    def process(self, frame: Frame, state: FlightState) -> DepthResult:
        """Run the model on one frame and publish the result (worker thread or synchronous)."""
        depth = self.calibration.apply(self.model.predict(frame.image))
        result = DepthResult(
            depth, frame.stamp, np.asarray(state.quat), np.asarray(state.pos), frame.index
        )
        with self._lock:
            self._result = result
        return result

    def latest_result(self) -> DepthResult | None:
        with self._lock:
            return self._result

    # ------------------------------------------------------------------ estimation

    def _camera_for(self, depth: np.ndarray) -> CameraModel:
        h, w = depth.shape[:2]
        if (w, h) != (self.camera.width, self.camera.height):
            self.camera = self.camera.scaled_to(w, h)
        return self.camera

    def measure_rays(
        self, result: DepthResult, rays_w: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalised free space per ray + in-FOV mask (no fill)."""
        depth = result.depth_m
        cam = self._camera_for(depth)
        r_cw = cam.world_to_camera(result.quat)
        dirs = rays_w.astype(np.float64) @ r_cw.T
        px, front = cam.project(dirs)
        in_fov = cam.in_image(px, front)
        r_max = float(self.contract.RAY_RANGE) * self.ray_range_scale
        values = np.ones(len(rays_w), np.float32)
        h, w = depth.shape[:2]
        k = self.patch_px
        for i in np.flatnonzero(in_fov):
            u, v = int(round(px[i, 0])), int(round(px[i, 1]))
            patch = depth[max(v - k, 0) : min(v + k + 1, h), max(u - k, 0) : min(u + k + 1, w)]
            z = float(np.percentile(patch, self.patch_percentile))
            cos = dirs[i, 2] / (np.linalg.norm(dirs[i]) + 1e-12)
            rng = self.scale.length(z / max(cos, 1e-6))
            if self.ground_z_m is not None:
                hit_z = float(result.pos[2] + rays_w[i, 2] * rng)
                if hit_z < self.scale.length(self.ground_z_m + self.ground_height_m):
                    continue  # ground, not an obstacle
            values[i] = min(rng, r_max) / r_max
        return values, in_fov

    def fill(
        self, values: np.ndarray, in_fov: np.ndarray, rays_w: np.ndarray, now: float
    ) -> np.ndarray:
        out = values.copy()
        missing = ~in_fov
        if not missing.any():
            return out
        if self.fill_policy == "clear":
            out[missing] = 1.0
        elif self.fill_policy == "nearest_in_fov":
            if in_fov.any():
                sims = rays_w[missing] @ rays_w[in_fov].T
                out[missing] = values[in_fov][np.argmax(sims, axis=1)]
            else:
                out[missing] = 1.0
        else:  # hold_last
            fresh = (now - self._last_seen) <= self.hold_max_age_s
            out[missing] = np.where(fresh[missing], self._last_values[missing], 1.0)
        return out

    def obstacle_vision(self, result: DepthResult) -> np.ndarray:
        if self.obstacle_mode == "zeros":
            return np.zeros(8, np.float32)
        from scipy import ndimage

        depth = result.depth_m
        cam = self._camera_for(depth)
        s = self.cluster_stride
        vs, us = np.mgrid[s // 2 : depth.shape[0] : s, s // 2 : depth.shape[1] : s]
        z = depth[vs, us]
        rays_c = cam.pixel_rays(np.stack([us.ravel(), vs.ravel()], axis=1))
        pts_c = rays_c * z.reshape(-1, 1)  # target metres, camera frame
        r_wc = quat_to_matrix(result.quat) @ cam.body_to_camera().T
        pts_w = pts_c @ r_wc.T / self.scale.factor + result.pos  # training scale
        dist = np.linalg.norm(pts_w - result.pos, axis=1)
        near = dist < self.scale.length(self.near_m)
        if self.ground_z_m is not None:
            near &= pts_w[:, 2] >= self.scale.length(self.ground_z_m + self.ground_height_m)
        labels, n = ndimage.label(near.reshape(z.shape))
        centers, sizes = [], []
        flat = labels.ravel()
        for lab in range(1, n + 1):
            sel = flat == lab
            if sel.sum() < self.min_cluster_cells:
                continue
            p = pts_w[sel]
            half = 0.5 * (p.max(axis=0) - p.min(axis=0))
            size = float(max(half.max(), 1e-3))
            surface = p.mean(axis=0)
            view = surface - result.pos
            view /= np.linalg.norm(view) + 1e-9
            centers.append(surface + view * size)  # visible surface -> body centre proxy
            sizes.append(size)
        return obstacle_vision_block(
            result.pos, centers=np.array(centers).reshape(-1, 3), sizes=np.array(sizes)
        )

    def update(
        self, frame: Frame | None, state: FlightState, target: np.ndarray, now: float
    ) -> FreeSpaceEstimate:
        if self.frame_source is None and frame is not None:
            last = self.latest_result()
            if last is None or last.frame_index != frame.index or last.stamp != frame.stamp:
                self.process(frame, state)  # synchronous mode (offline eval, tests)
        if self._worker_error is not None:
            raise RuntimeError("depth worker failed") from self._worker_error
        result = self.latest_result()
        n = int(self.contract.N_RAYS)
        if result is None:
            # No depth yet: an estimate stamped -inf, which the runner's age check rejects.
            return FreeSpaceEstimate(
                obstacle_vision=np.zeros(8, np.float32),
                raycast=np.ones(n, np.float32),
                in_fov_mask=np.zeros(n, bool),
                stamp=-np.inf,
                info={"depth": "none"},
            )
        rays_w = ray_directions(state.pos, state.vel, target, self.contract)
        values, in_fov = self.measure_rays(result, rays_w)
        self._last_values[in_fov] = values[in_fov]
        self._last_seen[in_fov] = now
        rays = self.fill(values, in_fov, rays_w, now)
        return FreeSpaceEstimate(
            obstacle_vision=self.obstacle_vision(result),
            raycast=rays.astype(np.float32),
            in_fov_mask=in_fov,
            stamp=result.stamp,
            info={"frame_index": result.frame_index, "fill_policy": self.fill_policy},
        )
