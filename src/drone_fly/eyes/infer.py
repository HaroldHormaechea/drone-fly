"""``EyesEstimator``: vision-only inference of the 40 rays (deployable).

``update(frame, state, target)`` consumes ONLY a camera frame, the telemetry
:class:`~drone_fly.harness.types.FlightState` (training scale) and the current gate target (contract
frame, the gate estimator's output; it orients the cone below 0.3 m/s exactly as
``inference.raycast`` does). It never sees track geometry, labels or oracle data: this module
imports numpy, torch, :mod:`drone_fly.harness.geometry` / :mod:`~drone_fly.harness.types` and the
deployable eyes modules only.

Per call
    1. ``state.discontinuity`` or a frame stamp going backwards -> :meth:`reset` (a new segment);
       the same frame stamp again returns the previous output (frames are never duplicated);
    2. the frame is prepared (:func:`~drone_fly.eyes.model.prepare_frame`) and buffered;
    3. history: :func:`~drone_fly.eyes.history.select_history` over the buffer (``K - 1`` frames
       at ``t - j dt`` +- 50 % of ``dt``). Missing (startup, a gap) -> ``history_ok = False``,
       ``free = 1`` and ``conf = 0`` for every ray: no estimate, never a guess from duplicated
       frames;
    4. rays (:func:`~drone_fly.eyes.camera_rays.ray_view`), ego-motion
       (:func:`~drone_fly.eyes.ego.ego_vector`, rates from the stride-1 history frame) and the
       actual history offsets (:func:`~drone_fly.eyes.history.history_dt`);
    5. one forward pass.

Wrapping this as a registered ``free_space_estimator`` part is a follow-up (UC-73).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drone_fly.eyes.camera_rays import camera_from_meta, cone_namespace, ray_view
from drone_fly.eyes.ego import ego_vector
from drone_fly.eyes.history import default_tol, history_dt, select_history
from drone_fly.eyes.model import EyesDepthNet, EyesNet, build_from_meta, prepare_frame
from drone_fly.harness.types import FlightState, Frame


@dataclass(frozen=True)
class EyesOutput:
    free: np.ndarray  # (R,) in [0, 1], 1 = clear to the scaled range
    conf: np.ndarray  # (R,) in [0, 1]
    in_view: np.ndarray  # (R,) bool
    history_ok: bool
    stamp: float  # capture time of the frame the estimate was computed from
    timings_s: dict[str, float] = field(default_factory=dict)


def resolve_device(device: str | None) -> str:
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(device)


def load_eyes_weights(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(meta, state_dict)`` of an ``eyes.pt`` file."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "meta" not in blob or "state_dict" not in blob:
        raise ValueError(f"{path} is not an eyes weights file (meta + state_dict)")
    return blob["meta"], blob["state_dict"]


class EyesEstimator:
    def __init__(
        self, net: EyesNet | EyesDepthNet, meta: dict[str, Any], device: str = "cpu"
    ) -> None:
        self.meta = meta
        self.device = resolve_device(device)
        # A v3 (depth-backbone) net holds its frozen backbone internally and is dispatched by
        # build_from_meta; the live/bench forward contract (6 args -> free, conf[, hit]) is kept.
        self.net = net.to(self.device).eval()
        inp = meta["input"]
        self.width, self.height = int(inp["width"]), int(inp["height"])
        self.k_frames = int(inp["k_frames"])
        self.dt_s = float(inp["dt_s"])
        self.tol_s = float(inp.get("hist_tol_s", default_tol(self.dt_s)))
        self.cone = cone_namespace(meta["cone"])
        self.camera = camera_from_meta(meta["camera"])
        self.masks = [list(m) for m in meta.get("masks") or []]
        self.n_rays = int(meta["n_rays"])
        self._keep_s = (self.k_frames - 1) * self.dt_s + self.tol_s + 0.25
        self.reset()

    @classmethod
    def from_weights(cls, path: str | Path, device: str = "auto") -> EyesEstimator:
        meta, state = load_eyes_weights(path)
        net = build_from_meta(meta)
        net.load_state_dict(state)
        return cls(net, meta, device)

    def reset(self) -> None:
        """Start a new segment: the buffered history is dropped."""
        self._times: list[float] = []
        self._frames: list[torch.Tensor] = []
        self._quats: list[np.ndarray] = []
        self._last: EyesOutput | None = None

    def _camera_for(self, image: np.ndarray) -> Any:
        h, w = image.shape[:2]
        if (w, h) != (self.camera.width, self.camera.height):
            self.camera = self.camera.scaled_to(w, h)
        return self.camera

    def update(self, frame: Frame, state: FlightState, target: Any | None = None) -> EyesOutput:
        t0 = time.perf_counter()
        stamp = float(frame.stamp)
        if self._times and stamp == self._times[-1] and self._last is not None:
            return self._last
        if state.discontinuity or (self._times and stamp < self._times[-1]):
            self.reset()
        cuda = self.device.startswith("cuda")
        x = prepare_frame(frame.image, self.width, self.height, self.masks, self.device)
        self._times.append(stamp)
        self._frames.append(x)
        self._quats.append(np.asarray(state.quat, np.float64))
        while self._times and self._times[0] < stamp - self._keep_s:
            self._times.pop(0)
            self._frames.pop(0)
            self._quats.pop(0)
        if cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        i = len(self._times) - 1
        hist = select_history(self._times, i, self.k_frames, self.dt_s, self.tol_s)
        camera = self._camera_for(frame.image)
        view = ray_view(state.pos, state.vel, state.quat, target, self.cone, camera, self.masks)
        if hist is None:
            out = EyesOutput(
                np.ones(self.n_rays, np.float32), np.zeros(self.n_rays, np.float32),
                view.in_view, False, stamp,
                {"preprocess_s": t1 - t0, "forward_s": 0.0, "total_s": time.perf_counter() - t0},
            )  # fmt: skip
            self._last = out
            return out
        j1 = hist[0] if hist else None
        if j1 is None:
            ego = ego_vector(state.quat, state.vel)
        else:
            ego = ego_vector(state.quat, state.vel, self._quats[j1], stamp - self._times[j1])
        frames = torch.cat([self._frames[i], *[self._frames[j] for j in hist]], dim=0)
        dev = self.device
        with torch.no_grad():
            # A v2 (hit-head) net returns a third output; deployment uses only free + conf (the
            # hit head's gain is carried into the sim via the re-derived ray-noise model), so we
            # take the first two outputs and ignore any extra.
            net_out = self.net(
                frames.unsqueeze(0),
                torch.from_numpy(view.uv).to(dev).unsqueeze(0),
                torch.from_numpy(view.rays_cam).to(dev).unsqueeze(0),
                torch.from_numpy(view.in_view.astype(np.float32)).to(dev).unsqueeze(0),
                torch.from_numpy(ego).to(dev).unsqueeze(0),
                torch.from_numpy(history_dt(self._times, i, hist)).to(dev).unsqueeze(0),
            )
            free, logit = net_out[0], net_out[1]
            free_np = free[0].float().cpu().numpy()
            conf_np = torch.sigmoid(logit[0]).float().cpu().numpy()
        t2 = time.perf_counter()
        out = EyesOutput(
            free_np.astype(np.float32), conf_np.astype(np.float32), view.in_view, True, stamp,
            {"preprocess_s": t1 - t0, "forward_s": t2 - t1, "total_s": t2 - t0},
        )  # fmt: skip
        self._last = out
        return out
