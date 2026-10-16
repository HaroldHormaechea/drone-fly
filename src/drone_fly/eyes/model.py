"""``EyesNet``: stacked-frame CNN + per-ray queries -> 40 free distances and confidences.

Input
    ``frames``: ``(B, 3K, H, W)`` uint8 RGB, the current frame first then the history frames
    (``t``, ``t - dt``, ...), each downscaled with :func:`prepare_frame` (area resize to
    ``192 x 80``, HUD masks zeroed at the low resolution). Per ray: ``uv`` ``(B, R, 2)`` in
    ``[-1, 1]``, the unit camera-frame direction ``rays_cam`` ``(B, R, 3)`` and ``in_view``
    ``(B, R)``; plus ``ego`` ``(B, 10)`` (:mod:`drone_fly.eyes.ego`) and ``hist_dt``
    ``(B, K - 1)``: the ACTUAL time offsets (seconds) of the history frames (the stride is chosen
    within +-50 %, so the net is told the true one).

Network
    Five stride-2 stages (32-48-64-96-128 channels; conv-BN-ReLU stride 2, then stride 1), a global
    average pool of the last stage, and per ray a bilinear ``grid_sample`` of the stride-8 and
    stride-16 maps at ``(u, v)`` -- **zeroed when the ray is out of view**, so an unseen ray is
    predicted from the global context, its direction and the ego-motion only (a learned prior).
    The query ``[stage-3 (64) || stage-4 (96) || global (128) || ray_cam (3) || in_view (1) ||
    ego (10) || hist_dt (K - 1)]`` goes through a shared 256-128 MLP to two outputs:
    ``free = sigmoid(.)`` in ``[0, 1]`` (1 = clear to the scaled range) and a confidence logit.

Stacking K frames lets the forward rays use looming (the growth of an obstacle's image between
frames, i.e. time-to-contact): parallax gives no depth at the focus of expansion, which is where
the velocity-aligned cone points.

:func:`build_from_meta` rebuilds the net from the weights meta (see :func:`default_meta`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

FORMAT = "drone-fly-eyes"
ARCH_VERSION = 2
CHANNELS = (32, 48, 64, 96, 128)
HIDDEN = (256, 128)
#: Backbone stages (0-based) whose maps are sampled per ray (stride 8 and stride 16).
SAMPLE_STAGES = (2, 3)
INPUT_WIDTH, INPUT_HEIGHT = 192, 80
K_FRAMES = 3
DT_S = 0.2
HIST_TOL_S = 0.1
#: Pixel normalisation after /255.
PIX_MEAN, PIX_STD = 0.45, 0.25


def _conv_bn(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
    )


class EyesNet(nn.Module):
    def __init__(
        self,
        k_frames: int = K_FRAMES,
        n_rays: int = 40,
        ego_dim: int = 10,
        channels: Sequence[int] = CHANNELS,
        hidden: Sequence[int] = HIDDEN,
        sample_stages: Sequence[int] = SAMPLE_STAGES,
    ) -> None:
        super().__init__()
        self.k_frames, self.n_rays, self.ego_dim = int(k_frames), int(n_rays), int(ego_dim)
        self.sample_stages = tuple(int(s) for s in sample_stages)
        stages, cin = [], 3 * self.k_frames
        for c in channels:
            stages.append(nn.Sequential(_conv_bn(cin, c, 2), _conv_bn(c, c, 1)))
            cin = c
        self.stages = nn.ModuleList(stages)
        q = sum(channels[s] for s in self.sample_stages) + channels[-1] + 3 + 1 + self.ego_dim
        q += self.k_frames - 1
        layers: list[nn.Module] = []
        for h in hidden:
            layers += [nn.Linear(q, h), nn.ReLU(inplace=True)]
            q = h
        layers.append(nn.Linear(q, 2))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        frames: torch.Tensor,
        uv: torch.Tensor,
        rays_cam: torch.Tensor,
        in_view: torch.Tensor,
        ego: torch.Tensor,
        hist_dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(free (B, R) in [0, 1], confidence logit (B, R))``."""
        x = normalise_frames(frames)
        maps = []
        for st in self.stages:
            x = st(x)
            maps.append(x)
        b, r = uv.shape[0], uv.shape[1]
        view = in_view.to(x.dtype).unsqueeze(-1)  # (B, R, 1)
        grid = uv.to(x.dtype).view(b, r, 1, 2)
        feats = []
        for s in self.sample_stages:
            f = F.grid_sample(maps[s], grid, mode="bilinear", padding_mode="zeros",
                              align_corners=True)  # fmt: skip
            feats.append(f.squeeze(-1).transpose(1, 2) * view)  # (B, R, C), zero when unseen
        g = maps[-1].mean(dim=(2, 3)).unsqueeze(1).expand(b, r, -1)
        e = ego.to(x.dtype).unsqueeze(1).expand(b, r, -1)
        h = hist_dt.to(x.dtype).reshape(b, -1).unsqueeze(1).expand(b, r, -1)
        q = torch.cat([*feats, g, rays_cam.to(x.dtype), view, e, h], dim=-1)
        out = self.head(q)
        return torch.sigmoid(out[..., 0]), out[..., 1]


def normalise_frames(frames: torch.Tensor) -> torch.Tensor:
    """uint8 (or 0-255 float) frames -> the network's normalised float input."""
    return (frames.float() / 255.0 - PIX_MEAN) / PIX_STD


def count_parameters(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))


# ----------------------------------------------------------------------------- frames


def low_res_mask(width: int, height: int, masks: Sequence[Sequence[float]]) -> np.ndarray:
    """``(height, width)`` bool, True where a low-resolution pixel's footprint touches a mask."""
    m = np.zeros((height, width), bool)
    for x0, y0, x1, y1 in masks:
        c0, r0 = int(np.floor(x0 * width)), int(np.floor(y0 * height))
        c1, r1 = int(np.ceil(x1 * width)), int(np.ceil(y1 * height))
        m[max(r0, 0) : min(r1, height), max(c0, 0) : min(c1, width)] = True
    return m


def prepare_frame(
    image_bgr: np.ndarray,
    width: int = INPUT_WIDTH,
    height: int = INPUT_HEIGHT,
    masks: Sequence[Sequence[float]] = (),
    device: Any = "cpu",
) -> torch.Tensor:
    """One BGR uint8 frame -> ``(3, height, width)`` uint8 RGB on ``device``.

    The ONE preprocessing of both the dataset cache and live inference: area resize, masked
    pixels zeroed at the low resolution, rounded back to uint8 (so a cached frame and a live frame
    are bit-comparable).
    """
    img = torch.from_numpy(np.ascontiguousarray(image_bgr)).to(device)
    x = img[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float()
    x = F.interpolate(x, size=(int(height), int(width)), mode="area")[0]
    if masks:
        m = torch.from_numpy(low_res_mask(int(width), int(height), masks)).to(x.device)
        x = x.masked_fill(m.unsqueeze(0), 0.0)
    return x.round().clamp_(0, 255).to(torch.uint8)


# ----------------------------------------------------------------------------- meta


def default_meta(
    *,
    cone: Any,
    camera: dict[str, Any],
    masks: Sequence[Sequence[float]],
    world_scale: float,
    ray_range_scale: float,
    ray_range: float,
    k_frames: int = K_FRAMES,
    dt_s: float = DT_S,
    hist_tol_s: float = HIST_TOL_S,
    width: int = INPUT_WIDTH,
    height: int = INPUT_HEIGHT,
    **extra: Any,
) -> dict[str, Any]:
    """The weights meta: everything inference needs, so it never reads labels or the contract."""
    c = np.asarray(cone, np.float32).reshape(-1, 3)
    return {
        "format": FORMAT,
        "version": ARCH_VERSION,
        "arch": {
            "channels": list(CHANNELS),
            "hidden": list(HIDDEN),
            "sample_stages": list(SAMPLE_STAGES),
        },  # fmt: skip
        "input": {
            "width": int(width),
            "height": int(height),
            "k_frames": int(k_frames),
            "dt_s": float(dt_s),
            "hist_tol_s": float(hist_tol_s),
            "hist_dt_input": True,
        },  # fmt: skip
        "n_rays": int(len(c)),
        "ego_dim": 10,
        "cone": [[float(v) for v in row] for row in c],
        "camera": dict(camera),
        "masks": [[float(v) for v in m] for m in masks],
        "world_scale": float(world_scale),
        "ray_range_scale": float(ray_range_scale),
        "ray_range": float(ray_range),
        **extra,
    }


def build_from_meta(meta: dict[str, Any]) -> EyesNet:
    if meta.get("format") != FORMAT:
        raise ValueError(f"not an eyes weights meta (format {meta.get('format')!r})")
    if int(meta.get("version", -1)) != ARCH_VERSION:
        raise ValueError(f"unsupported eyes arch version {meta.get('version')!r}")
    arch = meta.get("arch") or {}
    return EyesNet(
        k_frames=int(meta["input"]["k_frames"]),
        n_rays=int(meta["n_rays"]),
        ego_dim=int(meta.get("ego_dim", 10)),
        channels=tuple(arch.get("channels", CHANNELS)),
        hidden=tuple(arch.get("hidden", HIDDEN)),
        sample_stages=tuple(arch.get("sample_stages", SAMPLE_STAGES)),
    )
