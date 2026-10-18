"""UC-85: deployable frozen monocular-depth backbone for the eyes v3 estimator.

A small wrapper + factory around a pretrained monocular-depth model that turns a batch of RGB
frames into single-channel predicted depth maps. :class:`~drone_fly.eyes.model.EyesDepthNet` holds
one of these as a **plain (non-``nn.Module``) attribute**, so the frozen backbone is auto-excluded
from ``net.parameters()`` / the optimizer / ``MAX_PARAMS`` / ``state_dict()`` — checkpoints carry
only the learnable adapter + head, and strict ``load_state_dict`` round-trips without backbone keys.

Backbones (selected by ``backbone_id``):

* the default ``depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`` (DINOv2 ViT-S, ~24.8 M
  params / ~99 MB, Apache-2.0) via ``transformers`` — imported **lazily** (``transformers`` /
  ``pillow`` live only in the ``harness`` extra, never in the default/dev deps, so CI stays
  network-free), weights fetched once to the Hugging Face cache and reused;
* a pure-torch ``tiny-depth`` sentinel stub (no ``transformers``, no network, no download) that
  doubles as the **offline fallback** and the **hermetic CI/test backbone** (AC8).

Offline / already-cached runs (AC2): once the weights are in the HF cache, set ``HF_HUB_OFFLINE=1``
(and/or ``TRANSFORMERS_OFFLINE=1``) so ``from_pretrained`` reuses the cache without any network
access. :func:`provision_backbone` warms the cache once (``drone-fly eyes provision-backbone``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from drone_fly.harness.errors import HarnessDependencyError, lazy_import

if TYPE_CHECKING:  # annotations only
    import torch

#: Default pretrained depth backbone: Depth-Anything-V2 (metric, small), the one already used by
#: ``drone_fly.harness.liftoff.depth``. ~99 MB, Apache-2.0 — small enough for the ~17 GB disk.
DEFAULT_BACKBONE_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
#: Sentinel backbone id: a pure-torch stub (no transformers / network). Offline + hermetic tests.
TINY_DEPTH_ID = "tiny-depth"
#: Accepted ``arch.backbone_input`` adapter modes (only the predicted depth map today).
BACKBONE_INPUTS = ("depth-map",)
#: ViT patch size of the DINOv2 trunk: the model needs its input sized to a multiple of this.
_PATCH = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _imagenet_normalize(frames: torch.Tensor) -> torch.Tensor:
    """``(N, 3, H, W)`` uint8/float RGB in [0, 255] -> ImageNet-normalised float."""
    import torch

    x = frames.float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class DepthBackbone:
    """A frozen depth model: ``predict((N, 3, H, W)) -> (N, 1, h, w)`` depth; movable via ``to``.

    Deliberately **not** an ``nn.Module`` so an :class:`EyesDepthNet` that holds one does not
    register it as a submodule (the whole point of the non-registration guarantee).
    """

    backbone_id: str
    device: str

    def to(self, device: Any) -> DepthBackbone:  # pragma: no cover - trivial
        raise NotImplementedError

    def predict(self, frames: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        raise NotImplementedError


def _tiny_depth_module() -> Any:
    """A small, deterministic, frozen pure-torch depth net (stride-2, 1-channel output)."""
    import torch
    from torch import nn

    module = nn.Sequential(
        nn.Conv2d(3, 8, 3, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 8, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 1, 3, 1, 1),
    )
    # Deterministic weights from a LOCAL generator: identical every build and, crucially, it does
    # not touch the global RNG, so an EyesDepthNet's trainable adapter/head init (and training
    # resume) stay bit-reproducible regardless of the stub.
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.empty_like(p).uniform_(-0.1, 0.1, generator=g))
    return module


class TinyDepthBackbone(DepthBackbone):
    """Pure-torch depth stub: no transformers, no network, no download (offline + CI)."""

    backbone_id = TINY_DEPTH_ID

    def __init__(self, device: str = "cpu") -> None:
        self.device = str(device)
        self._module = _tiny_depth_module().to(self.device).eval()
        for p in self._module.parameters():
            p.requires_grad_(False)

    def to(self, device: Any) -> TinyDepthBackbone:
        self.device = str(device)
        self._module.to(self.device)
        return self

    def predict(self, frames: torch.Tensor) -> torch.Tensor:
        import torch

        x = _imagenet_normalize(frames.to(self.device))
        with torch.no_grad():
            return self._module(x).float()


class HfDepthBackbone(DepthBackbone):
    """A ``transformers`` ``AutoModelForDepthEstimation`` (lazy import, frozen, no grad)."""

    def __init__(self, backbone_id: str = DEFAULT_BACKBONE_ID, device: str = "cpu") -> None:
        tf = lazy_import("transformers", purpose="the eyes v3 depth backbone")
        self.backbone_id = str(backbone_id)
        self.device = str(device)
        self._model = tf.AutoModelForDepthEstimation.from_pretrained(self.backbone_id)
        self._model.to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    def to(self, device: Any) -> HfDepthBackbone:
        self.device = str(device)
        self._model.to(self.device)
        return self

    def predict(self, frames: torch.Tensor) -> torch.Tensor:
        import torch
        import torch.nn.functional as F

        x = _imagenet_normalize(frames.to(self.device))
        h, w = x.shape[-2:]
        th = max(_PATCH, int(round(h / _PATCH)) * _PATCH)
        tw = max(_PATCH, int(round(w / _PATCH)) * _PATCH)
        if (th, tw) != (h, w):
            x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
        with torch.no_grad():
            out = self._model(pixel_values=x).predicted_depth  # (N, h, w)
        if out.dim() == 3:
            out = out.unsqueeze(1)
        return out.float()


def _offline() -> bool:
    """True when the HF hub is pinned offline (so a missing model should not try to download)."""
    return any(
        os.environ.get(k, "").strip().lower() in ("1", "true", "yes")
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def make_depth_backbone(
    backbone_id: str | None = None,
    device: str = "cpu",
    *,
    fallback_to_tiny: bool = False,
) -> DepthBackbone:
    """Build the frozen depth backbone named by ``backbone_id`` (default DINOv2 ViT-S).

    ``tiny-depth`` -> the pure-torch stub. Any other id -> the ``transformers`` backbone (lazy).
    ``fallback_to_tiny`` falls back to the stub when ``transformers`` is unavailable (CI / offline
    without a cache); it is OFF by default because a model trained against the real backbone must
    not silently infer against the stub.
    """
    bid = backbone_id or DEFAULT_BACKBONE_ID
    if bid == TINY_DEPTH_ID:
        return TinyDepthBackbone(device)
    try:
        return HfDepthBackbone(bid, device)
    except HarnessDependencyError:
        if fallback_to_tiny:
            return TinyDepthBackbone(device)
        raise


def provision_backbone(model_id: str = DEFAULT_BACKBONE_ID) -> dict[str, Any]:
    """AC2: warm the Hugging Face cache for ``model_id`` once; a later run reuses it (offline-able).

    The ``tiny-depth`` stub needs no download. For a real backbone this lazily imports
    ``transformers`` (+ ``pillow``) and runs ``from_pretrained`` so the weights land in the HF
    cache; subsequent runs (even with ``HF_HUB_OFFLINE=1``) load from there.
    """
    if model_id == TINY_DEPTH_ID:
        return {
            "backbone_id": model_id,
            "cached": True,
            "download": False,
            "note": "pure-torch stub; nothing to download",
        }
    tf = lazy_import("transformers", purpose="provisioning the eyes v3 depth backbone")
    lazy_import("PIL.Image", package="pillow", purpose="provisioning the eyes v3 depth backbone")
    model = tf.AutoModelForDepthEstimation.from_pretrained(model_id)
    try:
        tf.AutoImageProcessor.from_pretrained(model_id)
    except Exception:  # noqa: BLE001 - some depth models ship no image processor; weights suffice
        pass
    n_params = int(sum(p.numel() for p in model.parameters()))
    return {
        "backbone_id": model_id,
        "cached": True,
        "download": True,
        "params": n_params,
        "offline": _offline(),
    }
