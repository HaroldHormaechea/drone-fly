"""Gate keypoint network (CenterNet-style), its decode step, targets and loss. Pure torch.

Architecture::

    image (B, 3, H, W) -> backbone (strides 8 / 16 / 32) -> light FPN -> P3 (stride 8)
        -> heads: heatmap (C)          per-class gate-centre heatmap (logits)
                  center_offset (2)    sub-cell centre offset
                  kp_offset (2K)       keypoint offsets from the centre, in cells
                  kp_vis (K)           keypoint visibility (logits)

``C`` = gate classes, ``K`` = the largest keypoint count of any class (a class with fewer
keypoints uses the first ``K_c`` slots). Backbones:

* ``resnet18-hf`` — ``transformers`` ``ResNetModel`` (``microsoft/resnet-18``, ImageNet
  pretrained, pure torch; one-time download at training). The backbone config is stored in the
  weights file, so loading trained weights never downloads anything.
* ``tiny-cnn`` — a small pure-torch CNN for CI and as an offline fallback.

Preprocessing (:func:`prepare_input`) is shared by training and runtime: BGR uint8 -> RGB float,
``torch.nn.functional.interpolate`` (bilinear, antialiased) to ``input_width`` (aspect kept),
ImageNet normalisation, bottom padding to a multiple of 32. No OpenCV.

Importing this module imports no torch: functions import it on call, and the ``nn.Module``
classes (:class:`GateKeypointNet` and its backbones) are defined on first attribute access.
"""

from __future__ import annotations

import math
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

from drone_fly.harness.errors import lazy_import

if TYPE_CHECKING:  # annotations only
    import torch

STRIDE = 8
BACKBONES = ("resnet18-hf", "tiny-cnn")
PRETRAINED_ID = "microsoft/resnet-18"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODULE_CLASSES = ("TinyBackbone", "HfResNetBackbone", "GateKeypointNet")


def _tf() -> tuple[Any, Any]:
    torch = lazy_import("torch", purpose="the gate keypoint network")
    return torch, torch.nn.functional


# ----------------------------------------------------------------------------- input


def input_geometry(width: int, height: int, input_width: int) -> tuple[int, int, int, float]:
    """``(in_w, in_h, padded_h, scale)`` for a ``width x height`` frame (``in_w`` % 32 == 0)."""
    in_w = max(32, int(round(input_width / 32.0)) * 32)
    scale = in_w / float(width)
    in_h = max(1, int(round(height * scale)))
    padded = int(math.ceil(in_h / 32.0)) * 32
    return in_w, in_h, padded, scale


def resize_images(images: torch.Tensor, in_w: int, in_h: int) -> torch.Tensor:
    """``(B, 3, H, W)`` uint8/float RGB -> float ``(B, 3, in_h, in_w)`` in [0, 255]."""
    torch, F = _tf()
    x = images.float()
    if x.shape[-2:] != (in_h, in_w):
        x = F.interpolate(
            x, size=(in_h, in_w), mode="bilinear", align_corners=False, antialias=True
        )
    return x


def normalize_pad(x: torch.Tensor, padded_h: int) -> torch.Tensor:
    """[0, 255] RGB -> ImageNet-normalised, zero-padded at the bottom to ``padded_h``."""
    torch, F = _tf()
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x / 255.0 - mean) / std
    if x.shape[-2] < padded_h:
        x = F.pad(x, (0, 0, 0, padded_h - x.shape[-2]))
    return x


def bgr_to_tensor(image: np.ndarray, device: Any) -> torch.Tensor:
    """``H x W x 3`` BGR uint8 -> ``(1, 3, H, W)`` RGB uint8 tensor on ``device``."""
    torch, F = _tf()
    t = torch.from_numpy(np.ascontiguousarray(image)).to(device)
    return t.permute(2, 0, 1).flip(0).unsqueeze(0)


def prepare_input(image: np.ndarray, input_width: int, device: Any) -> tuple[torch.Tensor, float]:
    """One BGR frame -> network input ``(1, 3, padded_h, in_w)`` and the resize scale."""
    h, w = image.shape[:2]
    in_w, in_h, padded, scale = input_geometry(w, h, input_width)
    x = resize_images(bgr_to_tensor(image, device), in_w, in_h)
    return normalize_pad(x, padded), scale


# ----------------------------------------------------------------------------- backbones


@cache
def _modules() -> dict[str, Any]:
    """Define the ``nn.Module`` classes (imports torch); cached."""
    torch, F = _tf()
    nn = torch.nn

    def _conv_bn(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class TinyBackbone(nn.Module):
        """Five stride-2 conv stages; returns the stride 8 / 16 / 32 maps."""

        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(_conv_bn(3, 16, 2), _conv_bn(16, 32, 2))
            self.s8 = nn.Sequential(_conv_bn(32, 64, 2), _conv_bn(64, 64))
            self.s16 = nn.Sequential(_conv_bn(64, 128, 2), _conv_bn(128, 128))
            self.s32 = nn.Sequential(_conv_bn(128, 192, 2), _conv_bn(192, 192))
            self.channels = (64, 128, 192)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            c3 = self.s8(self.stem(x))
            c4 = self.s16(c3)
            return c3, c4, self.s32(c4)

    class HfResNetBackbone(nn.Module):
        """``transformers`` ``ResNetModel``; returns its last three stages (strides 8 / 16 / 32)."""

        def __init__(self, config: dict[str, Any] | None = None, pretrained: bool = False) -> None:
            super().__init__()
            from drone_fly.harness.errors import HarnessDependencyError, lazy_import

            tf = lazy_import("transformers", purpose="the resnet18-hf gate-net backbone")
            if pretrained:
                try:
                    model = tf.ResNetModel.from_pretrained(PRETRAINED_ID)
                except Exception as exc:  # noqa: BLE001 - network / hub errors vary
                    if isinstance(exc, HarnessDependencyError):
                        raise
                    raise RuntimeError(
                        f"could not download the pretrained backbone {PRETRAINED_ID!r} ({exc}). "
                        "Check the network / Hugging Face access, or train with "
                        "--backbone tiny-cnn (no download, lower accuracy)."
                    ) from exc
            else:
                if config is None:
                    raise ValueError("an untrained resnet18-hf backbone needs its stored config")
                model = tf.ResNetModel(tf.ResNetConfig.from_dict(dict(config)))
            self.model = model
            sizes = list(model.config.hidden_sizes)
            self.channels = (sizes[-3], sizes[-2], sizes[-1])

        @property
        def config_dict(self) -> dict[str, Any]:
            return self.model.config.to_dict()

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hs = self.model(pixel_values=x, output_hidden_states=True).hidden_states
            return hs[-3], hs[-2], hs[-1]

    # ----------------------------------------------------------------------------- network

    def _head(cin: int, cout: int, bias: float = 0.0) -> nn.Sequential:
        head = nn.Sequential(
            nn.Conv2d(cin, cin, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(cin, cout, 1)
        )
        nn.init.constant_(head[-1].bias, bias)
        return head

    class GateKeypointNet(nn.Module):
        def __init__(
            self,
            num_classes: int,
            num_keypoints: int,
            backbone: str = "tiny-cnn",
            *,
            fpn_channels: int = 96,
            backbone_config: dict[str, Any] | None = None,
            pretrained: bool = False,
        ) -> None:
            super().__init__()
            if backbone not in BACKBONES:
                raise ValueError(
                    f"unknown gate-net backbone {backbone!r}; expected one of {BACKBONES}"
                )
            self.num_classes = int(num_classes)
            self.num_keypoints = int(num_keypoints)
            self.backbone_name = backbone
            self.backbone: nn.Module = (
                TinyBackbone()
                if backbone == "tiny-cnn"
                else HfResNetBackbone(backbone_config, pretrained=pretrained)
            )
            c3, c4, c5 = self.backbone.channels
            f = fpn_channels
            self.lat3, self.lat4, self.lat5 = (
                nn.Conv2d(c3, f, 1),
                nn.Conv2d(c4, f, 1),
                nn.Conv2d(c5, f, 1),
            )
            self.smooth = _conv_bn(f, f)
            self.heatmap = _head(f, self.num_classes, bias=-2.19)  # prior p ~= 0.1
            self.center_offset = _head(f, 2)
            self.kp_offset = _head(f, 2 * self.num_keypoints)
            self.kp_vis = _head(f, self.num_keypoints)

        @property
        def backbone_config(self) -> dict[str, Any] | None:
            return getattr(self.backbone, "config_dict", None)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            c3, c4, c5 = self.backbone(x)
            p5 = self.lat5(c5)
            p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
            p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
            p3 = self.smooth(p3)
            return {
                "heatmap": self.heatmap(p3),
                "center_offset": self.center_offset(p3),
                "kp_offset": self.kp_offset(p3),
                "kp_vis": self.kp_vis(p3),
            }

    return {
        "TinyBackbone": TinyBackbone,
        "HfResNetBackbone": HfResNetBackbone,
        "GateKeypointNet": GateKeypointNet,
    }


def __getattr__(name: str) -> Any:
    if name in MODULE_CLASSES:
        return _modules()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ----------------------------------------------------------------------------- decode


def decode(
    out: dict[str, torch.Tensor],
    *,
    score_threshold: float = 0.3,
    max_detections: int = 8,
    stride: int = STRIDE,
) -> list[list[dict[str, Any]]]:
    """Heatmap peaks (3x3 max-pool NMS) -> per image a list of detections in input pixels.

    Each detection: ``{"cls": index, "score", "center": (2,), "keypoints": (K, 2),
    "visibility": (K,)}`` (NumPy, float64).
    """
    torch, F = _tf()
    out = {k: v.detach() for k, v in out.items()}
    heat = torch.sigmoid(out["heatmap"].float())
    b, c, h, w = heat.shape
    peaks = heat * (F.max_pool2d(heat, 3, 1, 1) == heat)
    k = min(max_detections, c * h * w)
    scores, idx = peaks.view(b, -1).topk(k)
    cls = idx // (h * w)
    cell = idx % (h * w)
    ys, xs = (cell // w), (cell % w)
    off = out["center_offset"].float()
    kpo = out["kp_offset"].float()
    vis = torch.sigmoid(out["kp_vis"].float())
    n_kp = vis.shape[1]
    results: list[list[dict[str, Any]]] = []
    for i in range(b):
        dets = []
        for j in range(k):
            s = float(scores[i, j])
            if s < score_threshold:
                break
            y, x = int(ys[i, j]), int(xs[i, j])
            center = np.array(
                [(x + float(off[i, 0, y, x])) * stride, (y + float(off[i, 1, y, x])) * stride]
            )
            kp = kpo[i, :, y, x].view(n_kp, 2).cpu().numpy().astype(np.float64) * stride + center
            dets.append(
                {
                    "cls": int(cls[i, j]),
                    "score": s,
                    "center": center,
                    "keypoints": kp,
                    "visibility": vis[i, :, y, x].cpu().numpy().astype(np.float64),
                }
            )
        results.append(dets)
    return results


# ----------------------------------------------------------------------------- targets + loss


def gaussian_radius(height: float, width: float, min_overlap: float = 0.7) -> float:
    """CenterNet's Gaussian radius for a ``height x width`` box (cells)."""
    a1, b1 = 1.0, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1**2 - 4 * a1 * c1)) / 2
    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(b2**2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(b3**2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def _draw_gaussian(heat: np.ndarray, cx: int, cy: int, radius: int) -> None:
    sigma = (2 * radius + 1) / 6.0
    h, w = heat.shape
    ys, xs = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    g = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
    left, right = min(cx, radius), min(w - cx, radius + 1)
    top, bottom = min(cy, radius), min(h - cy, radius + 1)
    region = heat[cy - top : cy + bottom, cx - left : cx + right]
    patch = g[radius - top : radius + bottom, radius - left : radius + right]
    np.maximum(region, patch, out=region)


def build_targets(
    gates: list[dict[str, Any]],
    out_h: int,
    out_w: int,
    num_classes: int,
    num_keypoints: int,
    *,
    stride: int = STRIDE,
    max_objects: int = 16,
) -> dict[str, np.ndarray]:
    """Training targets for one image.

    ``gates``: ``{"cls": index, "center": (2,) input px, "keypoints": (K_c, 2) input px,
    "visible": (K_c,) bool}``. Centres outside the output grid are skipped.
    """
    k = num_keypoints
    t = {
        "heatmap": np.zeros((num_classes, out_h, out_w), np.float32),
        "ind": np.zeros(max_objects, np.int64),
        "mask": np.zeros(max_objects, np.float32),
        "center_offset": np.zeros((max_objects, 2), np.float32),
        "kp_offset": np.zeros((max_objects, 2 * k), np.float32),
        "kp_offset_mask": np.zeros((max_objects, k), np.float32),
        "kp_vis": np.zeros((max_objects, k), np.float32),
        "kp_vis_mask": np.zeros((max_objects, k), np.float32),
    }
    n = 0
    for g in gates:
        if n >= max_objects:
            break
        c = np.asarray(g["center"], np.float64) / stride
        cx, cy = int(np.floor(c[0])), int(np.floor(c[1]))
        if not (0 <= cx < out_w and 0 <= cy < out_h):
            continue
        kp = np.asarray(g["keypoints"], np.float64).reshape(-1, 2) / stride
        vis = np.asarray(g["visible"], bool).reshape(-1)
        lo = np.clip(kp.min(axis=0), [0, 0], [out_w, out_h])
        hi = np.clip(kp.max(axis=0), [0, 0], [out_w, out_h])
        bw, bh = max(float(hi[0] - lo[0]), 1.0), max(float(hi[1] - lo[1]), 1.0)
        radius = max(0, int(gaussian_radius(bh, bw)))
        _draw_gaussian(t["heatmap"][int(g["cls"])], cx, cy, radius)
        t["ind"][n] = cy * out_w + cx
        t["mask"][n] = 1.0
        t["center_offset"][n] = c - [cx, cy]
        kc = len(kp)
        t["kp_offset"][n, : 2 * kc] = (kp - c).reshape(-1)
        t["kp_offset_mask"][n, :kc] = vis.astype(np.float32)
        t["kp_vis"][n, :kc] = vis.astype(np.float32)
        t["kp_vis_mask"][n, :kc] = 1.0
        n += 1
    return t


def _gather(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """``(B, D, H, W)`` at flat indices ``(B, M)`` -> ``(B, M, D)``."""
    torch, F = _tf()
    b, d = feat.shape[:2]
    flat = feat.view(b, d, -1).permute(0, 2, 1)
    return flat.gather(1, ind.unsqueeze(-1).expand(-1, -1, d))


def focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss on Gaussian heatmaps."""
    torch, F = _tf()
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1.0 - pos
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p**2 * (1 - target) ** 4 * neg
    n_pos = pos.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / torch.clamp(n_pos, min=1.0)


def gate_net_loss(
    out: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: tuple[float, float, float, float] = (1.0, 1.0, 0.1, 0.5),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Focal heatmap + masked L1 (centre / keypoint offsets) + masked BCE (visibility)."""
    torch, F = _tf()
    w_hm, w_off, w_kp, w_vis = weights
    hm = focal_loss(out["heatmap"], targets["heatmap"])
    ind, mask = targets["ind"], targets["mask"]
    n_obj = torch.clamp(mask.sum(), min=1.0)
    off = _gather(out["center_offset"].float(), ind)
    off_l = (torch.abs(off - targets["center_offset"]).sum(-1) * mask).sum() / n_obj
    kpo = _gather(out["kp_offset"].float(), ind)
    b, m, _ = kpo.shape
    kp_err = torch.abs(kpo - targets["kp_offset"]).view(b, m, -1, 2).sum(-1)
    kp_mask = targets["kp_offset_mask"] * mask.unsqueeze(-1)
    kp_l = (kp_err * kp_mask).sum() / torch.clamp(kp_mask.sum(), min=1.0)
    vis = _gather(out["kp_vis"].float(), ind)
    vis_mask = targets["kp_vis_mask"] * mask.unsqueeze(-1)
    bce = F.binary_cross_entropy_with_logits(vis, targets["kp_vis"], reduction="none")
    vis_l = (bce * vis_mask).sum() / torch.clamp(vis_mask.sum(), min=1.0)
    total = w_hm * hm + w_off * off_l + w_kp * kp_l + w_vis * vis_l
    parts = {
        "heatmap": float(hm.detach()),
        "center_offset": float(off_l.detach()),
        "kp_offset": float(kp_l.detach()),
        "kp_vis": float(vis_l.detach()),
    }
    return total, parts


# ----------------------------------------------------------------------------- weights file


def build_from_meta(meta: dict[str, Any]) -> Any:
    """An untrained network with the architecture recorded in a weights file's metadata."""
    return _modules()["GateKeypointNet"](
        len(meta["classes"]),
        int(meta["num_keypoints"]),
        meta["backbone"],
        fpn_channels=int(meta.get("fpn_channels", 96)),
        backbone_config=meta.get("backbone_config"),
        pretrained=False,
    )
