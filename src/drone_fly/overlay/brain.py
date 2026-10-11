"""The brain panel: the visualizer's K1 activation heatmap, rendered live with NumPy (CPU only).

Same method as ``viz/viewer.js`` (UC-12 / UC-28 / UC-60), reimplemented — "as in the visualizer"
means the same look, not the same code:

* neuron positions are the recordings' (``drone_fly.record.coordinates.resolve_positions``:
  ``display3d``, projected on the view plane; default view ``top`` = ``xz``);
* the registered brain outline (``viz/brain_outline.js``) fixes a uniform, aspect-preserving
  voxel -> panel transform with the viewer's ``H - y`` flip;
* UC-60: neurons whose projected position lies **outside** the outline are not splatted; they
  render in tagged side boxes (vision / proprioceptive / hunger / other) as mini heatmaps, with the
  viewer's bucket order and titles. Non-anatomical positions or a missing outline degrade to
  auto-fit splats over every neuron, no outline, no boxes;
* per active neuron an additive Gaussian splat (radius ``MAP_KERNEL_R`` = 7 buffer pixels,
  sigma = R / 2.4) is stamped into a half-resolution buffer (``MAP_SS`` = 2), weighted by the
  neuron's activation **magnitude relative to rest** (``|a|``: a resting neuron is dark), and the
  buffer goes through the viewer's "hot" colormap with its soft alpha ramp, upscaled 2x.

The splat is a ``bincount`` of the weights at the neurons' buffer pixels followed by two
``scipy.ndimage.convolve1d`` passes: the 2-D kernel is ``exp(-(dx²+dy²)/2σ²) = g(dx)·g(dy)``, so
the separable convolution equals the viewer's per-neuron additive stamps exactly (neurons whose
centre falls outside the buffer are dropped rather than partially stamped).

**Live-gain deviation (documented, intentional).** The viewer stretches each neuron to its own
min -> max over the *whole recorded episode* and normalises against the episode's global peak —
both unavailable live. Here each neuron keeps a **running** ``lo`` / ``hi`` that relaxes toward
its current magnitude with ``a = 1 - exp(-dt / gain_window_s)`` (a decaying min / max window,
default ~5 s), and the colormap is normalised against a decaying running peak of the splat buffer.
A neuron's weight is ``clamp01((|a| - lo) / max(hi - lo, min_range))`` and weights below 2/255 are
skipped, as in the viewer. ``min_range`` is the viewer's 0.05 floor expressed in quantisation
steps of the frame (``0.05 = 6.375 codes x 2/255``): identical to the viewer for the recorder's
fixed scheme, and proportionally smaller for ``overlay.brain_scale: frame_peak`` frames, whose
codes are finer — otherwise that floor would hide every real (sub-0.05) K1 activation. The first
frame after start is dark (``lo = hi``); the picture is similar to, not identical with, an offline
recording of the same flight.

CPU / NumPy only: the asset loader reads the connectome and positions from disk and never touches
CUDA (no device query, no tensor on a GPU) — the GPU belongs to the game, the depth model and the
policy.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

PROJECTIONS = {"xz": (0, 2), "xy": (0, 1), "yz": (1, 2)}
VIEW_PLANES = {"front": "xy", "side": "yz", "top": "xz"}
MAP_SS = 2
MAP_KERNEL_R = 7
MAP_KERNEL_SIGMA = MAP_KERNEL_R / 2.4
MAP_MIN_WEIGHT = 2 / 255
MAP_GAIN_MIN_RANGE = 0.05
#: The recorder's quantisation step: ``MAP_GAIN_MIN_RANGE`` is expressed relative to it.
RECORDER_SCALE = 2.0 / 255.0
#: Running-peak floor so an all-dark frame does not divide by zero.
MIN_PEAK = 1e-9

#: UC-60 box buckets: (key, title), in the viewer's fixed order; anything else -> "other".
BOX_BUCKETS = (
    ("vision", "vision (external)"),
    ("proprioceptive", "proprioceptive"),
    ("hunger", "hunger"),
    ("other", "other (untagged)"),
)
#: Fraction of the panel height given to the box strip (when there are boxes).
BOX_STRIP_FRAC = 0.22
MIN_BOX_STRIP_PX = 12


# ----------------------------------------------------------------------------- asset


@dataclass
class BrainAsset:
    """Static per-connectome data the renderer needs (positions, tags, outline)."""

    n: int
    #: ``(N, 2)`` projected voxel-space positions on the view plane (NaN = none on this plane).
    points: np.ndarray
    anatomical: bool
    modality: list[str] = field(default_factory=list)
    #: ``(M, 2)`` outline polygon for the view (voxel space), or ``None``.
    outline_polygon: np.ndarray | None = None
    #: ``(u_min, u_max, v_min, v_max)`` of the outline bbox on the view plane, or ``None``.
    outline_bbox: tuple[float, float, float, float] | None = None
    source: str = ""
    view: str = "top"


def parse_outline_js(text: str) -> dict[str, Any]:
    """The ``BRAIN_OUTLINE`` object from ``viz/brain_outline.js``."""
    m = re.search(r"const\s+BRAIN_OUTLINE\s*=\s*(\{.*\})\s*;", text, re.S)
    if m is None:
        raise ValueError("brain_outline.js does not define BRAIN_OUTLINE")
    return json.loads(m.group(1))


def outline_for_view(
    outline: dict[str, Any] | None, view: str
) -> tuple[np.ndarray | None, tuple[float, float, float, float] | None]:
    """``(polygon, bbox)`` of the registered outline on ``view``'s plane (``None`` if absent)."""
    if not outline:
        return None, None
    a0, a1 = PROJECTIONS[VIEW_PLANES.get(view, "xz")]
    mn, mx = outline["bbox3d"]["min"], outline["bbox3d"]["max"]
    bbox = (float(mn[a0]), float(mx[a0]), float(mn[a1]), float(mx[a1]))
    plane = (outline.get("planes") or {}).get(view) or {}
    poly = plane.get("polygon")
    polygon = None if not poly else np.asarray(poly, np.float64).reshape(-1, 2)
    return polygon, bbox


def project_positions(positions: dict[str, Any], view: str = "top") -> np.ndarray:
    """``(N, 2)`` view-plane points from a positions dict (``display3d`` first, as the viewer)."""
    a0, a1 = PROJECTIONS[VIEW_PLANES.get(view, "xz")]
    is_default = VIEW_PLANES.get(view, "xz") == positions.get("projection", "xz")
    disp = positions.get("display3d")
    c3 = positions.get("coords3d") or []
    c2 = positions.get("coords2d") or []
    n = len(positions.get("has_position") or disp or c3 or c2)
    out = np.full((n, 2), np.nan)
    for i in range(n):
        d = disp[i] if disp is not None and i < len(disp) else None
        if d is not None:
            out[i] = (d[a0], d[a1])
        elif i < len(c3) and c3[i] is not None:
            out[i] = (c3[i][a0], c3[i][a1])
        elif is_default and i < len(c2) and c2[i] is not None:
            out[i] = c2[i]
    return out


def load_brain_asset(
    k1_path: str | Path, view: str = "top", *, outline_path: str | Path | None = None
) -> BrainAsset:
    """Load the K1 (or any connectome) brain asset from disk — CPU / NumPy only, no CUDA.

    Positions come from ``resolve_positions(persist=False)`` (the recordings' sidecar; never
    written here), modality tags from the recorder's ``RECORDED_MODALITIES`` via
    ``select_modality`` (fail-soft: an absent population is simply untagged), the outline from
    ``viz/brain_outline.js`` under the repo root (absent -> degraded auto-fit rendering).
    """
    if view not in VIEW_PLANES:
        raise ValueError(f"view must be one of {sorted(VIEW_PLANES)}, got {view!r}")
    from drone_fly.connectome.loader import load_connectome
    from drone_fly.controller.modality import ModalityError, select_modality
    from drone_fly.record.coordinates import resolve_positions
    from drone_fly.record.recorder import RECORDED_MODALITIES

    data = load_connectome(k1_path)
    positions = resolve_positions(data, persist=False, persist_strict=False)
    if positions is None:
        raise ValueError(f"no neuron positions could be resolved for {k1_path}")
    n = int(data.neuron_count)
    tags = [""] * n
    for name in RECORDED_MODALITIES:
        try:
            sel = select_modality(data, name)
        except ModalityError:
            continue
        for idx in np.asarray(sel.indices).tolist():
            if tags[idx] == "":
                tags[idx] = name

    outline = None
    if outline_path is None:
        from drone_fly.harness.config import find_repo_root

        root = find_repo_root()
        outline_path = None if root is None else root / "viz" / "brain_outline.js"
    if outline_path is not None and Path(outline_path).is_file():
        try:
            outline = parse_outline_js(Path(outline_path).read_text(encoding="utf-8"))
        except (ValueError, KeyError) as exc:
            log.warning("brain outline unreadable (%s); rendering without it", exc)
    polygon, bbox = outline_for_view(outline, view)
    source = str(positions.get("source") or "")
    return BrainAsset(
        n=n,
        points=project_positions(positions, view),
        anatomical=bool(re.match(r"^anatomical", source, re.I)),
        modality=tags,
        outline_polygon=polygon,
        outline_bbox=bbox,
        source=source,
        view=view,
    )


# ----------------------------------------------------------------------------- pure maths


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point-in-polygon (mirrors ``viewer.js pointInPolygon``)."""
    p = np.asarray(points, np.float64).reshape(-1, 2)
    poly = np.asarray(polygon, np.float64).reshape(-1, 2)
    inside = np.zeros(len(p), bool)
    if len(poly) < 3:
        return inside
    x, y = p[:, 0], p[:, 1]
    j = len(poly) - 1
    with np.errstate(invalid="ignore"):
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            dy = (yj - yi) or 1e-12
            cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / dy + xi)
            inside ^= cross
            j = i
    return inside & np.isfinite(p).all(axis=1)


def partition(
    points: np.ndarray, asset: BrainAsset
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """UC-60 ``(inside, boxes)``: in-outline neurons and the relocated neurons' tagged buckets."""
    pts = np.asarray(points, np.float64)
    finite = np.isfinite(pts).all(axis=1)
    if not asset.anatomical or asset.outline_bbox is None:
        return np.ones(len(pts), bool), []  # degrade: everything stays in the map, no boxes
    if asset.outline_polygon is not None and len(asset.outline_polygon) >= 3:
        inside = points_in_polygon(pts, asset.outline_polygon)
    else:
        u0, u1, v0, v1 = asset.outline_bbox
        with np.errstate(invalid="ignore"):
            inside = (pts[:, 0] >= u0) & (pts[:, 0] <= u1) & (pts[:, 1] >= v0) & (pts[:, 1] <= v1)
        inside &= finite
    tags = np.asarray(
        [t if t in ("vision", "proprioceptive", "hunger") else "other" for t in asset.modality]
        + ["other"] * max(0, len(pts) - len(asset.modality))
    )
    boxes = []
    for key, title in BOX_BUCKETS:
        idx = np.flatnonzero(~inside & (tags[: len(pts)] == key))
        if idx.size:
            boxes.append((title, idx))
    return inside, boxes


def hot_colormap(t: np.ndarray) -> np.ndarray:
    """The viewer's hot ramp: ``t`` in [0, 1] -> ``(..., 3)`` uint8 (black-red-yellow-white)."""
    t = np.asarray(t, np.float32)
    r = np.clip(t / 0.375, 0, 1)
    g = np.clip((t - 0.375) / 0.375, 0, 1)
    b = np.clip((t - 0.75) / 0.25, 0, 1)
    return np.round(np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def gaussian_kernel_1d(radius: int = MAP_KERNEL_R, sigma: float = MAP_KERNEL_SIGMA) -> np.ndarray:
    d = np.arange(-radius, radius + 1, dtype=np.float64)
    return np.exp(-(d * d) / (2 * sigma * sigma))


def make_transform(
    width: float, height: float, pad: float, u0: float, u1: float, v0: float, v1: float
) -> tuple[float, float, float]:
    """``(s, off_x, off_y)`` of the viewer's uniform fit: ``x = off_x + (u-u0)s``,
    ``y = height - off_y - (v-v0)s``."""
    ext_u = (u1 - u0) or 1.0
    ext_v = (v1 - v0) or 1.0
    s = min((width - 2 * pad) / ext_u, (height - 2 * pad) / ext_v)
    return s, (width - ext_u * s) / 2, (height - ext_v * s) / 2


# ----------------------------------------------------------------------------- renderer


@dataclass
class BrainRender:
    """One rendered brain panel (panel-local pixels)."""

    image: np.ndarray | None  # (H, W, 4) RGBA uint8, None when nothing can be drawn
    message: str | None = None
    #: ``(M, 2)`` outline polyline in panel pixels (closed), or ``None``.
    outline: np.ndarray | None = None
    #: ``[(title, x, y, w, h)]`` box rectangles (panel pixels) for the titles / frames.
    boxes: list[tuple[str, int, int, int, int]] = field(default_factory=list)
    #: ``(K, 2)`` tap-neuron positions (panel pixels) when tap outlining is on.
    taps: np.ndarray | None = None


class BrainRenderer:
    """Precomputes positions / kernel / partition once; renders one frame per brain datagram."""

    def __init__(
        self,
        asset: BrainAsset,
        width: int,
        height: int,
        *,
        gain_window_s: float = 5.0,
        pad: float = 6.0,
        tap_index: Any = None,
    ) -> None:
        from scipy.ndimage import convolve1d

        self._convolve1d = convolve1d
        self.asset = asset
        self.width, self.height = int(width), int(height)
        if self.width < 4 or self.height < 4:
            raise ValueError(f"brain panel too small: {width}x{height}")
        self.gain_window_s = float(gain_window_s)
        self.inside, self.box_groups = partition(asset.points, asset)
        strip = 0
        if self.box_groups:
            strip = max(MIN_BOX_STRIP_PX, int(round(self.height * BOX_STRIP_FRAC)))
        self.map_h = self.height - strip
        self.strip_h = strip

        pts = asset.points
        fixed = asset.anatomical and asset.outline_bbox is not None
        if fixed:
            u0, u1, v0, v1 = asset.outline_bbox  # type: ignore[misc]
        else:
            ok = np.isfinite(pts).all(axis=1)
            if ok.any():
                u0, u1 = float(pts[ok, 0].min()), float(pts[ok, 0].max())
                v0, v1 = float(pts[ok, 1].min()), float(pts[ok, 1].max())
            else:
                u0, u1, v0, v1 = 0.0, 1.0, 0.0, 1.0
        s, ox, oy = make_transform(self.width, self.map_h, pad, u0, u1, v0, v1)
        self._tf = (s, ox, oy, u0, v0)
        xy = self.to_panel(pts)
        self.bw = max(1, math.ceil(self.width / MAP_SS))
        self.bh = max(1, math.ceil(self.map_h / MAP_SS))
        with np.errstate(invalid="ignore"):
            sx = np.round(xy[:, 0] / MAP_SS)
            sy = np.round(xy[:, 1] / MAP_SS)
            valid = (
                self.inside
                & np.isfinite(sx)
                & np.isfinite(sy)
                & (sx >= 0)
                & (sx < self.bw)
                & (sy >= 0)
                & (sy < self.bh)
            )
        self.valid = valid
        self.flat_index = np.zeros(len(pts), np.int64)
        self.flat_index[valid] = (sy[valid] * self.bw + sx[valid]).astype(np.int64)
        self.kernel = gaussian_kernel_1d()
        self.outline = None
        if fixed and asset.outline_polygon is not None and len(asset.outline_polygon) > 1:
            self.outline = self.to_panel(asset.outline_polygon)
        self.taps = None
        if tap_index is not None:
            ti = np.asarray(tap_index, np.int64).reshape(-1)
            ti = ti[(ti >= 0) & (ti < len(pts))]
            ti = ti[valid[ti]]
            self.taps = xy[ti]
        self.box_rects = self._layout_boxes()
        self._lo: np.ndarray | None = None
        self._hi: np.ndarray | None = None
        self._peak = 0.0
        self._last_t: float | None = None

    # -- geometry --

    def to_panel(self, points: np.ndarray) -> np.ndarray:
        """Voxel-space view-plane points -> panel pixels (the viewer's transform, y flipped)."""
        s, ox, oy, u0, v0 = self._tf
        p = np.asarray(points, np.float64).reshape(-1, 2)
        return np.stack([ox + (p[:, 0] - u0) * s, self.map_h - oy - (p[:, 1] - v0) * s], axis=1)

    def screen_position(self, i: int) -> tuple[float, float]:
        """Panel pixel of neuron ``i`` (its splat centre, after the half-resolution rounding)."""
        idx = int(self.flat_index[i])
        return (idx % self.bw) * MAP_SS, (idx // self.bw) * MAP_SS

    def _layout_boxes(self) -> list[tuple[str, int, int, int, int]]:
        if not self.box_groups:
            return []
        k = len(self.box_groups)
        gap = 2
        w = max(1, (self.width - gap * (k - 1)) // k)
        return [
            (title, i * (w + gap), self.map_h, w, self.strip_h)
            for i, (title, _) in enumerate(self.box_groups)
        ]

    # -- per frame --

    def reset_gain(self) -> None:
        self._lo = self._hi = None
        self._peak = 0.0
        self._last_t = None

    def weights(self, values: np.ndarray, now: float, scale: float = RECORDER_SCALE) -> np.ndarray:
        """Per-neuron weights in [0, 1] from the running gain (updates the gain state)."""
        mag = np.abs(np.asarray(values, np.float32).reshape(-1))
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0)
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now
        if self._lo is None or self._hi is None:
            self._lo, self._hi = mag.copy(), mag.copy()
        else:
            a = np.float32(1.0 - math.exp(-dt / self.gain_window_s))
            self._lo = np.minimum(mag, self._lo + a * (mag - self._lo))
            self._hi = np.maximum(mag, self._hi + a * (mag - self._hi))
        min_range = np.float32(MAP_GAIN_MIN_RANGE * (float(scale) / RECORDER_SCALE))
        return np.clip((mag - self._lo) / np.maximum(self._hi - self._lo, min_range), 0, 1)

    def render(self, frame: Any, now: float) -> BrainRender:
        """Render a :class:`~drone_fly.overlay.protocol.BrainFrame` (or a raw activation vector)."""
        if hasattr(frame, "values") and hasattr(frame, "scale"):
            values, scale = frame.values(), float(frame.scale)
        else:
            values, scale = np.asarray(frame, np.float32).reshape(-1), RECORDER_SCALE
        if values.shape[0] != self.asset.n:
            return BrainRender(
                None,
                f"neuron count mismatch: frame has {values.shape[0]}, brain asset has "
                f"{self.asset.n} (overlay and harness use different K1 paths?)",
            )
        dt = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        w = self.weights(values, now, scale)
        active = self.valid & (w >= MAP_MIN_WEIGHT)
        buf = np.bincount(
            self.flat_index[active], weights=w[active], minlength=self.bw * self.bh
        ).reshape(self.bh, self.bw)
        buf = self._convolve1d(buf, self.kernel, axis=0, mode="constant", cval=0.0)
        buf = self._convolve1d(buf, self.kernel, axis=1, mode="constant", cval=0.0)
        frame_peak = float(buf.max()) if buf.size else 0.0
        decay = math.exp(-dt / self.gain_window_s)
        self._peak = max(frame_peak, self._peak * decay, MIN_PEAK)
        t = np.clip(buf / self._peak, 0, 1).astype(np.float32)
        img = np.zeros((self.height, self.width, 4), np.uint8)
        small = np.zeros((self.bh, self.bw, 4), np.uint8)
        small[..., :3] = hot_colormap(t)
        small[..., 3] = np.round(255 * np.clip(t * 1.25, 0, 1)).astype(np.uint8)
        small[t <= 0] = 0
        big = small.repeat(MAP_SS, axis=0).repeat(MAP_SS, axis=1)[: self.map_h, : self.width]
        img[: big.shape[0], : big.shape[1]] = big
        for (_, x, y, bw, bh), (_, idx) in zip(self.box_rects, self.box_groups, strict=True):
            img[y : y + bh, x : x + bw] = self._box_image(w[idx], bw, bh)
        return BrainRender(img, None, self.outline, list(self.box_rects), self.taps)

    @staticmethod
    def _box_image(weights: np.ndarray, width: int, height: int) -> np.ndarray:
        """A mini heatmap: members packed in a near-square grid of cells (``drawBoxes``)."""
        m = len(weights)
        out = np.zeros((height, width, 4), np.uint8)
        out[..., 3] = 255
        out[..., :3] = (10, 12, 23)  # the viewer's near-black box backdrop
        if m == 0 or width <= 0 or height <= 0:
            return out
        cols = max(1, int(round(math.sqrt(m * width / max(1, height)))))
        rows = math.ceil(m / cols)
        grid = np.zeros((rows * cols,), np.float32)
        grid[:m] = weights
        filled = np.zeros(rows * cols, bool)
        filled[:m] = True
        ys = np.minimum((np.arange(height) * rows) // height, rows - 1)
        xs = np.minimum((np.arange(width) * cols) // width, cols - 1)
        cell = ys[:, None] * cols + xs[None, :]
        colors = hot_colormap(grid)
        mask = filled[cell]
        out[..., :3][mask] = colors[cell][mask]
        return out
