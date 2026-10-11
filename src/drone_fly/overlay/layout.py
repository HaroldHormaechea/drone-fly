"""Overlay layouts and the capture-exclusion fallback decision (toolkit-free).

``default``
    Panels at the screen edges, the game centre clear: a left column (x 0.005-0.175 of the
    window) with action / brain / inputs and a right column (x 0.825-0.995) with the perception
    text / health / telemetry / raycast fan. The detected gate keypoints are drawn in game-screen
    space over the gate.

``mask_confined``
    Every panel inside a configured HUD mask (``camera.masks`` — regions the gate detector,
    auto-labeller and depth sampling already ignore), so overlay pixels can never reach
    perception even if they are captured. Greedy: panels in priority order (health > brain >
    action > telemetry > inputs > perception text) each take the largest remaining mask that is
    big enough; the ray fan and the in-game keypoints are disabled; panels that do not fit are
    hidden and logged.

:func:`decide_capture_mode` is the AC6 fallback: when ``SetWindowDisplayAffinity
(WDA_EXCLUDEFROMCAPTURE)`` fails, the overlay switches to ``mask_confined`` — or, with no masks
configured, hides every panel (a loud warning) rather than pollute the harness's capture.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

CAPTURE_EXCLUDED = "excluded"
CAPTURE_MASK_CONFINED = "mask_confined"
CAPTURE_HIDDEN = "hidden"

#: Default layout: panel -> (x0, y0, x1, y1) as fractions of the window.
DEFAULT_RECTS: dict[str, tuple[float, float, float, float]] = {
    "action": (0.005, 0.02, 0.175, 0.25),
    "brain": (0.005, 0.27, 0.175, 0.62),
    "inputs": (0.005, 0.64, 0.175, 0.98),
    "perception": (0.825, 0.02, 0.995, 0.17),
    "health": (0.825, 0.19, 0.995, 0.50),
    "telemetry": (0.825, 0.52, 0.995, 0.66),
    "raycast": (0.825, 0.68, 0.995, 0.98),
}
#: mask_confined placement priority (the ray fan is never placed in masks).
MASK_PRIORITY = ("health", "brain", "action", "telemetry", "inputs", "perception")
MIN_PANEL_W = 80
MIN_PANEL_H = 40


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def contains(self, other: Rect) -> bool:
        return (
            other.x >= self.x
            and other.y >= self.y
            and other.x + other.w <= self.x + self.w
            and other.y + other.h <= self.y + self.h
        )


@dataclass
class Layout:
    """Where each enabled panel goes (window-local pixels) and what may be drawn in-game."""

    mode: str
    rects: dict[str, Rect] = field(default_factory=dict)
    draw_keypoints: bool = True
    draw_ray_fan: bool = True
    hidden: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def decide_capture_mode(affinity_ok: bool, masks_present: bool) -> str:
    """``excluded`` (normal), ``mask_confined`` (affinity failed, masks set) or ``hidden``."""
    if affinity_ok:
        return CAPTURE_EXCLUDED
    return CAPTURE_MASK_CONFINED if masks_present else CAPTURE_HIDDEN


def _enabled(panels: Mapping[str, bool] | Iterable[tuple[str, bool]]) -> dict[str, bool]:
    return dict(panels.items() if isinstance(panels, Mapping) else panels)


def default_layout(
    width: int, height: int, panels: Mapping[str, bool] | Iterable[tuple[str, bool]]
) -> Layout:
    on = _enabled(panels)
    rects = {}
    for name, (x0, y0, x1, y1) in DEFAULT_RECTS.items():
        if on.get(name, False):
            rects[name] = Rect(
                int(round(x0 * width)),
                int(round(y0 * height)),
                int(round((x1 - x0) * width)),
                int(round((y1 - y0) * height)),
            )
    return Layout("default", rects, draw_keypoints=on.get("perception", False))


def mask_confined_layout(
    masks: Sequence[Rect],
    panels: Mapping[str, bool] | Iterable[tuple[str, bool]],
    *,
    min_w: int = MIN_PANEL_W,
    min_h: int = MIN_PANEL_H,
    inset: int = 2,
) -> Layout:
    """Greedy placement of panels into the largest masks (window-local pixels), one per mask."""
    on = _enabled(panels)
    free = sorted(masks, key=lambda r: r.area, reverse=True)
    layout = Layout("mask_confined", draw_keypoints=False, draw_ray_fan=False)
    if on.get("raycast"):
        layout.hidden.append("raycast")
    for name in MASK_PRIORITY:
        if not on.get(name):
            continue
        slot = next(
            (m for m in free if m.w - 2 * inset >= min_w and m.h - 2 * inset >= min_h), None
        )
        if slot is None:
            layout.hidden.append(name)
            continue
        free.remove(slot)
        layout.rects[name] = Rect(
            slot.x + inset, slot.y + inset, slot.w - 2 * inset, slot.h - 2 * inset
        )
    if layout.hidden:
        msg = f"mask_confined layout: no room for {', '.join(layout.hidden)} (hidden)"
        layout.warnings.append(msg)
        log.warning(msg)
    return layout


def hidden_layout(reason: str) -> Layout:
    msg = f"overlay panels hidden: {reason}"
    log.warning(msg)
    return Layout(CAPTURE_HIDDEN, draw_keypoints=False, draw_ray_fan=False, warnings=[msg])


def masks_to_window(
    mask_rects_px: np.ndarray, frame_size: Sequence[float], screen_map: object
) -> list[Rect]:
    """``FrameMasks.rects`` (frame pixels ``x0, y0, x1, y1``) -> window-local :class:`Rect` s."""
    out = []
    for x0, y0, x1, y1 in np.asarray(mask_rects_px, np.float64).reshape(-1, 4):
        corners = screen_map.map(np.array([[x0, y0], [x1, y1]]), frame_size)  # type: ignore[attr-defined]
        (a, b), (c, d) = corners
        x, y = int(np.ceil(min(a, c))), int(np.ceil(min(b, d)))
        w, h = int(np.floor(max(a, c))) - x, int(np.floor(max(b, d))) - y
        if w > 0 and h > 0:
            out.append(Rect(x, y, w, h))
    return out


def resolve_layout(
    layout_name: str,
    width: int,
    height: int,
    panels: Mapping[str, bool] | Iterable[tuple[str, bool]],
    *,
    affinity_ok: bool,
    masks: Sequence[Rect] = (),
) -> Layout:
    """The layout to use: the configured one, or the AC6 fallback when capture exclusion failed."""
    capture = decide_capture_mode(affinity_ok, bool(masks))
    if capture == CAPTURE_HIDDEN:
        log.error(
            "SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) failed and no camera.masks are "
            "configured: hiding every overlay panel so nothing can enter the harness's capture"
        )
        return hidden_layout("capture exclusion failed and no HUD masks are configured")
    if capture == CAPTURE_MASK_CONFINED:
        log.error(
            "SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) failed: switching to the "
            "mask_confined layout (panels only inside camera.masks, no in-game drawings)"
        )
        return mask_confined_layout(masks, panels)
    if layout_name == "mask_confined":
        if not masks:
            return hidden_layout("layout mask_confined needs camera.masks")
        return mask_confined_layout(masks, panels)
    return default_layout(width, height, panels)
