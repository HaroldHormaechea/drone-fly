"""Synthetic fixtures for the UC-74 HUD pointer reader tests.

No real footage is needed to build and prove the reader in-sandbox: these helpers render a fixed
HUD-icon glyph at a *known* pixel on a flat background (the convention
``tests/liftoff_capture_support.py`` uses for synthetic frames), build a
:class:`~drone_fly.harness.liftoff.pointer_cv.HudPointerReader`, and expose the ground truth the
offline eval harness scores against.

OpenCV is only needed to *render* the glyph (``cv2.circle``); importing this module does not require
it, so tests should ``pytest.importorskip("cv2")`` before calling :func:`render_icon`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drone_fly.harness.geometry import CameraModel
from drone_fly.harness.liftoff.pointer_cv import HudPointerReader, PointerRoi

#: A bright-green filled disc is the stand-in HUD glyph; this HSV band isolates it (OpenCV H is
#: 0-179, so pure green is ~60) and nothing else on the flat grey background matches.
GREEN_HSV: list[tuple[np.ndarray, np.ndarray]] = [
    (np.array([40, 100, 100], np.uint8), np.array([80, 255, 255], np.uint8))
]

#: A default synthetic capture size (wide HUD strip, same aspect family as the Arena preset).
DEFAULT_SIZE = (320, 180)
BG_GREY = 40


def default_camera(width: int = DEFAULT_SIZE[0], height: int = DEFAULT_SIZE[1],
                   hfov_deg: float = 90.0, uptilt_deg: float = 0.0) -> CameraModel:  # fmt: skip
    """A pinhole camera for the fixtures (``hfov 90`` → ``fx = width/2``)."""
    return CameraModel(width, height, hfov_deg, uptilt_deg)


def render_icon(center: tuple[int, int] | None, *, size: tuple[int, int] = DEFAULT_SIZE,
                radius: int = 6, color: tuple[int, int, int] = (0, 255, 0),
                bg: int = BG_GREY) -> np.ndarray:  # fmt: skip
    """A ``H x W x 3`` BGR frame with the glyph at ``center`` (``None`` → an empty/absent frame)."""
    import cv2

    w, h = size
    img = np.full((h, w, 3), bg, np.uint8)
    if center is not None:
        cv2.circle(img, (int(center[0]), int(center[1])), radius, color, -1)
    return img


def make_reader(camera: CameraModel | None = None, roi: PointerRoi | None = None,
                hsv_ranges: Any = None, **kw: Any) -> HudPointerReader:  # fmt: skip
    """A :class:`HudPointerReader` wired to detect the green fixture glyph."""
    cam = camera if camera is not None else default_camera()
    ranges = GREEN_HSV if hsv_ranges is None else hsv_ranges
    return HudPointerReader(cam, roi, hsv_ranges=ranges, **kw)
