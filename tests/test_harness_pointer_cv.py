"""UC-74: the classic-OpenCV HUD next-gate pointer reader (:mod:`.liftoff.pointer_cv`).

In-sandbox coverage of the build-now acceptance criteria that need no real footage:

* **AC1** — zero ML: the reader's code path reaches no torch / no model load / no
  ``gate_net`` / ``gate_learned`` import (static scan + a fresh-interpreter probe that actually
  runs a detection).
* **AC3 (hand-computed)** — a principal-point icon maps to the optical axis
  ``[cos uptilt, 0, sin uptilt]``; an offset icon maps to an independently hand-derived az/el.
* **AC4** — an absent / ambiguous frame yields the degraded block ``[0, 0, 0, 0]`` (bit 0), never a
  stale or hallucinated bearing.
* **AC5 (offline guardrail)** — the per-frame read cost over a small ROI is cheap and does not
  regress a whole-frame scan (``offline.pointer_eval.time_read``).
* **AC7** — the hybrid ROI (explicit ``camera.pointer_roi`` override vs a ROI derived from
  ``camera.masks``) and mask non-interference (the gate detector masks its own copy; the reader
  reads the raw frame).

Detection exercises real OpenCV where available; following the project convention it
``importorskip``\\s ``cv2`` so the suite stays green on hosts where OpenCV cannot load.
"""

from __future__ import annotations

import ast
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from drone_fly.harness.geometry import CameraModel
from drone_fly.harness.liftoff.masks import FrameMasks
from drone_fly.harness.liftoff.pointer_cv import (
    HudPointerReader,
    PointerRoi,
    pointer_reader_from_config,
)
from drone_fly.harness.offline.pointer_eval import time_read

from .pointer_fixtures import default_camera, make_reader, render_icon

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "src" / "drone_fly" / "harness" / "liftoff" / "pointer_cv.py"


@pytest.fixture
def cv2():
    return pytest.importorskip("cv2")


# ----------------------------------------------------------------------------- AC1: no ML


def _imports(path: Path) -> list[str]:
    out: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out


def test_ac1_pointer_reader_source_imports_no_ml() -> None:
    """Static guard: the reader module names no ML dependency (torch / learned gate code)."""
    forbidden = ("torch", "gate_net", "gate_learned")
    hits = [m for m in _imports(MODULE) if any(f in m for f in forbidden)]
    assert hits == [], hits
    # no file-based model load dressed up as something else, either
    src = MODULE.read_text(encoding="utf-8")
    assert "load_state_dict" not in src and "torch.load" not in src


_AC1_PROBE = """
import sys
import numpy as np
from drone_fly.harness.geometry import CameraModel
from drone_fly.harness.liftoff.pointer_cv import HudPointerReader

cam = CameraModel(320, 180, 90.0, 0.0)
ranges = [(np.array([40, 100, 100], np.uint8), np.array([80, 255, 255], np.uint8))]
reader = HudPointerReader(cam, None, hsv_ranges=ranges)
img = np.full((180, 320, 3), 40, np.uint8)
try:
    import cv2
    cv2.circle(img, (200, 90), 6, (0, 255, 0), -1)
except Exception:
    pass
reader.read(img)  # exercise the full detect -> bearing path
bad = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
bad += [m for m in ("drone_fly.harness.liftoff.gate_learned",) if m in sys.modules]
print(bad)
sys.exit(1 if bad else 0)
"""


def test_ac1_reading_a_pointer_loads_no_ml_runtime(cv2) -> None:
    """Dynamic guard: running a detection in a fresh interpreter imports neither torch nor the
    learned gate estimator."""
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-c", _AC1_PROBE], capture_output=True, text=True,
                       env=env, cwd=REPO, timeout=300)  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr


# ----------------------------------------------------------------------------- AC3: bearing (hand)


def test_ac3_principal_point_icon_maps_to_optical_axis() -> None:
    """A principal-point icon → the camera optical axis ``[cos uptilt, 0, sin uptilt]`` (FLU)."""
    for uptilt in (0.0, 10.0, 25.0):
        cam = CameraModel(320, 180, 90.0, uptilt)
        reader = HudPointerReader(cam, None)
        bearing = reader.body_bearing((cam.cx, cam.cy), cam.width, cam.height)
        t = math.radians(uptilt)
        np.testing.assert_allclose(bearing, [math.cos(t), 0.0, math.sin(t)], atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(bearing), 1.0, atol=1e-6)


def test_ac3_offset_icon_maps_to_hand_computed_bearing() -> None:
    """An icon one focal length to the right (uptilt 0) → 45° right: az = -45°, el = 0.

    Hand derivation (independent of the production helper): with ``uptilt = 0`` the body axes are
    forward ``[1,0,0]``, left ``[0,1,0]``, up ``[0,0,1]``; a pixel offset ``+fx`` in x is a +1 unit
    in the camera right direction (= body right = ``-y``), so the un-normalised body ray is
    ``[1, -1, 0]`` → ``az = atan2(-1, 1) = -45°``, ``el = 0``.
    """
    cam = CameraModel(320, 180, 90.0, 0.0)  # hfov 90 → fx = 160
    reader = HudPointerReader(cam, None)
    bearing = reader.body_bearing((cam.cx + cam.fx, cam.cy), cam.width, cam.height)
    s = 1.0 / math.sqrt(2.0)
    np.testing.assert_allclose(bearing, [s, -s, 0.0], atol=1e-6)
    az = math.atan2(bearing[1], bearing[0])
    el = math.asin(bearing[2])
    assert math.isclose(math.degrees(az), -45.0, abs_tol=1e-4)
    assert math.isclose(math.degrees(el), 0.0, abs_tol=1e-4)


def test_ac3_offset_icon_elevation_tracks_vertical_offset() -> None:
    """A purely-vertical icon offset (above the principal point) raises the bearing elevation."""
    cam = CameraModel(320, 180, 90.0, 0.0)  # fx == fy == 160
    reader = HudPointerReader(cam, None)
    # one focal length *above* centre (smaller y pixel) → 45° up, az 0.
    bearing = reader.body_bearing((cam.cx, cam.cy - cam.fy), cam.width, cam.height)
    s = 1.0 / math.sqrt(2.0)
    np.testing.assert_allclose(bearing, [s, 0.0, s], atol=1e-6)


def test_ac3_detect_then_bearing_roundtrip(cv2) -> None:
    """End to end on a rendered fixture: the icon at the principal point reads back the axis."""
    cam = default_camera(uptilt_deg=12.0)
    reader = make_reader(cam)
    img = render_icon((cam.width // 2, cam.height // 2), size=(cam.width, cam.height))
    block = reader.read(img)
    assert block.shape == (4,) and block.dtype == np.float32
    t = math.radians(12.0)
    np.testing.assert_allclose(block, [math.cos(t), 0.0, math.sin(t), 1.0], atol=2e-2)


# ----------------------------------------------------------------------------- AC4: availability


def test_ac4_absent_frame_is_degraded_block(cv2) -> None:
    """A frame with no icon → ``[0, 0, 0, 0]`` (zero bearing, availability bit 0)."""
    reader = make_reader()
    block = reader.read(render_icon(None))
    np.testing.assert_array_equal(block, np.zeros(4, np.float32))
    assert block[3] == 0.0


def test_ac4_ambiguous_frame_rejects_to_unavailable(cv2) -> None:
    """Two equally-good candidates → the best match is not unique → unavailable (never a guess)."""
    reader = make_reader()
    img = render_icon((90, 90))
    import cv2 as _cv2

    _cv2.circle(img, (230, 90), 6, (0, 255, 0), -1)  # a second, identical glyph
    block = reader.read(img)
    np.testing.assert_array_equal(block, np.zeros(4, np.float32))


def test_ac4_present_frame_sets_availability_bit(cv2) -> None:
    """Sanity: a single clear icon is available (bit 1) with a unit bearing."""
    reader = make_reader()
    block = reader.read(render_icon((210, 70)))
    assert block[3] == 1.0
    np.testing.assert_allclose(np.linalg.norm(block[:3]), 1.0, atol=1e-5)


# ------------------------------------------- center_exclude_radius_frac (Arena ACRO-reticle fix)


def _two_glyph_frame(cv2, *centers, size=(320, 180)):
    img = render_icon(None, size=size)
    for c in centers:
        cv2.circle(img, (int(c[0]), int(c[1])), 6, (0, 255, 0), -1)
    return img


def test_center_exclude_default_zero_keeps_a_centred_icon(cv2) -> None:
    """Backward compatible: the default (0.0) does not exclude anything — a centred icon reads."""
    cam = default_camera(320, 180)
    reader = make_reader(cam)  # center_exclude_radius_frac defaults to 0.0
    assert reader.center_exclude_radius_frac == 0.0
    px = reader.detect_icon(render_icon((160, 90), size=(320, 180)))
    assert px is not None
    np.testing.assert_allclose(px, [160, 90], atol=2.0)


def test_center_exclude_drops_a_principal_point_detection(cv2) -> None:
    """A detection within the exclusion radius of the principal point is dropped → unavailable;
    an off-centre icon outside the radius is still read."""
    cam = default_camera(320, 180)  # principal point (160, 90); min(w,h)=180
    reader = make_reader(cam, center_exclude_radius_frac=0.22)  # ~40 px disk
    assert reader.detect_icon(render_icon((160, 90), size=(320, 180))) is None
    off = reader.detect_icon(render_icon((260, 70), size=(320, 180)))
    assert off is not None
    np.testing.assert_allclose(off, [260, 70], atol=2.0)


def test_center_exclude_removes_reticle_before_the_uniqueness_test(cv2) -> None:
    """The Arena fix: a centred reticle + an off-centre pointer. WITHOUT exclusion the two equal
    candidates trip the ambiguity reject (→ unavailable); WITH exclusion the reticle is dropped
    *before* the uniqueness test so the real pointer wins."""
    cam = default_camera(320, 180)
    reticle, pointer = (160, 90), (260, 70)
    frame = _two_glyph_frame(cv2, reticle, pointer)

    plain = make_reader(cam)  # no exclusion → ambiguous → unavailable
    assert plain.detect_icon(frame) is None

    fixed = make_reader(cam, center_exclude_radius_frac=0.22)
    px = fixed.detect_icon(frame)
    assert px is not None
    np.testing.assert_allclose(px, [260, 70], atol=2.0)  # the off-centre pointer, not the reticle


# ----------------------------------------------------------------------------- AC5: latency guard


def test_ac5_offline_read_is_cheap_and_roi_does_not_regress(cv2) -> None:
    """Per-frame read cost over a small ROI is cheap and no worse than a whole-frame scan."""
    size = (1280, 720)
    cam = default_camera(size[0], size[1])
    # ROI covering a small HUD strip at the bottom-centre (where the fixture glyph sits).
    roi = PointerRoi((520, 600, 240, 110), (size[0], size[1]))
    icon = (640, 650)
    full = make_reader(cam, None)
    scoped = make_reader(cam, roi)
    # radius chosen so the glyph area clears min_area_frac against the WHOLE 1280x720 frame too
    # (the scoped reader measures area against its much smaller ROI, so it is unaffected).
    img = render_icon(icon, size=size, radius=16)
    # both must still detect the icon (the ROI contains it)
    assert full.read(img)[3] == 1.0
    assert scoped.read(img)[3] == 1.0
    t_full = time_read(full, img, passes=30)
    t_roi = time_read(scoped, img, passes=30)
    assert t_roi < 0.1, f"ROI read too slow: {t_roi:.4f}s"
    # restricting to the ROI processes far fewer pixels, so it must not regress the full scan.
    assert t_roi <= t_full * 1.5, (t_roi, t_full)


# ----------------------------------------------------------------------------- AC7: hybrid ROI


def test_ac7_explicit_pointer_roi_override_wins_over_masks() -> None:
    """``camera.pointer_roi`` takes precedence over a derived-from-``camera.masks`` ROI."""
    masks = FrameMasks.from_config(
        {"resolution": [1920, 1080], "masks": [{"name": "hud", "rect": [0, 980, 1920, 100]}]}
    )
    cam_cfg = {"resolution": [1920, 1080], "pointer_roi": [800, 980, 320, 100], "masks": masks}
    roi = PointerRoi.from_config(cam_cfg, masks)
    assert roi is not None
    assert roi.rect == (800, 980, 320, 100)  # the override, not the full-width mask bbox


def test_ac7_roi_derived_from_masks_when_no_override() -> None:
    """With no override, the ROI is the bounding box of the HUD ``camera.masks`` geometry."""
    masks = FrameMasks.from_config(
        {
            "resolution": [1920, 1080],
            "masks": [
                {"name": "a", "rect": [850, 980, 100, 80]},
                {"name": "b", "rect": [1000, 1000, 120, 60]},
            ],
        }
    )
    roi = PointerRoi.from_config({"resolution": [1920, 1080]}, masks)
    assert roi is not None
    # bbox: x0=850, y0=980, x1=1120, y1=1060 → (850, 980, 270, 80)
    assert roi.rect == (850, 980, 1120 - 850, 1060 - 980)


def test_ac7_named_mask_selects_only_that_region() -> None:
    """``camera.pointer_roi_mask`` narrows the derived ROI to the named mask region."""
    masks = FrameMasks.from_config(
        {
            "resolution": [1920, 1080],
            "masks": [
                {"name": "pointer", "rect": [900, 500, 120, 120]},
                {"name": "laptimer", "rect": [0, 0, 300, 60]},
            ],
        }
    )
    roi = PointerRoi.from_config({"resolution": [1920, 1080], "pointer_roi_mask": "pointer"}, masks)
    assert roi is not None and roi.rect == (900, 500, 120, 120)


def test_ac7_roi_scales_with_capture_resolution() -> None:
    """ROI pixels scale with the capture size, the same way masks do (``CameraModel.scaled_to``)."""
    roi = PointerRoi((100, 50, 40, 20), (200, 100))
    # at native resolution
    assert roi.rect_px(200, 100) == (100, 50, 140, 70)
    # doubled resolution → doubled pixel rect
    assert roi.rect_px(400, 200) == (200, 100, 280, 140)


def test_ac7_reader_reads_raw_frame_while_gate_detector_masks_its_copy(cv2) -> None:
    """Mask non-interference: the glyph lives inside the HUD mask region; the pointer reader still
    reads it from the raw frame, while ``FrameMasks.apply`` blanks it on the detector's own copy."""
    size = (320, 180)
    cam = default_camera(size[0], size[1])
    # a HUD mask covering the bottom strip; the glyph sits inside it.
    masks = FrameMasks.from_config(
        {"resolution": list(size), "masks": [{"name": "hud", "rect": [0, 120, 320, 60]}]}
    )
    roi = PointerRoi.from_config({"resolution": list(size)}, masks)
    reader = make_reader(cam, roi)
    icon = (160, 150)  # inside the masked strip
    img = render_icon(icon, size=size)

    # the pointer reader (raw frame, ROI) finds and reads it.
    block = reader.read(img)
    assert block[3] == 1.0

    # the gate detector's masked copy has that region blanked, so it could never detect there —
    # and masking does not mutate the raw frame the reader read.
    masked = masks.apply(img)
    assert masked is not img
    assert int(masked[150, 160].sum()) == 0  # glyph gone on the detector's copy
    assert int(img[150, 160].sum()) > 0  # raw frame untouched


# ----------------------------------------------------------------------------- config plumbing


def test_pointer_reader_from_config_disabled_or_absent_returns_none() -> None:
    """Legacy / sim configs (no ``camera.pointer`` or explicitly disabled) build no reader."""
    cam = default_camera()
    assert pointer_reader_from_config(cam, {"resolution": [320, 180]}) is None
    assert pointer_reader_from_config(cam, {"pointer": {}}) is None
    assert pointer_reader_from_config(cam, {"pointer": {"enabled": False}}) is None


def test_pointer_reader_from_config_builds_reader_from_camera_block() -> None:
    """A ``camera.pointer`` block with HSV ranges + an explicit ROI builds a wired reader."""
    cam = default_camera()
    cam_cfg = {
        "resolution": [320, 180],
        "pointer_roi": [100, 120, 120, 60],
        "pointer": {
            "enabled": True,
            "hsv_ranges": [[[40, 100, 100], [80, 255, 255]]],
            "min_circularity": 0.5,
            "center_exclude_radius_frac": 0.03,
        },
    }
    reader = pointer_reader_from_config(cam, cam_cfg)
    assert isinstance(reader, HudPointerReader)
    assert reader.roi is not None and reader.roi.rect == (100, 120, 120, 60)
    assert len(reader.hsv_ranges) == 1 and reader.min_circularity == 0.5
    assert reader.center_exclude_radius_frac == pytest.approx(0.03)  # from_config key parsed
