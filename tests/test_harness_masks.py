"""UC-64 AC7: screen-region masks before gate detection, auto-labelling and depth sampling.

``FrameMasks`` itself, a fake :class:`TrackingGateEstimator` subclass (no OpenCV, CI) showing that
content inside a mask yields no detection, and the depth estimator showing a masked near object
produces no ray hit and no obstacle cluster. (Auto-labelling: ``test_harness_autolabel.py``.)
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.geometry import CameraModel
from drone_fly.harness.liftoff.depth import MonocularDepthFreeSpaceEstimator
from drone_fly.harness.liftoff.gate_cv import GateDetection, TrackingGateEstimator
from drone_fly.harness.liftoff.masks import FrameMasks, MaskRegion
from drone_fly.harness.types import Frame
from tests.harness_support import hover_state

H, W = 108, 256  # 2560 x 1080 / 10


def _masks(*rects, resolution=(W, H)):
    return FrameMasks([MaskRegion(f"m{i}", r) for i, r in enumerate(rects)], resolution)


# ----------------------------------------------------------------------------- FrameMasks


def test_from_config_and_preset_masks():
    cam = load_config("liftoff", preset="liftoff").section("camera")
    m = FrameMasks.from_config(cam)
    assert len(m) == 6 and bool(m) and m.resolution == (2560, 1080)
    assert not FrameMasks.from_config({"masks": []})
    assert not FrameMasks.from_config({})


@pytest.mark.parametrize(
    ("camera", "match"),
    [
        ({"masks": [{"name": "a", "rect": [0, 0, 1, 1]}]}, "resolution"),
        ({"masks": "nope", "resolution": [10, 10]}, "list"),
        ({"masks": [{"name": "a"}], "resolution": [10, 10]}, "rect"),
        ({"masks": [{"rect": [0, 0, 0, 5]}], "resolution": [10, 10]}, "width/height"),
        ({"masks": [{"rect": [0, 0, 5]}], "resolution": [10, 10]}, "4 numbers"),
    ],
)
def test_from_config_validation(camera, match):
    with pytest.raises(HarnessConfigError, match=match):
        FrameMasks.from_config(camera)


def test_rects_scale_to_the_frame_and_clip():
    m = _masks((10, 20, 30, 40), (250, 100, 50, 50), resolution=(W, H))
    np.testing.assert_array_equal(m.rects(W, H), [[10, 20, 40, 60], [250, 100, 256, 108]])
    np.testing.assert_array_equal(m.rects(2 * W, 2 * H)[0], [20, 40, 80, 120])
    np.testing.assert_array_equal(m.rects(W // 2, H // 2)[0], [5, 10, 20, 30])


def test_apply_valid_mask_and_contains():
    m = _masks((10, 20, 30, 40))
    img = np.full((H, W, 3), 200, np.uint8)
    out = m.apply(img)
    assert out is not img and (img == 200).all()  # copy, input untouched
    assert (out[20:60, 10:40] == 0).all() and (out[:20] == 200).all()
    assert (m.apply(img, fill=7)[20:60, 10:40] == 7).all()
    ok = m.valid_mask(W, H)
    assert ok.shape == (H, W) and not ok[20:60, 10:40].any() and ok.sum() == W * H - 30 * 40
    pts = np.array([[10, 20], [39.9, 59.9], [40, 30], [9.9, 30]])
    np.testing.assert_array_equal(m.contains(pts, W, H), [True, True, False, False])
    empty = FrameMasks()
    assert empty.apply(img) is img and empty.valid_mask(W, H).all()
    assert not empty.contains(pts, W, H).any()


# ----------------------------------------------------------------------------- gate tracker


class BrightBlobEstimator(TrackingGateEstimator):
    """Fake detector: one gate per bright pixel column run, 10 m ahead, centred on the blob."""

    def __init__(self, masks):
        super().__init__(CameraModel(W, H, 120.0, 0.0), default_aperture=1.5, masks=masks)
        self.seen_images = []

    def detect(self, image, state):
        self.seen_images.append(image)
        ys, xs = np.nonzero(image[..., 0] > 128)
        if len(xs) == 0:
            return []
        out = []
        for lo, hi in ((0, W // 2), (W // 2, W)):
            sel = (xs >= lo) & (xs < hi)
            if not sel.any():
                continue
            c = np.array([xs[sel].mean(), ys[sel].mean()])
            ray = self.camera.pixel_rays(c[None])[0]
            t_cam = ray / np.linalg.norm(ray) * 10.0
            det = self.detection_from_pose(
                t_cam,
                np.array([0.0, 0.0, -1.0]),
                c[None],
                0.01,
                state,
                self.camera,
                1.5,
                cls="g",
                pixel_center=c,
            )
            out.append(det)
        return out


def _frame(blobs):
    img = np.zeros((H, W, 3), np.uint8)
    for x, y in blobs:
        img[y - 3 : y + 3, x - 3 : x + 3] = 255
    return Frame(img, 0.0, 1)


def test_content_inside_a_mask_is_never_seen_by_the_detector():
    est = BrightBlobEstimator(_masks((180, 10, 70, 40)))  # a "ghost list" region
    out = est.update(_frame([(215, 30)]), hover_state(), 0.0)
    assert out.info["detections"] == 0 and not out.info["matched"]
    assert (est.seen_images[-1][10:50, 180:250] == 0).all()  # masked before detect()
    assert out.status == "holding"  # acquiring, not tracking


def test_unmasked_gate_is_detected_next_to_a_masked_one():
    est = BrightBlobEstimator(_masks((180, 10, 70, 40)))
    out = est.update(_frame([(60, 60), (215, 30)]), hover_state(), 0.0)
    assert out.info["detections"] == 1 and out.status == "tracking"
    assert out.info["gate_class"] == "g"


class CentredInMaskEstimator(BrightBlobEstimator):
    """Reports its detection centre inside the mask although the evidence lies outside it."""

    def detect(self, image, state):
        dets = super().detect(image, state)
        return [
            GateDetection(**{**d.__dict__, "pixel_center": np.array([215.0, 30.0])}) for d in dets
        ]


def test_detection_centred_in_a_mask_is_dropped():
    est = CentredInMaskEstimator(_masks((180, 10, 70, 40)))
    out = est.update(_frame([(60, 60)]), hover_state(), 0.0)
    assert out.info["detections"] == 0


def test_without_masks_the_same_content_is_detected():
    est = BrightBlobEstimator(FrameMasks())
    out = est.update(_frame([(215, 30)]), hover_state(), 0.0)
    assert out.info["detections"] == 1


# ----------------------------------------------------------------------------- depth


class NearBlobDepth:
    inverse = False

    def predict(self, image):
        d = np.full(image.shape[:2], 100.0, np.float32)
        d[30:80, 100:160] = 2.0  # a near object dead ahead
        return d


def _depth_est(contract, masks):
    return MonocularDepthFreeSpaceEstimator(
        NearBlobDepth(),
        CameraModel(W, H, 120.0, 0.0),
        contract,
        ground_z_m=None,
        cluster_stride=4,
        near_m=3.0,
        masks=masks,
    )


def _depth_frame():
    return Frame(np.zeros((H, W, 3), np.uint8), 0.0, 1)


def test_masked_depth_gives_no_ray_hit_and_no_cluster(harness_contract):
    state = hover_state(vel=(1.0, 0.0, 0.0))
    target = np.array([5.0, 0.0, 1.0], np.float32)
    clear = _depth_est(harness_contract, FrameMasks()).update(_depth_frame(), state, target, 0.0)
    assert clear.raycast.min() < 1.0  # the object blocks some rays
    assert clear.obstacle_vision[3] > 0  # and forms a cluster

    masked_est = _depth_est(harness_contract, _masks((90, 20, 80, 70)))
    masked = masked_est.update(_depth_frame(), state, target, 0.0)
    np.testing.assert_array_equal(masked.raycast, np.ones_like(masked.raycast))
    np.testing.assert_array_equal(masked.obstacle_vision, np.zeros(8))
    depth = masked_est.latest_result().depth_m
    assert np.isinf(depth[20:90, 90:170]).all() and np.isfinite(depth[:20]).all()
