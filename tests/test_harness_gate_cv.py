"""UC-62 AC6: OpenCV gate estimator on synthetic rendered frames.

OpenCV is an optional backend, absent from default CI: the cv2 tests skip there. Run them with
the ephemeral overlay (changes neither pyproject nor uv.lock):

    uv run --with opencv-python-headless --extra dev pytest tests/test_harness_gate_cv.py
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.harness.config import load_config
from drone_fly.harness.geometry import CameraModel, quat_to_matrix
from drone_fly.harness.liftoff.gate_cv import OpenCvGateEstimator, order_corners
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import FlightState, Frame

RED = [([0, 120, 80], [10, 255, 255]), ([170, 120, 80], [179, 255, 255])]
W_M = H_M = 1.5  # nominal outer gate size (target metres)


@pytest.fixture
def cv2():
    return pytest.importorskip("cv2")


def _state(pos=(0.0, 0.0, 1.0), quat=(1.0, 0.0, 0.0, 0.0)) -> FlightState:
    return FlightState(
        pos=np.array(pos, np.float32),
        vel=np.zeros(3, np.float32),
        quat=np.array(quat, np.float32),
        omega=np.zeros(3, np.float32),
    )


def _render(cv2, camera: CameraModel, state: FlightState, gates, scale=1.0, size=(W_M, H_M)):
    """Draw red gate frames (outer quad red, inner opening dark) as seen from ``state``."""
    img = np.full((camera.height, camera.width, 3), 40, np.uint8)
    r_cw = camera.world_to_camera(state.quat)
    pos = np.asarray(state.pos, np.float64)
    # Draw far gates first so near ones occlude them.
    order = sorted(gates, key=lambda g: -np.linalg.norm(np.asarray(g[0]) - pos))
    for center, yaw in order:
        lat = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        for frac, colour in ((1.0, (0, 0, 255)), (0.7, (40, 40, 40))):
            hw, hh = frac * size[0] / 2 / scale, frac * size[1] / 2 / scale
            corners = [
                np.asarray(center) + a * hw * lat + b * hh * up
                for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))
            ]
            cam = np.array([scale * r_cw @ (c - pos) for c in corners])
            px, front = camera.project(cam)
            assert front.all()
            cv2.fillPoly(img, [np.round(px).astype(np.int32)], colour)
    return img


def _estimator(camera, scale=1.0, **kw):
    return OpenCvGateEstimator(camera, W_M, H_M, RED, WorldScale(scale), **kw)


# ----------------------------------------------------------------------------- pure helpers


def test_order_corners_tl_tr_br_bl():
    pts = np.array([[10, 90], [90, 10], [10, 10], [90, 90]])
    np.testing.assert_array_equal(order_corners(pts), [[10, 10], [90, 10], [90, 90], [10, 90]])


def test_constructor_validation():
    from drone_fly.harness.errors import HarnessConfigError

    cam = CameraModel(64, 48, 90.0)
    with pytest.raises(HarnessConfigError):
        OpenCvGateEstimator(cam, 0.0, 1.5, RED)
    with pytest.raises(HarnessConfigError):
        OpenCvGateEstimator(cam, 1.5, 1.5, [])


def test_from_config_builds_camera_and_scale():
    cfg = load_config("liftoff", {"camera": {"resolution": [800, 600], "hfov_deg": 100.0}})
    est = OpenCvGateEstimator.from_config(cfg.section("gates"), SimpleNamespace(config=cfg))
    assert (est.camera.width, est.camera.height, est.camera.hfov_deg) == (800, 600, 100.0)
    assert est.camera.uptilt_deg == 25.0 and est.scale == WorldScale(2.0)
    assert est.lost_timeout_s == 1.0 and len(est.hsv_ranges) == 2
    assert est.provides == frozenset({"target", "curr_gate", "next_lookahead"})


def test_camera_model_geometry():
    cam = CameraModel(640, 480, 90.0, uptilt_deg=0.0)
    assert cam.fx == pytest.approx(320.0)
    r = cam.body_to_camera()
    np.testing.assert_allclose(r @ [1, 0, 0], [0, 0, 1], atol=1e-12)  # forward -> optical axis
    np.testing.assert_allclose(r @ [0, -1, 0], [1, 0, 0], atol=1e-12)  # body right -> image right
    np.testing.assert_allclose(r @ [0, 0, 1], [0, -1, 0], atol=1e-12)  # up -> image up (-y)
    tilted = CameraModel(640, 480, 90.0, uptilt_deg=30.0).body_to_camera()
    np.testing.assert_allclose(tilted[2], [np.cos(np.radians(30)), 0, np.sin(np.radians(30))])
    px, front = cam.project(np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]]))
    np.testing.assert_allclose(px[0], [320, 240])
    assert list(front) == [True, False]
    assert cam.scaled_to(1280, 960).fx == pytest.approx(640.0)


# ----------------------------------------------------------------------------- detection + PnP


@pytest.mark.parametrize(
    "center,yaw,uptilt,quat,scale",
    [
        ((5.0, 0.5, 1.3), 0.0, 0.0, (1, 0, 0, 0), 1.0),
        ((4.0, -0.8, 0.9), 0.35, 0.0, (1, 0, 0, 0), 1.0),
        (
            (3.0, 0.4, 1.1),
            -0.3,
            25.0,
            (np.cos(np.radians(12.5)), 0, np.sin(np.radians(12.5)), 0),
            1.0,
        ),
        ((2.5, 0.2, 1.2), 0.2, 0.0, (1, 0, 0, 0), 2.0),
    ],
)
def test_pnp_pose_recovers_gate_centre_and_yaw(cv2, center, yaw, uptilt, quat, scale):
    camera = CameraModel(640, 480, 90.0, uptilt)
    state = _state(quat=quat)
    img = _render(cv2, camera, state, [(center, yaw)], scale=scale)
    est = _estimator(camera, scale)
    dets = est.detect(img, state)
    assert len(dets) == 1
    d = dets[0]
    assert np.linalg.norm(d.center - np.asarray(center)) < 0.05 * max(1.0, np.linalg.norm(center))
    assert abs(np.degrees(np.angle(np.exp(1j * (d.yaw - yaw))))) < 6.0
    assert d.aperture == pytest.approx(min(W_M, H_M) / scale)


def test_blank_frame_has_no_quads(cv2):
    est = _estimator(CameraModel(320, 240, 90.0))
    assert est.detect_quads(np.zeros((240, 320, 3), np.uint8)) == []


def test_capture_resolution_change_rescales_intrinsics(cv2):
    camera = CameraModel(640, 480, 90.0)
    state = _state()
    est = _estimator(CameraModel(1920, 1080, 90.0))  # configured size differs from the capture
    d = est.detect(_render(cv2, camera, state, [((5.0, 0.0, 1.0), 0.0)]), state)[0]
    assert est.camera.width == 640 and np.linalg.norm(d.center - [5.0, 0.0, 1.0]) < 0.2


# ----------------------------------------------------------------------------- tracking


GATE_A = ((4.0, -1.0, 1.0), 0.0)
GATE_B = ((9.0, 1.5, 1.2), 0.2)


def test_current_and_next_gate_with_vis_flag(cv2):
    camera = CameraModel(640, 480, 90.0)
    state = _state()
    est = _estimator(camera)
    g = est.update(Frame(_render(cv2, camera, state, [GATE_A, GATE_B]), 0.0, 1), state, 0.0)
    assert g.status == "tracking" and g.gate_index == 0 and not g.passed
    assert np.linalg.norm(g.target - np.asarray(GATE_A[0])) < 0.2
    np.testing.assert_allclose(g.curr_gate[:2], [np.cos(GATE_A[1]), np.sin(GATE_A[1])], atol=0.1)
    assert g.curr_gate[2] == pytest.approx(1.5)
    nl = g.next_lookahead
    assert nl[6] == 1.0  # next gate visible
    bearing = np.asarray(GATE_B[0]) - state.pos
    np.testing.assert_allclose(nl[:3], bearing / np.linalg.norm(bearing), atol=0.03)
    assert nl[5] == pytest.approx(min(np.linalg.norm(bearing) / 10, 1), abs=0.05)


def test_next_gate_unseen_without_history_bears_on_current_gate(cv2):
    camera = CameraModel(640, 480, 90.0)
    state = _state()
    est = _estimator(camera)
    g = est.update(Frame(_render(cv2, camera, state, [GATE_A]), 0.0, 1), state, 0.0)
    nl = g.next_lookahead
    assert nl[6] == 0.0 and np.all(nl[3:6] == 0.0)
    expected = np.asarray(GATE_A[0]) - state.pos
    np.testing.assert_allclose(nl[:3], expected / np.linalg.norm(expected), atol=0.03)
    assert np.linalg.norm(nl[:3]) == pytest.approx(1.0, abs=1e-5)


def test_pass_advances_to_the_known_next_gate(cv2):
    camera = CameraModel(640, 480, 90.0)
    s0 = _state()
    est = _estimator(camera)
    est.update(Frame(_render(cv2, camera, s0, [GATE_A, GATE_B]), 0.0, 1), s0, 0.0)
    at_gate = _state(pos=GATE_A[0])
    g = est.update(None, at_gate, 0.1)
    assert g.passed and g.gate_index == 1
    assert np.linalg.norm(g.target - np.asarray(GATE_B[0])) < 0.5
    assert g.status == "holding"


def test_hold_then_lost_after_timeout(cv2):
    camera = CameraModel(640, 480, 90.0)
    state = _state()
    est = _estimator(camera, lost_timeout_s=1.0)
    first = est.update(Frame(_render(cv2, camera, state, [GATE_A]), 0.0, 1), state, 0.0)
    blank = np.full((480, 640, 3), 40, np.uint8)
    held = est.update(Frame(blank, 0.5, 2), state, 0.5)
    assert held.status == "holding"
    np.testing.assert_array_equal(held.target, first.target)  # last world estimate held
    lost = est.update(Frame(blank, 1.2, 3), state, 1.2)
    assert lost.status == "lost"
    est.reset()
    assert est.current is None and est.gate_index == 0


def test_acquisition_hovers_then_fails_safe(cv2):
    camera = CameraModel(640, 480, 90.0)
    state = _state(pos=(1.0, 2.0, 1.5))
    est = _estimator(camera, acquire_timeout_s=3.0)
    blank = Frame(np.full((480, 640, 3), 40, np.uint8), 0.0, 1)
    g = est.update(blank, state, 10.0)
    assert g.status == "holding"
    np.testing.assert_array_equal(g.target, state.pos)  # hover in place while acquiring
    assert g.next_lookahead[6] == 0.0 and np.isfinite(g.next_lookahead).all()
    assert est.update(None, state, 12.9).status == "holding"
    assert est.update(None, state, 13.5).status == "lost"


def test_gates_per_lap_counts_laps(cv2):
    camera = CameraModel(640, 480, 90.0)
    est = _estimator(camera, gates_per_lap=1)
    s0 = _state()
    est.update(Frame(_render(cv2, camera, s0, [GATE_A]), 0.0, 1), s0, 0.0)
    g = est.update(None, _state(pos=GATE_A[0]), 0.1)
    assert g.passed and g.lap == 1


def test_detection_respects_world_frame_rotation(cv2):
    """A yawed drone sees the same world gate: the estimate is in world, not body, coordinates."""
    camera = CameraModel(640, 480, 90.0)
    h = np.radians(40) / 2
    yawed = _state(quat=(np.cos(h), 0, 0, np.sin(h)))
    fwd = quat_to_matrix(yawed.quat) @ [1, 0, 0]
    center = tuple(np.array([0.0, 0.0, 1.0]) + 5.0 * fwd)
    img = _render(cv2, camera, yawed, [(center, np.radians(40))])
    d = _estimator(camera).detect(img, yawed)[0]
    assert np.linalg.norm(d.center - np.asarray(center)) < 0.25
