"""UC-91: gate-centre back-projection, the size-range cross-check and the offline evaluator.

CPU-hermetic (no GPU, no Liftoff, no network): synthetic gates are projected through
:class:`CameraModel` and back-projected with IPPE PnP + ``detection_from_pose`` (AC5), the uptilt
sign and the principal point are exercised (AC3/AC6), ``gate_range`` (size range, truncation
fallback, ratio / tilt checks, config) is unit-tested, and the evaluator
(``offline/gate_backprojection_eval``) runs on synthetic run folders under ``tmp_path`` with every
directory listing call patched to raise. The AC4 measurement and the AC7 re-fly are data outcomes
on real flights and are not asserted here; the pre-registered AC4 rule is tested on synthetic
statistics instead.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.gate_types import GateTypeTable
from drone_fly.harness.geometry import (
    CameraModel,
    contract_to_unity_position,
    contract_to_unity_quat,
)
from drone_fly.harness.liftoff import gate_range
from drone_fly.harness.liftoff.gate_cv import (
    GateDetection,
    TrackingGateEstimator,
    camera_from_config,
    solve_planar_pnp,
)
from drone_fly.harness.liftoff.gate_range import (
    RangeCheck,
    check_detection,
    evaluate_detection,
    normal_tilt_deg,
    range_mode_from_config,
    size_range,
    size_range_valid,
)
from drone_fly.harness.liftoff.masks import FrameMasks, MaskRegion
from drone_fly.harness.offline import gate_backprojection_eval as gbe
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import FlightState
from tests.liftoff_capture_support import SMALL_CAMERA

cv2 = pytest.importorskip("cv2")

CFG = load_config("liftoff", SMALL_CAMERA, preset="liftoff")
TABLE = GateTypeTable.from_config(CFG.data["gate_types"])
ARCH = TABLE["inflatable_arch"]
OBJ = ARCH.object_points()
LIVE = CameraModel(2560, 1080, 120.0, 20.0)
OFF_PP = CameraModel(2560, 1080, 120.0, 20.0, principal_point=(1347.0, 498.0))


# ----------------------------------------------------------------------------- synthetic geometry


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def axis_quat(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    h = math.radians(deg) / 2
    return np.array([math.cos(h), *(math.sin(h) * a)])


def attitude(yaw=0.0, pitch=0.0, roll=0.0) -> np.ndarray:
    """``[w, x, y, z]`` body -> world: yaw about Z, then pitch about body Y, then roll about X."""
    q = qmul(axis_quat([0, 0, 1], yaw), axis_quat([0, 1, 0], pitch))
    return qmul(q, axis_quat([1, 0, 0], roll))


def gate_world_points(center, facing_yaw_deg: float, lean_deg: float = 0.0) -> tuple:
    """World points of the arch's object points (aperture-centre frame, y down) and its normal.

    ``facing_yaw_deg`` is the world yaw of the gate normal (object +z); ``lean_deg`` tilts the
    gate plane back about its own horizontal axis (0 = upright).
    """
    yaw = math.radians(facing_yaw_deg)
    lean = math.radians(lean_deg)
    n_h = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    normal = math.cos(lean) * n_h + math.sin(lean) * up
    down = -(math.cos(lean) * up - math.sin(lean) * n_h)
    right = np.cross(down, normal)  # x = y cross z for a right-handed object frame
    pts = np.asarray(center, np.float64) + np.outer(OBJ[:, 0], right) + np.outer(OBJ[:, 1], down)
    return pts, normal


def project(camera: CameraModel, pos, quat, world_pts) -> np.ndarray:
    p_cam = (np.asarray(world_pts) - np.asarray(pos)) @ camera.world_to_camera(quat).T
    px, front = camera.project(p_cam)
    assert camera.in_image(px, front).all(), "synthetic gate must be fully in view"
    return px


def state_of(pos, quat, received_at: float = 0.0) -> FlightState:
    return FlightState(
        pos=np.asarray(pos, np.float64),
        vel=np.zeros(3),
        quat=np.asarray(quat, np.float64),
        omega=np.zeros(3),
        received_at=received_at,
    )


class PoseOnly(TrackingGateEstimator):
    """The base tracker's ``detection_from_pose`` without a detector."""

    def detect(self, image, state):
        return []


def back_project(camera: CameraModel, pos, quat, px) -> GateDetection:
    t_cam, n_cam = solve_planar_pnp(OBJ, px, camera)
    est = PoseOnly(camera, WorldScale(1.0), default_aperture=ARCH.min_aperture)
    est._det_stamp = None  # current pose
    det = est.detection_from_pose(
        t_cam, n_cam, px, 0.01, state_of(pos, quat), camera, ARCH.min_aperture, cls=ARCH.cls
    )
    assert det is not None
    return det


# ----------------------------------------------------------------------------- AC5: round trip


@pytest.mark.parametrize("rng", [15.0, 40.0, 60.0])
@pytest.mark.parametrize("camera", [LIVE, OFF_PP], ids=["centred", "off_centre_pp"])
def test_synthetic_arch_back_projects_to_its_centre_including_z(camera, rng):
    # drone 6 m up, pitched nose-down 15 deg, rolled 8 deg, yawed 12 deg; gate 1.84 m up
    pos = np.array([3.0, -2.0, 6.0])
    quat = attitude(yaw=12.0, pitch=15.0, roll=8.0)
    bearing = math.radians(14.0)
    center = pos + np.array([rng * math.cos(bearing), rng * math.sin(bearing), 0.0])
    center[2] = 1.84
    pts, normal = gate_world_points(center, facing_yaw_deg=180.0 + 14.0 + 20.0)
    px = project(camera, pos, quat, pts)
    det = back_project(camera, pos, quat, px)
    np.testing.assert_allclose(det.center, center, atol=1e-3 * rng)
    assert abs(det.center[2] - 1.84) < 0.02
    assert det.distance == pytest.approx(np.linalg.norm(center - pos), rel=1e-4)
    # the world plane normal is carried on the detection (UC-91) and is the gate's (up to sign)
    assert det.n_world is not None
    assert abs(abs(float(np.dot(det.n_world, normal))) - 1.0) < 1e-4


def test_off_centre_principal_point_matters_when_ignored():
    """Back-projecting off-centre-pp pixels with a centred camera is visibly wrong (z moves)."""
    pos, quat = np.array([0.0, 0.0, 6.0]), attitude(pitch=15.0)
    center = np.array([40.0, 0.0, 1.84])
    px = project(OFF_PP, pos, quat, gate_world_points(center, 180.0)[0])
    right = back_project(OFF_PP, pos, quat, px)
    wrong = back_project(LIVE, pos, quat, px)
    assert abs(right.center[2] - 1.84) < 0.02
    # pp 42 px above centre at fx ~739 -> ~3.3 deg of elevation -> ~2.3 m at 40 m
    assert abs(wrong.center[2] - 1.84) > 1.5


def test_uptilt_sign_flip_shifts_z_by_twice_the_tilt():
    """A gate on the optical axis of a level drone: -uptilt puts it 2 r sin(t) lower."""
    t = 20.0
    pos, quat = np.array([0.0, 0.0, 2.0]), attitude()
    for r in (25.0, 40.0, 60.0):
        axis = np.array([math.cos(math.radians(t)), 0.0, math.sin(math.radians(t))])
        center = pos + r * axis
        px = project(LIVE, pos, quat, gate_world_points(center, 190.0)[0])
        good = back_project(LIVE, pos, quat, px)
        flipped = back_project(CameraModel(2560, 1080, 120.0, -t), pos, quat, px)
        np.testing.assert_allclose(good.center, center, atol=1e-3 * r)
        assert good.center[2] - flipped.center[2] == pytest.approx(
            2 * r * math.sin(math.radians(t)), rel=1e-3
        )


def test_one_degree_of_uptilt_error_is_about_0_66_m_at_38_m():
    pos, quat = np.array([0.0, 0.0, 2.0]), attitude()
    center = pos + 38.0 * np.array([math.cos(math.radians(20)), 0.0, math.sin(math.radians(20))])
    px = project(LIVE, pos, quat, gate_world_points(center, 190.0)[0])
    good = back_project(LIVE, pos, quat, px)
    off = back_project(CameraModel(2560, 1080, 120.0, 21.0), pos, quat, px)
    # the estimate swings 1 deg about the camera: 38 sin(1 deg) = 0.66 m, mostly vertical
    assert np.linalg.norm(off.center - good.center) == pytest.approx(0.66, abs=0.01)
    assert off.center[2] - good.center[2] == pytest.approx(0.62, abs=0.01)


# ----------------------------------------------------------------------------- principal point


def test_principal_point_defaults_to_the_image_centre():
    cam = camera_from_config({"resolution": [2560, 1080], "hfov_deg": 120.0, "uptilt_deg": 20.0})
    assert cam.principal_point is None
    assert (cam.cx, cam.cy) == (1280.0, 540.0)
    assert camera_from_config({"hfov_deg": 90.0}).principal_point is None  # 1920x1080 default
    # the shipped Liftoff config leaves it unset
    assert CFG.data["camera"].get("principal_point") is None


def test_principal_point_from_config_drives_intrinsics_and_rays():
    cam = camera_from_config(
        {
            "resolution": [2560, 1080],
            "hfov_deg": 120.0,
            "uptilt_deg": 20.0,
            "principal_point": [1300, 520.5],
        }
    )
    assert (cam.cx, cam.cy) == (1300.0, 520.5)
    k = cam.intrinsics()
    assert (k[0, 2], k[1, 2]) == (1300.0, 520.5)
    np.testing.assert_allclose(cam.pixel_rays(np.array([[1300.0, 520.5]])), [[0.0, 0.0, 1.0]])
    px, _ = cam.project(np.array([[0.0, 0.0, 5.0]]))
    np.testing.assert_allclose(px, [[1300.0, 520.5]])


@pytest.mark.parametrize(
    "pp, match",
    [
        ([3000, 500], "inside"),
        ([100, -1], "inside"),
        ([100], "principal_point"),
        ("centre", "principal_point"),
    ],
)
def test_principal_point_validation(pp, match):
    with pytest.raises(HarnessConfigError, match=match):
        camera_from_config({"resolution": [2560, 1080], "hfov_deg": 120.0, "principal_point": pp})


def test_scaled_to_scales_the_principal_point_proportionally():
    s = OFF_PP.scaled_to(1280, 540)
    assert s.principal_point == pytest.approx((1347.0 / 2, 498.0 / 2))
    assert (s.cx, s.cy) == pytest.approx((673.5, 249.0))
    assert (s.hfov_deg, s.uptilt_deg) == (OFF_PP.hfov_deg, OFF_PP.uptilt_deg)
    c = LIVE.scaled_to(1280, 540)
    assert c.principal_point is None and (c.cx, c.cy) == (640.0, 270.0)
    # positional construction (the pre-UC-91 signature) still works
    assert CameraModel(640, 270, 120.0, 20.0).principal_point is None


def test_gate_detection_n_world_is_an_optional_trailing_field():
    z = np.zeros(3)
    d = GateDetection(np.zeros((4, 2)), 0.1, z, z, z, 0.0, 1.0, 5.0, 0.0)
    assert d.n_world is None and d.cls is None


def _align_inputs():
    from drone_fly.harness.offline import align

    class Fr:
        stamp = 0.0

    class Track:
        def at(self, t):
            return (np.zeros(3), np.array([1.0, 0, 0, 0]))

    return align, [Fr()], Track()


def test_align_search_and_camera_check_preserve_the_principal_point(monkeypatch):
    align, frames, track = _align_inputs()
    seen: list[CameraModel] = []

    def score(frames, poses, cam, masks, edges):
        seen.append(cam)
        return 1.0, 99

    monkeypatch.setattr(align, "score_pose_set", score)
    monkeypatch.setattr(align, "baseline_score", lambda *a, **k: 0.5)
    align.search(
        frames,
        track,
        OFF_PP,
        None,
        [],
        offsets_s=np.array([0.0]),
        uptilts_deg=np.array([18.0, 22.0]),
    )
    align.camera_check(frames, track, OFF_PP, None, [], 0.0)
    assert len(seen) > 2
    assert {c.principal_point for c in seen} == {OFF_PP.principal_point}
    assert {c.uptilt_deg for c in seen} >= {18.0, 22.0}
    assert len({c.hfov_deg for c in seen}) > 1  # camera_check varies hfov too


# ----------------------------------------------------------------------------- size range


def test_size_range_is_exact_for_a_fronto_parallel_gate_anywhere_in_the_lens():
    """The tangent-plane projection makes it exact off-axis too (120 deg lens)."""
    quat = attitude()
    for cam in (LIVE, OFF_PP):
        for r, az, el in ((20.0, 0.0, 20.0), (50.0, 40.0, 20.0), (35.0, -45.0, 5.0)):
            a, e = math.radians(az), math.radians(el)
            los = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
            center = r * los
            # gate facing the line of sight: rotate the plane so its normal is -los
            pts = _facing(center, -los)
            px = project(cam, np.zeros(3), quat, pts)
            assert size_range(OBJ, px, cam) == pytest.approx(r, rel=0.01)


def _facing(center, normal) -> np.ndarray:
    n = np.asarray(normal, np.float64) / np.linalg.norm(normal)
    right = np.cross(n, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(n, right)
    return np.asarray(center) + np.outer(OBJ[:, 0], right) + np.outer(OBJ[:, 1], down)


def test_size_range_is_independent_of_the_principal_point():
    pos, quat = np.zeros(3), attitude()
    pts = gate_world_points([45.0, 5.0, 10.0], 180.0)[0]
    a = size_range(OBJ, project(LIVE, pos, quat, pts), LIVE)
    b = size_range(OBJ, project(OFF_PP, pos, quat, pts), OFF_PP)
    assert a == pytest.approx(b, rel=1e-9)


@pytest.mark.parametrize("phi", [30.0, 45.0, 60.0])
def test_oblique_gate_reads_farther_by_one_over_sqrt_cos_phi(phi):
    """The PnP / size ratio is sqrt(cos phi) <= 1, so foreshortening lives in the lower bound."""
    pos, quat = np.array([0.0, 0.0, 1.84]), attitude()
    center = np.array([50.0, 0.0, 1.84])
    pts = gate_world_points(center, 180.0 + phi)[0]
    px = project(LIVE, pos, quat, pts)
    r_size = size_range(OBJ, px, LIVE)
    t_cam, _ = solve_planar_pnp(OBJ, px, LIVE)
    ratio = float(np.linalg.norm(t_cam)) / r_size
    assert ratio == pytest.approx(math.sqrt(math.cos(math.radians(phi))), rel=0.02)
    # a correct oblique pose at 60 deg (0.71) sits inside the shipped band
    rc = RangeCheck.from_config(CFG.data["gates"]["learned"]["range_check"])
    assert rc.size_ratio_min <= ratio <= rc.size_ratio_max


def test_size_range_degenerate_hull_is_infinite():
    two = np.array([[500.0, 400.0], [520.0, 400.0], [500.0, 400.0], [520.0, 400.0]])
    assert size_range(OBJ[:4], two, LIVE) == math.inf
    assert gate_range.hull_area(np.array([[0, 0], [1, 1]])) == 0.0
    assert gate_range.hull_area(np.array([[0, 0], [2, 0], [2, 3], [0, 3], [1, 1]])) == 6.0


# ----------------------------------------------------------------------------- truncation fallback


def _arch_px(cam=LIVE, center=(40.0, 0.0, 1.84)):
    return project(cam, np.array([0.0, 0.0, 1.84]), attitude(), gate_world_points(center, 180.0)[0])


def test_size_range_valid_for_a_whole_gate():
    px = _arch_px()
    assert size_range_valid(px, OBJ, OBJ, 2560, 1080, RangeCheck()) == (True, None)


def test_size_range_invalid_for_few_points():
    px = _arch_px()
    assert size_range_valid(px[:3], OBJ[:3], OBJ, 2560, 1080, RangeCheck()) == (
        False,
        "few_points",
    )


def test_size_range_invalid_at_the_image_edge():
    px = _arch_px().copy()
    px[0] = [5.0, 600.0]  # within 8 px of the left border
    assert size_range_valid(px, OBJ, OBJ, 2560, 1080, RangeCheck()) == (False, "image_edge")
    px[0] = [2552.0, 600.0]  # within 8 px of the right border (w - 1 - m = 2551)
    assert size_range_valid(px, OBJ, OBJ, 2560, 1080, RangeCheck()) == (False, "image_edge")
    # with no margin the same point is fine
    assert size_range_valid(px, OBJ, OBJ, 2560, 1080, RangeCheck(edge_margin_px=0.0))[0]


def test_size_range_invalid_next_to_a_mask():
    px = _arch_px()
    u, v = px[8]  # the top keypoint
    near = FrameMasks(
        [MaskRegion("hud", (u - 50, v - 30, 100, 25))], resolution=(2560, 1080)
    )  # ends 5 px above the keypoint
    assert size_range_valid(px, OBJ, OBJ, 2560, 1080, RangeCheck(), near) == (False, "mask_edge")
    far = FrameMasks([MaskRegion("hud", (0, 0, 50, 50))], resolution=(2560, 1080))
    assert size_range_valid(px, OBJ, OBJ, 2560, 1080, RangeCheck(), far) == (True, None)


def test_size_range_invalid_for_a_small_kept_hull():
    px = _arch_px()
    top = [6, 7, 8, 9]  # the arch crown only
    assert size_range_valid(px[top], OBJ[top], OBJ, 2560, 1080, RangeCheck()) == (
        False,
        "small_hull",
    )
    assert size_range_valid(px[top], OBJ[top], OBJ, 2560, 1080, RangeCheck(min_hull_frac=0.0)) == (
        True,
        None,
    )


def test_truncated_detection_falls_back_to_the_tilt_check_only():
    """With an invalid size range a ratio that would fail is ignored; the tilt still applies."""
    px = _arch_px().copy()
    t_cam, n_cam = solve_planar_pnp(OBJ, px, LIVE)
    upright = np.array([1.0, 0.0, 0.0])
    strict = RangeCheck(size_ratio_min=1.2, size_ratio_max=1.5)  # rejects a correct frontal pose
    whole = evaluate_detection(t_cam, upright, px, OBJ, OBJ, LIVE, strict)
    assert (whole["accepted"], whole["reason"]) == (False, "size_ratio")
    assert whole["size_invalid"] is None and whole["r_size"] is not None
    px[0] = [3.0, px[0, 1]]  # truncated at the border
    cut = evaluate_detection(t_cam, upright, px, OBJ, OBJ, LIVE, strict)
    assert cut["accepted"] and cut["reason"] is None
    assert cut["size_invalid"] == "image_edge" and cut["r_size"] is None
    assert cut["r_size_raw"] is not None  # still reported for the evaluator's column
    leaning = np.array([0.5, 0.0, 0.866])  # 60 deg out of upright
    cut = evaluate_detection(t_cam, leaning, px, OBJ, OBJ, LIVE, strict)
    assert (cut["accepted"], cut["reason"]) == (False, "normal_tilt")


# ----------------------------------------------------------------------------- check_detection


def test_ratio_band_is_inclusive_and_rejects_outside():
    rc = RangeCheck(size_ratio_min=0.5, size_ratio_max=1.25, max_normal_tilt_deg=None)
    up = np.array([1.0, 0.0, 0.0])
    assert check_detection(50.0, 100.0, up, rc) == (True, None)
    assert check_detection(125.0, 100.0, up, rc) == (True, None)
    assert check_detection(49.0, 100.0, up, rc) == (False, "size_ratio")
    assert check_detection(126.0, 100.0, up, rc) == (False, "size_ratio")
    # an invalid / non-finite size range skips the ratio check
    assert check_detection(500.0, None, up, rc) == (True, None)
    assert check_detection(500.0, math.inf, up, rc) == (True, None)
    assert check_detection(500.0, 0.0, up, rc) == (True, None)


def test_tilt_check_and_its_null_switch():
    rc = RangeCheck(max_normal_tilt_deg=30.0)

    def tilt(deg: float) -> np.ndarray:
        return np.array([math.cos(math.radians(deg)), 0.0, math.sin(math.radians(deg))])

    assert normal_tilt_deg(tilt(29.0)) == pytest.approx(29.0)
    assert normal_tilt_deg(-tilt(29.0)) == pytest.approx(29.0)  # sign-agnostic
    assert check_detection(40.0, None, tilt(29.0), rc) == (True, None)
    assert check_detection(40.0, None, tilt(31.0), rc) == (False, "normal_tilt")
    assert check_detection(40.0, 40.0, tilt(80.0), RangeCheck(max_normal_tilt_deg=None)) == (
        True,
        None,
    )
    # the ratio is checked first
    assert check_detection(10.0, 40.0, tilt(80.0), rc) == (False, "size_ratio")


def test_ippe_wrong_branch_normal_is_rejected_and_the_right_one_accepted():
    """Seen steeply from above, IPPE's two planar solutions differ in normal tilt."""
    pos = np.array([0.0, 0.0, 22.0])
    # nose down so the 20 deg uptilt camera looks ~25 deg down; slightly yawed gate and roll
    # (an exactly symmetric pose makes OpenCV's IPPE return a zero rvec)
    quat = attitude(pitch=45.0, roll=2.0)
    center = np.array([28.0, 0.0, 1.84])
    pts, normal = gate_world_points(center, 183.0)
    px = project(LIVE, pos, quat, pts)
    n, rvecs, _, _ = cv2.solvePnPGeneric(
        OBJ, px.reshape(-1, 1, 2), LIVE.intrinsics(), None, flags=cv2.SOLVEPNP_IPPE
    )
    assert n == 2
    r_wc = (LIVE.world_to_camera(quat)).T
    verdicts = []
    for rv in rvecs:
        rot, _ = cv2.Rodrigues(rv)
        n_world = r_wc @ rot[:, 2]
        right = abs(abs(float(np.dot(n_world, normal))) - 1.0) < 1e-3
        verdicts.append((right, n_world))
    assert sorted(v[0] for v in verdicts) == [False, True]
    shipped = RangeCheck.from_config(CFG.data["gates"]["learned"]["range_check"])
    for rc in (RangeCheck(), shipped):
        for right, n_world in verdicts:
            got = check_detection(30.0, None, n_world, rc)
            assert got == ((True, None) if right else (False, "normal_tilt"))


# ----------------------------------------------------------------------------- config


def test_range_check_defaults_and_shipped_config():
    d = RangeCheck.from_config(None)
    assert d == RangeCheck() and d.max_normal_tilt_deg == 30.0
    lc = CFG.data["gates"]["learned"]
    assert lc["range_mode"] == "pnp"
    shipped = RangeCheck.from_config(lc["range_check"])
    assert shipped.max_normal_tilt_deg == 45.0
    assert (shipped.size_ratio_min, shipped.size_ratio_max) == (0.47, 1.33)
    assert (shipped.edge_margin_px, shipped.min_hull_frac) == (8.0, 0.5)
    assert shipped.size_ratio_max > 1.0 > shipped.size_ratio_min  # foreshortening below 1


def test_range_check_passthrough_and_null_tilt():
    rc = RangeCheck.from_config(
        {
            "size_ratio_min": 0.6,
            "size_ratio_max": 1.2,
            "max_normal_tilt_deg": None,
            "edge_margin_px": 3,
            "min_hull_frac": 0.25,
        }
    )
    assert rc == RangeCheck(0.6, 1.2, None, 3.0, 0.25)
    assert rc.as_dict()["max_normal_tilt_deg"] is None


@pytest.mark.parametrize(
    "cfg, match",
    [
        ({"size_ratio": 1.0}, "unknown"),
        ({"size_ratio_min": 1.5, "size_ratio_max": 1.2}, "size_ratio_min < size_ratio_max"),
        ({"size_ratio_min": 0.0}, "size_ratio_min"),
        ({"size_ratio_max": -1.0}, "size_ratio_max"),
        ({"max_normal_tilt_deg": 120.0}, "max_normal_tilt_deg"),
        ({"max_normal_tilt_deg": "steep"}, "max_normal_tilt_deg"),
        ({"edge_margin_px": -1.0}, "edge_margin_px"),
        ({"min_hull_frac": 1.5}, "min_hull_frac"),
    ],
)
def test_range_check_validation(cfg, match):
    with pytest.raises(HarnessConfigError, match=match):
        RangeCheck.from_config(cfg)


def test_range_mode_choices():
    assert range_mode_from_config("pnp") == "pnp"
    assert range_mode_from_config("pnp_checked") == "pnp_checked"
    for bad in ("size", "PNP", None, 1):
        with pytest.raises(HarnessConfigError, match="range_mode"):
            range_mode_from_config(bad)


# ----------------------------------------------------------------------------- matching


def test_match_by_bearing_clear_unmatched_and_ambiguous():
    pos = np.zeros(3)

    def deg(a: float, r: float = 40.0) -> np.ndarray:
        return np.array([r * math.cos(math.radians(a)), r * math.sin(math.radians(a)), 0.0])

    centers = np.array([deg(0.0), deg(30.0)])
    best, ang, amb = gbe.match_by_bearing(pos, deg(1.0, 38.0), centers)
    assert best == 0 and ang == pytest.approx(1.0) and not amb
    best, ang, amb = gbe.match_by_bearing(pos, deg(12.0), centers)
    assert best is None and ang == pytest.approx(12.0) and not amb
    # a second passage within the tolerance (e.g. a gate behind the matched one)
    behind = np.array([deg(0.0), deg(4.0, 70.0), deg(30.0)])
    best, _, amb = gbe.match_by_bearing(pos, deg(0.5), behind)
    assert best == 0 and amb
    # a second passage outside the tolerance but within the margin of the best angle
    margin = np.array([deg(0.0), deg(10.0)])
    best, ang, amb = gbe.match_by_bearing(pos, deg(4.0), margin)  # 4 vs 6 deg: 6 <= 4 + 3
    assert best == 0 and amb
    best, ang, amb = gbe.match_by_bearing(pos, deg(1.0), margin)  # 1 vs 9 deg
    assert best == 0 and not amb


def test_error_components_split_z_radial_and_tangential():
    pos, truth = np.zeros(3), np.array([40.0, 0.0, 0.0])
    e = gbe.error_components(pos, truth + [2.0, 0.0, 0.0], truth)
    assert e["radial_m"] == pytest.approx(2.0) and e["z_m"] == pytest.approx(0.0)
    e = gbe.error_components(pos, truth + [0.0, 0.0, -1.5], truth)
    assert e["z_m"] == pytest.approx(-1.5) and e["tan_v_m"] == pytest.approx(-1.5)
    assert e["radial_m"] == pytest.approx(0.0, abs=1e-12) and e["range_m"] == 40.0
    assert e["elev_err_deg"] == pytest.approx(math.degrees(math.atan2(-1.5, 40.0)))


def test_bin_labels_cover_the_plan_bins():
    assert gbe.bin_label(3.0) == "0-4"
    assert gbe.bin_label(39.9) == "25-40"
    assert gbe.bin_label(40.0) == "40-60"
    assert gbe.bin_label(75.0) == ">60"
    assert gbe.BIN_LABELS[-2:] == ("40-60", ">60")


# ----------------------------------------------------------------------------- telemetry pose


def test_telemetry_pose_lerps_position_and_slerps_attitude():
    t = np.array([0.0, 1.0])
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    quat = np.array([attitude(yaw=0.0), attitude(yaw=90.0)])
    tp = gbe.TelemetryPose(t, pos, quat)
    p, q = tp.at(0.5)
    np.testing.assert_allclose(p, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(q, attitude(yaw=45.0), atol=1e-9)
    p, q = tp.at(0.25)
    np.testing.assert_allclose(q, attitude(yaw=22.5), atol=1e-9)
    np.testing.assert_allclose(tp.at(1.0)[0], pos[1])
    np.testing.assert_allclose(tp.at(0.0)[0], pos[0])


def test_telemetry_pose_outside_the_span_is_none():
    tp = gbe.TelemetryPose(np.array([1.0, 2.0]), np.zeros((2, 3)), np.tile([1.0, 0, 0, 0], (2, 1)))
    assert tp.at(0.999) is None and tp.at(2.001) is None
    one = gbe.TelemetryPose(np.array([1.0]), np.zeros((1, 3)), np.array([[1.0, 0, 0, 0]]))
    assert one.at(1.0) is None


# ----------------------------------------------------------------------------- synthetic runs

RUN_CAM = {
    "resolution": [640, 270],
    "hfov_deg": 120.0,
    "uptilt_deg": 20.0,
    "principal_point": [330.0, 128.0],
    "masks": [],
}
RUN_CAMERA = camera_from_config(RUN_CAM)
GATE = np.array([45.0, 0.0, 1.84])
DECOY = np.array([10.0, 60.0, 1.84])  # another passage, far off the line of sight
DT = 0.05


def drone_pose(t: float) -> tuple[np.ndarray, np.ndarray]:
    """A slow level approach towards the gate, nose slightly down."""
    return np.array([2.0 * t, 0.3 * t, 6.0]), attitude(yaw=2.0 * t, pitch=5.0)


def write_run(
    d: Path,
    n_ticks: int = 20,
    *,
    partial_at: int | None = 3,
    late_at: int | None = 5,
    noise_px: float = 0.0,
    seed: int = 0,
) -> Path:
    """config.yaml + ticks.jsonl + telemetry.csv + frames.csv of a synthetic recorded run.

    Every tick logs one full-keypoint detection of ``GATE`` (frame ``k``, stamp ``k * DT``) and
    re-logs it on the next tick (the async cache), so de-duplication is exercised; tick
    ``partial_at`` adds a 6-point partial detection and tick ``late_at`` one whose source frame
    has no stamp (lag beyond the run start).
    """
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True)
    cfg = {
        "camera": RUN_CAM,
        "gates": {"pose_offset_s": 0.0, "lost_policy": "reacquire", "learned": {}},
        "gate_types": CFG.data["gate_types"],
        "scaling": {"world_scale": 1.0},
    }
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with (d / "telemetry.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["received_at", "pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
        )
        for k in range(2 * n_ticks + 3):  # telemetry at twice the frame rate
            t = k * DT / 2
            p, q = drone_pose(t)
            w.writerow([t, *contract_to_unity_position(p), *contract_to_unity_quat(q)])
    pts = gate_world_points(GATE, 186.0)[0]
    lines = []
    prev = None
    for k in range(n_ticks):
        t = k * DT
        p, q = drone_pose(t)
        px = project(RUN_CAMERA, p, q, pts) + rng.normal(0.0, noise_px, (len(pts), 2))
        det = {"cls": "inflatable_arch", "points": px.tolist(), "center": px.mean(0).tolist()}
        dets = [det]
        info = {"frame_index": k, "frame_size": [640, 270], "target_valid": True}
        if prev is not None and k % 2:  # an async tick: the previous frame's detection again
            dets, info["detection_lag_frames"] = prev, 1
        if k == partial_at:
            dets = dets + [
                {
                    "cls": "inflatable_arch",
                    "points": px[4:].tolist(),
                    "center": px[4:].mean(0).tolist(),
                }
            ]
        if k == late_at:
            info["detection_lag_frames"] = 99
        info["detections_px"] = dets
        prev = [det]
        lines.append(
            {
                "tick": k,
                "sim_time": t,
                "ages_s": {"telemetry": 0.0, "frame": 0.0},
                "state": {
                    "pos": p.tolist(),
                    "vel": [0.0, 0.0, 0.0],
                    "quat": q.tolist(),
                    "omega": [0.0, 0.0, 0.0],
                },
                "gate": {"stamp": t, "target": GATE.tolist(), "status": "tracking", "info": info},
                "events": [],
            }
        )
    (d / "ticks.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", "utf-8")
    with (d / "frames.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tick", "stamp", "file"])
        w.writerow([0, 0.0, "frames/000000.png"])
    return d


def truth() -> gbe.Truth:
    return gbe.Truth(
        centers=np.array([GATE, DECOY]),
        normals=np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        item_ids=("g0", "g1"),
        track="synthetic.xml",
    )


def test_load_run_reads_the_four_files(tmp_path):
    run = gbe.load_run(write_run(tmp_path / "r"))
    assert run.name == "r" and len(run.ticks) == 20 and len(run.frames) == 1
    assert run.world_scale == 1.0
    p, q = drone_pose(0.5)
    i = int(np.argmin(np.abs(run.tel_t - 0.5)))
    np.testing.assert_allclose(run.tel_pos[i], p, atol=1e-5)
    assert abs(abs(float(np.dot(run.tel_quat[i], q))) - 1.0) < 1e-6
    with pytest.raises(HarnessConfigError, match="ticks.jsonl"):
        (tmp_path / "r" / "ticks.jsonl").unlink()
        gbe.load_run(tmp_path / "r")


def test_logged_detections_deduplicate_and_count_missing_stamps(tmp_path):
    run = gbe.load_run(write_run(tmp_path / "r"))
    dets, counts = gbe.logged_detections(run)
    keys = [(d["src_index"], d["order"]) for d in dets]
    assert len(keys) == len(set(keys))
    # even ticks log their own frame, odd ticks re-log the previous frame's detection (lag 1);
    # the tick-3 partial is (frame 2, order 1); tick 5's lag-99 source frame has no stamp
    assert counts["no_source_stamp"] == 1
    assert sorted({k[0] for k in keys if k[0] >= 0}) == list(range(0, 20, 2))
    assert (2, 1) in keys and len(dets) == 11
    for d in dets:
        assert d["src_stamp"] == pytest.approx(d["src_index"] * DT)


def test_pass_a_resolves_full_detections_and_counts_partials(tmp_path):
    run = gbe.load_run(write_run(tmp_path / "r"))
    recs, counts = gbe.pass_a(run, truth(), RangeCheck())
    assert counts["partial_not_resolvable"] == 1
    assert counts["no_source_stamp"] == 1
    assert counts["resolved"] == len(recs) == 10
    for r in recs:
        assert r["category"] == "matched" and r["match"] == 0 and r["n_points"] == 10
        assert abs(r["z_m"]) < 0.05 and r["err3d_m"] < 0.2
        assert r["bin"] == "40-60" or r["bin"] == "25-40"
        assert r["r_size"] is not None  # a whole gate: the size column is filled
        assert abs(r["size_err_m"]) < 0.05 * r["range_m"]


def test_pass_a_skips_stamps_outside_the_telemetry_span(tmp_path):
    d = write_run(tmp_path / "r")
    rows = (d / "telemetry.csv").read_text("utf-8").splitlines()
    head, body = rows[0], rows[1:]
    keep = [r for r in body if float(r.split(",")[0]) >= 0.3]  # drop the first 0.3 s
    (d / "telemetry.csv").write_text("\n".join([head, *keep]) + "\n", "utf-8")
    recs, counts = gbe.pass_a(gbe.load_run(d), truth(), RangeCheck())
    assert counts["outside_telemetry"] == 3  # frames 0, 2, 4 at 0.0 / 0.1 / 0.2 s
    assert counts["outside_telemetry"] + counts["resolved"] == 10


def test_solve_detection_skips_non_finite_pnp(monkeypatch):
    from drone_fly.harness.liftoff import gate_cv

    px = _arch_px()
    pose = (np.zeros(3), np.array([1.0, 0, 0, 0]))
    ok = gbe.solve_detection(px, OBJ, OBJ, LIVE, pose, RangeCheck(), None)
    assert ok is not None and ok["r_pnp"] > 0
    monkeypatch.setattr(
        gate_cv, "solve_planar_pnp", lambda *a: (np.array([np.nan, 0.0, 1.0]), np.ones(3))
    )
    assert gbe.solve_detection(px, OBJ, OBJ, LIVE, pose, RangeCheck(), None) is None
    monkeypatch.setattr(
        gate_cv, "solve_planar_pnp", lambda *a: (np.array([0, 0, -5.0]), np.ones(3))
    )
    assert gbe.solve_detection(px, OBJ, OBJ, LIVE, pose, RangeCheck(), None) is None


def test_pass_b_replay_is_deterministic(tmp_path):
    run = gbe.load_run(write_run(tmp_path / "r", noise_px=0.3, seed=4))
    a = gbe.pass_b(run, gbe.tick_inputs(run, RangeCheck()), truth(), "pnp", RangeCheck())
    b = gbe.pass_b(run, gbe.tick_inputs(run, RangeCheck()), truth(), "pnp", RangeCheck())
    assert [r["valid"] for r in a] == [r["valid"] for r in b]
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x["target"], y["target"])


def test_pass_b_reproduces_the_logged_target_and_compares_modes_on_identical_inputs(tmp_path):
    run = gbe.load_run(write_run(tmp_path / "r"))  # noise-free: logged target = the truth
    inputs = gbe.tick_inputs(run, RangeCheck())
    a = gbe.pass_b(run, inputs, truth(), "pnp", RangeCheck())
    assert all(r["valid"] for r in a)
    f = gbe.fidelity(a)
    assert f["held_both"] == 20 and f["held_agreement"] == 1.0
    assert f["share_within_m"] >= 0.8 and not f["indicative_only"]
    assert f["median_dev_m"] < gbe.FIDELITY_TOL_M
    # both modes get identical inputs; a permissive check keeps the same targets
    loose = RangeCheck(size_ratio_min=0.01, size_ratio_max=100.0, max_normal_tilt_deg=None)
    c = gbe.pass_b(run, inputs, truth(), "pnp_checked", loose)
    for x, y in zip(a, c, strict=True):
        np.testing.assert_allclose(x["target"], y["target"])
    # a check that rejects everything never acquires
    never = RangeCheck(size_ratio_min=50.0, size_ratio_max=60.0)
    none = gbe.pass_b(run, inputs, truth(), "pnp_checked", never)
    assert not any(r["valid"] for r in none)


def test_fidelity_share_median_and_indicative_flag():
    def row(dev, valid=True, logged=True):
        return {
            "valid": valid,
            "logged_valid": logged,
            "target": np.array([dev, 0.0, 0.0]),
            "logged_target": np.zeros(3),
        }

    good = [row(0.1)] * 9 + [row(2.0)]
    f = gbe.fidelity(good)
    assert f["share_within_m"] == pytest.approx(0.9) and not f["indicative_only"]
    assert f["median_dev_m"] == pytest.approx(0.1)
    bad = [row(0.1)] * 5 + [row(2.0)] * 5 + [row(0.0, valid=False)]
    f = gbe.fidelity(bad)
    assert f["share_within_m"] == pytest.approx(0.5) and f["indicative_only"]
    assert f["held_both"] == 10 and f["held_agreement"] == pytest.approx(10 / 11)
    assert gbe.fidelity([])["indicative_only"]


# ----------------------------------------------------------------------------- rule + retention


def rec(range_m: float, z: float, *, ratio: float = 1.0, tilt: float = 0.0, cat="matched"):
    n = np.array([math.cos(math.radians(tilt)), 0.0, math.sin(math.radians(tilt))])
    return {
        "category": cat,
        "bin": gbe.bin_label(range_m),
        "range_m": range_m,
        "z_m": z,
        "radial_m": 0.0,
        "tan_h_m": 0.0,
        "tan_v_m": z,
        "err3d_m": abs(z),
        "elev_err_deg": 0.0,
        "r_pnp": range_m * ratio,
        "r_size": range_m,
        "n_world": n,
    }


def _records(n_far=90, n_out=10, out_ratio=0.2, near_z=0.5):
    """Far (50 m) inliers with |z| <= 1, far outliers (|z| 25, failing the ratio), near ones.

    The rule needs both a 30 % p90 drop and >= 90 % retention, so the cut outliers must be
    (just) the top 10 %: 10 of 100 puts the pre-cut p90 onto them.
    """
    far = [rec(50.0, float(z)) for z in np.linspace(-1.0, 1.0, n_far)]
    outliers = [rec(50.0, 25.0 * (-1) ** i, ratio=out_ratio) for i in range(n_out)]
    near = [rec(30.0, near_z + float(z)) for z in np.linspace(-0.3, 0.3, 40)]
    return far + outliers + near


def test_rule_passes_when_outliers_are_cut_and_retention_holds():
    v = gbe.rule_verdict(_records(), RangeCheck())
    assert v["pass"], v["reasons"]
    assert v["far"]["p90_drop"] >= 0.3 and v["far"]["sd_drop"] >= 0.3
    assert v["far"]["n_pnp"] == 100 and v["far"]["n_checked"] == 90
    assert {b["bin"]: b["kept_share"] for b in v["bins"]}["40-60"] == pytest.approx(0.9)


def test_rule_fails_on_retention_below_90_percent():
    v = gbe.rule_verdict(_records(n_out=20), RangeCheck())
    assert not v["pass"]
    assert any("40-60 m: retention" in r for r in v["reasons"])


def test_rule_fails_when_rejections_do_not_reduce_noise():
    recs = _records(n_out=0)
    recs += [rec(50.0, 0.0, ratio=0.2) for _ in range(5)]  # rejected but harmless
    v = gbe.rule_verdict(recs, RangeCheck())
    assert not v["pass"]
    assert any("p90 |z| drop" in r for r in v["reasons"])
    assert any("z sd drop" in r for r in v["reasons"])


def test_rule_fails_on_near_median_bias_and_thin_far_set():
    v = gbe.rule_verdict(_records(near_z=2.6), RangeCheck())
    assert not v["pass"] and any("25-40 m: |median z|" in r for r in v["reasons"])
    v = gbe.rule_verdict(_records(n_far=5, n_out=1), RangeCheck())
    assert not v["pass"] and any("only 6 matched" in r for r in v["reasons"])


def test_retention_counts_per_category_and_reason():
    recs = [
        rec(50.0, 0.0),
        rec(50.0, 0.0, ratio=0.2),
        rec(50.0, 0.0, tilt=60.0),
        rec(50.0, 0.0, ratio=3.0, cat="unmatched"),
        rec(30.0, 0.0, tilt=60.0, cat="ambiguous"),
    ]
    rc = RangeCheck()
    r = gbe.retention(recs, "pnp_checked", rc)
    far = r["40-60"]
    assert far["matched"] == {"kept": 1, "rejected": 2, "kept_share": pytest.approx(1 / 3)}
    assert far["unmatched"]["rejected"] == 1
    assert far["reasons"] == {"size_ratio": 2, "normal_tilt": 1}
    assert r["25-40"]["ambiguous"]["rejected"] == 1
    assert r["all"]["reasons"] == {"size_ratio": 2, "normal_tilt": 2}
    assert gbe.retention(recs, "pnp", rc)["all"]["matched"]["rejected"] == 0
    # ambiguous / unmatched records never enter the error statistics
    stats = gbe.binned_stats(recs)
    assert stats["40-60"]["n"] == 3 and stats["25-40"]["n"] == 0
    assert stats["25-40"]["ambiguous"] == 1


def test_tune_range_check_prefers_the_physics_start_among_equal_passing_cells():
    t = gbe.tune_range_check(_records())
    assert t["chosen"] == gbe.PHYSICS_START
    assert t["how"].startswith("passes the rule")
    assert len(t["grid"]) == 6 * 6 * 5


# ----------------------------------------------------------------------------- driver + hygiene


def test_a_run_in_both_tune_and_held_out_sets_is_an_error(tmp_path):
    d = write_run(tmp_path / "r")
    with pytest.raises(HarnessConfigError, match="both the tune and held-out"):
        gbe.run_backprojection_eval([d], "unused.xml", tune_runs=[tmp_path / "." / "r"])
    with pytest.raises(HarnessConfigError, match="unknown range mode"):
        gbe.run_backprojection_eval([d], "unused.xml", modes=("pnp", "size"))


def _forbid_listing(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the evaluator must never list or walk a run folder")

    for name in ("glob", "rglob", "iterdir", "walk"):
        if hasattr(Path, name):
            monkeypatch.setattr(Path, name, boom)
    for name in ("walk", "listdir", "scandir"):
        monkeypatch.setattr(os, name, boom)


def test_end_to_end_report_never_lists_and_opens_only_the_run_files(tmp_path, monkeypatch):
    tune = [write_run(tmp_path / f"t{i}", noise_px=0.4, seed=i) for i in range(2)]
    held = [write_run(tmp_path / f"h{i}") for i in range(2)]  # noise-free: fidelity 100 %
    monkeypatch.setattr(gbe, "truth_passages", lambda track, root=None: truth())
    # warm the lazy imports (cv2, tracker, yaml) before the listing guard goes up
    gbe.pass_a(gbe.load_run(held[0]), truth(), RangeCheck())

    opened: list[Path] = []
    real_open, real_read = Path.open, Path.read_text

    def spy_open(self, *a, **k):
        opened.append(self)
        return real_open(self, *a, **k)

    def spy_read(self, *a, **k):
        opened.append(self)
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "open", spy_open)
    monkeypatch.setattr(Path, "read_text", spy_read)
    _forbid_listing(monkeypatch)
    out = tmp_path / "out"
    rep = gbe.run_backprojection_eval(
        held, "synthetic.xml", tune_runs=tune, out_dir=out, invocation="drone-fly test"
    )
    run_dirs = {p.resolve() for p in tune + held}
    for p in opened:
        if p.resolve().parent in run_dirs:
            assert p.name in ("config.yaml", "ticks.jsonl", "telemetry.csv", "frames.csv")
    assert {p.name for p in opened} >= {"config.yaml", "ticks.jsonl", "telemetry.csv"}

    monkeypatch.undo()
    d = Path(rep["out_dir"])
    assert d.parent == out
    md = (d / "report.md").read_text("utf-8")
    js = json.loads((d / "report.json").read_text("utf-8"))
    assert "drone-fly test" in md and "Held-out runs: h0, h1" in md and "Tune runs: t0, t1" in md
    assert "| knob | physics start | final |" in md
    assert js["held_out"]["runs"] == ["h0", "h1"] and js["tune"]["runs"] == ["t0", "t1"]
    assert "rule" in js["held_out"] and "tuning" in js
    assert js["range_check"] == rep["range_check"].as_dict()
    a = js["held_out"]["pass_a"]
    assert a["categories"] == {"matched": 20}
    assert a["counts"]["h0"]["partial_not_resolvable"] == 1
    assert js["held_out"]["pass_b"]["fidelity_pooled"]["indicative_only"] is False
    assert "held-out runs: h0, h1" in gbe.summary(rep)


def test_report_without_tuning_uses_the_given_check(tmp_path, monkeypatch):
    held = [write_run(tmp_path / "h")]
    monkeypatch.setattr(gbe, "truth_passages", lambda track, root=None: truth())
    rc = RangeCheck(max_normal_tilt_deg=45.0)
    rep = gbe.run_backprojection_eval(held, "x.xml", modes=("pnp",), check=rc)
    assert rep["range_check"] == rc and "tuning" not in rep and "out_dir" not in rep
    assert "rule" not in rep["held_out"]  # the rule needs both modes


def test_cli_parser_has_the_backprojection_eval_subcommand():
    from drone_fly.cli.main import _cmd_gates_backprojection_eval, build_parser

    a = build_parser().parse_args(
        [
            "harness",
            "gates",
            "backprojection-eval",
            "--runs",
            "a",
            "b",
            "--track",
            "t.xml",
            "--tune-runs",
            "c",
            "--modes",
            "pnp",
            "--device",
            "cpu",
        ]
    )
    assert a.func is _cmd_gates_backprojection_eval
    assert a.runs == ["a", "b"] and a.track == "t.xml" and a.tune_runs == ["c"]
    assert a.modes == "pnp" and a.device == "cpu" and not a.with_frames
    d = build_parser().parse_args(
        ["harness", "gates", "backprojection-eval", "--runs", "a", "--track", "t.xml"]
    )
    assert d.modes == "pnp,pnp_checked" and d.out.endswith("uc91-backprojection")
    assert d.tune_runs is None
