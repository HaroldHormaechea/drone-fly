"""UC-64 AC9: depth calibration fitted on a capture against XML geometry + raycast scoring.

Ground-truth geometry (arch tubes, barrier faces, flag axes), visibility / z-buffer, the robust
fit and its variants are tested on synthetic data. ``run_depth_fit`` runs end to end on a
synthetic capture with an injected depth model that "sees" the true depth through a known
inverse calibration, which the fit must recover, and the 40-ray scan is scored against the
contract ``raycast()``. A local opt-in test runs the real depth model on the real capture.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.gate_types import GateTypeTable
from drone_fly.harness.geometry import CameraModel
from drone_fly.harness.liftoff.masks import FrameMasks, MaskRegion
from drone_fly.harness.offline.capture import load_capture
from drone_fly.harness.offline.depth_fit import (
    _bin_weights,
    build_truth_geometry,
    fit_variants,
    robust_affine_fit,
    run_depth_fit,
    truth_points,
    visible_truth,
)
from drone_fly.harness.offline.track_xml import parse_track_items
from tests.harness_support import liftoff_real_data
from tests.liftoff_capture_support import SMALL_CAMERA, straight_flight, write_manual_capture

PRESET = "liftoff"
CFG = load_config("liftoff", SMALL_CAMERA, preset=PRESET)
TABLE = GateTypeTable.from_config(CFG.data["gate_types"])
ITEM_TABLE = CFG.section("offline")["item_table"]
CAM = CameraModel(256, 108, 120.0, 20.0)
TRUE_SCALE, TRUE_SHIFT = -0.005, 0.25  # 1 / z = scale * raw + shift


def _track(path, extra=""):
    path.write_text(
        "<Track><blueprints>"
        + "".join(
            f"<TrackBlueprint><itemID>InflatableArchBrandless01</itemID><instanceID>{i}"
            f"</instanceID><position><x>0</x><y>0</y><z>{15 * i}</z></position>"
            "<rotation><x>0</x><y>0</y><z>0</z></rotation></TrackBlueprint>"
            for i in range(1, 4)
        )
        + "<TrackBlueprint><itemID>ConcreteBarrier01</itemID><instanceID>50</instanceID>"
        "<position><x>3</x><y>0</y><z>20</z></position>"
        "<rotation><x>0</x><y>0</y><z>0</z></rotation></TrackBlueprint>"
        + extra
        + "</blueprints></Track>",
        encoding="utf-8",
    )
    return path


# ----------------------------------------------------------------------------- geometry


def test_truth_geometry_primitives(tmp_path):
    flag = (
        "<TrackBlueprint><itemID>AirflagLiftoffSlalom01</itemID><instanceID>60</instanceID>"
        "<position><x>-5</x><y>0</y><z>10</z></position></TrackBlueprint>"
    )
    geom = build_truth_geometry(
        parse_track_items(_track(tmp_path / "t.xml", flag)), TABLE, ITEM_TABLE
    )
    assert len(geom["tube"]) and (geom["tube_r"] == 0.5).all()
    # the arch centre line spans x_u = +-3.5 (contract Y = -+3.5) and reaches 3.958 m up
    arch1 = geom["tube"][np.isclose(geom["tube"][:, 0], 15.0)]
    assert arch1[:, 1].min() == pytest.approx(-3.5) and arch1[:, 1].max() == pytest.approx(3.5)
    assert arch1[:, 2].max() == pytest.approx(3.958)
    assert len(geom["face"]) == 6 * 25 and len(geom["face_n"]) == len(geom["face"])
    assert len(geom["axis"]) and (geom["axis_r"] == 0.3).all()
    np.testing.assert_allclose(geom["axis"][:, :2].mean(axis=0), [10.0, 5.29], atol=1e-6)


def test_truth_points_face_the_camera(tmp_path):
    geom = build_truth_geometry(parse_track_items(_track(tmp_path / "t.xml")), TABLE, ITEM_TABLE)
    cam = np.array([0.0, 0.0, 1.74])
    pts = truth_points(geom, cam)
    n_tube = len(geom["tube"])
    # tube points move 0.5 m (tube radius) towards the camera
    d_line = np.linalg.norm(geom["tube"] - cam, axis=1)
    d_surf = np.linalg.norm(pts[:n_tube] - cam, axis=1)
    np.testing.assert_allclose(d_line - d_surf, 0.5, atol=1e-6)
    # only the barrier faces whose normal points at the camera are kept (3 of 6 here at most)
    faces = pts[n_tube:]
    assert 0 < len(faces) <= 3 * 25


def test_visible_truth_filters_range_masks_and_occlusion():
    pos = np.zeros(3)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    near = np.array([[5.0, 0.0, 1.0]])
    behind_near = np.array([[10.0, 0.0, 2.0]])  # same pixel ray, twice the distance
    too_close = np.array([[1.0, 0.0, 0.0]])
    too_far = np.array([[60.0, 0.0, 0.0]])
    behind = np.array([[-5.0, 0.0, 0.0]])
    pts = np.vstack([near, behind_near, too_close, too_far, behind])
    px, z = visible_truth(pts, CAM, FrameMasks(), pos, quat)
    assert len(z) == 1 and z[0] == pytest.approx(
        np.linalg.norm([5.0, 0, 1]) * np.cos(np.arctan2(1.0, 5.0) - np.radians(20)), rel=1e-6
    )
    u, v = px[0]
    mask = FrameMasks([MaskRegion("m", (u - 2, v - 2, 4, 4))], resolution=(256, 108))
    assert len(visible_truth(near, CAM, mask, pos, quat)[1]) == 0
    assert visible_truth(np.zeros((0, 3)), CAM, FrameMasks(), pos, quat)[1].shape == (0,)


# ----------------------------------------------------------------------------- fit


def test_robust_affine_fit_ignores_outliers():
    rng = np.random.default_rng(0)
    x = rng.uniform(1, 50, 400)
    y = 0.4 * x + 2.0
    y[:60] = rng.uniform(0, 100, 60)  # 15 % garbage
    a, b, inl = robust_affine_fit(x, y)
    assert a == pytest.approx(0.4, rel=1e-3) and b == pytest.approx(2.0, abs=0.05)
    assert inl >= 340
    with pytest.raises(HarnessConfigError, match="at least 2"):
        robust_affine_fit(np.array([1.0]), np.array([1.0]))


def test_bin_weights_give_each_range_bin_equal_weight():
    rng = np.r_[np.full(10, 3.0), np.full(1000, 7.0), np.full(100, 15.0), np.full(50, 30.0)]
    w = _bin_weights(rng, (2.0, 20.0))
    assert w[:10].sum() == pytest.approx(1.0) and w[10:1010].sum() == pytest.approx(1.0)
    assert w[1010:1110].sum() == pytest.approx(1.0) and (w[1110:] == 0).all()


def test_fit_variants_pick_the_form_the_data_follows():
    rng = np.random.default_rng(1)
    z = rng.uniform(2.0, 40.0, 3000)
    raw_inverse = (1.0 / z - TRUE_SHIFT) / TRUE_SCALE
    variants = fit_variants(raw_inverse, z, z)
    assert [v["name"] for v in variants][0] == "inverse_near_bin_weighted"
    best = variants[0]
    assert best["inverse"] is True and best["near_score"] < 1e-6
    assert best["scale"] == pytest.approx(TRUE_SCALE, rel=1e-4)
    assert best["shift"] == pytest.approx(TRUE_SHIFT, rel=1e-4)
    assert {v["name"] for v in variants} == {
        "affine_all_pairs",
        "affine_near_bin_weighted",
        "scale_only_near_bin_weighted",
        "inverse_near_bin_weighted",
    }
    raw_affine = (z - 1.0) / 0.5
    best_affine = fit_variants(raw_affine, z, z)[0]
    assert best_affine["inverse"] is False and best_affine["near_score"] < 1e-6
    assert best_affine["scale"] == pytest.approx(0.5, rel=1e-4)


# ----------------------------------------------------------------------------- end to end


class OracleDepth:
    """Paints the true depth of the XML geometry through the inverse calibration (by frame)."""

    inverse = True

    def __init__(self, states_by_frame, geom):
        self.states = states_by_frame
        self.geom = geom
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        st = self.states[int(image[0, 0, 0])]
        raw = np.full(image.shape[:2], (1.0 / 200.0 - TRUE_SHIFT) / TRUE_SCALE, np.float32)
        px, z = visible_truth(truth_points(self.geom, st.pos), CAM, FrameMasks(), st.pos, st.quat)
        order = np.argsort(-z)  # far first, near overwrites
        for (u, v), zz in zip(np.round(px[order]).astype(int), z[order], strict=True):
            raw[max(v - 3, 0) : v + 4, max(u - 3, 0) : u + 4] = (1.0 / zz - TRUE_SHIFT) / TRUE_SCALE
        return raw


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("uc64-depth")
    cap = write_manual_capture(
        tmp / "cap",
        straight_flight(duration_s=8.0),
        size=(256, 108),
        frame_times_s=[k / 10 for k in range(1, 80)],
    )
    track = _track(tmp / "track.xml")
    race = tmp / "race.xml"  # legacy flat race: arches 1 -> 2 -> 3
    race.write_text(
        "<Race>" + "".join(f"<instanceID>{i}</instanceID>" for i in (1, 2, 3)) + "</Race>"
    )
    c = load_capture(cap, CFG.section("telemetry"))
    geom = build_truth_geometry(parse_track_items(track), TABLE, ITEM_TABLE)
    return cap, track, race, {f.index: f.state for f in c.frames}, geom


def test_run_depth_fit_recovers_the_calibration_and_scores_rays(
    synthetic, tmp_path, harness_contract
):
    cap, track, race, states, geom = synthetic
    model = OracleDepth(states, geom)
    rep = run_depth_fit(
        cap,
        track,
        race,
        preset=PRESET,
        config=SMALL_CAMERA,
        every=2,
        report_dir=tmp_path,
        depth_model=model,
        contract=harness_contract,
    )
    fit = rep["fit"]
    assert model.calls == fit["frames"] == len(range(0, 71, 2))  # every 2nd of 71 paired frames
    assert fit["chosen"] == "inverse_near_bin_weighted" and fit["inverse"] is True
    assert fit["scale"] == pytest.approx(TRUE_SCALE, rel=0.05)
    assert fit["shift"] == pytest.approx(TRUE_SHIFT, rel=0.05)
    assert fit["after"]["near_score"] < 0.1 < fit["before"]["near_score"]
    assert fit["pairs"] >= fit["pairs_in_fit_range"] > 100
    assert set(fit["before"]["bins"]) == {"2-5 m", "5-10 m", "10-20 m", "20-40 m"}
    rays = rep["raycast"]
    for key in (
        "ray_error_m_median",
        "ray_within_0_5m",
        "ray_error_m_median_in_fov",
        "ray_within_0_5m_in_fov",
        "coverage_mean",
        "all_clear_reference",
        "truth_ray_hit_share",
    ):
        assert key in rays
    assert 0.0 <= rays["ray_within_0_5m"] <= 1.0 and rays["world_scale"] == 4.0
    assert all("raycast" in v for v in fit["variants"])
    assert "arches are gates" in rays["note"]
    assert "inverse: true" in rep["yaml_snippet"] and "scale: -0.00" in rep["yaml_snippet"]
    on_disk = json.loads((tmp_path / "depth_fit_report.json").read_text(encoding="utf-8"))
    assert on_disk["fit"]["chosen"] == fit["chosen"]


def test_depth_fit_argument_errors(synthetic, tmp_path, harness_contract):
    cap, track, race, states, geom = synthetic
    kw = dict(
        race=race,
        preset=PRESET,
        report_dir=tmp_path,
        depth_model=OracleDepth(states, geom),
        contract=harness_contract,
    )
    with pytest.raises(HarnessConfigError, match="--every"):
        run_depth_fit(cap, track, config=SMALL_CAMERA, every=0, **kw)
    with pytest.raises(HarnessConfigError, match="resolution"):
        run_depth_fit(cap, track, config={"camera": {"resolution": None, "masks": []}}, **kw)
    empty = tmp_path / "empty.xml"
    empty.write_text("<Track><blueprints/></Track>")
    with pytest.raises(HarnessConfigError, match="no ground-truth points"):
        run_depth_fit(cap, empty, config=SMALL_CAMERA, **kw)


def test_real_capture_depth_fit_local_opt_in(tmp_path):
    """Runs Depth-Anything-V2 (CPU) on every 60th real frame; slow, local only."""
    import os

    data = liftoff_real_data()
    if not os.environ.get("DRONE_FLY_LIFTOFF_DEPTH"):
        pytest.skip("set DRONE_FLY_LIFTOFF_DEPTH=1 to run the real depth model (minutes on CPU)")
    pytest.importorskip("transformers")
    pytest.importorskip("cv2")
    rep = run_depth_fit(
        data.capture,
        data.track,
        data.race,
        data.props,
        preset=PRESET,
        every=60,
        report_dir=tmp_path,
        device=os.environ.get("DRONE_FLY_EVAL_DEVICE", "cpu"),
    )
    fit = rep["fit"]
    assert fit["pairs_in_fit_range"] > 1000 and fit["frames"] > 20
    assert fit["after"]["near_score"] <= fit["before"]["near_score"]
    assert rep["props_warnings"] == []
