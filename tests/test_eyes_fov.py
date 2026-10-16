"""UC-78 AC1 (+ the deployable ray / ego geometry it rests on).

* camera_rays: projections through the uptilted FPV camera, in-view / mask rules, the cone basis
  parity with ``inference.raycast``;
* ego: ``gravity_body`` (sign pinned, parity with the contract's ``make_obs`` block) and body rates
  from attitude deltas;
* fov: the sweep classification, the report counts for the ``liftoff`` preset (5 always / 0 never
  / 35 attitude-dependent over the 10-40 deg band) and the ``eyes fov-report`` outputs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from drone_fly.eyes import camera_rays as cr
from drone_fly.eyes import ego as eg
from drone_fly.eyes import fov
from drone_fly.harness.geometry import CameraModel, ray_directions
from tests.eyes_support import IDENTITY, fib_cone, quat_axis

CAM = CameraModel(2560, 1080, 120.0, 20.0)


# ----------------------------------------------------------------------------- camera_rays


def test_straight_ahead_projects_below_centre_with_uptilt():
    """Level body, a world +x direction: the camera is pitched up 20 deg, so it lands below the
    image centre (OpenCV v > 0) at exactly the uptilt angle."""
    d, uv, view = cr.project_rays(np.array([[1.0, 0, 0]]), IDENTITY, CAM)
    assert view[0]
    np.testing.assert_allclose(
        d[0], [0.0, np.sin(np.radians(20)), np.cos(np.radians(20))], atol=1e-6
    )
    assert uv[0, 0] == pytest.approx(0.0, abs=1e-3) and uv[0, 1] > 0
    v_px = CAM.fy * np.tan(np.radians(20)) + CAM.cy
    assert uv[0, 1] == pytest.approx(2 * v_px / (CAM.height - 1) - 1, abs=1e-4)


def test_left_is_left_and_behind_is_out_of_view():
    d, uv, view = cr.project_rays(np.array([[1.0, 0.3, 0.36], [-1.0, 0, 0]]), IDENTITY, CAM)
    assert view[0] and uv[0, 0] < 0  # body +y (left) -> image left
    assert not view[1] and np.all(uv[1] == 0.0)  # behind: never in view, uv zeroed


def test_masks_hide_rays_and_normalise():
    m = cr.normalise_masks([[0, 0, 1280, 1080]], (2560, 1080))
    assert m == [[0.0, 0.0, 0.5, 1.0]]
    rays = np.array([[1.0, 0.3, 0.36], [1.0, -0.3, 0.36]])  # left half / right half
    _, _, view = cr.project_rays(rays, IDENTITY, CAM, m)
    assert view.tolist() == [False, True]
    # masks are fractions, so they apply at any resolution
    _, _, view_small = cr.project_rays(rays, IDENTITY, CAM.scaled_to(256, 108), m)
    assert view_small.tolist() == [False, True]


def test_masks_from_camera_config_needs_resolution():
    assert cr.masks_from_camera_config({}) == []
    cfg = {"resolution": [100, 50], "masks": [{"rect": [10, 5, 20, 10]}]}
    assert cr.masks_from_camera_config(cfg) == [[0.1, 0.1, 0.3, 0.3]]
    with pytest.raises(ValueError, match="resolution"):
        cr.masks_from_camera_config({"masks": [{"rect": [0, 0, 1, 1]}]})


def test_fwd_source_matches_the_heading_switch():
    assert cr.fwd_source([0.31, 0, 0], None) == cr.FWD_VELOCITY
    assert cr.fwd_source([0.29, 0, 0], [5, 0, 0]) == cr.FWD_TARGET
    assert cr.fwd_source([0.0, 0, 0], None) == cr.FWD_DEFAULT


def test_cone_namespace_is_frozen():
    ns = cr.cone_namespace(fib_cone().tolist())
    assert ns.CONE.shape == (40, 3)
    with pytest.raises(ValueError):
        ns.CONE[0, 0] = 1.0


def test_ray_basis_parity_with_contract_raycast(harness_contract):
    """The deployable basis equals ``inference.raycast``'s: a thin wall that only one ray's
    direction hits is reported hit by raycast on exactly that ray."""
    c = harness_contract
    ns = cr.cone_namespace(c.CONE)
    pos, vel = np.zeros(3, np.float32), np.array([1.2, 0.7, -0.3], np.float32)
    rays = ray_directions(pos, vel, None, ns)
    np.testing.assert_allclose(rays, ray_directions(pos, vel, None, c), atol=1e-7)
    for j in (0, 17, 39):
        p = rays[j] * 2.0  # a tiny box 2 m along ray j
        free = c.raycast(
            {"pos": pos, "vel": vel}, [{"center": p, "half_extents": [0.02] * 3, "type": "box"}]
        )
        assert free[j] == pytest.approx(
            (2.0 - 0.02 / np.abs(rays[j]).max()) / c.RAY_RANGE, abs=0.02
        )
        assert np.sum(free < 1.0) <= 2


def test_ray_view_low_speed_uses_target():
    ns = cr.cone_namespace(fib_cone())
    v = cr.ray_view([0, 0, 0], [0, 0, 0], IDENTITY, np.array([0, 5.0, 0]), ns, CAM)
    assert v.fwd_source == cr.FWD_TARGET
    # the cone axis now points to +y (left): its most-forward ray is roughly +y
    assert v.rays_world[np.argmax(fib_cone()[:, 2])][1] > 0.9


def test_camera_meta_round_trip():
    assert cr.camera_from_meta(cr.camera_meta(CAM)) == CAM


# ----------------------------------------------------------------------------- ego


def test_gravity_body_level_and_sign_pinned():
    np.testing.assert_allclose(eg.gravity_body(IDENTITY), [0, 0, -1], atol=1e-7)
    # nose DOWN 30 deg (rotation about body +y): gravity gains a +x (forward) body component
    q = fov.pitch_quat(30.0)
    np.testing.assert_allclose(eg.gravity_body(q), [0.5, 0.0, -np.cos(np.radians(30))], atol=1e-6)
    # +30 deg about body +x lifts the left (+y) side: gravity gains a body -y (right) component
    g = eg.gravity_body(quat_axis([1, 0, 0], 30.0))
    assert g[1] == pytest.approx(-0.5, abs=1e-6)


def test_gravity_body_parity_with_make_obs(harness_contract):
    rng = np.random.default_rng(3)
    for _ in range(5):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        s = {
            "pos": np.zeros(3, np.float32),
            "vel": np.zeros(3, np.float32),
            "quat": q.astype(np.float32),
            "omega": np.zeros(3, np.float32),
        }
        obs = harness_contract.make_obs(s, np.zeros(3, np.float32))
        np.testing.assert_allclose(eg.gravity_body(q), obs[3:6], atol=1e-5)
        np.testing.assert_allclose(eg.ego_vector(q, [0, 0, 0])[:3], obs[3:6], atol=1e-5)


def test_body_rates_from_attitude_deltas():
    np.testing.assert_allclose(
        eg.body_rates(IDENTITY, quat_axis([0, 0, 1], 20), 0.2),
        [0, 0, np.radians(20) / 0.2],
        atol=1e-5,
    )
    # body frame: a yawed drone that then rolls about ITS x axis reports a pure x rate
    q0 = quat_axis([0, 0, 1], 90)
    q1 = eg.quat_mul(q0, quat_axis([1, 0, 0], 10))
    np.testing.assert_allclose(eg.body_rates(q0, q1, 0.1), [np.radians(10) / 0.1, 0, 0], atol=1e-4)
    # q and -q are the same attitude: no spurious 360 deg rate
    np.testing.assert_allclose(eg.body_rates(q0, -q0, 0.1), [0, 0, 0], atol=1e-6)
    assert np.all(eg.body_rates(IDENTITY, q1, 0.0) == 0)


def test_ego_vector_layout():
    q = quat_axis([0, 0, 1], 90)  # yaw left 90: world +y is body +x
    e = eg.ego_vector(q, [0, 3, 0], IDENTITY, 0.5)
    assert e.shape == (eg.EGO_DIM,) == (len(eg.EGO_FIELDS),)
    np.testing.assert_allclose(e[3:6], [3, 0, 0], atol=1e-5)  # body velocity
    assert e[6] == pytest.approx(3.0)
    assert e[9] == pytest.approx(np.radians(90) / 0.5, rel=1e-4)
    assert np.all(eg.ego_vector(q, [0, 3, 0])[7:] == 0)  # no history -> zero rates


# ----------------------------------------------------------------------------- fov


def test_pitch_quat_is_nose_down():
    from drone_fly.harness.geometry import quat_to_matrix

    nose = quat_to_matrix(fov.pitch_quat(30.0)) @ np.array([1.0, 0, 0])
    assert nose[2] == pytest.approx(-0.5, abs=1e-6)


def test_ranges_and_classes():
    assert fov.pitch_range(0, 10, 5) == (0.0, 5.0, 10.0)
    assert fov.fpa_range(10, 5) == (-10.0, -5.0, 0.0, 5.0, 10.0)
    with pytest.raises(ValueError):
        fov.pitch_range(0, 10, 0)
    assert [fov.ray_class(f) for f in (1.0, 0.0, 0.4)] == list(fov.CLASSES)
    rows = [
        {"pitch_deg": 0.0, "in_view": [True, True, False]},
        {"pitch_deg": 20.0, "in_view": [True, False, False]},
        {"pitch_deg": 30.0, "in_view": [True, True, False]},
    ]
    np.testing.assert_allclose(fov.classify(rows, (10, 40)), [1.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="band"):
        fov.classify(rows, (50, 60))


def test_sweep_matches_direct_projection():
    """Every sweep row is the ray_view of its pose (no shortcut in the report)."""
    cone = fib_cone()
    rows = fov.sweep(cone, CAM, [], pitches=(20.0,), fpas=(-5.0, 0.0))
    assert [r["case"] for r in rows] == ["velocity", "velocity", "target"]
    ns = cr.cone_namespace(cone)
    v = cr.ray_view(
        np.zeros(3), fov.SWEEP_SPEED * np.array([1.0, 0, 0]), fov.pitch_quat(20.0), None, ns, CAM
    )
    assert rows[1]["in_view"] == v.in_view.tolist() and rows[1]["count"] == int(v.in_view.sum())


def test_fov_report_liftoff_counts(harness_contract):
    """AC1 on the real preset camera + HUD masks: 5 always / 0 never / 35 attitude-dependent."""
    view = fov.preset_view("liftoff")
    assert (
        view.camera.width,
        view.camera.height,
        view.camera.hfov_deg,
        view.camera.uptilt_deg,
    ) == (2560, 1080, 120.0, 20.0)
    rep = fov.fov_report(harness_contract.CONE, view.camera, view.masks, preset="liftoff")
    b = rep["band_summary"]
    assert b["always"] + b["never"] + b["attitude-dependent"] == 40 == rep["n_rays"]
    assert (b["always"], b["never"], b["attitude-dependent"]) == (5, 0, 35)
    assert len(rep["rays"]) == 40 and all(r["class"] in fov.CLASSES for r in rep["rays"])
    level0 = next(r for r in rep["per_pitch"] if r["pitch_deg"] == 0)
    assert level0["min"] <= level0["level_path"] <= level0["max"]
    md = fov.report_markdown(rep)
    assert "5 always" in md and "35 attitude-dependent" in md
    with pytest.raises(ValueError, match="include 0"):
        fov.fov_report(harness_contract.CONE, view.camera, fpas=(5.0,))


def test_masks_only_reduce_coverage(harness_contract):
    view = fov.preset_view("liftoff")
    a = fov.fov_report(harness_contract.CONE, view.camera, [])
    b = fov.fov_report(harness_contract.CONE, view.camera, view.masks)
    for ra, rb in zip(a["sweep"], b["sweep"], strict=True):
        assert all(x or not y for x, y in zip(ra["in_view"], rb["in_view"], strict=True))


def test_run_fov_report_writes_json_and_md(harness_contract, tmp_path):
    ldir = tmp_path / "lab"
    ldir.mkdir()
    np.savez(ldir / "eyes_labels.npz", in_view=np.array([[True] * 20 + [False] * 20] * 4))
    rep = fov.run_fov_report("liftoff", out=tmp_path / "out", labels=[ldir, tmp_path / "none"])
    j = json.loads((tmp_path / "out" / "fov_report.json").read_text())
    assert j["band_summary"] == rep["band_summary"]
    assert (tmp_path / "out" / "fov_report.md").read_text().startswith("# Eyes FOV coverage")
    emp = j["empirical"]
    assert emp[0]["frames"] == 4 and emp[0]["mean_in_view"] == 20.0
    assert emp[1]["error"] == "no eyes_labels.npz"
