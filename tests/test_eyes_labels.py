"""UC-78 AC2: exact ray labels on synthetic scenes (CPU, contract fixture, no track data).

* A box wall and a vertical cylinder at known Liftoff positions: every ray's labelled distance
  matches a hand-computed one within 5 cm, at ``W = 4`` with a non-zero course origin (the label
  position is ``state.pos - origin / W``, never ``to_course_point``).
* ``ray_range_scale``: the range is ``RAY_RANGE * s * W`` Liftoff m.
* The label is the training raycast (obstacles only): the floor and gate items never shorten it;
  they show up in the diagnostic ``aux_kind`` channel.
* Target fallback, warning time, the hit obstacle / size, label-vs-live parity of the camera rays
  and ego-motion, the settings hash, and the skip reasons of ``run_label``.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from drone_fly.eyes import labels as lb
from drone_fly.eyes.camera_rays import FWD_DEFAULT, FWD_TARGET, FWD_VELOCITY
from drone_fly.eyes.ego import ego_vector
from drone_fly.eyes.history import select_history
from drone_fly.harness.geometry import CameraModel
from tests.eyes_support import IDENTITY, label_settings, quat_axis, state, write_capture

W = 4.0
ORIGIN_L = np.array([40.0, -12.0, 3.0])  # course origin (Liftoff m, contract axes)
CAM = CameraModel(256, 108, 120.0, 20.0)


def _box(center_l, half_l):
    """A course-frame raycast box from Liftoff-metre centre / half extents."""
    return {
        "center": ((np.asarray(center_l) - ORIGIN_L) / W).astype(np.float32),
        "half_extents": (np.asarray(half_l) / W).astype(np.float32),
        "type": "box",
    }


def _cyl(center_l, radius_l, half_h_l):
    return {
        "center": ((np.asarray(center_l) - ORIGIN_L) / W).astype(np.float32),
        "half_extents": np.array([radius_l, radius_l, half_h_l], np.float32) / W,
        "type": "cylinder",
    }


def _label(
    contract, states, obstacles, *, s=1.0, targets=None, stamps=None, aux=None, settings=None
):
    st = settings or label_settings(contract, W, s)
    n = len(states)
    stamps = np.arange(n, dtype=np.float64) * 0.1 if stamps is None else stamps
    return lb.compute_labels(
        states,
        stamps,
        np.zeros(n, np.int64),
        targets,
        ORIGIN_L / W,
        obstacles,
        contract,
        st,
        CAM,
        [],
        aux,
    )


def _drone(p_l, v_l=(20.0, 0, 0), quat=IDENTITY):
    """A FlightState at Liftoff position / velocity (training scale = / W)."""
    return state(np.asarray(p_l) / W, np.asarray(v_l) / W, quat)


def test_box_wall_distances_within_5cm(harness_contract):
    c = harness_contract
    p = ORIGIN_L + np.array([5.0, 2.0, 1.5])
    wall = _box(p + [8.0, 0, 0], [0.5, 30.0, 30.0])  # face 7.5 m ahead, very wide
    out = _label(c, [_drone(p)], [wall])
    expect = 7.5 / c.CONE[:, 2].astype(np.float64)  # fwd = +x: the ray's x share = cone z
    assert np.all(out["hit"][0])
    np.testing.assert_allclose(out["dist_m"][0], expect, atol=0.05)
    np.testing.assert_allclose(out["free"][0], expect / (c.RAY_RANGE * W), atol=0.05 / 16)
    assert np.all(out["hit_obstacle"][0] == 0)
    np.testing.assert_allclose(out["hit_min_half_m"][0], 0.5, atol=1e-5)


def test_cylinder_distances_within_5cm(harness_contract):
    c = harness_contract
    p = ORIGIN_L + np.array([-3.0, 1.0, 2.0])
    centre = p + [6.0, 0.8, 0.0]
    r = 1.0
    out = _label(c, [_drone(p)], [_cyl(centre, r, 10.0)])
    rays = out["rays_world"][0].astype(np.float64)
    # hand computed: |(p + t d - centre)_xy| = r, smallest positive root, in Liftoff metres
    f = (p - centre)[:2]
    d = rays[:, :2]
    a = (d * d).sum(1)
    b = 2 * (d @ f)
    cc = f @ f - r * r
    disc = b * b - 4 * a * cc
    t = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
    hit = np.isfinite(t) & (t < c.RAY_RANGE * W)
    clear_margin = np.abs(disc) > 1e-2  # ignore grazing rays
    assert hit.sum() >= 3
    np.testing.assert_array_equal(out["hit"][0][clear_margin], hit[clear_margin])
    np.testing.assert_allclose(out["dist_m"][0][hit], t[hit], atol=0.05)
    assert np.all(out["free"][0][~hit & clear_margin] == 1.0)
    assert np.all(out["dist_m"][0][~hit & clear_margin] == pytest.approx(16.0))


def test_world_scale_is_a_pure_translation(harness_contract):
    """The same Liftoff scene gives the same Liftoff distances at W = 4 and W = 2."""
    c = harness_contract
    p = ORIGIN_L + np.array([1.0, 0.0, 2.0])
    a = _label(c, [_drone(p)], [_box(p + [6.0, 0, 0], [0.5, 30, 30])])
    st2 = label_settings(c, 2.0, 2.0)  # W = 2 with s = 2 keeps the Liftoff range at 16 m
    obs2 = [
        {
            "center": ((p + [6.0, 0, 0] - ORIGIN_L) / 2.0).astype(np.float32),
            "half_extents": np.array([0.5, 30, 30], np.float32) / 2.0,
            "type": "box",
        }
    ]
    s2 = state(p / 2.0, np.array([20.0, 0, 0]) / 2.0)
    b = lb.compute_labels(
        [s2], np.zeros(1), np.zeros(1, np.int64), None, ORIGIN_L / 2.0, obs2, c, st2, CAM
    )
    assert st2.range_m == pytest.approx(16.0)
    np.testing.assert_allclose(a["dist_m"], b["dist_m"], atol=0.02)


def test_ray_range_scale(harness_contract):
    """s = 1.5: range 4 * 1.5 * 4 = 24 Liftoff m (a wall 18 m ahead: clear at s = 1, hit at 1.5)."""
    c = harness_contract
    p = ORIGIN_L + np.array([0.0, 0.0, 2.0])
    wall = [_box(p + [18.5, 0, 0], [0.5, 60, 60])]
    a = _label(c, [_drone(p)], wall, s=1.0)
    assert not a["hit"].any() and np.all(a["dist_m"] == pytest.approx(16.0))
    b = _label(c, [_drone(p)], wall, s=1.5)
    st = label_settings(c, W, 1.5)
    assert st.range_m == pytest.approx(24.0) and st.range_train == pytest.approx(6.0)
    expect = 18.0 / c.CONE[:, 2].astype(np.float64)
    inside = expect < 24.0 - 0.1
    assert inside.sum() >= 5
    np.testing.assert_array_equal(b["hit"][0], expect < 24.0)
    np.testing.assert_allclose(b["dist_m"][0][inside], expect[inside], atol=0.05)
    np.testing.assert_allclose(b["free"][0][inside], expect[inside] / 24.0, atol=0.05 / 24)


def test_nearest_obstacle_index_and_size(harness_contract):
    c = harness_contract
    p = ORIGIN_L + np.array([0.0, 0.0, 2.0])
    far = _box(p + [12.0, 0, 0], [1.0, 40, 40])
    near = _cyl(p + [4.0, 0, 0], 0.15, 10.0)  # a thin pole straight ahead
    out = _label(c, [_drone(p)], [far, near])
    j = np.argmax(c.CONE[:, 2])  # the most-forward ray hits the pole first
    assert out["hit_obstacle"][0, j] == 1
    assert out["hit_min_half_m"][0, j] == pytest.approx(0.15, abs=1e-5)
    assert out["dist_m"][0, j] == pytest.approx(3.85, abs=0.05)
    assert np.all(out["hit_obstacle"][0][~out["hit"][0]] == -1)


def _empty_aux(**over):
    z3, z0 = np.zeros((0, 3)), np.zeros(0)
    base = dict(
        box_c=z3,
        box_rot=np.zeros((0, 3, 3)),
        box_half=z3,
        box_kind=np.zeros(0, np.int8),
        cap_c=z3,
        cap_axis=z3,
        cap_half=z0,
        cap_r=z0,
        cap_kind=np.zeros(0, np.int8),
        obs_lo=z3,
        obs_hi=z3,
        floor_z=0.0,
    )
    base.update(over)
    return lb.AuxGeometry(**base)


def test_floor_never_shortens_the_label(harness_contract):
    """Diving at the floor: the label stays clear (training raycast = obstacles only); aux sees
    the floor."""
    s = state([0.0, 0.0, 0.5], [2.0, 0.0, -1.5])
    out = lb.compute_labels(
        [s],
        np.zeros(1),
        np.zeros(1, np.int64),
        None,
        np.zeros(3),
        [],
        harness_contract,
        label_settings(harness_contract),
        CAM,
        [],
        _empty_aux(),
    )
    assert np.all(out["free"] == 1.0) and not out["hit"].any()
    floor = out["aux_kind"][0] == lb.AUX_FLOOR
    assert floor.sum() >= 10
    assert np.all(out["aux_free"][0][floor] < 1.0)


def test_gate_items_never_shorten_the_label(harness_contract):
    """A placed item box at a passage is ``gate`` in aux, clear in the label."""
    item = np.array([[3.0, 0.0, 1.0]])
    aux = _empty_aux(
        box_c=item,
        box_rot=np.eye(3)[None],
        box_half=np.array([[0.1, 2.0, 2.0]]),
        box_kind=np.array([lb.AUX_ITEM], np.int8),
        floor_z=-100.0,
        passage_c=np.array([[3.0, 0.0, 1.0]]),
        passage_reach=np.array([1.0]),
    )
    s = state([0.0, 0.0, 1.0], [2.0, 0.0, 0.0])
    out = lb.compute_labels(
        [s],
        np.zeros(1),
        np.zeros(1, np.int64),
        None,
        np.zeros(3),
        [],
        harness_contract,
        label_settings(harness_contract),
        CAM,
        [],
        aux,
    )
    assert np.all(out["free"] == 1.0)
    j = np.argmax(harness_contract.CONE[:, 2])
    assert out["aux_kind"][0, j] == lb.AUX_GATE
    assert out["aux_free"][0, j] == pytest.approx(2.9 / 4.0, abs=0.01)
    # the same item far from any passage is a plain item
    aux_far = replace(aux, passage_c=np.array([[30.0, 0.0, 1.0]]))
    out2 = lb.compute_labels(
        [s],
        np.zeros(1),
        np.zeros(1, np.int64),
        None,
        np.zeros(3),
        [],
        harness_contract,
        label_settings(harness_contract),
        CAM,
        [],
        aux_far,
    )
    assert out2["aux_kind"][0, j] == lb.AUX_ITEM


def test_aux_obstacle_takes_precedence(harness_contract):
    """The label's own hit is reported as ``obstacle`` by the aux channel too."""
    c = harness_contract
    p = ORIGIN_L + np.array([0.0, 0.0, 2.0])
    box = _box(p + [6.0, 0, 0], [0.5, 30, 30])
    aux = _empty_aux(
        obs_lo=(box["center"] - box["half_extents"])[None].astype(np.float64),
        obs_hi=(box["center"] + box["half_extents"])[None].astype(np.float64),
        floor_z=-100.0,
    )
    out = _label(c, [_drone(p)], [box], aux=aux)
    assert np.all(out["aux_kind"][0] == lb.AUX_OBSTACLE)
    np.testing.assert_allclose(out["aux_free"][0], out["free"][0], atol=1e-3)


def test_target_fallback_and_warning_time(harness_contract):
    c = harness_contract
    p = ORIGIN_L + np.array([0.0, 0.0, 2.0])
    slow = _drone(p, (0.4, 0, 0))  # 0.1 training m/s: below the 0.3 heading switch
    fast = _drone(p, (20.0, 0, 0))
    tgt = np.array([[0.0, 5.0, 0.0], [np.nan] * 3, [0.0, 5.0, 0.0]]) + (p - ORIGIN_L) / W
    out = _label(c, [slow, slow, fast], [], targets=tgt)
    assert out["fwd_source"].tolist() == [FWD_TARGET, FWD_DEFAULT, FWD_VELOCITY]
    j = np.argmax(c.CONE[:, 2])
    assert out["rays_world"][0, j, 1] > 0.9  # towards the target (+y)
    assert out["rays_world"][1, j, 0] > 0.9  # default +x
    # warning = range_m / speed (Liftoff); at <= 0.3 m/s it is inf
    assert out["speed_mps"][2] == pytest.approx(20.0)
    assert out["warning_s"][2] == pytest.approx(16.0 / 20.0)
    assert out["speed_mps"][0] == pytest.approx(0.4)
    assert out["warning_s"][0] == pytest.approx(16.0 / 0.4, rel=1e-4)
    still = _label(c, [_drone(p, (0.0, 0, 0))], [])
    assert np.isinf(still["warning_s"][0])
    shares = lb.warning_shares(np.array([0.2, 0.4, 0.6, np.inf]))
    assert shares == {
        "finite": 3,
        "share_lt_0_3": pytest.approx(1 / 3),
        "share_lt_0_5": pytest.approx(2 / 3),
    }


def test_label_camera_and_ego_match_the_live_path(harness_contract):
    """Labels store exactly what the live estimator computes from the same FlightStates:
    ray_view (camera dirs, uv, in_view) and the ego vector with its stride-1 history frame."""
    from drone_fly.eyes.camera_rays import cone_namespace, ray_view

    c = harness_contract
    rng = np.random.default_rng(5)
    states, stamps = [], np.cumsum(rng.uniform(0.08, 0.17, 12))
    for k in range(12):
        q = quat_axis([0.2, 1.0, 0.1 * k], 15.0 + 2 * k)
        states.append(_drone(ORIGIN_L + [k * 2.0, 0, 2.0], (18.0, 3.0 * np.sin(k), -1.0), q))
    out = _label(c, states, [], stamps=stamps)
    ns = cone_namespace(c.CONE)
    st = label_settings(c)
    assert st.ego_dt_s == 0.2 and st.hist_tol_s == pytest.approx(0.1)
    for i, s in enumerate(states):
        v = ray_view(s.pos, s.vel, s.quat, None, ns, CAM)
        np.testing.assert_allclose(out["rays_cam"][i], v.rays_cam, atol=1e-6)
        np.testing.assert_allclose(out["uv"][i], v.uv, atol=1e-6)
        np.testing.assert_array_equal(out["in_view"][i], v.in_view)
        h = select_history(stamps, i, 2, 0.2, 0.1)
        if h:
            e = ego_vector(s.quat, s.vel, states[h[0]].quat, stamps[i] - stamps[h[0]])
        else:
            e = ego_vector(s.quat, s.vel)
        np.testing.assert_allclose(out["ego"][i], e, atol=1e-6)
        assert bool(out["ego_history"][i]) == bool(h)
    assert out["ego_history"].sum() >= 6
    np.testing.assert_allclose(
        out["pos_c"], [np.asarray(s.pos) - ORIGIN_L / W for s in states], atol=1e-5
    )


def test_settings_hash():
    class C:
        CONE = np.ones((40, 3), np.float32)
        RAY_RANGE = 4.0
        N_RAYS = 40

    a = lb.settings_for(C, 4.0, 1.0)
    assert a.sha256() == lb.settings_for(C, 4.0, 1.0).sha256()
    for other in (
        lb.settings_for(C, 2.0, 1.0),
        lb.settings_for(C, 4.0, 1.5),
        replace(a, ego_dt_s=0.1),
        replace(a, pair_tol_s=0.05),
    ):
        assert other.sha256() != a.sha256()
    assert "lag_ms" not in a.as_dict()  # per-capture lag is not a label setting
    assert a.ego_dt_s == 0.2 and a.hist_tol_s == pytest.approx(0.1)
    assert a.range_m == pytest.approx(16.0)


def test_scale_obstacles_divides_lengths():
    o = lb.scale_obstacles([{"center": [3.0, 0, 1.5], "half_extents": [0.3, 0.3, 3.0]}], 1.5)
    np.testing.assert_allclose(o[0]["center"], [2.0, 0, 1.0])
    np.testing.assert_allclose(o[0]["half_extents"], [0.2, 0.2, 2.0])
    assert o[0]["type"] == "cylinder"


# ----------------------------------------------------------------------------- run_label skips


def test_run_label_skips_without_meta(harness_contract, tmp_path):
    cap = write_capture(tmp_path / "legacy", catalogue_id=None, map_name=None, meta=False)
    res = lb.run_label(cap, labels_root=tmp_path / "labels")
    assert res["skipped"] == "no-meta"
    assert not (tmp_path / "labels").exists()


def test_run_label_skips_unknown_catalogue_entry(harness_contract, tmp_path):
    cap = write_capture(tmp_path / "c", catalogue_id="no-such-track", map_name="Nowhere")
    res = lb.run_label(cap, labels_root=tmp_path / "labels")
    assert res["skipped"].startswith("no-geometry")
    assert res["catalogue_id"] == "no-such-track"


def test_legacy_capture_needs_both_overrides(tmp_path):
    from drone_fly.eyes.capture_meta import read_capture_meta

    cap = write_capture(tmp_path / "legacy", catalogue_id=None, map_name=None, meta=False)
    assert read_capture_meta(cap, catalogue_id="x") is None
    m = read_capture_meta(cap, catalogue_id="x", lag_ms=25)
    assert m.legacy and m.catalogue_id == "x" and m.lag_ms == 25.0


def test_capture_meta_reader(tmp_path):
    from drone_fly.eyes.capture_meta import capture_sha, discover_captures, read_capture_meta
    from drone_fly.harness.errors import HarnessConfigError

    a = write_capture(
        tmp_path / "root" / "MapA" / "c1", catalogue_id="id-a", map_name="MapA", verified=False
    )
    b = write_capture(tmp_path / "root" / "c2", catalogue_id="id-b", map_name="MapB", seed=1)
    assert sorted(discover_captures(tmp_path / "root")) == sorted([a, b])  # depth 1 and 2
    assert discover_captures(a) == [a]
    m = read_capture_meta(a)
    assert (m.catalogue_id, m.map, m.alignment_verified, m.lag_ms) == ("id-a", "MapA", False, 12.0)
    assert read_capture_meta(a, lag_ms=3).lag_ms == 3.0
    sha = capture_sha(a)
    (a / "frames.csv").write_text("t_wall_ns,file\n", encoding="utf-8")
    assert capture_sha(a) != sha  # re-captured folder: new identity
    meta = json.loads((b / "meta.json").read_text())
    (b / "meta.json").write_text(json.dumps({**meta, "schema_version": 2}), encoding="utf-8")
    with pytest.raises(HarnessConfigError, match="schema_version"):
        read_capture_meta(b)
