"""UC-64 AC4: auto-labelling gates by projecting XML/props geometry through the telemetry pose.

* The harness projection (Unity -> contract -> ``CameraModel``) is pinned to an independent
  restatement of the convention verified on the reference capture (``project_arches.py``: Unity
  frame, inverse body rotation, ``R_x(-uptilt)``, pinhole) to sub-pixel equality.
* Visibility (in front, in image, unmasked), view-relative left/right flipping, drop rules and the
  ``--props`` cross-check are tested on synthetic geometry; ``run_autolabel`` end to end on a
  synthetic capture. A local opt-in test labels the real capture.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.gate_types import GateTypeTable
from drone_fly.harness.liftoff.gate_cv import camera_from_config
from drone_fly.harness.liftoff.masks import FrameMasks, MaskRegion
from drone_fly.harness.liftoff.telemetry import LiftoffConverter
from drone_fly.harness.offline.autolabel import (
    check_gate_types_against_props,
    check_item_table_against_props,
    collider_bounds,
    label_capture,
    label_gate,
    place_gate,
    placed_gates,
    run_autolabel,
)
from drone_fly.harness.offline.capture import load_capture
from drone_fly.harness.offline.track_xml import parse_track_items
from drone_fly.harness.types import TelemetrySample
from tests.harness_support import liftoff_real_data
from tests.liftoff_capture_support import (
    Pose,
    arch_track,
    straight_flight,
    write_manual_capture,
)

PRESET = "liftoff_micro_arena"
CFG = load_config("liftoff", preset=PRESET)
TABLE = GateTypeTable.from_config(CFG.data["gate_types"])
ARCH = TABLE["inflatable_arch"]
CAMERA = camera_from_config(CFG.section("camera"))  # 2560 x 1080, 120 deg, uptilt 20
NO_MASKS = FrameMasks()


# ----------------------------------------------------------------------------- reference


def _qrot(q_xyzw, v):
    x, y, z, w = q_xyzw
    u = np.array([x, y, z])
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def reference_project(pts_unity_w, pos_u, q_xyzw, uptilt_deg, hfov_deg, width, height):
    """project_arches.py: world -> body (inverse rotation), R_x(-uptilt), pinhole (v up)."""
    qi = np.array([-q_xyzw[0], -q_xyzw[1], -q_xyzw[2], q_xyzw[3]])
    b = np.array([_qrot(qi, p - pos_u) for p in pts_unity_w])
    a = math.radians(uptilt_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    c = b @ rx.T
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    z = c[:, 2]
    uv = np.stack([width / 2 + fx * c[:, 0] / z, height / 2 - fx * c[:, 1] / z], axis=1)
    return uv, z


def _unity_yaw(yaw_deg):
    y = math.radians(yaw_deg)
    return np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])


def _random_quat(rng, max_deg=35.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = math.radians(rng.uniform(-max_deg, max_deg))
    return np.array([*(math.sin(ang / 2) * axis), math.cos(ang / 2)])


def _contract_state(pos_u, q_xyzw):
    s = TelemetrySample(
        t=0.0,
        position=np.asarray(pos_u, np.float32),
        quat_xyzw=np.asarray(q_xyzw, np.float32),
        velocity=np.zeros(3, np.float32),
        gyro=np.zeros(3, np.float32),
    )
    return LiftoffConverter().convert(s)


@pytest.mark.parametrize("seed", range(6))
def test_projection_matches_the_verified_reference_convention(tmp_path, seed):
    rng = np.random.default_rng(seed)
    arch_pos = (rng.uniform(-5, 5), 0.0, rng.uniform(12, 25))
    arch_yaw = rng.uniform(-40, 40)
    track = arch_track(tmp_path / "t.xml", [("1", arch_pos, arch_yaw)])
    gate = place_gate(parse_track_items(track)[0], ARCH)
    pos_u = np.array([rng.uniform(-2, 2), rng.uniform(1, 3), rng.uniform(-2, 2)])
    q = _random_quat(rng)
    st = _contract_state(pos_u, q)
    lab = label_gate(gate, CAMERA, NO_MASKS, st.pos, st.quat)
    assert lab is not None
    # Reference on the Unity-local keypoints (+ aperture centre) of the same placed arch.
    local = np.concatenate([ARCH.local_points(), [ARCH.center_offset]])
    pts_w = local @ _unity_yaw(arch_yaw).T + np.asarray(arch_pos)
    uv, z = reference_project(pts_w, pos_u, q, 20.0, 120.0, 2560, 1080)
    assert (z > 0).all()
    kp = np.asarray(lab["keypoints"])
    if lab["flipped"]:
        kp = kp[ARCH.flip_permutation()]
    # float32 telemetry + 2-decimal rounding in the label: well below a pixel
    np.testing.assert_allclose(kp, uv[:-1], atol=0.05)
    np.testing.assert_allclose(lab["center_px"], uv[-1], atol=0.05)
    assert lab["range_m"] == pytest.approx(float(np.linalg.norm(pts_w[-1] - pos_u)), rel=1e-5)


def test_unity_euler_placement_equals_the_reference_yaw_matrix(tmp_path):
    track = arch_track(tmp_path / "t.xml", [("1", (1.0, 0.0, 2.0), 30.0)])
    item = parse_track_items(track)[0]
    np.testing.assert_allclose(item.rotation, _unity_yaw(30.0), atol=1e-12)


# ----------------------------------------------------------------------------- label rules


def _gate(tmp_path, pos=(0.0, 0.0, 20.0), yaw=0.0):
    track = arch_track(tmp_path / "t.xml", [("1", pos, yaw)])
    return place_gate(parse_track_items(track)[0], ARCH)


LEVEL = np.array([0.0, 0.0, 0.0, 1.0])


def test_arch_ahead_is_fully_visible_with_range_and_bearing(tmp_path):
    g = _gate(tmp_path)
    st = _contract_state((0.0, 1.74, 0.0), LEVEL)
    lab = label_gate(g, CAMERA, NO_MASKS, st.pos, st.quat)
    assert lab["cls"] == "inflatable_arch" and lab["instance"] == "1"
    assert lab["item_id"] == "InflatableArchBrandless01"
    assert all(lab["visible"]) and all(lab["in_front"]) and lab["center_in_image"]
    assert lab["range_m"] == pytest.approx(20.0, abs=1e-4)
    assert lab["bearing_deg"][0] == pytest.approx(0.0, abs=1e-3)
    assert lab["bearing_deg"][1] == pytest.approx(-20.0, abs=1e-3)  # camera pitched up 20 deg
    assert lab["flipped"] is False
    np.testing.assert_allclose(lab["center_world"], [20.0, 0.0, 1.74], atol=1e-4)
    # left foot (keypoint 0, Unity x = -4) is on the image left
    assert lab["keypoints"][0][0] < 1280 < lab["keypoints"][1][0]


def test_gate_behind_or_out_of_range_is_not_labelled(tmp_path):
    g = _gate(tmp_path, pos=(0.0, 0.0, -20.0))
    st = _contract_state((0.0, 1.74, 0.0), LEVEL)
    assert label_gate(g, CAMERA, NO_MASKS, st.pos, st.quat) is None
    far = _gate(tmp_path, pos=(0.0, 0.0, 150.0))
    assert label_gate(far, CAMERA, NO_MASKS, st.pos, st.quat) is None
    assert label_gate(far, CAMERA, NO_MASKS, st.pos, st.quat, max_range_m=200.0) is not None


def test_gate_far_to_the_side_is_not_labelled(tmp_path):
    g = _gate(tmp_path, pos=(60.0, 0.0, 5.0))  # ~85 deg right: outside a 120 deg HFOV
    st = _contract_state((0.0, 1.74, 0.0), LEVEL)
    assert label_gate(g, CAMERA, NO_MASKS, st.pos, st.quat) is None


def test_partially_visible_gate_marks_off_image_keypoints_invisible(tmp_path):
    g = _gate(tmp_path, pos=(0.0, 0.0, 3.5))  # close: feet below the image bottom
    st = _contract_state((0.0, 1.74, 0.0), LEVEL)
    lab = label_gate(g, CAMERA, NO_MASKS, st.pos, st.quat)
    vis = np.asarray(lab["visible"])
    in_img = np.asarray(lab["in_image"])
    assert not vis.all() and vis.any()
    assert (vis == in_img).all()
    assert not vis[0] and not vis[1]  # feet out of view


def test_masked_keypoints_are_not_visible(tmp_path):
    g = _gate(tmp_path)
    st = _contract_state((0.0, 1.74, 0.0), LEVEL)
    clear = label_gate(g, CAMERA, NO_MASKS, st.pos, st.quat)
    x, y = clear["keypoints"][8]  # bar top
    masks = FrameMasks([MaskRegion("hud", (x - 20, y - 20, 40, 40))], resolution=(2560, 1080))
    lab = label_gate(g, CAMERA, masks, st.pos, st.quat)
    assert lab["visible"][8] is False and lab["in_image"][8] is True
    assert sum(lab["visible"]) == 9
    assert lab["center_masked"] is False
    cx, cy = clear["center_px"]
    centre_mask = FrameMasks([MaskRegion("c", (cx - 5, cy - 5, 10, 10))], resolution=(2560, 1080))
    assert label_gate(g, CAMERA, centre_mask, st.pos, st.quat)["center_masked"] is True


def test_gate_seen_from_behind_swaps_left_right_pairs(tmp_path):
    front = _gate(tmp_path, yaw=0.0)  # faces +z_u, drone at z=0 -> sees its back? no: front
    back = _gate(tmp_path, yaw=180.0)  # same arch turned around
    st = _contract_state((0.0, 1.74, 0.0), LEVEL)
    a = label_gate(front, CAMERA, NO_MASKS, st.pos, st.quat)
    b = label_gate(back, CAMERA, NO_MASKS, st.pos, st.quat)
    # the drone is on the -z_u side, travelling +z_u: the yaw-0 arch is flown "forwards"
    assert a["flipped"] is False and b["flipped"] is True
    # view-relative identity: keypoint 0 is the image-left foot in both cases
    np.testing.assert_allclose(a["keypoints"], b["keypoints"], atol=0.05)


# ----------------------------------------------------------------------------- capture


def _capture(tmp_path, frames=(1.0, 2.0, 3.0)):
    # straight +z_u flight at 1.74 m towards an arch at z = 30 (range 25 -> 15 m)
    return write_manual_capture(
        tmp_path / "cap", straight_flight(duration_s=4.0), frame_times_s=list(frames)
    )


def test_label_capture_rows_and_counts(tmp_path):
    cap = load_capture(_capture(tmp_path), CFG.section("telemetry"))
    gates = placed_gates(
        parse_track_items(
            arch_track(tmp_path / "t.xml", [("1", (0, 0, 30), 0), ("2", (0, 0, -30), 0)])
        ),
        TABLE,
    )
    rows, counts = label_capture(cap, gates, CAMERA, NO_MASKS)
    assert [r["frame"] for r in rows] == [0, 1, 2]
    assert all(len(r["gates"]) == 1 and r["gates"][0]["instance"] == "1" for r in rows)
    assert [round(r["gates"][0]["range_m"]) for r in rows] == [25, 20, 15]
    assert rows[0]["file"].startswith("frames/") and rows[0]["file"].endswith(".npy")
    assert counts["frames_with_gate"] == 3
    assert counts["gates:inflatable_arch"] == 3
    assert counts["gates_4_25m_centre_in_image:inflatable_arch"] == 3


def test_respawn_frames_are_dropped_before_labelling(tmp_path):
    a = straight_flight(duration_s=3.0)
    b = [Pose(p.t_sim + 3.01, (0.0, 1.74, 0.0), p.quat_xyzw) for p in a]  # respawn at origin
    walls = [p.t_sim for p in a] + [p.t_sim + 3.01 for p in a]
    cap_dir = write_manual_capture(
        tmp_path / "cap", a + b, wall_s=walls, frame_times_s=[1.5, 2.9, 3.1, 4.5]
    )
    track = arch_track(tmp_path / "t.xml", [("1", (0, 0, 30), 0)])
    meta = run_autolabel(cap_dir, track, preset=PRESET, out=tmp_path / "labels")
    rows = [json.loads(x) for x in (tmp_path / "labels" / "labels.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and [r["segment"] for r in rows] == [0, 1]
    assert meta["drop_reasons"] == {"near_cut": 2}
    assert meta["counts"]["frames_total"] == 4 and meta["counts"]["frames_labelled"] == 2


def test_run_autolabel_writes_labels_and_meta(tmp_path):
    cap_dir = _capture(tmp_path)
    track = arch_track(tmp_path / "t.xml", [("1", (0, 0, 30), 0)])
    out = tmp_path / "labels"
    meta = run_autolabel(cap_dir, track, preset=PRESET, out=out)
    on_disk = json.loads((out / "labels_meta.json").read_text(encoding="utf-8"))
    assert on_disk["gate_types_hash"] == TABLE.hash() == meta["gate_types_hash"]
    assert on_disk["capture_name"] == "cap" and on_disk["preset"] == PRESET
    assert on_disk["camera"] == {"resolution": [2560, 1080], "hfov_deg": 120.0, "uptilt_deg": 20.0}
    assert len(on_disk["masks"]) == 6
    assert on_disk["gate_instances"] == [
        {"instance": "1", "item_id": "InflatableArchBrandless01", "cls": "inflatable_arch"}
    ]
    assert on_disk["props_warnings"] == [] and on_disk["race"] is None
    rows = (out / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    g = json.loads(rows[0])["gates"][0]
    assert set(g) >= {
        "cls",
        "keypoints",
        "visible",
        "in_image",
        "in_front",
        "center_px",
        "range_m",
        "bearing_deg",
        "center_world",
        "normal_yaw_deg",
        "flipped",
    }
    assert len(g["keypoints"]) == 10 and meta["previews"] == []


def test_run_autolabel_with_race_records_route(tmp_path):
    from tests.harness_support import DATA_DIR

    cap_dir = _capture(tmp_path)
    meta = run_autolabel(
        cap_dir,
        DATA_DIR / "passages_track.xml",
        DATA_DIR / "race_passages.xml",
        preset=PRESET,
        out=tmp_path / "labels",
    )
    assert meta["route_checkpoints"] == ["8", "5", "7", "6", "8", "4", "3", "1"]
    assert len(meta["gate_instances"]) == 8  # every arch item, on route or not


def test_autolabel_refuses_without_resolution_or_gates(tmp_path):
    cap_dir = _capture(tmp_path)
    track = arch_track(tmp_path / "t.xml", [("1", (0, 0, 30), 0)])
    with pytest.raises(HarnessConfigError, match="resolution"):
        run_autolabel(
            cap_dir,
            track,
            preset=PRESET,
            out=tmp_path / "o",
            config={"camera": {"resolution": None, "masks": []}},
        )
    empty = tmp_path / "empty.xml"
    empty.write_text("<Track><blueprints/></Track>")
    with pytest.raises(HarnessConfigError, match="matches gate_types"):
        run_autolabel(cap_dir, empty, preset=PRESET, out=tmp_path / "o")


# ----------------------------------------------------------------------------- props check


def _arch_props(aperture_half=(3.09, 1.847), bar_y=3.958):
    """Collider shapes in liftoff-flight-analyzer `props` form (dimensions as in the preset)."""
    return {
        "InflatableArchBrandless01": {
            "colliders": [
                {
                    "t": "cap",
                    "p": [0.0, bar_y, 0.0],
                    "q": [0, 0, 0, 1],
                    "r": 0.5,
                    "h": 5.75,
                    "a": 0,
                },
                {
                    "t": "cap",
                    "p": [3.5, 1.58, 0.0],
                    "q": [0, 0, 0, 1],
                    "r": 0.5,
                    "h": 3.521,
                    "a": 1,
                },
                {
                    "t": "cap",
                    "p": [-3.5, 1.58, 0.0],
                    "q": [0, 0, 0, 1],
                    "r": 0.5,
                    "h": 3.521,
                    "a": 1,
                },
                {
                    "t": "box",
                    "p": [0.0, 1.74, 0.0],
                    "q": [0, 0, 0, 1],
                    "s": [aperture_half[0], aperture_half[1], 0.357],
                    "trig": 1,
                },
            ]
        }
    }


def test_collider_bounds_and_trigger_box():
    b = collider_bounds(_arch_props()["InflatableArchBrandless01"]["colliders"])
    np.testing.assert_allclose(b["solid"]["min"][:2], [-4.0, -0.1805], atol=1e-3)
    np.testing.assert_allclose(b["solid"]["max"][:2], [4.0, 4.458], atol=1e-3)
    assert b["trigger"]["center"] == [0.0, 1.74, 0.0]
    np.testing.assert_allclose(b["trigger"]["half"], [3.09, 1.847, 0.357])


def test_props_consistent_with_gate_types_gives_no_warning():
    ids = {"InflatableArchBrandless01", "SpectatorTent01"}
    assert check_gate_types_against_props(TABLE, ids, _arch_props()) == []


def test_props_mismatch_over_5_percent_warns():
    w = check_gate_types_against_props(
        TABLE, {"InflatableArchBrandless01"}, _arch_props(aperture_half=(2.5, 1.847))
    )
    assert len(w) == 1 and "aperture width" in w[0] and "6.180" in w[0]
    missing = check_gate_types_against_props(TABLE, {"InflatableArchBrandless01"}, {})
    assert missing and "no collider shapes" in missing[0]


def test_item_table_props_check_warns_on_size_mismatch():
    table = [{"match": "^Barrier", "kind": "box", "size": [1.0, 1.0, 1.0]}]
    props = {
        "Barrier01": {
            "colliders": [{"t": "box", "p": [0, 0.5, 0], "q": [0, 0, 0, 1], "s": [0.5, 0.5, 1.0]}]
        }
    }
    w = check_item_table_against_props(table, {"Barrier01"}, props)
    assert len(w) == 1 and "size z" in w[0]


def test_run_autolabel_props_warnings_land_in_meta(tmp_path):
    cap_dir = _capture(tmp_path)
    track = arch_track(tmp_path / "t.xml", [("1", (0, 0, 30), 0)])
    props = tmp_path / "props.json"
    props.write_text(json.dumps({"items": _arch_props(bar_y=5.0)}))
    meta = run_autolabel(cap_dir, track, props=props, preset=PRESET, out=tmp_path / "labels")
    assert any("outer height" in w for w in meta["props_warnings"])
    bad = tmp_path / "bad.json"
    bad.write_text("[]")
    with pytest.raises(HarnessConfigError, match="props"):
        run_autolabel(cap_dir, track, props=bad, preset=PRESET, out=tmp_path / "labels")


# ----------------------------------------------------------------------------- real data


def test_real_capture_labels_local_opt_in(tmp_path):
    data = liftoff_real_data()
    meta = run_autolabel(
        data.capture, data.track, data.race, data.props, preset=PRESET, out=tmp_path / "labels"
    )
    c = meta["counts"]
    assert c["frames_total"] == 2043 and c["frames_labelled"] >= 1500
    # ~380 frames with an arch 4-25 m ahead and its centre in the image (use case estimate)
    assert c["gates_4_25m_centre_in_image:inflatable_arch"] >= 300
    assert meta["props_warnings"] == []
    assert meta["route_checkpoints"] == ["8", "5", "7", "6", "8", "4", "3", "1"]
