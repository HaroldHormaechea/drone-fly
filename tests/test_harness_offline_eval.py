"""UC-62 AC8: offline evaluation against track/race-XML ground truth (synthetic fixtures).

Ground-truth estimators must score zero error; perturbed estimators must report the error they
were given. The XML parser is exercised against ``tests/data/harness/{track,race}.xml``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from drone_fly.harness.config import load_config
from drone_fly.harness.course import Course
from drone_fly.harness.course_senses import GateProgress, curr_gate_block, obstacle_vision_block
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.geometry import UNITY_TO_CONTRACT
from drone_fly.harness.interfaces import FreeSpaceEstimator, GateEstimator
from drone_fly.harness.offline.evaluate import (
    EvalReport,
    evaluate_frames,
    ground_truth_targets,
    load_session_frames,
    run_offline_eval,
)
from drone_fly.harness.offline.track_xml import (
    load_track_course,
    parse_race,
    parse_track_items,
    unity_euler_to_matrix,
)
from drone_fly.harness.recording import TELEMETRY_COLUMNS
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import FlightState, Frame, FreeSpaceEstimate, GateEstimate
from tests.harness_support import DATA_DIR

REPO = Path(__file__).resolve().parent.parent
TRACK = DATA_DIR / "track.xml"
RACE = DATA_DIR / "race.xml"
ITEM_TABLE = load_config("liftoff").section("offline")["item_table"]
WORLD_SCALE = 2.0  # the packaged liftoff default


def _course(scale: float = 1.0) -> Course:
    return load_track_course(TRACK, RACE, ITEM_TABLE, WorldScale(scale))


# ----------------------------------------------------------------------------- XML parsing


def test_parse_track_items_children_and_attribute_forms():
    items = parse_track_items(TRACK)
    assert [i.instance_id for i in items] == ["g1", "g2", "g3", "g4", "p1", "b1", "t1"]
    g3 = items[2]
    np.testing.assert_array_equal(g3.position, [0, 0, 15])  # attribute form
    np.testing.assert_allclose(g3.rotation @ [0, 0, 1], [-1, 0, 0], atol=1e-9)
    np.testing.assert_array_equal(items[4].scale, [1, 1, 1])
    np.testing.assert_array_equal(items[4].rotation, np.eye(3))


def test_parse_race_order_and_laps():
    ids, laps = parse_race(RACE)
    assert ids == ["g1", "g2", "g3", "g4"] and laps == 2
    assert parse_race(None) == ([], None)


def test_unity_euler_matches_unity_conventions():
    np.testing.assert_allclose(unity_euler_to_matrix([0, 90, 0]) @ [0, 0, 1], [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(unity_euler_to_matrix([90, 0, 0]) @ [0, 0, 1], [0, -1, 0], atol=1e-9)
    np.testing.assert_allclose(unity_euler_to_matrix([0, 0, 90]) @ [1, 0, 0], [0, 1, 0], atol=1e-9)
    # Z, then X, then Y (Unity order).
    r = unity_euler_to_matrix([90, 90, 0])
    np.testing.assert_allclose(
        r, unity_euler_to_matrix([0, 90, 0]) @ unity_euler_to_matrix([90, 0, 0]), atol=1e-9
    )


def test_track_course_geometry_in_contract_frame():
    c = _course()
    assert c.n_gates == 4 and c.laps == 2
    centers = np.array([g.center for g in c.gates])
    # Unity (x, y + h/2, z) -> contract (z, -x, y + 0.75)
    np.testing.assert_allclose(
        centers, [[5, 0, 0.75], [10, -5, 0.75], [15, 0, 0.75], [10, 5, 0.75]], atol=1e-5
    )
    yaws = [math.degrees(g.yaw) for g in c.gates]
    np.testing.assert_allclose(
        np.cos(np.radians(yaws)), np.cos(np.radians([-90, 0, 90, 180])), atol=1e-6
    )
    np.testing.assert_allclose(
        np.sin(np.radians(yaws)), np.sin(np.radians([-90, 0, 90, 180])), atol=1e-6
    )
    assert all(g.aperture == pytest.approx(1.5) for g in c.gates)
    np.testing.assert_allclose(c.start, centers[0])

    assert len(c.obstacles) == 2  # Tree_Pine is unmatched -> ignored
    pillar, box = c.obstacles
    assert pillar.type == "cylinder"
    np.testing.assert_allclose(pillar.center, [7.5, -1.0, 1.5], atol=1e-6)
    np.testing.assert_allclose(pillar.half_extents, [0.3, 0.3, 1.5])
    assert box.type == "box"
    np.testing.assert_allclose(box.center, [12.0, 1.0, 0.5], atol=1e-6)
    s = math.sqrt(0.5)
    np.testing.assert_allclose(box.half_extents, [s, s, 0.5], atol=1e-6)  # 45 deg -> AABB


def test_world_scale_applies_to_xml_geometry():
    a, b = _course(1.0), _course(WORLD_SCALE)
    for ga, gb in zip(a.gates, b.gates, strict=True):
        np.testing.assert_allclose(gb.center, ga.center / WORLD_SCALE, atol=1e-6)
        assert gb.aperture == pytest.approx(ga.aperture / WORLD_SCALE)
        assert gb.yaw == pytest.approx(ga.yaw)
    np.testing.assert_allclose(
        b.obstacles[0].half_extents, a.obstacles[0].half_extents / WORLD_SCALE
    )


def test_race_reorders_and_validates(tmp_path):
    race = tmp_path / "race.xml"
    race.write_text(
        "<Race><instanceID>g3</instanceID><instanceID>g1</instanceID></Race>", encoding="utf-8"
    )
    c = load_track_course(TRACK, race, ITEM_TABLE, WorldScale(1.0))
    np.testing.assert_allclose([g.center[0] for g in c.gates], [15, 5], atol=1e-5)
    assert c.laps == 1
    assert load_track_course(TRACK, race, ITEM_TABLE, WorldScale(1.0), laps=3).laps == 3

    bad = tmp_path / "bad_race.xml"
    bad.write_text(
        "<Race><instanceID>g1</instanceID><instanceID>zz</instanceID></Race>", encoding="utf-8"
    )
    with pytest.raises(HarnessConfigError, match="zz"):
        load_track_course(TRACK, bad, ITEM_TABLE, WorldScale(1.0))
    no_race = load_track_course(TRACK, None, ITEM_TABLE, WorldScale(1.0))
    assert no_race.n_gates == 4  # track order


def test_item_table_errors(tmp_path):
    with pytest.raises(HarnessConfigError, match="fewer than 2 gates"):
        load_track_course(
            TRACK,
            None,
            [{"match": "Pillar", "kind": "cylinder", "radius": 1, "height": 1}],
            WorldScale(1.0),
        )
    with pytest.raises(HarnessConfigError, match="kind"):
        load_track_course(
            TRACK,
            None,
            [{"match": "Gate", "kind": "gate"}, {"match": "Cube", "kind": "cone"}],
            WorldScale(1.0),
        )
    with pytest.raises(HarnessConfigError, match="match"):
        load_track_course(TRACK, None, [{"kind": "gate"}], WorldScale(1.0))
    ignore = load_track_course(
        TRACK,
        None,
        [{"match": "Gate", "kind": "gate"}, {"match": ".*", "kind": "ignore"}],
        WorldScale(1.0),
    )
    assert ignore.obstacles == ()
    broken = tmp_path / "broken.xml"
    broken.write_text("<Track><unclosed>", encoding="utf-8")
    with pytest.raises(HarnessConfigError, match="cannot read track XML"):
        parse_track_items(broken)


# ----------------------------------------------------------------------------- synthetic session


def _path_states(course: Course) -> list[tuple[np.ndarray, np.ndarray]]:
    """Training-scale (pos, vel) along a path through gate 0 toward gate 1, passing the pillar."""
    g0, g1 = course.gates[0].center.astype(np.float64), course.gates[1].center.astype(np.float64)
    start = g0 + np.array([-2.0, 0.0, 0.05])
    out = []
    for leg_a, leg_b in ((start, g0), (g0, g0 + 0.8 * (g1 - g0))):
        d = leg_b - leg_a
        for k in range(20):
            out.append((leg_a + d * k / 20, d / (20 * 0.02)))
    return out


def _write_session(tmp_path: Path, course: Course, scale: float) -> Path:
    session = tmp_path / "liftoff-20261011T000000000000Z"
    (session / "frames").mkdir(parents=True)
    m_t = UNITY_TO_CONTRACT.astype(np.float64).T
    with (session / "telemetry.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(TELEMETRY_COLUMNS)
        for i, (p, v) in enumerate(_path_states(course)):
            pu, vu = m_t @ (scale * p), m_t @ (scale * v)
            w.writerow([100.0 + i * 0.02, i * 0.02, *pu, 0, 0, 0, 1, *vu, 0, 0, 0] + [""] * 10)
    with (session / "frames.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tick", "frame_index", "stamp", "width", "height", "file"])
        for i in range(len(_path_states(course))):
            name = f"frames/{i:07d}.npy"
            np.save(session / name, np.zeros((24, 32, 3), np.uint8))
            w.writerow([i, i + 1, 100.0 + i * 0.02, 32, 24, name])
    return session


class TruthGate(GateEstimator):
    provides = frozenset({"target", "curr_gate", "next_lookahead"})

    def __init__(self, course: Course, offset=(0.0, 0.0, 0.0), yaw_offset=0.0):
        self.course, self.prog = course, GateProgress(course)
        self.offset, self.yaw_offset = np.array(offset, np.float32), yaw_offset

    def update(self, frame, state, now):
        self.prog.advance_if_passed(state.pos)
        g = self.prog.current
        cg = curr_gate_block(g)
        y = g.yaw + self.yaw_offset
        cg[:2] = [math.cos(y), math.sin(y)]
        return GateEstimate(
            target=g.center + self.offset,
            curr_gate=cg,
            next_lookahead=np.zeros(7, np.float32),
            info={"matched": True},
        )


class TruthFreeSpace(FreeSpaceEstimator):
    provides = frozenset({"obstacle_vision", "raycast"})

    def __init__(self, course: Course, contract, factor=1.0):
        self.course, self.contract, self.factor = course, contract, factor

    def update(self, frame, state, target, now):
        rays = self.contract.raycast(state.as_state_dict(), self.course.raycast_obstacles(), target)
        return FreeSpaceEstimate(
            obstacle_vision=obstacle_vision_block(state.pos, self.course.obstacles),
            raycast=(np.asarray(rays) * self.factor).astype(np.float32),
            in_fov_mask=np.ones(40, bool),
        )


def _triples(course: Course):
    states = [
        FlightState(
            pos=p.astype(np.float32),
            vel=v.astype(np.float32),
            quat=np.array([1, 0, 0, 0], np.float32),
            omega=np.zeros(3, np.float32),
        )
        for p, v in _path_states(course)
    ]
    truth = ground_truth_targets(states, course)
    return [
        (Frame(np.zeros((2, 2, 3), np.uint8), i * 0.02, i), s, t)
        for i, (s, t) in enumerate(zip(states, truth, strict=True))
    ], truth


def test_ground_truth_targets_follow_pass_rule_and_reset_on_discontinuity():
    course = _course(WORLD_SCALE)
    triples, truth = _triples(course)
    assert truth[0] == 0 and truth[-1] == 1  # gate 0 passed along the path
    s = [t[1] for t in triples]
    from dataclasses import replace

    s2 = [*s, replace(s[0], discontinuity=True)]
    assert ground_truth_targets(s2, course)[-1] == 0


def test_ground_truth_estimators_score_zero_error(harness_contract):
    course = _course(WORLD_SCALE)
    triples, _ = _triples(course)
    rep = evaluate_frames(
        triples,
        course,
        harness_contract,
        WorldScale(WORLD_SCALE),
        TruthGate(course),
        TruthFreeSpace(course, harness_contract),
    )
    assert rep.n_frames == 40 and rep.gate["detection_rate"] == 1.0
    assert rep.gate["position_error_m_mean"] == pytest.approx(0.0, abs=1e-6)
    assert rep.gate["yaw_error_deg_mean"] == pytest.approx(0.0, abs=1e-4)
    f = rep.free_space
    assert f["ray_mae"] == pytest.approx(0.0, abs=1e-7) and f["ray_mae_in_fov"] == pytest.approx(
        0.0, abs=1e-7
    )
    assert f["ray_near_weighted_error"] == pytest.approx(0.0, abs=1e-7)
    assert f["obstacle_vision_mae"] == pytest.approx(0.0, abs=1e-7) and f["coverage_mean"] == 1.0
    assert rep.depth_scale_fit == pytest.approx(1.0)  # the path sees the pillar


def test_perturbed_estimators_report_their_error(harness_contract):
    course = _course(WORLD_SCALE)
    triples, _ = _triples(course)
    rep = evaluate_frames(
        triples,
        course,
        harness_contract,
        WorldScale(WORLD_SCALE),
        TruthGate(course, offset=(0.1, 0.0, 0.0), yaw_offset=math.radians(10)),
        TruthFreeSpace(course, harness_contract, factor=0.9),
    )
    g = rep.gate
    assert g["position_error_training_m_mean"] == pytest.approx(0.1, abs=1e-5)
    assert g["position_error_m_mean"] == pytest.approx(0.2, abs=1e-5)  # target (Liftoff) metres
    assert g["yaw_error_deg_mean"] == pytest.approx(10.0, abs=1e-3)
    assert rep.free_space["ray_mae"] > 0.05
    assert rep.depth_scale_fit == pytest.approx(1 / 0.9, rel=1e-4)
    text = rep.summary()
    assert "detection rate 100.0%" in text and "suggested depth calibration scale" in text


def test_undetected_frames_lower_detection_rate(harness_contract):
    course = _course(WORLD_SCALE)
    triples, _ = _triples(course)

    class Blind(TruthGate):
        def update(self, frame, state, now):
            est = super().update(frame, state, now)
            from dataclasses import replace

            return replace(est, info={"matched": frame.index % 2 == 0})

    rep = evaluate_frames(
        triples, course, harness_contract, WorldScale(WORLD_SCALE), Blind(course), None
    )
    assert rep.gate["detection_rate"] == pytest.approx(0.5) and rep.free_space is None
    assert rep.depth_scale_fit is None


def test_run_offline_eval_end_to_end_writes_report(harness_contract, tmp_path):
    course = _course(WORLD_SCALE)
    session = _write_session(tmp_path, course, WORLD_SCALE)
    assert len(load_session_frames(session)) == 40
    report = run_offline_eval(
        session,
        TRACK,
        RACE,
        contract=harness_contract,
        gate_estimator=TruthGate(course),
        free_space_estimator=TruthFreeSpace(course, harness_contract),
    )
    assert isinstance(report, EvalReport) and report.n_frames == 40
    assert report.gate["detection_rate"] == 1.0
    assert report.gate["position_error_m_mean"] == pytest.approx(0.0, abs=1e-4)
    assert report.free_space["ray_mae"] == pytest.approx(0.0, abs=1e-4)
    assert report.gyro is not None and "suggested" in report.gyro
    out = session / "eval_report.json"
    assert report.report_path == str(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_frames"] == 40 and "suggested_depth_scale_multiplier" in data
    custom = tmp_path / "custom_report.json"
    run_offline_eval(
        session,
        TRACK,
        RACE,
        report_path=custom,
        contract=harness_contract,
        gate_estimator=TruthGate(course),
    )
    assert json.loads(custom.read_text(encoding="utf-8"))["free_space"] is None


def test_run_offline_eval_builds_depth_estimator_with_injected_model(harness_contract, tmp_path):
    from drone_fly.harness.interfaces import DepthModel

    class Far(DepthModel):
        def predict(self, image):
            return np.full(image.shape[:2], 50.0, np.float32)

    course = _course(WORLD_SCALE)
    session = _write_session(tmp_path, course, WORLD_SCALE)
    report = run_offline_eval(
        session,
        TRACK,
        RACE,
        config={"parts": {"free_space_estimator": "depth"}},
        contract=harness_contract,
        gate_estimator=TruthGate(course),
        depth_model=Far(),
    )
    f = report.free_space
    assert f is not None and 0.0 <= f["ray_mae"] < 1.0 and 0.0 <= f["coverage_mean"] <= 1.0


def test_session_folder_validation(tmp_path, harness_contract):
    with pytest.raises(HarnessConfigError, match="frames.csv"):
        load_session_frames(tmp_path)


def test_cli_eval_in_fresh_process(tmp_path):
    """`drone-fly harness eval` runs (exit 0 with OpenCV, exit 3 naming it without)."""
    import importlib.util

    session = _write_session(tmp_path, _course(WORLD_SCALE), WORLD_SCALE)
    report = tmp_path / "r.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "drone_fly.cli.main",
            "harness",
            "eval",
            "--session",
            str(session),
            "--track",
            str(TRACK),
            "--race",
            str(RACE),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if importlib.util.find_spec("cv2") is None:
        assert proc.returncode == 3, (proc.stdout, proc.stderr)
        assert "opencv-python" in proc.stderr and "uv sync --extra harness" in proc.stderr
    else:
        assert proc.returncode == 0, proc.stderr
        assert "frames evaluated: 40" in proc.stdout
        assert json.loads(report.read_text(encoding="utf-8"))["gate"]["detection_rate"] == 0.0
