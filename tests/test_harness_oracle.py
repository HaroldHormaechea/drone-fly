"""UC69 AC1 / AC3 / AC8: the oracle gate source on a synthetic course + synthetic telemetry.

Poses, the route (in-aperture travel-direction crossings, revisits, volumes, multi-lap circuits
with the finish re-crossing), the Unity -> course-frame mapping, the tick-0 start check, every
episode-monitor end, the respawn rule and the course-load refusals. Hermetic: the course is built
by hand (``tests/oracle_support.py``); the catalogue load is faked where the real one would read
game data. The two real-data anchors at the end are local opt-in (``$DRONE_FLY_TRACKDATA``).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from drone_fly.harness.course_senses import curr_gate_block, next_lookahead_block
from drone_fly.harness.errors import (
    EstimatorStop,
    HarnessConfigError,
    StartMisalignedError,
)
from drone_fly.harness.geometry import quat_to_matrix
from drone_fly.harness.liftoff.telemetry import GyroMap, LiftoffConverter
from drone_fly.harness.oracle import (
    EPISODE_END_REASONS,
    ORACLE_LABEL,
    PERCEPTION,
    EpisodeRules,
    OracleCourseError,
    OracleGateSource,
    StartTolerance,
    load_oracle_course,
    next_lookahead_fast,
)
from drone_fly.harness.scale import WorldScale
from drone_fly.tracks.teacher_obs_ref import to_harness_course
from tests.oracle_support import (
    ArmClock,
    StateSourceStub,
    conversion_record,
    loop_passages,
    make_course,
    opening,
    state,
    tilt_quat,
    unity_sample,
    volume,
    yaw_quat,
)

DT = 0.02


#: Most tests start in the air beside the route: a start check that accepts that.
LOOSE = StartTolerance(horizontal_m=50.0, dz_m=50.0, yaw_deg=180.0, speed_mps=100.0)


def source(course, *, arm=None, src=None, tol=LOOSE, rules=None):
    return OracleGateSource(
        course,
        purpose="eval",
        start_tolerance=tol,
        rules=rules,
        state_source=src if src is not None else StateSourceStub(),
        arm_clock=arm if arm is not None else ArmClock(0.0),
        course_info={"track_id": course.id},
    )


def fly(src, course, points, *, t0=0.0, dt=DT, **kw):
    """Feed course-frame positions (one per tick); returns the estimates."""
    out = []
    for i, p in enumerate(points):
        out.append(src.update(None, state(course, p, **kw), t0 + i * dt))
    return out


def line(a, b, n):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def hover_through(src, course, points, *, t0=0.0, dt=DT):
    """Like :func:`fly` with a velocity consistent with the step (keeps the monitor quiet)."""
    out = []
    for i, p in enumerate(points):
        v = (np.asarray(points[min(i + 1, len(points) - 1)]) - np.asarray(points[max(i - 1, 0)]))
        v = v / (2 * dt) if len(points) > 1 else v
        out.append(src.update(None, state(course, p, vel=v), t0 + i * dt))
    return out


# ----------------------------------------------------------------------------- poses


def test_estimate_poses_and_blocks():
    course = make_course()
    src = source(course)
    assert src.provides == {"target", "curr_gate", "next_lookahead", "oracle"}
    p = np.array([1.0, 0.5, 1.2])
    q = yaw_quat(0.3)
    est = src.update(None, state(course, p, quat=q), 0.0)
    # target: passage 0 centre in the CONTRACT frame
    np.testing.assert_allclose(est.target, course.frame.to_contract_point([5.0, 0.0, 1.5]))
    hc = to_harness_course(course)
    np.testing.assert_allclose(est.curr_gate, curr_gate_block(hc.gates[0]))
    np.testing.assert_allclose(
        est.next_lookahead, next_lookahead_block(p, hc.gates[1], hc.obstacles), atol=1e-6
    )
    assert (est.gate_index, est.lap, est.status, est.passed) == (0, 0, "tracking", False)
    info = est.info
    assert info["perception"] == PERCEPTION == "oracle"
    assert info["k"] == 0 and info["total"] == 2 and info["crossed"] is None
    np.testing.assert_allclose(info["pos_course"], p, atol=1e-5)
    assert info["altitude_m"] == pytest.approx(1.2, abs=1e-5)
    # body-frame poses of the next passages (3 rows; past the route end they are invalid)
    R = quat_to_matrix(q)
    rows = info["ahead"]
    assert [r["passage"] for r in rows] == [0, 1, 0]
    assert [r["valid"] for r in rows] == [True, True, False]
    np.testing.assert_allclose(rows[0]["rel_body"], R.T @ (np.array([5.0, 0, 1.5]) - p), atol=1e-5)
    np.testing.assert_allclose(rows[0]["normal_body"], R.T @ np.array([1.0, 0, 0]), atol=1e-6)
    assert rows[1]["dist_m"] == pytest.approx(np.linalg.norm([9.0, -0.5, 0.3]), abs=1e-5)


def test_state_course_maps_contract_to_course_frame():
    course = make_course()
    src = source(course)
    s = state(course, [2.0, 1.0, 1.0], vel=[1.0, 2.0, 3.0], omega=[0.1, 0.2, 0.3])
    src.update(None, s, 0.0)
    sc = src.state_course
    np.testing.assert_allclose(sc["pos"], [2.0, 1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(sc["vel"], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(sc["omega"], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(sc["quat"], [1, 0, 0, 0])


@pytest.mark.parametrize("seed", range(5))
def test_next_lookahead_fast_equals_reference(seed):
    """The vectorised occlusion test gives the course_senses result (visible and occluded)."""
    rng = np.random.default_rng(seed)
    course = make_course()
    hc = to_harness_course(course)
    centers = np.array([o.center for o in hc.obstacles], np.float32)
    sizes = np.array([np.max(o.half_extents) for o in hc.obstacles], np.float32)
    gate = opening((7.5, -6.0, 1.0), -math.pi / 2)
    from drone_fly.harness.course import Gate

    g = Gate(gate.center.astype(np.float32), -math.pi / 2, 2.0, None)
    seen = set()
    for _ in range(200):
        p = rng.uniform([4, -1, 0.2], [11, 1, 2.5]).astype(np.float32)
        a = next_lookahead_fast(p, g, centers, sizes)
        b = next_lookahead_block(p, g, hc.obstacles)
        np.testing.assert_allclose(a, b, atol=1e-6)
        seen.add(bool(b[-1]))
    assert seen == {True, False}  # both branches exercised (the box sits on the segment)


# ----------------------------------------------------------------------------- crossings


def test_crossing_inside_aperture_advances():
    course = make_course()
    src = source(course)
    est = hover_through(src, course, line([0, 0, 1.5], [6, 0.5, 1.5], 61))
    passed = [e for e in est if e.passed]
    assert len(passed) == 1 and passed[0].info["crossed"] == 0
    assert passed[0].info["k"] == 1 and passed[0].gate_index == 1
    assert est[-1].gate_index == 1
    np.testing.assert_allclose(est[-1].target, course.frame.to_contract_point([10, 0, 1.5]))
    assert src.crossings == [{"k": 1, "passage": 0, "t_s": pytest.approx(passed[0].stamp)}]


@pytest.mark.parametrize(
    "a,b",
    [
        ([0, 1.3, 1.5], [6, 1.3, 1.5]),  # beside the opening (half-width 1.0)
        ([0, 0, 2.7], [6, 0, 2.7]),  # above it
    ],
)
def test_crossing_outside_aperture_does_not_advance(a, b):
    course = make_course()
    src = source(course)
    est = hover_through(src, course, line(a, b, 61))
    assert not any(e.passed for e in est) and src.progress.k == 0


def test_backwards_crossing_does_not_count():
    course = make_course()
    src = source(course)
    est = hover_through(src, course, line([7, 0, 1.5], [3, 0, 1.5], 41))
    assert not any(e.passed for e in est) and src.progress.k == 0
    # ... and flying it forwards afterwards does
    est = hover_through(src, course, line([3, 0, 1.5], [7, 0, 1.5], 41), t0=1.0)
    assert sum(e.passed for e in est) == 1


def test_first_tick_never_crosses():
    """A first sample already past the plane is not a crossing (no previous position)."""
    course = make_course()
    src = source(course)
    est = src.update(None, state(course, [5.1, 0, 1.5]), 0.0)
    assert not est.passed and src.progress.k == 0


def test_only_the_target_passage_counts():
    """Crossing passage 1 while passage 0 is the target does nothing (route order)."""
    course = make_course()
    src = source(course)
    est = hover_through(src, course, line([8, 0, 1.5], [11, 0, 1.5], 31))
    assert not any(e.passed for e in est) and src.progress.k == 0


def test_revisit_route():
    """The same physical gate twice (out and back): the second visit is a separate passage."""
    passages = [
        opening((5.0, 0.0, 1.5), 0.0),
        opening((10.0, 0.0, 1.5), 0.0),
        opening((5.0, 0.0, 1.5), math.pi),  # back through gate 0
    ]
    course = make_course(passages)
    src = source(course)
    path = line([0, 0, 1.5], [11, 0, 1.5], 111) + line([11, 0, 1.5], [3, 0, 1.5], 81)[1:]
    est = hover_through(src, course, path)
    crossed = [e.info["crossed"] for e in est if e.passed]
    assert crossed == [0, 1, 2]
    assert est[-1].status == "finished" and src.progress.finished


def test_volume_passage_enters():
    passages = [volume((5.0, 0.0, 1.5), 0.0, (1.0, 1.0, 1.0)), opening((10.0, 0.0, 1.5), 0.0)]
    course = make_course(passages)
    src = source(course)
    est = hover_through(src, course, line([0, 0, 1.5], [6, 0, 1.5], 61))
    assert [e.info["crossed"] for e in est if e.passed] == [0]
    # beside the volume: no entry
    src = source(course)
    est = hover_through(src, course, line([0, 2.5, 1.5], [8, 2.5, 1.5], 81))
    assert not any(e.passed for e in est)


def _loop_path(laps: int) -> list[np.ndarray]:
    corners = [[0, 0, 1.5], [10, 0, 1.5], [10, 10, 1.5], [0, 10, 1.5], [0, 0, 1.5]]
    pts: list[np.ndarray] = []
    for _ in range(laps):
        for a, b in zip(corners[:-1], corners[1:], strict=True):
            pts += line(a, b, 51)[(1 if pts else 0) :]
    return pts


def test_multi_lap_circuit_and_finish_recrossing():
    course = make_course(loop_passages(), laps=2, circuit=True)
    assert course.total_crossings == 9
    src = source(course)
    path = _loop_path(2) + line([0, 0, 1.5], [7, 0, 1.5], 36)[1:]
    est = hover_through(src, course, path)
    passed = [e for e in est if e.passed]
    assert [e.info["crossed"] for e in passed] == [0, 1, 2, 3, 0, 1, 2, 3, 0]
    # lap counts: a circuit's first crossing starts lap 0
    assert [e.lap for e in passed] == [0, 0, 0, 0, 1, 1, 1, 1, 2]
    assert passed[-1].status == "finished"
    assert src.summary()["finished"] is True and src.summary()["k"] == 9


def test_finished_course_is_not_ended_by_the_monitor():
    """After the finish the episode monitor is off (the runner stops on ``finished``)."""
    course = make_course(time_limit_s=1.0)
    src = source(course)
    est = hover_through(src, course, line([0, 0, 1.5], [11, 0, 1.5], 45))
    assert est[-1].status == "finished"
    # past the time limit, finished: no EstimatorStop
    e = src.update(None, state(course, [11.2, 0, 1.5]), 5.0)
    assert e.status == "finished"


# ----------------------------------------------------------------------------- frames


def test_unity_telemetry_maps_through_the_converter_to_the_course_frame():
    course = make_course()
    conv = LiftoffConverter(GyroMap(), WorldScale(1.0))
    s = unity_sample(course, [3.0, -1.0, 2.0], t=0.0, vel=[1.0, 0.5, -0.2], quat=yaw_quat(0.4))
    fs = conv.convert(s)
    src = source(course, tol=StartTolerance(horizontal_m=10.0, dz_m=10.0, yaw_deg=90.0,
                                            speed_mps=5.0))  # fmt: skip
    src.update(None, fs, 0.0)
    np.testing.assert_allclose(src.state_course["pos"], [3.0, -1.0, 2.0], atol=1e-4)
    np.testing.assert_allclose(src.state_course["vel"], [1.0, 0.5, -0.2], atol=1e-5)
    q = src.state_course["quat"]
    assert math.atan2(*quat_to_matrix(q)[[1, 0], 0]) == pytest.approx(0.4, abs=1e-5)


def test_world_scale_four_regression_misaligns_the_start():
    """The preset's world_scale 4.0 divides telemetry positions: the oracle must run at 1.0.

    At 1.0 the telemetry spawn matches the XML start; through a 4.0 converter the same sample
    lands far from it and the start check refuses (the session forces 1.0, see
    ``test_harness_teacher_policy``)."""
    course = make_course(origin=(-37.7, 0.01, 0.87))
    s = unity_sample(course, [0.0, 0.0, 0.03], t=0.0)
    ok = LiftoffConverter(GyroMap(), WorldScale(1.0)).convert(s)
    source(course, tol=StartTolerance()).update(None, ok, 0.0)
    bad = LiftoffConverter(GyroMap(), WorldScale(4.0)).convert(s)
    with pytest.raises(StartMisalignedError, match="horizontal"):
        source(course, tol=StartTolerance()).update(None, bad, 0.0)


def test_gyro_map_anchor_roll_sign_and_units():
    """Liftoff's +90 deg/s roll is a NEGATIVE contract wx of pi/2 rad/s (UC64 measured)."""
    om = GyroMap().apply(np.array([0.0, 90.0, 0.0]))  # pitch, roll, yaw
    np.testing.assert_allclose(om, [-math.pi / 2, 0.0, 0.0], atol=1e-6)
    om = GyroMap().apply(np.array([90.0, 0.0, 90.0]))
    np.testing.assert_allclose(om, [0.0, math.pi / 2, -math.pi / 2], atol=1e-6)


# ----------------------------------------------------------------------------- start check


def test_start_alignment_passes_and_records_offsets():
    course = make_course()
    src = source(course, tol=StartTolerance())
    src.update(None, state(course, [0.5, 0.3, 0.04], quat=yaw_quat(math.radians(10))), 0.0)
    off = src.start_offsets
    assert off["horizontal_m"] == pytest.approx(math.hypot(0.5, 0.3), abs=1e-5)
    assert off["dz_m"] == pytest.approx(0.04, abs=1e-5)
    assert off["yaw_deg"] == pytest.approx(10.0, abs=1e-3)
    assert off["speed_mps"] == pytest.approx(0.0)
    assert src.summary()["start_offsets"] == off
    assert src.summary()["start_tolerance"] == StartTolerance().as_dict()


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"p": [1.2, 0.5, 0.0]}, "horizontal"),
        ({"p": [0.0, 0.0, 1.3]}, r"\|dz\|"),
        ({"p": [0.0, 0.0, -1.3]}, r"\|dz\|"),
        ({"quat": yaw_quat(math.radians(25))}, "yaw"),
        ({"quat": yaw_quat(math.radians(-170))}, "yaw"),
        ({"vel": [0.3, 0.0, 0.0]}, "not at rest"),
    ],
)
def test_start_misaligned_refuses_at_tick_zero(kw, match):
    course = make_course()
    src = source(course, tol=StartTolerance())
    p = kw.pop("p", [0.0, 0.0, 0.03])
    with pytest.raises(StartMisalignedError, match=match) as ei:
        src.update(None, state(course, p, **kw), 0.0)
    err = ei.value
    assert isinstance(err, EstimatorStop) and err.reason == "start_misaligned"
    assert src.end["reason"] == "start_misaligned" and src.end["t_s"] == 0.0
    assert src.start_offsets is not None  # recorded even on refusal


def test_start_check_runs_only_at_tick_zero():
    course = make_course()
    src = source(course, tol=StartTolerance())
    src.update(None, state(course, [0, 0, 0.03]), 0.0)
    # later ticks far from the start (flying) never re-check
    src.update(None, state(course, [1.0, 0.0, 1.5], vel=[3, 0, 0]), DT)


def test_start_tolerance_from_config_validation():
    tol = StartTolerance.from_config({"horizontal_m": 2.0})
    assert tol.horizontal_m == 2.0 and tol.yaw_deg == 20.0
    with pytest.raises(HarnessConfigError, match="unknown oracle.start_tolerance"):
        StartTolerance.from_config({"horizontal": 1.0})
    with pytest.raises(HarnessConfigError):
        StartTolerance.from_config({"dz_m": 0.0})


def test_packaged_config_tolerances_match_the_plan():
    from drone_fly.harness.config import load_config

    cfg = load_config("liftoff")
    tol = StartTolerance.from_config(cfg.get("oracle")["start_tolerance"])
    assert tol == StartTolerance(1.0, 1.0, 20.0, 0.3)


# ----------------------------------------------------------------------------- episode ends


def _stop(src, course, seq, t0=0.0):
    """Feed ``(t, pos, kw)`` ticks; return the EstimatorStop (fails if none)."""
    with pytest.raises(EstimatorStop) as ei:
        for t, p, kw in seq:
            src.update(None, state(course, p, **kw), t0 + t)
    return ei.value


def _takeoff(t_end=0.6, z=1.5):
    """Ground start then a climb to ``z`` by ``t_end`` (lifts off under both rules)."""
    seq = [(0.0, [0, 0, 0.03], {})]
    n = 20
    for i in range(1, n + 1):
        t = t_end * i / n
        seq.append((t, [0, 0, 0.03 + (z - 0.03) * i / n], {"vel": [0, 0, (z - 0.03) / t_end]}))
    return seq


def test_episode_end_time_limit():
    course = make_course(time_limit_s=1.0)
    src = source(course)
    seq = _takeoff() + [(0.6 + 0.02 * i, [0, 0, 1.5], {"vel": [0.6, 0, 0]}) for i in range(1, 40)]
    stop = _stop(src, course, seq)
    assert stop.reason == "episode_end:time_limit"
    assert src.end["reason"] == "time_limit" and src.end["t_s"] == pytest.approx(1.0, abs=0.03)


@pytest.mark.parametrize("p", [[0, 0, 30.0], [-15.0, -15.0, 1.5]])
def test_episode_end_out_of_bounds_and_corridor(p):
    course = make_course(corridor_m=10.0)
    src = source(course)
    seq = _takeoff() + [(0.62, p, {"vel": [0.6, 0, 0]})]
    # the jump guard is 5 m: approach in small steps
    start = np.array([0, 0, 1.5])
    path = line(start, p, 60)
    seq = _takeoff() + [(0.6 + 0.02 * (i + 1), q, {"vel": [1, 0, 0]}) for i, q in enumerate(path)]
    stop = _stop(src, course, seq)
    assert stop.reason == "episode_end:out_of_bounds"


def test_episode_end_crash_settled():
    course = make_course()
    src = source(course)
    seq = _takeoff()
    # comes down and sits still on the ground for > 2 s
    seq += [(0.6 + 0.02 * i, [0.5, 0, 0.1], {}) for i in range(1, 120)]
    stop = _stop(src, course, seq)
    assert stop.reason == "episode_end:crash_settled"
    assert src.end["t_s"] == pytest.approx(0.62 + 2.0, abs=0.03)


def test_episode_end_inverted():
    course = make_course()
    src = source(course)
    seq = _takeoff(z=0.6)
    q = tilt_quat("x", math.radians(150))
    seq += [(0.6 + 0.02 * i, [0, 0, 0.6], {"quat": q, "vel": [0.6, 0, 0]}) for i in range(1, 80)]
    stop = _stop(src, course, seq)
    assert stop.reason == "episode_end:inverted"
    assert src.end["t_s"] == pytest.approx(0.62 + 1.0, abs=0.03)


def test_inverted_high_up_does_not_end():
    """A flip above 1 m (a recoverable roll) is not an end."""
    course = make_course()
    src = source(course)
    seq = _takeoff(z=2.0)
    q = tilt_quat("x", math.radians(150))
    for t, p, kw in seq + [(0.6 + 0.02 * i, [0, 0, 2.0], {"quat": q, "vel": [0.6, 0, 0]})
                           for i in range(1, 80)]:  # fmt: skip
        src.update(None, state(course, p, **kw), t)
    assert src.end is None


def test_episode_end_no_liftoff_on_the_pad():
    """Armed, sitting on the (raised) pad: ends 5 s after arming ended, never before."""
    course = make_course(pad=True)
    arm = ArmClock(0.5, t_press=0.0)
    src = source(course, arm=arm)
    seq = [(0.02 * i, [0, 0, 0.03], {}) for i in range(0, 400)]
    stop = _stop(src, course, seq)
    assert stop.reason == "episode_end:no_liftoff"
    assert src.arm_end_s == pytest.approx(0.5, abs=0.021)
    assert src.end["t_s"] == pytest.approx(5.5, abs=0.03)
    assert src.liftoff == {"sim_rule_s": None, "threshold_s": None}


def test_episode_end_no_liftoff_tipped_over():
    course = make_course()
    src = source(course, arm=ArmClock(0.5))
    q = tilt_quat("y", math.radians(90))
    seq = [(0.02 * i, [0, 0, 0.05], {"quat": q}) for i in range(0, 400)]
    stop = _stop(src, course, seq)
    assert stop.reason == "episode_end:no_liftoff"


def test_no_liftoff_window_starts_when_arming_ends():
    """Before the Arm press the arming time is still full: the 5 s window has not started."""
    course = make_course()
    src = source(course, arm=ArmClock(0.5, t_press=10.0))
    for i in range(0, 400):  # 8 s on the ground, not yet armed
        src.update(None, state(course, [0, 0, 0.03]), 0.02 * i)
    assert src.end is None and src.arm_end_s is None


def test_liftoff_recorded_under_both_rules():
    course = make_course()
    src = source(course, arm=ArmClock(0.2))
    for t, p, kw in [(0.02 * i, [0, 0, 0.03], {}) for i in range(12)]:  # armed by t 0.2
        src.update(None, state(course, p, **kw), t)
    for t, p, kw in _takeoff(t_end=1.5, z=1.0)[1:]:
        src.update(None, state(course, p, **kw), 0.22 + t)
    lo = src.summary()["liftoff"]
    assert src.arm_end_s == pytest.approx(0.2, abs=0.021)
    assert lo["sim_rule_s"] is not None and lo["threshold_s"] is not None
    assert 0.2 <= lo["sim_rule_s"] < lo["threshold_s"]  # the sim rule fires first (2 cm)
    assert src.end is None


def test_sim_rule_liftoff_needs_a_climb():
    """Above the resting height but sinking is not a lift-off under the sim rule."""
    course = make_course()
    src = source(course, arm=ArmClock(0.0))
    src.update(None, state(course, [0, 0, 0.03]), 0.0)
    src.update(None, state(course, [0, 0, 0.1], vel=[0, 0, -0.2]), 0.02)
    assert src.liftoff["sim_rule_s"] is None
    src.update(None, state(course, [0, 0, 0.12], vel=[0, 0, 0.3]), 0.04)
    assert src.liftoff["sim_rule_s"] == pytest.approx(0.04)


def test_respawn_on_discontinuity_count():
    course = make_course()
    ss = StateSourceStub()
    src = source(course, src=ss)
    for t, p, kw in _takeoff():
        src.update(None, state(course, p, **kw), t)
    ss.discontinuity_count += 1
    with pytest.raises(EstimatorStop, match="respawn"):
        src.update(None, state(course, [0, 0, 1.5]), 0.7)
    assert src.end["reason"] == "respawn"


def test_respawn_on_runner_reset_hook():
    course = make_course()
    src = source(course)
    src.reset()  # before the first tick: nothing to end
    for t, p, kw in _takeoff():
        src.update(None, state(course, p, **kw), t)
    src.reset()
    with pytest.raises(EstimatorStop) as ei:
        src.update(None, state(course, [0, 0, 1.5]), 0.7)
    assert ei.value.reason == "episode_end:respawn"


def test_respawn_on_position_jump():
    course = make_course()
    src = source(course)
    for t, p, kw in _takeoff():
        src.update(None, state(course, p, **kw), t)
    with pytest.raises(EstimatorStop, match="jumped"):
        src.update(None, state(course, [5.5, 0, 1.5]), 0.62)


def test_respawn_teleport_through_the_aperture_never_advances():
    """A respawn whose segment crosses the target plane inside the opening is not a pass."""
    course = make_course()
    ss = StateSourceStub()
    src = source(course, src=ss)
    src.update(None, state(course, [3.0, 0, 1.5]), 0.0)
    ss.discontinuity_count = 1
    with pytest.raises(EstimatorStop):
        src.update(None, state(course, [7.0, 0, 1.5]), 0.02)  # 4 m: under the jump guard
    assert src.progress.k == 0 and src.crossings == []


def test_every_episode_end_reason_is_documented():
    assert set(EPISODE_END_REASONS) == {
        "time_limit", "out_of_bounds", "crash_settled", "inverted", "respawn", "no_liftoff"
    }  # fmt: skip
    assert EpisodeRules().as_dict()["respawn_jump_m"] == 5.0
    assert ORACLE_LABEL.startswith("ORACLE perception")


def test_summary_is_recorded_for_meta():
    course = make_course()
    src = source(course)
    hover_through(src, course, line([0, 0, 1.5], [6, 0, 1.5], 61))
    s = src.summary()
    assert s["perception"] == "oracle" and s["track"] == course.id
    assert s["k"] == 1 and s["total_crossings"] == 2 and s["finished"] is False
    assert s["ticks"] == 61 and s["end"] is None
    assert s["episode_rules"] == EpisodeRules().as_dict()


# ----------------------------------------------------------------------------- course loading


@pytest.fixture
def fake_catalogue(monkeypatch, tmp_path):
    """A one-entry Liftoff catalogue whose load returns the synthetic course."""
    from drone_fly.tracks import loader
    from drone_fly.tracks.catalogue import Catalogue

    monkeypatch.delenv("DRONE_FLY_TRACKDATA", raising=False)
    monkeypatch.delenv("DRONE_FLY_SCENES", raising=False)
    entry = {
        "id": "synthetic-01", "source": "liftoff-xml", "map": "SyntheticMap",
        "gate_types": [], "laps": 1, "split": "train", "status": "ok",
        "track": "t.xml", "race": "r.xml", "name": "Synthetic",
    }  # fmt: skip
    cat = Catalogue(
        None,
        {"source": "liftoff-xml", "maps": {"SyntheticMap": {"scene": {"status": "ok"}}}},
        [entry],
    )
    state_ = {"props": "props-sha", "scene": "scene-sha", "world_scale": 1.0, "calls": []}

    def fake_load_course(entry, *, catalogue, root, scenes_root, conversion, record, use_cache):
        state_["calls"].append({"root": root, "scenes_root": scenes_root, "conv": conversion})
        record.add("props", root / "props.json", state_["props"], {"build_id": "build-1"})
        if state_["scene"] is not None:
            record.add("scene", scenes_root / "SyntheticMap.scene", state_["scene"],
                       {"map": "SyntheticMap"})  # fmt: skip
        c = make_course()
        if state_["world_scale"] != 1.0:
            from drone_fly.sim.pool import CourseFrame

            c.frame = CourseFrame(c.frame.origin_contract, state_["world_scale"])
        c.meta["scene"] = "ok"
        return [c]

    monkeypatch.setattr(loader, "load_course", fake_load_course)
    state_["catalogue"] = cat
    state_["root"] = tmp_path
    return state_


def _sidecar(**conv):
    rec = conversion_record(scenes={"SyntheticMap": {"sha256": "scene-sha"}}, **conv)
    return {"course_conversion": rec, "world_scale": 1.0}


def test_load_oracle_course_ok(fake_catalogue):
    course, info = load_oracle_course(
        "synthetic-01", _sidecar(), root=fake_catalogue["root"],
        catalogue=fake_catalogue["catalogue"],
    )  # fmt: skip
    assert info["track_id"] == "synthetic-01" and info["split"] == "train"  # eval allows train
    assert info["scene_usable"] is True and info["world_scale"] == 1.0
    assert info["conversion_fingerprint"] == _sidecar()["course_conversion"]["fingerprint"]
    assert info["passages"] == course.n_passages == 2
    call = fake_catalogue["calls"][0]
    assert call["root"] == Path(fake_catalogue["root"])
    assert call["scenes_root"] == Path(fake_catalogue["root"]) / "scenes"
    assert call["conv"].fingerprint() == info["conversion_fingerprint"]


def test_load_oracle_course_roots_from_the_environment(fake_catalogue, monkeypatch, tmp_path):
    monkeypatch.setenv("DRONE_FLY_TRACKDATA", str(tmp_path / "td"))
    monkeypatch.setenv("DRONE_FLY_SCENES", str(tmp_path / "sc"))
    load_oracle_course("synthetic-01", _sidecar(), catalogue=fake_catalogue["catalogue"])
    call = fake_catalogue["calls"][0]
    assert call["root"] == tmp_path / "td" and call["scenes_root"] == tmp_path / "sc"


def test_load_oracle_course_refuses_props_mismatch(fake_catalogue):
    fake_catalogue["props"] = "other-props"
    with pytest.raises(OracleCourseError, match="props.json sha256"):
        load_oracle_course("synthetic-01", _sidecar(), root=fake_catalogue["root"],
                           catalogue=fake_catalogue["catalogue"])  # fmt: skip


def test_load_oracle_course_refuses_scene_mismatch(fake_catalogue):
    fake_catalogue["scene"] = "other-scene"
    with pytest.raises(OracleCourseError, match="scene sha256"):
        load_oracle_course("synthetic-01", _sidecar(), root=fake_catalogue["root"],
                           catalogue=fake_catalogue["catalogue"])  # fmt: skip


def test_load_oracle_course_refuses_missing_scene(fake_catalogue):
    fake_catalogue["scene"] = None
    with pytest.raises(OracleCourseError, match="none was loaded"):
        load_oracle_course("synthetic-01", _sidecar(), root=fake_catalogue["root"],
                           catalogue=fake_catalogue["catalogue"])  # fmt: skip


def test_load_oracle_course_refuses_fingerprint_mismatch(fake_catalogue):
    with pytest.raises(OracleCourseError, match="fingerprint"):
        load_oracle_course("synthetic-01", _sidecar(fingerprint="0123456789abcdef"),
                           root=fake_catalogue["root"], catalogue=fake_catalogue["catalogue"])


def test_load_oracle_course_refuses_world_scale_mismatch(fake_catalogue):
    side = _sidecar()
    side["world_scale"] = 4.0
    with pytest.raises(OracleCourseError, match="world_scale"):
        load_oracle_course("synthetic-01", side, root=fake_catalogue["root"],
                           catalogue=fake_catalogue["catalogue"])  # fmt: skip


def test_load_oracle_course_refuses_a_non_teacher_sidecar(fake_catalogue):
    with pytest.raises(OracleCourseError, match="no course_conversion"):
        load_oracle_course("synthetic-01", {}, catalogue=fake_catalogue["catalogue"])


def test_from_config_needs_the_session_course():
    from drone_fly.harness.interfaces import BuildContext

    ctx = BuildContext(config=None, contract=None, extra_dim=148, purpose="eval")
    with pytest.raises(HarnessConfigError, match="needs the track loaded"):
        OracleGateSource.from_config({}, ctx)


def test_from_config_shares_itself_through_resources():
    from drone_fly.harness.config import load_config
    from drone_fly.harness.interfaces import BuildContext

    course = make_course()
    cfg = load_config("liftoff", {"oracle": {"start_tolerance": {"yaw_deg": 5.0}}})
    ctx = BuildContext(config=cfg, contract=None, extra_dim=148, purpose="diagnostic")
    ss, step = StateSourceStub(), ArmClock(0.5)
    ctx.resources.update(
        {"oracle_course": course, "state_source": ss, "teacher_step": step,
         "oracle_course_info": {"track_id": "x-01"}}
    )  # fmt: skip
    src = OracleGateSource.from_config({}, ctx)
    assert ctx.resources["oracle"] is src
    assert src.tolerance.yaw_deg == 5.0 and src.state_source is ss and src.arm_clock is step
    assert src.summary()["track"] == "x-01"


# ----------------------------------------------------------------------------- real anchors


def _real_trackdata() -> Path:
    root = os.environ.get("DRONE_FLY_TRACKDATA")
    if not root or not (Path(root) / "LiftoffArenaTrack04_0001.xml").is_file():
        pytest.skip("real Liftoff trackdata not configured (set DRONE_FLY_TRACKDATA)")
    return Path(root)


def test_anchor_measured_infinity_loop_spawn_lands_on_the_xml_spawn():
    """UC66's measured Infinity Loop spawn (Unity, telemetry) through the REAL converter at
    world_scale 1 lands within 0.05 m (horizontally) of the course's spawn, and the XML yaw is
    the course's ``spawn_yaw`` (local opt-in: reads the gitignored trackdata)."""
    root = _real_trackdata()
    from drone_fly.harness.liftoff.ui_driver import QUICK_SETUP_LEVELS
    from drone_fly.tracks import catalogue as cat
    from drone_fly.tracks.liftoff import ConversionConfig
    from drone_fly.tracks.loader import load_course

    c = cat.load_named("liftoff")
    entry = cat.select(c, "liftoff:liftoff-arena-04", "eval")[0]
    course = load_course(entry, catalogue=c, root=root, conversion=ConversionConfig.from_dict(None),
                         use_cache=False)[0]  # fmt: skip
    spawn_u = np.array(QUICK_SETUP_LEVELS["infinity-loop"]["spawn"], np.float32)
    from drone_fly.harness.types import TelemetrySample

    s = TelemetrySample(0.0, spawn_u, np.array([0, 0, 0, 1], np.float32), np.zeros(3, np.float32),
                        np.zeros(3, np.float32))  # fmt: skip
    fs = LiftoffConverter(GyroMap(), WorldScale(1.0)).convert(s)
    p = course.frame.to_course_point(fs.pos)
    assert math.hypot(*(p - course.spawn_pos)[:2]) < 0.05
    assert abs(p[2] - course.spawn_pos[2]) < 1.0
    # the start check accepts it
    src = source(course, tol=StartTolerance(yaw_deg=180.0))
    src.update(None, fs, 0.0)
    assert src.start_offsets["horizontal_m"] < 0.05
