"""UC69 AC5 / AC6 / AC8: scoring a live attempt from telemetry + the course, NumPy only.

Crossings (the sim's route rules, stopping at a discontinuity), lap times, lift-off, the sim's
contact rules (``sim_rule_contact``: raw vs sim-rule completion), every failure mode and their
priority, the live / offline crossing comparison and a whole recorded attempt
(``load_attempt`` + ``score_attempt``). Synthetic courses and telemetry only.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from drone_fly.harness import scoring as sc
from drone_fly.harness.scoring import (
    ContactRules,
    Crossing,
    classify_failure,
    compare_live,
    detect_liftoff,
    first_window_stats,
    lap_times,
    laps_done,
    load_attempt,
    score_attempt,
    score_crossings,
    sim_rule_contact,
    trajectory_from_samples,
)
from tests.oracle_support import (
    loop_passages,
    make_course,
    path_samples,
    segments,
    tilt_quat,
    unity_sample,
    write_attempt,
)

DT = 0.01
RULES = ContactRules()


def traj_of(course, points, **kw):
    return trajectory_from_samples(path_samples(course, points, dt=DT, **kw), course)


def ground(n, p=(0.0, 0.0, 0.03)):
    return [np.asarray(p, float)] * n


#: ground 0.6 s, climb to 1.5 m in 0.5 s, then through both openings of the straight course
CLEAN = ground(60) + segments(
    ((0, 0, 0.03), (0, 0, 1.5), 51), ((0, 0, 1.5), (11, 0, 1.5), 111)
)[1:]
#: as CLEAN, but swerving to graze the placed box (0.08 m from its side) between the openings
GRAZE = ground(60) + segments(
    ((0, 0, 0.03), (0, 0, 1.5), 51),
    ((0, 0, 1.5), (5.5, 0, 1.5), 56),
    ((5.5, 0, 1.5), (7.5, -2.42, 1.0), 41),
    ((7.5, -2.42, 1.0), (9.0, 0.0, 1.5), 41),
    ((9.0, 0.0, 1.5), (11.0, 0.0, 1.5), 21),
)[1:]


# ----------------------------------------------------------------------------- trajectory


def test_trajectory_maps_unity_to_the_course_frame_and_flags_discontinuities():
    course = make_course()
    s = [
        unity_sample(course, [1.0, 2.0, 3.0], t=0.0, vel=[1.0, 0.0, -1.0], gyro=[0.0, 90.0, 0.0]),
        unity_sample(course, [1.1, 2.0, 3.0], t=0.01),
        unity_sample(course, [9.0, 2.0, 3.0], t=0.02),  # a 7.9 m jump
        unity_sample(course, [9.1, 2.0, 3.0], t=0.01),  # time going backwards
        unity_sample(course, [9.2, 2.0, 3.0], t=0.03),
    ]
    tr = trajectory_from_samples(s, course)
    np.testing.assert_allclose(tr.pos[0], [1.0, 2.0, 3.0], atol=1e-4)
    np.testing.assert_allclose(tr.vel[0], [1.0, 0.0, -1.0], atol=1e-5)
    np.testing.assert_allclose(tr.omega[0], [-math.pi / 2, 0, 0], atol=1e-6)  # +roll -> -wx
    assert tr.disc.tolist() == [False, False, True, True, False]
    assert tr.first_discontinuity() == 2
    assert tr.slice(3, 5).n == 2


# ----------------------------------------------------------------------------- crossings


def test_score_crossings_straight_course():
    course = make_course()
    tr = traj_of(course, CLEAN)
    cr = score_crossings(tr, course)
    assert [(c.k, c.passage) for c in cr] == [(1, 0), (2, 1)]
    assert cr[0].t == pytest.approx(1.1 + 5.0 / 11 * 1.1, abs=0.02)
    assert score_crossings(tr, course, end=cr[0].index) == []  # the end bound is exclusive
    assert laps_done(2, course) == 1


def test_score_crossings_stop_at_the_first_discontinuity():
    """A respawn teleporting through an opening is never tested (nor anything after it)."""
    course = make_course()
    pts = segments(((0, 0, 1.5), (3, 0, 1.5), 31)) + [np.array([8.5, 0, 1.5])] + segments(
        ((8.5, 0, 1.5), (11, 0, 1.5), 26)
    )[1:]
    tr = traj_of(course, pts)
    assert tr.first_discontinuity() == 31
    assert score_crossings(tr, course) == []


def test_score_crossings_out_of_aperture_and_backwards():
    course = make_course()
    assert score_crossings(traj_of(course, segments(((0, 1.4, 1.5), (11, 1.4, 1.5), 111))),
                           course) == []  # fmt: skip
    assert score_crossings(traj_of(course, segments(((11, 0, 1.5), (0, 0, 1.5), 111))),
                           course) == []  # fmt: skip


def _loop_points(laps, extra=True):
    corners = [(0, 0, 1.5), (10, 0, 1.5), (10, 10, 1.5), (0, 10, 1.5), (0, 0, 1.5)]
    legs = []
    for _ in range(laps):
        legs += [(a, b, 101) for a, b in zip(corners[:-1], corners[1:], strict=True)]
    if extra:
        legs.append(((0, 0, 1.5), (7, 0, 1.5), 71))
    return segments(*legs)


def test_lap_and_race_times_on_a_circuit():
    course = make_course(loop_passages(), laps=2, circuit=True)
    tr = traj_of(course, _loop_points(2))
    cr = score_crossings(tr, course)
    assert len(cr) == course.total_crossings == 9
    laps, full = lap_times(cr, course)
    assert len(laps) == 2 and laps == pytest.approx([4.0, 4.0], abs=0.02)  # 40 m at 10 m/s
    assert full == pytest.approx(8.0, abs=0.02)
    assert laps_done(9, course) == 2 and laps_done(5, course) == 1 and laps_done(1, course) == 0
    # incomplete: one lap, no race time
    laps, full = lap_times(cr[:6], course)
    assert len(laps) == 1 and full is None


# ----------------------------------------------------------------------------- lift-off


def test_detect_liftoff_after_arming():
    course = make_course()
    tr = traj_of(course, CLEAN)
    lo = detect_liftoff(tr, arm_end_clock=0.5)
    assert tr.clock[lo["sim_rule"]] == pytest.approx(0.61, abs=0.011)
    assert tr.clock[lo["threshold"]] > tr.clock[lo["sim_rule"]]
    # a climb before arming ended does not count
    lo = detect_liftoff(tr, arm_end_clock=5.0)
    assert lo == {"sim_rule": None, "threshold": None}


def test_first_window_stats():
    course = make_course()
    quats = [tilt_quat("x", math.radians(30 if 70 <= i < 80 else 0)) for i in range(len(CLEAN))]
    gyros = [(0.0, 300.0 if i == 75 else 0.0, 0.0) for i in range(len(CLEAN))]
    tr = traj_of(course, CLEAN, quats=quats, gyros=gyros)
    st = first_window_stats(tr, 0.5)
    assert st["max_tilt_deg"] == pytest.approx(30.0, abs=0.01)
    assert st["max_rate_rad_s"] == pytest.approx(math.radians(300), abs=1e-4)
    assert first_window_stats(tr, None) == {"max_tilt_deg": None, "max_rate_rad_s": None}


# ----------------------------------------------------------------------------- contact rules


def test_graze_then_finish_raw_complete_but_sim_rule_incomplete():
    course = make_course()
    tr = traj_of(course, GRAZE)
    cr = score_crossings(tr, course)
    assert len(cr) == 2
    lift = detect_liftoff(tr, 0.5)["sim_rule"]
    c = sim_rule_contact(tr, course, RULES, liftoff_index=lift)
    assert c is not None and c.kind == "collision" and c.source == 1  # the placed item
    assert c.distance_m == pytest.approx(0.08, abs=0.01)
    assert cr[0].index < c.index < cr[1].index


def test_clean_flight_has_no_contact():
    course = make_course()
    tr = traj_of(course, CLEAN)
    lift = detect_liftoff(tr, 0.5)["sim_rule"]
    assert sim_rule_contact(tr, course, RULES, liftoff_index=lift) is None


def test_resting_on_the_pad_is_not_a_floor_contact():
    """Grounded (never lifted off) on the raised pad and on the floor: nothing counts."""
    course = make_course(pad=True)
    tr = traj_of(course, ground(300))
    assert sim_rule_contact(tr, course, RULES, liftoff_index=None) is None
    course = make_course()
    tr = traj_of(course, ground(300))
    assert sim_rule_contact(tr, course, RULES, liftoff_index=None) is None


def test_touching_the_floor_after_the_grace_is_a_floor_contact():
    course = make_course()
    pts = ground(60) + segments(
        ((0, 0, 0.03), (0, 0, 0.6), 21), ((0, 0, 0.6), (0, 0, 0.05), 40)
    )[1:] + ground(50, (0, 0, 0.05))
    tr = traj_of(course, pts)
    lift = detect_liftoff(tr, 0.5)["sim_rule"]
    c = sim_rule_contact(tr, course, RULES, liftoff_index=lift)
    assert c is not None and c.kind == "floor" and c.source is None
    assert tr.t[c.index] >= tr.t[lift] + RULES.liftoff_grace_s
    assert c.pos[2] < RULES.floor_kill_m


def test_landing_on_the_pad_top_after_the_grace_is_a_floor_contact():
    """The pad is a primitive: its up-face counts once the grace is over (sim semantics)."""
    course = make_course(pad=True)
    pts = ground(60) + segments(((0, 0, 0.03), (0, 0, 0.6), 21), ((0, 0, 0.6), (0, 0, 0.05), 40))[
        1:
    ]
    tr = traj_of(course, pts)
    lift = detect_liftoff(tr, 0.5)["sim_rule"]
    c = sim_rule_contact(tr, course, RULES, liftoff_index=lift)
    assert c is not None and c.kind == "floor" and c.source == 0


def test_contact_within_the_grace_is_ignored():
    course = make_course()
    # touches the floor 0.1 s after lift-off (inside the 0.3 s grace), then climbs away
    pts = ground(60) + segments(
        ((0, 0, 0.03), (0, 0, 0.2), 6), ((0, 0, 0.2), (0, 0, 0.05), 6), ((0, 0, 0.05), (0, 0, 1.5), 60)
    )[1:]
    tr = traj_of(course, pts)
    lift = detect_liftoff(tr, 0.5)["sim_rule"]
    assert sim_rule_contact(tr, course, RULES, liftoff_index=lift) is None


@pytest.mark.parametrize("end", [(0, 0, 40.0), (-12.0, 12.0, 1.5)])
def test_out_of_bounds_and_corridor(end):
    course = make_course(corridor_m=10.0)
    pts = segments(((0, 0, 1.5), end, 400))
    tr = traj_of(course, pts)
    c = sim_rule_contact(tr, course, RULES, liftoff_index=0)
    assert c is not None and c.kind == "out_of_bounds"


def test_contact_rules_from_the_teacher_env():
    r = ContactRules.from_env({"floor_kill_m": 0.1, "drone_radius_m": 0.2, "liftoff_grace_s": 0.5})
    assert (r.floor_kill_m, r.drone_radius_m, r.liftoff_grace_s) == (0.1, 0.2, 0.5)
    d = r.as_dict(make_course())
    assert d["corridor_m"] == 25.0 and len(d["bounds_lo"]) == 3
    assert ContactRules.from_env(None) == ContactRules()


# ----------------------------------------------------------------------------- failure modes


def _classify(course, pts, *, reason="episode_end:crash_settled", arm=0.5, **kw):
    tr = traj_of(course, pts, **kw)
    cr = score_crossings(tr, course)
    lift = detect_liftoff(tr, arm)
    li = min([i for i in lift.values() if i is not None], default=None)
    contact = sim_rule_contact(tr, course, RULES, liftoff_index=li)
    return classify_failure(tr, course, cr, stop_reason=reason, arm_end_clock=arm,
                            contact=contact)  # fmt: skip


def test_takeoff_tumble():
    course = make_course()
    pts = ground(60) + segments(((0, 0, 0.03), (0, 0, 0.8), 81))[1:]
    quats = [tilt_quat("x", math.radians(80 if i > 100 else 0)) for i in range(len(pts))]
    f = _classify(course, pts, quats=quats)
    assert f["primary"] == "takeoff_tumble" and "tilt 80 deg" in f["details"]["takeoff_tumble"]


def test_takeoff_tumble_on_rate_alone():
    course = make_course()
    pts = ground(60) + segments(((0, 0, 0.03), (0, 0, 0.8), 81))[1:]
    gyros = [(0.0, 700.0 if i == 90 else 0.0, 0.0) for i in range(len(pts))]  # 12.2 rad/s
    f = _classify(course, pts, gyros=gyros)
    assert f["primary"] == "takeoff_tumble"


def test_tumble_after_the_first_crossing_is_not_a_takeoff_tumble():
    course = make_course()
    pts = segments(((0, 0, 1.5), (6, 0, 1.5), 61))  # through opening 0 at ~0.5 s
    quats = [tilt_quat("x", math.radians(80 if i > 55 else 0)) for i in range(len(pts))]
    f = _classify(course, pts, arm=0.0, quats=quats, reason="episode_end:inverted")
    assert "takeoff_tumble" not in f["details"]


def test_no_liftoff():
    course = make_course()
    f = _classify(course, ground(600), reason="episode_end:no_liftoff")
    assert f["primary"] == "no_liftoff"
    assert "altitude_loss" in f["secondary"]  # it ended on the ground too


def test_gate_frame_collision():
    """A 6 m/s stop within 0.15 s on the frame of opening 0 (between aperture and outer)."""
    course = make_course()
    pts = segments(((0, 1.3, 1.5), (5.0, 1.3, 1.5), 84)) + ground(30, (5.0, 1.3, 1.5))
    f = _classify(course, pts, arm=0.0)
    assert f["primary"] == "gate_frame_collision"
    assert "passage 0" in f["details"]["gate_frame_collision"]


def test_stop_in_the_open_is_not_a_gate_frame_collision():
    course = make_course()
    pts = segments(((0, 6.0, 1.5), (5.0, 6.0, 1.5), 84)) + ground(30, (5.0, 6.0, 1.5))
    assert "gate_frame_collision" not in _classify(course, pts, arm=0.0)["details"]


def test_altitude_loss_by_sinking():
    course = make_course()
    pts = segments(((0, 0, 3.5), (2, 0, 3.5), 51), ((2, 0, 3.5), (2, 0, 1.0), 151))
    f = _classify(course, pts, arm=0.0)
    assert f["primary"] == "altitude_loss" and "sinking" in f["details"]["altitude_loss"]


def test_overshoot_outside_the_outline():
    course = make_course()
    pts = segments(((0, 0, 1.5), (4, 1.6, 1.5), 41), ((4, 1.6, 1.5), (7, 1.6, 1.5), 31))
    f = _classify(course, pts, arm=0.0, reason="episode_end:out_of_bounds")
    assert f["primary"] == "overshoot" and "outside its outline" in f["details"]["overshoot"]


def test_overshoot_moving_away():
    course = make_course()
    pts = segments(((0, 0, 1.5), (4, 3.0, 1.5), 41), ((4, 3.0, 1.5), (0, 8.0, 1.5), 201))
    f = _classify(course, pts, arm=0.0, reason="episode_end:out_of_bounds")
    assert f["primary"] == "overshoot" and "moving away" in f["details"]["overshoot"]


def test_attitude_oscillation():
    course = make_course()
    pts = ground(60) + segments(((0, 0, 0.03), (0, 0, 1.5), 41), ((0, 0, 1.5), (1, 0, 1.5), 300))[1:]
    gyros = [(0.0, (250.0 if i % 4 < 2 else -250.0) if i > 250 else 0.0, 0.0)
             for i in range(len(pts))]  # fmt: skip
    f = _classify(course, pts, gyros=gyros, reason="episode_end:out_of_bounds")
    assert f["primary"] == "attitude_oscillation" and "roll rate" in f["details"][
        "attitude_oscillation"
    ]


def test_timeout_and_other():
    course = make_course()
    pts = ground(60) + segments(((0, 0, 0.03), (0, 0, 1.5), 41), ((0, 0, 1.5), (1, 0, 1.5), 300))[1:]
    assert _classify(course, pts, reason="episode_end:time_limit")["primary"] == "timeout"
    assert _classify(course, pts, reason="completed")["primary"] == "timeout"
    f = _classify(course, pts, reason="episode_end:respawn")
    assert f["primary"] == "other" and f["details"]["other"] == "stop episode_end:respawn"


def test_priority_order_tumble_over_floor():
    """A take-off tumble that ends on the floor: tumble primary, altitude loss secondary."""
    course = make_course()
    pts = ground(60) + segments(((0, 0, 0.03), (0, 0, 0.8), 31), ((0, 0, 0.8), (0, 0, 0.04), 60))[
        1:
    ]
    quats = [tilt_quat("x", math.radians(85 if i > 80 else 0)) for i in range(len(pts))]
    f = _classify(course, pts, quats=quats)
    assert f["primary"] == "takeoff_tumble"
    assert f["secondary"][0] == "altitude_loss"


# ----------------------------------------------------------------------------- power loss

#: hover at 3.5 m, then sink to 1.0 m (an altitude loss on its own)
SINK = segments(((0, 0, 3.5), (2, 0, 3.5), 51), ((2, 0, 3.5), (2, 0, 1.0), 151))


def _powered(tr, *, spin_at=20, cut_at=None, stick=1.0, live_motor=False):
    """``tr`` with motors at 1000 rpm, spun up to 25000 from ``spin_at`` and cut to 1000 from
    ``cut_at`` (``live_motor``: motor 3 keeps spinning), a constant stick and a sagging pack."""
    n = tr.n
    rpm = np.full((n, 4), 1000.0)
    rpm[spin_at:] = 25000.0
    if cut_at is not None:
        rpm[cut_at:, : 3 if live_motor else 4] = 1000.0
    return dataclasses.replace(
        tr, rpm=rpm, throttle=np.full(n, stick), volts=np.linspace(16.8, 13.2, n),
        charge=np.linspace(1.0, 0.2, n),
    )  # fmt: skip


def _classify_powered(course, tr, *, reason="episode_end:crash_settled", arm=0.0):
    cr = score_crossings(tr, course)
    lift = detect_liftoff(tr, arm)
    li = min([i for i in lift.values() if i is not None], default=None)
    contact = sim_rule_contact(tr, course, RULES, liftoff_index=li)
    return classify_failure(tr, course, cr, stop_reason=reason, arm_end_clock=arm,
                            contact=contact, power=sc.power_summary(tr))  # fmt: skip


def test_power_loss_fires_on_a_full_stick_motor_cut_after_spin_up():
    course = make_course()
    tr = _powered(traj_of(course, SINK), spin_at=20, cut_at=120, stick=0.95)
    ps = sc.power_summary(tr)
    assert ps["spin_up_clock"] == pytest.approx(tr.clock[20])
    assert ps["volts_at_spin_up"] == pytest.approx(tr.volts[20])
    loss = ps["loss"]
    assert loss["index"] == 120
    assert loss["t_from_spin_up_s"] == pytest.approx(tr.clock[120] - tr.clock[20])
    assert loss["t_from_spin_up_s"] == pytest.approx(1.0, abs=1e-6)
    assert loss["volts"] == pytest.approx(tr.volts[120])
    assert loss["charge"] == pytest.approx(tr.charge[120])
    assert loss["throttle"] == pytest.approx(0.95)
    assert loss["rpm_before"] == 25000.0 and loss["rpm_after"] == 1000.0
    np.testing.assert_allclose(loss["pos"], tr.pos[120])
    f = _classify_powered(course, tr)
    assert "power_loss" in f["details"]
    assert "motors cut to 1000 rpm at stick 0.95" in f["details"]["power_loss"]


def test_power_loss_needs_full_stick_all_motors_and_a_prior_spin_up():
    course = make_course()
    tr = traj_of(course, SINK)
    # low stick: an idle cut is a pilot input, not a power loss
    assert sc.power_summary(_powered(tr, cut_at=120, stick=0.5))["loss"] is None
    # just at the stick threshold (strictly above it is required)
    assert sc.power_summary(_powered(tr, cut_at=120, stick=sc.POWER_STICK))["loss"] is None
    # one motor still spinning
    assert sc.power_summary(_powered(tr, cut_at=120, live_motor=True))["loss"] is None
    # full stick with the motors low from the start: never spun up, so never a power loss
    ps = sc.power_summary(_powered(tr, spin_at=tr.n))
    assert ps == {"spin_up_clock": None, "loss": None}
    # low motors before the spin-up do not count, only those after it
    ps = sc.power_summary(_powered(tr, spin_at=100))
    assert ps["loss"] is None and ps["spin_up_clock"] == pytest.approx(tr.clock[100])
    assert "power_loss" not in _classify_powered(course, _powered(tr, cut_at=120, stick=0.5))[
        "details"
    ]
    assert sc.power_summary(dataclasses.replace(tr, rpm=None)) is None  # no RPM telemetry


def test_power_loss_takes_priority_over_altitude_loss():
    course = make_course()
    tr = _powered(traj_of(course, SINK), cut_at=120)
    f = _classify_powered(course, tr)
    assert f["primary"] == "power_loss"
    assert "altitude_loss" in f["secondary"]
    # the same flight without the cut is a plain altitude loss
    assert _classify_powered(course, _powered(tr))["primary"] == "altitude_loss"


def test_priority_table_and_thresholds():
    assert sc.FAILURE_MODES[0] == "takeoff_tumble" and sc.FAILURE_MODES[-1] == "other"
    assert len(sc.FAILURE_MODES) == 9
    assert [sc.MODE_PRIORITY[m] for m in sc.FAILURE_MODES] == ["1", "1b", "1c", "2", "3", "4",
                                                                 "5", "6", "7"]  # fmt: skip
    assert sc.FAILURE_MODES[1:4] == ("no_liftoff", "power_loss", "gate_frame_collision")
    th = sc.thresholds()
    assert th["tumble_tilt_deg"] == 60.0 and th["tumble_rate_rad_s"] == 10.0
    assert th["impact_speed_drop_mps"] == 3.0 and th["osc_sign_changes_per_s"] == 6.0
    assert th["power_loss_rpm"] == 3000.0 and th["power_stick"] == 0.9
    assert th["power_spin_rpm"] == 20000.0 and th["full_stick"] == 0.95
    assert len(th) == 21


# ----------------------------------------------------------------------------- live vs offline


def test_compare_live():
    off = [Crossing(1, 0, 10, 1.0, 1.0), Crossing(2, 1, 20, 2.0, 2.0), Crossing(3, 0, 30, 3, 3.0)]
    live = [
        {"k": 1, "passage": 0, "clock": 1.02},  # matches
        {"k": 2, "passage": 0, "clock": 2.0},  # wrong passage
        {"k": 4, "passage": 1, "clock": 4.0},  # live only
        {"k": None, "passage": None, "clock": 9.0},  # ignored
    ]
    mm = compare_live(off, live)
    assert {m["k"]: m["issue"] for m in mm} == {2: "passage", 3: "offline only", 4: "live only"}
    mm = compare_live(off[:1], [{"k": 1, "passage": 0, "clock": 1.5}])
    assert mm == [{"k": 1, "offline": 0, "live": 0, "issue": "dt +0.50 s"}]
    assert compare_live([], []) == []


# ----------------------------------------------------------------------------- a whole attempt


def test_score_attempt_graze_then_finish(tmp_path):
    course = make_course()
    samples = path_samples(course, GRAZE, dt=DT)
    tr = trajectory_from_samples(samples, course)
    cr = score_crossings(tr, course)
    live = [{"k": c.k, "passage": c.passage, "clock": c.clock} for c in cr]
    d = write_attempt(tmp_path, "a01", samples, stop_reason="course_finished",
                      stop_class="course_finished", live_crossings=live)  # fmt: skip
    rec = load_attempt(d)
    assert rec.arm_end_clock == pytest.approx(0.5, abs=0.021)
    assert rec.first_tick_clock == 0.0 and len(rec.live_crossings) == 2
    s = score_attempt(rec, course, RULES)
    assert s["k_raw"] == 2 and s["completion_raw"] is True
    assert s["k_sim_rule"] == 1 and s["completion_sim_rule"] is False
    assert s["contact"]["kind"] == "collision"
    assert s["failure"] is None  # raw-complete attempts are not classified
    assert s["live_mismatches"] == []
    assert s["time_to_first_gate_s"] == pytest.approx(cr[0].clock - rec.arm_end_clock)
    assert s["liftoff_after_arming_s"]["sim_rule"] == pytest.approx(0.1, abs=0.03)
    assert s["full_time_s"] is not None and s["stop_class"] == "course_finished"
    assert s["arm_confirm_delay_s"] == 0.1 and s["training_active"] is False


def test_score_attempt_logs_live_offline_mismatches_and_classifies(tmp_path):
    course = make_course()
    pts = ground(60) + segments(((0, 0, 0.03), (0, 0, 1.5), 51), ((0, 0, 1.5), (6, 0, 1.5), 61))[1:]
    samples = path_samples(course, pts, dt=DT)
    live = [{"k": 1, "passage": 0, "clock": 1.2}, {"k": 2, "passage": 1, "clock": 1.6}]
    d = write_attempt(tmp_path, "a02", samples, stop_reason="episode_end:time_limit",
                      stop_class="episode_end", live_crossings=live)  # fmt: skip
    s = score_attempt(load_attempt(d), course, RULES)
    assert s["k_raw"] == 1 and s["completion_raw"] is False
    # offline: opening 0 at 1.6 s; the live oracle claimed it at 1.2 s, and a second one
    issues = [m["issue"] for m in s["live_mismatches"]]
    assert issues[0] in ("dt -0.39 s", "dt -0.40 s") and issues[1:] == ["live only"]
    assert s["failure"]["primary"] == "timeout"


def test_score_attempt_ignores_telemetry_before_the_first_tick(tmp_path):
    """Samples before the run started (the receiver's backlog) are not scored."""
    course = make_course()
    pre = segments(((-1, 0, 1.5), (6, 0, 1.5), 71))  # an earlier pass through opening 0
    post = ground(100)
    samples = path_samples(course, pre + post, dt=DT)
    d = write_attempt(tmp_path, "a03", samples[71:], stop_reason="episode_end:no_liftoff",
                      stop_class="episode_end")  # fmt: skip
    # put the backlog into the CSV too, with the ticks starting at the ground part
    from tests.oracle_support import write_telemetry_csv

    write_telemetry_csv(d / "telemetry.csv", samples)
    s = score_attempt(load_attempt(d), course, RULES)
    assert s["k_raw"] == 0 and s["samples"] <= 106
