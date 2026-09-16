"""AC2 — waypoint logic: gate-before-finish + pass/cross detection on hand transitions.

Every function under test is pure (:mod:`drone_fly.env.geometry`), so these run on
hand-built position transitions with no sim, no torch, and no env state — the "unit-tested
on hand-built transitions" the acceptance criterion asks for.
"""

from __future__ import annotations

import numpy as np

from drone_fly.env.config import CourseConfig
from drone_fly.env.geometry import (
    DONE,
    TO_FINISH,
    TO_GATE,
    advance_phase,
    finish_crossed,
    gate_passed,
    target_position,
)

COURSE = CourseConfig()  # gate_x=3, finish_x=6, aperture=0.6, gate_center=(3,0,1)


# --- gate_passed --------------------------------------------------------------------
def test_gate_passed_through_aperture_center() -> None:
    prev = np.array([2.9, 0.0, 1.0])
    curr = np.array([3.1, 0.0, 1.0])
    assert gate_passed(prev, curr, COURSE) is True


def test_gate_passed_true_at_aperture_edge() -> None:
    # Cross exactly at the +y aperture rim (dy == aperture, dz == 0): inclusive boundary.
    y = COURSE.gate_center_y + COURSE.gate_aperture
    prev = np.array([2.9, y, 1.0])
    curr = np.array([3.1, y, 1.0])
    assert gate_passed(prev, curr, COURSE) is True


def test_gate_not_passed_outside_aperture() -> None:
    # Forward crossing of the gate plane but well outside the aperture disc.
    prev = np.array([2.9, 2.0, 1.0])
    curr = np.array([3.1, 2.0, 1.0])
    assert gate_passed(prev, curr, COURSE) is False


def test_gate_not_passed_outside_aperture_in_z() -> None:
    z = COURSE.gate_center_z + COURSE.gate_aperture + 0.5
    prev = np.array([2.9, 0.0, z])
    curr = np.array([3.1, 0.0, z])
    assert gate_passed(prev, curr, COURSE) is False


def test_gate_passed_is_forward_only() -> None:
    # A backward (-x) crossing never counts, even through the aperture centre.
    prev = np.array([3.1, 0.0, 1.0])
    curr = np.array([2.9, 0.0, 1.0])
    assert gate_passed(prev, curr, COURSE) is False


def test_gate_not_passed_when_segment_does_not_reach_plane() -> None:
    prev = np.array([2.0, 0.0, 1.0])
    curr = np.array([2.5, 0.0, 1.0])
    assert gate_passed(prev, curr, COURSE) is False


def test_gate_aperture_uses_interpolated_crossing_point() -> None:
    # Segment starts inside the disc laterally but drifts out; the *interpolated* point at
    # the gate plane is what matters. Here it crosses the plane while still inside.
    prev = np.array([2.99, 0.1, 1.0])
    curr = np.array([3.01, 0.1, 1.0])
    assert gate_passed(prev, curr, COURSE) is True


# --- finish_crossed -----------------------------------------------------------------
def test_finish_crossed_forward() -> None:
    prev = np.array([5.9, 0.0, 1.0])
    curr = np.array([6.1, 0.0, 1.0])
    assert finish_crossed(prev, curr, COURSE) is True


def test_finish_not_crossed_backward() -> None:
    prev = np.array([6.1, 0.0, 1.0])
    curr = np.array([5.9, 0.0, 1.0])
    assert finish_crossed(prev, curr, COURSE) is False


def test_finish_has_no_aperture() -> None:
    # The finish is a full plane: a far-off-axis forward crossing still counts.
    prev = np.array([5.9, 9.0, 9.0])
    curr = np.array([6.1, 9.0, 9.0])
    assert finish_crossed(prev, curr, COURSE) is True


# --- advance_phase: the gate-before-finish state machine ----------------------------
def test_advance_phase_passes_gate_then_advances_to_finish() -> None:
    prev = np.array([2.9, 0.0, 1.0])
    curr = np.array([3.1, 0.0, 1.0])
    phase, event = advance_phase(TO_GATE, prev, curr, COURSE)
    assert phase == TO_FINISH
    assert event == "gate"


def test_advance_phase_finish_before_gate_is_ignored() -> None:
    # Crossing the finish plane while still TO_GATE must NOT complete the course and must
    # not change phase — you cannot skip the gate.
    prev = np.array([5.9, 0.0, 1.0])
    curr = np.array([6.1, 0.0, 1.0])
    phase, event = advance_phase(TO_GATE, prev, curr, COURSE)
    assert phase == TO_GATE
    assert event is None


def test_advance_phase_completes_only_after_gate() -> None:
    prev = np.array([5.9, 0.0, 1.0])
    curr = np.array([6.1, 0.0, 1.0])
    phase, event = advance_phase(TO_FINISH, prev, curr, COURSE)
    assert phase == DONE
    assert event == "finish"


def test_advance_phase_full_ordered_sequence() -> None:
    # Drive a whole start→gate→finish trajectory through the machine and assert the gate
    # fires before the finish and completion happens exactly once, at the end.
    path = [
        (0.0, 0.0, 1.0),
        (2.9, 0.0, 1.0),
        (3.1, 0.0, 1.0),  # gate
        (5.9, 0.0, 1.0),
        (6.1, 0.0, 1.0),  # finish
    ]
    phase = TO_GATE
    events = []
    for prev, curr in zip(path, path[1:], strict=False):
        phase, event = advance_phase(phase, np.array(prev), np.array(curr), COURSE)
        events.append(event)
    assert events == [None, "gate", None, "finish"]
    assert phase == DONE


def test_advance_phase_stays_done() -> None:
    phase, event = advance_phase(DONE, np.array([6.1, 0, 1]), np.array([7.0, 0, 1]), COURSE)
    assert phase == DONE
    assert event is None


def test_target_position_switches_with_phase() -> None:
    np.testing.assert_array_equal(target_position(TO_GATE, COURSE), COURSE.gate_center)
    finish_pt = target_position(TO_FINISH, COURSE)
    assert finish_pt[0] == COURSE.finish_x
    np.testing.assert_array_equal(target_position(DONE, COURSE), finish_pt)
