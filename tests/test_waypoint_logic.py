"""UC-09 AC2 — waypoint logic: the indexed N-gate walk on hand-built transitions.

Every function under test is pure (:mod:`drone_fly.env.geometry`), so these run on
hand-built position transitions with no sim, no torch, and no env state.

UC-09 replaces the pre-UC-09 ``TO_GATE→TO_FINISH→DONE`` string phase machine with an
indexed walk over ``N`` ordered gates:

* gate passage is **3D proximity** — a gate is reached when the *segment* ``prev→curr``
  comes within ``gate.aperture`` of ``gate.center`` (segment-to-point distance, a hard
  anti-tunneling guard);
* gates must be passed **in order** — only the current-index gate is ever tested, so
  reaching gate *i+1* early is a no-op (out-of-order rejection is intrinsic);
* the **finish** is a forward +x plane crossing, counted only once all N gates are passed.
"""

from __future__ import annotations

import numpy as np

from drone_fly.env.config import CourseConfig, GateSpec, single_gate_course
from drone_fly.env.geometry import (
    advance,
    current_target,
    finish_crossed,
    gate_reached,
)

# Default 3-gate course: g0 (2.5,0,1.0) r0.6; g1 (4.0,0.6,1.3) r0.6; g2 (5.5,-0.5,0.9) r0.6;
# finish_x = 7.0.
COURSE = CourseConfig()
G0, G1, G2 = COURSE.gates
N = COURSE.num_gates  # 3


# --- gate_reached: 3D proximity (segment-to-point) -----------------------------------
def test_gate_reached_through_center() -> None:
    prev = np.array([2.4, 0.0, 1.0])
    curr = np.array([2.6, 0.0, 1.0])
    assert gate_reached(prev, curr, G0) is True


def test_gate_reached_at_aperture_edge_is_inclusive() -> None:
    # A near-stationary step exactly aperture away from the centre: the boundary is
    # inclusive (dist <= aperture). Use an exactly-representable gate (centre 0, aperture
    # 0.5, point at 0.5) so no float rounding pushes the distance over the edge.
    gate = GateSpec(center=(0.0, 0.0, 0.0), aperture=0.5)
    on_edge = np.array([0.0, 0.0, 0.5])
    assert gate_reached(on_edge, on_edge, gate) is True
    # And a hair beyond the edge is rejected.
    just_out = np.array([0.0, 0.0, 0.5 + 1e-6])
    assert gate_reached(just_out, just_out, gate) is False


def test_gate_not_reached_when_segment_stays_outside_sphere() -> None:
    # A step near the gate x but well outside the capture sphere laterally.
    prev = np.array([2.4, 2.0, 1.0])
    curr = np.array([2.6, 2.0, 1.0])
    assert gate_reached(prev, curr, G0) is False


def test_gate_reached_is_direction_agnostic() -> None:
    # Proximity is a sphere, not a plane: a backward step that passes through the sphere
    # still counts (unlike the forward-only finish plane).
    prev = np.array([2.6, 0.0, 1.0])
    curr = np.array([2.4, 0.0, 1.0])
    assert gate_reached(prev, curr, G0) is True


def test_gate_reached_segment_anti_tunneling_large_step() -> None:
    """AC2 robustness: one large step that flies straight through the sphere is credited.

    Endpoints are both far from the centre (dist ~2.5 each), but the segment passes right
    through gate0's centre — the segment-to-point distance is ~0, so the gate registers.
    """
    prev = np.array([0.0, 0.0, 1.0])
    curr = np.array([5.0, 0.0, 1.0])  # segment passes exactly through (2.5, 0, 1.0)
    # Endpoint checks would miss it: both endpoints are > aperture from the centre.
    assert float(np.linalg.norm(prev - G0.position)) > G0.aperture
    assert float(np.linalg.norm(curr - G0.position)) > G0.aperture
    assert gate_reached(prev, curr, G0) is True


def test_gate_reached_degrades_to_point_when_stationary() -> None:
    # A zero-length (hovering) step inside the sphere still registers; outside does not.
    inside = np.array([2.5, 0.0, 1.2])  # 0.2 from centre < 0.6
    outside = np.array([2.5, 0.0, 2.0])  # 1.0 from centre > 0.6
    assert gate_reached(inside, inside, G0) is True
    assert gate_reached(outside, outside, G0) is False


# --- finish_crossed: forward +x plane, no aperture -----------------------------------
def test_finish_crossed_forward() -> None:
    prev = np.array([6.9, 0.0, 1.0])
    curr = np.array([7.1, 0.0, 1.0])
    assert finish_crossed(prev, curr, COURSE) is True


def test_finish_not_crossed_backward() -> None:
    prev = np.array([7.1, 0.0, 1.0])
    curr = np.array([6.9, 0.0, 1.0])
    assert finish_crossed(prev, curr, COURSE) is False


def test_finish_has_no_aperture() -> None:
    # The finish is a full plane: a far-off-axis forward crossing still counts.
    prev = np.array([6.9, 9.0, 9.0])
    curr = np.array([7.1, 9.0, 9.0])
    assert finish_crossed(prev, curr, COURSE) is True


# --- current_target: gate-index → world target --------------------------------------
def test_current_target_points_at_each_gate_in_turn() -> None:
    np.testing.assert_array_equal(current_target(COURSE, 0), G0.position)
    np.testing.assert_array_equal(current_target(COURSE, 1), G1.position)
    np.testing.assert_array_equal(current_target(COURSE, 2), G2.position)


def test_current_target_is_finish_point_after_all_gates() -> None:
    # Once all N gates are passed the target is the finish x at the LAST gate's (y, z).
    finish_pt = current_target(COURSE, N)
    assert finish_pt[0] == COURSE.finish_x
    assert finish_pt[1] == G2.center[1]
    assert finish_pt[2] == G2.center[2]


# --- advance: the ordered indexed walk ----------------------------------------------
def test_advance_passes_current_gate_and_increments_index() -> None:
    prev = np.array([2.4, 0.0, 1.0])
    curr = np.array([2.6, 0.0, 1.0])
    gates_passed, done, event = advance(COURSE, 0, False, prev, curr)
    assert gates_passed == 1
    assert done is False
    assert event == "gate"


def test_advance_rejects_out_of_order_gate() -> None:
    """Reaching gate 1's vicinity while still on index 0 does NOT advance (AC2).

    Only the current gate (index 0) is tested, so flying through gate1's sphere before
    gate0 is a no-op — you cannot skip a gate.
    """
    # A step straight through gate1's centre while gates_passed == 0.
    prev = G1.position + np.array([-0.2, 0.0, 0.0])
    curr = G1.position + np.array([0.2, 0.0, 0.0])
    gates_passed, done, event = advance(COURSE, 0, False, prev, curr)
    assert gates_passed == 0
    assert event is None


def test_advance_finish_before_all_gates_is_ignored() -> None:
    """Crossing the finish plane before all gates are passed does nothing (AC2)."""
    prev = np.array([6.9, 0.0, 1.0])
    curr = np.array([7.1, 0.0, 1.0])
    # Still chasing gate 2 (index 2 < N); the finish must not fire.
    gates_passed, done, event = advance(COURSE, 2, False, prev, curr)
    assert gates_passed == 2
    assert done is False
    assert event is None


def test_advance_completes_only_after_all_gates_then_finish() -> None:
    prev = np.array([6.9, -0.5, 0.9])
    curr = np.array([7.1, -0.5, 0.9])
    gates_passed, done, event = advance(COURSE, N, False, prev, curr)
    assert gates_passed == N
    assert done is True
    assert event == "finish"


def test_advance_full_ordered_three_gate_sequence() -> None:
    """Drive a whole start→g0→g1→g2→finish trajectory; gates fire in order, finish last."""
    path = [
        (0.0, 0.0, 1.0),
        (2.5, 0.0, 1.0),  # g0
        (4.0, 0.6, 1.3),  # g1
        (5.5, -0.5, 0.9),  # g2
        (6.9, -0.5, 0.9),
        (7.1, -0.5, 0.9),  # finish
    ]
    gates_passed, done = 0, False
    events = []
    for prev, curr in zip(path, path[1:], strict=False):
        gates_passed, done, event = advance(
            COURSE, gates_passed, done, np.array(prev), np.array(curr)
        )
        events.append(event)
    assert events == ["gate", "gate", "gate", None, "finish"]
    assert gates_passed == N
    assert done is True


def test_advance_stays_done() -> None:
    gates_passed, done, event = advance(
        COURSE, N, True, np.array([7.1, 0.0, 1.0]), np.array([8.0, 0.0, 1.0])
    )
    assert done is True
    assert event is None


# --- N=1 single-gate course: reproduces a plain start→gate→finish race ---------------
def test_single_gate_course_advance_sequence() -> None:
    course = single_gate_course()  # gate at x=3, finish x=6, aperture 0.6
    assert course.num_gates == 1
    (gate,) = course.gates
    # target is the sole gate, then the finish.
    np.testing.assert_array_equal(current_target(course, 0), gate.position)
    assert current_target(course, 1)[0] == course.finish_x

    # pass the gate, then cross the finish.
    gp, done, ev = advance(course, 0, False, np.array([2.9, 0, 1.0]), np.array([3.1, 0, 1.0]))
    assert gp == 1 and ev == "gate" and done is False
    gp, done, ev = advance(course, 1, False, np.array([5.9, 0, 1.0]), np.array([6.1, 0, 1.0]))
    assert gp == 1 and ev == "finish" and done is True


def test_single_gate_finish_before_gate_ignored() -> None:
    course = single_gate_course()
    gp, done, ev = advance(course, 0, False, np.array([5.9, 0, 1.0]), np.array([6.1, 0, 1.0]))
    assert gp == 0 and done is False and ev is None
