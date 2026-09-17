"""Pure waypoint logic and the indexed N-gate → finish walk (UC-09 AC2).

No sim, no torch, no env state beyond what is passed in — every function here is pure and
unit-testable on hand-built position transitions. The env drives this each step to decide
gate advancement and course completion.

Rules
-----
* The course has **N ordered gates** (:class:`~drone_fly.env.config.GateSpec`) followed by
  a **finish** plane at ``finish_x``.
* A gate is passed by **3D proximity** (UC-09): the closest point of the step segment
  ``prev_pos→curr_pos`` comes within the gate's ``aperture`` (capture radius) of its
  centre. Using the *segment*-to-point distance (not just the endpoint) is a hard
  anti-tunneling guard — a fast step that flies straight through the sphere still counts —
  and it degrades to endpoint-proximity when the drone is near-stationary.
* Gates must be passed **in order**: the walk tracks ``gates_passed`` (0..N) and only ever
  tests the *current* gate ``gates[gates_passed]``. Reaching gate *i+1*'s vicinity before
  gate *i* therefore does nothing — out-of-order rejection is intrinsic to the index model.
* The **finish** is a forward ``+x`` plane crossing (as before), counted only once all N
  gates are passed (``gates_passed == N``). An early finish crossing is ignored.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from drone_fly.env.config import CourseConfig, GateSpec

Event = Literal["gate", "finish"] | None


def _forward_crossing_fraction(prev_x: float, curr_x: float, plane_x: float) -> float | None:
    """Return the interpolation fraction ``t`` in ``[0, 1]`` where a forward crossing of
    ``plane_x`` happens, or ``None`` if the segment does not cross it going ``+x``.

    Forward means ``prev_x < plane_x <= curr_x``. A stationary or backward segment yields
    ``None`` (backward crossings never count).
    """
    if curr_x <= prev_x:  # not moving forward across the plane
        return None
    if prev_x < plane_x <= curr_x:
        return (plane_x - prev_x) / (curr_x - prev_x)
    return None


def _segment_point_distance(
    prev_pos: np.ndarray, curr_pos: np.ndarray, point: np.ndarray
) -> float:
    """Closest distance from ``point`` to the segment ``prev_pos→curr_pos`` (3D).

    Degrades to the point-to-``prev_pos`` distance when the segment has ~zero length (a
    near-stationary step), so proximity detection still works while hovering.
    """
    seg = curr_pos - prev_pos
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq <= 1e-18:  # near-zero-length segment: treat as a point
        return float(np.linalg.norm(point - prev_pos))
    t = float(np.dot(point - prev_pos, seg) / seg_len_sq)
    t = min(1.0, max(0.0, t))  # clamp to the segment
    closest = prev_pos + t * seg
    return float(np.linalg.norm(point - closest))


def gate_reached(prev_pos: np.ndarray, curr_pos: np.ndarray, gate: GateSpec) -> bool:
    """Return ``True`` iff the step segment comes within ``gate.aperture`` of ``gate.center``.

    Uses the **segment-to-point closest distance** (UC-09 AC2): the drone need only pass
    *through* the capture sphere during the step, so a single fast step that would tunnel a
    naive endpoint check still registers the gate.
    """
    prev_pos = np.asarray(prev_pos, dtype=np.float64)
    curr_pos = np.asarray(curr_pos, dtype=np.float64)
    dist = _segment_point_distance(prev_pos, curr_pos, gate.position)
    return bool(dist <= gate.aperture)


def finish_crossed(prev_pos: np.ndarray, curr_pos: np.ndarray, course: CourseConfig) -> bool:
    """Return ``True`` iff the segment forward-crosses the finish plane at ``finish_x``.

    The finish is a full plane (no aperture); gate-ordering vs. the finish is enforced by
    the indexed walk in :func:`advance`, not here.
    """
    prev_pos = np.asarray(prev_pos, dtype=np.float64)
    curr_pos = np.asarray(curr_pos, dtype=np.float64)
    return _forward_crossing_fraction(prev_pos[0], curr_pos[0], course.finish_x) is not None


def current_target(course: CourseConfig, gates_passed: int) -> np.ndarray:
    """World-frame position of the current target waypoint (UC-09).

    Targets ``gates[gates_passed]`` while gates remain; once all N gates are passed it
    targets the finish-line point straight ahead, inheriting the **last gate's** ``(y, z)``
    so the finish sits on the natural continuation of the course. Used by the env to build
    the relative-waypoint pose in the observation and by the reward's progress term.
    """
    n = course.num_gates
    if gates_passed < n:
        return course.gates[gates_passed].position
    last = course.gates[-1]
    return np.asarray([course.finish_x, last.center[1], last.center[2]], dtype=np.float64)


def advance(
    course: CourseConfig,
    gates_passed: int,
    done: bool,
    prev_pos: np.ndarray,
    curr_pos: np.ndarray,
) -> tuple[int, bool, Event]:
    """Advance the indexed walk one step; return ``(gates_passed, done, event)``.

    ``event`` is ``"gate"`` on the step the current gate is passed, ``"finish"`` on the step
    the course is validly completed (all gates passed **then** the finish crossed), else
    ``None``. Only the *current* gate is tested, so gates can never be skipped (UC-09 AC2).

    Note: a single step must not span two consecutive gates — ``min_gate_spacing`` is kept
    ``≫`` the per-step travel (a step advances far less than the minimum 3D gate spacing),
    so at most one gate is credited per step even at high N.
    """
    n = course.num_gates
    if gates_passed < n:
        if gate_reached(prev_pos, curr_pos, course.gates[gates_passed]):
            return gates_passed + 1, done, "gate"
        return gates_passed, done, None
    if not done and finish_crossed(prev_pos, curr_pos, course):
        return gates_passed, True, "finish"
    return gates_passed, done, None
