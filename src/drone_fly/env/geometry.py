"""Pure waypoint logic and the start→gate→finish phase machine (AC2).

No sim, no torch, no env state beyond what is passed in — every function here is pure and
unit-testable on hand-built position transitions. The env drives this each step to decide
phase transitions and course completion.

Rules
-----
* The course has two ordered waypoints: a **gate** (a disc aperture at ``gate_x``) and a
  **finish** plane at ``finish_x``.
* A crossing is only counted **forward** (``+x``): the segment from the previous to the
  current position must go from before the plane to on/after it.
* The **gate** additionally requires the forward crossing point to fall within the
  aperture disc (radius ``gate_aperture``) in the y–z plane.
* **Gate-before-finish is enforced:** the finish only completes the course while the phase
  is ``TO_FINISH`` (i.e. the gate was already validly passed). A finish crossing while
  still ``TO_GATE`` is ignored — you cannot skip the gate.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from drone_fly.env.config import CourseConfig

#: Phase machine states.
TO_GATE = "to_gate"
TO_FINISH = "to_finish"
DONE = "done"

Phase = Literal["to_gate", "to_finish", "done"]
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


def gate_passed(prev_pos: np.ndarray, curr_pos: np.ndarray, course: CourseConfig) -> bool:
    """Return ``True`` iff the segment ``prev_pos→curr_pos`` passes through the gate disc.

    A forward crossing of the ``gate_x`` plane whose interpolated ``(y, z)`` lies within
    ``gate_aperture`` of the gate centre.
    """
    prev_pos = np.asarray(prev_pos, dtype=np.float64)
    curr_pos = np.asarray(curr_pos, dtype=np.float64)
    t = _forward_crossing_fraction(prev_pos[0], curr_pos[0], course.gate_x)
    if t is None:
        return False
    cross = prev_pos + t * (curr_pos - prev_pos)
    dy = cross[1] - course.gate_center_y
    dz = cross[2] - course.gate_center_z
    return bool(np.hypot(dy, dz) <= course.gate_aperture)


def finish_crossed(prev_pos: np.ndarray, curr_pos: np.ndarray, course: CourseConfig) -> bool:
    """Return ``True`` iff the segment forward-crosses the finish plane at ``finish_x``.

    The finish is a full plane (no aperture); ordering vs. the gate is enforced by the
    phase machine in :func:`advance_phase`, not here.
    """
    prev_pos = np.asarray(prev_pos, dtype=np.float64)
    curr_pos = np.asarray(curr_pos, dtype=np.float64)
    return _forward_crossing_fraction(prev_pos[0], curr_pos[0], course.finish_x) is not None


def advance_phase(
    phase: Phase,
    prev_pos: np.ndarray,
    curr_pos: np.ndarray,
    course: CourseConfig,
) -> tuple[Phase, Event]:
    """Advance the phase machine one step; return ``(new_phase, event)``.

    ``event`` is ``"gate"`` on the step the gate is validly passed, ``"finish"`` on the
    step the course is validly completed, else ``None``. Gate-before-finish is enforced:
    a finish crossing while ``TO_GATE`` produces no event and no phase change.
    """
    if phase == TO_GATE:
        if gate_passed(prev_pos, curr_pos, course):
            return TO_FINISH, "gate"
        return TO_GATE, None
    if phase == TO_FINISH:
        if finish_crossed(prev_pos, curr_pos, course):
            return DONE, "finish"
        return TO_FINISH, None
    return DONE, None


def target_position(phase: Phase, course: CourseConfig) -> np.ndarray:
    """World-frame position of the current target waypoint for the given phase.

    ``TO_GATE`` targets the gate centre; ``TO_FINISH`` / ``DONE`` target the finish-line
    point straight ahead of the gate. Used by the env to build the relative-waypoint pose
    in the observation and by the reward's progress term.
    """
    if phase == TO_GATE:
        return course.gate_center
    return np.asarray(
        [course.finish_x, course.gate_center_y, course.gate_center_z], dtype=np.float64
    )
