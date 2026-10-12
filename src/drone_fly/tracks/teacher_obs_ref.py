"""NumPy single-drone reference of the ``teacher-obs`` v1 observation (UC68 D2).

The batched torch builder (:mod:`drone_fly.sim.teacher_obs`) is tested against this, and the
UC69 Liftoff oracle builds its observation with it. ``harness30`` goes through the REAL harness
code: :class:`~drone_fly.harness.assembler.ContractObservationAssembler` (the contract's
``make_obs``) and :mod:`drone_fly.harness.course_senses` over :func:`to_harness_course` (the
course's harness obstacle list). The other blocks use the :mod:`drone_fly.sim.pool` references.

``state`` is ``{"pos", "vel", "quat" ([w, x, y, z], body -> world), "omega" (body)}`` in the
course frame (:class:`~drone_fly.sim.pool.CourseFrame`; the oracle maps telemetry through it).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.course import Course, Gate, Obstacle
from drone_fly.harness.course_senses import (
    curr_gate_block,
    next_lookahead_block,
    obstacle_vision_block,
)
from drone_fly.harness.geometry import quat_to_matrix
from drone_fly.harness.types import FlightState, FreeSpaceEstimate, GateEstimate
from drone_fly.sim.pool import (
    RouteProgress,
    SimCourse,
    floor_height,
    nearest_obstacles,
    obstacle_candidates,
    support_top,
)
from drone_fly.sim.teacher_obs import DIM, GATES_AHEAD, N_OBSTACLES, OBS_RANGE_M

_F = np.float32


def to_harness_course(course: SimCourse) -> Course:
    """The harness ``Course``: per-lap passages as gates + the harness obstacle list (AABBs)."""
    gates = tuple(Gate(c, float(y), float(a), cls) for c, y, a, cls in course.harness_gates())
    obstacles = tuple(
        Obstacle(np.asarray(c, _F), np.asarray(h, _F), "box")
        for c, h in zip(course.harness.center, course.harness.half, strict=True)
    )
    return Course(gates=gates, obstacles=obstacles, start=np.asarray(course.spawn_pos, _F),
                  laps=course.laps)  # fmt: skip


def default_contract() -> Any:
    from drone_fly.harness.contract import load_contract, loaded_contract

    return loaded_contract() or load_contract(device="cpu")


def harness30(
    state: dict[str, Any],
    course: SimCourse,
    progress: RouteProgress,
    contract: Any = None,
    harness_course: Course | None = None,
) -> np.ndarray:
    hc = harness_course or to_harness_course(course)
    pos = np.asarray(state["pos"], _F)
    n = course.n_passages
    gate = hc.gates[progress.k % n]
    nxt = hc.gates[(progress.k + 1) % n]
    est = GateEstimate(
        target=np.asarray(gate.center, _F),
        curr_gate=curr_gate_block(gate),
        next_lookahead=next_lookahead_block(pos, nxt, hc.obstacles),
    )
    fs = FreeSpaceEstimate(obstacle_vision=obstacle_vision_block(pos, list(hc.obstacles)))
    fl = FlightState.from_state_dict(state)
    asm = ContractObservationAssembler(contract or default_contract())
    _, extra = asm.assemble(fl, est, fs, 30)
    return extra


def build_teacher_obs(
    state: dict[str, Any],
    course: SimCourse,
    progress: RouteProgress,
    prev_action: Any,
    arm_remaining_s: float,
    *,
    contract: Any = None,
    harness_course: Course | None = None,
) -> np.ndarray:
    """The 148-value observation of one drone (see :mod:`drone_fly.sim.teacher_obs`)."""
    pos = np.asarray(state["pos"], np.float64)
    vel = np.asarray(state["vel"], np.float64)
    R = quat_to_matrix(np.asarray(state["quat"], np.float64))
    parts: list[np.ndarray] = [
        harness30(state, course, progress, contract, harness_course).astype(np.float64),
        R[:, 0],
        R[:, 1],
        R.T @ vel,
    ]
    for a in range(GATES_AHEAD):
        j, valid = progress.lookahead(a)
        p = course.passages[j]
        rel = p.center - pos
        row = np.concatenate([R.T @ rel, R.T @ p.rot[:, 2], R.T @ p.rot[:, 1],
                              [p.w, p.h, np.linalg.norm(rel), 1.0]])  # fmt: skip
        parts.append(row * float(valid))
    near = nearest_obstacles(course, pos, N_OBSTACLES, OBS_RANGE_M)
    ob = np.zeros((N_OBSTACLES, 8))
    for i, (pt, d, nrm) in enumerate(near):
        ob[i] = np.concatenate([R.T @ (pt - pos), [d], R.T @ nrm, [1.0]])
    parts.append(ob.reshape(-1))
    fh = float(floor_height(course, pos[0], pos[1]))
    idx = obstacle_candidates(course, pos)
    idx = idx[idx >= 0]
    top = None
    if len(idx):
        top = support_top(course.prims.subset(idx), pos[0], pos[1], pos[2], -math.inf)
    parts.append(np.array([pos[2] - max(fh, top if top is not None else -math.inf)]))
    parts.append(np.asarray(prev_action, np.float64).reshape(4))
    parts.append(np.array([max(0.0, float(arm_remaining_s))]))
    out = np.concatenate(parts).astype(_F)
    assert out.shape == (DIM,)
    return out
