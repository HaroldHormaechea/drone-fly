"""Deploying a UC70 student checkpoint in the live harness (`harness run --checkpoint`).

* :class:`StudentStepState` extends UC69's :class:`~drone_fly.harness.teacher.TeacherStepState`
  (the previous tick's RAW action, the sink's arming time left) with the observation history and
  the previous tick's clock reading (the EMA ``dt``; ``gate.stamp`` is a frame stamp and repeats
  while frames are reused).
* :class:`StudentObservationAssembler` builds ``student-obs`` v1 with
  :func:`drone_fly.student.obs_ref.build_student_obs` — the SAME function the training-side
  reference uses, so the two are bit-identical by construction. It needs the gate estimator's
  ``target`` / ``curr_gate`` / ``next_lookahead`` blocks (the learned detector); free space is
  never read (``obstacle_vision`` is zeros, as trained).

Nothing here imports torch at module level.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from drone_fly.harness.errors import PerceptionCoverageError
from drone_fly.harness.interfaces import ObservationAssembler
from drone_fly.harness.teacher import TeacherStepState
from drone_fly.harness.types import FlightState, FreeSpaceEstimate, GateEstimate


@dataclass
class StudentStepState(TeacherStepState):
    """Per-run state shared by the assembler and the student policy."""

    history: Any = None
    last_now: float | None = None
    #: The tick clock for the EMA ``dt`` (the runner's perf_counter; injectable for tests).
    clock: Any = time.perf_counter
    timings: dict[str, list[float]] = field(default_factory=lambda: {"assemble": [], "policy": []})

    def __post_init__(self) -> None:
        if self.history is None:
            from drone_fly.student.obs_ref import StudentHistory

            self.history = StudentHistory()

    def reset(self) -> None:
        """Session start: zero the action and the history."""
        self.prev_action = np.zeros(4, np.float32)
        self.history.reset()
        self.last_now = None
        self._t0 = None


class StudentObservationAssembler(ObservationAssembler):
    """``student-obs`` v1 (see the module docstring); ``state`` arrives in student units."""

    def __init__(self, contract: Any, step: StudentStepState, *, lost_timeout_s: float) -> None:
        self.contract = contract
        self.step = step
        self.lost_timeout_s = float(lost_timeout_s)

    def required_blocks(self, extra_dim: int) -> tuple[str, ...]:
        return ("target", "curr_gate", "next_lookahead")

    def check_coverage(self, extra_dim: int, provided: frozenset[str] | set[str]) -> None:
        missing = [b for b in self.required_blocks(extra_dim) if b not in provided]
        if missing:
            raise PerceptionCoverageError(missing, extra_dim)

    def assemble(
        self,
        state: FlightState,
        gate: GateEstimate,
        free_space: FreeSpaceEstimate | None,
        extra_dim: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        from drone_fly.student.obs_ref import build_student_obs

        t0 = time.perf_counter()
        now = float(self.step.clock())
        dt = 0.0 if self.step.last_now is None else max(now - self.step.last_now, 0.0)
        self.step.last_now = now
        extra = build_student_obs(
            self.contract,
            state,
            gate,
            prev_action=self.step.prev_action,
            arm_remaining_s=self.step.arm_remaining_s(float(gate.stamp)),
            lost_timeout_s=self.lost_timeout_s,
            history=self.step.history,
            dt=dt,
        )
        self.step.timings["assemble"].append(time.perf_counter() - t0)
        if extra.shape[0] != extra_dim:
            raise ValueError(f"student observation has {extra.shape[0]} values, not {extra_dim}")
        return extra[:12].copy(), extra
