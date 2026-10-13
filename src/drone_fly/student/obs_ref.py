"""``build_student_obs``: ``student-obs`` v1 through the REAL harness assembler (UC70 D4).

Both the training side (references, parity tests) and the live harness
(:class:`~drone_fly.harness.student.StudentObservationAssembler`) call :func:`build_student_obs`,
so the two are bit-identical by construction. Inputs are already in **student units** (the
harness's training scale ``k``): ``state_k`` is the harness :class:`FlightState` (positions /
velocities / k) and ``est_k`` the tracker's :class:`GateEstimate` with the UC70 ``info`` keys
(``matched``, ``target_age_s``, ``next_age_s``, ``target_valid``, ``hand_over``). The layout is
:mod:`drone_fly.sim.student_obs`'s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.geometry import quat_to_matrix
from drone_fly.harness.types import FlightState, FreeSpaceEstimate, GateEstimate
from drone_fly.sim.student_obs import DIM, EMA_TAU_S, TICKS_NORM

_F = np.float32


@dataclass
class StudentHistory:
    """The ``history`` block's state (reset at episode / session start)."""

    ema_rel: np.ndarray = field(default_factory=lambda: np.zeros(3, _F))
    ema_act: np.ndarray = field(default_factory=lambda: np.zeros(4, _F))
    prev_target: np.ndarray = field(default_factory=lambda: np.zeros(3, _F))
    prev_valid: bool = False
    ticks: int = 0
    init: bool = False

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    def step(
        self, R: np.ndarray, pos: np.ndarray, target: np.ndarray, prev_action: np.ndarray,
        matched: bool, hand_over: bool, valid: bool, dt: float,
    ) -> np.ndarray:  # fmt: skip
        """Advance and return the 12 history values (float32)."""
        R = R.astype(_F)
        rel_b = (R.T @ (np.asarray(target, _F) - np.asarray(pos, _F))).astype(_F)
        a = _F(1.0 - math.exp(-dt / EMA_TAU_S)) if dt > 0 else _F(1.0)
        pa = np.asarray(prev_action, _F)
        if not self.init:
            self.ema_rel, self.ema_act = rel_b.copy(), pa.copy()
            self.init = True
        else:
            self.ema_rel = (self.ema_rel + a * (rel_b - self.ema_rel)).astype(_F)
            self.ema_act = (self.ema_act + a * (pa - self.ema_act)).astype(_F)
        t = np.asarray(target, _F)
        use = matched and not hand_over and self.prev_valid
        delta = (R.T @ (t - self.prev_target)).astype(_F) if use else np.zeros(3, _F)
        self.prev_target = t.copy()
        self.prev_valid = bool(valid)
        self.ticks = 0 if matched else self.ticks + 1
        tn = _F(min(self.ticks, TICKS_NORM) / TICKS_NORM)
        return np.concatenate([self.ema_rel, self.ema_act, delta, [tn, _F(hand_over)]]).astype(_F)


def build_student_obs(
    contract: Any,
    state_k: FlightState,
    est_k: GateEstimate,
    *,
    prev_action: np.ndarray,
    arm_remaining_s: float,
    lost_timeout_s: float,
    history: StudentHistory,
    dt: float,
) -> np.ndarray:
    """The 59 float32 values (see :mod:`drone_fly.sim.student_obs`); advances ``history``."""
    asm = ContractObservationAssembler(contract)
    fs = FreeSpaceEstimate(obstacle_vision=np.zeros(8, _F))
    _, h30 = asm.assemble(state_k, est_k, fs, 30)
    R = quat_to_matrix(state_k.quat).astype(_F)
    vb = (R.T @ np.asarray(state_k.vel, _F)).astype(_F)
    info = est_k.info or {}
    T = float(lost_timeout_s)
    valid = bool(info.get("target_valid", est_k.status != "lost"))
    perc = np.array([
        min(float(info.get("target_age_s", T)), T) / T,
        1.0 if valid else 0.0,
        min(float(info.get("next_age_s", T)), T) / T,
    ], _F)  # fmt: skip
    hist = history.step(
        R, np.asarray(state_k.pos, _F), np.asarray(est_k.target, _F), prev_action,
        bool(info.get("matched", False)), bool(info.get("hand_over", False)), valid, dt,
    )  # fmt: skip
    out = np.concatenate([
        np.asarray(h30, _F), R[:, 0], R[:, 1], vb, np.asarray(prev_action, _F),
        [_F(arm_remaining_s)], perc, hist,
    ]).astype(_F)  # fmt: skip
    assert out.shape[0] == DIM
    return out


def scale_state(state: Any, k: float) -> FlightState:
    """A course-frame state dict / ``FlightState`` -> student units (``WorldScale(k)``)."""
    if isinstance(state, FlightState):
        s = state.as_state_dict()
        t = state.t
    else:
        s = state
        t = 0.0
    kk = _F(k)
    return FlightState(
        pos=(np.asarray(s["pos"], _F) / kk).astype(_F),
        vel=(np.asarray(s["vel"], _F) / kk).astype(_F),
        quat=np.asarray(s["quat"], _F),
        omega=np.asarray(s["omega"], _F),
        t=t,
    )
