"""Sysid safety checks: pure functions that name every violation and the action it calls for.

Actions (the session applies them; nothing here raises):

* ``abort_experiment`` — stop the running experiment, re-stabilize (land + disarm if that fails)
* ``failsafe``        — neutral sticks + throttle cut + RPM-confirmed disarm; the session ends
  resumable (stale telemetry, focus loss, F12)
* ``crash``           — fail-safe + disarm, clean end, the next experiment is recorded for resume

Bounds: the soft altitude floor / ceiling and the tilt bound apply outside the active experiment
window only; the hard floor and the horizontal geofence apply always; crash detection (a velocity
jump, a respawn, inverted near the floor) applies always.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drone_fly.harness.geometry import quat_to_matrix

ABORT_EXPERIMENT = "abort_experiment"
FAILSAFE = "failsafe"
CRASH = "crash"


@dataclass(frozen=True)
class Violation:
    name: str
    action: str
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "action": self.action, "detail": self.detail}


@dataclass(frozen=True)
class SafetyLimits:
    floor_m: float = 1.5
    ceiling_m: float = 12.0
    tilt_deg: float = 50.0
    hard_floor_m: float = 0.6
    radius_m: float = 30.0
    stale_after_s: float = 0.25
    crash_dv_m_s: float = 3.0
    crash_dt_s: float = 0.05
    respawn_jump_m: float = 5.0
    inverted_alt_m: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.hard_floor_m <= self.floor_m < self.ceiling_m:
            raise ValueError("need 0 <= hard_floor_m <= floor_m < ceiling_m")
        if self.radius_m <= 0 or self.tilt_deg <= 0:
            raise ValueError("radius_m and tilt_deg must be positive")


def check_state(
    state: Any,
    *,
    home: np.ndarray,
    ground_z: float,
    limits: SafetyLimits,
    soft: bool,
) -> list[Violation]:
    """Flight-envelope checks on one state (``soft``: outside the experiment window)."""
    out: list[Violation] = []
    alt = float(state.pos[2]) - ground_z
    r = quat_to_matrix(state.quat)
    up = float(r[2, 2])
    if up < 0.0 and alt < limits.inverted_alt_m:
        out.append(Violation("crash_inverted", CRASH, f"inverted at {alt:.2f} m"))
    if alt < limits.hard_floor_m:
        out.append(Violation("hard_floor", ABORT_EXPERIMENT, f"altitude {alt:.2f} m"))
    dist = float(np.linalg.norm(np.asarray(state.pos, np.float64)[:2] - np.asarray(home)[:2]))
    if dist > limits.radius_m:
        out.append(Violation("geofence", ABORT_EXPERIMENT, f"{dist:.1f} m from home"))
    if soft:
        if alt < limits.floor_m:
            out.append(Violation("altitude_floor", ABORT_EXPERIMENT, f"altitude {alt:.2f} m"))
        if alt > limits.ceiling_m:
            out.append(Violation("altitude_ceiling", ABORT_EXPERIMENT, f"altitude {alt:.2f} m"))
        tilt = float(np.degrees(np.arccos(np.clip(up, -1.0, 1.0))))
        if tilt > limits.tilt_deg:
            out.append(Violation("tilt", ABORT_EXPERIMENT, f"tilt {tilt:.0f} deg"))
    return out


def check_samples(prev: Any, samples: list[Any], limits: SafetyLimits) -> list[Violation]:
    """Crash detection over consecutive raw telemetry samples (``prev`` may be ``None``)."""
    out: list[Violation] = []
    last = prev
    for s in samples:
        if last is not None:
            dt = float(s.t) - float(last.t)
            jump = float(np.linalg.norm(np.asarray(s.position) - np.asarray(last.position)))
            if dt < 0 or jump > limits.respawn_jump_m:
                out.append(Violation("crash_respawn", CRASH, f"position jump {jump:.1f} m"))
            elif 0 < dt <= limits.crash_dt_s:
                dv = float(np.linalg.norm(np.asarray(s.velocity) - np.asarray(last.velocity)))
                if dv > limits.crash_dv_m_s:
                    out.append(
                        Violation(
                            "crash_impact", CRASH, f"velocity jump {dv:.1f} m/s in {dt:.3f} s"
                        )
                    )
        last = s
    return out


def check_link(
    *, age: float, focus: bool | None, abort: bool, limits: SafetyLimits
) -> list[Violation]:
    """Telemetry age, game focus, the F12 abort hotkey."""
    out: list[Violation] = []
    if abort:
        out.append(Violation("user_abort", FAILSAFE, "abort hotkey"))
    if focus is False:
        out.append(Violation("focus_lost", FAILSAFE, "the game window lost focus"))
    if age > limits.stale_after_s:
        out.append(Violation("stale_telemetry", FAILSAFE, f"telemetry age {age:.3f} s"))
    return out


def worst(violations: list[Violation]) -> Violation | None:
    """The violation to act on: crash > failsafe > abort_experiment."""
    for action in (CRASH, FAILSAFE, ABORT_EXPERIMENT):
        for v in violations:
            if v.action == action:
                return v
    return None
