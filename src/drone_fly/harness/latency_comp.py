"""Smith-predictor latency compensation for gpu_prototype contract policies (``runner.latency_*``).

A live Liftoff command reaches the measured state ``delay_s`` (D) after it is sent (gamepad,
game physics, telemetry). The policy is memoryless and was trained on the plant's own state, so
it reacts to a state that is already old and ignores the commands still "in flight". The
compensator forecasts the state the next command will act on:

* ``r`` is the measured state's ``received_at`` and ``now`` the time the observation is assembled;
  the horizon is ``H = (now - r) + D``, rolled as ``n = round(H / DT)`` contract ``plant_step``
  steps (capped at ``max_horizon_s``);
* at step ``j`` (time ``r + j*DT``) the plant applies the newest command sent at or before
  ``r + j*DT - D`` — the command the measured response is reacting to at that instant. Before the
  retained history starts it applies the pre-history command (the newest command evicted from the
  history); before any command at all the step is skipped (nothing was flying the drone yet).

``predict`` picks what is rolled: ``rotation`` (default) rolls ``plant_step`` but keeps only the
forecast attitude and body rates (scale-invariant), holding the measured velocity and
extrapolating the position as ``pos + vel * h``; ``full`` keeps the whole forecast state. The
training plant's translational dynamics do not match a scaled live world (V1).

Only perception keeps the measured state; the assembler alone gets the forecast. Training already
randomizes a 0-2 step (0-40 ms) action delay, so the delay is compensated in full rather than
tuned to "leave some". Raw actions are only meaningful to ``plant_step`` for action-spec v1, so a
v2 policy resolves ``auto`` off (warned) and refuses an explicit ``on``. See
docs/harness/liftoff-live-session.md.
"""

from __future__ import annotations

import collections
import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.types import FlightState

log = logging.getLogger(__name__)

MODES = ("off", "auto", "on")
PREDICTS = ("rotation", "full")
#: Default ``runner.latency_delay_s``: command sent -> measured gyro response on the 2026-10-18
#: recordings (tick -> gyro ~75 ms minus the ~35 ms loop before the send; the cleanest run's
#: attitude / rate replay is best at 35-45 ms).
DEFAULT_DELAY_S = 0.04
#: Whether ``auto`` may turn the compensator on for a live contract checkpoint. False: the V1
#: real-data replay (2026-10-18, five runs pooled) found the forecast worse than the stale state
#: on velocity (~10x: plant_step is the training whoop, the live world is scaled) and attitude
#: (beyond ~40 ms). ``on`` still forces it.
AUTO_LIVE = False
MAX_HORIZON_S = 0.2
HISTORY = 64


def predict_from(
    state: dict[str, np.ndarray],
    r: float,
    now: float,
    commands: Sequence[tuple[float, np.ndarray]],
    delay_s: float,
    dt: float,
    plant_step: Callable[[dict[str, np.ndarray], np.ndarray], dict[str, np.ndarray]],
    *,
    max_horizon_s: float = MAX_HORIZON_S,
    pre: np.ndarray | None = None,
    predict: str = "full",
) -> tuple[dict[str, np.ndarray], int]:
    """Roll ``state`` (measured at ``r``) forward to ``now + delay_s`` through the sent commands.

    ``commands`` are ``(sent_at, raw_action)`` pairs in send order. Returns the forecast state and
    the number of plant steps taken (``state`` itself and 0 when there is nothing to roll).
    ``predict="rotation"`` keeps only the forecast ``quat`` / ``omega``: ``vel`` stays measured and
    ``pos`` moves by ``vel`` over the steps taken.
    """
    if not commands:
        return state, 0
    horizon = min(float(now - r) + float(delay_s), float(max_horizon_s))
    n = int(round(horizon / dt))
    if n <= 0:
        return state, 0
    sent = np.fromiter((c[0] for c in commands), np.float64, len(commands))
    s = state
    taken = 0
    for j in range(n):
        i = int(np.searchsorted(sent, r + j * dt - delay_s, side="right")) - 1
        action = commands[i][1] if i >= 0 else pre
        if action is None:
            continue
        s = plant_step(s, action)
        taken += 1
    if predict == "rotation" and taken:
        vel = np.asarray(state["vel"], np.float32)
        s = {
            "pos": (np.asarray(state["pos"], np.float32) + vel * np.float32(taken * dt)),
            "vel": vel.copy(),
            "quat": s["quat"],
            "omega": s["omega"],
        }
    return s, taken


class LatencyCompensator:
    """Records sent commands and forecasts the policy's state (:func:`predict_from`)."""

    def __init__(
        self,
        mode: str = "auto",
        delay_s: float = DEFAULT_DELAY_S,
        *,
        plant_step: Callable[..., Any] | None = None,
        dt: float = 0.02,
        max_horizon_s: float = MAX_HORIZON_S,
        eligible: bool = True,
        predict: str = "rotation",
    ) -> None:
        if predict not in PREDICTS:
            raise HarnessConfigError(
                f"runner.latency_predict must be rotation or full, got {predict!r}"
            )
        self.predict_mode = predict
        if mode not in MODES:
            raise HarnessConfigError(
                f"runner.latency_compensation must be off, auto or on, got {mode!r}"
            )
        self.mode = mode
        self.delay_s = float(delay_s)
        self.plant_step = plant_step
        self.dt = float(dt)
        self.max_horizon_s = float(max_horizon_s)
        #: auto only: the session's verdict (contract checkpoint on the wall clock).
        self.eligible = bool(eligible)
        self.enabled = mode == "on" or (mode == "auto" and self.eligible and AUTO_LIVE)
        self.last_steps = 0
        self.reset()

    def resolve(self, contract_version: int | None) -> bool:
        """Final on/off once the checkpoint's action contract is known (after the model load)."""
        if contract_version is not None and int(contract_version) >= 2:
            if self.mode == "on":
                raise HarnessConfigError(
                    "runner.latency_compensation: on needs an action-spec v1 checkpoint (the "
                    f"predictor steps plant_step on raw actions); this one is v{contract_version}"
                )
            if self.enabled:
                log.warning("latency compensation off: action-spec v%s policy", contract_version)
            self.enabled = False
        if self.enabled and self.plant_step is None:
            raise HarnessConfigError("latency compensation needs the contract plant_step")
        return self.enabled

    def reset(self) -> None:
        self._commands: collections.deque[tuple[float, np.ndarray]] = collections.deque()
        self._pre: np.ndarray | None = None

    def record(self, sent_at: float, command: Any) -> None:
        """Remember a command actually sent at ``sent_at`` (its raw action)."""
        if not self.enabled:
            return
        raw = np.asarray(getattr(command, "raw", command), np.float32).reshape(-1).copy()
        if len(self._commands) >= HISTORY:
            self._pre = self._commands.popleft()[1]
        self._commands.append((float(sent_at), raw))

    def predict(self, state: FlightState, now: float) -> FlightState:
        """The forecast :class:`FlightState` (``state`` itself when disabled or nothing to roll)."""
        self.last_steps = 0
        if not self.enabled or not self._commands:
            return state
        out, n = predict_from(
            state.as_state_dict(),
            float(state.received_at),
            float(now),
            self._commands,
            self.delay_s,
            self.dt,
            self.plant_step,  # type: ignore[arg-type]
            max_horizon_s=self.max_horizon_s,
            pre=self._pre,
            predict=self.predict_mode,
        )
        self.last_steps = n
        if n == 0:
            return state
        return replace(
            state,
            pos=np.asarray(out["pos"], np.float32),
            vel=np.asarray(out["vel"], np.float32),
            quat=np.asarray(out["quat"], np.float32),
            omega=np.asarray(out["omega"], np.float32),
        )
