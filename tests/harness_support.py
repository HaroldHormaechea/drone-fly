"""Shared fakes and helpers for the UC-62 live-harness tests (``tests/test_harness_*.py``).

Nothing here touches ``gpu_prototype/``: tests that need the real contract use the session
fixture ``harness_contract`` from ``conftest.py``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from drone_fly.harness.interfaces import (
    AbortSignal,
    ActionSink,
    FreeSpaceEstimator,
    GateEstimator,
    PolicyRunner,
    StateSource,
)
from drone_fly.harness.types import (
    ContractCommand,
    FlightState,
    Frame,
    FreeSpaceEstimate,
    GateEstimate,
)

DATA_DIR = Path(__file__).parent / "data" / "harness"

#: The exact header of the owner's recorded Liftoff telemetry CSVs (AC5 synthetic fixtures).
RECORDED_CSV_HEADER = (
    "t,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w,vel_x,vel_y,vel_z,gyro_pitch,gyro_roll,"
    "gyro_yaw,in_throttle,in_yaw,in_pitch,in_roll,batt_charge,batt_volts,rpm_0,rpm_1,rpm_2,rpm_3"
)

#: UC-64 AC13: local opt-in real-data tests read the gitignored Liftoff data from the folder this
#: variable names (the main checkout's ``artifacts/harness``: ``captures/<name>/`` + ``trackdata/``)
#: and skip when it is unset or incomplete. Nothing from it is ever committed.
LIFTOFF_DATA_ENV = "DRONE_FLY_LIFTOFF_DATA"
REAL_CAPTURE_NAME = "2026-10-11-infinity-manual"
REAL_TRACK_XML = "LiftoffArenaTrack04_0001.xml"
REAL_RACE_XML = "LiftoffArenaRace04_0001.xml"


def liftoff_real_data() -> SimpleNamespace:
    """Paths of the real Infinity Loop data, or ``pytest.skip`` when absent (local opt-in)."""
    import os

    import pytest

    root = os.environ.get(LIFTOFF_DATA_ENV)
    if not root:
        pytest.skip(f"real Liftoff data not configured (set {LIFTOFF_DATA_ENV})")
    base = Path(root)
    data = SimpleNamespace(
        root=base,
        capture=base / "captures" / REAL_CAPTURE_NAME,
        track=base / "trackdata" / REAL_TRACK_XML,
        race=base / "trackdata" / REAL_RACE_XML,
        props=base / "trackdata" / "props.json",
    )
    missing = [
        str(p)
        for p in (data.capture / "telemetry_raw.csv", data.track, data.race, data.props)
        if not p.is_file()
    ]
    if missing:
        pytest.skip(f"real Liftoff data incomplete under {base}: missing {missing}")
    return data


#: Modules the hermetic (CI) path must never need.
HEAVY_MODULES = ("torch", "cv2", "mss", "dxcam", "transformers", "vgamepad", "PIL", "PySide6")


def fake_contract(**overrides: Any) -> SimpleNamespace:
    """A minimal stand-in for the contract facade (constants + a trivial ``make_obs``)."""

    def make_obs(s: dict[str, np.ndarray], target: np.ndarray) -> np.ndarray:
        rel = np.asarray(target, np.float32) - s["pos"]
        return np.concatenate([rel, [0.0, 0.0, -1.0], s["vel"], s["omega"]]).astype(np.float32)

    base = dict(
        MAX_THRUST=2.5 * 0.032 * 9.81,
        MAX_BODY_RATE=4.0,
        HOVER_THR=0.4,
        DT=0.02,
        N_RAYS=40,
        RAY_RANGE=4.0,
        make_obs=make_obs,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def hover_state(pos=(0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), t: float = 0.0) -> FlightState:
    return FlightState(
        pos=np.array(pos, np.float32),
        vel=np.array(vel, np.float32),
        quat=np.array([1.0, 0.0, 0.0, 0.0], np.float32),
        omega=np.zeros(3, np.float32),
        t=t,
    )


class FakeStateSource(StateSource):
    """Returns a fixed state; ``age_s`` / ``state`` / ``discontinuity_count`` are mutable."""

    def __init__(self, state: FlightState | None = None, age_s: float = 0.0) -> None:
        self.state = state if state is not None else hover_state()
        self.age_s = age_s
        self.discontinuity_count = 0
        self.started = self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def latest(self) -> FlightState | None:
        return self.state

    def age(self, now: float) -> float:
        return float("inf") if self.state is None else self.age_s


class FakeGateEstimator(GateEstimator):
    provides = frozenset({"target", "curr_gate", "next_lookahead"})
    provenance = "screen"  # UC-86: stands in for an on-screen detector (live guard)

    def __init__(self, status: str = "tracking", raises: BaseException | None = None) -> None:
        self.status = status
        self.raises = raises
        self.calls = 0
        self.resets = 0

    def update(self, frame: Frame | None, state: FlightState, now: float) -> GateEstimate:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return GateEstimate(
            target=np.array([2.0, 0.0, 1.0], np.float32),
            curr_gate=np.array([1.0, 0.0, 1.0], np.float32),
            next_lookahead=np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.2, 1.0], np.float32),
            status=self.status,  # type: ignore[arg-type]
            stamp=now,
        )

    def reset(self) -> None:
        self.resets += 1


class FakeFreeSpaceEstimator(FreeSpaceEstimator):
    provides = frozenset({"obstacle_vision", "raycast"})
    provenance = "screen"

    def __init__(self, age_s: float = 0.0, raises: BaseException | None = None) -> None:
        self.age_s = age_s
        self.raises = raises
        self.calls = 0
        self.resets = 0
        self.started = self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def update(
        self, frame: Frame | None, state: FlightState, target: np.ndarray, now: float
    ) -> FreeSpaceEstimate:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return FreeSpaceEstimate(
            obstacle_vision=np.zeros(8, np.float32),
            raycast=np.ones(40, np.float32),
            in_fov_mask=np.ones(40, bool),
            stamp=now - self.age_s,
        )

    def reset(self) -> None:
        self.resets += 1


class FakePolicy(PolicyRunner):
    """Constant raw action; optionally advances a SimClock (simulated inference time) or raises."""

    def __init__(
        self,
        extra_dim: int = 30,
        action=(0.0, 0.0, 0.0, 0.0),
        clock: Any = None,
        cost_s: float = 0.0,
        raises: BaseException | None = None,
        log: list[str] | None = None,
    ) -> None:
        self._extra_dim = extra_dim
        self.action = np.array(action, np.float32)
        self.clock = clock
        self.cost_s = cost_s
        self.raises = raises
        self.log = log
        self.loaded = False
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    @property
    def extra_dim(self) -> int:
        return self._extra_dim

    def load(self) -> None:
        self.loaded = True
        if self.log is not None:
            self.log.append("load")

    def act(self, obs: np.ndarray, extra: np.ndarray) -> np.ndarray:
        if self.raises is not None:
            raise self.raises
        if self.clock is not None and self.cost_s:
            self.clock.advance(self.cost_s)
        self.calls.append((obs, extra))
        return self.action.copy()


class FakeSink(ActionSink):
    def __init__(
        self,
        send_raises: BaseException | None = None,
        failsafe_failures: int = 0,
        log: list[str] | None = None,
    ) -> None:
        self.sent: list[ContractCommand] = []
        self.failsafe_calls = 0
        self.failsafe_failures = failsafe_failures
        self.send_raises = send_raises
        self.armed = False
        self.closed = 0
        self.log = log

    def arm(self) -> None:
        self.armed = True
        if self.log is not None:
            self.log.append("arm")

    def send(self, command: ContractCommand) -> dict[str, Any]:
        if self.send_raises is not None:
            raise self.send_raises
        self.sent.append(command)
        return {"ok": True}

    def failsafe(self) -> None:
        self.failsafe_calls += 1
        if self.failsafe_calls <= self.failsafe_failures:
            raise RuntimeError("failsafe failed")

    def close(self) -> None:
        self.closed += 1


class FakeAbort(AbortSignal):
    def __init__(self, set_after_checks: int | None = None) -> None:
        self.set_after_checks = set_after_checks
        self.checks = 0

    def is_set(self) -> bool:
        self.checks += 1
        return self.set_after_checks is not None and self.checks > self.set_after_checks


class ListRecorder:
    """In-memory stand-in for :class:`RunRecorder` (same producer API)."""

    def __init__(self) -> None:
        self.ticks: list[Any] = []
        self.telemetry: list[Any] = []
        self.frames: list[Any] = []

    def record_tick(self, record: Any) -> None:
        self.ticks.append(record)

    def record_telemetry(self, samples: list[Any]) -> None:
        self.telemetry.extend(samples)

    def record_frame(self, tick: int, frame: Frame, *, save_image: bool) -> None:
        self.frames.append((tick, frame, save_image))


class MockPad:
    """Records the vgamepad float-stick API calls."""

    def __init__(self) -> None:
        self.left = (None, None)
        self.right = (None, None)
        self.updates = 0
        self.history: list[dict[str, float]] = []
        self.buttons: set = set()
        self.button_log: list[tuple[str, object, int]] = []  # (press|release, button, updates)

    def press_button(self, button) -> None:
        self.buttons.add(button)
        self.button_log.append(("press", button, self.updates))

    def release_button(self, button) -> None:
        self.buttons.discard(button)
        self.button_log.append(("release", button, self.updates))

    def left_joystick_float(self, x_value_float: float, y_value_float: float) -> None:
        self.left = (x_value_float, y_value_float)

    def right_joystick_float(self, x_value_float: float, y_value_float: float) -> None:
        self.right = (x_value_float, y_value_float)

    def update(self) -> None:
        self.updates += 1
        self.history.append(self.sticks)

    @property
    def sticks(self) -> dict[str, float]:
        return {
            "left_x": self.left[0],
            "left_y": self.left[1],
            "right_x": self.right[0],
            "right_y": self.right[1],
        }
