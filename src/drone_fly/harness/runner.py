"""The fixed-rate perceive -> assemble -> act -> actuate loop, with a latched fail-safe.

Scheduling
    Tick ``k`` is due at ``start + k / rate_hz`` on an injectable clock (:class:`WallClock`:
    ``perf_counter`` + sleep + a final spin; :class:`SimClock`: virtual time). A tick that
    finishes after the next deadline counts as a **missed deadline**; the schedule then skips
    to the next future slot (no catch-up burst).

Tick
    abort? -> state (stale check, discontinuity -> reset estimators) -> gate update (lost ->
    fail-safe) -> free-space latest (age check) -> assemble -> policy -> decode -> sink.send ->
    record. Per-stage latencies (state, perception, inference, actuation, total), estimate ages
    and missed deadlines are recorded on every tick.

Fail-safe (latched)
    On stale telemetry, gate lost, a stale free-space estimate, any perception / policy / sink
    exception, the abort signal or Ctrl+C, the runner calls ``sink.failsafe()`` (neutral rates +
    throttle cut; retried once if it raises), records the reason and stops. The sink only ever
    receives the decoded contract command — no yaw shaping or rate override happens here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, NoReturn

import numpy as np

from drone_fly.harness.interfaces import (
    ActionSink,
    Harness,
    ObservationAssembler,
    PolicyRunner,
)
from drone_fly.harness.policy import decode_action
from drone_fly.harness.types import ContractCommand, FreeSpaceEstimate, GateEstimate, TickRecord

log = logging.getLogger(__name__)

# Stop reason codes. Everything except the NORMAL_STOPS set is a fail-safe stop.
STOP_COMPLETED = "completed"  # max_duration / max_ticks reached
STOP_COURSE_FINISHED = "course_finished"
STOP_NO_TELEMETRY = "no_telemetry"
STOP_STALE_TELEMETRY = "stale_telemetry"
STOP_GATE_LOST = "gate_lost"
STOP_PERCEPTION_STALE = "perception_stale"
STOP_PERCEPTION_ERROR = "perception_error"
STOP_POLICY_ERROR = "policy_error"
STOP_SINK_ERROR = "sink_error"
STOP_ABORT = "user_abort"
STOP_INTERRUPT = "keyboard_interrupt"
NORMAL_STOPS = frozenset({STOP_COMPLETED, STOP_COURSE_FINISHED})


class WallClock:
    """Real time: ``perf_counter`` with sleep, then a short spin for the last millisecond."""

    def __init__(self, spin_s: float = 0.001) -> None:
        self.spin_s = spin_s

    def now(self) -> float:
        return time.perf_counter()

    def sleep_until(self, deadline: float) -> None:
        remaining = deadline - time.perf_counter()
        if remaining > self.spin_s:
            time.sleep(remaining - self.spin_s)
        while time.perf_counter() < deadline:
            pass


class SimClock:
    """Virtual time: sleeping jumps the clock; work takes zero time unless ``advance`` is used."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = float(start)

    def now(self) -> float:
        return self.t

    def sleep_until(self, deadline: float) -> None:
        self.t = max(self.t, deadline)

    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass(frozen=True)
class RunnerSettings:
    rate_hz: float = 50.0
    stale_after_s: float = 0.25
    max_estimate_age_s: float = 0.3
    wait_for_state_s: float = 5.0
    max_duration_s: float | None = None
    max_ticks: int | None = None
    frame_every_n: int = 0  # recording: save every Nth frame image (0 = metadata only)

    @classmethod
    def from_config(cls, cfg: Any) -> RunnerSettings:
        r = cfg.runner
        return cls(
            rate_hz=r.rate_hz,
            stale_after_s=r.stale_after_s,
            max_estimate_age_s=r.max_estimate_age_s,
            wait_for_state_s=r.wait_for_state_s,
            max_duration_s=r.max_duration_s,
            max_ticks=r.max_ticks,
            frame_every_n=cfg.recording.frame_every_n,
        )


@dataclass
class RunResult:
    reason: str
    ticks: int = 0
    missed_deadlines: int = 0
    failsafe: bool = False
    failsafe_ok: bool = True
    error: str | None = None
    events: list[str] = field(default_factory=list)

    @property
    def is_failsafe_stop(self) -> bool:
        return self.reason not in NORMAL_STOPS


class RecordingOnlySink(ActionSink):
    """Dry-run sink: records the decoded command, actuates nothing."""

    def __init__(self, cfg: dict[str, Any] | None = None, ctx: Any = None) -> None:
        self.sent = 0
        self.failsafe_calls = 0

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> RecordingOnlySink:
        return cls(cfg, ctx)

    def send(self, command: ContractCommand) -> dict[str, Any]:
        self.sent += 1
        return {"dry_run": True}

    def failsafe(self) -> None:
        self.failsafe_calls += 1


class _Stop(Exception):
    def __init__(self, reason: str, error: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.error = error


class HarnessRunner:
    """Drives one composed :class:`Harness` with a loaded :class:`PolicyRunner`."""

    def __init__(
        self,
        harness: Harness,
        policy: PolicyRunner,
        assembler: ObservationAssembler,
        contract: Any,
        settings: RunnerSettings | None = None,
        recorder: Any = None,
        clock: Any = None,
    ) -> None:
        self.harness = harness
        self.policy = policy
        self.assembler = assembler
        self.contract = contract
        self.settings = settings or RunnerSettings()
        self.recorder = recorder
        self.clock = clock or harness.clock or WallClock()
        self.extra_dim = policy.extra_dim
        self._failsafe_latched = False
        self._last_disc = 0
        self._fs_required = any(
            b in ("obstacle_vision", "raycast") for b in assembler.required_blocks(self.extra_dim)
        )

    # -- lifecycle --

    def prepare(self, *, arm: bool = True) -> None:
        """Coverage check -> model load -> (optionally) arm the sink, in that order."""
        self.assembler.check_coverage(self.extra_dim, self.harness.provided_blocks())
        self.policy.load()
        if arm:
            self.arm()

    def arm(self) -> None:
        """Acquire the output device (call only once nothing else can refuse the run)."""
        self.harness.action_sink.arm()

    def _start_parts(self) -> None:
        h = self.harness
        for part in (h.state_source, h.frame_source, h.free_space_estimator, h.abort_signal):
            if part is not None:
                part.start()

    def shutdown(self) -> None:
        """Stop every part and close the sink (idempotent; safe on parts never started)."""
        h = self.harness
        for part in (h.abort_signal, h.free_space_estimator, h.frame_source, h.state_source):
            if part is None:
                continue
            try:
                part.stop()
            except Exception:  # noqa: BLE001
                log.exception("error stopping %s", type(part).__name__)
        try:
            h.action_sink.close()
        except Exception:  # noqa: BLE001
            log.exception("error closing the action sink")
        for cb in h.on_close:
            try:
                cb()
            except Exception:  # noqa: BLE001
                log.exception("error in harness close hook")

    def _failsafe(self, result: RunResult) -> None:
        if self._failsafe_latched:
            return
        self._failsafe_latched = True
        result.failsafe = True
        for attempt in (1, 2):
            try:
                self.harness.action_sink.failsafe()
                result.failsafe_ok = True
                return
            except Exception:  # noqa: BLE001
                result.failsafe_ok = False
                log.exception("sink.failsafe() failed (attempt %d)", attempt)

    def run(self, *, prepared: bool = False) -> RunResult:
        """Run until a stop condition; always ends with the sink in fail-safe."""
        if not prepared:
            self.prepare()
        result = RunResult(reason=STOP_COMPLETED)
        self._start_parts()
        try:
            self._loop(result)
        except _Stop as stop:
            result.reason, result.error = stop.reason, stop.error
        except KeyboardInterrupt:
            result.reason = STOP_INTERRUPT
        finally:
            self._failsafe(result)
            if result.is_failsafe_stop:
                log.warning("fail-safe stop: %s %s", result.reason, result.error or "")
            self.shutdown()
        return result

    # -- loop --

    def _wait_for_state(self) -> None:
        src = self.harness.state_source
        deadline = self.clock.now() + self.settings.wait_for_state_s
        while src.latest() is None:
            if self.clock.now() >= deadline:
                raise _Stop(STOP_NO_TELEMETRY, "no telemetry received before wait_for_state_s")
            self.clock.sleep_until(self.clock.now() + 0.01)

    def _loop(self, result: RunResult) -> None:
        s = self.settings
        period = 1.0 / s.rate_hz
        self._wait_for_state()
        start = self.clock.now()
        k = 0
        executed = 0
        while True:
            if s.max_ticks is not None and k >= s.max_ticks:
                raise _Stop(STOP_COMPLETED)
            if s.max_duration_s is not None and k * period >= s.max_duration_s - 1e-9:
                raise _Stop(STOP_COMPLETED)
            self.clock.sleep_until(start + k * period)
            record = self._tick(k, result)
            executed += 1
            result.ticks = executed
            next_deadline = start + (k + 1) * period
            now = self.clock.now()
            if now > next_deadline:
                record.missed_deadline = True
                result.missed_deadlines += 1
                k = int(np.floor((now - start) / period)) + 1  # skip to the next future slot
            else:
                k += 1
            if self.recorder is not None:
                self.recorder.record_tick(record)
            if record.gate is not None and record.gate.status == "finished":
                raise _Stop(STOP_COURSE_FINISHED)

    def _tick(self, k: int, result: RunResult) -> TickRecord:
        h = self.harness
        s = self.settings
        clock = self.clock
        t0 = clock.now()
        record = TickRecord(tick=k, wall_time=time.time(), sim_time=t0)

        if h.abort_signal is not None and h.abort_signal.is_set():
            self._finish(record, STOP_ABORT)

        # -- state --
        state = h.state_source.latest()
        age = h.state_source.age(t0)
        record.ages_s["telemetry"] = float(age)
        if self.recorder is not None:
            self.recorder.record_telemetry(h.state_source.drain_raw())
        if state is None or age > s.stale_after_s:
            self._finish(record, STOP_STALE_TELEMETRY, f"telemetry age {age:.3f}s")
        record.state = state.as_state_dict()
        disc = getattr(h.state_source, "discontinuity_count", 0)
        if disc != self._last_disc:
            self._last_disc = disc
            h.gate_estimator.reset()
            if h.free_space_estimator is not None:
                h.free_space_estimator.reset()
            record.events.append("discontinuity:estimators_reset")
        t1 = clock.now()

        # -- perception --
        frame = h.frame_source.latest() if h.frame_source is not None else None
        if frame is not None:
            record.ages_s["frame"] = float(t1 - frame.stamp)
            if self.recorder is not None and (s.frame_every_n == 0 or k % s.frame_every_n == 0):
                self.recorder.record_frame(k, frame, save_image=s.frame_every_n > 0)
        try:
            gate: GateEstimate = h.gate_estimator.update(frame, state, t1)
            fs: FreeSpaceEstimate | None = None
            if h.free_space_estimator is not None:
                fs = h.free_space_estimator.update(frame, state, gate.target, t1)
        except Exception as exc:  # noqa: BLE001
            log.exception("perception failed")
            self._finish(record, STOP_PERCEPTION_ERROR, repr(exc))
        record.gate = gate
        record.free_space = fs
        record.ages_s["gate"] = float(t1 - gate.stamp) if gate.stamp else 0.0
        if gate.passed:
            record.events.append(f"gate_passed:{gate.gate_index}:{gate.lap}")
        if gate.status == "lost":
            self._finish(record, STOP_GATE_LOST)
        if fs is not None:
            fs_age = float(t1 - fs.stamp)
            record.ages_s["free_space"] = fs_age
            if self._fs_required and fs_age > s.max_estimate_age_s:
                self._finish(record, STOP_PERCEPTION_STALE, f"free-space age {fs_age:.3f}s")
        t2 = clock.now()

        # -- assemble + act --
        try:
            obs, extra = self.assembler.assemble(state, gate, fs, self.extra_dim)
            record.obs, record.extra = obs, extra
            raw = self.policy.act(obs, extra)
            command = decode_action(raw, self.contract)
        except Exception as exc:  # noqa: BLE001
            log.exception("policy failed")
            self._finish(record, STOP_POLICY_ERROR, repr(exc))
        record.command = command
        t3 = clock.now()

        # -- actuate --
        try:
            record.sink_output = h.action_sink.send(command)
        except Exception as exc:  # noqa: BLE001
            log.exception("action sink failed")
            self._finish(record, STOP_SINK_ERROR, repr(exc))
        t4 = clock.now()

        record.latency_s = {
            "state": t1 - t0,
            "perception": t2 - t1,
            "inference": t3 - t2,
            "actuation": t4 - t3,
            "total": t4 - t0,
        }
        return record

    def _finish(self, record: TickRecord, reason: str, error: str | None = None) -> NoReturn:
        """Record the partial tick, then stop with ``reason`` (fail-safe follows in ``run``)."""
        record.events.append(f"stop:{reason}")
        if self.recorder is not None:
            self.recorder.record_tick(record)
        raise _Stop(reason, error)
