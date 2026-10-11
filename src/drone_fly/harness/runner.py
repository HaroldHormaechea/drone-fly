"""The fixed-rate perceive -> assemble -> act -> actuate loop, with a latched fail-safe.

Scheduling
    Tick ``k`` is due at ``start + k / rate_hz`` on an injectable clock (:class:`WallClock`:
    ``perf_counter`` + sleep + a final spin; :class:`SimClock`: virtual time). A tick that
    finishes after the next deadline counts as a **missed deadline**; the schedule then skips
    to the next future slot (no catch-up burst).

Start
    wait for the first telemetry sample, then — with a focus probe — wait (pad neutral) until the
    target window is focused (``focus.wait_for_focus_s``).

Tick
    abort? -> focus? -> state (stale check, discontinuity -> reset estimators) -> gate update
    (lost -> fail-safe) -> free-space latest (age check) -> assemble -> policy -> decode ->
    sink.send -> [overlay publish] -> record. Per-stage latencies (state, perception, inference,
    actuation, total; plus overlay when a live-overlay publisher is attached), the perception
    split per estimator (gate, free space), estimate ages and missed deadlines are recorded on
    every tick.

Live overlay (UC-65, optional)
    With a publisher, ticks for which ``publisher.wants_brain(k)`` is true call
    ``policy.act_with_brain`` (same ops, also returns the propagated K1 state); every other tick
    â€” and every tick without a publisher â€” calls ``policy.act`` unchanged. After actuation the
    tick is published (never raising: a publisher failure is logged and the run continues) and
    the cost lands in ``latency_s["overlay"]``; ``total`` stays state -> actuation, but the
    overlay cost still counts toward the missed-deadline check. The wait loops publish status
    and ``run()`` publishes the stop reason after the fail-safe. The publisher never calls the
    focus probe: the runner hands it its own focus result.

Fail-safe (latched)
    On stale telemetry, gate lost, a stale free-space estimate, loss of window focus, any
    perception / policy / sink exception, the abort signal or Ctrl+C, the runner calls
    ``sink.failsafe()`` (neutral rates + throttle cut; retried once if it raises), records the
    reason and what the sink was left at (``RunResult.failsafe_output``) and stops. The sink only
    ever receives the decoded contract command — no yaw shaping or rate override happens here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, NoReturn

import numpy as np

from drone_fly.harness.config import require_number
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
STOP_FOCUS_LOST = "focus_lost"
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
    wait_for_focus_s: float = 15.0
    max_duration_s: float | None = None
    max_ticks: int | None = None
    frame_every_n: int = 0  # recording: save every Nth frame image (0 = metadata only)

    @classmethod
    def from_config(cls, cfg: Any) -> RunnerSettings:
        r = cfg.runner
        focus = cfg.get("focus") or {}
        return cls(
            rate_hz=r.rate_hz,
            stale_after_s=r.stale_after_s,
            max_estimate_age_s=r.max_estimate_age_s,
            wait_for_state_s=r.wait_for_state_s,
            wait_for_focus_s=float(
                require_number(
                    focus.get("wait_for_focus_s", 15.0), "focus.wait_for_focus_s", min_value=0.0
                )
            ),
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
    #: What the sink was left at by the fail-safe (e.g. the gamepad's neutral sticks).
    failsafe_output: dict[str, Any] | None = None

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
        publisher: Any = None,
    ) -> None:
        self.harness = harness
        self.policy = policy
        self.assembler = assembler
        self.contract = contract
        self.settings = settings or RunnerSettings()
        self.recorder = recorder
        self.publisher = publisher
        self.clock = clock or harness.clock or WallClock()
        self.extra_dim = policy.extra_dim
        self._failsafe_latched = False
        self._last_disc = 0
        self._last_sample: Any = None  # newest raw telemetry sample (overlay extras)
        self._focus: bool | None = None  # this tick's focus-probe result (None = no probe)
        self._last_overlay_s: float | None = None
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
        for part in (
            h.state_source,
            h.frame_source,
            h.free_space_estimator,
            h.abort_signal,
            h.focus_probe,
        ):
            if part is not None:
                part.start()

    def shutdown(self) -> None:
        """Stop every part and close the sink (idempotent; safe on parts never started)."""
        h = self.harness
        for part in (
            h.focus_probe,
            h.abort_signal,
            h.free_space_estimator,
            h.frame_source,
            h.state_source,
        ):
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
                sink = self.harness.action_sink
                sink.failsafe()
                result.failsafe_ok = True
                neutral = getattr(sink, "neutral_sticks", None)
                result.failsafe_output = (
                    {"sticks": neutral()} if callable(neutral) else {"failsafe": True}
                )
                return
            except Exception:  # noqa: BLE001
                result.failsafe_ok = False
                log.exception("sink.failsafe() failed (attempt %d)", attempt)

    def run(self, *, prepared: bool = False) -> RunResult:
        """Run until a stop condition; always ends with the sink in fail-safe."""
        if not prepared:
            self.prepare()
        result = RunResult(reason=STOP_COMPLETED)
        try:
            # Inside the try: a part that refuses to start still leaves the sink in fail-safe.
            self._start_parts()
            self._loop(result)
        except _Stop as stop:
            result.reason, result.error = stop.reason, stop.error
        except KeyboardInterrupt:
            result.reason = STOP_INTERRUPT
        finally:
            self._failsafe(result)
            if self.publisher is not None:
                try:
                    self.publisher.publish_stop(result)
                except Exception:  # noqa: BLE001 - the overlay must never affect the run
                    log.exception("overlay publish_stop failed")
            if result.is_failsafe_stop:
                log.warning("fail-safe stop: %s %s", result.reason, result.error or "")
            self.shutdown()
        return result

    # -- loop --

    def _publish_status(self, run_state: str) -> None:
        if self.publisher is None:
            return
        try:
            self.publisher.publish_status(run_state)
        except Exception:  # noqa: BLE001 - the overlay must never affect the run
            log.exception("overlay publish_status failed")

    def _wait_for_state(self) -> None:
        src = self.harness.state_source
        deadline = self.clock.now() + self.settings.wait_for_state_s
        while src.latest() is None:
            self._publish_status("waiting_for_state")
            if self.clock.now() >= deadline:
                raise _Stop(STOP_NO_TELEMETRY, "no telemetry received before wait_for_state_s")
            self.clock.sleep_until(self.clock.now() + 0.01)

    def _wait_for_focus(self) -> None:
        """With a focus probe, wait (sink armed at neutral) until the target window is focused."""
        probe = self.harness.focus_probe
        if probe is None:
            return
        deadline = self.clock.now() + self.settings.wait_for_focus_s
        while not probe.has_focus():
            self._publish_status("waiting_for_focus")
            if self.harness.abort_signal is not None and self.harness.abort_signal.is_set():
                raise _Stop(STOP_ABORT)
            if self.clock.now() >= deadline:
                raise _Stop(
                    STOP_FOCUS_LOST,
                    f"target window not focused within {self.settings.wait_for_focus_s:.1f}s",
                )
            self.clock.sleep_until(self.clock.now() + 0.05)

    def _loop(self, result: RunResult) -> None:
        s = self.settings
        period = 1.0 / s.rate_hz
        self._wait_for_state()
        self._wait_for_focus()
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
        if h.focus_probe is not None:
            self._focus = h.focus_probe.has_focus()
            if not self._focus:
                self._finish(record, STOP_FOCUS_LOST, "target window lost focus")

        # -- state --
        state = h.state_source.latest()
        age = h.state_source.age(t0)
        record.ages_s["telemetry"] = float(age)
        if self.recorder is not None or self.publisher is not None:
            samples = h.state_source.drain_raw()
            if samples:
                self._last_sample = samples[-1]
            if self.recorder is not None:
                self.recorder.record_telemetry(samples)
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
            tg = clock.now()
            record.perception_s["gate"] = tg - t1
            fs: FreeSpaceEstimate | None = None
            if h.free_space_estimator is not None:
                fs = h.free_space_estimator.update(frame, state, gate.target, t1)
                record.perception_s["free_space"] = clock.now() - tg
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
        pub = self.publisher
        brain: Any = None
        try:
            obs, extra = self.assembler.assemble(state, gate, fs, self.extra_dim)
            record.obs, record.extra = obs, extra
            if pub is not None and pub.wants_brain(k):
                raw, brain = self.policy.act_with_brain(obs, extra)
            else:
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

        # -- live overlay (optional; never stops the run) --
        if pub is not None:
            try:
                fg = getattr(h.focus_probe, "last_foreground", None)  # (hwnd, process, title)
                pub.publish_tick(
                    record,
                    brain=brain,
                    sample=self._last_sample,
                    missed_deadlines=result.missed_deadlines,
                    focus=self._focus,
                    focus_window=fg[1] if isinstance(fg, tuple) and len(fg) > 1 else None,
                    overlay_prev_s=self._last_overlay_s,
                )
            except Exception:  # noqa: BLE001
                log.exception("overlay publish failed")
            self._last_overlay_s = clock.now() - t4
            record.latency_s["overlay"] = self._last_overlay_s
        return record

    def _finish(self, record: TickRecord, reason: str, error: str | None = None) -> NoReturn:
        """Record the partial tick, then stop with ``reason`` (fail-safe follows in ``run``)."""
        record.events.append(f"stop:{reason}")
        if self.recorder is not None:
            self.recorder.record_tick(record)
        raise _Stop(reason, error)
