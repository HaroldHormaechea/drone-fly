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
    (lost -> fail-safe) -> free-space latest (age check) -> [latency forecast] -> assemble ->
    policy -> decode -> sink.send -> [overlay publish] -> record. Per-stage latencies (state,
    perception, inference, actuation, total; plus overlay when a live-overlay publisher is
    attached), the perception
    split per estimator (gate, free space), estimate ages and missed deadlines are recorded on
    every tick.
    With a :class:`~drone_fly.harness.perception.PerceptionRuntime` (UC-86), right after the
    perception updates the first estimates are checked against their providers' contract (a
    violation stops with ``perception_error``) and per-provider latency / age rows are recorded.

Live overlay (UC-65, optional)
    With a publisher, ticks for which ``publisher.wants_brain(k)`` is true call
    ``policy.act_with_brain`` (same ops, also returns the propagated K1 state); every other tick
    — and every tick without a publisher — calls ``policy.act`` unchanged. After actuation the
    tick is published (never raising: a publisher failure is logged and the run continues) and
    the cost lands in ``latency_s["overlay"]``; ``total`` stays state -> actuation, but the
    overlay cost still counts toward the missed-deadline check. The wait loops publish status
    and ``run()`` publishes the stop reason after the fail-safe. The publisher never calls the
    focus probe: the runner hands it its own focus result.

Estimator stop (UC69)
    A gate source may end the attempt on purpose by raising
    :class:`~drone_fly.harness.errors.EstimatorStop` from ``update``: the runner stops with its
    reason before anything is sent on that tick (the fail-safe still follows).

Latency compensation (optional)
    With a :class:`~drone_fly.harness.latency_comp.LatencyCompensator`, perception (gate, free
    space, the spawn-yaw capture) keeps the measured state and only the assembler gets the state
    forecast through the commands still in flight; every successfully sent command is recorded.
    A discontinuity resets it with the estimators.

Fail-safe (latched)
    On stale telemetry, gate lost, a stale free-space estimate, loss of window focus, any
    perception / policy / sink exception, the abort signal or Ctrl+C, the runner calls
    ``sink.failsafe()`` (neutral rates + throttle cut; retried once if it raises), records the
    reason and what the sink was left at (``RunResult.failsafe_output``) and stops. The sink only
    ever receives the decoded contract command — no yaw shaping or rate override happens here.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, NoReturn

import numpy as np

from drone_fly.harness.config import require_number
from drone_fly.harness.errors import EstimatorStop, PerceptionContractError
from drone_fly.harness.interfaces import (
    ActionSink,
    Harness,
    ObservationAssembler,
    PolicyRunner,
)
from drone_fly.harness.policy import decode_action
from drone_fly.harness.types import ContractCommand, FreeSpaceEstimate, GateEstimate, TickRecord
from drone_fly.sim.action import legacy_reference, raw_of_command

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
STOP_NOT_ARMED = "not_armed"  # an arm button is configured but the motors never spun up
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


def motors_spinning(sample: Any) -> bool | None:
    """True when a telemetry sample shows non-zero motor RPM; None when it carries no RPM."""
    rpm = None if sample is None else getattr(sample, "rpm", None)
    if rpm is None:
        return None
    return float(np.max(np.abs(rpm))) > 0.0


def confirm_disarmed(
    sink: Any, source: Any, clock: Any, timeout_s: float, events: list[str]
) -> str:
    """Re-press Disarm every 0.5 s until telemetry shows the motors stopped (or give up).

    Appends ``disarmed`` or ``disarm_unconfirmed`` to ``events`` and returns the outcome:
    ``disarmed``, ``unconfirmed`` (then the sink keeps the throttle at minimum), ``no_rpm`` (the
    stream has no RPM block) or ``skipped`` (no timeout, or nothing to press).
    """
    disarm = getattr(sink, "disarm", None)
    if timeout_s <= 0 or disarm is None:
        return "skipped"
    start = clock.now()
    next_press = start + 0.5
    try:
        while clock.now() - start < timeout_s:
            samples = source.drain_raw()
            spinning = motors_spinning(samples[-1]) if samples else None
            if samples and spinning is None:
                return "no_rpm"  # stream without RPM: cannot confirm
            if spinning is False:
                events.append("disarmed")
                return "disarmed"
            if clock.now() >= next_press:
                if not disarm():
                    return "skipped"
                next_press = clock.now() + 0.5
            clock.sleep_until(clock.now() + 0.05)
    except Exception:  # noqa: BLE001 - never mask the stop reason
        log.exception("confirming the disarm failed")
    events.append("disarm_unconfirmed")
    hold = getattr(sink, "hold_throttle_min", None)
    if callable(hold):
        hold()  # never park the pad at centred sticks while the drone may be armed
    log.error("could not confirm the drone disarmed (motor RPM still > 0 or no telemetry)")
    return "unconfirmed"


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
    #: With an arm button configured: fail safe if telemetry shows no motor RPM this long after
    #: the first command (the Arm press did not arm). 0 = no check.
    arm_confirm_s: float = 0.0
    #: After the fail-safe (arm button configured): retry Disarm until telemetry shows the motors
    #: stopped, for at most this long.
    disarm_confirm_s: float = 2.0
    #: Spawn-yaw frame alignment switch ("auto" | True | False); session.py builds the aligner.
    align_spawn_yaw: str | bool = "auto"
    #: UC-88 command slew-rate override (full-range traversal seconds); None ⇒ use the checkpoint
    #: sidecar's ``action_filter``. session.py resolves it into the :class:`SlewLimiter`.
    slew_override_s: float | None = None

    @classmethod
    def from_config(cls, cfg: Any) -> RunnerSettings:
        r = cfg.runner
        focus = cfg.get("focus") or {}
        controls = cfg.get("controls") or {}
        arm_confirm = float(
            require_number(
                controls.get("arm_confirm_s", 3.0), "controls.arm_confirm_s", min_value=0.0
            )
        )
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
            arm_confirm_s=arm_confirm if controls.get("arm_button") else 0.0,
            disarm_confirm_s=float(
                require_number(
                    controls.get("disarm_confirm_s", 2.0),
                    "controls.disarm_confirm_s",
                    min_value=0.0,
                )
            ),
            align_spawn_yaw=getattr(r, "align_spawn_yaw", "auto"),
            slew_override_s=getattr(r, "slew_override_s", None),
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
        yaw_aligner: Any = None,
        latency_comp: Any = None,
        perception: Any = None,
        slew: Any = None,
    ) -> None:
        self.harness = harness
        #: Optional :class:`~drone_fly.harness.latency_comp.LatencyCompensator`.
        self.latency_comp = latency_comp
        #: Optional UC-88 :class:`~drone_fly.sim.action.SlewLimiter` on the command path (``None``
        #: or disabled ⇒ the raw command is sent and the policy memory sees the raw intent).
        self.slew = slew
        self._latency_logged = False
        #: Optional :class:`~drone_fly.harness.yaw_align.SpawnYawAligner` (contract policies).
        self.yaw_aligner = yaw_aligner
        #: Optional :class:`~drone_fly.harness.perception.PerceptionRuntime` (UC-86): validates
        #: the first estimates and records per-provider latency / age. ``None`` = not used.
        self.perception = perception
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
        self._arm_deadline: float | None = None  # pending arm confirmation (motor RPM)
        #: Called right after the sink's fail-safe, before the disarm confirmation (e.g. saving
        #: the game's flight recording while it still holds this attempt). Never raises out.
        self.on_stop: Any = None
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
        if self.latency_comp is not None:
            self.latency_comp.resolve(getattr(self.policy, "contract_version", None))
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
        # Optional gate-estimator hook (e.g. a background inference worker).
        start_gate = getattr(h.gate_estimator, "start", None)
        if start_gate is not None:
            start_gate()

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
        stop_gate = getattr(h.gate_estimator, "stop", None)
        if stop_gate is not None:
            try:
                stop_gate()
            except Exception:  # noqa: BLE001
                log.exception("error stopping the gate estimator")
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
                if self.on_stop is not None:
                    try:
                        self.on_stop(result)
                    except Exception:  # noqa: BLE001 - never mask the stop
                        log.exception("on_stop hook failed")
                        result.events.append("on_stop_failed")
                self._confirm_disarmed(result)
                return
            except Exception:  # noqa: BLE001
                result.failsafe_ok = False
                log.exception("sink.failsafe() failed (attempt %d)", attempt)

    def _confirm_disarmed(self, result: RunResult) -> None:
        """With an arm button: re-press Disarm every 0.5 s until motor RPM reads 0 (or give up).

        Records ``disarmed`` or ``disarm_unconfirmed`` in the run's events.
        """
        s = self.settings
        if s.arm_confirm_s <= 0:
            return
        confirm_disarmed(
            self.harness.action_sink,
            self.harness.state_source,
            self.clock,
            s.disarm_confirm_s,
            result.events,
        )

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
        requested = False
        while not probe.has_focus():
            if not requested:
                probe.request_focus()  # best effort (focus.auto_focus); the wait still applies
                requested = True
            self._publish_status("waiting_for_focus")
            if self.harness.abort_signal is not None and self.harness.abort_signal.is_set():
                raise _Stop(STOP_ABORT)
            if self.clock.now() >= deadline:
                raise _Stop(
                    STOP_FOCUS_LOST,
                    f"target window not focused within {self.settings.wait_for_focus_s:.1f}s",
                )
            self.clock.sleep_until(self.clock.now() + 0.05)

    def _check_armed(self, record: TickRecord, now: float) -> None:
        """Clear the pending arm check once telemetry shows motor RPM; fail safe at the deadline.

        A stream without RPM cannot confirm arming, so the check is dropped (logged once).
        """
        sample = self._last_sample
        spinning = motors_spinning(sample)
        if sample is not None and spinning is None:
            log.warning("telemetry has no MotorRPM block; cannot confirm arming")
            self._arm_deadline = None
            return
        if spinning:
            self._arm_deadline = None
            record.events.append("armed")
            return
        if self._arm_deadline is not None and now >= self._arm_deadline:
            self._finish(
                record,
                STOP_NOT_ARMED,
                "motors never spun up after the Arm press: check Liftoff's Arm binding "
                "(controls.arm_button) or restart Liftoff",
            )

    def _loop(self, result: RunResult) -> None:
        s = self.settings
        period = 1.0 / s.rate_hz
        if self.yaw_aligner is not None:
            self.yaw_aligner.reset()  # every run captures its own spawn yaw
        if self.latency_comp is not None:
            self.latency_comp.reset()
        mem = getattr(self.assembler, "memory", None)
        if mem is not None:
            mem.reset()  # UC-87 policy memory: zero-seed at run start (D3)
        if self.slew is not None:
            self.slew.reset()  # UC-88 slew carry: zero-seed at run start (same boundary as memory)
        self._wait_for_state()
        self._wait_for_focus()
        start = self.clock.now()
        self._arm_deadline = start + s.arm_confirm_s if s.arm_confirm_s > 0 else None
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
        period = 1.0 / s.rate_hz  # UC-88: the policy-step dt the slew cap is scaled by
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
        if (
            self.recorder is not None
            or self.publisher is not None
            or self._arm_deadline is not None
        ):
            samples = h.state_source.drain_raw()
            if samples:
                self._last_sample = samples[-1]
            if self.recorder is not None:
                self.recorder.record_telemetry(samples)
        if self._arm_deadline is not None:
            self._check_armed(record, t0)
        if state is None or age > s.stale_after_s:
            self._finish(record, STOP_STALE_TELEMETRY, f"telemetry age {age:.3f}s")
        record.state = state.as_state_dict()
        aligner = self.yaw_aligner
        disc = getattr(h.state_source, "discontinuity_count", 0)
        if disc != self._last_disc:
            self._last_disc = disc
            h.gate_estimator.reset()
            if h.free_space_estimator is not None:
                h.free_space_estimator.reset()
            record.events.append("discontinuity:estimators_reset")
            if aligner is not None:
                aligner.reset()  # a respawn: re-capture the spawn yaw from this tick
            if self.latency_comp is not None:
                self.latency_comp.reset()  # the in-flight commands belong to the old attempt
            mem = getattr(self.assembler, "memory", None)
            if mem is not None:
                mem.reset()  # UC-87: a respawn resets the policy memory (D3 boundary reset)
            if self.slew is not None:
                self.slew.reset()  # UC-88: a respawn resets the slew carry (same D3 boundary)
        if aligner is not None:
            psi0 = aligner.capture(state)
            if psi0 is not None:
                deg = math.degrees(psi0)
                record.events.append(f"spawn_yaw:{deg:.1f}")
                log.info("spawn yaw %.1f deg: policy inputs aligned to the training frame", deg)
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
            fs_wall: float | None = None
            if h.free_space_estimator is not None:
                fs = h.free_space_estimator.update(frame, state, gate.target, t1)
                fs_wall = clock.now() - tg
                record.perception_s["free_space"] = fs_wall
        except EstimatorStop as stop:  # an intended end of the attempt (oracle episode monitor)
            self._finish(record, stop.reason, stop.detail)
        except Exception as exc:  # noqa: BLE001
            log.exception("perception failed")
            self._finish(record, STOP_PERCEPTION_ERROR, repr(exc))
        record.gate = gate
        record.free_space = fs
        if self.perception is not None:
            try:
                self.perception.after_update(record, gate, fs, tg - t1, fs_wall, t1)
            except PerceptionContractError as exc:
                self._finish(record, STOP_PERCEPTION_ERROR, str(exc))
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
        comp = self.latency_comp
        policy_state = state
        if comp is not None and comp.enabled:
            policy_state = comp.predict(state, t2)
            if comp.last_steps:
                record.predicted_state = policy_state.as_state_dict()
                if not self._latency_logged:
                    self._latency_logged = True
                    record.events.append(f"latency_comp:{comp.last_steps}steps")
        try:
            obs, extra = self.assembler.assemble(policy_state, gate, fs, self.extra_dim)
            if aligner is not None:
                obs, extra = aligner.apply(obs, extra)  # recorded = what the policy sees
            record.obs, record.extra = obs, extra
            if pub is not None and pub.wants_brain(k):
                raw, brain = self.policy.act_with_brain(obs, extra)
            else:
                raw = self.policy.act(obs, extra)
            action_spec = getattr(self.policy, "action_spec", None)
            command = decode_action(raw, self.contract, action_spec)
            # UC-88: clamp the applied command's slew in command space (after decode, before send).
            # The ACTUATED command is the clamp output verbatim: collective_g / rates are exact (no
            # decode(raw_of_command(...)) round-trip in the sent values); fraction / thrust derive
            # from collective_g exactly as the decoder does. The effective-raw a' is computed ONLY
            # for the policy memory (AC4) + the record. Off => a' == raw, command unchanged.
            mem_action = raw
            if self.slew is not None and self.slew.enabled:
                applied = self.slew.step(
                    np.array([command.collective_g, *command.rates], dtype=np.float32), period
                )
                cg = float(applied[0])
                rates = np.asarray(applied[1:4], dtype=np.float32)
                mem_action = np.asarray(raw_of_command(applied, action_spec), dtype=np.float32)
                frac = float(np.clip(legacy_reference().fraction_for_g(cg), 0.0, 1.0))
                command = ContractCommand(
                    raw=mem_action,
                    collective=frac,
                    thrust=float(np.float32(frac) * self.contract.MAX_THRUST),
                    rates=rates,
                    collective_g=cg,
                    contract_version=command.contract_version,
                )
                record.command_raw = mem_action
            mem = getattr(self.assembler, "memory", None)
            if mem is not None:
                # UC-87/88 (D3): record the APPLIED raw action (== raw when the slew filter is off)
                # and the core obs the policy saw, so the next tick's memory block holds them — the
                # sim does the same in CourseEnv.step.
                mem.observe(obs, mem_action)
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
        if comp is not None:
            comp.record(clock.now(), command)
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
