"""``drone-fly sysid liftoff``: fly the seeded experiment schedule and record it.

Sequence (every part injected, like :class:`~drone_fly.harness.liftoff.pad_check.PadCheck`)::

    preflight (telemetry >= min_telemetry_hz, MotorRPM present) -> focus wait
    -> [resume: reset first] -> arm (RPM-confirmed) -> ground ramp -> h0
    -> AIRFRAME GUARD (h0 inside the expected hover band, else land + disarm, drone_mismatch)
    -> takeoff -> per pending experiment: settle -> open-loop window -> recover
    -> at each pass boundary: land -> disarm -> stop the UDP source -> anti-cheat check
       -> ui reset (full battery) -> restart the source -> re-arm -> ramp -> takeoff
    -> land -> RPM-confirmed disarm

Violations never raise out of :meth:`SysidSession.run` (see :mod:`~drone_fly.sysid.safety`):
tilt / altitude / geofence abort the experiment and re-stabilize (land + disarm if that fails);
stale telemetry, focus loss, F12 and Ctrl+C fail safe and end the session resumable; a crash
fails safe, ends cleanly and records ``next_experiment`` for ``--resume``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.runner import WallClock, confirm_disarmed, motors_spinning
from drone_fly.sysid.controller import (
    AltitudeLoop,
    ControllerGains,
    GroundRamp,
    Target,
    hold_command,
    tilt_deg,
    yaw_of,
)
from drone_fly.sysid.recorder import RecordingSource, SessionRecorder
from drone_fly.sysid.safety import (
    ABORT_EXPERIMENT,
    CRASH,
    SafetyLimits,
    Violation,
    check_link,
    check_samples,
    check_state,
    worst,
)
from drone_fly.sysid.schedule import Experiment

log = logging.getLogger(__name__)

AXES = ("roll", "pitch", "yaw")
OMEGA_INDEX = {"roll": 0, "pitch": 1, "yaw": 2}

STOP_COMPLETED = "completed"
STOP_INTERRUPT = "keyboard_interrupt"
STOP_NOT_ARMED = "not_armed"
STOP_NO_LIFTOFF = "no_liftoff"
STOP_MISMATCH = "drone_mismatch"
STOP_RESTABILIZE = "restabilize_failed"
STOP_TAKEOFF = "takeoff_timeout"
STOP_TRANSIT = "transit_failed"
STOP_ANTICHEAT = "anticheat"
STOP_ERROR = "error"


class SysidPreflightError(HarnessConfigError):
    """Telemetry too slow, or without the MotorRPM block the arm / disarm checks need."""


class DroneMismatchError(HarnessConfigError):
    """The ramp hover throttle says the loaded airframe is not the expected one."""

    def __init__(self, h0: float, band: tuple[float, float], source: str) -> None:
        self.h0, self.band, self.source = h0, band, source
        super().__init__(
            f"hover throttle {h0:.3f} outside the expected band [{band[0]:.3f}, {band[1]:.3f}] "
            f"({source}): the loaded drone is probably not the requested one"
        )


@dataclass(frozen=True)
class SessionSettings:
    rate_hz: float = 100.0
    hold_alt_m: float = 5.0
    limits: SafetyLimits = SafetyLimits()
    gains: ControllerGains = ControllerGains()
    wait_for_state_s: float = 10.0
    wait_for_focus_s: float = 15.0
    preflight_s: float = 1.0
    min_telemetry_hz: float = 80.0
    arm_confirm_s: float = 3.0
    disarm_confirm_s: float = 3.0
    takeoff_timeout_s: float = 12.0
    climb_m_s: float = 1.2
    settle_hold_s: float = 0.5
    settle_timeout_s: float = 12.0
    #: Settled = also within this horizontal distance of the hold point (after a coast).
    settle_radius_m: float = 2.0
    restabilize_s: float = 6.0
    recover_s: float = 1.2
    land_m_s: float = 0.7
    land_timeout_s: float = 20.0
    ramp_rate_per_s: float = 0.08
    #: Up-steps go to ``hover + a * step_headroom * (1 - hover)``.
    step_headroom: float = 0.6
    #: Throttle steps / bursts are cut short so the vertical speed change stays below this (a
    #: 5-inch at T/W 7 gains 10 m/s in a 0.25 s full-throttle burst). The excess acceleration is
    #: estimated as ``(level / hover - 1) g`` (thrust ~ proportional to throttle).
    max_dv_m_s: float = 4.0
    #: After take-off, fly this far forward (spawn heading) before the experiments; the
    #: geofence is centred there (a spawn at the edge of a map, e.g. Liftoff Arena).
    home_forward_m: float = 0.0
    transit_m_s: float = 3.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SysidResult:
    stop_reason: str = STOP_COMPLETED
    error: str | None = None
    session_dir: str | None = None
    next_experiment: str | None = None
    h0: float | None = None
    done: int = 0
    aborted: int = 0
    violations: list[dict[str, Any]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.stop_reason == STOP_COMPLETED

    @property
    def resumable(self) -> bool:
        return not self.completed and self.next_experiment is not None


class _Stop(Exception):
    def __init__(self, reason: str, detail: str = "", violation: Violation | None = None) -> None:
        super().__init__(reason)
        self.reason, self.detail, self.violation = reason, detail, violation


class _Abort(Exception):
    def __init__(self, violation: Violation) -> None:
        super().__init__(violation.name)
        self.violation = violation


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class SysidSession:
    """One sysid session over injected parts (see the module docstring).

    ``source`` is a state source in Liftoff metres (world scale 1); ``sink`` a
    :class:`~drone_fly.harness.liftoff.gamepad.VgamepadActionSink` (``send_sticks``, ``failsafe``,
    ``disarm``, ``close``, ``rearm``); ``rates`` / ``signs`` convert the controller's body rates
    to sticks; ``ui_reset`` resets the level (full battery) between passes; ``anticheat`` is the
    ``require_no_anticheat`` callable (``None`` skips the check).
    """

    def __init__(
        self,
        source: Any,
        sink: Any,
        settings: SessionSettings,
        *,
        drone_id: str,
        schedule: list[Experiment],
        recorder: SessionRecorder,
        rates: dict[str, Any],
        signs: dict[str, float],
        abort: Any = None,
        focus: Any = None,
        clock: Any = None,
        ui_reset: Any = None,
        anticheat: Any = None,
        expected_hover: tuple[float, float] | None = None,
        hover_band_source: str | None = None,
        resume: bool = False,
        meta_extra: dict[str, Any] | None = None,
    ) -> None:
        self.raw_source = source
        self.rec = recorder
        self.src = RecordingSource(source, recorder)
        self.sink = sink
        self.s = settings
        self.drone_id = drone_id
        self.schedule = schedule
        self.rates = rates
        self.signs = {a: float(signs.get(a, 1.0)) for a in AXES}
        self.abort = abort
        self.focus = focus
        self.clock = clock or WallClock()
        self.ui_reset = ui_reset
        self.anticheat = anticheat
        self.expected_hover = expected_hover
        self.hover_band_source = hover_band_source
        self.resume = resume
        self.meta_extra = meta_extra or {}
        self.gains = settings.gains
        self.period = 1.0 / settings.rate_hz
        self.home = np.zeros(3)
        self.ground_z = 0.0
        self.yaw_ref = 0.0
        self.loop = AltitudeLoop(0.4)
        self.airborne = False
        self.motors_may_spin = False
        self._prev_sample: Any = None
        self.meta: dict[str, Any] = {}
        self.result = SysidResult(session_dir=str(recorder.dir))

    # ------------------------------------------------------------------ meta

    def _init_meta(self) -> None:
        old = self.rec.meta if self.resume else {}
        statuses = (old.get("experiments") or {}) if old else {}
        segments = list(old.get("segments") or [])
        self.rec.segment = len(segments)
        segments.append({"index": self.rec.segment, "started_utc": _utc(), "resume": self.resume})
        self.meta = {
            **old,
            **self.meta_extra,
            "schema_version": 1,
            "drone_id": self.drone_id,
            "created_utc": old.get("created_utc", _utc()),
            "settings": self.s.as_dict(),
            "schedule": [e.as_dict() for e in self.schedule],
            "experiments": {e.id: statuses.get(e.id, {"status": "pending"}) for e in self.schedule},
            "events": list(old.get("events") or []),
            "violations": list(old.get("violations") or []),
            "segments": segments,
            "airframe_check": list(old.get("airframe_check") or []),
            "h0": list(old.get("h0") or []),
            "battery": dict(old.get("battery") or {}),
            "stop_reason": None,
            "next_experiment": old.get("next_experiment"),
        }
        self.rec.write_meta(self.meta)

    def _event(self, text: str) -> None:
        t = self.clock.now()
        self.meta["events"].append({"t": t, "segment": self.rec.segment, "event": text})
        self.result.events.append(text)
        log.info("sysid: %s", text)

    def _violation(self, v: Violation, experiment: str | None) -> None:
        entry = {**v.as_dict(), "t": self.clock.now(), "experiment": experiment}
        self.meta["violations"].append(entry)
        self.result.violations.append(entry)
        log.warning("sysid violation %s (%s): %s", v.name, v.action, v.detail)

    def _status(self, exp: Experiment, status: str, **kw: Any) -> None:
        entry = self.meta["experiments"].setdefault(exp.id, {})
        entry.update({"status": status, **kw})
        self.rec.write_meta(self.meta)

    def _battery(self, key: str) -> None:
        sample = self.src.last_sample
        bat = getattr(sample, "battery", None)
        if bat is not None:
            self.meta["battery"].setdefault(key, []).append(
                {"segment": self.rec.segment, "charge": float(bat[0]), "volts": float(bat[1])}
            )

    # ------------------------------------------------------------------ guards

    def _guard(self, envelope: str) -> Any:
        """Link + crash checks always; ``envelope`` = ``soft`` | ``hard`` | ``none``."""
        now = self.clock.now()
        focus = None if self.focus is None else bool(self.focus.has_focus())
        abort = self.abort is not None and self.abort.is_set()
        state = self.src.latest()
        age = self.src.age(now)
        vs = check_link(age=age, focus=focus, abort=abort, limits=self.s.limits)
        self.src.drain_raw()
        samples = self.src.take_pending()
        if samples:
            if self.airborne:
                vs += check_samples(self._prev_sample, samples, self.s.limits)
            self._prev_sample = samples[-1]
        if state is not None and self.airborne and envelope != "none":
            vs += check_state(
                state,
                home=self.home,
                ground_z=self.ground_z,
                limits=self.s.limits,
                soft=envelope == "soft",
            )
        v = worst(vs)
        if v is not None:
            if v.action == ABORT_EXPERIMENT:
                raise _Abort(v)
            raise _Stop(v.name if v.action != CRASH else "crash", v.detail, v)
        if state is None:
            raise _Stop("stale_telemetry", "no telemetry")
        return state

    def _loop(
        self,
        phase: str,
        fn: Any,
        *,
        duration: float | None = None,
        until: Any = None,
        envelope: str = "soft",
        experiment: str = "",
    ) -> Any:
        """Run ``fn(state, t, dt) -> sticks`` at ``rate_hz`` until ``until(state, t)`` or timeout.

        Returns the last state (``until`` met or ``duration`` elapsed).
        """
        t0 = self.clock.now()
        k = 0
        while True:
            self.clock.sleep_until(t0 + k * self.period)
            state = self._guard(envelope)
            t = self.clock.now() - t0
            if until is not None and until(state, t):
                return state
            if duration is not None and t >= duration - 1e-9:
                return state
            sticks = fn(state, t, self.period)
            self.sink.send_sticks(sticks)
            self.rec.record_command(self.clock.now(), experiment, phase, sticks)
            now = self.clock.now()
            if now > t0 + (k + 1) * self.period:
                k = int(math.floor((now - t0) / self.period)) + 1
            else:
                k += 1

    # ------------------------------------------------------------------ controller glue

    def _target(self, **kw: Any) -> Target:
        return Target(pos=self.home, z=self.ground_z + self.s.hold_alt_m, yaw=self.yaw_ref, **kw)

    def _hold(self, target: Target, throttle: float | None = None) -> Any:
        def fn(state: Any, t: float, dt: float) -> dict[str, float]:
            return hold_command(
                state,
                target,
                self.loop,
                self.gains,
                self.rates,
                self.signs,
                dt,
                throttle=throttle,
            )

        return fn

    def _settled(self, hold_for: float) -> Any:
        since: list[float | None] = [None]
        z_ref = self.ground_z + self.s.hold_alt_m

        def until(state: Any, t: float) -> bool:
            off = np.asarray(state.pos, np.float64)[:2] - np.asarray(self.home, np.float64)[:2]
            ok = (
                float(np.linalg.norm(off)) < self.s.settle_radius_m
                and abs(float(state.pos[2]) - z_ref) < 0.5
                and float(np.linalg.norm(state.vel)) < 0.7
                and tilt_deg(state.quat) < 10.0
                and float(np.linalg.norm(state.omega)) < 0.6
            )
            if not ok:
                since[0] = None
                return False
            if since[0] is None:
                since[0] = t
            return t - since[0] >= hold_for

        return until

    # ------------------------------------------------------------------ phases

    def _preflight(self) -> None:
        deadline = self.clock.now() + self.s.wait_for_state_s
        while self.src.latest() is None:
            if self.abort is not None and self.abort.is_set():
                raise _Stop("user_abort", "abort hotkey")
            if self.clock.now() >= deadline:
                raise SysidPreflightError(
                    "no telemetry: load a level with the drone (and enable Liftoff telemetry)"
                )
            self.clock.sleep_until(self.clock.now() + 0.01)
        self.src.drain_raw()
        self.src.take_pending()
        t0 = self.clock.now()
        samples: list[Any] = []
        while self.clock.now() - t0 < self.s.preflight_s:
            self.clock.sleep_until(self.clock.now() + 0.05)
            self.src.drain_raw()
            samples += self.src.take_pending()
        hz = len(samples) / self.s.preflight_s
        rpm_ok = bool(samples) and motors_spinning(samples[-1]) is not None
        self.meta["preflight"] = {"telemetry_hz": hz, "rpm_block": rpm_ok}
        if hz < self.s.min_telemetry_hz:
            raise SysidPreflightError(
                f"telemetry at {hz:.0f} Hz < {self.s.min_telemetry_hz:.0f} Hz: sysid needs ~100 Hz"
            )
        if not rpm_ok:
            raise SysidPreflightError(
                "telemetry has no MotorRPM block (StreamFormat): arming cannot be confirmed"
            )
        if motors_spinning(samples[-1]):
            raise SysidPreflightError("the motors are already spinning: disarm the drone first")
        state = self.src.latest()
        self._home_from(state)

    def _home_from(self, state: Any) -> None:
        self.home = np.asarray(state.pos, np.float64).copy()
        self.ground_z = float(state.pos[2])
        self.yaw_ref = yaw_of(state.quat)
        self._prev_sample = None

    def _wait_focus(self) -> None:
        if self.focus is None:
            return
        deadline = self.clock.now() + self.s.wait_for_focus_s
        request = getattr(self.focus, "request_focus", None)
        while not self.focus.has_focus():
            if request is not None:
                request()
                request = None
            if self.abort is not None and self.abort.is_set():
                raise _Stop("user_abort", "abort hotkey")
            if self.clock.now() >= deadline:
                raise _Stop("focus_lost", "game window not focused within wait_for_focus_s")
            self.clock.sleep_until(self.clock.now() + 0.05)

    def _arm(self) -> None:
        neutral = {"throttle": -1.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self.motors_may_spin = True

        def spinning(state: Any, t: float) -> bool:
            return bool(motors_spinning(self.src.last_sample))

        state = self._loop(
            "arm", lambda s, t, dt: neutral, until=spinning, duration=self.s.arm_confirm_s
        )
        if not motors_spinning(self.src.last_sample):
            raise _Stop(
                STOP_NOT_ARMED,
                "motors never spun up after the Arm press: check the Arm binding or restart "
                "Liftoff",
            )
        self._event("armed")
        settle = float(getattr(self.sink, "arm_settle_s", 0.0)) + 0.05
        state = self._loop("arm", lambda s, t, dt: neutral, duration=settle)
        self._home_from(state)

    def _ramp(self) -> float:
        ramp = GroundRamp(self.ground_z, rate_per_s=self.s.ramp_rate_per_s)
        lifted: list[bool] = [False]

        def fn(state: Any, t: float, dt: float) -> dict[str, float]:
            frac = ramp.step(state, t)
            if frac is None:
                lifted[0] = True
                frac = ramp.throttle
            return {"throttle": 2.0 * frac - 1.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

        self._loop(
            "ramp",
            fn,
            until=lambda s, t: lifted[0] or ramp.exhausted,
            envelope="none",
        )
        if ramp.h0 is None:
            raise _Stop(STOP_NO_LIFTOFF, f"no lift-off up to throttle {ramp.max_throttle:.2f}")
        self.result.h0 = ramp.h0
        self.meta["h0"].append({"segment": self.rec.segment, "h0": ramp.h0, "t": self.clock.now()})
        self._event(f"lift-off: h0 = {ramp.h0:.3f}")
        return ramp.h0

    def _check_airframe(self, h0: float) -> None:
        band = self.expected_hover
        if band is None:
            entry = {"status": "unverified", "h0": h0, "band": None, "source": None}
            self.meta["airframe_check"].append(entry)
            log.warning("no expected hover band: the airframe is not verified (h0 %.3f)", h0)
            return
        ok = band[0] <= h0 <= band[1]
        entry = {
            "status": "ok" if ok else "mismatch",
            "h0": h0,
            "band": list(band),
            "source": self.hover_band_source,
        }
        self.meta["airframe_check"].append(entry)
        if not ok:
            err = DroneMismatchError(h0, band, self.hover_band_source or "expected band")
            self._land()
            self._disarm()
            raise _Stop(STOP_MISMATCH, str(err))

    def _takeoff(self) -> None:
        self.airborne = True
        z0 = self.ground_z
        target = self._target()
        climb = self.s.climb_m_s

        def fn(state: Any, t: float, dt: float) -> dict[str, float]:
            target.z = min(z0 + 0.5 + climb * t, z0 + self.s.hold_alt_m)
            return self._hold(target)(state, t, dt)

        self._loop(
            "takeoff",
            fn,
            until=lambda s, t: float(s.pos[2]) - z0 > self.s.hold_alt_m - 0.4,
            duration=self.s.takeoff_timeout_s,
            envelope="none",  # climbing through the floors; crash checks still apply
        )
        state = self.src.latest()
        if float(state.pos[2]) - z0 <= self.s.hold_alt_m - 0.4:
            raise _Stop(STOP_TAKEOFF, "did not reach the hold altitude")
        self._event("airborne")

    def _fly_up(self) -> None:
        self._arm()
        h0 = self._ramp()
        self.loop = AltitudeLoop(h0)
        self._check_airframe(h0)
        self._takeoff()
        self._transit()

    def _transit(self) -> None:
        """Move the hold point ``home_forward_m`` ahead along the spawn heading."""
        dist = self.s.home_forward_m
        if dist <= 0:
            return
        start = self.home.copy()
        heading = np.array([math.cos(self.yaw_ref), math.sin(self.yaw_ref), 0.0])
        goal = start + dist * heading
        for _attempt in (1, 2):
            pos = np.asarray(self.src.latest().pos, np.float64)
            start = np.array([pos[0], pos[1], start[2]])
            left = float(np.linalg.norm((goal - start)[:2]))
            target = self._target()
            duration = left / self.s.transit_m_s

            def fn(
                state: Any, t: float, dt: float, start=start, target=target, duration=duration
            ) -> dict[str, float]:
                frac = min(t / duration, 1.0) if duration > 0 else 1.0
                target.pos = start + frac * (goal - start)
                self.home = target.pos  # the geofence moves with the hold point
                return self._hold(target)(state, t, dt)

            def arrived(state: Any, t: float) -> bool:
                off = np.asarray(state.pos, np.float64)[:2] - goal[:2]
                return float(np.linalg.norm(off)) < 1.0 and float(np.linalg.norm(state.vel)) < 0.7

            try:
                self._loop(
                    "transit",
                    fn,
                    until=arrived,
                    duration=duration + self.s.settle_timeout_s,
                    envelope="hard",
                )
            except _Abort as a:
                self._violation(a.violation, "transit")
                self._restabilize("transit")  # lands, disarms and stops if that fails
                continue
            self.home = goal
            if arrived(self.src.latest(), 0.0):
                self._event(f"transit {dist:.0f} m to the experiment hold point")
                return
            break
        self._event("transit failed: landing")
        self._land()
        self._disarm()
        raise _Stop(STOP_TRANSIT, "did not reach the experiment hold point")

    def _land(self) -> None:
        state = self.src.latest()
        z_start = float(state.pos[2]) if state is not None else self.ground_z
        target = self._target()
        touch: list[float | None] = [None]

        def fn(state: Any, t: float, dt: float) -> dict[str, float]:
            if touch[0] is not None:
                return {"throttle": -1.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
            target.z = max(z_start - self.s.land_m_s * t, self.ground_z - 0.5)
            target.pos = np.asarray(state.pos, np.float64)  # land where we are
            return self._hold(target)(state, t, dt)

        def until(state: Any, t: float) -> bool:
            if touch[0] is None:
                if float(state.pos[2]) - self.ground_z < 0.15 and abs(float(state.vel[2])) < 0.3:
                    touch[0] = t
                return False
            return t - touch[0] > 0.3

        self._loop("land", fn, until=until, duration=self.s.land_timeout_s, envelope="none")
        self.airborne = False

    def _disarm(self) -> None:
        try:
            self.sink.failsafe()
        finally:
            events: list[str] = []
            out = confirm_disarmed(self.sink, self.src, self.clock, self.s.disarm_confirm_s, events)
            for e in events:
                self._event(e)
            if out == "disarmed":
                self.motors_may_spin = False

    def _restabilize(self, experiment: str) -> None:
        try:
            self._loop(
                "restabilize",
                self._hold(self._target()),
                until=self._settled(self.s.settle_hold_s),
                duration=self.s.restabilize_s,
                envelope="none",  # may start below the floors; the timeout bounds it
                experiment=experiment,
            )
            if self._settled(0.0)(self.src.latest(), 0.0):
                return
        except _Abort as a:
            self._violation(a.violation, experiment)
        self._event("re-stabilize failed: landing")
        self._land()
        self._disarm()
        raise _Stop(STOP_RESTABILIZE, "could not re-stabilize after a violation")

    def _block_boundary(self) -> None:
        """Land, disarm, reset the level (full battery), fly up again."""
        self._land()
        self._disarm()
        self._battery("end")
        self._reset_level()
        self._battery("start")
        self._fly_up()

    def _reset_level(self) -> None:
        self.sink.close()  # parks the shared pad menu-safe (centred) after a confirmed disarm
        self.raw_source.stop()  # ui reset binds the telemetry port itself
        try:
            if self.anticheat is not None:
                try:
                    self.anticheat()
                except HarnessConfigError as exc:
                    raise _Stop(STOP_ANTICHEAT, str(exc)) from exc
            if self.ui_reset is not None:
                self.ui_reset()
                self._event("level reset")
        finally:
            self.raw_source.start()
            self.sink.rearm()
            self.sink.arm()
        self._prev_sample = None
        deadline = self.clock.now() + self.s.wait_for_state_s
        while self.src.latest() is None:
            if self.clock.now() >= deadline:
                raise _Stop("stale_telemetry", "no telemetry after the level reset")
            self.clock.sleep_until(self.clock.now() + 0.01)
        self.clock.sleep_until(self.clock.now() + 0.2)
        self._home_from(self.src.latest())

    # ------------------------------------------------------------------ experiments

    def _window_fn(self, exp: Experiment, h: float) -> tuple[Any, float]:
        """The open-loop window of ``exp`` as ``(fn, duration)``; ``h``: frozen hover throttle."""
        level = self._target(level_only=True)
        hold = self._hold(level)
        hs = self.s.step_headroom

        def with_throttle(frac_of_t: Any) -> Any:
            def fn(state: Any, t: float, dt: float) -> dict[str, float]:
                sticks = hold(state, t, dt)
                sticks["throttle"] = float(np.clip(2.0 * frac_of_t(t) - 1.0, -1.0, 1.0))
                return sticks

            return fn

        def capped(lvl: float) -> float:
            excess_g = abs(lvl / h - 1.0)
            return min(exp.hold_s, self.s.max_dv_m_s / (9.81 * max(excess_g, 1e-6)))

        if exp.kind in ("throttle_step", "throttle_burst"):
            if exp.kind == "throttle_burst":
                lvl = h + exp.amplitude * (1 - h)
            elif exp.sign > 0:
                lvl = h + exp.amplitude * hs * (1 - h)
            else:
                lvl = h * (1 - 0.8 * exp.amplitude)
            hold_s = capped(lvl)
            return with_throttle(lambda t: lvl if t < hold_s else h), hold_s + exp.tail_s
        if exp.kind == "throttle_chirp":
            f0 = float(exp.params.get("f0_hz", 0.5))
            f1 = float(exp.params.get("f1_hz", 5.0))
            T = exp.hold_s
            k = (f1 / f0) ** (1.0 / T)

            # The sweep rides on the altitude loop (a 3 s open-loop throttle would drift away);
            # the fit uses the commanded throttle, so the closed loop does not bias it.
            alt_hold = self._hold(self._target(level_only=True))

            def chirp(state: Any, t: float, dt: float) -> dict[str, float]:
                sticks = alt_hold(state, t, dt)
                if t < T:
                    phase = 2 * math.pi * f0 * (k**t - 1.0) / math.log(k)
                    frac = (sticks["throttle"] + 1.0) / 2.0
                    frac += h * 2.0 * exp.amplitude * math.sin(phase)
                    sticks["throttle"] = float(np.clip(2.0 * frac - 1.0, -1.0, 1.0))
                return sticks

            return chirp, T + exp.tail_s
        if exp.kind == "rate_doublet":
            axis = exp.axis or "roll"
            rate = abs(float(self.rates[axis].rate_deg_s(exp.amplitude)))
            max_angle = float(exp.params.get("max_angle_deg", 45.0))
            half = min(exp.hold_s, max_angle / max(rate, 1e-6))
            stick = exp.sign * exp.amplitude

            def doublet(state: Any, t: float, dt: float) -> dict[str, float]:
                s = {"throttle": 2.0 * h - 1.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
                if t < half:
                    s[axis] = stick
                elif t < 2 * half:
                    s[axis] = -stick
                return s

            return doublet, 2 * half + exp.tail_s
        if exp.kind == "coast":
            g = 9.81
            heading = np.array([math.cos(self.yaw_ref), math.sin(self.yaw_ref), 0.0])
            accel = exp.sign * g * math.tan(math.radians(exp.amplitude)) * heading
            push = self._hold(self._target(accel_override=accel))

            def coast(state: Any, t: float, dt: float) -> dict[str, float]:
                return push(state, t, dt) if t < exp.hold_s else hold(state, t, dt)

            return coast, exp.hold_s + exp.tail_s
        raise ValueError(f"unknown experiment kind {exp.kind!r}")

    def _run_experiment(self, exp: Experiment) -> None:
        try:
            hold = self._hold(self._target())
            level_thr: list[float] = []

            def settle(state: Any, t: float, dt: float) -> dict[str, float]:
                sticks = hold(state, t, dt)
                cos_tilt = math.cos(math.radians(tilt_deg(state.quat)))
                level_thr.append((sticks["throttle"] + 1.0) / 2.0 * cos_tilt)
                return sticks

            self._loop(
                "settle",
                settle,
                until=self._settled(self.s.settle_hold_s),
                duration=self.s.settle_timeout_s,
                experiment=exp.id,
            )
            if not self._settled(0.0)(self.src.latest(), 0.0):
                raise _Abort(Violation("settle_timeout", ABORT_EXPERIMENT, "did not settle"))
            # Frozen hover throttle for the open-loop window: the settled level throttle.
            n = max(1, int(self.s.settle_hold_s * self.s.rate_hz))
            h = float(np.clip(np.mean(level_thr[-n:]), 0.02, 0.95))
            fn, dur = self._window_fn(exp, h)
            start = self.clock.now()
            self._loop("window", fn, duration=dur, envelope="hard", experiment=exp.id)
            end = self.clock.now()
            self._loop(
                "recover",
                self._hold(self._target()),
                duration=self.s.recover_s,
                envelope="hard",
                experiment=exp.id,
            )
            self._status(exp, "done", start=start, end=end, hover=h, segment=self.rec.segment)
            self.result.done += 1
        except _Abort as a:
            self._violation(a.violation, exp.id)
            self._status(exp, f"aborted:{a.violation.name}", segment=self.rec.segment)
            self.result.aborted += 1
            self._restabilize(exp.id)

    # ------------------------------------------------------------------ run

    def _todo(self) -> list[Experiment]:
        ids = [e.id for e in self.schedule]
        nxt = self.meta.get("next_experiment")
        start = ids.index(nxt) if nxt in ids else 0
        return [
            e
            for e in self.schedule[start:]
            if self.meta["experiments"][e.id].get("status") == "pending"
        ]

    def _fly(self) -> None:
        todo = self._todo()
        if not todo:
            self._event("nothing pending: the schedule is complete")
            return
        self.meta["next_experiment"] = todo[0].id
        if self.resume:
            self._reset_level()
        self._battery("start")
        self._fly_up()
        current = todo[0].pass_index
        for i, exp in enumerate(todo):
            if exp.pass_index != current:
                self._block_boundary()
                current = exp.pass_index
            self.meta["next_experiment"] = exp.id
            self.rec.write_meta(self.meta)
            self._run_experiment(exp)
            self.meta["next_experiment"] = todo[i + 1].id if i + 1 < len(todo) else None
            self.rec.write_meta(self.meta)
        self._land()
        self._disarm()
        self._battery("end")

    def run(self) -> SysidResult:
        result = self.result
        self._init_meta()
        started = False
        try:
            self.sink.arm()
            for part in (self.raw_source, self.abort, self.focus):
                if part is not None and hasattr(part, "start"):
                    part.start()
            started = True
            self._preflight()
            self._wait_focus()
            self._fly()
        except _Stop as stop:
            result.stop_reason, result.error = stop.reason, stop.detail or None
            if stop.violation is not None:
                self._violation(stop.violation, self.meta.get("next_experiment"))
        except KeyboardInterrupt:
            result.stop_reason = STOP_INTERRUPT
        except SysidPreflightError as exc:
            result.stop_reason, result.error = "preflight", str(exc)
            self._finish(started)
            raise
        except Exception as exc:  # noqa: BLE001 - always end disarmed
            log.exception("sysid session failed")
            result.stop_reason, result.error = STOP_ERROR, repr(exc)
        self._finish(started)
        return result

    def _finish(self, started: bool) -> None:
        result = self.result
        try:
            if started and (self.motors_may_spin or self.airborne):
                try:
                    self._disarm()
                except Exception:  # noqa: BLE001
                    log.exception("fail-safe disarm failed")
            elif started:
                self.sink.failsafe()
        finally:
            for part in (self.focus, self.abort, self.raw_source):
                if part is not None and hasattr(part, "stop"):
                    try:
                        part.stop()
                    except Exception:  # noqa: BLE001
                        log.exception("error stopping %s", type(part).__name__)
            try:
                self.sink.close()
            except Exception:  # noqa: BLE001
                log.exception("error closing the sink")
            result.next_experiment = self.meta.get("next_experiment")
            if result.completed:
                result.next_experiment = None
            self.meta["stop_reason"] = result.stop_reason
            self.meta["error"] = result.error
            self.meta["next_experiment"] = result.next_experiment
            self.meta["segments"][-1]["ended_utc"] = _utc()
            self.meta["segments"][-1]["stop_reason"] = result.stop_reason
            self.rec.write_meta(self.meta)
            self.rec.close()


# ----------------------------------------------------------------------------- CLI entry


def expected_band(
    drone_id: str,
    liftoff_name: str | None,
    explicit: tuple[float, float] | None,
    *,
    root: Any = None,
    halfwidth: float = 0.08,
) -> tuple[tuple[float, float] | None, str | None]:
    """The airframe guard's hover band: ``--expect-hover`` > existing profile > ``DRONE_SLOTS``."""
    if explicit is not None:
        return explicit, "--expect-hover"
    from drone_fly import drones

    if drone_id in drones.list_ids(root):
        prof = drones.load(drone_id, root)
        h = prof.hover_throttle
        return (h - halfwidth, h + halfwidth), f"drones/{drone_id}/drone.yaml hover {h:.3f}"
    if liftoff_name:
        from drone_fly.harness.liftoff.ui_driver import DRONE_SLOTS

        slot = DRONE_SLOTS.get(liftoff_name)
        if slot and slot.get("hover_band"):
            lo, hi = slot["hover_band"]
            return (float(lo), float(hi)), f"ui_driver.DRONE_SLOTS[{liftoff_name!r}]"
    return None, None


def run_sysid_liftoff(
    drone_id: str,
    *,
    session: str | None = None,
    resume: bool = False,
    select: bool = False,
    liftoff_name: str | None = None,
    seed: int = 0,
    passes: int = 3,
    amplitudes: tuple[float, ...] | None = None,
    hold_alt: float = 5.0,
    home_forward: float = 0.0,
    ceiling: float = 12.0,
    radius: float = 30.0,
    expect_hover: tuple[float, float] | None = None,
    preset: str | None = "liftoff",
    config: Any = None,
    plant: str = "liftoff",
    out_root: Any = None,
    map_name: str | None = None,
    level: str | None = None,
) -> SysidResult:
    """Build the parts (live Liftoff, or a synthetic plant) and run one :class:`SysidSession`."""
    from pathlib import Path

    from drone_fly.drones.schema import ID_RE
    from drone_fly.harness.config import find_repo_root
    from drone_fly.harness.recording import git_head
    from drone_fly.sysid.recorder import META_FILE, default_session_name
    from drone_fly.sysid.schedule import ScheduleSettings, build_schedule

    if not ID_RE.match(drone_id):
        raise HarnessConfigError(
            f"--drone {drone_id!r} must be kebab-case (<source>-<class>-<model>)"
        )
    if plant not in ("liftoff", "synthetic", "synthetic-5inch"):
        raise HarnessConfigError(f"unknown --plant {plant!r}")
    live = plant == "liftoff"
    repo = find_repo_root()
    root = Path(out_root) if out_root is not None else (repo or Path.cwd()) / "artifacts" / "sysid"
    name = session or default_session_name()
    sdir = root / drone_id / name
    meta_path = sdir / META_FILE
    if resume:
        if not meta_path.is_file():
            raise HarnessConfigError(f"--resume: no session at {sdir} (pass --session NAME)")
    elif meta_path.is_file():
        raise HarnessConfigError(f"session {sdir} exists: pass --resume to continue it")

    if live:
        from drone_fly.harness.liftoff.anticheat import require_no_anticheat

        require_no_anticheat()  # before any input reaches the game
    recorder = SessionRecorder(sdir)
    if resume:
        schedule = [Experiment.from_dict(e) for e in recorder.meta.get("schedule") or []]
        seed = int(recorder.meta.get("seed", seed))
    else:
        sched_kw: dict[str, Any] = {"passes": passes}
        if amplitudes:
            sched_kw["amplitudes"] = tuple(amplitudes)
        schedule = build_schedule(seed, ScheduleSettings(**sched_kw))
    band, band_source = expected_band(drone_id, liftoff_name, expect_hover)
    settings = SessionSettings(
        hold_alt_m=hold_alt,
        home_forward_m=home_forward,
        limits=SafetyLimits(ceiling_m=ceiling, radius_m=radius),
    )
    meta_extra: dict[str, Any] = {
        "liftoff_name": liftoff_name,
        "map": map_name or level,
        "level": level,
        "git_head": git_head(repo),
        "seed": seed,
        "preset": preset,
        "plant": plant,
    }

    if not live:
        from drone_fly.sysid.model import FIVE_INCH_LIKE, MICRO_LIKE
        from drone_fly.sysid.synthetic import _truth, build_synthetic

        params = FIVE_INCH_LIKE if plant == "synthetic-5inch" else MICRO_LIKE
        clock, world, source, sink = build_synthetic(params, seed=seed)
        recorder.clock = clock
        meta_extra["truth"] = _truth(params)
        sess = SysidSession(
            source,
            sink,
            settings,
            drone_id=drone_id,
            schedule=schedule,
            recorder=recorder,
            rates={a: sink.profiles[a] for a in AXES},
            signs=dict(params.signs),
            clock=clock,
            ui_reset=world.reset,
            expected_hover=band,
            hover_band_source=band_source,
            resume=resume,
            meta_extra=meta_extra,
        )
        return sess.run()

    from drone_fly.harness.config import load_config
    from drone_fly.harness.interfaces import BuildContext
    from drone_fly.harness.liftoff import ui_driver
    from drone_fly.harness.liftoff.abort import HotkeyAbortSignal
    from drone_fly.harness.liftoff.anticheat import require_no_anticheat
    from drone_fly.harness.liftoff.focus import Win32ForegroundFocusProbe
    from drone_fly.harness.liftoff.gamepad import VgamepadActionSink
    from drone_fly.harness.liftoff.telemetry import (
        GyroMap,
        LiftoffConverter,
        LiftoffUdpStateSource,
        TelemetryFormat,
    )
    from drone_fly.harness.scale import WorldScale

    if select:
        if not liftoff_name:
            raise HarnessConfigError("--select needs --liftoff-name")
        ui_driver.select_drone(liftoff_name, level=level)
    cfg = load_config("liftoff", config, preset=preset)
    tcfg = cfg.section("telemetry")
    src = LiftoffUdpStateSource(
        TelemetryFormat.from_config(tcfg),
        LiftoffConverter(GyroMap.from_config(tcfg), WorldScale(1.0)),  # Liftoff metres
        host=str(tcfg.get("host", "127.0.0.1")),
        port=int(tcfg.get("port", 9001)),
    )
    ctx = BuildContext(config=cfg, contract=None)
    sink = VgamepadActionSink.from_config(cfg.section("controls"), ctx)
    abort = HotkeyAbortSignal.from_config(cfg.section("abort"), ctx)
    focus = None
    if cfg.parts.focus_probe is not None:
        focus = Win32ForegroundFocusProbe.from_config(cfg.get("focus") or {}, ctx)
    meta_extra["config"] = {"telemetry": tcfg, "controls": cfg.section("controls")}
    settings = SessionSettings(
        **{
            **settings.__dict__,
            "limits": SafetyLimits(
                ceiling_m=ceiling, radius_m=radius, stale_after_s=cfg.runner.stale_after_s
            ),
            "wait_for_state_s": cfg.runner.wait_for_state_s,
            "wait_for_focus_s": float(
                (cfg.get("focus") or {}).get("wait_for_focus_s", settings.wait_for_focus_s)
            ),
        }
    )
    sess = SysidSession(
        src,
        sink,
        settings,
        drone_id=drone_id,
        schedule=schedule,
        recorder=recorder,
        rates=sink.profiles,
        signs=sink.signs,
        abort=abort,
        focus=focus,
        ui_reset=ui_driver.reset,
        anticheat=require_no_anticheat,
        expected_hover=band,
        hover_band_source=band_source,
        resume=resume,
        meta_extra=meta_extra,
    )
    return sess.run()
