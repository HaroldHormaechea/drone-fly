"""Pad stick-direction check: does each stick move the drone the way the sink assumes?

``drone-fly harness pad-check [--preset ...] [--config ...] [--magnitude 0.15] [--pulse-s 0.2]
[--min-alt 0.5] [--max-alt 6] [--out ...]``

With Liftoff focused and telemetry streaming, the check arms the virtual pad, climbs, then holds
altitude with a P+D throttle loop around ``controls.hover_stick`` (from telemetry height). Per
axis (roll, pitch, yaw) it sends alternating short +/- stick pulses of ``magnitude`` and measures
the mean body rate (``FlightState.omega``, gyro map applied) over the second half of each pulse,
against the rate the configured Betaflight curve predicts for that stick
(``RateProfile.rate_deg_s``) with the configured ``stick_signs``. It reports per axis the sign
agreement and the magnitude ratio, and suggests a ``controls.stick_signs`` block.

It goes neutral and cuts the throttle on: the abort hotkey (F12), stale telemetry, loss of game
focus, altitude outside ``[min_alt, max_alt]`` (after the climb), completion or any error — and
records the reason. Every part (state source, sink, abort, focus probe, clock) is injectable, so
the procedure is unit-tested with fakes.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError

log = logging.getLogger(__name__)

AXES = ("roll", "pitch", "yaw")
OMEGA_INDEX = {"roll": 0, "pitch": 1, "yaw": 2}

STOP_COMPLETED = "completed"
STOP_ABORT = "user_abort"
STOP_STALE = "stale_telemetry"
STOP_NO_TELEMETRY = "no_telemetry"
STOP_FOCUS = "focus_lost"
STOP_ALTITUDE = "altitude_out_of_band"
STOP_ERROR = "error"


@dataclass(frozen=True)
class PadCheckSettings:
    magnitude: float = 0.15
    pulse_s: float = 0.2
    gap_s: float = 0.3
    pulses_per_axis: int = 6
    min_alt: float = 0.5
    max_alt: float = 6.0
    target_alt: float | None = None  # default: middle of the band, at most 2.5 m
    climb_timeout_s: float = 8.0
    settle_s: float = 1.0
    kp: float = 0.15
    kd: float = 0.12
    rate_hz: float = 100.0
    stale_after_s: float = 0.2
    wait_for_state_s: float = 10.0
    wait_for_focus_s: float = 15.0

    def __post_init__(self) -> None:
        if not 0.0 < self.magnitude <= 0.5:
            raise HarnessConfigError("--magnitude must be in (0, 0.5]")
        if self.pulse_s <= 0 or self.gap_s < 0 or self.pulses_per_axis < 2:
            raise HarnessConfigError("pulse settings must be positive (>= 2 pulses per axis)")
        if not 0.0 <= self.min_alt < self.max_alt:
            raise HarnessConfigError("--min-alt must be < --max-alt")

    @property
    def hold_alt(self) -> float:
        if self.target_alt is not None:
            return self.target_alt
        return min(0.5 * (self.min_alt + self.max_alt), 2.5, self.max_alt - 0.5)


@dataclass
class PadCheckResult:
    stop_reason: str = STOP_COMPLETED
    error: str | None = None
    axes: dict[str, Any] = field(default_factory=dict)
    suggested_stick_signs: dict[str, int] = field(default_factory=dict)
    neutral_output: dict[str, float] | None = None
    report_path: str | None = None

    @property
    def completed(self) -> bool:
        return self.stop_reason == STOP_COMPLETED

    def as_dict(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "error": self.error,
            "axes": self.axes,
            "suggested": {"controls": {"stick_signs": self.suggested_stick_signs}},
            "neutral_output": self.neutral_output,
        }


class _Abort(Exception):
    def __init__(self, reason: str, error: str | None = None) -> None:
        super().__init__(reason)
        self.reason, self.error = reason, error


class PadCheck:
    """The check procedure over injected parts (see the module docstring).

    ``state_source`` must produce states in Liftoff metres (world scale 1); its contract-frame
    ``pos[2]`` is the height. ``sink`` needs ``arm()``, ``send_sticks()``, ``failsafe()``,
    ``neutral_sticks()``, ``close()``.
    """

    def __init__(
        self,
        state_source: Any,
        sink: Any,
        profiles: dict[str, Any],
        signs: dict[str, float],
        hover_stick: float,
        settings: PadCheckSettings | None = None,
        *,
        abort: Any = None,
        focus: Any = None,
        clock: Any = None,
    ) -> None:
        from drone_fly.harness.runner import WallClock

        self.src = state_source
        self.sink = sink
        self.profiles = profiles
        self.signs = {a: float(signs.get(a, 1.0)) for a in AXES}
        self.hover_stick = float(hover_stick)
        self.s = settings or PadCheckSettings()
        self.abort = abort
        self.focus = focus
        self.clock = clock or WallClock()
        self.ground_z: float | None = None
        self.in_band = False

    # -- safety --

    def _check(self) -> Any:
        """Abort conditions shared by every phase; returns the latest state."""
        if self.abort is not None and self.abort.is_set():
            raise _Abort(STOP_ABORT)
        if self.focus is not None and not self.focus.has_focus():
            raise _Abort(STOP_FOCUS, "the game window lost focus")
        now = self.clock.now()
        state = self.src.latest()
        age = self.src.age(now)
        if state is None or age > self.s.stale_after_s:
            raise _Abort(STOP_STALE, f"telemetry age {age:.3f}s")
        if self.in_band:
            alt = self.altitude(state)
            if not self.s.min_alt <= alt <= self.s.max_alt:
                raise _Abort(STOP_ALTITUDE, f"altitude {alt:.2f} m")
        return state

    def altitude(self, state: Any) -> float:
        return float(state.pos[2]) - (self.ground_z or 0.0)

    def throttle(self, state: Any) -> float:
        err = self.s.hold_alt - self.altitude(state)
        stick = self.hover_stick + self.s.kp * err - self.s.kd * float(state.vel[2])
        return float(np.clip(stick, -1.0, 1.0))

    # -- phases --

    def _tick(self, k: int, t0: float) -> float:
        deadline = t0 + k / self.s.rate_hz
        self.clock.sleep_until(deadline)
        return deadline

    def _wait(self) -> None:
        deadline = self.clock.now() + self.s.wait_for_state_s
        while self.src.latest() is None:
            if self.abort is not None and self.abort.is_set():
                raise _Abort(STOP_ABORT)
            if self.clock.now() >= deadline:
                raise _Abort(STOP_NO_TELEMETRY, "no telemetry before wait_for_state_s")
            self.clock.sleep_until(self.clock.now() + 0.01)
        if self.focus is None:
            return
        deadline = self.clock.now() + self.s.wait_for_focus_s
        while not self.focus.has_focus():
            if self.abort is not None and self.abort.is_set():
                raise _Abort(STOP_ABORT)
            if self.clock.now() >= deadline:
                raise _Abort(STOP_FOCUS, "game window not focused within wait_for_focus_s")
            self.clock.sleep_until(self.clock.now() + 0.05)

    def _hold(self, duration: float, channel: str | None = None, stick: float = 0.0) -> list[float]:
        """Hold altitude for ``duration`` with ``channel`` at ``stick``; omega samples (deg/s)."""
        n = max(1, int(round(duration * self.s.rate_hz)))
        t0 = self.clock.now()
        samples = []
        for k in range(n):
            self._tick(k, t0)
            state = self._check()
            cmd = {"throttle": self.throttle(state)}
            if channel is not None:
                cmd[channel] = stick
            self.sink.send_sticks(cmd)
            if channel is not None and k >= n // 2:
                samples.append(math.degrees(float(state.omega[OMEGA_INDEX[channel]])))
        return samples

    def _climb(self) -> None:
        state = self._check()
        self.ground_z = float(state.pos[2])
        t0 = self.clock.now()
        k = 0
        while True:
            self._tick(k, t0)
            k += 1
            state = self._check()
            if self.altitude(state) >= self.s.min_alt + 0.3:
                break
            if self.clock.now() - t0 > self.s.climb_timeout_s:
                raise _Abort(STOP_ALTITUDE, "did not climb into the altitude band")
            self.sink.send_sticks({"throttle": max(self.throttle(state), self.hover_stick + 0.15)})
        self.in_band = True
        self._hold(self.s.settle_s)

    def _axis(self, axis: str) -> dict[str, Any]:
        prof = self.profiles[axis]
        sign = self.signs[axis]
        pulses = []
        for i in range(self.s.pulses_per_axis):
            stick = self.s.magnitude * (1.0 if i % 2 == 0 else -1.0)
            expected = sign * prof.rate_deg_s(stick)
            samples = self._hold(self.s.pulse_s, axis, stick)
            measured = float(np.mean(samples)) if samples else 0.0
            pulses.append({"stick": stick, "expected_deg_s": expected, "measured_deg_s": measured})
            self._hold(self.s.gap_s)
        exp = np.array([p["expected_deg_s"] for p in pulses])
        mea = np.array([p["measured_deg_s"] for p in pulses])
        agree = float(np.mean(np.sign(mea) == np.sign(exp)))
        ratio = float(np.median(np.abs(mea) / np.maximum(np.abs(exp), 1e-9)))
        return {
            "current_sign": int(sign),
            "sign_agreement": agree,
            "magnitude_ratio": ratio,
            "suggested_sign": int(sign if agree >= 0.5 else -sign),
            "pulses": pulses,
        }

    def run(self) -> PadCheckResult:
        result = PadCheckResult()
        try:
            self.sink.arm()
            for part in (self.src, self.abort, self.focus):
                if part is not None and hasattr(part, "start"):
                    part.start()
            self._wait()
            self._climb()
            for axis in AXES:
                result.axes[axis] = self._axis(axis)
        except _Abort as stop:
            result.stop_reason, result.error = stop.reason, stop.error
        except KeyboardInterrupt:
            result.stop_reason = STOP_ABORT
        except Exception as exc:  # noqa: BLE001 - always end neutral
            log.exception("pad check failed")
            result.stop_reason, result.error = STOP_ERROR, repr(exc)
        finally:
            try:
                self.sink.failsafe()
                result.neutral_output = self.sink.neutral_sticks()
            finally:
                for part in (self.focus, self.abort, self.src):
                    if part is not None and hasattr(part, "stop"):
                        try:
                            part.stop()
                        except Exception:  # noqa: BLE001
                            log.exception("error stopping %s", type(part).__name__)
                self.sink.close()
        result.suggested_stick_signs = {
            a: result.axes[a]["suggested_sign"] if a in result.axes else int(self.signs[a])
            for a in AXES
        }
        return result


def run_pad_check(
    *,
    config: str | Path | dict[str, Any] | None = None,
    preset: str | None = None,
    harness: str = "liftoff",
    settings: PadCheckSettings | None = None,
    out: str | Path | None = None,
) -> PadCheckResult:
    """Build the live parts from config, run :class:`PadCheck`, write ``report.json``."""
    from drone_fly.harness.config import find_repo_root, load_config
    from drone_fly.harness.interfaces import BuildContext
    from drone_fly.harness.liftoff.abort import HotkeyAbortSignal
    from drone_fly.harness.liftoff.focus import Win32ForegroundFocusProbe
    from drone_fly.harness.liftoff.gamepad import VgamepadActionSink
    from drone_fly.harness.liftoff.telemetry import (
        GyroMap,
        LiftoffConverter,
        LiftoffUdpStateSource,
        TelemetryFormat,
    )
    from drone_fly.harness.scale import WorldScale

    cfg = load_config(harness, config, preset=preset)
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
    base = settings or PadCheckSettings()
    focus_wait = float((cfg.get("focus") or {}).get("wait_for_focus_s", base.wait_for_focus_s))
    s = PadCheckSettings(
        **{
            **base.__dict__,
            "stale_after_s": cfg.runner.stale_after_s,
            "wait_for_state_s": cfg.runner.wait_for_state_s,
            "wait_for_focus_s": focus_wait,
        }
    )
    controls = cfg.section("controls")
    check = PadCheck(
        src,
        sink,
        sink.profiles,
        sink.signs,
        float(controls.get("hover_stick", -0.4)),
        s,
        abort=abort,
        focus=focus,
    )
    result = check.run()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if out is None:
        root = find_repo_root() or Path.cwd()
        out = root / "artifacts" / "harness" / "liftoff" / "reports" / "pad-check" / stamp
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": stamp,
        "preset": preset,
        "settings": s.__dict__,
        "rates": {a: sink.profiles[a].__dict__ for a in AXES},
        **result.as_dict(),
    }
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    result.report_path = str(path)
    return result
