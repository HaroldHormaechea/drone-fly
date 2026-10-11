"""UC-64 AC8: the pad stick-direction check, with fakes (no game, no virtual pad).

A simulated drone integrates the sticks the check sends (throttle -> climb rate around the hover
stick, each rate stick -> body rate through the Betaflight curve with the *game's* sign); the
check must report per-axis sign agreement / magnitude ratio, suggest ``stick_signs``, and go
neutral (rates 0, throttle cut) on F12, stale telemetry, focus loss, altitude out of band,
completion and errors. ``run_pad_check`` is exercised end to end with its live parts replaced.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.interfaces import StateSource
from drone_fly.harness.liftoff.gamepad import RateProfile, ThrottleMap, VgamepadActionSink
from drone_fly.harness.liftoff.pad_check import (
    STOP_ABORT,
    STOP_ALTITUDE,
    STOP_COMPLETED,
    STOP_ERROR,
    STOP_FOCUS,
    STOP_NO_TELEMETRY,
    STOP_STALE,
    PadCheck,
    PadCheckSettings,
)
from drone_fly.harness.runner import SimClock
from drone_fly.harness.types import FlightState
from tests.harness_support import FakeAbort, MockPad

HOVER = -0.06
PROFILES = {a: RateProfile(1.0, 0.7, 0.0) for a in ("roll", "pitch", "yaw")}
CONFIG_SIGNS = {"roll": 1.0, "pitch": 1.0, "yaw": -1.0}
NEUTRAL = {"throttle": -1.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}


class FakePadSink:
    """Records raw channel commands; ``failsafe`` / ``close`` leave it neutral."""

    def __init__(self, raise_on_send: int | None = None):
        self.channels = dict(NEUTRAL)
        self.history: list[dict[str, float]] = []
        self.armed = self.closed = False
        self.failsafe_calls = 0
        self.raise_on_send = raise_on_send

    def arm(self):
        self.armed = True

    def send_sticks(self, channels):
        if self.raise_on_send is not None and len(self.history) >= self.raise_on_send:
            raise RuntimeError("pad unplugged")
        self.channels = {**NEUTRAL, **channels}
        self.history.append(dict(self.channels))
        return self.channels

    def failsafe(self):
        self.failsafe_calls += 1
        self.channels = dict(NEUTRAL)

    def neutral_sticks(self):
        return {"left_x": 0.0, "left_y": -1.0, "right_x": 0.0, "right_y": 0.0}

    def close(self):
        self.closed = True


class SimDrone(StateSource):
    """Vertical dynamics + body rates from the sticks currently held (Liftoff metres, FLU)."""

    def __init__(
        self,
        clock,
        channels,
        game_signs=None,
        rate_gain: float = 1.0,
        thrust_gain: float = 8.0,
        freeze_at: float | None = None,
        drop_at: float | None = None,
        appear_at: float = 0.0,
        ground_z: float = 0.3,
    ):
        self.clock = clock
        self.channels = channels  # callable -> current {throttle, roll, pitch, yaw}
        self.game_signs = game_signs or dict(CONFIG_SIGNS)
        self.rate_gain = rate_gain
        self.thrust_gain = thrust_gain
        self.freeze_at = freeze_at
        self.drop_at = drop_at
        self.appear_at = appear_at
        self.z, self.vz = ground_z, 0.0
        self.ground_z = ground_z
        self.t = clock.now()
        self.started = self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def _advance(self, now):
        end = now if self.freeze_at is None else min(now, self.freeze_at)
        while self.t < end - 1e-12:
            dt = min(0.002, end - self.t)
            ch = self.channels()
            az = self.thrust_gain * (ch["throttle"] - HOVER) - 1.0 * self.vz
            self.vz += az * dt
            self.z += self.vz * dt
            if self.z < self.ground_z:
                self.z, self.vz = self.ground_z, max(self.vz, 0.0)
            if self.drop_at is not None and self.t < self.drop_at <= self.t + dt:
                self.z = self.ground_z + 0.1  # knocked down (e.g. a crash)
            self.t += dt

    def latest(self):
        now = self.clock.now()
        if now < self.appear_at:
            return None
        self._advance(now)
        ch = self.channels()
        omega = [
            math.radians(self.game_signs[a] * self.rate_gain * PROFILES[a].rate_deg_s(ch[a]))
            for a in ("roll", "pitch", "yaw")
        ]
        return FlightState(
            pos=np.array([0.0, 0.0, self.z], np.float32),
            vel=np.array([0.0, 0.0, self.vz], np.float32),
            quat=np.array([1.0, 0.0, 0.0, 0.0], np.float32),
            omega=np.array(omega, np.float32),
            t=self.t,
        )

    def age(self, now):
        if self.freeze_at is not None and now > self.freeze_at:
            return now - self.freeze_at
        return 0.0


class Focus:
    def __init__(self, clock, lose_at=None, gain_at=0.0):
        self.clock, self.lose_at, self.gain_at = clock, lose_at, gain_at
        self.started = self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def has_focus(self):
        t = self.clock.now()
        return t >= self.gain_at and (self.lose_at is None or t < self.lose_at)


def _check(sink=None, drone_kw=None, settings=None, abort=None, focus=None, signs=None):
    clock = SimClock()
    sink = sink or FakePadSink()
    drone = SimDrone(clock, lambda: sink.channels, **(drone_kw or {}))
    check = PadCheck(
        drone,
        sink,
        PROFILES,
        signs or CONFIG_SIGNS,
        HOVER,
        settings or PadCheckSettings(),
        abort=abort,
        focus=focus(clock) if callable(focus) else focus,
        clock=clock,
    )
    return check, sink, drone, clock


def _ended_neutral(result, sink):
    assert sink.failsafe_calls == 1 and sink.closed
    assert sink.channels == NEUTRAL
    assert result.neutral_output == {"left_x": 0.0, "left_y": -1.0, "right_x": 0.0, "right_y": 0.0}


# ----------------------------------------------------------------------------- measurement


def test_matching_signs_complete_with_full_agreement():
    check, sink, drone, _ = _check(focus=lambda c: Focus(c))
    r = check.run()
    assert r.stop_reason == STOP_COMPLETED and r.completed and r.error is None
    for axis in ("roll", "pitch", "yaw"):
        a = r.axes[axis]
        assert a["sign_agreement"] == 1.0
        assert a["magnitude_ratio"] == pytest.approx(1.0, rel=1e-3)
        assert a["current_sign"] == a["suggested_sign"] == int(CONFIG_SIGNS[axis])
        assert len(a["pulses"]) == 6
        assert {p["stick"] for p in a["pulses"]} == {0.15, -0.15}
    assert r.suggested_stick_signs == {"roll": 1, "pitch": 1, "yaw": -1}
    assert drone.started == 1 and drone.stopped == 1
    _ended_neutral(r, sink)


def test_pulses_go_through_one_channel_while_altitude_is_held():
    check, sink, drone, _ = _check()
    r = check.run()
    assert r.completed
    pulses = [h for h in sink.history if h["pitch"] != 0.0]
    assert pulses and all(h["roll"] == 0.0 and h["yaw"] == 0.0 for h in pulses)
    assert {abs(h["pitch"]) for h in pulses} == {0.15}
    # P+D throttle around the hover stick kept the drone inside the band
    assert 0.5 <= drone.z - drone.ground_z <= 6.0
    assert all(-1.0 <= h["throttle"] <= 1.0 for h in sink.history)


def test_inverted_game_axis_suggests_a_sign_flip():
    check, sink, _, _ = _check(drone_kw={"game_signs": {"roll": 1.0, "pitch": -1.0, "yaw": -1.0}})
    r = check.run()
    assert r.completed
    assert r.axes["pitch"]["sign_agreement"] == 0.0
    assert r.axes["pitch"]["suggested_sign"] == -1 and r.axes["roll"]["suggested_sign"] == 1
    assert r.suggested_stick_signs == {"roll": 1, "pitch": -1, "yaw": -1}
    d = r.as_dict()
    assert d["suggested"] == {"controls": {"stick_signs": {"roll": 1, "pitch": -1, "yaw": -1}}}


def test_magnitude_ratio_reports_a_rate_mismatch():
    check, _, _, _ = _check(drone_kw={"rate_gain": 0.5})
    r = check.run()
    for axis in ("roll", "pitch", "yaw"):
        assert r.axes[axis]["magnitude_ratio"] == pytest.approx(0.5, rel=1e-3)
        assert r.axes[axis]["sign_agreement"] == 1.0


# ----------------------------------------------------------------------------- abort reasons


def test_f12_aborts_and_goes_neutral():
    check, sink, _, _ = _check(abort=FakeAbort(set_after_checks=50))
    r = check.run()
    assert r.stop_reason == STOP_ABORT and not r.completed
    assert r.suggested_stick_signs == {"roll": 1, "pitch": 1, "yaw": -1}  # unchanged config
    _ended_neutral(r, sink)


def test_stale_telemetry_aborts():
    check, sink, _, _ = _check(drone_kw={"freeze_at": 3.0})
    r = check.run()
    assert r.stop_reason == STOP_STALE and "telemetry age" in r.error
    _ended_neutral(r, sink)


def test_no_telemetry_before_the_wait_times_out():
    check, sink, _, _ = _check(
        drone_kw={"appear_at": 100.0}, settings=PadCheckSettings(wait_for_state_s=1.0)
    )
    r = check.run()
    assert r.stop_reason == STOP_NO_TELEMETRY and sink.history == []
    _ended_neutral(r, sink)


def test_focus_never_gained_or_lost_mid_check():
    check, sink, _, _ = _check(
        focus=lambda c: Focus(c, gain_at=100.0), settings=PadCheckSettings(wait_for_focus_s=2.0)
    )
    r = check.run()
    assert r.stop_reason == STOP_FOCUS and sink.history == []
    _ended_neutral(r, sink)
    check, sink, _, _ = _check(focus=lambda c: Focus(c, lose_at=4.0))
    r = check.run()
    assert r.stop_reason == STOP_FOCUS and "lost focus" in r.error and sink.history
    _ended_neutral(r, sink)


def test_waits_for_focus_before_sending_anything():
    check, sink, _, _ = _check(focus=lambda c: Focus(c, gain_at=1.0))
    r = check.run()
    assert r.completed and sink.armed


def test_altitude_out_of_band_after_the_climb():
    check, sink, _, _ = _check(drone_kw={"drop_at": 4.0})
    r = check.run()
    assert r.stop_reason == STOP_ALTITUDE and "altitude" in r.error
    _ended_neutral(r, sink)


def test_failing_to_climb_aborts():
    check, sink, _, _ = _check(drone_kw={"thrust_gain": 0.0})
    r = check.run()
    assert r.stop_reason == STOP_ALTITUDE and "did not climb" in r.error
    _ended_neutral(r, sink)


def test_errors_end_neutral_and_are_reported():
    check, sink, _, _ = _check(sink=FakePadSink(raise_on_send=30))
    r = check.run()
    assert r.stop_reason == STOP_ERROR and "pad unplugged" in r.error
    _ended_neutral(r, sink)


def test_keyboard_interrupt_counts_as_abort():
    class Interrupting(FakePadSink):
        def send_sticks(self, channels):
            raise KeyboardInterrupt

    check, sink, _, _ = _check(sink=Interrupting())
    r = check.run()
    assert r.stop_reason == STOP_ABORT
    _ended_neutral(r, sink)


def test_settings_validation_and_hold_altitude():
    for kw in (
        {"magnitude": 0.0},
        {"magnitude": 0.6},
        {"pulse_s": 0},
        {"pulses_per_axis": 1},
        {"min_alt": 3.0, "max_alt": 2.0},
    ):
        with pytest.raises(HarnessConfigError):
            PadCheckSettings(**kw)
    assert PadCheckSettings().hold_alt == 2.5
    assert PadCheckSettings(min_alt=0.5, max_alt=2.0).hold_alt == 1.25
    assert PadCheckSettings(target_alt=1.0).hold_alt == 1.0


# ----------------------------------------------------------------------------- gamepad send_sticks


def _sink():
    pad = MockPad()
    sink = VgamepadActionSink(
        {a: RateProfile() for a in ("roll", "pitch", "yaw")},
        ThrottleMap(0.4, HOVER),
        pad_factory=lambda: pad,
    )
    return sink, pad


def test_send_sticks_maps_channels_raw_and_clips():
    sink, pad = _sink()
    with pytest.raises(RuntimeError, match="not armed"):
        sink.send_sticks({"roll": 0.1})
    sink.arm()
    out = sink.send_sticks({"throttle": 0.2, "pitch": -0.15, "roll": 3.0})
    # mode 2: throttle left_y, yaw left_x, pitch right_y, roll right_x; no rate curve / signs
    assert out == {"left_y": 0.2, "left_x": 0.0, "right_y": -0.15, "right_x": 1.0}
    assert pad.sticks == out
    assert sink.send_sticks({}) == sink.neutral_sticks()  # missing channels neutral
    with pytest.raises(HarnessConfigError, match="unknown stick channels"):
        sink.send_sticks({"aux": 1.0})


# ----------------------------------------------------------------------------- run_pad_check


def test_run_pad_check_end_to_end_writes_the_report(tmp_path, monkeypatch):
    """Live parts replaced: UDP source -> simulated drone, pad -> MockPad, hotkey/focus -> fakes."""
    from drone_fly.harness.liftoff import abort as abort_mod
    from drone_fly.harness.liftoff import focus as focus_mod
    from drone_fly.harness.liftoff import gamepad as gamepad_mod
    from drone_fly.harness.liftoff import telemetry as telemetry_mod
    from drone_fly.harness.liftoff.pad_check import run_pad_check

    class WallClock:
        def now(self):
            return time.perf_counter()

    holder = {}
    real_from_config = gamepad_mod.VgamepadActionSink.from_config.__func__

    def sink_from_config(cls, cfg, ctx):
        sink = real_from_config(cls, cfg, ctx)
        sink.pad_factory = MockPad
        holder["sink"] = sink
        return sink

    def channels():
        sink = holder["sink"]
        if sink.pad is None:
            return dict(NEUTRAL)
        inv = {v: k for k, v in sink.axis_map.items()}
        return {inv[ax]: val for ax, val in sink.pad.sticks.items()}

    def udp_source(fmt, converter, host="127.0.0.1", port=9001, **kw):
        holder["source_args"] = (fmt.rpm_count_prefix, converter.scale.factor, host, port)
        return SimDrone(WallClock(), channels)

    class NoAbort:
        def is_set(self):
            return False

    class AlwaysFocused:
        def has_focus(self):
            return True

    monkeypatch.setattr(
        gamepad_mod.VgamepadActionSink, "from_config", classmethod(sink_from_config)
    )
    monkeypatch.setattr(telemetry_mod, "LiftoffUdpStateSource", udp_source)
    monkeypatch.setattr(
        abort_mod.HotkeyAbortSignal, "from_config", classmethod(lambda c, a, b: NoAbort())
    )
    monkeypatch.setattr(
        focus_mod.Win32ForegroundFocusProbe,
        "from_config",
        classmethod(lambda c, a, b: AlwaysFocused()),
    )
    settings = PadCheckSettings(pulse_s=0.04, gap_s=0.0, pulses_per_axis=2, settle_s=0.05)
    result = run_pad_check(preset="liftoff_micro_arena", settings=settings, out=tmp_path / "pc")
    assert result.stop_reason == STOP_COMPLETED, (result.stop_reason, result.error)
    # the UDP source got the preset's datagram layout and Liftoff metres (world scale 1)
    assert holder["source_args"] == (True, 1.0, "127.0.0.1", 9001)
    report = json.loads((tmp_path / "pc" / "report.json").read_text(encoding="utf-8"))
    assert report["preset"] == "liftoff_micro_arena" and report["stop_reason"] == "completed"
    assert set(report["axes"]) == {"roll", "pitch", "yaw"}
    assert report["suggested"] == {"controls": {"stick_signs": {"roll": 1, "pitch": 1, "yaw": -1}}}
    assert report["settings"]["wait_for_focus_s"] == 15.0
    assert report["rates"]["roll"] == {"rc_rate": 1.0, "super_rate": 0.7, "expo": 0.0}
    assert report["neutral_output"] == {
        "left_x": 0.0,
        "left_y": -1.0,
        "right_x": 0.0,
        "right_y": 0.0,
    }
    assert result.report_path == str(tmp_path / "pc" / "report.json")
    assert holder["sink"].pad is None  # released
