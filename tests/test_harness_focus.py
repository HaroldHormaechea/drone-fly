"""UC-64 AC10: the focus fail-safe (pad input reaches whichever window is focused).

The runner waits, pad armed at neutral, until the game is focused (``focus.wait_for_focus_s``),
then fails safe — neutral rates + throttle cut — with ``focus_lost`` the moment it is not. Tested
with a fake focus probe; the ``win32-foreground`` probe with an injected foreground function; and
end to end through ``run_harness`` (run folder records the stop reason and neutral output).
"""

from __future__ import annotations

import json
import sys

import pytest

from drone_fly.harness import registry
from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.interfaces import BuildContext, FocusProbe, Harness
from drone_fly.harness.liftoff import build_liftoff_harness
from drone_fly.harness.liftoff.focus import Win32ForegroundFocusProbe
from drone_fly.harness.liftoff.gamepad import RateProfile, ThrottleMap, VgamepadActionSink
from drone_fly.harness.runner import (
    STOP_ABORT,
    STOP_COMPLETED,
    STOP_FOCUS_LOST,
    HarnessRunner,
    RunnerSettings,
    SimClock,
)
from drone_fly.harness.session import run_harness
from tests.harness_support import (
    DATA_DIR,
    FakeAbort,
    FakeFreeSpaceEstimator,
    FakeGateEstimator,
    FakePolicy,
    FakeSink,
    FakeStateSource,
    ListRecorder,
    MockPad,
    fake_contract,
)


class ScriptedFocus(FocusProbe):
    """``has_focus`` answers from a script (last value repeats); counts start/stop/calls."""

    def __init__(self, script=(True,)):
        self.script = list(script)
        self.calls = 0
        self.started = self.stopped = 0

    @classmethod
    def from_config(cls, cfg, ctx):
        return cls(cfg.get("script", (True,)))

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def has_focus(self):
        v = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return v


def _runner(focus, *, sink=None, abort=None, settings=None, recorder=None, policy=None):
    contract = fake_contract()
    clock = SimClock()
    h = Harness(
        name="fake",
        state_source=FakeStateSource(),
        gate_estimator=FakeGateEstimator(),
        action_sink=sink or FakeSink(),
        free_space_estimator=FakeFreeSpaceEstimator(),
        abort_signal=abort,
        focus_probe=focus,
        clock=clock,
    )
    return HarnessRunner(
        h,
        policy or FakePolicy(extra_dim=30, action=(0.8, 1.0, -0.5, 0.3)),
        ContractObservationAssembler(contract),
        contract,
        settings or RunnerSettings(max_ticks=10),
        recorder=recorder if recorder is not None else ListRecorder(),
    )


def _stops(rec):
    return [e for t in rec.ticks for e in t.events if e.startswith("stop:")]


# ----------------------------------------------------------------------------- runner


def test_waits_for_focus_with_the_pad_neutral_then_flies():
    sink, rec = FakeSink(), ListRecorder()
    focus = ScriptedFocus([False, False, False, True])
    r = _runner(focus, sink=sink, recorder=rec)
    result = r.run()
    assert result.reason == STOP_COMPLETED and result.ticks == 10
    assert focus.started == 1 and focus.stopped == 1
    assert len(sink.sent) == 10  # nothing was sent while waiting
    assert sink.armed and sink.failsafe_calls == 1


def test_focus_never_arrives_fails_safe_after_the_wait():
    sink = FakeSink()
    r = _runner(ScriptedFocus([False]), sink=sink, settings=RunnerSettings(wait_for_focus_s=2.0))
    result = r.run()
    assert result.reason == STOP_FOCUS_LOST and "not focused within 2.0s" in result.error
    assert result.is_failsafe_stop and result.ticks == 0
    assert sink.sent == [] and sink.failsafe_calls == 1


def test_abort_while_waiting_for_focus():
    sink = FakeSink()
    r = _runner(ScriptedFocus([False]), sink=sink, abort=FakeAbort(set_after_checks=3))
    result = r.run()
    assert result.reason == STOP_ABORT and sink.sent == [] and sink.failsafe_calls == 1


def test_focus_lost_mid_run_stops_with_neutral_rates_and_throttle_cut():
    pad = MockPad()
    sink = VgamepadActionSink(
        {ax: RateProfile() for ax in ("roll", "pitch", "yaw")},
        ThrottleMap(0.4, -0.4),
        pad_factory=lambda: pad,
    )
    rec = ListRecorder()
    # focus during the wait (1 call) and 3 ticks, then the user alt-tabs
    focus = ScriptedFocus([True, True, True, True, False])
    r = _runner(focus, sink=sink, recorder=rec, settings=RunnerSettings(max_ticks=50))
    result = r.run()
    assert result.reason == STOP_FOCUS_LOST and result.error == "target window lost focus"
    assert result.ticks == 3 and result.is_failsafe_stop and result.failsafe_ok
    assert _stops(rec) == ["stop:focus_lost"]
    assert rec.ticks[-1].command is None  # the focus check precedes perception and actuation
    flying = pad.history[1]
    assert flying["left_y"] > -1 and flying["right_x"] != 0
    neutral = {"left_x": 0.0, "left_y": -1.0, "right_x": 0.0, "right_y": 0.0}
    assert pad.sticks == neutral
    # neutral evidence of the fail-safe (written to meta.json by the session)
    assert result.failsafe_output == {"sticks": neutral}  # throttle (left_y) cut, rates 0


def test_no_probe_means_no_focus_checks():
    r = _runner(None)
    assert r.run().reason == STOP_COMPLETED


def test_wait_for_focus_setting_from_config():
    cfg = load_config("liftoff", {"focus": {"wait_for_focus_s": 3.5}})
    assert RunnerSettings.from_config(cfg).wait_for_focus_s == 3.5
    assert RunnerSettings.from_config(load_config("liftoff")).wait_for_focus_s == 15.0
    with pytest.raises(HarnessConfigError, match="wait_for_focus_s"):
        RunnerSettings.from_config(load_config("liftoff", {"focus": {"wait_for_focus_s": -1}}))


# ----------------------------------------------------------------------------- win32 probe


def test_foreground_probe_matches_the_game_process_case_insensitively():
    fg = {"v": (101, "liftoff.EXE", "Liftoff")}
    probe = Win32ForegroundFocusProbe(foreground=lambda: fg["v"])
    probe.start()  # injected foreground: nothing to resolve
    assert probe.has_focus() and probe.last_foreground == (101, "liftoff.EXE", "Liftoff")
    fg["v"] = (202, "explorer.exe", "Desktop")
    assert not probe.has_focus()
    fg["v"] = (0, "", "")
    assert not probe.has_focus()


def test_foreground_probe_title_regex():
    probe = Win32ForegroundFocusProbe(
        "Liftoff.exe", title_regex=r"^Liftoff$", foreground=lambda: (1, "Liftoff.exe", "Liftoff")
    )
    assert probe.has_focus()
    other = Win32ForegroundFocusProbe(
        "Liftoff.exe", title_regex=r"^Editor", foreground=lambda: (1, "Liftoff.exe", "Liftoff")
    )
    assert not other.has_focus()


def test_foreground_probe_validation():
    with pytest.raises(HarnessConfigError, match="process_name"):
        Win32ForegroundFocusProbe("")
    with pytest.raises(HarnessConfigError, match="title_regex"):
        Win32ForegroundFocusProbe("Liftoff.exe", title_regex="(")


def test_foreground_probe_refuses_to_start_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(HarnessConfigError, match="needs Windows"):
        Win32ForegroundFocusProbe().start()


@pytest.mark.skipif(sys.platform != "win32", reason="real Win32 foreground lookup")
def test_real_win32_foreground_lookup_runs():
    probe = Win32ForegroundFocusProbe("definitely-not-running.exe")
    probe.start()
    assert probe.has_focus() is False
    hwnd, name, title = probe.last_foreground
    assert isinstance(hwnd, int) and isinstance(name, str) and isinstance(title, str)


def test_from_config_and_registry():
    e = registry.entry("focus_probe", "win32-foreground")
    assert e.target.endswith("focus:Win32ForegroundFocusProbe")
    probe = Win32ForegroundFocusProbe.from_config(
        {"process_name": "Game.exe", "title_regex": "x"}, None
    )
    assert probe.process_name == "game.exe" and probe.title_re.pattern == "x"
    cfg = load_config("liftoff")
    assert cfg.parts.focus_probe == "win32-foreground"
    assert cfg.section("focus") == {
        "process_name": "Liftoff.exe",
        "title_regex": None,
        "wait_for_focus_s": 15.0,
        "auto_focus": True,
    }
    with pytest.raises(HarnessConfigError, match="auto_focus"):
        Win32ForegroundFocusProbe.from_config({"auto_focus": "yes"}, None)


# ----------------------------------------------------------------------------- auto-focus


class _RequestingFocus(ScriptedFocus):
    def __init__(self, script):
        super().__init__(script)
        self.requests = 0

    def request_focus(self):
        self.requests += 1
        return True


def test_runner_requests_focus_once_when_not_focused():
    focus = _RequestingFocus([False, False, True])
    assert _runner(focus).run().reason == STOP_COMPLETED
    assert focus.requests == 1


def test_runner_does_not_request_focus_when_already_focused():
    focus = _RequestingFocus([True])
    _runner(focus).run()
    assert focus.requests == 0


def test_probe_request_focus_uses_the_activator_with_its_target():
    seen = []

    def activator(name, title_re):
        seen.append((name, title_re.pattern if title_re else None))
        return True

    probe = Win32ForegroundFocusProbe("Liftoff.exe", title_regex="^Lift", activator=activator)
    assert probe.request_focus() is True and seen == [("liftoff.exe", "^Lift")]


def test_probe_request_focus_is_best_effort():
    def boom(name, title_re):
        raise OSError("access denied")

    assert Win32ForegroundFocusProbe(activator=boom).request_focus() is False
    assert Win32ForegroundFocusProbe(activator=lambda n, t: False).request_focus() is False
    never = Win32ForegroundFocusProbe(auto_focus=False, activator=lambda n, t: 1 / 0)
    assert never.request_focus() is False  # disabled: activator never called


@pytest.mark.skipif(sys.platform != "win32", reason="real Win32 window enumeration")
def test_real_activator_finds_no_window_for_a_missing_process():
    probe = Win32ForegroundFocusProbe("definitely-not-running.exe")
    assert probe.request_focus() is False


# ----------------------------------------------------------------------------- factory + session


class _NoFrames:
    @classmethod
    def from_config(cls, cfg, ctx):
        from drone_fly.harness.interfaces import FrameSource

        class NoFrames(FrameSource):
            def latest(self):
                return None

        return NoFrames()


@pytest.fixture
def fake_parts():
    registry.register("frame_source", "uc64-no-frames", _NoFrames)
    registry.register("focus_probe", "uc64-scripted", ScriptedFocus)
    yield
    registry.unregister("frame_source", "uc64-no-frames")
    registry.unregister("focus_probe", "uc64-scripted")


def _cfg(focus_probe="uc64-scripted", action_sink="recording"):
    return {
        "parts": {
            "state_source": "liftoff-csv",
            "frame_source": "uc64-no-frames",
            "abort_signal": None,
            "action_sink": action_sink,
            "focus_probe": focus_probe,
        },
        "telemetry": {"replay_csv": str(DATA_DIR / "liftoff_telemetry.csv")},
        "free_space": {"assume_no_obstacles": True},
    }


def test_factory_builds_the_probe_except_in_dry_runs(fake_parts):
    cfg = load_config("liftoff", _cfg())
    h = build_liftoff_harness(cfg, BuildContext(config=cfg, extra_dim=30, dry_run=False))
    assert isinstance(h.focus_probe, ScriptedFocus)
    dry = build_liftoff_harness(cfg, BuildContext(config=cfg, extra_dim=30, dry_run=True))
    assert dry.focus_probe is None
    off = load_config("liftoff", _cfg(focus_probe=None))
    assert build_liftoff_harness(off, BuildContext(config=off, extra_dim=30)).focus_probe is None


class _LosesFocus(ScriptedFocus):
    @classmethod
    def from_config(cls, cfg, ctx):
        return cls([True] * 6 + [False])


def test_session_run_records_focus_lost_and_neutral_output(
    harness_contract, harness_checkpoint_30, harness_k1_dir, tmp_path, fake_parts
):
    registry.register("focus_probe", "uc64-loses-focus", _LosesFocus, replace=True)
    try:
        session = run_harness(
            "liftoff",
            harness_checkpoint_30,
            30,
            config=_cfg(focus_probe="uc64-loses-focus"),
            k1_path=str(harness_k1_dir),
            device="cpu",
            duration_s=2.0,
            dry_run=False,  # the probe is only built for live runs; the sink records only
            output_root=str(tmp_path),
        )
    finally:
        registry.unregister("focus_probe", "uc64-loses-focus")
    r = session.result
    assert r.reason == STOP_FOCUS_LOST and r.ticks == 5 and r.failsafe_ok
    meta = json.loads((session.run_dir / "meta.json").read_text("utf-8"))
    assert meta["stop_reason"] == "focus_lost" and meta["failsafe_ok"] is True
    assert meta["failsafe_output"] == {"failsafe": True}  # RecordingOnlySink has no sticks
    last = (session.run_dir / "ticks.jsonl").read_text("utf-8").splitlines()[-1]
    assert "stop:focus_lost" in json.loads(last)["events"]
