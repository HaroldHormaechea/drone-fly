"""UC-62 AC10: the fixed-rate runner (scheduling, latency logging, latched fail-safe) with fakes."""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.errors import PerceptionCoverageError
from drone_fly.harness.interfaces import Harness
from drone_fly.harness.liftoff.abort import HotkeyAbortSignal, vk_for
from drone_fly.harness.liftoff.gamepad import RateProfile, ThrottleMap, VgamepadActionSink
from drone_fly.harness.runner import (
    NORMAL_STOPS,
    STOP_ABORT,
    STOP_COMPLETED,
    STOP_GATE_LOST,
    STOP_INTERRUPT,
    STOP_NO_TELEMETRY,
    STOP_PERCEPTION_ERROR,
    STOP_PERCEPTION_STALE,
    STOP_POLICY_ERROR,
    STOP_SINK_ERROR,
    STOP_STALE_TELEMETRY,
    HarnessRunner,
    RecordingOnlySink,
    RunnerSettings,
    SimClock,
    WallClock,
)
from tests.harness_support import (
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


def _runner(
    *,
    extra_dim=30,
    state=None,
    gate=None,
    fs=None,
    sink=None,
    policy=None,
    abort=None,
    settings=None,
    clock=None,
    recorder=None,
):
    clock = clock or SimClock()
    contract = fake_contract()
    h = Harness(
        name="fake",
        state_source=state or FakeStateSource(),
        gate_estimator=gate or FakeGateEstimator(),
        action_sink=sink or FakeSink(),
        free_space_estimator=fs if fs is not None else FakeFreeSpaceEstimator(),
        abort_signal=abort,
        clock=clock,
    )
    r = HarnessRunner(
        h,
        policy or FakePolicy(extra_dim=extra_dim),
        ContractObservationAssembler(contract),
        contract,
        settings or RunnerSettings(max_ticks=10),
        recorder=recorder if recorder is not None else ListRecorder(),
    )
    return r


def _stop_events(rec: ListRecorder) -> list[str]:
    return [e for t in rec.ticks for e in t.events if e.startswith("stop:")]


# ----------------------------------------------------------------------------- scheduling


def test_fixed_50hz_schedule_on_sim_clock():
    rec = ListRecorder()
    r = _runner(settings=RunnerSettings(max_ticks=10), recorder=rec)
    result = r.run()
    assert result.reason == STOP_COMPLETED and result.ticks == 10
    assert result.missed_deadlines == 0
    times = [t.sim_time for t in rec.ticks]
    np.testing.assert_allclose(np.diff(times), 0.02, atol=1e-12)
    assert [t.tick for t in rec.ticks] == list(range(10))


def test_rate_is_configurable_and_duration_limits_ticks():
    rec = ListRecorder()
    r = _runner(settings=RunnerSettings(rate_hz=100.0, max_duration_s=0.1), recorder=rec)
    result = r.run()
    assert result.ticks == 10
    np.testing.assert_allclose(np.diff([t.sim_time for t in rec.ticks]), 0.01, atol=1e-12)


def test_missed_deadlines_are_counted_and_skipped_without_catch_up():
    clock = SimClock()
    rec = ListRecorder()
    policy = FakePolicy(clock=clock, cost_s=0.05)  # inference takes 2.5 periods
    r = _runner(policy=policy, clock=clock, recorder=rec, settings=RunnerSettings(max_ticks=12))
    result = r.run()
    assert result.missed_deadlines == result.ticks > 0
    assert all(t.missed_deadline for t in rec.ticks)
    ks = [t.tick for t in rec.ticks]
    assert all(b - a == 3 for a, b in zip(ks, ks[1:], strict=False))  # skip to the next future slot
    # Each tick starts on its own slot (no burst of back-to-back catch-up ticks).
    np.testing.assert_allclose([t.sim_time for t in rec.ticks], [k * 0.02 for k in ks], atol=1e-9)
    assert rec.ticks[0].latency_s["inference"] == pytest.approx(0.05)


def test_latency_and_age_fields_recorded_every_tick():
    rec = ListRecorder()
    r = _runner(
        recorder=rec, state=FakeStateSource(age_s=0.01), settings=RunnerSettings(max_ticks=3)
    )
    r.run()
    for t in rec.ticks:
        assert set(t.latency_s) == {"state", "perception", "inference", "actuation", "total"}
        assert t.ages_s["telemetry"] == pytest.approx(0.01)
        assert "gate" in t.ages_s and "free_space" in t.ages_s
        assert t.command is not None and t.obs is not None and t.extra.shape == (30,)
        assert t.sink_output == {"ok": True}


def test_wall_clock_sleep_until_reaches_deadline():
    clock = WallClock()
    t0 = clock.now()
    clock.sleep_until(t0 + 0.01)
    assert clock.now() >= t0 + 0.01


def test_sim_clock_never_goes_backwards():
    c = SimClock(5.0)
    c.sleep_until(1.0)
    assert c.now() == 5.0
    c.advance(0.5)
    assert c.now() == 5.5


# ----------------------------------------------------------------------------- lifecycle


def test_prepare_order_coverage_then_load_then_arm():
    log: list[str] = []
    policy = FakePolicy(log=log)
    sink = FakeSink(log=log)
    r = _runner(policy=policy, sink=sink)
    r.prepare()
    assert log == ["load", "arm"]


def test_coverage_refusal_before_model_load_and_sink_arm():
    policy = FakePolicy(extra_dim=70)
    sink = FakeSink()
    fs = FakeFreeSpaceEstimator()
    fs.provides = frozenset({"obstacle_vision"})  # no raycast
    r = _runner(extra_dim=70, policy=policy, sink=sink, fs=fs)
    with pytest.raises(PerceptionCoverageError) as ei:
        r.run()
    assert ei.value.missing == ("raycast",)
    assert not policy.loaded and not sink.armed and sink.failsafe_calls == 0


def test_run_starts_and_shuts_down_every_part():
    state, fs, sink = FakeStateSource(), FakeFreeSpaceEstimator(), FakeSink()
    r = _runner(state=state, fs=fs, sink=sink, settings=RunnerSettings(max_ticks=2))
    r.run()
    assert state.started == state.stopped == 1
    assert fs.started == fs.stopped == 1
    assert sink.closed == 1
    r.shutdown()  # idempotent / safe to repeat
    assert sink.closed == 2


def test_normal_completion_still_leaves_sink_in_failsafe():
    sink = FakeSink()
    result = _runner(sink=sink, settings=RunnerSettings(max_ticks=3)).run()
    assert result.reason in NORMAL_STOPS and not result.is_failsafe_stop
    assert sink.failsafe_calls == 1 and result.failsafe


def test_dry_run_recording_sink():
    sink = RecordingOnlySink()
    r = _runner(sink=sink, settings=RunnerSettings(max_ticks=4))
    r.run()
    assert sink.sent == 4 and sink.failsafe_calls == 1
    assert RecordingOnlySink.from_config({}, None).send(None) == {"dry_run": True}  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- fail-safe


def _assert_failsafe(result, sink: FakeSink, reason: str, rec: ListRecorder | None = None):
    assert result.reason == reason
    assert result.is_failsafe_stop and result.failsafe and result.failsafe_ok
    assert sink.failsafe_calls == 1
    if rec is not None:
        assert _stop_events(rec) == [f"stop:{reason}"]


def test_failsafe_on_stale_telemetry():
    sink, rec = FakeSink(), ListRecorder()
    result = _runner(state=FakeStateSource(age_s=0.5), sink=sink, recorder=rec).run()
    _assert_failsafe(result, sink, STOP_STALE_TELEMETRY, rec)
    assert sink.sent == [] and "0.500" in result.error


def test_telemetry_going_stale_mid_run_stops_without_another_send():
    state, sink, rec = FakeStateSource(), FakeSink(), ListRecorder()

    class StalingPolicy(FakePolicy):
        def act(self, obs, extra):
            if len(self.calls) == 2:
                state.age_s = 1.0
            return super().act(obs, extra)

    result = _runner(state=state, sink=sink, policy=StalingPolicy(), recorder=rec).run()
    _assert_failsafe(result, sink, STOP_STALE_TELEMETRY, rec)
    assert len(sink.sent) == 3


def test_no_telemetry_before_wait_timeout():
    sink = FakeSink()
    st = FakeStateSource()
    st.state = None
    r = _runner(state=st, sink=sink, settings=RunnerSettings(max_ticks=5, wait_for_state_s=0.5))
    result = r.run()
    _assert_failsafe(result, sink, STOP_NO_TELEMETRY)
    assert r.clock.now() >= 0.5


def test_failsafe_on_gate_lost():
    sink, rec = FakeSink(), ListRecorder()
    result = _runner(gate=FakeGateEstimator(status="lost"), sink=sink, recorder=rec).run()
    _assert_failsafe(result, sink, STOP_GATE_LOST, rec)
    assert sink.sent == []


def test_failsafe_on_stale_free_space_estimate_when_model_needs_it():
    sink, rec = FakeSink(), ListRecorder()
    result = _runner(fs=FakeFreeSpaceEstimator(age_s=1.0), sink=sink, recorder=rec).run()
    _assert_failsafe(result, sink, STOP_PERCEPTION_STALE, rec)
    assert rec.ticks[-1].ages_s["free_space"] == pytest.approx(1.0)


def test_stale_free_space_ignored_when_model_does_not_use_it():
    sink = FakeSink()
    r = _runner(
        extra_dim=12,
        policy=FakePolicy(extra_dim=12),
        fs=FakeFreeSpaceEstimator(age_s=5.0),
        sink=sink,
        settings=RunnerSettings(max_ticks=3),
    )
    result = r.run()
    assert result.reason == STOP_COMPLETED and len(sink.sent) == 3


def test_depth_estimate_never_ready_fails_safe():
    """A free-space estimator with no depth map yet stamps -inf -> perception_stale."""

    class NotReady(FakeFreeSpaceEstimator):
        def update(self, frame, state, target, now):
            est = super().update(frame, state, target, now)
            from dataclasses import replace

            return replace(est, stamp=-np.inf)

    sink = FakeSink()
    result = _runner(extra_dim=70, policy=FakePolicy(extra_dim=70), fs=NotReady(), sink=sink).run()
    _assert_failsafe(result, sink, STOP_PERCEPTION_STALE)


@pytest.mark.parametrize("where", ["gate", "free_space"])
def test_failsafe_on_perception_exception(where):
    sink, rec = FakeSink(), ListRecorder()
    boom = ValueError("perception boom")
    kw = (
        {"gate": FakeGateEstimator(raises=boom)}
        if where == "gate"
        else {"fs": FakeFreeSpaceEstimator(raises=boom)}
    )
    result = _runner(sink=sink, recorder=rec, **kw).run()
    _assert_failsafe(result, sink, STOP_PERCEPTION_ERROR, rec)
    assert "perception boom" in result.error


def test_failsafe_on_policy_exception():
    sink = FakeSink()
    result = _runner(policy=FakePolicy(raises=RuntimeError("nan")), sink=sink).run()
    _assert_failsafe(result, sink, STOP_POLICY_ERROR)
    assert sink.sent == []


def test_failsafe_on_sink_exception():
    sink = FakeSink(send_raises=OSError("pad unplugged"))
    result = _runner(sink=sink).run()
    _assert_failsafe(result, sink, STOP_SINK_ERROR)
    assert "pad unplugged" in result.error


def test_failsafe_on_abort_signal():
    sink, rec = FakeSink(), ListRecorder()
    result = _runner(abort=FakeAbort(set_after_checks=3), sink=sink, recorder=rec).run()
    _assert_failsafe(result, sink, STOP_ABORT, rec)
    assert len(sink.sent) == 3


def test_keyboard_interrupt_fails_safe():
    sink = FakeSink()
    result = _runner(policy=FakePolicy(raises=KeyboardInterrupt()), sink=sink).run()
    assert result.reason == STOP_INTERRUPT and result.is_failsafe_stop
    assert sink.failsafe_calls == 1 and sink.closed == 1


def test_failsafe_retried_once_then_reported():
    ok_second = FakeSink(failsafe_failures=1)
    result = _runner(policy=FakePolicy(raises=RuntimeError("x")), sink=ok_second).run()
    assert ok_second.failsafe_calls == 2 and result.failsafe_ok

    never = FakeSink(failsafe_failures=5)
    result = _runner(policy=FakePolicy(raises=RuntimeError("x")), sink=never).run()
    assert never.failsafe_calls == 2 and not result.failsafe_ok


def test_failsafe_is_latched_single_call():
    sink = FakeSink()
    r = _runner(gate=FakeGateEstimator(status="lost"), sink=sink)
    r.run()
    from drone_fly.harness.runner import RunResult

    r._failsafe(RunResult(reason="x"))
    assert sink.failsafe_calls == 1


def test_failsafe_with_gamepad_sink_neutral_rates_and_throttle_cut():
    pad = MockPad()
    sink = VgamepadActionSink(
        {ax: RateProfile() for ax in ("roll", "pitch", "yaw")},
        ThrottleMap(0.4, -0.4),
        pad_factory=lambda: pad,
    )
    policy = FakePolicy(action=(0.8, 1.0, -0.5, 0.3))
    r = _runner(sink=sink, policy=policy, abort=FakeAbort(set_after_checks=2))
    result = r.run()
    assert result.reason == STOP_ABORT
    flying = pad.history[2]  # after arm (neutral) + 2 sends
    assert flying["left_y"] > -1 and flying["right_x"] != 0
    # failsafe() then close() both set neutral; the last pad state is neutral.
    assert pad.sticks == {"left_x": 0.0, "left_y": -1.0, "right_x": 0.0, "right_y": 0.0}
    assert pad.history[-2] == pad.sticks
    assert sink.pad is None  # released on close


def test_discontinuity_resets_estimators_and_is_logged():
    state, gate, fs, rec = (
        FakeStateSource(),
        FakeGateEstimator(),
        FakeFreeSpaceEstimator(),
        ListRecorder(),
    )

    class Respawn(FakePolicy):
        def act(self, obs, extra):
            if len(self.calls) == 1:
                state.discontinuity_count += 1
            return super().act(obs, extra)

    _runner(
        state=state,
        gate=gate,
        fs=fs,
        policy=Respawn(),
        recorder=rec,
        settings=RunnerSettings(max_ticks=4),
    ).run()
    assert gate.resets == 1 and fs.resets == 1
    events = [t.events for t in rec.ticks]
    assert events[2] == ["discontinuity:estimators_reset"]
    assert sum("discontinuity:estimators_reset" in e for e in events) == 1


def test_frames_are_recorded_every_nth_tick():
    from drone_fly.harness.interfaces import FrameSource
    from drone_fly.harness.types import Frame

    class Frames(FrameSource):
        def __init__(self):
            self.n = 0

        def latest(self):
            self.n += 1
            return Frame(np.zeros((4, 4, 3), np.uint8), stamp=0.0, index=self.n)

    rec = ListRecorder()
    r = _runner(recorder=rec, settings=RunnerSettings(max_ticks=6, frame_every_n=3))
    r.harness.frame_source = Frames()
    r.run()
    assert [(t, save) for t, _, save in rec.frames] == [(0, True), (3, True)]
    assert "frame" in rec.ticks[0].ages_s

    rec0 = ListRecorder()
    r0 = _runner(recorder=rec0, settings=RunnerSettings(max_ticks=3, frame_every_n=0))
    r0.harness.frame_source = Frames()
    r0.run()
    assert [(t, save) for t, _, save in rec0.frames] == [(0, False), (1, False), (2, False)]


def test_runner_settings_from_config():
    from drone_fly.harness.config import load_config

    cfg = load_config("liftoff")
    s = RunnerSettings.from_config(cfg)
    assert s.rate_hz == 50 and s.stale_after_s == 0.2 and s.frame_every_n == 10


# ----------------------------------------------------------------------------- abort hotkey


def test_hotkey_abort_signal_with_injected_key_state():
    pressed = {"down": False}
    seen = []

    def key_state(vk):
        seen.append(vk)
        return pressed["down"]

    sig = HotkeyAbortSignal("F12", poll_s=0.005, key_state=key_state)
    sig.start()
    try:
        import time

        time.sleep(0.03)
        assert not sig.is_set()
        pressed["down"] = True
        deadline = time.perf_counter() + 2.0
        while not sig.is_set() and time.perf_counter() < deadline:
            time.sleep(0.005)
        assert sig.is_set()
    finally:
        sig.stop()
    assert set(seen) == {0x7B}  # VK_F12
    pressed["down"] = False
    assert sig.is_set()  # latched


def test_vk_codes_and_bad_key():
    from drone_fly.harness.errors import HarnessConfigError

    assert vk_for("F1") == 0x70 and vk_for("f12") == 0x7B and vk_for("ESCAPE") == 0x1B
    assert vk_for(0x41) == 0x41
    with pytest.raises(HarnessConfigError, match="abort.key"):
        vk_for("NOPE")
    sig = HotkeyAbortSignal.from_config({"key": "F9", "poll_s": 0.05}, None)
    assert sig.vk == 0x78 and sig.poll_s == 0.05


# ----------------------------------------------------------------------------- UC69 estimator stop


class _StoppingGate(FakeGateEstimator):
    """Raises an EstimatorStop on update number ``at`` (1-based)."""

    def __init__(self, at: int, stop: BaseException) -> None:
        super().__init__()
        self.at = at
        self.stop = stop

    def update(self, frame, state, now):
        if self.calls + 1 == self.at:
            self.calls += 1
            raise self.stop
        return super().update(frame, state, now)


def test_estimator_stop_records_reason_sends_nothing_that_tick_and_fails_safe():
    from drone_fly.harness.errors import EstimatorStop

    rec = ListRecorder()
    sink = FakeSink()
    r = _runner(
        gate=_StoppingGate(4, EstimatorStop("episode_end:crash_settled", "settled at 0.1 m")),
        sink=sink, recorder=rec, settings=RunnerSettings(max_ticks=20),
    )  # fmt: skip
    result = r.run()
    assert result.reason == "episode_end:crash_settled"
    assert result.error == "settled at 0.1 m"
    assert len(sink.sent) == 3  # ticks 0-2 sent; nothing on the stopping tick
    assert sink.failsafe_calls == 1  # the fail-safe still runs
    assert _stop_events(rec) == ["stop:episode_end:crash_settled"]
    last = rec.ticks[-1]
    assert last.tick == 3 and last.command is None and last.sink_output is None
    assert result.is_failsafe_stop  # an episode end is not a normal stop (CLI exit 5)


def test_estimator_stop_is_not_reported_as_a_perception_error():
    from drone_fly.harness.errors import EstimatorStop

    r = _runner(gate=_StoppingGate(1, EstimatorStop("episode_end:respawn")))
    result = r.run()
    assert result.reason == "episode_end:respawn" and result.reason != STOP_PERCEPTION_ERROR


def test_oracle_start_misaligned_before_the_first_send():
    """The real oracle refuses at tick 0: no command ever reaches the sink (never armed)."""
    from drone_fly.harness.errors import StartMisalignedError
    from drone_fly.harness.oracle import OracleGateSource
    from tests.oracle_support import make_course, state

    course = make_course()
    src = OracleGateSource(course, purpose="eval")
    sink = FakeSink()
    rec = ListRecorder()
    r = _runner(
        state=FakeStateSource(state(course, [2.5, 0.0, 0.03])), gate=src, sink=sink, recorder=rec,
        fs=None, settings=RunnerSettings(max_ticks=5),
    )  # fmt: skip
    result = r.run()
    assert result.reason == StartMisalignedError.REASON
    assert sink.sent == [] and sink.failsafe_calls == 1
    assert len(rec.ticks) == 1 and rec.ticks[0].command is None


def test_oracle_course_finished_stops_normally():
    from drone_fly.harness.oracle import OracleGateSource, StartTolerance
    from drone_fly.harness.runner import STOP_COURSE_FINISHED
    from tests.oracle_support import make_course, state

    course = make_course()
    src = OracleGateSource(
        course, purpose="eval", start_tolerance=StartTolerance(50.0, 50.0, 180.0, 100.0)
    )
    ss = FakeStateSource(state(course, [0.0, 0.0, 1.5]))
    path = iter(np.linspace([0.0, 0.0, 1.5], [11.0, 0.0, 1.5], 60))

    class Moving(FakeStateSource):
        def latest(self):
            return state(course, next(path), vel=[9.0, 0.0, 0.0])

    sink = FakeSink()
    r = _runner(state=Moving(ss.state), gate=src, sink=sink, fs=None,
                settings=RunnerSettings(max_ticks=100))  # fmt: skip
    result = r.run()
    assert result.reason == STOP_COURSE_FINISHED and not result.is_failsafe_stop
    assert src.progress.finished and len(sink.sent) == result.ticks
