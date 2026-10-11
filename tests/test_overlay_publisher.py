"""UC-65 AC2 / AC3: the runner-side overlay publisher and the loop-budget guard.

A fake socket captures every datagram, which is decoded with the overlay's own protocol module,
so these tests pin the wire payload exactly as the overlay receives it: the tick schema (every
AC2 field), the brain decimation and quantisation (recorder scheme and the opt-in ``frame_peak``),
drop-on-error, the stop packet, and that a publisher failure can never stop a run. The SimClock
tests show the publish cost lands in ``latency_s["overlay"]`` and stays out of ``inference``.
"""

from __future__ import annotations

import logging
import socket
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from drone_fly.harness import overlay as overlay_mod
from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.interfaces import FocusProbe, Harness
from drone_fly.harness.overlay import (
    MIN_FRAME_PEAK,
    SEND_BUFFER_BYTES,
    OverlayPublisher,
    build_meta,
    contract_constants,
    quantize_frame_peak,
    sample_extras,
    stop_payload,
    tick_payload,
)
from drone_fly.harness.recording import summarize_ticks
from drone_fly.harness.runner import (
    STOP_COMPLETED,
    STOP_FOCUS_LOST,
    HarnessRunner,
    RunnerSettings,
    SimClock,
)
from drone_fly.harness.types import TelemetrySample
from drone_fly.overlay.protocol import (
    BRAIN_HEADER,
    DEFAULT_OFFSET,
    DEFAULT_SCALE,
    MAX_DATAGRAM,
    SCHEMA_VERSION,
)
from tests.harness_support import (
    FakeFreeSpaceEstimator,
    FakeGateEstimator,
    FakePolicy,
    FakeSink,
    FakeStateSource,
    ListRecorder,
    fake_contract,
)
from tests.overlay_support import FakeClock, FakeSocket

RUN_INFO = {
    "harness": "fake",
    "checkpoint": "model_readout.pt",
    "sha8": "deadbeef",
    "extra_dim": 70,
}


def _recorder_quantize(values: np.ndarray) -> np.ndarray:
    """The recorder's scheme without importing torch: round((a + 1) / (2 / 255))."""
    return np.clip(np.round((np.asarray(values) + 1.0) / DEFAULT_SCALE), 0, 255).astype(np.uint8)


def _publisher(sock: FakeSocket | None = None, **kw: Any) -> OverlayPublisher:
    kw.setdefault("quantize", _recorder_quantize)
    kw.setdefault("run_info", RUN_INFO)
    return OverlayPublisher(9100, kw.pop("brain_every_n", 3), sock=sock or FakeSocket(), **kw)


class SampleStateSource(FakeStateSource):
    """FakeStateSource whose ``drain_raw`` yields one telemetry sample per tick."""

    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def drain_raw(self) -> list[TelemetrySample]:
        self.n += 1
        return [
            TelemetrySample(
                t=float(self.n),
                position=np.zeros(3),
                quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                velocity=np.zeros(3),
                gyro=np.zeros(3),
                inputs=np.array([0.5, 0.0, -0.1, 0.1]),
                battery=np.array([0.87, 16.4]),
                rpm=np.array([1000.0, 1001.0, 1002.0, 1003.0]),
            )
        ]


class ConstFocus(FocusProbe):
    def __init__(self, value: bool = True) -> None:
        self.value = value
        self.calls = 0
        self.last_foreground = (1234, "liftoff.exe", "Liftoff")

    def has_focus(self) -> bool:
        self.calls += 1
        return self.value


class BrainPolicy(FakePolicy):
    """Returns a known brain vector so the round-trip can be checked."""

    def __init__(self, brain: np.ndarray, **kw: Any) -> None:
        super().__init__(**kw)
        self.brain = brain

    def act_with_brain(self, obs, extra):
        return self.act(obs, extra), self.brain


def _runner(
    publisher: Any,
    *,
    extra_dim: int = 70,
    policy: Any = None,
    clock: SimClock | None = None,
    state: Any = None,
    focus: Any = None,
    recorder: Any = None,
    max_ticks: int = 10,
) -> HarnessRunner:
    clock = clock or SimClock()
    contract = fake_contract()
    h = Harness(
        name="fake",
        state_source=state or FakeStateSource(),
        gate_estimator=FakeGateEstimator(),
        action_sink=FakeSink(),
        free_space_estimator=FakeFreeSpaceEstimator(),
        focus_probe=focus,
        clock=clock,
    )
    return HarnessRunner(
        h,
        policy or FakePolicy(extra_dim=extra_dim, action=(0.2, 0.5, -0.5, 0.1)),
        ContractObservationAssembler(contract),
        contract,
        RunnerSettings(max_ticks=max_ticks),
        recorder=recorder,
        publisher=publisher,
    )


# ----------------------------------------------------------------------------- tick schema


def test_tick_payload_carries_every_ac2_field():
    sock = FakeSocket()
    brain = np.linspace(-1, 1, 64).astype(np.float32)
    pub = _publisher(sock, meta=build_meta({"runner": {"rate_hz": 50}}, extra_dim=70))
    r = _runner(
        pub,
        policy=BrainPolicy(brain, extra_dim=70, action=(0.2, 0.5, -0.5, 0.1)),
        state=SampleStateSource(),
        focus=ConstFocus(True),
        max_ticks=4,
    )
    result = r.run()
    assert result.reason == STOP_COMPLETED
    ticks = sock.of_kind("tick")
    assert [p.data["tick"] for p in ticks] == [0, 1, 2, 3]
    d = ticks[1].data
    assert d["v"] == SCHEMA_VERSION and d["kind"] == "tick" and d["run_id"] == pub.run_id
    # tick, wall/sim time, run state, harness / checkpoint / sha / extra_dim
    for key in ("tick", "wall_time", "sim_time"):
        assert key in d
    assert d["run_state"] == "running"
    assert d["run_info"] == RUN_INFO
    # obs(12), extra(70), raw action(4), collective, body rates, sink output
    assert len(d["obs"]) == 12 and len(d["extra"]) == 70
    cmd = d["command"]
    assert len(cmd["raw"]) == 4 and len(cmd["rates"]) == 3 and 0 <= cmd["collective"] <= 1
    assert d["sink_output"] == {"ok": True}
    # gate estimate: status, index, lap, target, info, frame stamp
    gate = d["gate"]
    for key in ("status", "gate_index", "lap", "target", "info", "stamp"):
        assert key in gate
    # free space: obstacle_vision 8, raycast 40, in-FOV mask, coverage
    fs = d["free_space"]
    assert len(fs["obstacle_vision"]) == 8 and len(fs["raycast"]) == 40
    assert len(fs["in_fov_mask"]) == 40 and fs["coverage"] == pytest.approx(1.0)
    # latency / perception / ages, missed deadlines, focus, events
    assert {"state", "perception", "inference", "actuation", "total"} <= set(d["latency_s"])
    assert "overlay" in d["latency_s"]  # the previous tick's publish cost
    assert set(d["perception_s"]) == {"gate", "free_space"}
    assert {"telemetry", "gate", "free_space"} <= set(d["ages_s"])
    assert d["missed_deadlines"] == 0 and d["drops"] == 0
    assert d["focus"] is True and d["focus_window"] == "liftoff.exe"
    assert d["events"] == []
    # telemetry extras not in TickRecord.state come from the tick's TelemetrySample
    assert d["sample"] == {
        "battery": pytest.approx([0.87, 16.4]),
        "inputs": pytest.approx([0.5, 0.0, -0.1, 0.1]),
        "rpm": pytest.approx([1000.0, 1001.0, 1002.0, 1003.0]),
    }


def test_first_tick_has_no_overlay_cost_then_previous_ticks_cost_is_carried():
    sock = FakeSocket()
    r = _runner(_publisher(sock), max_ticks=3)
    r.run()
    ticks = sock.of_kind("tick")
    assert "overlay" not in ticks[0].data["latency_s"]
    assert all("overlay" in t.data["latency_s"] for t in ticks[1:])


def test_meta_layout_limits_and_constants():
    cfg = {
        "harness": "liftoff",
        "runner": {"rate_hz": 50, "stale_after_s": 0.25, "max_estimate_age_s": 0.3},
        "gates": {"lost_timeout_s": 1.0},
        "camera": {"monitor": 2, "region": [0, 0, 1280, 720], "resolution": [1280, 720]},
        "scaling": {"world_scale": 2.0},
    }
    contract = SimpleNamespace(
        MAX_BODY_RATE=4.0, MAX_THRUST=0.785, HOVER_THR=0.4, RAY_RANGE=4.0, CONE=np.ones((40, 3))
    )
    meta = build_meta(
        cfg,
        extra_dim=70,
        constants=contract_constants(contract),
        brain_info={"n_neurons": 25627, "tap_index": [1, 2]},
        k1_path="k1",
    )
    assert meta["extra_layout"] == [
        ["obs", 0, 12],
        ["curr_gate", 12, 3],
        ["next_lookahead", 15, 7],
        ["obstacle_vision", 22, 8],
        ["raycast", 30, 40],
    ]
    assert meta["age_limits"] == {"telemetry": 0.25, "free_space": 0.3, "gate": 1.0}
    assert meta["camera"] == {"monitor": 2, "region": [0, 0, 1280, 720], "resolution": [1280, 720]}
    assert meta["world_scale"] == 2.0 and meta["rate_hz"] == 50 and meta["k1_path"] == "k1"
    assert meta["brain"]["n_neurons"] == 25627
    assert meta["max_body_rate"] == 4.0 and meta["hover_thr"] == 0.4
    assert np.asarray(meta["cone_xy"]).shape == (40, 2)
    # block boundaries follow the contract widths for the smaller models
    assert build_meta({}, extra_dim=30)["extra_layout"][-1] == ["obstacle_vision", 22, 8]
    assert build_meta({}, extra_dim=12)["extra_layout"] == [["obs", 0, 12]]
    assert contract_constants(None) == {}


def test_meta_sent_on_first_tick_then_once_per_second():
    sock, clock = FakeSocket(), FakeClock()
    pub = _publisher(sock, clock=clock)
    rec = SimpleNamespace(tick=0, as_dict=lambda: {"tick": 0})
    for _ in range(10):
        pub.publish_tick(rec)
        clock.advance(0.25)
    kinds = sock.kinds()
    assert kinds[0] == "meta"
    assert kinds.count("meta") == 3  # t = 0, 1.0, 2.0
    assert kinds.count("tick") == 10
    meta = sock.of_kind("meta")[0].data
    assert meta["run_info"] == RUN_INFO


def test_status_is_rate_limited_to_4hz():
    sock, clock = FakeSocket(), FakeClock()
    pub = _publisher(sock, clock=clock)
    for _ in range(20):  # 1 s at 20 Hz
        pub.publish_status("waiting_for_focus")
        clock.advance(0.05)
    statuses = sock.of_kind("status")
    assert len(statuses) == 4
    assert statuses[0].data["run_state"] == "waiting_for_focus"


def test_runner_wait_loops_publish_status():
    class LateState(FakeStateSource):
        def __init__(self):
            super().__init__()
            self.polls = 0

        def latest(self):
            self.polls += 1
            return None if self.polls < 40 else self.state

    class LateFocus(ConstFocus):
        def has_focus(self):
            self.calls += 1
            return self.calls > 60

    sock, clock = FakeSocket(), SimClock()
    pub = _publisher(sock, clock=clock.now)  # the publisher's rate limit on the runner's clock
    r = _runner(pub, clock=clock, state=LateState(), focus=LateFocus(), max_ticks=2)
    assert r.run().reason == STOP_COMPLETED
    states = [p.data["run_state"] for p in sock.of_kind("status")]
    assert "waiting_for_state" in states and "waiting_for_focus" in states
    assert states.index("waiting_for_state") < states.index("waiting_for_focus")


# ----------------------------------------------------------------------------- brain frames


def test_brain_datagram_decimation_every_n_ticks():
    sock = FakeSocket()
    brain = np.zeros(64, np.float32)
    r = _runner(
        _publisher(sock, brain_every_n=3), policy=BrainPolicy(brain, extra_dim=70), max_ticks=10
    )
    r.run()
    assert [p.brain.tick for p in sock.of_kind("brain")] == [0, 3, 6, 9]
    assert len(sock.of_kind("tick")) == 10


@pytest.mark.parametrize("n", [1, 2, 5])
def test_wants_brain_is_tick_modulo_n(n):
    pub = _publisher(brain_every_n=n)
    assert [k for k in range(12) if pub.wants_brain(k)] == list(range(0, 12, n))


@pytest.mark.parametrize("bad", [0, -1, True])
def test_brain_every_n_must_be_positive(bad):
    with pytest.raises(ValueError, match="brain_every_n"):
        _publisher(brain_every_n=bad)


@pytest.mark.parametrize("bad", ["peak", "", None])
def test_brain_scale_is_validated(bad):
    with pytest.raises(ValueError, match="brain_scale"):
        _publisher(brain_scale=bad)


def test_brain_round_trip_recorder_scheme_within_one_code():
    sock = FakeSocket()
    pub = _publisher(sock)
    values = np.random.default_rng(1).uniform(-1, 1, 25627).astype(np.float32)
    pub.publish_tick(SimpleNamespace(tick=6, as_dict=lambda: {"tick": 6}), brain=values)
    (frame,) = [p.brain for p in sock.of_kind("brain")]
    assert frame.tick == 6 and frame.n == 25627
    assert frame.scale == pytest.approx(DEFAULT_SCALE) and frame.offset == pytest.approx(-1.0)
    assert np.max(np.abs(frame.values() - values)) <= DEFAULT_SCALE / 2 + 1e-6
    # ~25.6 KB for K1, one datagram
    ((data, _),) = [(d, a) for d, a in sock.sent if d[:4] == b"DFOB"]
    assert len(data) == BRAIN_HEADER.size + 25627 < MAX_DATAGRAM


def test_brain_round_trip_with_the_real_recorder_quantiser():
    """No injected quantiser: the publisher resolves ``quantize_activation`` from the recorder."""
    pytest.importorskip("torch")
    from drone_fly.record.recorder import ACTIVATION_OFFSET, ACTIVATION_SCALE, quantize_activation

    sock = FakeSocket()
    pub = OverlayPublisher(9100, 3, sock=sock)
    assert pub._quantize is quantize_activation
    values = np.linspace(-1, 1, 1001).astype(np.float32)
    pub.publish_tick(SimpleNamespace(tick=0, as_dict=lambda: {"tick": 0}), brain=values)
    frame = sock.of_kind("brain")[0].brain
    assert frame.scale == pytest.approx(ACTIVATION_SCALE)
    assert frame.offset == pytest.approx(ACTIVATION_OFFSET)
    np.testing.assert_array_equal(frame.codes, quantize_activation(values))
    assert np.max(np.abs(frame.values() - values)) <= ACTIVATION_SCALE / 2 + 1e-6


def test_torch_tensor_brain_is_copied_to_host():
    torch = pytest.importorskip("torch")
    sock = FakeSocket()
    pub = _publisher(sock)
    t = torch.linspace(-0.5, 0.5, 100)
    pub.publish_tick(SimpleNamespace(tick=0, as_dict=lambda: {"tick": 0}), brain=t)
    frame = sock.of_kind("brain")[0].brain
    np.testing.assert_allclose(frame.values(), t.numpy(), atol=DEFAULT_SCALE / 2 + 1e-6)


def test_quantize_frame_peak_spends_all_codes_on_the_frame_range():
    a = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    codes, scale, offset = quantize_frame_peak(a)
    assert codes.dtype == np.uint8 and codes[0] == 0 and codes[-1] == 255
    assert scale == pytest.approx(2 * 0.02 / 255) and offset == pytest.approx(-0.02)
    np.testing.assert_allclose(codes * scale + offset, a, atol=scale / 2 + 1e-12)


def test_quantize_frame_peak_all_zero_frame_has_finite_floored_scale():
    codes, scale, offset = quantize_frame_peak(np.zeros(100))
    assert np.isfinite(scale) and scale == pytest.approx(2 * MIN_FRAME_PEAK / 255)
    assert offset == pytest.approx(-MIN_FRAME_PEAK)
    assert np.all(np.abs(codes * scale + offset) <= scale)  # every neuron decodes to ~0
    codes, scale, _ = quantize_frame_peak(np.array([]))
    assert codes.size == 0 and np.isfinite(scale)


def test_quantize_frame_peak_maps_nan_and_inf_to_zero():
    a = np.array([np.nan, np.inf, -np.inf, 0.5, -0.25])
    codes, scale, offset = quantize_frame_peak(a)
    assert np.isfinite(scale) and scale == pytest.approx(2 * 0.5 / 255)  # peak ignores inf
    decoded = codes * scale + offset
    np.testing.assert_allclose(decoded[:3], 0.0, atol=scale)
    np.testing.assert_allclose(decoded[3:], [0.5, -0.25], atol=scale / 2 + 1e-12)


def test_recorder_vs_frame_peak_round_trip_on_real_scale_activations():
    """K1 activations sit far inside [-1, 1]: the recorder scheme collapses them onto a couple of
    codes, frame_peak resolves them; both decode through the same datagram scale / offset."""
    rng = np.random.default_rng(65)
    values = (rng.standard_normal(25627) * 0.004).astype(np.float32)
    out = {}
    for mode in ("recorder", "frame_peak"):
        sock = FakeSocket()
        pub = _publisher(sock, brain_scale=mode)
        pub.publish_tick(SimpleNamespace(tick=3, as_dict=lambda: {"tick": 3}), brain=values)
        out[mode] = sock.of_kind("brain")[0].brain
    rec, peak = out["recorder"], out["frame_peak"]
    assert len(np.unique(rec.codes)) <= 8
    assert len(np.unique(peak.codes)) > 100
    rec_err = np.max(np.abs(rec.values() - values))
    peak_err = np.max(np.abs(peak.values() - values))
    assert rec_err <= DEFAULT_SCALE / 2 + 1e-6
    assert peak_err <= peak.scale / 2 + 1e-6 and peak_err < rec_err / 10
    assert peak.offset == pytest.approx(-np.max(np.abs(values)), rel=1e-6)


def test_frame_peak_with_nan_brain_still_sends_a_finite_frame():
    sock = FakeSocket()
    pub = _publisher(sock, brain_scale="frame_peak")
    pub.publish_tick(
        SimpleNamespace(tick=0, as_dict=lambda: {"tick": 0}), brain=np.full(10, np.nan)
    )
    frame = sock.of_kind("brain")[0].brain
    assert np.all(np.isfinite(frame.values())) and pub.drops == 0


# ----------------------------------------------------------------------------- drop on error


@pytest.mark.parametrize("exc", [BlockingIOError(), OSError("network unreachable")])
def test_send_errors_drop_count_and_never_raise(exc, caplog):
    sock = FakeSocket(raises=exc)
    pub = _publisher(sock)
    rec = SimpleNamespace(tick=0, as_dict=lambda: {"tick": 0})
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            pub.publish_tick(rec, brain=np.zeros(8))  # meta + tick + brain on the first one
            pub.publish_status("waiting_for_focus")
        pub.publish_stop({"reason": "completed"})
    assert sock.sent == [] and pub.drops == sock.attempts > 0 and pub.sent == 0
    # logging is rate-limited (the fake publisher clock is the real monotonic clock here)
    assert sum("dropped" in r.getMessage() for r in caplog.records) <= 2


def test_drops_are_reported_in_later_ticks():
    sock = FakeSocket(raises=[None, BlockingIOError(), None])  # meta ok, tick 0 dropped
    pub = _publisher(sock)
    pub.publish_tick(SimpleNamespace(tick=0, as_dict=lambda: {"tick": 0}))
    pub.publish_tick(SimpleNamespace(tick=1, as_dict=lambda: {"tick": 1}))
    ticks = sock.of_kind("tick")
    assert [t.data["tick"] for t in ticks] == [1] and ticks[0].data["drops"] == 1


def test_serialisation_error_drops_the_datagram():
    sock = FakeSocket()
    pub = _publisher(sock)

    def boom():
        raise TypeError("not JSON")

    pub.publish_tick(SimpleNamespace(tick=0, as_dict=boom), brain=np.zeros(4))
    assert sock.kinds() == ["meta", "brain"] and pub.drops == 1
    pub.publish_tick(SimpleNamespace(tick=1, as_dict=lambda: {"tick": 1, "x": object()}))
    assert pub.drops == 2


def test_oversized_and_unquantisable_brains_are_dropped():
    sock = FakeSocket()
    pub = _publisher(sock)
    rec = SimpleNamespace(tick=0, as_dict=lambda: {"tick": 0})
    pub.publish_tick(rec, brain=np.zeros(MAX_DATAGRAM + 1))
    assert "brain" not in sock.kinds() and pub.drops == 1

    def bad_quantize(_v):
        raise RuntimeError("quantiser broke")

    pub2 = _publisher(FakeSocket(), quantize=bad_quantize)
    pub2.publish_tick(rec, brain=np.zeros(8))
    assert pub2.drops == 1 and pub2.sock.kinds() == ["meta", "tick"]


def test_closed_publisher_sends_nothing():
    sock = FakeSocket()
    pub = _publisher(sock)
    pub.close()
    pub.close()  # idempotent
    pub.publish_tick(SimpleNamespace(tick=0, as_dict=lambda: {"tick": 0}), brain=np.zeros(4))
    assert sock.sent == [] and sock.closed == 1


def test_publisher_send_failures_never_stop_a_run():
    sock = FakeSocket(raises=OSError("boom"))
    pub = _publisher(sock)
    rec = ListRecorder()
    r = _runner(pub, policy=BrainPolicy(np.zeros(8), extra_dim=70), recorder=rec)
    result = r.run()
    assert result.reason == STOP_COMPLETED and result.ticks == 10
    assert pub.drops > 10 and len(rec.ticks) == 10


class RaisingPublisher:
    """Every method raises: the runner must log and carry on."""

    def __init__(self) -> None:
        self.calls = 0

    def wants_brain(self, tick):
        return False

    def _boom(self, *a, **k):
        self.calls += 1
        raise RuntimeError("publisher exploded")

    publish_tick = publish_status = publish_stop = _boom


def test_a_raising_publisher_cannot_stop_a_run(caplog):
    pub = RaisingPublisher()
    rec = ListRecorder()
    with caplog.at_level(logging.ERROR):
        result = _runner(pub, recorder=rec).run()
    assert result.reason == STOP_COMPLETED and result.ticks == 10
    assert pub.calls == 11  # 10 ticks + the stop
    assert all("overlay" in t.latency_s for t in rec.ticks)
    assert any("overlay publish failed" in r.getMessage() for r in caplog.records)


def test_a_raising_brain_policy_is_still_a_policy_error_not_an_overlay_one():
    class Broken(FakePolicy):
        def act_with_brain(self, obs, extra):
            raise RuntimeError("model broke")

    r = _runner(_publisher(), policy=Broken(extra_dim=70))
    assert r.run().reason == "policy_error"


# ----------------------------------------------------------------------------- stop packet


def test_stop_packet_sent_twice_with_reason_and_failsafe_outcome():
    sock = FakeSocket()
    focus = ConstFocus(True)
    r = _runner(_publisher(sock), focus=focus, max_ticks=50)
    focus.value = True
    orig = focus.has_focus

    def flaky():
        return orig() and focus.calls < 5

    focus.has_focus = flaky  # type: ignore[method-assign]
    result = r.run()
    assert result.reason == STOP_FOCUS_LOST
    stops = sock.of_kind("stop")
    assert len(stops) == 2 and stops[0].data == stops[1].data
    s = stops[0].data
    assert s["reason"] == STOP_FOCUS_LOST and s["error"] == "target window lost focus"
    assert s["run_state"] == "stopped" and s["run_info"] == RUN_INFO
    assert s["ticks"] == result.ticks and s["failsafe_ok"] is True
    assert "failsafe_output" in s and "failsafe" in s
    assert sock.kinds()[-2:] == ["stop", "stop"]


def test_stop_payload_from_a_run_folder_meta_mapping():
    s = stop_payload({"stop_reason": "gate_lost", "stop_error": "x", "ticks": 3})
    assert s["reason"] == "gate_lost" and s["error"] == "x" and s["ticks"] == 3


# ----------------------------------------------------------------------------- socket setup


def test_real_socket_is_non_blocking_with_a_1mb_send_buffer(monkeypatch):
    made: list[Any] = []

    class SpySock:
        def __init__(self, family, kind):
            self.family, self.kind = family, kind
            self.blocking = None
            self.opts: dict[Any, Any] = {}
            made.append(self)

        def setblocking(self, flag):
            self.blocking = flag

        def setsockopt(self, level, opt, value):
            self.opts[(level, opt)] = value

        def close(self):
            pass

    monkeypatch.setattr(overlay_mod.socket, "socket", SpySock)
    pub = OverlayPublisher(9123, 3, quantize=_recorder_quantize)
    (s,) = made
    assert (s.family, s.kind) == (socket.AF_INET, socket.SOCK_DGRAM)
    assert s.blocking is False
    assert s.opts[(socket.SOL_SOCKET, socket.SO_SNDBUF)] == SEND_BUFFER_BYTES == 1 << 20
    assert pub.address == ("127.0.0.1", 9123)


def test_send_buffer_failure_is_logged_not_raised(monkeypatch, caplog):
    class NoBufSock:
        def __init__(self, *_a):
            pass

        def setblocking(self, _f):
            pass

        def setsockopt(self, *_a):
            raise OSError("not allowed")

    monkeypatch.setattr(overlay_mod.socket, "socket", NoBufSock)
    with caplog.at_level(logging.WARNING):
        OverlayPublisher(9123, 3, quantize=_recorder_quantize)
    assert any("SO_SNDBUF" in r.getMessage() for r in caplog.records)


def test_nobody_listening_on_a_real_localhost_socket_is_fine():
    """UDP to a closed localhost port: datagrams vanish, nothing raises, the run is unaffected."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    pub = OverlayPublisher(port, 3, quantize=_recorder_quantize, run_info=RUN_INFO)
    try:
        result = _runner(pub, policy=BrainPolicy(np.zeros(64), extra_dim=70)).run()
    finally:
        pub.close()
    assert result.reason == STOP_COMPLETED and result.ticks == 10


# ----------------------------------------------------------------------------- AC3 loop budget


class CostlyPublisher:
    """Advances the SimClock inside publish_tick (the measured overlay cost)."""

    def __init__(self, clock: SimClock, cost_s: float) -> None:
        self.clock, self.cost_s = clock, cost_s
        self.prev: list[float | None] = []

    def wants_brain(self, tick):
        return False

    def publish_tick(self, record, *, overlay_prev_s=None, **_kw):
        self.prev.append(overlay_prev_s)
        self.clock.advance(self.cost_s)

    def publish_status(self, _s):
        pass

    def publish_stop(self, _r):
        pass


def test_overlay_cost_is_recorded_and_excluded_from_inference_and_total():
    clock = SimClock()
    pub = CostlyPublisher(clock, 0.004)
    rec = ListRecorder()
    policy = FakePolicy(extra_dim=70, clock=clock, cost_s=0.003)
    r = _runner(pub, clock=clock, policy=policy, recorder=rec)
    result = r.run()
    assert result.missed_deadlines == 0
    for t in rec.ticks:
        assert t.latency_s["overlay"] == pytest.approx(0.004)
        assert t.latency_s["inference"] == pytest.approx(0.003)
        assert t.latency_s["total"] == pytest.approx(0.003)  # state -> actuation only
    assert pub.prev[0] is None and pub.prev[1:] == [pytest.approx(0.004)] * 9


def test_overlay_cost_still_counts_toward_the_missed_deadline_check():
    clock = SimClock()
    pub = CostlyPublisher(clock, 0.025)  # > one 20 ms period
    rec = ListRecorder()
    result = _runner(pub, clock=clock, recorder=rec, max_ticks=6).run()
    assert result.missed_deadlines == result.ticks > 0
    assert all(t.missed_deadline for t in rec.ticks)


def test_no_publisher_keeps_the_pre_uc65_latency_keys():
    rec = ListRecorder()
    _runner(None, recorder=rec).run()
    assert all("overlay" not in t.latency_s for t in rec.ticks)


def test_summarize_ticks_reports_the_overlay_row_only_when_ticks_carry_it():
    clock = SimClock()
    rec = ListRecorder()
    _runner(CostlyPublisher(clock, 0.0008), clock=clock, recorder=rec).run()
    with_overlay = summarize_ticks([t.as_dict() for t in rec.ticks], 50.0)
    assert with_overlay["latency_ms"]["overlay"]["p99"] == pytest.approx(0.8)
    assert with_overlay["latency_ms"]["policy"]["p99"] == pytest.approx(0.0)

    plain = ListRecorder()
    _runner(None, recorder=plain).run()
    without = summarize_ticks([t.as_dict() for t in plain.ticks], 50.0)
    assert "overlay" not in without["latency_ms"]
    assert {"state", "policy", "actuation", "total"} <= set(without["latency_ms"])


# ----------------------------------------------------------------------------- telemetry extras


def test_samples_are_drained_for_the_publisher_without_a_recorder():
    sock = FakeSocket()
    state = SampleStateSource()
    _runner(_publisher(sock), state=state, recorder=None, max_ticks=3).run()
    assert state.n == 3
    assert sock.of_kind("tick")[-1].data["sample"]["battery"] == pytest.approx([0.87, 16.4])


def test_samples_still_reach_the_recorder_when_both_are_attached():
    state, rec = SampleStateSource(), ListRecorder()
    _runner(_publisher(), state=state, recorder=rec, max_ticks=3).run()
    assert len(rec.telemetry) == 3


def test_no_sample_and_missing_optional_blocks():
    assert sample_extras(None) is None
    s = sample_extras(SimpleNamespace(battery=None, inputs=[1, 2, 3, 4]))
    assert s == {"battery": None, "inputs": [1.0, 2.0, 3.0, 4.0], "rpm": None}
    p = tick_payload({"tick": 1, "latency_s": {"overlay": 0.5}}, overlay_prev_s=0.9)
    assert p["latency_s"]["overlay"] == 0.5  # a recorded value is never overwritten
    assert p["sample"] is None and p["focus"] is None and p["run_state"] == "running"


def test_default_quantiser_constants_match_the_recorder():
    pytest.importorskip("torch")
    from drone_fly.record.recorder import ACTIVATION_OFFSET, ACTIVATION_SCALE

    assert DEFAULT_SCALE == pytest.approx(ACTIVATION_SCALE)
    assert DEFAULT_OFFSET == pytest.approx(ACTIVATION_OFFSET)
