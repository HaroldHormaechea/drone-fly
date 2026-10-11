"""UC-65 AC4 / AC10: the wire protocol, the latest-wins overlay state and the UDP receiver.

Pure and display-free: datagrams are built with the protocol encoders (or by the real
publisher), fed through ``OverlayState`` on a fake clock, and the receiver is driven with a fake
socket — plus one real localhost round trip publisher -> receiver.
"""

from __future__ import annotations

import json
import socket
import struct
import time

import numpy as np
import pytest

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.overlay import OverlayPublisher
from drone_fly.overlay.protocol import (
    BRAIN_HEADER,
    BRAIN_MAGIC,
    DEFAULT_PORT,
    SCHEMA_VERSION,
    BrainFrame,
    Packet,
    decode_datagram,
    encode_brain,
    encode_json,
)
from drone_fly.overlay.receiver import UdpReceiver
from drone_fly.overlay.state import (
    MODE_LIVE,
    MODE_STALE,
    MODE_STOPPED,
    MODE_WAITING,
    OverlayState,
)
from tests.overlay_support import FakeClock, FakeRecvSocket

RUN_A = "a" * 16
RUN_B = "b" * 16


def _tick(k: int, run: str = RUN_A, **extra) -> bytes:
    return encode_json("tick", run, {"tick": k, **extra})


def _brain(k: int, run: str = RUN_A, n: int = 8) -> bytes:
    return encode_brain(run, k, np.full(n, k % 256, np.uint8))


# ----------------------------------------------------------------------------- protocol


def test_text_round_trip_and_envelope_precedence():
    data = encode_json("tick", RUN_A, {"tick": 3, "kind": "spoof", "v": 99, "run_id": "x"})
    p = decode_datagram(data)
    assert isinstance(p, Packet)
    assert (p.kind, p.run_id, p.data["tick"], p.data["v"]) == ("tick", RUN_A, 3, SCHEMA_VERSION)
    for kind in ("meta", "status", "stop"):
        assert decode_datagram(encode_json(kind, RUN_A, {})).kind == kind


def test_brain_round_trip():
    codes = np.arange(300, dtype=np.int64) % 256
    p = decode_datagram(encode_brain(RUN_A, 42, codes, scale=0.5, offset=-3.0))
    assert p.kind == "brain" and p.run_id == RUN_A
    f = p.brain
    assert isinstance(f, BrainFrame) and f.tick == 42 and f.n == 300
    np.testing.assert_array_equal(f.codes, codes.astype(np.uint8))
    np.testing.assert_allclose(f.values(), codes * 0.5 - 3.0)
    assert f.values().dtype == np.float32


def test_encoders_reject_bad_input():
    with pytest.raises(ValueError, match="kind"):
        encode_json("brain", RUN_A, {})
    with pytest.raises(ValueError, match="run_id"):
        encode_brain("short", 0, np.zeros(4, np.uint8))


def _garbled() -> list[bytes]:
    good_brain = encode_brain(RUN_A, 1, np.zeros(10, np.uint8))
    bad_version = bytearray(good_brain)
    bad_version[4] = SCHEMA_VERSION + 1
    nan_scale = BRAIN_HEADER.pack(
        BRAIN_MAGIC, SCHEMA_VERSION, RUN_A.encode(), 1, 2, float("nan"), 0.0
    )
    return [
        b"",
        b"\x00\x01\x02",
        b"not json at all",
        b"\xff\xfe\xfa",  # not UTF-8
        b"[1, 2, 3]",  # JSON, not an object
        json.dumps({"v": SCHEMA_VERSION + 1, "kind": "tick", "run_id": RUN_A}).encode(),
        json.dumps({"v": SCHEMA_VERSION, "kind": "nope", "run_id": RUN_A}).encode(),
        json.dumps({"v": SCHEMA_VERSION, "kind": "tick", "run_id": 7}).encode(),
        good_brain[:-1],  # truncated
        good_brain + b"\x00",  # length mismatch
        good_brain[: BRAIN_HEADER.size - 1],  # header truncated
        bytes(bad_version),
        nan_scale + b"\x00\x00",
        BRAIN_MAGIC + struct.pack("<B", SCHEMA_VERSION),
    ]


@pytest.mark.parametrize("data", _garbled())
def test_garbled_datagrams_decode_to_none(data):
    assert decode_datagram(data) is None


def test_default_port():
    assert DEFAULT_PORT == 9100


# ----------------------------------------------------------------------------- state machine


def test_waiting_then_live_then_stale():
    clock = FakeClock()
    st = OverlayState(stale_s=1.0, clock=clock)
    s = st.snapshot()
    assert s.mode == MODE_WAITING and s.message == "waiting for harness" and s.tick is None
    st.ingest_bytes(_tick(0))
    s = st.snapshot()
    assert s.mode == MODE_LIVE and s.message == "live" and s.tick["tick"] == 0
    clock.advance(0.9)
    assert st.snapshot().mode == MODE_LIVE
    clock.advance(2.2)
    s = st.snapshot()
    assert s.mode == MODE_STALE and s.message == "no data for 3 s"
    assert s.tick["tick"] == 0  # the last data stays visible
    st.ingest_bytes(_tick(1))
    assert st.snapshot().mode == MODE_LIVE


def test_latest_wins_and_reordered_packets_are_ignored():
    st = OverlayState(clock=FakeClock())
    for k in (0, 1, 2, 5, 3, 4):
        st.ingest_bytes(_tick(k))
    assert st.snapshot().tick["tick"] == 5
    for k in (3, 6, 1):
        st.ingest_bytes(_brain(k))
    s = st.snapshot()
    assert s.brain.tick == 6 and s.brain_seq == 2  # the stale brain frames were dropped
    assert s.packets == 9


def test_meta_and_status_are_kept_and_status_message_shown_while_waiting():
    clock = FakeClock()
    st = OverlayState(clock=clock)
    st.ingest_bytes(encode_json("meta", RUN_A, {"extra_dim": 70}))
    st.ingest_bytes(encode_json("status", RUN_A, {"run_state": "waiting_for_focus"}))
    s = st.snapshot()
    assert s.meta["extra_dim"] == 70 and s.mode == MODE_LIVE
    assert s.message == "waiting for focus"
    clock.advance(0.01)
    st.ingest_bytes(_tick(0))
    assert st.snapshot().message == "live"  # a newer tick supersedes the status line


def test_last_stop_persists_until_a_new_run_id():
    clock = FakeClock()
    st = OverlayState(stale_s=1.0, clock=clock)
    st.ingest_bytes(_tick(9))
    st.ingest_bytes(encode_json("stop", RUN_A, {"reason": "focus_lost", "failsafe_ok": True}))
    s = st.snapshot()
    assert s.mode == MODE_STOPPED and "focus_lost" in s.message
    clock.advance(600.0)  # long after the stream went quiet: the stop reason stays visible
    s = st.snapshot()
    assert s.mode == MODE_STOPPED and s.last_stop["reason"] == "focus_lost"
    st.ingest_bytes(encode_json("stop", RUN_A, {"reason": "focus_lost"}))  # duplicate stop
    assert st.snapshot().mode == MODE_STOPPED
    # The next run starts: everything from the previous run is cleared.
    st.ingest_bytes(encode_json("status", RUN_B, {"run_state": "waiting_for_state"}))
    s = st.snapshot()
    assert s.mode == MODE_LIVE and s.run_id == RUN_B
    assert s.last_stop is None and s.tick is None and s.brain is None and s.meta is None


def test_a_new_run_resets_tick_ordering():
    st = OverlayState(clock=FakeClock())
    st.ingest_bytes(_tick(500, RUN_A))
    st.ingest_bytes(_tick(0, RUN_B))  # tick 0 of a new run is not "older"
    assert st.snapshot().tick["tick"] == 0
    st.ingest_bytes(_brain(300, RUN_A))
    st.ingest_bytes(_brain(0, RUN_B))
    assert st.snapshot().run_id == RUN_B and st.snapshot().brain.tick == 0


def test_garbled_datagrams_are_counted_and_ignored():
    st = OverlayState(clock=FakeClock())
    st.ingest_bytes(_tick(1))
    for data in _garbled():
        assert st.ingest_bytes(data) is None
    s = st.snapshot()
    assert s.garbled == len(_garbled()) and s.packets == 1 and s.tick["tick"] == 1


def test_tick_rate_ewma_tracks_50hz_and_ignores_long_gaps():
    clock = FakeClock()
    st = OverlayState(clock=clock)
    for k in range(200):
        st.ingest_bytes(_tick(k))
        clock.advance(0.02)
    assert st.snapshot().tick_rate_hz == pytest.approx(50.0, rel=1e-6)
    clock.advance(10.0)  # a pause does not crater the estimate
    st.ingest_bytes(_tick(201))
    assert st.snapshot().tick_rate_hz == pytest.approx(50.0, rel=1e-6)


def test_stale_s_must_be_positive():
    with pytest.raises(ValueError):
        OverlayState(stale_s=0)


# ----------------------------------------------------------------------------- receiver


def test_receiver_poll_drains_pending_datagrams_into_state():
    st = OverlayState(clock=FakeClock())
    sock = FakeRecvSocket([_tick(0), _tick(1), b"garbage", _brain(1), ConnectionResetError()])
    rx = UdpReceiver(9100, st, sock=sock)
    assert rx.poll() == 5  # the ICMP reset is swallowed and counts as "something pending"
    assert rx.received == 4
    s = st.snapshot()
    assert s.tick["tick"] == 1 and s.brain.tick == 1 and s.garbled == 1
    assert rx.poll() == 0


def test_receiver_thread_with_an_injected_socket_starts_and_stops():
    st = OverlayState()
    sock = FakeRecvSocket([_tick(k) for k in range(20)])
    rx = UdpReceiver(9100, st, sock=sock)
    rx.start()
    deadline = time.monotonic() + 5
    while st.snapshot().packets < 20 and time.monotonic() < deadline:
        time.sleep(0.01)
    rx.stop()
    assert st.snapshot().tick["tick"] == 19
    assert sock.closed == 1 and rx._thread is None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_bind_failure_names_the_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as holder:
        holder.bind(("127.0.0.1", 0))
        port = holder.getsockname()[1]
        rx = UdpReceiver(port, OverlayState())
        with pytest.raises(HarnessConfigError, match=f"port {port} in use"):
            rx.bind()


def test_real_localhost_round_trip_publisher_to_receiver():
    """The real publisher's non-blocking socket -> a bound receiver -> the overlay state."""
    port = _free_port()
    st = OverlayState()
    rx = UdpReceiver(port, st)
    rx.bind()
    pub = OverlayPublisher(
        port,
        3,
        run_info={"harness": "fake"},
        meta={"extra_dim": 70},
        quantize=lambda v: np.clip(np.round((v + 1) * 127.5), 0, 255).astype(np.uint8),
    )
    try:

        class Rec:
            def __init__(self, k):
                self.tick = k

            def as_dict(self):
                return {"tick": self.tick, "latency_s": {"total": 0.001}}

        brain = np.linspace(-1, 1, 25627).astype(np.float32)
        for k in range(6):
            pub.publish_tick(Rec(k), brain=brain if pub.wants_brain(k) else None)
        pub.publish_stop({"reason": "completed", "ticks": 6})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            rx.poll()
            if st.snapshot().last_stop is not None:
                break
            time.sleep(0.01)
    finally:
        pub.close()
        rx.stop()
    s = st.snapshot()
    assert pub.drops == 0
    assert s.run_id == pub.run_id and s.meta["extra_dim"] == 70
    assert s.tick["tick"] == 5 and s.brain.tick == 3 and s.brain.n == 25627
    np.testing.assert_allclose(s.brain.values(), brain, atol=1 / 255 + 1e-6)
    assert s.mode == MODE_STOPPED and s.last_stop["reason"] == "completed"
