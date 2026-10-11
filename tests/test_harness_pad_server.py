"""Long-lived virtual pad (``drone-fly harness pad``): server state handling, watchdog, client,
ping / auto-launch, and the sink's ``controls.pad: server`` mode. ``vgamepad`` is never imported:
the server gets a mock pad and every socket is a localhost one or a fake."""

from __future__ import annotations

import json
import socket
import threading
from types import SimpleNamespace

import pytest

from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.liftoff import pad_server
from drone_fly.harness.liftoff.gamepad import VgamepadActionSink
from drone_fly.harness.liftoff.pad_server import (
    NEUTRAL,
    PadServer,
    RemotePad,
    ensure_pad_server,
    ping_pad_server,
)
from tests.harness_support import MockPad


class FakeSock:
    def __init__(self):
        self.sent = []

    def sendto(self, data, addr):
        self.sent.append((json.loads(data.decode()), addr))

    def close(self):
        pass


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _server(**kw):
    pad, sock, clock = MockPad(), FakeSock(), Clock()
    srv = PadServer(pad, sock, lambda name: f"btn-{name}", clock=clock, **kw)
    return srv, pad, sock, clock


def _state_msg(state, buttons=()):
    return json.dumps({"v": 1, "state": state, "buttons": list(buttons)}).encode()


def test_server_starts_neutral_with_the_throttle_cut():
    _, pad, _, _ = _server()
    assert pad.sticks == NEUTRAL and pad.buttons == set()


def test_server_applies_states_and_button_changes():
    srv, pad, _, _ = _server()
    srv.handle(_state_msg({**NEUTRAL, "left_y": 0.4, "right_x": 2.0}, ["A"]), None)
    assert pad.sticks["left_y"] == 0.4 and pad.sticks["right_x"] == 1.0  # clipped
    assert pad.buttons == {"btn-A"}
    srv.handle(_state_msg(NEUTRAL, ["B"]), None)
    assert pad.buttons == {"btn-B"}
    assert [e[0] for e in pad.button_log] == ["press", "release", "press"]


def test_server_answers_pings_and_ignores_garbage():
    srv, pad, sock, _ = _server()
    srv.handle(json.dumps({"v": 1, "ping": 7}).encode(), ("127.0.0.1", 5555))
    assert sock.sent == [({"v": 1, "pong": 7}, ("127.0.0.1", 5555))]
    updates = pad.updates
    for junk in (b"\xff", b"[]", b'{"v": 2, "state": {}}', b'{"v": 1}', b'{"v": 1, "state": 3}'):
        srv.handle(junk, None)
    assert pad.updates == updates and srv.states == 0


def test_watchdog_goes_neutral_when_the_client_goes_silent():
    srv, pad, _, clock = _server(timeout_s=0.5)
    assert srv.check_watchdog() is False  # idle server: nothing to do
    srv.handle(_state_msg({**NEUTRAL, "left_y": 0.6}, ["A"]), None)
    clock.t = 0.4
    assert srv.check_watchdog() is False
    clock.t = 0.6
    assert srv.check_watchdog() is True
    assert pad.sticks == NEUTRAL and pad.buttons == set() and srv.watchdog_trips == 1
    clock.t = 5.0
    assert srv.check_watchdog() is False  # trips once per silence


def test_remote_pad_sends_the_full_state_on_update():
    sock = FakeSock()
    pad = RemotePad(9300, sock=sock)
    pad.left_joystick_float(x_value_float=0.1, y_value_float=0.2)
    pad.right_joystick_float(x_value_float=-0.3, y_value_float=0.4)
    pad.press_button(button="A")
    pad.update()
    pad.release_button(button="A")
    pad.update()
    (m1, addr), (m2, _) = sock.sent
    assert addr == ("127.0.0.1", 9300)
    assert m1 == {
        "v": 1,
        "state": {"left_x": 0.1, "left_y": 0.2, "right_x": -0.3, "right_y": 0.4},
        "buttons": ["A"],
    }
    assert m2["buttons"] == []


def _free_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_real_localhost_round_trip_ping_and_state():
    port = _free_udp_port()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(0.02)
    pad = MockPad()
    srv = PadServer(pad, sock, lambda n: n, timeout_s=5.0)
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            srv.serve_once()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    try:
        assert ping_pad_server(port, timeout_s=1.0) is True
        client = RemotePad(port)
        client.left_joystick_float(x_value_float=0.0, y_value_float=0.25)
        client.update()
        for _ in range(100):
            if srv.states:
                break
            threading.Event().wait(0.01)
        assert pad.sticks["left_y"] == 0.25
        client.close()
    finally:
        stop.set()
        t.join(timeout=1.0)
        sock.close()
    assert ping_pad_server(_free_udp_port(), timeout_s=0.1) is False  # nobody listening


def test_ensure_pad_server_launches_only_when_missing():
    launched = []
    assert (
        ensure_pad_server(9200, ping=lambda p: True, launch=lambda *a: launched.append(a)) is False
    )
    assert launched == []
    answers = iter([False, False, True])
    assert (
        ensure_pad_server(
            9200,
            0.5,
            ping=lambda p: next(answers),
            launch=lambda *a: launched.append(a),
            sleep=lambda s: None,
        )
        is True
    )
    assert launched == [(9200, 0.5)]
    with pytest.raises(HarnessConfigError, match="harness pad"):
        ensure_pad_server(
            9200, ping=lambda p: False, launch=lambda *a: None, wait_s=0.05, sleep=lambda s: None
        )


def test_sink_server_mode_drives_a_remote_pad_and_never_creates_a_local_one(monkeypatch):
    calls = []
    monkeypatch.setattr(pad_server, "ensure_pad_server", lambda port, t: calls.append((port, t)))
    sent = FakeSock()
    monkeypatch.setattr(pad_server, "RemotePad", lambda port: RemotePad(port, sock=sent))
    ctx = SimpleNamespace(contract=SimpleNamespace(HOVER_THR=0.4))
    cfg = load_config("liftoff", preset="liftoff").section("controls")
    assert cfg["pad"] == "server"
    sink = VgamepadActionSink.from_config(cfg, ctx)
    assert sink.keep_pad_connected is False and calls == []  # nothing until arm()
    sink.sleep = lambda s: None
    sink.arm()
    assert calls == [(9200, 0.5)]
    assert sent.sent[-1][0]["state"]["left_y"] == -1.0  # armed neutral, throttle cut
    sink.send_sticks({"throttle": 0.0})  # first command presses Arm (button *name*)
    assert "A" in sent.sent[-1][0]["buttons"]
    sink.failsafe()
    assert any("B" in m["buttons"] for m, _ in sent.sent)  # Disarm by name
    sink.close()
    assert sink.pad is None


def test_local_mode_is_the_default_and_validates():
    ctx = SimpleNamespace(contract=SimpleNamespace(HOVER_THR=0.4))
    cfg = load_config("liftoff").section("controls")
    assert cfg["pad"] == "local" and cfg["pad_port"] == 9200
    with pytest.raises(HarnessConfigError, match="controls.pad"):
        VgamepadActionSink.from_config({**cfg, "pad": "bluetooth"}, ctx)
    with pytest.raises(HarnessConfigError, match="pad_port"):
        VgamepadActionSink.from_config({**cfg, "pad": "server", "pad_port": 80}, ctx)
