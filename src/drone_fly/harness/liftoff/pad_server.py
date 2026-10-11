"""Long-lived virtual pad (``drone-fly harness pad``) shared by every run.

Liftoff re-arms and spins the motors up (centred sticks = about half throttle) whenever the
virtual Xbox 360 controller disconnects, and a second virtual controller may not get the user's
bindings. So one process owns **one** ViGEm pad for the whole session and keeps it connected,
neutral (throttle at minimum) until Ctrl+C. Runs and ``pad-check`` drive it over localhost UDP
through :class:`RemotePad`, a drop-in for the ``vgamepad`` object the action sink already uses
(``controls.pad: server``).

Protocol (JSON datagrams to ``127.0.0.1:<controls.pad_port>``):

* ``{"v": 1, "state": {"left_x", "left_y", "right_x", "right_y"}, "buttons": [names]}`` — the
  full pad state, sent on every ``update()``; applied as-is (stick floats in [-1, 1]).
* ``{"v": 1, "ping": <id>}`` — answered with ``{"v": 1, "pong": <id>}`` to the sender.

Watchdog: when a client has been driving the pad and nothing arrives for ``timeout_s``
(the run crashed or was killed), the server centres the sticks, sets the throttle to minimum and
releases every button. It never presses Disarm on its own (it does not know the bindings).
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

PROTOCOL = 1
DEFAULT_PORT = 9200
STICKS = ("left_x", "left_y", "right_x", "right_y")
#: Neutral pad: sticks centred, throttle (mode 2: left stick Y) at minimum.
NEUTRAL = {"left_x": 0.0, "left_y": -1.0, "right_x": 0.0, "right_y": 0.0}


def _clip(v: Any) -> float:
    return max(-1.0, min(1.0, float(v)))


class PadServer:
    """Applies client pad states to one virtual pad; neutral watchdog. Socket/pad injectable."""

    def __init__(
        self,
        pad: Any,
        sock: Any,
        button_value: Callable[[str], Any],
        *,
        timeout_s: float = 0.5,
        neutral: dict[str, float] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.pad = pad
        self.sock = sock
        self.button_value = button_value
        self.timeout_s = timeout_s
        self.neutral = dict(neutral or NEUTRAL)
        self.clock = clock
        self.pressed: set[str] = set()
        self.last_rx: float | None = None  # last state datagram (None = idle, already neutral)
        self.states = 0
        self.watchdog_trips = 0
        self.apply(self.neutral, [])

    def apply(self, sticks: dict[str, float], buttons: list[str]) -> None:
        s = {k: _clip(sticks.get(k, self.neutral[k])) for k in STICKS}
        self.pad.left_joystick_float(x_value_float=s["left_x"], y_value_float=s["left_y"])
        self.pad.right_joystick_float(x_value_float=s["right_x"], y_value_float=s["right_y"])
        wanted = set(buttons)
        for name in sorted(self.pressed - wanted):
            self.pad.release_button(button=self.button_value(name))
        for name in sorted(wanted - self.pressed):
            self.pad.press_button(button=self.button_value(name))
        self.pressed = wanted
        self.pad.update()

    def handle(self, data: bytes, addr: Any) -> None:
        try:
            msg = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(msg, dict) or msg.get("v") != PROTOCOL:
            return
        if "ping" in msg:
            self.sock.sendto(json.dumps({"v": PROTOCOL, "pong": msg["ping"]}).encode(), addr)
            return
        state = msg.get("state")
        if not isinstance(state, dict):
            return
        buttons = [b for b in msg.get("buttons") or [] if isinstance(b, str)]
        try:
            self.apply(state, buttons)
        except (KeyError, TypeError, ValueError, AttributeError):
            log.warning("ignoring a malformed pad state: %r", msg)
            return
        self.states += 1
        self.last_rx = self.clock()

    def check_watchdog(self) -> bool:
        """Go neutral when a driving client went silent; True if it tripped."""
        if self.last_rx is None or self.clock() - self.last_rx < self.timeout_s:
            return False
        self.apply(self.neutral, [])
        self.last_rx = None
        self.watchdog_trips += 1
        log.warning("no pad commands for %.2f s: sticks neutral, throttle minimum", self.timeout_s)
        return True

    def serve_once(self) -> None:
        """Handle every pending datagram (the socket has a short timeout), then the watchdog."""
        try:
            data, addr = self.sock.recvfrom(4096)
            self.handle(data, addr)
        except (TimeoutError, BlockingIOError, ConnectionResetError, OSError):
            pass
        self.check_watchdog()


def run_pad_server(port: int = DEFAULT_PORT, timeout_s: float = 0.5) -> int:
    """``drone-fly harness pad``: own one ViGEm pad until Ctrl+C (blocking)."""
    from drone_fly.harness.errors import HarnessConfigError, lazy_import
    from drone_fly.harness.liftoff.gamepad import ARM_BUTTONS

    vg = lazy_import("vgamepad", purpose="the ViGEm virtual gamepad (requires the ViGEmBus driver)")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", int(port)))
    except OSError as exc:
        sock.close()
        raise HarnessConfigError(
            f"pad port {port} in use: is another `drone-fly harness pad` already running?"
        ) from exc
    sock.settimeout(0.02)

    def button_value(name: str) -> Any:
        return getattr(vg.XUSB_BUTTON, ARM_BUTTONS[name])

    server = PadServer(vg.VX360Gamepad(), sock, button_value, timeout_s=timeout_s)
    print(
        f"virtual pad connected and held neutral (throttle minimum); runs drive it via "
        f"127.0.0.1:{port}. Keep this window open between runs. Ctrl+C disconnects the pad: "
        "Liftoff then re-arms, so reset or pause the drone first.",
        flush=True,
    )
    try:
        while True:
            server.serve_once()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.apply(server.neutral, [])
        finally:
            sock.close()
    print("pad disconnected")
    return 0


class RemotePad:
    """Client side: the ``vgamepad.VX360Gamepad`` methods the sink uses, sent to the pad server.

    ``press_button`` / ``release_button`` take button *names* (the sink's ``button_resolver`` is
    the identity in server mode); every ``update()`` sends the full state.
    """

    def __init__(self, port: int = DEFAULT_PORT, sock: Any = None) -> None:
        self.addr = ("127.0.0.1", int(port))
        self.sock = sock if sock is not None else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state = dict(NEUTRAL)
        self.buttons: set[str] = set()

    def left_joystick_float(self, x_value_float: float, y_value_float: float) -> None:
        self.state["left_x"], self.state["left_y"] = float(x_value_float), float(y_value_float)

    def right_joystick_float(self, x_value_float: float, y_value_float: float) -> None:
        self.state["right_x"], self.state["right_y"] = float(x_value_float), float(y_value_float)

    def press_button(self, button: Any) -> None:
        self.buttons.add(str(button))

    def release_button(self, button: Any) -> None:
        self.buttons.discard(str(button))

    def update(self) -> None:
        msg = {"v": PROTOCOL, "state": self.state, "buttons": sorted(self.buttons)}
        try:
            self.sock.sendto(json.dumps(msg).encode("utf-8"), self.addr)
        except OSError:
            log.warning("pad server unreachable on port %d", self.addr[1])

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def ping_pad_server(port: int = DEFAULT_PORT, timeout_s: float = 0.3, sock: Any = None) -> bool:
    """True when a pad server answers on ``port``."""
    own = sock is None
    s = sock if sock is not None else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.settimeout(timeout_s)
        token = int(time.monotonic() * 1e6) & 0x7FFFFFFF
        s.sendto(json.dumps({"v": PROTOCOL, "ping": token}).encode(), ("127.0.0.1", int(port)))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                data, _ = s.recvfrom(4096)
            except (TimeoutError, ConnectionResetError, OSError):
                return False
            try:
                msg = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(msg, dict) and msg.get("pong") == token:
                return True
        return False
    finally:
        if own:
            s.close()


def launch_pad_server(port: int = DEFAULT_PORT, timeout_s: float = 0.5) -> Any:
    """Start ``drone-fly harness pad`` in its own console window (it outlives this run)."""
    cmd = [sys.executable, "-m", "drone_fly.cli.main", "harness", "pad", "--port", str(port)]
    cmd += ["--timeout", str(timeout_s)]
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if sys.platform == "win32" else 0
    log.info("starting the long-lived pad: %s", " ".join(cmd))
    return subprocess.Popen(cmd, creationflags=flags)


def ensure_pad_server(
    port: int = DEFAULT_PORT,
    timeout_s: float = 0.5,
    *,
    ping: Callable[[int], bool] = ping_pad_server,
    launch: Callable[[int, float], Any] = launch_pad_server,
    wait_s: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Make sure a pad server answers on ``port``, launching one if needed. False on timeout."""
    if ping(port):
        return False  # already running: nothing launched
    launch(port, timeout_s)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if ping(port):
            return True
        sleep(0.2)
    from drone_fly.harness.errors import HarnessConfigError

    raise HarnessConfigError(
        f"the pad server did not come up on port {port}; run `drone-fly harness pad` yourself"
    )
