"""UDP receiver thread: drains every pending datagram on ``127.0.0.1:<port>`` into the state.

The socket gets a 4 MB receive buffer (best effort) so a redraw hiccup never makes the kernel
drop datagrams; the thread itself only decodes and hands each datagram to
:meth:`OverlayState.ingest_bytes`, which keeps the latest of each kind. A socket can be injected
for tests (anything with ``recvfrom`` raising ``BlockingIOError`` / ``TimeoutError`` when empty).
"""

from __future__ import annotations

import logging
import socket
import threading
from typing import Any

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.overlay.state import OverlayState

log = logging.getLogger(__name__)

RECV_BUFFER_BYTES = 4 << 20
MAX_RECV = 65536
POLL_TIMEOUT_S = 0.1


class UdpReceiver:
    """Receives overlay datagrams on a background thread (or synchronously via :meth:`poll`)."""

    def __init__(
        self, port: int, state: OverlayState, sock: Any = None, host: str = "127.0.0.1"
    ) -> None:
        self.address = (host, int(port))
        self.state = state
        self.sock = sock
        self.received = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def bind(self) -> None:
        """Create and bind the socket (no-op for an injected one)."""
        if self.sock is not None:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECV_BUFFER_BYTES)
        except OSError as exc:  # best effort
            log.warning("overlay receiver: could not set SO_RCVBUF: %s", exc)
        try:
            s.bind(self.address)
        except OSError as exc:
            s.close()
            raise HarnessConfigError(
                f"overlay port {self.address[1]} in use (or not bindable): {exc}. Is another "
                "overlay running? Choose another overlay.port / --port (and run the harness with "
                "the same one)."
            ) from exc
        s.settimeout(POLL_TIMEOUT_S)
        self.sock = s

    def _recv_once(self) -> bool:
        """Receive and ingest one datagram; ``False`` when nothing was pending."""
        try:
            data, _ = self.sock.recvfrom(MAX_RECV)
        except (BlockingIOError, TimeoutError):
            return False
        except ConnectionResetError:  # Windows reports ICMP errors on UDP recv; harmless
            return True
        self.received += 1
        self.state.ingest_bytes(data)
        return True

    def poll(self, max_datagrams: int = 100000) -> int:
        """Synchronously drain every pending datagram; returns how many were read."""
        self.bind()
        n = 0
        while n < max_datagrams and self._recv_once():
            n += 1
        return n

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if not self._recv_once():
                    self._stop.wait(0.005)  # an injected non-blocking socket must not spin
            except OSError:
                if self._stop.is_set():
                    return
                log.exception("overlay receiver error")
                self._stop.wait(0.1)

    def start(self) -> None:
        self.bind()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="overlay-receiver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:  # noqa: BLE001
                log.debug("closing the overlay receiver socket failed", exc_info=True)
