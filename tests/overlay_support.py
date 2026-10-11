"""Shared fakes for the UC-65 live-overlay tests (``tests/test_overlay_*.py``).

Toolkit-free: nothing here imports PySide6, torch or the contract, so the hermetic tests can use
it too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from drone_fly.overlay.protocol import Packet, decode_datagram
from drone_fly.overlay.win32 import MonitorRect

OVERLAY_RUN_DIR = Path(__file__).parent / "data" / "overlay" / "run"


class FakeSocket:
    """Records ``sendto`` datagrams; ``raises`` (an exception, or a list of them) fails sends."""

    def __init__(self, raises: Any = None) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self.raises = raises
        self.attempts = 0
        self.closed = 0

    def sendto(self, data: bytes, address: tuple[str, int]) -> int:
        self.attempts += 1
        exc = self.raises
        if isinstance(exc, list):
            exc = exc.pop(0) if exc else None
        if exc is not None:
            raise exc
        self.sent.append((bytes(data), address))
        return len(data)

    def close(self) -> None:
        self.closed += 1

    def packets(self) -> list[Packet]:
        out = [decode_datagram(d) for d, _ in self.sent]
        assert all(p is not None for p in out), "the publisher sent an undecodable datagram"
        return out  # type: ignore[return-value]

    def kinds(self) -> list[str]:
        return [p.kind for p in self.packets()]

    def of_kind(self, kind: str) -> list[Packet]:
        return [p for p in self.packets() if p.kind == kind]


class FakeRecvSocket:
    """``recvfrom`` pops queued datagrams (or exceptions), then raises ``BlockingIOError``."""

    def __init__(self, items: list[Any] | None = None) -> None:
        self.items = list(items or [])
        self.closed = 0

    def recvfrom(self, _n: int) -> tuple[bytes, tuple[str, int]]:
        if not self.items:
            raise BlockingIOError
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item, ("127.0.0.1", 50000)

    def close(self) -> None:
        self.closed += 1


class FakeClock:
    """A manual monotonic clock (``clock()`` returns ``t``)."""

    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeShim:
    """A :class:`~drone_fly.overlay.win32.Win32Shim` that records every call in order."""

    def __init__(
        self,
        rects: list[MonitorRect] | None = None,
        *,
        affinity_ok: bool = True,
        dpi_ok: bool = True,
        initial_style: int = 0,
        foreground: int = 0,
    ) -> None:
        self.rects = rects if rects is not None else [(0, 0, 2560, 1080)]
        self.affinity_ok = affinity_ok
        self.dpi_ok = dpi_ok
        self.style = initial_style
        self.foreground = foreground
        self.calls: list[tuple[Any, ...]] = []

    def set_dpi_awareness(self) -> bool:
        self.calls.append(("set_dpi_awareness",))
        return self.dpi_ok

    def monitor_rects(self) -> list[MonitorRect]:
        self.calls.append(("monitor_rects",))
        return list(self.rects)

    def get_ex_style(self, hwnd: int) -> int:
        self.calls.append(("get_ex_style", hwnd))
        return self.style

    def set_ex_style(self, hwnd: int, style: int) -> None:
        self.calls.append(("set_ex_style", hwnd, style))
        self.style = style

    def set_window_pos(
        self, hwnd: int, insert_after: int, x: int, y: int, w: int, h: int, flags: int
    ) -> None:
        self.calls.append(("set_window_pos", hwnd, insert_after, x, y, w, h, flags))

    def show_no_activate(self, hwnd: int) -> None:
        self.calls.append(("show_no_activate", hwnd))

    def set_display_affinity(self, hwnd: int, affinity: int) -> bool:
        self.calls.append(("set_display_affinity", hwnd, affinity))
        return self.affinity_ok

    def foreground_hwnd(self) -> int:
        return self.foreground

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]
