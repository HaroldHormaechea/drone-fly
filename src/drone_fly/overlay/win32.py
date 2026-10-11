"""Win32 window behaviour for the overlay, behind an injectable shim (UC-65 AC5 / AC6).

The overlay window must never take focus or input — the harness fails safe with ``focus_lost``
the moment the game is not the foreground window — and must stay out of the harness's own
screen capture. :func:`configure_overlay_window` applies, to the native handle:

1. extended styles ``WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE |
   WS_EX_TOOLWINDOW``, with ``WS_EX_APPWINDOW`` cleared (no taskbar / Alt-Tab entry; mouse
   clicks pass through to the window underneath);
2. ``SetWindowPos(HWND_TOPMOST, rect, SWP_NOACTIVATE | SWP_FRAMECHANGED)`` over the capture
   monitor;
3. ``ShowWindow(SW_SHOWNOACTIVATE)``;
4. ``SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`` — the result decides the AC6 fallback.

``SetLayeredWindowAttributes`` is deliberately **not** called: the toolkit already renders a
per-pixel-alpha layered window (``UpdateLayeredWindow``), and setting a colour key / constant
alpha would switch the window to the other layered mode and break per-pixel transparency.

:func:`prepare_process` makes the process per-monitor-DPI-aware (v2) **before** the GUI toolkit
starts, so window coordinates are physical pixels and match the game's.

Monitors: ``camera.monitor`` / ``--monitor`` follow the mss convention used by the harness's
screen capture — 1-based, ``0`` = the whole virtual desktop (dxcam uses ``output_idx =
monitor - 1``). :func:`monitor_rect` maps it onto ``EnumDisplayMonitors`` order, which is the
order mss enumerates; DXGI output order is assumed to match (see ADR 0004).

The module imports everywhere; :class:`CtypesWin32` refuses to start off Windows.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from drone_fly.harness.errors import HarnessConfigError

log = logging.getLogger(__name__)

WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
GWL_EXSTYLE = -20
WDA_EXCLUDEFROMCAPTURE = 0x00000011
SW_SHOWNOACTIVATE = 4
HWND_TOPMOST = -1
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4

#: Exactly the extended styles the overlay sets (AC5).
OVERLAY_EX_STYLES = (
    WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
)

MonitorRect = tuple[int, int, int, int]  # left, top, width, height (physical pixels)


class Win32Shim(Protocol):
    """The Win32 calls the overlay makes (a fake implements this for tests)."""

    def set_dpi_awareness(self) -> bool: ...

    def monitor_rects(self) -> list[MonitorRect]: ...

    def get_ex_style(self, hwnd: int) -> int: ...

    def set_ex_style(self, hwnd: int, style: int) -> None: ...

    def set_window_pos(
        self, hwnd: int, insert_after: int, x: int, y: int, w: int, h: int, flags: int
    ) -> None: ...

    def show_no_activate(self, hwnd: int) -> None: ...

    def set_display_affinity(self, hwnd: int, affinity: int) -> bool: ...

    def foreground_hwnd(self) -> int: ...


@dataclass(frozen=True)
class WindowSetup:
    ex_style: int
    affinity_ok: bool


def prepare_process(shim: Win32Shim) -> bool:
    """Per-monitor-DPI-aware (v2) process; call before the GUI toolkit is created."""
    ok = bool(shim.set_dpi_awareness())
    if not ok:
        log.warning("could not make the overlay per-monitor DPI aware; coordinates may be scaled")
    return ok


def monitor_rect(rects: Sequence[MonitorRect], monitor: object) -> MonitorRect:
    """The rect of mss-style monitor index ``monitor`` (1-based; 0 = all monitors' union)."""
    if isinstance(monitor, bool) or not isinstance(monitor, int):
        raise HarnessConfigError(f"monitor must be an integer (mss index), got {monitor!r}")
    if not rects:
        raise HarnessConfigError("no display monitors found")
    if monitor == 0:
        x0 = min(r[0] for r in rects)
        y0 = min(r[1] for r in rects)
        x1 = max(r[0] + r[2] for r in rects)
        y1 = max(r[1] + r[3] for r in rects)
        return (x0, y0, x1 - x0, y1 - y0)
    if not 1 <= monitor <= len(rects):
        raise HarnessConfigError(
            f"monitor {monitor} out of range: {len(rects)} monitor(s) (1..{len(rects)}, 0 = all)"
        )
    return tuple(int(v) for v in rects[monitor - 1])  # type: ignore[return-value]


def overlay_ex_style(current: int) -> int:
    """``current`` with the overlay styles OR-ed in and ``WS_EX_APPWINDOW`` cleared."""
    return (int(current) | OVERLAY_EX_STYLES) & ~WS_EX_APPWINDOW


def configure_overlay_window(hwnd: int, shim: Win32Shim, rect: MonitorRect) -> WindowSetup:
    """Apply the styles, topmost placement over ``rect``, show-no-activate and capture exclusion."""
    style = overlay_ex_style(shim.get_ex_style(hwnd))
    shim.set_ex_style(hwnd, style)
    x, y, w, h = rect
    shim.set_window_pos(hwnd, HWND_TOPMOST, x, y, w, h, SWP_NOACTIVATE | SWP_FRAMECHANGED)
    shim.show_no_activate(hwnd)
    affinity_ok = bool(shim.set_display_affinity(hwnd, WDA_EXCLUDEFROMCAPTURE))
    if not affinity_ok:
        log.error(
            "SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) FAILED: the overlay could appear in "
            "the harness's screen capture (needs Windows 10 2004+)"
        )
    return WindowSetup(style, affinity_ok)


class CtypesWin32:
    """The real :class:`Win32Shim` (``ctypes``; Windows only)."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise HarnessConfigError(
                "the overlay window needs Windows (Win32 styles + display affinity); use "
                "--replay on Windows, or the pure overlay modules elsewhere"
            )
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes
        self._wt = wintypes
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        u = self.user32
        u.GetForegroundWindow.restype = wintypes.HWND
        u.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        u.SetWindowPos.restype = wintypes.BOOL
        u.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        u.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
        u.SetWindowDisplayAffinity.restype = wintypes.BOOL
        self._get_long = getattr(u, "GetWindowLongPtrW", None) or u.GetWindowLongW
        self._set_long = getattr(u, "SetWindowLongPtrW", None) or u.SetWindowLongW
        self._get_long.argtypes = [wintypes.HWND, ctypes.c_int]
        self._get_long.restype = ctypes.c_ssize_t
        self._set_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
        self._set_long.restype = ctypes.c_ssize_t

    def set_dpi_awareness(self) -> bool:
        ct = self._ct
        try:
            fn = self.user32.SetProcessDpiAwarenessContext
            fn.argtypes = [ct.c_void_p]
            fn.restype = self._wt.BOOL
            if fn(ct.c_void_p(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)):
                return True
        except AttributeError:
            pass
        try:  # Windows 8.1 fallback: PROCESS_PER_MONITOR_DPI_AWARE
            return ct.windll.shcore.SetProcessDpiAwareness(2) == 0  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            return False

    def monitor_rects(self) -> list[MonitorRect]:
        ct, wt = self._ct, self._wt
        rects: list[MonitorRect] = []
        proc_type = ct.WINFUNCTYPE(wt.BOOL, wt.HMONITOR, wt.HDC, ct.POINTER(wt.RECT), wt.LPARAM)

        def cb(_hmon, _hdc, lprc, _data):  # noqa: ANN001
            r = lprc.contents
            rects.append((int(r.left), int(r.top), int(r.right - r.left), int(r.bottom - r.top)))
            return True

        self.user32.EnumDisplayMonitors(None, None, proc_type(cb), 0)
        return rects

    def get_ex_style(self, hwnd: int) -> int:
        return int(self._get_long(hwnd, GWL_EXSTYLE)) & 0xFFFFFFFF

    def set_ex_style(self, hwnd: int, style: int) -> None:
        self._set_long(hwnd, GWL_EXSTYLE, int(style))

    def set_window_pos(
        self, hwnd: int, insert_after: int, x: int, y: int, w: int, h: int, flags: int
    ) -> None:
        if not self.user32.SetWindowPos(hwnd, insert_after, x, y, w, h, flags):
            log.warning("SetWindowPos failed (error %d)", self._ct.get_last_error())

    def show_no_activate(self, hwnd: int) -> None:
        self.user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)

    def set_display_affinity(self, hwnd: int, affinity: int) -> bool:
        ok = bool(self.user32.SetWindowDisplayAffinity(hwnd, affinity))
        if not ok:
            log.warning("SetWindowDisplayAffinity error %d", self._ct.get_last_error())
        return ok

    def foreground_hwnd(self) -> int:
        return int(self.user32.GetForegroundWindow() or 0)
