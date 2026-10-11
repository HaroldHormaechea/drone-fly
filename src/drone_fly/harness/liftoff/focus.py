"""Focus probe: is the Liftoff window in the foreground? (``win32-foreground``)

The virtual pad's input reaches whichever window has focus, so a run that keeps sending sticks
after the user alt-tabs drives nothing in Liftoff — and something else on the desktop. The
runner waits (pad neutral) until the game is focused, then fails safe with ``focus_lost`` the
moment it is not.

``has_focus()`` calls ``GetForegroundWindow`` (cheap) every tick and caches the owning process
name per window handle, so the process lookup only runs when the foreground window changes. A
``foreground`` callable returning ``(hwnd, process_name, title)`` can be injected for tests.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from typing import Any

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.interfaces import FocusProbe

Foreground = Callable[[], tuple[int, str, str]]


def _win32_foreground() -> Foreground:
    if sys.platform != "win32":
        raise HarnessConfigError(
            "the win32-foreground focus probe needs Windows; set parts.focus_probe: null elsewhere"
        )
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    process_query_limited = 0x1000
    names: dict[int, str] = {}

    def process_name(hwnd: int) -> str:
        if hwnd in names:
            return names[hwnd]
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        name = ""
        handle = kernel32.OpenProcess(process_query_limited, False, pid.value)
        if handle:
            try:
                buf = ctypes.create_unicode_buffer(1024)
                size = wintypes.DWORD(len(buf))
                if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                    name = os.path.basename(buf.value)
            finally:
                kernel32.CloseHandle(handle)
        names[hwnd] = name
        return name

    def title(hwnd: int) -> str:
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, len(buf))
        return buf.value

    def foreground() -> tuple[int, str, str]:
        hwnd = int(user32.GetForegroundWindow() or 0)
        if not hwnd:
            return 0, "", ""
        return hwnd, process_name(hwnd), title(hwnd)

    return foreground


class Win32ForegroundFocusProbe(FocusProbe):
    """Foreground window's process name == ``focus.process_name`` (and title matches, if set)."""

    def __init__(
        self,
        process_name: str = "Liftoff.exe",
        title_regex: str | None = None,
        foreground: Foreground | None = None,
    ) -> None:
        if not process_name:
            raise HarnessConfigError("focus.process_name must name the game executable")
        self.process_name = process_name.lower()
        try:
            self.title_re = re.compile(title_regex) if title_regex else None
        except re.error as exc:
            raise HarnessConfigError(f"focus.title_regex is not a valid regex: {exc}") from exc
        self._foreground = foreground
        self.last_foreground: tuple[int, str, str] | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> Win32ForegroundFocusProbe:
        return cls(
            process_name=str(cfg.get("process_name") or "Liftoff.exe"),
            title_regex=cfg.get("title_regex"),
        )

    def start(self) -> None:
        if self._foreground is None:
            self._foreground = _win32_foreground()

    def has_focus(self) -> bool:
        if self._foreground is None:
            self.start()
        assert self._foreground is not None
        hwnd, name, title = self._foreground()
        self.last_foreground = (hwnd, name, title)
        if not hwnd or name.lower() != self.process_name:
            return False
        return self.title_re is None or bool(self.title_re.search(title))
