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

import logging
import os
import re
import sys
from collections.abc import Callable
from typing import Any

from drone_fly.harness.config import require_bool
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.interfaces import FocusProbe

log = logging.getLogger(__name__)

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


Activator = Callable[[str, "re.Pattern[str] | None"], bool]


def _win32_activator() -> Activator:
    """``activate(process_name, title_re)``: raise the game's top-level window to the foreground.

    Windows only lets the foreground process change the foreground window, so this uses the
    usual workaround: a synthetic Alt key tap (which unlocks ``SetForegroundWindow``), restore the
    window if minimised, then ``SetForegroundWindow`` + ``BringWindowToTop``. Best effort: it
    returns False when no matching window exists; the caller still waits for real focus.
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    vk_menu, keyup, sw_restore = 0x12, 0x0002, 9

    def exe_of(hwnd: int) -> str:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        handle = kernel32.OpenProcess(0x1000, False, pid.value)
        if not handle:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(len(buf))
            ok = kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
            return os.path.basename(buf.value) if ok else ""
        finally:
            kernel32.CloseHandle(handle)

    def activate(process_name: str, title_re: re.Pattern[str] | None) -> bool:
        found: list[int] = []

        def visit(hwnd: int, _lparam: int) -> bool:
            if not user32.IsWindowVisible(hwnd) or user32.GetWindow(hwnd, 4):  # 4 = GW_OWNER
                return True
            if exe_of(hwnd).lower() != process_name:
                return True
            if title_re is not None:
                buf = ctypes.create_unicode_buffer(512)
                user32.GetWindowTextW(hwnd, buf, len(buf))
                if not title_re.search(buf.value):
                    return True
            found.append(hwnd)
            return False

        user32.EnumWindows(enum_proc(visit), 0)
        if not found:
            return False
        hwnd = found[0]
        user32.ShowWindow(hwnd, sw_restore if user32.IsIconic(hwnd) else 5)  # 5 = SW_SHOW
        # Windows only lets the foreground thread change the foreground window: attach to its
        # input queue while switching (plus the Alt-tap unlock), then fall back to
        # SwitchToThisWindow if the switch still did not take.
        fg = user32.GetForegroundWindow()
        fg_thread = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        me = kernel32.GetCurrentThreadId()
        attached = bool(
            fg_thread and fg_thread != me and user32.AttachThreadInput(me, fg_thread, True)
        )
        try:
            user32.keybd_event(vk_menu, 0, 0, 0)
            user32.keybd_event(vk_menu, 0, keyup, 0)
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
            user32.SetActiveWindow(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(me, fg_thread, False)
        if user32.GetForegroundWindow() != hwnd:
            user32.SwitchToThisWindow(hwnd, True)
        return True

    return activate


class Win32ForegroundFocusProbe(FocusProbe):
    """Foreground window's process name == ``focus.process_name`` (and title matches, if set)."""

    def __init__(
        self,
        process_name: str = "Liftoff.exe",
        title_regex: str | None = None,
        foreground: Foreground | None = None,
        auto_focus: bool = True,
        activator: Activator | None = None,
    ) -> None:
        if not process_name:
            raise HarnessConfigError("focus.process_name must name the game executable")
        self.process_name = process_name.lower()
        try:
            self.title_re = re.compile(title_regex) if title_regex else None
        except re.error as exc:
            raise HarnessConfigError(f"focus.title_regex is not a valid regex: {exc}") from exc
        self._foreground = foreground
        self.auto_focus = auto_focus
        self._activator = activator
        self.last_foreground: tuple[int, str, str] | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> Win32ForegroundFocusProbe:
        return cls(
            process_name=str(cfg.get("process_name") or "Liftoff.exe"),
            title_regex=cfg.get("title_regex"),
            auto_focus=require_bool(cfg.get("auto_focus", True), "focus.auto_focus"),
        )

    def request_focus(self) -> bool:
        """Raise the game window (``focus.auto_focus``); never raises, the wait still applies."""
        if not self.auto_focus:
            return False
        try:
            if self._activator is None:
                self._activator = _win32_activator()
            ok = bool(self._activator(self.process_name, self.title_re))
        except Exception:  # noqa: BLE001 - best effort only
            log.warning("could not bring %s to the foreground", self.process_name, exc_info=True)
            return False
        if ok:
            log.info("switched focus to %s", self.process_name)
        else:
            log.warning("no %s window found to focus; click into the game", self.process_name)
        return ok

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
