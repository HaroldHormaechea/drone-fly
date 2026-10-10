"""Abort hotkey via Win32 ``GetAsyncKeyState`` (polled on a background thread).

Works while the game window has focus (the key state is global). Ctrl+C in the terminal is
always honoured by the runner independently. A ``key_state`` callable can be injected for tests.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from typing import Any

from drone_fly.harness.config import require_number
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.interfaces import AbortSignal

#: Virtual-key codes for the accepted key names (F1-F12 plus a few others).
VK_CODES: dict[str, int] = {
    **{f"F{i}": 0x6F + i for i in range(1, 13)},
    "ESCAPE": 0x1B,
    "SPACE": 0x20,
    "PAUSE": 0x13,
    "BACKSPACE": 0x08,
    "END": 0x23,
}


def vk_for(key: str | int) -> int:
    if isinstance(key, int) and not isinstance(key, bool):
        return key
    code = VK_CODES.get(str(key).upper())
    if code is None:
        raise HarnessConfigError(f"abort.key {key!r} unknown; use one of {sorted(VK_CODES)}")
    return code


def _win32_key_state() -> Callable[[int], bool]:
    if sys.platform != "win32":
        raise HarnessConfigError(
            "the hotkey abort signal needs Windows; set parts.abort_signal: null elsewhere"
        )
    import ctypes

    get = ctypes.windll.user32.GetAsyncKeyState  # type: ignore[attr-defined]
    return lambda vk: bool(get(vk) & 0x8000)


class HotkeyAbortSignal(AbortSignal):
    def __init__(
        self,
        key: str | int = "F12",
        poll_s: float = 0.02,
        key_state: Callable[[int], bool] | None = None,
    ) -> None:
        self.vk = vk_for(key)
        self.poll_s = poll_s
        self._key_state = key_state
        self._set = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> HotkeyAbortSignal:
        return cls(
            key=cfg.get("key", "F12"),
            poll_s=float(require_number(cfg.get("poll_s", 0.02), "abort.poll_s", min_value=0.001)),
        )

    def start(self) -> None:
        if self._key_state is None:
            self._key_state = _win32_key_state()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="abort-hotkey", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._key_state is not None
        while not self._stop.wait(self.poll_s):
            if self._key_state(self.vk):
                self._set.set()

    def is_set(self) -> bool:
        return self._set.is_set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
