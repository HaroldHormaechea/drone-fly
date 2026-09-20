"""Route Python logging into the TUI logs pane and silence console handlers (UC-32, Windows).

Two Windows-only defects this fixes, both keyed off the fact that on Windows the live TUI has
commandeered the console (macOS/Linux route Python logging through :class:`FdLogCapture`'s
fd-1/2 pipe into the pane, so neither defect exists there — AC-9):

* **AC-4** — the root logger's ``StreamHandler`` (installed by ``logging.basicConfig`` in the
  CLI) keeps writing to that commandeered console, throwing ``OSError [WinError 1]`` on every
  emit. :class:`LogBridge` removes every root-logger handler whose stream is ``sys.stdout`` /
  ``sys.stderr`` for the duration of the live session and restores them exactly on teardown.
* **AC-7** — with those console handlers gone, Python log records would vanish; the bridge
  attaches a :class:`DashboardLogHandler` that formats each record into a bounded deque the
  dashboard renders in its logs pane.

The bridge is installed ONLY on Windows (guarded at its call site in
:mod:`~drone_fly.train.tui.dashboard`), so macOS/Linux behaviour is byte-identical (AC-9). It is
stdlib-only (no Rich), so importing it is cheap.
"""

from __future__ import annotations

import logging
import sys
from collections import deque

#: Default number of formatted log records retained for the pane (bounded like the scrollback).
DEFAULT_LOG_BUFFER = 500


class DashboardLogHandler(logging.Handler):
    """A logging handler that buffers formatted records into a bounded deque (AC-7)."""

    def __init__(self, maxlen: int = DEFAULT_LOG_BUFFER) -> None:
        super().__init__()
        self.buffer: deque[str] = deque(maxlen=max(int(maxlen), 1))
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.append(self.format(record))
        except Exception:  # noqa: BLE001 - a handler must never raise into the emitting code
            pass

    def lines(self) -> list[str]:
        """Snapshot the buffered log lines (oldest first)."""
        return list(self.buffer)


def _is_console_stream(handler: logging.Handler) -> bool:
    """True when ``handler`` writes to the current ``sys.stdout`` / ``sys.stderr`` (AC-4)."""
    stream = getattr(handler, "stream", None)
    return stream is not None and stream in (sys.stdout, sys.stderr)


class LogBridge:
    """Install/restore the dashboard log handler + console-handler suppression (Windows).

    Idempotent: :meth:`install` and :meth:`uninstall` are safe to call more than once. On
    teardown the exact set of removed console handlers is re-attached, so logging behaves as it
    did before the session.
    """

    def __init__(self, *, maxlen: int = DEFAULT_LOG_BUFFER) -> None:
        self.handler = DashboardLogHandler(maxlen=maxlen)
        self._removed: list[tuple[logging.Logger, logging.Handler]] = []
        self._installed = False

    def lines(self) -> list[str]:
        return self.handler.lines()

    def install(self) -> None:
        if self._installed:
            return
        root = logging.getLogger()
        # Remove every console (stdout/stderr) handler on the root logger so no emit hits the
        # commandeered console (AC-4). Record each for an exact restore on teardown.
        for h in list(root.handlers):
            if _is_console_stream(h):
                root.removeHandler(h)
                self._removed.append((root, h))
        root.addHandler(self.handler)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        root = logging.getLogger()
        try:
            root.removeHandler(self.handler)
        except Exception:  # noqa: BLE001 - teardown must never raise
            pass
        for lg, h in self._removed:
            lg.addHandler(h)
        self._removed = []
        self._installed = False


__all__ = ["DashboardLogHandler", "LogBridge", "DEFAULT_LOG_BUFFER"]
