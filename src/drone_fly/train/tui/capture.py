"""C-level (file-descriptor) stdout/stderr capture for the training TUI (UC-22, AC5).

The right-hand log pane must show what pybullet's **native** code prints — those writes go
straight to file descriptors 1/2, bypassing Python's ``sys.stdout``. :class:`FdLogCapture`
therefore redirects the real fds with ``os.dup2`` into a pipe and drains the read end on a
daemon thread into a bounded scrollback ``deque``. It also hands back a ``display_stdout``
stream bound to the *original* terminal, so Rich can keep drawing the dashboard while every
other write is captured.

Teardown restores the saved fds in ``_restore``, which runs from ``__exit__`` on normal exit,
on an exception, and on ``KeyboardInterrupt`` alike — the terminal is always left usable and
no final output is lost (AC8). The reader thread cannot deadlock a full pipe because it drains
continuously; the scrollback is bounded so unbounded native spew cannot exhaust memory.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections import deque

logger = logging.getLogger(__name__)

#: Default number of captured lines retained (scrollback bound).
DEFAULT_SCROLLBACK = 500
#: Read chunk size (large, so a busy pipe drains in few syscalls).
_READ_CHUNK = 65536


class FdLogCapture:
    """Context manager redirecting real fd 1/2 into a bounded, thread-drained scrollback."""

    def __init__(self, maxlen: int = DEFAULT_SCROLLBACK) -> None:
        self.scrollback: deque[str] = deque(maxlen=max(int(maxlen), 1))
        self.display_stdout = None  # text stream bound to the ORIGINAL terminal (for Rich)

        self._active = False
        self._saved_stdout_fd: int | None = None
        self._saved_stderr_fd: int | None = None
        self._read_fd: int | None = None
        self._write_fd: int | None = None
        self._thread: threading.Thread | None = None
        self._buf = b""

    # -- public ----------------------------------------------------------------------------

    def lines(self) -> list[str]:
        """Snapshot the current scrollback (oldest first)."""
        return list(self.scrollback)

    # -- context manager -------------------------------------------------------------------

    def __enter__(self) -> FdLogCapture:
        try:
            self._read_fd, self._write_fd = os.pipe()
            # Save the real terminal fds BEFORE redirecting.
            self._saved_stdout_fd = os.dup(1)
            self._saved_stderr_fd = os.dup(2)
            # An independent stream to the original terminal so the TUI can still draw.
            self.display_stdout = os.fdopen(
                os.dup(self._saved_stdout_fd),
                "w",
                buffering=1,
                encoding="utf-8",
                errors="replace",
                closefd=True,
            )
            # Flush Python-level buffers, then point fd 1/2 at the pipe write end.
            self._flush_std()
            os.dup2(self._write_fd, 1)
            os.dup2(self._write_fd, 2)
            self._active = True
            self._thread = threading.Thread(target=self._reader, name="fd-log-capture", daemon=True)
            self._thread.start()
        except Exception:
            # Partial setup must not leave the process with hijacked fds.
            self._restore()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._restore()
        return False  # never suppress (incl. KeyboardInterrupt / exceptions)

    # -- internals -------------------------------------------------------------------------

    def _reader(self) -> None:
        assert self._read_fd is not None
        while True:
            try:
                chunk = os.read(self._read_fd, _READ_CHUNK)
            except OSError:
                break
            if not chunk:  # EOF: all write ends closed
                break
            self._ingest(chunk)

    def _ingest(self, chunk: bytes) -> None:
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self.scrollback.append(line.decode("utf-8", "replace"))

    @staticmethod
    def _flush_std() -> None:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001 - flushing must never raise out of teardown
                pass

    def _restore(self) -> None:
        """Restore fds, stop the reader, drain the tail — idempotent and exception-safe."""
        self._flush_std()

        # 1) Put the real terminal back on fd 1/2 so later writes hit the screen, not the pipe.
        if self._active:
            for saved, target in (
                (self._saved_stdout_fd, 1),
                (self._saved_stderr_fd, 2),
            ):
                if saved is not None:
                    try:
                        os.dup2(saved, target)
                    except OSError:
                        pass
            self._active = False

        # 2) Close our write end so the reader sees EOF, then join it.
        if self._write_fd is not None:
            try:
                os.close(self._write_fd)
            except OSError:
                pass
            self._write_fd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        # 3) Bank any trailing partial line (reader has stopped — no race on ``_buf``).
        if self._buf:
            self.scrollback.append(self._buf.decode("utf-8", "replace"))
            self._buf = b""

        # 4) Close the remaining fds / streams.
        if self._read_fd is not None:
            try:
                os.close(self._read_fd)
            except OSError:
                pass
            self._read_fd = None
        if self.display_stdout is not None:
            try:
                self.display_stdout.flush()
                self.display_stdout.close()
            except Exception:  # noqa: BLE001
                pass
            self.display_stdout = None
        for fd in (self._saved_stdout_fd, self._saved_stderr_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._saved_stdout_fd = None
        self._saved_stderr_fd = None


__all__ = ["FdLogCapture", "DEFAULT_SCROLLBACK"]
