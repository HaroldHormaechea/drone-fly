"""Bounded, filtered tail of a training run's raw log file (UC-61 follow-up item 5).

Training launches redirect their stdout+stderr to ``training/<name>/logs/app.log`` (see
:func:`app.runs._default_spawn`). The desktop app polls this file so the Status view can show a live
process-log panel. This module is the read side:

* **Bounded byte-offset tail** (:func:`tail`) — the app passes the byte offset it last saw as
  ``since`` and gets back the new complete lines plus the ``next`` offset to poll from. Reads are
  capped at :data:`_MAX_READ` bytes per poll (a burst can't return an unbounded slice); the cursor
  advances only over **complete** newline-terminated lines (a torn in-flight tail is left for the
  next poll); a ``since`` past the current size (file rotated/truncated) resets to ``0``; an absent
  file yields an empty result. When the cap forces a skip over unread bytes, a single truncation
  marker line is emitted so the skip is visible.
* **Read-time deny-filter** (:data:`_DENY_RE`) — the RAW log is kept on disk untouched (full record
  for debugging); only the *returned* lines are filtered. The owner-confirmed deny-list drops known
  pybullet / OpenGL / thread-init boot noise (anchored, case-insensitive — never a bare
  ``semaphore`` / ``OpenGL`` substring) and keeps everything else (SB3 rollout tables, checkpoint
  saves, health verdicts, warnings, tracebacks, and our own lines).

Stdlib-only (``os`` + ``re``) so it is cheap and hermetically testable with a temp file.
"""

from __future__ import annotations

import os
import re

#: Max bytes read per poll. Caps the response payload and bounds work when a run floods stdout.
_MAX_READ = 64 * 1024

#: Emitted once (in place of the skipped bytes) when the cap forces the cursor to jump forward.
_TRUNCATION_MARKER = "… earlier output skipped (log advanced faster than the poll) …"

#: OWNER-CONFIRMED deny-list (Q1, 2026-09-27): drop known pybullet/GL/thread boot noise, keep
#: everything else. Anchored at line start and case-insensitive — deliberately NOT bare
#: ``semaphore`` / ``OpenGL`` substrings, so an SB3 line that merely mentions those words survives.
_DENY_PATTERNS: tuple[str, ...] = (
    r"pybullet build time:",
    r"argv\[0\]=",
    r"b3Printf",
    r"b3Warning",
    r"b3.*semaphore",
    r"ven\s?=",
    r"GL_(VENDOR|RENDERER|VERSION)",
    r"(Vendor|Renderer|Version)\s?=",
    r"(startThreads|stopThreads|numActiveThreads|Thread with id)",
    r"(EGL|GLX|MesaGL)",
    r"Workaround for some crash in the Intel OpenGL driver",
    r"ExampleBrowserThreadFunc",
)
#: Compiled once. Each alternative is anchored via ``re.match`` (which matches at the string start).
_DENY_RE = re.compile("|".join(_DENY_PATTERNS), re.IGNORECASE)


def log_path_for(project_root: str, name: str) -> str:
    """Return ``<project_root>/training/<name>/logs/app.log`` (the redirected run log)."""
    return os.path.join(project_root, "training", name, "logs", "app.log")


def _is_noise(line: str) -> bool:
    """Return ``True`` for a line that should be filtered out (blank or a deny-list boot line)."""
    if line.strip() == "":
        return True
    return _DENY_RE.match(line) is not None


def tail(path: str, since: int = 0) -> dict:
    """Return the filtered new lines in ``path`` from byte offset ``since``.

    Returns ``{"lines": [<filtered strings>], "next": <byte offset>}``. ``next`` is the offset the
    caller should pass as ``since`` on the following poll; it is only ever advanced over complete
    newline-terminated lines, so an in-flight partial line is re-read (whole) next time. The read is
    bounded to the last :data:`_MAX_READ` bytes; when that skips unread output a single
    :data:`_TRUNCATION_MARKER` line is prepended. Guards: ``since`` < 0 or beyond the file size
    (rotation/truncation) restarts at ``0``; an absent/unreadable file returns
    ``{"lines": [], "next": since}``.
    """
    if not os.path.isfile(path):
        return {"lines": [], "next": since}
    try:
        size = os.path.getsize(path)
    except OSError:
        return {"lines": [], "next": since}

    if since < 0 or since > size:
        since = 0  # cursor invalid (file rotated/truncated) → restart from the top.

    start = since
    skipped = False
    if size - start > _MAX_READ:
        start = size - _MAX_READ
        skipped = True

    try:
        with open(path, "rb") as fh:
            fh.seek(start)
            data = fh.read(size - start)
    except OSError:
        return {"lines": [], "next": since}

    last_nl = data.rfind(b"\n")
    if last_nl < 0:
        # No complete line in the window yet. Don't advance past a torn tail; if we skipped we still
        # can't safely emit a fragment, so hold the cursor at the window start to retry next poll.
        return {"lines": [], "next": start if skipped else since}

    complete = data[: last_nl + 1]
    next_offset = start + last_nl + 1

    raw_lines = complete.decode("utf-8", errors="replace").split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()  # trailing "" from the final newline.
    if skipped and raw_lines:
        raw_lines = raw_lines[1:]  # first line is a torn fragment (we started mid-line).

    out: list[str] = []
    if skipped:
        out.append(_TRUNCATION_MARKER)
    out.extend(line for line in raw_lines if not _is_noise(line))
    return {"lines": out, "next": next_offset}


__all__ = ["log_path_for", "tail"]
