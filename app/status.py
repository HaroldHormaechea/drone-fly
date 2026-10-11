"""Parse a reservoir run's captured stdout into live progress (UC-63 AC5).

``train_aug`` (``gpu_prototype/reservoir_aug.py``) prints, with ``flush=True``:

* a header ``K1ReservoirAug: feat_dim=... extra=... trainable=...k frozen=...M N=...`` and, when
  warm-started, ``warm-started from <path> (src_extra_dim=..., new senses zero-init)``;
* every 20 iterations ``[aug] it %4d steps %6.1fM fps %6.0f ep_rew/step %6.3f tilt %5.1fdeg
  <extra>``, where ``<extra>`` is ``env.metric()`` (``compl %4.1f%% lap %.1f`` for
  ``BatchedProcCourse``) or empty for envs without a metric;
* at the end ``[aug] saved <path>``.

The app redirects that stdout to ``training/<name>/logs/app.log`` (see :mod:`app.runs`), and writes
an ``[app] launch ...`` marker line before each launch so a relaunch's output can be told apart from
the previous attempt's. This module is a pure parser over that file: malformed or partial lines are
skipped (never raised), reads are bounded, and the byte cursor only ever advances over complete,
newline-terminated lines (a torn in-flight tail is re-read whole on the next poll).
"""

from __future__ import annotations

import os
import re
from typing import Any

from app import logs

#: Max bytes read per poll (bounds work when a long log is read from the start).
_MAX_READ = 256 * 1024
#: Bytes read from the end of the log to find the latest point / saved line / launch marker.
_TAIL_READ = 64 * 1024

LAUNCH_MARKER = "[app] launch"

_NUM = r"-?\d+(?:\.\d+)?"
_AUG_RE = re.compile(
    rf"^\[aug\] it\s+(\d+)\s+steps\s+({_NUM})M\s+fps\s+({_NUM})\s+ep_rew/step\s+({_NUM})"
    rf"\s+tilt\s+({_NUM})deg(?:\s+(.*))?$"
)
_COMPL_RE = re.compile(rf"compl\s+({_NUM})%\s+lap\s+({_NUM})")
_SAVED_RE = re.compile(r"^\[aug\] saved (.+?)\s*$")
_HEADER_RE = re.compile(r"^K1ReservoirAug: feat_dim=")
_WARM_RE = re.compile(r"^warm-started from ")


def parse_aug_line(line: str) -> dict[str, Any] | None:
    """Parse one ``[aug] it ...`` progress line, or return ``None`` if it is not a complete one.

    Returns ``{it, msteps, fps, ep_rew_per_step, tilt_deg, compl_pct, lap}``; ``compl_pct`` and
    ``lap`` are ``None`` when the env printed no (recognisable) metric. Never raises.
    """
    if not isinstance(line, str):
        return None
    m = _AUG_RE.match(line.strip())
    if m is None:
        return None
    try:
        point: dict[str, Any] = {
            "it": int(m.group(1)),
            "msteps": float(m.group(2)),
            "fps": float(m.group(3)),
            "ep_rew_per_step": float(m.group(4)),
            "tilt_deg": float(m.group(5)),
            "compl_pct": None,
            "lap": None,
        }
    except ValueError:
        return None
    extra = m.group(6) or ""
    cm = _COMPL_RE.search(extra)
    if cm is not None:
        point["compl_pct"] = float(cm.group(1))
        point["lap"] = float(cm.group(2))
    return point


def parse_saved_line(line: str) -> str | None:
    """Return the path from an ``[aug] saved <path>`` line, else ``None``."""
    if not isinstance(line, str):
        return None
    m = _SAVED_RE.match(line.strip())
    return m.group(1) if m else None


def is_header_line(line: str) -> bool:
    """Return ``True`` for the ``K1ReservoirAug: ...`` / ``warm-started from ...`` header lines."""
    s = line.strip()
    return bool(_HEADER_RE.match(s) or _WARM_RE.match(s))


def _read_window(path: str, start: int, end: int) -> bytes:
    with open(path, "rb") as fh:
        fh.seek(start)
        return fh.read(end - start)


def _complete_lines(data: bytes, *, drop_first: bool) -> tuple[list[str], int]:
    """Split ``data`` into complete lines; return ``(lines, bytes_consumed)``."""
    last_nl = data.rfind(b"\n")
    if last_nl < 0:
        return [], 0
    lines = data[: last_nl + 1].decode("utf-8", errors="replace").split("\n")
    lines.pop()  # trailing "" after the final newline
    if drop_first and lines:
        lines = lines[1:]  # started mid-line
    return [ln.rstrip("\r") for ln in lines], last_nl + 1


def current_attempt_tail(log_path: str) -> list[str]:
    """Return the complete lines of the log's tail that belong to the latest launch attempt."""
    if not os.path.isfile(log_path):
        return []
    try:
        size = os.path.getsize(log_path)
        start = max(0, size - _TAIL_READ)
        data = _read_window(log_path, start, size)
    except OSError:
        return []
    lines, _ = _complete_lines(data, drop_first=start > 0)
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].startswith(LAUNCH_MARKER):
            return lines[idx + 1 :]
    return lines


def saved_path_in(lines: list[str]) -> str | None:
    """Return the last ``[aug] saved`` path among ``lines``, else ``None``."""
    found = None
    for line in lines:
        path = parse_saved_line(line)
        if path is not None:
            found = path
    return found


def progress_snapshot(project_root: str, name: str, since: int = 0) -> dict[str, Any]:
    """Return the parsed progress of ``training/<name>/logs/app.log`` from byte offset ``since``.

    Returns ``{latest, points, saved, header, reset, next}``:

    * ``points`` — progress points parsed from complete lines in ``[since, next)`` (bounded; when a
      backlog exceeds the read cap the oldest bytes are skipped);
    * ``reset`` — ``True`` when a launch marker was seen in that window, i.e. ``points`` start a new
      attempt and the caller should drop any older curve;
    * ``latest`` — the latest point of the current attempt (read from the log tail, so it is set on
      every poll even when nothing new arrived), or ``None``;
    * ``saved`` — the ``[aug] saved`` path of the current attempt, or ``None``;
    * ``header`` — the current attempt's header lines;
    * ``next`` — the cursor to pass as ``since`` next time.

    An absent log gives an empty snapshot with ``next == since``.
    """
    path = logs.log_path_for(project_root, name)
    empty = {"latest": None, "points": [], "saved": None, "header": [], "reset": False}
    if not os.path.isfile(path):
        return {**empty, "next": since}
    try:
        size = os.path.getsize(path)
    except OSError:
        return {**empty, "next": since}
    if since < 0 or since > size:
        since = 0  # rotated / truncated → restart from the top

    start = max(since, size - _MAX_READ)
    try:
        data = _read_window(path, start, size)
    except OSError:
        return {**empty, "next": since}
    lines, consumed = _complete_lines(data, drop_first=start > since)
    next_offset = start + consumed if consumed else (start if start > since else since)

    points: list[dict[str, Any]] = []
    reset = False
    for line in lines:
        if line.startswith(LAUNCH_MARKER):
            points = []
            reset = True
            continue
        point = parse_aug_line(line)
        if point is not None:
            points.append(point)

    attempt = current_attempt_tail(path)
    latest = None
    for line in reversed(attempt):
        latest = parse_aug_line(line)
        if latest is not None:
            break
    return {
        "latest": latest,
        "points": points,
        "saved": saved_path_in(attempt),
        "header": [ln.strip() for ln in attempt if is_header_line(ln)],
        "reset": reset,
        "next": next_offset,
    }


__all__ = [
    "LAUNCH_MARKER",
    "current_attempt_tail",
    "is_header_line",
    "parse_aug_line",
    "parse_saved_line",
    "progress_snapshot",
    "saved_path_in",
]
