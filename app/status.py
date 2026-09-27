"""Read the training status stream the emitter writes (UC-61 AC6).

The :class:`~drone_fly.train.status_emitter.StatusEmitterCallback` appends one JSON object per
rollout to ``training/<name>/status.jsonl``. This module reads that file for the backend's
progress endpoints, plus a CSV-tail fallback from SB3's ``progress.csv`` when no JSONL exists yet
(an older run, or the very first rollout before the first line lands).

All reads are **partial-write safe**: only complete, newline-terminated lines are parsed, and each
line is ``json.loads``-guarded so a half-flushed final line (the writer appends ``line + "\\n"``
in one ``write``, but a reader may still catch a torn tail on some filesystems) is skipped rather
than raised. The file is treated as append-only, so line index doubles as a stream cursor.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any


def status_path_for(project_root: str, name: str) -> str:
    """Return ``<project_root>/training/<name>/status.jsonl``."""
    return os.path.join(project_root, "training", name, "status.jsonl")


def progress_csv_for(project_root: str, name: str) -> str:
    """Return ``<project_root>/training/<name>/logs/progress.csv`` (SB3's CSV logger output)."""
    return os.path.join(project_root, "training", name, "logs", "progress.csv")


def read_status_lines(status_path: str, *, since: int = 0) -> list[dict[str, Any]]:
    """Return parsed status records at line index ``>= since`` (partial-write safe).

    Only complete newline-terminated lines are considered; a trailing line without a newline (an
    in-flight append) is ignored. Any line that fails to parse as JSON is skipped, never raised.
    Each record is annotated with its 0-based ``line`` index so the caller can advance a cursor.
    """
    if not os.path.isfile(status_path):
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(status_path, encoding="utf-8") as fh:
            content = fh.read()
    except OSError:
        return []
    # Split on newlines; a complete line is one that was terminated. ``splitlines`` would also
    # split a torn tail — instead, drop everything after the last newline (the incomplete tail).
    last_nl = content.rfind("\n")
    if last_nl < 0:
        return []
    complete = content[: last_nl + 1]
    for idx, raw in enumerate(complete.splitlines()):
        if idx < since:
            continue
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if isinstance(rec, dict):
            rec["line"] = idx
            out.append(rec)
    return out


def latest_status(status_path: str) -> dict[str, Any] | None:
    """Return the most recent complete status record, or ``None`` if there are none."""
    lines = read_status_lines(status_path)
    return lines[-1] if lines else None


def read_progress_csv(csv_path: str) -> list[dict[str, Any]]:
    """Parse SB3's ``progress.csv`` into a list of numeric-coerced row dicts (CSV fallback, AC6).

    Best-effort: a missing/short file returns ``[]``. Empty cells become ``None`` and numeric-
    looking cells are coerced to ``float`` so the front-end can plot without re-parsing.
    """
    if not os.path.isfile(csv_path):
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                coerced: dict[str, Any] = {}
                for key, value in row.items():
                    if value is None or value == "":
                        coerced[key] = None
                        continue
                    try:
                        coerced[key] = float(value)
                    except ValueError:
                        coerced[key] = value
                rows.append(coerced)
    except OSError:
        return []
    return rows


def progress_snapshot(project_root: str, name: str, *, since: int = 0) -> dict[str, Any]:
    """Assemble a progress snapshot for one run: JSONL primary, CSV fallback (AC6).

    Returns ``{"source": "jsonl"|"csv"|"none", "latest": <record|None>, "lines": [...],
    "next": <cursor>}``. When JSONL exists it is authoritative; otherwise the CSV rows are
    surfaced under ``lines`` with ``source == "csv"`` so the UI can still draw a curve.
    """
    sp = status_path_for(project_root, name)
    lines = read_status_lines(sp, since=since)
    if lines or os.path.isfile(sp):
        latest = lines[-1] if lines else (latest_status(sp))
        next_cursor = (lines[-1]["line"] + 1) if lines else since
        return {"source": "jsonl", "latest": latest, "lines": lines, "next": next_cursor}
    csv_rows = read_progress_csv(progress_csv_for(project_root, name))
    if csv_rows:
        return {
            "source": "csv",
            "latest": csv_rows[-1],
            "lines": csv_rows[since:] if since < len(csv_rows) else [],
            "next": len(csv_rows),
        }
    return {"source": "none", "latest": None, "lines": [], "next": since}


__all__ = [
    "status_path_for",
    "progress_csv_for",
    "read_status_lines",
    "latest_status",
    "read_progress_csv",
    "progress_snapshot",
]
