"""Recording enumeration + server-side gunzip for the embedded viewer (UC-61 AC8).

The training/eval recorder writes ``episode_<n>.json`` or ``episode_<n>.json.gz`` under
``training/<name>/recordings/``. The desktop app lists them and, on demand, reads one and returns
its **parsed JSON** to the front-end, decompressing ``.gz`` **server-side** (per the owner
decision) so playback never depends on the system webview's ``DecompressionStream``. The front-end
hands the parsed object to the reused ``viz/viewer.js`` via ``window.loadDocument``.

Everything is filesystem-only and path-safe: episode identifiers are integers, and the resolved
path is confirmed to sit inside the run's recordings directory before it is opened.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from typing import Any

_EPISODE_RE = re.compile(r"^episode_(\d+)\.json(\.gz)?$")


def recordings_dir_for(project_root: str, name: str) -> str:
    """Return ``<project_root>/training/<name>/recordings``."""
    return os.path.join(project_root, "training", name, "recordings")


def list_recordings(project_root: str, name: str) -> list[dict[str, Any]]:
    """List recorded episodes for a run, sorted by episode index (AC8).

    Each entry: ``{"episode": int, "file": <basename>, "gzip": bool, "bytes": int}``. A directory
    with no recordings (or none yet) returns ``[]``.
    """
    directory = recordings_dir_for(project_root, name)
    if not os.path.isdir(directory):
        return []
    out: list[dict[str, Any]] = []
    for fn in os.listdir(directory):
        m = _EPISODE_RE.match(fn)
        if not m:
            continue
        full = os.path.join(directory, fn)
        try:
            size = os.path.getsize(full)
        except OSError:
            size = 0
        out.append(
            {
                "episode": int(m.group(1)),
                "file": fn,
                "gzip": bool(m.group(2)),
                "bytes": size,
            }
        )
    out.sort(key=lambda e: e["episode"])
    return out


def _resolve_episode_path(project_root: str, name: str, episode: int) -> str:
    """Resolve the on-disk path for ``episode`` (prefers plain ``.json``, then ``.json.gz``).

    Raises :class:`FileNotFoundError` if neither exists. The resolved path is confirmed to live
    inside the run's recordings directory (defence against a crafted ``name``/``episode``).
    """
    directory = os.path.realpath(recordings_dir_for(project_root, name))
    for candidate in (f"episode_{episode}.json", f"episode_{episode}.json.gz"):
        path = os.path.realpath(os.path.join(directory, candidate))
        if os.path.commonpath([directory, path]) != directory:
            raise FileNotFoundError(f"recording path escapes the recordings dir: {candidate}")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"no recording for episode {episode} of run {name!r}")


def load_recording(project_root: str, name: str, episode: int) -> dict[str, Any]:
    """Read + parse one episode recording, gunzipping ``.gz`` server-side (AC8).

    Returns the parsed JSON object (the document ``window.loadDocument`` expects). Raises
    :class:`FileNotFoundError` when absent and :class:`ValueError` when the file is not valid JSON.
    """
    path = _resolve_episode_path(project_root, name, episode)
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            text = fh.read()
    else:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    try:
        return json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"recording for episode {episode} of run {name!r} is not valid JSON"
        ) from exc


__all__ = [
    "recordings_dir_for",
    "list_recordings",
    "load_recording",
]
