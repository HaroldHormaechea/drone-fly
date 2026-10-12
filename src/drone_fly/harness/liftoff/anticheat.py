"""Refuse to send any input while a kernel anti-cheat game is running (League / Riot Vanguard).

The unattended Liftoff tooling injects a virtual gamepad, mouse clicks and key presses. The user
allows that only with League of Legends and Riot Vanguard closed, so every entry point that
drives the game (``sysid liftoff`` on the live plant, ``ui select-drone``, the sysid reset between
blocks) calls :func:`require_no_anticheat` first. The process list comes from
``tasklist /FO CSV /NH``; the enumerator is injectable for tests.
"""

from __future__ import annotations

import csv
import fnmatch
import io
import subprocess
from collections.abc import Callable, Iterable

from drone_fly.harness.errors import HarnessConfigError

#: Case-insensitive image-name patterns (with or without ``.exe``).
ANTICHEAT_PATTERNS = (
    "leagueclient*",
    "league of legends*",
    "riotclient*",
    "vgc",
    "vgtray",
)


class AntiCheatRunningError(HarnessConfigError):
    """A League / Riot / Vanguard process is running: no input may be sent."""

    def __init__(self, processes: list[str]) -> None:
        self.processes = processes
        super().__init__(
            "refusing to drive Liftoff while anti-cheat software is running: "
            f"{', '.join(processes)} (close League of Legends / the Riot client / Vanguard first)"
        )


def anticheat_processes(names: Iterable[str]) -> list[str]:
    """The names in ``names`` that match :data:`ANTICHEAT_PATTERNS` (input order, no repeats)."""
    out: list[str] = []
    for name in names:
        stem = name.strip()
        low = stem.lower()
        if low.endswith(".exe"):
            low = low[: -len(".exe")]
        if any(fnmatch.fnmatchcase(low, pat) for pat in ANTICHEAT_PATTERNS) and stem not in out:
            out.append(stem)
    return out


def running_process_names() -> list[str]:
    """Image names of every running process (``tasklist /FO CSV /NH``; Windows only)."""
    try:
        text = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, check=False
        ).stdout
    except FileNotFoundError as exc:
        raise HarnessConfigError(
            "cannot list processes (tasklist not found; Windows only)"
        ) from exc
    return [row[0] for row in csv.reader(io.StringIO(text)) if row]


def require_no_anticheat(enumerator: Callable[[], Iterable[str]] | None = None) -> None:
    """Raise :class:`AntiCheatRunningError` if any anti-cheat process is running."""
    found = anticheat_processes((enumerator or running_process_names)())
    if found:
        raise AntiCheatRunningError(found)
