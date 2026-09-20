"""Windows terminal-capability gate for the live TUI (UC-32, AC-8).

On Windows the full-screen alternate-screen buffer + ~1 Hz redraw require a VT-capable
terminal (Windows Terminal, the VS Code integrated terminal, or a ``conhost`` new enough to
enable virtual-terminal processing). Legacy ``conhost.exe`` cannot drive it, so the TUI must
fail fast with an actionable message *before* it engages an alt-buffer it can't drive, rather
than degrading or crashing.

The gate keys the hard-error on Rich's own reliable signal (``Console.legacy_windows``), never
on environment variables. ``WT_SESSION`` / ``TERM_PROGRAM`` are **corroboration only**: they can
prevent a hard-error (when Rich's read looks incapable but a modern-terminal marker is present),
but can never *cause* one — so a capable-but-unlabelled terminal is never wrongly rejected (no
false-negative, the AC-8 risk note). Non-Windows platforms are always capable here (AC-9: they
are untouched).
"""

from __future__ import annotations

import os
import sys


class TuiUnsupportedError(RuntimeError):
    """The current Windows terminal cannot drive the full-screen live TUI (AC-8)."""


ACTIONABLE_MESSAGE = (
    "The live training dashboard needs a VT-capable terminal on Windows, but this terminal "
    "reports it cannot support the full-screen alternate-screen buffer / per-second redraw "
    "(e.g. legacy conhost.exe). Use Windows Terminal or the VS Code integrated terminal, or "
    "pass --no-tui to run with the plain logger."
)


def assert_tui_capable(*, platform: str | None = None, console=None) -> None:
    """Raise :class:`TuiUnsupportedError` on a Windows terminal that can't drive the TUI (AC-8).

    Parameters (all injectable for hermetic testing — no real Windows/terminal needed, AC-11):

    * ``platform`` — overrides ``sys.platform`` (defaults to it). A no-op unless ``"win32"``.
    * ``console`` — an object exposing ``legacy_windows`` (defaults to a fresh
      :class:`rich.console.Console`, which reads the real terminal). Injecting a stub lets tests
      exercise both the capable and incapable branches.

    Decision (Windows only): hard-error **iff** ``console.legacy_windows`` is truthy AND no
    modern-terminal env marker (``WT_SESSION`` / ``TERM_PROGRAM``) is present. Any other case is
    considered capable, so a modern terminal is never wrongly rejected.
    """
    plat = platform if platform is not None else sys.platform
    if plat != "win32":
        return

    if console is None:
        from rich.console import Console

        console = Console()

    legacy = bool(getattr(console, "legacy_windows", False))
    if not legacy:
        return
    # Rich read the terminal as legacy/incapable. Corroborate: if a modern-terminal marker is
    # present we almost certainly misread — defer to it and do NOT hard-error (no false-negative).
    if os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM"):
        return
    raise TuiUnsupportedError(ACTIONABLE_MESSAGE)


__all__ = ["TuiUnsupportedError", "assert_tui_capable", "ACTIONABLE_MESSAGE"]
