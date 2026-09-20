"""UC-32 AC-8 — Windows terminal-capability gate (``drone_fly.train.tui.capability``).

The full-screen alternate-screen buffer + ~1 Hz redraw need a VT-capable Windows terminal. On an
incapable one (legacy ``conhost.exe``) the TUI must fail fast with an actionable error *before* it
engages an alt-buffer it can't drive — but a capable-but-unlabelled terminal must NEVER be wrongly
rejected (no false-negative, the AC-8 risk note).

All hermetic and platform-mockable (AC-11): ``sys.platform`` is injected via the ``platform=``
argument and Rich's capability read is injected via a ``console`` stub — no real Windows, no real
terminal. The decision keys on Rich's own ``legacy_windows`` signal; ``WT_SESSION`` /
``TERM_PROGRAM`` are corroboration-only and can only *prevent* a hard-error, never cause one.
"""

from __future__ import annotations

import types

import pytest

from drone_fly.train.tui.capability import (
    ACTIONABLE_MESSAGE,
    TuiUnsupportedError,
    assert_tui_capable,
)


def _console(*, legacy_windows: bool):
    """A Rich-Console stand-in exposing only the ``legacy_windows`` signal the gate reads."""
    return types.SimpleNamespace(legacy_windows=legacy_windows)


@pytest.fixture(autouse=True)
def _clean_terminal_env(monkeypatch):
    """Ensure no ambient modern-terminal marker leaks in from the CI environment."""
    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.delenv("TERM_PROGRAM", raising=False)


# --------------------------------------------------------------------------- #
# Non-Windows is always capable (AC-9: macOS/Linux untouched)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("plat", ["linux", "darwin"])
def test_non_windows_is_always_capable_even_with_a_legacy_console(plat) -> None:
    """Off win32 the gate is a no-op regardless of the console signal — it never raises."""
    assert assert_tui_capable(platform=plat, console=_console(legacy_windows=True)) is None


def test_non_windows_does_not_even_construct_a_console() -> None:
    """Off win32 the gate returns before touching ``console`` (None is fine, no real Console)."""
    assert assert_tui_capable(platform="linux", console=None) is None


# --------------------------------------------------------------------------- #
# Windows — capable terminals never raise
# --------------------------------------------------------------------------- #
def test_windows_capable_console_does_not_raise() -> None:
    """AC-8: a modern Windows terminal (Rich reads it as NON-legacy) is accepted silently."""
    assert assert_tui_capable(platform="win32", console=_console(legacy_windows=False)) is None


def test_windows_bare_but_capable_terminal_is_not_rejected() -> None:
    """No false-negative: legacy=False with NO env markers is still capable (bare-but-capable)."""
    assert assert_tui_capable(platform="win32", console=_console(legacy_windows=False)) is None


# --------------------------------------------------------------------------- #
# Windows — incapable terminal fails fast (AC-8)
# --------------------------------------------------------------------------- #
def test_windows_legacy_console_without_markers_raises_actionable_error() -> None:
    """AC-8: legacy conhost (Rich reads legacy=True) with no modern marker -> hard-error."""
    with pytest.raises(TuiUnsupportedError) as ei:
        assert_tui_capable(platform="win32", console=_console(legacy_windows=True))
    msg = str(ei.value)
    assert msg == ACTIONABLE_MESSAGE
    # actionable: names the remedy (Windows Terminal / --no-tui), never a bare traceback.
    assert "Windows Terminal" in msg
    assert "--no-tui" in msg


# --------------------------------------------------------------------------- #
# Windows — env markers are corroboration-only: they can PREVENT a false-negative
# --------------------------------------------------------------------------- #
def test_windows_terminal_marker_prevents_false_negative(monkeypatch) -> None:
    """WT_SESSION present + a (mis)read legacy=True -> defer to the marker, do NOT hard-error."""
    monkeypatch.setenv("WT_SESSION", "abc-123")
    assert assert_tui_capable(platform="win32", console=_console(legacy_windows=True)) is None


def test_vscode_term_program_marker_prevents_false_negative(monkeypatch) -> None:
    """TERM_PROGRAM=vscode present + legacy=True -> defer to the marker, do NOT hard-error."""
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    assert assert_tui_capable(platform="win32", console=_console(legacy_windows=True)) is None


def test_env_markers_cannot_cause_a_hard_error_on_a_capable_terminal(monkeypatch) -> None:
    """The markers only ever PREVENT a raise. On a capable (legacy=False) console, their absence
    (or presence) is irrelevant — never a hard-error. Guards against markers *causing* rejection."""
    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    assert assert_tui_capable(platform="win32", console=_console(legacy_windows=False)) is None
