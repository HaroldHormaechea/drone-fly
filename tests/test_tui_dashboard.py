"""UC-32 — ``TrainingDashboard`` orchestration seams (``drone_fly.train.tui.dashboard``).

Covers the dashboard-level behaviours UC-32 adds, all hermetic (AC-11 — no real Windows/terminal;
fd work is strictly saved/restored so it can never corrupt the runner):

* **Bounded redraw failure (E)** — a transient redraw error self-heals; only N consecutive
  *consecutive* failures disable live updates, and the failure warning is rate-limited to
  ``_REDRAW_WARN_INTERVAL`` so a persistent error can't re-flood the logs (which would re-introduce
  the AC-4 spam this use case removes).
* **Single lock** — every model-mutation+redraw path (tick / update / redraw / set_verdict / the
  timer's _refresh_tick) is serialised through the one ``self._lock`` so the timer thread and the
  SB3 callback thread never race on ``build_layout``.
* **Windows native-fd redirect (AC-6/AC-7)** — native fd 1/2 go to ``<logs_dir>/native.log`` and
  BOTH fds are restored on teardown (the challenger's non-blocking note).
* **Log-source select** — Windows reads the log-bridge buffer (AC-7); macOS/Linux the FdLogCapture
  scrollback (AC-9).
* **Capability-gate-first (AC-8)** — an incapable Windows terminal fails fast BEFORE any capture /
  alt-buffer is engaged.
"""

from __future__ import annotations

import logging
import os
import sys
import threading

import pytest

from drone_fly.train.tui.dashboard import (
    _MAX_REDRAW_FAILURES,
    _REDRAW_WARN_INTERVAL,
    TrainingDashboard,
)


class _FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _OkLive:
    def __init__(self) -> None:
        self.renderables: list = []

    def update(self, renderable) -> None:
        self.renderables.append(renderable)


class _FailingLive:
    """A Live whose every ``update`` raises — a persistent redraw error (WinError-like)."""

    def update(self, renderable) -> None:
        raise RuntimeError("redraw exploded")


# --------------------------------------------------------------------------- #
# Bounded redraw failure (E) — N=3 consecutive + rate-limited warning
# --------------------------------------------------------------------------- #
def test_redraw_tolerates_failures_up_to_the_bound_then_disables_live() -> None:
    """E: live updates survive the first N-1 consecutive failures, disabled only on the Nth."""
    assert _MAX_REDRAW_FAILURES == 3
    clock = _FakeClock()
    dash = TrainingDashboard(scheduled_iters=10, capture=False, now=clock)
    dash._live = _FailingLive()

    for i in range(_MAX_REDRAW_FAILURES - 1):
        clock.advance(1.0)
        dash.redraw()
        assert dash._consecutive_redraw_failures == i + 1
        assert dash._live is not None  # not yet disabled

    clock.advance(1.0)
    dash.redraw()  # the Nth consecutive failure
    assert dash._consecutive_redraw_failures == _MAX_REDRAW_FAILURES
    assert dash._live is None  # live updates disabled


def test_redraw_failure_warning_is_rate_limited(caplog) -> None:
    """E: three consecutive failures inside the 30 s window emit only ONE 'redraw failed' warning
    (rate-limited) plus the single 'disabling live updates' notice — never a per-failure flood."""
    clock = _FakeClock()
    dash = TrainingDashboard(scheduled_iters=10, capture=False, now=clock)
    dash._live = _FailingLive()

    with caplog.at_level(logging.WARNING, logger="drone_fly.train.tui.dashboard"):
        for _ in range(_MAX_REDRAW_FAILURES):
            clock.advance(1.0)  # all within the _REDRAW_WARN_INTERVAL (30 s) window
            dash.redraw()

    failed_warnings = [r for r in caplog.records if "redraw failed" in r.message]
    disabling = [r for r in caplog.records if "disabling live updates" in r.message]
    assert len(failed_warnings) == 1  # rate-limited: NOT one per failure (no AC-4 re-flood)
    assert len(disabling) == 1
    assert _REDRAW_WARN_INTERVAL == 30.0


def test_transient_redraw_failure_self_heals_and_does_not_disable_live() -> None:
    """E: a blip (2 failures) followed by a success resets the counter, so live stays enabled —
    persistent-only disabling, transient blips self-heal."""

    class _FlakyLive:
        def __init__(self) -> None:
            self.calls = 0

        def update(self, renderable) -> None:
            self.calls += 1
            if self.calls <= 2:
                raise RuntimeError("blip")

    clock = _FakeClock()
    dash = TrainingDashboard(scheduled_iters=10, capture=False, now=clock)
    dash._live = _FlakyLive()

    clock.advance(1.0)
    dash.redraw()  # fail 1
    clock.advance(1.0)
    dash.redraw()  # fail 2
    assert dash._consecutive_redraw_failures == 2
    clock.advance(1.0)
    dash.redraw()  # success -> reset

    assert dash._consecutive_redraw_failures == 0
    assert dash._live is not None  # never disabled (blip self-healed)


# --------------------------------------------------------------------------- #
# Single lock — serialises every mutation+redraw path
# --------------------------------------------------------------------------- #
def test_all_mutation_paths_go_through_the_single_lock() -> None:
    """The timer thread and the SB3 callback thread must never race on build_layout: every
    model-mutation+redraw entry point (tick / update / redraw / set_verdict / _refresh_tick) is
    serialised through the one ``self._lock``."""

    class _TrackingLock:
        def __init__(self) -> None:
            self._l = threading.RLock()
            self.enters = 0

        def __enter__(self):
            self.enters += 1
            return self._l.__enter__()

        def __exit__(self, *a):
            return self._l.__exit__(*a)

    dash = TrainingDashboard(scheduled_iters=10, capture=False)
    lock = _TrackingLock()
    dash._lock = lock
    dash._live = _OkLive()

    dash.tick(elapsed_seconds=1.0, rollout_steps=0, rollout_target=0)
    dash.update(n_updates=1, elapsed_seconds=1.0)
    dash.redraw()
    dash.set_verdict(None)
    dash._timer_start_monotonic = 0.0
    dash._refresh_tick(1.0)

    assert lock.enters == 5  # every path acquired the lock exactly once (no unguarded mutation)


# --------------------------------------------------------------------------- #
# Windows native-fd redirect (AC-6/AC-7) — file target + both-fd restore
# --------------------------------------------------------------------------- #
def _fd_identity(fd: int) -> tuple[int, int]:
    st = os.fstat(fd)
    return (st.st_dev, st.st_ino)


def test_windows_native_redirect_writes_to_native_log_and_restores_both_fds(tmp_path) -> None:
    """AC-7: pybullet's native banner (fd 1/2 writes) lands in <logs_dir>/native.log, and teardown
    restores BOTH fd 1 and fd 2 to the original terminal (the challenger's non-blocking note)."""
    dash = TrainingDashboard(capture=False, logs_dir=str(tmp_path))
    dash._is_win = True

    before1, before2 = _fd_identity(1), _fd_identity(2)
    saved1, saved2 = os.dup(1), os.dup(2)  # our own safety net, independent of the dashboard's
    native_id1 = native_id2 = None
    restored1 = restored2 = None
    try:
        dash._start_windows_capture()
        native_id1, native_id2 = _fd_identity(1), _fd_identity(2)
        os.write(1, b"pybullet build time: 42\n")
        os.write(2, b"Version = 3.2.5\n")
        dash._restore_windows_capture()
        restored1, restored2 = _fd_identity(1), _fd_identity(2)
    finally:
        os.dup2(saved1, 1)
        os.dup2(saved2, 2)
        os.close(saved1)
        os.close(saved2)

    # while redirected, fd 1/2 pointed at native.log (not the terminal)
    assert native_id1 != before1
    assert native_id2 != before2
    # both fds restored to the ORIGINAL terminal on teardown (AC-6/AC-7, both-fd note)
    assert restored1 == before1
    assert restored2 == before2
    # all Windows capture state cleared
    assert dash._win_saved_fd1 is None
    assert dash._win_saved_fd2 is None
    assert dash._win_native_fd is None
    assert dash._win_display is None
    # the native banner landed in the file, off the screen (AC-7)
    native_log = os.path.join(str(tmp_path), "native.log")
    assert os.path.exists(native_log)
    content = open(native_log, encoding="utf-8").read()
    assert "pybullet build time: 42" in content
    assert "Version = 3.2.5" in content


# --------------------------------------------------------------------------- #
# Log-source select — logbridge (Windows) vs FdLogCapture (macOS/Linux)
# --------------------------------------------------------------------------- #
def test_log_lines_prefers_the_logbridge_buffer_when_present() -> None:
    """AC-7: on Windows the pane reads the Python-record buffer from the log bridge."""

    class _Bridge:
        def lines(self):
            return ["INFO drone_fly: a", "WARNING drone_fly: b"]

    dash = TrainingDashboard(capture=False)
    dash._logbridge = _Bridge()
    assert dash._log_lines() == ["INFO drone_fly: a", "WARNING drone_fly: b"]


def test_log_lines_uses_fd_capture_scrollback_without_logbridge() -> None:
    """AC-9: macOS/Linux keep the FdLogCapture pipe scrollback, unchanged."""

    class _Cap:
        def lines(self):
            return ["native scrollback line"]

    dash = TrainingDashboard(capture=False)
    dash._capture = _Cap()
    dash._logbridge = None
    assert dash._log_lines() == ["native scrollback line"]


def test_log_lines_empty_when_no_source() -> None:
    dash = TrainingDashboard(capture=False)
    dash._capture = None
    dash._logbridge = None
    assert dash._log_lines() == []


# --------------------------------------------------------------------------- #
# Capability-gate-first (AC-8) — fail fast BEFORE any capture / alt-buffer
# --------------------------------------------------------------------------- #
def test_start_fails_fast_before_engaging_capture_on_incapable_windows(monkeypatch) -> None:
    """AC-8 ordering: on an incapable Windows terminal, _start raises from assert_tui_capable
    BEFORE _start_windows_capture / any Console/Live is built — so no alt-buffer is ever engaged."""
    from drone_fly.train.tui.capability import TuiUnsupportedError

    monkeypatch.setattr(sys, "platform", "win32")

    def _raise(*a, **k):
        raise TuiUnsupportedError("legacy conhost")

    monkeypatch.setattr("drone_fly.train.tui.capability.assert_tui_capable", _raise)

    dash = TrainingDashboard(capture=False)
    touched = {"capture": False, "console": False}
    monkeypatch.setattr(
        dash, "_start_windows_capture", lambda: touched.__setitem__("capture", True)
    )
    monkeypatch.setattr(
        dash, "_build_windows_console", lambda display: touched.__setitem__("console", True)
    )

    with pytest.raises(TuiUnsupportedError):
        dash._start()

    assert touched["capture"] is False  # never redirected fds / engaged the screen
    assert touched["console"] is False
    assert dash._live is None


# --------------------------------------------------------------------------- #
# UC-33 Item 4 — Windows terminal-size resolution (hermetic, AC-18)
#
# The size-resolution helper is called directly with a sentinel saved fd (no sys.platform
# dependency): it must read the SAVED real terminal fd (not the redirected bare fd 1), degrade
# gracefully to None on failure, and never lock a fixed size on the constructed Console.
# --------------------------------------------------------------------------- #
_SENTINEL_SAVED_FD = 987654  # a distinctive value that is unmistakably NOT the bare fd 1


def test_resolve_win_size_reads_the_saved_fd_not_bare_fd_1(monkeypatch) -> None:
    """AC-15/AC-18: the helper queries ``os.get_terminal_size(self._win_saved_fd1)`` — the SAVED
    real terminal fd — not the bare fd 1 (which UC-32 repointed at native.log)."""
    calls: list = []

    def _fake_get_terminal_size(fd):
        calls.append(fd)
        return os.terminal_size((203, 51))

    monkeypatch.setattr(os, "get_terminal_size", _fake_get_terminal_size)

    dash = TrainingDashboard(capture=False)
    dash._win_saved_fd1 = _SENTINEL_SAVED_FD
    size = dash._resolve_win_terminal_size()

    assert size == (203, 51)  # (columns, lines)
    assert calls == [_SENTINEL_SAVED_FD]  # queried the saved fd exactly once...
    assert 1 not in calls  # ...and never the bare, redirected fd 1


def test_resolve_win_size_degrades_to_none_on_oserror(monkeypatch) -> None:
    """AC-18 graceful degrade: if ``os.get_terminal_size`` on the saved fd raises OSError (e.g. the
    fd is not a tty), the helper returns None so callers fall back to Rich self-detection, not a
    crash."""

    def _raise(fd):
        raise OSError("not a tty")

    monkeypatch.setattr(os, "get_terminal_size", _raise)

    dash = TrainingDashboard(capture=False)
    dash._win_saved_fd1 = _SENTINEL_SAVED_FD
    assert dash._resolve_win_terminal_size() is None


def test_resolve_win_size_returns_none_when_saved_fd_missing(monkeypatch) -> None:
    """AC-18: with no saved fd (helper never captured one), the helper returns None WITHOUT even
    calling ``os.get_terminal_size`` — it never falls back to probing the bare fd 1."""
    called = {"n": 0}

    def _fake(fd):
        called["n"] += 1
        return os.terminal_size((80, 25))

    monkeypatch.setattr(os, "get_terminal_size", _fake)

    dash = TrainingDashboard(capture=False)
    dash._win_saved_fd1 = None
    assert dash._resolve_win_terminal_size() is None
    assert called["n"] == 0  # never probed any fd


def test_build_windows_console_applies_size_without_locking_a_fixed_size(monkeypatch) -> None:
    """AC-16/AC-18: the Windows Console is built WITHOUT a locking ``width``/``height`` kwarg (which
    would freeze the size so resizes aren't followed); the resolved size is applied afterwards via
    the mutable ``console.size`` property instead."""
    import rich.console as rc

    captured: dict = {}
    real_console_cls = rc.Console

    class _SpyConsole(real_console_cls):
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = dict(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(rc, "Console", _SpyConsole)
    monkeypatch.setattr(os, "get_terminal_size", lambda fd: os.terminal_size((177, 44)))

    dash = TrainingDashboard(capture=False)
    dash._win_saved_fd1 = _SENTINEL_SAVED_FD
    console = dash._build_windows_console(display=None)

    # No locking fixed size passed to the constructor...
    assert "width" not in captured["kwargs"]
    assert "height" not in captured["kwargs"]
    # ...but force_terminal is still engaged (so screen=True works over the saved fd).
    assert captured["kwargs"].get("force_terminal") is True
    # ...and the resolved size was applied via the mutable property afterwards.
    assert console.size == (177, 44)


def test_build_windows_console_leaves_size_to_rich_when_unresolved(monkeypatch) -> None:
    """AC-18: if the size can't be resolved (no saved fd), the console is built with no fixed size —
    Rich self-detects rather than being locked to a wrong value."""
    monkeypatch.setattr(os, "get_terminal_size", lambda fd: os.terminal_size((80, 25)))

    dash = TrainingDashboard(capture=False)
    dash._win_saved_fd1 = None  # helper -> None
    console = dash._build_windows_console(display=None)
    # Constructed successfully; size was NOT forced from our helper (it returned None).
    assert console is not None
