"""Live-dashboard orchestration for the training TUI (UC-22, extended by UC-32).

:class:`TrainingDashboard` binds the halves together: it owns the pure
:class:`~drone_fly.train.tui.metrics.DashboardModel`, an
:class:`~drone_fly.train.tui.capture.FdLogCapture` (macOS/Linux) *or* a direct Windows
native-fd redirect (UC-32), and (inside :meth:`live_session`) a Rich ``Live``. The callback
feeds ``model`` and calls the lock-guarded :meth:`tick` / :meth:`update` / :meth:`redraw`; the
status bar reads ``model.latest_verdict`` via the injected :meth:`set_verdict`.

``live_session`` is the single entry/teardown boundary. On **macOS/Linux** it behaves exactly
as it did under UC-22/UC-30 — fd capture then Rich ``Live`` (drawing to the *original* terminal
via ``capture.display_stdout``), torn down ``Live``-first then fds — so behaviour is
byte-identical (AC-9). On **Windows** (UC-32) it additionally, all ``sys.platform=="win32"``
guarded:

* fails fast via :func:`~drone_fly.train.tui.capability.assert_tui_capable` **before** any
  ``Console``/``Live`` is built, so an incapable terminal never engages an alt-buffer (AC-8);
* redirects native fd 1/2 to ``<logs_dir>/native.log`` (saving BOTH real fds for restore) so
  pybullet's native banner goes to a file, not the screen (AC-7), while Rich draws to the saved
  terminal fd with ``force_terminal=True`` so ``screen=True`` engages full-screen (AC-6);
* installs a :class:`~drone_fly.train.tui.logbridge.LogBridge` that silences the console log
  handlers (no ``WinError 1`` — AC-4) and routes Python records into the logs pane (AC-7);
* runs a ~1 Hz background refresh timer so the elapsed clock + liveness tick even during
  ``PPO.train()`` (AC-5).

A single lock serialises every model-mutation+redraw section (callback tick/update, verdict,
timer clock-tick) so the timer thread and the SB3 callback thread never race on ``build_layout``
(uncontended, hence invisible, on macOS/Linux). Everything is defensive: a Rich failure degrades
to plain logging and never crashes training; redraw failures are tolerated up to N consecutive
before live updates are disabled (with a rate-limited warning) so a transient blip self-heals.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import contextmanager

from drone_fly.train.tui.capture import FdLogCapture
from drone_fly.train.tui.metrics import DashboardModel

logger = logging.getLogger(__name__)

#: Rich Live refresh cap; the callback drives explicit redraws once per rollout anyway.
_REFRESH_PER_SECOND = 4
#: Windows refresh-timer cadence (seconds) — the ~1 Hz clock/liveness tick (AC-5).
_TIMER_INTERVAL = 1.0
#: Consecutive redraw failures tolerated before live updates are disabled (E).
_MAX_REDRAW_FAILURES = 3
#: Minimum wall-clock gap between redraw-failure warnings, so a persistent error can't re-flood
#: the logs and re-introduce the AC-4 spam (E).
_REDRAW_WARN_INTERVAL = 30.0


class TrainingDashboard:
    """Own the model + capture + Rich Live, exposing lock-guarded ``tick``/``update``/``redraw``."""

    def __init__(
        self,
        *,
        scheduled_iters: int = 0,
        capture: bool = True,
        n_envs: int = 1,
        backend: str = "dummy",
        logs_dir: str | None = None,
        now=None,
        sleep=None,
    ) -> None:
        # UC-26 AC-11: forward the resolved rollout parallelism into the pure model so the
        # values panel can render it (static per run).
        self.model = DashboardModel(scheduled_iters=scheduled_iters, n_envs=n_envs, backend=backend)
        self._capture = FdLogCapture() if capture else None
        self._live = None
        self._console = None
        self._enabled = True
        # UC-32: single lock guarding every model-mutation+redraw section (uncontended on
        # macOS/Linux, where no timer thread exists — so no behaviour change, AC-9).
        self._lock = threading.RLock()

        self._logs_dir = logs_dir
        #: Injectable clocks for the Windows refresh timer (hermetic tests, AC-11).
        self._now = now if now is not None else time.monotonic
        self._sleep = sleep if sleep is not None else time.sleep

        # Windows-only state (created in _start when on win32; all no-ops elsewhere).
        self._is_win = False
        self._logbridge = None
        self._timer: threading.Thread | None = None
        self._timer_stop: threading.Event | None = None
        self._timer_start_monotonic: float | None = None
        self._win_saved_fd1: int | None = None
        self._win_saved_fd2: int | None = None
        self._win_native_fd: int | None = None
        self._win_display = None

        # UC-32 (E): bounded redraw-failure tracking.
        self._consecutive_redraw_failures = 0
        self._last_redraw_warning: float | None = None

    # -- verdict seam (injected into the existing HealthCallback) --------------------------

    def set_verdict(self, verdict) -> None:
        """Store the latest health verdict; never raises into the callback."""
        try:
            with self._lock:
                self.model.set_verdict(verdict)
        except Exception as exc:  # noqa: BLE001 - a consumer must not crash training
            logger.warning("dashboard set_verdict failed: %s", exc)

    # -- callback-facing, lock-guarded mutation+redraw (UC-32) -----------------------------

    def tick(self, **kwargs) -> None:
        """Ingest an intra-rollout heartbeat then redraw, atomically under the lock (UC-30/32)."""
        with self._lock:
            self.model.tick(**kwargs)
            self._redraw_locked()

    def update(self, **kwargs) -> None:
        """Ingest a rollout-end snapshot then redraw, atomically under the lock (UC-22/32)."""
        with self._lock:
            self.model.update(**kwargs)
            self._redraw_locked()

    # -- live session ----------------------------------------------------------------------

    @contextmanager
    def live_session(self):
        """Enter capture + Rich Live; guarantee ordered teardown even on Ctrl-C (AC8)."""
        self._start()
        try:
            yield self
        finally:
            self._stop()

    def _start(self) -> None:
        self._is_win = sys.platform == "win32"

        # AC-8: on Windows, fail FAST before any Console/Live/alt-buffer is built. A hard-error
        # here propagates straight out of live_session() (no partial screen state to unwind).
        if self._is_win:
            from drone_fly.train.tui.capability import assert_tui_capable

            assert_tui_capable()

        # 1) Capture first (so the Live console can draw to the saved real terminal).
        if self._is_win:
            self._start_windows_capture()
            self._install_logbridge()
        elif self._capture is not None:
            try:
                self._capture.__enter__()
            except Exception as exc:  # noqa: BLE001
                logger.warning("fd log capture unavailable, continuing without it: %s", exc)
                self._capture = None

        # 2) Rich Live, defensively (degrade to plain logging on any failure, AC8).
        try:
            from rich.console import Console
            from rich.live import Live

            from drone_fly.train.tui.render import build_layout

            if self._is_win:
                self._console = self._build_windows_console(self._win_display)
            else:
                display = getattr(self._capture, "display_stdout", None) if self._capture else None
                self._console = Console(file=display) if display is not None else Console()
            self._live = Live(
                build_layout(self.model, self._log_lines()),
                console=self._console,
                screen=True,
                refresh_per_second=_REFRESH_PER_SECOND,
                transient=False,
            )
            self._live.__enter__()
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the run
            logger.warning("Live dashboard unavailable, falling back to logs: %s", exc)
            self._live = None
            self._enabled = False

        # 3) Windows ~1 Hz refresh timer, only once Live is actually up (AC-5).
        if self._is_win and self._live is not None:
            self._start_timer()

    def _stop(self) -> None:
        # 0) Stop the refresh timer FIRST so no redraw races the teardown.
        self._stop_timer()

        # 1) Tear down Live BEFORE restoring fds, so its final frame flushes to the terminal.
        if self._live is not None:
            try:
                self._live.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Live teardown error (ignored): %s", exc)
            self._live = None
        self._console = None

        # 2) Restore capture. On Windows this restores BOTH fd 1 and fd 2 (challenger's note) and
        # then un-installs the log bridge (re-attaching the console handlers).
        if self._is_win:
            self._restore_windows_capture()
            self._uninstall_logbridge()
        elif self._capture is not None:
            try:
                self._capture.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("fd capture teardown error (ignored): %s", exc)

    # -- redraw ----------------------------------------------------------------------------

    def redraw(self) -> None:
        """Rebuild the layout under the lock; never crashes training."""
        with self._lock:
            self._redraw_locked()

    def _redraw_locked(self) -> None:
        """Rebuild the layout from the current model + logs. Caller must hold ``self._lock``.

        Bounded-failure policy (E): a transient redraw error is tolerated; only after
        :data:`_MAX_REDRAW_FAILURES` consecutive failures are live updates disabled. Warnings are
        rate-limited to :data:`_REDRAW_WARN_INTERVAL` so a persistent error cannot re-flood the
        logs (which would re-introduce the AC-4 spam this use case removes).
        """
        if self._live is None:
            return
        try:
            from drone_fly.train.tui.render import build_layout

            self._live.update(build_layout(self.model, self._log_lines()))
            self._consecutive_redraw_failures = 0
        except Exception as exc:  # noqa: BLE001
            self._consecutive_redraw_failures += 1
            now = self._now()
            if (
                self._last_redraw_warning is None
                or (now - self._last_redraw_warning) >= _REDRAW_WARN_INTERVAL
            ):
                logger.warning(
                    "dashboard redraw failed (%d consecutive): %s",
                    self._consecutive_redraw_failures,
                    exc,
                )
                self._last_redraw_warning = now
            if self._consecutive_redraw_failures >= _MAX_REDRAW_FAILURES:
                logger.warning(
                    "disabling live updates after %d consecutive redraw failures",
                    self._consecutive_redraw_failures,
                )
                self._live = None

    def _log_lines(self) -> list[str]:
        # Windows: Python records come from the log bridge buffer (AC-7). macOS/Linux: the
        # FdLogCapture pipe scrollback, unchanged (AC-9).
        if self._logbridge is not None:
            return self._logbridge.lines()
        return self._capture.lines() if self._capture is not None else []

    # -- Windows refresh timer (AC-5) ------------------------------------------------------

    def _refresh_tick(self, now: float) -> None:
        """Advance the live clock from a monotonic ``now`` and redraw (pure; hermetically tested).

        Guarded by the caller (started only on win32). Uses the timer's own start reference so
        the elapsed clock keeps advancing between rollout boundaries; the callback's rollout-end
        ``update`` remains the authoritative elapsed value.
        """
        if self._timer_start_monotonic is None:
            return
        with self._lock:
            self.model.tick_clock(elapsed_seconds=now - self._timer_start_monotonic)
            self._redraw_locked()

    def _start_timer(self) -> None:
        self._timer_start_monotonic = self._now()
        self._timer_stop = threading.Event()
        stop = self._timer_stop

        def _run() -> None:
            # Event.wait returns True once set (stop requested), False on timeout — so this loops
            # every _TIMER_INTERVAL until _stop_timer sets the event.
            while not stop.wait(_TIMER_INTERVAL):
                try:
                    self._refresh_tick(self._now())
                except Exception as exc:  # noqa: BLE001 - never crash training over the TUI
                    logger.warning("TUI refresh timer stopped after error: %s", exc)
                    break

        self._timer = threading.Thread(target=_run, name="tui-refresh-timer", daemon=True)
        self._timer.start()

    def _stop_timer(self) -> None:
        if self._timer_stop is not None:
            self._timer_stop.set()
        if self._timer is not None:
            self._timer.join(timeout=2.0)
        self._timer = None
        self._timer_stop = None
        self._timer_start_monotonic = None

    # -- Windows native-fd redirect (AC-6/AC-7) --------------------------------------------

    def _build_windows_console(self, display):
        """Build the Rich Console for Live on Windows: force_terminal + real terminal size (AC-6).

        ``force_terminal=True`` makes Rich treat the plain ``display`` stream as a terminal so
        ``screen=True`` engages the full-screen alt-buffer even though we draw to a saved fd;
        passing the real terminal size makes the layout fill the screen (worst case: legacy
        conhost, already rejected by the AC-8 capability gate before we reach here).
        """
        from rich.console import Console

        kwargs: dict = {"force_terminal": True}
        if display is not None:
            kwargs["file"] = display
        try:
            size = os.get_terminal_size()
            kwargs["width"] = size.columns
            kwargs["height"] = size.lines
        except OSError:
            pass
        return Console(**kwargs)

    def _start_windows_capture(self) -> None:
        """Redirect native fd 1/2 to ``<logs_dir>/native.log``, saving BOTH real fds for restore.

        The saved fd 1 is duplicated into a text ``display`` stream so Rich keeps drawing to the
        real terminal; pybullet's native banner (and any other native write) then lands in
        ``native.log`` instead of the screen (AC-7). Degrades quietly on any error (restores what
        it set) so it can never crash the run.
        """
        try:
            self._flush_std()
            self._win_saved_fd1 = os.dup(1)
            self._win_saved_fd2 = os.dup(2)
            self._win_display = os.fdopen(
                os.dup(self._win_saved_fd1),
                "w",
                buffering=1,
                encoding="utf-8",
                errors="replace",
                closefd=True,
            )
            if self._logs_dir:
                os.makedirs(self._logs_dir, exist_ok=True)
                native_path = os.path.join(self._logs_dir, "native.log")
            else:
                native_path = os.devnull
            self._win_native_fd = os.open(native_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.dup2(self._win_native_fd, 1)
            os.dup2(self._win_native_fd, 2)
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the run
            logger.warning("Windows native-output redirect unavailable: %s", exc)
            self._restore_windows_capture()

    def _restore_windows_capture(self) -> None:
        """Restore BOTH fd 1 and fd 2 to the real terminal and close everything (idempotent).

        Restoring both fds is the challenger's non-blocking note: mirror ``FdLogCapture._restore``,
        which saves and restores fd 1 AND fd 2.
        """
        self._flush_std()
        for saved, target in ((self._win_saved_fd1, 1), (self._win_saved_fd2, 2)):
            if saved is not None:
                try:
                    os.dup2(saved, target)
                except OSError:
                    pass
        if self._win_native_fd is not None:
            try:
                os.close(self._win_native_fd)
            except OSError:
                pass
            self._win_native_fd = None
        if self._win_display is not None:
            try:
                self._win_display.flush()
                self._win_display.close()
            except Exception:  # noqa: BLE001
                pass
            self._win_display = None
        for fd in (self._win_saved_fd1, self._win_saved_fd2):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._win_saved_fd1 = None
        self._win_saved_fd2 = None

    # -- log bridge (AC-4/AC-7) ------------------------------------------------------------

    def _install_logbridge(self) -> None:
        try:
            from drone_fly.train.tui.logbridge import LogBridge

            self._logbridge = LogBridge()
            self._logbridge.install()
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the run
            logger.warning("TUI log bridge unavailable: %s", exc)
            self._logbridge = None

    def _uninstall_logbridge(self) -> None:
        if self._logbridge is not None:
            try:
                self._logbridge.uninstall()
            except Exception as exc:  # noqa: BLE001
                logger.warning("TUI log bridge teardown error (ignored): %s", exc)
            self._logbridge = None

    # -- helpers ---------------------------------------------------------------------------

    @staticmethod
    def _flush_std() -> None:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001 - flushing must never raise out of teardown
                pass


__all__ = ["TrainingDashboard"]
