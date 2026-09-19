"""Live-dashboard orchestration for the training TUI (UC-22).

:class:`TrainingDashboard` binds the three halves together: it owns the pure
:class:`~drone_fly.train.tui.metrics.DashboardModel`, an
:class:`~drone_fly.train.tui.capture.FdLogCapture`, and (inside :meth:`live_session`) a Rich
``Live``. The callback feeds ``model`` and calls :meth:`redraw`; the status bar reads
``model.latest_verdict`` via the injected :meth:`set_verdict` (the UC-23 seam).

``live_session`` is the single entry/teardown boundary: it enters fd-capture and ``Live``
(drawing to the *original* terminal via ``capture.display_stdout``) and guarantees teardown —
``Live`` stopped first, then fds restored — on normal exit, on an exception, and on
``KeyboardInterrupt`` (AC8). Everything is defensive: a Rich failure (too-small terminal,
dumb/``NO_COLOR`` terminal, or any render error) degrades to plain capture / line logging and
never crashes the training run.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from drone_fly.train.tui.capture import FdLogCapture
from drone_fly.train.tui.metrics import DashboardModel

logger = logging.getLogger(__name__)

#: Rich Live refresh cap; the callback drives explicit redraws once per rollout anyway.
_REFRESH_PER_SECOND = 4


class TrainingDashboard:
    """Own the model + fd-capture + Rich Live, and expose ``set_verdict`` / ``redraw``."""

    def __init__(
        self,
        *,
        scheduled_iters: int = 0,
        capture: bool = True,
        n_envs: int = 1,
        backend: str = "dummy",
    ) -> None:
        # UC-26 AC-11: forward the resolved rollout parallelism into the pure model so the
        # values panel can render it (static per run).
        self.model = DashboardModel(scheduled_iters=scheduled_iters, n_envs=n_envs, backend=backend)
        self._capture = FdLogCapture() if capture else None
        self._live = None
        self._console = None
        self._enabled = True

    # -- verdict seam (injected into the existing HealthCallback) --------------------------

    def set_verdict(self, verdict) -> None:
        """Store the latest health verdict; never raises into the callback."""
        try:
            self.model.set_verdict(verdict)
        except Exception as exc:  # noqa: BLE001 - a consumer must not crash training
            logger.warning("dashboard set_verdict failed: %s", exc)

    # -- live session ----------------------------------------------------------------------

    @contextmanager
    def live_session(self):
        """Enter fd-capture + Rich Live; guarantee ordered teardown even on Ctrl-C (AC8)."""
        self._start()
        try:
            yield self
        finally:
            self._stop()

    def _start(self) -> None:
        # 1) fd capture first (so the Live console can draw to the saved real terminal).
        if self._capture is not None:
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

    def _stop(self) -> None:
        # Tear down Live BEFORE restoring fds, so its final frame flushes to the terminal.
        if self._live is not None:
            try:
                self._live.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Live teardown error (ignored): %s", exc)
            self._live = None
        self._console = None
        if self._capture is not None:
            try:
                self._capture.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("fd capture teardown error (ignored): %s", exc)

    def redraw(self) -> None:
        """Rebuild the layout from the current model + scrollback; never crashes training."""
        if self._live is None:
            return
        try:
            from drone_fly.train.tui.render import build_layout

            self._live.update(build_layout(self.model, self._log_lines()))
        except Exception as exc:  # noqa: BLE001
            logger.warning("dashboard redraw failed, disabling live updates: %s", exc)
            self._live = None

    def _log_lines(self) -> list[str]:
        return self._capture.lines() if self._capture is not None else []


__all__ = ["TrainingDashboard"]
