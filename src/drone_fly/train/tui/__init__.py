"""Live full-screen training TUI (UC-22): a Rich dashboard redrawn from a SB3 callback.

Strict data⟂render split:

* :mod:`metrics` — pure, headless data layer (rolling history, trend/sparkline/ETA math,
  :class:`~drone_fly.train.tui.metrics.DashboardModel`). No Rich.
* :mod:`capture` — C-level (fd) stdout/stderr capture into a bounded scrollback. Stdlib only.
* :mod:`render` — Rich ``model → renderable`` builders (the untested draw boundary).
* :mod:`dashboard` — orchestration: model + fd-capture + Rich ``Live``.
* :mod:`callback` — the SB3 ``BaseCallback`` that feeds the model each rollout.

Only the pure/stdlib halves (:mod:`metrics`, :mod:`capture`) are re-exported here: their source
is Rich-free and importing them does not itself reach the :mod:`render` / :mod:`dashboard` /
:mod:`callback` layer, keeping the data⟂render split intact. Those three are imported directly
(and lazily by the training loop) at their point of use. (Note: the parent
``drone_fly.train`` package eagerly imports :mod:`~drone_fly.train.loop`, which pulls in SB3 —
and hence Rich — as a pre-existing side effect independent of this package.)
"""

from __future__ import annotations

from drone_fly.train.tui.capture import FdLogCapture
from drone_fly.train.tui.metrics import (
    TRACKED_METRICS,
    DashboardModel,
    MetricDirection,
    MetricHistory,
    MetricSpec,
    TrendRow,
    classify_trend,
    eta,
    format_duration,
    format_pct,
    progress_fraction,
    sparkline,
    tendency_icon,
)

__all__ = [
    "DashboardModel",
    "MetricDirection",
    "MetricSpec",
    "MetricHistory",
    "TrendRow",
    "TRACKED_METRICS",
    "classify_trend",
    "tendency_icon",
    "sparkline",
    "progress_fraction",
    "eta",
    "format_duration",
    "format_pct",
    "FdLogCapture",
]
