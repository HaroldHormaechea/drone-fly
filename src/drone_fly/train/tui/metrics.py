"""Pure, headless data layer for the live training TUI (UC-22).

This module is the **testable half** of the dashboard: it owns the per-metric rolling
histories, the trend/sparkline/ETA/progress math, and the :class:`DashboardModel` that a SB3
callback feeds one rollout-snapshot at a time. It imports **no Rich** and touches no terminal
— the strict data⟂render split the use case mandates (AC4). :mod:`render` and :mod:`dashboard`
are the only modules allowed to import Rich.

Timing contract (None/nan-tolerant by design). A ROLLOUT value that is ``nan`` or a TRAIN
value that is ``None``/missing is *not* appended to its history and renders as the placeholder
``—`` — never ``0``, never a crash. This covers the first-rollout warm-up (SB3 collects the
episode buffers before that iteration's ``PPO.train()`` records any ``train/*`` key) and an
absent ``train/std`` (only logged for a continuous action space). The cross-source off-by-one
(fresh ROLLOUT vs one-iteration-lagged TRAIN) is cosmetic and matches the HealthCallback.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

#: Sparkline block glyphs, low→high (8 levels).
SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
#: Rendered for a perfectly flat / single-point / no-variance series.
FLAT_LINE_CHAR = "─"
#: Placeholder for a not-yet-observed value (never rendered as ``0``).
PLACEHOLDER = "—"

#: Rolling window kept per metric (the "last ~100 updates" of AC4).
DEFAULT_WINDOW = 100
#: Fixed sparkline width the window is rebucketed to.
DEFAULT_SPARK_WIDTH = 8
#: Relative-change magnitude within which a series counts as "flat".
DEFAULT_FLAT_TOL = 0.05
#: A ``success_rate`` at or below this counts as pinned at ~0 (concern).
SUCCESS_EPS = 0.02

#: Tendency icons for the three trend classes.
TREND_ICONS = {"improving": "▲", "worsening": "▼", "flat": "►"}


class MetricDirection(Enum):
    """Which way a metric "improves", used to make trend classification direction-aware."""

    HIGHER_BETTER = "higher_better"  # success_rate, ep_rew_mean, explained_var
    LOWER_BETTER = "lower_better"  # value_loss; entropy/std where shrinking is expected
    BAND = "band"  # approx_kl — flat/near-target is healthy, drift up is bad


# --- Small pure helpers -------------------------------------------------------------------


def _finite(x) -> float | None:
    """Coerce ``x`` to a finite float, or return ``None`` for None/nan/inf/garbage."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _clean(values: Sequence[float] | None) -> list[float]:
    """Drop None/nan/inf entries; keep order (oldest first, current last)."""
    out: list[float] = []
    for x in values or ():
        v = _finite(x)
        if v is not None:
            out.append(v)
    return out


def _rel_change(values: Sequence[float] | None) -> float:
    """Signed relative change first→last, ``(last-first)/max(|first|,1e-9→1)``.

    Returns ``0.0`` with fewer than two finite observations (no trend yet). A tiny/zero
    baseline can't blow the ratio up — mirrors the health engine's own ``_rel_change``.
    """
    vals = _clean(values)
    if len(vals) < 2:
        return 0.0
    first, last = vals[0], vals[-1]
    denom = abs(first) if abs(first) > 1e-9 else 1.0
    return (last - first) / denom


def classify_trend(
    values: Sequence[float] | None,
    direction: MetricDirection,
    flat_tol: float = DEFAULT_FLAT_TOL,
) -> str:
    """Classify a series as ``"improving"`` / ``"worsening"`` / ``"flat"`` (AC4).

    Direction-aware: a rising series is *improving* for :attr:`MetricDirection.HIGHER_BETTER`
    but *worsening* for :attr:`MetricDirection.LOWER_BETTER`. For :attr:`MetricDirection.BAND`
    (``approx_kl``) near-target/flat reads as ``"flat"`` (healthy) — never ``"improving"`` — and
    a drift *up* out of the band reads as ``"worsening"``.
    """
    change = _rel_change(values)
    if abs(change) <= flat_tol:
        return "flat"
    rising = change > 0
    if direction is MetricDirection.HIGHER_BETTER:
        return "improving" if rising else "worsening"
    if direction is MetricDirection.LOWER_BETTER:
        return "worsening" if rising else "improving"
    # BAND: flat already handled; upward drift out of band is bad, back down is good.
    return "worsening" if rising else "improving"


def tendency_icon(trend: str) -> str:
    """Map a trend class to its glyph (▲ improving / ▼ worsening / ► flat)."""
    return TREND_ICONS.get(trend, "►")


def _resample(vals: list[float], width: int) -> list[float]:
    """Rebucket ``vals`` to exactly ``width`` points (stretch if shorter, average if longer)."""
    n = len(vals)
    if n == width:
        return list(vals)
    if n < width:
        return [vals[int(i * n / width)] for i in range(width)]
    out: list[float] = []
    for i in range(width):
        start = int(i * n / width)
        end = max(int((i + 1) * n / width), start + 1)
        chunk = vals[start:end]
        out.append(sum(chunk) / len(chunk))
    return out


def sparkline(values: Sequence[float] | None, width: int = DEFAULT_SPARK_WIDTH) -> str:
    """Render ``values`` as a fixed-``width`` block sparkline over the rolling window (AC4).

    Empty/all-missing → ``width`` spaces. A single point or a zero-variance series → a flat
    line (``─`` × ``width``). Otherwise the window is rebucketed to ``width`` points and
    min–max normalised onto the 8 block glyphs.
    """
    width = max(int(width), 1)
    vals = _clean(values)
    if not vals:
        return " " * width
    lo, hi = min(vals), max(vals)
    if hi - lo <= 0:  # single point or perfectly flat
        return FLAT_LINE_CHAR * width
    buckets = _resample(vals, width)
    blo, bhi = min(buckets), max(buckets)
    span = bhi - blo
    if span <= 0:
        return FLAT_LINE_CHAR * width
    levels = len(SPARK_BLOCKS)
    out = []
    for b in buckets:
        idx = int((b - blo) / span * (levels - 1) + 0.5)
        out.append(SPARK_BLOCKS[max(0, min(levels - 1, idx))])
    return "".join(out)


def progress_fraction(current: int, scheduled: int) -> float:
    """Fraction of scheduled iterations completed, clamped to ``[0, 1]`` (0 when unknown)."""
    if not scheduled or scheduled <= 0:
        return 0.0
    return max(0.0, min(1.0, current / scheduled))


def eta(elapsed_seconds: float | None, fraction: float | None) -> float | None:
    """Estimate remaining seconds from ``elapsed`` and the completed ``fraction``.

    Returns ``None`` before there is enough signal (no progress yet / no elapsed time); ``0``
    once complete. Linear extrapolation: ``elapsed * (1 - f) / f``.
    """
    if fraction is None or fraction <= 0 or elapsed_seconds is None or elapsed_seconds < 0:
        return None
    if fraction >= 1.0:
        return 0.0
    return elapsed_seconds * (1.0 - fraction) / fraction


def format_duration(seconds: float | None) -> str:
    """Compact ``2h15m`` / ``5m30s`` / ``45s`` duration; ``—`` when unknown."""
    if seconds is None:
        return PLACEHOLDER
    total = int(max(0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def format_pct(fraction: float | None) -> str:
    """Render a 0..1 fraction as an integer percentage, or ``—`` when unknown."""
    if fraction is None:
        return PLACEHOLDER
    return f"{int(round(fraction * 100))}%"


# --- Metric registry ----------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """Static description of one tracked trend row."""

    key: str
    label: str
    direction: MetricDirection
    fmt: Callable[[float], str]
    #: Optional ``(values, trend) -> bool`` predicate raising the ⚠ concern marker.
    concern: Callable[[list[float], str], bool] | None = None


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _fmt_int(v: float) -> str:
    return f"{v:.0f}"


def _fmt_2f(v: float) -> str:
    return f"{v:.2f}"


def _fmt_3f(v: float) -> str:
    return f"{v:.3f}"


def _fmt_1f(v: float) -> str:
    return f"{v:.1f}"


def _concern_success(vals: list[float], trend: str) -> bool:
    """Success pinned at ~0 is a concern regardless of a tiny wobble."""
    return bool(vals) and vals[-1] <= SUCCESS_EPS


def _concern_entropy_flat(vals: list[float], trend: str) -> bool:
    """A flat (non-shrinking) action entropy is the "actor not committing" signature."""
    return trend == "flat"


#: The 7 trend rows, in wireframe order (success_rate first), followed at render time by the
#: iterations progress bar. ``entropy`` is derived from ``train/entropy_loss`` (see
#: :meth:`DashboardModel.update`); ``std`` is ``train/std``.
TRACKED_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "success_rate", "success_rate", MetricDirection.HIGHER_BETTER, _fmt_pct, _concern_success
    ),
    MetricSpec("ep_rew_mean", "ep_rew_mean", MetricDirection.HIGHER_BETTER, _fmt_int),
    MetricSpec("entropy", "entropy", MetricDirection.LOWER_BETTER, _fmt_2f, _concern_entropy_flat),
    MetricSpec("std", "std", MetricDirection.LOWER_BETTER, _fmt_3f),
    MetricSpec("value_loss", "value_loss", MetricDirection.LOWER_BETTER, _fmt_1f),
    MetricSpec("approx_kl", "approx_kl", MetricDirection.BAND, _fmt_3f),
    MetricSpec("explained_var", "explained_var", MetricDirection.HIGHER_BETTER, _fmt_2f),
)


@dataclass(frozen=True)
class TrendRow:
    """One rendered trend row: label + value + sparkline + tendency + concern flag."""

    key: str
    label: str
    value_text: str
    sparkline: str
    trend: str
    icon: str
    concern: bool


class MetricHistory:
    """A bounded rolling window that silently drops None/nan on append."""

    def __init__(self, maxlen: int = DEFAULT_WINDOW) -> None:
        self._d: deque[float] = deque(maxlen=max(int(maxlen), 1))

    def append(self, value) -> None:
        v = _finite(value)
        if v is not None:
            self._d.append(v)

    def values(self) -> list[float]:
        return list(self._d)

    def latest(self) -> float | None:
        return self._d[-1] if self._d else None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._d)


class DashboardModel:
    """Mutable model the callback feeds and the renderer reads (pure, no Rich).

    Holds a rolling :class:`MetricHistory` per tracked metric, the latest raw grouped values
    for the TIME/TRAIN/ROLLOUT panel, the iteration progress, and the latest health verdict.
    """

    def __init__(
        self,
        *,
        scheduled_iters: int = 0,
        window: int = DEFAULT_WINDOW,
        spark_width: int = DEFAULT_SPARK_WIDTH,
        flat_tol: float = DEFAULT_FLAT_TOL,
        n_envs: int = 1,
        backend: str = "dummy",
    ) -> None:
        self.scheduled_iters = int(scheduled_iters or 0)
        # UC-26 AC-11: the RESOLVED rollout parallelism (worker count + active vec-env backend),
        # shown static in the values panel so the operator sees how many cores are in use.
        self.n_envs = int(n_envs)
        self.backend = str(backend)
        self.window = window
        self.spark_width = spark_width
        self.flat_tol = flat_tol
        self.history: dict[str, MetricHistory] = {
            spec.key: MetricHistory(window) for spec in TRACKED_METRICS
        }
        self.current_iter = 0
        self.elapsed_seconds = 0.0
        # UC-30 intra-rollout heartbeat: steps collected in the current rollout and its target
        # (``n_steps × n_envs``). Fed by :meth:`tick` only — never through :meth:`update`.
        self.rollout_steps = 0
        self.rollout_target = 0
        # Latest raw snapshot for the grouped value panel (None → placeholder, never 0).
        self.raw: dict[str, float | None] = {
            "ep_rew": None,
            "ep_len": None,
            "success": None,
            "entropy_loss": None,
            "value_loss": None,
            "explained_variance": None,
        }
        self.latest_verdict = None

    def update(
        self,
        *,
        n_updates: int | None = None,
        elapsed_seconds: float | None = None,
        ep_rew_mean=None,
        ep_len_mean=None,
        success_rate=None,
        entropy_loss=None,
        std=None,
        value_loss=None,
        approx_kl=None,
        explained_variance=None,
    ) -> None:
        """Ingest one rollout snapshot. Every value is None/nan-tolerant (see module docstring).

        ``entropy`` is derived as ``-entropy_loss`` (for a continuous action space SB3 logs
        ``entropy_loss = -mean(entropy)``), so a *shrinking* entropy = a committing actor.
        """
        if n_updates is not None:
            self.current_iter = int(n_updates)
        if elapsed_seconds is not None:
            self.elapsed_seconds = float(elapsed_seconds)

        el = _finite(entropy_loss)
        entropy = -el if el is not None else None

        # Histories (append() itself skips None/nan, so the placeholder contract holds).
        self.history["success_rate"].append(success_rate)
        self.history["ep_rew_mean"].append(ep_rew_mean)
        self.history["entropy"].append(entropy)
        self.history["std"].append(std)
        self.history["value_loss"].append(value_loss)
        self.history["approx_kl"].append(approx_kl)
        self.history["explained_var"].append(explained_variance)

        # Raw panel snapshot (finite-or-None; the renderer shows a placeholder for None).
        self.raw["ep_rew"] = _finite(ep_rew_mean)
        self.raw["ep_len"] = _finite(ep_len_mean)
        self.raw["success"] = _finite(success_rate)
        self.raw["entropy_loss"] = el
        self.raw["value_loss"] = _finite(value_loss)
        self.raw["explained_variance"] = _finite(explained_variance)

    def tick(self, *, elapsed_seconds, rollout_steps, rollout_target) -> None:
        """Ingest one intra-rollout heartbeat (UC-30): live elapsed + within-rollout progress.

        Deliberately touches **neither** ``history`` **nor** ``raw`` — unlike :meth:`update`,
        whose unconditional ``raw[...]`` assignments would blank the six values-panel entries to
        the placeholder after the first rollout. It only refreshes the live clock and the
        collecting-progress counters, so the last rollout-end snapshot persists through the next
        collection (AC-4).
        """
        self.elapsed_seconds = float(elapsed_seconds)
        self.rollout_steps = int(rollout_steps)
        self.rollout_target = int(rollout_target)

    def tick_clock(self, elapsed_seconds) -> None:
        """Advance ONLY the live elapsed clock (and hence ETA) — no counters, no history (UC-32).

        Fed by the Windows ~1 Hz refresh timer (AC-5) so the elapsed timer and ETA keep ticking
        during ``PPO.train()``, which fires no ``on_step``. Deliberately narrower than
        :meth:`tick`: it must **not** move the collecting-progress counters (``rollout_steps`` /
        ``rollout_target``) — the timer has observed no new steps, so fabricating progress would
        be wrong — and it touches neither ``history`` nor ``raw`` (so the last rollout-end
        snapshot persists). Uncontended on macOS/Linux, where no timer runs (AC-9).
        """
        self.elapsed_seconds = float(elapsed_seconds)

    def set_verdict(self, verdict) -> None:
        """Store the latest health verdict (the UC-23 seam the status bar renders)."""
        self.latest_verdict = verdict

    def rows(self) -> list[TrendRow]:
        """Build the 7 :class:`TrendRow`s in registry order."""
        rows: list[TrendRow] = []
        for spec in TRACKED_METRICS:
            hist = self.history[spec.key]
            vals = hist.values()
            latest = hist.latest()
            trend = classify_trend(vals, spec.direction, self.flat_tol)
            value_text = spec.fmt(latest) if latest is not None else PLACEHOLDER
            concern = bool(spec.concern(vals, trend)) if spec.concern is not None else False
            rows.append(
                TrendRow(
                    key=spec.key,
                    label=spec.label,
                    value_text=value_text,
                    sparkline=sparkline(vals, self.spark_width),
                    trend=trend,
                    icon=tendency_icon(trend),
                    concern=concern,
                )
            )
        return rows

    def progress(self) -> tuple[int, int, float]:
        """``(current_iter, scheduled_iters, fraction)`` for the progress bar."""
        return (
            self.current_iter,
            self.scheduled_iters,
            progress_fraction(self.current_iter, self.scheduled_iters),
        )

    def rollout_progress(self) -> tuple[int, int, float]:
        """``(rollout_steps, rollout_target, fraction)`` for the UC-30 collecting bar.

        The fraction reuses :func:`progress_fraction` (clamped to ``[0, 1]``, ``0`` when the
        target is unknown), so a pre-tick / missing-attr rollout degrades to an empty bar.
        """
        return (
            self.rollout_steps,
            self.rollout_target,
            progress_fraction(self.rollout_steps, self.rollout_target),
        )

    def eta_seconds(self) -> float | None:
        """Estimated remaining seconds, or ``None`` before there is enough signal."""
        _, _, frac = self.progress()
        return eta(self.elapsed_seconds, frac)


__all__ = [
    "MetricDirection",
    "MetricSpec",
    "TrackedMetric",
    "TRACKED_METRICS",
    "TrendRow",
    "MetricHistory",
    "DashboardModel",
    "classify_trend",
    "tendency_icon",
    "sparkline",
    "progress_fraction",
    "eta",
    "format_duration",
    "format_pct",
    "SPARK_BLOCKS",
    "FLAT_LINE_CHAR",
    "PLACEHOLDER",
    "DEFAULT_WINDOW",
    "DEFAULT_SPARK_WIDTH",
]
# Backwards-friendly alias (the registry entries are ``MetricSpec`` instances).
TrackedMetric = MetricSpec
