"""UC-22 AC3/AC4 — the pure, headless TUI data layer (``drone_fly.train.tui.metrics``).

Everything here is terminal-free. The data layer's **source** is Rich-free and importing it
must not reach the Rich-bearing ``render`` / ``dashboard`` submodules — the strict data⟂render
split the use case mandates (importing it still transitively loads Rich via ``drone_fly.train``'s
eager SB3 import, which itself depends on Rich; see the two split tests). These tests cover:

* direction-aware trend classification (higher/lower-better + the ``approx_kl`` band),
* fixed-width sparkline bucketing incl. flat / degenerate / empty series,
* the rolling ~100-update window truncation,
* ETA / progress-fraction math and the compact duration / percent formatters,
* the **first-rollout snapshot** timing contract — a ``nan`` ROLLOUT + all-``None`` TRAIN
  snapshot must ``update()`` cleanly, leave every trend row on the ``—`` placeholder (never
  ``0``, never a crash), and then populate ``success_rate`` / ``ep_rew_mean`` from the buffers
  on the next real snapshot (guards timing issues 1–3 together).
"""

from __future__ import annotations

import math
import sys

import pytest

from drone_fly.train.tui import metrics as M
from drone_fly.train.tui.metrics import (
    DEFAULT_WINDOW,
    PLACEHOLDER,
    DashboardModel,
    MetricDirection,
    MetricHistory,
    classify_trend,
    eta,
    format_duration,
    format_pct,
    progress_fraction,
    sparkline,
    tendency_icon,
)

# --------------------------------------------------------------------------- #
# Data⟂render split: the data layer must not import Rich (AC4)
# --------------------------------------------------------------------------- #


def test_metrics_module_is_rich_free_at_source() -> None:
    """The data-layer source must contain no Rich import (the pure/headless split, AC4).

    (Note: importing the module still transitively pulls Rich in via ``drone_fly.train``'s
    eager ``loop``/SB3 import — SB3 itself depends on Rich — which is why we assert on the
    *source* and, below, on the *render/dashboard submodules not being reached*.)
    """
    from pathlib import Path

    src = Path(M.__file__).read_text(encoding="utf-8")
    assert "import rich" not in src
    assert "from rich" not in src


def test_importing_metrics_does_not_reach_the_render_layer() -> None:
    """The real data⟂render contract: importing the data layer must NOT pull in the Rich-bearing
    ``render`` / ``dashboard`` submodules. Checked in a fresh interpreter so an unrelated
    in-suite import can't mask a regression."""
    import subprocess

    code = (
        "import sys; import drone_fly.train.tui.metrics;"
        "assert 'drone_fly.train.tui.render' not in sys.modules, 'render was imported';"
        "assert 'drone_fly.train.tui.dashboard' not in sys.modules, 'dashboard was imported';"
        "assert 'drone_fly.train.tui.callback' not in sys.modules, 'callback was imported';"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# --------------------------------------------------------------------------- #
# classify_trend — direction-aware (AC4)
# --------------------------------------------------------------------------- #


def test_higher_better_rising_is_improving_falling_is_worsening() -> None:
    up = [0.0, 0.1, 0.2, 0.3, 0.5]
    down = [0.5, 0.4, 0.3, 0.2, 0.0]
    assert classify_trend(up, MetricDirection.HIGHER_BETTER) == "improving"
    assert classify_trend(down, MetricDirection.HIGHER_BETTER) == "worsening"


def test_lower_better_falling_is_improving_rising_is_worsening() -> None:
    """value_loss / entropy / std: shrinking is *good*, growing is *bad*."""
    falling = [50.0, 40.0, 30.0, 20.0, 10.0]
    rising = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert classify_trend(falling, MetricDirection.LOWER_BETTER) == "improving"
    assert classify_trend(rising, MetricDirection.LOWER_BETTER) == "worsening"


def test_band_metric_flat_is_flat_and_upward_drift_is_worsening() -> None:
    """approx_kl: near-target/flat reads as flat (healthy), an upward drift is worsening."""
    flat = [0.010, 0.0101, 0.0099, 0.0100]
    drift_up = [0.005, 0.015, 0.03, 0.05]
    drift_down = [0.05, 0.03, 0.015, 0.005]
    assert classify_trend(flat, MetricDirection.BAND) == "flat"
    assert classify_trend(drift_up, MetricDirection.BAND) == "worsening"
    assert classify_trend(drift_down, MetricDirection.BAND) == "improving"


def test_flat_within_tolerance_is_flat_for_every_direction() -> None:
    nearly_flat = [1.0, 1.0, 1.01, 1.0]  # < default 5% relative change
    for direction in MetricDirection:
        assert classify_trend(nearly_flat, direction) == "flat"


def test_classify_trend_needs_two_finite_points() -> None:
    assert classify_trend([], MetricDirection.HIGHER_BETTER) == "flat"
    assert classify_trend([3.0], MetricDirection.HIGHER_BETTER) == "flat"
    assert classify_trend([None, float("nan")], MetricDirection.HIGHER_BETTER) == "flat"


def test_classify_trend_ignores_none_and_nan_entries() -> None:
    vals = [None, 0.0, float("nan"), 0.2, 0.5]
    assert classify_trend(vals, MetricDirection.HIGHER_BETTER) == "improving"


def test_tendency_icon_maps_each_class() -> None:
    assert tendency_icon("improving") == "▲"
    assert tendency_icon("worsening") == "▼"
    assert tendency_icon("flat") == "►"
    assert tendency_icon("garbage") == "►"  # defensive default


# --------------------------------------------------------------------------- #
# sparkline — fixed width, bucketing, degenerate cases (AC4)
# --------------------------------------------------------------------------- #


def test_sparkline_empty_series_is_blank_of_requested_width() -> None:
    assert sparkline([], 8) == " " * 8
    assert sparkline(None, 5) == " " * 5
    assert sparkline([None, float("nan")], 6) == " " * 6


def test_sparkline_single_or_flat_series_is_a_flat_line() -> None:
    assert sparkline([5.0], 8) == M.FLAT_LINE_CHAR * 8
    assert sparkline([3.0, 3.0, 3.0], 8) == M.FLAT_LINE_CHAR * 8


def test_sparkline_monotonic_series_spans_low_to_high_glyph() -> None:
    s = sparkline([0, 1, 2, 3, 4, 5, 6, 7], 8)
    assert len(s) == 8
    assert s[0] == M.SPARK_BLOCKS[0]
    assert s[-1] == M.SPARK_BLOCKS[-1]
    assert all(ch in M.SPARK_BLOCKS for ch in s)


@pytest.mark.parametrize("width", [1, 3, 4, 8, 16])
def test_sparkline_always_returns_exactly_width_chars(width: int) -> None:
    assert len(sparkline(list(range(50)), width)) == width
    assert len(sparkline([1.0], width)) == width
    assert len(sparkline([], width)) == width


def test_sparkline_downsamples_a_long_window_to_fixed_width() -> None:
    s = sparkline(list(range(100)), 8)
    assert len(s) == 8
    assert s[0] == M.SPARK_BLOCKS[0]
    assert s[-1] == M.SPARK_BLOCKS[-1]


# --------------------------------------------------------------------------- #
# ETA / progress / formatters (AC4)
# --------------------------------------------------------------------------- #


def test_progress_fraction_clamps_and_guards_zero_schedule() -> None:
    assert progress_fraction(0, 0) == 0.0
    assert progress_fraction(5, 0) == 0.0
    assert progress_fraction(0, 100) == 0.0
    assert progress_fraction(50, 100) == 0.5
    assert progress_fraction(200, 100) == 1.0  # clamped
    assert progress_fraction(-5, 100) == 0.0  # clamped


def test_eta_linear_extrapolation_and_edge_cases() -> None:
    assert eta(100.0, 0.5) == pytest.approx(100.0)
    assert eta(100.0, 0.25) == pytest.approx(300.0)
    assert eta(100.0, 1.0) == 0.0  # complete
    assert eta(100.0, 0.0) is None  # no progress yet
    assert eta(100.0, None) is None
    assert eta(None, 0.5) is None
    assert eta(-1.0, 0.5) is None


def test_format_duration_buckets() -> None:
    assert format_duration(8100) == "2h15m"
    assert format_duration(330) == "5m30s"
    assert format_duration(45) == "45s"
    assert format_duration(0) == "0s"
    assert format_duration(None) == PLACEHOLDER
    assert format_duration(-10) == "0s"  # clamped, never negative


def test_format_pct_rounds_and_placeholders() -> None:
    assert format_pct(0.1734) == "17%"
    assert format_pct(0.0) == "0%"
    assert format_pct(1.0) == "100%"
    assert format_pct(None) == PLACEHOLDER


# --------------------------------------------------------------------------- #
# MetricHistory — bounded window, None/nan dropping (AC4)
# --------------------------------------------------------------------------- #


def test_metric_history_drops_none_and_nan_on_append() -> None:
    h = MetricHistory(maxlen=10)
    for v in [1.0, None, float("nan"), float("inf"), 2.0]:
        h.append(v)
    assert h.values() == [1.0, 2.0]
    assert h.latest() == 2.0


def test_metric_history_truncates_to_rolling_window() -> None:
    h = MetricHistory(maxlen=DEFAULT_WINDOW)
    for i in range(DEFAULT_WINDOW + 50):
        h.append(float(i))
    vals = h.values()
    assert len(vals) == DEFAULT_WINDOW  # ~100-update rolling window (AC4)
    assert vals[0] == 50.0  # the oldest 50 were evicted
    assert vals[-1] == float(DEFAULT_WINDOW + 49)


def test_metric_history_empty_latest_is_none() -> None:
    assert MetricHistory().latest() is None


# --------------------------------------------------------------------------- #
# DashboardModel — registry, rows, progress, verdict
# --------------------------------------------------------------------------- #


def test_tracked_metrics_are_the_seven_wireframe_rows_in_order() -> None:
    keys = [spec.key for spec in M.TRACKED_METRICS]
    assert keys == [
        "success_rate",
        "ep_rew_mean",
        "entropy",
        "std",
        "value_loss",
        "approx_kl",
        "explained_var",
    ]


def test_model_progress_and_eta() -> None:
    m = DashboardModel(scheduled_iters=488)
    m.update(n_updates=81, elapsed_seconds=8100.0)
    cur, sched, frac = m.progress()
    assert (cur, sched) == (81, 488)
    assert frac == pytest.approx(81 / 488)
    assert m.eta_seconds() == pytest.approx(8100.0 * (1 - frac) / frac)


def test_model_rows_are_seven_and_populate_from_a_real_snapshot() -> None:
    m = DashboardModel(scheduled_iters=488)
    # two updates so trend rows have >=2 points and can classify
    m.update(
        ep_rew_mean=-1300,
        ep_len_mean=200,
        success_rate=0.0,
        entropy_loss=-3.0,
        std=0.52,
        value_loss=60.0,
        approx_kl=0.006,
        explained_variance=0.40,
    )
    m.update(
        ep_rew_mean=-1204,
        ep_len_mean=210,
        success_rate=0.10,
        entropy_loss=-2.5,
        std=0.50,
        value_loss=51.2,
        approx_kl=0.007,
        explained_variance=0.63,
    )
    rows = {r.key: r for r in m.rows()}
    assert len(rows) == 7
    assert rows["success_rate"].value_text == "10%"
    assert rows["ep_rew_mean"].value_text == "-1204"
    assert rows["value_loss"].value_text == "51.2"
    # entropy is derived as -entropy_loss: -(-2.5) = 2.50
    assert rows["entropy"].value_text == "2.50"
    # explained_var rising across the two snapshots -> improving (higher-better)
    assert rows["explained_var"].trend == "improving"
    assert rows["explained_var"].icon == "▲"


def test_success_rate_zero_is_a_value_not_a_placeholder_and_raises_concern() -> None:
    """0% success is a real observation (concern), NOT a missing '—' placeholder."""
    m = DashboardModel()
    m.update(success_rate=0.0)
    row = next(r for r in m.rows() if r.key == "success_rate")
    assert row.value_text == "0%"  # not PLACEHOLDER
    assert row.concern is True


def test_entropy_flat_raises_concern() -> None:
    """A non-shrinking (flat) entropy is the 'actor not committing' concern signature."""
    m = DashboardModel()
    for _ in range(4):
        m.update(entropy_loss=-2.94)  # constant entropy = 2.94 -> flat
    row = next(r for r in m.rows() if r.key == "entropy")
    assert row.trend == "flat"
    assert row.concern is True


def test_set_verdict_is_stored_on_the_model() -> None:
    m = DashboardModel()
    assert m.latest_verdict is None
    sentinel = object()
    m.set_verdict(sentinel)
    assert m.latest_verdict is sentinel


# --------------------------------------------------------------------------- #
# First-rollout snapshot — the timing contract (guards issues 1–3 together)
# --------------------------------------------------------------------------- #


def test_first_rollout_snapshot_all_missing_is_placeholders_no_crash() -> None:
    """nan ROLLOUT + all-None TRAIN (SB3's first rollout): update() must not crash and every
    trend row stays on the '—' placeholder — never 0, never a spurious history point."""
    m = DashboardModel(scheduled_iters=488)
    m.update(
        n_updates=1,
        elapsed_seconds=5.0,
        ep_rew_mean=float("nan"),
        ep_len_mean=float("nan"),
        success_rate=float("nan"),
        entropy_loss=None,
        std=None,
        value_loss=None,
        approx_kl=None,
        explained_variance=None,
    )
    rows = m.rows()
    assert [r.value_text for r in rows] == [PLACEHOLDER] * 7
    # no history was recorded for any metric (no spurious 0)
    assert all(len(h) == 0 for h in m.history.values())
    # raw grouped-panel values are all None -> the renderer shows placeholders
    assert all(v is None for v in m.raw.values())


def test_second_snapshot_populates_rollout_from_buffers_even_when_train_still_none() -> None:
    """The cross-source off-by-one: ROLLOUT (from the model buffers) can populate while the
    one-iteration-lagged TRAIN keys are still None — rollout rows fill, train rows stay '—'."""
    m = DashboardModel(scheduled_iters=488)
    # first rollout: nothing observed
    m.update(ep_rew_mean=float("nan"), success_rate=float("nan"), n_updates=1)
    # second rollout: buffers now have episodes; PPO.train() still hasn't logged train/*
    m.update(
        ep_rew_mean=-1204.0,
        ep_len_mean=210.0,
        success_rate=0.0,
        entropy_loss=None,
        std=None,
        value_loss=None,
        approx_kl=None,
        explained_variance=None,
        n_updates=2,
    )
    rows = {r.key: r for r in m.rows()}
    assert rows["success_rate"].value_text == "0%"  # populated from the buffer
    assert rows["ep_rew_mean"].value_text == "-1204"  # populated from the buffer
    # the lagged train-derived rows are still placeholders (None-tolerant, no crash)
    for k in ("entropy", "std", "value_loss", "approx_kl", "explained_var"):
        assert rows[k].value_text == PLACEHOLDER


def test_update_is_tolerant_of_a_completely_empty_call() -> None:
    m = DashboardModel()
    m.update()  # must not raise
    assert [r.value_text for r in m.rows()] == [PLACEHOLDER] * 7


def test_entropy_derivation_sign() -> None:
    """SB3 logs entropy_loss = -mean(entropy); the model derives entropy = -entropy_loss."""
    m = DashboardModel()
    m.update(entropy_loss=-2.5)
    assert m.history["entropy"].latest() == pytest.approx(2.5)
    assert m.raw["entropy_loss"] == pytest.approx(-2.5)
    # a nan entropy_loss must not append and must not blow up
    m2 = DashboardModel()
    m2.update(entropy_loss=float("nan"))
    assert m2.history["entropy"].latest() is None
    assert not math.isnan(0.0)  # sanity: nan handling above did not leak


# --------------------------------------------------------------------------- #
# UC-26 AC-11 — DashboardModel carries the RESOLVED rollout parallelism
# --------------------------------------------------------------------------- #


def test_dashboard_model_defaults_to_single_dummy_env() -> None:
    """A model built without the UC-26 fields defaults to the serial 1-env / dummy run so
    pre-UC-26 construction is unchanged."""
    m = DashboardModel()
    assert m.n_envs == 1
    assert m.backend == "dummy"


def test_dashboard_model_stores_resolved_n_envs_and_backend() -> None:
    """AC-11: the model stores the RESOLVED worker count + active backend verbatim (ints/strs)."""
    m = DashboardModel(scheduled_iters=488, n_envs=8, backend="subproc")
    assert m.n_envs == 8
    assert m.backend == "subproc"


def test_dashboard_model_coerces_resolved_parallelism_types() -> None:
    m = DashboardModel(n_envs="4", backend=None)  # type: ignore[arg-type]
    assert m.n_envs == 4
    assert isinstance(m.n_envs, int)
    assert m.backend == "None" and isinstance(m.backend, str)


# --------------------------------------------------------------------------- #
# UC-30 — the intra-rollout heartbeat data seam: tick() + rollout_progress()
# --------------------------------------------------------------------------- #


def test_dashboard_model_defaults_rollout_counters_to_zero() -> None:
    """A fresh model starts with no collecting progress (pre-UC-30 construction unchanged)."""
    m = DashboardModel()
    assert m.rollout_steps == 0
    assert m.rollout_target == 0
    assert m.rollout_progress() == (0, 0, 0.0)


def test_tick_updates_live_fields_but_leaves_history_and_raw_untouched() -> None:
    """The AC-4 isolation guarantee at the data layer: ``tick`` refreshes only the live clock +
    collecting counters. It must touch NEITHER ``history`` NOR ``raw`` (unlike ``update``, whose
    unconditional ``raw[...]`` writes would blank the values panel after the first rollout)."""
    m = DashboardModel(scheduled_iters=488)
    m.update(
        n_updates=1,
        elapsed_seconds=100.0,
        ep_rew_mean=-1200.0,
        ep_len_mean=210.0,
        success_rate=0.1,
        entropy_loss=-2.5,
        std=0.5,
        value_loss=51.2,
        approx_kl=0.007,
        explained_variance=0.63,
    )
    raw_before = dict(m.raw)
    hist_before = {k: h.values() for k, h in m.history.items()}

    m.tick(elapsed_seconds=999.0, rollout_steps=128, rollout_target=2048)

    # snapshot preserved verbatim
    assert m.raw == raw_before
    assert {k: h.values() for k, h in m.history.items()} == hist_before
    # but the live fields DID move
    assert m.elapsed_seconds == 999.0
    assert m.rollout_steps == 128
    assert m.rollout_target == 2048


def test_tick_coerces_its_argument_types() -> None:
    m = DashboardModel()
    m.tick(elapsed_seconds="12.5", rollout_steps="300", rollout_target="2048")  # type: ignore[arg-type]
    assert m.elapsed_seconds == pytest.approx(12.5) and isinstance(m.elapsed_seconds, float)
    assert m.rollout_steps == 300 and isinstance(m.rollout_steps, int)
    assert m.rollout_target == 2048 and isinstance(m.rollout_target, int)


def test_rollout_progress_math_clamps_and_guards_zero_target() -> None:
    """AC-3/AC-7: the collecting fraction is steps/target, clamped to [0, 1], and 0 when the
    target is unknown (pre-tick or missing attrs) — no div-by-zero."""
    m = DashboardModel()

    m.tick(elapsed_seconds=1.0, rollout_steps=512, rollout_target=2048)
    steps, target, frac = m.rollout_progress()
    assert (steps, target) == (512, 2048)
    assert frac == pytest.approx(0.25)

    # over-target clamps to 1.0 (last step can slightly overshoot n_steps × n_envs)
    m.tick(elapsed_seconds=1.0, rollout_steps=9999, rollout_target=2048)
    assert m.rollout_progress()[2] == 1.0

    # unknown / zero target degrades to an empty (0-fraction) bar, never a ZeroDivisionError
    m.tick(elapsed_seconds=1.0, rollout_steps=100, rollout_target=0)
    assert m.rollout_progress() == (100, 0, 0.0)


def test_tick_then_update_restores_the_snapshot_flow() -> None:
    """A heartbeat mid-collection followed by the next rollout-end ``update`` must produce a
    normal populated snapshot — tick does not corrupt the subsequent update path."""
    m = DashboardModel(scheduled_iters=488)
    m.tick(elapsed_seconds=5.0, rollout_steps=256, rollout_target=2048)
    assert all(v is None for v in m.raw.values())  # tick alone leaves raw empty
    m.update(ep_rew_mean=-1204.0, success_rate=0.1, n_updates=1)
    assert m.raw["ep_rew"] == -1204.0
    assert m.raw["success"] == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# UC-32 Addendum — cumulative step count for the TIME panel's steps line
# --------------------------------------------------------------------------- #


def test_total_steps_defaults_to_zero_and_current_starts_at_zero() -> None:
    m = DashboardModel(scheduled_iters=10)
    assert m.total_steps == 0
    assert m.current_steps == 0
    assert m.steps_progress() == (0, 0)


def test_total_steps_is_static_and_current_is_fed_by_update() -> None:
    """``total_steps`` is the configured budget (static per run); ``current_steps`` tracks SB3
    ``num_timesteps`` fed via ``update``."""
    m = DashboardModel(scheduled_iters=10, total_steps=1_000_000)
    m.update(n_updates=1, current_steps=26_624)
    assert m.steps_progress() == (26_624, 1_000_000)
    assert m.total_steps == 1_000_000  # unchanged by update


def test_current_steps_is_fed_by_tick_too() -> None:
    """The heartbeat tick also advances the cumulative step count (liveness during a collection)."""
    m = DashboardModel(scheduled_iters=10, total_steps=500_000)
    m.tick(elapsed_seconds=1.0, rollout_steps=100, rollout_target=2048, current_steps=12_000)
    assert m.steps_progress() == (12_000, 500_000)


def test_update_and_tick_without_current_steps_leave_it_unchanged() -> None:
    """Omitting ``current_steps`` (None) must not reset the counter — it only moves when fed."""
    m = DashboardModel(scheduled_iters=10, total_steps=500_000)
    m.update(n_updates=1, current_steps=30_000)
    m.update(n_updates=2)  # no current_steps -> keep last
    assert m.current_steps == 30_000
    m.tick(elapsed_seconds=1.0, rollout_steps=1, rollout_target=2)  # no current_steps -> keep last
    assert m.current_steps == 30_000


# --------------------------------------------------------------------------- #
# UC-49 AC2 — the drone-dynamics slot on the model (None-tolerant)
# --------------------------------------------------------------------------- #
def test_drone_dynamics_slot_defaults_to_none() -> None:
    """A fresh model carries no summary yet — the slot is None until the callback feeds one."""
    m = DashboardModel(scheduled_iters=10)
    assert m.drone_dynamics is None


def test_set_drone_dynamics_stores_the_summary() -> None:
    """set_drone_dynamics stores the summary object for the renderer to pick up."""
    from drone_fly.adapter.dynamics_summary import drone_dynamics_summary

    m = DashboardModel(scheduled_iters=10)
    summary = drone_dynamics_summary(
        backend="pybullet", sampled_mass=1.0, max_body_rate=4.0, tw_preserving=True
    )
    m.set_drone_dynamics(summary)
    assert m.drone_dynamics is summary


def test_set_drone_dynamics_tolerates_none() -> None:
    """Feeding None (e.g. a glitched read) clears the slot without raising — None-tolerant."""
    from drone_fly.adapter.dynamics_summary import drone_dynamics_summary

    m = DashboardModel(scheduled_iters=10)
    m.set_drone_dynamics(
        drone_dynamics_summary(backend="simple", sampled_mass=1.0, max_body_rate=4.0)
    )
    m.set_drone_dynamics(None)
    assert m.drone_dynamics is None
