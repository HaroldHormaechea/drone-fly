"""UC-22 AC3/AC6/AC8 — the Rich render layer (``drone_fly.train.tui.render``).

The live *draw loop* is a documented untested boundary (no interactive TTY in CI, exactly like
the viewer and pybullet). What IS testable — and tested here — is that the ``model → renderable``
builders construct **and render to a string headlessly** without raising:

* ``build_layout`` assembles the four wireframe regions for a representative model (AC3),
* the bottom status bar renders a ``HealthVerdict`` with severity styling (AC6),
* a placeholder / first-rollout model renders (no crash on all-missing values),
* graceful degradation: a too-small terminal and a ``NO_COLOR``/no-color console still render
  without raising (AC8).

We render through a ``rich.console.Console`` backed by ``io.StringIO`` (never a pty) so any
lazy Rich construction error surfaces, while staying fully headless. **No ``Live``, no pty.**
"""

from __future__ import annotations

import io

import pytest

from drone_fly.adapter.dynamics_summary import drone_dynamics_summary
from drone_fly.train.health import HealthReason, HealthVerdict
from drone_fly.train.tui import metrics as M
from drone_fly.train.tui.render import (
    SEVERITY_STYLE,
    STATUS_BORDER_OVERHEAD,
    STATUS_MAX_CONTENT_LINES,
    build_drone_panel,
    build_layout,
    build_status_bar,
    build_trends_panel,
    build_values_panel,
    status_region_size,
)


def _render(
    renderable, *, width: int = 120, height: int | None = None, no_color: bool = False
) -> str:
    """Render to a string via a headless Console (surfaces lazy Rich errors); never a pty.

    ``height`` is passed for full-``build_layout`` renders so the ratio-sized bottom trends panel
    (grown by UC-52's optimizing bar + the values-panel size bump) isn't squeezed below its content
    by Rich's default 25-line console.
    """
    from rich.console import Console

    buf = io.StringIO()
    Console(file=buf, width=width, height=height, no_color=no_color, legacy_windows=False).print(
        renderable
    )
    return buf.getvalue()


def _populated_model() -> M.DashboardModel:
    m = M.DashboardModel(scheduled_iters=488)
    for i in range(6):
        m.update(
            n_updates=i + 1,
            elapsed_seconds=100.0 * (i + 1),
            ep_rew_mean=-1300 + i * 20,
            ep_len_mean=200 + i,
            success_rate=0.02 * i,
            entropy_loss=-(3.0 - 0.1 * i),
            std=0.52 - 0.005 * i,
            value_loss=60.0 - i,
            approx_kl=0.006 + 0.0001 * i,
            explained_variance=0.4 + 0.03 * i,
        )
    return m


# --------------------------------------------------------------------------- #
# AC3 — the four regions render
# --------------------------------------------------------------------------- #


def test_build_layout_renders_for_a_representative_model() -> None:
    m = _populated_model()
    out = _render(build_layout(m, ["pybullet build", "Version = 3.2.5"]))
    assert out.strip()  # produced content, did not raise
    # region titles present
    assert "training" in out
    assert "TRENDS" in out
    assert "logs" in out


def test_build_layout_renders_a_fresh_placeholder_model() -> None:
    """A brand-new model (no update yet / first rollout) must render, not crash."""
    m = M.DashboardModel(scheduled_iters=488)
    out = _render(build_layout(m))
    assert out.strip()
    assert M.PLACEHOLDER in out  # missing values shown as '—', never 0


def test_values_panel_shows_grouped_time_train_rollout() -> None:
    out = _render(build_values_panel(_populated_model()))
    assert "TIME" in out
    assert "TRAIN" in out
    assert "ROLLOUT" in out


def test_trend_rows_and_progress_render() -> None:
    # height=60: UC-52 added a third bar (optimizing) to the trends panel and bumped the values
    # panel to size=9, so the default 25-line console clips the bottom "iterations" row.
    out = _render(build_layout(_populated_model()), height=60)
    # the 7 metric labels appear as trend rows
    for label in (
        "success_rate",
        "ep_rew_mean",
        "entropy",
        "std",
        "value_loss",
        "approx_kl",
        "explained_var",
    ):
        assert label in out
    assert "iterations" in out  # the progress bar row


# --------------------------------------------------------------------------- #
# AC6 — health verdict rendering with severity styling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "status,message",
    [
        ("normal", "Training progressing normally."),
        ("warning", "actor entropy flat & 0% success"),
        ("critical", "value loss diverging"),
    ],
)
def test_status_bar_renders_each_verdict_severity(status: str, message: str) -> None:
    reasons = [HealthReason("r", message, status)] if status != "normal" else []
    verdict = HealthVerdict(status=status, message=message, reasons=reasons)
    panel = build_status_bar(verdict)
    out = _render(panel)
    assert message.split()[0] in out  # the verdict text is rendered
    assert status in SEVERITY_STYLE  # a style is defined for this severity


def test_status_bar_none_verdict_falls_back_to_normal() -> None:
    out = _render(build_status_bar(None))
    assert "normally" in out  # the stubbed "Training progressing normally." message


def test_layout_renders_the_models_verdict_in_the_status_bar() -> None:
    m = _populated_model()
    m.set_verdict(HealthVerdict(status="warning", message="connectome slice likely too small"))
    out = _render(build_layout(m))
    assert "connectome" in out


# --------------------------------------------------------------------------- #
# AC8 — graceful degradation (small / no-color terminal still renders)
# --------------------------------------------------------------------------- #


def test_renders_in_a_tiny_terminal_without_raising() -> None:
    m = _populated_model()
    out = _render(build_layout(m), width=20)  # far too small for the wireframe
    assert isinstance(out, str)  # degraded, but did not raise


def test_renders_with_no_color_without_raising() -> None:
    m = _populated_model()
    out = _render(build_layout(m), no_color=True)
    assert out.strip()


# --------------------------------------------------------------------------- #
# UC-26 AC-11 — the values panel surfaces the resolved n_envs + backend
# --------------------------------------------------------------------------- #


def test_values_panel_shows_resolved_parallelism_for_a_parallel_run() -> None:
    """AC-11: a parallel run renders the resolved worker count + 'subproc' backend, not the raw
    config input, in the values panel (verified against the pure render layer — no live Live)."""
    m = M.DashboardModel(scheduled_iters=488, n_envs=8, backend="subproc")
    out = _render(build_values_panel(m))
    assert "envs" in out
    assert "8 (subproc)" in out


def test_values_panel_shows_resolved_parallelism_for_a_serial_run() -> None:
    """AC-11: the serial (dummy) run shows its resolved 1-env / dummy parallelism too."""
    m = M.DashboardModel(scheduled_iters=488, n_envs=1, backend="dummy")
    out = _render(build_values_panel(m))
    assert "envs" in out
    assert "1 (dummy)" in out


# --------------------------------------------------------------------------- #
# UC-30 — the intra-rollout collecting bar renders, distinct from the UC-22 bar
# --------------------------------------------------------------------------- #


def test_trends_panel_renders_both_the_iterations_and_collecting_bars() -> None:
    """AC-7: the additive UC-30 'collecting' bar renders headlessly AND the UC-22 'iterations'
    bar is still present — two DISTINCT labelled bars, so neither obscures the other."""
    m = _populated_model()
    m.tick(elapsed_seconds=123.0, rollout_steps=512, rollout_target=2048)
    out = _render(build_trends_panel(m), width=200)
    assert "iterations" in out  # UC-22 bar preserved
    assert "collecting" in out  # UC-30 bar added
    assert "512/2048" in out  # sourced from rollout_progress()
    assert "25%" in out


def test_collecting_bar_degrades_to_empty_when_target_unknown() -> None:
    """A pre-tick / missing-attr rollout (target 0) still renders — an empty collecting bar,
    no crash, 0%."""
    m = M.DashboardModel(scheduled_iters=488)  # never ticked -> rollout_target == 0
    out = _render(build_trends_panel(m), width=200)
    assert "collecting" in out
    assert "0/0" in out


def test_heartbeat_tick_does_not_blank_the_values_panel_render() -> None:
    """AC-4 at the render layer: after a populated rollout snapshot, a ``tick`` (live clock +
    collecting counters) must NOT blank the six values-panel entries to the '—' placeholder —
    the last rollout's values stay on screen, only the elapsed clock advances."""
    m = _populated_model()  # 6 full updates -> raw fully populated
    # UC-52 added an always-present collect/optimize "split" line that shows '—' until the first
    # durations are observed; set them so the ONLY thing that could add a placeholder here is a
    # (regression) blanking of the six snapshot entries — the invariant this test guards.
    m.set_collect_duration(12.0)
    m.set_optimize_duration(46.0)
    before = _render(build_values_panel(m))
    assert M.PLACEHOLDER not in before  # every value present after the snapshot
    assert "-1200" in before  # last ep_rew_mean (-1300 + 5*20)
    assert "10m00s" in before  # last elapsed (100 * 6 = 600 s)

    m.tick(elapsed_seconds=999.0, rollout_steps=100, rollout_target=2048)
    after = _render(build_values_panel(m))

    assert M.PLACEHOLDER not in after  # STILL no blanking — the snapshot persists
    assert "-1200" in after  # the rollout value is unchanged
    assert "16m39s" in after  # but the live elapsed clock advanced (999 s)


# --------------------------------------------------------------------------- #
# UC-32 Addendum — the TIME panel's cross-platform 'steps current/total' line
# --------------------------------------------------------------------------- #
#
# A DELIBERATE cross-platform display change (shown on macOS/Linux AND Windows): the TIME panel
# gains a ``steps current/total`` line (thousands-separated) from SB3 ``num_timesteps`` and the
# configured ``total_timesteps``. Per the Addendum this is NOT an AC-9 regression — verify it
# renders (on this non-win32 CI host too) with the correct values.


def test_values_panel_renders_the_steps_line_thousands_separated() -> None:
    """Addendum: the TIME panel shows ``steps 26,624 / 1,000,000`` with thousands separators."""
    m = M.DashboardModel(scheduled_iters=488, total_steps=1_000_000)
    m.update(n_updates=1, elapsed_seconds=10.0, current_steps=26_624)
    out = _render(build_values_panel(m), width=200)
    assert "steps" in out
    assert "26,624" in out  # current (thousands-separated)
    assert "1,000,000" in out  # total (thousands-separated)


def test_steps_line_updates_via_both_tick_and_update() -> None:
    """Addendum: the steps line tracks ``current_steps`` fed by BOTH the heartbeat tick and the
    rollout-end update (it is not frozen between iterations)."""
    m = M.DashboardModel(scheduled_iters=488, total_steps=500_000)
    m.tick(elapsed_seconds=5.0, rollout_steps=100, rollout_target=2048, current_steps=12_000)
    out_tick = _render(build_values_panel(m), width=200)
    assert "12,000" in out_tick  # fed by tick

    m.update(n_updates=2, elapsed_seconds=20.0, current_steps=48_000)
    out_update = _render(build_values_panel(m), width=200)
    assert "48,000" in out_update  # fed by update


def test_steps_line_is_untouched_by_tick_clock() -> None:
    """Addendum: the Windows ~1 Hz clock tick advances elapsed only — the steps line keeps its
    last real value (the timer observed no new steps)."""
    m = M.DashboardModel(scheduled_iters=488, total_steps=1_000_000)
    m.update(n_updates=1, elapsed_seconds=10.0, current_steps=26_624)
    m.tick_clock(999.0)
    out = _render(build_values_panel(m), width=200)
    assert "26,624" in out  # steps unchanged by a pure clock tick
    assert "1,000,000" in out


def test_values_panel_steps_line_not_clipped_in_full_layout() -> None:
    """Addendum: the values panel height was bumped (7→8) for the extra steps line — assert the
    steps line survives inside the full four-region layout (i.e. not clipped by the panel size)."""
    m = M.DashboardModel(scheduled_iters=488, total_steps=1_000_000)
    m.update(n_updates=1, elapsed_seconds=10.0, current_steps=26_624)
    out = _render(build_layout(m, ["log line"]), width=200)
    assert "steps" in out
    assert "26,624" in out


# --------------------------------------------------------------------------- #
# UC-33 Item 1 — bottom status-region autosize (status_region_size)
#
# Headless, pure function: given (message, width) it returns the Rich Layout height for the
# status region. No live terminal draw (AC-4). Height == content rows + STATUS_BORDER_OVERHEAD.
# --------------------------------------------------------------------------- #


def test_status_region_baseline_single_line_is_size_3() -> None:
    """AC-3: a one-line verdict (no width given) yields 1 content row + border overhead == the
    historical hardcoded ``size=3``, so the common case has zero visual regression."""
    assert STATUS_BORDER_OVERHEAD == 2
    assert status_region_size("Training progressing normally.") == 1 + STATUS_BORDER_OVERHEAD
    assert status_region_size("Training progressing normally.") == 3


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_status_region_k_logical_lines_grows_to_min_k_plus_overhead(k: int) -> None:
    """AC-1: a K-line message (split on ``\\n``, no width) allocates min(K, MAX) content rows +
    overhead, so a multi-line health warning renders all its lines instead of clipping to one."""
    message = "\n".join(f"line {i}" for i in range(k))
    expected = min(k, STATUS_MAX_CONTENT_LINES) + STATUS_BORDER_OVERHEAD
    assert status_region_size(message) == expected


def test_status_region_empty_message_still_has_one_content_row() -> None:
    """An empty message clamps to a minimum of one content row (never a zero-height region)."""
    assert status_region_size("") == 1 + STATUS_BORDER_OVERHEAD


def test_status_region_clamps_oversized_message_to_max() -> None:
    """AC-2: a pathologically long message (far more than the cap of logical lines) clamps to
    STATUS_MAX_CONTENT_LINES content rows + overhead — it can never grow to swallow the terminal."""
    huge = "\n".join(f"line {i}" for i in range(STATUS_MAX_CONTENT_LINES * 5))
    assert status_region_size(huge) == STATUS_MAX_CONTENT_LINES + STATUS_BORDER_OVERHEAD


def test_status_region_width_aware_wrap_grows_a_long_single_line() -> None:
    """AC-1: a single logical line longer than the inner content width wraps to multiple rows when
    a width is supplied, so the region grows to show the wrapped text (not just 1 row)."""
    # One long logical line (no embedded newline), rendered against a narrow width so it must wrap.
    message = "word " * 40  # ~200 chars of word-wrappable text
    narrow = status_region_size(message, width=30)
    # With a width it wraps to >1 content row -> taller than the no-width (logical-line) baseline.
    assert narrow > 1 + STATUS_BORDER_OVERHEAD
    # Still bounded by the documented cap (AC-2).
    assert narrow <= STATUS_MAX_CONTENT_LINES + STATUS_BORDER_OVERHEAD


def test_status_region_wider_width_wraps_to_fewer_rows_than_narrow() -> None:
    """Width-aware wrapping is monotonic: the same message needs no more rows at a wider width."""
    message = "word " * 40
    assert status_region_size(message, width=200) <= status_region_size(message, width=30)


def test_status_region_is_deterministic_for_fixed_message_and_width() -> None:
    """AC-5: 'platform-independent' == deterministic given a fixed (message, width) pair, NOT
    identical runtime widths across OSes. Repeated calls return the same height."""
    message = "actor entropy flat & 0% success\nvalue loss trending up\ncheck the connectome slice"
    first = status_region_size(message, width=80)
    for _ in range(5):
        assert status_region_size(message, width=80) == first


def test_status_region_narrow_width_does_not_undercount_wrapped_line() -> None:
    """Challenger rec: Rich-native (word-aware) wrap measurement must not UNDER-count. A single
    logical line that clearly exceeds the inner width reports at least 2 content rows."""
    inner = 20
    message = "x" * (inner * 3)  # 3x the inner content width -> must wrap to multiple rows
    # width = inner + STATUS_PANEL_CHROME_WIDTH(4); the helper subtracts chrome internally.
    size = status_region_size(message, width=inner + 4)
    assert size - STATUS_BORDER_OVERHEAD >= 2


def test_status_region_width_none_uses_logical_line_count() -> None:
    """A single (unwrapped) logical line with width=None counts as exactly one content row,
    regardless of how long the string is — the no-width path never wraps."""
    long_single_line = "x" * 500
    assert status_region_size(long_single_line, width=None) == 1 + STATUS_BORDER_OVERHEAD


def test_layout_status_region_autosizes_for_a_multiline_verdict() -> None:
    """End-to-end at the layout level: a multi-line verdict makes build_layout allocate a taller
    status region (still renders headlessly, AC-4). Uses a message with embedded newlines so the
    height derives from the verdict rather than a hardcoded size."""
    m = _populated_model()
    m.set_verdict(HealthVerdict(status="warning", message="line one\nline two\nline three"))
    # width=None path: 3 logical lines -> 3 + overhead.
    assert status_region_size("line one\nline two\nline three") == 3 + STATUS_BORDER_OVERHEAD
    out = _render(build_layout(m), width=120)
    assert out.strip()  # assembles + renders without raising


# --------------------------------------------------------------------------- #
# UC-49 AC2 — the drone-dynamics top segment
# --------------------------------------------------------------------------- #
def _summary(**overrides):
    """A known DroneDynamicsSummary for the render assertions (pybullet default is T/W 2.25)."""
    kwargs = dict(
        backend="pybullet",
        sampled_mass=1.0,
        max_body_rate=4.0,
        tw_preserving=True,
        spawn_z=1.5,
    )
    kwargs.update(overrides)
    return drone_dynamics_summary(**kwargs)


def test_drone_panel_shows_all_five_fields_from_a_known_summary() -> None:
    """build_drone_panel renders Weight/T/W/hover/body-rate/spawn-z headlessly. (UC-55 dropped
    the attitude-authority field along with the curriculum it displayed.)"""
    m = M.DashboardModel(scheduled_iters=488)
    m.set_drone_dynamics(_summary())
    out = _render(build_drone_panel(m))
    assert out.strip()
    # The five fields' formatted values appear (weight 0.027*9.8=0.26 N, T/W 2.25, hover 0.50,
    # rate 4.0 rad/s, spawn_z 1.50 m).
    assert "0.26 N" in out  # weight
    assert "2.25" in out  # T/W
    assert "0.50" in out  # hover throttle
    assert "4.0 rad/s" in out  # max body rate
    assert "1.50 m" in out  # spawn-z
    # Field labels present so an operator can read the segment.
    for label in ("weight", "T/W", "hover", "rate", "spawn_z"):
        assert label in out


def test_drone_panel_none_summary_renders_placeholders_without_raising() -> None:
    """A model with no summary yet renders '—' placeholders and never crashes (graceful degrade)."""
    m = M.DashboardModel(scheduled_iters=488)  # drone_dynamics defaults to None
    out = _render(build_drone_panel(m))
    assert out.strip()
    assert M.PLACEHOLDER in out


def test_drone_panel_none_spawn_z_renders_placeholder() -> None:
    """spawn_z=None (unknown) renders the placeholder for that field, not a crash."""
    m = M.DashboardModel(scheduled_iters=488)
    m.set_drone_dynamics(_summary(spawn_z=None))
    out = _render(build_drone_panel(m))
    assert out.strip()
    assert "2.25" in out  # other fields still render
    assert M.PLACEHOLDER in out  # the spawn_z field degrades to placeholder


def test_build_layout_includes_the_drone_segment() -> None:
    """The full layout carries the drone segment values when a summary is set (end-to-end AC2)."""
    m = _populated_model()
    m.set_drone_dynamics(_summary())
    out = _render(build_layout(m, ["pybullet build"]))
    assert out.strip()
    assert "T/W" in out
    assert "2.25" in out


def test_build_layout_renders_without_drone_summary() -> None:
    """A layout whose model has no drone summary still renders (segment shows placeholders)."""
    m = _populated_model()  # no set_drone_dynamics call
    out = _render(build_layout(m))
    assert out.strip()
