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

from drone_fly.train.health import HealthReason, HealthVerdict
from drone_fly.train.tui import metrics as M
from drone_fly.train.tui.render import (
    SEVERITY_STYLE,
    build_layout,
    build_status_bar,
    build_trends_panel,
    build_values_panel,
)


def _render(renderable, *, width: int = 120, no_color: bool = False) -> str:
    """Render to a string via a headless Console (surfaces lazy Rich errors); never a pty."""
    from rich.console import Console

    buf = io.StringIO()
    Console(file=buf, width=width, no_color=no_color, legacy_windows=False).print(renderable)
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
    out = _render(build_layout(_populated_model()))
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
