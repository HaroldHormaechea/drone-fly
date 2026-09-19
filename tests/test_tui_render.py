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
