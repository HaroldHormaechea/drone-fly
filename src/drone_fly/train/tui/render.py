"""Rich render layer for the training TUI (UC-22, AC3).

Pure ``model → renderable`` functions: given a :class:`~drone_fly.train.tui.metrics.
DashboardModel` (and the captured log scrollback) they build the four wireframe regions —
grouped TIME/TRAIN/ROLLOUT value panel (top-left), the 7 trend rows + iterations progress bar
(middle-left), the raw-log pane (right ~30%), and the full-width health status bar (bottom).
This module is the **untested boundary**: it is constructible headlessly (tests build a layout
without a terminal), but the live draw loop itself is not exercised in CI — exactly like the
viewer and pybullet sim. Rich is imported here and in :mod:`dashboard` only.
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from drone_fly.train.tui import metrics as M

#: Health-verdict severity → Rich style for the bottom status bar.
SEVERITY_STYLE = {"normal": "green", "warning": "yellow", "critical": "bold red"}
CONCERN_MARK = "⚠"
#: Log lines shown in the right pane (tail); the scrollback itself is separately bounded.
_LOG_TAIL = 200


def _fmt_or_dash(value, fmt) -> str:
    return fmt(value) if value is not None else M.PLACEHOLDER


def build_values_panel(model: M.DashboardModel) -> Panel:
    """Top-left grouped raw-value panel: TIME / TRAIN / ROLLOUT (three columns)."""
    cur, sched, _ = model.progress()

    time_col = Text()
    time_col.append("TIME\n", style="bold")
    time_col.append(f" iters   {cur}/{sched}\n")
    time_col.append(f" elapsed {M.format_duration(model.elapsed_seconds)}\n")
    time_col.append(f" eta     {M.format_duration(model.eta_seconds())}\n")
    # UC-26 AC-11: resolved rollout parallelism (worker count + active backend), e.g.
    # " envs  8 (subproc)" — the value shown is the RESOLVED one, not the raw config input.
    time_col.append(f" envs    {model.n_envs} ({model.backend})")

    ent_loss = _fmt_or_dash(model.raw["entropy_loss"], lambda v: f"{v:.2f}")
    value_loss = _fmt_or_dash(model.raw["value_loss"], lambda v: f"{v:.1f}")
    expl_var = _fmt_or_dash(model.raw["explained_variance"], lambda v: f"{v:.2f}")
    train_col = Text()
    train_col.append("TRAIN\n", style="bold")
    train_col.append(f" ent_loss   {ent_loss}\n")
    train_col.append(f" value_loss {value_loss}\n")
    train_col.append(f" expl_var   {expl_var}")

    ep_rew = _fmt_or_dash(model.raw["ep_rew"], lambda v: f"{v:.0f}")
    ep_len = _fmt_or_dash(model.raw["ep_len"], lambda v: f"{v:.0f}")
    success = _fmt_or_dash(model.raw["success"], lambda v: f"{v * 100:.0f}%")
    rollout_col = Text()
    rollout_col.append("ROLLOUT\n", style="bold")
    rollout_col.append(f" ep_rew  {ep_rew}\n")
    rollout_col.append(f" ep_len  {ep_len}\n")
    rollout_col.append(f" success {success}")

    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column()
    grid.add_column()
    grid.add_column()
    grid.add_row(time_col, train_col, rollout_col)
    return Panel(grid, title="drone-fly · training", border_style="cyan")


def build_trends_panel(model: M.DashboardModel) -> Panel:
    """Middle-left panel: the 7 trend rows, the iterations bar, and the UC-30 collecting bar."""
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(justify="left", ratio=2)  # label
    table.add_column(justify="right")  # value
    table.add_column(justify="left")  # sparkline
    table.add_column(justify="left")  # tendency
    table.add_column(justify="left")  # concern
    for row in model.rows():
        tendency = Text(f"{row.icon} {row.trend}")
        mark = Text(CONCERN_MARK, style="yellow") if row.concern else Text("")
        table.add_row(row.label, row.value_text, row.sparkline, tendency, mark)

    cur, sched, frac = model.progress()
    total = max(sched, 1)
    bar = ProgressBar(total=total, completed=min(cur, total))
    prog = Table.grid(expand=True, padding=(0, 1))
    prog.add_column()
    prog.add_column(justify="right")
    prog.add_row(bar, f"{cur}/{sched} ({M.format_pct(frac)})")

    # UC-30: a distinct within-rollout "collecting" bar below the iterations bar, so a slow
    # rollout shows live step-progress instead of a frozen screen. Label is deliberately
    # distinct from "iterations" so both bars stay legible; sourced from tick()-fed counters.
    rcur, rtgt, rfrac = model.rollout_progress()
    rtotal = max(rtgt, 1)
    collecting_bar = ProgressBar(total=rtotal, completed=min(rcur, rtotal))
    collecting = Table.grid(expand=True, padding=(0, 1))
    collecting.add_column()
    collecting.add_column(justify="right")
    collecting.add_row(collecting_bar, f"collecting {rcur}/{rtgt} ({M.format_pct(rfrac)})")

    body = Group(table, Text("iterations", style="bold"), prog, collecting)
    return Panel(body, title="TRENDS", border_style="cyan")


def build_logs_panel(log_lines: list[str] | None) -> Panel:
    """Right-hand raw-log pane (the pybullet/SB3 fd spew), tailing the scrollback."""
    lines = (log_lines or [])[-_LOG_TAIL:]
    text = Text("\n".join(lines), no_wrap=False, overflow="fold")
    return Panel(text, title="logs", border_style="dim")


def build_status_bar(verdict) -> Panel:
    """Full-width bottom status bar rendering the health verdict with severity styling."""
    if verdict is None:
        message, status = "Training progressing normally.", "normal"
    else:
        message = getattr(verdict, "message", str(verdict))
        status = getattr(verdict, "status", "normal")
    style = SEVERITY_STYLE.get(status, "green")
    return Panel(Text(message, style=style), border_style=style)


def build_layout(model: M.DashboardModel, log_lines: list[str] | None = None) -> RenderableType:
    """Assemble the four regions into a Rich :class:`~rich.layout.Layout` (AC3).

    Left column ~70% (values on top, trends below), right log pane ~30%, full-width status bar
    at the bottom. Constructible headlessly; the live draw is the documented untested boundary.
    """
    layout = Layout()
    layout.split_column(
        Layout(name="main", ratio=1),
        Layout(name="status", size=3),
    )
    layout["main"].split_row(
        Layout(name="left", ratio=7),
        Layout(name="logs", ratio=3),
    )
    layout["left"].split_column(
        Layout(build_values_panel(model), name="values", size=7),
        Layout(build_trends_panel(model), name="trends", ratio=1),
    )
    layout["logs"].update(build_logs_panel(log_lines))
    layout["status"].update(build_status_bar(model.latest_verdict))
    return layout


__all__ = [
    "build_layout",
    "build_values_panel",
    "build_trends_panel",
    "build_logs_panel",
    "build_status_bar",
    "SEVERITY_STYLE",
    "CONCERN_MARK",
]
