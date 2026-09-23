"""UC-52 — optimize-phase progress + collect/optimize timing in the training TUI.

All hermetic (no GPU/pybullet). Covers the six-part test plan from
``use-cases/plans/52-optimize-phase-tui-progress.md`` § "Test plan (AC-10)":

* (a) render assertions — the ``optimizing`` bar (``build_trends_panel``) and the collect/optimize
  ``split`` line (``build_values_panel``) across synthetic begin/tick/end sequences (fresh
  placeholder, 0%, mid, 100%, divergent-shape clamp), plus a headless full-layout build.
* (b) the pure :func:`~drone_fly.train.tui... progress_ppo.counting_get_proxy` helper with fake
  iterables + a recording ``on_tick`` → exactly ``N×M`` ticks with the right ``(epoch, minibatch)``
  indices (incl. a partial final minibatch), samples forwarded untouched; and — via a fake sink
  through :meth:`ProgressReportingPPO.train` — **exactly one begin / one end** (single-owner).
* (c) a hermetic smoke run: a real :class:`ProgressReportingPPO` on the committed connectome
  fixture (numpy ``simple`` adapter, CPU, 256 steps) + a recording sink proving each iteration
  emits ``begin_optimize → tick_optimize(≥1) → end_optimize`` with the correct ``N``/``M``.
* (d) the 1 Hz throttle — ``dashboard.tick_optimize`` gated to ≈1 redraw/sec via an injected
  clock, while ``begin_optimize`` / ``end_optimize`` force an immediate redraw regardless.
* (e) unattached-sink byte-identity — ``train()`` with no sink calls ``super().train()`` and never
  wraps ``rollout_buffer.get``.

Plus: non-clobber (begin/tick/end_optimize leave ``raw``/``history`` untouched); the ``min(cur,
total)`` clamp (AC-7); and AC-3 (``_OPTIMIZE_MIN_INTERVAL`` distinct, ``_HEARTBEAT_MIN_INTERVAL``
unchanged).
"""

from __future__ import annotations

import io
import math

import pytest

from drone_fly.train.progress_ppo import ProgressReportingPPO, counting_get_proxy
from drone_fly.train.tui import metrics as M
from drone_fly.train.tui.render import build_layout, build_trends_panel, build_values_panel


# --------------------------------------------------------------------------- #
# Shared test doubles
# --------------------------------------------------------------------------- #
class _FakeClock:
    """A controllable monotonic source (per ``test_tui_refresh_clock``): advance by hand."""

    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeLive:
    """Headless Rich ``Live`` sink: records each rebuilt renderable, never a terminal."""

    def __init__(self) -> None:
        self.renderables: list = []

    def update(self, renderable) -> None:
        self.renderables.append(renderable)


class _RecordingSink:
    """A duck-typed optimize-phase sink that records the exact call sequence."""

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.collect_durations: list = []
        self.optimize_durations: list = []

    def begin_optimize(self, n_epochs, total_minibatches) -> None:
        self.events.append(("begin", n_epochs, total_minibatches))

    def tick_optimize(self, epoch, minibatch) -> None:
        self.events.append(("tick", epoch, minibatch))

    def end_optimize(self) -> None:
        self.events.append(("end",))

    def set_collect_duration(self, seconds) -> None:
        self.collect_durations.append(seconds)

    def set_optimize_duration(self, seconds) -> None:
        self.optimize_durations.append(seconds)


def _render(
    renderable, *, width: int = 200, height: int | None = None, no_color: bool = False
) -> str:
    """Render to a string via a headless Console (surfaces lazy Rich errors); never a pty.

    ``height`` is given for full-``build_layout`` renders so the ratio-sized bottom trends panel
    isn't squeezed below its content by Rich's default 25-line console (the individual panel
    builders render unclipped without it).
    """
    from rich.console import Console

    buf = io.StringIO()
    console = Console(
        file=buf,
        width=width,
        height=height,
        no_color=no_color,
        legacy_windows=False,
    )
    console.print(renderable)
    return buf.getvalue()


def _group_iterations(events: list[tuple]) -> list[list[tuple]]:
    """Split a flat event stream into per-``begin`` iteration groups."""
    groups: list[list[tuple]] = []
    for ev in events:
        if ev[0] == "begin":
            groups.append([])
        if groups:
            groups[-1].append(ev)
    return groups


# =========================================================================== #
# (a) render — the optimizing bar (build_trends_panel)
# =========================================================================== #
def test_trends_panel_placeholder_when_not_optimizing() -> None:
    """A fresh model (never entered optimize) shows the ``optimizing —`` placeholder, no crash."""
    m = M.DashboardModel(scheduled_iters=100)
    out = _render(build_trends_panel(m))
    assert "optimizing" in out
    assert "—" in out
    # the collecting bar (UC-30) is still present — the optimizing bar is additive, not a replace
    assert "collecting" in out


def test_trends_panel_optimizing_bar_at_zero_percent() -> None:
    """After ``begin_optimize`` but before the first tick: epoch 1-based, minibatch 0/M, 0%."""
    m = M.DashboardModel(scheduled_iters=100)
    m.begin_optimize(10, 4)
    out = _render(build_trends_panel(m))
    assert "optimizing epoch 1/10 · minibatch 0/4 (0%)" in out


def test_trends_panel_optimizing_bar_midway() -> None:
    """Mid-optimize: cumulative fraction ``(epoch*M+minibatch)/(N*M)`` and 1-based epoch display."""
    m = M.DashboardModel(scheduled_iters=100)
    m.begin_optimize(10, 4)
    m.tick_optimize(4, 2)  # cur = 4*4 + 2 = 18 ; total = 40 -> 45%
    out = _render(build_trends_panel(m))
    assert "optimizing epoch 5/10 · minibatch 2/4 (45%)" in out


def test_trends_panel_optimizing_bar_reaches_exactly_100_on_final_minibatch() -> None:
    """The 0-based epoch / 1-based minibatch convention hits exactly 100% on the last minibatch."""
    m = M.DashboardModel(scheduled_iters=100)
    m.begin_optimize(10, 4)
    m.tick_optimize(9, 4)  # cur = 9*4 + 4 = 40 == total -> 100%
    out = _render(build_trends_panel(m))
    assert "optimizing epoch 10/10 · minibatch 4/4 (100%)" in out


def test_trends_panel_clamps_divergent_buffer_shape() -> None:
    """AC-7: a divergent buffer shape can never render e.g. ``5/4`` — text is ``min(cur,total)``."""
    m = M.DashboardModel(scheduled_iters=100)
    m.begin_optimize(10, 4)
    m.tick_optimize(9, 5)  # minibatch 5 > M=4, cur = 41 > total 40
    out = _render(build_trends_panel(m))
    assert "minibatch 4/4" in out  # clamped, NOT "5/4"
    assert "5/4" not in out
    assert "(100%)" in out  # fraction clamped to 1.0
    assert "epoch 10/10" in out  # epoch clamped too (min(9+1, 10))


# =========================================================================== #
# (a) render — the collect/optimize split line (build_values_panel)
# =========================================================================== #
def test_values_panel_split_line_placeholder_before_any_timing() -> None:
    """Before any duration is observed the split line shows placeholders, never 0s / a crash."""
    m = M.DashboardModel(scheduled_iters=100)
    out = _render(build_values_panel(m))
    assert "split" in out
    assert "c — · o — (—)" in out


def test_values_panel_split_line_with_both_durations() -> None:
    """The split renders ``c <collect> · o <optimize> (<optimize-share>)`` from ``phase_split``."""
    m = M.DashboardModel(scheduled_iters=100)
    m.set_collect_duration(12.0)
    m.set_optimize_duration(46.0)  # optimize share = 46/58 = 79%
    out = _render(build_values_panel(m))
    assert "split   c 12s · o 46s (79%)" in out


def test_values_panel_split_line_partial_timing_degrades_gracefully() -> None:
    """Only one side observed → the share is ``—`` (guarded), the known side still shows."""
    m = M.DashboardModel(scheduled_iters=100)
    m.set_collect_duration(12.0)  # optimize still None
    out = _render(build_values_panel(m))
    assert "c 12s · o — (—)" in out


def test_full_layout_builds_headlessly_with_optimize_state() -> None:
    """The whole four-region layout assembles + renders with an active optimize phase (AC-10a)."""
    m = M.DashboardModel(scheduled_iters=100, total_steps=1_000_000)
    m.update(n_updates=1, elapsed_seconds=10.0, ep_rew_mean=-100.0, current_steps=2048)
    m.begin_optimize(10, 4)
    m.tick_optimize(3, 1)
    m.set_collect_duration(12.0)
    m.set_optimize_duration(46.0)
    out = _render(build_layout(m, ["log line one", "log line two"]), height=60)
    assert out.strip()
    assert "optimizing" in out  # optimize bar survives inside the full layout
    assert "split" in out  # split line survives (values panel bumped 8->9)
    assert "c 12s" in out  # the collect/optimize split values render in-layout


def test_values_panel_split_line_not_clipped_in_full_layout() -> None:
    """The values panel size bump (8->9) keeps the UC-52 split line visible in the full layout."""
    m = M.DashboardModel(scheduled_iters=100, total_steps=1_000_000)
    m.update(n_updates=1, elapsed_seconds=10.0, current_steps=26_624)
    m.set_collect_duration(9.0)
    m.set_optimize_duration(41.0)
    out = _render(build_layout(m, ["log"]))
    assert "split" in out
    assert "c 9s" in out


# =========================================================================== #
# model getters — optimize_progress / phase_split math (AC-1, AC-2, AC-7)
# =========================================================================== #
def test_optimize_progress_cumulative_math() -> None:
    m = M.DashboardModel(scheduled_iters=100)
    m.begin_optimize(10, 4)
    m.tick_optimize(4, 2)
    cur, total, frac = m.optimize_progress()
    assert (cur, total) == (18, 40)
    assert frac == pytest.approx(0.45)


def test_optimize_progress_fraction_clamped_to_one() -> None:
    """AC-7: a divergent shape may push cur past total, but the fraction clamps to 1.0."""
    m = M.DashboardModel(scheduled_iters=100)
    m.begin_optimize(10, 4)
    m.tick_optimize(9, 5)  # cur = 41 > total 40
    cur, total, frac = m.optimize_progress()
    assert cur == 41 and total == 40  # cur itself returned unclamped
    assert frac == 1.0  # fraction clamped


def test_phase_split_optimize_share_and_none_guards() -> None:
    m = M.DashboardModel(scheduled_iters=100)
    # both None -> share None
    assert m.phase_split() == (None, None, None)
    m.set_collect_duration(12.0)
    m.set_optimize_duration(46.0)
    collect_s, optimize_s, frac = m.phase_split()
    assert (collect_s, optimize_s) == (12.0, 46.0)
    assert frac == pytest.approx(46.0 / 58.0)


def test_phase_split_zero_sum_guarded() -> None:
    """A zero-sum pair (both 0) yields ``None`` for the share rather than dividing by zero."""
    m = M.DashboardModel(scheduled_iters=100)
    m.set_collect_duration(0.0)
    m.set_optimize_duration(0.0)
    _, _, frac = m.phase_split()
    assert frac is None


# =========================================================================== #
# non-clobber — begin/tick/end_optimize leave raw/history untouched
# =========================================================================== #
def test_optimize_methods_do_not_clobber_raw_or_history() -> None:
    """Like ``tick``, the optimize methods must not blank the values panel snapshot (AC-4)."""
    m = M.DashboardModel(scheduled_iters=100)
    m.update(
        n_updates=3,
        elapsed_seconds=50.0,
        ep_rew_mean=-200.0,
        ep_len_mean=210.0,
        success_rate=0.5,
        entropy_loss=-2.0,
        value_loss=40.0,
        explained_variance=0.3,
    )
    raw_before = dict(m.raw)
    hist_before = {k: h.values() for k, h in m.history.items()}

    m.begin_optimize(10, 4)
    m.tick_optimize(2, 3)
    m.set_collect_duration(12.0)
    m.set_optimize_duration(46.0)
    m.end_optimize()

    assert m.raw == raw_before  # values-panel snapshot persists through optimize
    assert {k: h.values() for k, h in m.history.items()} == hist_before  # no fabricated history


def test_end_optimize_keeps_last_counts_for_a_clean_final_frame() -> None:
    m = M.DashboardModel(scheduled_iters=100)
    m.begin_optimize(10, 4)
    m.tick_optimize(9, 4)
    m.end_optimize()
    assert m.is_optimizing is False
    assert (m.optimize_epoch, m.optimize_minibatch) == (9, 4)  # last counts retained


# =========================================================================== #
# (b) counting_get_proxy — pure helper
# =========================================================================== #
def _fake_get_factory(minibatches_per_epoch: int):
    """Return a fake ``rollout_buffer.get``: each call yields distinct sentinel minibatches."""
    call = {"epoch": 0}

    def fake_get(batch_size=None):
        e = call["epoch"]
        call["epoch"] += 1
        for i in range(minibatches_per_epoch):
            yield ("sample", e, i)

    return fake_get


def test_counting_proxy_emits_exactly_n_times_m_ticks_with_correct_indices() -> None:
    N, Mn = 3, 4
    ticks: list[tuple[int, int]] = []
    proxy = counting_get_proxy(_fake_get_factory(Mn), N, Mn, lambda e, mb: ticks.append((e, mb)))

    consumed = []
    for _ in range(N):  # each call is one epoch
        consumed.extend(list(proxy(32)))

    assert len(ticks) == N * Mn
    expected = [(e, mb) for e in range(N) for mb in range(1, Mn + 1)]
    assert ticks == expected  # 0-based epoch, 1-based minibatch


def test_counting_proxy_forwards_samples_untouched() -> None:
    """The proxy must yield the underlying samples byte-identically (never alters training data)."""
    N, Mn = 2, 3
    real = _fake_get_factory(Mn)
    # Capture what the real generator would yield, independently.
    reference = [list(real(32)) for _ in range(N)]

    real2 = _fake_get_factory(Mn)
    proxy = counting_get_proxy(real2, N, Mn, lambda e, mb: None)
    got = [list(proxy(32)) for _ in range(N)]

    assert got == reference


def test_counting_proxy_handles_partial_final_minibatch() -> None:
    """A partial final minibatch (fewer real yields than M) still ticks per real minibatch."""
    # total_minibatches computed elsewhere as M=3, but the buffer only yields 3 rows (last partial).
    N, Mn = 1, 3
    ticks: list[tuple[int, int]] = []
    proxy = counting_get_proxy(_fake_get_factory(3), N, Mn, lambda e, mb: ticks.append((e, mb)))
    list(proxy(32))
    assert ticks == [(0, 1), (0, 2), (0, 3)]


def test_counting_proxy_disables_on_excess_epochs_but_keeps_yielding() -> None:
    """AC-6: more get() calls than N stops reporting for the extra epoch — samples still flow."""
    N, Mn = 2, 2
    ticks: list[tuple[int, int]] = []
    proxy = counting_get_proxy(_fake_get_factory(Mn), N, Mn, lambda e, mb: ticks.append((e, mb)))
    all_samples = []
    for _ in range(N + 1):  # one extra epoch beyond N
        all_samples.extend(list(proxy(32)))
    # ticks only for the first N epochs; the excess epoch reports nothing.
    assert ticks == [(0, 1), (0, 2), (1, 1), (1, 2)]
    # but every sample (incl. the excess epoch's) is still yielded — training never perturbed.
    assert len(all_samples) == (N + 1) * Mn


def test_counting_proxy_disables_on_excess_minibatches_but_keeps_yielding() -> None:
    """AC-6/AC-7: an epoch yielding more than M minibatches stops reporting, keeps yielding."""
    N, Mn = 1, 2
    ticks: list[tuple[int, int]] = []
    proxy = counting_get_proxy(_fake_get_factory(4), N, Mn, lambda e, mb: ticks.append((e, mb)))
    samples = list(proxy(32))  # real yields 4 > M=2
    assert ticks == [(0, 1), (0, 2)]  # stopped reporting after M
    assert len(samples) == 4  # all four still yielded


def test_counting_proxy_sink_error_never_propagates_out_of_yield() -> None:
    """A sink glitch inside ``on_tick`` disables reporting but never breaks the yield stream.

    (The production wiring guards each sink call, but the proxy also defends itself.)
    """
    N, Mn = 1, 3

    def boom(e, mb):
        raise RuntimeError("sink blew up")

    proxy = counting_get_proxy(_fake_get_factory(Mn), N, Mn, boom)
    samples = list(proxy(32))  # must not raise
    assert len(samples) == Mn


# =========================================================================== #
# (b) single-owner begin/end via ProgressReportingPPO.train()
# =========================================================================== #
def _make_bare_reporting_ppo(sink, *, n_epochs, n_steps, n_envs, batch_size, rollout_buffer):
    """Build a ProgressReportingPPO WITHOUT running PPO.__init__ (no torch/env), for train() unit
    tests. Only the attributes ``train()`` reads are populated."""
    model = ProgressReportingPPO.__new__(ProgressReportingPPO)
    model._progress_sink = sink
    model.n_epochs = n_epochs
    model.n_steps = n_steps
    model.n_envs = n_envs
    model.batch_size = batch_size
    model.rollout_buffer = rollout_buffer
    return model


class _FakeBuffer:
    """Minimal rollout buffer whose ``get`` yields ``m`` sentinel minibatches per call."""

    def __init__(self, m: int) -> None:
        self._m = m
        self.get = self._real_get

    def _real_get(self, batch_size=None):
        for i in range(self._m):
            yield ("mb", i)


def test_train_is_single_owner_of_begin_and_end(monkeypatch) -> None:
    """AC-10b: through ``train()`` exactly ONE begin and ONE end fire; ticks come from the proxy.

    ``super().train()`` (the real PPO optimize body) is stubbed to just drain
    ``rollout_buffer.get`` N times — so we exercise the wrapping/ownership logic, not gradients.
    """
    from stable_baselines3 import PPO

    def fake_ppo_train(self):
        for _ in range(self.n_epochs):
            for _mb in self.rollout_buffer.get(self.batch_size):
                pass

    monkeypatch.setattr(PPO, "train", fake_ppo_train)

    sink = _RecordingSink()
    # n_steps=4, n_envs=1, batch_size=2 -> M = ceil(4/2) = 2 ; N = 3
    buf = _FakeBuffer(m=2)
    model = _make_bare_reporting_ppo(
        sink, n_epochs=3, n_steps=4, n_envs=1, batch_size=2, rollout_buffer=buf
    )

    model.train()

    begins = [e for e in sink.events if e[0] == "begin"]
    ends = [e for e in sink.events if e[0] == "end"]
    ticks = [e for e in sink.events if e[0] == "tick"]
    assert len(begins) == 1
    assert len(ends) == 1
    assert begins[0] == ("begin", 3, 2)  # N, M forwarded
    assert len(ticks) == 3 * 2  # N * M ticks from the proxy
    assert sink.events[0] == ("begin", 3, 2)  # begin first
    assert sink.events[-1] == ("end",)  # end last
    assert len(sink.optimize_durations) == 1  # duration set exactly once, before end
    # buffer.get restored after train() (the finally clause).
    assert buf.get == buf._real_get


def test_train_restores_get_and_ends_even_when_super_raises(monkeypatch) -> None:
    """AC-6: if the optimize body raises, ``rollout_buffer.get`` is still restored and end fires."""
    from stable_baselines3 import PPO

    def boom_train(self):
        # touch get so the wrap is installed, then fail like a real training error
        next(iter(self.rollout_buffer.get(self.batch_size)))
        raise RuntimeError("gradient explosion")

    monkeypatch.setattr(PPO, "train", boom_train)

    sink = _RecordingSink()
    buf = _FakeBuffer(m=2)
    model = _make_bare_reporting_ppo(
        sink, n_epochs=2, n_steps=4, n_envs=1, batch_size=2, rollout_buffer=buf
    )

    with pytest.raises(RuntimeError, match="gradient explosion"):
        model.train()

    assert buf.get == buf._real_get  # restored despite the exception
    assert ("end",) in sink.events  # end still fired exactly once
    assert sum(1 for e in sink.events if e[0] == "end") == 1


# =========================================================================== #
# (e) unattached-sink byte-identity
# =========================================================================== #
def test_train_with_no_sink_calls_super_and_never_wraps_get(monkeypatch) -> None:
    """AC-4: with no sink, ``train()`` returns ``super().train()`` and never touches ``get``."""
    from stable_baselines3 import PPO

    called = {"n": 0}

    def fake_ppo_train(self):
        called["n"] += 1

    monkeypatch.setattr(PPO, "train", fake_ppo_train)

    sentinel_get = object()

    class _SpyBuffer:
        pass

    buf = _SpyBuffer()
    buf.get = sentinel_get  # a non-callable sentinel: if train() wrapped it, identity changes

    model = ProgressReportingPPO.__new__(ProgressReportingPPO)
    # deliberately DO NOT set _progress_sink (getattr(..., None) -> None => inert path)
    model.rollout_buffer = buf

    model.train()

    assert called["n"] == 1  # super().train() ran
    assert buf.get is sentinel_get  # get untouched — no proxy wrapping on the inert path


def test_train_with_missing_buffer_degrades_to_super(monkeypatch) -> None:
    """A sink attached but no usable ``rollout_buffer.get`` still degrades to plain training."""
    from stable_baselines3 import PPO

    called = {"n": 0}
    monkeypatch.setattr(PPO, "train", lambda self: called.__setitem__("n", called["n"] + 1))

    model = ProgressReportingPPO.__new__(ProgressReportingPPO)
    model._progress_sink = _RecordingSink()
    model.rollout_buffer = None  # nothing safe to wrap

    model.train()
    assert called["n"] == 1  # ran plain PPO.train, no crash


# =========================================================================== #
# (d) 1 Hz throttle — dashboard.tick_optimize
# =========================================================================== #
def _make_dashboard_with_fake_live(clock):
    from drone_fly.train.tui.dashboard import TrainingDashboard

    dash = TrainingDashboard(scheduled_iters=100, capture=False, now=clock)
    dash._live = _FakeLive()
    return dash


def test_tick_optimize_is_throttled_to_one_hz_but_begin_end_force_redraw() -> None:
    """AC-3: begin + end redraw at once; ticks in between are gated to ~1/sec via ``self._now``."""
    from drone_fly.train.tui.dashboard import _OPTIMIZE_MIN_INTERVAL

    assert _OPTIMIZE_MIN_INTERVAL == 1.0
    clock = _FakeClock(t=100.0)
    dash = _make_dashboard_with_fake_live(clock)
    live = dash._live

    dash.begin_optimize(2, 3)  # immediate redraw (#1), gate reset to None
    assert len(live.renderables) == 1

    dash.tick_optimize(0, 1)  # gate None -> first tick redraws (#2)
    assert len(live.renderables) == 2

    dash.tick_optimize(0, 2)  # +0s -> gated, no redraw
    clock.advance(0.4)
    dash.tick_optimize(0, 3)  # +0.4s -> still gated
    assert len(live.renderables) == 2

    clock.advance(0.7)  # now 1.1s since last redraw -> passes the gate
    dash.tick_optimize(1, 1)  # redraws (#3)
    assert len(live.renderables) == 3

    dash.end_optimize()  # immediate redraw regardless of gate (#4)
    assert len(live.renderables) == 4


def test_tick_optimize_mutates_counters_even_when_redraw_is_gated() -> None:
    """The counter mutation is unconditional/cheap; only the Rich rebuild is throttled."""
    clock = _FakeClock(t=100.0)
    dash = _make_dashboard_with_fake_live(clock)

    dash.begin_optimize(10, 4)
    dash.tick_optimize(0, 1)  # redraws + records
    dash.tick_optimize(0, 2)  # gated (no redraw) ...
    dash.tick_optimize(0, 3)  # ... but the model still advances

    assert dash.model.optimize_epoch == 0
    assert dash.model.optimize_minibatch == 3  # last (gated) tick still applied to the model


def test_begin_optimize_resets_the_gate_so_first_tick_of_each_phase_redraws() -> None:
    """Each ``begin_optimize`` resets ``_last_optimize_redraw`` to None so the next tick redraws."""
    clock = _FakeClock(t=100.0)
    dash = _make_dashboard_with_fake_live(clock)

    dash.begin_optimize(2, 2)
    dash.tick_optimize(0, 1)
    base = len(dash._live.renderables)

    # Second phase, clock has NOT advanced a full second — but begin resets the gate.
    dash.begin_optimize(2, 2)  # immediate redraw
    dash.tick_optimize(0, 1)  # gate was reset -> redraws immediately again
    assert len(dash._live.renderables) == base + 2


def test_optimize_sink_methods_never_raise_into_training() -> None:
    """A redraw failure inside the sink methods is swallowed (never perturbs PPO.train())."""

    class _BoomLive:
        def update(self, renderable):
            raise RuntimeError("render blew up")

    clock = _FakeClock(t=100.0)
    from drone_fly.train.tui.dashboard import TrainingDashboard

    dash = TrainingDashboard(scheduled_iters=100, capture=False, now=clock)
    dash._live = _BoomLive()
    # None of these may raise even though every redraw fails.
    dash.begin_optimize(2, 2)
    dash.tick_optimize(0, 1)
    dash.end_optimize()
    dash.set_collect_duration(1.0)
    dash.set_optimize_duration(2.0)


def test_duration_setters_do_not_redraw() -> None:
    """The two duration setters mutate only — no redraw (mirroring ``set_verdict``)."""
    clock = _FakeClock(t=100.0)
    dash = _make_dashboard_with_fake_live(clock)
    dash.set_collect_duration(12.0)
    dash.set_optimize_duration(46.0)
    assert dash._live.renderables == []  # neither triggered a redraw
    assert dash.model.collect_seconds == 12.0
    assert dash.model.optimize_seconds == 46.0


# =========================================================================== #
# AC-3 — the optimize throttle constant is DISTINCT and the heartbeat is UNCHANGED
# =========================================================================== #
def test_optimize_interval_distinct_from_heartbeat_which_is_unchanged() -> None:
    from drone_fly.train.tui.callback import _HEARTBEAT_MIN_INTERVAL
    from drone_fly.train.tui.dashboard import _OPTIMIZE_MIN_INTERVAL

    assert _OPTIMIZE_MIN_INTERVAL == 1.0
    assert _HEARTBEAT_MIN_INTERVAL == 0.25  # collection heartbeat untouched by UC-52
    assert _OPTIMIZE_MIN_INTERVAL != _HEARTBEAT_MIN_INTERVAL


# =========================================================================== #
# (c) hermetic smoke run — real ProgressReportingPPO on the committed fixture
# =========================================================================== #
pytest.importorskip("stable_baselines3")


def _M(n_steps: int, n_envs: int, batch_size: int) -> int:
    return math.ceil(n_steps * n_envs / batch_size)


def test_smoke_reporting_ppo_emits_begin_ticks_end_per_iteration(connectome) -> None:
    """AC-10c: a real (CPU, numpy-adapter, seeded) ProgressReportingPPO with a recording sink
    emits ``begin_optimize → tick_optimize(≥1) → end_optimize`` with the right N/M per iter."""
    from drone_fly.env.racing_env import build_vec_env
    from drone_fly.train.config import TrainConfig
    from drone_fly.train.loop import build_policy_kwargs

    n_steps, batch_size, n_epochs, n_envs = 64, 32, 2, 1
    venv = build_vec_env(adapter="simple", n_envs=n_envs, seed=0, training=True)
    cfg = TrainConfig(seed=0)
    model = ProgressReportingPPO(
        "MlpPolicy",
        venv,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        seed=0,
        device="cpu",
        policy_kwargs=build_policy_kwargs(connectome, cfg),
    )
    sink = _RecordingSink()
    model.attach_progress_sink(sink)

    total_timesteps = 256  # -> 256/64 = 4 optimize iterations
    model.learn(total_timesteps=total_timesteps)

    expected_M = _M(n_steps, n_envs, batch_size)  # ceil(64/32) = 2
    groups = _group_iterations(sink.events)
    assert len(groups) >= 1  # at least one full optimize phase ran

    # begins and ends are balanced (single-owner, one each per iteration).
    assert sum(1 for e in sink.events if e[0] == "begin") == len(groups)
    assert sum(1 for e in sink.events if e[0] == "end") == len(groups)

    for grp in groups:
        assert grp[0] == ("begin", n_epochs, expected_M)  # correct N/M
        assert grp[-1] == ("end",)  # end closes the phase
        ticks = [e for e in grp if e[0] == "tick"]
        assert len(ticks) >= 1  # at least one tick (AC-10c)
        # target_kl is None in the drone-fly config, so no early-stop: exactly N*M ticks.
        assert len(ticks) == n_epochs * expected_M
        # indices are the expected 0-based-epoch / 1-based-minibatch grid, in order.
        assert [(e[1], e[2]) for e in ticks] == [
            (ep, mb) for ep in range(n_epochs) for mb in range(1, expected_M + 1)
        ]

    # the optimize-duration setter fired once per iteration too.
    assert len(sink.optimize_durations) == len(groups)


def test_smoke_reporting_ppo_without_sink_trains_clean(connectome) -> None:
    """AC-4 end-to-end: with no sink attached the subclass learns exactly like plain PPO (no crash,
    no sink calls) — the inert path stays byte-identical in behaviour."""
    from drone_fly.env.racing_env import build_vec_env
    from drone_fly.train.config import TrainConfig
    from drone_fly.train.loop import build_policy_kwargs

    venv = build_vec_env(adapter="simple", n_envs=1, seed=0, training=True)
    cfg = TrainConfig(seed=0)
    model = ProgressReportingPPO(
        "MlpPolicy",
        venv,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        seed=0,
        device="cpu",
        policy_kwargs=build_policy_kwargs(connectome, cfg),
    )
    # no attach_progress_sink -> inert
    model.learn(total_timesteps=128)  # must complete without raising
    assert model.num_timesteps >= 128
