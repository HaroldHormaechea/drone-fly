"""UC-32 AC-5 — the Windows ~1 Hz refresh clock (independent of PPO iteration boundaries).

On Windows a 2048-step rollout can take ~100 s, and the dashboard only redrew at PPO update
boundaries — so the elapsed timer looked frozen during ``PPO.train()`` (which fires no ``on_step``).
UC-32 adds a background ~1 Hz timer whose pure seam is
:meth:`TrainingDashboard._refresh_tick`: it advances the live elapsed clock (via the model's
clock-only :meth:`DashboardModel.tick_clock`) and redraws, WITHOUT touching the rollout/step
counters or the metric history (the timer has observed no new steps). The timer is Windows-guarded
so macOS/Linux behaviour is byte-identical (AC-9).

Hermetic (AC-11): the monotonic clock is injected, so ``elapsed`` is driven deterministically with
zero real sleeps; the thread wrapper is the documented untested boundary — the ``_refresh_tick``
seam and the win32 guard are what these tests pin. ``sys.platform`` is monkeypatched; no real
Windows/terminal.
"""

from __future__ import annotations

import sys

import pytest

from drone_fly.train.tui.dashboard import TrainingDashboard
from drone_fly.train.tui.metrics import DashboardModel


class _FakeClock:
    """A controllable monotonic source: advance it by hand, never sleep."""

    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeLive:
    """Headless Live sink: records each rebuilt renderable, no terminal."""

    def __init__(self) -> None:
        self.renderables: list = []

    def update(self, renderable) -> None:
        self.renderables.append(renderable)


# --------------------------------------------------------------------------- #
# DashboardModel.tick_clock — advances ONLY the elapsed clock (AC-5)
# --------------------------------------------------------------------------- #
def test_tick_clock_advances_elapsed_only() -> None:
    m = DashboardModel(scheduled_iters=100)
    m.tick_clock(42.0)
    assert m.elapsed_seconds == 42.0


def test_tick_clock_does_not_move_step_or_rollout_counters() -> None:
    """The timer has observed no new steps, so tick_clock must fabricate no progress: rollout and
    cumulative-step counters, history, and raw all stay put — even after an update filled them."""
    m = DashboardModel(scheduled_iters=100, total_steps=1_000_000)
    m.update(
        n_updates=3,
        elapsed_seconds=50.0,
        ep_rew_mean=-200.0,
        ep_len_mean=210.0,
        success_rate=0.5,
        current_steps=26_624,
    )
    m.tick(elapsed_seconds=51.0, rollout_steps=512, rollout_target=8192, current_steps=27_000)
    raw_before = dict(m.raw)
    hist_before = {k: h.values() for k, h in m.history.items()}

    m.tick_clock(999.0)

    assert m.elapsed_seconds == 999.0  # clock advanced
    assert m.rollout_steps == 512  # collecting counter untouched
    assert m.rollout_target == 8192
    assert m.current_steps == 27_000  # cumulative step count untouched (no fabricated progress)
    assert m.current_iter == 3
    assert m.raw == raw_before  # values-panel snapshot persists
    assert {k: h.values() for k, h in m.history.items()} == hist_before  # no fabricated history


def test_tick_clock_updates_eta_via_elapsed() -> None:
    """ETA is derived from elapsed × progress, so a clock tick keeps ETA fresh mid-train."""
    m = DashboardModel(scheduled_iters=10)
    m.update(n_updates=2, elapsed_seconds=10.0)  # 2/10 done in 10 s
    eta_before = m.eta_seconds()
    m.tick_clock(20.0)  # same progress, more elapsed -> larger ETA estimate
    assert m.eta_seconds() is not None
    assert m.eta_seconds() > eta_before


# --------------------------------------------------------------------------- #
# _refresh_tick — the pure timer seam advances elapsed + redraws (AC-5)
# --------------------------------------------------------------------------- #
def test_refresh_tick_advances_elapsed_at_one_hz_with_zero_on_step() -> None:
    """AC-5: driving _refresh_tick once per simulated second advances elapsed ~1 Hz and redraws
    each time — with NO callback / _on_step involved, so the clock ticks during PPO.train()."""
    clock = _FakeClock(t=100.0)
    dash = TrainingDashboard(scheduled_iters=100, capture=False, total_steps=1_000_000)
    live = _FakeLive()
    dash._live = live
    dash._timer_start_monotonic = clock()  # timer baseline (set by _start_timer in production)

    for expected in (1.0, 2.0, 3.0):
        clock.advance(1.0)
        dash._refresh_tick(clock())
        assert dash.model.elapsed_seconds == pytest.approx(expected)

    assert len(live.renderables) == 3  # one redraw per tick, liveness during a slow rollout
    # zero rollout progress fabricated (no _on_step ever fired)
    assert dash.model.rollout_steps == 0
    assert dash.model.current_steps == 0


def test_refresh_tick_is_a_noop_before_the_timer_baseline_is_set() -> None:
    """Guard: with no _timer_start_monotonic (timer not started), a stray tick does nothing."""
    dash = TrainingDashboard(scheduled_iters=100, capture=False)
    live = _FakeLive()
    dash._live = live
    dash._timer_start_monotonic = None
    dash._refresh_tick(123.0)
    assert dash.model.elapsed_seconds == 0.0
    assert live.renderables == []


# --------------------------------------------------------------------------- #
# Timer thread lifecycle — starts/stops cleanly (wrapper around the pure seam)
# --------------------------------------------------------------------------- #
def test_start_timer_spawns_a_daemon_thread_and_stop_timer_clears_it() -> None:
    """The background timer is a named daemon thread; _stop_timer joins + clears all timer state
    (using the injected clock for the baseline; the event stops it at once, no real 1 s wait)."""
    clock = _FakeClock(t=500.0)
    dash = TrainingDashboard(scheduled_iters=100, capture=False, now=clock)
    dash._start_timer()
    try:
        assert dash._timer is not None
        assert dash._timer.is_alive()
        assert dash._timer.daemon is True
        assert dash._timer.name == "tui-refresh-timer"
        assert dash._timer_start_monotonic == 500.0
    finally:
        dash._stop_timer()

    assert dash._timer is None
    assert dash._timer_stop is None
    assert dash._timer_start_monotonic is None


# --------------------------------------------------------------------------- #
# Windows-guard — the timer only runs on win32 (AC-9: macOS/Linux untouched)
# --------------------------------------------------------------------------- #
def _stub_rich(monkeypatch):
    """Replace Rich's Console/Live with headless fakes so _start never touches a terminal."""

    class _FakeConsole:
        def __init__(self, *a, **k) -> None:
            pass

    class _FakeLiveCtx:
        def __init__(self, *a, **k) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a) -> bool:
            return False

        def update(self, *a, **k) -> None:
            pass

    monkeypatch.setattr("rich.console.Console", _FakeConsole)
    monkeypatch.setattr("rich.live.Live", _FakeLiveCtx)


def test_timer_starts_on_win32(monkeypatch) -> None:
    """AC-5: on win32 (capable terminal), _start engages the refresh timer."""
    monkeypatch.setattr(sys, "platform", "win32")
    _stub_rich(monkeypatch)
    monkeypatch.setattr("drone_fly.train.tui.capability.assert_tui_capable", lambda *a, **k: None)

    dash = TrainingDashboard(scheduled_iters=100, capture=False)
    started = {"n": 0}
    monkeypatch.setattr(dash, "_start_windows_capture", lambda: None)
    monkeypatch.setattr(dash, "_install_logbridge", lambda: None)
    monkeypatch.setattr(dash, "_build_windows_console", lambda display: object())
    monkeypatch.setattr(dash, "_start_timer", lambda: started.__setitem__("n", started["n"] + 1))

    dash._start()
    try:
        assert started["n"] == 1  # timer engaged on Windows
        assert dash._is_win is True
    finally:
        dash._live = None  # avoid teardown touching the fake
        dash._stop()


def test_timer_does_not_start_off_win32(monkeypatch) -> None:
    """AC-9: on macOS/Linux the timer never starts — behaviour is byte-identical to pre-UC-32."""
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_rich(monkeypatch)

    dash = TrainingDashboard(scheduled_iters=100, capture=False)
    started = {"n": 0}
    monkeypatch.setattr(dash, "_start_timer", lambda: started.__setitem__("n", started["n"] + 1))

    dash._start()
    try:
        assert started["n"] == 0  # NO timer off Windows
        assert dash._is_win is False
        assert dash._logbridge is None  # no Windows log bridge either
    finally:
        dash._live = None
        dash._stop()
