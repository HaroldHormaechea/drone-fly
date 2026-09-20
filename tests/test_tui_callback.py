"""UC-22 — the SB3→dashboard bridge callback (``drone_fly.train.tui.callback.TuiCallback``).

Headless unit tests with a fake SB3 model (no PPO, no env). They pin the two timing
subtleties the plan calls out:

* **ROLLOUT stats come from the model's episode buffers**, not the logger: a populated
  ``ep_info_buffer`` / ``ep_success_buffer`` feeds ``ep_rew_mean`` / ``ep_len_mean`` /
  ``success_rate``; empty buffers (the first rollout / no completed episodes) yield ``nan`` and
  the data layer treats them as "not observed".
* **TRAIN stats come from ``logger.name_to_value`` and are one-iteration lagged / None on the
  first rollout** — the callback forwards ``None`` for absent keys, never fabricating a ``0``.

Plus the defensive contract: any error inside the callback disables it and never crashes
training.
"""

from __future__ import annotations

import io
import math
import types

import pytest

from drone_fly.train.tui.callback import TuiCallback
from drone_fly.train.tui.metrics import DashboardModel


class _RecordingDashboard:
    """Captures the kwargs fed to the lock-guarded ``update`` / ``tick`` and counts redraws.

    UC-32 rewired the callback to call the dashboard's lock-guarded ``update`` / ``tick`` (each of
    which mutates the model AND redraws atomically under the single lock) rather than poking
    ``model.update`` + ``redraw`` separately — so this harness mirrors that shape.
    """

    def __init__(self) -> None:
        self.updates: list[dict] = []
        self.ticks: list[dict] = []
        self.redraws = 0

        parent = self

        class _Model:
            def update(self, **kwargs) -> None:
                parent.updates.append(kwargs)

            def tick(self, **kwargs) -> None:
                parent.ticks.append(kwargs)

        self.model = _Model()

    def update(self, **kwargs) -> None:
        self.model.update(**kwargs)
        self.redraws += 1

    def tick(self, **kwargs) -> None:
        self.model.tick(**kwargs)
        self.redraws += 1

    def redraw(self) -> None:
        self.redraws += 1


def _fake_sb3_model(*, ep_info=None, ep_success=None, name_to_value=None):
    logger = types.SimpleNamespace(name_to_value=name_to_value or {})
    return types.SimpleNamespace(
        ep_info_buffer=ep_info,
        ep_success_buffer=ep_success,
        logger=logger,
    )


def _fire_rollout(cb: TuiCallback, model) -> None:
    cb.model = model
    cb._on_training_start()
    cb._on_rollout_end()


def test_rollout_stats_come_from_the_episode_buffers() -> None:
    dash = _RecordingDashboard()
    cb = TuiCallback(dash)
    model = _fake_sb3_model(
        ep_info=[{"r": -100.0, "l": 200}, {"r": -300.0, "l": 220}],
        ep_success=[0.0, 1.0],
        name_to_value={
            "train/entropy_loss": -2.5,
            "train/std": 0.5,
            "train/value_loss": 51.2,
            "train/approx_kl": 0.007,
            "train/explained_variance": 0.63,
        },
    )
    _fire_rollout(cb, model)

    assert dash.redraws == 1
    (kw,) = dash.updates
    assert kw["ep_rew_mean"] == -200.0  # mean of the two episode returns
    assert kw["ep_len_mean"] == 210.0
    assert kw["success_rate"] == 0.5
    # train keys forwarded verbatim from name_to_value
    assert kw["entropy_loss"] == -2.5
    assert kw["std"] == 0.5
    assert kw["value_loss"] == 51.2
    assert kw["approx_kl"] == 0.007
    assert kw["explained_variance"] == 0.63
    assert kw["n_updates"] == 1


def test_empty_buffers_yield_nan_and_missing_train_keys_are_none() -> None:
    """First rollout: no completed episodes and no train/* logged yet."""
    dash = _RecordingDashboard()
    cb = TuiCallback(dash)
    model = _fake_sb3_model(ep_info=[], ep_success=[], name_to_value={})
    _fire_rollout(cb, model)

    (kw,) = dash.updates
    assert math.isnan(kw["ep_rew_mean"])
    assert math.isnan(kw["ep_len_mean"])
    assert math.isnan(kw["success_rate"])
    # absent train keys -> None (never a fabricated 0)
    for k in ("entropy_loss", "std", "value_loss", "approx_kl", "explained_variance"):
        assert kw[k] is None


def test_none_buffers_are_tolerated() -> None:
    dash = _RecordingDashboard()
    cb = TuiCallback(dash)
    model = _fake_sb3_model(ep_info=None, ep_success=None, name_to_value=None)
    _fire_rollout(cb, model)  # must not raise
    (kw,) = dash.updates
    assert math.isnan(kw["success_rate"])


def test_n_updates_increments_per_rollout() -> None:
    dash = _RecordingDashboard()
    cb = TuiCallback(dash)
    model = _fake_sb3_model(ep_info=[{"r": 1.0, "l": 1}], ep_success=[1.0])
    cb.model = model
    cb._on_training_start()
    cb._on_rollout_end()
    cb._on_rollout_end()
    cb._on_rollout_end()
    assert [u["n_updates"] for u in dash.updates] == [1, 2, 3]


def test_on_step_returns_true() -> None:
    cb = TuiCallback(_RecordingDashboard())
    assert cb._on_step() is True


def test_callback_disables_itself_on_error_never_crashes() -> None:
    """A redraw failure must disable the callback and NOT propagate into training."""

    class _BoomDashboard:
        def __init__(self) -> None:
            class _Model:
                def update(self, **kwargs) -> None:
                    pass

            self.model = _Model()

        def update(self, **kwargs) -> None:
            # UC-32: the callback calls the lock-guarded dashboard.update (mutation+redraw); a
            # failure here (e.g. a redraw explosion) must disable the callback, never crash.
            raise RuntimeError("render exploded")

        def redraw(self) -> None:  # pragma: no cover - not called under the UC-32 wiring
            raise RuntimeError("render exploded")

    cb = TuiCallback(_BoomDashboard())
    model = _fake_sb3_model(ep_info=[{"r": 1.0, "l": 1}], ep_success=[1.0])
    cb.model = model
    cb._on_training_start()
    cb._on_rollout_end()  # must not raise
    assert cb._enabled is False
    # once disabled it is a no-op
    cb._on_rollout_end()


# --------------------------------------------------------------------------- #
# UC-30 — intra-rollout heartbeat (AC-1..AC-6)
#
# A fully synthetic harness: a fake SB3 model (``n_steps`` / ``get_env().num_envs`` /
# ``num_timesteps``) plus an *injected* monotonic clock so the 0.25 s throttle is driven
# deterministically — NO live Rich ``Live``, NO pty, NO ``time.sleep``. The private
# ``_on_*`` hooks are called directly (as the UC-22 tests above do); ``num_timesteps`` is a
# plain callback attribute (SB3 copies it from the model each real ``on_step``), so the harness
# sets it explicitly to simulate step progression.
# --------------------------------------------------------------------------- #


class _FakeClock:
    """A controllable monotonic source: advance it by hand, never sleep."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeSB3Model:
    """Minimal SB3 stand-in exposing what the heartbeat reads: ``n_steps``, ``get_env()``
    (→ ``num_envs``), ``num_timesteps``, and (for the rollout-end path) the episode buffers +
    logger. ``num_envs=None`` yields an env WITHOUT ``num_envs`` so the fallback to
    ``model.n_envs`` (Dummy/Subproc-safe) can be exercised."""

    def __init__(
        self,
        *,
        n_steps: int = 2048,
        num_envs: int | None = 1,
        num_timesteps: int = 0,
        model_n_envs: int | None = None,
        ep_info=None,
        ep_success=None,
        name_to_value=None,
    ) -> None:
        self.n_steps = n_steps
        self.num_timesteps = num_timesteps
        if num_envs is None:
            self._env = types.SimpleNamespace()  # no num_envs -> fallback to model.n_envs
        else:
            self._env = types.SimpleNamespace(num_envs=num_envs)
        if model_n_envs is not None:
            self.n_envs = model_n_envs
        self.ep_info_buffer = ep_info
        self.ep_success_buffer = ep_success
        self.logger = types.SimpleNamespace(name_to_value=name_to_value or {})

    def get_env(self):
        return self._env


class _RealModelDashboard:
    """A dashboard backed by a REAL :class:`DashboardModel` (so ``tick`` / ``update`` /
    ``raw`` behave exactly as in production) that only counts redraws."""

    def __init__(self, *, scheduled_iters: int = 100, n_envs: int = 1, backend: str = "dummy"):
        self.model = DashboardModel(scheduled_iters=scheduled_iters, n_envs=n_envs, backend=backend)
        self.redraws = 0

    def tick(self, **kwargs) -> None:
        self.model.tick(**kwargs)
        self.redraws += 1

    def update(self, **kwargs) -> None:
        self.model.update(**kwargs)
        self.redraws += 1

    def redraw(self) -> None:
        self.redraws += 1


def _render_to_str(renderable, *, width: int = 120) -> str:
    from rich.console import Console

    buf = io.StringIO()
    Console(file=buf, width=width, legacy_windows=False).print(renderable)
    return buf.getvalue()


def _begin(cb: TuiCallback, model: _FakeSB3Model, clock: _FakeClock) -> None:
    """Wire the model + clock and run training-start → rollout-start (baseline captured)."""
    cb.model = model
    cb.num_timesteps = model.num_timesteps
    cb._on_training_start()
    cb._on_rollout_start()


def _step_at(cb: TuiCallback, timesteps: int) -> bool:
    """Simulate SB3's per-step ``num_timesteps`` bump, then fire the heartbeat hook."""
    cb.num_timesteps = timesteps
    return cb._on_step()


def test_ac1_heartbeat_is_live_during_collection_before_any_rollout_end() -> None:
    """AC-1: before any ``_on_rollout_end``, a driven ``_on_step`` gives a non-zero elapsed +
    a within-rollout step-progress value, with zero completed rollouts and ``raw`` untouched."""
    clock = _FakeClock()
    dash = _RealModelDashboard(n_envs=4, backend="subproc")
    cb = TuiCallback(dash, now=clock)
    model = _FakeSB3Model(n_steps=2048, num_envs=4, num_timesteps=0)
    _begin(cb, model, clock)

    clock.advance(3.0)
    assert _step_at(cb, 256) is True

    assert dash.redraws >= 1  # the heartbeat redrew
    assert dash.model.elapsed_seconds == pytest.approx(3.0)  # live clock ticked, non-zero
    assert dash.model.rollout_steps == 256  # within-rollout progress > 0
    assert dash.model.rollout_target == 2048 * 4
    # zero rollout-ends -> the values-panel snapshot is still all-None placeholders
    assert all(v is None for v in dash.model.raw.values())
    # and no trend history was fabricated
    assert all(len(h) == 0 for h in dash.model.history.values())


def test_ac2_heartbeat_is_throttled_via_the_injected_clock() -> None:
    """AC-2: many rapid ``_on_step`` calls redraw at most ~4×/sec (the 0.25 s gate), asserted
    through the fake clock — NOT real sleeps. 50 steps over 5 s must yield far fewer redraws."""
    clock = _FakeClock()
    dash = _RealModelDashboard()
    cb = TuiCallback(dash, now=clock)
    model = _FakeSB3Model(n_steps=2048, num_envs=1)
    _begin(cb, model, clock)

    ts = 0
    for _ in range(50):  # advance 0.1 s each -> 5 s of simulated collection
        clock.advance(0.1)
        ts += 1
        _step_at(cb, ts)

    # theoretical ceiling over 5 s at a 0.25 s gate: 5.0 / 0.25 + 1 = 21
    assert dash.redraws <= 21
    assert dash.redraws < 50  # decisively throttled below the step count
    assert dash.redraws >= 10  # still visibly live during the slow rollout


def test_ac3_step_progress_target_and_reset_at_each_rollout_start() -> None:
    """AC-3: current = num_timesteps - rollout_start; target = n_steps × n_envs; the counter
    resets on a second ``_on_rollout_start``."""
    clock = _FakeClock()
    dash = _RealModelDashboard()
    cb = TuiCallback(dash, now=clock)
    model = _FakeSB3Model(n_steps=2048, num_envs=4, num_timesteps=1000)
    _begin(cb, model, clock)  # baseline = 1000

    clock.advance(1.0)
    _step_at(cb, 1000 + 512)
    steps, target, frac = dash.model.rollout_progress()
    assert steps == 512
    assert target == 2048 * 4
    assert frac == pytest.approx(512 / (2048 * 4))

    # a second rollout re-baselines: progress restarts from the new num_timesteps
    cb.num_timesteps = 1000 + 2048 * 4
    cb._on_rollout_start()
    clock.advance(1.0)
    _step_at(cb, 1000 + 2048 * 4 + 128)
    steps2, _, _ = dash.model.rollout_progress()
    assert steps2 == 128  # reset worked (not 512 + …)


def test_ac3_n_envs_falls_back_to_model_when_env_lacks_num_envs() -> None:
    """AC-3: ``n_envs`` reads ``training_env.num_envs`` first, else falls back to ``model.n_envs``
    (Dummy/Subproc-safe) so the target is correct on either vec-env backend."""
    clock = _FakeClock()
    dash = _RealModelDashboard()
    cb = TuiCallback(dash, now=clock)
    model = _FakeSB3Model(n_steps=100, num_envs=None, model_n_envs=6)  # env has no num_envs
    _begin(cb, model, clock)

    clock.advance(0.5)
    _step_at(cb, 50)
    _, target, _ = dash.model.rollout_progress()
    assert target == 100 * 6  # fell back to model.n_envs


def test_ac4_heartbeat_never_blanks_the_rollout_end_snapshot() -> None:
    """AC-4 (the KEY regression): after a full ``_on_rollout_end`` populates ``raw``, several
    heartbeats in the next collection must leave ``raw`` byte-for-byte unchanged — every entry
    still its last-rollout value, never blanked to a placeholder."""
    clock = _FakeClock()
    dash = _RealModelDashboard(n_envs=2)
    cb = TuiCallback(dash, now=clock)
    model = _FakeSB3Model(
        n_steps=2048,
        num_envs=2,
        num_timesteps=8192,
        ep_info=[{"r": -200.0, "l": 210}],
        ep_success=[1.0],
        name_to_value={
            "train/entropy_loss": -2.5,
            "train/std": 0.5,
            "train/value_loss": 51.2,
            "train/approx_kl": 0.007,
            "train/explained_variance": 0.63,
        },
    )
    _begin(cb, model, clock)

    # complete a rollout: the full snapshot lands in raw
    cb._on_rollout_end()
    snapshot = dict(dash.model.raw)
    assert snapshot["ep_rew"] == -200.0  # sanity: it really populated
    assert snapshot["success"] == 1.0
    assert snapshot["value_loss"] == 51.2
    assert all(v is not None for v in snapshot.values())

    # next collection: fire several heartbeats with an advancing clock
    cb._on_rollout_start()
    for i in range(5):
        clock.advance(0.5)
        _step_at(cb, 8192 + (i + 1) * 256)

    assert dash.model.raw == snapshot  # UNCHANGED — tick() touched neither raw nor history
    assert all(v is not None for v in dash.model.raw.values())
    assert dash.model.elapsed_seconds > 0  # but the live clock did advance
    assert dash.model.rollout_steps == 5 * 256  # and the collecting counter moved


def test_ac5_heartbeat_disables_itself_on_error_and_never_raises() -> None:
    """AC-5: an exception inside the heartbeat disables it (``_enabled=False``) and never
    propagates into training; once disabled it is a fast no-op."""

    class _BoomModel:
        def tick(self, **kwargs) -> None:
            raise RuntimeError("heartbeat boom")

    class _BoomDashboard:
        def __init__(self) -> None:
            self.model = _BoomModel()
            self.redraws = 0

        def tick(self, **kwargs) -> None:
            # UC-32: lock-guarded tick mutates the model then redraws; the mutation raises here.
            self.model.tick(**kwargs)
            self.redraws += 1  # pragma: no cover - never reached (tick raises first)

        def redraw(self) -> None:  # pragma: no cover - never reached (tick raises first)
            self.redraws += 1

    clock = _FakeClock()
    dash = _BoomDashboard()
    cb = TuiCallback(dash, now=clock)
    model = _FakeSB3Model(n_steps=10, num_envs=1)
    cb.model = model
    cb.num_timesteps = 0
    cb._on_training_start()
    cb._on_rollout_start()

    clock.advance(1.0)
    assert cb._on_step() is True  # must not raise
    assert cb._enabled is False
    assert dash.redraws == 0  # blew up before the redraw

    # once disabled, further steps short-circuit (no second tick attempt)
    clock.advance(1.0)
    assert _step_at(cb, 5) is True


def test_ac6_heartbeat_redraw_refreshes_the_live_log_pane() -> None:
    """AC-6: the heartbeat's ``dashboard.redraw()`` rebuilds the layout from the CURRENT log
    scrollback, so lines captured mid-collection appear in the log pane before any rollout end.
    Uses the real :class:`TrainingDashboard` with a fake ``Live`` + capture (no pty, no Live)."""
    from drone_fly.train.tui.dashboard import TrainingDashboard

    class _FakeLive:
        def __init__(self) -> None:
            self.renderables: list = []

        def update(self, renderable) -> None:
            self.renderables.append(renderable)

    class _GrowingCapture:
        def __init__(self) -> None:
            self._lines: list[str] = []

        def add(self, line: str) -> None:
            self._lines.append(line)

        def lines(self) -> list[str]:
            return list(self._lines)

    clock = _FakeClock()
    dash = TrainingDashboard(scheduled_iters=100, capture=False, n_envs=1)
    live = _FakeLive()
    cap = _GrowingCapture()
    dash._live = live  # inject a headless Live sink
    dash._capture = cap  # inject a growing log source

    cb = TuiCallback(dash, now=clock)
    model = _FakeSB3Model(n_steps=64, num_envs=1)
    cb.model = model
    cb.num_timesteps = 0
    cb._on_training_start()
    cb._on_rollout_start()

    cap.add("pybullet build")
    clock.advance(0.5)
    _step_at(cb, 16)

    # a NEW log line arrives during collection (before any rollout end)
    cap.add("Version = 3.2.5")
    clock.advance(0.5)
    _step_at(cb, 32)

    out = _render_to_str(live.renderables[-1])
    assert "Version = 3.2.5" in out  # the freshly captured line is live in the pane
    assert "pybullet build" in out


# --------------------------------------------------------------------------- #
# UC-32 — the callback drives the dashboard's lock-guarded tick/update and feeds
# the cumulative num_timesteps into the model's steps line (Addendum).
# --------------------------------------------------------------------------- #


def test_uc32_heartbeat_routes_through_the_lock_guarded_tick() -> None:
    """UC-32: a heartbeat calls ``dashboard.tick`` (mutation+redraw atomic under the lock) rather
    than poking ``model.tick`` + ``redraw`` separately — so the callback can never mutate the model
    outside the lock the Windows refresh timer also contends for."""
    clock = _FakeClock()
    dash = _RecordingDashboard()
    cb = TuiCallback(dash, now=clock)
    model = _FakeSB3Model(n_steps=2048, num_envs=2, num_timesteps=4096)
    _begin(cb, model, clock)

    clock.advance(1.0)
    _step_at(cb, 4096 + 512)

    assert len(dash.ticks) == 1  # went through the lock-guarded tick
    assert dash.updates == []  # a heartbeat is not a rollout-end update


def test_uc32_current_steps_is_fed_on_both_tick_and_update() -> None:
    """Addendum: the model's cumulative step count (for the TIME panel's ``steps`` line) is fed
    from SB3 ``num_timesteps`` by BOTH the heartbeat tick and the rollout-end update."""
    clock = _FakeClock()
    dash = _RealModelDashboard(n_envs=2)
    cb = TuiCallback(dash, now=clock)
    model = _FakeSB3Model(
        n_steps=2048,
        num_envs=2,
        num_timesteps=0,
        ep_info=[{"r": -200.0, "l": 210}],
        ep_success=[1.0],
    )
    _begin(cb, model, clock)

    clock.advance(1.0)
    _step_at(cb, 26_624)  # a heartbeat mid-collection
    assert dash.model.current_steps == 26_624  # tick fed num_timesteps

    cb.num_timesteps = 40_960
    cb._on_rollout_end()  # a rollout-end snapshot
    assert dash.model.current_steps == 40_960  # update fed num_timesteps too
