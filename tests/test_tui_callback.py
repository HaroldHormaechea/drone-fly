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

import math
import types

from drone_fly.train.tui.callback import TuiCallback


class _RecordingDashboard:
    """Captures the kwargs fed to ``model.update`` and counts redraws."""

    def __init__(self) -> None:
        self.updates: list[dict] = []
        self.redraws = 0

        parent = self

        class _Model:
            def update(self, **kwargs) -> None:
                parent.updates.append(kwargs)

        self.model = _Model()

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

        def redraw(self) -> None:
            raise RuntimeError("render exploded")

    cb = TuiCallback(_BoomDashboard())
    model = _fake_sb3_model(ep_info=[{"r": 1.0, "l": 1}], ep_success=[1.0])
    cb.model = model
    cb._on_training_start()
    cb._on_rollout_end()  # must not raise
    assert cb._enabled is False
    # once disabled it is a no-op
    cb._on_rollout_end()
