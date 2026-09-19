"""UC-22 AC2/AC7 — TTY gating & fallback at the training-loop level.

The dashboard is **default-on only when stdout is an interactive TTY**. This module proves the
gate ``tui_enabled = (tui is not False) and sys.stdout.isatty()`` end-to-end on a real (tiny)
PPO run, without ever spawning a pty:

* **non-TTY** (piped / CI, the normal test environment): the TUI is auto-disabled, no
  ``TrainingDashboard`` is constructed, and SB3's stdout logger is kept — byte-identical to the
  pre-UC-22 run (AC2).
* **``tui=False``** hard-off: no dashboard even if stdout were a TTY.
* **interactive TTY + default**: the dashboard IS constructed and SB3's stdout
  ``HumanOutputFormat`` is dropped so Rich and SB3 don't fight over the screen (AC2/AC7).

The interactive path uses a **fake** dashboard (a plain object with a no-op ``live_session``),
so no real Rich ``Live`` and no terminal are involved — the untested draw boundary is never
touched. ``capacity_floor=1`` keeps the pre-train capacity guard silent.

Hermetic: ``simple`` numpy adapter on the committed fixture, a couple of PPO iterations.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

import pytest

import drone_fly.train.loop as loop
from drone_fly.train.config import TrainConfig


@pytest.fixture
def tiny_cfg(tmp_path) -> TrainConfig:
    return TrainConfig(
        models_dir=str(tmp_path / "models"),
        logs_dir=str(tmp_path / "logs"),
        checkpoint_freq=64,
        n_envs=1,
        n_steps=64,
        batch_size=32,
        seed=0,
    )


class _StdoutProxy:
    """Delegates everything to the real stdout but forces a chosen ``isatty()`` result."""

    def __init__(self, real, isatty: bool) -> None:
        self._real = real
        self._isatty = isatty

    def isatty(self) -> bool:
        return self._isatty

    def __getattr__(self, name):
        return getattr(self._real, name)


def _spy_make_logger(monkeypatch) -> list[bool]:
    """Record the ``include_stdout`` value each time the loop builds its SB3 logger."""
    seen: list[bool] = []
    orig = loop._make_logger

    def spy(logs_dir, include_stdout=True):
        seen.append(include_stdout)
        return orig(logs_dir, include_stdout=include_stdout)

    monkeypatch.setattr(loop, "_make_logger", spy)
    return seen


class _BoomDashboard:
    """A TrainingDashboard stand-in that must never be constructed on the disabled path."""

    instances = 0

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - asserted not-called
        type(self).instances += 1
        raise AssertionError("TrainingDashboard was constructed while the TUI should be OFF")


class _FakeDashboard:
    """A headless dashboard: records construction, provides a no-op live session (no Rich)."""

    instances = 0

    def __init__(self, *, scheduled_iters: int = 0) -> None:
        type(self).instances += 1
        self.scheduled_iters = scheduled_iters
        self.redraws = 0
        self.verdicts: list[object] = []

        class _Model:
            def update(self, **kwargs) -> None:
                pass

            def set_verdict(self, verdict) -> None:
                pass

        self.model = _Model()

    def set_verdict(self, verdict) -> None:
        self.verdicts.append(verdict)

    @contextmanager
    def live_session(self):
        yield self

    def redraw(self) -> None:
        self.redraws += 1


def test_non_tty_disables_tui_and_keeps_stdout_logger(connectome, tiny_cfg, monkeypatch) -> None:
    """AC2: on a non-TTY run the TUI never engages and SB3's stdout logger is preserved."""
    monkeypatch.setattr(sys, "stdout", _StdoutProxy(sys.stdout, isatty=False))
    monkeypatch.setattr("drone_fly.train.tui.dashboard.TrainingDashboard", _BoomDashboard)
    _BoomDashboard.instances = 0
    seen = _spy_make_logger(monkeypatch)

    loop.train(
        tiny_cfg,
        connectome=connectome,
        adapter="simple",
        device="cpu",
        total_timesteps=128,
        tui=True,  # requested, but a non-TTY overrides it
        capacity_floor=1,
    )

    assert _BoomDashboard.instances == 0  # dashboard never built
    assert seen and all(inc is True for inc in seen)  # SB3 stdout logger kept


def test_tui_false_forces_off_even_on_a_tty(connectome, tiny_cfg, monkeypatch) -> None:
    """AC2: ``--no-tui`` (tui=False) is a hard off switch regardless of TTY state."""
    monkeypatch.setattr(sys, "stdout", _StdoutProxy(sys.stdout, isatty=True))
    monkeypatch.setattr("drone_fly.train.tui.dashboard.TrainingDashboard", _BoomDashboard)
    _BoomDashboard.instances = 0
    seen = _spy_make_logger(monkeypatch)

    loop.train(
        tiny_cfg,
        connectome=connectome,
        adapter="simple",
        device="cpu",
        total_timesteps=128,
        tui=False,
        capacity_floor=1,
    )

    assert _BoomDashboard.instances == 0
    assert seen and all(inc is True for inc in seen)  # stdout logger kept when TUI off


def test_interactive_tty_engages_tui_and_drops_stdout_logger(
    connectome, tiny_cfg, monkeypatch
) -> None:
    """AC2/AC7: with an interactive TTY the dashboard is built and SB3's stdout format is
    dropped (CSV/TensorBoard kept) so Rich and SB3 don't fight over the screen."""
    monkeypatch.setattr(sys, "stdout", _StdoutProxy(sys.stdout, isatty=True))
    monkeypatch.setattr("drone_fly.train.tui.dashboard.TrainingDashboard", _FakeDashboard)
    _FakeDashboard.instances = 0
    seen = _spy_make_logger(monkeypatch)

    loop.train(
        tiny_cfg,
        connectome=connectome,
        adapter="simple",
        device="cpu",
        total_timesteps=128,
        tui=None,  # default -> on, because stdout is (forced) a TTY
        capacity_floor=1,
    )

    assert _FakeDashboard.instances == 1  # dashboard engaged
    assert seen and all(inc is False for inc in seen)  # SB3 stdout logger dropped (AC7)


def test_csv_and_tensorboard_survive_when_tui_is_on(connectome, tiny_cfg, monkeypatch) -> None:
    """AC7: dropping the stdout format must NOT drop CSV / TensorBoard — the learning curve
    (and the --no-tui/non-TTY parity) depends on them."""
    import os

    monkeypatch.setattr(sys, "stdout", _StdoutProxy(sys.stdout, isatty=True))
    monkeypatch.setattr("drone_fly.train.tui.dashboard.TrainingDashboard", _FakeDashboard)
    _FakeDashboard.instances = 0

    loop.train(
        tiny_cfg,
        connectome=connectome,
        adapter="simple",
        device="cpu",
        total_timesteps=128,
        tui=None,
        capacity_floor=1,
    )

    logs = os.listdir(tiny_cfg.logs_dir)
    assert "progress.csv" in logs
    assert any(f.startswith("events.out.tfevents") for f in logs), "TensorBoard events missing"
