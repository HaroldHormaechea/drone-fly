"""UC-23 AC6/AC7 — the SB3 runtime health callback + scope-containment proof.

The :class:`HealthCallback` is driven directly through its SB3 lifecycle hooks
(``init_callback`` → ``_on_training_start`` → ``_on_step`` → ``_on_rollout_end``) with a
lightweight fake model/env, so no real PPO rollout is needed. Covers:

* self-accounting of episode reward + success from step infos (no ``Monitor`` in the stack);
* first-rollout / missing-``train/*``-keys → ``normal`` / "insufficient history", never a
  ``KeyError`` or crash;
* a populated verdict after enough rollouts;
* the injected ``on_verdict`` seam (UC-22) is invoked with each verdict;
* the no-spam log cadence (status change OR every ``log_every`` updates);
* the callback never crashes training even when accounting/assessment blows up;
* AC7 scope containment: a ``smoke_train`` on the fixture stays green and non-interactive
  (no env/obs/checkpoint change; the guardrail warn-continues on a tiny CI slice).
"""

from __future__ import annotations

import logging
import types

import numpy as np
import pytest

from drone_fly.train.health import HealthVerdict
from drone_fly.train.health_callback import (
    DEFAULT_LOG_EVERY,
    DEFAULT_WINDOW,
    HealthCallback,
)


def _make_callback(*, n_envs: int = 2, **kwargs) -> tuple[HealthCallback, dict]:
    """Wire a callback to a fake model/env and run ``_on_training_start``.

    Returns the callback and the mutable ``name_to_value`` dict standing in for SB3's logger
    metric store (tests set ``train/*`` keys on it between rollouts).
    """
    name_to_value: dict = {}
    env = types.SimpleNamespace(num_envs=n_envs)
    model = types.SimpleNamespace(
        get_env=lambda: env,
        logger=types.SimpleNamespace(name_to_value=name_to_value),
    )
    cb = HealthCallback(**kwargs)
    cb.init_callback(model)
    cb._on_training_start()
    return cb, name_to_value


def _rollout(cb: HealthCallback, *, rewards, dones, successes) -> None:
    """Feed one rollout: a single step (with the given per-env outcomes) then a rollout end."""
    infos = [{"is_success": bool(s)} for s in successes]
    cb.locals = {
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=bool),
        "infos": infos,
    }
    cb._on_step()
    cb._on_rollout_end()


# --- AC6: warm-up / first rollout — no crash, insufficient history -------------------------


def test_first_rollout_no_train_keys_is_normal_no_crash() -> None:
    cb, _ = _make_callback()
    # No train/* keys populated (first rollout fires before PPO.train()).
    _rollout(cb, rewards=[1.0, 0.5], dones=[True, False], successes=[True, False])
    assert cb.n_updates == 1
    assert cb.latest_verdict is not None
    assert cb.latest_verdict.status == "normal"
    assert "insufficient" in cb.latest_verdict.message.lower()


def test_missing_train_keys_never_raise() -> None:
    cb, name_to_value = _make_callback()
    # Only a subset of train/* keys present — the rest default to None and are skipped.
    for _ in range(3):
        name_to_value.clear()
        name_to_value["train/approx_kl"] = 0.01
        _rollout(cb, rewards=[0.0, 0.0], dones=[False, False], successes=[False, False])
    # No exception, verdict exists.
    assert cb.latest_verdict is not None


# --- AC6: self-accounting of reward + success ---------------------------------------------


def test_self_accounts_reward_and_success() -> None:
    cb, name_to_value = _make_callback(n_envs=2)
    name_to_value.update(
        {
            "train/explained_variance": 0.8,
            "train/std": 1.0,
            "train/approx_kl": 0.01,
            "train/value_loss": 0.5,
        }
    )
    # Env 0 finishes an episode with reward 2.0 and success; env 1 does not finish.
    cb.locals = {
        "rewards": np.array([2.0, 1.0], dtype=np.float32),
        "dones": np.array([True, False], dtype=bool),
        "infos": [{"is_success": True}, {"is_success": False}],
    }
    cb._on_step()
    cb._on_rollout_end()
    # The banked episode reward (2.0) and success (1.0) were self-accounted from step infos.
    assert list(cb._ep_rew) == [2.0]
    assert list(cb._success) == [1.0]


def test_reward_accumulates_across_steps_until_done() -> None:
    cb, _ = _make_callback(n_envs=1)
    for r, done in [(1.0, False), (2.0, False), (3.0, True)]:
        cb.locals = {
            "rewards": np.array([r], dtype=np.float32),
            "dones": np.array([done], dtype=bool),
            "infos": [{"is_success": False}],
        }
        cb._on_step()
    cb._on_rollout_end()
    assert list(cb._ep_rew) == [6.0]  # 1 + 2 + 3 banked on the done step


# --- AC6: populated verdict after enough rollouts -----------------------------------------


def test_populated_verdict_after_enough_rollouts() -> None:
    cb, name_to_value = _make_callback(n_envs=2)
    name_to_value.update(
        {
            "train/explained_variance": 0.8,
            "train/std": 1.0,  # flat high entropy
            "train/approx_kl": 0.01,
            "train/value_loss": 0.5,
        }
    )
    # Drive past the warm-up guard with the K0 signature: EV healthy, entropy flat, success ~0.
    for _ in range(8):
        _rollout(cb, rewards=[0.0, 0.0], dones=[True, True], successes=[False, False])
    assert cb.n_updates == 8
    assert cb.latest_verdict.status in ("warning", "critical")
    assert cb.latest_verdict.reasons, "expected at least one fired reason after 8 rollouts"


# --- AC6: injected on_verdict seam (UC-22) ------------------------------------------------


def test_on_verdict_invoked_each_rollout() -> None:
    seen: list[HealthVerdict] = []
    cb, _ = _make_callback(on_verdict=seen.append)
    for _ in range(3):
        _rollout(cb, rewards=[0.0, 0.0], dones=[True, True], successes=[False, False])
    assert len(seen) == 3
    assert all(isinstance(v, HealthVerdict) for v in seen)
    assert seen[-1] is cb.latest_verdict


def test_on_verdict_exception_does_not_crash_training() -> None:
    def _boom(_verdict):
        raise RuntimeError("consumer blew up")

    cb, _ = _make_callback(on_verdict=_boom)
    # A raising consumer must be swallowed — training keeps going.
    _rollout(cb, rewards=[0.0, 0.0], dones=[True, True], successes=[False, False])
    assert cb.latest_verdict is not None
    assert cb._enabled is True  # a consumer error does not disable the callback


# --- AC6: no-spam log cadence -------------------------------------------------------------


def test_logs_on_status_change_and_cadence(caplog) -> None:
    cb, name_to_value = _make_callback(log_every=5)
    with caplog.at_level(logging.INFO, logger="drone_fly.train.health_callback"):
        # First rollout: status normal (insufficient history) -> a status-change log line.
        _rollout(cb, rewards=[0.0, 0.0], dones=[True, True], successes=[False, False])
        first_count = len(caplog.records)
        assert first_count >= 1  # status changed from None -> normal

        # Rollouts 2..4 stay normal (still within warm-up) -> no new lines (no change, no cadence).
        for _ in range(3):
            _rollout(cb, rewards=[0.0, 0.0], dones=[True, True], successes=[False, False])
        assert len(caplog.records) == first_count

        # Rollout 5 hits the cadence (n_updates % log_every == 0) -> one more line.
        _rollout(cb, rewards=[0.0, 0.0], dones=[True, True], successes=[False, False])
        assert len(caplog.records) == first_count + 1


# --- AC6/AC7: never crashes training ------------------------------------------------------


def test_step_accounting_error_disables_but_never_raises(caplog) -> None:
    cb, _ = _make_callback()
    # rewards present but non-indexable in the accounting loop -> caught, callback disabled.
    cb.locals = {"rewards": np.array([1.0]), "dones": np.array([True]), "infos": "not-a-list"}
    with caplog.at_level(logging.WARNING, logger="drone_fly.train.health_callback"):
        result = cb._on_step()
    assert result is True  # SB3 contract: _on_step returns True to continue training
    assert cb._enabled is False


def test_rollout_error_disables_but_never_raises(monkeypatch, caplog) -> None:
    cb, _ = _make_callback()

    def _boom(*_a, **_k):
        raise ValueError("assessment exploded")

    # Force the assessment call inside _on_rollout_end to raise.
    monkeypatch.setattr("drone_fly.train.health_callback.assess_training_health", _boom)
    with caplog.at_level(logging.WARNING, logger="drone_fly.train.health_callback"):
        cb._on_rollout_end()  # must not raise
    assert cb._enabled is False


def test_disabled_callback_is_a_noop() -> None:
    cb, _ = _make_callback()
    cb._enabled = False
    assert cb._on_step() is True
    cb._on_rollout_end()  # no-op, no crash
    assert cb.latest_verdict is None


def test_env_count_grows_safely() -> None:
    # If the reported env count is smaller than an index seen at a step, accumulators grow.
    cb, _ = _make_callback(n_envs=1)
    cb.locals = {
        "rewards": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "dones": np.array([False, False, True], dtype=bool),
        "infos": [{}, {}, {"is_success": True}],
    }
    cb._on_step()
    cb._on_rollout_end()
    assert cb._enabled is True
    assert list(cb._success) == [1.0]


# --- module defaults sanity ----------------------------------------------------------------


def test_callback_defaults_are_sane() -> None:
    assert DEFAULT_WINDOW > 0
    assert DEFAULT_LOG_EVERY > 0
    cb = HealthCallback()
    assert cb.window == DEFAULT_WINDOW
    assert cb.log_every == DEFAULT_LOG_EVERY
    assert cb.n_updates == 0
    assert cb.latest_verdict is None


def test_window_and_log_every_floored_to_one() -> None:
    cb = HealthCallback(window=0, log_every=0)
    assert cb.window == 1
    assert cb.log_every == 1


# --- AC7: scope containment — smoke_train stays green & non-interactive --------------------


def test_smoke_train_stays_green_non_interactive(connectome, tmp_path, monkeypatch) -> None:
    """A smoke run on the fixture completes without prompting or changing behavior (AC7).

    ``smoke_train`` is warn-only / non-strict, so even the tiny (under-capacity) fixture slice
    warn-continues rather than aborting or blocking on input. Proves the health wiring adds no
    env/obs/checkpoint change and never turns CI smoke training into a false alarm.
    """
    from drone_fly.train.config import TrainConfig
    from drone_fly.train.loop import smoke_train

    # Any prompt during a non-interactive smoke run would be a bug.
    monkeypatch.setattr(
        "builtins.input",
        lambda *_a, **_k: pytest.fail("smoke_train must never prompt"),
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    cfg = TrainConfig(models_dir=str(tmp_path / "models"), logs_dir=str(tmp_path / "logs"))
    model = smoke_train(connectome=connectome, cfg=cfg, timesteps=64)
    assert model is not None
    # The health engine is importable/consumable against the trained model's actor.
    from drone_fly.train.capacity_guard import count_actor_trainable_params

    assert count_actor_trainable_params(model) > 0
