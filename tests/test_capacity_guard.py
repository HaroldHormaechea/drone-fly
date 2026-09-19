"""UC-23 AC4/AC5 — pre-train capacity guardrail.

Covers:

* ``count_actor_trainable_params`` reads the resolved actor's trainable-param count.
* **Floor calibration on the committed fixture** — prune the real ``mcns_fixture.npz`` at
  k=0 and k=2, build the actor for each, and assert
  ``k0_params < DEFAULT_CAPACITY_FLOOR <= k2_params`` with ``DEFAULT_PRUNE_K == 2``. The
  numbers are *computed*, not hardcoded, so the test catches drift in the pruner/actor.
* ``enforce_capacity`` policy matrix (AC5): strict aborts in both modes; interactive TTY
  confirm/decline (monkeypatched ``isatty`` + ``input``); non-interactive warn-and-continue;
  at/above floor never prompts and never aborts; ``capacity_floor`` override.
* venv is closed on an in-process ``CapacityAbort`` (no leaked envs) — exercised through the
  real ``train`` loop.

The guard only needs ``model.policy.features_extractor.actor`` (a real
:class:`ConnectomeActorNetwork`) plus obs/action spaces, so most cases use a lightweight
fake model wrapping a real actor built from the fixture — no full PPO required.
"""

from __future__ import annotations

import types

import gymnasium as gym
import numpy as np
import pytest

from drone_fly.connectome.prune import DEFAULT_PRUNE_K, prune_to_subcircuit
from drone_fly.controller.actor import ConnectomeActorNetwork
from drone_fly.controller.encoding import ACTION_DIM, OBS_DIM
from drone_fly.train.capacity_guard import (
    CapacityAbort,
    count_actor_trainable_params,
    enforce_capacity,
)
from drone_fly.train.health import DEFAULT_CAPACITY_FLOOR, DEFAULT_THRESHOLDS, HealthThresholds


def _fake_model_for(connectome) -> object:
    """A minimal stand-in exposing ``policy.features_extractor.actor`` + obs/action spaces."""
    actor = ConnectomeActorNetwork(connectome)
    policy = types.SimpleNamespace(features_extractor=types.SimpleNamespace(actor=actor))
    return types.SimpleNamespace(
        policy=policy,
        observation_space=gym.spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32),
    )


@pytest.fixture
def under_capacity_model(connectome):
    """A fake model whose actor (k=0 slice) is under the default floor."""
    return _fake_model_for(prune_to_subcircuit(connectome, k=0))


@pytest.fixture
def sufficient_model(connectome):
    """A fake model whose actor (k=2 default slice) is at/above the default floor."""
    return _fake_model_for(prune_to_subcircuit(connectome, k=DEFAULT_PRUNE_K))


# --- AC4: capacity signal + floor calibration ---------------------------------------------


def test_count_actor_trainable_params(connectome) -> None:
    model = _fake_model_for(prune_to_subcircuit(connectome, k=0))
    actor = model.policy.features_extractor.actor
    expected = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    assert count_actor_trainable_params(model) == expected
    assert count_actor_trainable_params(model) > 0


def test_capacity_floor_calibrated_against_fixture(connectome) -> None:
    """The default floor sits between the k=0 and k=2 fixture actor param counts (AC4).

    Numbers are computed from the committed fixture, so a change to the pruner or actor that
    breaks the ordering fails here instead of silently mis-calibrating the guardrail.
    """
    assert DEFAULT_PRUNE_K == 2

    k0_params = count_actor_trainable_params(_fake_model_for(prune_to_subcircuit(connectome, k=0)))
    k2_params = count_actor_trainable_params(
        _fake_model_for(prune_to_subcircuit(connectome, k=DEFAULT_PRUNE_K))
    )

    # k=0 minimal corridor is undersized; the k=2 default slice clears the floor with margin.
    assert k0_params < DEFAULT_CAPACITY_FLOOR <= k2_params
    # Sanity: the two slices are genuinely different sizes (the pruner actually did something).
    assert k0_params < k2_params


# --- AC4: always logs the verdict on every start ------------------------------------------


def test_enforce_logs_verdict_when_normal(sufficient_model, caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="drone_fly.train.capacity_guard"):
        verdict = enforce_capacity(sufficient_model, strict=True, is_interactive=False)
    assert verdict.status == "normal"
    assert any("capacity check" in r.message.lower() for r in caplog.records)


def test_enforce_logs_verdict_when_under_capacity(under_capacity_model, caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="drone_fly.train.capacity_guard"):
        enforce_capacity(under_capacity_model, strict=False, is_interactive=False)
    assert any("capacity check" in r.message.lower() for r in caplog.records)


# --- AC5: sufficient capacity never prompts, never aborts ----------------------------------


def test_sufficient_capacity_never_prompts_or_aborts(sufficient_model, monkeypatch) -> None:
    def _boom(*_a, **_k):  # a prompt here would be a bug: capacity is fine
        raise AssertionError("enforce_capacity must not prompt when at/above the floor")

    monkeypatch.setattr("builtins.input", _boom)
    # Even with is_interactive=True, a sufficient actor is never questioned.
    verdict = enforce_capacity(sufficient_model, strict=True, is_interactive=True)
    assert verdict.status == "normal"


# --- AC5: strict aborts in BOTH modes -----------------------------------------------------


@pytest.mark.parametrize("interactive", [True, False])
def test_strict_aborts_in_both_modes(under_capacity_model, interactive, monkeypatch) -> None:
    # In strict mode the abort happens before any prompt, regardless of TTY state.
    monkeypatch.setattr(
        "builtins.input",
        lambda *_a, **_k: pytest.fail("strict must abort before prompting"),
    )
    with pytest.raises(CapacityAbort):
        enforce_capacity(under_capacity_model, strict=True, is_interactive=interactive)


# --- AC5: interactive TTY confirm / decline -----------------------------------------------


def test_interactive_confirm_proceeds(under_capacity_model, monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "y")
    verdict = enforce_capacity(under_capacity_model, strict=False, is_interactive=True)
    assert verdict.status in ("warning", "critical")  # proceeded despite under-capacity


def test_interactive_decline_aborts(under_capacity_model, monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "n")
    with pytest.raises(CapacityAbort):
        enforce_capacity(under_capacity_model, strict=False, is_interactive=True)


def test_interactive_empty_answer_defaults_to_decline(under_capacity_model, monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "")
    with pytest.raises(CapacityAbort):
        enforce_capacity(under_capacity_model, strict=False, is_interactive=True)


def test_interactive_eof_defaults_to_decline(under_capacity_model, monkeypatch) -> None:
    def _eof(*_a, **_k):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    with pytest.raises(CapacityAbort):
        enforce_capacity(under_capacity_model, strict=False, is_interactive=True)


# --- AC5: TTY detection via monkeypatched isatty ------------------------------------------


def test_tty_detection_prompts_when_both_streams_are_ttys(
    under_capacity_model, monkeypatch
) -> None:
    # is_interactive=None -> the guard consults sys.stdin/stdout.isatty().
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "n")
    with pytest.raises(CapacityAbort):
        enforce_capacity(under_capacity_model, strict=False, is_interactive=None)


def test_non_tty_warn_and_continues(under_capacity_model, monkeypatch) -> None:
    # No TTY -> cannot block on input; degrade to warn-and-continue (AC5 fallback).
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr(
        "builtins.input",
        lambda *_a, **_k: pytest.fail("non-interactive start must not prompt"),
    )
    verdict = enforce_capacity(under_capacity_model, strict=False, is_interactive=None)
    assert verdict.status in ("warning", "critical")  # continued, no abort


# --- AC5: capacity_floor override ---------------------------------------------------------


def test_capacity_floor_override_lets_small_actor_pass(under_capacity_model) -> None:
    params = count_actor_trainable_params(under_capacity_model)
    lax = HealthThresholds(capacity_floor=params)  # floor <= params -> OK
    verdict = enforce_capacity(under_capacity_model, strict=True, thresholds=lax)
    assert verdict.status == "normal"


def test_capacity_floor_override_can_trip_a_large_actor(sufficient_model) -> None:
    params = count_actor_trainable_params(sufficient_model)
    strict_floor = HealthThresholds(capacity_floor=params + 1)  # floor > params -> under-capacity
    with pytest.raises(CapacityAbort):
        enforce_capacity(sufficient_model, strict=True, thresholds=strict_floor)


def test_default_thresholds_used_when_unspecified(under_capacity_model) -> None:
    # Sanity: the default floor (no thresholds arg) trips the k=0 actor.
    assert DEFAULT_THRESHOLDS.capacity_floor == DEFAULT_CAPACITY_FLOOR
    with pytest.raises(CapacityAbort):
        enforce_capacity(under_capacity_model, strict=True)


# --- AC5: venv closed on abort (real train loop) ------------------------------------------


def test_train_strict_abort_closes_venv(connectome, monkeypatch, tmp_path) -> None:
    """A strict under-capacity abort through ``train`` must close the vec env (no leak)."""
    import drone_fly.train.loop as loop
    from drone_fly.train.config import TrainConfig

    closed = {"value": False}
    real_build = loop.build_vec_env

    def spy_build(*args, **kwargs):
        venv = real_build(*args, **kwargs)
        original_close = venv.close

        def _close(*a, **k):
            closed["value"] = True
            return original_close(*a, **k)

        venv.close = _close
        return venv

    monkeypatch.setattr(loop, "build_vec_env", spy_build)

    cfg = TrainConfig(models_dir=str(tmp_path / "models"), logs_dir=str(tmp_path / "logs"))
    with pytest.raises(CapacityAbort):
        loop.train(
            cfg,
            connectome=connectome,
            prune=True,
            prune_k=0,
            strict_capacity=True,
            adapter="simple",
            device="cpu",
            total_timesteps=64,
            n_envs=1,
        )
    assert closed["value"], "vec env was not closed on CapacityAbort — envs would leak"
