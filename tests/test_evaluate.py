"""AC5/AC10/AC12 — checkpoint evaluation with pinned metric semantics.

Covers the ``evaluate`` entrypoint's contract:

* **AC5** — over N episodes, report course-completion rate and mean start→gate→finish time.
* **AC10** — ``meets_mastery = completion_rate >= threshold``; the threshold + episode
  count are tunable constants (``TrainConfig``), never a hard gate.
* **AC12** — for a fixed seed and a given checkpoint the evaluation is reproducible.

The pinned metric semantics are also unit-tested directly on :class:`EvalMetrics` so the
denominator/sample-size disclosure rules hold regardless of what the policy does:

* ``completion_rate`` = completed / **all** N (timeouts/crashes are non-completions),
* ``mean_completion_time`` averaged over **completed episodes only**, ``None`` when zero
  completed,
* ``completed_count`` / ``n_episodes`` always disclosed.
"""

from __future__ import annotations

import os

import pytest

from drone_fly.evaluate.evaluator import EvalMetrics, evaluate_checkpoint
from drone_fly.train.config import TrainConfig
from drone_fly.train.loop import CHECKPOINT_PREFIX, smoke_train


@pytest.fixture
def trained_checkpoint(connectome, tmp_path):
    """Run a tiny hermetic smoke-train and return (checkpoint_zip, vecnormalize_pkl)."""
    cfg = TrainConfig(
        models_dir=str(tmp_path / "models"),
        logs_dir=str(tmp_path / "logs"),
        checkpoint_freq=64,
        n_envs=1,
        n_steps=64,
        batch_size=32,
        seed=0,
    )
    smoke_train(connectome=connectome, cfg=cfg, timesteps=128)
    final = os.path.join(cfg.models_dir, f"{CHECKPOINT_PREFIX}_final.zip")
    stats = os.path.join(cfg.models_dir, cfg.vecnormalize_name)
    assert os.path.isfile(final) and os.path.isfile(stats)
    return final, stats


# --- EvalMetrics semantics (pure, no training) --------------------------------------
def test_metrics_completion_rate_over_all_episodes() -> None:
    m = EvalMetrics(
        n_episodes=20,
        completed_count=5,
        completion_rate=5 / 20,
        mean_completion_time=3.0,
        threshold=0.8,
        meets_mastery=False,
        backend="simple",
    )
    assert m.completion_rate == 0.25
    assert m.completed_count == 5
    assert m.n_episodes == 20


def test_metrics_mean_time_none_when_nothing_completed() -> None:
    m = EvalMetrics(
        n_episodes=20,
        completed_count=0,
        completion_rate=0.0,
        mean_completion_time=None,
        threshold=0.8,
        meets_mastery=False,
        backend="simple",
    )
    assert m.mean_completion_time is None
    assert "n/a" in m.summary()


def test_metrics_meets_mastery_threshold_boundary() -> None:
    at_bar = EvalMetrics(20, 16, 16 / 20, 2.0, 0.8, 16 / 20 >= 0.8, "simple")
    below = EvalMetrics(20, 15, 15 / 20, 2.0, 0.8, 15 / 20 >= 0.8, "simple")
    assert at_bar.meets_mastery is True  # 0.80 >= 0.80
    assert below.meets_mastery is False  # 0.75 < 0.80


# --- evaluate_checkpoint end-to-end (hermetic) --------------------------------------
def test_evaluate_reports_all_disclosure_fields(trained_checkpoint) -> None:
    ckpt, stats = trained_checkpoint
    m = evaluate_checkpoint(
        ckpt, vecnormalize_path=stats, episodes=4, seed=0, adapter="simple", device="cpu"
    )
    assert isinstance(m, EvalMetrics)
    assert m.n_episodes == 4
    assert 0 <= m.completed_count <= 4
    assert m.completion_rate == m.completed_count / 4
    assert m.threshold == TrainConfig().mastery_threshold  # 0.80
    assert m.meets_mastery == (m.completion_rate >= m.threshold)
    assert m.backend == "simple"
    # mean time is disclosed consistently with completed_count.
    if m.completed_count == 0:
        assert m.mean_completion_time is None
    else:
        assert m.mean_completion_time is not None


def test_evaluate_episode_count_is_configurable(trained_checkpoint) -> None:
    ckpt, stats = trained_checkpoint
    m = evaluate_checkpoint(
        ckpt, vecnormalize_path=stats, episodes=2, seed=0, adapter="simple", device="cpu"
    )
    assert m.n_episodes == 2


def test_evaluate_deterministic_for_fixed_seed(trained_checkpoint) -> None:
    ckpt, stats = trained_checkpoint
    a = evaluate_checkpoint(
        ckpt, vecnormalize_path=stats, episodes=3, seed=0, adapter="simple", device="cpu"
    )
    b = evaluate_checkpoint(
        ckpt, vecnormalize_path=stats, episodes=3, seed=0, adapter="simple", device="cpu"
    )
    assert a.completion_rate == b.completion_rate
    assert a.completed_count == b.completed_count
    assert a.mean_completion_time == b.mean_completion_time
