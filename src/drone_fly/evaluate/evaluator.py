"""Checkpoint evaluation with pinned metric semantics (AC5, AC10, AC12).

Loads a trained checkpoint + its VecNormalize stats (frozen: ``training=False``,
``norm_reward=False``) and runs N deterministic episodes (``predict(deterministic=True)``),
reporting course-completion and timing with **explicitly documented** semantics so a single
number is never ambiguous:

* ``completion_rate`` — completed / **total** over all N episodes (timeouts and crashes
  count as non-completions).
* ``mean_completion_time`` — mean start→gate→finish time over **completed episodes only**;
  ``None`` when ``completed_count == 0`` (no valid time to average).
* ``completed_count`` / ``n_episodes`` — always disclosed so the rate's denominator and the
  mean's sample size are transparent.
* ``meets_mastery`` — ``completion_rate >= threshold`` (AC10); the bar is a tunable
  constant, never a hard CI gate.

Determinism (AC12): for a fixed seed and a given checkpoint on CPU the evaluation is
reproducible; bit-identical reproducibility *across a resume boundary* is best-effort and
documented (on-policy PPO keeps no replay buffer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from drone_fly.env.config import EnvConfig
from drone_fly.env.racing_env import build_vec_env
from drone_fly.train.config import TrainConfig
from drone_fly.train.device import resolve_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalMetrics:
    """Evaluation result with pinned, unambiguous semantics."""

    n_episodes: int
    completed_count: int
    completion_rate: float  # completed / n_episodes (all N)
    mean_completion_time: float | None  # over completed episodes only; None if none completed
    threshold: float
    meets_mastery: bool
    backend: str

    def summary(self) -> str:
        """One-line human summary (used by the CLI)."""
        mct = (
            f"{self.mean_completion_time:.2f}s"
            if self.mean_completion_time is not None
            else "n/a (0 completed)"
        )
        verdict = "MASTERY" if self.meets_mastery else "below bar"
        return (
            f"completion_rate={self.completion_rate:.1%} "
            f"({self.completed_count}/{self.n_episodes}) [{verdict} @ {self.threshold:.0%}]; "
            f"mean_completion_time={mct}; backend={self.backend}"
        )


def evaluate_checkpoint(
    checkpoint: str,
    *,
    vecnormalize_path: str | None = None,
    episodes: int | None = None,
    seed: int = 0,
    adapter: str = "auto",
    env_config: EnvConfig | None = None,
    train_config: TrainConfig | None = None,
    device: str | None = None,
) -> EvalMetrics:
    """Evaluate ``checkpoint`` over N deterministic episodes; return :class:`EvalMetrics`."""
    from stable_baselines3 import PPO

    tcfg = train_config or TrainConfig()
    n = episodes if episodes is not None else tcfg.eval_episodes
    resolved_device = resolve_device(device)

    venv = build_vec_env(
        config=env_config,
        adapter=adapter,
        n_envs=1,
        seed=seed,
        training=False,
        norm_reward=False,
        vecnormalize_path=vecnormalize_path,
    )
    model = PPO.load(checkpoint, env=venv, device=resolved_device)
    backend = venv.get_attr("backend")[0]

    completed_times: list[float] = []
    completed_count = 0

    for _ep in range(n):
        obs = venv.reset()
        done = False
        info: dict = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, dones, infos = venv.step(action)
            done = bool(dones[0])
            info = infos[0]
        if info.get("completed"):
            completed_count += 1
            ct = info.get("completion_time")
            if ct is not None:
                completed_times.append(float(ct))

    completion_rate = completed_count / n if n else 0.0
    mean_time = float(np.mean(completed_times)) if completed_times else None
    venv.close()

    metrics = EvalMetrics(
        n_episodes=n,
        completed_count=completed_count,
        completion_rate=completion_rate,
        mean_completion_time=mean_time,
        threshold=tcfg.mastery_threshold,
        meets_mastery=completion_rate >= tcfg.mastery_threshold,
        backend=backend,
    )
    logger.info("Evaluation: %s", metrics.summary())
    return metrics
