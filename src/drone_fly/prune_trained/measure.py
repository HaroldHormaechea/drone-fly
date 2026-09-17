"""Per-neuron activation/importance measurement over a representative episode set (UC-07 AC1).

Post-training pruning needs a per-neuron *importance* signal measured over a set of episodes.
This module measures it **non-invasively**, reusing the exact activation quantity the UC-05
recorder captures: the actor's post-propagation neuron state, pulled through the actor's opt-in
``sink`` hook. Nothing here alters the policy numerics — the sink receives a detached, cloned
copy and the measurement pass runs under ``torch.no_grad`` / ``predict(deterministic=True)``.

Two entry points mirror the UC-05 raw-vs-checkpoint split:

* :func:`measure_importance` — drives a **raw**
  :class:`~drone_fly.controller.actor.ConnectomeActorNetwork`
  over a single Gymnasium env (the pure-numpy :class:`~drone_fly.adapter.simple.SimpleDroneAdapter`
  path), so the whole measure→prune pipeline is unit-testable on the committed fixture with no
  checkpoint, no pybullet, no network.
* :func:`measure_importance_checkpoint` — drives a **trained** SB3 ``PPO`` checkpoint over a
  ``VecNormalize``-wrapped env (the user-facing path), enforcing the hard
  ``connectome.neuron_count == actor.n_neurons`` alignment assertion before any frame is folded.

Importance metric (AC1, resolves the use-case design fork)
----------------------------------------------------------
The default metric is ``mean_abs`` — the per-neuron mean ``|activation|`` over all measured
frames. A secondary ``active_fraction`` (fraction of frames with ``|activation| > eps``) is
always computed and reported. Both are cheap, intuitive activation-magnitude statistics; the
use case notes a gradient/ablation metric would be more faithful but costlier, so it is left as
a documented future option and pruning aggressiveness is always judged by completion-rate-after
(see :mod:`drone_fly.prune_trained.workflow`), never by the proxy alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Default importance metric (per-neuron mean |activation|).
DEFAULT_METRIC = "mean_abs"

#: Secondary reported metric (fraction of frames the neuron is above ``eps``).
ACTIVE_FRACTION_METRIC = "active_fraction"

#: Supported importance metrics.
METRICS = (DEFAULT_METRIC, ACTIVE_FRACTION_METRIC)

#: Default activity epsilon for ``active_fraction`` (a neuron counts as "active" in a frame
#: when ``|activation| > eps``). The activations are ``tanh``-bounded to ``[-1, 1]``, so a
#: small absolute floor cleanly separates genuinely-firing neurons from numerical dust.
DEFAULT_EPS = 1e-6


@dataclass
class ImportanceResult:
    """Per-neuron importance statistics measured over an episode set.

    Attributes
    ----------
    metric:
        The metric selected for thresholding (one of :data:`METRICS`).
    values:
        Length-``N`` array of the selected metric per neuron (this is what the pruner
        thresholds against).
    mean_abs / active_fraction:
        Both magnitude statistics, always computed for the report regardless of ``metric``.
    frames / n_episodes / completed:
        Total frames folded, episodes run, and episodes that reached a valid completion
        (``completion_before`` in the run report).
    eps:
        The activity epsilon used for ``active_fraction``.
    """

    metric: str
    values: np.ndarray
    mean_abs: np.ndarray
    active_fraction: np.ndarray
    frames: int
    n_episodes: int
    completed: int
    eps: float

    @property
    def completion_rate(self) -> float:
        """Completed / n_episodes over the measured set (the completion-before number)."""
        return self.completed / self.n_episodes if self.n_episodes else 0.0


class ImportanceAccumulator:
    """Pull-based per-neuron activation accumulator mirroring the UC-05 recorder's sink.

    :meth:`sink` stashes the latest post-propagation activation (wired onto ``actor.sink``);
    :meth:`commit` folds exactly one sample per env step into running sums. Storing (not
    appending) in :meth:`sink` makes capture double-safe: if the policy forward runs more than
    once per step (e.g. an SB3 value pass), only the last activation before :meth:`commit` is
    folded, so each step contributes exactly one frame.
    """

    def __init__(self, n_neurons: int, *, eps: float = DEFAULT_EPS) -> None:
        self.n_neurons = int(n_neurons)
        self.eps = float(eps)
        self._pending: np.ndarray | None = None
        self._sum_abs = np.zeros(self.n_neurons, dtype=np.float64)
        self._count_active = np.zeros(self.n_neurons, dtype=np.int64)
        self.frames = 0

    def sink(self, activation: np.ndarray) -> None:
        """Actor hook: stash the latest post-propagation activation (pull-based capture)."""
        self._pending = activation

    def commit(self) -> None:
        """Fold the pending activation into the running sums (one frame per env step)."""
        if self._pending is None:
            raise RuntimeError(
                "No activation captured this step: wire actor.sink = accumulator.sink before "
                "the policy forward pass."
            )
        activation = np.asarray(self._pending, dtype=np.float64)
        if activation.ndim == 2:  # (B, N) — measurement assumes a single env (B == 1)
            activation = activation[0]
        if activation.shape[-1] != self.n_neurons:
            raise ValueError(
                f"Activation width ({activation.shape[-1]}) != connectome neuron count "
                f"({self.n_neurons}); the checkpoint's actor and the measurement connectome "
                f"are misaligned."
            )
        mag = np.abs(activation)
        self._sum_abs += mag
        self._count_active += (mag > self.eps).astype(np.int64)
        self.frames += 1
        self._pending = None

    @property
    def mean_abs(self) -> np.ndarray:
        """Per-neuron mean ``|activation|`` over the folded frames."""
        return self._sum_abs / max(self.frames, 1)

    @property
    def active_fraction(self) -> np.ndarray:
        """Per-neuron fraction of frames with ``|activation| > eps``."""
        return self._count_active / max(self.frames, 1)

    def metric_values(self, metric: str) -> np.ndarray:
        """Return the per-neuron array for ``metric`` (one of :data:`METRICS`)."""
        if metric == DEFAULT_METRIC:
            return self.mean_abs
        if metric == ACTIVE_FRACTION_METRIC:
            return self.active_fraction
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}.")

    def result(self, metric: str, n_episodes: int, completed: int) -> ImportanceResult:
        """Snapshot the accumulator into an :class:`ImportanceResult`."""
        return ImportanceResult(
            metric=metric,
            values=self.metric_values(metric),
            mean_abs=self.mean_abs,
            active_fraction=self.active_fraction,
            frames=self.frames,
            n_episodes=n_episodes,
            completed=completed,
            eps=self.eps,
        )


def _validate_measure_args(n_episodes: int, metric: str) -> None:
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}.")
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}.")


def measure_importance(
    actor,
    env,
    *,
    n_episodes: int,
    seed: int = 0,
    metric: str = DEFAULT_METRIC,
    eps: float = DEFAULT_EPS,
) -> ImportanceResult:
    """Measure per-neuron importance by rolling a **raw actor** over ``env`` (AC1, hermetic).

    Mirrors :func:`drone_fly.record.rollout.record_rollout`: wires ``actor.sink`` to an
    :class:`ImportanceAccumulator`, runs ``n_episodes`` episodes (episode ``e`` seeded
    ``seed + e``), and folds one activation frame per env step. Counts ``info["completed"]``
    for the completion-before number. Raises if no frames are measured (AC10: pruning with no
    representative episodes is an error).

    Parameters
    ----------
    actor:
        A :class:`~drone_fly.controller.actor.ConnectomeActorNetwork`.
    env:
        A single-agent Gymnasium env exposing the racing contract (``reset(seed=…)`` →
        ``(obs, info)``; ``step`` → ``(obs, reward, terminated, truncated, info)``).
    """
    _validate_measure_args(n_episodes, metric)
    acc = ImportanceAccumulator(actor.n_neurons, eps=eps)
    was_training = actor.training
    actor.eval()
    completed = 0
    try:
        actor.sink = acc.sink
        for episode in range(n_episodes):
            obs, _info = env.reset(seed=seed + episode)
            terminated = truncated = False
            info: dict = {}
            while not (terminated or truncated):
                with torch.no_grad():
                    action = actor(torch.as_tensor(np.asarray(obs), dtype=torch.float32))
                action_np = action.detach().cpu().numpy()
                obs, _reward, terminated, truncated, info = env.step(action_np)
                acc.commit()
            if info.get("completed"):
                completed += 1
    finally:
        actor.sink = None
        if was_training:
            actor.train()

    if acc.frames == 0:
        raise ValueError(
            "No activation frames were measured (0 env steps across all episodes); cannot "
            "compute per-neuron importance. Increase --episodes or check the env."
        )
    logger.info(
        "Measured importance over %d episodes / %d frames (metric=%s); completion-before=%.1f%%.",
        n_episodes,
        acc.frames,
        metric,
        100.0 * (completed / n_episodes),
    )
    return acc.result(metric, n_episodes, completed)


def measure_importance_checkpoint(
    model,
    connectome,
    *,
    n_episodes: int,
    seed: int = 0,
    metric: str = DEFAULT_METRIC,
    eps: float = DEFAULT_EPS,
    env_config=None,
    adapter: str = "simple",
    vecnormalize_path: str | None = None,
) -> ImportanceResult:
    """Measure per-neuron importance for a **trained checkpoint** over ``n_episodes`` (AC1).

    The checkpoint carries a :class:`~drone_fly.controller.sb3.ConnectomeFeaturesExtractor`;
    this pulls its actor via :func:`~drone_fly.controller.sb3.actor_from_model`, asserts the
    supplied ``connectome`` aligns with the actor (``neuron_count == n_neurons``), builds a
    frozen ``VecNormalize`` eval env, and folds one activation frame per env step of a
    ``predict(deterministic=True)`` rollout. Counts completions for completion-before.
    """
    _validate_measure_args(n_episodes, metric)
    from drone_fly.controller.sb3 import actor_from_model
    from drone_fly.env.racing_env import build_vec_env

    actor = actor_from_model(model)
    if connectome.neuron_count != actor.n_neurons:
        raise ValueError(
            f"Measurement alignment failure: the connectome has {connectome.neuron_count} "
            f"neurons but the checkpoint's actor has {actor.n_neurons}. Pass the SAME "
            f"--connectome/--prune/--prune-k used to train this checkpoint."
        )

    venv = build_vec_env(
        config=env_config,
        adapter=adapter,
        n_envs=1,
        seed=seed,
        training=False,
        norm_reward=False,
        vecnormalize_path=vecnormalize_path,
    )
    acc = ImportanceAccumulator(actor.n_neurons, eps=eps)
    completed = 0
    prev_sink = actor.sink
    try:
        actor.sink = acc.sink
        for _ep in range(n_episodes):
            obs = venv.reset()
            done = False
            info: dict = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, dones, infos = venv.step(action)
                done = bool(dones[0])
                info = infos[0]
                acc.commit()
            if info.get("completed"):
                completed += 1
    finally:
        actor.sink = prev_sink
        venv.close()

    if acc.frames == 0:
        raise ValueError(
            "No activation frames were measured (0 env steps across all episodes); cannot "
            "compute per-neuron importance. Increase --episodes or check the env."
        )
    logger.info(
        "Measured checkpoint importance over %d episodes / %d frames (metric=%s); "
        "completion-before=%.1f%%.",
        n_episodes,
        acc.frames,
        metric,
        100.0 * (completed / n_episodes),
    )
    return acc.result(metric, n_episodes, completed)
