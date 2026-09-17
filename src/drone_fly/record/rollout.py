"""Recording rollout driver (UC-05, AC1/AC4).

:func:`record_rollout` runs an episode loop over a single :class:`~gymnasium.Env` (the
racing env), driving a raw :class:`~drone_fly.controller.actor.ConnectomeActorNetwork`
directly and pulling one activation frame per env step into an
:class:`~drone_fly.record.recorder.ActivationRecorder`. It records **every Nth episode**
(``record_every``) and emits one self-contained file per recorded episode.

This path deliberately works from a **raw actor** (not a trained PPO checkpoint), so the
whole record→schema pipeline is unit-testable on the committed fixture with the pure-numpy
:class:`~drone_fly.adapter.simple.SimpleDroneAdapter` — no checkpoint, no pybullet, no
network. The evaluator wires the same recorder into the trained-checkpoint eval loop for
the primary user-facing path (see :mod:`drone_fly.evaluate.evaluator`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from drone_fly.controller.actor import ConnectomeActorNetwork
from drone_fly.record.recorder import ActivationRecorder

logger = logging.getLogger(__name__)


def record_rollout(
    actor: ConnectomeActorNetwork,
    env,
    recorder: ActivationRecorder,
    *,
    n_episodes: int,
    seed: int = 0,
    record_every: int = 1,
) -> list[Path]:
    """Roll ``actor`` through ``env`` for ``n_episodes``, recording every ``record_every``-th.

    Parameters
    ----------
    actor:
        A :class:`ConnectomeActorNetwork` whose ``sink`` attribute this function toggles.
    env:
        A single-agent Gymnasium env exposing the racing contract (``reset(seed=…)`` →
        ``(obs, info)``; ``step(action)`` → ``(obs, reward, terminated, truncated, info)``
        with ``info["position"]`` present — see :class:`~drone_fly.env.racing_env.RaceEnv`).
    recorder:
        The :class:`ActivationRecorder` collecting frames and writing files.
    n_episodes:
        Number of episodes to run.
    seed:
        Base seed; episode ``e`` uses ``seed + e`` for reproducible playback.
    record_every:
        Cadence — record episodes ``0, record_every, 2*record_every, …`` (``1`` = every one).

    Returns the paths of the files written (one per recorded episode).
    """
    if record_every < 1:
        raise ValueError(f"record_every must be >= 1, got {record_every}.")
    if len(recorder.neuron_ids) != actor.n_neurons:
        raise ValueError(
            f"Recorder connectome ({len(recorder.neuron_ids)} neurons) is not aligned with "
            f"the actor ({actor.n_neurons} neurons); they must be built from the same graph."
        )

    was_training = actor.training
    actor.eval()
    written: list[Path] = []
    try:
        for episode in range(n_episodes):
            capturing = (episode % record_every) == 0
            episode_seed = seed + episode
            obs, _info = env.reset(seed=episode_seed)
            if capturing:
                recorder.start_episode(episode, episode_seed)
                actor.sink = recorder.sink
            else:
                actor.sink = None

            terminated = truncated = False
            total_reward = 0.0
            steps = 0
            info: dict = {}
            while not (terminated or truncated):
                with torch.no_grad():
                    action = actor(torch.as_tensor(np.asarray(obs), dtype=torch.float32))
                action_np = action.detach().cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action_np)
                steps += 1
                total_reward += float(reward)
                if capturing:
                    recorder.capture_frame(action_np, info["position"])

            if capturing:
                actor.sink = None
                path = recorder.finish_episode(
                    completed=bool(info.get("completed", False)),
                    completion_time=info.get("completion_time"),
                    total_reward=total_reward,
                    steps=steps,
                )
                written.append(path)
    finally:
        actor.sink = None
        if was_training:
            actor.train()

    logger.info(
        "record_rollout: %d/%d episodes recorded to %s.", len(written), n_episodes, recorder.out_dir
    )
    return written
