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

from drone_fly.env.config import CourseConfig, EnvConfig
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
    randomized: bool = False  # UC-08: True when this eval ran over randomized courses/dynamics

    def summary(self) -> str:
        """One-line human summary (used by the CLI)."""
        mct = (
            f"{self.mean_completion_time:.2f}s"
            if self.mean_completion_time is not None
            else "n/a (0 completed)"
        )
        verdict = "MASTERY" if self.meets_mastery else "below bar"
        # The randomized number is the HONEST robustness metric (over N different courses);
        # the fixed-course number is the memorization-inflated one. Flagged so they're never
        # conflated (UC-08 AC8).
        mode = "randomized" if self.randomized else "fixed-course"
        return (
            f"completion_rate={self.completion_rate:.1%} "
            f"({self.completed_count}/{self.n_episodes}) [{verdict} @ {self.threshold:.0%}]; "
            f"mean_completion_time={mct}; mode={mode}; backend={self.backend}"
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
    record: bool = False,
    record_every: int = 1,
    record_dir: str | None = None,
    connectome_path: str | None = None,
    prune: bool = False,
    prune_k: int | None = None,
) -> EvalMetrics:
    """Evaluate ``checkpoint`` over N deterministic episodes; return :class:`EvalMetrics`.

    Recording (UC-05, opt-in, default off → behaviour byte-identical to prior UCs)
    ------------------------------------------------------------------------------
    When ``record`` is set, every ``record_every``-th episode's per-frame neuron
    activations, actions, and drone path are written to ``record_dir`` (default
    ``artifacts/activations/``) as a self-contained playback file (see
    :class:`~drone_fly.record.recorder.ActivationRecorder`). The recorder needs the
    ``neuron_ids`` / ``superclass`` / soma positions the checkpoint does **not** carry, so
    it re-loads the connectome from ``connectome_path`` (applying ``--prune``/``--prune-k``
    exactly as at training time) and a hard alignment assertion
    (``len(neuron_ids) == actor.n_neurons``) guards against a checkpoint/connectome mismatch.
    """
    from stable_baselines3 import PPO

    tcfg = train_config or TrainConfig()
    n = episodes if episodes is not None else tcfg.eval_episodes
    resolved_device = resolve_device(device)
    ecfg = env_config or EnvConfig()

    venv = build_vec_env(
        config=ecfg,
        adapter=adapter,
        n_envs=1,
        seed=seed,
        training=False,
        norm_reward=False,
        vecnormalize_path=vecnormalize_path,
    )
    model = PPO.load(checkpoint, env=venv, device=resolved_device)
    backend = venv.get_attr("backend")[0]

    recorder = _build_recorder(
        model=model,
        record=record,
        record_dir=record_dir,
        backend=backend,
        checkpoint=checkpoint,
        dt=ecfg.episode.dt,
        course=ecfg.course,
        connectome_path=connectome_path,
        prune=prune,
        prune_k=prune_k,
    )
    actor = None
    if recorder is not None:
        from drone_fly.controller.sb3 import actor_from_model

        actor = actor_from_model(model)

    completed_times: list[float] = []
    completed_count = 0

    for ep in range(n):
        capturing = recorder is not None and (ep % max(record_every, 1)) == 0
        if actor is not None:
            actor.sink = recorder.sink if capturing else None
        if capturing:
            recorder.start_episode(ep, seed)

        obs = venv.reset()
        if capturing:
            # Stamp THIS episode's actually-sampled course (UC-08 AC9): with course
            # randomization on, each reset() drew a fresh course off the seeded stream, so
            # read it back from the env rather than recording the static config default.
            recorder.set_course(venv.get_attr("active_course")[0])
        done = False
        info: dict = {}
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            done = bool(dones[0])
            info = infos[0]
            total_reward += float(reward[0])
            steps += 1
            if capturing:
                recorder.capture_frame(action[0], info["position"])
        if info.get("completed"):
            completed_count += 1
            ct = info.get("completion_time")
            if ct is not None:
                completed_times.append(float(ct))
        if capturing:
            recorder.finish_episode(
                completed=bool(info.get("completed", False)),
                completion_time=info.get("completion_time"),
                total_reward=total_reward,
                steps=steps,
            )
    if actor is not None:
        actor.sink = None

    completion_rate = completed_count / n if n else 0.0
    mean_time = float(np.mean(completed_times)) if completed_times else None
    venv.close()

    rcfg = ecfg.randomization
    metrics = EvalMetrics(
        n_episodes=n,
        completed_count=completed_count,
        completion_rate=completion_rate,
        mean_completion_time=mean_time,
        threshold=tcfg.mastery_threshold,
        meets_mastery=completion_rate >= tcfg.mastery_threshold,
        backend=backend,
        randomized=bool(rcfg.enable_course or rcfg.enable_dynamics),
    )
    logger.info("Evaluation: %s", metrics.summary())
    return metrics


def _build_recorder(
    *,
    model,
    record: bool,
    record_dir: str | None,
    backend: str,
    checkpoint: str,
    dt: float,
    course: CourseConfig | None = None,
    connectome_path: str | None,
    prune: bool,
    prune_k: int | None,
):
    """Construct an :class:`ActivationRecorder` for a recording eval run, or ``None``.

    Re-loads the connectome the checkpoint was trained on (the checkpoint does not retain
    ``neuron_ids`` / ``superclass`` / adjacency), applies the same pruning, and enforces the
    hard alignment assertion against the model's actor before any frame is captured.
    """
    if not record:
        return None

    from drone_fly.connectome import load_connectome
    from drone_fly.connectome.prune import DEFAULT_PRUNE_K, prune_to_subcircuit
    from drone_fly.controller.sb3 import actor_from_model
    from drone_fly.record.recorder import DEFAULT_RECORD_DIR, ActivationRecorder

    connectome = load_connectome(connectome_path)
    if prune:
        connectome = prune_to_subcircuit(
            connectome, k=prune_k if prune_k is not None else DEFAULT_PRUNE_K
        )

    actor = actor_from_model(model)
    if connectome.neuron_count != actor.n_neurons:
        raise ValueError(
            f"Recording alignment failure: the re-loaded connectome has "
            f"{connectome.neuron_count} neurons but the checkpoint's actor has "
            f"{actor.n_neurons}. Pass the SAME --connectome/--prune/--prune-k used to train "
            f"this checkpoint so the recorded activations align to neuron_ids."
        )

    logger.info(
        "Recording enabled: %d-neuron connectome '%s' aligned to the checkpoint actor.",
        connectome.neuron_count,
        getattr(connectome, "source", "?"),
    )
    return ActivationRecorder(
        connectome,
        record_dir or DEFAULT_RECORD_DIR,
        backend=str(backend),
        checkpoint=checkpoint,
        dt=dt,
        course=course,
    )
