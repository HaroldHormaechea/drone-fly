"""Training configuration: PPO hyperparameters, checkpointing, and the mastery bar.

All tunables the ``train`` / ``evaluate`` entrypoints need live here as a documented
dataclass with defaults. The mastery threshold (AC10) and evaluation episode count (AC5)
are **configurable constants** here — if PPO plateaus below the bar, the tradeoff is
surfaced by tuning these, never by running unbounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """PPO training + checkpoint configuration.

    Attributes
    ----------
    total_timesteps:
        Environment steps for a full ``train`` run (the owner's dev-time mastery run;
        large). Overridable from the CLI.
    checkpoint_freq:
        SB3 ``CheckpointCallback`` save frequency **in env steps per env** (AC4/AC6). A
        run interrupted between checkpoints loses at most this many steps.
    n_envs:
        Number of vectorised envs.
    smoke_timesteps:
        Tiny step budget for ``smoke-train`` / CI (AC9) — just enough to prove the loop
        wires together and stays finite.
    seed:
        Base RNG seed (AC12).
    vf_arch:
        Value-head MLP widths. The policy head is empty (``pi=[]``) so the 4-channel
        connectome features stay load-bearing (see :mod:`drone_fly.train.loop`); the value
        head gets a small MLP on top of those features. Developer-tunable.
    mastery_threshold:
        Course-completion fraction that counts as mastery (AC10). Default 0.80.
    eval_episodes:
        Number of evaluation episodes (AC5/AC10). Default 20.
    """

    total_timesteps: int = 1_000_000
    checkpoint_freq: int = 25_000
    n_envs: int = 1
    smoke_timesteps: int = 256

    # PPO hyperparameters (SB3 defaults, lightly tuned for a small continuous-control task).
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    # UC-38 (AC8): bumped 0.0 -> 0.01 to sustain exploration during takeoff discovery. With
    # ent_coef=0 the policy's entropy/std collapse before it discovers throttle-up, so a small
    # positive coefficient keeps the action distribution exploring long enough to find takeoff
    # (paired with the UC-38 decoupling that restores the +airborne_bonus survival gradient).
    ent_coef: float = 0.01

    # UC-39 — training-time collision-penalty CURRICULUM (crash-cliff relief, default on). The
    # genuine floor/ceiling/OOB collision penalty is ramped LINEARLY from
    # ``collision_penalty_start``
    # to ``collision_penalty_end`` over the first ``collision_curriculum_warmup_fraction`` of
    # ``total_timesteps``, then held at the end value. Rationale: PPO propagates the −100 crash
    # terminal back onto the "throttle up" actions that begin any takeoff, giving them negative
    # advantage; starting the penalty low (10) while the policy learns to fly removes that barrier,
    # and ramping it back to full strength (100) restores precision so the drone doesn't learn
    # permanently-sloppy floor/ceiling-clipping flight. Applied at rollout time via the env's
    # ``set_collision_penalty`` — the env DEFAULT ``RewardConfig.collision_penalty`` (100) is never
    # changed, so every reward test that asserts 100 is unaffected (minimal test blast radius). The
    # schedule is a function of ``num_timesteps`` only (stateless), so it is resume-correct. End
    # value 100 keeps AC6 (floor-shortcut still loses to completion) and AC8 holds at every value
    # (start 10 ≥ any hover net). Set ``collision_curriculum_enabled=False`` to train at the
    # constant env default (byte-identical to pre-UC-39).
    collision_penalty_start: float = 10.0
    collision_penalty_end: float = 100.0
    collision_curriculum_warmup_fraction: float = 0.5
    collision_curriculum_enabled: bool = True

    seed: int = 0
    vf_arch: list[int] = field(default_factory=lambda: [64, 64])

    # Mastery bar (AC10) + evaluation (AC5) — tunable constants, never a hard-coded gate.
    mastery_threshold: float = 0.80
    eval_episodes: int = 20

    # Artifact locations (relative to CWD). Checkpoints + VecNormalize stats + logs.
    models_dir: str = "artifacts/models"
    logs_dir: str = "artifacts/logs"

    #: Filename of the VecNormalize stats saved alongside each checkpoint / at run end.
    vecnormalize_name: str = "vecnormalize.pkl"
