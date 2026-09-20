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
