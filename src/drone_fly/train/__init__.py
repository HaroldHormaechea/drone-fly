"""Training stage: the reinforcement-learning training loop (UC-03).

Wires the connectome-seeded controller policy (UC-02) to the racing environment and trains
it with Stable-Baselines3 PPO. Writes periodic checkpoints (+ VecNormalize stats) to
``artifacts/models/`` and a TensorBoard + CSV learning curve to ``artifacts/logs/``.
Supports checkpoint/resume and an Apple-Silicon-aware device auto-detect.

Exports the training entrypoints, the config dataclass, and the device resolver.
"""

from __future__ import annotations

from drone_fly.train.config import TrainConfig
from drone_fly.train.device import APPLE_SILICON_NOTE, CUDA_OOM_HINT, resolve_device
from drone_fly.train.loop import (
    build_policy_kwargs,
    find_latest_checkpoint,
    smoke_train,
    train,
)

__all__ = [
    "TrainConfig",
    "resolve_device",
    "APPLE_SILICON_NOTE",
    "CUDA_OOM_HINT",
    "train",
    "smoke_train",
    "build_policy_kwargs",
    "find_latest_checkpoint",
]
