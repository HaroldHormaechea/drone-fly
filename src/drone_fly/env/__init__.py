"""Environment stage: course/dynamics config dataclasses used by the reservoir pipeline.

The Gymnasium racing environment, geometry and reward helpers (the Stable-Baselines3 training
lineage) were removed in the 2026-10 cleanup; the shipped reservoir models train in gpu_prototype's
hand-rolled batched envs. What remains here is the course/obstacle/dynamics config surface that the
recorder and course generators still read.
"""

from __future__ import annotations

from drone_fly.env.config import (
    CourseConfig,
    DynamicsParams,
    EnvConfig,
    EpisodeConfig,
    GateSpec,
    RandomizationConfig,
    RewardConfig,
    single_gate_course,
)
from drone_fly.env.randomization import (
    is_course_solvable,
    sample_course,
    sample_dynamics,
)

__all__ = [
    "CourseConfig",
    "GateSpec",
    "single_gate_course",
    "DynamicsParams",
    "RandomizationConfig",
    "EnvConfig",
    "EpisodeConfig",
    "RewardConfig",
    "sample_course",
    "sample_dynamics",
    "is_course_solvable",
]
