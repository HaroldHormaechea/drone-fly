"""Environment stage: the Gymnasium quadrotor racing environment (UC-03).

Wraps a quadrotor simulator behind the sim-agnostic adapter (pybullet for mastery, a pure
numpy model for hermetic CI) as a Gymnasium env with a start→gate→finish course. Penalizes
floor/ceiling collisions; rewards minimal start→gate→finish time.

Exposes ``RaceEnv``, the ``make_env`` / ``build_vec_env`` factories, the pure geometry +
reward helpers, and the tunable config dataclasses.
"""

from __future__ import annotations

from drone_fly.env.config import (
    CourseConfig,
    DynamicsParams,
    EnvConfig,
    EpisodeConfig,
    RandomizationConfig,
    RewardConfig,
)
from drone_fly.env.geometry import (
    DONE,
    TO_FINISH,
    TO_GATE,
    advance_phase,
    finish_crossed,
    gate_passed,
    target_position,
)
from drone_fly.env.racing_env import RaceEnv, build_vec_env, make_env
from drone_fly.env.randomization import (
    is_course_solvable,
    sample_course,
    sample_dynamics,
)
from drone_fly.env.reward import compute_reward

__all__ = [
    "CourseConfig",
    "DynamicsParams",
    "RandomizationConfig",
    "EnvConfig",
    "EpisodeConfig",
    "RewardConfig",
    "RaceEnv",
    "make_env",
    "build_vec_env",
    "compute_reward",
    "advance_phase",
    "gate_passed",
    "finish_crossed",
    "target_position",
    "sample_course",
    "sample_dynamics",
    "is_course_solvable",
    "TO_GATE",
    "TO_FINISH",
    "DONE",
]
