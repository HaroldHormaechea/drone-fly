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
    GateSpec,
    RandomizationConfig,
    RewardConfig,
    single_gate_course,
)
from drone_fly.env.geometry import (
    advance,
    current_target,
    finish_crossed,
    gate_reached,
)
from drone_fly.env.racing_env import (
    DEFAULT_PARALLEL_N_ENVS,
    VEC_ENV_START_METHOD,
    RaceEnv,
    build_vec_env,
    make_env,
    resolve_vec_env,
)
from drone_fly.env.randomization import (
    is_course_solvable,
    sample_course,
    sample_dynamics,
)
from drone_fly.env.reward import compute_reward

__all__ = [
    "CourseConfig",
    "GateSpec",
    "single_gate_course",
    "DynamicsParams",
    "RandomizationConfig",
    "EnvConfig",
    "EpisodeConfig",
    "RewardConfig",
    "RaceEnv",
    "make_env",
    "build_vec_env",
    "resolve_vec_env",
    "DEFAULT_PARALLEL_N_ENVS",
    "VEC_ENV_START_METHOD",
    "compute_reward",
    "advance",
    "current_target",
    "gate_reached",
    "finish_crossed",
    "sample_course",
    "sample_dynamics",
    "is_course_solvable",
]
