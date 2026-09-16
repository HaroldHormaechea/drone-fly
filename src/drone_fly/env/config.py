"""Tunable course, arena, reward, and episode constants for the racing env.

Everything that shapes difficulty lives here as documented dataclasses with defaults, so
the course geometry (AC-edge-case: geometry is otherwise unspecified) is transparent and
later-tunable rather than magic numbers scattered through the env. The dynamics are
**fixed** for UC-03 (AC11) — there is no randomization knob here by design.

Course layout (a straight start→gate→finish along +x)
-----------------------------------------------------
* Start at the origin-ish spawn ``start_position``.
* One **gate** at ``gate_x`` whose aperture is a disc of radius ``gate_aperture`` centred
  at ``(gate_x, gate_center_y, gate_center_z)`` in the y–z plane. The gate must be passed
  *through the aperture* to count.
* A **finish** plane at ``finish_x``. Crossing it counts only after the gate is passed
  (gate-before-finish is enforced in :mod:`drone_fly.env.geometry`).
* Arena vertical bounds ``floor_z`` / ``ceiling_z``; touching either is a collision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CourseConfig:
    """Fixed course geometry and arena bounds (AC11: no randomization)."""

    start_position: tuple[float, float, float] = (0.0, 0.0, 1.0)
    gate_x: float = 3.0
    gate_center_y: float = 0.0
    gate_center_z: float = 1.0
    gate_aperture: float = 0.6  # disc radius in the y-z plane
    finish_x: float = 6.0
    floor_z: float = 0.0
    ceiling_z: float = 2.5

    @property
    def start(self) -> np.ndarray:
        return np.asarray(self.start_position, dtype=np.float64)

    @property
    def gate_center(self) -> np.ndarray:
        return np.asarray([self.gate_x, self.gate_center_y, self.gate_center_z], dtype=np.float64)


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping weights (AC3). See :func:`drone_fly.env.reward.compute_reward`.

    Chosen so a *valid* completion always dominates any collision shortcut:
    ``completion_bonus`` is large and positive while ``collision_penalty`` is large and
    negative, and the per-step ``time_penalty`` makes faster completions score higher.
    """

    time_penalty: float = 0.05  # subtracted every step -> faster start→gate→finish wins
    progress_weight: float = 1.0  # reward for closing distance to the current target
    gate_bonus: float = 10.0  # one-off reward when the gate is validly passed
    completion_bonus: float = 100.0  # one-off reward on a VALID gate-then-finish
    collision_penalty: float = 100.0  # subtracted on floor/ceiling contact (episode ends)


@dataclass(frozen=True)
class EpisodeConfig:
    """Episode timing (AC1 timeout termination)."""

    dt: float = 0.05  # control timestep (s) -> 20 Hz
    max_steps: int = 400  # timeout; 400 * 0.05s = 20s of simulated flight


@dataclass(frozen=True)
class EnvConfig:
    """Bundle of the three config groups, so an env is configured by one object."""

    course: CourseConfig = field(default_factory=CourseConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
