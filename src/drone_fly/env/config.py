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
class DynamicsParams:
    """Resolved per-episode drone dynamics handed to the adapter (UC-08 AC5).

    These are **absolute** resolved values (not multipliers): the adapter applies them
    verbatim. The defaults are exactly the ``SimpleDroneAdapter`` module constants, so a
    ``DynamicsParams()`` reconfigure is byte-identical to the fixed UC-01..06 dynamics.
    :func:`drone_fly.env.randomization.sample_dynamics` multiplies a base
    ``DynamicsParams`` by the configured factors to produce a randomized instance.
    """

    mass: float = 1.0  # kg
    drag: float = 0.15  # 1/s linear velocity damping
    max_body_rate: float = 4.0  # rad/s at full stick
    max_thrust: float = 2.0 * 1.0 * 9.81  # N (== 2 * BASE_MASS * GRAVITY == 19.62)
    latency_steps: int = 0  # control-latency delay in steps (0 == no delay)


@dataclass(frozen=True)
class RandomizationConfig:
    """Per-episode domain-randomization ranges and enable flags (UC-08 AC1, AC5).

    **Off by default** on both axes so ``EnvConfig()`` — and therefore every existing
    caller — is byte-identical to UC-01..06 (AC7). Ranges are documented ``(lo, hi)``
    tuples centred on the current :class:`CourseConfig` / dynamics defaults and are a
    tunable **difficulty knob** (AC-ambiguity: wider ⇒ harder, narrower ⇒ near-memorization).

    Two independently-toggleable axes:

    * **course** (``enable_course``) — primary anti-memorization axis: samples a fresh
      start / gate / finish geometry each ``reset()`` (AC2).
    * **dynamics** (``enable_dynamics``) — secondary robustness / sim-to-sim axis: samples
      mass / drag / thrust / body-rate / control-latency each ``reset()`` (AC5).

    Solvability parameters bound the course sampler so every sampled course is flyable
    (AC3); degenerate draws are rejected-and-resampled up to ``max_resample_attempts`` and
    then clamped to the base course (guaranteed solvable, can never loop forever).
    """

    # -- enable flags (both off by default -> byte-identical when unset) -----------------
    enable_course: bool = False
    enable_dynamics: bool = False

    # -- course sampling ranges (lo, hi); defaults centred on CourseConfig ---------------
    start_x_range: tuple[float, float] = (-0.5, 0.5)
    start_y_range: tuple[float, float] = (-1.0, 1.0)
    start_z_range: tuple[float, float] = (0.7, 1.5)
    gate_x_range: tuple[float, float] = (2.0, 4.0)
    gate_center_y_range: tuple[float, float] = (-1.0, 1.0)
    gate_center_z_range: tuple[float, float] = (0.8, 1.8)
    gate_aperture_range: tuple[float, float] = (0.4, 0.8)
    finish_x_range: tuple[float, float] = (5.0, 7.0)
    aperture_min: float = 0.4

    # -- solvability guard --------------------------------------------------------------
    min_start_gate_gap: float = 1.0  # gate_x - start_x must be at least this
    min_gate_finish_gap: float = 1.0  # finish_x - gate_x must be at least this
    z_margin: float = 0.2  # keep waypoints strictly this far off floor/ceiling
    lateral_bound: float = 2.0  # |y| bound for start / gate centre
    max_resample_attempts: int = 50  # hard cap -> clamp to base course, never loop forever

    # -- dynamics sampling factors (lo, hi); multiply the base DynamicsParams -----------
    mass_factor_range: tuple[float, float] = (0.8, 1.2)
    drag_factor_range: tuple[float, float] = (0.8, 1.2)
    thrust_factor_range: tuple[float, float] = (0.8, 1.2)
    rate_factor_range: tuple[float, float] = (0.8, 1.2)
    latency_steps_range: tuple[int, int] = (0, 2)


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
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
