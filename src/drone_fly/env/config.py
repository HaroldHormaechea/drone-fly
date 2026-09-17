"""Tunable course, arena, reward, and episode constants for the racing env.

Everything that shapes difficulty lives here as documented dataclasses with defaults, so
the course geometry (AC-edge-case: geometry is otherwise unspecified) is transparent and
later-tunable rather than magic numbers scattered through the env.

Course layout (a sequence of N free-3D waypoint gates along +x, UC-09)
----------------------------------------------------------------------
* Start at the spawn ``start_position``.
* A sequence of **gates** (``gates``), each a :class:`GateSpec` — a 3D ``center`` with an
  ``aperture`` that doubles as the 3D capture radius. Gates must be passed **in order**;
  the aperture is a sphere (proximity detection), so a waypoint may sit anywhere and the
  drone may have to turn to reach it. Gates are **strictly x-monotonic** ("free 3D" means
  free y/z, monotonically-increasing x) so the finish plane stays reachable.
* A **finish** plane at ``finish_x``. Crossing it counts only after **all** gates are
  passed (ordering is enforced in :mod:`drone_fly.env.geometry`).
* Arena vertical bounds ``floor_z`` / ``ceiling_z``; touching either is a collision.

The default course is a **3-gate** course (UC-09 AC1); :func:`single_gate_course` builds a
one-gate course (N=1 reproduces the pre-UC-09 single-gate behaviour).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class GateSpec:
    """A single waypoint gate: a 3D centre and an aperture (UC-09 AC1).

    The ``aperture`` is both the visual ring radius (in the y–z plane, for the viewer) and
    the **3D capture radius** — a gate is passed when the drone comes within ``aperture`` of
    ``center`` (see :func:`drone_fly.env.geometry.gate_reached`). Smaller apertures force
    more accurate flight; no separate capture tolerance is introduced.
    """

    center: tuple[float, float, float]
    aperture: float = 0.6

    @property
    def position(self) -> np.ndarray:
        """The gate centre as a float64 ``ndarray`` (the target the policy homes on)."""
        return np.asarray(self.center, dtype=np.float64)


#: The default 3-gate course geometry (UC-09 AC1). Verified solvable: every gate sits inside
#: the (floor_z + margin, ceiling_z - margin) corridor, apertures are ≥ the flyable minimum,
#: x is strictly increasing, and the finish is beyond the last gate.
_DEFAULT_GATES: tuple[GateSpec, ...] = (
    GateSpec(center=(2.5, 0.0, 1.0), aperture=0.6),
    GateSpec(center=(4.0, 0.6, 1.3), aperture=0.6),
    GateSpec(center=(5.5, -0.5, 0.9), aperture=0.6),
)


@dataclass(frozen=True)
class CourseConfig:
    """Course geometry and arena bounds: N ordered 3D gates + a finish plane (UC-09 AC1).

    ``gates`` is a tuple of :class:`GateSpec` (default: the 3-gate course). Gates are
    **strictly x-monotonic** (ordered by increasing ``center.x``); the drone passes them in
    order before the finish plane at ``finish_x`` counts.
    """

    start_position: tuple[float, float, float] = (0.0, 0.0, 1.0)
    gates: tuple[GateSpec, ...] = _DEFAULT_GATES
    finish_x: float = 7.0
    floor_z: float = 0.0
    ceiling_z: float = 2.5

    @property
    def start(self) -> np.ndarray:
        return np.asarray(self.start_position, dtype=np.float64)

    @property
    def num_gates(self) -> int:
        """Number of gates in the course (UC-09 AC1)."""
        return len(self.gates)


def single_gate_course(
    *,
    start_position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    gate_center: tuple[float, float, float] = (3.0, 0.0, 1.0),
    gate_aperture: float = 0.6,
    finish_x: float = 6.0,
    floor_z: float = 0.0,
    ceiling_z: float = 2.5,
) -> CourseConfig:
    """Build a valid **one-gate** course (UC-09 AC1; N=1 == pre-UC-09 single-gate race).

    The defaults reproduce the pre-UC-09 scalar-gate geometry (gate at ``x=3``, finish at
    ``x=6``), so ``single_gate_course()`` is the drop-in single-gate course.
    """
    return CourseConfig(
        start_position=start_position,
        gates=(GateSpec(center=gate_center, aperture=gate_aperture),),
        finish_x=finish_x,
        floor_z=floor_z,
        ceiling_z=ceiling_z,
    )


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
    """Per-episode domain-randomization ranges and enable flags (UC-08 AC1/AC5, UC-09 AC5).

    **Off by default** on both axes so ``EnvConfig()`` — and therefore every existing
    caller — samples nothing. Ranges are documented ``(lo, hi)`` tuples and are a tunable
    **difficulty knob** (wider ⇒ harder, narrower ⇒ near-memorization).

    Two independently-toggleable axes:

    * **course** (``enable_course``) — primary anti-memorization axis: samples a fresh
      ``num_gates`` (UC-09 AC5, default range ``[1, 10]``), start, and per-gate 3D centre +
      aperture each ``reset()`` (AC2).
    * **dynamics** (``enable_dynamics``) — secondary robustness / sim-to-sim axis: samples
      mass / drag / thrust / body-rate / control-latency each ``reset()`` (AC5).

    The course sampler walks x forward from the start (incremental deltas), so ordering and
    spacing are structural rather than checked-after-the-fact. Solvability parameters bound
    the sampler so every sampled course is flyable (AC3, UC-09 AC5); degenerate draws are
    rejected-and-resampled up to ``max_resample_attempts`` and then replaced by a
    deterministic, zero-RNG **fallback course** that is solvable-by-construction for every N.
    """

    # -- enable flags (both off by default) ---------------------------------------------
    enable_course: bool = False
    enable_dynamics: bool = False

    # -- gate count (UC-09 AC5) ---------------------------------------------------------
    num_gates_range: tuple[int, int] = (1, 10)  # inclusive integer range for N

    # -- start sampling ranges (lo, hi) -------------------------------------------------
    start_x_range: tuple[float, float] = (-0.5, 0.5)
    start_y_range: tuple[float, float] = (-1.0, 1.0)
    start_z_range: tuple[float, float] = (0.7, 1.5)

    # -- per-gate forward-delta walk along +x (UC-09) -----------------------------------
    # x_0 = start_x + draw(first_gate_gap_range); x_i = x_{i-1} + draw(gate_gap_range);
    # finish_x = x_{N-1} + draw(finish_gap_range). Deltas keep gates x-monotonic and spaced
    # by construction (no absolute gate_x/finish_x ranges any more).
    first_gate_gap_range: tuple[float, float] = (1.5, 2.5)
    gate_gap_range: tuple[float, float] = (1.2, 2.0)
    finish_gap_range: tuple[float, float] = (1.0, 2.0)

    # -- per-gate lateral / vertical / aperture ranges ----------------------------------
    gate_center_y_range: tuple[float, float] = (-1.0, 1.0)
    gate_center_z_range: tuple[float, float] = (0.8, 1.8)
    gate_aperture_range: tuple[float, float] = (0.4, 0.8)
    aperture_min: float = 0.4

    # -- solvability guard --------------------------------------------------------------
    min_start_gate_gap: float = 1.0  # first gate must be at least this far ahead of start
    min_gate_spacing: float = 1.0  # adjacent gates ≥ this apart in 3D (also anti-tunneling)
    min_gate_finish_gap: float = 1.0  # finish_x - last gate_x must be at least this
    z_margin: float = 0.2  # keep waypoints strictly this far off floor/ceiling
    lateral_bound: float = 2.0  # |y| bound for start / gate centres
    max_resample_attempts: int = 50  # hard cap -> deterministic fallback course, never loops

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
    gate_bonus: float = 10.0  # per-gate reward, NORMALISED by num_gates (UC-09 AC4)
    completion_bonus: float = 100.0  # one-off reward on a VALID all-gates-then-finish
    collision_penalty: float = 100.0  # subtracted on floor/ceiling contact (episode ends)


@dataclass(frozen=True)
class EpisodeConfig:
    """Episode timing (AC1 timeout termination).

    The **effective** per-episode step budget scales with the number of gates (UC-09): the
    env truncates at ``max_steps + steps_per_gate * (num_gates - 1)``, so N=1 keeps the
    ``max_steps`` floor (400) exactly while longer courses get proportionally more time.
    """

    dt: float = 0.05  # control timestep (s) -> 20 Hz
    max_steps: int = 400  # base/floor timeout for a 1-gate course (400 * 0.05s = 20s)
    steps_per_gate: int = 200  # extra step budget granted per gate beyond the first (UC-09)


@dataclass(frozen=True)
class EnvConfig:
    """Bundle of the config groups, so an env is configured by one object."""

    course: CourseConfig = field(default_factory=CourseConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
