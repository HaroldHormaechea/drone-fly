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

from drone_fly.env.obstacles import OBSTACLE_VISION_K


@dataclass(frozen=True)
class ObstacleSpec:
    """A cylindrical **pillar** obstacle (UC-15 AC1): floor-anchored, ``center + radius + height``.

    The pillar is anchored on the floor: its base sits at the course ``floor_z`` and its top at
    ``floor_z + height``, with a circular horizontal footprint of ``radius`` about the vertical
    axis at ``center`` (an ``(x, y)`` world position). Collision and vision resolve this against
    the numpy adapter's point position via :mod:`drone_fly.env.obstacles` (hermetic, offline).
    """

    center: tuple[float, float]
    radius: float
    height: float

    @property
    def axis_xy(self) -> np.ndarray:
        """The pillar's vertical-axis ``(x, y)`` position as a float64 ``ndarray``."""
        return np.asarray(self.center, dtype=np.float64)


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
    # Cylindrical pillar obstacles (UC-15 AC1). Appended **last** (after ``ceiling_z``) so every
    # positional constructor call from UC-01..14 is unshifted; the default is **empty**, so a
    # ``CourseConfig()`` is byte-identical to UC-14 (no obstacles, no collision, no vision block).
    obstacles: tuple[ObstacleSpec, ...] = ()

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


#: The fixed default obstacle set (UC-15 AC1): two full-height pillars placed **off** the
#: default 3-gate corridor (which runs start→(2.5,0)→(4.0,0.6)→(5.5,-0.5)→finish). Each pillar
#: sits well clear of every gate centre and of the start→gates→finish polyline, so
#: :func:`default_obstacle_course` is solvable-by-construction (asserted at build time). Values
#: are documented, tunable constants — the "manually placed" set AC1 requires.
_DEFAULT_OBSTACLES: tuple[ObstacleSpec, ...] = (
    ObstacleSpec(center=(3.25, 1.5), radius=0.3, height=2.5),
    ObstacleSpec(center=(4.75, -1.6), radius=0.3, height=2.5),
)


def default_obstacle_course() -> CourseConfig:
    """Build the default 3-gate course **with** the fixed :data:`_DEFAULT_OBSTACLES` (UC-15 AC1).

    The manually-placed obstacle set for AC1: the standard default course plus two off-corridor
    pillars. Asserted solvable-by-construction against the default solvability bounds (no start /
    gate / finish inside a pillar and a collision-free start→gates→finish polyline). The
    :func:`~drone_fly.env.randomization.is_course_solvable` import is deferred to call time to
    avoid an import cycle (randomization imports this module).
    """
    from drone_fly.env.randomization import is_course_solvable

    course = CourseConfig(obstacles=_DEFAULT_OBSTACLES)
    assert is_course_solvable(course, RandomizationConfig()), (
        "default_obstacle_course must be solvable by construction"
    )
    return course


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

    # -- enable flags (all off by default) ----------------------------------------------
    enable_course: bool = False
    enable_dynamics: bool = False
    # Obstacle axis (UC-15): sample pillars each reset() when on. Off by default and its draws
    # are pinned **after** the gate walk, so a disabled obstacle axis never perturbs the UC-08/09
    # course/dynamics RNG stream (byte-identity, AC7-style). Only sampled when a course is drawn.
    enable_obstacles: bool = False

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

    # -- obstacle sampling (UC-15 AC3) --------------------------------------------------
    # Pillars are drawn off the corridor (lateral y-offset from a random gate), reject-resampled
    # against the extended solvability guard. ``obstacle_clearance`` is the horizontal margin the
    # start→gates→finish polyline must keep beyond each pillar's radius (constructively
    # guarantees ≥1 collision-free path). All appended last so field order is UC-14-compatible.
    obstacle_count_range: tuple[int, int] = (1, 3)  # inclusive count of pillars per course
    obstacle_radius_range: tuple[float, float] = (0.3, 0.6)
    obstacle_height_range: tuple[float, float] = (1.0, 2.5)
    obstacle_lateral_offset_range: tuple[float, float] = (1.0, 1.8)  # |y| offset off a gate
    obstacle_clearance: float = 0.3  # polyline must clear each pillar by radius + this


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
    # SEVERE, NON-terminating obstacle-contact penalty (UC-15 AC2/AC9). Documented, tunable.
    # Applied **edge-triggered** (once per distinct contact, not per overlapping step), so a
    # sustained graze cannot stack an unbounded per-frame penalty. Sized well above a single
    # normalised gate_bonus (severe) yet below the terminal collision_penalty, and it never
    # feeds ``terminated`` — the drone may recover aerially and still complete the course.
    obstacle_penalty: float = 50.0


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
class ObstacleVisionConfig:
    """Obstacle-vision observation-block settings (UC-15 AC4/AC5).

    **Off by default** so ``EnvConfig()`` — and therefore every UC-01..14 caller — emits the
    unchanged 12-d observation. When ``enabled`` the env appends a ``4 * k`` obstacle-vision
    block (nearest-``k`` egocentric encoding) to the observation, widening it to
    ``OBS_DIM + 4 * k``. ``k`` defaults to :data:`~drone_fly.env.obstacles.OBSTACLE_VISION_K`
    so it matches the ``obstacle_vision_v2`` schema block's width (``12 == 4 * 3``); the training
    coordinator derives ``k`` name-based from the schema, so the two never drift.
    """

    enabled: bool = False
    k: int = OBSTACLE_VISION_K


@dataclass(frozen=True)
class EnvConfig:
    """Bundle of the config groups, so an env is configured by one object."""

    course: CourseConfig = field(default_factory=CourseConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    # Appended last with an all-off default, so ``EnvConfig()`` stays byte-identical to UC-14.
    obstacle_vision: ObstacleVisionConfig = field(default_factory=ObstacleVisionConfig)
