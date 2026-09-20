"""AC1/AC11/AC12 — RaceEnv contract: spaces, termination modes, fixed dynamics, seeding.

All tests use ``adapter="simple"`` so they are hermetic (no pybullet). Completion
termination is exercised with a small **scripted adapter** that walks the drone through a
hand-built start→gate→finish trajectory, so we can assert the "terminate on course
completion" branch deterministically without relying on an untrained policy to actually
fly the course.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drone_fly.adapter.base import DroneState
from drone_fly.adapter.simple import SimpleDroneAdapter
from drone_fly.controller.encoding import ACTION_DIM, OBS_DIM
from drone_fly.env import EnvConfig, make_env
from drone_fly.env.config import (
    CourseConfig,
    DamageConfig,
    DockConfig,
    EarlyTerminationConfig,
    EpisodeConfig,
    GateSpec,
    ObstacleSpec,
    ObstacleVisionConfig,
    RandomizationConfig,
    default_obstacle_course,
    default_pad_course,
    default_repair_course,
    single_gate_course,
    single_pad_course,
    single_repair_pad_course,
)
from drone_fly.env.randomization import is_course_solvable

HOVER = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)

_BASELINE_ROLLOUT = Path(__file__).parent / "data" / "uc08_baseline_rollout.npz"


class _ScriptedAdapter:
    """A stand-in adapter that replays a fixed list of positions, for env-level tests."""

    backend = "scripted"

    def __init__(self, positions, *, collide_at=None):
        self._positions = [np.asarray(p, dtype=np.float64) for p in positions]
        self._collide_at = collide_at
        self._i = 0

    def _state(self, idx) -> DroneState:
        return DroneState(
            position=self._positions[min(idx, len(self._positions) - 1)].copy(),
            velocity=np.zeros(3),
            attitude=np.zeros(3),
            angular_velocity=np.zeros(3),
            collided=(idx == self._collide_at),
        )

    def reset(self, seed=None) -> DroneState:
        self._i = 0
        return self._state(0)

    def step(self, action) -> DroneState:
        self._i += 1
        return self._state(self._i)

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def _env_with(adapter, config=None):
    env = make_env(config, adapter="simple")
    env.adapter = adapter
    env.backend = adapter.backend
    return env


# --- AC1: observation / action spaces -----------------------------------------------
def test_observation_space_is_box_12_float32() -> None:
    env = make_env(adapter="simple")
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.observation_space.shape == (12,)
    assert env.observation_space.dtype == np.float32


def test_action_space_is_ctbr_box() -> None:
    env = make_env(adapter="simple")
    assert env.action_space.shape == (ACTION_DIM,)
    np.testing.assert_array_equal(env.action_space.low, np.array([0.0, -1.0, -1.0, -1.0]))
    np.testing.assert_array_equal(env.action_space.high, np.array([1.0, 1.0, 1.0, 1.0]))
    assert env.action_space.dtype == np.float32


def test_reset_returns_valid_observation() -> None:
    env = make_env(adapter="simple")
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert info["backend"] == "simple"


def test_step_returns_five_tuple_with_finite_obs() -> None:
    env = make_env(adapter="simple")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(HOVER)
    assert obs.shape == (OBS_DIM,)
    assert np.isfinite(obs).all()
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# --- AC1: termination on completion / collision / timeout ---------------------------
# Default 3-gate course: g0 (2.5,0,1.0), g1 (4.0,0.6,1.3), g2 (5.5,-0.5,0.9); finish x=7.
_THREE_GATE_PATH = [
    (0.0, 0.0, 1.0),
    (2.5, 0.0, 1.0),  # g0
    (4.0, 0.6, 1.3),  # g1
    (5.5, -0.5, 0.9),  # g2
    (6.9, -0.5, 0.9),
    (7.1, -0.5, 0.9),  # finish
]


def test_terminates_on_course_completion() -> None:
    # Scripted walk through all three default gates in order, then the finish plane.
    env = _env_with(_ScriptedAdapter(_THREE_GATE_PATH))
    env.reset()
    terminated = False
    info = {}
    for _ in range(len(_THREE_GATE_PATH)):
        _obs, _r, terminated, truncated, info = env.step(HOVER)
        if terminated or truncated:
            break
    assert terminated is True
    assert info["completed"] is True
    assert info["is_success"] is True
    assert info["completion_time"] is not None


def test_terminates_on_completion_advances_target_gate_in_order() -> None:
    """UC-09 AC2: ``info['target_gate']`` steps 0→1→2→3 as gates are passed in order."""
    env = _env_with(_ScriptedAdapter(_THREE_GATE_PATH))
    env.reset()
    seen = []
    for _ in range(len(_THREE_GATE_PATH)):
        _obs, _r, terminated, truncated, info = env.step(HOVER)
        seen.append(info["target_gate"])
        if terminated or truncated:
            break
    # After g0,g1,g2 the target index climbs to 3 (== N, now chasing the finish).
    assert seen == [1, 2, 3, 3, 3]


def test_out_of_order_gate_is_rejected_at_env_level() -> None:
    """UC-09 AC2: flying through gate 1's vicinity before gate 0 does not advance.

    The path hovers at gate 1's centre (never near gate 0), then crosses the finish plane.
    Because only the current gate (index 0) is ever tested, nothing advances and the course
    never completes — you cannot skip a gate.
    """
    path = [
        (3.8, 0.6, 1.3),  # start already near g1 but g0 (2.5,0,1) is untouched
        (4.0, 0.6, 1.3),  # through g1's centre — but current gate is still g0
        (6.9, 0.6, 1.3),
        (7.1, 0.6, 1.3),  # crosses the finish plane
    ]
    env = _env_with(_ScriptedAdapter(path))
    env.reset()
    completed_any = False
    target_gates = []
    for _ in range(len(path)):
        _obs, _r, terminated, truncated, info = env.step(HOVER)
        completed_any = completed_any or info["completed"]
        target_gates.append(info["target_gate"])
        if terminated or truncated:
            break
    assert completed_any is False
    assert set(target_gates) == {0}  # never advanced past gate 0


def test_finish_before_gate_does_not_complete() -> None:
    # Fly the whole trajectory off-axis (y=2.0, well outside the 0.6 aperture) so no gate is
    # ever validly passed, yet the drone still forward-crosses the finish plane. Crossing
    # the finish before all gates are passed must not complete-as-success.
    path = [(0, 2, 1), (3.1, 2, 1), (6.1, 2, 1), (7.1, 2, 1)]
    env = _env_with(_ScriptedAdapter(path))
    env.reset()
    completed_any = False
    for _ in range(len(path)):
        _obs, _r, terminated, truncated, info = env.step(HOVER)
        completed_any = completed_any or info["completed"]
        if terminated or truncated:
            break
    assert completed_any is False


def test_single_gate_course_completes_like_a_one_gate_race() -> None:
    """UC-09 AC1: an N=1 course is a valid one-gate start→gate→finish race."""
    cfg = EnvConfig(course=single_gate_course())  # gate x=3, finish x=6
    path = [(0, 0, 1), (2.9, 0, 1), (3.1, 0, 1), (5.9, 0, 1), (6.1, 0, 1)]
    env = _env_with(_ScriptedAdapter(path), cfg)
    env.reset()
    terminated = False
    info = {}
    for _ in range(5):
        _obs, _r, terminated, truncated, info = env.step(HOVER)
        if terminated or truncated:
            break
    assert terminated is True
    assert info["completed"] is True


def test_terminates_on_collision() -> None:
    env = make_env(adapter="simple")
    env.reset(seed=0)
    terminated = False
    info = {}
    for _ in range(500):
        _obs, _r, terminated, truncated, info = env.step(
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # no thrust → hits the floor
        )
        if terminated or truncated:
            break
    assert terminated is True
    assert info["collided"] is True
    assert info["completed"] is False


def test_truncates_on_timeout() -> None:
    # Small custom EpisodeConfig, kept fast: steps_per_gate=0 pins the budget to max_steps=8
    # regardless of the default 3-gate course, so hover truncates promptly at step 8.
    cfg = EnvConfig(episode=EpisodeConfig(dt=0.05, max_steps=8, steps_per_gate=0))
    env = make_env(cfg, adapter="simple")
    env.reset(seed=0)
    truncated = False
    steps = 0
    for _ in range(50):
        _obs, _r, terminated, truncated, _info = env.step(HOVER)  # hover: no crash, no finish
        steps += 1
        if terminated or truncated:
            break
    assert truncated is True
    assert steps == 8


# --- UC-09 AC1/AC5: the effective step budget scales with the gate count -------------
def _n_gate_course(n: int) -> CourseConfig:
    """Build an N-gate course (geometry need only be well-formed for the budget check)."""
    gates = tuple(GateSpec(center=(2.0 + 1.5 * i, 0.0, 1.0), aperture=0.6) for i in range(n))
    return CourseConfig(gates=gates, finish_x=2.0 + 1.5 * n + 1.0)


def test_effective_step_budget_scales_with_num_gates() -> None:
    """N=1⇒400, N=3⇒800, N=10⇒2200 (max_steps + steps_per_gate*(N-1)); N=1 == base (AC1)."""
    # N=1 via the single-gate factory keeps the 400 floor exactly.
    env1 = make_env(EnvConfig(course=single_gate_course()), adapter="simple")
    env1.reset(seed=0)
    assert env1._max_steps == 400

    # N=3 (the default course).
    env3 = make_env(EnvConfig(), adapter="simple")
    env3.reset(seed=0)
    assert env3._course.num_gates == 3
    assert env3._max_steps == 800

    # N=10.
    env10 = make_env(EnvConfig(course=_n_gate_course(10)), adapter="simple")
    env10.reset(seed=0)
    assert env10._course.num_gates == 10
    assert env10._max_steps == 2200


def test_truncation_fires_at_the_scaled_budget() -> None:
    """A hovering 2-gate episode truncates at max_steps + steps_per_gate*(N-1) (AC5).

    Keep it fast with a small custom EpisodeConfig: max_steps=5, steps_per_gate=3 ⇒ for the
    2-gate course the budget is 5 + 3*(2-1) = 8.
    """
    cfg = EnvConfig(
        course=_n_gate_course(2),
        episode=EpisodeConfig(dt=0.05, max_steps=5, steps_per_gate=3),
    )
    env = make_env(cfg, adapter="simple")
    env.reset(seed=0)
    assert env._max_steps == 8
    truncated = False
    steps = 0
    for _ in range(50):
        _obs, _r, terminated, truncated, _info = env.step(HOVER)
        steps += 1
        if terminated or truncated:
            break
    assert truncated is True
    assert steps == 8


def test_reset_info_exposes_target_gate() -> None:
    """UC-09 AC7 wiring: reset()'s info carries the initial target-gate index (0)."""
    env = make_env(EnvConfig(), adapter="simple")
    _obs, info = env.reset(seed=0)
    assert info["target_gate"] == 0


# --- AC11: fixed dynamics -----------------------------------------------------------
def test_dynamics_are_fixed_same_seed_same_actions_identical_rollout() -> None:
    rng = np.random.default_rng(0)
    actions = [rng.uniform([0, -1, -1, -1], [1, 1, 1, 1]).astype(np.float32) for _ in range(30)]

    def rollout():
        env = make_env(adapter="simple")
        obs, _ = env.reset(seed=42)
        trace = [obs.copy()]
        for a in actions:
            obs, r, t, tr, _ = env.step(a)
            trace.append(obs.copy())
            if t or tr:
                break
        return np.array(trace)

    np.testing.assert_array_equal(rollout(), rollout())


def test_different_seed_field_is_accepted_and_deterministic_per_seed() -> None:
    # The simple backend is noise-free, so seeding only reseeds a private RNG; the same
    # seed must always reproduce, which is the reproducibility contract (AC12) relies on.
    def first_obs(seed):
        env = make_env(adapter="simple")
        obs, _ = env.reset(seed=seed)
        return obs

    np.testing.assert_array_equal(first_obs(1), first_obs(1))


# --- AC12: reproducibility ----------------------------------------------------------
def test_full_episode_reproducible_for_fixed_seed() -> None:
    def episode_reward(seed):
        env = make_env(adapter="simple")
        env.reset(seed=seed)
        total = 0.0
        for _ in range(50):
            _obs, r, t, tr, _ = env.step(np.array([0.55, 0.02, 0.02, 0.0], dtype=np.float32))
            total += r
            if t or tr:
                break
        return total

    assert episode_reward(3) == episode_reward(3)


# ===========================================================================
# UC-08 — domain randomization at the env level (AC2, AC4, AC5, AC7)
# ===========================================================================
def _rollout_obs(config, seed, actions):
    """Roll a numpy-backed env through ``actions`` and return the stacked obs trace."""
    env = make_env(config, adapter="simple")
    obs, _ = env.reset(seed=seed)
    trace = [obs.copy()]
    for a in actions:
        obs, _r, terminated, truncated, _info = env.step(np.asarray(a, dtype=np.float32))
        trace.append(obs.copy())
        if terminated or truncated:
            break
    return np.array(trace, dtype=np.float32)


# --- UC-09 AC6: seed-42 determinism regression guard vs the REGENERATED baseline ------
# The baseline fixture ``tests/data/uc08_baseline_rollout.npz`` was REGENERATED under the
# new UC-09 default 3-gate course (see scripts/regen_uc08_baseline.py, AC6). The guard is no
# longer "matches a frozen pre-UC-09 trace" — it is "the current default env @ seed 42
# reproduces the committed baseline" (same seed → identical rollout). It still FAILS if any
# future change perturbs the seed-42 rollout, which is exactly the regression it guards.
def test_default_env_seed42_reproduces_the_committed_baseline() -> None:
    """The current default env @ seed 42 reproduces the committed (regenerated) baseline (AC6).

    Any RNG draw or dynamics change leaking into the disabled path — or any perturbation of
    the default course/geometry/reward wiring — breaks this same-seed determinism guard.
    """
    golden = np.load(_BASELINE_ROLLOUT)
    actions = list(golden["actions"])

    # EnvConfig() defaults have randomization OFF on both axes.
    trace = _rollout_obs(EnvConfig(), seed=42, actions=actions)

    assert trace.shape == golden["env_trace"].shape
    np.testing.assert_array_equal(trace, golden["env_trace"])


def test_explicit_disabled_randomization_matches_default() -> None:
    """An explicitly-disabled RandomizationConfig reproduces the same baseline (AC6/AC7)."""
    golden = np.load(_BASELINE_ROLLOUT)
    actions = list(golden["actions"])
    cfg = EnvConfig(randomization=RandomizationConfig(enable_course=False, enable_dynamics=False))
    trace = _rollout_obs(cfg, seed=42, actions=actions)
    np.testing.assert_array_equal(trace, golden["env_trace"])


# --- AC2: per-episode course varies across resets -----------------------------------
def test_course_varies_across_resets_when_enabled() -> None:
    cfg = EnvConfig(randomization=RandomizationConfig(enable_course=True))
    env = make_env(cfg, adapter="simple")
    env.reset(seed=0)
    starts = [tuple(np.round(env.active_course.start_position, 6))]
    for _ in range(9):
        env.reset()
        starts.append(tuple(np.round(env.active_course.start_position, 6)))
    assert len(set(starts)) > 1  # the course genuinely changes between episodes


def test_course_fixed_across_resets_when_disabled() -> None:
    """With course randomization off, active_course stays the fixed config course (AC7)."""
    env = make_env(EnvConfig(), adapter="simple")
    env.reset(seed=0)
    assert env.active_course == env.config.course
    env.reset()
    assert env.active_course == env.config.course


# --- active_course property ---------------------------------------------------------
def test_active_course_reflects_the_sampled_course() -> None:
    cfg = EnvConfig(randomization=RandomizationConfig(enable_course=True))
    env = make_env(cfg, adapter="simple")
    env.reset(seed=3)
    ac = env.active_course
    # It is a freshly-sampled, solvable course (not the static default in general).
    assert is_course_solvable(ac, cfg.randomization)
    # The env observation/geometry read this course, not config.course.
    assert env._course is ac


# --- AC3 wiring: no sampled course collides at step 0 -------------------------------
def test_no_sampled_course_collides_at_step_zero() -> None:
    """Every enabled-course reset spawns strictly inside the arena — no step-0 collision."""
    cfg = EnvConfig(randomization=RandomizationConfig(enable_course=True))
    rcfg = cfg.randomization
    env = make_env(cfg, adapter="simple")
    for i in range(200):
        env.reset(seed=i)
        course = env.active_course
        sz = course.start_position[2]
        assert course.floor_z + rcfg.z_margin < sz < course.ceiling_z - rcfg.z_margin
        # A fresh adapter spawned at that start does not flag a collision.
        ad = SimpleDroneAdapter(
            course.start, floor_z=course.floor_z, ceiling_z=course.ceiling_z, dt=0.05
        )
        assert not ad.reset(seed=i).collided


# --- AC5: dynamics on changes the rollout; off leaves it unchanged -------------------
def test_dynamics_randomization_changes_the_rollout() -> None:
    """Enabling ONLY the dynamics axis perturbs the trajectory vs the fixed baseline (AC5)."""
    rng = np.random.default_rng(1)
    actions = [rng.uniform([0, -1, -1, -1], [1, 1, 1, 1]) for _ in range(40)]

    base = _rollout_obs(EnvConfig(), seed=0, actions=actions)
    dyn = _rollout_obs(
        EnvConfig(randomization=RandomizationConfig(enable_dynamics=True)), seed=0, actions=actions
    )
    assert not (base.shape == dyn.shape and np.array_equal(base, dyn))


def test_dynamics_off_leaves_rollout_unchanged() -> None:
    """Dynamics off (default) is byte-identical to the fixed dynamics path (AC7)."""
    rng = np.random.default_rng(2)
    actions = [rng.uniform([0, -1, -1, -1], [1, 1, 1, 1]) for _ in range(40)]
    a = _rollout_obs(EnvConfig(), seed=0, actions=actions)
    b = _rollout_obs(
        EnvConfig(randomization=RandomizationConfig(enable_dynamics=False)), seed=0, actions=actions
    )
    np.testing.assert_array_equal(a, b)


# --- AC4/AC7: per-episode seeding keeps the axes independent -------------------------
def test_course_stream_is_reproducible_for_a_fixed_seed() -> None:
    """reset(seed=S) then repeated resets yield a deterministic course stream (AC4)."""
    cfg = EnvConfig(randomization=RandomizationConfig(enable_course=True))

    def stream(seed, k=6):
        env = make_env(cfg, adapter="simple")
        env.reset(seed=seed)
        seq = [env.active_course]
        for _ in range(k - 1):
            env.reset()
            seq.append(env.active_course)
        return seq

    assert stream(11) == stream(11)
    assert stream(11) != stream(12)


def test_dynamics_toggle_does_not_shift_the_per_seed_sampled_course() -> None:
    """Course is drawn BEFORE dynamics, so toggling the dynamics axis never changes the
    course sampled for a given per-episode seed (AC4/AC7 independence).

    (Across a *continuous* stream the shared RNG advances differently, which is why the
    canonical mode seeds per episode; here we assert the per-seed invariance the design
    guarantees.)
    """

    def first_course(enable_dynamics, seed):
        cfg = EnvConfig(
            randomization=RandomizationConfig(enable_course=True, enable_dynamics=enable_dynamics)
        )
        env = make_env(cfg, adapter="simple")
        env.reset(seed=seed)
        return env.active_course

    for seed in (0, 1, 2, 7, 13):
        assert first_course(False, seed) == first_course(True, seed)


# ===========================================================================
# UC-15 — obstacle vision widens the obs; contact penalises but never terminates
# ===========================================================================
def test_obstacle_vision_widens_observation_to_24d() -> None:
    """AC4: enabling the obstacle-vision block emits a 24-d obs (12 base + 4*k, k=3)."""
    cfg = EnvConfig(
        course=default_obstacle_course(),
        obstacle_vision=ObstacleVisionConfig(enabled=True, k=3),
    )
    env = make_env(cfg, adapter="simple")
    assert env.observation_space.shape == (OBS_DIM + 12,)
    assert env.observation_space.shape == (24,)
    assert env.obs_width == 24
    obs, _ = env.reset(seed=0)
    assert obs.shape == (24,)
    assert np.isfinite(obs).all()  # the obstacle block is inside the nan_to_num sanitation


def test_obs_width_tracks_k() -> None:
    """AC4: ``obs_width`` == OBS_DIM + 4*k for the configured k."""
    for k in (1, 2, 5):
        env = make_env(
            EnvConfig(obstacle_vision=ObstacleVisionConfig(enabled=True, k=k)), adapter="simple"
        )
        assert env.obs_width == OBS_DIM + 4 * k


def test_legacy_env_stays_12d_when_obstacle_vision_disabled() -> None:
    """AC4/back-compat: the default env (obstacle vision off) is the locked 12-d contract."""
    env = make_env(EnvConfig(), adapter="simple")
    assert env.observation_space.shape == (OBS_DIM,) == (12,)
    assert env.obs_width == 12
    # Even a course that HAS obstacles stays 12-d while the vision block is off.
    course = CourseConfig(obstacles=(ObstacleSpec(center=(3.0, 1.5), radius=0.3, height=2.5),))
    env2 = make_env(EnvConfig(course=course), adapter="simple")
    assert env2.obs_width == 12


# A single-gate course with one pillar sitting on the flight path at x=4, y=0.
_PILLAR_ON_PATH = CourseConfig(
    start_position=(0.0, 0.0, 1.0),
    gates=(GateSpec(center=(2.5, 0.0, 1.0), aperture=0.6),),
    finish_x=7.0,
    obstacles=(ObstacleSpec(center=(4.0, 0.0), radius=0.5, height=2.5),),
)
# start → g0 → into the pillar → still inside → out toward finish → cross finish.
_GRAZE_PATH = [
    (0.0, 0.0, 1.0),
    (2.5, 0.0, 1.0),  # g0
    (4.0, 0.0, 1.0),  # contact begins (inside the pillar)
    (4.2, 0.0, 1.0),  # still overlapping
    (6.9, 0.0, 1.0),
    (7.1, 0.0, 1.0),  # finish
]


def test_obstacle_contact_penalises_but_does_not_terminate() -> None:
    """AC2/AC9: contact raises the penalty flag and NEVER sets ``terminated``.

    The scripted walk flies through a pillar (no floor/ceiling collision), so the only
    possible terminations are the finish crossing; the contact steps themselves stay live.
    """
    env = _env_with(_ScriptedAdapter(_GRAZE_PATH), EnvConfig(course=_PILLAR_ON_PATH))
    env.reset()
    contact_step = None
    terminated_before_finish = False
    for i in range(len(_GRAZE_PATH)):
        _obs, reward, terminated, _trunc, info = env.step(HOVER)
        if info["obstacle_contact"]:
            contact_step = i
            # The contact step subtracted the severe penalty and did NOT terminate.
            assert terminated is False
            # The severe -50 penalty dominates the small progress/time terms → strongly negative.
            assert reward < -40
        if info["completed"]:
            break
        if terminated:
            terminated_before_finish = True
    assert contact_step is not None, "the scripted fly-through must register a contact"
    assert terminated_before_finish is False


def test_glancing_contact_still_completes_the_course() -> None:
    """AC9: a drone that grazes a pillar can recover aerially and still finish (terminated)."""
    env = _env_with(_ScriptedAdapter(_GRAZE_PATH), EnvConfig(course=_PILLAR_ON_PATH))
    env.reset()
    saw_contact = False
    completed = False
    for _ in range(len(_GRAZE_PATH)):
        _obs, _r, terminated, _trunc, info = env.step(HOVER)
        saw_contact = saw_contact or info["obstacle_contact"]
        if terminated:
            completed = info["completed"]
            break
    assert saw_contact is True
    assert completed is True  # grazing a pillar did not stop the course from completing


def test_obstacle_contact_is_edge_triggered_once_per_contact() -> None:
    """AC2: a sustained overlap raises the flag ONCE (edge-triggered), not every step.

    The scripted path enters the pillar and stays overlapping for several steps; only the
    first overlapping step is flagged (so the penalty can't stack unboundedly per frame).
    """
    env = _env_with(_ScriptedAdapter(_GRAZE_PATH), EnvConfig(course=_PILLAR_ON_PATH))
    env.reset()
    flags = []
    for _ in range(len(_GRAZE_PATH)):
        _obs, _r, terminated, _trunc, info = env.step(HOVER)
        flags.append(info["obstacle_contact"])
        if terminated:
            break
    assert sum(1 for f in flags if f) == 1, f"expected exactly one edge-trigger, got {flags}"


def test_prev_contact_edge_state_resets_each_episode() -> None:
    """The edge-trigger state is reset on ``reset()`` so a new episode re-fires on contact."""
    env = _env_with(_ScriptedAdapter(_GRAZE_PATH), EnvConfig(course=_PILLAR_ON_PATH))
    for _ in range(2):
        env.reset()  # rewinds the scripted replay (adapter.reset sets the index to 0)
        fired = False
        for _ in range(len(_GRAZE_PATH)):
            _obs, _r, terminated, _trunc, info = env.step(HOVER)
            fired = fired or info["obstacle_contact"]
            if terminated:
                break
        assert fired is True


def test_no_obstacle_course_never_flags_contact() -> None:
    """A course with no pillars never raises ``obstacle_contact`` (back-compat)."""
    env = _env_with(_ScriptedAdapter(_THREE_GATE_PATH))
    env.reset()
    for _ in range(len(_THREE_GATE_PATH)):
        _obs, _r, terminated, _trunc, info = env.step(HOVER)
        assert info["obstacle_contact"] is False
        if terminated:
            break


# ===========================================================================
# UC-16 — pad docking: land / dwell / takeoff, docked via info only, byte-identity
# ===========================================================================
class _DockScriptedAdapter:
    """A scripted adapter with **per-step** contact flags and attitudes, for dock tests.

    ``_ScriptedAdapter`` only supports a single ``collide_at`` and always-level attitude; a
    dock→dwell→takeoff trajectory needs contact sustained across several steps (dwell) and a
    controllable attitude (to prove upright landings dock). Positions/attitudes/contact are
    replayed frame by frame; the env infers descent speed from ``(prev_z - curr_z)/dt`` across
    the replayed positions, so the trajectory drives the dock classifier deterministically.
    """

    backend = "scripted"

    def __init__(self, positions, collided_flags, attitudes=None):
        self._positions = [np.asarray(p, dtype=np.float64) for p in positions]
        self._collided = list(collided_flags)
        if attitudes is None:
            attitudes = [(0.0, 0.0, 0.0)] * len(positions)
        self._attitudes = [np.asarray(a, dtype=np.float64) for a in attitudes]
        self._i = 0

    def _state(self, idx) -> DroneState:
        j = min(idx, len(self._positions) - 1)
        return DroneState(
            position=self._positions[j].copy(),
            velocity=np.zeros(3),
            attitude=self._attitudes[j].copy(),
            angular_velocity=np.zeros(3),
            collided=bool(self._collided[j]),
        )

    def reset(self, seed=None) -> DroneState:
        self._i = 0
        return self._state(0)

    def step(self, action) -> DroneState:
        self._i += 1
        return self._state(self._i)

    def close(self) -> None:  # pragma: no cover - trivial
        pass


# A one-gate, one-pad course with the pad at the spawn (0,0). Floor z=0; gate far away at x=3
# and finish at x=6, so an at-origin dock/dwell/takeoff never passes a gate or the finish.
_DOCK_COURSE = single_pad_course(pad_center=(0.0, 0.0), pad_radius=0.5)

# dt = 0.05 (EpisodeConfig default) ⇒ descent = (prev_z - curr_z)/0.05.
# Frames: 0 spawn (just above floor, over pad) → 1 land (descent 0.4<0.5) → 2,3 dwell (settled)
# → 4 takeoff (airborne, no contact) → 5 descend → 6 re-dock (descent 0.4<0.5). All upright.
_DOCK_POSITIONS = [
    (0.0, 0.0, 0.02),  # 0: spawn, over the pad, just above the floor
    (0.0, 0.0, 0.00),  # 1: DOCK — floor contact, |vz|≈0.4 m/s, upright, over pad
    (0.0, 0.0, 0.00),  # 2: DWELL — settled on the pad
    (0.0, 0.0, 0.00),  # 3: DWELL — still settled
    (0.0, 0.0, 0.30),  # 4: TAKEOFF — lifted off the floor, no contact
    (0.0, 0.0, 0.02),  # 5: descending back toward the pad (airborne)
    (0.0, 0.0, 0.00),  # 6: RE-DOCK — controlled touchdown again
]
_DOCK_CONTACT = [False, True, True, True, False, False, True]


def _dock_env():
    env = _env_with(
        _DockScriptedAdapter(_DOCK_POSITIONS, _DOCK_CONTACT), EnvConfig(course=_DOCK_COURSE)
    )
    env.reset()
    return env


def test_controlled_landing_docks_without_terminating() -> None:
    """AC2: a slow, upright, over-pad floor contact sets ``docked`` and does NOT terminate."""
    env = _dock_env()
    _obs, _r, terminated, truncated, info = env.step(HOVER)  # frame 1: the landing
    assert info["docked"] is True
    assert terminated is False
    assert truncated is False
    # A dock is NOT reported as a collision (crash) — the two are disjoint (AC3 wiring).
    assert info["collided"] is False
    assert info["completed"] is False


def test_dock_dwell_persists_across_steps() -> None:
    """AC4: ``docked`` stays True across several dwell steps; the episode stays alive."""
    env = _dock_env()
    docked_flags = []
    for _ in range(4):  # frames 1 (land) + 2,3 (dwell) + ...
        _obs, _r, terminated, truncated, info = env.step(HOVER)
        docked_flags.append(info["docked"])
        assert terminated is False and truncated is False
    # Land + two dwell steps are all docked (level info, not edge-triggered).
    assert docked_flags[:3] == [True, True, True]


def test_takeoff_clears_dock_then_redocks() -> None:
    """AC5: throttling off the pad clears ``docked`` (normal flight), a re-touchdown re-docks.

    Dock↔fly is repeatable within one episode and never terminates — the whole
    land→dwell→takeoff→re-dock sequence stays live.
    """
    env = _dock_env()
    seq = []
    for _ in range(len(_DOCK_POSITIONS) - 1):  # frames 1..6
        _obs, _r, terminated, _trunc, info = env.step(HOVER)
        seq.append(info["docked"])
        assert terminated is False  # nothing in the sequence crashes or completes
    # land, dwell, dwell, takeoff(clear), fly, re-dock
    assert seq == [True, True, True, False, False, True]


def test_docked_state_surfaced_via_info_only_observation_byte_identical() -> None:
    """AC6: docking changes ``info`` only — the observation is byte-identical to the no-pad run.

    Two envs replay the *same* scripted positions/attitudes/contact; one course has the pad
    (so frame 1 docks), the other has none (so frame 1 crashes). The emitted observation at the
    contact step is bit-for-bit identical — the divergence lives entirely in ``info['docked']``
    / ``info['collided']`` / ``terminated``, never in the 12-d observation vector.
    """
    no_pad_course = single_gate_course(
        gate_center=(3.0, 0.0, 1.0), finish_x=6.0
    )  # identical geo, 0 pads
    pad_env = _env_with(
        _DockScriptedAdapter(_DOCK_POSITIONS, _DOCK_CONTACT), EnvConfig(course=_DOCK_COURSE)
    )
    bare_env = _env_with(
        _DockScriptedAdapter(_DOCK_POSITIONS, _DOCK_CONTACT), EnvConfig(course=no_pad_course)
    )
    obs_pad, _ = pad_env.reset()
    obs_bare, _ = bare_env.reset()
    # Reset observation identical (no pad influence on the obs vector).
    assert obs_pad.shape == (OBS_DIM,)
    np.testing.assert_array_equal(obs_pad, obs_bare)

    # Frame 1 — the contact step: obs identical, but behaviour diverges only in info/termination.
    obs_pad, _r, term_pad, _t, info_pad = pad_env.step(HOVER)
    obs_bare, _r, term_bare, _t, info_bare = bare_env.step(HOVER)
    np.testing.assert_array_equal(obs_pad, obs_bare)  # byte-identical observation
    assert obs_pad.shape == (OBS_DIM,)  # still the locked 12-d contract (no obs-schema block)
    assert info_pad["docked"] is True and term_pad is False  # pad → dock, alive
    assert info_bare["docked"] is False and term_bare is True  # no pad → crash, terminated
    assert "docked" in info_pad  # surfaced via info, not the observation


# --- AC8: off-by-default byte-identity (no pads, no dock, no RNG draw) ---------------
def _full_stream(config, seed, actions):
    """Run a numpy-backed env and capture the full (obs, reward, terminated, truncated) stream
    plus the final RNG bit-generator state — the byte-identity fingerprint for AC8.
    """
    env = make_env(config, adapter="simple")
    obs, _ = env.reset(seed=seed)
    obs_trace = [obs.copy()]
    rewards, terms, truncs = [], [], []
    for a in actions:
        obs, r, terminated, truncated, _info = env.step(np.asarray(a, dtype=np.float32))
        obs_trace.append(obs.copy())
        rewards.append(r)
        terms.append(terminated)
        truncs.append(truncated)
        if terminated or truncated:
            break
    rng_state = env.np_random.bit_generator.state
    return np.array(obs_trace, dtype=np.float32), rewards, terms, truncs, rng_state


def test_no_pad_env_is_byte_identical_to_committed_baseline() -> None:
    """AC8: with no pads, a fixed-seed episode reproduces the pre-UC-16 committed baseline obs.

    ``EnvConfig()`` has an empty ``course.pads`` and an all-default ``dock``; the dock predicate
    short-circuits ``False`` every step, so the observation stream is bit-identical to the
    regenerated seed-42 golden rollout (which predates the UC-16 dock wiring).
    """
    golden = np.load(_BASELINE_ROLLOUT)
    actions = list(golden["actions"])
    obs_trace, _r, _t, _tr, _rng = _full_stream(EnvConfig(), seed=42, actions=actions)
    assert obs_trace.shape == golden["env_trace"].shape
    np.testing.assert_array_equal(obs_trace, golden["env_trace"])


def test_dock_config_and_present_pads_do_not_perturb_stream_or_rng() -> None:
    """AC8: pads/dock are inert when no floor contact occurs — obs, reward, termination step,
    and RNG consumption are all byte-identical to the default env.

    Compares three configs under the same seed + action stream: the default env (no pads), a
    custom ``DockConfig`` with a different threshold (still no pads), and ``default_pad_course``
    (pads PRESENT but never touched — the drone never lands). All three must match bit-for-bit,
    proving the dock thresholds are inert without contact and pads draw **zero** RNG.
    """
    rng = np.random.default_rng(7)
    actions = [rng.uniform([0, -1, -1, -1], [1, 1, 1, 1]) for _ in range(60)]

    base = _full_stream(EnvConfig(), seed=0, actions=actions)
    custom_dock = _full_stream(
        EnvConfig(dock=DockConfig(max_dock_descent_speed=99.0, max_dock_tilt=3.0)),
        seed=0,
        actions=actions,
    )
    with_pads = _full_stream(EnvConfig(course=default_pad_course()), seed=0, actions=actions)

    for other in (custom_dock, with_pads):
        np.testing.assert_array_equal(base[0], other[0])  # observation stream
        assert base[1] == other[1]  # reward stream
        assert base[2] == other[2]  # terminated stream (same termination step)
        assert base[3] == other[3]  # truncated stream
        assert base[4] == other[4]  # final RNG state — identical draw count (no pad RNG draw)


def test_no_pad_stream_is_deterministic_for_a_fixed_seed() -> None:
    """AC8: same seed → identical obs/reward/termination/RNG stream (determinism guard)."""
    rng = np.random.default_rng(3)
    actions = [rng.uniform([0, -1, -1, -1], [1, 1, 1, 1]) for _ in range(40)]
    a = _full_stream(EnvConfig(), seed=5, actions=actions)
    b = _full_stream(EnvConfig(), seed=5, actions=actions)
    np.testing.assert_array_equal(a[0], b[0])
    assert (a[1], a[2], a[3], a[4]) == (b[1], b[2], b[3], b[4])


# --- AC9: hermetic finite docking trajectory (scripted + real-adapter smoke) ---------
def test_scripted_dock_trajectory_is_finite_and_bounded() -> None:
    """AC9: the offline scripted dock→dwell→takeoff→re-dock trajectory runs finite and clean.

    No pybullet, no network — every step yields a finite 12-d observation and a finite reward,
    the run never raises, and it stays live (a dock never terminates) through the whole script.
    """
    env = _dock_env()
    docked_any = False
    for _ in range(len(_DOCK_POSITIONS) - 1):
        obs, reward, terminated, truncated, info = env.step(HOVER)
        assert obs.shape == (OBS_DIM,)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert terminated is False and truncated is False
        docked_any = docked_any or info["docked"]
    assert docked_any is True  # the scripted trajectory did dock at least once


def test_real_adapter_docking_smoke_run_completes_finitely() -> None:
    """AC9: a real numpy-adapter episode on a pad course terminates/truncates within budget.

    Hover over the spawn pad on the ``simple`` adapter for the whole step budget: the run is
    hermetic, every observation/reward stays finite, and the episode ends (truncation) in a
    bounded number of steps — the docking wiring never makes the env hang or diverge.
    """
    # Small budget so the smoke run is fast: steps_per_gate=0 pins it to max_steps for N gates.
    cfg = EnvConfig(
        course=default_pad_course(), episode=EpisodeConfig(max_steps=25, steps_per_gate=0)
    )
    env = make_env(cfg, adapter="simple")
    env.reset(seed=0)
    ended = False
    steps = 0
    for _ in range(200):  # hard cap well above the 25-step budget → proves it is finite
        obs, reward, terminated, truncated, _info = env.step(HOVER)
        steps += 1
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if terminated or truncated:
            ended = True
            break
    assert ended is True
    assert steps <= 25


# =====================================================================================
# UC-17 — battery observation block + soft-depletion termination (AC3/AC4/AC5)
# =====================================================================================
from drone_fly.env.config import BatteryConfig  # noqa: E402


def test_battery_off_by_default_obs_width_unchanged() -> None:
    """AC5: EnvConfig() has battery off → observation stays the locked 12-d contract."""
    env = make_env(EnvConfig(), adapter="simple")
    assert env.observation_space.shape == (OBS_DIM,) == (12,)


def test_battery_enabled_appends_exactly_one_dim() -> None:
    """AC4: enabling battery appends EXACTLY one observation dim (the width-1 battery block)."""
    off = make_env(EnvConfig(), adapter="simple")
    on = make_env(EnvConfig(battery=BatteryConfig(enabled=True)), adapter="simple")
    assert on.observation_space.shape[0] == off.observation_space.shape[0] + 1


def test_battery_block_is_the_last_obs_dim_and_encodes_depletion() -> None:
    """AC4: the battery dim is appended LAST and carries depletion = 1 - charge (0 at full)."""
    env = make_env(EnvConfig(battery=BatteryConfig(enabled=True)), adapter="simple")
    obs, _info = env.reset(seed=0)
    # At reset the battery is full → depletion 0.0 in the last dim (matches the zero-init graft
    # baseline so a warm-started actor sees its trained baseline).
    assert obs[-1] == pytest.approx(0.0)
    # After a throttle step the battery drains → the last dim rises above 0 (depletion grows).
    obs2, *_ = env.step(np.array([0.8, 0.0, 0.0, 0.0], dtype=np.float32))
    assert obs2[-1] > 0.0


def test_full_battery_hunger_v3_width_and_block_order() -> None:
    """AC4: with obstacle-vision AND battery enabled the env emits 25 dims in block order
    (vision, proprioception, obstacle_vision, battery) — matching ``battery_hunger_v3``."""
    from drone_fly.controller.obs_schema import BATTERY_HUNGER_V3

    cfg = EnvConfig(
        course=default_obstacle_course(),
        obstacle_vision=ObstacleVisionConfig(enabled=True),
        battery=BatteryConfig(enabled=True),
    )
    env = make_env(cfg, adapter="simple")
    assert env.observation_space.shape == (25,)
    assert env.obs_width == BATTERY_HUNGER_V3.total_width == 25
    obs, _info = env.reset(seed=0)
    # Battery dim is strictly last; at reset it is full-charge depletion (0.0).
    assert obs.shape == (25,)
    assert obs[-1] == pytest.approx(0.0)


def test_battery_depletion_triggers_soft_crash_termination() -> None:
    """AC3: a drained battery cannot hover → the drone sinks to the floor and the episode
    terminates via the EXISTING crash path (terminated AND info['collided'], NOT a success, NOT a
    timeout). Uses a fast idle drain so depletion is reached deterministically within the budget."""
    cfg = EnvConfig(
        # Fast idle drain empties the battery in a few steps; no throttle term needed.
        battery=BatteryConfig(enabled=True, idle_rate=10.0, throttle_rate=0.0),
    )
    env = make_env(cfg, adapter="simple")
    env.reset(seed=0)
    terminated = truncated = False
    info: dict = {}
    for _ in range(cfg.episode.max_steps + cfg.episode.steps_per_gate * 3):
        # Full throttle: with an empty battery the ceiling is below hover, so it still sinks.
        _obs, _reward, terminated, truncated, info = env.step(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        if terminated or truncated:
            break
    assert terminated is True, "an empty battery must end the episode via the crash path"
    assert truncated is False, "the soft crash terminates before the timeout, not by truncation"
    assert info["collided"] is True, "the soft depletion is reported as a (crash) collision"
    assert info.get("is_success") is False, "sinking on an empty battery is not a course success"


# ===========================================================================
# UC-18 — recharge pads: dock-to-recharge, course-variation, load-bearing (AC1/AC2/AC5/AC6)
# ===========================================================================
from drone_fly.env.config import PadSpec  # noqa: E402


class _BatteryDockScriptedAdapter:
    """A scripted-position dock adapter that carries a REAL numpy battery ledger (UC-18 AC6).

    Extends the UC-16 :class:`_DockScriptedAdapter` idea: positions / contact / attitude are
    replayed frame by frame (so the dock classifier is driven deterministically), but the battery
    is a genuine :class:`SimpleDroneAdapter` — the SAME numpy drain (applied every ``step``) and
    ``recharge`` (clamp-at-1.0) arithmetic the flight adapter uses. This lets a hermetic scripted
    trajectory *measure* the battery in the numpy sim: the env drains on every step and tops up via
    ``recharge`` only on a docked step over a ``rechargeable`` pad — exactly as in real flight — so
    "the pad is load-bearing" is a measured claim about the numpy battery ledger, not a stub.
    """

    backend = "scripted"

    def __init__(
        self,
        positions,
        collided_flags,
        battery_cfg,
        *,
        dt=0.05,
        floor_z=0.0,
        ceiling_z=2.5,
        start_position=(0.0, 0.0, 1.0),
        attitudes=None,
    ):
        self._positions = [np.asarray(p, dtype=np.float64) for p in positions]
        self._collided = list(collided_flags)
        if attitudes is None:
            attitudes = [(0.0, 0.0, 0.0)] * len(positions)
        self._attitudes = [np.asarray(a, dtype=np.float64) for a in attitudes]
        self._i = 0
        # A real numpy adapter, used ONLY as the battery ledger (its integrated position is
        # ignored — positions are scripted). This is the numpy sim doing the battery arithmetic.
        self._battery_adapter = SimpleDroneAdapter(
            np.asarray(start_position, dtype=np.float64),
            floor_z=floor_z,
            ceiling_z=ceiling_z,
            dt=dt,
            battery=battery_cfg,
        )

    def _state(self, idx) -> DroneState:
        j = min(idx, len(self._positions) - 1)
        return DroneState(
            position=self._positions[j].copy(),
            velocity=np.zeros(3),
            attitude=self._attitudes[j].copy(),
            angular_velocity=np.zeros(3),
            collided=bool(self._collided[j]),
            battery=self._battery_adapter._battery,
        )

    def reset(self, seed=None) -> DroneState:
        self._i = 0
        self._battery_adapter.reset(seed=seed)
        return self._state(0)

    def step(self, action) -> DroneState:
        self._i += 1
        self._battery_adapter.step(action)  # genuine numpy drain for this step
        return self._state(self._i)

    def recharge(self, delta: float) -> float:
        return self._battery_adapter.recharge(delta)

    def close(self) -> None:  # pragma: no cover - trivial
        pass


# A tuned, energy-constrained fixture (per the plan's "tuned drain" note). dt=0.05 → drain
# 0.10/step (idle 2.0, throttle term 0); recharge_rate 8.0 → +0.40 per docked step (net +0.30).
# The net-positive invariant (8.0 > 2.0) holds with wide margins so the arithmetic is robust.
_RCHG_BATTERY = BatteryConfig(enabled=True, idle_rate=2.0, throttle_rate=0.0, recharge_rate=8.0)


def _recharge_course(rechargeable: bool) -> CourseConfig:
    """A one-gate course with a single floor pad at (2, 0); ``rechargeable`` toggles the flag."""
    return CourseConfig(
        start_position=(0.0, 0.0, 1.0),
        gates=(GateSpec(center=(4.0, 0.0, 1.0), aperture=0.6),),
        finish_x=5.0,
        floor_z=0.0,
        ceiling_z=2.5,
        pads=(PadSpec(center=(2.0, 0.0), radius=0.5, rechargeable=rechargeable),),
    )


# Scripted trajectory: fly+drain → slow descent onto the pad → dwell (refill) → take off →
# pass the gate → cross the finish. Contact is True only on the five docked frames (5..9).
_RCHG_POSITIONS = [
    (0.0, 0.0, 1.0),  # 0 spawn (reset)
    (0.7, 0.0, 1.0),  # 1 fly forward (airborne, drains)
    (1.4, 0.0, 0.8),  # 2 descend toward the pad
    (2.0, 0.0, 0.3),  # 3 airborne OVER the pad (no contact → AC2: no refill while hovering)
    (2.0, 0.0, 0.02),  # 4 slow final approach, still airborne
    (2.0, 0.0, 0.0),  # 5 DOCK — floor contact, descent 0.4<0.5, upright, over pad
    (2.0, 0.0, 0.0),  # 6 dwell (refill)
    (2.0, 0.0, 0.0),  # 7 dwell
    (2.0, 0.0, 0.0),  # 8 dwell
    (2.0, 0.0, 0.0),  # 9 dwell
    (2.0, 0.0, 0.3),  # 10 take off (airborne again)
    (3.0, 0.0, 0.8),  # 11 fly toward the gate
    (4.0, 0.0, 1.0),  # 12 pass the gate (segment reaches the gate centre, within aperture)
    (5.1, 0.0, 1.0),  # 13 cross the finish plane (x 4.0 → 5.1 forward-crosses finish_x=5.0)
]
_RCHG_CONTACT = [False] * 5 + [True] * 5 + [False] * 4
_RCHG_DOCK_FRAMES = range(5, 10)  # step indices where the drone is docked on the pad


def _run_recharge_trajectory(rechargeable: bool):
    """Replay the scripted trajectory on the battery-ledger adapter; return per-step battery
    charge (decoded from the width-1 battery obs), whether the course completed, and dock flags.
    """
    env = _env_with(
        _BatteryDockScriptedAdapter(_RCHG_POSITIONS, _RCHG_CONTACT, _RCHG_BATTERY),
        EnvConfig(course=_recharge_course(rechargeable), battery=_RCHG_BATTERY),
    )
    obs, _info = env.reset(seed=0)
    # Battery obs is the last dim, encoded as depletion = 1 - charge.
    charges = [1.0 - float(obs[-1])]
    docked = [False]
    completed = False
    completed_at = None
    for i in range(1, len(_RCHG_POSITIONS)):
        obs, _r, terminated, truncated, info = env.step(HOVER)
        charges.append(1.0 - float(obs[-1]))
        docked.append(bool(info["docked"]))
        if info.get("completed") and completed_at is None:
            completed = True
            completed_at = i
        if terminated or truncated:
            break
    return charges, docked, completed, completed_at


# --- AC1: recharge accrues ONLY while docked on a rechargeable pad; clamps at 1.0 ---------
def test_recharge_accrues_only_while_docked_on_a_rechargeable_pad() -> None:
    """AC1: the battery drains while airborne, then RISES across the docked dwell on a
    rechargeable pad (each docked step nets a gain), then drains again after take off."""
    charges, docked, _completed, _at = _run_recharge_trajectory(rechargeable=True)
    # Airborne approach (frames 1..4): strictly draining.
    assert charges[4] < charges[1] < charges[0]
    # Dock (frame 5) refills: the observed charge jumps up versus the pre-dock low.
    assert charges[5] > charges[4]
    # Every docked frame is flagged docked and never below the pre-dock charge (net-positive).
    assert all(docked[i] for i in _RCHG_DOCK_FRAMES)
    assert min(charges[i] for i in _RCHG_DOCK_FRAMES) >= charges[4]
    # After take off (frames 10..) the battery drains again (no more refill).
    assert charges[11] < charges[10]


def test_recharge_clamps_at_full_charge_over_a_long_dwell() -> None:
    """AC1: a sustained dwell tops the battery up toward 1.0 and never exceeds it (clamp)."""
    charges, _docked, _c, _a = _run_recharge_trajectory(rechargeable=True)
    assert max(charges) <= 1.0 + 1e-9
    # The dwell actually reached (clamped at) full charge on this tuned fixture.
    assert max(charges[i] for i in _RCHG_DOCK_FRAMES) == pytest.approx(1.0)


def test_non_rechargeable_pad_never_refills_even_while_docked() -> None:
    """AC1/AC5: docking on a plain (non-recharge) pad never refills — the battery is monotone
    non-increasing across the whole trajectory, drain-only exactly as UC-16/17."""
    charges, docked, _c, _a = _run_recharge_trajectory(rechargeable=False)
    # It still docks (the flag does not change dock geometry)...
    assert all(docked[i] for i in _RCHG_DOCK_FRAMES)
    # ...but the charge only ever falls (no refill on a non-recharge pad).
    assert all(y <= x + 1e-12 for x, y in zip(charges, charges[1:], strict=False))


# --- AC2: hovering over a recharge pad (airborne, not docked) does NOT recharge ------------
def test_hover_over_recharge_pad_does_not_recharge() -> None:
    """AC2: full landing required — frames 3 and 4 are airborne directly over the recharge pad
    (no floor contact ⇒ not docked), so the battery keeps draining; no charge accrues in the air."""
    charges, docked, _c, _a = _run_recharge_trajectory(rechargeable=True)
    # Frames 3,4 sit over the pad horizontally but are airborne — not docked, so no refill.
    assert docked[3] is False and docked[4] is False
    assert charges[3] < charges[2]  # still draining while hovering over the pad
    assert charges[4] < charges[3]


# --- AC6: the pad is LOAD-BEARING — completes WITH the dwell, dies WITHOUT it -------------
def test_recharge_pad_is_load_bearing_completes_with_dwell_and_dies_without() -> None:
    """AC6: on an energy-constrained course the SAME scripted trajectory completes when the pad
    recharges (battery stays strictly positive to the finish) and would fail without it (the
    battery is fully depleted well before the finish). Both measured in the numpy battery sim.
    """
    # WITH recharge: genuine completion with charge to spare at every frame.
    charges_on, docked_on, completed_on, at_on = _run_recharge_trajectory(rechargeable=True)
    assert completed_on is True, "the course must complete when the recharge dwell tops up"
    assert at_on == len(_RCHG_POSITIONS) - 1  # completed on the final (finish) frame
    assert min(charges_on) > 0.0, "with recharge the battery never depletes en route"

    # WITHOUT recharge: the battery is fully depleted (0.0) BEFORE the finish frame — a real drone
    # could not sustain flight the rest of the way (an empty battery cannot hover; see below).
    charges_off, _docked_off, _completed_off, _at_off = _run_recharge_trajectory(rechargeable=False)
    finish_frame = len(_RCHG_POSITIONS) - 1
    depleted_frames = [i for i, c in enumerate(charges_off) if c == pytest.approx(0.0)]
    assert depleted_frames, "without recharge the battery must fully deplete"
    assert depleted_frames[0] < finish_frame, "depletion must occur strictly before the finish"
    # Tie the measured depletion to real inability to fly: an empty battery cannot hover
    # (ceiling_factor(0) < the 0.5 hover threshold at base TWR 2), so the pad is load-bearing.
    assert _RCHG_BATTERY.ceiling_factor(0.0) < 0.5


# --- AC5: a rechargeable pad is inert when battery physics are disabled (byte-identity) ----
def test_rechargeable_flag_is_inert_when_battery_disabled() -> None:
    """AC5: with battery DISABLED the recharge path is gated off entirely, so a course carrying a
    rechargeable pad is byte-identical (obs / reward / termination / RNG) to the same course whose
    pad is plain — and the observation stays the locked 12-d contract (no battery block)."""

    def stream(rechargeable: bool):
        env = make_env(EnvConfig(course=_recharge_course(rechargeable)), adapter="simple")
        rng = np.random.default_rng(4)
        actions = [rng.uniform([0, -1, -1, -1], [1, 1, 1, 1]) for _ in range(40)]
        obs, _ = env.reset(seed=0)
        trace = [obs.copy()]
        rewards, terms, truncs = [], [], []
        for a in actions:
            obs, r, terminated, truncated, _info = env.step(np.asarray(a, dtype=np.float32))
            trace.append(obs.copy())
            rewards.append(r)
            terms.append(terminated)
            truncs.append(truncated)
            if terminated or truncated:
                break
        return np.array(trace), rewards, terms, truncs, env.np_random.bit_generator.state

    on = stream(rechargeable=True)
    off = stream(rechargeable=False)
    assert on[0].shape[1] == OBS_DIM  # still the locked 12-d observation (battery off)
    np.testing.assert_array_equal(on[0], off[0])  # observation stream
    assert on[1] == off[1]  # reward stream
    assert on[2] == off[2] and on[3] == off[3]  # termination / truncation streams
    assert on[4] == off[4]  # identical RNG consumption (the flag draws nothing)


def test_rechargeable_pad_extends_the_step_budget() -> None:
    """UC-18 (step-budget allowance): the env grants ``recharge_step_allowance`` extra steps per
    rechargeable pad on the active course (so a legitimate recharge detour still finishes in time),
    and grants NONE when the pad is plain (byte-identical budget)."""
    with_recharge = make_env(
        EnvConfig(course=_recharge_course(True), battery=_RCHG_BATTERY), adapter="simple"
    )
    without = make_env(
        EnvConfig(course=_recharge_course(False), battery=_RCHG_BATTERY), adapter="simple"
    )
    with_recharge.reset(seed=0)
    without.reset(seed=0)
    episode = EpisodeConfig()
    base_budget = episode.max_steps + episode.steps_per_gate * (1 - 1)  # N=1 course
    assert without._max_steps == base_budget  # plain pad → no extra budget
    assert with_recharge._max_steps == base_budget + episode.recharge_step_allowance


# ===========================================================================
# UC-19 — damage/integrity + repair pads (AC1/AC2/AC4/AC7)
# ===========================================================================
class _DamageDockScriptedAdapter:
    """A scripted-position dock adapter that carries a REAL numpy INTEGRITY ledger (UC-19).

    The exact counterpart of :class:`_BatteryDockScriptedAdapter` for integrity: positions /
    contact / attitude are replayed frame by frame (so the env's obstacle-contact and dock
    classifiers are driven deterministically), but integrity is a genuine ``SimpleDroneAdapter`` —
    the SAME numpy ``damage`` (clamp-at-0) / ``repair`` (clamp-at-1) arithmetic the flight adapter
    uses. The env calls :meth:`damage` on each UC-15 obstacle-contact edge event and :meth:`repair`
    on each docked step over a ``repairable`` pad — exactly as in real flight — so integrity is a
    *measured* claim about the numpy ledger, not a stub.
    """

    backend = "scripted"

    def __init__(
        self,
        positions,
        collided_flags,
        damage_cfg,
        *,
        dt=0.05,
        floor_z=0.0,
        ceiling_z=2.5,
        start_position=(0.0, 0.0, 1.0),
        attitudes=None,
    ):
        self._positions = [np.asarray(p, dtype=np.float64) for p in positions]
        self._collided = list(collided_flags)
        if attitudes is None:
            attitudes = [(0.0, 0.0, 0.0)] * len(positions)
        self._attitudes = [np.asarray(a, dtype=np.float64) for a in attitudes]
        self._i = 0
        # A real numpy adapter used ONLY as the integrity ledger (its integrated position is
        # ignored — positions are scripted). This is the numpy sim doing the damage/repair math.
        self._ledger = SimpleDroneAdapter(
            np.asarray(start_position, dtype=np.float64),
            floor_z=floor_z,
            ceiling_z=ceiling_z,
            dt=dt,
            damage=damage_cfg,
        )

    def _state(self, idx) -> DroneState:
        j = min(idx, len(self._positions) - 1)
        return DroneState(
            position=self._positions[j].copy(),
            velocity=np.zeros(3),
            attitude=self._attitudes[j].copy(),
            angular_velocity=np.zeros(3),
            collided=bool(self._collided[j]),
            integrity=self._ledger._integrity,
        )

    def reset(self, seed=None) -> DroneState:
        self._i = 0
        self._ledger.reset(seed=seed)
        return self._state(0)

    def step(self, action) -> DroneState:
        self._i += 1
        self._ledger.step(action)  # genuine numpy step for this frame
        return self._state(self._i)

    def damage(self, amount: float) -> float:
        return self._ledger.damage(amount)

    def repair(self, delta: float) -> float:
        return self._ledger.repair(delta)

    def close(self) -> None:  # pragma: no cover - trivial
        pass


# A tuned fixture: 0.3 integrity shed per contact; repair_rate 4.0 → +0.20 per docked step (dt=.05).
_DMG_CFG = DamageConfig(enabled=True, damage_per_contact=0.3, repair_rate=4.0)


def _damage_course(repairable: bool) -> CourseConfig:
    """A one-gate course with ONE on-path pillar (the damage source) at (2,0) and ONE floor pad at
    (4,0); ``repairable`` toggles that pad's flag."""
    return CourseConfig(
        start_position=(0.0, 0.0, 1.0),
        gates=(GateSpec(center=(6.0, 0.0, 1.0), aperture=0.6),),
        finish_x=8.0,
        floor_z=0.0,
        ceiling_z=2.5,
        obstacles=(ObstacleSpec(center=(2.0, 0.0), radius=0.5, height=2.5),),
        pads=(PadSpec(center=(4.0, 0.0), radius=0.6, repairable=repairable),),
    )


# Scripted trajectory: fly into the pillar (contact begins, sustained overlap) → exit → descend to
# the pad → dwell (repair) → take off. Contact is True only on the docked frames (7..10).
_DMG_POSITIONS = [
    (0.0, 0.0, 1.0),  # 0 spawn (reset)
    (1.0, 0.0, 1.0),  # 1 fly (airborne, no obstacle)
    (2.0, 0.0, 1.0),  # 2 enter the pillar — obstacle-contact edge fires (1 decrement)
    (2.3, 0.0, 1.0),  # 3 still overlapping (NO new edge — continuous overlap)
    (3.0, 0.0, 1.0),  # 4 exit the pillar (contact clears)
    (4.0, 0.0, 0.3),  # 5 airborne OVER the pad (no contact → AC4: no repair while hovering)
    (4.0, 0.0, 0.02),  # 6 slow final approach, still airborne
    (4.0, 0.0, 0.0),  # 7 DOCK — floor contact, descent 0.4<0.5, upright, over pad → repair
    (4.0, 0.0, 0.0),  # 8 dwell (repair)
    (4.0, 0.0, 0.0),  # 9 dwell
    (4.0, 0.0, 0.0),  # 10 dwell
    (4.0, 0.0, 0.3),  # 11 take off (airborne again)
]
_DMG_CONTACT = [False] * 7 + [True] * 4 + [False]
_DMG_DOCK_FRAMES = range(7, 11)  # step indices where the drone is docked on the pad


def _run_damage_trajectory(repairable: bool):
    """Replay the scripted trajectory on the integrity-ledger adapter; return per-step integrity
    (decoded from the width-1 damage obs = ``1 - integrity``), obstacle-contact edges, and dock
    flags."""
    env = _env_with(
        _DamageDockScriptedAdapter(_DMG_POSITIONS, _DMG_CONTACT, _DMG_CFG),
        EnvConfig(course=_damage_course(repairable), damage=_DMG_CFG),
    )
    obs, _info = env.reset(seed=0)
    integrity = [1.0 - float(obs[-1])]
    contacts = [False]
    docked = [False]
    for _i in range(1, len(_DMG_POSITIONS)):
        obs, _r, terminated, truncated, info = env.step(HOVER)
        integrity.append(1.0 - float(obs[-1]))
        contacts.append(bool(info["obstacle_contact"]))
        docked.append(bool(info["docked"]))
        if terminated or truncated:
            break
    return integrity, contacts, docked


# --- AC2: damage accrues from the UC-15 obstacle-contact edge event; clamps ≥ 0 ------------
def test_damage_accrues_on_obstacle_contact_edge_trigger() -> None:
    """AC2: integrity drops by ``damage_per_contact`` on the step the obstacle contact BEGINS, and
    a sustained overlap decrements ONCE (edge-triggered, not per overlapping frame)."""
    integrity, contacts, _docked = _run_damage_trajectory(repairable=True)
    # Exactly one edge over the sustained pillar overlap (frames 2..4).
    assert sum(1 for c in contacts if c) == 1
    edge_frame = contacts.index(True)
    assert integrity[edge_frame - 1] == pytest.approx(1.0)  # pristine right up to the contact
    assert integrity[edge_frame] == pytest.approx(1.0 - _DMG_CFG.damage_per_contact)
    # The overlapping-but-no-new-edge frames do not shed further integrity.
    assert integrity[edge_frame + 1] == pytest.approx(integrity[edge_frame])


def test_n_distinct_contacts_give_n_decrements() -> None:
    """AC2: N distinct obstacle contacts → N decrements (two separated pillars, path exits contact
    between them so each re-enters as a fresh edge)."""
    cfg = DamageConfig(enabled=True, damage_per_contact=0.2)
    course = CourseConfig(
        start_position=(0.0, 0.0, 1.0),
        gates=(GateSpec(center=(8.0, 0.0, 1.0), aperture=0.6),),
        finish_x=10.0,
        floor_z=0.0,
        ceiling_z=2.5,
        obstacles=(
            ObstacleSpec(center=(2.0, 0.0), radius=0.3, height=2.5),
            ObstacleSpec(center=(4.0, 0.0), radius=0.3, height=2.5),
        ),
    )
    positions = [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),  # enter pillar 1 → edge 1
        (2.6, 0.0, 1.0),  # exit pillar 1
        (3.4, 0.0, 1.0),  # gap (contact clears)
        (4.0, 0.0, 1.0),  # enter pillar 2 → edge 2
        (4.6, 0.0, 1.0),  # exit pillar 2
    ]
    env = _env_with(
        _DamageDockScriptedAdapter(positions, [False] * len(positions), cfg),
        EnvConfig(course=course, damage=cfg),
    )
    obs, _ = env.reset(seed=0)
    integrity = [1.0 - float(obs[-1])]
    edges = 0
    for _ in range(1, len(positions)):
        obs, _r, _t, _tr, info = env.step(HOVER)
        integrity.append(1.0 - float(obs[-1]))
        edges += int(info["obstacle_contact"])
    assert edges == 2  # two distinct contacts
    assert integrity[-1] == pytest.approx(1.0 - 2 * cfg.damage_per_contact)  # two decrements


def test_damage_clamps_at_zero_over_many_contacts() -> None:
    """AC2: integrity clamps at 0.0 — repeated contacts never push it negative (measured on the
    numpy ledger via the same repeated-contact course)."""
    cfg = DamageConfig(enabled=True, damage_per_contact=0.4)
    course = CourseConfig(
        start_position=(0.0, 0.0, 1.0),
        gates=(GateSpec(center=(12.0, 0.0, 1.0), aperture=0.6),),
        finish_x=14.0,
        floor_z=0.0,
        ceiling_z=2.5,
        obstacles=tuple(
            ObstacleSpec(center=(2.0 + 2.0 * k, 0.0), radius=0.3, height=2.5) for k in range(4)
        ),
    )
    positions = [(0.0, 0.0, 1.0)]
    for k in range(4):  # enter/exit each of the four pillars
        positions += [
            (2.0 + 2.0 * k, 0.0, 1.0),
            (2.6 + 2.0 * k, 0.0, 1.0),
            (3.4 + 2.0 * k, 0.0, 1.0),
        ]
    env = _env_with(
        _DamageDockScriptedAdapter(positions, [False] * len(positions), cfg),
        EnvConfig(course=course, damage=cfg),
    )
    obs, _ = env.reset(seed=0)
    integrity = [1.0 - float(obs[-1])]
    for _ in range(1, len(positions)):
        obs, _r, _t, _tr, _info = env.step(HOVER)
        integrity.append(1.0 - float(obs[-1]))
    # 4 contacts * 0.4 = 1.6 nominal loss, but integrity floors at 0.0.
    assert min(integrity) == pytest.approx(0.0)
    assert all(v >= -1e-9 for v in integrity)


# --- AC4: repair accrues ONLY while docked on a repairable pad; clamps at 1.0 --------------
def test_repair_accrues_only_while_docked_on_a_repairable_pad() -> None:
    """AC4: integrity is shed at the pillar, then RISES across the docked dwell on a repairable pad
    (each docked step nets a gain), toward 1.0."""
    integrity, _contacts, docked = _run_damage_trajectory(repairable=True)
    edge_frame = 2
    docked_lo = min(_DMG_DOCK_FRAMES)
    # Damaged before docking; every docked frame is flagged docked.
    assert integrity[edge_frame] < 1.0
    assert all(docked[i] for i in _DMG_DOCK_FRAMES)
    # Integrity strictly rises on the first docked step versus the pre-dock (damaged) level.
    assert integrity[docked_lo] > integrity[docked_lo - 1]


def test_repair_clamps_at_full_integrity_over_a_long_dwell() -> None:
    """AC4: a sustained dwell restores integrity toward 1.0 and never exceeds it (clamp)."""
    integrity, _contacts, _docked = _run_damage_trajectory(repairable=True)
    assert max(integrity) <= 1.0 + 1e-9
    # The dwell actually reached (clamped at) full integrity on this tuned fixture.
    assert max(integrity[i] for i in _DMG_DOCK_FRAMES) == pytest.approx(1.0)


def test_non_repair_pad_never_restores_even_while_docked() -> None:
    """AC4: docking on a plain (non-repair) pad never restores — integrity is monotone
    non-increasing across the whole trajectory (damage-only, exactly as if the pad were absent)."""
    integrity, _contacts, docked = _run_damage_trajectory(repairable=False)
    assert all(docked[i] for i in _DMG_DOCK_FRAMES)  # it still docks (the flag ≠ dock geometry)
    assert all(y <= x + 1e-12 for x, y in zip(integrity, integrity[1:], strict=False))


def test_hover_over_repair_pad_does_not_restore() -> None:
    """AC4: full landing required — frames 5,6 are airborne directly over the repair pad (no floor
    contact ⇒ not docked), so integrity does NOT rise while hovering; only the dwell repairs."""
    integrity, _contacts, docked = _run_damage_trajectory(repairable=True)
    assert docked[5] is False and docked[6] is False
    # Hovering over the pad keeps integrity at the post-damage level (no repair in the air).
    assert integrity[5] == pytest.approx(integrity[4])
    assert integrity[6] == pytest.approx(integrity[5])


# --- AC2: damage is reward-neutral and never terminates ------------------------------------
def test_damage_never_terminates_and_is_reward_neutral() -> None:
    """AC2: integrity is a sensory/handicap signal — it feeds NEITHER termination NOR the reward.

    Two envs replay the identical scripted flight through the pillar; one has damage enabled (so a
    contact sheds integrity), the other disabled. The reward at the contact step is identical (the
    UC-15 obstacle penalty is unchanged; the integrity loss adds nothing), and neither run
    terminates on the contact."""

    def stream(damage_enabled: bool):
        cfg = DamageConfig(enabled=damage_enabled, damage_per_contact=0.3)
        env = _env_with(
            _DamageDockScriptedAdapter(_DMG_POSITIONS, _DMG_CONTACT, cfg),
            EnvConfig(course=_damage_course(False), damage=cfg),
        )
        env.reset(seed=0)
        rewards, terms = [], []
        for _ in range(1, len(_DMG_POSITIONS)):
            _obs, r, terminated, _t, _info = env.step(HOVER)
            rewards.append(r)
            terms.append(terminated)
            if terminated:
                break
        return rewards, terms

    r_on, t_on = stream(True)
    r_off, t_off = stream(False)
    assert r_on == r_off, "integrity loss must not perturb the reward stream"
    assert not any(t_on), "an obstacle contact (damage) must never terminate the episode"
    assert not any(t_off)


# --- AC1: off-by-default byte-identity (a repair pad is inert when damage disabled) --------
def test_repairable_flag_is_inert_when_damage_disabled() -> None:
    """AC1: with damage DISABLED the whole damage/repair path is gated off, so a course carrying a
    repairable pad + an obstacle is byte-identical (obs / reward / termination / RNG) to the same
    course whose pad is plain — and the observation stays the locked 12-d contract (no damage
    block)."""

    def stream(repairable: bool):
        env = make_env(EnvConfig(course=_damage_course(repairable)), adapter="simple")
        rng = np.random.default_rng(4)
        actions = [rng.uniform([0, -1, -1, -1], [1, 1, 1, 1]) for _ in range(40)]
        obs, _ = env.reset(seed=0)
        trace = [obs.copy()]
        rewards, terms, truncs = [], [], []
        for a in actions:
            obs, r, terminated, truncated, _info = env.step(np.asarray(a, dtype=np.float32))
            trace.append(obs.copy())
            rewards.append(r)
            terms.append(terminated)
            truncs.append(truncated)
            if terminated or truncated:
                break
        return np.array(trace), rewards, terms, truncs, env.np_random.bit_generator.state

    on = stream(repairable=True)
    off = stream(repairable=False)
    assert on[0].shape[1] == OBS_DIM  # still the locked 12-d observation (damage off)
    np.testing.assert_array_equal(on[0], off[0])  # observation stream
    assert on[1] == off[1]  # reward stream
    assert on[2] == off[2] and on[3] == off[3]  # termination / truncation streams
    assert on[4] == off[4]  # identical RNG consumption (the flag draws nothing)


def test_repairable_pad_extends_the_step_budget() -> None:
    """UC-19 (step-budget allowance): the env grants ``repair_step_allowance`` extra steps per
    repairable pad on the active course (so a legitimate repair detour still finishes in time), and
    grants NONE when the pad is plain (byte-identical budget). Independent of the recharge
    allowance."""
    with_repair = make_env(
        EnvConfig(course=_damage_course(True), damage=_DMG_CFG), adapter="simple"
    )
    without = make_env(EnvConfig(course=_damage_course(False), damage=_DMG_CFG), adapter="simple")
    with_repair.reset(seed=0)
    without.reset(seed=0)
    episode = EpisodeConfig()
    base_budget = episode.max_steps + episode.steps_per_gate * (1 - 1)  # N=1 course
    assert without._max_steps == base_budget  # plain pad → no extra budget
    assert with_repair._max_steps == base_budget + episode.repair_step_allowance


# ===========================================================================
# UC-19 AC7 (top risk) — repair is LOAD-BEARING for an agility maneuver, and the
# repair-necessity course-variation axis (damage-heavy vs light).
# ===========================================================================
# The two-sided property lives in AGILITY (short-window / rapid-reversal) maneuvers: a degraded
# ``max_body_rate`` builds attitude more slowly, so in a short window it reaches LESS laterally.
# Sustained single-direction tilt saturates at the attitude limit and hides the difference, so the
# test scripts an open-loop AGILITY swing on the REAL numpy adapter and measures lateral reach. The
# "gate" is a documented lateral-reach threshold that a pristine (or repaired) drone clears and a
# degraded one misses — so REPAIR (what a pad does) is exactly what turns a miss into a pass.
_AC7_SWING = np.array([0.5, 1.0, 0.0, 0.0])  # hover throttle, full roll stick
_AC7_SWING_STEPS = 6  # short window (before the tilt saturates, where authority matters most)
_AC7_LATERAL_GATE = 0.15  # metres — sits strictly between the degraded and pristine reaches


def _agility_reach(mode: str) -> float:
    """Lateral reach (|y| after a short scripted roll swing) on the REAL numpy adapter, with
    integrity driven ONLY through the env-facing ``damage`` / ``repair`` hooks (as the env does)."""
    cfg = DamageConfig(enabled=True)  # DEFAULT tuning band (the plan's empirical claim)
    ad = SimpleDroneAdapter(
        np.array([0.0, 0.0, 5.0]), floor_z=0.0, ceiling_z=100.0, dt=0.05, damage=cfg
    )
    ad.reset(seed=0)
    if mode in ("degraded", "repaired"):
        for _ in range(3):  # three default contacts drive integrity to the 0.0 floor
            ad.damage(cfg.damage_per_contact)
    if mode == "repaired":
        for _ in range(500):  # a long pad dwell restores integrity to full
            ad.repair(cfg.repair_rate * 0.05)
    st = None
    for _ in range(_AC7_SWING_STEPS):
        st = ad.step(_AC7_SWING)
    return abs(float(st.position[1]))


def test_ac7_repair_is_load_bearing_for_an_agility_gate() -> None:
    """AC7 (load-bearing, two-sided, EMPIRICAL): under the DEFAULT ``DamageConfig`` a short agility
    swing clears a fixed lateral gate when PRISTINE, MISSES it when fully DEGRADED, and clears it
    again after a repair dwell. So repair is load-bearing: it is exactly what restores the maneuver.
    Measured on the real numpy adapter (open-loop scripted actions), integrity driven only through
    the env's ``damage`` / ``repair`` hooks.

    If this band ever collapses (no gate separates the two sides), that is an escalation trigger —
    NOT a silent weakening of the assert.
    """
    pristine = _agility_reach("pristine")
    degraded = _agility_reach("degraded")
    repaired = _agility_reach("repaired")
    # The band exists and is wide (the plan's ~3.6× measurement).
    assert pristine > degraded
    assert pristine / max(degraded, 1e-9) > 3.0
    # The gate is load-bearing: pristine/repaired clear it, degraded misses it.
    assert degraded < _AC7_LATERAL_GATE < pristine
    assert repaired > _AC7_LATERAL_GATE
    # Repair fully restores the maneuver (a full dwell returns to the pristine reach).
    assert repaired == pytest.approx(pristine)


def test_ac7_default_repair_course_damage_is_real_and_pad_restores() -> None:
    """AC7 (env integration on the exact fixture): on ``default_repair_course()`` with damage
    ENABLED, a real obstacle contact sheds integrity (damage is real on this course) and a dwell on
    the mid-corridor **repair** pad restores it — the pad is load-bearing in the env, not just at
    the adapter level. Explicit ``EnvConfig(damage=DamageConfig(enabled=True), course=...)``."""
    cfg = DamageConfig(enabled=True, repair_rate=4.0)  # default per-contact loss; brisk repair
    positions = [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.5, 0.0, 1.0),  # enter the first on-corridor pillar → damage
        (2.2, 0.0, 1.0),  # exit
        (3.0, 0.0, 0.3),  # descend toward the repair pad at (3.0, 0)
        (3.0, 0.0, 0.02),  # slow approach (airborne)
        (3.0, 0.0, 0.0),  # DOCK on the repair pad
        (3.0, 0.0, 0.0),  # dwell
        (3.0, 0.0, 0.0),  # dwell
        (3.0, 0.0, 0.0),  # dwell
        (3.0, 0.0, 0.0),  # dwell
    ]
    contact = [False] * 6 + [True] * 5
    env = _env_with(
        _DamageDockScriptedAdapter(positions, contact, cfg),
        EnvConfig(course=default_repair_course(), damage=cfg),
    )
    obs, _ = env.reset(seed=0)
    integrity = [1.0 - float(obs[-1])]
    for _ in range(1, len(positions)):
        obs, _r, term, trunc, _info = env.step(HOVER)
        integrity.append(1.0 - float(obs[-1]))
        if term or trunc:
            break
    assert min(integrity) < 1.0, "damage must be real on default_repair_course()"
    assert integrity[-1] > min(integrity), "the repair pad must restore integrity"
    assert integrity[-1] == pytest.approx(1.0)  # the dwell fully recovered


def test_ac7_light_course_completable_without_repair() -> None:
    """AC7 (the other side of the axis): a light course with NO damage source is completable without
    ever repairing — integrity stays pinned at 1.0 and the drone crosses the finish. Damage is
    ENABLED (mechanics are gated by ``damage.enabled``, not by the course), so this proves the
    course-variation axis: repair-necessity depends on obstacle density, not on the flag."""
    cfg = DamageConfig(enabled=True)
    course = single_repair_pad_course(
        pad_center=(0.0, 0.0), gate_center=(3.0, 0.0, 1.0), finish_x=6.0
    )
    positions = [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),
        (3.0, 0.0, 1.0),  # pass the gate
        (4.0, 0.0, 1.0),
        (5.0, 0.0, 1.0),
        (6.1, 0.0, 1.0),  # cross the finish plane
    ]
    env = _env_with(
        _DamageDockScriptedAdapter(positions, [False] * len(positions), cfg),
        EnvConfig(course=course, damage=cfg),
    )
    obs, _ = env.reset(seed=0)
    integrity = [1.0 - float(obs[-1])]
    completed = False
    for _ in range(1, len(positions)):
        obs, _r, terminated, truncated, info = env.step(HOVER)
        integrity.append(1.0 - float(obs[-1]))
        if info.get("completed"):
            completed = True
        if terminated or truncated:
            break
    assert completed is True, "a light (no-obstacle) course completes without any repair"
    assert all(v == pytest.approx(1.0) for v in integrity), "no damage source ⇒ integrity stays 1.0"


# ===========================================================================
# UC-25 — grounded / no-progress early termination (AC1–AC7)
# ===========================================================================
# Each detector needs a full window of consecutive qualifying steps to fire. UC-36 split the two
# windows: the no-progress (stuck) detector keeps ``stuck_window`` (default 100), while the grounded
# detector now has its own shorter ``grounded_window`` (default 10). The committed golden fixtures
# (baseline 24, reproducibility 50, dynamics 30 steps) are untouched by BOTH defaults — the
# no-progress window exceeds every fixture length (AC6), and the fixtures never enter the grounded
# state, so the short grounded window cannot fire in them either. The tests below drive the rule
# with small windows so the scripted trajectories stay short and readable (challenger's non-blocking
# note). Scripted adapters hardcode ``velocity=zeros`` — perfect for the grounded/stuck cuts, but
# the low-but-progressing case (AC5) MUST use the REAL ``simple`` adapter, whose flight velocity is
# far above the rest band.


def test_uc25_grounded_episode_is_cut_as_a_crash() -> None:
    """AC1: a drone that falls and rests ~8 mm above the floor at zero speed is cut as a floor
    crash well under the step budget (grounded detector), not run to ``_max_steps``."""
    # Fall from spawn, then rest inside the floor band (z=0.008 < floor_epsilon=0.05) at zero speed.
    positions = [(0.0, 0.0, 1.0), (0.0, 0.0, 0.3)] + [(0.0, 0.0, 0.008)] * 8
    # UC-36: the grounded detector now has its OWN window; match it to the legacy UC-25 value of 4
    # so this test's timing is preserved. ``stuck_window`` is set equally but is irrelevant here —
    # a grounded drone fires on ``grounded_window`` first (grounded priority).
    env = _env_with(
        _ScriptedAdapter(positions),
        EnvConfig(early_termination=EarlyTerminationConfig(stuck_window=4, grounded_window=4)),
    )
    env.reset()
    terminated = truncated = False
    info: dict = {}
    steps = 0
    for _ in range(env._max_steps + 1):
        _obs, _r, terminated, truncated, info = env.step(HOVER)
        steps += 1
        if terminated or truncated:
            break
    assert terminated is True, "a grounded episode must terminate (crash), not truncate"
    assert truncated is False, "it ends promptly via the crash path, not at the budget"
    assert info["collided"] is True, "a grounded cut is reported as a crash collision (AC3)"
    assert info["completed"] is False, "resting on the floor is not a course success"
    assert info["early_termination"] == "grounded"
    # Warm-up: it takes a full window of grounded steps (spawn + fall are not grounded).
    assert steps == 5, "cut fires exactly on the 4th consecutive grounded step (window=4)"
    assert steps < env._max_steps, "the cut is well under the full step budget"


def test_uc25_no_progress_episode_is_cut_as_a_crash() -> None:
    """AC2: an airborne drone that hovers in place — never reducing its distance to the target
    gate for a full window — is cut as a crash by the no-progress (stuck) detector."""
    # Airborne hover (z=1.0, above the floor band) that never closes on gate g0.
    positions = [(0.0, 0.0, 1.0)] + [(0.5, 0.0, 1.0)] * 10
    env = _env_with(
        _ScriptedAdapter(positions),
        EnvConfig(early_termination=EarlyTerminationConfig(stuck_window=4)),
    )
    env.reset()
    terminated = truncated = False
    info: dict = {}
    for _ in range(env._max_steps + 1):
        _obs, _r, terminated, truncated, info = env.step(HOVER)
        if terminated or truncated:
            break
    assert terminated is True and truncated is False
    assert info["collided"] is True, "a stuck cut is reported as a crash collision (AC3)"
    assert info["early_termination"] == "stuck"
    assert info["completed"] is False


def test_uc25_closing_path_is_not_cut() -> None:
    """AC2 (negative): a drone that keeps closing the gap to each gate completes the course and is
    never cut by the stuck detector, even under an aggressively small window."""
    env = _env_with(
        _ScriptedAdapter(_THREE_GATE_PATH),
        EnvConfig(early_termination=EarlyTerminationConfig(stuck_window=4)),
    )
    env.reset()
    terminated = truncated = False
    info: dict = {}
    reasons = []
    for _ in range(len(_THREE_GATE_PATH) + 2):
        _obs, _r, terminated, truncated, info = env.step(HOVER)
        reasons.append(info["early_termination"])
        if terminated or truncated:
            break
    assert info["completed"] is True, "a monotonically-closing path completes the course"
    assert info["collided"] is False, "a completion is not a crash"
    assert all(r is None for r in reasons), "a progressing flight is never cut early"


def test_uc25_grounded_cut_applies_the_collision_penalty() -> None:
    """AC3: a grounded/stuck cut earns the EXISTING collision penalty in the reward, exactly like a
    floor/ceiling crash — the cut step's reward is dominated by ``collision_penalty`` (100)."""
    positions = [(0.0, 0.0, 1.0)] + [(0.0, 0.0, 0.008)] * 8
    # UC-36: match the grounded window to the legacy value of 3 (grounded fires first anyway).
    cfg = EnvConfig(early_termination=EarlyTerminationConfig(stuck_window=3, grounded_window=3))
    env = _env_with(_ScriptedAdapter(positions), cfg)
    env.reset()
    reward = 0.0
    terminated = False
    for _ in range(env._max_steps + 1):
        _obs, reward, terminated, _tr, info = env.step(HOVER)
        if terminated:
            break
    assert terminated is True and info["collided"] is True
    # The default collision_penalty is 100; the cut step's reward must reflect that big negative.
    assert reward <= -cfg.reward.collision_penalty + 1.0, (
        "the cut step must eat the collision penalty (AC3)"
    )


def test_uc25_docked_exemption_is_two_sided_and_non_vacuous() -> None:
    """AC4: a drone on a rechargeable pad is NOT cut while service is productive (battery rising)
    across MORE than a full window of docked steps — but once the battery clamps at 1.0 the dwell
    is no longer productive and the stuck detector cuts it after the window. One run, both sides.

    This is the non-loophole test: mere presence on a pad does not exempt (the drone stays docked
    the whole time); only *productive* service keeps it alive. It FAILS if the exemption is keyed on
    docking rather than on charge/integrity actually improving.
    """
    # drain 0.10/airborne-step, net +0.10/docked-step (recharge 0.20 gross − 0.10 drain).
    battery = BatteryConfig(enabled=True, idle_rate=2.0, throttle_rate=0.0, recharge_rate=4.0)
    # Airborne approach draining to ~0.40, a valid slow dock, then a long dwell.
    positions = [
        (0.0, 0.0, 1.0),
        (0.4, 0.0, 1.0),
        (0.8, 0.0, 1.0),
        (1.2, 0.0, 0.9),
        (1.6, 0.0, 0.7),
        (2.0, 0.0, 0.4),
        (2.0, 0.0, 0.02),  # final slow approach (airborne, over the pad)
    ] + [(2.0, 0.0, 0.0)] * 12  # DOCK + long dwell on the pad
    contact = [False] * 7 + [True] * 12
    cfg = EnvConfig(
        course=_recharge_course(True),
        battery=battery,
        early_termination=EarlyTerminationConfig(stuck_window=4),
    )
    env = _env_with(_BatteryDockScriptedAdapter(positions, contact, battery), cfg)
    obs, _ = env.reset(seed=0)

    docked_frames = 0
    rising_docked_alive = 0
    prev_charge = 1.0 - float(obs[-1])
    terminated = False
    cut_info: dict = {}
    for _ in range(1, len(positions)):
        obs, _r, terminated, _tr, info = env.step(HOVER)
        charge = 1.0 - float(obs[-1])
        if info["docked"]:
            docked_frames += 1
            # While docked AND charging, the episode must stay alive (productive exemption).
            if charge > prev_charge + 1e-9:
                rising_docked_alive += 1
                assert terminated is False, "a productive docked dwell must not be cut (AC4)"
        prev_charge = charge
        if terminated:
            cut_info = info
            break

    assert rising_docked_alive > 4, (
        "the productive dwell must outlast a full window (non-vacuous alive side)"
    )
    assert docked_frames > 4, "the drone stays docked for more than a window before the cut"
    assert terminated is True, "once the battery clamps, the idle dock is cut (loophole closed)"
    assert cut_info["collided"] is True
    assert cut_info["docked"] is True, "the cut happens while still docked — it is the idle-cut"
    assert cut_info["early_termination"] == "stuck", "the idle-after-full cut is a stuck cut"


def test_uc25_normal_flight_is_byte_identical_to_rule_disabled() -> None:
    """AC5/AC6: default (rule ON) per-step outcomes on a real-adapter flight are byte-identical to
    the rule explicitly OFF — a normal flight never accumulates a window, so nothing fires and obs/
    reward/termination/RNG are untouched (warm-up guard + no-spurious-firing)."""
    golden = np.load(_BASELINE_ROLLOUT)
    actions = list(golden["actions"])
    on = _full_stream(EnvConfig(), seed=42, actions=actions)
    off = _full_stream(
        EnvConfig(early_termination=EarlyTerminationConfig(enabled=False)),
        seed=42,
        actions=actions,
    )
    np.testing.assert_array_equal(on[0], off[0])  # observation stream
    assert on[1] == off[1]  # reward stream
    assert on[2] == off[2]  # terminated stream
    assert on[3] == off[3]  # truncated stream
    assert on[4] == off[4]  # final RNG state


def test_uc25_committed_baseline_still_matches_with_no_regen() -> None:
    """AC6: with the rule ON by default, the committed seed-42 golden rollout still reproduces
    byte-for-byte and the obs width stays 12 — no fixture regeneration, no observation-schema/
    checkpoint impact. Neither detector fires: stuck_window=100 exceeds the 24-step baseline, and
    the default grounded_window=10 cannot fire either because the never-grounded baseline never
    enters the floor band (UC-36)."""
    golden = np.load(_BASELINE_ROLLOUT)
    actions = list(golden["actions"])
    obs_trace, _r, _t, _tr, _rng = _full_stream(EnvConfig(), seed=42, actions=actions)
    assert obs_trace.shape == golden["env_trace"].shape
    np.testing.assert_array_equal(obs_trace, golden["env_trace"])
    assert make_env(EnvConfig(), adapter="simple").obs_width == OBS_DIM == 12


def test_uc25_low_but_progressing_real_flight_is_not_cut() -> None:
    """AC5: a low-but-progressing flight on the REAL ``simple`` adapter is NOT cut. The drone dips
    into the floor band while moving fast toward a low gate; the grounded detector's speed guard
    (real velocity ≫ rest_speed_epsilon) keeps it from being mis-cut, and its steady progress keeps
    the stuck counter at zero. Uses the real adapter — scripted adapters hardcode velocity=zeros."""
    course = CourseConfig(
        start_position=(0.0, 0.0, 0.3),
        gates=(GateSpec(center=(3.0, 0.0, 0.2), aperture=0.6),),
        finish_x=5.0,
        floor_z=0.0,
        ceiling_z=2.5,
    )
    cfg = EnvConfig(course=course, early_termination=EarlyTerminationConfig(stuck_window=4))
    env = make_env(cfg, adapter="simple")
    env.reset(seed=0)
    fe = cfg.early_termination.floor_epsilon
    rse = cfg.early_termination.rest_speed_epsilon
    saw_low_and_moving = False
    reasons = []
    forward = np.array([0.52, 0.0, 0.4, 0.0], dtype=np.float32)  # pitch forward, ~hover throttle
    for _ in range(30):
        _obs, _r, terminated, truncated, info = env.step(forward)
        z = float(info["position"][2])
        speed = float(np.linalg.norm(env.adapter._velocity))
        reasons.append(info["early_termination"])
        if not terminated and z <= course.floor_z + fe and speed > rse:
            saw_low_and_moving = True
        if terminated or truncated:
            # If it ends, it is a REAL floor collision, never a grounded/stuck cut.
            assert info["early_termination"] is None
            break
    assert saw_low_and_moving, "the flight must actually enter the floor band while moving fast"
    assert all(r is None for r in reasons), "a low-but-progressing real flight is never cut early"


def test_uc25_config_defaults_are_named_and_documented() -> None:
    """AC7: the early-termination constants are named config fields with the documented defaults,
    and ``EnvConfig()`` keeps ``early_termination`` ON by default (the fix)."""
    et = EarlyTerminationConfig()
    assert et.floor_epsilon == 0.05
    assert et.stuck_window == 100
    assert et.grounded_window == 10  # UC-36: grounded detector's own shorter window (0.5 s @ 20 Hz)
    assert et.progress_epsilon == 0.01
    assert et.rest_speed_epsilon == 0.05
    assert et.enabled is True
    assert EnvConfig().early_termination == et, "early_termination is appended with all defaults"


def test_uc36_grounded_window_tunes_cut_timing() -> None:
    """AC3/UC-36: shrinking ``grounded_window`` cuts a grounded episode sooner — the grounded
    detector's own window (independent of ``stuck_window``) is a live, tunable knob on the cut
    timing. (Was UC-25's ``stuck_window`` tuning test; UC-36 gave the grounded detector its own
    window, so the grounded cut is now driven by ``grounded_window``.)"""
    positions = [(0.0, 0.0, 1.0)] + [(0.0, 0.0, 0.008)] * 20

    def cut_step(window: int) -> int:
        env = _env_with(
            _ScriptedAdapter(positions),
            # A large ``stuck_window`` proves the cut is driven by ``grounded_window`` alone.
            EnvConfig(
                early_termination=EarlyTerminationConfig(grounded_window=window, stuck_window=100)
            ),
        )
        env.reset()
        for i in range(1, 25):
            _obs, _r, terminated, truncated, info = env.step(HOVER)
            if terminated or truncated:
                assert info["early_termination"] == "grounded"
                return i
        raise AssertionError("expected a grounded cut")

    assert cut_step(2) == 2
    assert cut_step(6) == 6
    assert cut_step(2) < cut_step(6), "a smaller grounded window cuts sooner (tunable)"


def test_uc36_stuck_window_tunes_no_progress_timing() -> None:
    """AC3/UC-36: the no-progress (stuck) detector still has its own independently-tunable window.
    An airborne drone hovering statically above the floor band (never grounded) is cut purely by
    ``stuck_window``; shrinking it cuts sooner. This keeps the no-progress tuning coverage that the
    converted grounded test above no longer provides."""
    # Airborne static hover at z=1.0 (well above the floor band), never moving, never closing on the
    # gate. ``_best_dist`` starts at +inf, so the very first step always registers "progress" (the
    # warm-up guard) and the no-progress counter only starts climbing on step 2 ⇒ the cut lands on
    # step ``window + 1``.
    positions = [(0.0, 0.0, 1.0)] * 21

    def cut_step(window: int) -> int:
        env = _env_with(
            _ScriptedAdapter(positions),
            # A large ``grounded_window`` (irrelevant — the drone is airborne) proves the cut is
            # driven by ``stuck_window`` alone.
            EnvConfig(
                early_termination=EarlyTerminationConfig(stuck_window=window, grounded_window=100)
            ),
        )
        env.reset()
        for i in range(1, 25):
            _obs, _r, terminated, truncated, info = env.step(HOVER)
            if terminated or truncated:
                assert info["early_termination"] == "stuck"
                return i
        raise AssertionError("expected a stuck cut")

    assert cut_step(2) == 3  # warm-up step + 2 no-progress steps
    assert cut_step(6) == 7  # warm-up step + 6 no-progress steps
    assert cut_step(2) < cut_step(6), "a smaller stuck window cuts sooner (tunable)"


def test_uc25_floor_epsilon_tunes_the_grounded_band() -> None:
    """AC7: the floor-epsilon band is tunable — a drone resting at z=0.1 is OUTSIDE the default
    0.05 band (so it is not 'grounded', only eventually 'stuck'), but WITHIN a widened 0.2 band (so
    it is classified 'grounded')."""
    positions = [(0.0, 0.0, 1.0)] + [(0.0, 0.0, 0.1)] * 20

    def cut_reason(floor_epsilon: float) -> str:
        env = _env_with(
            _ScriptedAdapter(positions),
            EnvConfig(
                # UC-36: keep grounded and stuck windows equal (=3) so the classification (grounded
                # vs. stuck) — not a window-length race — is what the widened band changes.
                early_termination=EarlyTerminationConfig(
                    stuck_window=3, grounded_window=3, floor_epsilon=floor_epsilon
                )
            ),
        )
        env.reset()
        for _ in range(25):
            _obs, _r, terminated, truncated, info = env.step(HOVER)
            if terminated or truncated:
                return info["early_termination"]
        raise AssertionError("expected an early cut")

    assert cut_reason(0.05) == "stuck", "z=0.1 is above the default band ⇒ not grounded"
    assert cut_reason(0.2) == "grounded", "a widened band captures z=0.1 as grounded"


# ===========================================================================
# UC-36 — grounded early termination reaches the recording / eval path
# ===========================================================================
# UC-36 extends UC-25. The bug it fixes: the grounded detector reused ``stuck_window`` (100) as its
# firing window, so a drone that floors at ~frame 12 would only be cut at ~frame 112 — past the
# ~101-frame recording horizon — leaving a recording that is ~87 % dead drone on the floor. Giving
# the grounded detector its own short ``grounded_window`` (default 10) cuts the recording near the
# crash. The recording/eval loop already honours ``terminated`` (``record_rollout`` loops on
# ``while not (terminated or truncated)``), so no loop change is needed; the window shrink suffices.

# The recording drone floors at this frame and stays there (matches the reported failure recording).
_UC36_FLOOR_FRAME = 12
_UC36_DEFAULT_GROUNDED_WINDOW = 10  # EarlyTerminationConfig.grounded_window default


def test_uc36_recording_is_bounded_near_crash_not_horizon(connectome, tmp_path: Path) -> None:
    """AC2: a recorded episode of a drone that floors at ~frame 12 ends near the crash (frame 12 +
    ``grounded_window``), NOT at the ~400-step horizon. End-to-end through the REAL recording
    driver (:func:`record_rollout`) on the hermetic numpy stack — the same path the failing Drive
    recording used. Proves the fix reaches recording, not just training."""
    import json

    import torch  # noqa: F401  (imported for parity with the record stack; kept local + hermetic)

    from drone_fly.controller.actor import ConnectomeActorNetwork
    from drone_fly.record.recorder import ActivationRecorder
    from drone_fly.record.rollout import record_rollout

    # Fall straight down from z=1.0 (frames 0..11, all above the 0.05 floor band), then rest on the
    # floor from frame 12 onward (the scripted adapter clamps to its last position). Velocity is
    # zeros ⇒ the grounded speed guard is satisfied ⇒ the grounded detector arms at frame 12.
    descend = [(0.0, 0.0, round(1.0 - 0.08 * k, 4)) for k in range(_UC36_FLOOR_FRAME)]
    positions = descend + [(0.0, 0.0, 0.008)]
    assert positions[_UC36_FLOOR_FRAME - 1][2] > 0.05  # last airborne frame is above the band
    assert positions[_UC36_FLOOR_FRAME][2] <= 0.05  # floored from frame 12

    # Default EnvConfig ⇒ grounded_window=10, and a large (default 400) horizon to fall back to.
    env = _env_with(_ScriptedAdapter(positions), EnvConfig())
    horizon = env._max_steps
    assert horizon >= 100, "the horizon must dwarf the crash frame for the assertion to matter"

    torch.manual_seed(0)
    actor = ConnectomeActorNetwork(connectome)  # raw actor; the scripted adapter ignores its action
    recorder = ActivationRecorder(connectome, tmp_path / "act", backend="simple", dt=0.05)
    written = record_rollout(actor, env, recorder, n_episodes=1, seed=0, record_every=1)

    assert len(written) == 1
    doc = json.loads(written[0].read_text())
    steps = doc["outcome"]["steps"]
    n_frames = doc["meta"]["n_frames"]

    assert n_frames == steps, "one captured activation frame per env step"
    assert doc["outcome"]["completed"] is False, "a drone resting on the floor never completes"
    # Bounded near the crash: the episode is cut a full grounded_window after flooring, plus a tiny
    # slack — emphatically NOT run out to the horizon (the ~101-frame bug the UC reported).
    assert steps >= _UC36_FLOOR_FRAME, "the cut cannot precede the drone reaching the floor"
    assert steps <= _UC36_FLOOR_FRAME + _UC36_DEFAULT_GROUNDED_WINDOW + 2, (
        "the recording must be bounded near crash + grounded_window (0.5 s grace), not the horizon"
    )
    assert steps < horizon, "the recorded episode is cut well before the full step budget"


def test_uc36_healthy_flight_is_not_clipped_by_short_grounded_window(connectome) -> None:
    """AC5: the new short default ``grounded_window=10`` does NOT clip a legitimately-flying
    episode. A real-adapter healthy flight (the committed seed-42 golden actions) is never cut by
    the grounded/no-progress detector, and it ends at exactly the same step — via the same
    natural event — as the identical flight with the rule turned OFF. The short grounded window
    changes nothing about a healthy episode's length."""
    golden = np.load(_BASELINE_ROLLOUT)
    actions = list(golden["actions"])

    def run(config):
        env = make_env(config, adapter="simple")
        env.reset(seed=42)
        reasons, steps = [], 0
        terminated = truncated = False
        for a in actions:
            _obs, _r, terminated, truncated, info = env.step(np.asarray(a, dtype=np.float32))
            reasons.append(info["early_termination"])
            steps += 1
            if terminated or truncated:
                break
        return steps, terminated, truncated, reasons

    # Rule ON with the UC-36 default grounded_window=10; real ``simple`` adapter (real velocities).
    on_steps, on_term, on_trunc, on_reasons = run(EnvConfig())
    # Rule fully OFF — the reference "before UC-25/36" behaviour.
    off_steps, off_term, off_trunc, _ = run(
        EnvConfig(early_termination=EarlyTerminationConfig(enabled=False))
    )

    assert all(r is None for r in on_reasons), (
        "a healthy flight is never cut by the grounded/no-progress detector (short window and all)"
    )
    assert (on_steps, on_term, on_trunc) == (off_steps, off_term, off_trunc), (
        "the short grounded window does not shorten or alter the end of a healthy episode"
    )


def test_uc36_short_grounded_window_preserves_baseline_byte_identity(connectome) -> None:
    """AC6: the UC-36 default ``grounded_window=10`` is byte-identical to the legacy long window on
    the committed baseline — the shorter grounded window introduced by UC-36 perturbs nothing. The
    seed-42 golden flight never enters the grounded state, so shrinking the grounded window from 100
    to 10 leaves the obs / reward / termination / RNG streams bit-for-bit unchanged (no fixture
    regeneration, no checkpoint/schema impact)."""
    golden = np.load(_BASELINE_ROLLOUT)
    actions = list(golden["actions"])

    short = _full_stream(EnvConfig(), seed=42, actions=actions)  # UC-36 default grounded_window=10
    legacy = _full_stream(
        EnvConfig(early_termination=EarlyTerminationConfig(grounded_window=100)),
        seed=42,
        actions=actions,
    )
    np.testing.assert_array_equal(short[0], legacy[0])  # observation stream
    assert short[1] == legacy[1]  # reward stream
    assert short[2] == legacy[2]  # terminated stream
    assert short[3] == legacy[3]  # truncated stream
    assert short[4] == legacy[4]  # final RNG state
    # And it still matches the committed golden fixture exactly.
    np.testing.assert_array_equal(short[0], golden["env_trace"])
