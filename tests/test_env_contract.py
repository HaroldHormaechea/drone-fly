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
    DockConfig,
    EpisodeConfig,
    GateSpec,
    ObstacleSpec,
    ObstacleVisionConfig,
    RandomizationConfig,
    default_obstacle_course,
    default_pad_course,
    single_gate_course,
    single_pad_course,
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
