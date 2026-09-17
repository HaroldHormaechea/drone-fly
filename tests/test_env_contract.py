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

from drone_fly.adapter.base import DroneState
from drone_fly.adapter.simple import SimpleDroneAdapter
from drone_fly.controller.encoding import ACTION_DIM, OBS_DIM
from drone_fly.env import EnvConfig, make_env
from drone_fly.env.config import EpisodeConfig, RandomizationConfig
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
def test_terminates_on_course_completion() -> None:
    # Scripted walk: start → through gate (x=3) → through finish (x=6).
    path = [(0, 0, 1), (2.9, 0, 1), (3.1, 0, 1), (5.9, 0, 1), (6.1, 0, 1)]
    env = _env_with(_ScriptedAdapter(path))
    env.reset()
    terminated = False
    info = {}
    for _ in range(len(path)):
        _obs, _r, terminated, truncated, info = env.step(HOVER)
        if terminated or truncated:
            break
    assert terminated is True
    assert info["completed"] is True
    assert info["is_success"] is True
    assert info["completion_time"] is not None


def test_finish_before_gate_does_not_complete() -> None:
    # Fly the whole trajectory off-axis (y=2.0, well outside the 0.6 aperture) so the gate
    # is NEVER validly passed, yet the drone still forward-crosses both the gate plane and
    # the finish plane. Crossing the finish while still TO_GATE must not complete-as-success.
    path = [(0, 2, 1), (3.1, 2, 1), (6.1, 2, 1), (7.0, 2, 1)]
    env = _env_with(_ScriptedAdapter(path))
    env.reset()
    completed_any = False
    for _ in range(len(path)):
        _obs, _r, terminated, truncated, info = env.step(HOVER)
        completed_any = completed_any or info["completed"]
        if terminated or truncated:
            break
    assert completed_any is False


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
    cfg = EnvConfig(episode=EpisodeConfig(dt=0.05, max_steps=8))
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


# --- AC7: byte-identity vs a pre-UC-08 baseline rollout ------------------------------
def test_randomization_off_is_byte_identical_to_pre_uc08_baseline() -> None:
    """With both axes off, the env reproduces a fixed-seed PRE-UC-08 rollout bit-for-bit.

    The golden trace in ``fixtures/uc08_baseline_rollout.npz`` was captured by running the
    fixed-course env at the commit **before** UC-08 (seed=42, the stored action script).
    Any RNG draw or dynamics change leaking into the disabled path would break this. This is
    the load-bearing AC7 guarantee (the challenger's explicit verification hook).
    """
    golden = np.load(_BASELINE_ROLLOUT)
    actions = list(golden["actions"])

    # EnvConfig() defaults have randomization OFF on both axes.
    trace = _rollout_obs(EnvConfig(), seed=42, actions=actions)

    assert trace.shape == golden["env_trace"].shape
    np.testing.assert_array_equal(trace, golden["env_trace"])


def test_explicit_disabled_randomization_matches_default() -> None:
    """An explicitly-disabled RandomizationConfig is identical to the default (AC7)."""
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
