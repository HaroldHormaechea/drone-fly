"""AC1/AC11/AC12 — RaceEnv contract: spaces, termination modes, fixed dynamics, seeding.

All tests use ``adapter="simple"`` so they are hermetic (no pybullet). Completion
termination is exercised with a small **scripted adapter** that walks the drone through a
hand-built start→gate→finish trajectory, so we can assert the "terminate on course
completion" branch deterministically without relying on an untrained policy to actually
fly the course.
"""

from __future__ import annotations

import numpy as np

from drone_fly.adapter.base import DroneState
from drone_fly.controller.encoding import ACTION_DIM, OBS_DIM
from drone_fly.env import EnvConfig, make_env
from drone_fly.env.config import EpisodeConfig

HOVER = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)


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
