"""``RaceEnv`` — the Gymnasium start→gate→finish racing environment (AC1, AC11, AC12).

Wires the sim-agnostic adapter (:mod:`drone_fly.adapter`), the pure geometry phase machine
(:mod:`drone_fly.env.geometry`), and the pure reward (:mod:`drone_fly.env.reward`) into a
single ``gymnasium.Env`` the connectome PPO policy trains against.

Observation / action contract (locked to UC-01/UC-02)
-----------------------------------------------------
* **Observation** — ``Box`` of shape ``(OBS_DIM,)`` == ``(12,)``: relative next-waypoint
  pose ``(3)`` + attitude ``(3)`` + linear velocity ``(3)`` + angular velocity ``(3)``.
  This matches the fixed 12-d contract the connectome actor consumes.
* **Action** — ``Box(low=[0,-1,-1,-1], high=[1,1,1,1])`` == the canonical CTBR channels
  ``(THROTTLE, ROLL, PITCH, YAW)``. PPO's Gaussian head is unbounded; the adapter clips.

Termination (AC1)
-----------------
An episode ``terminated`` on a valid course completion (gate passed **and** finish
crossed) or a floor/ceiling collision; it is ``truncated`` on timeout (``max_steps``).

Determinism (AC12)
------------------
With :class:`~drone_fly.adapter.simple.SimpleDroneAdapter` the dynamics are fixed (AC11)
and noise-free, so a fixed ``seed`` yields a bit-identical episode for a given policy.
"""

from __future__ import annotations

import logging

import gymnasium as gym
import numpy as np

from drone_fly.adapter import make_adapter
from drone_fly.controller.encoding import ACTION_DIM, OBS_DIM
from drone_fly.env.config import EnvConfig
from drone_fly.env.geometry import TO_GATE, advance_phase, target_position
from drone_fly.env.reward import compute_reward

logger = logging.getLogger(__name__)


class RaceEnv(gym.Env):
    """Single-drone start→gate→finish racing environment.

    Parameters
    ----------
    config:
        Course / reward / episode configuration. Defaults to :class:`EnvConfig`.
    adapter:
        Backend selector passed to :func:`drone_fly.adapter.make_adapter`
        (``"auto"`` | ``"simple"`` | ``"pybullet"``).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None, *, adapter: str = "auto") -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self._adapter_choice = adapter
        course = self.config.course

        self.adapter = make_adapter(
            adapter,
            course.start,
            floor_z=course.floor_z,
            ceiling_z=course.ceiling_z,
            dt=self.config.episode.dt,
        )
        self.backend = self.adapter.backend

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

        self._phase = TO_GATE
        self._prev_pos = course.start.copy()
        self._step_count = 0

    # -- observation encoding -----------------------------------------------------------
    def _observation(self, state) -> np.ndarray:
        target = target_position(self._phase, self.config.course)
        rel = target - state.position
        obs = np.concatenate([rel, state.attitude, state.velocity, state.angular_velocity]).astype(
            np.float32
        )
        # Belt-and-braces: the env never emits a non-finite observation to the policy.
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

    def _dist_to_target(self, position: np.ndarray) -> float:
        target = target_position(self._phase, self.config.course)
        return float(np.linalg.norm(target - position))

    # -- gymnasium API ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        state = self.adapter.reset(seed=seed)
        self._phase = TO_GATE
        self._prev_pos = state.position.copy()
        self._step_count = 0
        info = {"phase": self._phase, "backend": self.backend}
        return self._observation(state), info

    def step(self, action):
        course = self.config.course
        dist_prev = self._dist_to_target(self._prev_pos)

        state = self.adapter.step(np.asarray(action, dtype=np.float64))
        self._step_count += 1

        new_phase, event = advance_phase(self._phase, self._prev_pos, state.position, course)
        self._phase = new_phase
        completed = event == "finish"

        # Distance to the (possibly newly-advanced) target, for the progress term.
        dist_curr = self._dist_to_target(state.position)

        reward = compute_reward(
            dist_to_target_prev=dist_prev,
            dist_to_target_curr=dist_curr,
            event=event,
            collided=state.collided,
            completed=completed,
            cfg=self.config.reward,
        )

        terminated = bool(completed or state.collided)
        truncated = bool(not terminated and self._step_count >= self.config.episode.max_steps)

        info = {
            "phase": self._phase,
            "backend": self.backend,
            "event": event,
            "collided": bool(state.collided),
            "completed": bool(completed),
            "steps": self._step_count,
            # World-frame drone position this step (UC-05 recording draws the flight path).
            # Additive key; existing tests assert membership, so this stays back-compatible.
            "position": state.position.copy(),
        }
        if terminated or truncated:
            info["completion_time"] = (
                self._step_count * self.config.episode.dt if completed else None
            )
            info["is_success"] = bool(completed)

        self._prev_pos = state.position.copy()
        return self._observation(state), float(reward), terminated, truncated, info

    def close(self) -> None:
        self.adapter.close()


def make_env(config: EnvConfig | None = None, *, adapter: str = "auto") -> RaceEnv:
    """Factory for a single :class:`RaceEnv` (AC1)."""
    return RaceEnv(config, adapter=adapter)


def build_vec_env(
    *,
    config: EnvConfig | None = None,
    adapter: str = "auto",
    n_envs: int = 1,
    seed: int | None = None,
    training: bool = True,
    norm_reward: bool | None = None,
    vecnormalize_path: str | None = None,
):
    """Build a ``VecNormalize``-wrapped vectorised env for SB3 (AC4/AC6).

    Parameters
    ----------
    training:
        ``True`` for training (VecNormalize updates its running stats and normalises
        reward); ``False`` for evaluation (frozen stats, raw reward).
    norm_reward:
        Override reward normalisation; defaults to ``training``.
    vecnormalize_path:
        If given, load saved VecNormalize stats from this path (checkpoint resume / eval)
        instead of starting fresh.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cfg = config or EnvConfig()
    norm_reward = training if norm_reward is None else norm_reward

    def _factory():
        return make_env(cfg, adapter=adapter)

    venv = DummyVecEnv([_factory for _ in range(max(1, n_envs))])
    if seed is not None:
        venv.seed(seed)

    if vecnormalize_path is not None:
        venv = VecNormalize.load(vecnormalize_path, venv)
        venv.training = training
        venv.norm_reward = norm_reward
    else:
        venv = VecNormalize(venv, training=training, norm_obs=True, norm_reward=norm_reward)
    return venv
