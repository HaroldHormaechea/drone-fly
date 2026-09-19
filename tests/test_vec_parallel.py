"""UC-26 AC-3 — real cross-core parallelism, plus the worker-crash cleanup pitfall.

These tests exercise the ``SubprocVecEnv`` + ``start_method="spawn"`` machinery directly with
module-level, picklable helper envs (spawn re-imports this module in each worker, so anything a
worker builds MUST be defined at module top level — never a local closure/lambda/nested class).

* **AC-3** — sub-envs run in *separate* worker processes: their PIDs differ from each other and
  from the parent, and none of them import the live-TUI package (worker isolation, AC-13's
  worker half). The real factory's functional cross-core step is proven by
  ``test_build_vec_env.py::test_forced_subproc_reset_and_step_are_finite_and_correct_shape``.
* **Pitfall (worker crash / cleanup)** — a worker raising inside ``step`` surfaces promptly to
  the parent (it does not hang the run), and the vec-env can still be torn down.

Kept tiny and hermetic (numpy Box spaces only, no pybullet / connectome / network).
"""

from __future__ import annotations

import os
import sys
import time

import gymnasium as gym
import numpy as np
import pytest

from drone_fly.env.racing_env import VEC_ENV_START_METHOD


class _PidProbeEnv(gym.Env):
    """A trivial module-level (picklable) env that reports its worker's PID + TUI imports."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, np.float32), {}

    def step(self, action):
        return np.zeros(1, np.float32), 0.0, False, False, {}

    def worker_pid(self) -> int:
        return os.getpid()

    def tui_imported(self) -> bool:
        """Whether the live-TUI package was pulled into THIS worker's interpreter."""
        return any(m == "drone_fly.train" or m.startswith("drone_fly.train") for m in sys.modules)


class _CrashOnStepEnv(gym.Env):
    """A module-level env whose ``step`` raises inside the worker (crash pitfall)."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, np.float32), {}

    def step(self, action):
        raise RuntimeError("induced worker crash")


def test_workers_run_in_distinct_processes_from_each_other_and_the_parent() -> None:
    """AC-3: each sub-env lives in its own spawned worker process (distinct PIDs)."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    venv = SubprocVecEnv(
        [_PidProbeEnv, _PidProbeEnv, _PidProbeEnv], start_method=VEC_ENV_START_METHOD
    )
    try:
        pids = venv.env_method("worker_pid")
        assert len(pids) == 3
        assert len(set(pids)) == 3  # all three workers are separate processes
        assert os.getpid() not in pids  # ...and none is the main process
    finally:
        venv.close()


def test_workers_do_not_import_the_live_tui_package() -> None:
    """AC-13 (worker half): spawned rollout workers never import ``drone_fly.train`` / its TUI —
    the dashboard is a main-process-only concern, so it cannot collide with the workers."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    venv = SubprocVecEnv([_PidProbeEnv, _PidProbeEnv], start_method=VEC_ENV_START_METHOD)
    try:
        assert venv.env_method("tui_imported") == [False, False]
    finally:
        venv.close()


def test_worker_crash_surfaces_promptly_and_does_not_hang() -> None:
    """Pitfall: a worker dying in ``step`` propagates to the parent quickly (it does not hang
    the rollout), and teardown is still attempted. We bound the elapsed time to catch a hang."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    venv = SubprocVecEnv([_CrashOnStepEnv, _CrashOnStepEnv], start_method=VEC_ENV_START_METHOD)
    venv.reset()
    start = time.time()
    # A dead worker surfaces to the parent as an EOF / broken-pipe on the command socket
    # (ConnectionResetError is an OSError subclass) — never a silent hang.
    with pytest.raises((EOFError, OSError)):
        venv.step(np.zeros((2, 1), np.float32))
    assert time.time() - start < 30.0, "a crashed worker hung the parent instead of surfacing"
    # Best-effort teardown: workers are already gone, so close() may raise — it must not hang.
    try:
        venv.close()
    except Exception:
        pass
