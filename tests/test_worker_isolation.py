"""UC-26 AC-13 — worker native-output suppression (the log-pane-safety gate).

Under the subproc backend, a spawned worker's native ``stdout``/``stderr`` (fd 1/2) would leak
onto — and corrupt — the live TUI display. The fix (``build_vec_env(..., suppress_worker_output=
tui_enabled)``) has each worker ``os.dup2`` its OWN fd 1/2 to ``os.devnull`` *before* building
its env, and ONLY inside spawned workers (never the main process, whose fds carry the TUI).

There is no way to inject a probe env into the real ``build_vec_env`` under ``spawn`` (the worker
re-imports the module fresh, so a parent monkeypatch of ``make_env`` never reaches it). So the
gate is tested two ways:

1. **Mechanism, both directions** — a module-level factory that mirrors the production
   ``_subproc_factory`` dup2 gate line-for-line, wrapping a probe env that writes a sentinel to
   fd 1 in ``reset``/``step``. With the parent's fd 1 redirected to a capture file (workers
   inherit it at spawn), the sentinel is absent when suppression is on and present when it is
   off — proving the gate flips behaviour and that the redirect happens in the *worker*, not the
   parent.
2. **Production wiring** — the real ``build_vec_env`` subproc path resets/steps/closes cleanly
   with ``suppress_worker_output`` both True and False, and never redirects the *main* process's
   fd 1 (the parent keeps its own stdout, which is what protects the TUI).

Hermetic: numpy Box spaces only; no pybullet / connectome / network.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np

from drone_fly.env.racing_env import VEC_ENV_START_METHOD, build_vec_env

_SENTINEL = b"WORKER_NATIVE_SPEW\n"


class _LeakyProbeEnv(gym.Env):
    """Module-level (picklable) env that writes a sentinel to fd 1 on reset and step."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def reset(self, *, seed=None, options=None):
        os.write(1, _SENTINEL)
        return np.zeros(1, np.float32), {}

    def step(self, action):
        os.write(1, _SENTINEL)
        return np.zeros(1, np.float32), 0.0, True, False, {}


class _GateFactory:
    """Module-level picklable factory mirroring production ``_subproc_factory``'s dup2 gate.

    Kept byte-for-byte equivalent to the redirect in
    ``drone_fly.env.racing_env.build_vec_env._subproc_factory`` so this test tracks the real
    gate: redirect the worker's own fd 1/2 to devnull IFF ``suppress`` is set, then build the env.
    """

    def __init__(self, suppress: bool) -> None:
        self.suppress = suppress

    def __call__(self):
        if self.suppress:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
            finally:
                os.close(devnull_fd)
        return _LeakyProbeEnv()


def _capture_worker_stdout(suppress: bool, capture_path: str) -> bytes:
    """Redirect the parent's fd 1 to ``capture_path`` (workers inherit it at spawn), run a
    reset+step across two spawned workers, restore fd 1, and return what landed in the file."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    saved_fd1 = os.dup(1)
    cap_fd = os.open(capture_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(cap_fd, 1)
    os.close(cap_fd)
    try:
        venv = SubprocVecEnv(
            [_GateFactory(suppress), _GateFactory(suppress)], start_method=VEC_ENV_START_METHOD
        )
        venv.reset()
        venv.step(np.zeros((2, 1), np.float32))
        venv.close()
    finally:
        os.dup2(saved_fd1, 1)  # restore the parent's real stdout no matter what
        os.close(saved_fd1)
    with open(capture_path, "rb") as fh:
        return fh.read()


def test_worker_output_is_suppressed_when_flag_set(tmp_path) -> None:
    """AC-13: with suppression on, no worker sentinel reaches the (inherited) parent stdout."""
    captured = _capture_worker_stdout(True, str(tmp_path / "cap_suppress.out"))
    assert _SENTINEL not in captured
    assert captured == b""


def test_worker_output_is_not_suppressed_by_default(tmp_path) -> None:
    """AC-13 (the other direction): with suppression off, worker output DOES reach stdout — so
    non-TUI subproc runs keep worker spew and the gate is proven to actually flip behaviour."""
    captured = _capture_worker_stdout(False, str(tmp_path / "cap_nosuppress.out"))
    assert _SENTINEL in captured
    assert captured.count(_SENTINEL) >= 2  # both workers, at least the reset write each


def test_build_vec_env_subproc_suppress_true_runs_and_leaves_parent_stdout_intact() -> None:
    """AC-13 production wiring: the real subproc path threads suppress_worker_output=True and
    still resets/steps/closes; the MAIN process keeps its own fd 1 (dup2 runs only in workers)."""
    parent_fd1_before = os.fstat(1).st_ino
    venv = build_vec_env(
        adapter="simple",
        n_envs=2,
        training=True,
        seed=0,
        vec_backend="subproc",
        suppress_worker_output=True,
    )
    try:
        obs = venv.reset()
        assert obs.shape[0] == 2
        venv.step(np.zeros((2, *venv.action_space.shape), dtype=np.float32))
    finally:
        venv.close()
    # The main process's stdout was never redirected to devnull by the worker-only gate.
    assert os.fstat(1).st_ino == parent_fd1_before


def test_build_vec_env_subproc_suppress_false_runs() -> None:
    """AC-13 production wiring: the default (suppress off) subproc path also runs cleanly."""
    venv = build_vec_env(
        adapter="simple",
        n_envs=2,
        training=True,
        seed=0,
        vec_backend="subproc",
        suppress_worker_output=False,
    )
    try:
        obs = venv.reset()
        assert obs.shape[0] == 2
    finally:
        venv.close()
