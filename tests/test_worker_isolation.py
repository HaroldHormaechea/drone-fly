"""UC-26 AC-13 / UC-32 AC-1·AC-2 — worker native-output isolation (the log-pane-safety gate).

Under the subproc backend a spawned worker's native ``stdout``/``stderr`` (fd 1/2) would leak onto —
and corrupt — the live TUI display. Pre-UC-32 the fix ``os.dup2``'d each worker's fd 1/2 to
``os.devnull`` before building its env. **UC-32 replaces devnull with a per-worker log FILE**
(``redirect_worker_fds`` → ``<worker_log_dir>/worker_<idx>.log``): the worker survives Windows
``spawn`` (AC-1), its output stays off the display yet is inspectable on disk (AC-2), and — as
before — the redirect happens ONLY inside the spawned worker, never the main process (whose fds
carry the TUI).

There is no way to inject a probe env into the real ``build_vec_env`` under ``spawn`` (the worker
re-imports the module fresh, so a parent monkeypatch of ``make_env`` never reaches it). So the gate
is tested two ways:

1. **Mechanism, both directions** — a module-level factory that mirrors the production per-index
   subproc factory, calling the REAL ``redirect_worker_fds`` when ``suppress`` is set, wrapping a
   probe env that writes a sentinel to fd 1. With the parent's fd 1 redirected to a capture file
   (workers inherit it at spawn), the sentinel is ABSENT from the parent capture when suppression is
   on (it went to the worker's own FILE, which we then read back) and PRESENT when it is off.
2. **Production wiring** — the real ``build_vec_env`` subproc path resets/steps/closes cleanly with
   ``suppress_worker_output`` both True and False, creates the per-worker files under
   ``worker_log_dir``, and never redirects the *main* process's fd 1.

Hermetic: numpy Box spaces only; no pybullet / connectome / network.
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np

from drone_fly.env.racing_env import VEC_ENV_START_METHOD, build_vec_env
from drone_fly.env.worker_output import redirect_worker_fds, worker_log_path

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
    """Module-level picklable factory mirroring the production per-index subproc factory.

    UC-32: redirect the worker's own fd 1/2 to its per-worker log FILE (via the REAL
    ``redirect_worker_fds``) IFF ``suppress`` is set and a ``worker_log_dir`` is given, then build
    the env — byte-for-byte the gate in ``drone_fly.env.racing_env.build_vec_env``'s
    ``_make_subproc_factory``.
    """

    def __init__(self, suppress: bool, worker_log_dir: str | None, idx: int) -> None:
        self.suppress = suppress
        self.worker_log_dir = worker_log_dir
        self.idx = idx

    def __call__(self):
        if self.suppress and self.worker_log_dir is not None:
            redirect_worker_fds(worker_log_path(self.worker_log_dir, self.idx))
        return _LeakyProbeEnv()


def _capture_worker_stdout(suppress: bool, capture_path: str, worker_log_dir: str | None) -> bytes:
    """Redirect the parent's fd 1 to ``capture_path`` (workers inherit it at spawn), run a
    reset+step across two spawned workers, restore fd 1, and return what landed in the parent."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    if worker_log_dir is not None:
        os.makedirs(worker_log_dir, exist_ok=True)

    saved_fd1 = os.dup(1)
    cap_fd = os.open(capture_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(cap_fd, 1)
    os.close(cap_fd)
    try:
        venv = SubprocVecEnv(
            [
                _GateFactory(suppress, worker_log_dir, 0),
                _GateFactory(suppress, worker_log_dir, 1),
            ],
            start_method=VEC_ENV_START_METHOD,
        )
        venv.reset()
        venv.step(np.zeros((2, 1), np.float32))
        venv.close()
    finally:
        os.dup2(saved_fd1, 1)  # restore the parent's real stdout no matter what
        os.close(saved_fd1)
    with open(capture_path, "rb") as fh:
        return fh.read()


def test_worker_output_goes_to_per_worker_file_not_parent_stdout(tmp_path) -> None:
    """AC-1/AC-2: with suppression on, the worker sentinel is ABSENT from the (inherited) parent
    stdout — it landed in the worker's OWN per-worker FILE (Windows-safe, inspectable), which we
    then read back to prove the output was captured, not discarded (devnull) and not leaked."""
    worker_dir = str(tmp_path / "workers")
    captured = _capture_worker_stdout(True, str(tmp_path / "cap_suppress.out"), worker_dir)
    assert _SENTINEL not in captured  # nothing leaked to the parent (TUI-safe)
    assert captured == b""

    # ...and the sentinel is inspectable in the per-worker files (AC-2), not swallowed by devnull.
    f0 = worker_log_path(worker_dir, 0)
    f1 = worker_log_path(worker_dir, 1)
    assert os.path.exists(f0)
    assert os.path.exists(f1)
    combined = open(f0, "rb").read() + open(f1, "rb").read()
    assert _SENTINEL in combined


def test_worker_output_is_not_suppressed_by_default(tmp_path) -> None:
    """AC-13 (the other direction): with suppression off, worker output DOES reach stdout — so
    non-TUI subproc runs keep worker spew and the gate is proven to actually flip behaviour."""
    captured = _capture_worker_stdout(False, str(tmp_path / "cap_nosuppress.out"), None)
    assert _SENTINEL in captured
    assert captured.count(_SENTINEL) >= 2  # both workers, at least the reset write each


def test_build_vec_env_subproc_suppress_true_runs_and_leaves_parent_stdout_intact(tmp_path) -> None:
    """AC-1/AC-13 production wiring: the real subproc path threads suppress_worker_output=True +
    worker_log_dir and still resets/steps/closes; the MAIN process keeps its own fd 1 (dup2 runs
    only in workers), and the per-worker files are created under the run's logs dir (AC-2)."""
    parent_fd1_before = os.fstat(1).st_ino
    worker_dir = str(tmp_path / "workers")
    venv = build_vec_env(
        adapter="simple",
        n_envs=2,
        training=True,
        seed=0,
        vec_backend="subproc",
        suppress_worker_output=True,
        worker_log_dir=worker_dir,
    )
    try:
        obs = venv.reset()
        assert obs.shape[0] == 2
        venv.step(np.zeros((2, *venv.action_space.shape), dtype=np.float32))
    finally:
        venv.close()
    # The main process's stdout was never redirected by the worker-only gate.
    assert os.fstat(1).st_ino == parent_fd1_before
    # Per-worker files exist and live under the run's logs dir (AC-2).
    assert os.path.exists(worker_log_path(worker_dir, 0))
    assert os.path.exists(worker_log_path(worker_dir, 1))


def test_build_vec_env_subproc_suppress_false_runs_without_worker_files(tmp_path) -> None:
    """AC-13/AC-10 production wiring: the default (suppress off, no worker_log_dir) subproc path
    also runs cleanly and creates NO worker files — byte-identical to the pre-UC-32 leak path."""
    worker_dir = tmp_path / "workers"
    venv = build_vec_env(
        adapter="simple",
        n_envs=2,
        training=True,
        seed=0,
        vec_backend="subproc",
        suppress_worker_output=False,
        worker_log_dir=None,
    )
    try:
        obs = venv.reset()
        assert obs.shape[0] == 2
    finally:
        venv.close()
    assert not worker_dir.exists()  # no redirect target created when suppression is off
