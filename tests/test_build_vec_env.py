"""UC-22 AC1 — the enabling data-layer change: ``VecMonitor`` in ``build_vec_env``.

Without a Monitor, SB3 never populates ``ep_info_buffer`` / ``ep_success_buffer``, so
``rollout/{ep_rew_mean,ep_len_mean,success_rate}`` are never collected — the data the TUI (and
the CSV/TensorBoard learning curve) needs. UC-22 wraps the **training** vec-env in
``VecMonitor(info_keywords=("is_success",))``, and it must sit **inside** ``VecNormalize`` so
the reported episode reward/length are the *raw* (un-normalised) values. Eval/non-training
construction must stay byte-identical (no Monitor).

Hermetic: the ``simple`` numpy adapter (no pybullet), offline, no network.
"""

from __future__ import annotations

import logging

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from drone_fly.env import racing_env
from drone_fly.env.racing_env import (
    DEFAULT_PARALLEL_N_ENVS,
    VEC_ENV_START_METHOD,
    build_vec_env,
    resolve_vec_env,
)


def test_training_env_is_vecmonitor_inside_vecnormalize() -> None:
    """training=True: VecNormalize → VecMonitor → DummyVecEnv (Monitor wraps raw episodes)."""
    venv = build_vec_env(adapter="simple", n_envs=1, training=True, seed=0)
    try:
        assert isinstance(venv, VecNormalize)
        assert isinstance(venv.venv, VecMonitor)  # Monitor is INSIDE Normalize (AC1 order)
        assert isinstance(venv.venv.venv, DummyVecEnv)  # ...and wraps the raw env
    finally:
        venv.close()


def test_training_vecmonitor_tracks_is_success() -> None:
    """The success_rate row needs ``is_success`` promoted from info into the episode buffer."""
    venv = build_vec_env(adapter="simple", n_envs=1, training=True, seed=0)
    try:
        assert venv.venv.info_keywords == ("is_success",)
    finally:
        venv.close()


def test_eval_env_has_no_vecmonitor_and_is_unchanged() -> None:
    """training=False must be byte-identical to before UC-22: no Monitor layer at all."""
    venv = build_vec_env(adapter="simple", n_envs=1, training=False, seed=0)
    try:
        assert isinstance(venv, VecNormalize)
        assert isinstance(venv.venv, DummyVecEnv)  # Normalize wraps the raw env directly
        assert not isinstance(venv.venv, VecMonitor)
    finally:
        venv.close()


def test_no_vecmonitor_anywhere_in_the_eval_stack() -> None:
    venv = build_vec_env(adapter="simple", n_envs=1, training=False, seed=0)
    try:
        node = venv
        while hasattr(node, "venv"):
            assert not isinstance(node, VecMonitor)
            node = node.venv
    finally:
        venv.close()


def test_training_env_reports_raw_episode_reward_from_monitor() -> None:
    """VecMonitor sits under VecNormalize, so the episode reward it records is the RAW reward,
    not the normalised one. Step to the end of an episode and check the Monitor-populated
    ``episode`` info carries a finite (un-normalised) reward."""
    import numpy as np

    venv = build_vec_env(adapter="simple", n_envs=1, training=True, seed=0)
    try:
        venv.reset()
        saw_episode = False
        for _ in range(2000):  # simple adapter episodes are short; this is ample
            actions = np.zeros((1,) + venv.action_space.shape, dtype=np.float32)
            _, _, dones, infos = venv.step(actions)
            for info in infos:
                if "episode" in info:  # VecMonitor writes this at episode end
                    r = info["episode"]["r"]
                    assert np.isfinite(r)  # a real, un-normalised episode return
                    saw_episode = True
            if saw_episode:
                break
        assert saw_episode, "VecMonitor never emitted an 'episode' record over 2000 steps"
    finally:
        venv.close()


# =========================================================================== #
# UC-26 — parallel vec-env rollout backend
# =========================================================================== #
#
# ``resolve_vec_env`` truth table (AC-1 / AC-5), the identical wrapper stack under a forced
# subproc backend (AC-2), and a forced subproc-on-simple reset+step (AC-4 spawn correctness).
# Only the ``"auto"`` adapter branch consults :func:`pybullet_available`; an explicit
# ``"pybullet"`` adapter is parallel-capable *without* pybullet being installed, so the
# pybullet-branch truth-table rows need no monkeypatch. The two ``"auto"`` rows monkeypatch
# ``pybullet_available`` to exercise both sides of the detection.


# --------------------------------------------------------------------------- #
# resolve_vec_env — the 3-branch resolver truth table (AC-1 / AC-5)
# --------------------------------------------------------------------------- #
def test_resolve_pybullet_unset_n_envs_defaults_to_8_subproc() -> None:
    """AC-5: parallel-capable adapter + unset n_envs -> DEFAULT_PARALLEL_N_ENVS (8), subproc."""
    resolved_adapter, n, backend = resolve_vec_env("pybullet", None, cfg_n_envs=1)
    assert (resolved_adapter, n, backend) == ("pybullet", DEFAULT_PARALLEL_N_ENVS, "subproc")
    assert DEFAULT_PARALLEL_N_ENVS == 8


def test_resolve_pybullet_explicit_n_envs_wins_and_is_subproc() -> None:
    """AC-5: an explicit n_envs overrides the default-8; >1 + pybullet -> subproc."""
    assert resolve_vec_env("pybullet", 4, cfg_n_envs=1) == ("pybullet", 4, "subproc")


def test_resolve_pybullet_single_env_is_dummy() -> None:
    """AC-1: n_envs==1 stays on DummyVecEnv even for the parallel-capable adapter."""
    assert resolve_vec_env("pybullet", 1, cfg_n_envs=8) == ("pybullet", 1, "dummy")


def test_resolve_simple_unset_falls_back_to_cfg_n_envs_dummy() -> None:
    """AC-1/AC-5: the load-bearing pre-UC-26 branch — simple + unset -> cfg_n_envs, dummy."""
    assert resolve_vec_env("simple", None, cfg_n_envs=2) == ("simple", 2, "dummy")


def test_resolve_simple_multi_env_stays_dummy() -> None:
    """AC-1: the simple adapter is never parallel-capable, so it stays serial on DummyVecEnv."""
    assert resolve_vec_env("simple", 3, cfg_n_envs=1) == ("simple", 3, "dummy")


def test_resolve_clamps_non_positive_n_envs_to_one() -> None:
    assert resolve_vec_env("pybullet", 0, cfg_n_envs=1) == ("pybullet", 1, "dummy")


def test_resolve_auto_becomes_pybullet_subproc_when_available(monkeypatch) -> None:
    """AC-1: adapter='auto' resolves to pybullet when the sim stack imports -> subproc + 8."""
    monkeypatch.setattr(racing_env, "pybullet_available", lambda: True)
    assert resolve_vec_env("auto", None, cfg_n_envs=1) == ("pybullet", 8, "subproc")


def test_resolve_auto_becomes_simple_dummy_when_unavailable(monkeypatch) -> None:
    """AC-1/AC-6: adapter='auto' resolves to simple when pybullet is absent -> dummy (CI path)."""
    monkeypatch.setattr(racing_env, "pybullet_available", lambda: False)
    assert resolve_vec_env("auto", None, cfg_n_envs=1) == ("simple", 1, "dummy")


def test_resolve_warns_only_when_multi_env_requested_on_non_parallel_adapter(caplog) -> None:
    """The subproc-on-simple pitfall surface: an explicit n_envs>1 on a non-parallel adapter
    warns (and stays on DummyVecEnv); the pybullet path and the unset path do NOT warn."""
    with caplog.at_level(logging.WARNING, logger="drone_fly.env.racing_env"):
        resolve_vec_env("simple", 4, cfg_n_envs=1)
    assert any("DummyVecEnv" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="drone_fly.env.racing_env"):
        resolve_vec_env("pybullet", 4, cfg_n_envs=1)  # parallel-capable -> no warning
        resolve_vec_env("simple", None, cfg_n_envs=2)  # unset -> no warning
        resolve_vec_env("simple", 1, cfg_n_envs=1)  # single env -> no warning
    assert caplog.records == []


# --------------------------------------------------------------------------- #
# build_vec_env(vec_backend="subproc") — identical wrapper stack (AC-2) + spawn (AC-4)
# --------------------------------------------------------------------------- #
def test_forced_subproc_wrapper_stack_is_vecmonitor_inside_vecnormalize() -> None:
    """AC-2: the wrapping order is backend-independent — VecNormalize -> VecMonitor ->
    SubprocVecEnv, mirroring the DummyVecEnv stack asserted at the top of this file."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    venv = build_vec_env(adapter="simple", n_envs=2, training=True, seed=0, vec_backend="subproc")
    try:
        assert isinstance(venv, VecNormalize)
        assert isinstance(venv.venv, VecMonitor)
        assert isinstance(venv.venv.venv, SubprocVecEnv)
    finally:
        venv.close()


def test_forced_subproc_eval_stack_has_no_vecmonitor() -> None:
    """AC-2: training=False stays Monitor-free for the subproc backend too (byte-identical
    wrapping rule), so eval construction is unchanged regardless of backend."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    venv = build_vec_env(adapter="simple", n_envs=2, training=False, seed=0, vec_backend="subproc")
    try:
        assert isinstance(venv, VecNormalize)
        assert isinstance(venv.venv, SubprocVecEnv)
        assert not isinstance(venv.venv, VecMonitor)
    finally:
        venv.close()


def test_forced_subproc_reset_and_step_are_finite_and_correct_shape() -> None:
    """AC-4: the env factory is picklable under 'spawn' and each worker rebuilds a functioning
    env — a forced subproc-on-simple stack resets and steps to finite, correctly-shaped obs."""
    n_envs = 2
    venv = build_vec_env(
        adapter="simple", n_envs=n_envs, training=True, seed=0, vec_backend="subproc"
    )
    try:
        obs = venv.reset()
        assert obs.shape == (n_envs, *venv.observation_space.shape)
        assert np.isfinite(obs).all()

        actions = np.zeros((n_envs, *venv.action_space.shape), dtype=np.float32)
        obs2, rewards, dones, infos = venv.step(actions)
        assert obs2.shape == (n_envs, *venv.observation_space.shape)
        assert np.isfinite(obs2).all()
        assert rewards.shape == (n_envs,)
        assert np.isfinite(rewards).all()
        assert dones.shape == (n_envs,)
        assert len(infos) == n_envs
    finally:
        venv.close()


def test_subproc_uses_the_spawn_start_method_constant() -> None:
    """AC-4/AC-10: the multiprocessing start method is the named 'spawn' constant."""
    assert VEC_ENV_START_METHOD == "spawn"
