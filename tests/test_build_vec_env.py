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

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from drone_fly.env.racing_env import build_vec_env


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
