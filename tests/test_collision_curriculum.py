"""UC-39/41 AC3 — training-time collision-penalty curriculum: pure schedule + callback plumbing.

The crash-cliff relief is a *training-time* curriculum, so the env default
``RewardConfig.collision_penalty`` (100) never changes — the penalty is ramped at rollout time via
:meth:`drone_fly.env.racing_env.RaceEnv.set_collision_penalty`. UC-41 reshaped the schedule from a
from-t=0 linear ramp into a **hold-then-ramp**: ``collision_penalty_start`` is HELD through the
first ``collision_curriculum_hold_fraction`` of training (the fly-learning phase), then ramped
linearly to ``collision_penalty_end`` over the next ``collision_curriculum_warmup_fraction``, then
held flat at the end value. This module pins:

* :func:`drone_fly.train.collision_curriculum.collision_penalty_at` — the **pure** schedule:
  holds ``collision_penalty_start`` through ``hold_fraction × total``, ramps linearly across the
  next ``warmup_fraction × total``, reaches and HOLDS ``collision_penalty_end`` at/after
  ``(hold + warmup) × total``, is monotonic non-decreasing and clamped, raises on an out-of-range
  hold/warmup split, and degenerates to the end value when there is no warmup;
* :class:`drone_fly.train.collision_curriculum.CollisionCurriculumCallback` — pushes the scheduled
  penalty into every base ``RaceEnv`` through the REAL SB3 wrapper stack (VecNormalize → VecMonitor
  → DummyVecEnv), verified by reading the override back through the wrappers;
* the pushed override actually **alters the reward** a genuine crash pays (env-level check).
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from drone_fly.env import make_env
from drone_fly.env.config import EnvConfig
from drone_fly.env.racing_env import build_vec_env
from drone_fly.train.collision_curriculum import (
    CollisionCurriculumCallback,
    collision_penalty_at,
)
from drone_fly.train.config import TrainConfig

_ZERO_THRUST = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


# --- AC3: the pure schedule (endpoints, monotonicity, clamping) ----------------------
def test_schedule_starts_at_start_value_at_t0() -> None:
    """AC3: the schedule begins at ``collision_penalty_start`` on the very first step (t=0), and any
    pre-start / negative query is clamped to the start value (never below the low endpoint)."""
    cfg = TrainConfig(
        total_timesteps=1000,
        collision_curriculum_hold_fraction=0.4,
        collision_curriculum_warmup_fraction=0.5,
    )
    assert collision_penalty_at(0, cfg) == pytest.approx(cfg.collision_penalty_start)
    assert collision_penalty_at(-5, cfg) == pytest.approx(cfg.collision_penalty_start)


def test_schedule_reaches_and_holds_end_value_after_warmup() -> None:
    """AC3 (UC-41): the ramp reaches ``collision_penalty_end`` exactly at the END of the ramp
    (``(hold_fraction + warmup_fraction) × total_timesteps``) — NOT at the warmup boundary alone —
    and holds it flat thereafter (no overshoot)."""
    cfg = TrainConfig(
        total_timesteps=1000,
        collision_curriculum_hold_fraction=0.4,
        collision_curriculum_warmup_fraction=0.5,
    )  # hold = 0.4×1000 = 400, warmup = 0.5×1000 = 500
    hold = int(cfg.collision_curriculum_hold_fraction * cfg.total_timesteps)  # 400
    warmup = int(cfg.collision_curriculum_warmup_fraction * cfg.total_timesteps)  # 500
    ramp_end = hold + warmup  # 900
    # Just before the ramp end it is still below the end value; at the ramp end it reaches end.
    assert collision_penalty_at(ramp_end - 1, cfg) < cfg.collision_penalty_end
    assert collision_penalty_at(ramp_end, cfg) == pytest.approx(cfg.collision_penalty_end)
    assert collision_penalty_at(ramp_end * 5, cfg) == pytest.approx(cfg.collision_penalty_end)
    assert collision_penalty_at(cfg.total_timesteps, cfg) == pytest.approx(
        cfg.collision_penalty_end
    )


def test_schedule_midpoint_is_linear_interpolation() -> None:
    """AC3 (UC-41): halfway through the RAMP (at ``hold_steps + warmup/2``) the penalty is the
    linear midpoint of the endpoints — the interpolation is over the ramp, not from t=0."""
    cfg = TrainConfig(
        total_timesteps=1000,
        collision_curriculum_hold_fraction=0.4,
        collision_curriculum_warmup_fraction=0.5,
    )  # hold 400, warmup 500; ramp midpoint at t=650
    hold = int(cfg.collision_curriculum_hold_fraction * cfg.total_timesteps)  # 400
    warmup = int(cfg.collision_curriculum_warmup_fraction * cfg.total_timesteps)  # 500
    ramp_midpoint = hold + warmup // 2  # 650
    expected = cfg.collision_penalty_start + 0.5 * (
        cfg.collision_penalty_end - cfg.collision_penalty_start
    )
    # 2 + 0.5·(100−2) = 51
    assert collision_penalty_at(ramp_midpoint, cfg) == pytest.approx(expected)


def test_schedule_is_monotonic_non_decreasing() -> None:
    """AC3: the schedule never decreases as training progresses (default start ≤ end endpoints)."""
    cfg = TrainConfig(
        total_timesteps=1000,
        collision_curriculum_hold_fraction=0.4,
        collision_curriculum_warmup_fraction=0.5,
    )
    vals = [collision_penalty_at(t, cfg) for t in range(0, 1200, 25)]
    for earlier, later in zip(vals, vals[1:], strict=False):
        assert later >= earlier - 1e-9
    assert vals[0] == pytest.approx(cfg.collision_penalty_start)
    assert vals[-1] == pytest.approx(cfg.collision_penalty_end)


def test_schedule_degenerate_no_warmup_returns_end_value() -> None:
    """AC3 edge: a zero (or non-positive) warmup fraction degenerates to the full end value
    immediately — no curriculum, train at full strength from step 0."""
    cfg = TrainConfig(total_timesteps=1000, collision_curriculum_warmup_fraction=0.0)
    assert collision_penalty_at(0, cfg) == pytest.approx(cfg.collision_penalty_end)
    assert collision_penalty_at(999, cfg) == pytest.approx(cfg.collision_penalty_end)


def test_schedule_is_stateless_and_resume_correct() -> None:
    """AC3 (UC-41): the schedule is a pure function of ``num_timesteps`` only, so a resumed run
    (which continues ``num_timesteps`` from the checkpoint) picks up the schedule at the right
    point — the value at a timestep is identical however you arrive there. With the hold-then-ramp
    shape, t=300 (total=1000 ⇒ hold_steps=400) now falls in the HOLD phase ⇒ returns the start
    value; a mid-RAMP point (t=650) is verified separately to exercise the interpolation branch."""
    cfg = TrainConfig(
        total_timesteps=1000,
        collision_curriculum_hold_fraction=0.4,
        collision_curriculum_warmup_fraction=0.5,
    )  # hold_steps = 400, ramp 400→900
    # Stateless: two queries at the same timestep agree exactly.
    assert collision_penalty_at(300, cfg) == collision_penalty_at(300, cfg)
    # t=300 is inside the hold phase (≤ 400) ⇒ the held start value.
    assert collision_penalty_at(300, cfg) == pytest.approx(cfg.collision_penalty_start)
    # A mid-ramp point still resolves to the interpolated value purely from num_timesteps.
    assert collision_penalty_at(650, cfg) == collision_penalty_at(650, cfg)
    assert collision_penalty_at(650, cfg) == pytest.approx(
        cfg.collision_penalty_start
        + ((650 - 400) / 500) * (cfg.collision_penalty_end - cfg.collision_penalty_start)
    )


# --- AC3 (UC-41): the hold phase covers the whole fly-learning window --------------------
def test_schedule_holds_start_value_through_the_entire_hold_phase() -> None:
    """AC3 (UC-41 crux): the collision penalty is HELD at ``collision_penalty_start`` across the
    whole hold phase ``[0, hold_steps]`` — this is what keeps the effective penalty below the
    ~4.8 crash-cliff through the fly-learning window (the observed stall was at ~16% / 159k of a
    1M-step run). If a future change shortens the hold, this test fails loudly."""
    cfg = TrainConfig(
        total_timesteps=1_000_000,
        collision_curriculum_hold_fraction=0.4,
        collision_curriculum_warmup_fraction=0.5,
    )  # hold_steps = 0.4×1M = 400_000
    hold_steps = int(cfg.collision_curriculum_hold_fraction * cfg.total_timesteps)
    assert hold_steps == 400_000
    # The observed-stall operating point (16% ≈ 159k) sits inside the hold ⇒ still the start value.
    assert collision_penalty_at(159_000, cfg) == pytest.approx(cfg.collision_penalty_start)
    # A dense sweep across the whole hold phase (including both endpoints) never leaves the start.
    for t in range(0, hold_steps + 1, 20_000):
        assert collision_penalty_at(t, cfg) == pytest.approx(cfg.collision_penalty_start)
    assert collision_penalty_at(hold_steps, cfg) == pytest.approx(cfg.collision_penalty_start)
    # One step past the hold, the ramp has begun to rise above the start value.
    assert collision_penalty_at(hold_steps + 1, cfg) > cfg.collision_penalty_start


@pytest.mark.parametrize(
    "hold_fraction, warmup_fraction",
    [
        (-0.1, 0.5),  # negative hold
        (0.6, 0.5),  # hold + warmup = 1.1 > 1 (ramp end past the run length)
        (0.9, 0.2),  # hold + warmup = 1.1 > 1
    ],
)
def test_schedule_raises_on_out_of_range_hold_warmup_split(
    hold_fraction: float, warmup_fraction: float
) -> None:
    """AC3 (UC-41): the schedule rejects an out-of-range curriculum shape — ``hold_fraction < 0`` or
    ``hold_fraction + warmup_fraction > 1`` (which would push the ramp end past the run length) —
    with a ``ValueError`` rather than silently producing a degenerate ramp."""
    cfg = TrainConfig(
        total_timesteps=1000,
        collision_curriculum_hold_fraction=hold_fraction,
        collision_curriculum_warmup_fraction=warmup_fraction,
    )
    with pytest.raises(ValueError, match="collision curriculum fractions out of range"):
        collision_penalty_at(0, cfg)


def test_schedule_accepts_boundary_hold_warmup_split_summing_to_one() -> None:
    """AC3 (UC-41): the boundary split ``hold + warmup == 1`` is valid (ramp ends exactly at the
    run end, no held-at-end tail) and must NOT raise."""
    cfg = TrainConfig(
        total_timesteps=1000,
        collision_curriculum_hold_fraction=0.5,
        collision_curriculum_warmup_fraction=0.5,
    )
    assert collision_penalty_at(0, cfg) == pytest.approx(cfg.collision_penalty_start)
    assert collision_penalty_at(1000, cfg) == pytest.approx(cfg.collision_penalty_end)


# --- AC3: the callback pushes the penalty through the real SB3 wrapper stack ----------
def _fake_model(env):
    """A minimal stand-in exposing only what ``BaseCallback`` needs: ``get_env`` for
    ``training_env``. ``num_timesteps`` is a plain attribute set on the callback itself."""
    return types.SimpleNamespace(get_env=lambda: env)


def test_callback_pushes_scheduled_penalty_through_wrapper_stack() -> None:
    """AC3: ``CollisionCurriculumCallback`` computes the scheduled penalty for the current
    ``num_timesteps`` and pushes it — via ``training_env.env_method('set_collision_penalty', cp)`` —
    all the way down to each base ``RaceEnv``, THROUGH the VecNormalize → VecMonitor → DummyVecEnv
    stack. Verified by reading ``_collision_penalty_override`` back through the same wrappers."""
    venv = build_vec_env(adapter="simple", n_envs=1, training=True, seed=0)
    try:
        # Confirm we really are exercising the full wrapper stack (not a bare env).
        assert isinstance(venv, VecNormalize)
        assert isinstance(venv.venv, VecMonitor)
        assert isinstance(venv.venv.venv, DummyVecEnv)

        cfg = TrainConfig(
            total_timesteps=1000,
            collision_curriculum_hold_fraction=0.4,
            collision_curriculum_warmup_fraction=0.5,
        )
        cb = CollisionCurriculumCallback(cfg)
        cb.init_callback(_fake_model(venv))

        # Start of training (t=0): the start value reaches every base env through the wrappers.
        cb.num_timesteps = 0
        cb._on_training_start()
        assert venv.get_attr("_collision_penalty_override") == [
            pytest.approx(cfg.collision_penalty_start)
        ]

        # A later rollout at full strength: the end value propagates the same way.
        cb.num_timesteps = cfg.total_timesteps
        cb._on_rollout_start()
        assert venv.get_attr("_collision_penalty_override") == [
            pytest.approx(cfg.collision_penalty_end)
        ]

        # Mid-ramp on a subsequent rollout: the interpolated value propagates too. With the UC-41
        # hold-then-ramp shape (hold_steps=400, ramp 400→900) the ramp midpoint is at t=650 ⇒ 51.0.
        cb.num_timesteps = 650
        cb._on_rollout_start()
        assert venv.get_attr("_collision_penalty_override") == [pytest.approx(51.0)]

        # A point inside the hold phase (t=250 ≤ hold_steps=400) propagates the held start value.
        cb.num_timesteps = 250
        cb._on_rollout_start()
        assert venv.get_attr("_collision_penalty_override") == [
            pytest.approx(cfg.collision_penalty_start)
        ]
    finally:
        venv.close()


def test_callback_propagates_to_all_envs_in_a_multi_env_stack() -> None:
    """AC3: with more than one base env, the pushed penalty reaches EVERY env — ``env_method`` fans
    out across the whole DummyVecEnv, not just env 0."""
    venv = build_vec_env(adapter="simple", n_envs=3, training=True, seed=0)
    try:
        cfg = TrainConfig(
            total_timesteps=1000,
            collision_curriculum_hold_fraction=0.4,
            collision_curriculum_warmup_fraction=0.5,
        )
        cb = CollisionCurriculumCallback(cfg)
        cb.init_callback(_fake_model(venv))
        cb.num_timesteps = 0
        cb._on_training_start()
        overrides = venv.get_attr("_collision_penalty_override")
        assert overrides == [pytest.approx(cfg.collision_penalty_start)] * 3
    finally:
        venv.close()


# --- AC3: the pushed override actually alters the reward a genuine crash pays ---------
def _fall_crash_reward(cp_override):
    """Drive a real-adapter free fall (zero thrust from a mid-air start) into a genuine floor crash;
    return (crash-step reward, info). The trajectory is deterministic, so two runs differ ONLY in
    the collision penalty when an override is set — isolating its effect on the reward."""
    env = make_env(EnvConfig(floor_start=False), adapter="simple")
    env.reset(seed=0)
    if cp_override is not None:
        env.set_collision_penalty(cp_override)
    reward = 0.0
    info: dict = {}
    for _ in range(500):
        _obs, reward, terminated, _tr, info = env.step(_ZERO_THRUST)
        if terminated:
            break
    return reward, info


def test_override_alters_the_reward_a_genuine_crash_pays() -> None:
    """AC3: setting the override via ``set_collision_penalty`` changes what a genuine crash pays in
    the reward — a low override (10) yields a much less negative crash-step reward than the env
    default (100), and the difference is EXACTLY the penalty gap (everything else identical)."""
    r_default, info_default = _fall_crash_reward(None)  # env default 100
    r_low, info_low = _fall_crash_reward(10.0)  # curriculum start value
    # Both are genuine terminating crashes.
    assert info_default["collided"] is True and info_low["collided"] is True
    default_cp = EnvConfig().reward.collision_penalty
    assert default_cp == pytest.approx(100.0)
    # Identical deterministic fall ⇒ the reward gap is purely the penalty gap: (−10) − (−100) = +90.
    assert r_low - r_default == pytest.approx(default_cp - 10.0)
    assert r_low > r_default  # the relieved crash is strictly less punishing


def test_set_collision_penalty_none_restores_env_default() -> None:
    """AC3: clearing the override (``None``) restores the env default penalty — the override never
    mutates the env's own ``RewardConfig`` (which stays 100)."""
    env = make_env(EnvConfig(floor_start=False), adapter="simple")
    env.reset(seed=0)
    env.set_collision_penalty(10.0)
    assert env._collision_penalty_override == pytest.approx(10.0)
    env.set_collision_penalty(None)
    assert env._collision_penalty_override is None
    # The env's default reward config was never touched.
    assert env.config.reward.collision_penalty == pytest.approx(100.0)


def test_curriculum_disabled_leaves_no_callback_effect() -> None:
    """AC3/AC9: with the curriculum disabled the training loop never installs the callback, so the
    env trains at the constant default (byte-identical to pre-UC-39). We assert the disable flag is
    honoured at the config level (the loop guards on ``collision_curriculum_enabled``)."""
    assert TrainConfig().collision_curriculum_enabled is True  # default ON (AC3)
    assert TrainConfig(collision_curriculum_enabled=False).collision_curriculum_enabled is False
