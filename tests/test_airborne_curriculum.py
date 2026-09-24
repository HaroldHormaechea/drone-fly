"""UC-44 — airborne-start reverse curriculum: the pure schedule, its training-only plumbing, the
early-termination compatibility with airborne spawns, and a hermetic numpy-adapter smoke.

The takeoff-discovery relief is a *training-time* curriculum: it changes only the SPAWN STATE, not
the reward function, so every reward-math / doc-contract test stays green (AC6, pinned elsewhere).
Instead of always spawning on the floor (UC-37 forces spawn z → ``course.floor_z``), the training
envs spawn the drone airborne early in training and anneal the spawn z linearly down to the floor
as training progresses, pushed in each rollout via
:meth:`drone_fly.env.racing_env.RaceEnv.set_spawn_z`. This module pins:

* **AC1** — :func:`drone_fly.train.airborne_curriculum.spawn_z_at`, the **pure** schedule:
  ``high_z`` at fraction 0, EXACTLY ``floor_z`` at/after the anneal end (UC-37 byte-identity at the
  terminal state), monotone non-increasing, clamped ``[floor_z, high_z]``, ``ValueError`` on an
  out-of-range anneal fraction, and the degenerate ``anneal_fraction == 0`` / ``total_timesteps
  <= 0`` → ``floor_z`` (curriculum off).
* **AC2** — training-only application: the callback pushes the scheduled spawn z through the REAL
  SB3 wrapper stack onto the training envs, while an eval/recording env (built with
  ``training=False``, and NEVER given the callback) keeps the floored UC-37 spawn — as does a base
  ``RaceEnv`` with no override set (default ``None``).
* **AC5** — early-termination compatibility: an airborne spawn (``set_spawn_z`` / a high spawn
  state) is taken-off at step 0 and is NOT grounded/stuck cut within the warm-up window, while a
  drone that then falls to the floor and rests IS grounded-cut (UC-36 semantics preserved).
* **AC8** — a hermetic numpy-adapter smoke: an airborne spawn collects airborne/climb reward and
  the episode return clears the −5 time-penalty floor, and a short seeded PPO train under the
  curriculum keeps the action std off 0 (no collapse).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from drone_fly.adapter.base import DroneState
from drone_fly.env.config import EarlyTerminationConfig, EnvConfig
from drone_fly.env.racing_env import build_vec_env, make_env
from drone_fly.train.airborne_curriculum import AirborneStartCurriculumCallback, spawn_z_at
from drone_fly.train.config import TrainConfig

# High/low endpoints used by the pure-schedule tests. floor_z = 0.0 (the course floor), high_z = 1.0
# (floor + altitude_target — UC-58; was climb_target_height), the same endpoints the training loop
# derives at wire time.
_FLOOR_Z = 0.0
_HIGH_Z = 1.0


# ============================================================================================
# AC1 — the pure schedule: endpoints, monotonicity, clamping, degenerate + out-of-range cases.
# ============================================================================================
def test_ac1_returns_high_z_at_fraction_zero() -> None:
    """AC1: at ``num_timesteps == 0`` the spawn is the full airborne start (the high endpoint)."""
    cfg = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_anneal_fraction=0.5,
        airborne_curriculum_warmup_fraction=0.0,
    )
    assert spawn_z_at(0, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == pytest.approx(_HIGH_Z)


def test_ac1_returns_exactly_floor_z_at_and_after_anneal_end() -> None:
    """AC1: at the anneal end (``anneal_fraction × total_timesteps``) and forever after, the spawn
    is held at EXACTLY ``floor_z`` — a strict ``==`` (not ``approx``) so the terminal state is
    byte-identical to the UC-37 floored start (the eval/recording spawn)."""
    cfg = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_anneal_fraction=0.5,
        airborne_curriculum_warmup_fraction=0.0,
    )
    anneal_steps = 500  # 0.5 × 1000
    assert spawn_z_at(anneal_steps, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == _FLOOR_Z
    assert spawn_z_at(anneal_steps + 1, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == _FLOOR_Z
    assert spawn_z_at(10 * anneal_steps, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == _FLOOR_Z


def test_ac1_midpoint_is_linear_interpolation() -> None:
    """AC1: halfway through the anneal window the spawn is the linear midpoint of the endpoints."""
    cfg = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_anneal_fraction=0.5,
        airborne_curriculum_warmup_fraction=0.0,
    )
    # anneal_steps = 500; halfway = 250 ⇒ (high + floor) / 2 = 0.5.
    assert spawn_z_at(250, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == pytest.approx(0.5)


def test_ac1_monotone_non_increasing_across_the_anneal() -> None:
    """AC1: the schedule is monotone NON-INCREASING in ``num_timesteps`` (never climbs back up)."""
    cfg = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_anneal_fraction=0.5,
        airborne_curriculum_warmup_fraction=0.0,
    )
    samples = [spawn_z_at(t, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) for t in range(0, 1001, 25)]
    for earlier, later in zip(samples, samples[1:], strict=False):
        assert later <= earlier + 1e-12, f"schedule rose: {earlier} → {later}"


def test_ac1_clamped_to_floor_high_band() -> None:
    """AC1: the result never dips below ``floor_z`` nor overshoots ``high_z`` — a negative query
    clamps to the high endpoint and any query past the anneal clamps to the floor."""
    cfg = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_anneal_fraction=0.5,
        airborne_curriculum_warmup_fraction=0.0,
    )
    for t in (-100, -1, 0, 1, 123, 250, 499, 500, 501, 5000, 10_000_000):
        z = spawn_z_at(t, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z)
        assert _FLOOR_Z <= z <= _HIGH_Z, f"t={t}: {z} escaped [{_FLOOR_Z}, {_HIGH_Z}]"
    # A negative (pre-start) query is the full airborne start, clamped to the high endpoint.
    assert spawn_z_at(-5, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == pytest.approx(_HIGH_Z)


@pytest.mark.parametrize("bad_fraction", [-0.1, -1.0, 1.0001, 2.0])
def test_ac1_raises_on_out_of_range_anneal_fraction(bad_fraction: float) -> None:
    """AC1: an anneal fraction outside ``[0, 1]`` (a window that runs backwards or past the end of
    training) is a config error — the schedule raises ``ValueError`` rather than silently clamp."""
    cfg = TrainConfig(total_timesteps=1000, airborne_curriculum_anneal_fraction=bad_fraction)
    with pytest.raises(ValueError, match="anneal fraction"):
        spawn_z_at(0, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z)


def test_ac1_boundary_anneal_fractions_are_accepted() -> None:
    """AC1: the inclusive endpoints ``0`` and ``1`` are valid (0 ⇒ curriculum off; 1 ⇒ anneal over
    the whole run). Neither raises."""
    off = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_anneal_fraction=0.0,
        airborne_curriculum_warmup_fraction=0.0,
    )
    full = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_anneal_fraction=1.0,
        airborne_curriculum_warmup_fraction=0.0,
    )
    assert spawn_z_at(0, off, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == _FLOOR_Z  # off ⇒ floored
    assert spawn_z_at(0, full, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == pytest.approx(_HIGH_Z)
    assert spawn_z_at(1000, full, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == _FLOOR_Z


def test_ac1_degenerate_zero_fraction_is_floored_curriculum_off() -> None:
    """AC1: ``anneal_fraction == 0`` degenerates to the floored UC-37 start at every step — the
    curriculum is off and training spawns on the floor exactly as UC-43."""
    cfg = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_anneal_fraction=0.0,
        airborne_curriculum_warmup_fraction=0.0,
    )
    for t in (0, 1, 500, 1000, 50_000):
        assert spawn_z_at(t, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == _FLOOR_Z


def test_ac1_degenerate_nonpositive_total_timesteps_is_floored() -> None:
    """AC1: a non-positive ``total_timesteps`` (no real anneal window) degenerates to ``floor_z``
    — the schedule can't divide by a zero/negative horizon, so it turns the curriculum off."""
    cfg = TrainConfig(
        total_timesteps=0,
        airborne_curriculum_anneal_fraction=0.5,
        airborne_curriculum_warmup_fraction=0.0,
    )
    assert spawn_z_at(0, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == _FLOOR_Z


def test_ac1_env_defensively_clamps_a_below_floor_override_to_the_floor() -> None:
    """AC1/AC2 (defensive): the env's ``floor_start`` reset applies ``max(override, floor_z)``, so
    even a hand-set below-floor ``set_spawn_z`` value never spawns the drone underground — it is
    pinned to the floor. (The schedule itself is already floor-bounded; this is belt-and-braces.)"""
    floor_z = EnvConfig().course.floor_z
    env = make_env(EnvConfig(), adapter="simple")
    env.set_spawn_z(floor_z - 1.0)  # below the floor — must be clamped up
    env.reset(seed=0)
    assert env._prev_pos[2] == pytest.approx(floor_z, abs=env._et_floor_epsilon)


def test_ac1_is_stateless_and_resume_correct() -> None:
    """AC1: the schedule is a pure function of ``num_timesteps`` (+ cfg / endpoints) — the same
    query returns the same value regardless of call order, so a resumed run continues correctly."""
    cfg = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_anneal_fraction=0.5,
        airborne_curriculum_warmup_fraction=0.0,
    )
    first = [spawn_z_at(t, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) for t in (0, 250, 500, 750)]
    # Query out of order; each value must be identical to the in-order query (no hidden state).
    assert spawn_z_at(750, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == first[3]
    assert spawn_z_at(0, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == first[0]
    assert spawn_z_at(500, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == first[2]


# ============================================================================================
# AC2 — training-only application: pushed onto training envs; eval / no-override stay floored.
# ============================================================================================
def _fake_model(env):
    """A minimal stand-in exposing only what ``BaseCallback`` needs: ``get_env`` for
    ``training_env``. ``num_timesteps`` is a plain attribute set on the callback itself (mirrors the
    collision-curriculum test's helper)."""
    return types.SimpleNamespace(get_env=lambda: env)


def test_ac2_callback_pushes_scheduled_spawn_z_through_wrapper_stack() -> None:
    """AC2: ``AirborneStartCurriculumCallback`` computes the scheduled spawn z for the current
    ``num_timesteps`` and pushes it — via ``training_env.env_method('set_spawn_z', z)`` — all the
    way down to each base ``RaceEnv`` THROUGH the VecNormalize → VecMonitor → DummyVecEnv stack.
    Verified by reading ``_spawn_z_override`` back through the same wrappers."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    venv = build_vec_env(adapter="simple", n_envs=1, training=True, seed=0)
    try:
        # Confirm we really are exercising the full wrapper stack (not a bare env).
        assert isinstance(venv, VecNormalize)
        assert isinstance(venv.venv, VecMonitor)
        assert isinstance(venv.venv.venv, DummyVecEnv)

        cfg = TrainConfig(
            total_timesteps=1000,
            airborne_curriculum_anneal_fraction=0.5,
            airborne_curriculum_warmup_fraction=0.0,
        )
        cb = AirborneStartCurriculumCallback(cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z)
        cb.init_callback(_fake_model(venv))

        # Start of training (t=0): the high airborne spawn reaches the base env through the stack.
        cb.num_timesteps = 0
        cb._on_training_start()
        assert venv.get_attr("_spawn_z_override") == [pytest.approx(_HIGH_Z)]

        # Mid-anneal on a later rollout: the interpolated spawn propagates the same way (t=250⇒0.5).
        cb.num_timesteps = 250
        cb._on_rollout_start()
        assert venv.get_attr("_spawn_z_override") == [pytest.approx(0.5)]

        # At/after the anneal end the pushed value is exactly the floor (terminal floored spawn).
        cb.num_timesteps = 500
        cb._on_rollout_start()
        assert venv.get_attr("_spawn_z_override") == [_FLOOR_Z]
    finally:
        venv.close()


def test_ac2_callback_propagates_to_all_envs_in_a_multi_env_stack() -> None:
    """AC2: with more than one base env the pushed spawn reaches EVERY env — ``env_method`` fans
    out across the whole DummyVecEnv, not just env 0."""
    venv = build_vec_env(adapter="simple", n_envs=3, training=True, seed=0)
    try:
        cfg = TrainConfig(
            total_timesteps=1000,
            airborne_curriculum_anneal_fraction=0.5,
            airborne_curriculum_warmup_fraction=0.0,
        )
        cb = AirborneStartCurriculumCallback(cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z)
        cb.init_callback(_fake_model(venv))
        cb.num_timesteps = 0
        cb._on_training_start()
        assert venv.get_attr("_spawn_z_override") == [pytest.approx(_HIGH_Z)] * 3
    finally:
        venv.close()


def test_ac2_base_raceenv_without_override_spawns_on_the_floor() -> None:
    """AC2: a base ``RaceEnv`` with the default (``None``) override — i.e. any env that never
    received the curriculum callback — resets to the UC-37 floored spawn (z ≈ floor_z), not
    airborne. This is the takeoff-measurement guarantee."""
    env = make_env(EnvConfig(), adapter="simple")  # floor_start default True, no override
    assert env._spawn_z_override is None
    env.reset(seed=0)
    floor_z = env._course.floor_z
    assert env._prev_pos[2] == pytest.approx(floor_z, abs=env._et_floor_epsilon)
    assert env._took_off is False  # a floored spawn has not taken off


def test_ac2_set_spawn_z_none_restores_the_floored_spawn() -> None:
    """AC2: clearing the override with ``set_spawn_z(None)`` restores the floored spawn even after a
    prior airborne override — the None default is genuinely floor, not a stale raised value."""
    env = make_env(EnvConfig(), adapter="simple")
    env.set_spawn_z(5.0)
    env.reset(seed=0)
    assert env._prev_pos[2] == pytest.approx(5.0, abs=env._et_floor_epsilon)  # override took effect
    env.set_spawn_z(None)
    env.reset(seed=0)
    assert env._prev_pos[2] == pytest.approx(env._course.floor_z, abs=env._et_floor_epsilon)


def test_ac2_eval_vec_env_has_no_override_and_spawns_on_the_floor() -> None:
    """AC2: the eval/recording path — ``build_vec_env(training=False)``, which the evaluator uses
    and which is NEVER handed the curriculum callback — carries no spawn override and resets on the
    floor, so the takeoff measurement is unchanged regardless of the schedule."""
    venv = build_vec_env(adapter="simple", n_envs=1, training=False, seed=0)
    try:
        assert venv.get_attr("_spawn_z_override") == [None]
        venv.reset()
        floor_z = venv.get_attr("_course")[0].floor_z
        spawn_z = venv.get_attr("_prev_pos")[0][2]
        assert spawn_z == pytest.approx(floor_z, abs=venv.get_attr("_et_floor_epsilon")[0])
    finally:
        venv.close()


# ============================================================================================
# AC5 — early-termination compatibility with airborne spawns.
# ============================================================================================
class _ScriptedAdapter:
    """A stand-in adapter that replays a fixed list of positions (velocity ≡ 0), for env-level
    early-termination tests. Zero velocity means the grounded detector's rest-speed guard is always
    satisfied, so the detector's behaviour is driven purely by the scripted altitude band."""

    backend = "scripted"

    def __init__(self, positions):
        self._positions = [np.asarray(p, dtype=np.float64) for p in positions]
        self._i = 0

    def _state(self, idx) -> DroneState:
        return DroneState(
            position=self._positions[min(idx, len(self._positions) - 1)].copy(),
            velocity=np.zeros(3),
            attitude=np.zeros(3),
            angular_velocity=np.zeros(3),
            collided=False,
        )

    def reset(self, seed=None) -> DroneState:
        self._i = 0
        return self._state(0)

    def step(self, action) -> DroneState:
        self._i += 1
        return self._state(self._i)

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def _env_with(adapter, config):
    env = make_env(config, adapter="simple")
    env.adapter = adapter
    env.backend = adapter.backend
    return env


_HOVER = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)


def test_ac5_airborne_spawn_is_taken_off_at_step_zero() -> None:
    """AC5: a spawn above the floor band (``set_spawn_z`` raised it, or the spawn state is high) is
    considered taken-off at step 0, so the grounded detector arms immediately (UC-36 semantics)
    rather than the pre-takeoff suppression a floored start gets. Exercised through the REAL
    simple-adapter ``set_spawn_z`` override."""
    env = make_env(EnvConfig(), adapter="simple")
    env.set_spawn_z(1.0)
    env.reset(seed=0)
    assert env._prev_pos[2] == pytest.approx(1.0, abs=env._et_floor_epsilon)
    assert env._took_off is True


def test_ac5_airborne_spawn_not_grounded_or_stuck_cut_within_warmup_window() -> None:
    """AC5: a drone that spawns high and hovers/descends slowly (staying above the floor band) is
    NOT cut within its warm-up window — the grounded counter never arms while it is airborne, and
    neither detector can fire before it accumulates a full window. Directly addresses the pitfall
    that a high spawn must not be instantly flagged grounded/stuck."""
    # Spawn airborne and descend slowly, staying ABOVE the floor band (floor_epsilon = 0.05) the
    # whole time. No horizontal progress ⇒ the stuck counter climbs, but stays below its window.
    positions = [(0.0, 0.0, z) for z in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)]
    env = _env_with(
        _ScriptedAdapter(positions),
        EnvConfig(early_termination=EarlyTerminationConfig(stuck_window=8, grounded_window=5)),
    )
    env.reset()
    assert env._took_off is True  # armed from the high spawn
    for step in range(4):  # 4 < grounded_window(5) and < stuck_window(8): inside the warm-up
        _obs, _r, terminated, truncated, info = env.step(_HOVER)
        assert not terminated, f"airborne drone cut at step {step}: {info.get('early_termination')}"
        assert not truncated
        # It is airborne (above the floor band) every step, so the grounded counter never arms.
        assert env._grounded_counter == 0


def test_ac5_airborne_spawn_then_fall_to_floor_and_rest_is_grounded_cut() -> None:
    """AC5: grounded TERMINATION semantics are preserved — a drone that spawned airborne (took-off
    at step 0) and then falls to the floor and RESTS there is grounded-cut exactly as UC-36, once
    the grounded counter fills its window. UC-58: the cut is now PENALTY-FREE (``collided=False``) —
    a gentle grounded-rest after a failed takeoff must not pay −collision_penalty, or it would
    re-create the early-termination trap. The grounded detector still BOUNDS the episode; only a
    genuine crash pays the penalty."""
    # Airborne spawn (z=1.0 ⇒ took_off), then rest on the floor (z=0.0, in the band, zero velocity).
    positions = [(0.0, 0.0, 1.0)] + [(0.0, 0.0, 0.0)] * 6
    env = _env_with(
        _ScriptedAdapter(positions),
        EnvConfig(early_termination=EarlyTerminationConfig(stuck_window=100, grounded_window=3)),
    )
    env.reset()
    assert env._took_off is True
    terminated = truncated = False
    info: dict = {}
    for _ in range(env._max_steps + 1):
        _obs, _r, terminated, truncated, info = env.step(_HOVER)
        if terminated or truncated:
            break
    assert terminated is True and truncated is False
    assert info["early_termination"] == "grounded", (
        "an airborne-spawned drone that falls to the floor and rests is grounded-cut (UC-36)"
    )
    # UC-58: the grounded cut is penalty-free — it terminates the episode but does NOT pay
    # −collision_penalty (only a genuine crash does). This is the anti-suicide guarantee (AC1/AC2).
    assert info["collided"] is False


# ============================================================================================
# AC8 — hermetic numpy-adapter smoke: airborne spawn collects reward > −5 floor; std no-collapse.
# ============================================================================================
def test_ac8_airborne_spawn_collects_reward_above_time_penalty_floor() -> None:
    """AC8 (reward clause, hermetic — numpy adapter, no pybullet): an airborne spawn that holds
    altitude collects the sustained UC-58 altitude reward, so the episode return is strictly
    positive (well clear of the 0 a floored do-nothing episode now earns — ``time_penalty`` is
    retired). The cut is the penalty-free stuck cut (it stayed airborne but made no forward
    progress), NOT a paid crash — so the positive altitude gradient is not swamped."""
    env = make_env(EnvConfig(), adapter="simple")
    env.set_spawn_z(1.0)  # spawn at altitude_target (the curriculum's early high endpoint)
    env.reset(seed=0)
    band = env._course.floor_z + env._et_floor_epsilon
    total = 0.0
    airborne_steps = 0
    info: dict = {}
    terminated = truncated = False
    # Hover throttle (0.5) maintains altitude in the point-mass adapter ⇒ the drone stays airborne.
    for _ in range(400):
        _obs, reward, terminated, truncated, info = env.step(_HOVER)
        total += reward
        if info["position"][2] > band:
            airborne_steps += 1
        if terminated or truncated:
            break
    assert airborne_steps > 0, "the drone must actually spend time airborne to collect the reward"
    assert total > 0.0, (
        f"airborne-spawn episode return {total:.3f} must be strictly positive — the sustained "
        f"UC-58 altitude reward pays every airborne step (a floored do-nothing episode earns ≈ 0)"
    )
    # It stayed airborne but stopped progressing ⇒ penalty-free stuck cut, not a paid crash.
    assert info.get("early_termination") == "stuck"
    assert info["collided"] is False


def test_ac8_smoke_train_airborne_spawn_std_does_not_collapse(connectome) -> None:
    """AC8 (std clause, lab-scoped): a short seeded CPU train under the REAL climb-bias init + the
    REAL airborne curriculum callback completes with the action std finite and OFF 0 (no collapse),
    and the callback genuinely raised the spawn into the airborne region over the smoke window.
    Mirrors the UC-40/41 smoke-train precedent (real SB3 PPO, committed connectome fixture, numpy
    ``simple`` adapter, CPU, seeded). The ~12h GPU retrain remains the user's, not a gate (AC8)."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    import torch
    from stable_baselines3 import PPO

    from drone_fly.train import loop as loop_mod
    from drone_fly.train.loop import build_policy_kwargs

    ecfg = EnvConfig()
    cfg = TrainConfig(seed=0)
    venv = build_vec_env(adapter="simple", n_envs=1, seed=0, training=True)
    try:
        model = PPO(
            "MlpPolicy",
            venv,
            n_steps=128,
            batch_size=64,
            n_epochs=10,
            ent_coef=cfg.ent_coef,
            seed=0,
            device="cpu",
            policy_kwargs=build_policy_kwargs(connectome, cfg),
        )
        loop_mod._apply_climb_bias(model)  # fresh-build init exactly as the training loop does

        std0 = float(torch.exp(model.policy.log_std.detach()).mean())
        assert std0 == pytest.approx(1.0, abs=1e-6)  # frozen at init before any update

        cb = AirborneStartCurriculumCallback(
            cfg,
            floor_z=ecfg.course.floor_z,
            high_z=ecfg.course.floor_z + ecfg.reward.altitude_target,
        )
        model.learn(total_timesteps=4096, callback=cb)

        # The curriculum raised the spawn into the airborne region over the (num_timesteps≈0) smoke
        # window — the effect AC8 asks the smoke to demonstrate.
        override = venv.get_attr("_spawn_z_override")[0]
        assert override is not None
        assert override > ecfg.course.floor_z + ecfg.early_termination.floor_epsilon

        log_std = model.policy.log_std.detach()
        assert torch.isfinite(log_std).all(), "log_std diverged to non-finite under training"
        std1 = float(torch.exp(log_std).mean())
        assert std1 > 0.1, (
            f"action std collapsed toward 0 (std {std0:.5f}→{std1:.5f}) over the smoke window — "
            f"the premature-collapse mode the curriculum must not trigger"
        )
    finally:
        venv.close()


# ============================================================================================
# UC-51 (AC5 airborne piece) — the warmup HOLD / start-delay reshapes the spawn schedule so the
# floor descent is the last, isolated stage. The airborne ``warmup_fraction`` holds the spawn fully
# airborne through the warmup, THEN anneals ``high_z`` -> ``floor_z`` across ``[warmup, anneal]``.
# (Contrast the collision curriculum, where ``warmup_fraction`` is the RAMP WIDTH.)
# ============================================================================================
def test_uc51_spawn_held_high_through_the_entire_warmup_window() -> None:
    """AC5: with a non-zero warmup the spawn is HELD at ``high_z`` across the whole warmup window
    ``[0, warmup_steps]`` — the descent does not begin until the warmup ends (the isolation the
    restagger buys: floor-takeoff comes last)."""
    cfg = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_warmup_fraction=0.6,
        airborne_curriculum_anneal_fraction=1.0,
    )
    warmup_steps = 600
    for t in (0, 1, 300, warmup_steps):  # inclusive of the warmup boundary
        assert spawn_z_at(t, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == pytest.approx(_HIGH_Z)
    # One step past the warmup the descent has begun (strictly below the high endpoint).
    assert spawn_z_at(warmup_steps + 1, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) < _HIGH_Z


def test_uc51_linear_descent_across_the_warmup_to_anneal_window() -> None:
    """AC5: between ``warmup_steps`` and ``anneal_steps`` the spawn anneals LINEARLY ``high_z`` ->
    ``floor_z`` — the ramp midpoint is the linear midpoint of the endpoints."""
    cfg = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_warmup_fraction=0.6,
        airborne_curriculum_anneal_fraction=1.0,
    )
    # ramp = [600, 1000]; midpoint t=800 ⇒ (high + floor) / 2 = 0.5.
    assert spawn_z_at(800, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == pytest.approx(0.5)
    # A quarter into the ramp (t=700) ⇒ 0.75 of the way from floor to high.
    assert spawn_z_at(700, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == pytest.approx(0.75)


def test_uc51_spawn_reaches_and_holds_floor_at_and_after_anneal_end() -> None:
    """AC5: at the anneal end (``anneal_fraction × total``) and forever after, the spawn is held at
    EXACTLY ``floor_z`` — the terminal state is byte-identical to the UC-37 floored start."""
    cfg = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_warmup_fraction=0.6,
        airborne_curriculum_anneal_fraction=1.0,
    )
    assert spawn_z_at(1000, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == _FLOOR_Z
    assert spawn_z_at(1001, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == _FLOOR_Z
    assert spawn_z_at(50_000, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == _FLOOR_Z


def test_uc51_warmup_schedule_is_monotone_non_increasing() -> None:
    """AC5: the warmup+anneal schedule is still monotone NON-INCREASING (holds, then only ever
    descends) — it never climbs back up."""
    cfg = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_warmup_fraction=0.6,
        airborne_curriculum_anneal_fraction=1.0,
    )
    samples = [spawn_z_at(t, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) for t in range(0, 1001, 25)]
    for earlier, later in zip(samples, samples[1:], strict=False):
        assert later <= earlier + 1e-12, f"schedule rose: {earlier} → {later}"


def test_uc51_warmup_past_anneal_raises() -> None:
    """AC5 edge / composition backstop: a warmup that runs PAST the anneal window
    (``warmup_fraction > anneal_fraction``) is a config error — the pure schedule raises
    ``ValueError`` (defence-in-depth behind the ``ConfigError`` at YAML load)."""
    cfg = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_warmup_fraction=0.7,
        airborne_curriculum_anneal_fraction=0.5,
    )
    with pytest.raises(ValueError, match="warmup"):
        spawn_z_at(0, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z)


def test_uc51_negative_warmup_raises() -> None:
    """AC5 edge: a negative warmup fraction is rejected by the pure schedule."""
    cfg = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_warmup_fraction=-0.1,
        airborne_curriculum_anneal_fraction=0.5,
    )
    with pytest.raises(ValueError, match="warmup"):
        spawn_z_at(0, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z)


def test_uc51_warmup_zero_is_byte_identical_to_the_pre_uc51_single_window_anneal() -> None:
    """AC5 (byte-identity, AC2 plumbing): ``warmup_fraction == 0.0`` reproduces the pre-UC-51
    single-window anneal from step 0 EXACTLY — the same linear ``high_z`` -> ``floor_z`` over
    ``[0, anneal_steps]`` the old ``spawn_z_at`` produced (so an explicit ``warmup: 0.0`` is a
    lossless opt-back-in to the old shape)."""
    cfg = TrainConfig(
        total_timesteps=1000,
        airborne_curriculum_warmup_fraction=0.0,
        airborne_curriculum_anneal_fraction=0.5,
    )
    anneal_steps = 500  # 0.5 × 1000

    def old_single_window(t: int) -> float:
        # The pre-UC-51 shape: linear from step 0 across [0, anneal_steps], clamped, floored after.
        frac = min(max(t / anneal_steps, 0.0), 1.0)
        return min(max(_HIGH_Z + (_FLOOR_Z - _HIGH_Z) * frac, _FLOOR_Z), _HIGH_Z)

    for t in (0, 1, 125, 250, 375, 499, 500, 501, 1000):
        assert spawn_z_at(t, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z) == pytest.approx(
            old_single_window(t)
        ), f"warmup=0 diverged from the old single-window anneal at t={t}"
