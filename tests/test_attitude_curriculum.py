"""UC-46 — attitude-authority curriculum: the pure schedule, its in-``RaceEnv.step`` scaling
(clip-then-scale, throttle never touched), the training-only plumbing, and the enable flag.

The tumbling relief is a *training-time* curriculum: it scales only the roll/pitch/yaw command
channels (indices 1, 2, 3 — never throttle) by an authority factor that starts low (so a noisy
early policy cannot flip the drone) and anneals up to full authority as training progresses. The
factor is pushed into each training env every rollout via
:meth:`drone_fly.env.racing_env.RaceEnv.set_attitude_authority`; eval and recording envs never
receive the callback, so they run at full authority (1.0) and measure true flight. No reward /
mixer / exploration change. This module pins:

* **AC1** — :func:`drone_fly.train.attitude_curriculum.attitude_authority_at`, the **pure**
  schedule: ``attitude_authority_start`` at fraction 0, EXACTLY ``1.0`` at/after the anneal end,
  monotone NON-DECREASING, clamped ``[start, 1.0]``, ``ValueError`` on an out-of-range anneal
  fraction AND on an out-of-range start, degenerate off-cases (anneal_fraction 0 / total ≤ 0 /
  disabled) → constant ``1.0``, and stateless/resume-correct.
* **AC2** — in-``RaceEnv.step`` scaling: ``set_attitude_authority(a)`` scales ONLY the rpy channels
  (1, 2, 3); throttle (0) is never scaled; the default (unset = 1.0) path is a byte-identical raw
  passthrough (pre-UC-46).
* **AC2b** — the saturated-input ordering lock: raw rpy = 5.0 at factor 0.25 yields an *effective*
  ``|rpy| = 0.25`` per channel (clip→1.0, then ×0.25). This PASSES under clip-then-scale and would
  FAIL under scale-then-clip (which would leak ``clip(1.25) = 1.0``) — a genuine ordering
  discriminator.
* **AC3** — training-only application: the callback pushes the scheduled factor through the REAL
  VecNormalize → VecMonitor → DummyVecEnv stack onto EVERY training env, while an eval/recording
  env (``training=False``, never given the callback) reports effective authority ``1.0`` regardless
  of the schedule.
* **AC4** — the enable flag: disabled → constant ``1.0`` (pre-UC-46 restored); the callback is
  resume-correct on both the fresh (t=0) and resumed (t>0) paths.
* **AC5** — a lightweight guard that the default env step path stays byte-identical to pre-UC-46
  (no reward / mixer disturbance), complementing the existing reward-math / recording / UC-44
  suites which must stay green unedited.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from drone_fly.adapter.base import DroneState
from drone_fly.env.config import EnvConfig
from drone_fly.env.racing_env import build_vec_env, make_env
from drone_fly.train.attitude_curriculum import (
    AttitudeAuthorityCurriculumCallback,
    attitude_authority_at,
)
from drone_fly.train.config import TrainConfig

_START = 0.25  # the default early authority (TrainConfig.attitude_authority_start)


# ============================================================================================
# AC1 — the pure schedule: endpoints, monotonicity, clamping, degenerate + out-of-range cases.
# ============================================================================================
def test_ac1_returns_start_at_fraction_zero() -> None:
    """AC1: at ``num_timesteps == 0`` the authority is the configured low start endpoint."""
    cfg = TrainConfig(
        total_timesteps=1000,
        attitude_authority_start=_START,
        attitude_authority_anneal_fraction=0.5,
    )
    assert attitude_authority_at(0, cfg) == pytest.approx(_START)


def test_ac1_returns_exactly_one_at_and_after_anneal_end() -> None:
    """AC1: at the anneal end (``anneal_fraction × total_timesteps``) and forever after, the
    authority is held at EXACTLY ``1.0`` — a strict ``==`` (not ``approx``) so the terminal / eval
    state is byte-identical to full authority (pre-UC-46)."""
    cfg = TrainConfig(
        total_timesteps=1000,
        attitude_authority_start=_START,
        attitude_authority_anneal_fraction=0.5,
    )
    anneal_steps = 500  # 0.5 × 1000
    assert attitude_authority_at(anneal_steps, cfg) == 1.0
    assert attitude_authority_at(anneal_steps + 1, cfg) == 1.0
    assert attitude_authority_at(10 * anneal_steps, cfg) == 1.0


def test_ac1_midpoint_is_linear_interpolation() -> None:
    """AC1: halfway through the anneal window the authority is the linear midpoint of the endpoints
    (``start`` → ``1.0``)."""
    cfg = TrainConfig(
        total_timesteps=1000,
        attitude_authority_start=_START,
        attitude_authority_anneal_fraction=0.5,
    )
    # anneal_steps = 500; halfway = 250 ⇒ start + (1 - start) × 0.5 = 0.25 + 0.375 = 0.625.
    assert attitude_authority_at(250, cfg) == pytest.approx(_START + (1.0 - _START) * 0.5)


def test_ac1_monotone_non_decreasing_across_the_anneal() -> None:
    """AC1: the schedule is monotone NON-DECREASING in ``num_timesteps`` (authority never drops back
    down as training progresses)."""
    cfg = TrainConfig(
        total_timesteps=1000,
        attitude_authority_start=_START,
        attitude_authority_anneal_fraction=0.5,
    )
    samples = [attitude_authority_at(t, cfg) for t in range(0, 1001, 25)]
    for earlier, later in zip(samples, samples[1:], strict=False):
        assert later >= earlier - 1e-12, f"schedule fell: {earlier} → {later}"


def test_ac1_clamped_to_start_one_band() -> None:
    """AC1: the result never dips below ``start`` nor overshoots ``1.0`` — a negative (pre-start)
    query clamps to the low start endpoint and a query past the anneal clamps to full authority."""
    cfg = TrainConfig(
        total_timesteps=1000,
        attitude_authority_start=_START,
        attitude_authority_anneal_fraction=0.5,
    )
    for t in (-100, -1, 0, 1, 123, 250, 499, 500, 501, 5000, 10_000_000):
        a = attitude_authority_at(t, cfg)
        assert _START <= a <= 1.0, f"t={t}: {a} escaped [{_START}, 1.0]"
    # A negative (pre-start) query is the lowest authority, clamped to the start endpoint.
    assert attitude_authority_at(-5, cfg) == pytest.approx(_START)


@pytest.mark.parametrize("bad_fraction", [-0.1, -1.0, 1.0001, 2.0])
def test_ac1_raises_on_out_of_range_anneal_fraction(bad_fraction: float) -> None:
    """AC1: an anneal fraction outside ``[0, 1]`` (a window that runs backwards or past the end of
    training) is a config error — the schedule raises ``ValueError`` rather than silently clamp."""
    cfg = TrainConfig(total_timesteps=1000, attitude_authority_anneal_fraction=bad_fraction)
    with pytest.raises(ValueError, match="anneal fraction"):
        attitude_authority_at(0, cfg)


@pytest.mark.parametrize("bad_start", [-0.1, -1.0, 1.0001, 2.0])
def test_ac1_raises_on_out_of_range_start(bad_start: float) -> None:
    """AC1: an ``attitude_authority_start`` outside ``[0, 1]`` is a config error — a negative start
    would invert the command channels, a start above 1.0 would amplify commands beyond the
    sanitized envelope. Fail loud on either rather than silently clamp."""
    cfg = TrainConfig(
        total_timesteps=1000,
        attitude_authority_start=bad_start,
        attitude_authority_anneal_fraction=0.5,
    )
    with pytest.raises(ValueError, match="start"):
        attitude_authority_at(0, cfg)


def test_ac1_boundary_fractions_and_starts_are_accepted() -> None:
    """AC1: the inclusive endpoints ``0`` and ``1`` are valid for BOTH the anneal fraction (0 ⇒
    curriculum off; 1 ⇒ anneal over the whole run) and the start (0 ⇒ zero early authority; 1 ⇒
    already full). None raises."""
    # anneal fraction boundaries
    off = TrainConfig(
        total_timesteps=1000,
        attitude_authority_start=_START,
        attitude_authority_anneal_fraction=0.0,
    )
    full = TrainConfig(
        total_timesteps=1000,
        attitude_authority_start=_START,
        attitude_authority_anneal_fraction=1.0,
    )
    assert attitude_authority_at(0, off) == 1.0  # fraction 0 ⇒ curriculum off ⇒ full authority
    assert attitude_authority_at(0, full) == pytest.approx(_START)
    assert attitude_authority_at(1000, full) == 1.0
    # start boundaries
    start0 = TrainConfig(
        total_timesteps=1000, attitude_authority_start=0.0, attitude_authority_anneal_fraction=0.5
    )
    start1 = TrainConfig(
        total_timesteps=1000, attitude_authority_start=1.0, attitude_authority_anneal_fraction=0.5
    )
    assert attitude_authority_at(0, start0) == pytest.approx(0.0)
    assert attitude_authority_at(0, start1) == pytest.approx(1.0)  # already full ⇒ constant


def test_ac1_degenerate_zero_fraction_is_constant_full_authority() -> None:
    """AC1: ``anneal_fraction == 0`` degenerates to constant ``1.0`` at every step — the curriculum
    is off (no-op, byte-identical to pre-UC-46)."""
    cfg = TrainConfig(
        total_timesteps=1000,
        attitude_authority_start=_START,
        attitude_authority_anneal_fraction=0.0,
    )
    for t in (0, 1, 500, 1000, 50_000):
        assert attitude_authority_at(t, cfg) == 1.0


def test_ac1_degenerate_nonpositive_total_timesteps_is_full_authority() -> None:
    """AC1: a non-positive ``total_timesteps`` (no real anneal window) degenerates to constant
    ``1.0`` — the schedule can't divide by a zero/negative horizon, so it turns the curriculum
    off."""
    cfg = TrainConfig(
        total_timesteps=0, attitude_authority_start=_START, attitude_authority_anneal_fraction=0.5
    )
    assert attitude_authority_at(0, cfg) == 1.0


def test_ac1_disabled_flag_is_constant_full_authority() -> None:
    """AC1/AC4: with ``attitude_authority_curriculum_enabled=False`` the schedule is constant
    ``1.0`` regardless of ``num_timesteps`` — the disable path restores pre-UC-46 behaviour."""
    cfg = TrainConfig(
        total_timesteps=1000,
        attitude_authority_start=_START,
        attitude_authority_anneal_fraction=0.5,
        attitude_authority_curriculum_enabled=False,
    )
    for t in (0, 1, 250, 500, 1000, 50_000):
        assert attitude_authority_at(t, cfg) == 1.0


def test_ac1_is_stateless_and_resume_correct() -> None:
    """AC1: the schedule is a pure function of ``num_timesteps`` (+ cfg) — the same query returns
    the same value regardless of call order, so a resumed run continues correctly."""
    cfg = TrainConfig(
        total_timesteps=1000,
        attitude_authority_start=_START,
        attitude_authority_anneal_fraction=0.5,
    )
    first = [attitude_authority_at(t, cfg) for t in (0, 250, 500, 750)]
    # Query out of order; each value must be identical to the in-order query (no hidden state).
    assert attitude_authority_at(750, cfg) == first[3]
    assert attitude_authority_at(0, cfg) == first[0]
    assert attitude_authority_at(500, cfg) == first[2]


# ============================================================================================
# AC2 — in-``RaceEnv.step`` scaling: only rpy scaled, throttle untouched, default = raw passthrough.
# ============================================================================================
class _RecordingAdapter:
    """A stand-in adapter that records the exact action array handed to ``step`` (the *effective*,
    post-scaling command that reaches the sim) and returns a fixed benign airborne state. Lets a
    test assert precisely which channels ``RaceEnv.step`` scaled before the adapter saw them."""

    backend = "recording"

    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []

    def _state(self) -> DroneState:
        return DroneState(
            position=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            velocity=np.zeros(3),
            attitude=np.zeros(3),
            angular_velocity=np.zeros(3),
            collided=False,
        )

    def reset(self, seed=None) -> DroneState:
        return self._state()

    def step(self, action) -> DroneState:
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        return self._state()

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def _env_with_recording(authority: float | None = None):
    """Build a ``RaceEnv`` whose adapter is a recording spy, optionally with an attitude-authority
    override set. Returns ``(env, spy)``; ``spy.actions[-1]`` is the last command the sim saw."""
    env = make_env(EnvConfig(), adapter="simple")
    spy = _RecordingAdapter()
    env.adapter = spy
    env.backend = spy.backend
    if authority is not None:
        env.set_attitude_authority(authority)
    env.reset(seed=0)
    return env, spy


def test_ac2_default_unset_is_byte_identical_raw_passthrough() -> None:
    """AC2: an env with the default (unset) authority ``1.0`` hands the adapter the RAW action
    verbatim — the pre-UC-46 ``adapter.step(np.asarray(action, float64))`` path. Proven with an
    OUT-OF-BOX action: it reaches the sim UNCLIPPED (no sanitize on the default path), i.e. exactly
    ``np.asarray(action, float64)`` — byte-identical to before UC-46."""
    env, spy = _env_with_recording(authority=None)
    assert env._attitude_authority == 1.0
    action = np.array([0.7, 5.0, -5.0, 0.3], dtype=np.float64)  # rpy out of the [-1, 1] box
    env.step(action)
    assert np.array_equal(spy.actions[-1], np.asarray(action, dtype=np.float64))


def test_ac2_scales_only_rpy_channels_throttle_untouched() -> None:
    """AC2: ``set_attitude_authority(factor)`` scales ONLY the roll/pitch/yaw channels (indices 1,
    2, 3); the throttle channel (index 0) is never scaled. In-box action so sanitize is a no-op and
    the scaling is isolated."""
    factor = 0.5
    env, spy = _env_with_recording(authority=factor)
    action = np.array([0.8, 0.6, -0.4, 0.2], dtype=np.float64)  # all in the canonical box
    env.step(action)
    seen = spy.actions[-1]
    assert seen[0] == pytest.approx(0.8)  # throttle NOT scaled
    assert seen[1] == pytest.approx(0.6 * factor)
    assert seen[2] == pytest.approx(-0.4 * factor)
    assert seen[3] == pytest.approx(0.2 * factor)


def test_ac2b_saturated_input_ordering_lock_clip_then_scale() -> None:
    """AC2b (ordering discriminator): raw rpy = ±5.0 at factor 0.25 must reach the sim as an
    *effective* ``|rpy| = 0.25`` per channel — ``sanitize_action`` clips 5.0 → 1.0 FIRST, THEN the
    ×0.25 scaling is applied (clip-then-scale), so the authority is a STRUCTURAL cap that holds even
    for saturated commands (the exact ones that flip the drone). Under scale-then-clip the effective
    value would be ``clip(5.0 × 0.25) = clip(1.25) = 1.0`` — full authority leaking through — so
    this test PASSES only for the correct ordering. Throttle (0.7, in box) is untouched."""
    factor = 0.25
    env, spy = _env_with_recording(authority=factor)
    action = np.array([0.7, 5.0, -5.0, 3.0], dtype=np.float64)  # rpy saturated well past ±1
    env.step(action)
    seen = spy.actions[-1]
    assert seen[0] == pytest.approx(0.7)  # throttle untouched
    # clip-then-scale ⇒ |rpy| == 0.25 exactly (NOT 1.0, which scale-then-clip would leak).
    assert seen[1] == pytest.approx(0.25)
    assert seen[2] == pytest.approx(-0.25)
    assert seen[3] == pytest.approx(0.25)
    assert abs(seen[1]) < 0.5 and abs(seen[2]) < 0.5 and abs(seen[3]) < 0.5, (
        "effective |rpy| leaked toward full authority — scale-then-clip ordering regression"
    )


def test_ac2_scaled_command_stays_in_the_canonical_box() -> None:
    """AC2 (pitfall): the scaled attitude command still lies inside the canonical ``[-1, 1]`` box —
    scaling a clipped value by a factor ≤ 1 can only shrink it, so it never re-escapes the
    envelope; throttle stays in ``[0, 1]``."""
    env, spy = _env_with_recording(authority=0.75)
    env.step(np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float64))
    seen = spy.actions[-1]
    assert 0.0 <= seen[0] <= 1.0
    assert np.all(seen[1:4] >= -1.0) and np.all(seen[1:4] <= 1.0)


# ============================================================================================
# AC3 — training-only application: pushed onto EVERY training env; eval / no-callback stay at 1.0.
# ============================================================================================
def _fake_model(env):
    """A minimal stand-in exposing only what ``BaseCallback`` needs: ``get_env`` for
    ``training_env``. ``num_timesteps`` is a plain attribute set on the callback itself (mirrors the
    airborne-curriculum test's helper)."""
    return types.SimpleNamespace(get_env=lambda: env)


def test_ac3_callback_pushes_scheduled_factor_through_wrapper_stack() -> None:
    """AC3: ``AttitudeAuthorityCurriculumCallback`` computes the scheduled factor for the current
    ``num_timesteps`` and pushes it — via ``training_env.env_method('set_attitude_authority', a)`` —
    all the way down to each base ``RaceEnv`` THROUGH the VecNormalize → VecMonitor → DummyVecEnv
    stack. Verified by reading ``_attitude_authority`` back through the same wrappers."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    venv = build_vec_env(adapter="simple", n_envs=1, training=True, seed=0)
    try:
        # Confirm we really are exercising the full wrapper stack (not a bare env).
        assert isinstance(venv, VecNormalize)
        assert isinstance(venv.venv, VecMonitor)
        assert isinstance(venv.venv.venv, DummyVecEnv)

        cfg = TrainConfig(
            total_timesteps=1000,
            attitude_authority_start=_START,
            attitude_authority_anneal_fraction=0.5,
        )
        cb = AttitudeAuthorityCurriculumCallback(cfg)
        cb.init_callback(_fake_model(venv))

        # Start of training (t=0): the low start authority reaches the base env through the stack.
        cb.num_timesteps = 0
        cb._on_training_start()
        assert venv.get_attr("_attitude_authority") == [pytest.approx(_START)]

        # Mid-anneal on a later rollout: the interpolated factor propagates the same way.
        cb.num_timesteps = 250
        cb._on_rollout_start()
        assert venv.get_attr("_attitude_authority") == [
            pytest.approx(_START + (1.0 - _START) * 0.5)
        ]

        # At/after the anneal end the pushed value is exactly full authority.
        cb.num_timesteps = 500
        cb._on_rollout_start()
        assert venv.get_attr("_attitude_authority") == [1.0]
    finally:
        venv.close()


def test_ac3_callback_propagates_to_all_envs_in_a_multi_env_stack() -> None:
    """AC3: with more than one base env the pushed factor reaches EVERY env — ``env_method`` fans
    out across the whole DummyVecEnv, not just env 0."""
    venv = build_vec_env(adapter="simple", n_envs=3, training=True, seed=0)
    try:
        cfg = TrainConfig(
            total_timesteps=1000,
            attitude_authority_start=_START,
            attitude_authority_anneal_fraction=0.5,
        )
        cb = AttitudeAuthorityCurriculumCallback(cfg)
        cb.init_callback(_fake_model(venv))
        cb.num_timesteps = 0
        cb._on_training_start()
        assert venv.get_attr("_attitude_authority") == [pytest.approx(_START)] * 3
    finally:
        venv.close()


def test_ac3_base_raceenv_without_callback_is_full_authority() -> None:
    """AC3: a base ``RaceEnv`` that never received the curriculum callback keeps the default full
    authority (1.0) — the flight-measurement guarantee at the single-env level."""
    env = make_env(EnvConfig(), adapter="simple")
    assert env._attitude_authority == 1.0


def test_ac3_eval_vec_env_is_full_authority_regardless_of_schedule() -> None:
    """AC3: the eval/recording path — ``build_vec_env(training=False)``, which the evaluator uses
    and which is NEVER handed the curriculum callback — reports effective attitude authority 1.0, so
    true flight is measured regardless of the (training) schedule. Contrast: a sibling TRAINING venv
    driven by the callback at the same ``num_timesteps`` sits at the reduced start authority."""
    eval_venv = build_vec_env(adapter="simple", n_envs=1, training=False, seed=0)
    train_venv = build_vec_env(adapter="simple", n_envs=1, training=True, seed=0)
    try:
        # Eval env carries full authority with no callback ever applied.
        assert eval_venv.get_attr("_attitude_authority") == [1.0]

        # Drive the schedule hard on the TRAINING venv only; the eval venv must be unaffected.
        cfg = TrainConfig(
            total_timesteps=1000,
            attitude_authority_start=_START,
            attitude_authority_anneal_fraction=0.5,
        )
        cb = AttitudeAuthorityCurriculumCallback(cfg)
        cb.init_callback(_fake_model(train_venv))
        cb.num_timesteps = 0
        cb._on_training_start()
        assert train_venv.get_attr("_attitude_authority") == [pytest.approx(_START)]
        # Eval env is STILL full authority — the curriculum cannot leak into the measurement.
        assert eval_venv.get_attr("_attitude_authority") == [1.0]
    finally:
        eval_venv.close()
        train_venv.close()


# ============================================================================================
# AC4 — the enable flag + fresh/resume wiring correctness.
# ============================================================================================
def test_ac4_disabled_callback_pushes_constant_full_authority() -> None:
    """AC4: with the curriculum DISABLED, the callback pushes constant ``1.0`` regardless of
    ``num_timesteps`` — pre-UC-46 behaviour is fully restored even if the callback is (harmlessly)
    attached."""
    venv = build_vec_env(adapter="simple", n_envs=1, training=True, seed=0)
    try:
        cfg = TrainConfig(
            total_timesteps=1000,
            attitude_authority_start=_START,
            attitude_authority_anneal_fraction=0.5,
            attitude_authority_curriculum_enabled=False,
        )
        cb = AttitudeAuthorityCurriculumCallback(cfg)
        cb.init_callback(_fake_model(venv))
        for t in (0, 250, 500):
            cb.num_timesteps = t
            cb._on_rollout_start()
            assert venv.get_attr("_attitude_authority") == [1.0], (
                f"leaked reduced authority at t={t}"
            )
    finally:
        venv.close()


def test_ac4_callback_is_resume_correct_on_fresh_and_resumed_paths() -> None:
    """AC4: the callback is stateless in ``num_timesteps`` — a FRESH run (t=0) pushes the start
    authority and a RESUMED run (t continues from a checkpoint, e.g. 250) picks the schedule up at
    exactly the right point, with no extra bookkeeping. This is what makes the single callback list
    correct on both the fresh and resume ``model.learn`` paths."""
    venv = build_vec_env(adapter="simple", n_envs=1, training=True, seed=0)
    try:
        cfg = TrainConfig(
            total_timesteps=1000,
            attitude_authority_start=_START,
            attitude_authority_anneal_fraction=0.5,
        )
        # Fresh path: a brand-new callback at t=0 → start authority.
        fresh = AttitudeAuthorityCurriculumCallback(cfg)
        fresh.init_callback(_fake_model(venv))
        fresh.num_timesteps = 0
        fresh._on_training_start()
        assert venv.get_attr("_attitude_authority") == [pytest.approx(_START)]

        # Resume path: a fresh callback whose num_timesteps already reflects the checkpoint (250)
        # lands on the mid-anneal value immediately at _on_training_start — no replay needed.
        resumed = AttitudeAuthorityCurriculumCallback(cfg)
        resumed.init_callback(_fake_model(venv))
        resumed.num_timesteps = 250
        resumed._on_training_start()
        assert venv.get_attr("_attitude_authority") == [
            pytest.approx(_START + (1.0 - _START) * 0.5)
        ]
    finally:
        venv.close()


# ============================================================================================
# AC5 — the default env step path stays byte-identical to pre-UC-46 (no reward/mixer disturbance).
# ============================================================================================
def test_ac5_default_step_path_is_unchanged_for_a_full_authority_env() -> None:
    """AC5: at the default full authority (1.0) the ``!= 1.0`` guard takes the straight
    ``adapter.step(np.asarray(action, float64))`` branch — the reward/termination machinery sees the
    same command as pre-UC-46. A fixed-seed episode under the real simple adapter is reproducible
    across two identical envs (no hidden per-step state introduced by the guard)."""
    env_a = make_env(EnvConfig(), adapter="simple")
    env_b = make_env(EnvConfig(), adapter="simple")
    env_a.reset(seed=7)
    env_b.reset(seed=7)
    hover = np.array([0.5, 0.1, -0.1, 0.05], dtype=np.float32)
    for _ in range(25):
        oa, ra, ta, tra, _ia = env_a.step(hover)
        ob, rb, tb, trb, _ib = env_b.step(hover)
        assert np.array_equal(oa, ob)
        assert ra == rb
        assert ta == tb and tra == trb
        if ta or tra:
            break
