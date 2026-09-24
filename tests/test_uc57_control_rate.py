"""UC-57 — raise & decouple the control frequency (policy Hz vs inner-loop/physics Hz).

Fully hermetic (AC7): no pybullet, no GPU, no training. The behavioural verdict (does a 50 Hz
policy actually fly better) is the owner's fresh GPU retrain — out of scope here. What IS tested is
the *plumbing* that UC-57 adds, exactly the surfaces the plan enumerates:

* the pure timing helpers in :mod:`drone_fly.env.timing` — rate↔dt conversion, composite step-budget
  scaling (episode SECONDS invariance), latency ms→steps conversion (standing + sampled combine),
  the decoupling ratio / inner-dt / iteration-count, and the CtrlAviary frequency resolution
  (pyb_freq an exact multiple of ctrl_freq; ratio=1 byte-identity);
* the decoupled inner control loop
  (:func:`drone_fly.adapter.pybullet_adapter.run_inner_control_loop`) driven with injected fakes —
  the PID runs ``physics_ratio`` times per policy action at the finer ``inner_dt``, and collision
  OR-latches + breaks on the first collided inner tick;
* the command-latency FIFO — counted in POLICY steps, converted at the POLICY dt, and sitting
  UPSTREAM of the inner loop (asserted via the simple backend + the resolver);
* the reward ``per_step_scale`` episode-integral invariance;
* the run-layer 50 Hz default and train/eval symmetry;
* YAML validation + exposure of ``control_hz`` / ``physics_ratio`` / ``command_latency_ms``.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.adapter.base import DroneState
from drone_fly.adapter.pybullet_adapter import run_inner_control_loop
from drone_fly.adapter.rate_controller import RateControllerConfig
from drone_fly.adapter.simple import SimpleDroneAdapter
from drone_fly.config import ConfigError, EvaluateRunConfig, TrainRunConfig
from drone_fly.env.timing import (
    BASELINE_DT,
    dt_from_hz,
    hz_from_dt,
    inner_dt,
    inner_rate,
    iteration_count,
    pybullet_freqs,
    resolve_latency_steps,
    scale_step_budget,
)


# =========================================================================== #
# Timing helpers — rate ↔ dt conversion (AC1)
# =========================================================================== #
def test_baseline_dt_is_the_20hz_constant() -> None:
    """The single source of the baseline: 20 Hz ⇒ dt = 0.05 s."""
    assert BASELINE_DT == 0.05


def test_dt_hz_roundtrip() -> None:
    """AC1: ``dt = 1/hz`` and ``hz = 1/dt`` are exact inverses at the two rates of interest."""
    assert dt_from_hz(20.0) == pytest.approx(0.05)
    assert dt_from_hz(50.0) == pytest.approx(0.02)
    assert hz_from_dt(0.05) == pytest.approx(20.0)
    assert hz_from_dt(0.02) == pytest.approx(50.0)
    for hz in (20.0, 33.0, 50.0, 100.0):
        assert hz_from_dt(dt_from_hz(hz)) == pytest.approx(hz)


# =========================================================================== #
# scale_step_budget — episode SECONDS invariance (AC3)
# =========================================================================== #
def test_scale_step_budget_is_identity_at_baseline() -> None:
    """AC3 byte-identity: at the 20 Hz baseline the budget is unchanged (the identity map)."""
    for budget in (400, 800, 2200, 1):
        assert scale_step_budget(budget, BASELINE_DT) == budget


def test_scale_step_budget_scales_inversely_with_dt() -> None:
    """AC3: 50 Hz (dt=0.02) grants 2.5× the baseline steps — 400 → 1000, 800 → 2000."""
    assert scale_step_budget(400, 0.02) == 1000
    assert scale_step_budget(800, 0.02) == 2000


def test_scale_step_budget_keeps_seconds_invariant() -> None:
    """AC3 core property: ``steps · dt`` (episode seconds) is invariant across the rate change."""
    baseline_seconds = 800 * BASELINE_DT
    for hz in (20.0, 25.0, 50.0, 100.0):
        dt = dt_from_hz(hz)
        steps = scale_step_budget(800, dt)
        assert steps * dt == pytest.approx(baseline_seconds, rel=1e-3)


# =========================================================================== #
# resolve_latency_steps — Hz-invariant latency, single authoritative site (AC4)
# =========================================================================== #
def test_latency_100ms_reproduces_2_steps_at_20hz() -> None:
    """AC4 conversion property: a 100 ms command reproduces the historical fixed 2-step latency at
    20 Hz — and 5 steps at 50 Hz (same real-time delay, more steps at the higher rate)."""
    assert resolve_latency_steps(0.05, command_latency_ms=100.0) == 2
    assert resolve_latency_steps(0.02, command_latency_ms=100.0) == 5


def test_latency_default_zero_is_byte_identical() -> None:
    """AC4/AC7: the default (0 ms standing, no sampled baseline) resolves to 0 steps ⇒ no FIFO ⇒
    byte-identical to the pre-UC-57 no-latency path."""
    assert resolve_latency_steps(0.05, command_latency_ms=0.0) == 0
    assert resolve_latency_steps(0.02, command_latency_ms=0.0) == 0


def test_latency_sampled_baseline_reduces_to_the_sampled_int_at_baseline() -> None:
    """Byte-identity: with 0 ms standing latency at the 20 Hz baseline, a per-episode sampled
    baseline latency (drawn in 20 Hz steps) passes through as exactly that integer — matching the
    pre-UC-57 fixed-step handling."""
    for sampled in (0, 1, 2, 3):
        assert (
            resolve_latency_steps(
                0.05, command_latency_ms=0.0, sampled_latency_steps_baseline=sampled
            )
            == sampled
        )


def test_latency_combines_standing_and_sampled_as_durations_one_round() -> None:
    """AC4: the standing ms latency and the sampled-baseline-step latency are summed as DURATIONS
    and converted with a SINGLE round (never double-rounded). 100 ms + 2 baseline steps at 50 Hz:
    (0.1 + 2·0.05) / 0.02 = 0.2/0.02 = 10 steps."""
    assert (
        resolve_latency_steps(0.02, command_latency_ms=100.0, sampled_latency_steps_baseline=2)
        == 10
    )
    # Single-round guard: the two sources are summed as durations THEN rounded once. 10 ms + 1
    # baseline step at 50 Hz: (0.01 + 0.05)/0.02 = 0.06/0.02 = 3. A per-source round would instead
    # give round(0.01/0.02)=round(0.5)=0 (banker's) + round(0.05/0.02)=2 = 2 — a different result,
    # proving the conversion sums-then-rounds rather than rounding each addend independently.
    assert (
        resolve_latency_steps(0.02, command_latency_ms=10.0, sampled_latency_steps_baseline=1) == 3
    )


def test_latency_is_converted_at_policy_dt_not_inner_dt() -> None:
    """Plan nuance #2: the CTBR command stream updates at the POLICY rate and the FIFO sits upstream
    of the inner loop, so latency is resolved at the POLICY dt — NOT the (finer) inner dt. A 100 ms
    latency at 50 Hz is 5 POLICY steps, regardless of physics_ratio."""
    policy_dt = dt_from_hz(50.0)
    assert resolve_latency_steps(policy_dt, command_latency_ms=100.0) == 5
    # If it were (wrongly) resolved at the inner dt (physics_ratio=10 ⇒ inner_dt 0.002) it would be
    # 50 — so the correct policy-dt resolution is an order of magnitude smaller.
    assert resolve_latency_steps(inner_dt(policy_dt, 10), command_latency_ms=100.0) == 50


# =========================================================================== #
# Decoupling: inner_rate / inner_dt / iteration_count (AC2)
# =========================================================================== #
def test_inner_rate_and_ratio_at_baseline_are_single_rate() -> None:
    """AC2 byte-identity: at ``(20 Hz, ratio 1)`` the inner rate is 20 Hz, inner_dt == dt, and
    there is exactly one inner iteration — the pre-UC-57 single-rate configuration."""
    assert inner_rate(20.0, 1) == 20
    assert inner_dt(0.05, 1) == 0.05
    assert iteration_count(1) == 1


def test_inner_rate_and_dt_decouple_at_50hz_ratio_10() -> None:
    """AC2: 50 Hz policy × ratio 10 ⇒ 500 Hz inner loop, inner_dt = 0.002 s, 10 iterations."""
    assert inner_rate(50.0, 10) == 500
    assert inner_dt(0.02, 10) == pytest.approx(0.002)
    assert iteration_count(10) == 10


def test_inner_dt_is_reciprocal_of_inner_rate() -> None:
    """Consistency: ``inner_dt`` equals ``1 / inner_rate`` (up to rounding in inner_rate)."""
    for hz, ratio in ((20.0, 1), (50.0, 10), (50.0, 4), (33.0, 3)):
        dt = dt_from_hz(hz)
        assert inner_dt(dt, ratio) == pytest.approx(1.0 / inner_rate(hz, ratio), rel=1e-9)


# =========================================================================== #
# pybullet_freqs — CtrlAviary frequency resolution (AC2)
# =========================================================================== #
def test_pybullet_freqs_baseline_is_byte_identical() -> None:
    """AC2 byte-identity: ``(20, 1)`` ⇒ ctrl 20 / pyb 240 / 12 substeps — today's CtrlAviary."""
    ctrl, pyb, mult = pybullet_freqs(20.0, 1)
    assert (ctrl, pyb, mult) == (20, 240, 12)


def test_pybullet_freqs_pyb_is_always_a_multiple_of_ctrl() -> None:
    """AC2: ``pyb_freq`` is an exact integer multiple of ``ctrl_freq`` for every rate/ratio, so
    pybullet's ``pyb_freq % ctrl_freq == 0`` guard always holds (fixes the latent divisibility bug
    in the old ``max(ctrl·4, 240)`` form). Also ``pyb_freq ≥ 240`` and ``pyb_freq ≥ 4·ctrl``."""
    for hz in (20.0, 33.0, 50.0, 70.0, 100.0):
        for ratio in (1, 2, 4, 10):
            ctrl, pyb, mult = pybullet_freqs(hz, ratio)
            assert ctrl == inner_rate(hz, ratio)
            assert pyb == ctrl * mult
            assert pyb % ctrl == 0
            assert pyb >= 240
            assert pyb >= 4 * ctrl


def test_pybullet_freqs_odd_ctrl_freq_still_divides() -> None:
    """Regression: an inner rate that does NOT divide 240 (e.g. 70 Hz) still yields a pyb_freq that
    is a clean multiple of ctrl_freq — the old ``max(ctrl·4, 240)`` = 280? no, 280 (4×70) here, but
    e.g. 50 Hz gave max(200,240)=240 which is NOT a multiple of 50. The new form always divides."""
    ctrl, pyb, _ = pybullet_freqs(50.0, 1)
    assert pyb % ctrl == 0  # old form: 240 % 50 == 40 (the latent bug); new: exact multiple


# =========================================================================== #
# Decoupled inner control loop — driven with injected fakes (AC2)
# =========================================================================== #
class _RecordingController:
    """A fake rate controller that records the ``dt`` of every PID update (to assert inner_dt)."""

    def __init__(self) -> None:
        self.dts: list[float] = []
        self.setpoints: list[np.ndarray] = []

    def update(self, setpoint, measured, dt):
        self.dts.append(float(dt))
        self.setpoints.append(np.asarray(setpoint, dtype=np.float64).copy())
        return np.zeros(3, dtype=np.float64)


def _run_loop(*, iterations, i_dt, collide_at=None):
    """Drive ``run_inner_control_loop`` with fakes; return (state, collided, controller, counters).

    ``read_gyro`` / ``step_physics`` count their calls; ``step_physics`` reports collision on the
    inner tick index ``collide_at`` (0-based) when given.
    """
    counters = {"gyro": 0, "physics": 0}

    def read_gyro():
        counters["gyro"] += 1
        return np.zeros(3, dtype=np.float64)

    def step_physics(rpm):
        i = counters["physics"]
        counters["physics"] += 1
        collided = collide_at is not None and i == collide_at
        # Encode the tick index into the position so we can prove WHICH tick's state is returned.
        state = DroneState(
            np.array([float(i), 0.0, 0.0]), np.zeros(3), np.zeros(3), np.zeros(3), collided
        )
        return state, collided

    controller = _RecordingController()
    state, collided = run_inner_control_loop(
        np.array([0.5, 0.1, -0.2, 0.3]),
        iterations=iterations,
        inner_dt=i_dt,
        hover_rpm=1000.0,
        max_rpm=2000.0,
        rate_config=RateControllerConfig(),
        rate_controller=controller,
        read_gyro=read_gyro,
        step_physics=step_physics,
    )
    return state, collided, controller, counters


def test_inner_loop_runs_physics_ratio_iterations_at_inner_dt() -> None:
    """AC2: for one policy action the loop runs exactly ``physics_ratio`` inner ticks, each reading
    the gyro once and stepping physics once, and the PID integrates at the finer ``inner_dt``."""
    ratio, i_dt = 10, inner_dt(0.02, 10)  # 50 Hz policy, 500 Hz inner
    _state, collided, controller, counters = _run_loop(iterations=ratio, i_dt=i_dt)
    assert not collided
    assert counters["gyro"] == ratio
    assert counters["physics"] == ratio
    assert len(controller.dts) == ratio
    assert all(d == pytest.approx(i_dt) for d in controller.dts)  # PID at inner_dt, not policy dt


def test_inner_loop_is_single_tick_at_ratio_one() -> None:
    """AC2 byte-identity: at ``physics_ratio == 1`` there is exactly one gyro read, one PID update
    at ``dt``, one physics step — identical to the pre-UC-57 single physics tick."""
    _state, collided, controller, counters = _run_loop(iterations=1, i_dt=0.05)
    assert not collided
    assert counters["gyro"] == 1
    assert counters["physics"] == 1
    assert controller.dts == [pytest.approx(0.05)]


def test_inner_loop_holds_the_setpoint_across_inner_ticks() -> None:
    """AC2: the policy decides ONCE per policy step — the body-rate setpoint is constant across all
    inner ticks (the connectome policy runs at ``control_hz``, only the PID/physics run faster)."""
    ratio = 5
    _state, _collided, controller, _counters = _run_loop(iterations=ratio, i_dt=inner_dt(0.02, 5))
    assert len(controller.setpoints) == ratio
    first = controller.setpoints[0]
    for sp in controller.setpoints[1:]:
        np.testing.assert_array_equal(sp, first)


def test_inner_loop_collision_or_latches_and_breaks_on_first_collided_tick() -> None:
    """AC2 (collision semantics): a collision on an inner tick OR-latches the policy-step collided
    flag AND breaks the loop immediately, returning THAT tick's state — so the env's crash-speed
    proxy / dock / crash classifiers see the contact instant, not a settled post-contact pose."""
    ratio = 10
    collide_at = 3  # 4th inner tick collides
    state, collided, _controller, counters = _run_loop(
        iterations=ratio, i_dt=inner_dt(0.02, 10), collide_at=collide_at
    )
    assert collided is True
    # Broke on the first collided tick: only collide_at+1 physics steps ran (not all 10).
    assert counters["physics"] == collide_at + 1
    assert counters["gyro"] == collide_at + 1
    # Returned state is the collided tick's state (position encodes the tick index).
    assert state.position[0] == float(collide_at)
    assert state.collided is True


# =========================================================================== #
# Command-latency FIFO — POLICY steps, upstream of the inner loop (simple backend)
# =========================================================================== #
def _simple(latency_steps: int = 0) -> SimpleDroneAdapter:
    return SimpleDroneAdapter(
        np.array([0.0, 0.0, 1.0]),
        floor_z=0.0,
        ceiling_z=2.5,
        dt=0.02,  # 50 Hz policy dt
        command_latency_steps=latency_steps,
    )


def test_fifo_delays_command_stream_by_exactly_n_policy_steps() -> None:
    """Plan nuance #2: the FIFO is counted in POLICY steps — a latency-N simple adapter applying
    ``[a0, a1, ...]`` equals a latency-0 adapter applying ``[hover]*N ++ [a0, a1, ...]`` (the
    applied stream is shifted by exactly N policy steps, warming up with hover)."""
    warm = np.array([0.5, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(11)
    actions = [rng.uniform([0.2, -0.5, -0.5, -0.5], [1.0, 0.5, 0.5, 0.5]) for _ in range(20)]

    for n in (1, 2, 5):
        delayed = _simple(latency_steps=n)
        delayed.reset(seed=1)
        delayed_pos = [delayed.step(a).position.copy() for a in actions]

        ref = _simple(latency_steps=0)
        ref.reset(seed=1)
        combined = [warm] * n + list(actions)
        ref_pos = [ref.step(a).position.copy() for a in combined]

        for i, d in enumerate(delayed_pos):
            np.testing.assert_array_equal(d, ref_pos[i], err_msg=f"latency={n} mismatch at {i}")


def test_fifo_zero_latency_is_byte_identical() -> None:
    """AC7: a zero-latency FIFO is bit-identical to the no-buffer passthrough."""
    rng = np.random.default_rng(12)
    actions = [rng.uniform([0.2, -0.5, -0.5, -0.5], [1.0, 0.5, 0.5, 0.5]) for _ in range(20)]
    a = _simple(latency_steps=0)
    a.reset(seed=1)
    pos_a = [a.step(x).position.copy() for x in actions]
    # A fresh adapter constructed with no latency kwarg at all must match.
    b = SimpleDroneAdapter(np.array([0.0, 0.0, 1.0]), floor_z=0.0, ceiling_z=2.5, dt=0.02)
    b.reset(seed=1)
    pos_b = [b.step(x).position.copy() for x in actions]
    for x, y in zip(pos_a, pos_b, strict=True):
        np.testing.assert_array_equal(x, y)


# =========================================================================== #
# Reward per_step_scale — episode-integral invariance (AC5)  [cross-checks test_reward.py]
# =========================================================================== #
def test_reward_per_step_scale_integral_is_rate_invariant() -> None:
    """AC5: the reward's ``per_step_scale`` (= dt/BASELINE_DT) makes the summed per-step terms
    invariant over a fixed-SECONDS episode. Here, the isolated scaled contribution (vs scale=0)
    times the step count is equal at 20 Hz (800 steps × 1.0) and 50 Hz (2000 steps × 0.4)."""
    from drone_fly.env.config import RewardConfig
    from drone_fly.env.reward import compute_reward

    cfg = RewardConfig(enable_altitude_decoupling=False, altitude_hold_weight=0.0)
    h = cfg.climb_target_height

    def scaled(scale):
        kw = dict(
            dist_to_target_prev=0.0,
            dist_to_target_curr=0.0,
            event=None,
            collided=False,
            completed=False,
            cfg=cfg,
            airborne=True,
            height_above_floor_prev=h,
            height_above_floor_curr=h,
        )
        return compute_reward(**kw, per_step_scale=scale) - compute_reward(**kw, per_step_scale=0.0)

    assert 800 * scaled(dt_from_hz(20.0) / BASELINE_DT) == pytest.approx(
        2000 * scaled(dt_from_hz(50.0) / BASELINE_DT)
    )


# =========================================================================== #
# Run-layer 50 Hz default + train/eval symmetry (AC1)
# =========================================================================== #
def test_run_layer_defaults_to_50hz_ratio_10() -> None:
    """AC1: a bare CLI run defaults to 50 Hz / ratio 10 / 0 ms at the RUN layer (train + eval),
    even though the ``EpisodeConfig`` dataclass default stays 20 Hz for byte-identity."""
    t = TrainRunConfig.from_mapping({"name": "x"})
    assert (t.control_hz, t.physics_ratio, t.command_latency_ms) == (50.0, 10, 0.0)
    e = EvaluateRunConfig.from_mapping({"name": "x", "checkpoint": "c.zip"})
    assert (e.control_hz, e.physics_ratio, e.command_latency_ms) == (50.0, 10, 0.0)


def test_apply_control_rate_yields_identical_episode_for_train_and_eval() -> None:
    """AC1 (train/eval symmetry): ``_apply_control_rate`` produces the SAME episode timing for a
    default train and a default eval run — so a 50 Hz-trained policy is never evaluated on a 20 Hz
    plant. The resolved episode is 50 Hz (dt 0.02), ratio 10, 0 ms."""
    from drone_fly.cli import _apply_control_rate

    t = TrainRunConfig.from_mapping({"name": "x"})
    e = EvaluateRunConfig.from_mapping({"name": "x", "checkpoint": "c.zip"})
    t_ep = _apply_control_rate(None, t).episode
    e_ep = _apply_control_rate(None, e).episode
    assert t_ep.dt == e_ep.dt == pytest.approx(0.02)
    assert t_ep.physics_ratio == e_ep.physics_ratio == 10
    assert t_ep.command_latency_ms == e_ep.command_latency_ms == 0.0


def test_apply_control_rate_converts_control_hz_to_dt() -> None:
    """AC1: the friendly ``control_hz`` YAML knob is converted to the authoritative ``dt = 1/hz``;
    an explicit 20 Hz reproduces the baseline dt exactly."""
    from drone_fly.cli import _apply_control_rate

    cfg20 = TrainRunConfig.from_mapping({"name": "x", "control_hz": 20.0, "physics_ratio": 1})
    ep = _apply_control_rate(None, cfg20).episode
    assert ep.dt == pytest.approx(0.05)
    assert ep.physics_ratio == 1


# =========================================================================== #
# YAML exposure + validation of the three knobs (AC6)
# =========================================================================== #
def test_yaml_exposes_and_accepts_all_three_knobs() -> None:
    """AC6: ``control_hz`` / ``physics_ratio`` / ``command_latency_ms`` are settable in the train
    YAML (UC-51 exposure pattern) and round-trip through ``from_mapping``."""
    cfg = TrainRunConfig.from_mapping(
        {"name": "x", "control_hz": 100.0, "physics_ratio": 5, "command_latency_ms": 40.0}
    )
    assert cfg.control_hz == 100.0
    assert cfg.physics_ratio == 5
    assert cfg.command_latency_ms == 40.0


def test_yaml_exposes_the_three_knobs_on_evaluate_too() -> None:
    """AC6: the same three knobs are exposed on the evaluate config (train/eval symmetry)."""
    cfg = EvaluateRunConfig.from_mapping(
        {
            "name": "x",
            "checkpoint": "c.zip",
            "control_hz": 25.0,
            "physics_ratio": 2,
            "command_latency_ms": 10.0,
        }
    )
    assert (cfg.control_hz, cfg.physics_ratio, cfg.command_latency_ms) == (25.0, 2, 10.0)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"control_hz": 0.0}, "control_hz"),
        ({"control_hz": -50.0}, "control_hz"),
        ({"physics_ratio": 0}, "physics_ratio"),
        ({"physics_ratio": -1}, "physics_ratio"),
        ({"command_latency_ms": -1.0}, "command_latency_ms"),
    ],
)
def test_yaml_validation_rejects_out_of_range_values(overrides, match) -> None:
    """AC6: control_hz > 0, physics_ratio ≥ 1, command_latency_ms ≥ 0 are enforced (fail loud)."""
    mapping = {"name": "x", **overrides}
    with pytest.raises(ConfigError, match=match):
        TrainRunConfig.from_mapping(mapping)


def test_yaml_validation_rejects_bad_types() -> None:
    """AC6: a non-numeric control_hz / a bool for int physics_ratio are rejected (type guard)."""
    with pytest.raises(ConfigError, match="control_hz"):
        TrainRunConfig.from_mapping({"name": "x", "control_hz": "fast"})
    with pytest.raises(ConfigError, match="physics_ratio"):
        TrainRunConfig.from_mapping({"name": "x", "physics_ratio": True})


def test_yaml_validation_applies_to_evaluate() -> None:
    """AC6: the same validation guards the evaluate config (shared ``_validate_control_rate``)."""
    with pytest.raises(ConfigError, match="physics_ratio"):
        EvaluateRunConfig.from_mapping({"name": "x", "checkpoint": "c.zip", "physics_ratio": 0})
