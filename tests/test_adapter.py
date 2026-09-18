"""Adapter contract: CTBR→RPM mixing (pure, no pybullet) + SimpleDroneAdapter dynamics.

Two independent surfaces are covered:

* :func:`drone_fly.adapter.pybullet_adapter.ctbr_to_rpm` — a **pure** function unit-tested
  without pybullet (the CTBR→RPM mapping the pybullet path relies on). This is the
  "if its native action is motor RPMs, the adapter performs the CTBR→RPM mapping and it is
  tested" pitfall from the use case.
* :class:`drone_fly.adapter.simple.SimpleDroneAdapter` — determinism, floor/ceiling
  contact, and action clipping for the hermetic backend.

Constructing :class:`PyBulletAdapter` requires the sim; that test is **skipped** when
``pybullet`` is not importable so CI stays hermetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.adapter import (
    ADAPTER_CHOICES,
    SimpleDroneAdapter,
    make_adapter,
    pybullet_available,
)
from drone_fly.adapter.base import DroneState, sanitize_action
from drone_fly.adapter.pybullet_adapter import ctbr_to_rpm
from drone_fly.controller.encoding import ACTION_DIM

HOVER = 1000.0
MAXR = 2000.0


# --- sanitize_action (canonical CTBR clipping) --------------------------------------
def test_sanitize_action_clips_to_bounds() -> None:
    out = sanitize_action(np.array([5.0, 5.0, -5.0, 0.3]))
    np.testing.assert_array_equal(out, np.array([1.0, 1.0, -1.0, 0.3]))


def test_sanitize_action_throttle_floor_is_zero() -> None:
    out = sanitize_action(np.array([-2.0, 0.0, 0.0, 0.0]))
    assert out[0] == 0.0  # throttle clamps at 0, not -1


def test_sanitize_action_rejects_wrong_width() -> None:
    with pytest.raises(ValueError):
        sanitize_action(np.array([0.5, 0.0, 0.0]))


# --- ctbr_to_rpm: pure mixing, unit-tested without pybullet -------------------------
def test_ctbr_hover_maps_to_base_rpm() -> None:
    # throttle=0.5, no attitude command → every rotor sits at hover RPM.
    rpm = ctbr_to_rpm(np.array([0.5, 0.0, 0.0, 0.0]), hover_rpm=HOVER, max_rpm=MAXR)
    np.testing.assert_allclose(rpm, np.full(4, HOVER))


def test_ctbr_full_throttle_above_hover_below_max() -> None:
    rpm = ctbr_to_rpm(np.array([1.0, 0.0, 0.0, 0.0]), hover_rpm=HOVER, max_rpm=MAXR)
    assert np.all(rpm > HOVER)
    assert np.all(rpm <= MAXR)


def test_ctbr_zero_throttle_below_hover() -> None:
    rpm = ctbr_to_rpm(np.array([0.0, 0.0, 0.0, 0.0]), hover_rpm=HOVER, max_rpm=MAXR)
    assert np.all(rpm < HOVER)


def test_ctbr_output_clipped_to_valid_range() -> None:
    rpm = ctbr_to_rpm(np.array([1.0, 1.0, 1.0, 1.0]), hover_rpm=HOVER, max_rpm=MAXR)
    assert np.all(rpm >= 0.0)
    assert np.all(rpm <= MAXR)
    # A hard negative command floors at 0 rather than going negative.
    rpm2 = ctbr_to_rpm(np.array([0.0, -1.0, -1.0, -1.0]), hover_rpm=HOVER, max_rpm=MAXR)
    assert np.all(rpm2 >= 0.0)


def test_ctbr_roll_is_differential_left_vs_right() -> None:
    # Motor order: [front-right, back-right, back-left, front-left]. +roll lifts the LEFT
    # rotors (2,3) relative to the RIGHT (0,1) — a genuine differential, not a common shift.
    rpm = ctbr_to_rpm(np.array([0.5, 1.0, 0.0, 0.0]), hover_rpm=HOVER, max_rpm=MAXR)
    left = rpm[[2, 3]].mean()
    right = rpm[[0, 1]].mean()
    assert left > right


def test_ctbr_pitch_is_differential_front_vs_back() -> None:
    # +pitch lifts the FRONT rotors (0,3) relative to the BACK (1,2).
    rpm = ctbr_to_rpm(np.array([0.5, 0.0, 1.0, 0.0]), hover_rpm=HOVER, max_rpm=MAXR)
    front = rpm[[0, 3]].mean()
    back = rpm[[1, 2]].mean()
    assert front > back


def test_ctbr_yaw_is_differential_by_spin_direction() -> None:
    # +yaw speeds the CW rotors (1,3) relative to the CCW rotors (0,2).
    rpm = ctbr_to_rpm(np.array([0.5, 0.0, 0.0, 1.0]), hover_rpm=HOVER, max_rpm=MAXR)
    cw = rpm[[1, 3]].mean()
    ccw = rpm[[0, 2]].mean()
    assert cw > ccw


def test_ctbr_returns_four_channels() -> None:
    rpm = ctbr_to_rpm(np.zeros(ACTION_DIM), hover_rpm=HOVER, max_rpm=MAXR)
    assert rpm.shape == (4,)


# --- SimpleDroneAdapter: determinism + floor/ceiling contact ------------------------
def _adapter() -> SimpleDroneAdapter:
    return SimpleDroneAdapter(np.array([0.0, 0.0, 1.0]), floor_z=0.0, ceiling_z=2.5, dt=0.05)


def test_simple_adapter_reset_returns_state_at_start() -> None:
    ad = _adapter()
    st = ad.reset(seed=0)
    assert isinstance(st, DroneState)
    np.testing.assert_array_equal(st.position, np.array([0.0, 0.0, 1.0]))
    assert not st.collided
    assert ad.backend == "simple"


def test_simple_adapter_deterministic_for_same_seed_and_actions() -> None:
    actions = [np.array([0.6, 0.1, -0.1, 0.05]) for _ in range(40)]
    a, b = _adapter(), _adapter()
    a.reset(seed=7)
    b.reset(seed=7)
    for act in actions:
        sa, sb = a.step(act), b.step(act)
        np.testing.assert_array_equal(sa.position, sb.position)
        np.testing.assert_array_equal(sa.velocity, sb.velocity)
        np.testing.assert_array_equal(sa.attitude, sb.attitude)


def test_simple_adapter_detects_floor_contact() -> None:
    ad = _adapter()
    ad.reset(seed=0)
    hit = False
    for _ in range(400):
        st = ad.step(np.array([0.0, 0.0, 0.0, 0.0]))  # no thrust → falls to the floor
        if st.collided:
            hit = True
            assert st.position[2] == pytest.approx(0.0)
            break
    assert hit, "drone with zero thrust should reach the floor and flag a collision"


def test_simple_adapter_detects_ceiling_contact() -> None:
    ad = _adapter()
    ad.reset(seed=0)
    hit = False
    for _ in range(400):
        st = ad.step(np.array([1.0, 0.0, 0.0, 0.0]))  # full thrust → rises to the ceiling
        if st.collided:
            hit = True
            assert st.position[2] == pytest.approx(2.5)
            break
    assert hit, "drone at full thrust should reach the ceiling and flag a collision"


def test_simple_adapter_state_stays_finite_under_extreme_action() -> None:
    ad = _adapter()
    ad.reset(seed=0)
    for _ in range(200):
        st = ad.step(np.array([1.0, 1.0, 1.0, 1.0]))
        assert np.isfinite(st.position).all()
        assert np.isfinite(st.velocity).all()


# --- make_adapter factory -----------------------------------------------------------
def test_make_adapter_simple_is_hermetic() -> None:
    ad = make_adapter("simple", np.array([0.0, 0.0, 1.0]), floor_z=0.0, ceiling_z=2.5, dt=0.05)
    assert isinstance(ad, SimpleDroneAdapter)
    assert ad.backend == "simple"


def test_make_adapter_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        make_adapter("nope", np.array([0.0, 0.0, 1.0]), floor_z=0.0, ceiling_z=2.5, dt=0.05)


def test_make_adapter_auto_falls_back_to_simple_without_pybullet() -> None:
    ad = make_adapter("auto", np.array([0.0, 0.0, 1.0]), floor_z=0.0, ceiling_z=2.5, dt=0.05)
    if not pybullet_available():
        assert ad.backend == "simple"
    else:  # pragma: no cover - only on a host with the sim installed
        assert ad.backend == "pybullet"


def test_adapter_choices_constant() -> None:
    assert ADAPTER_CHOICES == ("auto", "simple", "pybullet")


# --- PyBulletAdapter construction: skipped without the sim --------------------------
@pytest.mark.skipif(
    not pybullet_available(),
    reason="pybullet/gym-pybullet-drones not installed (hermetic CI); sim path verified on macOS",
)
def test_pybullet_adapter_constructs_when_sim_present() -> None:  # pragma: no cover - sim path
    from drone_fly.adapter.pybullet_adapter import PyBulletAdapter

    ad = PyBulletAdapter(np.array([0.0, 0.0, 1.0]), floor_z=0.0, ceiling_z=2.5, dt=0.05)
    assert ad.backend == "pybullet"


def test_pybullet_adapter_construction_raises_actionable_error_without_sim() -> None:
    if pybullet_available():  # pragma: no cover - only on a host with the sim
        pytest.skip("pybullet installed; the actionable-ImportError path is for hermetic hosts")
    from drone_fly.adapter.pybullet_adapter import PyBulletAdapter

    with pytest.raises(ImportError) as exc:
        PyBulletAdapter(np.array([0.0, 0.0, 1.0]), floor_z=0.0, ceiling_z=2.5, dt=0.05)
    # The message must point the user at the bootstrap, not be a cryptic import failure.
    assert "train.sh" in str(exc.value) or "adapter='simple'" in str(exc.value)


# ===========================================================================
# UC-08 — reconfigure() spawn/dynamics + control-latency (AC5, AC7)
# ===========================================================================
from drone_fly.adapter.base import DroneAdapter  # noqa: E402
from drone_fly.env.config import DynamicsParams  # noqa: E402


def _rollout(adapter: SimpleDroneAdapter, actions, seed: int = 3) -> list[np.ndarray]:
    adapter.reset(seed=seed)
    return [adapter.step(a).position.copy() for a in actions]


def test_reconfigure_updates_start_mass_drag_and_rate() -> None:
    """reconfigure() applies a new spawn and each dynamics knob independently (AC5)."""
    ad = _adapter()
    ad.reconfigure(
        start=np.array([1.0, -2.0, 1.5]),
        dynamics=DynamicsParams(
            mass=1.3, drag=0.25, max_body_rate=6.0, max_thrust=25.0, latency_steps=2
        ),
    )
    # Spawn reflected at the next reset.
    st = ad.reset(seed=0)
    np.testing.assert_array_equal(st.position, np.array([1.0, -2.0, 1.5]))
    # Independent instance knobs updated (mass/thrust do NOT recompute each other).
    assert ad._mass == 1.3
    assert ad._drag == 0.25
    assert ad._max_body_rate == 6.0
    assert ad._max_thrust == 25.0
    assert ad._latency == 2


def test_reconfigure_none_args_are_a_no_op() -> None:
    """A reconfigure(start=None, dynamics=None) leaves spawn and dynamics untouched (AC7)."""
    ad = _adapter()
    ad.reconfigure(start=None, dynamics=None)
    st = ad.reset(seed=0)
    np.testing.assert_array_equal(st.position, np.array([0.0, 0.0, 1.0]))
    assert ad._mass == 1.0
    assert ad._latency == 0


def test_defaults_are_byte_identical_to_never_reconfigured() -> None:
    """A DynamicsParams() reconfigure reproduces the fixed UC-01..06 dynamics exactly (AC7)."""
    rng = np.random.default_rng(0)
    actions = [rng.uniform([0, -1, -1, -1], [1, 1, 1, 1]) for _ in range(40)]

    plain = _adapter()
    reconf = _adapter()
    reconf.reconfigure(dynamics=DynamicsParams())  # latency 0, base constants

    for pa, pb in zip(_rollout(plain, actions), _rollout(reconf, actions), strict=True):
        np.testing.assert_array_equal(pa, pb)


def test_latency_zero_is_bit_identical_to_no_buffer_path() -> None:
    """Reconfiguring latency=0 keeps the no-buffer passthrough — bit-identical (AC5/AC7)."""
    rng = np.random.default_rng(4)
    actions = [rng.uniform([0, -1, -1, -1], [1, 1, 1, 1]) for _ in range(40)]

    plain = _adapter()
    zero = _adapter()
    zero.reconfigure(dynamics=DynamicsParams(latency_steps=0))
    for pa, pb in zip(_rollout(plain, actions), _rollout(zero, actions), strict=True):
        np.testing.assert_array_equal(pa, pb)


def test_latency_n_delays_the_command_stream_by_exactly_n() -> None:
    """latency=N delays real commands by exactly N steps, warming up with hover (AC5).

    A latency=N adapter applying ``[a0, a1, ...]`` must equal a latency-0 adapter applying
    ``[hover]*N + [a0, a1, ...]`` — i.e. the applied stream is shifted by exactly N.
    """
    warm = np.array([0.5, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(6)
    actions = [rng.uniform([0.2, -0.5, -0.5, -0.5], [1.0, 0.5, 0.5, 0.5]) for _ in range(30)]

    for n in (1, 2, 3):
        delayed = _adapter()
        delayed.reconfigure(dynamics=DynamicsParams(latency_steps=n))
        delayed_pos = _rollout(delayed, actions, seed=1)

        # Reference (latency 0): the APPLIED stream is [warm]*N ++ actions. The delayed
        # adapter applies exactly the first len(actions) elements of that stream, so its
        # i-th position must equal the reference's i-th position over the combined stream.
        combined = [warm] * n + list(actions)
        ref = _adapter()
        ref.reset(seed=1)
        ref_all = [ref.step(a).position.copy() for a in combined]

        for i, d in enumerate(delayed_pos):
            np.testing.assert_array_equal(
                d, ref_all[i], err_msg=f"latency={n} mismatch at step {i}"
            )


def test_latency_n_actually_differs_from_latency_zero() -> None:
    """Sanity: a non-zero latency genuinely changes the trajectory (guards a silent no-op)."""
    rng = np.random.default_rng(8)
    actions = [rng.uniform([0.2, -0.5, -0.5, -0.5], [1.0, 0.5, 0.5, 0.5]) for _ in range(30)]
    zero = _adapter()
    two = _adapter()
    two.reconfigure(dynamics=DynamicsParams(latency_steps=2))
    assert not np.array_equal(
        np.array(_rollout(zero, actions, seed=1)), np.array(_rollout(two, actions, seed=1))
    )


def test_base_adapter_reconfigure_is_a_silent_no_op() -> None:
    """The DroneAdapter base ``reconfigure`` default ignores both knobs (AC5 no-op default).

    A backend that cannot honour a knob safely inherits this no-op — a call with a spawn and
    dynamics must not raise and must change nothing observable.
    """

    class _Minimal(DroneAdapter):
        backend = "minimal"

        def reset(self, seed=None):
            return DroneState(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), False)

        def step(self, action):
            return DroneState(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), False)

    ad = _Minimal()
    # Must not raise, and returns None (a pure hook).
    assert (
        ad.reconfigure(start=np.array([9.0, 9.0, 9.0]), dynamics=DynamicsParams(mass=2.0)) is None
    )
    # State is unaffected — the base class stores nothing.
    np.testing.assert_array_equal(ad.reset().position, np.zeros(3))


# ===========================================================================
# UC-17 — battery drain + thrust-impact (AC1/AC2/AC3/AC5/AC6)
# ===========================================================================
from drone_fly.adapter.simple import GRAVITY  # noqa: E402


def _battery_adapter(cfg) -> SimpleDroneAdapter:
    """A hermetic adapter with the given BatteryConfig (or None for disabled)."""
    return SimpleDroneAdapter(
        np.array([0.0, 0.0, 1.0]), floor_z=0.0, ceiling_z=2.5, dt=0.05, battery=cfg
    )


LEVEL_FULL_THROTTLE = np.array([1.0, 0.0, 0.0, 0.0])


# --- AC1: drain is monotone non-increasing; higher throttle drains strictly faster --------
def test_battery_resets_to_full_charge() -> None:
    from drone_fly.env.config import BatteryConfig

    ad = _battery_adapter(BatteryConfig(enabled=True))
    st = ad.reset(seed=0)
    assert st.battery == 1.0


def test_battery_drains_monotone_non_increasing() -> None:
    """AC1: enabled → the battery is monotone non-increasing each step and clamped ≥ 0."""
    from drone_fly.env.config import BatteryConfig

    ad = _battery_adapter(BatteryConfig(enabled=True))
    ad.reset(seed=0)
    charges = []
    for _ in range(30):
        st = ad.step(np.array([0.5, 0.0, 0.0, 0.0]))
        charges.append(st.battery)
    assert all(y <= x for x, y in zip(charges, charges[1:], strict=False))
    assert charges[-1] < charges[0]  # it genuinely dropped
    assert all(c >= 0.0 for c in charges)


def test_battery_drain_matches_documented_rate() -> None:
    """AC1: the first-step drop equals (idle_rate + throttle_rate*throttle) * dt exactly."""
    from drone_fly.env.config import BatteryConfig

    cfg = BatteryConfig(enabled=True)
    ad = _battery_adapter(cfg)
    ad.reset(seed=0)
    throttle = 0.7
    st = ad.step(np.array([throttle, 0.0, 0.0, 0.0]))
    expected_drop = (cfg.idle_rate + cfg.throttle_rate * throttle) * 0.05
    assert st.battery == pytest.approx(1.0 - expected_drop)


def test_higher_throttle_drains_strictly_faster() -> None:
    """AC1: over equal steps, a higher-throttle trajectory drains strictly more charge."""
    from drone_fly.env.config import BatteryConfig

    high = _battery_adapter(BatteryConfig(enabled=True))
    low = _battery_adapter(BatteryConfig(enabled=True))
    high.reset(seed=0)
    low.reset(seed=0)
    for _ in range(20):
        sh = high.step(np.array([1.0, 0.0, 0.0, 0.0]))
        sl = low.step(np.array([0.0, 0.0, 0.0, 0.0]))
    assert sh.battery < sl.battery


# --- AC2: achievable vertical acceleration is strictly lower at low battery ----------------
def _vertical_accel_first_step(ad: SimpleDroneAdapter, battery: float) -> float:
    """Vertical acceleration on the first step from rest at a fixed start-of-step charge."""
    ad.reset(seed=0)
    ad._battery = battery  # set start-of-step charge directly (private test hook)
    st = ad.step(LEVEL_FULL_THROTTLE)
    # From rest with level attitude and zero drag, vz = accel_z * dt.
    return st.velocity[2] / 0.05


def test_lower_battery_gives_lower_vertical_accel() -> None:
    """AC2: holding action/dynamics fixed, vertical accel is strictly lower at low charge."""
    from drone_fly.env.config import BatteryConfig

    ad = _battery_adapter(BatteryConfig(enabled=True))
    accel_full = _vertical_accel_first_step(ad, battery=1.0)
    accel_low = _vertical_accel_first_step(ad, battery=0.02)
    assert accel_low < accel_full


# --- AC3: soft depletion — an empty battery cannot hover and sinks to a floor crash --------
def test_empty_battery_cannot_hover_and_crashes_to_floor() -> None:
    """AC3 (adapter-level soft depletion): with an empty battery even full throttle produces a
    ceiling below hover, so the drone sinks and eventually flags a floor collision — no new
    hard-terminate branch, just the existing crash path."""
    from drone_fly.env.config import BatteryConfig

    ad = _battery_adapter(BatteryConfig(enabled=True))
    ad.reset(seed=0)
    ad._battery = 0.0  # fully depleted at start of every step (idle drain keeps it there)
    crashed = False
    for _ in range(400):
        st = ad.step(LEVEL_FULL_THROTTLE)
        ad._battery = 0.0  # keep it pinned empty for a deterministic soft-crash
        if st.collided:
            crashed = True
            assert st.position[2] == pytest.approx(0.0)  # sank to the FLOOR, not the ceiling
            break
    assert crashed, "an empty battery must be unable to hover and sink to a floor crash"


# --- AC5: off-by-default byte-identity + no extra RNG consumption --------------------------
def test_battery_disabled_reports_full_charge_always() -> None:
    """AC5: disabled (or no battery) → state.battery stays 1.0 and is never drained."""
    from drone_fly.env.config import BatteryConfig

    for cfg in (None, BatteryConfig(enabled=False)):
        ad = _battery_adapter(cfg)
        ad.reset(seed=0)
        for _ in range(50):
            st = ad.step(np.array([0.9, 0.1, -0.1, 0.05]))
        assert st.battery == 1.0


def test_battery_disabled_is_bit_identical_to_no_battery() -> None:
    """AC5: a BatteryConfig(enabled=False) adapter is bit-identical to a no-battery adapter over a
    fixed-seed episode — the disabled path never reads a battery, drains, or draws np_random."""
    from drone_fly.env.config import BatteryConfig

    rng = np.random.default_rng(11)
    actions = [rng.uniform([0, -1, -1, -1], [1, 1, 1, 1]) for _ in range(60)]

    plain = _battery_adapter(None)
    disabled = _battery_adapter(BatteryConfig(enabled=False))
    plain.reset(seed=5)
    disabled.reset(seed=5)
    for act in actions:
        sp_, sd = plain.step(act), disabled.step(act)
        np.testing.assert_array_equal(sp_.position, sd.position)
        np.testing.assert_array_equal(sp_.velocity, sd.velocity)
        np.testing.assert_array_equal(sp_.attitude, sd.attitude)


def test_battery_disabled_thrust_is_exactly_throttle_times_max_thrust() -> None:
    """AC5/AC6: on the disabled path the effective thrust is EXACTLY throttle * _max_thrust — the
    enabled ceiling_factor never multiplies in (disabled skips the battery branch entirely)."""
    plain = _battery_adapter(None)
    plain.reset(seed=0)
    st = plain.step(LEVEL_FULL_THROTTLE)
    accel_z = st.velocity[2] / 0.05
    # thrust_acc = throttle * _max_thrust / mass ; vz = (thrust_acc - GRAVITY) * dt at rest, level.
    expected_accel = 1.0 * plain._max_thrust / plain._mass - GRAVITY
    assert accel_z == pytest.approx(expected_accel)


# --- AC6: battery composes with UC-08 thrust_factor via a documented product order ---------
def test_battery_is_third_multiplicative_factor_after_uc08_thrust_factor() -> None:
    """AC6: effective_max_thrust == (base * thrust_factor) * ceiling_factor(battery). UC-08 sets
    _max_thrust = base * thrust_factor via reconfigure; battery is the THIRD factor on top."""
    from drone_fly.env.config import BatteryConfig

    cfg = BatteryConfig(enabled=True)
    ad = _battery_adapter(cfg)
    # UC-08 thrust_factor=0.8 folds into max_thrust (base * 0.8).
    thrust_factor = 0.8
    base_max_thrust = 2.0 * ad._mass * GRAVITY  # BASE_MAX_THRUST (TWR 2)
    ad.reconfigure(dynamics=DynamicsParams(max_thrust=base_max_thrust * thrust_factor))
    ad.reset(seed=0)
    battery = 0.1
    ad._battery = battery
    st = ad.step(LEVEL_FULL_THROTTLE)  # throttle 1.0, level attitude
    # Recover the effective ceiling that acted this step from the vertical accel.
    accel_z = st.velocity[2] / 0.05
    effective_max_thrust = (accel_z + GRAVITY) * ad._mass / 1.0
    expected = (base_max_thrust * thrust_factor) * cfg.ceiling_factor(battery)
    assert effective_max_thrust == pytest.approx(expected)


# ===========================================================================
# UC-18 — recharge primitive: clamp at 1.0 + net-positive over a docked step (AC1)
# ===========================================================================
def test_recharge_adds_charge_and_returns_new_value() -> None:
    """AC1: ``recharge`` adds the increment and returns the resulting charge."""
    from drone_fly.env.config import BatteryConfig

    ad = _battery_adapter(BatteryConfig(enabled=True))
    ad.reset(seed=0)
    ad._battery = 0.4
    out = ad.recharge(0.25)
    assert out == pytest.approx(0.65)
    assert ad._battery == pytest.approx(0.65)


def test_recharge_clamps_at_full_charge() -> None:
    """AC1: charge clamps at ``1.0`` — a full battery cannot overcharge past the ceiling."""
    from drone_fly.env.config import BatteryConfig

    ad = _battery_adapter(BatteryConfig(enabled=True))
    ad.reset(seed=0)
    ad._battery = 0.9
    assert ad.recharge(0.5) == 1.0  # 0.9 + 0.5 would be 1.4 → clamped
    assert ad._battery == 1.0
    # Recharging an already-full battery is a no-op (still clamped at 1.0).
    assert ad.recharge(0.3) == 1.0


def test_recharge_floors_a_negative_increment_at_zero() -> None:
    """AC1: a negative ``delta`` can never *drain* through the recharge path (floored at 0)."""
    from drone_fly.env.config import BatteryConfig

    ad = _battery_adapter(BatteryConfig(enabled=True))
    ad.reset(seed=0)
    ad._battery = 0.5
    assert ad.recharge(-0.3) == 0.5  # unchanged
    assert ad._battery == 0.5


def test_docked_step_nets_a_gain_under_the_net_positive_invariant() -> None:
    """AC1: mirroring the env's docked step (adapter drains, then ``recharge(recharge_rate*dt)``
    tops up), a docked step is a strict *gain* when ``recharge_rate > docked drain`` — so the pad
    actually fills instead of leaking away."""
    from drone_fly.env.config import BatteryConfig

    cfg = BatteryConfig(enabled=True)
    ad = _battery_adapter(cfg)
    ad.reset(seed=0)
    ad._battery = 0.5
    before = ad._battery
    # The env sequence on a docked step: adapter.step drains, then env calls recharge(rate*dt).
    ad.step(np.array([0.5, 0.0, 0.0, 0.0]))  # applies this step's drain
    after = ad.recharge(cfg.recharge_rate * 0.05)
    assert after > before  # net-positive: the docked step gained charge
    # And the gain equals recharge_rate*dt minus the (idle + throttle*rate)*dt drain.
    drain = (cfg.idle_rate + cfg.throttle_rate * 0.5) * 0.05
    expected = before - drain + cfg.recharge_rate * 0.05
    assert after == pytest.approx(expected)


def test_recharge_safe_on_the_disabled_path() -> None:
    """``recharge`` is safe to call even with battery disabled (``_battery`` stays a plain float);
    the env only calls it when enabled, but the primitive never raises."""
    ad = _battery_adapter(None)
    ad.reset(seed=0)
    assert ad.recharge(0.1) == 1.0  # already full → clamped, no error


@pytest.mark.parametrize(
    ("idle_rate", "throttle_rate", "recharge_rate", "throttle"),
    [
        (0.005, 0.01, 0.5, 1.0),  # defaults, full throttle
        (0.08, 0.05, 0.9, 0.5),  # the UC-18 tuned-drain fixture
        (0.2, 0.3, 0.6, 1.0),  # aggressive drain, still net-positive
    ],
)
def test_net_positive_invariant_holds_across_drain_configs(
    idle_rate, throttle_rate, recharge_rate, throttle
) -> None:
    """AC1 (general net-positive invariant): whenever ``recharge_rate > idle_rate + throttle_rate``
    the battery STRICTLY rises over a docked step (adapter drains, env tops up by recharge_rate*dt)
    for any throttle — so a recharge pad always fills, never leaks, across drain configs."""
    from drone_fly.env.config import BatteryConfig

    cfg = BatteryConfig(
        enabled=True,
        idle_rate=idle_rate,
        throttle_rate=throttle_rate,
        recharge_rate=recharge_rate,
    )
    assert cfg.recharge_rate > cfg.idle_rate + cfg.throttle_rate  # invariant precondition
    ad = _battery_adapter(cfg)
    ad.reset(seed=0)
    ad._battery = 0.5
    before = ad._battery
    ad.step(np.array([throttle, 0.0, 0.0, 0.0]))  # this docked step's drain
    after = ad.recharge(cfg.recharge_rate * 0.05)  # env's per-step top-up
    assert after > before  # strictly rose despite the drain


# ===========================================================================
# UC-19 — integrity damage + control-authority impact (AC1/AC3)
# ===========================================================================
def _damage_adapter(cfg) -> SimpleDroneAdapter:
    """A hermetic adapter with the given DamageConfig (or None for disabled)."""
    return SimpleDroneAdapter(
        np.array([0.0, 0.0, 1.0]), floor_z=0.0, ceiling_z=2.5, dt=0.05, damage=cfg
    )


# --- integrity ledger: reset / damage (clamp ≥0) / repair (clamp ≤1) -----------------------
def test_integrity_resets_to_full() -> None:
    """AC1: integrity resets to ``1.0`` (pristine) on reset."""
    from drone_fly.env.config import DamageConfig

    ad = _damage_adapter(DamageConfig(enabled=True))
    st = ad.reset(seed=0)
    assert st.integrity == 1.0
    assert ad._integrity == 1.0


def test_damage_decrements_and_returns_new_integrity() -> None:
    """AC2: ``damage`` subtracts the amount and returns the resulting integrity."""
    from drone_fly.env.config import DamageConfig

    ad = _damage_adapter(DamageConfig(enabled=True))
    ad.reset(seed=0)
    ad._integrity = 0.8
    out = ad.damage(0.3)
    assert out == pytest.approx(0.5)
    assert ad._integrity == pytest.approx(0.5)


def test_damage_clamps_at_zero() -> None:
    """AC2: integrity clamps at ``0.0`` — repeated contact cannot push it negative."""
    from drone_fly.env.config import DamageConfig

    ad = _damage_adapter(DamageConfig(enabled=True))
    ad.reset(seed=0)
    ad._integrity = 0.2
    assert ad.damage(0.5) == 0.0  # 0.2 - 0.5 would be -0.3 → clamped
    assert ad._integrity == 0.0
    assert ad.damage(0.1) == 0.0  # already wrecked → stays clamped


def test_damage_floors_a_negative_amount_at_zero() -> None:
    """AC2: a negative ``amount`` can never *heal* through the damage path (floored at 0)."""
    from drone_fly.env.config import DamageConfig

    ad = _damage_adapter(DamageConfig(enabled=True))
    ad.reset(seed=0)
    ad._integrity = 0.5
    assert ad.damage(-0.3) == 0.5  # unchanged
    assert ad._integrity == 0.5


def test_repair_increments_and_clamps_at_full() -> None:
    """AC4: ``repair`` adds the delta, returns the new integrity, and clamps at ``1.0``."""
    from drone_fly.env.config import DamageConfig

    ad = _damage_adapter(DamageConfig(enabled=True))
    ad.reset(seed=0)
    ad._integrity = 0.4
    assert ad.repair(0.25) == pytest.approx(0.65)
    assert ad._integrity == pytest.approx(0.65)
    ad._integrity = 0.9
    assert ad.repair(0.5) == 1.0  # 0.9 + 0.5 would be 1.4 → clamped
    assert ad.repair(0.3) == 1.0  # already full → no overheal


def test_repair_floors_a_negative_delta_at_zero() -> None:
    """AC4: a negative ``delta`` can never *damage* through the repair path (floored at 0)."""
    from drone_fly.env.config import DamageConfig

    ad = _damage_adapter(DamageConfig(enabled=True))
    ad.reset(seed=0)
    ad._integrity = 0.5
    assert ad.repair(-0.3) == 0.5
    assert ad._integrity == 0.5


def test_damage_repair_safe_on_the_disabled_path() -> None:
    """The primitives never raise even with damage disabled (``_integrity`` stays a plain float);
    the env only calls them when enabled."""
    ad = _damage_adapter(None)
    ad.reset(seed=0)
    assert ad.damage(0.2) == pytest.approx(0.8)
    assert ad.repair(0.5) == pytest.approx(1.0)


# --- AC3: a damaged instance shows a SMALLER attitude / angular-velocity response ----------
_AGILITY_ACTION = np.array([0.5, 1.0, 0.0, 0.0])  # hover throttle, full roll stick


def test_damaged_shows_smaller_attitude_and_angular_response() -> None:
    """AC3: for a fixed action sequence, a degraded instance builds LESS roll attitude and a
    smaller angular velocity than a pristine one — the effective ``max_body_rate`` is scaled by
    ``authority_factor(integrity)`` (integrity is read start-of-step, mutated only via the hooks,
    so it stays fixed across these steps)."""
    from drone_fly.env.config import DamageConfig

    pristine = _damage_adapter(DamageConfig(enabled=True))
    degraded = _damage_adapter(DamageConfig(enabled=True))
    pristine.reset(seed=0)
    degraded.reset(seed=0)
    degraded._integrity = 0.1  # heavily damaged (start-of-step, held: step never mutates it)
    for _ in range(4):
        sp = pristine.step(_AGILITY_ACTION)
        sd = degraded.step(_AGILITY_ACTION)
    # Roll attitude and roll-rate magnitudes are strictly smaller when degraded.
    assert abs(sd.attitude[0]) < abs(sp.attitude[0])
    assert abs(sd.angular_velocity[0]) < abs(sp.angular_velocity[0])


def test_degraded_body_rate_is_floored_at_min_authority() -> None:
    """AC3: at integrity ``0.0`` the effective body rate is exactly ``min_authority`` (the floor),
    recovered from the first-step roll rate under full stick."""
    from drone_fly.env.config import DamageConfig

    cfg = DamageConfig(enabled=True)
    ad = _damage_adapter(cfg)
    ad.reset(seed=0)
    ad._integrity = 0.0  # authority_factor(0)=0 → max(min_authority, 0) == min_authority
    st = ad.step(np.array([0.5, 1.0, 0.0, 0.0]))  # full roll stick
    # angular_velocity[0] == roll_cmd * effective_max_body_rate == 1.0 * min_authority.
    assert st.angular_velocity[0] == pytest.approx(cfg.min_authority)


# --- AC3: integrity == 1.0 ENABLED path is byte-identical to the DISABLED path -------------
def test_full_integrity_enabled_is_bit_identical_to_disabled() -> None:
    """AC3: with ``integrity == 1.0`` and ``min_authority < BASE_MAX_BODY_RATE``, the enabled path
    (``max(min_authority, max_body_rate * 1.0) == max_body_rate``) is bit-for-bit identical to the
    disabled path over a fixed action sequence. Hermetic — no dynamics randomization (no
    ``reconfigure``), same seed."""
    from drone_fly.env.config import DamageConfig

    rng = np.random.default_rng(19)
    actions = [rng.uniform([0, -1, -1, -1], [1, 1, 1, 1]) for _ in range(60)]

    enabled = _damage_adapter(DamageConfig(enabled=True))  # integrity stays 1.0 (never damaged)
    disabled = _damage_adapter(DamageConfig(enabled=False))
    enabled.reset(seed=7)
    disabled.reset(seed=7)
    for act in actions:
        se, sd = enabled.step(act), disabled.step(act)
        np.testing.assert_array_equal(se.position, sd.position)
        np.testing.assert_array_equal(se.velocity, sd.velocity)
        np.testing.assert_array_equal(se.attitude, sd.attitude)
        np.testing.assert_array_equal(se.angular_velocity, sd.angular_velocity)
        assert se.integrity == 1.0  # enabled but pristine reports full integrity


def test_disabled_reports_full_integrity_always() -> None:
    """AC1: disabled (or no damage) → ``state.integrity`` stays 1.0 and is never read/mutated."""
    from drone_fly.env.config import DamageConfig

    for cfg in (None, DamageConfig(enabled=False)):
        ad = _damage_adapter(cfg)
        ad.reset(seed=0)
        for _ in range(30):
            st = ad.step(np.array([0.9, 0.5, -0.3, 0.1]))
        assert st.integrity == 1.0


# --- AC3: ONLY control authority is degraded — thrust / mass / drag are untouched ----------
def test_damage_leaves_thrust_mass_drag_untouched() -> None:
    """AC3: damage degrades ONLY ``max_body_rate``; ``max_thrust`` / ``mass`` / ``drag`` and the
    achieved vertical thrust are identical between a pristine and a heavily-degraded instance under
    a pure-throttle (no-tilt) command — a damaged drone is sluggish-but-flyable, not sinking."""
    from drone_fly.env.config import DamageConfig

    pristine = _damage_adapter(DamageConfig(enabled=True))
    degraded = _damage_adapter(DamageConfig(enabled=True))
    pristine.reset(seed=0)
    degraded.reset(seed=0)
    degraded._integrity = 0.05
    # Knobs are unchanged by damage.
    assert pristine._max_thrust == degraded._max_thrust
    assert pristine._mass == degraded._mass
    assert pristine._drag == degraded._drag
    # Level full-throttle: no roll/pitch command ⇒ body-rate scaling is irrelevant ⇒ the vertical
    # response is identical (thrust path is byte-untouched by integrity).
    sp = pristine.step(np.array([1.0, 0.0, 0.0, 0.0]))
    sd = degraded.step(np.array([1.0, 0.0, 0.0, 0.0]))
    assert sd.velocity[2] == pytest.approx(sp.velocity[2])
    assert sd.position[2] == pytest.approx(sp.position[2])
