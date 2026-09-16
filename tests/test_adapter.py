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
