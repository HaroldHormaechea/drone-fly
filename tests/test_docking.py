"""UC-16 — pure pad-docking predicates for controlled landing/takeoff.

Covers :mod:`drone_fly.env.docking` (hermetic numpy, no env / sim / torch / network): the
five component predicates and the composite :func:`evaluate_dock`, exercised on **hand-driven
states** so every threshold is hit unambiguously — in particular the AC3 "too fast" branch is
a **pure-predicate** test (hand-chosen ``prev``/``curr`` z so the inferred descent speed is
exact), never a scripted-dynamics trajectory.

Mapped ACs:
* **AC2** — a slow, upright, over-pad floor contact docks; the ``<=`` boundaries
  (descent == max, tilt == max, horizontal == radius, z == floor + eps) are all inclusive.
* **AC3** — each failing condition **in isolation** re-crashes: too fast, too tilted (roll
  and pitch independently), off-pad, and a bare non-collision.
* **AC7** — a ceiling contact (``curr_z`` at the ceiling, well above the floor band) never
  docks even when it is over a pad and slow/upright.
* **drift re-crash** — the predicate is stateless, so an off-pad floor contact returns
  ``False`` regardless of any earlier dock (lateral drift off a pad edge is a crash).
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.env.config import PadSpec
from drone_fly.env.docking import (
    FLOOR_CONTACT_EPS,
    descent_speed,
    evaluate_dock,
    is_floor_contact,
    is_upright,
    over_pad,
    pad_under,
)

# Shared fixtures for the composite tests: one pad on the floor at the origin, floor at z=0.
_PAD = PadSpec(center=(0.0, 0.0), radius=0.5)
_PADS = (_PAD,)
_FLOOR_Z = 0.0
_DT = 0.05
_MAX_DESCENT = 0.5  # m/s
_MAX_TILT = 0.2618  # rad (~15 deg)
_LEVEL = (0.0, 0.0, 0.0)  # roll, pitch, yaw — perfectly upright


# =====================================================================================
# over_pad / pad_under — horizontal containment (AC2)
# =====================================================================================
def test_over_pad_inside_radius() -> None:
    """A point horizontally within the pad radius is over the pad (z is ignored)."""
    assert over_pad((0.2, 0.1, 5.0), _PAD) is True


def test_over_pad_outside_radius() -> None:
    assert over_pad((1.0, 0.0, 0.0), _PAD) is False


def test_over_pad_on_radius_edge_is_inclusive() -> None:
    """Horizontal containment is ``hypot(dx, dy) <= radius`` — exactly on the edge counts."""
    assert over_pad((0.5, 0.0, 0.0), _PAD) is True  # hypot == radius
    assert over_pad((0.3, 0.4, 0.0), _PAD) is True  # 3-4-5 → hypot == 0.5 == radius


def test_over_pad_respects_a_nonzero_center() -> None:
    pad = PadSpec(center=(2.0, -1.0), radius=0.5)
    assert over_pad((2.4, -1.0, 0.0), pad) is True
    assert over_pad((0.0, 0.0, 0.0), pad) is False


def test_pad_under_returns_first_matching_pad() -> None:
    """``pad_under`` returns the first pad the point is over (stable, deterministic)."""
    a = PadSpec(center=(0.0, 0.0), radius=0.5)
    b = PadSpec(center=(0.1, 0.0), radius=0.5)  # also covers the origin
    assert pad_under((0.0, 0.0, 0.0), (a, b)) is a


def test_pad_under_none_when_off_every_pad() -> None:
    assert pad_under((5.0, 5.0, 0.0), _PADS) is None


def test_pad_under_empty_is_none() -> None:
    """No pads → never over a pad (the off-by-default AC1/AC8 short-circuit)."""
    assert pad_under((0.0, 0.0, 0.0), ()) is None


# =====================================================================================
# descent_speed — inferred impact-velocity proxy (AC3, AC7 load-bearing decision)
# =====================================================================================
def test_descent_speed_positive_when_descending() -> None:
    """``(prev_z - curr_z) / dt`` is positive while dropping; hand-chosen for an exact value."""
    assert descent_speed((0.0, 0.0, 0.02), (0.0, 0.0, 0.0), _DT) == pytest.approx(0.4)


def test_descent_speed_negative_when_ascending() -> None:
    """A rising step (takeoff) reads a negative descent speed."""
    assert descent_speed((0.0, 0.0, 0.0), (0.0, 0.0, 0.30), _DT) == pytest.approx(-6.0)


def test_descent_speed_zero_when_level() -> None:
    assert descent_speed((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), _DT) == 0.0


def test_descent_speed_ignores_horizontal_motion() -> None:
    """Only the vertical drop feeds the proxy; horizontal travel is irrelevant."""
    assert descent_speed((1.0, 2.0, 0.02), (-3.0, 4.0, 0.0), _DT) == pytest.approx(0.4)


# =====================================================================================
# is_upright — roll/pitch within max_tilt, yaw irrelevant (AC2)
# =====================================================================================
def test_is_upright_when_level() -> None:
    assert is_upright(_LEVEL, _MAX_TILT) is True


def test_is_upright_ignores_yaw() -> None:
    """Yaw is not part of uprightness — a fully-spun-around drone is still upright."""
    assert is_upright((0.0, 0.0, 3.0), _MAX_TILT) is True


def test_is_upright_on_tilt_limit_is_inclusive() -> None:
    """``<=`` — a drone exactly at the tilt limit (roll or pitch) still counts as upright."""
    assert is_upright((_MAX_TILT, 0.0, 0.0), _MAX_TILT) is True
    assert is_upright((0.0, -_MAX_TILT, 0.0), _MAX_TILT) is True


def test_is_upright_past_tilt_limit_is_false() -> None:
    assert is_upright((_MAX_TILT + 0.01, 0.0, 0.0), _MAX_TILT) is False
    assert is_upright((0.0, _MAX_TILT + 0.01, 0.0), _MAX_TILT) is False


# =====================================================================================
# is_floor_contact — floor-vs-ceiling disambiguation (AC7)
# =====================================================================================
def test_is_floor_contact_at_floor() -> None:
    assert is_floor_contact((0.0, 0.0, 0.0), _FLOOR_Z) is True


def test_is_floor_contact_within_eps_is_inclusive() -> None:
    """A contact at (or within) ``FLOOR_CONTACT_EPS`` of the floor counts as floor contact."""
    assert is_floor_contact((0.0, 0.0, _FLOOR_Z + FLOOR_CONTACT_EPS), _FLOOR_Z) is True


def test_is_floor_contact_at_ceiling_is_false() -> None:
    """A ceiling contact (``curr_z`` well above the floor band) is not a floor contact (AC7)."""
    assert is_floor_contact((0.0, 0.0, 2.5), _FLOOR_Z) is False


def test_is_floor_contact_respects_a_nonzero_floor() -> None:
    assert is_floor_contact((0.0, 0.0, 1.0), floor_z=1.0) is True
    assert is_floor_contact((0.0, 0.0, 1.5), floor_z=1.0) is False


# =====================================================================================
# evaluate_dock — composite: dock success + each failing condition in isolation
# =====================================================================================
def _dock(prev, curr, attitude=_LEVEL, collided=True, pads=_PADS, max_descent=_MAX_DESCENT):
    """Call ``evaluate_dock`` with the shared floor/dt/tilt fixtures pinned."""
    return evaluate_dock(
        prev,
        curr,
        attitude,
        collided,
        _FLOOR_Z,
        pads,
        _DT,
        max_descent,
        _MAX_TILT,
    )


def test_evaluate_dock_success() -> None:
    """AC2: a slow, upright, over-pad floor contact is a controlled dock."""
    # prev z 0.02 → curr z 0.0 over dt 0.05 = 0.4 m/s descent (< 0.5), upright, over the pad.
    assert _dock((0.0, 0.0, 0.02), (0.0, 0.0, 0.0)) is True


def test_evaluate_dock_descent_on_limit_is_inclusive() -> None:
    """AC2 boundary: descent speed exactly ``max_descent`` still docks (predicate uses ``>``)."""
    # prev z = max_descent * dt = 0.025 → curr 0.0 → exactly 0.5 m/s.
    assert _dock((0.0, 0.0, _MAX_DESCENT * _DT), (0.0, 0.0, 0.0)) is True


def test_evaluate_dock_too_fast_crashes() -> None:
    """AC3 (pure predicate): a fast floor contact fails the descent gate → not a dock.

    Hand-chosen z drop (0.1 m over one 0.05 s step = 2.0 m/s) so the threshold is unambiguous;
    this is NOT a scripted-dynamics run — it directly exercises the descent-speed branch.
    """
    assert _dock((0.0, 0.0, 0.10), (0.0, 0.0, 0.0)) is False


def test_evaluate_dock_too_tilted_crashes() -> None:
    """AC3: an over-tilted (roll OR pitch past max_tilt) but otherwise-perfect landing crashes."""
    assert _dock((0.0, 0.0, 0.02), (0.0, 0.0, 0.0), attitude=(_MAX_TILT + 0.1, 0.0, 0.0)) is False
    assert _dock((0.0, 0.0, 0.02), (0.0, 0.0, 0.0), attitude=(0.0, _MAX_TILT + 0.1, 0.0)) is False


def test_evaluate_dock_off_pad_crashes() -> None:
    """AC3: a slow, upright floor contact that is NOT over any pad still crashes."""
    assert _dock((5.0, 5.0, 0.02), (5.0, 5.0, 0.0)) is False


def test_evaluate_dock_not_collided_is_false() -> None:
    """No contact this step → not a dock (short-circuits before any geometry)."""
    assert _dock((0.0, 0.0, 0.02), (0.0, 0.0, 0.0), collided=False) is False


def test_evaluate_dock_no_pads_short_circuits_false() -> None:
    """AC8: with ``pads == ()`` a floor contact is never a dock → crash path byte-identical."""
    assert _dock((0.0, 0.0, 0.02), (0.0, 0.0, 0.0), pads=()) is False


def test_evaluate_dock_ceiling_never_docks() -> None:
    """AC7: a ceiling contact never docks, even directly over a pad and slow + upright.

    The drone rises gently into the ceiling at (0,0) — horizontally over the pad, |descent|
    tiny, level — yet ``curr_z`` is at the ceiling (not the floor band), so it is a crash.
    """
    assert _dock((0.0, 0.0, 2.48), (0.0, 0.0, 2.5)) is False


def test_evaluate_dock_is_stateless_off_pad_drift_recrashes() -> None:
    """Drift re-crash: the predicate carries NO state, so a previous dock cannot rescue an
    off-pad floor contact — evaluated fresh, the drifted contact returns ``False`` (AC3/AC5).
    """
    prev_curr = ((0.0, 0.0, 0.02), (0.0, 0.0, 0.0))
    drifted = ((0.6, 0.0, 0.02), (0.6, 0.0, 0.0))  # 0.6 m out — just past the 0.5 radius
    assert _dock(*prev_curr) is True  # docked here...
    assert _dock(*drifted) is False  # ...but a fresh off-pad contact re-crashes regardless


def test_evaluate_dock_boundaries_compose() -> None:
    """AC2: all three ``<=`` boundaries at once (descent == max, tilt == max, horizontal ==
    radius) still docks — the inclusive edges compose into a valid dock.
    """
    on_edge_xy = (_PAD.radius, 0.0)  # exactly radius away horizontally
    prev = (*on_edge_xy, _MAX_DESCENT * _DT)
    curr = (*on_edge_xy, 0.0)
    assert _dock(prev, curr, attitude=(_MAX_TILT, 0.0, 0.0)) is True


def test_evaluate_dock_returns_plain_bool() -> None:
    """Predicates return real ``bool``/``float`` (not numpy) so ``info['docked']`` is JSON-safe."""
    assert type(_dock((0.0, 0.0, 0.02), (0.0, 0.0, 0.0))) is bool  # noqa: E721
    assert isinstance(over_pad((0.0, 0.0, 0.0), _PAD), bool)
    assert isinstance(is_upright(_LEVEL, _MAX_TILT), bool)
    assert isinstance(is_floor_contact((0.0, 0.0, 0.0), _FLOOR_Z), bool)
    assert isinstance(descent_speed((0.0, 0.0, 0.02), (0.0, 0.0, 0.0), _DT), float)


def test_over_pad_accepts_ndarray_positions() -> None:
    """The predicates coerce array-likes, so an env passing a numpy position works unchanged."""
    assert over_pad(np.array([0.2, 0.1, 0.0]), _PAD) is True
    assert _dock(np.array([0.0, 0.0, 0.02]), np.array([0.0, 0.0, 0.0])) is True


# =====================================================================================
# UC-18 — the ``rechargeable`` flag does NOT alter the dock geometry (AC1)
# =====================================================================================
# A recharge pad is a UC-16 docking pad tagged ``rechargeable``; the flag re-classifies what a
# dock *does* (refill vs. not), never *whether* a floor contact docks. The dock predicate does
# not read the flag, so a recharge pad and a plain pad with identical geometry dock identically.
_RECHARGE_PAD = PadSpec(center=(0.0, 0.0), radius=0.5, rechargeable=True)


def test_recharge_pad_docks_like_a_plain_pad() -> None:
    """AC1: a slow, upright, over-pad floor contact docks on a ``rechargeable`` pad exactly as on
    a plain pad — the dock predicate is geometry-only and ignores the flag."""
    prev, curr = (0.0, 0.0, _MAX_DESCENT * _DT), (0.0, 0.0, 0.0)
    assert _dock(prev, curr, pads=(_RECHARGE_PAD,)) is True
    # Identical geometry, flag flipped → identical dock verdict.
    assert _dock(prev, curr, pads=(_RECHARGE_PAD,)) == _dock(prev, curr, pads=(_PAD,))


def test_recharge_pad_fast_landing_still_re_crashes() -> None:
    """AC1: the flag does not relax the descent gate — a too-fast landing on a recharge pad is
    still a crash, exactly as on a plain pad."""
    prev, curr = (0.0, 0.0, (_MAX_DESCENT + 1.0) * _DT), (0.0, 0.0, 0.0)
    assert _dock(prev, curr, pads=(_RECHARGE_PAD,)) is False


def test_pad_under_returns_the_rechargeable_pad_with_its_flag() -> None:
    """AC1: ``pad_under`` returns the pad object (flag intact) so the env can read
    ``pad.rechargeable`` to decide whether to refill."""
    found = pad_under((0.0, 0.0, 0.0), (_RECHARGE_PAD,))
    assert found is _RECHARGE_PAD
    assert found.rechargeable is True
