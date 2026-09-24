"""Core unit tests for :func:`drone_fly.env.reward.compute_reward` (UC-58 redesign).

``compute_reward`` is pure (scalars + bools + a config), so its "documented and unit-tested on
hand-constructed transitions" contract is exercised here without any env or sim. UC-58 rebuilt the
reward from scratch — ONE coherent composition::

    reward = progress + gate_bonus/N + completion_bonus + altitude_reward
             − collision_penalty(genuine crash) − obstacle_penalty

The four accreted altitude terms (``airborne_bonus`` / ``climb_*`` / ``ground_break_*`` /
``altitude_hold_*``) and the per-step ``time_penalty`` were RETIRED; the single sustained,
level-based, saturating ``altitude_reward`` supplies the takeoff/hold signal. This module pins the
component algebra and the ordering guarantees; the AC1–AC5 rollout-level invariants live in
:mod:`tests.test_uc58_takeoff_reward`.
"""

from __future__ import annotations

import pytest

from drone_fly.env.config import RewardConfig
from drone_fly.env.reward import compute_reward

CFG = RewardConfig()


def _step(
    *,
    progress_prev=0.0,
    progress_curr=0.0,
    event=None,
    collided=False,
    completed=False,
    num_gates=1,
    obstacle_contact=False,
    h=0.0,
    per_step_scale=1.0,
    cfg=CFG,
) -> float:
    """Thin keyword wrapper around the UC-58 ``compute_reward`` signature."""
    return compute_reward(
        dist_to_target_prev=progress_prev,
        dist_to_target_curr=progress_curr,
        event=event,
        collided=collided,
        completed=completed,
        cfg=cfg,
        num_gates=num_gates,
        obstacle_contact=obstacle_contact,
        height_above_floor_curr=h,
        per_step_scale=per_step_scale,
    )


# --------------------------------------------------------------------------- #
# Existence cost is GONE (UC-58): an idle floor step is exactly 0 — no time_penalty.
# --------------------------------------------------------------------------- #
def test_idle_floor_step_is_zero_no_time_penalty() -> None:
    """UC-58: with ``time_penalty`` retired and the altitude reward paying 0 at the floor, a bare
    idle step (no progress, no events, on the floor) is EXACTLY 0 — no per-step existence bleed."""
    assert _step() == 0.0


def test_no_negative_reward_without_an_explicit_penalty() -> None:
    """UC-58 anti-suicide algebra: absent a genuine crash / obstacle contact, a step is never
    negative (progress can only add, the altitude term is ≥ 0). Nothing is "saved" by ending."""
    assert _step(h=0.5) > 0.0
    assert _step(progress_prev=2.0, progress_curr=1.0) > 0.0


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #
def test_progress_toward_target_is_rewarded() -> None:
    closing = _step(progress_prev=2.0, progress_curr=1.0)  # closed 1.0
    receding = _step(progress_prev=1.0, progress_curr=2.0)  # opened 1.0
    assert closing > receding
    assert closing == pytest.approx(CFG.progress_weight * 1.0)
    assert receding == pytest.approx(-CFG.progress_weight * 1.0)


# --------------------------------------------------------------------------- #
# Altitude reward — the sole per-step term (UC-58)
# --------------------------------------------------------------------------- #
def test_altitude_reward_is_zero_at_the_floor() -> None:
    """AC3: the altitude reward pays exactly 0 at ``h = 0`` (its floor value) — no offset."""
    assert _step(h=0.0) == pytest.approx(0.0)


def test_altitude_reward_strictly_increases_from_floor_to_target() -> None:
    """AC3/AC4: strictly increasing on ``[0, altitude_target]`` — the takeoff gradient, no dead
    zone. Every centimetre of lift below the target pays strictly more than the one below it."""
    heights = [0.0, 0.1, 0.25, 0.5, 0.75, CFG.altitude_target]
    vals = [_step(h=h) for h in heights]
    for lo, hi in zip(vals, vals[1:], strict=False):
        assert hi > lo


def test_altitude_reward_saturates_at_and_above_target() -> None:
    """AC4: level + saturating — at/above ``altitude_target`` the term is flat at its ceiling
    ``altitude_weight`` (no ceiling-seeking, no farm-by-overshoot)."""
    at_target = _step(h=CFG.altitude_target)
    above = _step(h=CFG.altitude_target * 3.0)
    assert at_target == pytest.approx(CFG.altitude_weight)
    assert above == pytest.approx(CFG.altitude_weight)
    assert above == pytest.approx(at_target)


def test_altitude_reward_value_matches_the_linear_formula() -> None:
    """The exact algebra: ``altitude_weight · clamp(h,0,target)/target`` at a mid height."""
    h = 0.4
    expected = CFG.altitude_weight * (h / CFG.altitude_target)
    assert _step(h=h) == pytest.approx(expected)


def test_altitude_reward_clamps_negative_height_to_zero() -> None:
    """A drone reported below the floor (h < 0) earns 0 altitude reward, never a negative one."""
    assert _step(h=-0.3) == pytest.approx(0.0)


def test_altitude_reward_scaled_by_per_step_scale() -> None:
    """UC-57 rate-invariance: ``per_step_scale`` multiplies ONLY the altitude term (the sole
    per-step term left after the redesign). At scale 0.4 the per-step payout scales to 0.4×."""
    full = _step(h=CFG.altitude_target, per_step_scale=1.0)
    scaled = _step(h=CFG.altitude_target, per_step_scale=0.4)
    assert scaled == pytest.approx(0.4 * full)
    assert full == pytest.approx(CFG.altitude_weight)


def test_per_step_scale_does_not_touch_event_terms() -> None:
    """UC-57: progress / gate / completion / collision / obstacle are per-event or path-extensive —
    ``per_step_scale`` must NOT rescale them (only the time-extensive altitude term). Here h=0 so
    the altitude term is 0 and the two scales must yield an identical event-only reward."""
    base = dict(progress_prev=2.0, progress_curr=1.0, event="gate", completed=True, h=0.0)
    assert _step(**base, per_step_scale=1.0) == pytest.approx(_step(**base, per_step_scale=0.4))


# --------------------------------------------------------------------------- #
# Collision / obstacle penalties
# --------------------------------------------------------------------------- #
def test_collision_is_penalised_and_negative() -> None:
    r = _step(collided=True)
    assert r == pytest.approx(-CFG.collision_penalty)
    assert r < 0.0


def test_collision_penalty_is_only_charged_when_collided() -> None:
    """UC-58: the collision penalty is charged ONLY on a genuine crash flag — a penalty-free
    grounded-rest (``collided=False``) at the floor costs nothing (the anti-suicide guarantee)."""
    assert _step(collided=False, h=0.0) == pytest.approx(0.0)


def test_obstacle_penalty_subtracted_iff_contact_flag() -> None:
    with_contact = _step(obstacle_contact=True)
    without = _step(obstacle_contact=False)
    assert with_contact == pytest.approx(-CFG.obstacle_penalty)
    assert without == pytest.approx(0.0)
    assert without - with_contact == pytest.approx(CFG.obstacle_penalty)


def test_obstacle_penalty_defaults_to_no_contact() -> None:
    """The ``obstacle_contact`` kwarg defaults to False, so pre-UC-15 callers pay no penalty."""
    r = compute_reward(
        dist_to_target_prev=0.0,
        dist_to_target_curr=0.0,
        event=None,
        collided=False,
        completed=False,
        cfg=CFG,
    )
    assert r == pytest.approx(0.0)


def test_obstacle_penalty_default_is_severe_but_below_terminal_collision() -> None:
    assert 0 < CFG.obstacle_penalty < CFG.collision_penalty


def test_obstacle_penalty_magnitude_is_tunable() -> None:
    cfg = RewardConfig(obstacle_penalty=12.5)
    assert _step(obstacle_contact=True, cfg=cfg) == pytest.approx(-12.5)


# --------------------------------------------------------------------------- #
# Gate bonus + normalisation (UC-09 AC4)
# --------------------------------------------------------------------------- #
def test_gate_bonus_awarded_on_gate_event() -> None:
    assert _step(event="gate") == pytest.approx(CFG.gate_bonus)
    assert _step(event=None) == pytest.approx(0.0)


def test_single_gate_bonus_equals_gate_bonus_exactly() -> None:
    assert _step(event="gate", num_gates=1) == pytest.approx(CFG.gate_bonus)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_per_gate_bonus_is_gate_bonus_over_n(n: int) -> None:
    assert _step(event="gate", num_gates=n) == pytest.approx(CFG.gate_bonus / n)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_n_gate_bonuses_sum_to_gate_bonus(n: int) -> None:
    total = sum(_step(event="gate", num_gates=n) for _ in range(n))
    assert total == pytest.approx(CFG.gate_bonus)


def test_larger_n_gives_smaller_per_gate_bonus() -> None:
    assert _step(event="gate", num_gates=2) > _step(event="gate", num_gates=5)


def test_num_gates_zero_is_guarded_against_div_by_zero() -> None:
    # max(1, num_gates) guards the divisor.
    assert _step(event="gate", num_gates=0) == pytest.approx(CFG.gate_bonus)


# --------------------------------------------------------------------------- #
# Completion bonus — dominant, un-normalised (objective hierarchy, AC5)
# --------------------------------------------------------------------------- #
def test_completion_bonus_only_on_valid_completion() -> None:
    assert _step(completed=True) == pytest.approx(CFG.completion_bonus)
    assert _step(completed=False) == pytest.approx(0.0)


def test_completion_bonus_is_not_normalised_by_n() -> None:
    """The completion bonus is a one-off, independent of the gate count (unlike the gate bonus)."""
    for n in (1, 3, 7):
        assert _step(completed=True, num_gates=n) == pytest.approx(CFG.completion_bonus)


def test_completion_dominates_collision_magnitude() -> None:
    assert CFG.completion_bonus > CFG.collision_penalty


def test_valid_completion_beats_shortcut_through_floor() -> None:
    """A valid completion strictly out-scores a same-step floor crash (completion + no penalty vs a
    −collision_penalty crash), so diving through the floor is never a shortcut to a better score."""
    completing = _step(event="finish", completed=True)
    crashing = _step(collided=True)
    assert completing > crashing


# --------------------------------------------------------------------------- #
# Return type
# --------------------------------------------------------------------------- #
def test_reward_is_a_python_float() -> None:
    assert isinstance(_step(h=0.5, event="gate"), float)
