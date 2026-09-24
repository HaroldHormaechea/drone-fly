"""UC-58 — takeoff-oriented reward redesign: AC1–AC5 rollout-level invariants (hermetic).

These are the acceptance-criteria tests for the from-scratch reward redesign. They operate on
hand-constructed synthetic trajectories through the PURE :func:`drone_fly.env.reward.compute_reward`
(no env, no sim, no GPU/pybullet/training — AC10), summing per-step rewards into a rollout return.

DESIGN NOTE (challenger carry-forward). Reward VecNormalize is ON in training and the behavioral
takeoff verdict is deferred to the owner's fresh GPU retrain (AC10). So these tests are deliberately
**STRUCTURAL** — they assert *orderings*, *monotonicity*, and *correlation SIGN*, never absolute
reward margins. The one arithmetic assertion (AC5's ``0.2 × 800 = 160 < 200``) is a relationship
between shipped config CONSTANTS, not a claim about normalized returns, so it is legitimate.
"""

from __future__ import annotations

import pytest

from drone_fly.env.config import CourseConfig, EpisodeConfig, RewardConfig
from drone_fly.env.reward import compute_reward

CFG = RewardConfig()


def _default_step_budget() -> int:
    """The default per-episode step budget the reward bounds are anchored to (UC-09 scaling):
    ``max_steps + steps_per_gate · (num_gates − 1)`` = 400 + 200·2 = 800 on the default 3-gate
    course."""
    ep = EpisodeConfig()
    course = CourseConfig()
    return ep.max_steps + ep.steps_per_gate * (course.num_gates - 1)


def _reward_at(h: float, *, collided: bool = False, completed: bool = False, scale: float = 1.0):
    """A single bare step at height ``h`` (no progress / gate), optionally a crash or completion."""
    return compute_reward(
        dist_to_target_prev=0.0,
        dist_to_target_curr=0.0,
        event=None,
        collided=collided,
        completed=completed,
        cfg=CFG,
        height_above_floor_curr=h,
        per_step_scale=scale,
    )


def _rollout_return(
    heights, *, crash_at_end: bool = False, complete_at_end: bool = False, scale: float = 1.0
) -> float:
    """Sum the per-step reward over a height trajectory. The final step optionally flags a genuine
    crash (pays −collision_penalty) or a valid completion (pays +completion_bonus). A rollout that
    simply *ends* (neither flag) models the penalty-free grounded/stuck/timeout cut (UC-58)."""
    total = 0.0
    n = len(heights)
    for i, h in enumerate(heights):
        last = i == n - 1
        total += _reward_at(
            h,
            collided=crash_at_end and last,
            completed=complete_at_end and last,
            scale=scale,
        )
    return total


def _pearson(xs, ys) -> float:
    """Pearson correlation coefficient (pure python, no scipy). Returns 0.0 for a degenerate
    (zero-variance) series."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0.0 or vy == 0.0:
        return 0.0
    return cov / (vx * vy) ** 0.5


# =========================================================================== #
# AC1 — Anti-suicide invariant (both descent paths) + reward-vs-length correlation ≥ 0
# =========================================================================== #
def test_ac1_descending_to_floor_scores_below_holding_altitude() -> None:
    """AC1: a rollout that descends to the ground scores STRICTLY LOWER than an otherwise-identical
    one that holds altitude over the same horizon — the descent forgoes the sustained altitude
    reward. Tested for the penalty-free grounded-rest descent (``collided=False``)."""
    horizon = 20
    hold = [CFG.altitude_target] * horizon
    descend = [CFG.altitude_target * (1.0 - i / (horizon - 1)) for i in range(horizon)]  # → 0
    assert _rollout_return(hold) > _rollout_return(descend)


def test_ac1_genuine_crash_path_scores_below_holding_altitude() -> None:
    """AC1 (the second descent path): a rollout that descends AND ends in a genuine crash
    (``collided=True`` — a real floor/ceiling/OOB crash, not a gentle grounded-rest) scores far
    below one that holds altitude. The crash both forgoes altitude reward and pays −penalty."""
    horizon = 20
    hold = [CFG.altitude_target] * horizon
    dive_and_crash = [CFG.altitude_target * (1.0 - i / (horizon - 1)) for i in range(horizon)]
    assert _rollout_return(hold) > _rollout_return(dive_and_crash, crash_at_end=True)


def test_ac1_shorter_suicide_rollout_does_not_out_score_longer_hold() -> None:
    """AC1 (the −0.87 trap, inverted): the historical failure was SHORTER episodes scoring BETTER
    (reward negatively correlated with length). Here a short dive-to-floor rollout must NOT beat a
    longer altitude-holding rollout — the anti-suicide guarantee removes the incentive to end
    early."""
    short_suicide = [CFG.altitude_target, CFG.altitude_target / 2, 0.0]  # ends fast on the floor
    long_hold = [CFG.altitude_target] * 40
    assert long_hold and _rollout_return(long_hold) > _rollout_return(short_suicide)


def test_ac1_reward_vs_episode_length_correlation_is_non_negative() -> None:
    """AC1 (correlation SIGN, structural): across altitude-holding rollouts of increasing length,
    cumulative reward correlates NON-NEGATIVELY with episode length — the −0.87 suicide correlation
    is removed/inverted. (Longer time aloft accrues more sustained altitude reward, never less.)"""
    lengths = [5, 10, 20, 40, 80, 160]
    returns = [_rollout_return([CFG.altitude_target] * n) for n in lengths]
    assert _pearson(lengths, returns) >= 0.0
    # And concretely positive here (holding pays every step), not merely non-negative.
    assert _pearson(lengths, returns) > 0.5


# =========================================================================== #
# AC2 — No existence-penalty dominance: ending early is never net-positive vs staying up
# =========================================================================== #
def test_ac2_early_floor_cut_is_never_net_positive_versus_a_hold() -> None:
    """AC2: ending an episode early by contacting the floor is never net-positive versus staying
    airborne over the same-or-longer horizon. With ``time_penalty`` retired there is NO per-step
    cost to escape, and the grounded cut is penalty-free, so the early-floor rollout can at best
    tie (all-zero) and never exceed the altitude-holding rollout."""
    horizon = 30
    early_floor = [0.0] * horizon  # sits on the floor the whole time, then the penalty-free cut
    hold = [CFG.altitude_target] * horizon
    assert _rollout_return(early_floor) <= _rollout_return(hold)
    assert _rollout_return(early_floor) == pytest.approx(0.0)  # no existence penalty to "save"


def test_ac2_terminal_ground_cost_does_not_reward_early_termination() -> None:
    """AC2: the grounded-rest terminal is penalty-free (0), and because there is no per-step
    existence penalty, the "penalty saved by terminating early" is also 0 — so 0 ≥ 0 holds and a
    failed takeoff that settles onto the floor is not punished into a suicide optimum (the trap the
    retired UC-39/41 crash-cliff curriculum papered over)."""
    penalty_saved_per_step = _reward_at(0.0)  # a floor step costs nothing now
    terminal_ground_cost = 0.0  # grounded-rest is collided=False ⇒ no collision_penalty
    assert terminal_ground_cost >= -penalty_saved_per_step
    assert penalty_saved_per_step == pytest.approx(0.0)


# =========================================================================== #
# AC3 — Takeoff gradient from the floor (climb > grounded; positive gradient at h=0)
# =========================================================================== #
def test_ac3_climbing_from_floor_scores_above_staying_grounded() -> None:
    """AC3: a trajectory that leaves the ground and climbs toward the target scores STRICTLY ABOVE
    one that stays grounded over the same horizon — a smooth ground → airborne → hold signal."""
    horizon = 20
    grounded = [0.0] * horizon
    climb = [
        min(CFG.altitude_target, i * CFG.altitude_target / (horizon - 1)) for i in range(horizon)
    ]
    assert _rollout_return(climb) > _rollout_return(grounded)


def test_ac3_positive_gradient_immediately_off_the_floor_no_dead_zone() -> None:
    """AC3: the reward is strictly increasing the instant the drone leaves the floor — any nudge
    upward from h=0 pays strictly more than staying at h=0 (no dead zone, the UC-43 failure the
    ground-break potential patched is now structural)."""
    at_floor = _reward_at(0.0)
    for h in (0.001, 0.01, 0.05, 0.1):
        assert _reward_at(h) > at_floor


def test_ac3_gradient_holds_at_50hz_run_default_scale() -> None:
    """AC3 (rate robustness): the takeoff gradient is scale-invariant — at the 50 Hz run default
    (``per_step_scale = 0.4``) leaving the floor still pays strictly more than staying grounded."""
    assert _reward_at(0.1, scale=0.4) > _reward_at(0.0, scale=0.4)


# =========================================================================== #
# AC4 — Climb rewarded but non-farmable (loiter/oscillate < genuine climb; saturation)
# =========================================================================== #
def test_ac4_genuine_climb_beats_hovering_low() -> None:
    """AC4: climbing to the target and holding out-scores hovering low (the UC-50 loiter trap) —
    the level-based reward pays more the higher you are, so parking at low altitude is worse."""
    horizon = 30
    climb_hold = [min(CFG.altitude_target, i * CFG.altitude_target / 10) for i in range(horizon)]
    hover_low = [0.15] * horizon
    assert _rollout_return(climb_hold) > _rollout_return(hover_low)


def test_ac4_oscillation_cannot_farm_more_than_holding_at_target() -> None:
    """AC4 (non-farmable): oscillating above/below the target cannot out-earn simply holding AT the
    target — the reward saturates, so the above-target excursions are clamped (earn no bonus) while
    the below-target dips earn strictly less. Holding at the target is the per-step optimum."""
    horizon = 40
    hold_target = [CFG.altitude_target] * horizon
    # Oscillate between the floor and 2× the target (mean = target, but the highs are clamped).
    oscillate = [0.0 if i % 2 else 2 * CFG.altitude_target for i in range(horizon)]
    assert _rollout_return(hold_target) > _rollout_return(oscillate)


def test_ac4_no_bonus_for_climbing_above_the_target() -> None:
    """AC4 (saturation): flying above the target earns no more than sitting exactly at it — there
    is no ceiling-seeking incentive."""
    assert _reward_at(CFG.altitude_target * 2.5) == pytest.approx(_reward_at(CFG.altitude_target))


# =========================================================================== #
# AC5 — Objective hierarchy preserved (completion dominates takeoff/hold shaping)
# =========================================================================== #
def test_ac5_altitude_ceiling_is_below_completion_bonus() -> None:
    """AC5: the per-episode altitude-reward ceiling over the DEFAULT step budget
    (``altitude_weight × 800 = 160``) is STRICTLY below ``completion_bonus`` (200), so a policy that
    merely loiters at altitude scores below one that completes the course. A relationship between
    shipped CONSTANTS (not a normalized-return margin), so an exact assertion is legitimate."""
    budget = _default_step_budget()
    altitude_ceiling = CFG.altitude_weight * budget
    assert altitude_ceiling == pytest.approx(160.0)
    assert altitude_ceiling < CFG.completion_bonus


def test_ac5_a_completing_rollout_beats_any_pure_hover_rollout() -> None:
    """AC5: a rollout that completes the course out-scores the best-case pure-hover rollout (the
    full budget spent saturated at the target, earning the 160 altitude ceiling) — traversal stays
    at the top of the objective hierarchy above takeoff/hold shaping."""
    budget = _default_step_budget()
    best_hover = _rollout_return([CFG.altitude_target] * budget)
    # A completing rollout: fly at altitude for the budget AND bank the completion bonus at the end.
    completing = _rollout_return([CFG.altitude_target] * budget, complete_at_end=True)
    assert completing > best_hover
    # Even a short completing sprint beats the maxed-out hover (completion dominates shaping).
    short_complete = _rollout_return([CFG.altitude_target] * 10, complete_at_end=True)
    assert short_complete > best_hover
