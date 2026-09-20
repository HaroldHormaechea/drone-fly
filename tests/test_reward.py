"""AC3 (UC-03) + AC4 (UC-09) — reward: fastest valid completion + normalised gate bonus.

:func:`drone_fly.env.reward.compute_reward` is pure (scalars + bools + a config), so the
"documented and unit-tested on hand-constructed transitions" contract is exercised here
without any env or sim. The key ordering guarantees:

* faster completion → higher return (per-step time penalty),
* a collision → strictly penalised,
* a shortcut through the floor scores strictly worse than a valid completion,
* the completion bonus fires *only* on a valid gate-then-finish (``completed=True``).

UC-09 AC4 adds the **normalised per-gate bonus**: each gate awards ``gate_bonus / N`` so an
N-gate episode awards ``gate_bonus`` in total (comparable as N is randomized), and N=1 is
exactly today's ``gate_bonus``.
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
) -> float:
    return compute_reward(
        dist_to_target_prev=progress_prev,
        dist_to_target_curr=progress_curr,
        event=event,
        collided=collided,
        completed=completed,
        cfg=CFG,
        num_gates=num_gates,
    )


def _step_air(
    *,
    progress_prev=0.0,
    progress_curr=0.0,
    airborne=False,
) -> float:
    """UC-37 helper: a bare step with the ``airborne`` survival flag threaded through."""
    return compute_reward(
        dist_to_target_prev=progress_prev,
        dist_to_target_curr=progress_curr,
        event=None,
        collided=False,
        completed=False,
        cfg=CFG,
        airborne=airborne,
    )


def test_time_penalty_applied_every_step() -> None:
    # An idle step (no progress, no events) costs exactly the time penalty.
    assert _step() == -CFG.time_penalty


def test_progress_toward_target_is_rewarded() -> None:
    closing = _step(progress_prev=2.0, progress_curr=1.0)  # closed 1.0
    receding = _step(progress_prev=1.0, progress_curr=2.0)  # opened 1.0
    assert closing > receding
    assert closing == -CFG.time_penalty + CFG.progress_weight * 1.0


def test_faster_completion_yields_higher_return() -> None:
    # Model two completions with identical shaping but different step counts: the return is
    # the summed per-step reward, so fewer steps => less accumulated time penalty => higher.
    def episode_return(n_steps: int) -> float:
        total = 0.0
        for _ in range(n_steps - 1):
            total += _step()  # cruising steps
        total += _step(event="finish", completed=True)  # completing step
        return total

    fast = episode_return(10)
    slow = episode_return(30)
    assert fast > slow


def test_collision_is_penalised_and_negative() -> None:
    r = _step(collided=True)
    assert r < 0
    assert r == -CFG.time_penalty - CFG.collision_penalty


def test_valid_completion_beats_shortcut_through_floor() -> None:
    # A valid completion earns the completion bonus and no collision penalty.
    valid = _step(progress_prev=1.0, progress_curr=0.0, event="finish", completed=True)
    # A "shortcut through the floor" reaches the goal region but is a collision, never a
    # valid completion: no bonus AND a collision penalty.
    shortcut = _step(progress_prev=1.0, progress_curr=0.0, collided=True, completed=False)
    assert valid > shortcut
    assert shortcut < 0 < valid


def test_completion_bonus_only_on_valid_completion() -> None:
    # Same transition, but completed flips. The bonus is exactly the delta, and it never
    # fires on a finish-before-gate (completed=False even if an event were present).
    with_bonus = _step(event="finish", completed=True)
    without_bonus = _step(event="finish", completed=False)
    assert with_bonus - without_bonus == CFG.completion_bonus


def test_gate_bonus_awarded_on_gate_event() -> None:
    # UC-09 AC4: for N=1 (the default) the per-gate bonus equals gate_bonus exactly.
    gate = _step(event="gate", num_gates=1)
    none = _step(event=None, num_gates=1)
    assert gate - none == CFG.gate_bonus


def test_completion_dominates_collision_magnitude() -> None:
    # Config invariant that makes the ordering robust: a valid completion's bonus outweighs
    # a single collision penalty, so a clean finish always beats crashing into the goal.
    assert CFG.completion_bonus >= CFG.collision_penalty


# --- UC-09 AC4: normalised per-gate bonus -------------------------------------------
def test_single_gate_bonus_equals_gate_bonus_exactly() -> None:
    """N=1 == pre-UC-09 behaviour: one gate event awards the full ``gate_bonus`` (AC4)."""
    delta = _step(event="gate", num_gates=1) - _step(event=None, num_gates=1)
    assert delta == CFG.gate_bonus


@pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
def test_per_gate_bonus_is_gate_bonus_over_n(n: int) -> None:
    """A single gate event awards exactly ``gate_bonus / N`` for the active N (AC4)."""
    delta = _step(event="gate", num_gates=n) - _step(event=None, num_gates=n)
    assert delta == pytest.approx(CFG.gate_bonus / n)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
def test_n_gate_bonuses_sum_to_gate_bonus(n: int) -> None:
    """N gate events over an N-gate episode sum to exactly ``gate_bonus`` total (AC4).

    This is the load-bearing invariant: total gate reward stays comparable as N varies.
    """
    baseline = _step(event=None, num_gates=n)  # per-step floor (time penalty only)
    per_gate = _step(event="gate", num_gates=n) - baseline
    total = n * per_gate
    assert total == pytest.approx(CFG.gate_bonus)


def test_larger_n_gives_smaller_per_gate_bonus() -> None:
    """More gates ⇒ each individual gate is worth proportionally less (AC4)."""
    one = _step(event="gate", num_gates=1) - _step(event=None, num_gates=1)
    ten = _step(event="gate", num_gates=10) - _step(event=None, num_gates=10)
    assert one > ten
    assert one == pytest.approx(10 * ten)


def test_num_gates_zero_is_guarded_against_div_by_zero() -> None:
    """The ``max(1, num_gates)`` guard means N=0 degrades to the N=1 bonus, never a crash."""
    delta = _step(event="gate", num_gates=0) - _step(event=None, num_gates=0)
    assert delta == CFG.gate_bonus


def test_completion_bonus_is_not_normalised_by_n() -> None:
    """Only the per-gate bonus is normalised; the one-off completion bonus is N-independent."""
    for n in (1, 4, 10):
        with_bonus = _step(completed=True, num_gates=n)
        without = _step(completed=False, num_gates=n)
        assert with_bonus - without == CFG.completion_bonus


# --- UC-15 AC2/AC9: the severe, tunable, non-terminating obstacle penalty ------------
def test_obstacle_penalty_subtracted_iff_contact_flag() -> None:
    """``obstacle_contact=True`` subtracts exactly ``obstacle_penalty``; False does not (AC2)."""
    hit = compute_reward(
        dist_to_target_prev=0.0,
        dist_to_target_curr=0.0,
        event=None,
        collided=False,
        completed=False,
        cfg=CFG,
        obstacle_contact=True,
    )
    no_hit = compute_reward(
        dist_to_target_prev=0.0,
        dist_to_target_curr=0.0,
        event=None,
        collided=False,
        completed=False,
        cfg=CFG,
        obstacle_contact=False,
    )
    assert no_hit - hit == CFG.obstacle_penalty
    assert hit == -CFG.time_penalty - CFG.obstacle_penalty


def test_obstacle_penalty_defaults_to_no_contact() -> None:
    """The ``obstacle_contact`` param defaults to ``False`` — pre-UC-15 callers unchanged."""
    default_call = compute_reward(
        dist_to_target_prev=1.0,
        dist_to_target_curr=1.0,
        event=None,
        collided=False,
        completed=False,
        cfg=CFG,
    )
    explicit_no = compute_reward(
        dist_to_target_prev=1.0,
        dist_to_target_curr=1.0,
        event=None,
        collided=False,
        completed=False,
        cfg=CFG,
        obstacle_contact=False,
    )
    assert default_call == explicit_no


def test_obstacle_penalty_magnitude_is_tunable() -> None:
    """The penalty magnitude is read from the config — a documented, tunable constant (AC9)."""
    for magnitude in (10.0, 50.0, 250.0):
        cfg = RewardConfig(obstacle_penalty=magnitude)
        hit = compute_reward(
            dist_to_target_prev=0.0,
            dist_to_target_curr=0.0,
            event=None,
            collided=False,
            completed=False,
            cfg=cfg,
            obstacle_contact=True,
        )
        assert hit == -cfg.time_penalty - magnitude


def test_obstacle_penalty_default_is_severe_but_below_terminal_collision() -> None:
    """Sized well above one normalised gate bonus (severe) yet below the terminal penalty (AC9)."""
    assert CFG.obstacle_penalty > CFG.gate_bonus  # severe vs a single gate reward
    assert CFG.obstacle_penalty < CFG.collision_penalty  # but not a terminal crash


def test_obstacle_contact_does_not_grant_or_block_completion_bonus() -> None:
    """Obstacle contact is a pure additive penalty — a completing step still earns the bonus.

    A glancing contact on the completing step penalises but does not cancel the completion
    bonus, so the drone can graze a pillar and still be rewarded for finishing (AC9).
    """
    completed_clean = compute_reward(
        dist_to_target_prev=1.0,
        dist_to_target_curr=0.0,
        event="finish",
        collided=False,
        completed=True,
        cfg=CFG,
        obstacle_contact=False,
    )
    completed_grazing = compute_reward(
        dist_to_target_prev=1.0,
        dist_to_target_curr=0.0,
        event="finish",
        collided=False,
        completed=True,
        cfg=CFG,
        obstacle_contact=True,
    )
    assert completed_clean - completed_grazing == CFG.obstacle_penalty
    # Even after the severe penalty a completion still nets strongly positive.
    assert completed_grazing > 0


# --- UC-37 AC5/AC6: airborne-only survival reward ------------------------------------
def test_airborne_step_yields_the_bonus_grounded_step_yields_none() -> None:
    """AC5: an airborne step adds exactly ``airborne_bonus``; a grounded (on/at-floor) step adds
    nothing — the survival reward is paid ONLY while off the ground, so sitting still earns zero."""
    airborne = _step_air(airborne=True)
    grounded = _step_air(airborne=False)
    assert airborne - grounded == pytest.approx(CFG.airborne_bonus)
    # A grounded step earns nothing beyond the usual time penalty (bonus is exactly zero on floor).
    assert grounded == pytest.approx(-CFG.time_penalty)
    assert airborne == pytest.approx(-CFG.time_penalty + CFG.airborne_bonus)


def test_airborne_defaults_to_grounded_for_pre_uc37_callers() -> None:
    """AC5/backward-compat: ``airborne`` defaults to False, so every pre-UC-37 caller (which never
    passes it) is byte-identical — no survival term leaks into the existing reward paths."""
    default_call = compute_reward(
        dist_to_target_prev=1.0,
        dist_to_target_curr=1.0,
        event=None,
        collided=False,
        completed=False,
        cfg=CFG,
    )
    explicit_grounded = _step_air(progress_prev=1.0, progress_curr=1.0, airborne=False)
    assert default_call == explicit_grounded


def test_net_per_airborne_step_is_strictly_positive() -> None:
    """AC6a: the net per-airborne-step reward (``airborne_bonus − time_penalty``) is strictly
    positive, so staying airborne beats sinking/crashing and there is a gradient toward takeoff."""
    net = CFG.airborne_bonus - CFG.time_penalty
    assert net > 0
    assert net == pytest.approx(0.05)  # 0.10 − 0.05, the shipped constants


def test_max_episode_survival_reward_is_below_completion_bonus() -> None:
    """AC6b: over the DEFAULT 3-gate step budget, the maximum survival reward accruable
    (``airborne_bonus × budget``) is strictly below ``completion_bonus``, so a policy that merely
    loiters airborne scores worse than one that reaches gates and finishes.

    The budget is computed exactly as :class:`RaceEnv` does at reset
    (``max_steps + steps_per_gate × (num_gates − 1)``) for the shipped default env, so this bound
    tracks the real default rather than a hard-coded magic number.
    """
    from drone_fly.env.config import EnvConfig, EpisodeConfig

    ep = EpisodeConfig()
    course = EnvConfig().course
    budget = ep.max_steps + ep.steps_per_gate * (course.num_gates - 1)
    assert budget == 800  # 400 + 200 × (3 − 1): the default 3-gate budget the plan anchors AC6b to
    max_survival = CFG.airborne_bonus * budget
    assert max_survival == pytest.approx(80.0)
    assert CFG.completion_bonus == pytest.approx(100.0)
    assert max_survival < CFG.completion_bonus  # loitering < completing (AC6b)


# --- UC-38 AC1: no-progress / timeout cut carries no collision penalty ----------------
def test_uc38_stuck_or_timeout_cut_reward_has_no_collision_penalty() -> None:
    """UC-38 AC1: a no-progress ("stuck") cut or a pure ``max_steps`` timeout is decoupled from the
    collision penalty at the reward level. The env passes ``collided=False`` for those cuts (only a
    genuine crash or a grounded drop passes True), so the step reward is the ORDINARY non-collision
    reward — time penalty + progress + any airborne survival bonus — with NO −collision_penalty
    term. This is the reward-side pin for the −105-trap fix."""
    # Airborne no-progress hover at the moment of the stuck cut: no progress, no event, airborne ⇒
    # survival bonus; collided is False because the cut is decoupled from the crash penalty.
    stuck_cut = compute_reward(
        dist_to_target_prev=1.0,
        dist_to_target_curr=1.0,
        event=None,
        collided=False,
        completed=False,
        cfg=CFG,
        airborne=True,
    )
    assert stuck_cut == pytest.approx(-CFG.time_penalty + CFG.airborne_bonus)
    assert stuck_cut > -CFG.collision_penalty, "no −collision_penalty on a decoupled stuck cut"
    # A pure timeout truncation step (here on/at the floor, no bonus) likewise carries no penalty —
    # it is just the per-step time penalty, nowhere near the −collision_penalty terminal.
    timeout_cut = _step(progress_prev=1.0, progress_curr=1.0)  # collided defaults False
    assert timeout_cut == pytest.approx(-CFG.time_penalty)
    assert timeout_cut > -CFG.collision_penalty
    # Sanity: a GENUINE collision (collided=True) still eats the full penalty (contrast, AC2).
    genuine = _step(progress_prev=1.0, progress_curr=1.0, collided=True)
    assert genuine == pytest.approx(-CFG.time_penalty - CFG.collision_penalty)
