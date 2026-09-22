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

# UC-50 shipped enabled-by-default; these base-component tests pin the pre-UC-50 reward pieces
# in isolation, so they use an explicit feature-OFF config.
CFG = RewardConfig(enable_altitude_decoupling=False, altitude_hold_weight=0.0)


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


def _step_climb(
    *,
    h_prev=0.0,
    h_curr=0.0,
    airborne=False,
    progress_prev=0.0,
    progress_curr=0.0,
    collided=False,
    completed=False,
    event=None,
    cfg=CFG,
) -> float:
    """UC-39 helper: a step with the two climb heights (above the floor) threaded through."""
    return compute_reward(
        dist_to_target_prev=progress_prev,
        dist_to_target_curr=progress_curr,
        event=event,
        collided=collided,
        completed=completed,
        cfg=cfg,
        airborne=airborne,
        height_above_floor_prev=h_prev,
        height_above_floor_curr=h_curr,
    )


def _climb_contribution(h_prev: float, h_curr: float) -> float:
    """The climb term ALONE for a height transition: (step with heights) − (step at h=0)."""
    return _step_climb(h_prev=h_prev, h_curr=h_curr) - _step_climb(h_prev=0.0, h_curr=0.0)


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
        cfg = RewardConfig(
            obstacle_penalty=magnitude,
            enable_altitude_decoupling=False,
            altitude_hold_weight=0.0,
        )
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
    """AC5 (UC-42 altitude-graded): at/above the climb target the airborne survival payout
    SATURATES at exactly ``airborne_bonus``; a grounded (on/at-floor) step adds nothing. The
    airborne and not-airborne steps are BOTH evaluated at ``h == target`` so the shared
    potential-based climb term cancels and their difference isolates the (now graded) airborne
    payout at its saturated maximum. UC-42 also makes an airborne step AT THE FLOOR pay ≈0 (the
    payout is graded by fractional height), where UC-37's flat bonus paid the full amount there."""
    target = CFG.climb_target_height
    airborne_at_target = _step_climb(h_prev=target, h_curr=target, airborne=True)
    not_airborne_at_target = _step_climb(h_prev=target, h_curr=target, airborne=False)
    # Graded payout saturates at ``airborne_bonus`` for h ≥ target (== UC-37's flat value there).
    assert airborne_at_target - not_airborne_at_target == pytest.approx(CFG.airborne_bonus)
    # A grounded step on the floor earns nothing beyond the usual time penalty.
    assert _step_air(airborne=False) == pytest.approx(-CFG.time_penalty)
    # UC-42 contract change: an airborne step AT THE FLOOR (h ≈ 0) now pays ≈0 (graded), not a flat
    # bonus — so hovering just off the ground is no longer rewarded like holding the target.
    assert _step_climb(h_prev=0.0, h_curr=0.0, airborne=True) == pytest.approx(-CFG.time_penalty)


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
    """AC6a: the net per-airborne-step reward AT THE TARGET (``airborne_bonus − time_penalty``) is
    strictly positive, so staying airborne beats sinking/crashing and there is a gradient toward
    takeoff. UC-42 raised ``airborne_bonus`` 0.1 → 0.2, so this net rose 0.05 → 0.15."""
    net = CFG.airborne_bonus - CFG.time_penalty
    assert net > 0
    assert net == pytest.approx(0.15)  # 0.20 − 0.05, the UC-42 constants


def test_max_episode_survival_reward_is_below_completion_bonus() -> None:
    """UC-37 AC6b + UC-39 AC5 (extended): over the DEFAULT 3-gate step budget, the maximum
    NON-completion reward accruable — airborne survival (``airborne_bonus × budget``) PLUS the whole
    climb reward attainable in one episode (``climb_gamma × climb_weight × climb_target_height`` =
    1.98, since the climb term telescopes and caps at the target) — is STILL strictly below
    ``completion_bonus``. So even a policy that loiters airborne AND farms every last drop of climb
    reward scores worse than one that reaches gates and finishes (UC-39 Note 3).

    The budget is computed exactly as :class:`RaceEnv` does at reset
    (``max_steps + steps_per_gate × (num_gates − 1)``) for the shipped default env, so this bound
    tracks the real default rather than a hard-coded magic number.
    """
    from drone_fly.env.config import EnvConfig, EpisodeConfig

    ep = EpisodeConfig()
    course = EnvConfig().course
    budget = ep.max_steps + ep.steps_per_gate * (course.num_gates - 1)
    assert budget == 800  # 400 + 200 × (3 − 1): the default 3-gate budget the plan anchors AC6b to
    # UC-42: max survival = ``airborne_bonus`` per step (the payout saturates at the target, so a
    # policy hovering AT the target for the whole budget banks the full per-step bonus each step).
    max_survival = CFG.airborne_bonus * budget
    assert max_survival == pytest.approx(160.0)  # 0.20 × 800 (was 0.10 × 800 = 80 pre-UC-42)
    # UC-39: the per-episode climb bound (telescoping ⇒ ≈ γ·w·target, reached by a from-floor jump
    # to the target height) is the MOST climb reward any single episode can accrue.
    max_climb = CFG.climb_gamma * CFG.climb_weight * CFG.climb_target_height
    assert max_climb == pytest.approx(1.98)
    combined_max_non_completion = max_survival + max_climb
    assert combined_max_non_completion == pytest.approx(161.98)  # 160 + 1.98 (UC-42; was 81.98)
    # UC-42 raised ``completion_bonus`` 100 → 200 precisely to keep this loiter < completion bound
    # after the airborne bump (161.98 < 200, headroom ≈ 1.23 ≈ the original ≈ 1.22).
    assert CFG.completion_bonus == pytest.approx(200.0)
    assert combined_max_non_completion < CFG.completion_bonus  # loiter+climb < completing (AC5)


# --- UC-38 AC1: no-progress / timeout cut carries no collision penalty ----------------
def test_uc38_stuck_or_timeout_cut_reward_has_no_collision_penalty() -> None:
    """UC-38 AC1: a no-progress ("stuck") cut or a pure ``max_steps`` timeout is decoupled from the
    collision penalty at the reward level. The env passes ``collided=False`` for those cuts (only a
    genuine crash or a grounded drop passes True), so the step reward is the ORDINARY non-collision
    reward — time penalty + progress + any airborne survival bonus — with NO −collision_penalty
    term. This is the reward-side pin for the −105-trap fix."""
    # Airborne no-progress hover at the moment of the stuck cut: no progress, no event, airborne ⇒
    # survival bonus; collided is False because the cut is decoupled from the crash penalty. UC-42's
    # airborne payout is altitude-graded, so thread a REAL height (hover AT the target) — the
    # saturated payout is ``airborne_bonus`` and the shared climb potential contributes only its
    # tiny standing tax ((γ−1)·w·target = −0.02); crucially there is still NO −collision_penalty.
    target = CFG.climb_target_height
    standing_tax = (CFG.climb_gamma - 1.0) * CFG.climb_weight * CFG.climb_target_height
    # UC-43: the ground-break potential also levies a constant above-threshold standing tax
    # ((γ−1)·ground_break_weight = −0.005/step) at this above-threshold hover.
    gb_standing_tax = (CFG.climb_gamma - 1.0) * CFG.ground_break_weight
    stuck_cut = compute_reward(
        dist_to_target_prev=1.0,
        dist_to_target_curr=1.0,
        event=None,
        collided=False,
        completed=False,
        cfg=CFG,
        airborne=True,
        height_above_floor_prev=target,
        height_above_floor_curr=target,
    )
    assert stuck_cut == pytest.approx(
        -CFG.time_penalty + CFG.airborne_bonus + standing_tax + gb_standing_tax
    )
    assert stuck_cut == pytest.approx(0.125)  # −0.05+0.20−0.02−0.005 = UC-43 net-hold at target
    assert stuck_cut > -CFG.collision_penalty, "no −collision_penalty on a decoupled stuck cut"
    # A pure timeout truncation step (here on/at the floor, no bonus) likewise carries no penalty —
    # it is just the per-step time penalty, nowhere near the −collision_penalty terminal.
    timeout_cut = _step(progress_prev=1.0, progress_curr=1.0)  # collided defaults False
    assert timeout_cut == pytest.approx(-CFG.time_penalty)
    assert timeout_cut > -CFG.collision_penalty
    # Sanity: a GENUINE collision (collided=True) still eats the full penalty (contrast, AC2).
    genuine = _step(progress_prev=1.0, progress_curr=1.0, collided=True)
    assert genuine == pytest.approx(-CFG.time_penalty - CFG.collision_penalty)


# --- UC-39 AC1: dense potential-based climb reward -----------------------------------
def test_uc39_from_floor_climb_yields_positive_contribution() -> None:
    """AC1: a step that climbs (a substantial amount) from a floor start toward the target height
    yields a POSITIVE climb contribution, paid on the very step altitude is gained — before/indep.
    of any later crash. Uses a substantial climb (0 → 0.3 m), not a sub-1% knife-edge, so the
    positive signal is unambiguous."""
    contrib = _climb_contribution(h_prev=0.0, h_curr=0.3)
    assert contrib > 0.0
    # ``_climb_contribution`` isolates ALL height-driven shaping, which since UC-43 is the climb
    # potential PLUS the ground-break potential. For a 0 → 0.3 m step (0.3 clears the 0.05 m
    # ground-break threshold), both contribute their full from-floor amount:
    #   climb:        F   = γ·Φ(0.3) − Φ(0)     = 0.99·(2.0·0.3) − 0 = 0.594
    #   ground-break: F_gb = γ·Φ_gb(0.3) − Φ_gb(0) = 0.99·0.5     − 0 = 0.495 (saturated)
    climb_part = CFG.climb_gamma * (CFG.climb_weight * 0.3)
    ground_break_part = CFG.climb_gamma * CFG.ground_break_weight
    assert contrib == pytest.approx(climb_part + ground_break_part, abs=1e-9)
    assert contrib == pytest.approx(1.089, abs=1e-6)


def test_uc39_resting_on_floor_yields_no_climb_reward() -> None:
    """AC1: a step resting on the floor (no altitude gained, h ≈ 0) yields ≈0 climb reward, so
    sitting on the ground earns nothing from the climb term."""
    assert _climb_contribution(h_prev=0.0, h_curr=0.0) == pytest.approx(0.0, abs=1e-12)


def test_uc39_step_above_target_yields_no_additional_climb_reward() -> None:
    """AC1: a step taken entirely ABOVE the target height yields ≤0 additional climb reward — the
    potential saturates at ``climb_target_height`` so there is no incentive to climb into the
    ceiling. Both a flat above-target step and a climb-higher-above-target step are capped."""
    # Both endpoints above target ⇒ BOTH potentials saturate ⇒ the isolated height shaping is the
    # sum of the two constant standing taxes: climb (γ−1)·w·target = −0.02 and ground-break
    # (γ−1)·ground_break_weight = −0.005, i.e. −0.025 ≤ 0.
    flat_above = _climb_contribution(h_prev=1.2, h_curr=1.5)
    assert flat_above <= 0.0
    assert flat_above == pytest.approx(
        (CFG.climb_gamma - 1.0) * CFG.climb_weight
        + (CFG.climb_gamma - 1.0) * CFG.ground_break_weight,
        abs=1e-9,
    )
    # Climbing FROM the target further UP into the ceiling earns nothing extra (also capped ≤ 0).
    into_ceiling = _climb_contribution(h_prev=1.0, h_curr=2.4)
    assert into_ceiling <= 0.0


# --- UC-39 AC2: not farmable (telescoping) + per-episode bound -----------------------
def test_uc39_climb_round_trip_nets_approximately_zero() -> None:
    """AC2a: a round trip — climb by X then descend the same X — nets ≈0 climb reward (telescoping,
    so bobbing up and down cannot be farmed into a loiter optimum). The tiny residual is the γ<1
    discount and is ≤ 0 (never a positive farmable gain)."""
    up = _climb_contribution(h_prev=0.0, h_curr=0.5)
    down = _climb_contribution(h_prev=0.5, h_curr=0.0)
    round_trip = up + down
    assert round_trip == pytest.approx(0.0, abs=0.05)
    assert round_trip <= 0.0, "the γ<1 residual makes a round trip non-positive — not farmable"


def test_uc39_per_episode_climb_reward_is_bounded_below_completion_and_gate() -> None:
    """AC2b: the maximum total climb reward accruable over any single episode is
    ``climb_gamma × climb_weight × climb_target_height`` = 1.98 (a from-floor jump to the target;
    any further step at/above target pays ≤0, any descent pays negative). It is strictly below
    ``completion_bonus`` and at/below a single normalised 3-gate ``gate_bonus`` (3.33)."""
    # The single largest climb step (floor → target in one step) realises the whole bound. Since
    # UC-43 ``_climb_contribution`` also captures the ground-break potential, whose full from-floor
    # amount (γ·ground_break_weight = 0.495) is banked on that step, the isolated height shaping
    # is climb bound (1.98) + ground-break bound (0.495) = 2.475.
    max_single_step = _climb_contribution(h_prev=0.0, h_curr=CFG.climb_target_height)
    bound = CFG.climb_gamma * CFG.climb_weight * CFG.climb_target_height
    ground_break_bound = CFG.climb_gamma * CFG.ground_break_weight
    assert max_single_step == pytest.approx(bound + ground_break_bound)
    assert bound == pytest.approx(1.98)
    assert ground_break_bound == pytest.approx(0.495)
    # A further step held at the target adds ≤ 0 (both standing taxes), so the bound is the ceiling.
    assert _climb_contribution(h_prev=1.0, h_curr=1.0) <= 0.0
    # The CLIMB bound alone stays ≪ completion and ≤ a normalised 3-gate gate_bonus (unchanged); the
    # ground-break bound is tiny (0.495) and the combined-shaping < completion invariant is pinned
    # in test_uc43_ground_break_reward.py.
    assert bound < CFG.completion_bonus  # ≪ completion (200)
    assert bound <= CFG.gate_bonus / 3  # ≤ normalised gate bonus on the default 3-gate course
    assert bound + ground_break_bound < CFG.completion_bonus


# --- UC-39 AC6: floor-shortcut still loses to a valid completion ---------------------
def test_uc39_floor_shortcut_still_loses_to_valid_completion_even_with_climb() -> None:
    """AC6: a trajectory that dives through the floor to reach the finish still scores worse
    than a valid completion — even when the shortcut is credited the airborne survival bonus AND the
    maximum climb reward on that step. It eats ``collision_penalty`` and never earns
    ``completion_bonus``; the small climb/airborne credits cannot offset a 100-point crash."""
    # A valid completion at a useful altitude: earns the completion bonus, no collision.
    valid = _step_climb(
        progress_prev=1.0,
        progress_curr=0.0,
        event="finish",
        completed=True,
        h_prev=0.7,
        h_curr=1.0,
        airborne=True,
    )
    # A floor-shortcut that ALSO banks a full from-floor climb + airborne bonus, then crashes.
    shortcut = _step_climb(
        progress_prev=1.0,
        progress_curr=0.0,
        collided=True,
        completed=False,
        h_prev=0.0,
        h_curr=1.0,
        airborne=True,
    )
    assert valid > shortcut
    assert shortcut < 0 < valid


# --- UC-39 AC7: hover-at-target still beats sitting (climb "standing tax" accounted) -
def test_uc39_hover_at_target_nets_above_sitting_on_floor() -> None:
    """AC7: taking off and hovering AT the target height still returns strictly more per step than
    sitting on the floor — even after the potential-based climb term's tiny standing tax
    ((1−γ)·w·target = 0.02/step) AND the UC-43 ground-break term's constant above-threshold standing
    tax ((1−γ)·ground_break_weight = 0.005/step) are subtracted. UC-42 raised ``airborne_bonus``
    0.1 → 0.2 and UC-43 added the ground-break leak, so the hover net is 0.03 → 0.13 → 0.125: hover
    net = airborne_bonus − time_penalty − climb_tax − gb_tax = 0.2 − 0.05 − 0.02 − 0.005 =
    +0.125/step > sit = −time_penalty = −0.05/step."""
    hover_at_target = _step_climb(h_prev=1.0, h_curr=1.0, airborne=True)
    sit_on_floor = _step_climb(h_prev=0.0, h_curr=0.0, airborne=False)
    assert hover_at_target == pytest.approx(0.125, abs=1e-9)
    assert sit_on_floor == pytest.approx(-CFG.time_penalty)
    assert hover_at_target > sit_on_floor


# --- UC-39 AC8: anti-suicide (hover episode beats takeoff-then-immediate-crash) ------
def test_uc39_hover_episode_beats_takeoff_then_immediate_crash_at_worst_case_cp() -> None:
    """AC8: a full hovering episode must return strictly MORE than a take-off-then-immediately-crash
    episode, so the policy is never incentivised to end an episode early via a deliberate crash. The
    guarantee is tightest at the LOW curriculum endpoint (``collision_penalty_start``, UC-41 = 2.0),
    so this test evaluates the crash episode at the held curriculum start value — the worst case for
    anti-suicide (the smaller the crash penalty, the more attractive a deliberate crash becomes).

    Both episodes share an identical takeoff prefix (climb floor→target over K steps); they diverge
    only afterwards, so the comparison reduces to (continue hovering) vs (crash once at the held
    start CP). The inequality is collision-penalty-independent — the crash-step climb give-back
    (≈ −climb_weight·target) dominates — so it holds with margin at the lowered UC-41 start too."""
    from drone_fly.env.config import RewardConfig
    from drone_fly.train.config import TrainConfig

    cp_low = TrainConfig().collision_penalty_start
    assert cp_low == pytest.approx(2.0)
    crash_cfg = RewardConfig(
        collision_penalty=cp_low,  # worst-case low curriculum endpoint
        enable_altitude_decoupling=False,
        altitude_hold_weight=0.0,
    )

    target = CFG.climb_target_height

    def _takeoff_prefix_return(steps: int) -> float:
        """Climb from the floor to the target over ``steps`` equal increments (airborne)."""
        total = 0.0
        prev = 0.0
        for k in range(1, steps + 1):
            curr = target * k / steps
            total += _step_climb(h_prev=prev, h_curr=curr, airborne=True)
            prev = curr
        return total

    k = 5
    prefix = _takeoff_prefix_return(k)

    # Suicide episode: takeoff prefix, then a deliberate crash back to the floor at CP = 10.
    crash_step = _step_climb(
        h_prev=target, h_curr=0.0, airborne=False, collided=True, cfg=crash_cfg
    )
    suicide_return = prefix + crash_step

    # Hover episode: same takeoff prefix, then keep hovering at the target for M steps (no crash).
    m = 20
    hover_tail = sum(_step_climb(h_prev=target, h_curr=target, airborne=True) for _ in range(m))
    hover_return = prefix + hover_tail

    assert hover_return > suicide_return, (
        "hovering must beat a deliberate crash (anti-suicide, AC8)"
    )
    # And the crash episode is genuinely worse than even a SINGLE further hover step after takeoff.
    one_more_hover = prefix + _step_climb(h_prev=target, h_curr=target, airborne=True)
    assert one_more_hover > suicide_return
