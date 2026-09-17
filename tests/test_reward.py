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
