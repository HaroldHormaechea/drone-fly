"""AC3 — reward function: fastest valid completion wins; collisions penalised.

:func:`drone_fly.env.reward.compute_reward` is pure (scalars + bools + a config), so the
"documented and unit-tested on hand-constructed transitions" contract is exercised here
without any env or sim. The key ordering guarantees:

* faster completion → higher return (per-step time penalty),
* a collision → strictly penalised,
* a shortcut through the floor scores strictly worse than a valid completion,
* the completion bonus fires *only* on a valid gate-then-finish (``completed=True``).
"""

from __future__ import annotations

from drone_fly.env.config import RewardConfig
from drone_fly.env.reward import compute_reward

CFG = RewardConfig()


def _step(
    *, progress_prev=0.0, progress_curr=0.0, event=None, collided=False, completed=False
) -> float:
    return compute_reward(
        dist_to_target_prev=progress_prev,
        dist_to_target_curr=progress_curr,
        event=event,
        collided=collided,
        completed=completed,
        cfg=CFG,
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
    gate = _step(event="gate")
    none = _step(event=None)
    assert gate - none == CFG.gate_bonus


def test_completion_dominates_collision_magnitude() -> None:
    # Config invariant that makes the ordering robust: a valid completion's bonus outweighs
    # a single collision penalty, so a clean finish always beats crashing into the goal.
    assert CFG.completion_bonus >= CFG.collision_penalty
