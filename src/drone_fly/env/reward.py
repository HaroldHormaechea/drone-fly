"""Pure reward function for the racing env (AC3).

Encodes "fastest valid start→gate→finish" with explicit floor/ceiling penalties. Pure and
unit-testable on hand-constructed transitions — it takes scalars/bools, not env or sim
objects, and the completion signal is passed in **explicitly** so the bonus can *only*
fire on a valid gate-then-finish (the phase machine in :mod:`drone_fly.env.geometry` owns
that validity; this function never re-derives it).

Design guarantees (all unit-testable)
--------------------------------------
* **Faster completion → higher return.** A per-step ``time_penalty`` is subtracted every
  step, so an episode that completes in fewer steps accumulates less penalty and returns
  more.
* **Collision is penalised.** A floor/ceiling contact subtracts ``collision_penalty``.
* **Shortcut-through-floor scores worse than a valid completion.** A trajectory that dives
  through the floor to reach the finish never earns ``completion_bonus`` (it is not a
  valid completion) *and* eats ``collision_penalty``, so its return is strictly below a
  valid completion's. ``collision_penalty`` and ``completion_bonus`` are sized in
  :class:`~drone_fly.env.config.RewardConfig` to keep this ordering under any shaping.
* **Bonus only on valid completion.** ``completed`` must be the phase machine's DONE
  signal; the bonus is added iff ``completed`` is ``True``.
"""

from __future__ import annotations

from drone_fly.env.config import RewardConfig


def compute_reward(
    *,
    dist_to_target_prev: float,
    dist_to_target_curr: float,
    event: str | None,
    collided: bool,
    completed: bool,
    cfg: RewardConfig,
) -> float:
    """Return the scalar step reward.

    Parameters
    ----------
    dist_to_target_prev, dist_to_target_curr:
        Euclidean distance to the *current* target waypoint before and after the step.
        Their difference is the progress term (positive when closing in).
    event:
        ``"gate"`` on the step the gate is validly passed, ``"finish"`` on completion,
        else ``None`` — from :func:`drone_fly.env.geometry.advance_phase`.
    collided:
        Whether the drone touched the floor or ceiling this step.
    completed:
        Whether the course was validly completed this step (phase reached DONE). Passed in
        explicitly so the completion bonus can never fire on a finish-before-gate.
    cfg:
        Reward weights.
    """
    reward = -cfg.time_penalty
    reward += cfg.progress_weight * (dist_to_target_prev - dist_to_target_curr)
    if event == "gate":
        reward += cfg.gate_bonus
    if completed:
        reward += cfg.completion_bonus
    if collided:
        reward -= cfg.collision_penalty
    return float(reward)
