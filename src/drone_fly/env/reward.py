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
    num_gates: int = 1,
    obstacle_contact: bool = False,
) -> float:
    """Return the scalar step reward.

    Parameters
    ----------
    dist_to_target_prev, dist_to_target_curr:
        Euclidean distance to the *current* target waypoint before and after the step.
        Their difference is the progress term (positive when closing in). The progress term
        needs **no** N-dependence: it telescopes across gate transitions, so its total over
        an episode is independent of the number of gates.
    event:
        ``"gate"`` on the step a gate is validly passed, ``"finish"`` on completion,
        else ``None`` — from :func:`drone_fly.env.geometry.advance`.
    collided:
        Whether the drone touched the floor or ceiling this step.
    completed:
        Whether the course was validly completed this step (all gates passed then finish).
        Passed in explicitly so the completion bonus can never fire before all gates.
    cfg:
        Reward weights.
    num_gates:
        Number of gates on the active course (UC-09 AC4). The per-gate bonus is
        **normalised** to ``gate_bonus / num_gates`` so an N-gate episode awards
        ``gate_bonus`` in total across its N gates, keeping gate reward comparable as N is
        randomized. For ``num_gates == 1`` the per-gate bonus equals ``gate_bonus`` exactly
        (backward compatible). ``max(1, num_gates)`` guards against a zero divisor.
    obstacle_contact:
        Whether the drone contacted a pillar obstacle this step (UC-15 AC2/AC9). When ``True``
        the SEVERE ``cfg.obstacle_penalty`` is subtracted. This flag is **edge-triggered by the
        env** (raised only on the step contact *begins*, not every overlapping step), so the
        total obstacle penalty over an episode is bounded by ``obstacle_penalty × distinct
        contacts`` and a sustained graze cannot stack unboundedly. Obstacle contact is a pure
        reward signal — it NEVER feeds episode termination (the env owns that), so a penalised
        drone may recover aerially and still complete the course. Default ``False`` keeps every
        pre-UC-15 caller byte-identical.
    """
    reward = -cfg.time_penalty
    reward += cfg.progress_weight * (dist_to_target_prev - dist_to_target_curr)
    if event == "gate":
        reward += cfg.gate_bonus / max(1, num_gates)
    if completed:
        reward += cfg.completion_bonus
    if collided:
        reward -= cfg.collision_penalty
    if obstacle_contact:
        reward -= cfg.obstacle_penalty
    return float(reward)
