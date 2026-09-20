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
    airborne: bool = False,
    height_above_floor_prev: float = 0.0,
    height_above_floor_curr: float = 0.0,
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
    airborne:
        Whether the drone is above the floor band this step (UC-37 AC5). When ``True`` the per-step
        ``cfg.airborne_bonus`` survival reward is added; it is exactly zero on/at the floor, so
        sitting on the ground earns nothing and the only path to reward is to take off and stay up.
        The env raises this flag only when the drone's altitude exceeds ``floor_z + floor_epsilon``.
        Default ``False`` keeps every pre-UC-37 caller byte-identical (no survival term).
    height_above_floor_prev, height_above_floor_curr:
        Drone altitude ABOVE the course floor (``position[2] − floor_z``) before and after the
        step (UC-39 AC1/AC2). They drive a **dense potential-based climb reward** that pays for
        upward progress toward ``cfg.climb_target_height`` on the very step altitude is gained —
        before and independent of any later crash — so PPO's per-action advantage for the initial
        "throttle up" actions of a takeoff is positive even when the attempt later crashes. The term
        is potential-based (Ng et al. 1999): Φ(h) = ``cfg.climb_weight`` · min(max(h, 0),
        ``cfg.climb_target_height``); the per-step contribution is
        F = ``cfg.climb_gamma`` · Φ(curr) − Φ(prev). Consequences (all unit-testable): it
        telescopes so a climb-then-descend round trip nets ≈0 (non-farmable, no loiter optimum);
        the per-episode total is bounded by ≈ ``climb_gamma`` · ``climb_weight`` ·
        ``climb_target_height`` ≪ ``completion_bonus``; it is ≈0 on the floor (h ≈ 0 ⇒ Φ ≈ 0) and
        ≈0 for steps taken above the target (Φ saturates ⇒ no ceiling-seeking). ``cfg.climb_gamma``
        MUST equal the training γ for the invariance to hold (see
        :class:`~drone_fly.env.config.RewardConfig`). Both default to ``0.0`` ⇒ Φ_prev = Φ_curr = 0
        ⇒ F = 0, so every pre-UC-39 caller is byte-identical.
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
    if airborne:
        reward += cfg.airborne_bonus
    # UC-39: dense potential-based climb shaping (F = γΦ' − Φ, Φ = w·min(max(h,0), target)).
    # Telescoping ⇒ non-farmable; capped at the target ⇒ no ceiling-seeking; ≈0 on the floor.
    phi_prev = cfg.climb_weight * min(max(height_above_floor_prev, 0.0), cfg.climb_target_height)
    phi_curr = cfg.climb_weight * min(max(height_above_floor_curr, 0.0), cfg.climb_target_height)
    reward += cfg.climb_gamma * phi_curr - phi_prev
    return float(reward)
