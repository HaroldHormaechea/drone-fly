"""Pure reward function for the racing env (AC3).

Encodes "take off, hold altitude, then fly the course start→gate→finish" as ONE coherent,
first-principles reward (UC-58 redesign). Pure and unit-testable on hand-constructed
transitions — it takes scalars/bools, not env or sim objects, and the completion signal is
passed in **explicitly** so the bonus can *only* fire on a valid gate-then-finish (the phase
machine in :mod:`drone_fly.env.geometry` owns that validity; this function never re-derives it).

UC-58 — takeoff-oriented redesign (from scratch)
------------------------------------------------
The reward accreted ~15 use-cases of overlapping altitude machinery while the drone could not
fly (``time_penalty``, ``airborne_bonus``, the UC-39 ``climb_*`` potential, the UC-43
``ground_break_*`` potential, and the UC-50 ``altitude_hold_*`` decoupling group). Once the
plant became flight-capable (UC-55 rate loop + UC-56 Meteor75 T/W 2.5) that tangle was shown to
*incentivize early termination*: the per-step ``time_penalty`` made existence net-negative below
a useful altitude, and the potential-based terms telescoped to ≈0 net — so diving to the floor
to stop the bleed was optimal (reward-vs-episode-length correlation −0.87). This module replaces
all of it with a single **sustained, level-based, saturating altitude reward** paid from ``h=0``.

Reward composition (every term documented)
------------------------------------------
``reward = progress + gate_bonus/N + completion_bonus + altitude_reward
           − collision_penalty(genuine crash) − obstacle_penalty``

* **progress** (``progress_weight · Δdist``) — dense shaping toward the current target waypoint.
  It telescopes across gate transitions, so its episode total is independent of the gate count.
* **gate_bonus / N** — per-gate reward, normalised by ``num_gates`` so an N-gate course awards
  ``gate_bonus`` in total (UC-09 AC4).
* **completion_bonus** — one-off on a VALID all-gates-then-finish. Dominant positive reward: the
  objective hierarchy keeps traversal above takeoff/hold shaping (AC5). Sized so it strictly
  exceeds the maximum altitude reward accruable over a default episode (see ``RewardConfig``).
* **altitude_reward** — the takeoff/hold signal (UC-58, replaces the four retired altitude terms):
  ``altitude_weight · clamp(h, 0, altitude_target) / altitude_target``, scaled by
  ``per_step_scale`` and paid **every step from h=0**. It is:
    - *level-based* (a function of the current height, NOT a potential difference) ⇒ a SUSTAINED
      positive signal for being/staying up, so descending or terminating early is strictly worse
      than holding altitude over the same-or-longer horizon (AC1/AC2 — nothing is "saved" by
      ending the episode now that ``time_penalty`` is gone);
    - *paid from h=0* ⇒ a smooth takeoff gradient off the floor, no dead zone (AC3);
    - *saturating + level* ⇒ non-farmable by oscillation, and climbing to the target strictly
      beats loitering low (AC4).
* **collision_penalty** — subtracted on a *genuine* crash (floor/ceiling/OOB), passed in via
  ``collided``. A gentle grounded-rest after a failed takeoff is NOT a crash (the env sets
  ``collided=False`` for it, UC-58), so it is penalty-free and cannot re-create the suicide trap.
* **obstacle_penalty** — severe, non-terminating pillar-contact penalty (edge-triggered by the
  env, so a sustained graze cannot stack).

Design guarantees (all unit-testable)
--------------------------------------
* **Anti-suicide.** With no per-step existence cost and a sustained positive altitude reward, a
  rollout that stays/gains altitude out-scores an otherwise-identical one that descends or ends
  early (AC1/AC2).
* **Takeoff gradient from the floor.** ``altitude_reward`` is strictly increasing in ``h`` on
  ``[0, altitude_target]`` (AC3).
* **Non-farmable.** ``altitude_reward`` is bounded by ``altitude_weight`` per step and saturates
  at ``altitude_target``; hovering/oscillating cannot exceed genuine climb-to-target (AC4).
* **Hierarchy preserved.** ``completion_bonus`` > the per-episode altitude-reward ceiling (AC5).
* **Bonus only on valid completion.** ``completed`` must be the phase machine's DONE signal.
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
    height_above_floor_curr: float = 0.0,
    per_step_scale: float = 1.0,
) -> float:
    """Return the scalar step reward (UC-58 redesign).

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
        Whether a **genuine crash** penalty should be charged this step. The env passes ``True``
        only for a real floor/ceiling/OOB crash (UC-58: a gentle grounded-rest after a failed
        takeoff is NOT a crash and is passed as ``False``, so ending an episode by settling onto
        the floor is penalty-free — this is what keeps the anti-suicide invariant from being
        re-broken by a terminal cost).
    completed:
        Whether the course was validly completed this step (all gates passed then finish).
        Passed in explicitly so the completion bonus can never fire before all gates.
    cfg:
        Reward weights.
    num_gates:
        Number of gates on the active course (UC-09 AC4). The per-gate bonus is
        **normalised** to ``gate_bonus / num_gates`` so an N-gate episode awards
        ``gate_bonus`` in total across its N gates, keeping gate reward comparable as N is
        randomized. For ``num_gates == 1`` the per-gate bonus equals ``gate_bonus`` exactly.
        ``max(1, num_gates)`` guards against a zero divisor.
    obstacle_contact:
        Whether the drone contacted a pillar obstacle this step (UC-15 AC2/AC9). When ``True``
        the SEVERE ``cfg.obstacle_penalty`` is subtracted. This flag is **edge-triggered by the
        env** (raised only on the step contact *begins*, not every overlapping step), so the
        total obstacle penalty over an episode is bounded by ``obstacle_penalty × distinct
        contacts``. Obstacle contact is a pure reward signal — it NEVER feeds episode termination.
    height_above_floor_curr:
        Drone altitude ABOVE the course floor (``position[2] − floor_z``) AFTER the step (UC-58).
        Drives the sustained, level-based, saturating altitude reward
        ``altitude_reward(h) = altitude_weight · clamp(h, 0, altitude_target) / altitude_target``
        (scaled by ``per_step_scale``), paid on EVERY step from ``h=0``:

        * *Level-based* (a function of the current height, not a potential difference) ⇒ it is a
          SUSTAINED positive signal for holding altitude, so a descent / early termination scores
          strictly below holding-or-climbing over the same-or-longer horizon (AC1/AC2).
        * *Paid from h=0* ⇒ a smooth takeoff gradient off the floor, with no dead zone (AC3).
        * *Strictly increasing on ``[0, altitude_target]`` and flat at/above it* ⇒ climbing to the
          target strictly beats loitering low, and there is no ceiling-seeking above the target
          (AC4). Bounded by ``altitude_weight`` per step ⇒ non-farmable by oscillation.
    per_step_scale:
        Per-step time-scaling factor for the **rate/time-extensive** altitude reward (UC-57).
        When the control rate changes, an episode spans a different NUMBER of steps for the same
        wall-clock seconds, so any reward paid *per step* would silently rescale the return. The
        env passes ``per_step_scale = dt / BASELINE_DT`` (= 0.4 at 50 Hz, 1.0 at the 20 Hz
        baseline); it multiplies ONLY the altitude reward — the sole per-step term left after the
        UC-58 redesign — so its per-episode integral is rate-invariant (2.5× more steps × 0.4
        per-step == unchanged). The progress, gate, completion, collision, and obstacle terms are
        left UNSCALED: progress telescopes over distance (path-extensive), and the event
        bonuses/penalties are per-event. Default ``1.0`` ⇒ byte-identical at the 20 Hz baseline.
    """
    # Dense progress toward the current target waypoint (telescopes across gate transitions).
    reward = cfg.progress_weight * (dist_to_target_prev - dist_to_target_curr)
    # Per-gate reward, normalised so an N-gate course awards ``gate_bonus`` in total (UC-09 AC4).
    if event == "gate":
        reward += cfg.gate_bonus / max(1, num_gates)
    # One-off dominant reward on a VALID all-gates-then-finish (objective hierarchy top, AC5).
    if completed:
        reward += cfg.completion_bonus
    # Genuine floor/ceiling/OOB crash penalty (a penalty-free grounded-rest passes collided=False).
    if collided:
        reward -= cfg.collision_penalty
    # Severe, non-terminating pillar-contact penalty (edge-triggered by the env).
    if obstacle_contact:
        reward -= cfg.obstacle_penalty
    # UC-58: sustained, level-based, saturating altitude reward — the takeoff/hold signal. Paid
    # EVERY step from h=0 (no dead zone, AC3), strictly increasing on [0, altitude_target] then
    # flat (climb-to-target beats loiter, non-farmable, AC4), and — being level-based rather than a
    # telescoping potential — a genuinely sustained positive signal so holding/gaining altitude
    # beats descending or terminating early (AC1/AC2). Scaled by ``per_step_scale`` for UC-57
    # rate-invariance (the only per-step term left after the redesign; default 1.0 ⇒ byte-identical
    # at the 20 Hz baseline).
    h_frac = min(max(height_above_floor_curr, 0.0), cfg.altitude_target) / cfg.altitude_target
    reward += per_step_scale * cfg.altitude_weight * h_frac
    return float(reward)
