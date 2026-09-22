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
    target_height_above_floor_prev: float = 0.0,
    target_height_above_floor_curr: float = 0.0,
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
        survival reward is added; it is exactly zero on/at the floor, so sitting on the ground earns
        nothing and the only path to reward is to take off and stay up. The env raises this flag
        only when the drone's altitude exceeds ``floor_z + floor_epsilon``. Default ``False`` keeps
        every pre-UC-37 caller byte-identical (no survival term).

        **UC-42 contract change — the survival reward is now altitude-GRADED, not bool-only.** The
        payout is ``cfg.airborne_bonus · min(max(height_above_floor_curr, 0),
        cfg.climb_target_height) / cfg.climb_target_height`` — ``airborne_bonus`` scaled by
        fractional height toward the climb target, clamped to [0, 1] and saturating (flat) at/above
        the target. This makes holding a higher altitude net strictly better than hovering just off
        the floor (a flat bonus made it slightly *worse*, net of ``time_penalty`` and the discounted
        climb-potential leak), giving a monotone climb-to-target pull with no ceiling-seeking above
        the target. At ``h == target`` the payout equals ``airborne_bonus`` exactly, so UC-37's
        flat-bonus behaviour at the target is backward compatible.
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

        **UC-43 — the same two heights ALSO drive a sub-threshold "ground-breaking" potential.**
        UC-42's graded airborne bonus is gated off below the airborne threshold (the env only
        raises ``airborne`` above ``floor_z + floor_epsilon`` ≈ 0.05 m) and the UC-39 climb term is
        too weak there to overcome ``time_penalty``, so a fresh policy rests on the floor forever.
        A SECOND potential-based term supplies a strong positive gradient in the band
        [0, ``cfg.ground_break_height``]: Φ_gb(h) = ``cfg.ground_break_weight`` · min(max(h, 0),
        ``cfg.ground_break_height``) / ``cfg.ground_break_height``; per-step
        F_gb = ``cfg.climb_gamma`` · Φ_gb(curr) − Φ_gb(prev). It telescopes (non-farmable — a bob
        nets ≈0), is ≈0 at rest, and SATURATES at ``ground_break_height`` (chosen equal to the
        airborne threshold ``EarlyTerminationConfig.floor_epsilon``), so above the threshold Φ_gb is
        flat ⇒ F_gb = (γ−1)·``ground_break_weight`` per step, height-independent — a continuous
        handoff to the UC-42 bonus with no double-count. Its per-episode contribution is bounded by
        ≈ ``climb_gamma`` · ``ground_break_weight`` ≪ ``completion_bonus``.
    target_height_above_floor_prev, target_height_above_floor_curr:
        Height ABOVE the course floor of the CURRENT TARGET waypoint (``current_target(...)[2] −
        floor_z``) before and after the step's gate ``advance`` (UC-50). They straddle ``advance``
        exactly as ``dist_to_target_*`` do, so the potential-based altitude-shortfall term below
        telescopes across gate transitions. Both default ``0.0`` and are read ONLY when
        ``cfg.enable_altitude_decoupling`` is ``True`` ⇒ every pre-UC-50 caller is byte-identical.

        **UC-50 — altitude-holding forward-flight decoupling (default OFF).** When
        ``cfg.enable_altitude_decoupling`` is ``True`` the target-gate height drives two coupled
        mechanisms that make LEVEL, altitude-holding flight the rewarded path (the safe altitude
        tracks each gate's own z; the finish leg inherits the last gate's z via ``current_target``):

        * **Progress hard-gate.** The dense progress term is paid only when
          ``height_above_floor_curr ≥ max(ref_curr − cfg.altitude_band, cfg.ground_break_height)``,
          where ``ref_curr`` is the target-gate height LOWER-clamped to
          ``cfg.ground_break_height``. Below that band the POSITIVE
          progress reward is withheld (subtracted back) — "rush the gate while sinking" earns 0. The
          withhold only ever REDUCES reward (a below-band retreat, i.e. negative progress, keeps its
          penalty), so an approach/retreat loop cannot be farmed. The lower edge
          ``cfg.ground_break_height`` keeps the gate from ever demanding LESS altitude than the
          UC-43 bootstrap band, so a fresh floor-start policy still takes off (no chicken-and-egg).
        * **Altitude-hold shaping (potential-based).** Φ_track(h, ref) =
          ``cfg.altitude_hold_weight`` ·
          clamp(h − (ref − ``cfg.altitude_band``), 0, ``cfg.altitude_band``) — a NON-NEGATIVE
          altitude "credit" that is 0 at/below the band's lower edge, rises with height, and
          saturates at ``altitude_band`` once the drone reaches the reference. One-sided (flat above
          the reference — no overshoot reward). Per-step F = ``cfg.climb_gamma`` · Φ_track(curr) −
          Φ_track(prev). It is anchored non-negative on purpose — exactly like the UC-39 climb and
          UC-43 ground-break potentials — so the shaping leak (γ−1)·Φ is ≤ 0: a climb-then-descend /
          bob round trip nets ≤ 0 (non-farmable, NO loiter optimum) and hovering BELOW the reference
          is never positively rewarded. (The naive negative "penalise the shortfall" form −w·(ref−h)
          would invert this — its positive leak pays a drone to loiter below the reference — so it
          is deliberately NOT used.) Telescoping holds because the refs straddle ``advance`` and Φ
          is a pure function of state. The per-episode total is bounded by
          ≈ ``altitude_hold_weight`` · ``altitude_band`` ≪ ``completion_bonus``. ``cfg.climb_gamma``
          MUST equal the training γ
          (same coupling as the UC-39 climb term). ``ref`` is LOWER-clamped to
          ``cfg.ground_break_height`` (keeps the band non-degenerate for floor-band gates + hands
          off cleanly to the UC-43 ground-break band); the sizing invariant ``ref − band ≤
          climb_target_height`` is a documented ``RewardConfig`` constraint, not a runtime clamp.

        When the flag is off none of the above executes and the function is identical to UC-43.
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
        # UC-42: altitude-GRADED survival reward (was a flat per-step bonus). Pay ``airborne_bonus``
        # scaled by fractional height toward ``climb_target_height`` (clamped to [0, 1], flat
        # at/above the target). This flips the previously-perverse net-hold gradient — a flat bonus
        # made holding higher slightly WORSE net of ``time_penalty`` and the discounted climb-
        # potential leak — into a monotone climb-to-target pull, with no ceiling-seeking above the
        # target. At h == target the payout equals ``airborne_bonus`` (UC-37 behaviour preserved).
        h_frac = min(max(height_above_floor_curr, 0.0), cfg.climb_target_height)
        reward += cfg.airborne_bonus * (h_frac / cfg.climb_target_height)
    # UC-39: dense potential-based climb shaping (F = γΦ' − Φ, Φ = w·min(max(h,0), target)).
    # Telescoping ⇒ non-farmable; capped at the target ⇒ no ceiling-seeking; ≈0 on the floor.
    phi_prev = cfg.climb_weight * min(max(height_above_floor_prev, 0.0), cfg.climb_target_height)
    phi_curr = cfg.climb_weight * min(max(height_above_floor_curr, 0.0), cfg.climb_target_height)
    reward += cfg.climb_gamma * phi_curr - phi_prev
    # UC-43: dense potential-based SUB-THRESHOLD "ground-breaking" shaping (F_gb = γΦ_gb' − Φ_gb,
    # Φ_gb = w·min(max(h,0), gb_height)/gb_height). Pays a positive gradient in the band
    # [0, ground_break_height] — where UC-42's graded airborne bonus is gated OFF (the airborne
    # flag only trips above floor_z + floor_epsilon) — so PPO earns reward for the first few
    # centimetres of lift and can break the takeoff chicken-and-egg. Like the climb term it
    # telescopes (non-farmable: a bob nets ≈0), is ≈0 at rest (Φ_gb(0) = 0), and SATURATES at
    # ``ground_break_height`` so it hands off continuously to UC-42 at the airborne threshold
    # (above it Φ_gb is flat ⇒ F_gb = (γ−1)·w = −0.005/step, height-independent, no double-count).
    # ``ground_break_height`` MUST equal ``EarlyTerminationConfig.floor_epsilon`` (see the config).
    phi_gb_prev = cfg.ground_break_weight * (
        min(max(height_above_floor_prev, 0.0), cfg.ground_break_height) / cfg.ground_break_height
    )
    phi_gb_curr = cfg.ground_break_weight * (
        min(max(height_above_floor_curr, 0.0), cfg.ground_break_height) / cfg.ground_break_height
    )
    reward += cfg.climb_gamma * phi_gb_curr - phi_gb_prev
    # UC-50: altitude-holding forward-flight decoupling (default OFF ⇒ this whole block is skipped
    # and the function is byte-identical to UC-43). See the ``target_height_above_floor_*`` params'
    # docstring for the full contract. Two coupled mechanisms, both keyed off the TARGET GATE height
    # so the safe altitude tracks each gate's own z (and the finish leg via ``current_target``).
    if cfg.enable_altitude_decoupling:
        # Reference altitude = target-gate height above floor, LOWER-clamped to
        # ``ground_break_height``: this keeps the band non-degenerate for a floor-band gate (whose
        # raw height above floor may be ≈0) and preserves the clean handoff to the UC-43
        # ground-break bootstrap. No UPPER/ceiling clamp is applied — the band is ONE-SIDED (it only
        # penalises being BELOW the reference), and the sizing invariant ``ref − band ≤
        # climb_target_height`` is a DOCUMENTED config constraint (see ``RewardConfig``), not a
        # runtime clamp. The clamp is a pure function of the target height, so the potential still
        # telescopes across ``advance`` (refs straddle it, mirroring the climb term).
        ref_prev = max(target_height_above_floor_prev, cfg.ground_break_height)
        ref_curr = max(target_height_above_floor_curr, cfg.ground_break_height)
        # (a) Progress hard-gate. Pay progress only when height ≥ max(ref_curr − band,
        # ground_break_height); below that band withhold the POSITIVE progress reward (subtract it
        # back — only ever reduces reward). A below-band retreat (progress ≤ 0) keeps its penalty ⇒
        # the gate cannot be farmed by dropping below it and re-approaching. The
        # ``ground_break_height`` lower edge keeps takeoff bootstrapping (AC-5: no chicken-and-egg).
        progress = cfg.progress_weight * (dist_to_target_prev - dist_to_target_curr)
        gate_floor = max(ref_curr - cfg.altitude_band, cfg.ground_break_height)
        if height_above_floor_curr < gate_floor and progress > 0.0:
            reward -= progress
        # (b) Potential-based altitude-hold shaping. Φ_track(h, ref) = altitude_hold_weight ·
        # clamp(h − (ref − altitude_band), 0, altitude_band) — a NON-NEGATIVE altitude "credit" that
        # is 0 at/below the band's lower edge (ref − band), rises linearly with height, and
        # saturates at ``altitude_band`` once the drone reaches the reference. It is the mirror of
        # the altitude *shortfall* (credit = band − shortfall inside the band) but anchored
        # non-negative on purpose, exactly like the UC-39 climb and UC-43 ground-break potentials:
        # a non-negative Φ gives a SAFE leak (γ−1)·Φ ≤ 0, so a bob / climb-then-descend round trip
        # nets ≤ 0 (no loiter optimum) and holding BELOW the reference is never positively rewarded.
        # (A negative Φ = −w·shortfall — the naive "penalise the shortfall" form — inverts this: its
        # leak is POSITIVE, paying a drone to hover below the reference and paying MORE the larger
        # the shortfall, a farmable loiter incentive; hence the non-negative anchoring here.)
        # One-sided: flat above the reference (no overshoot reward). Per-step
        # F = climb_gamma · Φ_track(curr) − Φ_track(prev); telescopes (refs straddle ``advance``,
        # Φ is a pure function of state) ⇒ non-farmable, round trip nets ≤ 0; per-episode total
        # bounded by ≈ altitude_hold_weight · altitude_band ≪ completion_bonus.
        lo_prev = ref_prev - cfg.altitude_band
        lo_curr = ref_curr - cfg.altitude_band
        phi_track_prev = cfg.altitude_hold_weight * min(
            cfg.altitude_band, max(0.0, height_above_floor_prev - lo_prev)
        )
        phi_track_curr = cfg.altitude_hold_weight * min(
            cfg.altitude_band, max(0.0, height_above_floor_curr - lo_curr)
        )
        reward += cfg.climb_gamma * phi_track_curr - phi_track_prev
    return float(reward)
