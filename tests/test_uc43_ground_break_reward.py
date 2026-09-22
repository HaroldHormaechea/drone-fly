"""UC-43 — the sub-threshold "ground-breaking" reward gate + the no-dead-zone audit contract.

Why this test is structural / relative, not absolute (read first)
-----------------------------------------------------------------
The env trains with reward ``VecNormalize`` ON (``racing_env.py``), so any GLOBAL rescale of the
reward is a no-op — the normaliser divides it straight back out. This is the UC-38/UC-42 trap. So
every assertion here gates on a RELATIVE / STRUCTURAL property of the reward math:

  * a monotone sub-threshold rise earns strictly MORE cumulative shaping than staying planted (a
    positive gradient now exists in the band [0, ground_break_height] where none did before);
  * the term is potential-based ⇒ a bob (up then back) telescopes to ≈0 (non-farmable), and is
    ≈0 at rest (Φ_gb(0) = 0);
  * it SATURATES at ``ground_break_height`` (≡ the airborne threshold ``floor_epsilon``) so it
    hands off continuously to UC-42's graded airborne bonus with no double-count — above the
    threshold F_gb is a height-INDEPENDENT constant ``(γ−1)·ground_break_weight``;
  * marginal upward reward is strictly positive on BOTH sides of the seam (AC7 "no dead zone");
  * an EXPLICIT assertion that every OTHER reward term is UNCHANGED — proving the fix is an
    additive sub-threshold term, not a global rescale (which VecNormalize erases).

Behavioural takeoff is the user's GPU retrain, explicitly NOT this gate (the UC-40/42 lesson):
potential-based shaping is return-invariant, so a stronger sub-threshold slope is
necessary-but-maybe-not-sufficient. The numpy ``simple``-adapter smoke-train is likewise not
evidence of takeoff. This file only pins the reward MATH.
"""

from __future__ import annotations

import pytest

from drone_fly.env.config import (
    EarlyTerminationConfig,
    EnvConfig,
    EpisodeConfig,
    RewardConfig,
)
from drone_fly.env.reward import compute_reward

# UC-50 ships enabled-by-default; these UC-43 ground-break tests pin the pre-UC-50 shaping in
# isolation, so use an explicit feature-OFF config.
CFG = RewardConfig(enable_altitude_decoupling=False, altitude_hold_weight=0.0)
GAMMA = CFG.climb_gamma  # 0.99 — must equal training γ (asserted in the unchanged-terms test)
THRESHOLD = CFG.ground_break_height  # 0.05 m — saturation height ≡ floor_epsilon (asserted below)

# A config with the UC-39 climb potential switched OFF, so a step's height-driven shaping is the
# ground-break term ALONE. Used to pin the ground-break term's own properties (saturation,
# above-threshold height-independence, telescoping) in isolation from the climb potential, which is
# also active in the sub-threshold band. Only ``climb_weight`` is zeroed; every other knob (and the
# ground-break knobs in particular) is the shipped default.
GB_ONLY = RewardConfig(climb_weight=0.0, enable_altitude_decoupling=False, altitude_hold_weight=0.0)


def _reward(cfg: RewardConfig = CFG, **over: object) -> float:
    """A bare step through :func:`compute_reward` with overridable kwargs."""
    kwargs: dict[str, object] = dict(
        dist_to_target_prev=0.0,
        dist_to_target_curr=0.0,
        event=None,
        collided=False,
        completed=False,
        cfg=cfg,
    )
    kwargs.update(over)
    return compute_reward(**kwargs)  # type: ignore[arg-type]


def _gb_term(h_prev: float, h_curr: float) -> float:
    """The ground-breaking term F_gb ALONE for a height transition.

    Evaluated with the climb potential zeroed and ``airborne=False`` (so the airborne graded bonus
    never fires), then the ``h=0`` baseline (pure ``−time_penalty``) is differenced out, leaving
    exactly F_gb = γ·Φ_gb(curr) − Φ_gb(prev)."""
    step = _reward(
        cfg=GB_ONLY,
        airborne=False,
        height_above_floor_prev=h_prev,
        height_above_floor_curr=h_curr,
    )
    base = _reward(
        cfg=GB_ONLY, airborne=False, height_above_floor_prev=0.0, height_above_floor_curr=0.0
    )
    return step - base


def _phi_gb(h: float) -> float:
    """Reconstruct Φ_gb(h) from the term: F_gb(0→h) = γ·Φ_gb(h) − Φ_gb(0) = γ·Φ_gb(h)."""
    return _gb_term(0.0, h) / GAMMA


# --- Directive 1 / AC2: the coupling is asserted against its SOURCE, not a 0.05 pin ---------
def test_ground_break_height_is_coupled_to_floor_epsilon_at_source() -> None:
    """Directive 1 / Risk 2: ``ground_break_height`` MUST equal the airborne threshold
    ``EarlyTerminationConfig.floor_epsilon`` so the sub-threshold term saturates EXACTLY where
    UC-42's airborne bonus begins (a continuous handoff, no dead zone, no double-count). Asserted
    against the live source — NOT an independent ``0.05`` literal — so a future ``floor_epsilon``
    change can't silently reopen a dead zone or shift the seam without breaking this gate."""
    assert CFG.ground_break_height == EarlyTerminationConfig().floor_epsilon


# --- AC2: ≈0 at rest, non-farmable floor -----------------------------------------------------
def test_ground_break_is_zero_at_rest() -> None:
    """AC2: Φ_gb(0) = 0 ⇒ F_gb = 0 for a step that stays planted on the floor — sitting still earns
    nothing from the ground-break term, so the floor is non-farmable."""
    assert _gb_term(0.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    # And the FULL reward at rest (climb + ground-break both active) is just the time penalty.
    at_rest = _reward(airborne=False, height_above_floor_prev=0.0, height_above_floor_curr=0.0)
    assert at_rest == pytest.approx(-CFG.time_penalty)


# --- AC2: sub-threshold monotone rise → strictly-increasing return; planted earns less -------
def test_sub_threshold_monotone_rise_out_earns_staying_planted() -> None:
    """AC2 (the crux): a monotone rise from rest toward the threshold accrues strictly-INCREASING
    cumulative shaping, and staying planted earns strictly less. Uses the SHIPPED config (climb +
    ground-break together, ``airborne=False`` because the drone is still below the airborne
    threshold), so this is the real sub-threshold signal PPO sees. The positive gradient in
    [0, threshold] is exactly what was missing before UC-43."""
    steps = [i / 10.0 * THRESHOLD for i in range(11)]  # 0, .1t, ..., threshold (fine grid)
    # Per-step shaping over the pure time-penalty floor, as the drone rises one increment per step.
    planted_floor = _reward(
        airborne=False, height_above_floor_prev=0.0, height_above_floor_curr=0.0
    )
    cumulative = 0.0
    prev_cumulative = None
    for prev, curr in zip(steps, steps[1:], strict=False):
        step_reward = _reward(
            airborne=False, height_above_floor_prev=prev, height_above_floor_curr=curr
        )
        # Every rising step out-earns the planted step (its shaping over the floor is positive).
        assert step_reward > planted_floor, "a rising sub-threshold step must beat staying planted"
        cumulative += step_reward - planted_floor  # shaping accrued, net of the shared time penalty
        if prev_cumulative is not None:
            assert cumulative > prev_cumulative, "cumulative sub-threshold shaping must keep rising"
        prev_cumulative = cumulative
    # Total shaping of the full rise 0 → threshold is strictly positive; planted earns exactly 0.
    assert cumulative > 0.0


# --- AC2, pitfall 1: non-farmable — a bob telescopes to ≈0 -----------------------------------
def test_ground_break_bob_telescopes_to_zero_non_farmable() -> None:
    """AC2 / pitfall 1: a bob (rise then fall back to start) inside the sub-threshold band nets ≈0
    (only the γ<1 telescoping residual, which is ≤0), so upward-downward bobbing cannot be farmed
    into a loiter optimum."""
    up = _gb_term(0.0, 0.03)
    down = _gb_term(0.03, 0.0)
    round_trip = up + down
    assert round_trip == pytest.approx(0.0, abs=0.05)
    assert round_trip <= 0.0, "the γ<1 residual keeps a bob non-positive — not farmable"


# --- AC2: saturation at the threshold + height-independent handoff above it ------------------
def test_phi_gb_saturates_at_weight_on_the_threshold() -> None:
    """AC2: Φ_gb saturates at ``ground_break_weight`` exactly at ``ground_break_height`` — reaching
    the threshold has captured the term's whole potential, and climbing further adds no more Φ_gb
    (so it hands off cleanly to the airborne bonus which takes over above the threshold)."""
    assert _phi_gb(THRESHOLD) == pytest.approx(CFG.ground_break_weight)
    # Above the threshold Φ_gb is flat: a from-floor step to 2× / 4× the threshold captures the SAME
    # saturated potential as a step to exactly the threshold.
    assert _phi_gb(2 * THRESHOLD) == pytest.approx(CFG.ground_break_weight)
    assert _phi_gb(4 * THRESHOLD) == pytest.approx(CFG.ground_break_weight)


def test_above_threshold_step_is_height_independent_no_double_count() -> None:
    """AC2 / pitfall 2: for any step taken FULLY above the threshold, F_gb collapses to the
    height-INDEPENDENT constant ``(γ−1)·ground_break_weight`` (= −0.005/step) — it adds no
    height-dependent reward above the threshold, so it never double-counts on top of UC-42's
    graded airborne bonus or the climb potential. A continuous, no-jump handoff at the seam."""
    expected = (GAMMA - 1.0) * CFG.ground_break_weight
    assert expected == pytest.approx(-0.005)
    # Different above-threshold transitions all yield the SAME constant (height-independent).
    assert _gb_term(2 * THRESHOLD, 2 * THRESHOLD) == pytest.approx(expected)
    assert _gb_term(2 * THRESHOLD, 10 * THRESHOLD) == pytest.approx(expected)
    assert _gb_term(0.5, 0.9) == pytest.approx(expected)
    assert _gb_term(1.0, 2.4) == pytest.approx(expected)


# --- AC2/AC7: strictly-positive marginal upward reward on BOTH sides of the seam -------------
def _marginal_upward(h: float, dh: float) -> float:
    """Reward of stepping UP from ``h`` to ``h+dh`` minus the reward of holding at ``h``, using the
    SHIPPED config with the real airborne gate (airborne iff the altitude clears the threshold).
    Positive ⇒ moving up pays more than not moving ⇒ no dead zone at that altitude."""
    step_up = _reward(
        airborne=(h + dh) > THRESHOLD,
        height_above_floor_prev=h,
        height_above_floor_curr=h + dh,
    )
    hold = _reward(airborne=h > THRESHOLD, height_above_floor_prev=h, height_above_floor_curr=h)
    return step_up - hold


def test_marginal_upward_reward_is_strictly_positive_across_the_seam() -> None:
    """AC7 (no dead zone) / AC2 (continuity): moving up pays strictly more than holding at EVERY
    altitude across the airborne-threshold seam — below it (ground-break term), crossing it, and
    above it up to the climb target (airborne graded bonus + climb potential). There is no altitude
    band where genuine upward progress earns zero or negative marginal reward."""
    # Below the seam (both endpoints sub-threshold): the ground-break term supplies the gradient.
    assert _marginal_upward(0.0, 0.01) > 0.0
    assert _marginal_upward(0.02, 0.02) > 0.0
    # Crossing the seam (below → above): still strictly positive, no discontinuous drop.
    assert _marginal_upward(0.03, 0.2) > 0.0
    # Above the seam, up to the climb target: airborne graded bonus + climb potential.
    assert _marginal_upward(0.1, 0.3) > 0.0
    assert _marginal_upward(0.5, 0.4) > 0.0  # 0.5 → 0.9, still below target


# --- AC4: combined shaping bound stays below the completion bonus ----------------------------
def test_combined_shaping_bound_below_completion_bonus() -> None:
    """AC4: the maximum NON-completion shaping over the default 800-step budget — airborne survival
    (``airborne_bonus × budget`` = 160, saturating) + the whole climb bound (1.98) + the whole
    ground-break bound (``climb_gamma × ground_break_weight`` = 0.495, telescoping) — stays strictly
    below ``completion_bonus`` (200). Adding the ground-break term does NOT break the loiter <
    completion ordering (headroom ≈ 37.5)."""
    ep = EpisodeConfig()
    course = EnvConfig().course
    budget = ep.max_steps + ep.steps_per_gate * (course.num_gates - 1)
    assert budget == 800
    max_survival = CFG.airborne_bonus * budget
    max_climb = CFG.climb_gamma * CFG.climb_weight * CFG.climb_target_height
    max_ground_break = CFG.climb_gamma * CFG.ground_break_weight
    assert max_survival == pytest.approx(160.0)
    assert max_climb == pytest.approx(1.98)
    assert max_ground_break == pytest.approx(0.495)
    combined = max_survival + max_climb + max_ground_break
    assert combined == pytest.approx(162.475)  # 160 + 1.98 + 0.495 (≈162.48 in the README prose)
    assert combined < CFG.completion_bonus, "combined shaping must stay below a valid completion"


# --- AC4: seam-monotonicity sizing bound (ground_break_weight < 0.9) -------------------------
def test_ground_break_weight_respects_seam_monotonicity_bound() -> None:
    """AC4 / sizing: the net-hold band slope is +0.18/m and the ground-break term subtracts
    0.2·w from it, so net-hold monotonicity above the seam holds iff ``ground_break_weight`` < 0.9.
    The shipped 0.5 leaves the net slope at +0.08/m (still strictly increasing)."""
    assert CFG.ground_break_weight < 0.9
    net_slope = 0.18 - 0.2 * CFG.ground_break_weight
    assert net_slope > 0.0
    assert net_slope == pytest.approx(0.08)


# --- AC4: every OTHER reward term is UNCHANGED (guard against a disguised global rescale) -----
def test_all_other_reward_terms_unchanged() -> None:
    """AC4: EXPLICITLY pin every reward term the ground-break fix must NOT touch. UC-43 adds only
    the two ``ground_break_*`` knobs; everything setting the relative geometry of the reward is
    byte-identical to UC-42, so the fix is a purely-additive sub-threshold term (not the
    VecNormalize-erasable global rescale)."""
    from drone_fly.train.config import TrainConfig

    assert CFG.time_penalty == pytest.approx(0.05)
    assert CFG.progress_weight == pytest.approx(1.0)
    assert CFG.gate_bonus == pytest.approx(10.0)
    assert CFG.completion_bonus == pytest.approx(200.0)
    assert CFG.collision_penalty == pytest.approx(100.0)
    assert CFG.obstacle_penalty == pytest.approx(50.0)
    assert CFG.airborne_bonus == pytest.approx(0.2)
    assert CFG.climb_weight == pytest.approx(2.0)
    assert CFG.climb_target_height == pytest.approx(1.0)
    assert CFG.climb_gamma == pytest.approx(0.99)
    # The two NEW knobs, pinned so a silent drift is caught.
    assert CFG.ground_break_weight == pytest.approx(0.5)
    assert CFG.ground_break_height == pytest.approx(0.05)
    # climb_gamma must still track the training discount for potential-based invariance to hold.
    assert CFG.climb_gamma == pytest.approx(TrainConfig().gamma)


# --- AC7: every desired outcome credits incremental progress densely, and round-trips net ≈0 --
def test_ac7_break_ground_step_pays_and_round_trip_nets_zero() -> None:
    """AC7: a small step toward BREAKING GROUND (rising inside the sub-threshold band) yields
    strictly-positive shaping; a round trip back to rest nets ≈0 (non-farmable)."""
    forward = _gb_term(0.0, 0.02)
    assert forward > 0.0
    round_trip = _gb_term(0.0, 0.02) + _gb_term(0.02, 0.0)
    assert round_trip == pytest.approx(0.0, abs=0.05)
    assert round_trip <= 0.0


def test_ac7_climb_step_pays_and_round_trip_nets_zero() -> None:
    """AC7: a small step toward CLIMBING to target (above the seam) yields strictly-positive climb
    shaping; a round trip nets ≈0 (the UC-39 potential telescopes)."""

    def _climb_only(h_prev: float, h_curr: float) -> float:
        # Isolate the climb POTENTIAL: ground-break zeroed and ``airborne=False`` (the airborne
        # graded bonus is a state function of the current height, NOT a potential difference, so it
        # would not telescope), differenced against the h=0 baseline.
        cfg = RewardConfig(
            ground_break_weight=0.0,
            enable_altitude_decoupling=False,
            altitude_hold_weight=0.0,
        )
        step = _reward(
            cfg=cfg, airborne=False, height_above_floor_prev=h_prev, height_above_floor_curr=h_curr
        )
        base = _reward(
            cfg=cfg, airborne=False, height_above_floor_prev=0.0, height_above_floor_curr=0.0
        )
        return step - base

    forward = _climb_only(0.3, 0.6)
    assert forward > 0.0
    round_trip = _climb_only(0.3, 0.6) + _climb_only(0.6, 0.3)
    assert round_trip == pytest.approx(0.0, abs=0.05)
    assert round_trip <= 0.0


def test_ac7_reduce_gate_distance_pays_and_round_trip_nets_zero() -> None:
    """AC7: a step that REDUCES distance to the next gate yields strictly-positive progress reward;
    an approach-then-retreat round trip nets exactly 0 (the progress term telescopes, so loitering
    toward-and-away from a gate cannot be farmed)."""

    def _progress(dist_prev: float, dist_curr: float) -> float:
        step = _reward(dist_to_target_prev=dist_prev, dist_to_target_curr=dist_curr)
        hold = _reward(dist_to_target_prev=dist_prev, dist_to_target_curr=dist_prev)
        return step - hold

    forward = _progress(1.0, 0.5)  # closing distance to the gate
    assert forward > 0.0
    round_trip = _progress(1.0, 0.5) + _progress(0.5, 1.0)
    assert round_trip == pytest.approx(0.0, abs=1e-12)
