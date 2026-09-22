"""UC-42 — the SOLE committed acceptance gate: sustained climb decisively out-rewards hover-rest.

Why this test is structural, not absolute (read first)
------------------------------------------------------
The env trains with reward ``VecNormalize`` ON (``racing_env.py``), so any GLOBAL rescale of
the reward is a no-op — the normaliser divides it straight back out. An acceptance test that
asserted an absolute reward *margin* would therefore be meaningless (this is the UC-38 trap that
already failed). So every assertion here gates on a RELATIVE / STRUCTURAL property of the math:

  (a) the durable net-hold-per-airborne-step at the target rose from the old +0.03 to ~+0.13/step
      — asserted as ``>= ~0.12`` AND strictly greater than the OLD +0.03 (a real restructuring,
      not a rescale);
  (b) the net-hold-per-step is MONOTONICALLY INCREASING in altitude up to the target and FLAT
      above it (no ceiling-seeking), and ~0 (no airborne credit over sitting) at the floor — i.e.
      the previously-perverse "holding higher is slightly worse" gradient is flipped into a
      strictly increasing climb-to-target pull;
  (c) an EXPLICIT assertion that ``time_penalty``, ``progress_weight``, ``collision_penalty`` and
      ``gate_bonus`` are UNCHANGED — proving the fix is a relative restructuring of the airborne
      term (plus a forced completion-bound bump), NOT a global rescale that VecNormalize erases.

Any absolute number below is illustrative of the shipped constants only; the load-bearing
assertions are the relative/structural ones. The numpy ``simple``-adapter smoke-train is
explicitly NOT evidence of takeoff (UC-40 over-promised on it) and real-pybullet behaviour is the
user's retrain, not a gate.
"""

from __future__ import annotations

import pytest

from drone_fly.env.config import EnvConfig, EpisodeConfig, RewardConfig
from drone_fly.env.reward import compute_reward

# UC-50 ships enabled-by-default; these UC-42 survival/altitude-grading tests pin the pre-UC-50
# behavior in isolation, so use an explicit feature-OFF config.
CFG = RewardConfig(enable_altitude_decoupling=False, altitude_hold_weight=0.0)

# The pre-UC-42 durable net-hold-per-airborne-step AT THE TARGET, with the OLD flat bonus:
#   airborne_bonus(0.10) - time_penalty(0.05) - standing_tax(0.02) = +0.03/step.
# This is exactly the UC-38 margin that already failed to produce takeoff — the fix must beat it.
OLD_NET_HOLD_AT_TARGET = 0.03


def _net_hold(h: float) -> float:
    """Durable per-step reward of HOLDING altitude ``h`` while airborne (``h_prev == h_curr == h``).

    This is the "net-hold-per-airborne-step": -time_penalty + the (now altitude-graded) airborne
    payout + the potential-based climb term's standing contribution ((gamma-1)*w*min(h,target) for
    a hold). It is the quantity a policy that has climbed to ``h`` and holds there actually banks
    each step, so it is the right thing to compare against sitting on the floor (-time_penalty).
    """
    return compute_reward(
        dist_to_target_prev=0.0,
        dist_to_target_curr=0.0,
        event=None,
        collided=False,
        completed=False,
        cfg=CFG,
        airborne=True,
        height_above_floor_prev=h,
        height_above_floor_curr=h,
    )


SIT_ON_FLOOR = -CFG.time_penalty  # what sitting still on the ground banks per step


# --- (a) the durable net-hold differential rose from +0.03 to ~+0.13/step -------------------
def test_uc42_net_hold_at_target_rose_from_old_margin() -> None:
    """AC2 (a): holding at the climb target now nets ~+0.13/step, up from the old +0.03 that failed.

    Asserted RELATIVELY: ``>= ~0.12`` (a healthy, normalization-surviving hold reward) AND strictly
    greater than the OLD +0.03 flat-bonus margin. The ~4.3x jump is the structural fix — not a
    global rescale (which VecNormalize would erase), because the penalty/progress/gate knobs are
    unchanged (proven in test_uc42_unchanged_knobs_prove_relative_restructuring)."""
    net_hold_target = _net_hold(CFG.climb_target_height)
    # Illustrative absolute value of the shipped constants: -0.05 + 0.20 - 0.02 - 0.005 = +0.125
    # (UC-43 added the ground-break constant above-threshold leak -(1-gamma)*w = -0.005/step).
    assert net_hold_target == pytest.approx(0.125, abs=1e-9)
    # Load-bearing RELATIVE assertions:
    assert net_hold_target >= 0.12, "net-hold at target must clear ~+0.12/step"
    assert net_hold_target > OLD_NET_HOLD_AT_TARGET, (
        "the fix must strictly beat the old +0.03 margin that already failed (UC-38 trap)"
    )
    # And holding at the target decisively out-rewards sitting on the floor.
    assert net_hold_target > SIT_ON_FLOOR


def test_uc42_hold_differential_over_floor_is_large_and_positive() -> None:
    """AC2: the differential (hold-at-target minus sit-on-floor) is large and positive — climbing
    and holding altitude is strictly, and substantially, better than resting on the ground."""
    differential = _net_hold(CFG.climb_target_height) - SIT_ON_FLOOR
    assert differential > 0
    assert differential == pytest.approx(0.175, abs=1e-9)  # 0.125 - (-0.05) (UC-43: -0.005 leak)


# --- (b) net-hold is monotone increasing to target, flat above, ~0 at the floor -------------
def test_uc42_net_hold_is_monotonically_increasing_up_to_target() -> None:
    """AC2/AC3 (b): the net-hold-per-step is STRICTLY INCREASING in altitude from the floor up to
    the target. This flips UC-37's perverse gradient (with a flat bonus, holding higher was
    slightly WORSE, net of the time penalty and the discounted climb-potential leak) into a
    monotone climb-to-target pull — the drone is rewarded MORE the higher it holds, up to target."""
    target = CFG.climb_target_height
    heights = [i / 20.0 * target for i in range(21)]  # 0, 0.05*t, ..., target (fine grid)
    holds = [_net_hold(h) for h in heights]
    # strict=False is intentional: pairwise (consecutive) iteration, so the tail is 1 shorter.
    for lower, higher in zip(holds, holds[1:], strict=False):
        assert higher > lower, "net-hold must strictly increase with altitude up to the target"
    # Endpoints: at the floor it equals sitting (no airborne credit); at the target it is the max.
    assert holds[0] == pytest.approx(SIT_ON_FLOOR)
    assert holds[-1] == pytest.approx(0.125, abs=1e-9)  # UC-43: 0.13 - the 0.005 ground-break leak


def test_uc42_net_hold_is_flat_above_target_no_ceiling_seeking() -> None:
    """AC3 (b): above the climb target the net-hold SATURATES (both the airborne payout and the
    climb potential cap at the target), so there is no incentive to climb into the ceiling — the
    pull is "climb to the target and hold", not "climb forever"."""
    target = CFG.climb_target_height
    at_target = _net_hold(target)
    for h in (target * 1.5, target * 2.0, 2.4):  # progressively deeper into ceiling territory
        assert _net_hold(h) == pytest.approx(at_target, abs=1e-9), "net-hold flat above target"


def test_uc42_net_hold_zero_floor_credit_and_positive_break_even() -> None:
    """AC3 (b): at the floor an airborne step earns ~no credit over sitting (the graded payout -> 0
    as h -> 0). Above the airborne threshold the differential over sitting is ``0.18*h - 0.005``
    (UC-43's ground-break term adds a constant -0.005/step leak above threshold) — still strictly
    positive for any h above ~0.028 m and monotone, so there is no "hover just off the floor" local
    optimum: climbing always pays more than staying lower. Separately, the ABSOLUTE per-step reward
    crosses zero at the break-even altitude h ~= 0.055 / 0.18 ~= 0.306 m (raised from UC-42's ~0.278
    by the -0.005 leak): below it the step is still net-negative in absolute terms; above it the
    drone banks positive reward every step it holds."""
    # ~0 credit over sitting at the floor; strictly-positive, altitude-proportional pull above it.
    assert _net_hold(0.0) == pytest.approx(SIT_ON_FLOOR)
    # h=0.1 is above the 0.05 threshold, so the differential is 0.18*h minus the 0.005 leak.
    assert _net_hold(0.1) - SIT_ON_FLOOR == pytest.approx(0.18 * 0.1 - 0.005, abs=1e-9)
    assert _net_hold(0.1) > SIT_ON_FLOOR  # any positive altitude beats sitting (monotone pull)
    # Absolute break-even of the per-step reward at h ~= 0.306 m (0.055 / 0.18, UC-43-shifted).
    assert _net_hold(0.2) < 0.0  # below break-even: absolute per-step reward still negative
    assert _net_hold(0.4) > 0.0  # above break-even: absolute per-step reward positive
    assert _net_hold(0.055 / 0.18) == pytest.approx(0.0, abs=2e-3)


# --- (c) the unchanged knobs prove a RELATIVE restructuring, not a global rescale ------------
def test_uc42_unchanged_knobs_prove_relative_restructuring() -> None:
    """AC2/AC3 (c): EXPLICITLY assert the penalty/progress/gate knobs are UNCHANGED. This is the
    guard against the UC-38/VecNormalize trap: because reward normalization is ON, a global rescale
    would be a no-op — so the takeoff fix MUST be a relative restructuring. The only reward-config
    changes are ``airborne_bonus`` (0.1 -> 0.2, the graded structural fix) and ``completion_bonus``
    (100 -> 200, forced bound-preservation); everything setting RELATIVE geometry is untouched."""
    assert CFG.time_penalty == pytest.approx(0.05), "time_penalty unchanged"
    assert CFG.progress_weight == pytest.approx(1.0), "progress_weight unchanged"
    assert CFG.collision_penalty == pytest.approx(100.0), "collision_penalty unchanged"
    assert CFG.gate_bonus == pytest.approx(10.0), "gate_bonus unchanged"
    # The climb-shaping geometry is likewise untouched (only the airborne term was restructured).
    assert CFG.climb_weight == pytest.approx(2.0), "climb_weight unchanged"
    assert CFG.climb_target_height == pytest.approx(1.0), "climb_target_height unchanged"
    assert CFG.climb_gamma == pytest.approx(0.99), "climb_gamma unchanged"
    # The two DELIBERATE changes, pinned so a silent drift is caught.
    assert CFG.airborne_bonus == pytest.approx(0.2), "airborne_bonus raised 0.1 -> 0.2 (UC-42)"
    assert CFG.completion_bonus == pytest.approx(200.0), "completion_bonus raised 100->200 (UC-42)"
    # UC-43 added the two ground-break knobs (the ONLY new reward config; everything else above is
    # byte-identical to UC-42), pinned here so a silent drift is caught.
    assert CFG.ground_break_weight == pytest.approx(0.5), "ground_break_weight added (UC-43)"
    assert CFG.ground_break_height == pytest.approx(0.05), "ground_break_height added (UC-43)"


# --- invariant re-proofs (UC-37/38/39 must all still hold after the restructuring) ----------
def _climb_contribution(h_prev: float, h_curr: float) -> float:
    """The climb term ALONE for a height transition (airborne payout differenced out via h=0)."""
    base = compute_reward(
        dist_to_target_prev=0.0,
        dist_to_target_curr=0.0,
        event=None,
        collided=False,
        completed=False,
        cfg=CFG,
        height_above_floor_prev=0.0,
        height_above_floor_curr=0.0,
    )
    with_heights = compute_reward(
        dist_to_target_prev=0.0,
        dist_to_target_curr=0.0,
        event=None,
        collided=False,
        completed=False,
        cfg=CFG,
        height_above_floor_prev=h_prev,
        height_above_floor_curr=h_curr,
    )
    return with_heights - base


def test_uc42_climb_term_still_telescopes_non_farmable() -> None:
    """Invariant (UC-39 AC2a): the potential-based climb term still telescopes — a
    climb-then-descend round trip nets <= 0 (the gamma<1 residual), so bobbing up and down cannot
    be farmed. Unchanged by UC-42 (the airborne term was restructured; climb weights untouched)."""
    up = _climb_contribution(0.0, 0.5)
    down = _climb_contribution(0.5, 0.0)
    round_trip = up + down
    assert round_trip == pytest.approx(0.0, abs=0.05)
    assert round_trip <= 0.0, "the gamma<1 residual keeps a round trip non-positive (non-farmable)"


def test_uc42_graded_survival_is_zero_at_rest_on_the_floor() -> None:
    """Invariant (UC-37 AC5, tightened by UC-42): the survival reward is ~0 at rest on the floor —
    and UNLIKE UC-37's flat bonus, an airborne step AT the floor also pays ~0 now (graded). So the
    floor is non-farmable: the only way to earn survival reward is to actually climb."""
    on_floor_grounded = compute_reward(
        dist_to_target_prev=0.0,
        dist_to_target_curr=0.0,
        event=None,
        collided=False,
        completed=False,
        cfg=CFG,
        airborne=False,
        height_above_floor_curr=0.0,
    )
    airborne_at_floor = compute_reward(
        dist_to_target_prev=0.0,
        dist_to_target_curr=0.0,
        event=None,
        collided=False,
        completed=False,
        cfg=CFG,
        airborne=True,
        height_above_floor_curr=0.0,
    )
    assert on_floor_grounded == pytest.approx(-CFG.time_penalty)
    assert airborne_at_floor == pytest.approx(-CFG.time_penalty)


def test_uc42_per_episode_loiter_still_below_completion_bonus() -> None:
    """Invariant (UC-37 AC6b / UC-39 AC5): after the airborne bump the max NON-completion reward
    over the default 800-step budget — max survival (``airborne_bonus * budget`` = 160, since the
    payout saturates at ``airborne_bonus``) plus the whole climb bound (1.98) — is STILL strictly
    below ``completion_bonus``. ``completion_bonus`` was raised 100 -> 200 in lockstep precisely to
    preserve this ordering; the headroom (~1.23) is kept tight because reward VecNormalize is ON."""
    ep = EpisodeConfig()
    course = EnvConfig().course
    budget = ep.max_steps + ep.steps_per_gate * (course.num_gates - 1)
    assert budget == 800
    max_survival = CFG.airborne_bonus * budget
    max_climb = CFG.climb_gamma * CFG.climb_weight * CFG.climb_target_height
    assert max_survival == pytest.approx(160.0)
    assert max_climb == pytest.approx(1.98)
    combined = max_survival + max_climb
    assert combined == pytest.approx(161.98)
    assert combined < CFG.completion_bonus, "loiter+climb must stay below a valid completion"


def test_uc42_climb_gamma_still_tracks_training_gamma() -> None:
    """Invariant (UC-39 Note 4): the potential-based climb shaping stays policy-invariant only if
    ``climb_gamma`` equals the training discount gamma. UC-42 touched neither, so it holds."""
    from drone_fly.train.config import TrainConfig

    assert CFG.climb_gamma == pytest.approx(TrainConfig().gamma)


def test_uc42_climb_bound_below_completion_and_normalised_gate() -> None:
    """Invariant (UC-39 AC2b): the per-episode climb bound (1.98) stays much less than
    ``completion_bonus`` and <= a normalised 3-gate ``gate_bonus`` (10/3 ~= 3.33) — the completion
    bump did not break it."""
    climb_bound = CFG.climb_gamma * CFG.climb_weight * CFG.climb_target_height
    assert climb_bound == pytest.approx(1.98)
    assert climb_bound < CFG.completion_bonus
    assert climb_bound <= CFG.gate_bonus / 3
