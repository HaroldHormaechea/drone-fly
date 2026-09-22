"""UC-50 — reward altitude-holding forward flight: stop trading height for gate-progress.

Hermetic, relative/structural unit tests on :func:`drone_fly.env.reward.compute_reward` (no
pybullet, no GPU). Because the reward feeds VecNormalize, every assertion is an ordering, a
sign, a round-trip net, or an equality — **never** an absolute magnitude margin (the UC-42
lesson).

UC-50 ships ENABLED by default (``RewardConfig().enable_altitude_decoupling is True``); the
``OFF`` config below constructs the explicit feature-off variant used to pin the byte-identical
path — with the feature off the new ``target_height_above_floor_*`` inputs are ignored and the
function is byte-identical to UC-43 (AC-6). The enabled configuration under test mirrors the
shipped/recommended training values (``altitude_hold_weight = 2.0``, ``altitude_band = 0.6``).

Two coupled mechanisms are exercised (see the ``compute_reward`` docstring for the contract):

* **Progress hard-gate** — the POSITIVE progress reward is withheld while the drone is below
  ``max(ref_eff − altitude_band, ground_break_height)`` (``ref_eff = max(target_height,
  ground_break_height)``); a below-band retreat keeps its penalty (un-farmable).
* **Altitude-hold shaping** — a NON-NEGATIVE potential ``Φ_track(h, ref) = w · clamp(h −
  (ref_eff − band), 0, band)``, per-step ``F = γ·Φ(curr) − Φ(prev)``: climbing toward the
  reference pays, sinking pays back, telescoping ⇒ round trips net ≤ 0 (no loiter optimum).

Isolation strategy (mirrors the UC-43 ``_gb_term`` difference-from-baseline helper): term
contributions are recovered by differencing two ``compute_reward`` calls that share every input
EXCEPT the one dimension under test, so the untouched terms (``time_penalty``, ``airborne``,
UC-39 climb, UC-43 ground-break) cancel exactly.
"""

from __future__ import annotations

import pytest

from drone_fly.env.config import RewardConfig
from drone_fly.env.reward import compute_reward

# --- Configs under test ----------------------------------------------------------------------
OFF = RewardConfig(  # explicit feature-off variant (UC-50 now ships enabled by default)
    enable_altitude_decoupling=False, altitude_hold_weight=0.0
)
# Enabled with the recommended training values (documented in RewardConfig).
EN = RewardConfig(enable_altitude_decoupling=True, altitude_hold_weight=2.0, altitude_band=0.6)
# Enabled but with the altitude-hold WEIGHT zeroed: the progress hard-gate still runs, the
# potential-based track term is identically 0. Used to isolate the track term (EN − EN_NOTRACK)
# and the effective (gated) progress term without the track term interfering.
EN_NOTRACK = RewardConfig(
    enable_altitude_decoupling=True, altitude_hold_weight=0.0, altitude_band=0.6
)

GAMMA = OFF.climb_gamma  # 0.99 — must equal training γ (the potential invariance depends on it)
BAND = EN.altitude_band  # 0.6
WEIGHT = EN.altitude_hold_weight  # 2.0
GB = OFF.ground_break_height  # 0.05 — the ref_eff / gate-edge lower clamp


def _reward(cfg: RewardConfig = EN, **over: object) -> float:
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


def _track_term(h_prev: float, h_curr: float, ref_prev: float, ref_curr: float) -> float:
    """Isolate the altitude-hold potential term F = γ·Φ_track(curr) − Φ_track(prev) ALONE.

    EN and EN_NOTRACK share every input (no horizontal progress ⇒ the gate is inert and cancels;
    climb / ground-break / airborne / time depend only on the shared heights and cancel), so the
    difference is exactly the track term contributed by ``altitude_hold_weight = 2.0``."""
    common = dict(
        height_above_floor_prev=h_prev,
        height_above_floor_curr=h_curr,
        target_height_above_floor_prev=ref_prev,
        target_height_above_floor_curr=ref_curr,
    )
    return _reward(EN, **common) - _reward(EN_NOTRACK, **common)


def _phi_track(h: float, ref: float) -> float:
    """Reconstruct Φ_track(h; ref) via the anchor at the band's lower edge lo = ref_eff − band
    (where Φ_track = 0): F(lo→h) = γ·Φ(h) − Φ(lo) = γ·Φ(h) ⇒ Φ(h) = F(lo→h)/γ."""
    ref_eff = max(ref, GB)
    lo = ref_eff - BAND
    return _track_term(lo, h, ref, ref) / GAMMA


def _progress_eff(
    dist_prev: float,
    dist_curr: float,
    h: float,
    ref: float,
    cfg: RewardConfig = EN_NOTRACK,
) -> float:
    """Effective (post-gate) progress contribution for a step at constant height ``h``.

    Differences a closing step (``dist_curr``) against a held-distance step (``dist_curr =
    dist_prev``) with identical heights, so every height-driven term (track, climb, ground-break,
    airborne, time) cancels and only the (possibly-withheld) progress term survives. Run on
    EN_NOTRACK by default so the track term never leaks into the isolation."""
    common = dict(
        height_above_floor_prev=h,
        height_above_floor_curr=h,
        target_height_above_floor_prev=ref,
        target_height_above_floor_curr=ref,
    )
    closing = _reward(cfg, dist_to_target_prev=dist_prev, dist_to_target_curr=dist_curr, **common)
    held = _reward(cfg, dist_to_target_prev=dist_prev, dist_to_target_curr=dist_prev, **common)
    return closing - held


# =============================================================================================
# AC-1: progress is NOT earnable while sinking below the safe band.
# =============================================================================================
def test_ac1_sinking_below_band_while_closing_nets_non_positive() -> None:
    """AC-1: a step that closes horizontal distance to the target but descends BELOW the safe band
    (referenced to the gate's z) nets ≤ 0 from the combined progress + altitude terms.

    Reference gate at z = 1.0 ⇒ band lower edge = ref − band = 0.4. Descend from inside the band
    (h_prev = 0.5) to below it (h_curr = 0.3) while closing distance (2.0 → 1.5). The progress
    reward is withheld (h_curr < 0.4) ⇒ effective progress = 0, and the track term is strictly
    negative (Φ drops toward 0). Their sum is ≤ 0, and here strictly negative."""
    ref = 1.0
    progress = _progress_eff(dist_prev=2.0, dist_curr=1.5, h=0.3, ref=ref)  # below-band end state
    track = _track_term(h_prev=0.5, h_curr=0.3, ref_prev=ref, ref_curr=ref)
    assert progress == pytest.approx(0.0, abs=1e-12)  # positive progress fully withheld
    assert track < 0.0  # sinking pays back the shaping credit
    assert progress + track <= 0.0
    assert progress + track < 0.0  # genuinely negative, not the degenerate 0 = 0 case


def test_ac1_companion_sinking_total_below_level_total() -> None:
    """AC-1 companion: the FULL reward of a sinking-below-band closing step is strictly less than a
    level-in-band closing step over identical horizontal progress."""
    ref = 1.0
    sinking = _reward(
        EN,
        dist_to_target_prev=2.0,
        dist_to_target_curr=1.5,
        airborne=True,
        height_above_floor_prev=0.5,
        height_above_floor_curr=0.3,  # sinks below the 0.4 band edge
        target_height_above_floor_prev=ref,
        target_height_above_floor_curr=ref,
    )
    level = _reward(
        EN,
        dist_to_target_prev=2.0,
        dist_to_target_curr=1.5,
        airborne=True,
        height_above_floor_prev=0.7,
        height_above_floor_curr=0.7,  # holds inside the band
        target_height_above_floor_prev=ref,
        target_height_above_floor_curr=ref,
    )
    assert sinking < level


# =============================================================================================
# AC-2: level beats descending for equal horizontal progress.
# =============================================================================================
def test_ac2_level_in_band_strictly_beats_descend_below_ref() -> None:
    """AC-2: for two steps with identical horizontal closing distance, holding altitude within the
    safe band nets STRICTLY higher reward than losing altitude below the reference."""
    ref = 1.0
    common = dict(
        dist_to_target_prev=2.0,
        dist_to_target_curr=1.4,  # identical horizontal closing for both
        airborne=True,
        target_height_above_floor_prev=ref,
        target_height_above_floor_curr=ref,
    )
    level = _reward(EN, height_above_floor_prev=0.8, height_above_floor_curr=0.8, **common)
    descend = _reward(EN, height_above_floor_prev=0.8, height_above_floor_curr=0.25, **common)
    assert level > descend


# =============================================================================================
# AC-3: the safe reference tracks the current target gate's z.
# =============================================================================================
def test_ac3_reference_shifts_with_target_height() -> None:
    """AC-3: the band / reference is derived from the current target gate's z, not a global
    constant. At a FIXED drone height h = 0.5, a LOW gate (ref = 0.8, edge = 0.2) leaves progress
    ungated and pays the altitude credit, while a HIGH gate (ref = 1.3, edge = 0.7) gates progress
    off and pays no credit — the same state yields different behaviour purely from the gate z."""
    h = 0.5
    low_ref, high_ref = 0.8, 1.3  # low: edge 0.2 (h above) · high: edge 0.7 (h below)

    prog_low = _progress_eff(dist_prev=2.0, dist_curr=1.5, h=h, ref=low_ref)
    prog_high = _progress_eff(dist_prev=2.0, dist_curr=1.5, h=h, ref=high_ref)
    assert prog_low > 0.0  # above the low gate's band → progress paid
    assert prog_high == pytest.approx(0.0, abs=1e-12)  # below the high gate's band → withheld

    phi_low = _phi_track(h, low_ref)
    phi_high = _phi_track(h, high_ref)
    assert phi_low > 0.0  # inside the low gate's band → positive altitude credit
    assert phi_high == pytest.approx(0.0, abs=1e-12)  # below the high gate's band → zero credit
    assert phi_low != pytest.approx(phi_high)  # the reference genuinely shifted the shaping


# =============================================================================================
# AC-4: non-farmability — non-negative Φ, safe leak, telescoping round trips, bounded total.
# =============================================================================================
def test_ac4a_phi_track_non_negative_across_the_band() -> None:
    """AC-4(a): the corrected anchor makes Φ_track ≥ 0 everywhere (the negative-shortfall form that
    was caught and fixed would go negative and pay a farmable loiter income). Swept from below the
    band edge, through the band, to above the reference."""
    ref = 1.0  # band edge = 0.4, reference reached at 1.0
    for h in (0.0, 0.2, 0.4, 0.5, 0.7, 0.9, 1.0, 1.3):
        assert _phi_track(h, ref) >= -1e-9, f"Φ_track went negative at h={h}"


def test_ac4b_stationary_in_band_step_nets_non_positive() -> None:
    """AC-4(b): a stationary in-band step leaks only the discounted-potential term (γ−1)·Φ ≤ 0 —
    holding position is never positively rewarded by the track term (no loiter income)."""
    ref = 1.0
    for h in (0.5, 0.7, 0.9):
        leak = _track_term(h_prev=h, h_curr=h, ref_prev=ref, ref_curr=ref)
        assert leak <= 1e-12, f"stationary track leak positive at h={h}"


def test_ac4c_bob_round_trip_nets_non_positive() -> None:
    """AC-4(c): a climb-then-descend bob round trip nets ≤ 0 and ≈ 0 (telescoping potential) — the
    guard against the negative-potential loiter bug that was caught and fixed in implementation."""
    ref = 1.0
    up = _track_term(h_prev=0.5, h_curr=0.9, ref_prev=ref, ref_curr=ref)
    down = _track_term(h_prev=0.9, h_curr=0.5, ref_prev=ref, ref_curr=ref)
    assert up > 0.0  # climbing toward the reference pays
    assert down < 0.0  # sinking pays it back
    round_trip = up + down
    assert round_trip <= 0.0  # never a net gain from a round trip (no loiter optimum)
    assert round_trip == pytest.approx(0.0, abs=0.05)  # telescopes to ≈ 0 (only the tiny leak)


def test_ac4_per_episode_track_magnitude_far_below_completion_bonus() -> None:
    """AC-4: the per-episode altitude-hold total is bounded by ≈ altitude_hold_weight · band
    (the potential saturates at ``altitude_band``), which is ≪ ``completion_bonus`` — so no amount
    of altitude shaping can rival, let alone out-score, a course completion."""
    per_episode_bound = WEIGHT * BAND  # 2.0 · 0.6 = 1.2
    # The reconstructed Φ never exceeds the analytic saturation bound anywhere in/above the band.
    ref = 1.0
    for h in (0.4, 0.7, 1.0, 1.5):
        assert _phi_track(h, ref) <= per_episode_bound + 1e-9
    assert per_episode_bound < 0.05 * OFF.completion_bonus  # 1.2 ≪ 200 (≪, not merely <)


# =============================================================================================
# AC-5: takeoff still bootstraps (UC-43 ground-break / UC-39 climb untouched; no chicken-and-egg).
# =============================================================================================
def test_ac5_first_cm_of_lift_is_net_positive_and_progress_independent() -> None:
    """AC-5: a first-cm-of-lift step (0 → 0.02 m, below the ground-break saturation 0.05 and far
    below any gate band) is still net-positive from the UC-43 ground-break term, and its reward is
    independent of horizontal progress (progress is gated off that low), so takeoff bootstraps
    without a new chicken-and-egg."""
    ref = 1.0
    lift = _reward(
        EN,
        airborne=False,  # below floor_epsilon ⇒ the airborne bonus is not yet active
        height_above_floor_prev=0.0,
        height_above_floor_curr=0.02,
        target_height_above_floor_prev=ref,
        target_height_above_floor_curr=ref,
    )
    assert lift > 0.0  # ground-break gradient beats the time penalty on the very first lift

    # Progress-independent: closing horizontal distance during that sub-band lift adds nothing,
    # because the progress term is withheld until the drone reaches the band edge.
    gated = _progress_eff(dist_prev=2.0, dist_curr=1.5, h=0.02, ref=ref)
    assert gated == pytest.approx(0.0, abs=1e-12)


def test_ac5_ground_break_and_climb_terms_untouched_when_below_band() -> None:
    """AC-5: with the flag ON but the drone in the sub-band bootstrap zone, the height-driven reward
    equals the flag-OFF reward — the UC-43 ground-break and UC-39 climb terms are unchanged (the
    track term is 0 below the edge and the gate only ever withholds a POSITIVE progress reward,
    which is 0 here)."""
    ref = 1.0
    for h_curr in (0.02, 0.04, 0.05):
        common = dict(
            height_above_floor_prev=0.0,
            height_above_floor_curr=h_curr,
            target_height_above_floor_prev=ref,
            target_height_above_floor_curr=ref,
        )
        assert _reward(EN, airborne=False, **common) == pytest.approx(
            _reward(OFF, airborne=False, **common)
        )


def test_ac5_progress_ungates_once_at_or_above_the_band_edge() -> None:
    """AC-5: the gate is one-sided and opens exactly at the band edge — at/above ``max(ref − band,
    ground_break_height)`` the full progress reward is paid (no residual suppression)."""
    ref = 1.0  # edge = max(1.0 − 0.6, 0.05) = 0.4
    paid = _progress_eff(dist_prev=2.0, dist_curr=1.5, h=0.5, ref=ref)  # 0.5 ≥ 0.4 → ungated
    assert paid == pytest.approx(OFF.progress_weight * (2.0 - 1.5))


# =============================================================================================
# AC-6: byte-identity regression — flag OFF ignores the new inputs entirely.
# =============================================================================================
# Representative pre-UC-50 transitions spanning progress / gate / collision / completion / airborne
# / climb / ground-break, each a full kwargs dict for the pre-change signature.
_REPRESENTATIVE_CALLS = (
    dict(
        dist_to_target_prev=1.0,
        dist_to_target_curr=0.5,
        event=None,
        collided=False,
        completed=False,
    ),
    dict(
        dist_to_target_prev=1.0,
        dist_to_target_curr=0.5,
        event="gate",
        collided=False,
        completed=False,
        num_gates=3,
    ),
    dict(
        dist_to_target_prev=0.5,
        dist_to_target_curr=0.0,
        event="finish",
        collided=False,
        completed=True,
    ),
    dict(
        dist_to_target_prev=1.0, dist_to_target_curr=1.2, event=None, collided=True, completed=False
    ),
    dict(
        dist_to_target_prev=1.0,
        dist_to_target_curr=0.8,
        event=None,
        collided=False,
        completed=False,
        obstacle_contact=True,
    ),
    dict(
        dist_to_target_prev=1.0,
        dist_to_target_curr=0.9,
        event=None,
        collided=False,
        completed=False,
        airborne=True,
        height_above_floor_prev=0.3,
        height_above_floor_curr=0.6,
    ),
    dict(
        dist_to_target_prev=1.0,
        dist_to_target_curr=0.9,
        event=None,
        collided=False,
        completed=False,
        height_above_floor_prev=0.0,
        height_above_floor_curr=0.03,
    ),
)


@pytest.mark.parametrize("call", _REPRESENTATIVE_CALLS)
def test_ac6_flag_off_is_byte_identical_regardless_of_target_height(call: dict) -> None:
    """AC-6: with ``enable_altitude_decoupling`` explicitly False, the new
    ``target_height_above_floor_*`` inputs are ignored — the result is identical whether they are
    omitted (pre-change callers) or set to arbitrary non-zero values. Pins the additive,
    feature-off, byte-compatible contract mirrored from every prior reward UC (UC-50 itself now
    ships enabled by default)."""
    baseline = compute_reward(cfg=OFF, **call)
    with_targets = compute_reward(
        cfg=OFF,
        target_height_above_floor_prev=1.3,
        target_height_above_floor_curr=0.9,
        **call,
    )
    assert with_targets == baseline
    # Equality is exact float identity, not approximate — this is a byte-compatibility guarantee.
    assert repr(with_targets) == repr(baseline)


# =============================================================================================
# Formula consistency: ref_eff = max(target_height, ground_break_height) used for BOTH the gate
# edge and the track term (pitfall #5 — floor-band gates must not degenerate the band).
# =============================================================================================
def test_ref_eff_lower_clamp_is_consistent_for_floor_band_gates() -> None:
    """Pitfall #5: any target height at or below ``ground_break_height`` clamps to the same
    ``ref_eff = ground_break_height``, so a floor-band gate never demands less altitude than the
    UC-43 bootstrap band. All such gates must produce identical gate-edge AND track behaviour."""
    baseline_ref = GB  # exactly ground_break_height (0.05)
    for tiny_ref in (0.0, 0.02, GB):
        # Track term identical across all sub-clamp target heights (same ref_eff ⇒ same Φ_track).
        assert _track_term(0.2, 0.5, tiny_ref, tiny_ref) == pytest.approx(
            _track_term(0.2, 0.5, baseline_ref, baseline_ref)
        )
        # Progress-gate behaviour identical too (same ref_eff ⇒ same edge = ground_break_height).
        assert _progress_eff(2.0, 1.5, h=0.03, ref=tiny_ref) == pytest.approx(
            _progress_eff(2.0, 1.5, h=0.03, ref=baseline_ref)
        )
