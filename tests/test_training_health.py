"""UC-23 AC1/AC2/AC3 — the pure training-health assessment engine.

Every rule is exercised with a triggering AND a non-triggering synthetic history, the
under-capacity K0 signature is checked positive and (twice) negative, the warm-up guard is
proven to short-circuit before any metric series is inspected, and the verdict shape +
status precedence (normal < warning < critical) are locked in.

The engine is pure (no I/O, no global state), so every case is a plain data-in / verdict-out
assertion — no fixtures, no mocking.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from drone_fly.train.health import (
    DEFAULT_CAPACITY_FLOOR,
    DEFAULT_THRESHOLDS,
    CapacityFacts,
    HealthReason,
    HealthThresholds,
    HealthVerdict,
    TrainingSignals,
    assess_training_health,
    worst_status,
)


def _rule_ids(verdict: HealthVerdict) -> set[str]:
    return {r.rule_id for r in verdict.reasons}


# --- AC1: verdict shape + purity ----------------------------------------------------------


def test_verdict_and_reason_shape_and_frozen() -> None:
    reason = HealthReason("some_rule", "human text", "warning")
    verdict = HealthVerdict("warning", "a message", [reason])
    assert verdict.status == "warning"
    assert verdict.message == "a message"
    assert verdict.reasons == [reason]
    assert (reason.rule_id, reason.text, reason.status) == ("some_rule", "human text", "warning")
    # Frozen dataclasses — the verdict object UC-22 renders must be immutable.
    with pytest.raises(FrozenInstanceError):
        verdict.status = "critical"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        reason.text = "changed"  # type: ignore[misc]


def test_default_verdict_has_empty_reasons() -> None:
    assert HealthVerdict("normal", "ok").reasons == []


def test_both_none_returns_trivial_normal() -> None:
    verdict = assess_training_health()
    assert verdict.status == "normal"
    assert verdict.reasons == []


def test_engine_is_pure_same_input_same_verdict() -> None:
    signals = TrainingSignals(
        n_updates=10,
        ep_rew_mean=(5.0, 5.0, 5.0),
        success_rate=(0.3, 0.3, 0.3),
        entropy_std=(1.0, 0.99, 0.99),
    )
    a = assess_training_health(signals=signals)
    b = assess_training_health(signals=signals)
    assert (a.status, a.message, _rule_ids(a)) == (b.status, b.message, _rule_ids(b))


# --- AC2: under-capacity K0 signature -----------------------------------------------------


def _under_capacity_signature_signals(**overrides) -> TrainingSignals:
    """Healthy/rising EV + flat-high entropy + success pinned at ~0 (the K0 signature)."""
    base = dict(
        n_updates=10,
        ep_rew_mean=(1.0, 1.0, 1.0),
        success_rate=(0.0, 0.0, 0.0),
        explained_variance=(0.80, 0.82, 0.81),
        entropy_std=(1.00, 1.00, 1.01),
        approx_kl=(0.01, 0.01),
        value_loss=(0.5, 0.5),
    )
    base.update(overrides)
    return TrainingSignals(**base)


def test_under_capacity_signature_fires() -> None:
    verdict = assess_training_health(signals=_under_capacity_signature_signals())
    assert "under_capacity_signature" in _rule_ids(verdict)
    assert verdict.status in ("warning", "critical")
    reason = next(r for r in verdict.reasons if r.rule_id == "under_capacity_signature")
    assert "slice" in reason.text.lower()


def test_under_capacity_signature_escalates_to_critical_after_warmup() -> None:
    # n_updates >= min_updates * 4 -> the persistent signature escalates to critical.
    long_run = _under_capacity_signature_signals(n_updates=DEFAULT_THRESHOLDS.min_updates * 4)
    reason = next(
        r
        for r in assess_training_health(signals=long_run).reasons
        if r.rule_id == "under_capacity_signature"
    )
    assert reason.status == "critical"


def test_under_capacity_signature_suppressed_when_entropy_falling() -> None:
    signals = _under_capacity_signature_signals(entropy_std=(1.0, 0.7, 0.5))
    assert "under_capacity_signature" not in _rule_ids(assess_training_health(signals=signals))


def test_under_capacity_signature_suppressed_when_success_climbing() -> None:
    signals = _under_capacity_signature_signals(
        success_rate=(0.0, 0.2, 0.5), ep_rew_mean=(1.0, 2.0, 3.0)
    )
    assert "under_capacity_signature" not in _rule_ids(assess_training_health(signals=signals))


# --- UC-40 (AC2): EV criterion is ABSOLUTE, not "rising" ----------------------------------
# The K0 signature must gate "critic is healthy" on the ABSOLUTE explained_variance clearing
# ev_healthy — never on a mere upward trend. The old OR-branch (`_rel_change(EV) > 0`) fired the
# rule at an absolute EV≈0.03 that happened to be rising, mislabeling a near-zero critic as
# "high/rising" and producing the false K0 "undersized slice" verdict UC-40 exists to fix.


def test_under_capacity_signature_not_fired_when_ev_rising_but_near_zero() -> None:
    """AC2 regression: EV rising (0.01→0.02→0.03) but mean ≈0.02 ≪ ev_healthy must NOT fire.

    This is the exact observed trace (abs EV ~0.03, trending up) that the pre-UC-40 trend-branch
    mis-classified as a healthy critic. With the trend-branch dropped, a near-zero critic — even
    a rising one — is no longer the K0 undersized-slice signature.
    """
    near_zero_rising = (0.01, 0.02, 0.03)
    # Guard the premise: the series really is below the healthy threshold on absolute terms.
    assert sum(near_zero_rising) / len(near_zero_rising) < DEFAULT_THRESHOLDS.ev_healthy
    signals = _under_capacity_signature_signals(explained_variance=near_zero_rising)
    assert "under_capacity_signature" not in _rule_ids(assess_training_health(signals=signals))


def test_under_capacity_signature_fires_when_ev_absolutely_healthy() -> None:
    """AC2: a genuinely healthy critic (abs EV at/above ev_healthy) with the flat-high-std /
    zero-success actor still fires — the fix must not suppress the genuine K0 signature."""
    signals = _under_capacity_signature_signals(explained_variance=(0.55, 0.60, 0.58))
    verdict = assess_training_health(signals=signals)
    assert "under_capacity_signature" in _rule_ids(verdict)
    reason = next(r for r in verdict.reasons if r.rule_id == "under_capacity_signature")
    assert "slice" in reason.text.lower()


def test_under_capacity_signature_ev_boundary_at_ev_healthy() -> None:
    """AC2: the decision is a clean threshold at ``ev_healthy`` — mean EV exactly at the
    threshold fires; a hair below does not (all other K0 conditions held constant)."""
    ev = DEFAULT_THRESHOLDS.ev_healthy
    at = _under_capacity_signature_signals(explained_variance=(ev, ev, ev))
    below = _under_capacity_signature_signals(explained_variance=(ev - 0.01, ev - 0.01, ev - 0.01))
    assert "under_capacity_signature" in _rule_ids(assess_training_health(signals=at))
    assert "under_capacity_signature" not in _rule_ids(assess_training_health(signals=below))


def test_under_capacity_signature_message_states_absolute_not_high_rising() -> None:
    """AC2: the corrected reason text no longer claims the critic is 'high/rising' — it states
    absolute health — while still naming the (possible) undersized 'slice'."""
    verdict = assess_training_health(signals=_under_capacity_signature_signals())
    text = next(
        r.text for r in verdict.reasons if r.rule_id == "under_capacity_signature"
    ).lower()
    assert "slice" in text
    assert "high/rising" not in text
    assert "rising" not in text
    assert "absolute" in text


# --- AC3: reward stalled ------------------------------------------------------------------


def test_reward_stalled_fires_when_flat() -> None:
    signals = TrainingSignals(
        n_updates=10,
        ep_rew_mean=(5.0, 5.0, 5.0, 4.9),
        success_rate=(0.3, 0.3),
        entropy_std=(1.0, 0.99),
    )
    assert "reward_stalled" in _rule_ids(assess_training_health(signals=signals))


def test_reward_stalled_not_fired_when_improving() -> None:
    signals = TrainingSignals(
        n_updates=10,
        ep_rew_mean=(1.0, 2.0, 3.0, 4.0),
        success_rate=(0.3, 0.4),
        entropy_std=(1.0, 0.99),
    )
    assert "reward_stalled" not in _rule_ids(assess_training_health(signals=signals))


def test_reward_stalled_needs_two_observations() -> None:
    signals = TrainingSignals(n_updates=10, ep_rew_mean=(5.0,))
    assert "reward_stalled" not in _rule_ids(assess_training_health(signals=signals))


# --- AC3: approx_kl runaway ---------------------------------------------------------------


def test_approx_kl_runaway_fires_and_escalates() -> None:
    # mean KL > 2 * band -> critical.
    signals = TrainingSignals(
        n_updates=10,
        approx_kl=(0.20, 0.25, 0.22),
        ep_rew_mean=(1.0, 2.0, 3.0),
        success_rate=(0.3, 0.4),
        entropy_std=(1.0, 0.99),
    )
    verdict = assess_training_health(signals=signals)
    assert "approx_kl_runaway" in _rule_ids(verdict)
    reason = next(r for r in verdict.reasons if r.rule_id == "approx_kl_runaway")
    assert reason.status == "critical"


def test_approx_kl_runaway_warning_band() -> None:
    # band < mean KL <= 2 * band -> warning.
    signals = TrainingSignals(
        n_updates=10,
        approx_kl=(0.06, 0.07, 0.065),
        ep_rew_mean=(1.0, 2.0, 3.0),
        success_rate=(0.3, 0.4),
        entropy_std=(1.0, 0.99),
    )
    reason = next(
        r
        for r in assess_training_health(signals=signals).reasons
        if r.rule_id == "approx_kl_runaway"
    )
    assert reason.status == "warning"


def test_approx_kl_not_fired_within_band() -> None:
    signals = TrainingSignals(
        n_updates=10,
        approx_kl=(0.01, 0.02, 0.015),
        ep_rew_mean=(1.0, 2.0, 3.0),
        success_rate=(0.3, 0.4),
        entropy_std=(1.0, 0.99),
    )
    assert "approx_kl_runaway" not in _rule_ids(assess_training_health(signals=signals))


# --- AC3: value_loss divergence -----------------------------------------------------------


def test_value_loss_divergence_fires_on_growth_and_ev_collapse() -> None:
    signals = TrainingSignals(
        n_updates=10,
        value_loss=(0.5, 2.0, 5.0),
        explained_variance=(0.8, 0.1, -0.2),
        ep_rew_mean=(1.0, 2.0, 3.0),
        success_rate=(0.3, 0.4),
        entropy_std=(1.0, 0.99),
    )
    verdict = assess_training_health(signals=signals)
    assert "value_loss_divergence" in _rule_ids(verdict)
    # both growth AND EV collapse -> critical.
    reason = next(r for r in verdict.reasons if r.rule_id == "value_loss_divergence")
    assert reason.status == "critical"


def test_value_loss_divergence_warning_on_growth_only() -> None:
    signals = TrainingSignals(
        n_updates=10,
        value_loss=(0.5, 2.0, 5.0),
        explained_variance=(0.8, 0.82, 0.81),
        ep_rew_mean=(1.0, 2.0, 3.0),
        success_rate=(0.3, 0.4),
        entropy_std=(1.0, 0.99),
    )
    reason = next(
        r
        for r in assess_training_health(signals=signals).reasons
        if r.rule_id == "value_loss_divergence"
    )
    assert reason.status == "warning"


def test_value_loss_not_fired_when_stable() -> None:
    signals = TrainingSignals(
        n_updates=10,
        value_loss=(0.5, 0.5, 0.48),
        explained_variance=(0.8, 0.82, 0.81),
        ep_rew_mean=(1.0, 2.0, 3.0),
        success_rate=(0.3, 0.4),
        entropy_std=(1.0, 0.99),
    )
    assert "value_loss_divergence" not in _rule_ids(assess_training_health(signals=signals))


# --- AC3: premature entropy collapse ------------------------------------------------------


def test_premature_entropy_collapse_fires() -> None:
    signals = TrainingSignals(
        n_updates=10,
        entropy_std=(1.0, 0.5, 0.1),
        success_rate=(0.0, 0.0, 0.0),
        ep_rew_mean=(1.0, 1.0, 1.0),
        explained_variance=(0.1, 0.1),
    )
    assert "premature_entropy_collapse" in _rule_ids(assess_training_health(signals=signals))


def test_premature_entropy_collapse_suppressed_when_reward_climbing() -> None:
    # A collapse alongside real progress is not premature — it should not fire.
    signals = TrainingSignals(
        n_updates=10,
        entropy_std=(1.0, 0.5, 0.1),
        success_rate=(0.0, 0.0, 0.0),
        ep_rew_mean=(1.0, 5.0, 10.0),
        explained_variance=(0.1, 0.1),
    )
    assert "premature_entropy_collapse" not in _rule_ids(assess_training_health(signals=signals))


def test_premature_entropy_collapse_not_fired_when_std_flat() -> None:
    signals = TrainingSignals(
        n_updates=10,
        entropy_std=(1.0, 0.99, 1.0),
        success_rate=(0.0, 0.0, 0.0),
        ep_rew_mean=(1.0, 1.0, 1.0),
    )
    assert "premature_entropy_collapse" not in _rule_ids(assess_training_health(signals=signals))


# --- AC3 pitfall: warm-up guard checked FIRST ---------------------------------------------


def test_warmup_guard_short_circuits_before_any_rule() -> None:
    # A history that would trip several rules, but below min_updates -> normal, no reasons.
    signals = TrainingSignals(
        n_updates=DEFAULT_THRESHOLDS.min_updates - 1,
        ep_rew_mean=(10.0, 1.0),
        success_rate=(0.0, 0.0),
        entropy_std=(1.0, 0.05),
        approx_kl=(0.5, 0.9),
        value_loss=(0.5, 100.0),
        explained_variance=(0.9, -0.9),
    )
    verdict = assess_training_health(signals=signals)
    assert verdict.status == "normal"
    assert verdict.reasons == []
    assert "insufficient" in verdict.message.lower()


def test_at_min_updates_rules_are_evaluated() -> None:
    signals = TrainingSignals(
        n_updates=DEFAULT_THRESHOLDS.min_updates,
        ep_rew_mean=(5.0, 5.0, 5.0),
        success_rate=(0.3, 0.3),
        entropy_std=(1.0, 0.99),
    )
    assert "reward_stalled" in _rule_ids(assess_training_health(signals=signals))


# --- status precedence + summary ----------------------------------------------------------


def test_worst_status_precedence() -> None:
    assert worst_status([]) == "normal"
    assert worst_status([HealthReason("a", "", "normal")]) == "normal"
    assert worst_status([HealthReason("a", "", "warning")]) == "warning"
    assert (
        worst_status([HealthReason("a", "", "warning"), HealthReason("b", "", "critical")])
        == "critical"
    )
    assert (
        worst_status([HealthReason("b", "", "critical"), HealthReason("a", "", "warning")])
        == "critical"
    )


def test_verdict_status_is_worst_of_reasons() -> None:
    # KL runaway (critical) + reward stalled (warning) -> overall critical.
    signals = TrainingSignals(
        n_updates=10,
        approx_kl=(0.20, 0.25, 0.22),
        ep_rew_mean=(5.0, 5.0, 5.0),
        success_rate=(0.3, 0.3),
        entropy_std=(1.0, 0.99),
    )
    verdict = assess_training_health(signals=signals)
    statuses = {r.status for r in verdict.reasons}
    assert "critical" in statuses and "warning" in statuses
    assert verdict.status == "critical"


def test_summary_message_reflects_status_and_extra_count() -> None:
    signals = TrainingSignals(
        n_updates=10,
        approx_kl=(0.20, 0.25, 0.22),
        ep_rew_mean=(5.0, 5.0, 5.0),
        success_rate=(0.3, 0.3),
        entropy_std=(1.0, 0.99),
    )
    message = assess_training_health(signals=signals).message
    assert message.startswith("CRITICAL:")
    assert "more issue" in message  # >1 reason -> "(+N more issues)"


def test_normal_run_message() -> None:
    signals = TrainingSignals(
        n_updates=10,
        ep_rew_mean=(1.0, 2.0, 3.0),
        success_rate=(0.3, 0.5),
        entropy_std=(1.0, 0.9),
        explained_variance=(0.7, 0.75),
        approx_kl=(0.01, 0.01),
        value_loss=(0.5, 0.45),
    )
    verdict = assess_training_health(signals=signals)
    assert verdict.status == "normal"
    assert verdict.reasons == []
    assert verdict.message == "Training progressing normally."


# --- AC4: capacity-only path (pure engine side) -------------------------------------------


def test_capacity_only_under_floor_is_critical() -> None:
    facts = CapacityFacts(trainable_params=942, obs_dim=12, action_dim=4, neuron_count=45)
    verdict = assess_training_health(capacity=facts)
    assert verdict.status == "critical"  # params * 2 < floor -> critical
    assert _rule_ids(verdict) == {"under_capacity"}
    assert "942" in verdict.message and str(DEFAULT_CAPACITY_FLOOR) in verdict.message


def test_capacity_only_near_floor_is_warning() -> None:
    # floor/2 <= params < floor -> warning, not critical.
    params = DEFAULT_CAPACITY_FLOOR - 100
    facts = CapacityFacts(trainable_params=params, obs_dim=12, action_dim=4)
    assert assess_training_health(capacity=facts).status == "warning"


def test_capacity_only_above_floor_is_normal() -> None:
    facts = CapacityFacts(trainable_params=7918, obs_dim=12, action_dim=4)
    verdict = assess_training_health(capacity=facts)
    assert verdict.status == "normal"
    assert verdict.reasons == []


def test_capacity_folded_into_runtime_verdict() -> None:
    # When both signals and capacity are passed, the under-capacity reason is folded in.
    signals = TrainingSignals(
        n_updates=10,
        ep_rew_mean=(1.0, 2.0, 3.0),
        success_rate=(0.3, 0.5),
        entropy_std=(1.0, 0.9),
    )
    facts = CapacityFacts(trainable_params=942, obs_dim=12, action_dim=4)
    verdict = assess_training_health(signals=signals, capacity=facts)
    assert "under_capacity" in _rule_ids(verdict)
    assert verdict.status == "critical"


def test_capacity_floor_override_via_thresholds() -> None:
    facts = CapacityFacts(trainable_params=942, obs_dim=12, action_dim=4)
    lax = replace(DEFAULT_THRESHOLDS, capacity_floor=500)
    assert assess_training_health(capacity=facts, thresholds=lax).status == "normal"


# --- pitfall: thresholds are named constants, not magic numbers ---------------------------


def test_thresholds_have_documented_defaults() -> None:
    t = HealthThresholds()
    assert t.min_updates > 0
    assert t.capacity_floor == DEFAULT_CAPACITY_FLOOR
    assert 0 < t.success_eps < 1
    assert DEFAULT_THRESHOLDS.capacity_floor == DEFAULT_CAPACITY_FLOOR
