"""Headless, pure-logic training-health assessment engine (UC-23).

This module answers one question with **no I/O and no global/process state**: given a
snapshot of PPO training signals (and/or the seeded actor's capacity), is the run healthy,
worth a warning, or in trouble — and *why*? The answer is a structured
:class:`HealthVerdict` (``status`` + human ``message`` + a list of contributing
:class:`HealthReason`), which UC-22's status bar renders and which the runtime callback also
emits as a plain log line when no TUI is attached.

It codifies the diagnosis this project keeps hitting: a **healthy critic with a flat,
non-committing actor and ~0% success is the signature of an under-capacity connectome slice**
(the k=0 minimal corridor), plus the classic PPO failure modes — reward stalled over a long
window, ``approx_kl`` runaway, ``value_loss`` divergence, and premature entropy collapse.

Design contract
---------------
* Pure: every rule is a threshold-driven function of the passed-in snapshot. No file, no
  clock, no RNG, no SB3 import. Fully unit-testable by feeding synthetic histories.
* Stable shape: :class:`HealthVerdict` / :class:`HealthReason` are the interface UC-22 plugs
  into (it renders the verdict; it never re-derives it).
* Named thresholds: every number lives on :class:`HealthThresholds` with a documented
  default — there are no magic numbers scattered through the rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

#: Verdict / reason severity. Ordered normal < warning < critical.
HealthStatus = Literal["normal", "warning", "critical"]

_STATUS_RANK: dict[str, int] = {"normal": 0, "warning": 1, "critical": 2}


# --- Verdict shape (the UC-22 interface) --------------------------------------------------


@dataclass(frozen=True)
class HealthReason:
    """One rule's firing: a stable ``rule_id``, a human ``text``, and its ``status``."""

    rule_id: str
    text: str
    status: HealthStatus


@dataclass(frozen=True)
class HealthVerdict:
    """Overall training-health verdict.

    ``status`` is the worst status across ``reasons`` (``normal`` when none fired);
    ``message`` is a short human summary; ``reasons`` lists every rule that fired (empty on a
    clean/insufficient-history verdict). This is exactly the object UC-22's status bar renders.
    """

    status: HealthStatus
    message: str
    reasons: list[HealthReason] = field(default_factory=list)


# --- Inputs -------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingSignals:
    """A rolling-window snapshot of PPO training signals.

    Each metric is a short sequence of recent values (oldest first, current value last); an
    empty sequence means "not observed yet" (e.g. a ``train/*`` key absent on the first
    rollout). ``n_updates`` is the callback-owned count of completed rollouts/updates — it is
    NOT read from SB3's ``name_to_value`` (which has no such stable key).

    ``entropy_std`` is SB3's ``train/std`` (the Gaussian action std) used as the entropy /
    action-commitment proxy: SB3 logs no raw entropy for a continuous action space, and a
    flat, high ``std`` is exactly the "actor not committing" signature.
    """

    n_updates: int = 0
    ep_rew_mean: Sequence[float] = ()
    success_rate: Sequence[float] = ()
    explained_variance: Sequence[float] = ()
    entropy_std: Sequence[float] = ()
    approx_kl: Sequence[float] = ()
    value_loss: Sequence[float] = ()


@dataclass(frozen=True)
class CapacityFacts:
    """Facts about the resolved (post-prune), seeded actor for the pre-train guardrail.

    ``trainable_params`` is the sum of ``numel()`` over the actor's ``requires_grad``
    parameters (dominated by the sparse per-edge connectome weights). ``neuron_count`` /
    ``edge_count`` are optional and only enrich the message.
    """

    trainable_params: int
    obs_dim: int
    action_dim: int
    neuron_count: int | None = None
    edge_count: int | None = None


@dataclass(frozen=True)
class HealthThresholds:
    """Named, documented thresholds for every rule (no magic numbers in the rules).

    Attributes
    ----------
    min_updates:
        Warm-up guard: below this many completed updates the engine returns ``normal`` with an
        "insufficient history" message and inspects no metric series (avoids early-run
        false positives).
    success_eps:
        A ``success_rate`` at or below this counts as "~0" (pinned at no completions).
    entropy_flat_tol:
        Relative change of ``entropy_std`` within ``±entropy_flat_tol`` over the window counts
        as "flat" (actor not committing); a drop steeper than ``-entropy_flat_tol`` counts as
        "falling".
    ev_healthy:
        Mean ``explained_variance`` at or above this counts as a healthy critic.
    kl_target_band:
        Sustained ``approx_kl`` above this band is a runaway (updates too aggressive).
    value_loss_divergence_factor:
        ``value_loss`` growing by at least this factor across the window counts as diverging.
    entropy_collapse_frac:
        A relative ``entropy_std`` drop steeper than this (crashing toward zero) counts as a
        collapse.
    reward_stall_tol:
        Relative ``ep_rew_mean`` improvement at or below this counts as stalled.
    capacity_floor:
        Minimum acceptable actor ``trainable_params``; below it the pre-train guardrail flags
        under-capacity. Default :data:`DEFAULT_CAPACITY_FLOOR` (calibrated against the
        committed fixture — see below).
    """

    min_updates: int = 5
    success_eps: float = 0.02
    entropy_flat_tol: float = 0.05
    ev_healthy: float = 0.5
    kl_target_band: float = 0.05
    value_loss_divergence_factor: float = 3.0
    entropy_collapse_frac: float = 0.30
    reward_stall_tol: float = 0.02
    capacity_floor: int = 3000


#: Actor trainable-parameter floor, calibrated on the committed ``tests/fixtures`` connectome:
#: the k=0 minimal corridor yields 942 trainable params (undersized) while the k=2 default
#: slice (``connectome.prune.DEFAULT_PRUNE_K == 2``) yields 7918. A floor of 3000 sits cleanly
#: between them (k0 trips it, the default slice passes with margin). The calibration test in
#: ``tests/test_capacity_guard.py`` locks this ordering in.
DEFAULT_CAPACITY_FLOOR: int = 3000

#: The default threshold set used everywhere unless a caller overrides it.
DEFAULT_THRESHOLDS = HealthThresholds(capacity_floor=DEFAULT_CAPACITY_FLOOR)


# --- Small pure helpers -------------------------------------------------------------------


def _clean(seq: Sequence[float] | None) -> list[float]:
    """Drop ``None``/non-finite entries and coerce to floats (oldest first, current last)."""
    out: list[float] = []
    for x in seq or ():
        if x is None:
            continue
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if v == v and v not in (float("inf"), float("-inf")):  # NaN/inf guard
            out.append(v)
    return out


def _mean(seq: Sequence[float]) -> float:
    vals = _clean(seq)
    return sum(vals) / len(vals) if vals else 0.0


def _rel_change(seq: Sequence[float]) -> float:
    """Signed relative change from the first to the last value of the window.

    ``(last - first) / max(|first|, 1)`` so a tiny/zero baseline can't blow the ratio up.
    Returns ``0.0`` when there are fewer than two observations (no trend yet).
    """
    vals = _clean(seq)
    if len(vals) < 2:
        return 0.0
    first, last = vals[0], vals[-1]
    denom = abs(first) if abs(first) > 1e-9 else 1.0
    return (last - first) / denom


def worst_status(reasons: Sequence[HealthReason]) -> HealthStatus:
    """Return the highest-severity status across ``reasons`` (``normal`` when empty)."""
    status: HealthStatus = "normal"
    for r in reasons:
        if _STATUS_RANK[r.status] > _STATUS_RANK[status]:
            status = r.status
    return status


# --- Capacity assessment (pre-train guardrail) --------------------------------------------


def _assess_capacity(capacity: CapacityFacts, thresholds: HealthThresholds) -> HealthVerdict:
    """Pre-train capacity verdict: under the floor → under-capacity, else ``normal`` (AC4).

    Severity escalates to ``critical`` when the count is less than half the floor (a badly
    undersized slice), otherwise ``warning``.
    """
    params = int(capacity.trainable_params)
    floor = int(thresholds.capacity_floor)
    detail = f"{params} trainable actor params"
    if capacity.neuron_count is not None or capacity.edge_count is not None:
        bits = []
        if capacity.neuron_count is not None:
            bits.append(f"{capacity.neuron_count} neurons")
        if capacity.edge_count is not None:
            bits.append(f"{capacity.edge_count} edges")
        detail += f" ({', '.join(bits)})"

    if params >= floor:
        return HealthVerdict(
            "normal",
            f"Actor capacity OK: {detail} >= floor {floor}.",
            [],
        )
    status: HealthStatus = "critical" if params * 2 < floor else "warning"
    text = (
        f"Under-capacity actor: {detail} < floor {floor} — connectome slice likely too small; "
        f"the actor may be unable to learn. Consider a larger --prune-k slice."
    )
    return HealthVerdict(
        status,
        text,
        [HealthReason("under_capacity", text, status)],
    )


# --- Runtime assessment (five independent rules) ------------------------------------------


def _rule_under_capacity_signature(
    signals: TrainingSignals, thresholds: HealthThresholds
) -> HealthReason | None:
    """K0 signature: healthy/rising EV + flat-high entropy + ~0 success over the window (AC2).

    Suppressed when entropy is falling or when success/reward is climbing (those mean the
    policy IS committing / learning, so it is not the stuck-undersized signature).
    """
    ev_ok = _mean(signals.explained_variance) >= thresholds.ev_healthy or (
        _rel_change(signals.explained_variance) > 0
    )
    entropy_change = _rel_change(signals.entropy_std)
    entropy_flat = abs(entropy_change) <= thresholds.entropy_flat_tol
    entropy_falling = entropy_change < -thresholds.entropy_flat_tol
    success_zero = _mean(signals.success_rate) <= thresholds.success_eps
    reward_climbing = _rel_change(signals.ep_rew_mean) > thresholds.reward_stall_tol
    success_climbing = _rel_change(signals.success_rate) > 0

    if entropy_falling or reward_climbing or success_climbing:
        return None
    if not (ev_ok and entropy_flat and success_zero):
        return None

    # Escalate to critical once the signature has persisted well past warm-up.
    status: HealthStatus = (
        "critical" if signals.n_updates >= thresholds.min_updates * 4 else "warning"
    )
    text = (
        "Likely-undersized connectome slice (K0 signature): critic is healthy "
        "(explained_variance high/rising) but the actor is not committing (action std flat "
        "and high) and success_rate is pinned at ~0. The slice may lack the capacity to learn."
    )
    return HealthReason("under_capacity_signature", text, status)


def _rule_reward_stalled(
    signals: TrainingSignals, thresholds: HealthThresholds
) -> HealthReason | None:
    """Reward stalled/declining: ``ep_rew_mean`` not improving over the window (AC3)."""
    if len(_clean(signals.ep_rew_mean)) < 2:
        return None
    change = _rel_change(signals.ep_rew_mean)
    if change > thresholds.reward_stall_tol:
        return None  # improving — not stalled
    text = (
        "Reward stalled: ep_rew_mean is not improving (or declining) over the recent window "
        "despite ongoing updates."
    )
    return HealthReason("reward_stalled", text, "warning")


def _rule_approx_kl_runaway(
    signals: TrainingSignals, thresholds: HealthThresholds
) -> HealthReason | None:
    """approx_kl runaway: KL sustained above the target band (AC3)."""
    vals = _clean(signals.approx_kl)
    if not vals:
        return None
    mean_kl = _mean(signals.approx_kl)
    if mean_kl <= thresholds.kl_target_band or vals[-1] <= thresholds.kl_target_band:
        return None  # not sustained above the band
    status: HealthStatus = "critical" if mean_kl > 2 * thresholds.kl_target_band else "warning"
    text = (
        f"approx_kl runaway: KL sustained above the target band ({thresholds.kl_target_band:.3g}); "
        "policy updates are too aggressive and may be diverging."
    )
    return HealthReason("approx_kl_runaway", text, status)


def _rule_value_loss_divergence(
    signals: TrainingSignals, thresholds: HealthThresholds
) -> HealthReason | None:
    """value_loss divergence: value_loss growing unbounded and/or EV collapsing (AC3)."""
    vloss = _clean(signals.value_loss)
    ev = _clean(signals.explained_variance)

    vloss_diverging = (
        len(vloss) >= 2
        and vloss[0] > 1e-9
        and vloss[-1] >= vloss[0] * thresholds.value_loss_divergence_factor
    )
    # Critic breaking down: EV was positive and has dropped sharply / gone negative.
    ev_collapsing = len(ev) >= 2 and ev[0] > 0 and (ev[-1] <= 0 or ev[-1] < ev[0] - 0.5)

    if not (vloss_diverging or ev_collapsing):
        return None
    status: HealthStatus = "critical" if (vloss_diverging and ev_collapsing) else "warning"
    parts = []
    if vloss_diverging:
        parts.append("value_loss growing without bound")
    if ev_collapsing:
        parts.append("explained_variance collapsing")
    text = f"value_loss divergence: {' and '.join(parts)} — the critic is breaking down."
    return HealthReason("value_loss_divergence", text, status)


def _rule_premature_entropy_collapse(
    signals: TrainingSignals, thresholds: HealthThresholds
) -> HealthReason | None:
    """Premature entropy collapse: std crashing near zero before any success/reward gain (AC3)."""
    if len(_clean(signals.entropy_std)) < 2:
        return None
    entropy_change = _rel_change(signals.entropy_std)
    if entropy_change > -thresholds.entropy_collapse_frac:
        return None  # not collapsing
    success_zero = _mean(signals.success_rate) <= thresholds.success_eps
    reward_climbing = _rel_change(signals.ep_rew_mean) > thresholds.reward_stall_tol
    if not success_zero or reward_climbing:
        return None  # collapse alongside real progress is not premature
    text = (
        "Premature entropy collapse: action std is crashing toward zero before any success or "
        "reward gain — the policy is going deterministic too early."
    )
    return HealthReason("premature_entropy_collapse", text, "warning")


_RUNTIME_RULES = (
    _rule_under_capacity_signature,
    _rule_reward_stalled,
    _rule_approx_kl_runaway,
    _rule_value_loss_divergence,
    _rule_premature_entropy_collapse,
)


def _assess_runtime(signals: TrainingSignals, thresholds: HealthThresholds) -> HealthVerdict:
    """Runtime verdict from the five rules, warm-up guard checked FIRST (AC3, pitfall).

    Below ``min_updates`` completed updates the engine returns ``normal``/"insufficient
    history" without inspecting any metric series, so early-run noise cannot fire a rule.
    """
    if signals.n_updates < thresholds.min_updates:
        return HealthVerdict(
            "normal",
            f"Insufficient training history ({signals.n_updates} updates); no assessment yet.",
            [],
        )
    reasons = [r for rule in _RUNTIME_RULES if (r := rule(signals, thresholds)) is not None]
    status = worst_status(reasons)
    return HealthVerdict(status, _summarize(status, reasons), reasons)


# --- Public entry point -------------------------------------------------------------------


def _summarize(status: HealthStatus, reasons: Sequence[HealthReason]) -> str:
    """Build a short human message from the fired reasons."""
    if not reasons:
        return "Training progressing normally."
    primary = next((r for r in reasons if r.status == status), reasons[0])
    extra = len(reasons) - 1
    suffix = f" (+{extra} more issue{'s' if extra != 1 else ''})" if extra > 0 else ""
    return f"{status.upper()}: {primary.text}{suffix}"


def assess_training_health(
    signals: TrainingSignals | None = None,
    capacity: CapacityFacts | None = None,
    thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
) -> HealthVerdict:
    """Assess training health and return a structured :class:`HealthVerdict` (AC1/AC4).

    Modes:

    * ``signals is None`` → **pre-train / capacity-only** path (the guardrail): requires
      ``capacity``; returns its verdict.
    * ``signals`` present → **runtime** assessment; if ``capacity`` is also passed its
      under-capacity reason is folded in as an extra reason.
    * both ``None`` → a trivial ``normal`` verdict (nothing to assess).

    Pure: no I/O, no global state — the same inputs always yield the same verdict.
    """
    if signals is None and capacity is None:
        return HealthVerdict("normal", "No training signals provided.", [])

    reasons: list[HealthReason] = []
    runtime_message: str | None = None

    if capacity is not None:
        reasons.extend(_assess_capacity(capacity, thresholds).reasons)

    if signals is not None:
        runtime = _assess_runtime(signals, thresholds)
        reasons.extend(runtime.reasons)
        if not runtime.reasons:
            # Preserve the "insufficient history" / normal runtime message when nothing fired.
            runtime_message = runtime.message

    status = worst_status(reasons)
    if reasons:
        message = _summarize(status, reasons)
    elif runtime_message is not None:
        message = runtime_message
    elif capacity is not None:
        message = _assess_capacity(capacity, thresholds).message
    else:  # pragma: no cover - covered by the both-None guard above
        message = "Training progressing normally."
    return HealthVerdict(status, message, reasons)


__all__ = [
    "HealthStatus",
    "HealthReason",
    "HealthVerdict",
    "TrainingSignals",
    "CapacityFacts",
    "HealthThresholds",
    "DEFAULT_THRESHOLDS",
    "DEFAULT_CAPACITY_FLOOR",
    "assess_training_health",
    "worst_status",
]
