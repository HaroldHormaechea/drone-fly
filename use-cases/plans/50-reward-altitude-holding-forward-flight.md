---
plan_for: use-cases/50-reward-altitude-holding-forward-flight.md
work_branch: feat/uc-50-reward-altitude-holding-forward-flight
team: drone-fly-uc-50
approved: 2026-09-22
---

# UC-50 — FINAL APPROVED PROPOSAL (analyst, challenger-approved Revision 2 + pitfall-#5 addendum)

Approach **C**: progress-gate + potential-based altitude-track, feature-gated **OFF** by default.

## Analysis
Diagnose-first confirmed the reward-gradient conflict in source:
- **Progress bug real.** `reward.py` pays `progress_weight * (dist_prev − dist_curr)` where `racing_env._dist_to_target` (l.347–349) is a **3D** Euclidean norm to `current_target`. Closing horizontally faster than opening vertically nets positive progress *while sinking* (matches the ep_250 evidence). `progress_weight=1.0`.
- **Altitude-holding pays ~nothing.** `airborne_bonus=0.2` graded to `climb_target_height=1.0`; UC-39 climb + UC-43 ground-break are potential-based/telescoping → reward altitude *change*, ≈0 for level flight. Nothing rewards *holding* near gate-z or penalizes sinking-before-crash.
- **Finish leg free:** `geometry.current_target` (l.88–100) inherits the last gate's (y,z) on the finish leg → a reference from `current_target[2]` handles pitfall #4 with no special-casing.
- Default course: gates z={1.0,1.3,0.9}, floor_z=0, ceiling_z=2.5, `climb_target_height=1.0` (aligns with gate-0).

## Proposed Solution — Approach C (progress-gate + potential-based altitude-track), feature-gated OFF by default
1. **New `compute_reward` inputs:** `target_height_above_floor_prev` / `..._curr` (both default `0.0`). Env supplies `current_target(course, gates_passed)[2] − floor_z`.
2. **Progress gate (A):** when enabled, pay progress only if `height_above_floor_curr ≥ max(ref_eff − altitude_band, ground_break_height)`, where `ref_eff = max(target_height_above_floor_curr, ground_break_height)`. One-sided (lower edge only). Guarantees AC-1 for arbitrary step size (only zeroing progress beats an unbounded progress term).
3. **Altitude-shortfall track term (B), potential-based — NON-NEGATIVE anchor (corrected during implementation, analyst+challenger approved):** `Φ_track(h;ref) = altitude_hold_weight · clamp(h − (ref_eff − altitude_band), 0, altitude_band)`; per-step `F = climb_gamma·Φ(h_curr;ref_curr) − Φ(h_prev;ref_prev)`. Reuses `climb_gamma` (==training γ). Climbing pays +, sinking pays −, telescopes (round trip ≈0, no loiter optimum), per-episode magnitude ≤ `altitude_hold_weight·altitude_band`. Provides the AC-2 tiebreak; standing positive for being airborne stays the existing `airborne_bonus`.
   - **Sign-correction note:** the originally-drafted `Φ_track = −altitude_hold_weight·min(band, max(0, ref−h))` was NEGATIVE, which makes the potential-based steady leak `(γ−1)·Φ` POSITIVE (γ<1) → a farmable loiter income (bob nets +0.018/cycle), violating AC-4 and inverting the UC-39/43 non-negative-potential convention. The corrected form equals the original plus the constant `altitude_hold_weight·band`: identical climbing gradient (AC-1/AC-2 preserved, AC-1 strengthened), non-negative Φ ⇒ leak `(γ−1)Φ ≤ 0` (bob nets −0.006, no loiter income), inert below the band edge (F=0 ⇒ cleaner AC-5 bootstrap), per-episode bound unchanged (`≤ climb_gamma·w·band ≈ 1.2 ≪ completion_bonus`). `ref_eff = max(target_height_above_floor, ground_break_height)`.

**New `RewardConfig` fields (appended last; defaults keep byte-identity):**
- `enable_altitude_decoupling: bool = False` — master gate; when False, `compute_reward` runs the exact pre-change statements → byte-identical (AC-6).
- `altitude_hold_weight: float = 0.0` (recommended enabled ~2.0).
- `altitude_band: float` (recommended ~0.6) — half-width/shortfall clamp; comment MUST document the sizing invariant `ref − band ≤ climb_target_height` (else high-first-gate re-introduces the UC-40/41 takeoff chicken-and-egg) and the enabled-value combined bound (2.0×0.6=1.2 → 163.68 < completion_bonus 200).

## Files Affected
**Production code (developer):**
- `src/drone_fly/env/reward.py` — new params (default 0.0); enabled-gated progress gate + potential-based shortfall term with `ref_eff` lower-clamp; extend docstring (incl. one-sided rationale / no ceiling clamp needed).
- `src/drone_fly/env/config.py` — new `RewardConfig` fields, fully documented (UC-39/43 comment style; sizing invariant; enabled-value bound).
- `src/drone_fly/env/racing_env.py` — **call-site wiring ONLY** (team-lead-approved relaxation): capture `target_height_above_floor_prev = current_target(course, self._gates_passed)[2] − course.floor_z` **at ~l.482 alongside `dist_prev`** (PRE-advance gates_passed), and `..._curr` **at ~l.515 alongside `dist_curr`** (POST-advance), pass both into `compute_reward(...)` at ~l.706. **Do NOT read `self._gates_passed` at l.706 — it is clobbered by `advance()` at l.509.** No other change; byte-identical when feature off.
- `README.md` (repo root) — reward-section table rows for the new term(s) — MANDATORY (AC-9).

**Test code (qa):**
- New `tests/test_uc50_altitude_holding_reward.py` — hermetic, relative/structural (AC-8):
  - **AC-1:** below-band, horizontally-closing, descending step → `progress_term + altitude_track_term ≤ 0` (isolated via difference-from-baseline, like `_climb_contribution`); companion: sinking-below-band total < level-in-band total.
  - **AC-2:** equal horizontal closing → level-in-band strictly > descend-below-ref.
  - **AC-3:** reference shifts with `target_height_above_floor` (two different values).
  - **AC-4:** climb-then-descend round trip ≈0; per-episode track bound ≪ completion_bonus; no loiter optimum.
  - **AC-5:** default-course bootstrap — below-edge step net-positive & progress-independent from untouched ground-break/climb; progress un-gates at h ≥ edge.
  - **AC-6:** byte-identity regression — representative pre-change calls equal with defaults.

## Risks & Considerations
- **Scope:** three production files (reward.py, config.py, racing_env.py wiring-only) + README + test — per team-lead ruling.
- **High-first-gate stranding:** documented sizing invariant `ref − band ≤ climb_target_height`; default course safe (edge 0.4 ≤ 1.0); AC-5 test pins it.
- **Floor/ceiling-adjacent gates:** lower clamp via `ref_eff`/edge `max(…, ground_break_height)`; one-sided band needs no ceiling clamp (gate-z is a reachable target by construction).
- **Behavioral verdict** needs a fresh GPU retrain (out of scope; hermetic tests encode the reward property only). Checkpoints from the current run are invalidated by this reward change.

**Challenger verdict:** APPROVE (peer loop closed at round 2; pitfall-#5 addendum accepted, approval unchanged).
