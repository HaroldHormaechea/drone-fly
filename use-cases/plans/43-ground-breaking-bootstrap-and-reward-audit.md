---
plan_for: use-cases/43-ground-breaking-bootstrap-and-reward-audit.md
work_branch: feat/uc-43-ground-breaking-bootstrap-and-reward-audit
team: drone-fly-uc-43
approved: 2026-09-21
---

# UC-43 — Final Approved Implementation Plan
_Sub-threshold ground-breaking reward + reward-system audit. Challenger APPROVED v1 (no revision round). All four Minor directives folded in. All paths inside TARGET_DIR `/workspace/drone-fly-uc-43-ground-breaking-bootstrap-and-reward-audit`._

## Analysis

### Problem summary
UC-40 (hover-bias), UC-41 (hold-then-ramp collision curriculum), UC-42 (altitude-graded airborne reward) are merged, yet a fresh ~99k-step run shows the drone NEVER breaks ground: z pinned at ~0.0135 m in every recording, `ep_rew_mean` glued to exactly −5, throttle mean flat at hover ~0.48. This plan (a) adds a non-farmable sub-threshold "ground-breaking" reward that creates a positive gradient below the airborne threshold, and (b) audits every reward term for dead zones / perverse incentives, fixing only the one clearly-wrong, evidence-backed takeoff-blocker and documenting the rest.

### Diagnosis with evidence (AC1)
**Planted-population reading confirmed.** `ep_rew_mean = −5 = −time_penalty(0.05) × stuck_window(100)`. `stuck_window=100` (`config.py:834`). Chain: floored start (racing_env reset drops z to `floor_z`) → drone never takes off → `grounded_window`(10) never *arms* (it arms only post-takeoff, racing_env.py ~508-516) → the ungated stuck detector cuts at 100 steps → UC-38 makes a stuck cut **penalty-free** (`racing_env.py:571`, `penalize_collision = crash or (early_termination == "grounded")`) → pure −0.05×100 = −5, with **zero** airborne bonus / progress / climb credit. Any airborne time would add `airborne_bonus·(h/target)` and lift the mean off −5. So −5 exact ⇒ the entire episode population is planted, not just the sampled recordings.

**Constants (read from code).** `floor_z=0.0` (`config.py:136`), `floor_epsilon=0.05` (`config.py:833`) ⇒ airborne threshold = `floor_z + floor_epsilon` = 0.05 m. The simple adapter clamps z to `floor_z` and zeroes vz on floor contact (`simple.py:293-295`), so the resting state is z=0.0; the "0.0135 m" seen in recordings is the *peak* of tiny noise-driven hops that never reach 0.05.

**Why no sub-threshold gradient bootstraps takeoff.**
- **Airborne graded bonus (UC-42)** — `reward.py:120-128`: `if airborne: airborne_bonus·min(max(h,0),target)/target`. The `airborne` flag is the boolean gate at **`racing_env.py:384`** (`z > floor_z + floor_epsilon`). At the hop peak 0.0135 < 0.05 the flag never trips, so this term contributes **exactly 0** for the whole planted population. This is the canonical dead zone.
- **Climb potential (UC-39)** — `reward.py:129-133`: `Φ(h)=climb_weight·min(max(h,0),target)`, `F=γ·Φ(curr)−Φ(prev)`, heights passed **unconditionally** (`racing_env.py:601-602`). So it IS active below threshold with slope `climb_weight=2.0`/m. BUT (a) it is **potential-based ⇒ return-invariant** (Ng et al. 1999): a resting drone produces only transient up-then-back hops, which telescope to ≈0 net return, so PPO's episodic advantage for "throttle up" is ≈0; (b) its per-step magnitude for a realistic 0→0.0135 hop is `γ·Φ(0.0135)−0 ≈ 0.99·2·0.0135 ≈ 0.027` — smaller than `time_penalty` 0.05 and buried under reward-VecNormalize noise. So below 0.05 m the *only* net signal is negative (−0.05 time penalty + a tiny climb leak), and throttle has no gradient to rise.

**Root cause:** a dead/negative zone in `h ∈ [0, floor_epsilon]` where genuine upward progress earns no positive dense signal. The `racing_env.py:384` gate is **correct by design** (it gates the airborne latch + survival term) — it is NOT a bug, so per the hard constraint `racing_env.py` stays UNCHANGED. The fix is purely additive in `reward.py`/`config.py`.

### Reward-system audit (AC3) + dead-zone enumeration (AC7)
**Every desired outcome, with its dense non-farmable gradient confirmed:**
1. **Break ground** `h∈[0,0.05]` — WAS a dead/negative zone → **NOW** the ground-breaking potential (dense, telescoping). ✔ FIXED — the sole clearly-wrong, evidence-backed takeoff-blocker.
2. **Climb toward target** `h∈[0.05,1.0]` — climb potential (slope 2/m, telescoping) + airborne graded bonus (dense). ✔ existing.
3. **Reduce distance to next required gate** — progress term `progress_weight·(dist_prev−dist_curr)`, dense and telescoping across gate transitions (toward-then-away nets ≈0). ✔ existing.
4. **Pass a gate** — `gate_bonus/num_gates` event, with the progress term supplying the dense approach gradient. ✔ existing.
5. **Complete the course** — `completion_bonus` event + the progress/gate gradient leading in. ✔ existing.
   (Climbing above target / into the ceiling is intentionally saturated — NOT a desired outcome; the ceiling is a crash. ✔ correct non-reward, not a dead-zone violation.)

**Per-term verdicts (fix only clearly-wrong takeoff-blockers; document the rest):**
- **time_penalty (−0.05/step):** correct — keep. It is the *sole* net sub-threshold signal today (negative); the fix adds the missing positive term rather than removing the cost.
- **progress term:** healthy, dense, telescoping/non-farmable. A weak, geometry-dependent vertical pull toward an overhead gate exists but is not a reliable takeoff bootstrap. No change.
- **climb potential (UC-39):** correct but return-invariant with a weak sub-threshold slope — keep byte-identical, supplement below threshold with the new term.
- **airborne graded bonus (UC-42):** correct above threshold; its boolean gate makes it a dead zone below — remedied by the new term, UC-42 itself unchanged.
- **collision_penalty + UC-41 curriculum:** healthy — no perverse residual in scope. During the fly-learning phase (0–40% of training) the curriculum holds the penalty at `collision_penalty_start=2.0` (`train/config.py:99`), so a failed takeoff (grounded cut, ~−2 over a short episode) already out-scores the penalty-free planted stuck cut (−5). Post-UC-38 stuck/timeout cuts are penalty-free. No change.
- **obstacle_penalty (50):** edge-triggered, bounded, default course has none; no takeoff relevance. No change.
- **gate/completion bonuses:** sparse events; the progress term supplies the dense approach gradient. No change.
- **anti-suicide (pre-takeoff floor-contact suppression, racing_env ~427-437):** healthy — prevents an insta-crash at step 1 for a floored start. No change.
- **stuck/grounded early-termination:** `stuck_window=100` (penalty-free) bounds planted episodes; `grounded_window=10` arms post-takeoff and pays the penalty (a genuine failed flight). Healthy. No change.

**Documented follow-ups (out of scope; NO speculative rewrite):**
- Potential-based shaping is return-invariant, so a steeper sub-threshold potential strengthens the per-step learning signal but does not by itself change the episodic optimum for an *un-sustained* hop. If the GPU retrain shows exploration still can't produce sustained above-hover throttle, the levers are temporally-correlated exploration (OU/pink noise) and/or a higher fresh-build throttle-bias init (extending UC-40 hover-bias >0.5). `ent_coef` is already 0.005 (`train/config.py:73`).
- The progress term's geometry-dependent weak vertical pull — could add an explicit toward-gate-altitude component; deferred to avoid double-shaping.

## Proposed Solution
Add a **second potential-based "ground-breaking" shaping term**, active in the sub-threshold band and saturating exactly at the airborne threshold so it hands off cleanly to the UC-42 airborne term:
```
Φ_gb(h) = ground_break_weight · min(max(h, 0), ground_break_height) / ground_break_height
F_gb    = climb_gamma · Φ_gb(curr) − Φ_gb(prev)      # reuses climb_gamma (== training γ = 0.99)
reward += F_gb
```
New `RewardConfig` fields (appended **last** so every positional `RewardConfig` caller is unshifted):
- `ground_break_weight: float = 0.5` — the term's peak potential (tunable). **Keep 0.5 — do not shrink** (you'd lose transient strength); `w < 0.9` is the only hard bound.
- `ground_break_height: float = 0.05` — saturation height. **Documented coupling: MUST equal `EarlyTerminationConfig.floor_epsilon`** (mirrors the accepted `climb_gamma == training γ` precedent). This keeps `racing_env.py` untouched — `compute_reward` already receives `height_above_floor_{prev,curr}`, so the term is fully computable from cfg + existing args.

**Provable properties (challenger independently re-derived; all reward-math-testable, VecNormalize-safe):**
- **≈0 at rest / non-farmable floor:** Φ_gb(0)=0 ⇒ F_gb=0 at h=0. No reward for sitting.
- **Dense positive sub-threshold gradient (AC2/AC7):** slope `ground_break_weight/ground_break_height = 0.5/0.05 = 10`/m — 5× the climb slope, concentrated in the critical band. A monotone rise 0→0.05 yields strictly-increasing cumulative return `≈ γ^k·Φ_gb(h_k) − Φ_gb(0) > 0`; staying planted earns strictly less.
- **Non-farmable (AC2, pitfall 1):** potential-based ⇒ telescopes ⇒ a bob (up then back to start) nets ≈0 (only the `(γ−1)ΣΦ ≤ 0` residual).
- **Continuous handoff at threshold, no double-count (AC2, pitfall 2):** Φ_gb saturates at `ground_break_weight` exactly at h=0.05 where the airborne graded bonus begins; for any step fully above threshold `F_gb = (γ−1)·ground_break_weight = −0.005/step` — a **height-independent** constant, so the term adds NO height-dependent reward above threshold and never stacks on the airborne/climb terms. Marginal upward reward is strictly positive on both sides of the seam (below: F_gb; above: airborne increment + climb F).
- **Bounds preserved (AC4):** per-episode telescoping bound ≈ `climb_gamma·ground_break_weight = 0.495`. Combined shaping over the default 800-step budget = airborne 160 + climb 1.98 + ground-break 0.495 = **162.48 < completion_bonus 200** (headroom ~37.5). `climb_gamma==γ` invariance holds and gb-bound 0.495 ≤ gate_bonus/3.
- **Seam monotonicity bound:** existing net-hold band slope ≈ +0.18/m; the term adds −0.2·w, so net 0.18−0.2w>0 ⟺ **w<0.9**. At w=0.5 → +0.08/m, monotonicity holds.

## Files Affected

### Production code (developer)
- **`src/drone_fly/env/config.py`** — add `ground_break_weight` and `ground_break_height` to `RewardConfig` with docstrings + the `ground_break_height == EarlyTerminationConfig.floor_epsilon` coupling note. **[Directive 3] Do NOT mutate the historical `161.98` in the `completion_bonus` comment (that is UC-42's rationale for the 100→200 bump) — ADD a new UC-43 line noting the new combined bound 160+1.98+0.495 = 162.48 < 200.**
- **`src/drone_fly/env/reward.py`** — add the ground-breaking potential term immediately after the UC-39 climb term (~line 133); extend the module/function docstring to describe it (potential-based, saturates at threshold, hands off to airborne term, ≈0 at rest, bound).
- **`README.md`** — add a "Ground-breaking" row to the `Action | Reward | Condition` table (value cells: `ground_break_weight` / `ground_break_height`); add a UC-43 lever/section; write the reward-system audit (per-term verdicts + the AC7 desired-outcome enumeration + follow-ups). **[Directive 4] Keep the existing HONEST large-N caveat intact** (budget→2200, airborne 440 > completion; loiter-domination rests on the stuck detector) — adding gb's 0.495 doesn't change that story. **[Directive 1 / Risk 1] Keep the return-invariance honesty verbatim** — necessary-but-maybe-not-sufficient; behavioral takeoff is the user's GPU retrain, not the gate (the UC-40/42 lesson).
- **`src/drone_fly/env/racing_env.py`** — **UNCHANGED** (audit verdict: the `:384` airborne gate is correct-by-design, not a bug; heights already passed; threshold supplied via the cfg coupling). Endorsed by challenger.

### Test code (qa)
- **NEW `tests/test_uc43_ground_break_reward.py`** — the reward-math acceptance gate (structural/relative only, VecNormalize-safe):
  - sub-threshold monotone rise → strictly-increasing cumulative return; staying planted earns strictly less;
  - bob (0→0.03→0) telescopes to ≈0 (non-farmable); ≈0 at rest (h=0);
  - saturation `Φ_gb(0.05) == ground_break_weight`; for a step fully above threshold `F_gb == (γ−1)·ground_break_weight` (height-independent ⇒ no double-count);
  - strictly-positive marginal upward reward on both sides of the seam (no dead zone at the boundary);
  - combined-shaping bound (airborne + climb + ground-break) < `completion_bonus`;
  - explicit **unchanged-terms** assertions: `time_penalty`, `progress_weight`, `gate_bonus`, `completion_bonus`, `collision_penalty`, `obstacle_penalty`, `airborne_bonus`, `climb_weight`, `climb_target_height`, `climb_gamma`;
  - AC7 per-outcome dense-crediting: a small step toward break-ground / climb / reduce-dist-to-gate yields strictly-positive shaping, and a round-trip nets ≈0.
  - **[Directive 1] Assert the coupling against its source:** `RewardConfig().ground_break_height == EarlyTerminationConfig().floor_epsilon` (import + compare — NOT an independent `0.05` pin), so a future `floor_epsilon` change can't silently reopen a dead zone / shift the seam.
- **`tests/test_reward_docs.py`** — add a "ground" row to the `expectations` list (`ground_break_weight` / `ground_break_height`) so the new constants are doc-synced (AC6).
- **`tests/test_uc42_takeoff_reward.py` — exact-value updates (AC4; legitimate contract change, challenger signed off).** The constant standing leak `−(1−γ)·ground_break_weight = −0.005/step` (h≥threshold) shifts pinned scalars: `0.13 → 0.125`; differential `0.18 → 0.175`; `_net_hold(0.1)−SIT 0.018 → 0.013`. **[Directive 2 — must-do, red-suite risk] Also fix `test_uc42_net_hold_zero_floor_credit_and_positive_break_even`: the break-even moves to `0.055/0.18 ≈ 0.3056`, so `_net_hold(0.05/0.18) == pytest.approx(0.0, abs=2e-3)` FAILS unless recomputed** (net_hold(0.278) ≈ −0.005). (`_net_hold(0.2)<0` = −0.019 and `_net_hold(0.4)>0` = +0.017 still hold — verified.) Add a `ground_break_weight`/`ground_break_height` line to the unchanged-knobs test. Structural assertions (monotone-increasing, flat-above-target, ≈0-at-floor, telescoping round-trip) stay.
- QA runs the full `uv run --extra dev pytest`, including checking `test_uc39*`/`test_reward.py` for any exact-reward-with-height assertions that also move by the constant leak (AC6).

## Risks & Considerations
1. **Return-invariance caveat (honest — necessary-but-maybe-not-sufficient).** The reward-math gate is fully satisfied, but behavioral takeoff is the user's GPU retrain (explicitly not the gate, AC5). The README audit states this plainly (kept verbatim) rather than over-promising as UC-40 did on the smoke-train.
2. **Coupling `ground_break_height == floor_epsilon`.** Not auto-derived (RewardConfig can't see EarlyTerminationConfig without a layering change). Mirrors the accepted `climb_gamma==γ` pattern; asserted against source in the new test (Directive 1). If `floor_epsilon` changes, update in lockstep.
3. **Overlap of the ground-break and climb potentials in [0,0.05].** Both potential-based; their sum is a potential ⇒ still telescopes, no non-farmability issue. Intentional — it strengthens the critical band.
4. **Default-ON shifts pinned UC-42 (and possibly UC-39/test_reward) scalars by the benign constant leak.** Unlike UC-39/42 which could default their weights to 0 for byte-identity, ground-break must ship ON (it *is* the fix). Legitimate contract change (challenger signed off) — the structure (telescoping, non-farmable, bounds, monotonicity) is preserved; only pinned constants move by a constant. Keep w=0.5.

## Challenger's Verdict — APPROVE
Independently verified the load-bearing math against the actual code (reward.py, config.py RewardConfig, racing_env.py:384 gate, train/config.py γ=0.99 & ent_coef=0.005, and the UC-42 test file). Every Critical-scrutiny axis defused: diagnosis airtight (ep_rew_mean=−5 = −0.05×100, penalty-free via `crash or grounded`, grounded arms only post-takeoff); design provably non-farmable and continuous (slope 10/m below threshold; round-trip telescopes to (γ−1)ΣΦ ≤ 0; above-threshold F_gb=−0.005/step height-independent ⇒ no double-count; combined bound 162.48 < 200; climb_gamma==γ; gb bound 0.495 ≤ gate_bonus/3); w<0.9 seam bound exactly right (net-hold band slope 0.18−0.2w>0 ⇒ +0.08/m at w=0.5); AC7 enumeration + audit correct; racing_env.py untouched respects minimality. Risk 2 → endorse cfg-coupling (do NOT thread floor_epsilon into compute_reward), assert against source. Risk 4 → signed off as legitimate contract change, keep w=0.5. Four Minor directives (all folded in). Approved to proceed — no revision round needed.
