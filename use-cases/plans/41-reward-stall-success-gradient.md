---
plan_for: use-cases/41-reward-stall-success-gradient.md
work_branch: feat/uc-41-reward-stall-success-gradient
team: drone-fly-uc-41
approved: 2026-09-21
---

# UC-41 — Final Approved Implementation Plan (analyst↔challenger, approved round 3)

**Use case:** `use-cases/41-reward-stall-success-gradient.md`
**TARGET_DIR / WORKDIR:** `/workspace/drone-fly-uc-41-reward-stall-success-gradient` (git worktree on `feat/uc-41-reward-stall-success-gradient`)
**Brief contract:** profiles=[] (none active); paths.production=`src/drone_fly/**`; paths.test=`tests/**`; build.test=`uv run --extra dev pytest`, lint=`uv run ruff check .`, format=`uv run ruff format .`.

---

## Analysis — problem summary + relevant context

**Observed symptom (fresh post-UC-40 run).** At update ~13/81 (16%, ~159k/1M steps): `ep_rew_mean` glued to ≈ −5, `ep_len` pinned at 101, `success_rate` 0%, `health_callback` fires "Reward stalled". UC-40 successfully UNFROZE the actor (entropy/std now trend down, K0 verdict gone), but the now-committing actor is converging to a do-little / 0-success local optimum.

**Why the return is EXACTLY the time-penalty floor.** Step reward (`src/drone_fly/env/reward.py:100-116`) = `−time_penalty(0.05)` + `progress_weight·(dist_prev−dist_curr)` + (airborne? `+airborne_bonus 0.1`) + climb potential `F=γΦ′−Φ` + gate/completion bonuses. A return of −0.05×101 EXACTLY means every non-time term nets ~0: the drone **never leaves the floor band** (airborne flag `racing_env.py:384`: `z > floor_z + floor_epsilon` — never set), the climb potential telescopes to ≈0, and progress nets ≈0. The episode ends via the **no-progress "stuck" cut** at `stuck_window=100` (`racing_env.py:520-523`; ep_len 101 = 1 + 100), which is **penalty-free** (`racing_env.py:571`: `penalize_collision = crash or early_termination=="grounded"` — "stuck" is neither). So the return is pure accumulated time penalty ⇒ a do-nothing local optimum.

**Root mechanism = a crash-cliff ASYMMETRY, not an absent gradient.**
- Do nothing → penalty-free stuck cut → ≈ −5.
- Attempt takeoff and fail (lift, then drop back and rest on the floor) → the drone is `_took_off=True` (`racing_env.py:385`); the **grounded** detector arms only post-takeoff (`racing_env.py:527-529`) and a grounded cut **PAYS the collision penalty** (`racing_env.py:571`). Under the UC-39 collision-penalty curriculum (`train/collision_curriculum.py:35-57`, warmup_fraction 0.5), that penalty at 16% training ≈ 10 + 90×(0.16/0.5) ≈ **−38.8** (and ≥10 even at the curriculum start). Meanwhile the climb term telescopes back to ≈0 on the descent (round-trip ≈0, `test_reward.py` round-trip test) and airborne credit is only +0.05 net/step.

So a failed takeoff nets far below do-nothing's −5. PPO assigns the "throttle-up" actions strongly negative advantage vs the −5 baseline, and with `ent_coef=0.001` (`train/config.py`, set by UC-40) the action std commits onto the do-nothing policy before exploration can chain takeoff→climb→sustained-flight→progress. The gradient toward a *successful* sustained flight is real but small and only pays after many consecutive correct steps, whereas the penalty for a failed attempt is immediate and large. **That asymmetry — not the absence of a gradient — is the stall.**

Quantified: a ~35-step failed-takeoff nets ≈ +airborne(~1.5) + climb(~0 round-trip) − time(1.75) − penalty = **−0.25 − penalty**, vs do-nothing −5.05. Takeoff beats do-nothing iff effective penalty < ~4.8. A genuinely airborne-flailing episode (stuck cut, penalty-free) already beats −5 once up. So the ONLY thing punishing takeoff is the grounded/genuine-crash penalty, governed by the curriculum.

---

## Three levers — each confirmed / ruled out with evidence

**Lever 1 — reward gradient (CONFIRMED, primary).** The pull toward *sustained* flight is far too weak relative to the downside of a failed attempt (grounded-cut collision penalty ≥10 even at curriculum start, vs whole-episode climb bound only 1.98 and airborne +0.05/step). The invariant-safe minimal knob is the crash-cliff itself: UC-39's collision-penalty curriculum already exists to relieve this, but its ramp is too steep — the effective penalty leaves the safe band almost immediately. **Fix = reshape the curriculum ramp** (never touches `RewardConfig`/telescoping, so the protected non-farmability / per-episode<completion bounds are untouched).

**Lever 2 — ent_coef=0.001 (CONFIRMED, companion guard for AC4).** The run shows std trending DOWN with success at 0% — the premature-collapse precondition (`health.py:_rule_premature_entropy_collapse`: entropy dropping >30% AND success≈0 AND reward not climbing). `ent_coef` is a fixed SB3 coefficient (`loop.py:502`), no floor/schedule. It cannot create a success gradient — it only preserves exploration long enough for the (unblocked) gradient to take. Bumped modestly to satisfy AC4 (fix must demonstrably NOT cause premature collapse; higher ent_coef monotonically REDUCES collapse risk).

**Lever 3 — stuck_window=100 (RULED OUT, with hard evidence — stakeholder hypothesis explicitly tested).** Hypothesis: the no-progress cut ends episodes before the drone has runway to fly / vertical takeoff doesn't count as progress.
1. **What counts as progress:** `_dist_to_target` (`racing_env.py:252-254`) = `np.linalg.norm(current_target − position)` — full **3D** Euclidean distance. `current_target` (`geometry.py current_target`) = `gates[gates_passed]` — the **next REQUIRED gate in order**, NOT nearest (so the "nearest-vs-next / skip the sequence" concern is already handled correctly). The stuck detector resets when `dist_curr < best_dist − progress_epsilon(0.01)` (`racing_env.py:520-523`).
2. **Geometry (default course, floor_start ON):** start is overridden to `(0,0,0)` (`racing_env.py:281-285`); first required gate `gates[0]` = `(2.5, 0.0, 1.0)` (`config.py:118`). The gate is **up AND forward.** 3D distance from floor-start = √(2.5²+1.0²) = **2.693 m**; a PURE VERTICAL climb to z=1.0 shrinks the z-component → distance drops to √6.25 = 2.5 m (~0.371 m per m of climb near the floor). Any climb gaining >~1 cm net toward the gate height exceeds `progress_epsilon` and **resets the counter**. Forward (+x) motion also always reduces distance. ⇒ **Takeoff DOES register as progress.** The hypothesis would only bite if the gate sat at the start's height (zero z-component) — it does not.
3. **Reset vs fixed budget:** `self._stuck_counter = 0` on any progress (`racing_env.py:522`) ⇒ a rolling no-progress GRACE PERIOD, not a fixed ceiling.
4. **Wall-clock:** dt=0.05 s / 20 Hz ⇒ 100 steps = **5.0 s** of no-progress grace; a TWR-2 drone (50% throttle hovers) takes off in <1 s ⇒ ample runway.
**Conclusion:** the step-100 cut is a **symptom** of the do-nothing policy (zero net progress because it never leaves the floor), not the cause. Relaxing stuck_window would only let a do-nothing drone sit for 10 s instead of 5 s — it creates no gradient toward takeoff and cannot unstick the optimum. **No change** (also preserves the ≥~65 golden-fixture floor). **Fallback caveat:** if a later run shows the drone DOES take off but gets cut mid-climb before chaining to forward progress, the TRAINING-time stuck_window relaxation becomes the fallback lever — current evidence shows it isn't needed.

**Phased-curriculum candidate (stakeholder design input) — NOT ADOPTED.** Proposal: Phase 1 disable/lengthen the no-progress window for takeoff runway; Phase 2 switch objective to "approach next waypoint" — or a single piecewise composite potential Φ (altitude when low, waypoint-distance when high).
1. **The ordering is ALREADY realized, continuously, as a SUM of potentials** — no regime switch needed. The **progress term** `progress_weight·(dist_prev−dist_curr)` is itself a telescoping potential Φ_progress = −progress_weight·(3D dist to next-required gate); because the gate is up+forward it already rewards altitude-toward-gate AND forward motion. The **UC-39 climb potential** adds dense 0→1 m takeoff shaping. Φ_total = Φ_climb + Φ_progress is itself a potential ⇒ still policy-invariant / telescoping / non-farmable. A smooth sum is strictly safer than a piecewise switch at the airborne boundary (no value-function discontinuity, no new anti-farming risk).
2. **It does NOT subsume lever 1.** The problem is not a mis-encoded ordering; it's that the correctly-encoded upside is bounded (per-episode shaping 81.98 < completion 100, `test_reward.py:341-366`) and small, while the failed-takeoff downside dominates during exploration. No admissible potential can offset a −10..−39 penalty without breaking the bound/non-farming invariants. The crash-cliff fix removes the barrier directly; a richer potential neither removes it nor is needed.
3. **Minimality.** A phased curriculum / piecewise Φ is more complex and risks the UC-37/38/39 contract; the evidence shows it is unnecessary. Documented as a follow-up only if a future run proves staging is genuinely needed.

---

## Proposed Solution

Reshape the collision-penalty curriculum from a from-t=0 linear ramp to **hold → ramp → hold-at-end**, so the effective penalty stays in the invariant-safe band (≤ ~climb-reward bound ≈ 2–5) through the entire fly-learning phase, then rises to full strength for late-training precision. No `RewardConfig` value changes (the curriculum overrides collision penalty at reward time only; the env default 100 is never mutated).

**Config changes — `src/drone_fly/train/config.py` (`TrainConfig`):**
- `collision_penalty_start: 10.0 → 2.0` — the HELD value during fly-learning (token crash signal, well below the ~4.8 cliff).
- **NEW** `collision_curriculum_hold_fraction: float = 0.4` — hold at `start` for the first 40% of training before the ramp begins.
- `collision_curriculum_warmup_fraction: 0.5` — UNCHANGED value; now semantically the ramp DURATION after the hold (ramp runs 40%→90%).
- `collision_penalty_end: 100.0` — UNCHANGED (AC6 + late-training precision preserved).
- `ent_coef: 0.001 → 0.005` — companion exploration guard (strictly between UC-40's 0.001 and UC-38's 0.01; monotone-safe for AC4).
- Add a validation guard (e.g. `__post_init__`): `0 ≤ hold_fraction` and `hold_fraction + warmup_fraction ≤ 1`, so a bad split can't push the ramp end past `total_timesteps`.

**Schedule change — `src/drone_fly/train/collision_curriculum.py` (`collision_penalty_at`):**
New shape (keep the existing `warmup_steps <= 0 → end` degenerate guard FIRST so the no-curriculum edge stays byte-identical):
```
hold_steps = hold_fraction · total ; warmup_steps = warmup_fraction · total
if warmup_steps <= 0: return end          # degenerate: no curriculum (unchanged)
if t <= hold_steps:   return start        # hold phase
if t >= hold_steps + warmup_steps: return end   # held at end
else: frac = (t − hold_steps)/warmup_steps ; return start + (end−start)·frac
```
Still a pure function of `num_timesteps` ⇒ resume-correct; monotonic non-decreasing (flat-low → up → flat-high). The callback (`CollisionCurriculumCallback`) is unchanged.

**Effective-penalty curve (total=1M, hold=0.4, warmup=0.5):**

| step | % train | effective cp | (old start=10) |
|---|---|---|---|
| 0 | 0% | **2.0** | 10 |
| 14.3k | 1.4% | **2.0** | 12.5 |
| **159k** | **16% (observed stall)** | **2.0** ✓ | 38.6 |
| 400k | 40% (hold ends) | **2.0** | 100 |
| 650k | 65% | 51.0 | 100 |
| 900k | 90% (ramp ends) | **100.0** | 100 |
| 1M | 100% | 100.0 | 100 |

The penalty stays at 2.0 (below the ~4.8 cliff) through the entire first 40% of training (~400k steps ≈ 4000 episodes at n_envs=1: generous runway vs the observed no-flight-at-16%), then ramps to full 100 by 90% for precision. At penalty=2 a ~35-step failed-takeoff nets ≈ −2.25 > do-nothing −5.05 ⇒ takeoff strictly wins across realistic attempt lengths (`0.1·A − 0.05·L − 2 > −5.05`). AC6 (floor-shortcut < completion) holds at EVERY t (crash always < completion 100 and never earns it).

**Prose/doc updates (production):** `collision_curriculum.py` module + `collision_penalty_at` docstrings (~lines 12-14, 38-42) and the `loop.py:544-549` comment currently describe a from-t=0 linear ramp — update to the hold-then-ramp shape. `README.md` curriculum PROSE updated to the hold-then-ramp shape with final numbers (2 held → ramp 40%→90% → 100). The README reward TABLE is NOT touched (no RewardConfig value changed).

---

## Files Affected

**Production code (developer) — under `src/drone_fly/**`:**
- `src/drone_fly/train/config.py` — the field changes above + docstrings + the hold/warmup split guard.
- `src/drone_fly/train/collision_curriculum.py` — `collision_penalty_at` reshape + module/function docstring prose.
- `src/drone_fly/train/loop.py` — update the stale from-t=0-ramp comment (~544-549).
- `README.md` — curriculum prose → hold-then-ramp final numbers (reward table untouched).

**Test code (qa) — under `tests/**`:**
- `tests/test_collision_curriculum.py`:
  - UPDATE `test_schedule_reaches_and_holds_end_value_after_warmup` — end now reached at `hold_steps + warmup`, not `warmup`.
  - UPDATE `test_schedule_midpoint_is_linear_interpolation` — midpoint is of the RAMP, at `hold_steps + warmup/2` (expected 51 with default endpoints).
  - UPDATE `test_schedule_is_stateless_and_resume_correct` — its t=300 (total=1000, hold=0.4 ⇒ hold_steps=400) now falls in the HOLD ⇒ returns `start`; fix the expectation.
  - UNCHANGED / stay green: `test_schedule_starts_at_start_value_at_t0`, `test_schedule_is_monotonic_non_decreasing`, `test_schedule_degenerate_no_warmup_returns_end_value` (guard kept first), and the callback wrapper-stack test.
  - NEW hold-phase test: `collision_penalty_at(t) == start` for all t in `[0, hold_steps]`, specifically at the 16%/159k timestep.
- `tests/test_reward.py:516` — anti-suicide worst-case pin `cp_low == 10.0` → `2.0`; the inequality (hover beats takeoff-then-crash) is cp-independent (crash-step give-back ≈ −w·target = −2.0 dominates), verified — stays green.
- `tests/test_config.py:422` — `ent_coef == 0.001` → `0.005` (keep `> 0.0`); ADD `collision_curriculum_hold_fraction` default (0.4) + range coverage.
- **NEW deterministic AC3 operating-point test:** assert `failed_takeoff_return > do_nothing_stuck_cut_return` computed via `compute_reward` with `collision_penalty = collision_penalty_at(t, cfg)` **parametrized over t ∈ {0, 159k (16%), 400k (hold-end)}** — the fly-learning window. Because cp=2 across that whole window the inequality holds with margin; the parametrization fails loudly on any future schedule regression that shortens the hold. Document that BEYOND hold-end the property intentionally lapses (precision phase; drone assumed flying).
- **AC4 guard:** assert `TrainConfig().ent_coef ≥` prior; short CPU smoke-train assertion that `train/std` stays finite and does not crash toward 0 while success is 0 (num_timesteps≈0 ⇒ within the hold; this is a wires-together/finite/std-not-collapsing proof only, NOT the AC3 evidence).

---

## Risks & Considerations

- **Diagnosis→fix linkage at the operating point (the challenger's Critical, resolved).** A `start`-only change is inert: the old linear ramp re-crosses the ~4.8 cliff by ~1.4% of training, so at 16% the penalty would still be ~33. The hold-then-ramp is what keeps the penalty at 2 THROUGH the fly-learning phase. This is the crux — do not regress it to a start-only tweak.
- **Validation realism (AC1/AC3).** The 256-step CPU smoke-train and a t=0 evaluation both sit at num_timesteps≈0 (in the hold) and would go green even if the fix didn't work at 16% — a false positive. Hence the AC3 evidence MUST be the deterministic operating-point tests parametrized across the ramp, not the smoke-train.
- **AC6 precision tradeoff (quantified).** Precision penalty is DELAYED, not weakened: low penalty for 40%, then rising to full 100 held for the final ~10%. AC6 holds at every t. The 0.4/0.5 split is a documented tunable erring toward runway (the observed run needed >16% even at the higher cliff); the user's 12h GPU run can retune it.
- **Invariants (UC-37/38/39) — all preserved.** No `reward.py`/`RewardConfig` change ⇒ potential-based/telescoping F=γΦ′−Φ, non-farmable round-trip ≈0, ≈0-on-floor, capped at target, per-episode shaping 81.98 < completion 100, `climb_gamma == gamma`, penalty-free stuck-cut, and anti-suicide all intact. Reward table + doc-contract tests (`test_reward_docs`, `test_bootstrap_docs`, `test_viz_contract`) untouched. `stuck_window` unchanged ⇒ golden-fixture ≥~65 floor untouched.
- **Lever interaction / over-correction (pitfall).** Only two levers changed (curriculum shape + a modest ent_coef bump); lever 3 deliberately untouched; no reward magnitude inflation; no phased Φ. Minimal set the evidence supports.
- **Scope reach (AC8).** Lab-validation only (unit tests + short CPU smoke-train). The full ~12h GPU retrain remains the user's on Windows and is NOT a gate.
- **Non-blocking follow-ups (from the challenger, do them):** stale-prose cleanup in `collision_curriculum.py`/`loop.py`; the `hold_fraction` config guard + `test_config.py` coverage; README prose to the final numbers.

---

## Challenger's Final Verdict (round 3) — APPROVE

Revision 1 resolves both Criticals. Critical #1 (cliff inert at operating point) RESOLVED — hold-then-ramp holds the effective penalty at 2.0 (below the ~4.8 cliff) through the first 40% (~400k steps), covering the 16%/159k stall with margin, then ramps to 100 by 90%; re-derived the table, correct; monotonic and still a pure function of num_timesteps ⇒ resume-correct; degenerate guard kept first. Critical #2 (rigged validation) RESOLVED — parametrizing across t ∈ {0,159k,400k} under the ramped penalty + a pure-schedule pin exercises the real operating point and fails loudly on regression. Invariants intact (no RewardConfig change; anti-suicide pin 10→2 safe, cp-independent; end=100 preserves AC6). ent_coef→0.005 kept. Lever-3 rule-out accepted (3D-distance/next-required-gate/rolling-reset evidence convincing). Phased curriculum correctly not adopted (already the smooth Φ_climb+Φ_progress sum). Test-churn list verified accurate against the files; new field additive-safe. Success-gate alternative rejection accepted (statefulness would break resume-correctness + deterministic testing).
