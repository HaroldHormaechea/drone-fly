---
plan_for: use-cases/40-diagnose-k0-actor-no-commit.md
work_branch: feat/uc-40-diagnose-k0-actor-no-commit
team: drone-fly-uc-40
approved: 2026-09-21
---

# UC-40 — Approved Implementation Plan (analyst↔challenger, approved round 2)

Target: /workspace/drone-fly-uc-40-diagnose-k0-actor-no-commit · Use case: use-cases/40-diagnose-k0-actor-no-commit.md · Profiles: none (`profiles: []`).

## Analysis

A 1M-step run fires health_callback's K0 "likely-undersized connectome slice" signature (WARNING at update 11 → CRITICAL by update 20), concluding the slice lacks capacity. The verdict is wrong; the true cause is diagnosed below with evidence, confirming/ruling out each of the three named suspects.

**Suspect 3 — health_callback logic: CONFIRMED. This is why the K0 verdict mis-fires.**
`src/drone_fly/train/health.py:255-256`, `_rule_under_capacity_signature`, computes:
`ev_ok = _mean(explained_variance) >= ev_healthy (0.5) OR _rel_change(explained_variance) > 0`.
The OR trend-branch forces `ev_ok=True` whenever EV merely trends upward, regardless of absolute value. At the observed abs EV ≈0.03 rising, `_rel_change = (0.03−0.01)/0.01 = 2.0 > 0` ⇒ `ev_ok=True` despite mean 0.02 ≪ 0.5 ⇒ the K0 signature fires and its message asserts "critic is healthy (explained_variance high/rising)" (health.py:276) — "high" is false at 0.03. A genuine threshold/normalization bug (AC2).

**Suspect 2 — flat reward gradient / no learning signal: CONFIRMED root cause of the pinned actor.**
- `ep_rew_mean ≈ −5` = `time_penalty 0.05` (env/config.py:557) × ep_len ~101. `ep_len 101` = `stuck_window 100` (env/config.py:818) + 1: the no-progress ("stuck") detector cuts every episode, and that cut is penalty-free (UC-38, racing_env.py:493-494), so the return is pure time penalty.
- `entropy 5.72` at `std 1.014` = the exact differential entropy of a 4-D diagonal Gaussian (action space `Box` dim 4, racing_env.py:116) at std≈1.014 → the actor is still at policy initialization; std/entropy have not moved.
- Mechanism: every floor-bound trajectory returns the same −5 → after PPO advantage normalization, advantages ≈ noise → the policy-gradient term vanishes → the actor's mean/std never update → the only surviving gradient is the small entropy bonus, which holds std at its high init value → exactly the observed "flat high std / flat high entropy / 0% success".
- **Quantified root-cause mechanism (why the drone never takes off):** hover requires **throttle ≈ 0.5** (`BASE_MAX_THRUST = 2·BASE_MASS·GRAVITY`, adapter/simple.py:46; `_WARMUP_ACTION = [0.5,0,0,0]` documented "exact hover: 0.5·19.62/1.0 = 9.81 = g", simple.py:58-60). But `net_arch=dict(pi=[])` (loop.py:64) makes the action mean a linear readout initialized ≈0 (ortho gain ~0.01, log_std=0 → std=1); throttle is an unbounded Gaussian mean clipped to [0,1] (adapter/base.py:86-98, not tanh-squashed). So the deterministic policy commands ~0 throttle and mean-0/std-1 exploration clipped to [0,1] averages ≈0.29 — both below the 0.5 hover point → net downward thrust → the drone never sustains takeoff → never reaches the altitude where the real, correctly-sized potential-based climb/airborne gradient applies (F ≈ 0.099/step at 5 cm > time_penalty 0.05) → flat returns → frozen actor, `success_rate` structurally pinned at 0.

**Suspect 1 — ent_coef / entropy schedule over-rewarding randomness: RULED OUT as primary; secondary amplifier only.**
`ent_coef = 0.01` (train/config.py:64, the UC-38 bump; wired at loop.py:451) contributes 0.01×5.72 ≈ 0.057 to the loss — small in absolute terms, not "over-rewarding" in isolation. It only *appears* dominant because suspect 2 has zeroed the competing policy gradient. It is a lever for the fix, not the cause. (SB3 logs `entropy_loss = −mean(entropy)` unscaled; the TUI `entropy = −entropy_loss`, tui/metrics.py:379-380, so "entropy 5.72" is the mean policy entropy, not an ent_coef artifact.)

**AC5 — the slice is NOT undersized.** The run recorded ~25,627 neurons ≫ `DEFAULT_CAPACITY_FLOOR = 3000` (health.py:154); `_assess_capacity` passes. The runtime K0 rule fired only via the EV mislabel (suspect 3) + genuine flat entropy (suspect 2). AC5's conditional ("if the evidence shows the slice genuinely is undersized") is therefore **not met** → no slice bump; `DEFAULT_CAPACITY_FLOOR` and `tests/test_capacity_guard.py` calibration stay untouched.

## Proposed Solution

**1. PRIMARY (root cause) — hover-bias the initial throttle mean.**
In `src/drone_fly/train/loop.py`, after `PPO(...)` is constructed (~line 441), set the throttle channel of `model.policy.action_net.bias` to the adapter's hover throttle (0.5, matching `_WARMUP_ACTION[0]`). Because throttle is an unbounded Gaussian mean clipped to [0,1] (not tanh-squashed), a bias of 0.5 makes the initial deterministic action hover and centers exploration on 0.5 instead of 0. The drone then floats, the existing potential-based climb/airborne gradient becomes reachable, returns vary across policies, PPO gets real advantage signal, and — paired with the ent_coef change — std can narrow onto a *useful* policy so `success_rate` can leave 0 (satisfies AC3's *both* clauses: entropy/std down AND success able to leave 0). This is the lever that addresses flat-reward-at-root; lowering ent_coef alone cannot (it would only narrow std onto noise). Policy-side only, at model-build time — no env/adapter/reward change, so the large test_adapter / test_env_contract / reward-doc suites are untouched. Accepted by the challenger as in-scope "training config."

Developer notes (challenger, non-blocking):
- Do not hardcode `0.5` — tie the bias to the adapter's hover constant (`_WARMUP_ACTION[0]` or a shared named constant) so it can't silently desync if base dynamics change.
- Treat the "initial mean throttle ≈0.5" print as a **real gate**, not decoration: the deterministic mean is `bias[0] + W·features`; with ortho gain ~0.01 the `W·features` term is small but nonzero and depends on `ConnectomeFeaturesExtractor` normalization — empirically confirm the post-bias mean lands ≈0.5.
- Add a one-line guard that `squash_output` is False for this policy, keeping the "unbounded mean" assumption honest against future SB3/policy changes.

**2. COMPLEMENTARY SECONDARY — `ent_coef` → a lowered STRICTLY-POSITIVE constant.**
In `src/drone_fly/train/config.py`, lower `TrainConfig.ent_coef` to a smaller strictly-positive value, wired unchanged at loop.py:451. This must NOT be a constructor schedule: SB3 stores `ent_coef` raw (`on_policy_algorithm.py:106`) and uses it directly in the loss (`ppo.py:256`, `self.ent_coef * entropy_loss`); only `learning_rate`/`clip_range` go through `get_schedule_fn`, so a callable would break. If a true decay is ever wanted, implement it as a rollout callback that mutates `model.ent_coef` between updates (re-read each `train()`), never a constructor callable. The value must stay **strictly positive** (AC8 — config.py:57 warns that at `ent_coef=0` std collapses before takeoff is discovered; UC-38 fixed exactly that). Once the hover bias supplies a gradient, a smaller-but-positive ent_coef lets std commit without re-introducing the UC-38 collapse.

**3. CALLBACK EV FIX (AC2).**
In `_rule_under_capacity_signature` (health.py:255-256) replace `ev_ok = _mean(explained_variance) >= ev_healthy OR _rel_change(explained_variance) > 0` with `ev_ok = _mean(explained_variance) >= ev_healthy` — drop the pure-trend OR-branch so the K0 "healthy critic" gate requires genuine absolute health. Correct the reason `text` (health.py:276) so it no longer claims "high/rising" unconditionally (state absolute health, e.g. "critic is genuinely healthy — explained_variance ≥ threshold"). Everything else in the rule (flat-high entropy + ~0 success + suppression clauses) is unchanged. No new `ev_min_for_rising` threshold (dropping the branch is cleaner and keeps every existing K0 test green).

**4. OPTIONAL, EVIDENCE-GATED REWARD FALLBACK** (apply only if the smoke-train shows hover-bias + ent_coef still don't sustain ascent).
Strengthen the low-altitude climb potential (`RewardConfig.climb_weight` / `climb_target_height` in `src/drone_fly/env/config.py`) and/or relax the *training-time* no-progress `stuck_window` so exploration has more time to chain sustained thrust — **preserving all UC-37/39 invariants**: still potential-based (F = γΦ′ − Φ), telescoping/non-farmable, ≈0 on the floor, capped at target (no ceiling-seeking), per-episode total < `completion_bonus` (100), and `climb_gamma == training gamma`; and respecting the `stuck_window ≥ ~65` golden-fixture floor (change a training value, not the fixture default). Any `RewardConfig` value change auto-flows to the README reward table via `tests/test_reward_docs.py`, so README §"Reward table" must be updated in the same change to keep `test_reward_docs.py` / `test_bootstrap_docs.py` / `test_viz_contract.py` green (AC6).

**5. VALIDATION (AC4).**
Baseline-vs-fixed CPU smoke-train (numpy backend, a few hundred–thousand steps): assert `train/std` (and derived entropy) trend **DOWN** under the fix and are ~flat/high in baseline, AND an airborne-rate / success signal leaves 0 (or, at minimum, the drone gets airborne — direct evidence the gradient is now live). Also print the post-bias initial mean throttle as a real gate (≈0.5). The full ~12h GPU retrain remains the user's to run on Windows and is NOT a gate. Full suite green: `uv run --extra dev pytest` (AC7).

## Files Affected

**Production code — for the developer** (`paths.production = src/drone_fly/**`):
- `src/drone_fly/train/loop.py` — [new] hover-bias `model.policy.action_net.bias` throttle channel after PPO build; add the `squash_output is False` guard; ent_coef stays wired as a constant at :451.
- `src/drone_fly/train/config.py` — `TrainConfig.ent_coef` lowered strictly-positive constant.
- `src/drone_fly/train/health.py` — `_rule_under_capacity_signature` EV criterion (drop trend-branch) + message-text fix.
- (fallback only) `src/drone_fly/env/config.py` — `RewardConfig.climb_weight`/`climb_target_height` bump.
- (fallback only) `README.md` — reward-table sync.

**Test code — for QA** (`paths.test = tests/**`):
- `tests/test_training_health.py` — add K0 boundary tests: EV (0.01,0.02,0.03) rising must NOT fire; EV ≥0.5 fires; boundary at `ev_healthy`. Existing K0 tests (EV 0.80,0.82,0.81) stay green; keep the `"slice" in reason.text.lower()` assertion (line ~97), adjusting only if the text wording changes.
- `tests/test_config.py:413-420` — update the exact-value assertion to the new positive ent_coef default; **keep** the `ent_coef > 0.0` assertion (AC8).
- `tests/test_policy.py` / `tests/test_sb3_compat.py` — assert the `action_net` throttle bias initializes to hover; verify no existing init assumption breaks.
- `tests/test_health_callback.py` — verify no EV-value assumption breaks under the criterion change.
- New smoke-train validation (AC4): baseline-vs-fixed `train/std` delta + airborne/success-off-0.
- (fallback only) `tests/test_reward_docs.py` — auto-syncs to new RewardConfig values; confirm README table updated.

## Risks & Considerations
- **Hover bias only sets the START point** — training can move it; it touches no reward/env semantics, so no UC-37/39 invariant is affected. Resume (`PPO.load`) rebuilds from checkpoint, so the bias only seeds fresh runs (the user's retrain is fresh) — acceptable.
- **ent_coef floor > 0** guards the UC-38 collapse (AC8); lowering it too far risks the opposite failure that `_rule_premature_entropy_collapse` detects — validate the chosen value with the smoke-train.
- **Callback fix must not suppress genuine K0** — abs EV ≥0.5 must still fire; covered by keeping the absolute `ev_healthy` gate and the new boundary tests. If the reason text changes, update the asserted text.
- **UC-37/39 protected work** — any fallback reward change must preserve the potential-based/non-farmable invariants and the anti-suicide tests (extend, don't append naively) and keep the README reward-table doc contracts green.
- **Smoke-train limits** — CPU/no-GPU sandbox can only show entropy/std *responding* (and the drone getting airborne / success leaving 0), not task mastery; the full 12h GPU retrain stays the user's, not a gate.
- **AC5 conditional not met** — deliberately no slice bump; the 25,627-neuron slice ≫ the 3000-param capacity floor rules out under-capacity as the real cause.

## Challenger's final verdict
**APPROVE (round 2).** Diagnosis proven at file:line for all three suspects; hover-bias accepted as the in-scope primary root-cause fix; EV callback fix satisfies AC2; slice bump correctly declined (AC5 conditional not met); reward fallback is evidence-gated and preserves UC-37/39 + doc-contract invariants; validation is lab-scoped per AC4.
