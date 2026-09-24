---
plan_for: use-cases/58-takeoff-reward-redesign.md
work_branch: feat/uc-58-takeoff-reward-redesign
team: drone-fly-uc-58
approved: 2026-09-24
---

# Approved implementation plan — UC-58 takeoff-oriented reward redesign

Challenger-approved (analyst v2, after one revision round). All claims verified against the tree.

## Root cause (confirmed line-by-line)
The −0.87 reward-vs-length suicide trap is driven by three compounding factors in the LIVE default reward:
1. **`time_penalty` (0.05/step)** — unconditional per-step existence cost; makes existence net-negative below h≈0.25 m (where the graded `airborne_bonus` 0.2·h_frac < 0.05).
2. **UC-50 progress hard-gate is LIVE** (`enable_altitude_decoupling` defaults `True`) — withholds the positive progress reward whenever the drone is below the gate-referenced altitude band, so a low floor-start policy moving toward a gate is penalized on top of the time bleed.
3. **No sustained positive altitude term** — `airborne_bonus` is dead-zoned (boolean-gated above floor_epsilon) and net-negative low; the three potential-based terms (climb/ground_break/altitude_hold) telescope to ≈0 net by construction, giving only transient nudges clawed back on descent.
Net: diving to the floor to stop the bleed is optimal → throttle decays 0.53→0.30 → −0.87.

## Approved solution (from-scratch, one coherent reward)
Replace the four overlapping altitude terms with ONE **sustained, level-based, saturating altitude reward** paid from h=0:
`altitude_reward(h) = altitude_weight · clamp(h,0,altitude_target)/altitude_target`, scaled by `per_step_scale` (UC-57 rate-invariance). Values: **altitude_weight=0.2, altitude_target=1.0 m**.

New `compute_reward` = progress (kept, telescoping) + gate_bonus/N + completion_bonus + altitude_reward − collision_penalty(genuine crash) − obstacle_penalty.

Why it satisfies the contract: level-based (not potential) ⇒ sustained signal (AC1/AC2 anti-suicide from positive reward, robustly — nothing saved by terminating since time_penalty is gone); paid from h=0 ⇒ takeoff gradient from the floor, no dead zone (AC3); saturating+level ⇒ non-farmable by oscillation, climb-to-target beats loiter-low (AC4); completion 200 > max altitude accrual 0.2×800=160 ⇒ hierarchy preserved (AC5).

**Retire (each documented, AC6/AC7):** `time_penalty`; `airborne_bonus`+boolean gate; `climb_weight/climb_target_height/climb_gamma` (eliminates the fragile γ-coupling entirely); `ground_break_weight/height`; the whole UC-50 group `enable_altitude_decoupling/altitude_hold_weight/altitude_band` + its progress hard-gate (a LIVE path); the UC-39 crash-cliff `set_collision_penalty` override + `CollisionCurriculumCallback`. Change `penalize_collision = crash or grounded` → `penalize_collision = crash` so a gentle post-takeoff grounded-rest is penalty-free (else a failed takeoff attempt pays −100 vs free do-nothing — re-creating the trap the crash-cliff curriculum papered over). **Keep** the UC-25 grounded/stuck early-termination detectors (termination bound, not reward; both cuts now penalty-free) and the UC-44/51 airborne-start SPAWN curriculum (separate `set_spawn_z` callback, reward-decoupled).

**Deferred (AC6 known limitation):** no explicit gate-diving guard — completion dominance makes dive-to-gate unable to beat completion; revisit in a future UC if the retrain shows horizontal-dive-and-sink.

## Files Affected
**Production (developer):**
- `src/drone_fly/env/reward.py` — rewrite `compute_reward` body+docstring; new keyword-only signature: `(*, dist_to_target_prev, dist_to_target_curr, event, collided, completed, cfg, num_gates=1, obstacle_contact=False, height_above_floor_curr=0.0, per_step_scale=1.0)`.
- `src/drone_fly/env/config.py` — RewardConfig: delete the retired fields, add `altitude_weight=0.2`, `altitude_target=1.0`; rewrite docstring/bound comments. Leave EarlyTerminationConfig intact.
- `src/drone_fly/env/racing_env.py` — `penalize_collision = crash`; drop retired kwargs + the `target_h_prev/curr` captures from the call site; remove the `set_collision_penalty` method + `_collision_penalty_override` path (keep `_took_off` — still arms the grounded detector).
- `src/drone_fly/train/collision_curriculum.py` — retire; remove its registration at loop.py:564–567.
- `src/drone_fly/train/config.py` — remove collision-curriculum fields + the `climb_gamma==gamma` note.
- `src/drone_fly/config.py` — add reward `_Spec`s (altitude_weight≥0, altitude_target>0) to train & evaluate; drop collision-curriculum specs.
- `src/drone_fly/cli/__init__.py` — add `_apply_reward(env_config, cfg)` (mirror `_apply_control_rate`, fold via `dataclasses.replace`); wire in `_run_train` AND the evaluate run path; drop collision-curriculum overrides.
- `configs/train/example.yaml`, `configs/evaluate/example.yaml` — expose `altitude_weight`/`altitude_target`; delete collision-curriculum block.
- `README.md` — rewrite the `Action|Reward|Condition` table (~L393) + surrounding UC-37/39/42/43/50 prose + the L585 "desired outcome" table.

**Test (QA):** `tests/test_reward.py` (core rewrite), `tests/test_reward_docs.py` (table expectations — enforcing doc-gate), `tests/test_config.py` (drop collision-curriculum + `climb_gamma==gamma` specs; add reward-key validation), `tests/test_cli.py` (drop collision overrides; add `_apply_reward` round-trip+validation), `tests/test_uc51_curriculum_schedule.py` (drop deleted keys), `tests/test_collision_curriculum.py` (retire), `tests/test_uc42/43/50/41_*` (retire/rewrite as UC-58 invariant tests), `tests/test_env_config.py`, `tests/test_env_contract.py`, `tests/test_uc57_control_rate.py` (per_step_scale still scales the altitude term), `tests/test_airborne_curriculum.py` (verify no reward coupling). NEW hermetic tests: AC1 (both descent paths — penalty-free grounded-rest AND crash — multi-horizon, corr≥0), AC2, AC3 (gradient from h=0), AC4 (loiter/oscillate < climb, saturation), AC5 (160<200 + completion>hover), AC8 (YAML round-trip + validation, train & evaluate). All GPU/pybullet/training-free (AC10).

## Risks
1. Removing `time_penalty` drops the explicit "fastest completion" incentive (not in ACs; near-term target is takeoff+hold; anti-dawdle from stuck detector + dominant completion). Future speed incentive → completion-time bonus, never a per-step cost.
2. γ-coupling fully resolved (no potential shaping remains).
3. Large-N caveat: budget >800 ⇒ 0.2×budget can exceed 200; carried forward, mitigated by stuck detector + forgone bonuses.
4. Large test/README churn is expected (default-on reward change, checkpoints invalidated); behavioral takeoff verdict deferred to owner's fresh GPU retrain (AC10).

**Challenger's carry-forward for developer/QA (not blocking):** keep the AC1/AC3/AC5 tests STRUCTURAL (orderings, monotonicity, correlation sign) — NOT absolute-margin — because reward VecNormalize is on and the behavioral verdict is deferred to the fresh GPU retrain.

## Challenger verdict
APPROVE (v2, after one revision round). Verified the analyst read the code correctly (root cause = time_penalty net-negative existence + live UC-50 progress-gate withholding near-floor progress + no sustained positive term ⇒ −0.87 reward-vs-length corr); independently checked that UC-50 is default-ON (not OFF as v1 claimed — fixed in v2), the collision-curriculum retirement doesn't touch the separate airborne/spawn curriculum, and the full test-retirement surface. Anti-suicide/AC5 invariants are hermetic; behavioral takeoff verdict deferred to the owner's fresh GPU retrain (AC10).
