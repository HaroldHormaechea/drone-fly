---
plan_for: use-cases/90-gate-approach-altitude-tracking.md
work_branch: feat/uc-90-gate-approach-altitude-tracking
team: drone-fly-uc-90
approved: 2026-10-19
---

# UC-90 — gate-approach altitude-tracking reward — APPROVED PLAN

Challenger approved first round (verdict: Approve, no Critical/Major; all code-level claims verified
against the worktree). This is the fixed spec for the developer/QA.

## Task
Add a gate-approach altitude-tracking reward to `src/drone_fly/sim/course.py` so the policy settles its
altitude to the **current target gate's** z on approach (live, the UC-89 deploy policy flies a median
+11 m OVER gate 0 → 0/33). Must generalize to all gates, be proximity-gated, be a configurable
`CourseConfig` field, **default 0 ⇒ byte-identical**, and not double-count progress/clearance/floor or
fight the grounded launch.

## File to modify (single production file)
`src/drone_fly/sim/course.py`

### 1. Two new `CourseConfig` fields (add after the UC-89 block, ~L206)
- `gate_alt_weight: float = 0.0` — weight on the vertical-alignment penalty. Default 0 ⇒ term skipped ⇒ byte-identical.
- `gate_alt_radius_m: float = 3.0` — horizontal proximity window in **sim world units** (same units as gate centers / progress distance / PASS_SLACK / ARENA — NOT the metre-denominated FLOOR constants; `_m` mirrors `spawn_distance_m`'s convention). Term fades linearly from full at the gate to 0 at this radius. Default is a starting point; the retrain tunes weight+radius.
- **Docstring line (required, per challenger):** note AC4 (gate-z observation-noise DR) is intentionally deferred — the existing `gate_pos_noise_m` DR path already perturbs the observed gate rel in all 3 axes incl. z. Record it as a documented decision, not a silent omission. Also state `gate_alt_radius_m` is sim world units.

### 2. Validation in `__post_init__` (~L238)
Require `gate_alt_weight >= 0` and `gate_alt_radius_m > 0` (raise `ValueError` on violation, matching the existing style).

### 3. Reward block in `step_command` — insert right after the `cfg.floor` block (after L1190), before the UC-87 penalty block
```
if cfg.gate_alt_weight > 0:
    gp = self._target_pos()                              # post-advance current target gate
    h = (gp[:, :2] - self.pos[:, :2]).norm(dim=-1)       # horizontal distance to the gate
    prox = (1.0 - h / cfg.gate_alt_radius_m).clamp(0.0, 1.0)
    zerr = (self.pos[:, 2] - gp[:, 2]).abs()             # symmetric vertical error
    reward = reward - cfg.gate_alt_weight * prox * zerr * rscale
```
(Pseudocode — developer writes idiomatic torch matching the file.) Key points, all challenger-verified:
- `_target_pos()` is POST gate-advance (advance is L1147–1150, block is after L1190) ⇒ keys on the gate now being approached, same `self.tgt` as progress/pass ⇒ generalizes to all gates.
- `_apply_gstart_rot` (L650) mutates only x/y; z passes through ⇒ the gate-z term is invariant to the UC-89 scene rotation.
- `rscale = dt/reward_reference_dt_s` (L1113) ⇒ rate-scaled like the other dense terms; folds into `ep_return` at L1243.
- **Stateless** (no potential/`prev_` state) + `if gate_alt_weight > 0` guard ⇒ no reset-state change, no RNG drawn when off ⇒ default-off byte-identical.
- Symmetric penalty: above OR below the gate penalized equally (AC2 "vertical-alignment specific").
- Linear in `zerr` — **do NOT clamp zerr** (challenger: the +11 m overshoot is exactly the large-zerr case we want full descent gradient on; proximity gating + modest retrain weight bound it).

### 4. Short code comment documenting composition (AC2, no double-counting)
Progress rewards 3D distance reduction across all axes over the whole course; this term adds an independent, additive, z-only, proximity-gated alignment pressure that bites only near the gate. Clearance = obstacle-surface distance; floor = height above the kill plane (absolute, not gate-relative). None overlap a gate-relative z-alignment term; it is a separate summand like gap/floor.

### Not doing (scoped out, documented)
- **No new `EPISODE_TAIL` column** — captured in `ep_return` like progress/clearance/floor; a new log column would risk the log's default-off byte-identity for no AC value. AC5 "recorded in run config" is satisfied by `asdict` serialization (config.py L174) + `from_dict` passthrough (`__dataclass_fields__`, L238) — automatic for the new fields.
- **AC4 (gate-z-noise DR) deferred** — optional/secondary; existing `gate_pos_noise_m` path (`_update_input_dr` L1325+) already covers observed-gate z DR. Documented in the field docstring (item 1).

## Test file (new, CPU-hermetic) — for QA
`tests/test_uc90_gate_altitude.py`, mirroring `tests/test_reward_penalties.py`'s "two same-seed envs, reward delta isolates the term" pattern (`meteor75()` from `tests.sim_support`, `no_obs=True`, small pool, fixed timing):
1. **Default-off byte-identity** — `gate_alt_weight=0` env vs a baseline with the field unset: reward, `ep_return`, and `extra()` identical across a multi-step rollout (same seed).
2. **Pure penalty / maximal-at-alignment** — two same-seed envs differing only in `gate_alt_weight`; per-step reward delta == `−w·prox·zerr·rscale` computed from `_target_pos()`/`pos`; delta→0 as zerr→0, grows with zerr.
3. **Proximity gating** — ≈0 contribution when `h ≥ radius`; active when `h < radius`.
4. **Keys on current target gate** — after a gate pass (`tgt` increments), the term tracks the NEW gate's z.
5. **Grounded-launch non-interference** (challenger-requested regression guard) — at a grounded spawn with `spawn_distance_m` set and `gate_alt_radius_m ≲ spawn_distance LOWER bound`, the term contribution is ≈0 (prox≈0), so it does not pull altitude off the pad.
6. **Config** — `from_dict` passthrough of both fields; validation rejects `gate_alt_radius_m <= 0` and negative weight.

## Files Affected
- **Production (developer):** `src/drone_fly/sim/course.py` — 2 `CourseConfig` fields + validation + reward block + docstrings/comment.
- **Test (qa):** new `tests/test_uc90_gate_altitude.py` (6 cases above).

## Notes for the orchestrator (out of code scope, AC7 retrain)
- Constraint to honor when choosing config: `gate_alt_radius_m ≲ spawn_distance_m LOWER bound` (spawn distance is a range) so prox≈0 at launch across the spawn distribution.
- Optional de-risk lever: anneal the altitude weight mild→target like UC-87's action-rate λ so completion establishes before altitude pressure (challenger suggestion; not a code requirement).
- AC3 (no regression) + AC7 (settling checkpoint, live 768 re-fly) are the retrain's job; the code only owes default-off byte-identity + tunability + tests, all covered above.

## Challenger verdict
**APPROVE** (no Critical/Major, no revision rounds). Verified: gstart rotation leaves z untouched (term rotation-invariant); insertion post gate-advance ⇒ keys on the approached gate, generalizes; stateless + `if weight>0` guard ⇒ default-off byte-identical (reward/ep_return/extra), no new RNG; `from_dict`/`asdict` passthrough automatic; existing `gate_pos_noise_m` justifies the AC4 defer. Resolved for the developer: AC-literal penalty form (not positive-bounded); zerr unclamped; `gate_alt_radius_m` name kept; AC4 deferred (documented). Requested the extra grounded-spawn≈0 CPU test.
