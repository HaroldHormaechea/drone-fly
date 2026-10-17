---
plan_for: use-cases/81-combined-distillation-fix.md
work_branch: feat/uc-81-combined-distillation-fix
team: drone-fly-uc-81
approved: 2026-10-16
---

# UC-81 — Combined distillation fix — Approved implementation plan

Challenger APPROVED (round 2; AC2 Critical resolved via explicit `on_policy_beta_floor` flag). All claims grounded in real code; all 8 ACs covered.

## Analysis
- **Reward path:** `RewardConfig` frozen dataclass at `sim/multi_course.py:134`; reward assembled in `MultiCourseEnv.step` :892–907, dense block at :894. Floor already computed in-step: `fh = self.floor_height(pos)` (:874); `FLOOR_KILL_M=0.08` (pool.py:54); liftoff `grace` (:864) already suppresses the floor kill. Spawn air band `air_z_m=(0.8,1.5)`.
- **Byte-identity:** snapshots use `asdict` (config.py:152/:366); established pop idiom at config.py:371 (UC-77 N4) + `_DISTILL_UC77_DEFAULTS` loop (:649). Student `config.yaml` written by `StudentConfig.snapshot()` (:644) → `base.snapshot()` = `TeacherConfig.snapshot()` (:365), which emits `out["env"]["rewards"]`.
- **AC2 root cause (key finding):** `distill.py:1311–1316` — under `on_policy_relabel`, `act = s_mean + noise_` **ignores β**; β-mix only in `else`. UC-80 `onpolicy` (beta_end 0.0, pure student) and `beta` (beta_end 0.15, non-on-policy) are mutually exclusive by construction → "on-policy + β-floor" needs a minimal code change, not pure config.
- **AC3:** `ppo_finetune` (:1574) reads reward straight from `senv.step()` (:1641) → GAE, so any `env.rewards` term auto-flows into the PPO objective; no PPO code change.

## Proposed Solution

**AC1 — opt-in floor-clearance penalty (the one real new code).** New field `floor_clearance: float = 0.0` on `RewardConfig` + module const `SAFE_CLEARANCE_M = 0.5`. One-sided bounded dense penalty: `clearance = pos[:,2]-fh`; `deficit = (SAFE_CLEARANCE_M - clearance).clamp(min=0)/SAFE_CLEARANCE_M` ∈[0,1]; add `- rw.floor_clearance * deficit` into `r_dense` (:894, scaled by rscale) **gated by `~grace`**. No positive bonus ⇒ can't be hover-hacked; 0 above the band; bounded [−w,0]. **Chosen weight = 0.03**. Default 0.0 ⇒ byte-identical to UC-77/79/80. Byte-identity pop in `TeacherConfig.snapshot` (config.py:365, beside :371): `if self.env.rewards.floor_clearance == 0.0: out["env"]["rewards"].pop("floor_clearance", None)`.

**AC2 — explicit `on_policy_beta_floor: bool = False` flag** (new `DistillSection` field at config.py:530, added to `_DISTILL_UC77_DEFAULTS` :567 → popped at default). Rollout branch (distill.py:1311):
```
if d.on_policy_relabel and not d.on_policy_beta_floor:
    act = s_mean + noise_                                            # pure student — UC-80, byte-identical
elif d.on_policy_relabel:                                            # UC-81 combined: student-driven + β-floor
    act = torch.where(use_t.unsqueeze(-1), t_mean, s_mean + noise_); teacher_share += ...
else:
    act = torch.where(use_t.unsqueeze(-1), t_mean, s_mean + noise_); teacher_share += ...   # unchanged baseline
```
Preserves both locked UC-80 probe tests unchanged (`use_t`/`noise_` drawn before the branch ⇒ RNG stream unchanged).

**AC3 — combined+PPO.** Combined distill + `finetune.steps: 3.0e7` + `env.rewards.floor_clearance: 0.03`. No PPO code change (auto-flow via :1641).

**AC4–AC6 — 3 configs + driver + doc** (modeled on UC-80):
- `src/drone_fly/training/configs/student-liftoff-open-uc81-baseline.yaml` — extends `student-liftoff-open`, mlp-student, final_report, budget 1.8e7 (control).
- `…-uc81-combined.yaml` — `distill:{on_policy_relabel:true, on_policy_beta_floor:true, beta_start:0.15, beta_end:0.15, beta_over:1.0}`, budget 1.8e7 (constant 15% teacher floor; student drives ~85% from step 0).
- `…-uc81-combined-ppo.yaml` — combined distill + `finetune.steps:3.0e7` + `env.rewards.floor_clearance:0.03`; **total ≈1.8e7+3e7=4.8e7 stated explicitly** (equal-budget relaxed per AC4).
- `scripts/uc81_combined_fix.sh` — resumable back-to-back driver, `VARIANTS=(baseline combined combined-ppo)`, same filesystem-state / skip-if-eval-exists / resume contract as `scripts/uc80_distillation_fix.sh` (TEACHER/NOISE/DEVICE/STATE_ROOT; teacher `teacher_it1374`, noise model, track cache).
- `docs/experiments/uc81-combined-distillation-fix.md` — one comparison table (rows: teacher, baseline, combined, combined+ppo, + UC-77 / UC-79 teacher_full / UC-80 best; cols: held-out + Infinity macro_completion exact & noise + outcome breakdown, 32 ep seed 1234) + recommendation + AC8 pivot branch. **Must document the combined-ppo confound** (bundles extended PPO budget AND floor-clearance reward; a ≥5% result can't attribute cause — note follow-up to isolate) and **report the PPO reward curve** (hover-hack sanity check).

**AC7 — tests** (`tests/test_training_student_uc81.py`, CPU-hermetic, reuse the UC-80/UC-70 tiny-DAgger harness):
1. AC1 default-off byte-identity — assert all three: `"floor_clearance" not in snap["env"]["rewards"]`; `rebuilt.snapshot()==baseline.snapshot()`; **reward-tensor bit-equality** (step reward floor_clearance-unset == explicitly-0.0); student-obs schema unchanged.
2. AC1 wiring — identical state below vs above the 0.5 m band → reward strictly lower when enabled, **equal to default-off when clearance ≥ band** (one-sided + no-leak); grace-suppression.
3. AC2 combined — extend `_run_rollout_probe` with `on_policy_beta_floor`: both flags true + β-floor → executed action blends teacher (teacher_share>0) while label stays teacher ⇒ both levers active; flag-off stays pure student.
4. AC3 — `floor_clearance>0` → PPO rollout reward reflects the penalty (same below/above-band assertion on the env step reward feeding :1641); config test combined-ppo has `finetune.steps≈3e7` + `env.rewards.floor_clearance>0`.
5. CPU 1-iteration smoke train for combined + combined-ppo paths.
- Note: `tests/test_reward_docs.py` gates `drone_fly.env.config.RewardConfig` (UC-58 single-env), **not** `sim.multi_course.RewardConfig` — no doc-gate change needed.

**AC8** — doc records ≥5% held-out (noise) as success for any variant; else records the negative result + the stated pivot to obstacle detection with the 70-input connectome model (UC-73/UC-78 line); UC-81 is the last distillation attempt before that pivot.

## Files Affected
- **Production (developer, `src/drone_fly/**`):**
  - `sim/multi_course.py` — `floor_clearance` field + `SAFE_CLEARANCE_M` const + dense-term wiring (~:894) gated by `~grace`.
  - `training/config.py` — `on_policy_beta_floor` field (DistillSection ~:530) + `_DISTILL_UC77_DEFAULTS` entry (:567) + byte-identity pop in `TeacherConfig.snapshot` (~:365/:371).
  - `training/distill.py` — AC2 branch (~:1311–1318) + teacher_share tracking.
  - `training/configs/student-liftoff-open-uc81-{baseline,combined,combined-ppo}.yaml` (3 new).
  - `scripts/uc81_combined_fix.sh` (developer-authorized `scripts/**`).
  - `docs/experiments/uc81-combined-distillation-fix.md` (developer-authorized `docs/experiments/**`).
- **Test (qa, `tests/**`):** `tests/test_training_student_uc81.py`.

## Risks & Considerations
- Reward-hacking / PPO destabilization: mitigated by one-sided bounded form + weight 0.03 off-by-default; doc reports reward curve.
- Equal-budget relaxed (combined-ppo ≈4.8e7 vs 1.8e7): deliberate, documented confound.
- 3e7 may still undertrain: doc must note if PPO reward still rising; AC8 pivot likely triggers first.
- `SAFE_CLEARANCE_M=0.5` is course-scale-coupled: fine for obstacle-free `student-liftoff-open` scope; revisit for other geometries (noted, out of scope).
- combined-ppo attribution confound documented (AC6) with follow-up to isolate the lever if it clears 5%.

## Challenger verdict (APPROVE, round 2)
Round 1 Critical (AC2 keying `on_policy_relabel and beta_end<=0` would break the locked UC-80 `test_on_policy_executes_the_student_while_the_teacher_labels` and silently redefine the flag) resolved by adopting an explicit `on_policy_beta_floor` flag. Re-traced: existing on_policy=True/False tests pass unchanged, combined config blends while labels stay teacher, RNG stream unchanged, new field popped at default. Non-blocking recs folded in: floor_clearance=0.03 pinned, three-way byte-identity test, combined-ppo confound documented, test_reward_docs hedge dropped (different RewardConfig). Cleared for implementation.
