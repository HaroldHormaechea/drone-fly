---
plan_for: use-cases/80-fix-student-distillation-training.md
work_branch: feat/uc-80-fix-student-distillation-training
team: drone-fly-uc-80
approved: 2026-10-16
---

# UC-80 — Fix student distillation training — Approved implementation plan

Challenger APPROVED (round 1; 4 minor doc/test refinements folded in; all line-refs verified against the tree).

## Key finding
Two of the three interventions ALREADY exist in code — only on-policy relabelling needs new logic. The rest is config + a resumable driver + the experiment doc + tests.

## Proposed Solution

### (AC2) On-policy DAgger relabelling — the one new code path
- `src/drone_fly/training/config.py`: add `on_policy_relabel: bool = False` to `DistillSection` (near config.py:530-557) AND add it to `_DISTILL_UC77_DEFAULTS` (config.py:562). The `snapshot()` pop (config.py:644-647) then drops it at default → `config.yaml` byte-identical (AC1). Sidecar is structurally unaffected (`student_sidecar` doesn't dump the distill config).
- `src/drone_fly/training/distill.py` (`_train_student` rollout, ~1308-1312): when `d.on_policy_relabel`, set `act = s_mean + noise_` unconditionally (student-driven execution), decoupling execution from β. Teacher still labels every visited state (`buf.add(..., decoded_label(t_mean,...))` unchanged). ~3 flag-guarded lines; flag-off path is bit-identical. Zero extra teacher queries (teacher already queried every tick, distill.py:1305).

### (AC4) β schedule — config-only
Already parameterised (`beta_start/beta_end/beta_schedule/beta_over`, `beta_at` distill.py:196-204). Variant config sets `distill.beta_end: 0.15`, `distill.beta_over: 1.0` (gentler/longer; floors teacher correction at ~15% across the whole budget). No code change.

### (AC3) PPO fine-tune — config-only + budget split
Already exists (`FinetuneSection`, `ppo_finetune` distill.py:1569, trigger at 1411-1415). Optimises the env reward `r,done = senv.step(a)` = `RewardConfig` gate 10 / finish 50 / progress 1.0 (multi_course.py:139-146). Variant sets `finetune.steps`. **Budget split = 1.2e7 DAgger + 0.6e7 PPO = 1.8e7 total env-steps** (both loops count env-steps: DAgger `steps += rollout*n` @1328, PPO `steps += roll*n` @1654 — challenger verified units match). Doc must state equal-budget = TOTAL env-steps (DAgger+PPO) and flag that PPO's arm gets fewer imitation steps (comparability confound) + sparse-reward undertraining risk.

### (AC5) Four configs under `src/drone_fly/training/configs/`
All extend `student-liftoff-open`, set `student.actor: mlp-student` (confirmed valid vs `STUDENT_ACTORS`, config.py:456; default is k1-student so must be explicit), `budget: 1.8e7`, `final_report: true` (eval already wired: 32 ep / seed 1234, held-out + Infinity-Loop, exact+noise, outcome breakdown):
- `student-liftoff-open-uc80-baseline.yaml` (no new fields — byte-identical distill)
- `student-liftoff-open-uc80-onpolicy.yaml` (`on_policy_relabel: true`)
- `student-liftoff-open-uc80-beta.yaml` (`beta_end: 0.15`, `beta_over: 1.0`)
- `student-liftoff-open-uc80-ppo.yaml` (`budget: 1.2e7`, `finetune.steps: 6.0e6`)

### (AC6) `scripts/uc80_distillation_fix.sh`
Resumable driver looping the 4 configs; resume via `--resume <folder>` if the run exists else fresh (resume supported, cli.py:660-664). Document artifacts to copy to any CUDA/no-Liftoff box: teacher run `teacher_it1374`, noise-model YAML, Liftoff track-conversion cache.

### (AC7) `docs/experiments/uc80-distillation-training-fix.md`
One comparison table (rows: teacher, baseline, on_policy, ppo, beta, + UC-77 & UC-79 `teacher_full` reference; columns: held-out macro_completion exact+noise, Infinity-Loop exact+noise, outcome breakdown) + recommendation (lever(s) that move completion, or the next lever if none hit ≥5%). Mirror `uc77-obstacle-free-student.md`. **Include:** equal-budget = total env-steps note; the 1.2+0.6 split justification; the on-policy "no teacher-forcing warmup → relies on warm-start; β-floor combo is the follow-up" note.

## Files Affected
**Production (developer; `src/drone_fly/**`):** `config.py` (field + defaults entry), `distill.py` (guarded execution), 4 new `configs/*.yaml`.
**Non-production (developer-authorized):** `scripts/uc80_distillation_fix.sh`, `docs/experiments/uc80-distillation-training-fix.md`.
**Test (qa; `tests/**`), extend `test_training_student.py` or new `test_training_student_uc80.py`:**
- AC1: snapshot byte-identity (default drops `on_policy_relabel`, snapshot == reference) + a cheap sidecar-unchanged assertion; obs DIM/schema unchanged.
- AC2: relabel-source spy (mirror `test_labels_come_from_teacher_obs...` :362) — flag ON: `senv.step` gets the STUDENT action & buffer label == teacher mean; flag OFF: executes the β-mix (locks the gate).
- AC3: PPO-reward wiring (existing `test_ppo_iteration...` :457 covers "one update moves policy"; assert the optimised reward is the env gate/completion reward).
- AC4: extend `test_beta_schedules` :293 — gentler params give strictly higher β at a given frac.
- AC9: CPU 1-iter smoke train per new path (on_policy / ppo / beta) via `train(..., max_iterations=1)`; config-load test for the 4 UC-80 configs (mirror `test_run2_config_plumbs...` :129).

## Risks
PPO sparse-reward undertraining (doc flags, doesn't fail-conclude) · on-policy loses teacher-forcing warmup (warm-start mitigates; β-floor combo = follow-up) · on-policy zero throughput overhead · budget comparability confound (documented) · success bar ≥5% held-out noise for ≥1 variant (AC8), else record negative result + next lever.

## Challenger verdict (APPROVE, round 1)
AC1 default-off byte-identity verified (config.py:644-647 snapshot pop; sidecar at distill.py:964 doesn't serialize distill config). AC2 is genuine on-policy DAgger (student-driven execution, teacher labels student-visited states; test asserts target source). AC3 reward = env RewardConfig reused at distill.py:1636. AC4 β already parameterised. AC5/AC8 equal-budget math sound (1.2e7 DAgger + 6e6 PPO = 1.8e7 true total; both count env-steps). Paths correct. 4 minor doc/test refinements folded in; none block implementation.
