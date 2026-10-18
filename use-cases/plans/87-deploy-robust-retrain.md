---
plan_for: use-cases/87-deploy-robust-retrain.md
work_branch: feat/uc-87-deploy-robust-retrain-2
team: drone-fly-uc-87
approved: 2026-10-18
supersedes: the run-1 fine-tune plan (failed AC7 — completion collapse); this is the fresh-train redesign
---

# UC-87 Redesign — FINAL APPROVED Implementation Proposal
_(Analyst → challenger-approved round 1; incorporates the 4 minor recs + the path-scope resolution.)_

## Analysis — problem + context

**What failed (run-1/run-2, `training/13-deploy-robust/README.md`).** Warm-starting model 11 straight into FULL domain-randomisation from step 0 — above all an **unramped spawn-yaw 0–360°** model 11 never saw, plus the model-11 **obstacle-count curriculum discarded** (fracs null) — collapsed completion (0.0%/0.8% vs model 11's 54.7% perfect). Run-2 quartered the penalties and completion barely moved (0.0→0.8%), proving the blocker is the training *task shock*, not the penalty weights.

**Revised approach (use-case file, revised 2026-10-18):** train **fresh** (no warm-start), **smooth by construction**, with a **staged curriculum**, keeping the `SpawnYawAligner` and **no spawn-yaw randomisation**.

**Infrastructure already built on this branch (REUSE, do not rebuild):**
- Policy memory — `src/drone_fly/sim/course.py`: `CourseConfig.memory/history_len`, `memory_dim`/`extra_dim` props (extra_dim 98 at history_len 2), `prev_action` maintained, `_memory_block`/`_update_memory`, end-append invariant. Obs-contract v2 in `src/drone_fly/sim/action.py`.
- Smoothness reward — `course.py` reward block reads `cfg.action_rate_lambda`/`saturation_weight`/`saturation_threshold`; per-episode `ep_arate`/`ep_sat` in EPISODE_TAIL.
- Input DR — `course.py` gyro `_update_input_dr`, gate noise, `raycast_floor`, `obstacle_vision_zero`; all read fixed `cfg.*` magnitudes (no ramp today).
- Deploy timing — `src/drone_fly/sim/timing.py` `TimingConfig`/`ControlTiming`; tau via `src/drone_fly/sim/randomization.py` `DynamicsRandomizer` + `src/drone_fly/sim/plant.py` `range_overrides` (per-key absolute override, D4). Scale about nominal: `lo = n + scale·(rng_min − n)` — scale set once at __init__; not runtime-mutable today.
- Eval harness — `src/drone_fly/training/smooth_metrics.py` + `smooth_eval.py` (`preset_config` perfect/live-like, `evaluate_one`, `evaluate_bars` 5 bars, `sweep_grid`/`sweep_cell_config`). `training/export.py` `export_deploy_robust`.
- Aligner — `src/drone_fly/harness/yaw_align.py` `SpawnYawAligner`, `runner.align_spawn_yaw`. The CORRECT spawn-yaw solution; KEEP ON.
- Curriculum machinery — `src/drone_fly/training/ppo.py`: `_ramp(frac, over)`, per-iteration ramp loop (`setattr(env, attr, v)` for difficulty/obs_pos_curr/obs_count_curr), inline log_std anneal.
- **Model-11 completion recipe (verified source):** `gpu_prototype/exp11_count-curriculum.py` — `count_curr_frac=0.5`, logstd `init=-0.5, final=-1.0, anneal_start=0.5`, clearance-shaping + terminal-contact. Note: exp11 RESUMED a baseline and did NOT re-run the position curriculum; a fresh run must restore position + difficulty + count together.

## Proposed Solution

New curriculum/ramp knobs + a fresh-train config + eval-bar fixes. Everything defaults to byte-identical behaviour (new ramp scales default 1.0 = full; new config fracs default None = off), so existing configs, warm-start/zero-init back-compat, and the eval presets are unaffected.

**1. Three runtime ramp scales on `CourseEnv` (`src/drone_fly/sim/course.py`), each default 1.0.**
- `self.deploy_dr_scale` — ramps deployment DR (AC1/AC6 "Deployment"): control timing/delay/skip AND the plant dynamics-randomisation scale (carries the tau-override span). Read at `_spawn` when calling `self.timing.resample` and when (re)applying the plant scale.
- `self.input_dr_scale` — ramps input noise (AC5/AC6 "Perception"): multiplies effective gyro white+vibration σ, gate pos-noise/outlier/staleness/dropout, and ray-noise magnitude. At scale 0 → zero noise contribution; at 1 → full cfg magnitudes.
- `self.penalty_scale` — multiplies ONLY `cfg.action_rate_lambda` in the reward (AC3 λ anneal mild→target). `saturation_weight` is intentionally un-annealed (already small/constant guard-rail; annealing adds no AC value) — documented in code comment + config (rec #3).

**2. Make timing ramp-able (`src/drone_fly/sim/timing.py`).** Add a `scale` field to `ControlTiming` (default 1.0). In `resample`, interpolate between a CLEAN baseline and full sampled values: `period = clean + scale·(sampled − clean)`, `delay = scale·sampled_delay`, `skip = scale·sampled_skip`. The clean baseline rate is `TimingConfig.fixed_hz` — the same anchor the `perfect` eval preset uses (`TimingConfig.fixed(PERFECT_HZ)`); this coupling must be called out in a code comment so a future `fixed_hz`/`PERFECT_HZ` change stays consistent (rec #4). Env sets `self.timing.scale = self.deploy_dr_scale` before each resample. Scale 0 → clean; 1 → full range.

**3. Make dynamics scale runtime-mutable (`randomization.py` + `plant.py`).** Add `DynamicsRandomizer.set_scale(s)` recomputing `_lo/_span` (same formula as `bounds`/`__init__`), exposed as `Plant.set_dr_scale(s)`. Env calls it with `cfg.randomization_scale · deploy_dr_scale` when the scale changes. Scale 0 → nominal plant (tau = profile nominal, no DR); ramps to `cfg.randomization_scale` (tau override span opens). Eval presets collapse tau via range_overrides, so eval is unchanged.

**4. New schedule knobs in `CurriculumConfig` (`training/config.py`), each default None/off.** Each new ramp is a `[start_frac, end_frac]` pair (linear 0→1 between start and end; None = off ⇒ scale pinned at 1.0):
- `timing_dr_frac: [start,end] | None` (AC1/AC6 Deployment)
- `input_dr_frac: [start,end] | None` (AC5/AC6 Perception)
- `penalty_frac: [start,end] | None` + `penalty_init_scale: float = 1.0` (AC3): effective λ-scale ramps `penalty_init_scale → 1.0`.
(Existing difficulty/obs_* fracs stay single-float ramp-from-0 — unchanged.)

**5. Wire the ramps into the train loop (`training/ppo.py`).** Add `_ramp_between(frac, start, end)` beside `_ramp`. In the per-iteration block set `env.deploy_dr_scale`, `env.input_dr_scale`, `env.penalty_scale` from the new knobs (default 1.0 when None). Log the three scales in the metrics row (next to `log_std`/`difficulty`) so the schedule + λ-vs-completion trade-off are visible (AC3/AC6).

**6. Fresh-train staged config — rewrite `training/configs/uc87-deploy-robust.yaml` (AC2/AC4/AC6).**
- `warm_start.checkpoint: null` (fresh; readout incl. memory columns randomly init).
- `extra_dim 98`, `memory: true`, `history_len: 2`.
- Completion recipe seeded from model 11 (`exp11_count-curriculum.py`) + lineage: `obs_count_frac: 0.5`; `obs_pos_frac: ~0.5` and `difficulty_frac: 0.6` (restored — fresh run has no prior stage); logstd `logstd_init: -0.5, logstd_final: -1.0, anneal_start: 0.5`; full `shaping`/`obs_terminal`/`floor` + `ceiling_z 5.0` (rec #2).
- Aligned spawn: `spawn_yaw: false`, `spawn_distance_m: null` (AC4).
- Staged schedule: `timing_dr_frac: [~0.3, ~0.6]`, `input_dr_frac: [~0.6, ~0.85]`, `penalty_frac: [~0.5, ~0.8]`, `penalty_init_scale: ~0.1`; `action_rate_lambda` = target, `saturation_weight` small/constant.
- Larger `budget` than the 8M fine-tune (fresh needs more — size to the curriculum; report the completion curve). Keep timing/plant DR ranges as before (16.7–50 Hz, 0–100 ms triangular@75 ms, tau override 0.005–0.083) — now ramped in, not from step 0.
- Documented fallback: add `training/configs/uc87-deploy-robust-warmstart.yaml` that `extends` the fresh config and only sets `warm_start.checkpoint` = model 11 + `src_extra_dim: 70` (zero-init memory cols), to reach the completion base if fresh can't — ramps unchanged.

**7. Eval-bar fixes (`smooth_eval.py`).**
- AC9: tighten the `perfect` bar from "not worse by >5 pp" to ≥ model-11 perfect (`AC7_PERFECT_REGRESS_PP → 0` / change the comparison). Completion-gap bar (within 10 pp) unchanged.
- AC10: remove the spawn-yaw axis from `sweep_grid` and `sweep_cell_config` — spawn is aligned.

**8. Deliverable (AC11) — `training/13-deploy-robust/`.**
- `harness-liftoff.yaml`: flip `runner.align_spawn_yaw: true` (the aligner IS the spawn-yaw solution; it was wrongly false); latency-comp stays off.
- `README.md`: rewrite to the fresh-train redesign (curriculum schedule table, config, metrics placeholders), KEEP the run-1/run-2 empirical failure section as history, document the warm-start fallback.

## Files Affected

**Production code (developer):**
- `src/drone_fly/sim/course.py` — 3 ramp-scale attrs + application points.
- `src/drone_fly/sim/timing.py` — `ControlTiming.scale` + interpolated resample (fixed_hz base-rate comment).
- `src/drone_fly/sim/randomization.py` — `DynamicsRandomizer.set_scale`.
- `src/drone_fly/sim/plant.py` — `Plant.set_dr_scale` passthrough.
- `src/drone_fly/training/config.py` — `CurriculumConfig` new fields.
- `src/drone_fly/training/ppo.py` — `_ramp_between` + wire + log the three scales.
- `src/drone_fly/training/smooth_eval.py` — AC9 perfect bar, AC10 drop yaw axis.
- `src/drone_fly/training/configs/uc87-deploy-robust.yaml` (rewrite) + `uc87-deploy-robust-warmstart.yaml` (new fallback).
- `training/13-deploy-robust/harness-liftoff.yaml` (flip align_spawn_yaw) + `training/13-deploy-robust/README.md` (rewrite). Path-scope resolved: developer owns `training/13-deploy-robust/**` per CLAUDE.md "Dev-team write authorizations".

**Test code (qa):**
- `tests/test_sim_timing_zoh.py` / `test_sim_timing.py` — timing ramp: scale 0 → clean (no delay/skip, fixed_hz), scale 1 → full, monotone.
- `tests/test_sim_randomization.py` / `test_tau_override.py` — `set_scale` recompute: scale 0 → nominal, ramps to full override span.
- `tests/test_input_randomizers.py` — `input_dr_scale` scales gyro/gate/ray stats; at scale 0 assert ZERO noise EFFECT on the observation (observed omega == true omega, gate estimate == true rel), NOT identical-RNG/byte-identity (rec #1).
- `tests/test_reward_penalties.py` — penalty anneal: effective λ = `penalty_init_scale·target` → target; assert `saturation_weight` unaffected by `penalty_scale` (rec #3).
- New `tests/test_curriculum_ramp.py` (or extend `test_training_cli.py`) — `_ramp_between` as fn of fraction + train-loop wiring (short `max_iterations`).
- `tests/test_sim_course.py` — aligned-spawn: with `spawn_yaw false` the spawn quat is identity/aligned.
- `tests/test_smooth_eval.py` — AC9 tightened perfect bar; AC10 sweep grid has no yaw axis.
- Keep green: `test_reservoir_memory.py`, `test_harness_memory_parity.py`, `test_old_contract_load.py`, `test_altitude_envelope.py`, `test_export_deploy_robust.py`.

## Risks & Considerations
- Fresh-train budget: fresh may need a bigger budget than a fine-tune to hit model-11-level completion; size to the curriculum, with the documented warm-start-for-completion-base fallback.
- λ/completion trade-off: per-iteration λ/completion logging makes the curve reportable to tune the target on the GPU run.
- Dynamics-scale ramp granularity: `deploy_dr_scale` ramps the ENTIRE plant DR scale (thrust/drag/rate_scale too), consistent with AC1/AC6 "Deployment" (nominal plant in base, full realism in deployment); one knob.
- Penalty-knob placement: λ anneal lives in `CurriculumConfig` (cohesion with other budget-fraction ramps).
- Byte-identical guarantee: new scales default 1.0, new fracs default None ⇒ eval presets, warm-start back-compat, existing configs/tests unchanged; default-off covered by a zero-effect assertion (not RNG identity).
- spawn_yaw code retained (dormant, off) for back-compat; only the AC10 sweep axis removed.

## Challenger verdict
APPROVE (round 1). Verified every material claim against the branch code. Reverses all four documented run-1/2 failure causes and reintroduces none. New knobs real, threaded through the existing ramp machinery, CPU-unit-testable without a GPU run. 4 minor recs folded in (zero-effect assertion not RNG identity; seed fractions from exp11; saturation un-annealed documented; fixed_hz base-rate comment). Existing infra reused, not rebuilt.
