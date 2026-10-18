---
plan_for: "(free-form task) UC70 run-2 — connectome-student: pointer obs + gate memory + FP handling"
work_branch: feat/uc-70-student-distillation-detector-noise
team: drone-fly
approved: 2026-10-14
---

# FINAL APPROVED PROPOSAL — UC70 run-2 (CODE ONLY; the dev-team does NOT run the 80M-step training). Challenger approved after one revision cycle (all 3 Majors resolved + verified).

## Goal
Close run-1's information/false-positive gap (ADR 0009 "Run 1 result": memoryless 2-gate perceived view → completion follows beta→0, ~65% OOB; gates39 closes 72% of the exact-to-teacher gap; FALSE POSITIVES are the costliest noise component) WITHOUT touching the teacher or the frozen K1. Three **independently-toggleable** features (so the experiment can attribute gains), all default-OFF ⇒ run-1 behavior, except the appended pointer obs block.

## The three changes

### 1. Next-gate pointer observation (NEW obs block; schema v2→v3)
- A **body-frame unit bearing to the current target gate (3) + availability bit (1)** = 4 values. Direction only (no range → not a route prior). Models Liftoff's on-screen next-gate HUD pointer.
- **Single-sourced in perception**: computed in `GatePerception.outputs()` AND the numpy twin `ReferencePerception.outputs()` (both are truth-driven, both hold `target["center"]`), flows `raw["pointer"] → estimate_from_raw → est.info["pointer"]`; torch builder reads `perc["pointer"]`, numpy builder reads `est.info["pointer"]` — one value, parity at the perception layer.
- **Live pre-UC72**: the learned `TrackingGateEstimator` doesn't populate `info["pointer"]` → None → zeros/bit=0 (the correct degraded behavior; the live HUD reader is UC72).
- **Train/serve robustness**: per-episode pointer DROPOUT (`pointer.dropout_p`) forces whole episodes to bit=0, so the student learns to fly without the pointer and stays deployable pre-UC72.
- **No truth-leakage**: the sim pointer is a NOISY HUD proxy (per-episode bias + per-tick jitter); the SAME noisy pointer feeds both the obs and the FP-suppression reference, so sim-suppression ≈ live-suppression.
- Appending obs columns needs NO change to controller/student.py (encoder/pi/residual auto-size from `student_obs.DIM`; warm-start zeros the new columns → student still == 07 on the legacy 30). DIM 61→65, VERSION 2→3.

### 2. World-frame gate memory (perception-internal; feeds EXISTING obs fields — no new columns)
- Decayed memory of each slot's last CONFIDENT (non-FP) world pose; **hand-over seeding** shifts next-slot memory → target on a pass, so a briefly-lost gate stays re-targetable. Fallback order: live-held → memory(fresh) → reacquire-released `t_last` → drone pos. Obs reports `t − mem_stamp` as the age when memory supplies the target.
- Guarded so `gate_memory.enabled=false` ⇒ `outputs()` byte-identical to run-1.

### 3. False-positive handling (perception-internal; no obs change)
- Three sub-gates on the `pick_fp`/`mis` acquisition paths: (a) pointer-consistency (reject candidates off the noisy pointer by > `bearing_tol_deg`), (b) memory-consistency (reject candidates far from remembered pose), (c) temporal persistence (require N consecutive frames before promoting a candidate). Each toggleable. No new random draws beyond the pointer-noise slots.

## Draw-layout change
`EPISODE_DRAWS` += {ptr_drop(1), ptr_bias(2)}; `TICK_DRAWS` += {ptr_jit(2)} — all always-drawn, config-scaled (p=0/sigma=0 ⇒ parity-strict). Run-1 is already flown and comparability is metric-based, so this doesn't affect it. **QA must regenerate all draw/parity fixtures in lockstep across the torch and numpy twins.**

## Files Affected
**Production (developer):**
- src/drone_fly/sim/perception.py — PerceptionConfig fields+validation+from_model; compute+emit `pointer` in outputs(); extended draw layout (dropout/bias/jitter); GateMemory state+decay+hand-over seeding; FP suppression gates; outputs() target fallback; thread target-valid into step.
- src/drone_fly/student/perception_ref.py — mirror ALL of the above (line-for-line parity).
- src/drone_fly/sim/student_obs.py — pointer block, DIM 61→65, VERSION 2→3, input_scale, schema() pointer sub-block, build_student_obs pointer arg.
- src/drone_fly/student/obs_ref.py — numpy build_student_obs pointer arg (from est.info).
- src/drone_fly/sim/student_env.py — forward config + target-valid via _passages (NO pointer computation here).
- src/drone_fly/harness/student.py — assembler passes gate.info.get("pointer") (None⇒zeros; UC72 doc).
- src/drone_fly/training/config.py — PerceptionSection: pointer/gate_memory/fp_suppression sub-configs + validation.
- src/drone_fly/training/distill.py — perception_config overrides + build_student_env wiring.
- src/drone_fly/student/checkpoint.py — expected_schema sources pointer sub-config from sch.get("pointer") (deep-equal guard).
- src/drone_fly/training/export.py — schema() built with pointer params.
- src/drone_fly/training/configs/student-liftoff-run2.yaml — NEW (extends student-liftoff; features ON; VRAM note).
- (controller/student.py — NO change; verified auto-sizing.)

**Test (qa):**
- tests/test_sim_perception.py, test_perception_tracker_parity.py — torch↔numpy parity incl. pointer/memory/FP-suppression; regenerate fixtures for new draw widths; parity runs use dropout_p=0/sigma=0.
- tests/test_perception_lost_policy.py — memory + suppression interplay with reacquire; hand-over seeding.
- tests/test_student_obs_parity.py — pointer parity (torch↔numpy↔live assembler); update (NAME,VERSION,DIM)→(student-obs,3,65).
- tests/test_obs_schema.py — v3 layout/offsets, input_scale length, pointer sub-schema.
- tests/test_student_checkpoint.py — expected_schema deep-equal w/ pointer; version guard v3.
- tests/test_controller_student.py — encoder/pi/residual auto-size; warm-start zeros pointer columns.
- tests/test_training_student.py, test_distill_paths.py — config plumbing + tiny-n_envs few-iteration smoke (QA's heaviest allowed).
- NEW: pointer-off episodes (dropout) still fly via gate-memory fallback; a synthetic FP off the (noisy) pointer is rejected w/ suppression on but hijacks the target off; bearing_tol_deg doesn't suppress true detections near the noisy pointer.

## Flags for the orchestrator (runtime, not code)
- **VRAM**: run-1's 4096 envs peaked ~3.1 GB (csr32) student-process on an 8 GB GPU (total incl. game contention peaked ~6.5 GB); on a 6 GB RTX A3000, 4096 fits only on a DEDICATED GPU. Set `n_envs: 2048` for run-2 (overridable; ~2× gradient-updates/sample retained, acceptable).
- **Live-pre-UC72 reality**: a run-2 student deployed live BEFORE UC72 degrades to {run-1 + gate-memory + FP-suppression (temporal+memory sub-gates only)}; the pointer OBS gain and the pointer FP-suppression sub-gate are unrealized until the UC72 HUD reader exists. Dropout keeps it deployable in that degraded regime.
- Pointer-noise sigmas and bearing_tol_deg are provisional defaults pending real HUD characterization in UC72.

## Challenger's final verdict: APPROVE (round 2). All 3 round-1 Majors (pointer computed in env → broke parity; train/serve skew; truth-leakage) resolved and re-verified. Controller needs no change; K1 stays frozen; scope clean (src/** + tests/**); run-1 preserved by default-OFF toggles.
