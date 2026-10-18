---
plan_for: use-cases/87-deploy-robust-retrain.md
work_branch: feat/uc-87-deploy-robust-retrain-2
team: drone-fly-uc-87
approved: 2026-10-18
---

# UC-87 Implementation Plan — deploy-robust retraining of the 70-dim connectome policy (sim only)

_Approved by the challenger (round 2 of 2); all 9 review rulings folded in; two code-level claims re-verified by the challenger. Incorporates the challenger's two non-blocking refinements (ceiling_z default 5.0; D-test-b exact/atol-1e-6 tolerance)._

TARGET_DIR = /workspace/drone-fly-uc-87-deploy-robust-retrain-2. All paths below are under it.

## Analysis
Model 11 (`training/11-count-curriculum/model_readout.pt`, 70-dim obstacle-aware connectome reservoir, extra_dim=70) flies smoothly in sim with perfect inputs but chatters/flips live: ~70% control timing (command hold + ~75ms actuation delay) on a memoryless high-gain rate feedback, ~10% perception noise, ~12-15% intrinsic. Fix = retrain with deploy-timing + input randomisation, add policy memory, add an action-rate penalty, prove smoothness offline at live-like conditions. No live Liftoff on this machine.

## Frozen decisions
- **D0 — Landing.** All work lands in the promoted `src/drone_fly/**` path (`sim/`, `training/`, `controller/`, `harness/`) + `tests/**` + `training/configs/` + `docs/`. `gpu_prototype/` is left untouched (legacy, CUDA-hardcoded). Developer note: there are TWO env/config trees — the older `src/drone_fly/env/*` (pybullet-adapter lineage) and the active `src/drone_fly/sim/*` (model-11 lineage). **Edit only `sim/*`; do not touch `env/*`.** Confirmed: uc82/uc83 configs already warm-start model 11 into `CourseEnv` at extra_dim=70, so this is the established path.
- **D1 — Memory-block layout.** Append, as the **last** block of the extra vector, in this exact order: `prev_action(4)` then `obs_history(2×12=24)`. `prev_action` = the 4 raw action dims (3 body rates + collective) the policy OUTPUT on the previous policy-step, pre-decode/pre-ZOH/pre-delay. `obs_history` = the last **2** policy-step 12-dim core flight-obs vectors (`rel(3)+gravity_body(3)+vel(3)+omega(3)`), raw-stacked (not deltas, not EMA), sampled at the policy-step rate. Gate/next/obstacle_vision/raycast blocks excluded from history. **New `extra_dim = 70 + 4 + 24 = 98`.** H is config-exposed; the fine-tune run uses H=2 → 98.
- **D2 — End-append invariant.** The memory block is strictly appended AFTER raycast in BOTH sim `extra()` and harness `EXTRA_LAYOUT`; columns 0..69 stay byte-identical in meaning/order to model 11. This is what makes `controller/reservoir.py:warm_start`'s zero-init reproduce model 11. Hard invariant, enforced by test (D-test-b).
- **D3 — Parity semantics.** `prev_action` and `obs_history` seeded to **zeros** at episode (sim) / run (harness) start, ring-reset on episode/run boundary. History sampled at the **policy-step** rate, never the physics substep. The contract runner path becomes **stateful**, mirroring `harness/student.py`'s prev_action/history tracking. Parity test drives an identical input sequence through sim `extra()` and the harness assembler and asserts equal memory blocks.
- **D4 — τ override, no profile edit.** Verified: `DynamicsRandomizer.bounds()` (sim/randomization.py) samples only along the `nominal→profile-range` axis and can't independently target a fast-Betaflight τ. Add an optional per-key `range_overrides: dict[str,(min,max)]` to `DynamicsRandomizer.__init__` that takes precedence over `profile.randomization[key]` in `bounds()`, threaded `Plant.__init__` → `CourseEnv.__init__` → `PlantConfig`/`TrainConfig` → `ppo.build_env`. UC-87 config sets `rates.{roll,pitch,yaw}.tau_s` override to ~[0.005, 0.083] (fast Betaflight → soft 55ms+). **drones/meteor75/drone.yaml is NOT edited.**
- **D5 — Eval single-source-of-truth.** The eval's `perfect`/`live-like`/`sweep` presets reference the SAME TimingConfig/PlantConfig(+τ-override) fields the training DR uses — no hardcoded eval timing. `live-like` = widened τ (fast loop) + 50ms control period + 75ms delay + all input noise on. `perfect` = training defaults (fixed 50Hz, zero delay, nominal τ, no noise).
- **D6 — GPU boundary.** Dev-team "done" = all code/config green on CPU + a runnable, unit-tested eval harness + CPU-hermetic tests (AC10). The fine-tune training run and empirical AC7/AC8 numbers are produced post-merge on the GPU box. The deliverable README explicitly flags AC7/AC8 numbers as deferred-to-GPU-run (not silently omitted). QA does not block on empirical bars.

## Proposed Solution — files to change (production; developer). All under `src/drone_fly/` unless noted.
- **`sim/course.py`** — (a) append the D1 memory block to `extra()` (last block, +28 dims, zero-seed + per-episode ring-reset, policy-step-rate sampling); update `BASE_EXTRA_DIM` accounting (raycast→70, memory→98). (b) In `step_command`: add action-rate penalty `λ·‖tanh(a_t)−tanh(a_{t-1})‖²` + a small saturation penalty; log both per-episode (new episode-log columns). (c) Make `CEIL` a config field `ceiling_z` (**default 5.0**, AC5 4–6m) + add a floor-margin reward term. (d) Randomise spawn yaw 0–360° and spawn distance 1.5–10m to the first gate (config ranges). (e) Add gyro/omega observation noise (σ≈0.3–0.5 + coloured/AR(1) vibration) injected into the **observed** `omega` slice of the obs the policy consumes, NOT the plant's true state — training-only DR, NOT mirrored by the harness. (f) Make the floor visible to the raycast.
- **`sim/timing.py`** — widen `control_hz` to reach 60ms (≈16.7–50Hz); widen `action_delay_s` to 0–100ms with a new distribution option centred ~75ms; keep per-episode logging.
- **`sim/randomization.py`** — add the D4 `range_overrides` hook (`__init__` + `bounds`).
- **`sim/plant.py`** — thread `range_overrides` from `PlantConfig`/`Plant.__init__` into `DynamicsRandomizer`.
- **`controller/reservoir.py`** — confirm `K1ReservoirAug(extra_dim=98)` builds and `warm_start(src_extra_dim=70)` zero-inits the 28 new `pi.0`/`vf.0` columns (mechanism present); set `input_scale` for the new columns (raw/1.0, student convention).
- **`sim/action.py`** — bump `contract_version` in the sidecar (extra_dim already recorded).
- **`training/config.py`** — new fields on `CourseConfig`/`ActorConfig`/`PlantConfig`/`TrainConfig`: memory toggles (`prev_action`, `history_len`=2, content=core-obs), reward weights (`action_rate_lambda` ~0.05–0.1 relative to spin term, `saturation_weight`), `ceiling_z`, spawn-yaw/spawn-distance ranges, τ `range_overrides`; wire into `build_env`.
- **`training/ppo.py`** — pass new config to env; warm-start src_extra_dim=70 → extra_dim=98; log new reward terms.
- **`harness/assembler.py`** — extend `ContractObservationAssembler` to accept prev_action + obs-history state and append the D1 memory block; extend `EXTRA_LAYOUT`; new assembler/contract version.
- **`harness/runner.py`** — make the contract-policy step path stateful (store raw action output per policy-step + ring of last H policy-step obs vectors; zero-seed + boundary reset; mirror `harness/student.py`).
- **`harness/policy.py`** — add `98` to `SUPPORTED_EXTRA_DIMS`; sidecar validation handles it.
- **`harness/config.py`** — new-contract defaults: `align_spawn_yaw: false` + latency compensation off.
- **`sim/smooth_metrics.py`** (new) — pure CPU metrics: saturation share (|rate|≥3.8), per-axis sign-flips/s (>0.4 rad/s), mean |Δaction|/policy-tick, >5Hz command-power share, completion/gates/crashes; raw-action convention `tanh(a)*MAX_BODY_RATE` used consistently.
- **`training/smooth_eval.py`** (new; builds on `training/proc_eval.py`) — deterministic mean-action rollouts capturing the action trace, for model 11 AND the fine-tuned policy, under the D5 presets. Emit explicit PASS/FAIL for the FIVE AC7 bars: (1) live-like mean|Δaction|≤0.6, (2) live-like sat≤10%, (3) live-like yaw-flips≤3/s, (4) live-like completion within 10pp of model-11-perfect, (5) fine-tuned-perfect not regressing model-11-perfect by >5pp. AC8 sweep = grid over control-period × delay × τ-bandwidth (+ spawn yaw) vs model 11. AC3 report includes the λ trade-off curve.
- **`training/export.py` / `run_dir.py`** — AC9 export: checkpoint + `.contract.json` sidecar (extra_dim=98, bumped contract_version) + README + live harness config.
- **`training/configs/uc87-deploy-robust.yaml`** (new) — extends `uc83-obstacle-rays-v2`; warm_start from model 11 at src_extra_dim=70; extra_dim=98; live-like timing ranges; τ `range_overrides`; action-rate+saturation weights; `ceiling_z: 5.0`; spawn-yaw/dist DR; gyro noise.
- **`docs/INFERENCE_CONTRACT.md`** — document the new contract version + 98-dim layout (70 unchanged + prev_action(4) + obs_history(24)) + the end-append invariant.
- **New `training/NN-deploy-robust/` dir** — deliverable README (training config, metrics table, AC7/AC8-deferred flag) + live harness yaml (align_spawn_yaw:false, latency off) + instructions to verify live passes via `harness/offline/recordings.py:completeness_from_pose` (NOT the overflight `gate_passed`).

## Files Affected
**Production (developer):** `sim/course.py`, `sim/timing.py`, `sim/randomization.py`, `sim/plant.py`, `sim/action.py`, `sim/smooth_metrics.py`(new), `controller/reservoir.py`, `training/config.py`, `training/ppo.py`, `training/smooth_eval.py`(new), `training/export.py`, `training/run_dir.py`, `harness/assembler.py`, `harness/runner.py`, `harness/policy.py`, `harness/config.py`, `training/configs/uc87-deploy-robust.yaml`(new), `docs/INFERENCE_CONTRACT.md`, `training/NN-deploy-robust/` README + live yaml(new). All within `paths.production=src/drone_fly/**` except the doc/config/deliverable artifacts (`docs/`, `training/configs/`, `training/NN-deploy-robust/`) which are training-lineage repo-data (not session-dir, not `gpu_prototype/`).

**Test (qa), `tests/`:**
- `test_sim_timing_zoh.py` — ZOH + delay-buffer application times (AC10).
- `test_reservoir_memory.py` — (a) prev-action/history assembly in sim at policy-step rate + zero-seed/reset; **(D-test-b)** warm-started 98-dim policy produces actions identical to model 11 on a fixed batch when memory inputs are zero — assert exact where zero-padding makes it exact, else tight `atol=1e-6` (AC2 crux).
- `test_old_contract_load.py` — model-11 checkpoint still loads at extra_dim=70 unchanged (AC2 backward-compat).
- `test_harness_memory_parity.py` — identical input sequence → equal memory blocks sim `extra()` vs harness assembler (AC2 pitfall).
- `test_reward_penalties.py` — action-rate + saturation terms (AC3).
- `test_input_randomizers.py` — statistics of each randomiser: gyro/omega noise, floor-visible rays, spawn yaw 0–360°, spawn distance 1.5–10m, obstacle_vision zeroed, parameterized gate staleness/outliers/Markov-dropout (AC4).
- `test_smooth_metrics.py` — metrics on synthetic traces (AC6/AC10).
- `test_tau_override.py` — `range_overrides` widens sampled τ beyond the profile range (D4).

## AC → coverage map
AC1: timing-range widening + τ override (D4) + existing Plant ZOH/delay → test_sim_timing_zoh, test_tau_override. AC2: D1/D2/D3 memory + warm_start zero-init + contract bump → test_reservoir_memory, test_old_contract_load, test_harness_memory_parity. AC3: penalties+logging+λ-curve → test_reward_penalties. AC4: gyro noise/floor rays/spawn yaw/spawn dist = must-add; gate staleness(0–160ms)/10%-outliers@0.4m/30%-Markov-dropout = PARTIAL-exists in sim/perception+noise_model, parameterize to AC4 numbers; obstacle_vision-zero = toggle → test_input_randomizers. AC5: `ceiling_z`+floor margin. AC6: smooth_metrics + smooth_eval presets (D5) → test_smooth_metrics. AC7: five PASS/FAIL bars in smooth_eval. AC8: sweep grid. AC9: export + README + live yaml + completeness_from_pose instructions. AC10: all tests above, CPU-hermetic.

## Risks
1. Backward-compat crux (D2) — end-append invariant; enforced by D-test-b + test_old_contract_load.
2. Parity (D3) — policy-step-rate sampling + action-actually-output + zero-seed/reset; gyro noise is training-only, deliberately NOT mirrored live.
3. τ override (D4) — new hook threads through 4 layers (randomization→plant→course→config/ppo).
4. λ trade-off (AC3) — penalty can cost completion; report includes λ-vs-completion curve; config λ is a starting point to tune on the GPU run.
5. GPU boundary (D6) — empirical AC7/AC8 numbers deferred post-merge; tested metric functions + runnable harness make the bars trustworthy; README flags deferral.

## Challenger verdict
APPROVE (round 2 of 2). Round 1 was Request-revision over 6 Major + 3 Minor items (open design decisions the analyst had deferred); the analyst folded all 9 in faithfully. Big factual claims verified against the actual code in both rounds — nothing hand-waved. Two non-blocking Minor notes left for the developer: pick a concrete `ceiling_z` default (resolved: 5.0); use a tight atol if bit-identical is brittle (resolved: exact where zero-padding makes it exact, else atol=1e-6). Developer-"done" is CPU-green code + runnable eval harness + CPU-hermetic tests; empirical AC7/AC8 numbers deferred to a post-merge GPU run and flagged in the deliverable README — QA must NOT block on empirical bars.
