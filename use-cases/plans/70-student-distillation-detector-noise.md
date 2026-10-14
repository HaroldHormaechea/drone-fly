---
plan_for: use-cases/70-student-distillation-detector-noise.md
work_branch: feat/uc-70-student-distillation-detector-noise
run: drone-fly-uc-70
approved: 2026-10-13
---

# UC70 approved implementation plan

Approved by challenger-70 in round 2, with its four developer Minors folded in (see the end of this plan). This plan also applies the orchestrator's ruling: the feasibility gate does not block, and the user's capture is optional. It is self-contained and supersedes v1, v2 and v2.1.

## Run context

- WT = C:\dev\drone-fly-uc-70-student-distillation-detector-noise, stacked on UC69 fdcf541, which sits on UC68.
- paths.production `src/drone_fly/**`; paths.test `tests/**`; no profiles.
- Tests run with C:\dev\drone-fly\.venv-cuda\Scripts\python.exe and PYTHONPATH=src. Never `uv sync`.

**USER ASK (Risk 1):** a 3–5 min hand-flown Infinity Loop capture that is never used for training, flown once the GPU is free.
- It gives honest held-out data for BOTH detector weights, plus more samples.
- UC70 does NOT depend on it. It works on provisional data and refits in place when the capture arrives.

# Analysis

## A1. Detector data (measured)

There is one capture: `C:\dev\drone-fly\artifacts\harness\captures\2026-10-11-infinity-manual`.
- 2,043 frames over 182 s, hand-flown.
- 1,579 auto-labelled rows in `artifacts/harness/liftoff/labels/<same>/`.
- One class: `inflatable_arch`.

Weights in `artifacts/harness/liftoff/models/`:
- `liftoff-gates.pt` (purpose: evaluate). This is what the `liftoff` preset flies.
  - Split: train 1,082 / val 149 / **holdout 324** (the last ~54 s) / guard 24.
  - UC64 held-out evaluation, gates at 4–25 m:

    | Metric | Value |
    |---|---|
    | Detection | **19.6 % (29/148)** |
    | Median position error | 2.19 m (11 % of range) |
    | Median yaw error | 11.6° |
    | False positives | **1.03 per frame** |
    | CUDA latency | p50 8.1 ms, p95 22.2 ms |
- `liftoff-gates-deploy.pt` was trained on every frame, so no held-out data exists for it. Its 96 % detection is in-sample.

Live 07 runs (`artifacts/harness/liftoff/runs/*`, 6 runs, learned detector, world_scale 10):
- every run ended `gate_lost` after 96–115 ticks;
- the loop ran at 30–37 Hz effective against a 50 Hz target;
- 35–64 % of deadlines were missed;
- `ticks.jsonl` carries the perception ages.

## A2. What the live harness gives the policy

`harness/liftoff/gate_cv.py` `TrackingGateEstimator`:
- `target` is the held world centre of the tracked gate.
- `curr_gate` = [cos yaw, sin yaw, class aperture].
- `next_lookahead` uses the visible form or the unseen form.
- Status is tracking / holding / lost.
- "Next" is the nearest farther candidate, not route-aware.
- It has its own pass rule.
- It holds the last estimate for `lost_timeout_s` (1.0 s), then reports `gate_lost`.
- Before acquisition it targets its own position until `acquire_timeout_s` (3 s).

Detector outputs:
- Aperture is the class nominal, so size error is 0 while there is only one class.
- Yaw comes from the PnP normal.

Timing quirks:
- `last_seen` is the tick time, not the frame's capture time.
- The async path may use an older frame.
- The world transform uses the CURRENT pose for an image taken L seconds earlier. That error (v·L, ω·L·range) is invisible to capture evaluations.

Obstacles and obs12:
- `free_space_estimator: none` gives obstacle_vision zeros only when `assume_no_obstacles: true`.
- obs12 = rel_world, g_body, vel_world, ω_body. It is heading-blind.

## A3. 07 and K1

- `training/07-proc-course-flight/model_readout.pt`:
  - v1 action, extra_dim 30;
  - `pi.0` is (128, 1054);
  - `body.input_projection` (32×12) is frozen.
- `K1ReservoirAug.features()` runs under no_grad, so nothing in front of the body can be trained.
- `reservoir.warm_start(..., src_extra_dim=30, legacy_v1_to_scale_g=...)` is the UC67 v1→v2 path.
- **CPU batch-1 inference measured p50 223 ms, p99 268 ms**, so the K1 student MUST run on CUDA live.
  - 07's live compute was p50 17.3 ms, p95 25.5 ms.
  - The GPU cost wasn't measured while the teacher was training; it is a bench step.

## A4. The simulator

- `MultiCourseEnv.step` decides terminals internally and respawns inside `step`.
- `OUTCOMES` = none, floor, collision, out-of-bounds, timeout, completed.
- UC67 timing defaults: 20–50 Hz, 0–60 ms delay, skip 0–0.35.
- The train tracks have ~52 checkpoint kinds and ~1,000 passages through INVISIBLE volumes. Infinity Loop has arches only.

# Proposed Solution

## D1. `drone-fly train detector-noise` (AC1)

**CLI:**
`train detector-noise --detector <weights> --captures <dir>… [--labels <dir>…] [--live-runs <run>…] [--preset liftoff] [--device auto|cpu|cuda] [--out <dir>] [--latency-only]`

Implemented in `harness/offline/detector_noise.py`, imported lazily from `training/cli.py`.

### Held-out enforcement

Checked per capture:
- If the weights' meta records a holdout window for the capture, only frames inside that window are used (`check_disjoint` is reused).
- If the weights never trained on the capture, all its frames are used.
- Deploy weights trained on the capture, or a split with no holdout, are refused with `NoiseLeakageError`.

Today only `liftoff-gates.pt` with its 324 held-out frames qualifies. There is no in-sample mode at all: the loader refuses `held_out: false`.

### Visibility predicate, shared with the simulator (challenger Minor 3)

ONE function, `camera_visible(gate, pose, camera)`, built on the real `harness/geometry.CameraModel`. A gate is visible only if all of these hold:
- its centre is in front of the camera (z_cam > 0.1 m);
- its projected centre is inside the image;
- at least `min_in_frame_frac` (0.5) of its projected outline corners are inside the image;
- its centre is not inside a HUD mask (normalised rectangles).

D1 selects records with exactly this predicate. The report counts two cases separately: "centre in image but outline below the fraction", and out-of-view frames.

### Per frame

1. Run the real estimator's `detect_frame` with the frame's telemetry state.
2. Match detections to labels by class, within max(1 m, 0.25·range), greedily.
3. Each visible labelled gate becomes a record with camera-frame range, azimuth, elevation and a detected flag. If detected, it also gets:
   - the position error in the line-of-sight frame (radial, tangential-h, tangential-v);
   - the signed yaw error, folded to ±90°;
   - whether the class was correct.
4. Unmatched detections become false-positive records (range, azimuth, elevation).
5. Per-instance sequences give the temporal statistics.

### Fit

`fit_noise_model(records, cfg)` is a pure function.

**Binned tables**, always computed and reported (AC1):
- Bins: camera range [0, 4, 8, 12, 18, 25, 40, max_range], |azimuth_cam| [0, 20, 40, 60]°, elevation_cam, and per class.
- Each bin holds:
  - p_det;
  - bias and covariance of [radial, tangential-h, tangential-v, yaw];
  - the misclassification rate.
- Bins with n < 15 are pooled, bearing first and then range, with `pooled_from` recorded. Extrapolated bins are flagged.

**Parametric fallback**, fitted by maximum likelihood per class:
- p_det = σ(a + b·range + c·edge_distance_norm);
- radial bias = c_r·range;
- diagonal σ_i = s_i·range (yaw σ constant).

The simulator uses `model: parametric` for a class whenever any sim-relevant bin was pooled below n_min, which is the case today. The choice is recorded per class as `model: binned|parametric`, with the reason.

**Temporal:**
- a two-state Markov dropout;
- the mean miss-burst length L_miss(range);
- the AR(1) error correlation time τ.

**False positives:** the rate per frame, plus an empirical (range, azimuth) histogram.

**Size:** `class-nominal`, error 0, plus a confusion matrix.

**Latency:** CUDA-synced inference p50/p95. `--latency-only` reruns just this on the GPU later; fitting can run on CPU now.

**`timing`**, from `--live-runs`, otherwise null:
- tick period p5/p50/p95;
- missed-deadline rate;
- frame age;
- frame period (median gap between distinct `frame_index` values, plus jitter σ);
- frame-skip probability (index gaps > 1);
- the detection-lag distribution;
- observation→action delay.

### Outputs

Written to `artifacts/harness/liftoff/models/<detector-id>/`, where detector-id = `<weights stem>-<sha256[:8]>`. The weights stay where they are.

**`noise_model.yaml`** (schema `detector-noise` v1) contains:
- `detector`: name, sha256, git_head, purpose;
- `captures`: name, map, frames used, window, `held_out: true`;
- `coverage.maps: [LiftoffArena]`;
- `camera`: hfov, resolution, uptilt, normalised masks, `min_in_frame_frac` and `max_range_m`. Taken from the preset and the labels meta, and verified to agree between them;
- tracker parameters: `assoc_radius`, `lost_timeout_s`, `acquire_timeout_s`, `score_threshold`;
- the bins, parametric, temporal, false-positive, latency and timing sections;
- `revision` and `history`;
- `fitted_utc` and `git_head`.

Re-running with more captures refits in place. The previous file is kept as `noise_model.r<N>.yaml`.

**`noise_report.md` / `.json`**, plus `plots/*.png` (matplotlib is installed):
- Plots: p_det vs range, errors vs range, yaw, the burst-length histogram, the false-positive map and latency.
- n per bin, with bootstrap 95 % confidence intervals.
- Pooled and extrapolated bins, and the maps covered.
- A "LOW-n, provisional" banner when there are fewer than 200 detected samples, as today.
- The observed miss-burst distribution next to the chosen `lost_timeout_s`.

### Validation replay

The real `TrackingGateEstimator` replays the held-out sequence, compared against the reference perception (D2) on the same truth trajectory. It compares:
- the target-error distribution;
- hold durations, including out-of-view holds and lost;
- lost events;
- tracker passes against truth crossings (truth comes from the UC69 `scoring.py` RouteProgress + `gate_crossed`).

## D2. Applying the noise in the simulator (AC2)

**`sim/noise_model.py`** (NumPy only): load, validate (refuses `held_out: false`), `to_tables(device)`, and sha256. It never imports harness.

**`sim/perception.py` `GatePerception`**: batched torch, with two slots per env (target and next). Each control tick:
1. **Frame arrival.** A per-env phase accumulator with period T_f, jitter σ, a per-frame skip probability and the detection-lag distribution. All come from `timing`, with defaults when it is null. Latency L is drawn per episode.
2. **Delayed pose**, from a per-env ring of (pos, quat) covering up to 16 ticks.
3. **Frustum.** `camera_visible` (a torch twin of the D1 predicate, parity-tested against `CameraModel`) is evaluated on the delayed pose. **p_det = 0 outside the frustum.**
4. **Markov dropout** on the in-frustum p_det, parametric or binned.
5. **If detected:** estimate = pos_now + R_now·R_{t−L}ᵀ(g_true − pos_{t−L}) + bias + Chol·z, with AR(1) z.
   - Yaw is computed the same way.
   - Aperture = the class nominal.
   - `last_seen` = now.
6. **Hold-last** semantics, copied from `TrackingGateEstimator.update`:
   - hold, and track age;
   - lost after `lost_timeout_s`;
   - before acquisition: target = own position, `curr_gate` = [1, 0, default_aperture], and lost after `acquire_timeout_s`.
7. **Hand-over.** On a truth k advance:
   - target ← the next estimate if known (`last_seen` = now), otherwise the acquire state;
   - the next slot restarts unseen;
   - the unseen form is the bearing to the last-known position, otherwise to the current gate.
8. **False positives:**
   - While the target is unacquired, a false positive can be acquired as the target (sampled from the FP histogram) and is then held.
   - At a measured next-gate mis-association rate, the next slot is replaced by a false positive. This is a documented simplification, quantified by the replay.

**Uncovered types and invisible volumes:** `perception.uncovered: proxy` (default: the arch's model, flagged, optimistic) or `exclude`. Evaluation reports three coverage strata: `covered`, `proxy` and `invisible`.

**Modes:**
- `noise`.
- `exact`: the same tracker semantics and frustum, but every in-frustum frame is detected, with no error and no latency.

There is no `optimistic` mode.

**`MultiCourseEnv`** gets ONE minimal hook, `step(action, terminate=None)`. `gate_lost` is appended to the END of `OUTCOMES`. With `terminate=None` the env is bit-identical to before.

**`student/perception_ref.py`** is a NumPy single-drone reference driven by explicit random draws. It returns a real harness `GateEstimate`.

## D3. Student architecture (AC4): `controller/student.py`

### `K1Student`

**Body:** `K1ReservoirAug`, unchanged. The frozen K1 body sees obs12 in student units: Liftoff metres ÷ k for rel and vel, with g and ω unscaled. That is exactly the harness `WorldScale(k)`.

**Readout:** `pi` over [LayerNorm(tap) ‖ extra(59) ‖ enc].

**`sense_encoder`** (L1, on): an MLP over the 29 non-legacy senses → 64 → 32, ELU. The output layer and its readout columns are zero-initialised.

**`readout_residual`** (L2, optional): a zero-initialised MLP 256–128 over [LN(tap) ‖ extra ‖ enc], added to `pi`.

**Warm start** from 07 via `warm_start(src_extra_dim=30, legacy_v1_to_scale_g=action.collective_scale_g)`. At init the student equals 07 on the legacy 30 columns.

**What trains and what stays frozen** (tested):
- Trainable: `tap_norm`, `pi`, the encoder, the residual and `log_std`.
- Frozen: the body and `input_projection`.

### `MLPStudent`

A comparison baseline only: 256–256–128 ELU over the same 59 dims.

### No-swap rule

- The deployable export is ALWAYS the K1 student.
- `train export` refuses `mlp-student` (`ExportError`) unless `--baseline-export` is given. That flag writes `deployable: false` and `role: baseline`.
- `harness run` refuses a `deployable: false` checkpoint unless `--allow-baseline` is given.
- If the MLP meets the targets and K1 does not, that is reported and escalated, never swapped.

### k (`body_world_scale`)

Chosen a priori before run 1 by the CPU-only `train student --scale-report`.
- Inputs: the TRAIN-split aperture and spacing p10–p90 (`tracks stats --split train`), and the teacher's val speed p50 and p90.
- Each is divided by k for k ∈ {4, 6, 10} and compared with 07's training ranges (the `gpu_prototype` proc-course constants: aperture 0.45–1.5 m, spacing, v_ref).
- The chosen k is the one whose aperture median and speed p50 fall inside those ranges, with the minimum summed log-ratio.
- It is recorded in the ADR parameter-source table. For reference, 07 live used 10 and the preset uses 4.

## D4. `student-obs` v1: 59 float32 (AC3)

| Indices | Block | Contents |
|---|---|---|
| [0:30) | `harness30` | `ContractObservationAssembler.assemble(state_k, GateEstimate_k, FreeSpaceEstimate(zeros), 30)` |
| [30:36) | `attitude6` | R(body→world) columns 0 and 1 |
| [36:39) | `vel_body` | ÷k |
| [39:43) | `prev_action` | raw output, before decode, clip or arm override |
| [43] | `arm_remaining_s` | |
| [44:47) | `perception` | `target_age` = min(age, lost_timeout)/lost_timeout; `target_valid`; `next_age` (same normalisation) |
| [47:59) | `history` | see below |

**`history` block** (12 values):
- EMA (τ 0.25 s, α = 1 − exp(−dt/τ)) of the target relative vector, in body frame ÷k (3);
- EMA of `prev_action` (4);
- **Δ target estimate** (3), computed on the WORLD-frame estimate in student units and then rotated into the current body frame. It is zero while holding, even under pure rotation (challenger Minor 2; tested);
- `ticks_since_detection`, normalised (1);
- the `hand_over` flag (1).

The history resets at episode start and at harness session start.

**Age** = now − last_seen + L_src, where L_src = now − the capture stamp of the frame behind the matched detection. Harness changes for this, with no behaviour change:
- the tracker records the source-frame capture stamp;
- the learned estimator's async path carries (index, stamp);
- `GateEstimate.info` gains `target_age_s`, `next_age_s`, `target_valid` and `hand_over`.

**`sim/student_obs.py`:**
- `VERSION = 1`, `BLOCKS` and `schema()`. The schema pins k, the timeouts, the camera block, `obstacle_vision: zeros`, the assembler source and version, and the requirement for a noise model.
- The batched builder: torch twins of `make_obs`, `curr_gate_block`, `lookahead_visible` and `lookahead_unseen`.

**Parity:**
- `student/obs_ref.py` `build_student_obs(...)` calls the REAL assembler.
- The harness `StudentObservationAssembler` calls the SAME function, so the two are bit-identical by construction; tested with `np.array_equal`.
- The torch twin matches to atol 1e-5. That tolerance is stated; it is not bit-identical.

**The `student/` package** holds the training↔harness contract: references, the checkpoint format and constants. sim never imports harness or `student/`, and the import-graph test is extended to check this.

## D5. `drone-fly train student` (AC5): `training/distill.py`

**CLI:**
`train student --teacher <run|pt> --noise <noise_model.yaml> --drone <id> --tracks liftoff:train --procedural 0.4 [--actor k1-student|mlp-student] [--config student-liftoff] [--budget] [--n-envs] [--resume] [--ppo-steps N] [--feasibility-only] [--scale-report] [--allow-gpu-contention] [--k1-path] [--device]`

### Contention guard

Refuses with exit 2 and names the condition when any of these is present:
- the teacher run is active (UC69 rules);
- Liftoff.exe is running;
- a live `harness run` is running.

`--allow-gpu-contention` overrides this and labels the run.

### Teacher and course pool

- The teacher comes from UC69's `TeacherCheckpoint.resolve`. It is frozen and uses its mean action.
- The course conversion comes from the teacher sidecar's `course_conversion`.
- The pool uses `purpose="train"`.

### Feasibility stage (non-blocking; orchestrator ruling plus challenger Minor 1)

- The teacher is evaluated under exact observations on held-out and val (N=32, seed 1234). `eval/feasibility.json` is always written.
- If held-out completion is below 82.4 %:
  - a WARNING is logged;
  - the report gets a banner: "AC7 absolute target infeasible by construction";
  - the operative gates become ≥ 85 % of the teacher, plus the exact-vs-noise comparison;
  - the run CONTINUES.
- `--feasibility-only` runs this stage alone and exits 0. There is no blocking flag.

### Env: `sim/student_env.py` `StudentCourseEnv`

Step order:
1. decode;
2. physics;
3. truth progress;
4. pose-ring push;
5. perception update;
6. lost check → `terminate` (`gate_lost`);
7. env terminals and log;
8. respawn;
9. perception reset for respawned envs: clear the slots, refill the ring with the spawn pose, redraw the Markov, AR and phase state, zero the history;
10. observation.

It returns both `student_obs` and `teacher_obs`.

### DAgger

- **Rollouts:** 64 ticks × N envs per rollout.
- **Who acts:** a per-step Bernoulli mix. The teacher acts with probability β; otherwise the student's mean plus `explore_std` 0.05.
- **β schedule:** `{start 1.0, end 0.0, schedule linear|exp, over 0.5}`.
- **Labels:** the teacher's mean on the exact `teacher_obs` (prev_action = the executed action), **clipped to the decode range**.
- **Loss:** computed on DECODED quantities (collective_g/max_g, tanh(rates)).
  - On the student side, collective uses a **straight-through clip**, plus λ_raw·relu(out-of-range)² with λ_raw 0.01, so gradients never die (challenger Minor 4).
  - The ADR states this choice.
- **Buffer:** FIFO, 2e6 samples, CPU-pinned. For K1 each sample stores fp16 taps (1024), obs (59) and the label (4), about 2.3 GB in total.
- **Updates:** 8 × 8192 per iteration.

### Mechanics

The same as UC68:
- atomic checkpoints every 600 s;
- `resume.pt` (without the buffer);
- STOP or Ctrl-C to stop cleanly;
- `progress.json`;
- CSV, JSONL and TensorBoard logs.

Checkpoint selection uses the val evaluation in `noise` mode, seed 4321.

### PPO fine-tune (L4, optional via `--ppo-steps`)

- `K1StudentAdapter`, with an asymmetric critic MLP 512–512 over `teacher_obs` ‖ `critic_extra`. The critic is never exported.
- Critic warm-up: 20 iterations.
- A behaviour-cloning term toward the teacher, λ 0.1, decaying to 0 over the first half.
- `log_std` −1.5.
- The `training/ppo.py` teacher loop is generalised with `obs_fn`, `critic_fn` and `aux_loss` hooks. Legacy and teacher behaviour are unchanged.

### Run directory

`artifacts/training/<drone-id>/<run-id>/`, with the same layout as the teacher:
- config and drone;
- checkpoints: `student.pt`, `student_it<N>.pt`, `resume.pt`, each with `.contract.json`;
- logs and eval;
- a `tracks/` snapshot;
- `inputs/`: `teacher.json` plus a frozen `noise_model.yaml`, with its sha and detector-id.

**Sidecar keys:** `obs_schema`, `world_scale` = k, `action_spec` v2, `actor`, `actor_arch`, `teacher`, `noise_model`, `perception`, `camera`, `course_conversion` and `timing`.

### `configs/student-liftoff.yaml`

Extends `teacher-liftoff`:
- course mix: liftoff:train 0.6 (`scene_weight` 2) + procedural 0.4;
- randomization 1;
- `ground_fraction` 0.5;
- slack 0;
- timing from the noise model;
- `perception.mode: noise`;
- `distill` and `ppo` sections.

## D6. Latency fit (AC8)

**Student timing** comes from `noise_model.timing`, otherwise from the config:
- `control_hz` = [p5, p95], about [28, 45] from the 07 runs;
- the measured delay;
- skip in [0, measured].

It is written to the sidecar.

**`liftoff` preset changes:**
- `runner.rate_hz: 40`;
- `free_space.assume_no_obstacles: true`;
- `gates.lost_timeout_s` = the fitted miss-burst p99 at < 25 m, clamped to [1, 3] s.

A preset comment and the docs note that legacy 07/K1 runs now tick at 40 Hz; `--set runner.rate_hz=50` restores 50.

**Harness behaviour:**
- `lost_timeout_s`, `acquire_timeout_s` and world_scale are overridden from the checkpoint, and the overrides are logged.
- A K1 student on `contract.device: cpu` is refused (`StudentDeviceError`) unless `--allow-slow-cpu` is given.
- A live camera config that differs from the checkpoint's camera block is refused (`StudentSchemaError`).

**Export bench:** CUDA batch-1 p50/p99 over 200 passes. The manifest records `latency: {p50, p99, budget_ms 10, pass}`. A failure warns but does not block.

**Test:** the preset's `rate_hz` lies inside the student's `control_hz` range, and the session applies the overrides.

## D7. Export and loader (AC6)

**`train export <run> [--checkpoint val|final|it<N>] --out <path> [--baseline-export]`** writes ONE file:
- `format: drone-fly-student`, `format_version: 1`;
- `state_dict`: the readout only, with `body.layer.*` stripped;
- `meta`:
  - the sidecar, plus the K1 prune provenance sha;
  - teacher and noise provenance;
  - latency;
  - `deployable` and `role`;
  - the required harness parts: the learned estimator, the detector sha, `free_space none` + `assume_no_obstacles`, and the camera.

It also writes a `.contract.json`.

**`harness run --checkpoint student.pt`** detects the format and takes the `harness/student.py` path. `--extra-dim` is forbidden.

**Version guard:** the embedded `obs_schema` must deep-equal `sim.student_obs.schema()`, and the `action_spec` version must be supported. Otherwise `StudentSchemaError` names both versions and the run exits 2 before anything is armed.

**The student path then uses:**
- `StudentObservationAssembler`;
- `StudentStepState`, which extends UC69's `TeacherStepState` (prev_action, arm, history);
- `StudentPolicyRunner`: K1 on CUDA, mean action;
- the checkpoint overrides.

A detector-sha mismatch with the noise model only warns, and is recorded as `noise_model_detector_mismatch`. `--overlay` works with the student.

**Test:** export → `harness run` dry run with fakes → one tick, giving a finite command and the expected meta keys.

## D8. Report (AC7)

`train eval <student-run> [--baseline <mlp-run>]` runs the UC68 per-track engine on `StudentCourseEnv`.

**Settings:**
- modes `noise` (the headline) and `exact`;
- randomization 1, random timing;
- `ground_fraction` 1, slack 0;
- N=32, seed 1234.

**Report contents:**
- per-track held-out completion;
- the teacher (exact) on the same tracks and seeds, with the ratio;
- K1 vs MLP;
- exact vs noise;
- outcome shares, including `gate_lost`;
- burst lengths vs `lost_timeout_s`, and the `gate_lost` fraction;
- scene and coverage strata;
- the feasibility banner.

**Bounds:** teacher-exact, MLP-noise, K1-exact and K1-noise. The gap is attributed to the teacher, to K1's capacity, or to perception.

There is no held-out tuning.

## D9. Order of operations and GPU budget

**Now (no GPU):**
- all code plus the hermetic tests;
- `train detector-noise --device cpu` on the 324 held-out frames (~1–3 min). This produces the provisional noise model;
- `train student --scale-report` (CPU), which fixes k.

**After UC68 training (~06:30), its val selection, and the UC69 live stage:**
1. `detector-noise --latency-only --device cuda` (2 min).
2. The feasibility stage (~10 min).
3. `train bench --student` (5 min). It sets n_envs; estimated 40–120k env-steps/s.
4. Pilot: 2e7 steps (~10 min).
5. **K1 DAgger run 1** (L1 + L3): 2e8 steps, capped at 2.5 h.
6. MLP baseline DAgger, same budget (~30–45 min).
7. Evaluations (~30 min).
8. **Run 2: ONE step**, chosen on VAL:
   - cap_gap = MLP_exact − K1_exact; noise_gap = K1_exact − K1_noise (both as fractions of the teacher).
   - If cap_gap ≥ 0.10 and K1_exact < 0.85 × teacher: L2 (`readout_residual`) DAgger run 2, capped at 2.5 h.
   - Otherwise, if noise_gap dominates: L4 PPO, ≤ 3e8 steps, capped at 3 h.
   - If both gaps are < 0.05: stop and report a teacher or detector limit.
   - The other step is allowed only if its gap persists after the first.
9. Final evaluation.

**Cap:** at most 2 DAgger runs and 1 PPO run. Total about 4 h in the base case, at most ~7.5 h in the worst case.

**L5** (a trainable input adapter in front of the body) is named but not used. The gradient cost through 25.6k neurons on the 3050 is high, and it would break the cached-tap buffer. It is a lever for UC72, and the ADR records why.

## D10. Readiness for UC71 and UC72

**UC71:**
- the noise model is per detector-id, multi-class, multi-map and revisioned, and refits in place;
- each student records the noise model's sha;
- a new detector only triggers a warning.

**UC72:**
- the export format plus `harness run --checkpoint`;
- `resume.pt`;
- the UC69 `scoring.py` is reusable.

Not in UC70: `--track` with the student, and any new scoring code.

# Files Affected

## Production (developer)

**New:**
- `src/drone_fly/sim/noise_model.py`, `sim/perception.py`, `sim/student_obs.py`, `sim/student_env.py`
- `src/drone_fly/student/__init__.py`, `perception_ref.py`, `obs_ref.py`, `checkpoint.py`
- `src/drone_fly/controller/student.py`: `K1Student` (with the encoder and residual), `MLPStudent`, `K1StudentAdapter`
- `src/drone_fly/training/distill.py`, `training/export.py`, `training/configs/student-liftoff.yaml`
- `src/drone_fly/harness/student.py`
- `src/drone_fly/harness/offline/detector_noise.py`: records, the visibility predicate, the binned and parametric fits, the report, plots and the replay

**Modified:**
- `sim/multi_course.py`: the `terminate` hook, plus `gate_lost` appended to `OUTCOMES`
- `sim/__init__.py`
- `sim/action.py`: new sidecar keys
- `harness/geometry.py`: only if a projection helper is needed for `camera_visible`
- `harness/liftoff/gate_cv.py`: the capture stamp, info ages, `target_valid`, `hand_over`
- `harness/liftoff/gate_learned.py`: the async path carries (index, stamp)
- `harness/session.py`: student dispatch, overrides, the camera check, and the baseline and CPU refusals
- `harness/errors.py`: `StudentSchemaError`, `StudentDeviceError`, `NoiseLeakageError`, `ExportError`
- `harness/configs/liftoff.yaml`: `rate_hz` 40, `assume_no_obstacles`, `lost_timeout_s`
- `training/cli.py`, `training/config.py` (`StudentConfig`), `training/ppo.py` (hooks), `training/evaluate.py`, `training/run_dir.py`
- `cli/main.py`

**Untouched:** `controller/reservoir.py`, `sim/course.py`, `sim/teacher_obs.py`, `tracks/*`.

**Docs:**
- `docs/adr/0009-student-distillation-and-detector-noise.md`: the k table, the fallback ladder, L5, the loss choice and the results
- `docs/INFERENCE_CONTRACT.md`: `student-obs` v1 and the export format
- `docs/harness/liftoff-live-session.md`: the student run and the preset changes

No game data, captures, weights or noise models are committed.

## Tests (QA)

All synthetic.

- **`test_detector_noise_fit.py`:**
  - recovery of the binned model: p_det, bias and covariance, burst length, τ, FP;
  - pooling and extrapolation;
  - held-out enforcement;
  - YAML round trip;
  - revisions;
  - report shape.
- **`test_detector_noise_parametric.py`:** the parametric form is recovered.
- **`test_sim_perception.py`** (≥ 1e5 samples, ~4σ tolerance):
  - p_det, burst length, bias and covariance, AR(1), FP acquisition;
  - latency error against the analytic formula;
  - frame phase and skips;
  - hold, lost, acquire and hand-over;
  - `exact` mode.
- **`test_sim_perception_frustum.py`:**
  - a gate out of the frustum (behind, azimuth > 60°, elevation −20°) is never detected;
  - uptilt asymmetry: +40° is visible, −20° is not;
  - a masked centre;
  - the in-frame fraction;
  - the torch frustum matches `CameraModel`;
  - D1 and the simulator use the same predicate function.
- **`test_perception_tracker_parity.py`:**
  - the real `TrackingGateEstimator` vs `perception_ref` on scripted detections;
  - the torch twin vs the reference with injected draws.
- **`test_student_obs_parity.py`:**
  - harness vs reference, `np.array_equal`;
  - torch twin, atol 1e-5;
  - k-scaling equals `WorldScale(k)`.
- **`test_student_history_parity.py`:**
  - multiple ticks;
  - hold;
  - **Δ = 0 under pure rotation while holding**;
  - hand-over;
  - reset.
- **`test_student_env_reset.py`:**
  - no leakage after a respawn (sentinel test);
  - the `gate_lost` outcome;
  - `terminate=None` leaves the env identical;
  - uncovered `proxy` and `exclude`.
- **`test_controller_student.py`:**
  - warm start: legacy columns are copied and the new columns, encoder and residual are zero, so the output equals the source at init;
  - the v1→v2 conversion;
  - the body and `input_projection` are frozen and unchanged after a training step;
  - the MLP actor protocol.
- **`test_training_student.py`:**
  - CPU DAgger for 1–2 iterations;
  - β;
  - teacher labels come from `teacher_obs` and are clipped;
  - the straight-through and raw-penalty gradients are non-zero out of range;
  - the FIFO buffer;
  - the run-dir `inputs/`;
  - resume and STOP;
  - one PPO iteration with critic warm-up and the BC term;
  - a leakage spy;
  - the contention guard.
- **`test_training_feasibility.py`:**
  - the flag, banner and warning are written;
  - training CONTINUES;
  - `--feasibility-only` exits 0.
- **`test_student_checkpoint.py`:**
  - version guard on name, version, dim, k, camera and action_spec → exit 2 with clear messages;
  - an export round trip with fakes, one tick;
  - `--extra-dim` is rejected;
  - the CPU refusal;
  - the detector-mismatch warning.
- **`test_export_no_swap.py`:**
  - MLP export is refused without the flag;
  - with the flag it is labelled `deployable: false`;
  - the harness refuses it without `--allow-baseline`.
- **`test_training_eval_student.py`:** modes, strata, the teacher and baseline columns, the ratio, bounds and the banner.
- **`test_harness_timing_alignment.py`:** the preset's `rate_hz` lies inside the student's `control_hz`, and the session's overrides apply.
- **`test_tracks_import_graph.py`** (extended): `sim.*` loads neither harness nor `student`.

These must stay green and untouched:
- the UC68 and UC69 suites;
- `test_sim_course` and `test_sim_meteor75_equivalence`;
- `test_harness_*`;
- the legacy `--checkpoint` path for extra_dim 12/30/70.

# Risks & Considerations

1. **Detector quality dominates AC7.**
   - 19.6 % detection, about 1 false positive per frame, only 29 error samples, one map.
   - The absolute target will likely be missed, and the feasibility banner may fire.
   - The report attributes the gap. UC71 is the fix, and the user's capture helps.
2. **Same-map optimism:** `coverage.maps` contains LiftoffArena only.
3. **Uncovered and invisible passages** use proxies, which is optimistic. The strata are reported.
4. **The teacher bounds the student:** it was at about 44 % training completion at 34 % of its budget. The feasibility stage checks this.
5. **K1 readout capacity:** bounded by the L2 residual and by the MLP baseline.
6. **K1 needs CUDA live** (CPU is 223 ms). It shares the 3050 with the detector and Liftoff. The export bench checks the budget, and UC72 re-measures.
7. **Remaining perception approximations:**
   - "next" is not route-aware;
   - false-positive geometry;
   - the tracker's own pass rule.

   The replay quantifies them, and the frustum covers out-of-view dropouts.
8. **k is chosen a priori** by the scale report.
9. **Bit-identity** holds only between the harness and the reference. The torch twin is tolerance-tested.
10. **GPU budget:** about 4 h, at most ~7.5 h, strictly after UC68 training and the UC69 live stage. Guarded by the contention guard.
11. **The single `MultiCourseEnv` hook** is append-only and identity-tested.
12. **The preset change to 40 Hz** also slows legacy runs; documented.
13. **The 59-dim schema** means more zero-initialised inputs at warm start. Harmless, and tested.
14. **Use-case pitfalls covered:**
    - Infinity Loop only: coverage is reported, and the model refits;
    - held-out only: `NoiseLeakageError`;
    - the 30-dim target: obstacle_vision is zeros;
    - `gate_lost`: terminated in the sim, with the timeout fitted from measured bursts.

# Challenger Minors folded in (the developer MUST apply these)

1. The non-blocking feasibility gate (D5).
2. History Δ is computed on the world-frame estimate and rotated into the body frame; it is zero while holding (D4, tested).
3. The same visibility predicate is used in the fit and in the simulator (D1, D2, tested).
4. A straight-through clip plus a raw out-of-range penalty on the student side, with teacher labels clipped (D5, ADR).
