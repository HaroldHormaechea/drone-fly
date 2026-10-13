# ADR 0009: A deployable student distilled from the teacher under a measured detector-noise model

- **Status:** accepted (UC-70). The code and the provisional noise model are in; the GPU stages
  (latency, feasibility, runs, evaluations) and their results are pending (§ Results).
- **Date:** 2026-10-13
- **Builds on:** [ADR 0007](0007-track-library-and-teacher.md) (the teacher, `teacher-obs` v1, the
  multi-course env), [ADR 0008](0008-oracle-teacher-in-liftoff.md) (the oracle session and
  `scoring.py`), [ADR 0003](0003-real-liftoff-calibration.md) (the learned gate detector, the
  auto-labelled capture), [ADR 0006](0006-profile-driven-batched-sim.md) (timing, action contract v2)
- **Code:** `src/drone_fly/sim/` (`noise_model`, `perception`, `student_obs`, `student_env`, the
  `MultiCourseEnv` terminate hook), `src/drone_fly/student/` (`perception_ref`, `obs_ref`,
  `checkpoint`), `src/drone_fly/controller/student.py`, `src/drone_fly/training/` (`distill`,
  `export`, `configs/student-liftoff.yaml`), `src/drone_fly/harness/` (`student`,
  `offline/detector_noise`, `liftoff/gate_cv` + `gate_learned` ages, `session`,
  `configs/liftoff.yaml`)

## Context

The teacher sees exact gate poses. Live, the harness only has the learned detector's estimates:
noisy, biased with range, often missing, and late. The user's goal is that **their own connectome
model** (the 07 K1-reservoir lineage) flies Liftoff. So the deployable model is the K1 reservoir
actor, warm-started from 07 and distilled from the teacher in the simulator under a noise model
**measured** on the real detector, with an observation that the live harness can assemble exactly.

## Decision

### 1. Measure the detector (`train detector-noise`)

- **Held-out frames only.** A capture the weights trained on contributes only its recorded held-out
  window (`check_disjoint`); deploy weights or a split without a held-out window raise
  `NoiseLeakageError`. There is no in-sample mode. Today only `liftoff-gates.pt` qualifies (324
  held-out frames of `2026-10-11-infinity-manual`).
- **One visibility predicate** (`sim.noise_model.visibility`, duck-typed over the real
  `CameraModel`): centre in front, centre in the image, at least half the aperture corners in the
  image, centre outside the HUD masks, range at most 60 m. The fit selects records with it; the
  simulator applies its torch twin (parity-tested).
- **Records:** the real estimator's `detect_frame` per frame (Liftoff metres), same-class greedy
  matching within `max(1 m, 0.25 x range)`; errors in the line-of-sight frame (radial,
  tangential-h, tangential-v) and the signed yaw error folded to +-90 deg.
- **Fit** (pure): binned tables (range x |azimuth| x elevation, pooled bearing-first then range
  below n = 15, flagged), a parametric maximum-likelihood fallback, a two-state Markov dropout with
  mean miss-burst seconds per range bin, an AR(1) correlation time, the false-positive rate +
  (range, azimuth) histogram + next-gate mis-association rate, class-nominal size, inference
  latency and the live frame / tick timing of `--live-runs`.
- **Output** next to the weights, `<stem>-<sha8>/noise_model.yaml` (revisioned in place, the
  previous kept as `noise_model.r<N>.yaml`), `noise_report.md/.json`, plots, and a **validation
  replay** (the real `TrackingGateEstimator` over the recorded detections vs. the reference
  perception on the same truth route).

### 2. Apply it in the simulator (`sim.perception.GatePerception`)

Two slots (target, next) per env with the tracker's semantics, driven by the TRUE route progress:
hand-over on a truth crossing, a frame clock (period, jitter, skips), the detection latency drawn
per episode and applied as a delayed pose from a 16-tick ring, the frustum, the Markov dropout,
the AR(1) error re-placed every tick with the CURRENT pose (the harness's own "current pose for a
late image" transform), hold / lost / acquire, false positives. `exact` mode keeps the tracker,
frustum and frame clock but detects every in-frustum frame with no error, no latency and no false
positive. `student.perception_ref` is the NumPy single-drone twin driven by the same explicit
draws.

### 3. The student (`controller.student.K1Student`)

The frozen K1 body sees the observation's first 12 values in student units (`WorldScale(k)`); the
readout sees `[LayerNorm(tap) || 59 values || enc(29 senses)]` and starts equal to 07 on the
legacy 30 columns (v1 -> v2 collective conversion). Trainable: `tap_norm`, `pi`, the sense encoder
(L1), the optional residual (L2), `log_std`. Frozen: the body incl. `input_projection`.
`MLPStudent` is a comparison baseline only (the **no-swap rule**: `train export` refuses it without
`--baseline-export`, which labels it `deployable: false`; `harness run` refuses that without
`--allow-baseline`).

### 4. `student-obs` v1 (59 float32)

`harness30` (the real `ContractObservationAssembler`, `obstacle_vision` zeros), `attitude6`,
`vel_body`, `prev_action` (raw), `arm_remaining_s`, `perception` (ages normalised by
`lost_timeout_s`, target valid), `history` (EMAs, the world-frame target change rotated into the
body — zero while holding — ticks since a match, the hand-over flag). Age = now minus the capture
stamp of the frame behind the last match: the tracker now records that stamp (the async learned
estimator carries its frame's). `student.obs_ref.build_student_obs` is called by both the
training reference and the harness assembler (bit-identical by construction); the torch builder
matches to 1e-5.

### 5. DAgger (`train student`)

Teacher frozen, mean action on its exact observation; beta 1 -> 0 linearly over half the budget;
labels clipped to the decode range; the loss on DECODED quantities with a straight-through
collective clip plus `0.01 x relu(out of range)^2`, so student gradients never die; a 2e6-sample
CPU FIFO (K1 stores fp16 taps); 8 x 8192 updates per 64-tick rollout. Checkpoint selection on val
in noise mode (seed 4321). A non-blocking feasibility stage evaluates the teacher on held-out and
val first. A contention guard refuses (exit 2) while the teacher trains, Liftoff runs or a live
harness run runs.

### 6. Latency fit and the preset

The student's control timing comes from the noise model's measured live timing. The `liftoff`
preset is aligned: `runner.rate_hz: 40`, `free_space.assume_no_obstacles: true`,
`gates.lost_timeout_s: 1.3`. A student checkpoint overrides the world scale, the tracker timeouts
and association radius, and the free-space settings, and refuses a different camera, a
non-learned detector and (K1) the CPU.

### 7. Export and the version guard

`train export` writes one `drone-fly-student` v1 file (readout state + meta) and its sidecar;
`harness run --checkpoint student.pt` detects it, forbids `--extra-dim`, and refuses (exit 2,
before arming) any schema / action-contract / format mismatch with a message naming both versions.

## Parameter sources

| Parameter | Value | Source |
|---|---|---|
| Student world scale `k` | **6** | Scale report 2026-10-13 (`student-scale-report.md`): TRAIN-split aperture p50 3.69 m, spacing p50 30.0 m against 07's ranges (aperture 0.45-1.5 m, spacing 1-5 m). No k is inside both: k = 6 gives aperture 0.61 (in) and spacing 5.01 (out by 0.01); k = 10 gives aperture 0.37 (out) and spacing 3.00 (in). The mechanical rule (minimum summed \|log ratio\|) picked **10 by 5e-4** (1.096 vs 1.096). **Overridden to 6** (orchestrator, 2026-10-13): apertures out of the frozen body's distribution (p50 below 07's minimum, p10 0.22) matter more than spacing, which only changes time between gates. No teacher speed was used (no val-selected teacher yet); the report is re-run with teacher speeds once one exists, and a clearly different k is reported before run 1, not switched silently. References: 07 flew live at 10; the preset uses 4. |
| `lost_timeout_s` | 1.30 s | Noise model r5: miss-burst p99 below 25 m on the held-out frames, clamped to [1, 3] s |
| `acquire_timeout_s` | 3.0 s | The tracker default (unchanged) |
| Control rate range | [26.7, 51.2] Hz | Live runs 231059, 232549, 232606, 233038, 233054, 233116 (07, 2026-10-11): p5 of the 8-tick rolling effective rate, p95 of the on-deadline tick rate. The sim's nominal rate; missed deadlines are its skipped ticks |
| Skip probability | [0, 0.497] | Same runs: missed-deadline rate 49.7 % |
| Detection latency | [0.046, 0.127] s | Same runs: frame age + detection lag frames x frame period, p5-p95 |
| Frame period / jitter / skip | 20 ms / 8.7 ms / 0.40 | Same runs: distinct `frame_index` gaps |
| `runner.rate_hz` | 40 | Inside the control range; legacy 07 / K1 runs on this preset now also tick at 40 Hz (overlay `runner: {rate_hz: 50}` via `--config` to restore) |
| FP acquisition | P(wrong candidate) 0.73, P(pick wrong \| target detected) 0.15 | Validation replay fit (§ Deviations 2) |
| Feasibility threshold | 82.4 % | 0.70 / 0.85: below it the absolute AC7 target is unreachable at 85 % of the teacher |
| DAgger beta | 1 -> 0, linear, over 0.5 of the budget | A priori |
| Loss | decoded MSE + 0.01 x relu(out of range)^2 | A priori (challenger Minor 4) |
| EMA tau / ticks norm | 0.25 s / 50 | A priori |
| Sense encoder | 29 -> 64 -> 32 ELU | Plan D3 |

## Provisional noise model (r5, `liftoff-gates-161f7a83`)

Fitted on CPU on 324 held-out frames of one map (LiftoffArena); **LOW-n, provisional**.

- 549 visible records, 191 detected (34.8 %); 4-25 m: 29/148 = 19.6 % (reproduces UC64).
- Parametric (bins pooled): P(det) = sigmoid(-2.27 + 0.027 range + 1.37 edge) — detection RISES
  with range (0 % below 8 m); position sigma = [0.128, 0.042, 0.027] x range; yaw sigma 52 deg.
- Markov P(miss | det) 0.29, P(miss | miss) 0.84; mean miss burst 0.57 s; AR(1) tau 0.31 s.
- False positives 1.03 per frame; next-gate mis-association 0.57.
- Latency: CPU only so far (98 ms p50, not representative): rerun `--latency-only --device cuda`.

## Deviations from the approved plan

1. **Tracker association gating in the simulated perception.** The replay showed the plain
   truth-associated reference far more optimistic than the real tracker, which only re-associates
   a detection within `assoc_radius x max(1, aperture)` (in training scale) of the held estimate.
   `PerceptionConfig.assoc_limit = k x assoc_radius x max(1, nominal / k)` (course units) is
   applied in both the torch module and the NumPy reference.
2. **False-positive acquisition fitted, not 0.5.** On each held-out frame the tracker's own
   acquisition rule (`min distance x (1 + off_axis)`) is applied to the recorded detections:
   P(a wrong candidate — false positive or another gate — is present) = 0.73 (238/324, CI
   [0.68, 0.78]); P(the rule picks wrong | the target is detected too) = 0.15 (8/55, CI
   [0.08, 0.26]). Below n = 30 the conservative upper Wilson bound is used. Recorded in
   `noise_model.yaml` (`acquisition`). It did **not** close the replay gap (tracking share 0.37 in
   the reference either way vs. 0.03 for the real tracker; see § Known gap).
3. **Sense-encoder initialisation.** Only the encoder's readout columns start at zero. Zeroing its
   output layer too (as written in the plan) would leave both with a zero gradient forever.
4. **PPO fine-tune in `training/distill.py`**, not by generalising `training/ppo.py` with hooks, so
   the running UC68 teacher loop stays untouched.
5. **Route starts kept** (the teacher's 30 % of air starts before a route passage); perception
   starts unacquired there, as after a respawn.
6. **`perception.uncovered: exclude`** drops courses with proxy / invisible passages from the pool
   (`proxy`, the default, flies them with the measured class's model and reports the stratum).

## Known gap: the live tracker is worse than the simulated one

The validation replay (324 held-out frames, 8 truth crossings) compares the real
`TrackingGateEstimator` on the recorded detections with the reference perception on the same
route: **2 tracker passes vs. 8 truth crossings**, tracking share 0.03 vs. 0.37, target-error
median 43.6 m vs. 20.4 m, and a next-gate mis-association rate of 0.57. The real tracker acquires a
wrong detection early, its own pass rule rarely fires, and hand-overs go to mis-associated next
gates. The simulator keeps **truth-driven hand-over** (decided 2026-10-13): a truth-driven sim
cannot reproduce tracker-logic failures, and an n = 8 pass-miss probability would mostly end
every episode as `gate_lost`. So **the student trains with a better tracker than the live one.**
**Top follow-up for UC72 (or a new use case after it): the live tracker's pass / hand-over logic**
(route-aware next selection, a pass rule that matches the sim's).

## Fallback ladder (D9)

- **L1** (on): the sense encoder. **L3**: DAgger run 1 (2e8 steps, at most 2.5 h).
- After run 1 and the MLP baseline, ONE step chosen on val: `cap_gap = MLP_exact - K1_exact`,
  `noise_gap = K1_exact - K1_noise` (fractions of the teacher). If `cap_gap >= 0.10` and K1 exact
  is below 85 % of the teacher: **L2** (the readout residual) DAgger run 2. Otherwise, if the noise
  gap dominates: **L4** (PPO fine-tune, at most 3e8 steps). Both gaps below 0.05: stop and report a
  teacher or detector limit.
- **L5** (a trainable input adapter in front of the frozen body) is named but not used: back-prop
  through 25.6k neurons on the RTX 3050 is expensive, and it would invalidate the cached-tap
  buffer. It is a lever for UC72.

## Results

Pending the GPU stages (after UC68's training and val selection and UC69's live stage): CUDA
latency, feasibility, bench, pilot, K1 DAgger run 1, the MLP baseline, evaluations, run 2 and the
final evaluation. The report (`train eval <student run> --baseline <mlp run>`) gives per-track
held-out completion in noise (headline) and exact modes, the teacher on the same tracks and seeds,
the ratio, outcome shares incl. `gate_lost`, coverage strata and the gap attribution.

## Consequences

- Every detector version carries its own noise model (per detector id, revisioned, multi-class
  and multi-map ready for UC71); each student records the noise model's sha and detector id, and a
  different live detector only warns.
- Legacy runs on the `liftoff` preset now tick at 40 Hz and get `assume_no_obstacles: true`.
- The noise model is LOW-n from one map; the user's extra hand-flown capture (never trained on)
  refits it in place.
