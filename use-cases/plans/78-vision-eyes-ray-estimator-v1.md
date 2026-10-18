---
plan_for: use-cases/78-vision-eyes-ray-estimator-v1.md
work_branch: feat/uc-78-vision-eyes-ray-estimator-v1
run: drone-fly-uc-78
approved: 2026-10-16
---

# UC-78 implementation plan: vision "eyes" v1

Challenger (challenger-78) verdict: **APPROVED** after 2 rounds. v1 had 3 Major issues, all resolved in v2, and 6 Minor ones, also resolved:
- **MAJ1:** the deployable path imported `drone_fly.sim`, which loads `sim.pool`.
- **MAJ2:** the 4 m range-vs-speed pitfall was not measured.
- **MAJ3:** train/live parity of the ego and ray inputs was not pinned.

Plan approval was waived by the user for this autonomous night run. Orchestrator rulings:
- labels follow the training raycast (obstacles only); the AC2 hit kind is delivered as the diagnostic `aux_kind` channel;
- disk is not tight (~89 GB free after the captures download).

Verified in code by the challenger:
- the training raycast hits only obstacle cylinders and AABBs (`sim/course.py:615-653`, `gpu_prototype/inference.py:161`);
- `harness.geometry` imports only numpy;
- CourseFrame = (p − origin)/W and is never rotated (`sim/pool.py:67-91`);
- `load_capture` shares `converter_from_config` with the live state source (`offline/capture.py:160`).

## Binding developer notes
- **(a)** The raycast position is `state.pos − origin/W` (FlightState is already divided by W). Never apply `to_course_point` to it. The W=4 equivalence test guards this.
- **(b)** The `gravity_body` parity test must pin the sign.

## Analysis
- **Rays.** `contract.raycast` (`gpu_prototype/inference.py:161`) and the `sim/course.py:592-653` port define 40 Fibonacci-cone rays (±45°, RAY_RANGE 4).
  - The cone is centred on the world velocity when speed > 0.3 m/s, else on the target gate, else +x. Roll uses world up.
  - Rays hit only obstacle vertical cylinders and AABBs, never gates or the floor. Values are normalised to [0,1], 1 = clear.
- **Liftoff obstacles.** They come from `tracks.loader.load_course` → `to_harness_course(course).raycast_obstacles()`, the same list the UC69 oracle uses. The full geometry (`course.prims`, floor/heightfield) feeds only the diagnostic channel.
- **Scale.** The preset sets world_scale W (4.0 in liftoff.yaml) and ray_range_scale s. s is honoured by dividing lengths by s before `contract.raycast`. Liftoff metres = free * RAY_RANGE * s * W.
- **Camera.** 2560x1080, hfov 120, uptilt 20. Between 24 and 37 of the 40 rays are in view, depending on pitch, so the in-view set depends on attitude and velocity.
- **Captures.**
  - `load_capture` pairs frames within 30 ms. UC71 `meta.json` is read by a minimal, schema-compatible reader, because UC-71 is not on this branch.
  - Replay captures have gyro=0 and 10 Hz interpolated poses.
  - Captures run at ~10-12 fps, so history is selected by time stride.
- **Catalogue split.** `tracks/liftoff.yaml` `maps.<Map>.split`, with TrackLeakageError.

## Proposed Solution: new package `src/drone_fly/eyes/`

### Deployable modules
They import only numpy, torch, `harness.geometry`, `harness.types` and each other. No `drone_fly.sim` at all.
- **B2 `camera_rays.py`:**
  - world ray basis via `harness.geometry.ray_directions` with a frozen CONE namespace built from the weights meta;
  - `CameraModel.world_to_camera` / `project` / `in_image`;
  - per ray: camera-frame direction, (u,v) in [−1,1], and `in_view` (in front, in image, outside the mask rects).
- **B2b `history.py`:** picks the history frame nearest to t − kΔt within ±40 ms, in the same segment. Shared by the dataset and the estimator.
- **B4 `ego.py`:** a 10-vector:
  - gravity in body, R^T(0,0,−1);
  - body velocity R^T v (training-scale);
  - speed;
  - body rates from q(t−Δt)^−1 q(t) over the actual dt. Gyro is not used.
- **B5 `model.py`:** `EyesNet`.
  - Input: K=3 frames at Δt=0.1 s, masked, RGB, 192x80, stacked as 3K channels.
  - Backbone: 5 stride-2 conv stages (32-48-64-96-128, BN+ReLU) plus global average pooling.
  - Per-ray query: `grid_sample` of features at (u,v), zeroed when out of view; the global vector; ray_cam(3); in_view(1); ego(10).
  - A shared 256-128 MLP outputs free (sigmoid) plus a confidence logit.
  - Under 1.5 M params.
  - `build_from_meta(meta)`. The meta stores input size, K, Δt, cone, camera, W, s, label-settings hash, dataset manifest sha and split maps.
- **B6 `infer.py`:** `EyesEstimator`.
  - `from_weights(path, device)`.
  - `update(frame, state, target)` returns free(40), conf(40), in_view(40), `history_ok` and stamp.
  - When history is missing (startup or a gap), `history_ok=False`, conf=0 and free=1. Frames are never duplicated.
  - `reset()` is called on a discontinuity.
  - Wrapping it as a registered free_space_estimator part is out of scope.

### Offline modules
- **B3 `fov.py` (AC1):** static sweep over pitch 0..60 in 5° steps, ±15° flight-path, and the low-speed target case.
  - Per-ray always / never / attitude-dependent over the 10-40° band.
  - Optional empirical per-capture in-view fractions.
  - Writes `fov_report.json` and `.md`.
- **B7 `capture_meta.py`:** UC71 `meta.json` reader (catalogue_id, map, camera, alignment.{verified, lag_ms}, pose_source).
  - Capture discovery at depth 1-2.
  - Capture sha.
  - Legacy captures only via `--catalogue-id` / `--lag-ms`; otherwise skipped as no-meta.
- **B8 `labels.py` (AC2):**
  - `load_capture(..., world_scale=W)` with the preset telemetry config, so each FlightState is identical to what the live state source produces. `ConversionConfig.world_scale` must equal W, otherwise labelling is refused.
  - Ego, rays_cam, uv and in_view come from that FlightState via the deployable functions.
  - The raycast position is `pos − origin/W` (a pure translation). The target passage comes from RouteProgress + gate_crossed per segment.
  - free/dist_m/hit_kind come from `contract.raycast` with the s trick, so the label is obstacles only.
  - A diagnostic pass against the full geometry (OBB slab, capsule, floor/heightfield march) gives `aux_free` and `aux_kind` (none / obstacle / gate / scene / item / floor).
  - Per frame it stores: fwd_source, pair |Δt|, speed_mps, range_m, warning_s, the hit obstacle index and its minimum half-extent.
  - It skips captures with a reason (no meta, unconvertible, or no geometry).
  - Writes `eyes_labels.npz` + `eyes_labels_meta.json`. The meta holds:
    - capture sha, catalogue id/map, conversion fingerprint;
    - settings and their sha256 (W, s, RAY_RANGE, N_RAYS, cone sha, lag_ms, pair_tol, contract sha);
    - `hit_rule` (label = training raycast, obstacles only; AC2 hit kind = aux_kind);
    - both world-scale values, counts and drop reasons;
    - alignment p50/p95/max;
    - hit-kind breakdown.
  - Optional `--preview N` (≤ 20, OpenCV).
- **B9 `dataset.py` (AC3):**
  - The split comes from catalogue `maps.<map>.split`, never from meta. A map mismatch is refused.
  - Unverified alignment is excluded. An uptilt mismatch over 2° is train-only on train maps.
  - Leakage guard: TrackLeakageError.
  - Samples need K−1 history frames, otherwise no-history.
  - Resumable frame cache (uint8 `.npy` memmap per capture, keyed by sha + size + mask) with a projected-size disk guard.
  - `manifest.json`/`.md`: per map and split, counts, hit shares, label vs aux kinds, drop reasons, label hash (must agree across captures), catalogue sha, warning-time p05/p50/p95, and the share with warning < 0.3 s and < 0.5 s.
- **B10 `train.py` (AC4):**
  - AdamW + cosine, AMP on CUDA, photometric and blur augmentation, no flips.
  - Loss: smooth-L1 with w_hit=4 on hit rays, plus BCE on confidence vs |err| < tol_m (0.5 training m).
  - Atomic `resume.pt` with RNG states; STOP file / Ctrl-C.
  - Outputs `eyes.pt`, `eyes.meta.json` and `training_log.json`.
  - `--resume` refuses another dataset. `--device cpu` works for the smoke run.
- **B11 `evaluate.py` (AC6):**
  - Refuses train maps.
  - Per ray and aggregated: median abs error (training m and Liftoff m), miss rate and false-alarm rate.
  - Every figure is split in-view vs out-of-view and by distance bin.
  - Calibration over 10 bins plus ECE, overall and per distance bin.
  - Per-map and obstacle-size bins (thin < 0.2 Liftoff m / medium / large).
  - Warning-time stats, plus the miss rate on frames with warning < 0.5 s.
  - Baselines: always-clear and the train-mean per ray.
  - Writes `eval_<split>.json` and `.md`.
- **B12 `bench.py` (AC7):**
  - Times `EyesEstimator.update` on synthetic 2560x1080 frames: preprocess, forward and total, as p50/p95/p99, against the 25 ms budget.
  - Optional interleaving with the learned gate detector (skipped with a reason if its weights are missing).
  - Peak CUDA memory for the eyes alone and for eyes + detector (max_allocated and reserved); these fields are null on CPU.
  - Explicit note: the connectome footprint is not measured (follow-up).
  - Writes `bench.json` and `.md`.
- **B13 `cli.py`:** `add_parser(sub)` with fov-report, label, dataset, train, eval and bench. Registered in `cli/main.py` next to tracks.
- **B14 `docs/experiments/uc78-vision-eyes.md` (AC9):** inputs and outputs, velocity-aligned rays, the label rule vs the diagnostic kinds, looming, time stride and derived rates, out-of-view handling, scale, the warning-time evidence for UC-73 AC10, known limits (blur, untextured walls), the commands, GPU and disk notes, and an empty Results section.

## Files Affected

**Production (developer):**
- `src/drone_fly/eyes/`: `__init__`, camera_rays, history, ego, model, infer, capture_meta, fov, labels, dataset, train, evaluate, bench, cli
- `src/drone_fly/cli/main.py`
- `docs/experiments/uc78-vision-eyes.md`

**Tests (QA, CPU, hermetic):**
- `tests/eyes_support.py`
- `test_eyes_fov.py`: projections, sweep counts, noise_model parity, gravity_body parity (sign pinned).
- `test_eyes_labels.py`: box and cylinder within 5 cm; W and s; floor and gates never shorten the label; target fallback; skip reasons; hash stability; label vs live equivalence at W=4, s=1.5; warning_s.
- `test_eyes_dataset.py`: split from the catalogue; leakage; exclusions; no-history; manifest; disk guard; warning percentiles.
- `test_eyes_model.py`: shapes; out-of-view features zeroed; meta round-trip; param budget.
- `test_eyes_train_smoke.py`: 2 epochs; bit-identical resume (num_workers=0, deterministic, per-epoch seed); refuses another dataset.
- `test_eyes_infer.py`: history selection, reset/gap, a tiny end-to-end run.
- `test_eyes_vision_only.py`: static AST plus a fresh-interpreter check of update, with no drone_fly.sim / tracks / harness.offline / oracle / contract / course / loopback / liftoff modules or offline eyes modules loaded; signature checks.
- `test_eyes_eval_metrics.py`: metric maths, bins, ECE, per-map / size / warning breakdowns, train-map refusal.
- `test_eyes_bench.py`: CPU 3-iteration run; memory fields null with a reason; detector skip.
- `test_eyes_cli.py`
- `test_eyes_docs.py`

## Risks & Considerations
1. The label rule is obstacles only, per the orchestrator's ruling. Report it to the user.
2. UC-71 is not on this branch; unify later.
3. Replay captures have gyro=0 (handled) and finite-difference velocity (documented).
4. A low-speed target-basis case exists; fwd_source is recorded.
5. The 8-value obstacle_vision block is not produced (follow-up).
6. The cache-size guard exists (~89 GB free).
7. Val or held-out maps may have no captures; this is reported as "no frames".
8. Monocular range is hard; v1 is a measured baseline against always-clear.
9. Label fidelity: covered by the aux channel and previews.
10. Latency is dominated by preprocessing; the bench measures it.
11. Tests use the harness_contract fixture.
12. A future heavy builtins import would be caught by the probe.
