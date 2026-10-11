---
plan_for: use-cases/62-liftoff-vision-harness.md
work_branch: feat/uc-62-liftoff-vision-harness
run: drone-fly-uc-62
approved: 2026-10-11
---

# UC-62 Approved Implementation Plan — pluggable live-harness framework + vision-driven Liftoff harness

Challenger verdict: **APPROVE** (round 2). Round 1 → 2 Major (world_scale inconsistency; yaw_mode scope creep altering the contract's action) + 8 Minor; v2 resolved all ten. No forbidden-path edits; no contract reimplementations.

**Needs user sign-off (pre/during implementation):**
- `pyproject.toml`: `drone-fly` console script + `harness` optional extra.
- `uv.lock`: plain `uv lock`, no `--upgrade` (stale `app` extra drops out; re-run `uv lock` on rebase, don't hand-merge).
- `docs/adr/0002-live-harness-layering.md`: new (docs/ not in the target CLAUDE.md developer allow-list).
- `README.md`: "Live harness" section.
- `tests/test_smoke.py`: drop train/evaluate/cli from `test_subpackages_import` (red at baseline — cc817f4 removed them); QA records the full baseline suite first.
- AC5 deviation: telemetry fixtures are synthetic with the identical CSV layout, not "cut from the existing recorded CSVs".

TARGET_DIR = C:\dev\drone-fly-uc-62-liftoff-vision-harness (branch feat/uc-62-liftoff-vision-harness, base cc817f4). paths.production = src/drone_fly/**, paths.test = tests/**, profiles = [].

## Hard constraints
- Never edit gpu_prototype/, training/, docs/INFERENCE_CONTRACT.md, USE_CASES.md, PROJECT_BRIEF.md, or any training code/config/artifacts.
- Reuse the contract by reference; never reimplement: load_policy (strict=False + body.layer.* assert), Policy.act, make_obs, plant_step, raycast, init_state, quat_rotate/quat_rotate_inv, CONE, N_RAYS, RAY_RANGE, MAX_BODY_RATE, MAX_THRUST, HOVER_THR, DT.
- Do not reuse legacy `drone_fly.adapter` (its throttle/Euler semantics contradict contract §2).
- CI hermetic: importing `drone_fly.harness`, the CLI and `harness list` must not import torch, cv2, mss/dxcam, transformers or vgamepad, and must not touch gpu_prototype. Heavy imports are lazy and raise `HarnessDependencyError` naming the package and `uv sync --extra harness`.

## Key facts
- `inference.py` lacks curr_gate / next_lookahead / obstacle_vision; they exist only as batched torch code in gates_proc_env.py (extra() L329-335, _next_lookahead L234-259, _obstacle_vision L218-232, pass rule L361-366).
- Import traps: the gpu_prototype modules are top-level siblings, not a package. The K1 path is hardcoded: `reservoir_aug.K1_PATH` is an overridable module global, but `gates_k1_reservoir` uses a literal, so the pure reservoir is refused. `hover_gpu.dev = "cuda"`, and gates_gpu, gates_k1_reservoir and reservoir_aug bind `dev` by value at import; gates_gpu and gates_proc_env build tensors at import. A pre-import override is required for CPU-only CI.
- The model is stateless per tick. `artifacts/` (the K1 connectome) is absent from the worktree: live runs need `--k1-path`, and CI uses the 322-neuron fixture connectome `tests/fixtures/mcns_fixture.npz`.
- Target model: `training/10-proc-obstacle-avoid/model_readout.pt` = K1ReservoirAug(extra_dim=70), trained with PROC_TRACTABLE + PROC_RAYCAST.

## Stage A — framework, contract bridge, sim-loopback parity (AC1-4, AC10 core, AC11, AC12 list/run-loopback, AC14)

`src/drone_fly/harness/`:
- `errors.py`: HarnessDependencyError, HarnessConfigError, PerceptionCoverageError (carries missing block names).
- `types.py`: frozen dataclasses with float32 numpy vectors: TelemetrySample (+ receive monotonic time); FlightState (contract world frame, Z-up right-handed: pos, vel, quat [w,x,y,z], body omega, t, received_at, discontinuity flag, plus the state-dict shape pos/vel/quat/omega); Frame; GateEstimate (target, curr_gate(3), next_lookahead(7), gate_index, lap, status tracking/holding/lost, stamp); FreeSpaceEstimate (obstacle_vision(8), raycast(40) or None, in_fov_mask(40), stamp); TickRecord.
- `interfaces.py`: ABCs with documented threading contracts: StateSource (start/stop/latest/age), FrameSource, GateEstimator (`provides` ⊆ {target, curr_gate, next_lookahead}; update(frame, state, now); reset), FreeSpaceEstimator (`provides` ⊆ {obstacle_vision, raycast}; update(frame, state, target, now); reset; optional start/stop), ObservationAssembler, ActionSink (send/failsafe/close), PolicyRunner (load/act/extra_dim), AbortSignal, DepthModel. Harness is a composition object built by a factory from config, never subclassed per target.
- `registry.py` + `builtins.py`: per-kind name registry (harness, state_source, frame_source, gate_estimator, free_space_estimator, action_sink). Entries are lazy "module:attr" strings; resolve() imports on demand and turns ImportError into HarnessDependencyError. Programmatic `register(kind, name, "module:attr")`; no entry-point group. AC13: implement + register, no runner edits.
- `contract.py`: the ONLY gpu_prototype toucher, read-only. `load_contract(gpu_prototype_dir, k1_path, device)`:
  1. Validates paths with actionable errors.
  2. Inspects sys.modules for hover_gpu, gates_gpu, gates_k1_reservoir, reservoir_aug, gates_proc_env, course_gen and inference; raises HarnessConfigError naming the module if any is already imported with a different `dev`.
  3. Repeat call with the same device and paths → idempotent no-op returning the cached facade.
  4. Inserts the dir into sys.path, pre-imports hover_gpu and sets `dev` (default cuda if available, else cpu; cuda requested without CUDA → cpu with a warning).
  5. Imports reservoir_aug and sets K1_PATH, then imports inference.
  6. Collision guard: modules must resolve inside the configured dir.
  7. Returns a ContractModule facade (re-exports by reference, exposes the resolved device). `import_reference(name)` is the only path for gates_proc_env and course_gen.
  - Path resolution — gpu_prototype_dir: CLI → config → env DRONE_FLY_GPU_PROTOTYPE → repo-root walk-up. k1_path: CLI → config → env DRONE_FLY_K1_PATH → <repo-root>/artifacts/pruned/k1. The pure reservoir (extra_dim None) is refused.
- `policy.py`: ContractPolicyRunner. Validates extra_dim ∈ {12, 30, 70}, calls the contract's load_policy with the facade's device only, records the checkpoint sha256. Pure `decode_action(raw, contract)`: collective fraction = clip((a0+1)/2, 0, 1), thrust = fraction × MAX_THRUST, rates = tanh(a1:4) × MAX_BODY_RATE.
- `assembler.py`: ContractObservationAssembler. obs = contract.make_obs(state_dict, target); extra = obs ‖ curr_gate ‖ next_lookahead ‖ obstacle_vision ‖ raycast, truncated to extra_dim. required_blocks: 12 → none; 30 → curr_gate, next_lookahead, obstacle_vision; 70 → + raycast; target always. check_coverage raises PerceptionCoverageError naming every missing block; the runner calls it before loading the model and before arming the sink.
- `scale.py`: WorldScale, the ONLY place world_scale appears. Rule: training_m = liftoff_m / world_scale. Applies to pos, vel, gate centre, aperture (nominal min(w,h) / world_scale), obstacle centres and half-extents, obstacle_vision rel and size, lookahead dist (before /10), depth ranges (before /RAY_RANGE), and track-XML geometry. Rates and quaternions are never scaled; the gamepad rate_scale is a separate knob.
- `course.py`: course schema + `normalise_obstacle` (shared by YAML, course_gen and track-XML converters). Cylinder → half_extents [r, r, half_h] (mirrors gates_proc_env L78); box → [hx, hy, hz]; raycast dict schema (center, half_extents, type).
- `course_senses.py`: pure-NumPy single-drone port of the gates_proc_env blocks. curr_gate_block = [cos yaw, sin yaw, aperture]. next_lookahead_block with a segment occlusion test, dist/10 clamped at 1. obstacle_vision_block: nearest 2, rel + max(half_extents), zero padded. Pass rule: 3D dist < aperture/2, tgt wrap/lap. Difficulty and curricula fixed at the eval value 1.0. Documented deviations (also in the ADR): training-time occlusion bearing noise omitted; an unseen/occluded next gate never gets a zero bearing (last-known next-gate pos re-bearing'd from the current pos and renormalised; with no history, the unit vector to the current gate; orient=0, dist=0, vis=0); rotated XML obstacles → AABB.
- `geometry.py`: `ray_directions(state, target, contract)` copies the ~6-line cone basis from inference.raycast (fwd = vel if speed > 0.3, else target direction, else +x), pinned by test against `raycast()`; ground-truth casting always calls contract.raycast(). Camera intrinsics from FOV + resolution; camera-from-body rotation from uptilt; Unity → contract converter.
- `runner.py`: HarnessRunner.
  - Fixed rate (default 50 Hz, configurable) with deadline scheduling on an injectable clock: WallClock (perf_counter + sleep + final-ms spin) or SimClock. No catch-up burst; missed deadlines are counted.
  - Tick: state → stale check → gate update → free-space latest → assemble → act → decode → sink.send → record. Per-stage latency (state, perception, inference, actuation, total), estimate ages and missed deadlines are recorded.
  - Latched fail-safe, then stop with a reason code, on: telemetry age > stale_after_s, gate status "lost", depth estimate age > max_estimate_age_s, any perception/policy/sink exception, abort hotkey, or Ctrl+C. Action: sink.failsafe() (neutral rates + throttle cut), retried once if it raises, failure logged.
  - A discontinuity resets both estimators and is logged as an event. Also: max_duration_s, dry_run (recording-only sink).
  - **No** yaw shaping or wz override: the sink applies the decoded contract command only. The camera-heading risk goes in the ADR and README as a candidate follow-up UC.
- `recording.py`: RunRecorder → `artifacts/harness/<harness>-<UTC stamp>/`.
  - Default root resolves against the repo root (else --output-root is required). Refuses any root whose real path is inside training/.
  - Files: config.yaml; meta.json (checkpoint path + sha256, extra_dim, device, git HEAD, contract dir, dropped-record count); telemetry.csv; ticks.jsonl (obs, extra, raw action, decoded command, sink output, estimator outputs/status/ages, latencies, events); frames/ + frames.csv (every N ticks, 0 = metadata only; written with cv2, else .npy).
  - Writes on a bounded-queue background thread.
- `loopback.py`: sim-loopback harness.
  - LoopbackPlant holds the contract state from init_state. PlantStateSource returns it; PlantActionSink applies contract.plant_step in send().
  - CourseGateEstimator gives ground truth via course_senses + the pass rule; RaycastFreeSpaceEstimator gives ground truth via contract.raycast + the obstacle_vision block.
  - Courses come from YAML or from course_gen.sample_course via import_reference, through normalise_obstacle. Runs on SimClock with no frame source.
  - Tick order equals env step(): plant_step at the end of tick k; tick k+1 reads that state, runs pass/advance, then assembles. No one-step target lag.
- `config.py`: dataclasses + YAML loader that overlays the user's --config on a packaged default, rejects unknown keys and validates ranges. Package data `harness/configs/loopback.yaml` and `harness/configs/liftoff_5inch.yaml`. Liftoff keys:
  - telemetry: host, port, stream-format list, gyro units and axis/sign table
  - camera: backend, monitor/region, FOV, uptilt, resolution
  - gates: nominal w/h, HSV ranges
  - scaling and controls: world_scale, per-axis rate profile, throttle map, rate_scale, explicit FLU stick signs
  - fail-safe timeouts, depth backend and calibration, fill policy, recording
- `src/drone_fly/cli/__init__.py` + `cli/main.py` (new, argparse, no heavy imports):
  - `drone-fly harness list`: probes registered parts with find_spec and shows whether each extra is available.
  - `drone-fly harness run --harness --checkpoint --extra-dim {12,30,70} --config [--k1-path] [--gpu-prototype-dir] [--device] [--duration] [--dry-run] [--output-root]`
  - `drone-fly harness eval` (Stage C)
  - Distinct exit codes for config error, missing dependency, coverage refusal and fail-safe stop.

**Checkpoint A:** parity + contract tests green, ruff clean.

## Stage B — Liftoff obstacle-free path, 30-dim (AC5, AC6, AC9, AC10 full, AC12 liftoff, AC13 ADR)

`src/drone_fly/harness/liftoff/`:
- `telemetry.py`:
  - LiftoffTelemetryParser: packed little-endian float32 per the configured stream format. Default order: timestamp, pos(3), quat xyzw(4), vel(3), gyro pitch/roll/yaw(3), inputs throttle/yaw/pitch/roll(4), battery(2), rpm(4); block widths configurable.
  - LiftoffCsvTelemetryReader: the recorded layout t, pos_x..z, quat_x..w, vel_x..z, gyro_pitch/roll/yaw, in_throttle/yaw/pitch/roll, batt_charge, batt_volts, rpm_0..3.
  - LiftoffUdpStateSource: stdlib socket on a background thread, latest wins, ring buffer. Discontinuity when t goes backwards or position jumps beyond a threshold.
- Unity → contract converter (geometry.py):
  - Position: X = z_u, Y = −x_u, Z = y_u. This is a reflection (det −1), so the quaternion vector part maps through the same matrix and is then negated.
  - Body frame FLU; g_body via contract.quat_rotate_inv.
  - omega via the configurable axis/sign/unit table (deg/s default → rad/s).
  - WorldScale on positions and velocities.
- `gyro_check.py`: check_gyro_convention finite-differences consecutive converted quaternions into body rates and compares them per axis with the mapped gyro: sign agreement, least-squares scale, correlation, and a suggested table correction. Runs in the CSV reader (optional report) and in offline eval.
- `frames.py`: ScreenFrameSource using dxcam/mss (lazy); background thread, latest wins, monitor/region, target fps.
- `gate_cv.py`: OpenCvGateEstimator (cv2 lazy).
  1. HSV threshold + morphology → contours → 4-corner quads (area/convexity filters).
  2. solvePnP against the nominal gate rectangle, intrinsics from FOV.
  3. camera → body (uptilt) → world (state): world gate centre and yaw, normal along the direction of travel.
  4. Tracking: current = tracked gate, else the most central/nearest candidate; next = the next candidate farther along. A pass (signed plane crossing within the aperture, and/or image area peaking then vanishing) advances the gate.
  5. Lost: hold the last world estimate (rel recomputed from live pos) for lost_timeout_s, then status "lost" → runner fail-safe.
  6. Encodings via course_senses + WorldScale; an unseen next gate uses the occluded form above.
- free-space estimator `none`: obstacle_vision zeros, NEVER provides raycast; accepted only with `assume_no_obstacles: true`. So 30-dim passes coverage and 70-dim is refused, naming raycast.
- `gamepad.py`: VgamepadActionSink (vgamepad lazy; injectable pad factory).
  - Body rate in deg/s = rad/s × rate_scale × 180/π, inverted per axis through the Betaflight rc_rate/super_rate/expo curve (monotonic; bisection or LUT), saturating at the profile max.
  - Throttle: piecewise-linear map anchoring HOVER_THR to the configured hover stick, 0 → −1, 1 → +1.
  - Default axis assignment mode 2. Explicit FLU signs: +wx (roll, right wing down) → roll stick right; +wy (about the left axis, nose down) → pitch stick forward; +wz (about up, CCW) → yaw stick left.
  - failsafe(): centre sticks, throttle min; close() releases the pad.
- `abort.py`: HotkeyAbortSignal via ctypes GetAsyncKeyState (lazy); Ctrl+C is always honoured.
- Registration `liftoff` = udp + screen + opencv + (none | depth) + vgamepad.
- ADR `docs/adr/0002-live-harness-layering.md`: layering, bridge rules, registry, fail-safe semantics, frame/scale conventions, documented deviations, how to add a harness or estimator, the camera-heading known risk and follow-up, and Liftoff as the worked example.

**Checkpoint B:** tests for the parser/converter, gyro check, gate estimator (synthetic images), sink (mock pad), runner fail-safe and coverage refusal; the `--harness liftoff --extra-dim 30` path wired end-to-end (the live check is manual, by the user).

## Stage C — depth free-space + raycast reconstruction (70-dim), offline eval (AC7, AC8)

- `liftoff/depth.py`: MonocularDepthFreeSpaceEstimator, registered as `depth`.
  - Pluggable DepthModel; first backend is Depth-Anything-V2 metric via transformers (lazy, configurable model id), with affine scale/shift calibration and an optional inverse-depth flag.
  - Own worker thread, latest frame wins, publishes a timestamped depth map.
  - Synchronous update():
    1. Rays from geometry.ray_directions, including the target fallback below 0.3 m/s.
    2. world → body → camera → pixel.
    3. In-front, in-FOV rays take a robust patch minimum; z-depth → range along the ray via z/cos → WorldScale → clamp at RAY_RANGE × ray_range_scale → normalise.
    4. Out-of-FOV rays use the fill policy: clear | nearest_in_fov (default) | hold_last (max age).
    5. In-FOV mask and coverage fraction are recorded.
  - obstacle_vision: near-depth connected components (excluding a horizon band) back-projected to centroid rel + half-extent proxy, nearest 2; config switch depth_clusters | zeros.
  - Estimate age is tracked; the runner fails safe beyond max_estimate_age_s.
- `harness/offline/` (the ONLY XML reader; the runtime never imports it):
  - `track_xml.py`: race XML (gate order) + track XML (placed items) via a configurable item-id table; items go through the Unity converter, WorldScale and normalise_obstacle; rotated obstacles → AABB.
  - `evaluate.py`: replays a run folder (each sampled frame paired with the nearest telemetry; estimators re-run synchronously). Reports gate position error (m), yaw error (deg) and detection rate; free-space per-ray MAE and near-weighted error against contract.raycast() on the XML obstacles; obstacle_vision error; the gyro check; and a suggested depth-scale fit. Writes eval_report.json and prints a summary.
  - CLI: `drone-fly harness eval --session --track --race --config [--report]`.

**Checkpoint C:** depth tested with a fake DepthModel; XML and evaluator with synthetic fixtures; 70-dim + `depth` passes coverage.

## pyproject
`[project.scripts] drone-fly = "drone_fly.cli.main:main"`; `[project.optional-dependencies] harness` = opencv-python, mss, dxcam (win32), vgamepad (win32), transformers (+ pillow if needed). Core dependencies unchanged. `uv lock` with no --upgrade.

## Tests (QA: tests/, flat test_harness_*.py; synthetic fixtures in tests/data/harness/; one session-scoped CPU contract fixture)
- **registry_imports:** with heavy deps blocked, the package, CLI and `list` still import (AC1); a missing dep raises HarnessDependencyError naming the extra; a programmatically registered fake estimator runs through the runner unchanged (AC13); runtime modules don't import `offline`.
- **contract:** fixture-connectome K1ReservoirAug(70) seeded, stripped and saved; ContractPolicyRunner == inference.load_policy(...).act. decode_action matches the linear clamp and tanh·MAX_BODY_RATE, including clamping, and is not a sigmoid. Pure reservoir and bad extra_dim refused; missing paths give actionable errors. CPU import works, a pre-imported mismatched dev raises, a repeat call is idempotent.
- **assembler:** bit-exact layout and prefix truncation at 12/30/70; obs == make_obs. Coverage refusal names the missing blocks; `none` + 70 refused, + 30 accepted.
- **course_senses:** drift test against gates_proc_env via import_reference: non-occluded cases exact, occluded cases on orient/dist/vis only. Covers curr_gate, the pass rule, cylinder [r, r, half_h] normalisation with a differing raw [1], and the unseen-bearing rule.
- **scale:** a Liftoff-scale scene (scale s, world_scale=s) gives the same 70-dim extra as the training-scale scene, and the pass rule fires at the same point.
- **loopback_parity (AC4):** the reference loop uses only contract functions + course_senses, does its own concatenation and follows env order (plant_step → pass → advance with wrap/lap → obs). Per-step obs/extra/action match within tolerance and the target index sequence matches exactly, on a course with ≥ 2 passes. ray_directions == raycast() basis.
- **runner:** 50 Hz scheduling, missed deadlines, latency fields. Fail-safe on stale telemetry, gate lost, depth age, perception/policy/sink exceptions and abort, each calling failsafe() with neutral rates + throttle cut and logging the reason. A discontinuity resets the estimators.
- **recording:** run-folder layout; refusal of training/ (including a relative path that traverses into it); repo-root default.
- **liftoff_telemetry:** synthetic CSV + matching packed UDP datagram. Unity → contract property tests (rotate-then-map == map-then-rotate; up → +Z; identity attitude gives g_body (0,0,−1)); gyro unit/axis mapping; discontinuity; a UDP round trip on 127.0.0.1.
- **gyro_check:** gyro generated from the fixture's own quaternions gives sign +1 and scale ≈ 1; a flipped axis is detected.
- **gate_cv** (importorskip cv2): PnP pose error on synthetic quads, vis flag, pass advance, hold → lost.
- **gamepad:** inverse rate curve round-trips; throttle anchors; axis assignment; all three FLU signs + throttle direction pinned; failsafe values.
- **depth:** a fake DepthModel covers projection, z→range, clamp/normalise, each fill policy, the low-speed fallback, latest-frame-wins and age.
- **offline_eval:** synthetic XML + session folder; ground-truth estimators give zero error, perturbed estimators report error.
- **cli:** `list` output; `run --harness sim-loopback` with the fixture model writes a run folder; distinct exit codes.

## Risks (acknowledged)
- The smoke test is red at baseline; editing it is gated on user sign-off.
- The CPU override is load-bearing.
- The real K1 connectome is not in the worktree: CI proves the I/O path with the fixture connectome; a real-model parity test runs only when DRONE_FLY_K1_PATH + the checkpoint exist.
- sys.path collisions: covered by the guard. Contract drift: caught by the parity/drift tests.
- Scale/plant mismatch: flight success in Liftoff is not an acceptance criterion.
- Owner calibration inputs: telemetry format, gyro convention (now measured), rate profile, hover stick, XML item table, in-game virtual-pad binding.
- Latency: async perception + age-based fail-safe.
- Out-of-FOV rays: fill policy + coverage metric.
- Respawns: detected via discontinuity; Player.log tailing deferred.
- Camera heading: the policy is yaw-agnostic. Known risk and candidate follow-up; no yaw shaping in this UC.
- The brief is stale on SB3/gymnasium/cli/train/evaluate; fix later via revise-brief.

## AC map
AC1 interfaces/registry/errors · AC2 contract + policy · AC3 assembler · AC4 loopback + parity · AC5 telemetry + converter (+ gyro_check) · AC6 gate_cv · AC7 depth · AC8 offline · AC9 gamepad · AC10 runner · AC11 recording · AC12 cli + configs · AC13 registry + ADR · AC14 extra + baseline note · AC15 Stages A/B/C with checkpoints.

## Files Affected (from the analyst's final approved proposal)

**Production (developer)**, under `src/drone_fly/`:
- `harness/__init__.py`, `errors.py`, `types.py`, `interfaces.py`, `registry.py`, `builtins.py`, `contract.py`, `policy.py`, `assembler.py`, `scale.py`, `course.py`, `course_senses.py`, `geometry.py`, `runner.py`, `recording.py`, `loopback.py`, `config.py`
- `harness/configs/loopback.yaml`, `harness/configs/liftoff_5inch.yaml` (package data inside paths.production; does not recreate the deleted top-level `configs/`)
- `harness/liftoff/__init__.py`, `telemetry.py`, `gyro_check.py`, `frames.py`, `gate_cv.py`, `gamepad.py`, `abort.py`, `depth.py` (+ `unity_frames.py` if not in geometry.py)
- `harness/offline/__init__.py`, `track_xml.py`, `evaluate.py`
- `cli/__init__.py`, `cli/main.py`

**Test (QA)**, under `tests/`:
- `test_harness_registry_imports.py`, `test_harness_contract.py`, `test_harness_assembler.py`, `test_harness_course_senses.py`, `test_harness_scale.py`, `test_harness_loopback_parity.py`, `test_harness_runner.py`, `test_harness_recording.py`, `test_harness_liftoff_telemetry.py`, `test_harness_gyro_check.py`, `test_harness_gate_cv.py`, `test_harness_gamepad.py`, `test_harness_depth.py`, `test_harness_offline_eval.py`, `test_harness_cli.py`
- Synthetic fixtures under `tests/data/harness/`: a telemetry CSV with the exact recorded header, a track/race XML pair, and a course YAML.

**Outside declared paths (user sign-off):** `pyproject.toml`, `uv.lock`, `docs/adr/0002-live-harness-layering.md`, `README.md`, `tests/test_smoke.py` (QA), AC5 synthetic-fixture deviation.

**Never touched:** `gpu_prototype/`, `training/`, `docs/INFERENCE_CONTRACT.md`, `PROJECT_BRIEF.md`, `USE_CASES.md`.

## User sign-off (2026-10-11)
- Plan accepted for implementation.
- APPROVED outside declared paths: `pyproject.toml` + `uv.lock`; `docs/adr/0002-live-harness-layering.md` + `README.md` "Live harness" section; AC5 synthetic telemetry fixtures.
- NOT approved: editing `tests/test_smoke.py`. Leave it untouched; report its failure as pre-existing at baseline (cc817f4), not caused by this branch. AC14 is judged against the baseline result.
- Checkpoint policy: commit + push to `feat/uc-62-liftoff-vision-harness` at each stage checkpoint (A, B, C).
