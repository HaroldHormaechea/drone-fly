---
plan_for: (free-form task)
work_branch: feat/eyes-live-free-space-estimator
run: drone-fly
approved: 2026-10-18
---

# Eyes as a live `free_space_estimator` part — approved plan

Challenger APPROVED in round 2 with 4 non-blocking conditions (at the end). The orchestrator reviewed and accepted the plan; the user is away and asked for autonomous work. Orchestrator authorizations for this run:
- `docs/harness/liftoff-live-session.md` (outside `paths.production`).
- `src/drone_fly/harness/configs/liftoff_5inch.yaml` is under `src/`.

## Analysis (summary)

**Latency on the RTX 3050** (bench without Liftoff, 2560x1080):
- v1 (artifacts/uc84/v1/eyes.pt): eyes p50 9.5 / p95 12.9 ms; with the gate detector, p50 23.8 / p95 28.2 ms.
- v2 (artifacts/uc84/v2/eyes.pt): eyes p50 6.5 / p95 13.1 ms; with the detector, p50 12.6 / p95 15.3 ms; about 110 MB.
- v3 (EyesDepthNet, real DA-V2-Small backbone): p50 34.4 / p95 36.1 ms, 188 MB. No trained v3 checkpoint exists locally yet.
- So the eyes run on an async worker, and the tick reads the latest published result.
- Expected estimate age: about 50–100 ms for v1/v2, about 100–150 ms for v3. max_estimate_age_s is 0.25.

**The 165 ms policy step in today's depth run** is GPU contention (Depth-Anything at 513–660 ms per frame, and the detector stretched to 470 ms over the same ticks). It is not warm-up and not a policy defect. No policy change is in scope. The first eyes flight's latency_summary is the check.

**Harness facts:**
- builtins.py registers only none and depth;
- SECTION_FOR_KIND maps free_space_estimator to the section `free_space`;
- the depth part reads its own `depth` section via ctx.config.section;
- the runner fails safe with perception_stale when t1 − fs.stamp > max_estimate_age_s (runner.py:567-586);
- reset() is called on a discontinuity;
- the config is strict, so new keys must exist in liftoff_5inch.yaml;
- torch must not be imported at module level in harness modules (test_harness_registry_imports.py).

**Contract match:**
- the v1/v2 meta cone == contract.CONE, N_RAYS 40, RAY_RANGE 4.0;
- world_scale 4.0, ray_range_scale 1.0 and the camera (2560x1080, hfov 120, uptilt 20) match the preset;
- raycast = free with no transform; conf does not gate the rays.

**obstacle_vision (8 values):** the two nearest obstacles, each as a world-frame offset (dx, dy, dz) plus max(half_extents), in training metres, zero-padded, gates never included (gpu_prototype/gates_proc_env.py:232-246; harness/course_senses.py:101). Live, it may only be zeros or derived from screen data.

## Solution

1. **New module** `src/drone_fly/harness/liftoff/eyes.py` with `EyesFreeSpaceEstimator(FreeSpaceEstimator)`, provides {"obstacle_vision", "raycast"}.
   - **Provider-contract docstring** (the basis for the follow-up pluggable-perception use case):
     - Inputs: screen Frame + telemetry FlightState + the gate target from an on-screen detector.
     - Outputs: raycast float32 (40,) in [0, 1] in CONE order; in_fov_mask bool (40,); obstacle_vision float32 (8,).
     - stamp = capture time (perf_counter) of the newest frame that produced a valid estimate, -inf when none.
     - Latency rule: worker GPU duty ≤ 50%, estimate age ≤ runner.max_estimate_age_s.
     - info keys: frame_index and inference_s, which the recorder reads.
   - Lazy torch through errors.lazy_import, so a missing torch raises HarnessDependencyError.
   - The constructor takes an injected EyesEstimator, frame_source, state_source, max_rate_hz, warmup_timeout_s and clock.
2. **from_config** reads ctx.config.section("eyes"):
   - `weights` is required; the path is absolute or relative to the repo root (else the cwd); a missing file is HarnessConfigError.
   - `device`: auto | cpu | cuda.
   - `max_rate_hz`: auto (default) or a number > 0.
     - auto resolves from the meta: EyesDepthNet → 10 Hz, EyesNet v1/v2 → 30 Hz (AUTO_RATE_HZ).
     - A numeric value below min_rate = 1/(2·tol_s), from meta input.hist_tol_s (default 0.5·dt_s, so 5 Hz), is HarnessConfigError.
   - `warmup_timeout_s` ≥ 0, default 30.
   - `obstacle_vision`: zeros | ray_clusters (default per condition 3).
   - `cluster_link_m` > 0 (0.5) and `min_obstacle_size_m` > 0 (0.15), both training metres.
   - Load errors become HarnessConfigError.
   - Meta checks against the config, each a HarnessConfigError naming both values: n_rays and cone (allclose, atol 1e-5), ray_range, ray_range_scale == 1.0, world_scale, camera hfov and uptilt (resolution may differ).
   - **Screen-only guard:** parts.gate_estimator of oracle or course is refused (course tested; oracle kept as defence in depth).
   - frame_source and state_source come from ctx.resources.
3. **CUDA stream isolation:**
   - On CUDA the loader creates `torch.cuda.Stream(device)`. The worker's estimator.update and the start() warm-up run under `with torch.cuda.stream(...)`.
   - Sync mode uses the caller's current stream.
   - Only host numpy results cross threads.
   - **infer.py:** torch.cuda.synchronize() becomes current_stream(device).synchronize().
4. **Lifecycle:**
   - start(): 3 warm-up passes on a blank frame at the meta resolution (in the stream, synced). It records the median forward_s and logs the resolved rate, stream on/off and forward_s. Then it resets the estimator.
   - Duty guard: pure functions gpu_duty / check_duty log a WARNING when duty > MAX_GPU_DUTY 0.5 (warn, not refuse).
   - Sync mode returns at that point.
   - Async mode starts the daemon thread "eyes-worker" and waits up to warmup_timeout_s for the first history_ok estimate. A timeout is a warning, after which the runner fails safe; a worker error is re-raised. stop() sets the event and joins with a 2 s timeout.
   - The worker:
     - polls frame/state sources about every 2 ms, skipping None or an unchanged frame index;
     - throttles to 1/rate;
     - applies a pending reset on the worker thread;
     - updates with the latest target, stored under a lock;
     - publishes under the lock;
     - sends errors to _worker_error.
   - A single thread owns the estimator.
5. **Staleness:**
   - Only history_ok outputs are published, stamped with their frame's capture time; others never replace the held estimate.
   - Before the first valid estimate or after reset(): raycast ones, in_fov all false, obstacle_vision zeros, stamp -inf, info {"eyes": "none"}.
   - reset() clears the estimate, flags a worker reset, and clears the sync-mode last-processed stamp.
6. **update():**
   - Stores target. In sync mode it processes the frame unless its stamp equals the last one processed.
   - If the worker failed, it raises RuntimeError → perception_error.
   - It returns a FreeSpaceEstimate with:
     - raycast = clip(free, 0, 1) float32;
     - in_fov_mask = in_view;
     - obstacle_vision per mode;
     - stamp;
     - info {frame_index, inference_s ("eyes worker wall time per frame, includes stream sync"), preprocess_s, forward_s, history_ok, conf_mean, conf (40 values, 3 dp), coverage, weights, rate_hz, stream, gpu_duty_est}.
7. **ray_clusters:** a pure numpy function obstacle_vision_from_rays(pos, rays_world, free, ray_range, link_m, min_size).
   - Hit rays are free < 1 − 1e-3; hit points = pos + dir·free·ray_range.
   - Greedy nearest-first clustering, done twice: size = max(half-extent, min_size); centre = surface mean + view_dir·size (as depth.py:410-415).
   - The result goes through course_senses.obstacle_vision_block(pos, centers=, sizes=).
   - EyesOutput gains `rays_world`.
8. **builtins.py:** register ("free_space_estimator", "eyes", f"{_PKG}.liftoff.eyes:EyesFreeSpaceEstimator", requires torch). v3 needs transformers/pillow lazily (documented).
9. **liftoff_5inch.yaml:** new commented `eyes:` section after `depth:` (weights null, device auto, max_rate_hz auto, warmup_timeout_s 30, obstacle_vision <default>, cluster_link_m 0.5, min_obstacle_size_m 0.15). The parts comment becomes `none | depth | eyes`. The preset is unchanged.
10. **infer.py:** the stream sync, the rays_world field, and the docstring pointer.
11. **Docs** (`docs/harness/liftoff-live-session.md`):
    - New section "Stage (c): 70-dim obstacle model with the vision eyes".
    - Example overlay artifacts/harness/live-eyes.yaml: gates.learned.weights detector-v1 deploy path (data root), parts.free_space_estimator eyes, eyes.weights artifacts/uc84/v2/eyes.pt; worktrees need absolute paths.
    - Command: `.venv-cuda\Scripts\drone-fly.exe harness run --harness liftoff --preset liftoff --config artifacts/harness/live-eyes.yaml --extra-dim 70 --checkpoint training/11-count-curriculum/model_readout.pt --output-root artifacts/harness/liftoff/runs --overlay`.
    - It also states: the screen-only rule and the refusal; the measured latencies and duty; acceptance via latency_summary.json; the 165 ms explanation; the obstacle_vision mismatch and phantom-obstacle trade-off; that a student checkpoint forces free_space none (student.py:196-205); provision-backbone for v3.
    - Fix the stale paths in steps 0.3/0.4 and the stage (b) command.
    - Troubleshooting rows.

## Files
- **Production:** src/drone_fly/harness/liftoff/eyes.py (new), src/drone_fly/harness/builtins.py, src/drone_fly/harness/configs/liftoff_5inch.yaml, src/drone_fly/eyes/infer.py.
- **Docs:** docs/harness/liftoff-live-session.md.
- **Tests (QA, CPU, hermetic):** tests/test_harness_eyes.py (new); update the expected registry set at tests/test_harness_registry_imports.py:176; re-run test_eyes_infer / test_eyes_bench / test_eyes_vision_only and the harness suite.
- **Test coverage:**
  - registration and selection;
  - all config validation (rate floor, auto rate v1/v2 → 30, v3 tiny-depth → 10, cluster keys);
  - course refusal;
  - output contract and 70-dim assembly;
  - equivalence with a reference estimator;
  - sync staleness (-inf before history, the held stamp during a gap, -inf after reset, a same-stamp frame after reset);
  - real HarnessRunner perception_stale;
  - async worker (start waits, latest wins, throttle, errors surface, stop joins);
  - duty warning;
  - _stream None on CPU;
  - ray_clusters on synthetic rays;
  - v1/v2/v3 tiny-depth load;
  - torch blocked → HarnessDependencyError;
  - optional AST import guard;
  - the doc contains the command.

## Risks
- Train/live input mismatch (noisy eyes rays; derived or zero obstacle_vision): this makes the flight possible, not good avoidance.
- Latency under Liftoff rendering is unmeasured; the first flight is the acceptance check.
- v3 backbone caching is a follow-up.
- About 2 ms state/frame skew.
- About 0.4 s history warm-up; a respawn mid-run fails safe.

## Challenger conditions (developer and QA must honour)
1. After the weights are loaded on the default stream, call `self._stream.wait_stream(torch.cuda.current_stream(device))` (or synchronize once) before the first use on the worker stream.
2. EyesOutput.rays_world: last field, default None (or keyword in both constructors, infer.py:134 and :164).
3. obstacle_vision default:
   - ray_clusters only if it beats zeros by ≥ 2 pp in both completion and collision, toward "true", on exact AND noised rays; otherwise zeros is the default and ray_clusters is opt-in.
   - Audit pending from the analyst; the orchestrator relays the decision. Until then, implement both and default to **zeros**.
4. Docs: next to ray_clusters, say that eyes false alarms become phantom obstacles.
