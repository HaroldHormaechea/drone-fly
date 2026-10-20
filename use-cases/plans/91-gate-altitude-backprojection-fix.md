---
plan_for: use-cases/91-gate-altitude-backprojection-fix.md
work_branch: feat/uc-91-gate-altitude-backprojection
run: drone-fly-uc-91
approved: 2026-10-20
---

# UC-91 implementation plan (challenger-approved, 2 rounds)

Analyst v1 with the v2 amendments merged in, plus 3 non-blocking challenger notes at the end.

## 0. Top-line framing (it heads the plan and the doc): the use case's premise is refuted
- **Gate 0 is NOT 10-18 m up.** All 7 `InflatableArchBrandless01` in
  `artifacts/harness/trackdata/LiftoffArenaTrack04_0001.xml` sit at Unity y=0.1, which puts the aperture centre at
  **1.84 m** (centre_offset 1.74). `load_catalogue_course` confirms this: passages are at course z 0.98 with the frame
  origin at z 0.86, giving 1.84 contract. The "10-18 m" figure is the drone's own altitude over the gate. UC-89 notes
  say "passing 7-13 m too high", and UC-90 says "+11 m above the perceived gate".
- **The live estimate is correct.** On uc-89 attempt 5, static on the pad, the estimate is (0.06, −0.34, 1.85) m
  against truth (0.065, −0.065, 1.84) at 37.8 m.
- **The policy is told the truth.** At tick 170 the drone is at 13.5 m, tracking gate 0 with 0.9 m 3-D error, and its
  observation rel z = −0.942 training units = **−11.3 m** (the gate is 11 m below it). The overflight is a policy
  problem. Model 14 predates the UC-90 altitude-reward work.
- **Expected AC7 outcome: about no change in gate passes.** The AC7 report must say this plainly, and must not credit
  or blame perception.
- **AC4's bias clause is already met by the baseline.** Re-solving every logged 10-keypoint detection from all 6 uc-89
  runs (Liftoff m):
  - At 25-40 m, the z-error median is about 0 (sd 1.2-2.7).
  - At 40-60 m, the z median is −1.4…+0.1 and the sd is 2.2-3.0 (p90 |z| 3.6-6.3, 3-D median 5.5-8).
  - At 40-60 m there is a radial over-estimate of +0.7…+7.4 m, which may partly be a mirrored-arch matching artifact.
  - 20-45 % of detections point at no arch (false positives or wild PnP).
  - The logged target spikes (z 29, 38, −11.6 m) are such outliers, acquired and then held by
    `lost_policy: reacquire` / `lost_hard_stop_s: 10`.

## AC1: current derivation (verified in code; no monocular depth)
`harness/liftoff/depth.py` feeds only the free-space eyes.
1. `gate_learned.LearnedGateEstimator.detect` (L~395-440) runs `infer` (gate_net prepare_input/decode, keypoints
   rescaled to frame px). It keeps keypoints with visibility ≥0.5 that are in the image and outside the masks, and
   needs at least `min_keypoints` (4).
2. `gate_cv.solve_planar_pnp` (L124-140) solves OpenCV IPPE PnP against `GateType.object_points()` (gate_types.py
   L104-112: aperture-centre-relative, y-down, z=0). **Range already comes from known gate geometry.**
   inflatable_arch: 10 kp, outer 8.0×4.46 m, aperture 6.18×3.69 m, centre_offset 1.74 m.
3. Intrinsics: `gate_cv.camera_from_config` builds a `geometry.CameraModel` (hfov pinhole, square pixels, principal
   point at the image centre). `_camera_for` rescales it to the capture size.
4. `TrackingGateEstimator.detection_from_pose` (gate_cv.py L266-304) takes the pose at det_stamp + pose_offset_s and
   computes `r_wc = quat_to_matrix(q) @ body_to_camera().T` and `center = pos + r_wc @ (t_cam / world_scale)`.
5. The tracker `update` (L387+) produces `GateEstimate.target`. The runner's observation is target − pos, rotated by the
   spawn-yaw aligner, with z untouched.

## AC3: camera
- Config: 2560×1080, hfov 120°, uptilt 20.0° (configs/liftoff.yaml L43-46; the drone file says 20.0315°).
- **Sign, analytically:** the optical axis (cos t, 0, sin t) in FLU pitches up for positive uptilt.
- **Sign, empirically:** see the pad check in section 0. A flipped sign would put z about 25 m off at 38 m, and 1° of
  error is about 0.66 m.
- The developer also re-runs `offline/align.py`'s CP6b camera check on one uc-89 run and records the result.

## Proposed Solution
Default behaviour stays bit-identical.

1. **`src/drone_fly/harness/geometry.py` `CameraModel`:**
   - Add `principal_point: tuple[float, float] | None = None` as a trailing default, so positional constructors stay
     valid.
   - `cx`/`cy` honour it, and `scaled_to` scales it proportionally.
   - In `offline/align.py` L352/406/534/553, the `CameraModel(...)` rebuilds become `dataclasses.replace(camera, ...)`.
   - Liftoff stays centred, which is documented.
2. **`gate_cv.camera_from_config`:** reads an optional `camera.principal_point: [cx, cy]`, validated with
   `require_vector` and required to lie inside the image. If it is absent, the image centre is used.
3. **New pure numpy module `src/drone_fly/harness/liftoff/gate_range.py`:**
   - `size_range(object_pts, image_pts, fx)` = f·sqrt(A_obj/A_img) over the convex hulls of the kept keypoints, with
     dimensions taken from the gate_types table.
   - `size_range_valid(...)` returns False when any kept keypoint is within `edge_margin_px` of the image border or a
     mask, when the hull covers less than `min_hull_frac` of the full class hull, or when there are fewer than 4 points.
   - `check_detection(...)` returns (accepted, reason):
     - (a) r_pnp/r_size must lie in [size_ratio_min, size_ratio_max]. The band is asymmetric to allow for
       foreshortening.
     - (b) The world gate-plane normal's tilt must be ≤ `max_normal_tilt_deg` (null disables). Its primary purpose is
       rejecting the IPPE planar-ambiguity flip.
     - When the size range is invalid, only (b) applies. This is the documented truncated-box fallback.
4. **`gate_learned.py`:**
   - New `gates.learned.range_mode` ∈ {`pnp` (default, today's behaviour, bit-identical), `pnp_checked`}, with knobs
     under `gates.learned.range_check`, all validated with `require_*`, plus the `from_config` passthrough.
   - `pnp_checked` drops rejected detections.
   - Per-reason rejection counters are exposed via `extra_info()`, so they land in ticks.jsonl.
   - Added cost is microseconds, so there is no measurable latency at 26-31 Hz.
5. **New `src/drone_fly/harness/offline/gate_backprojection_eval.py`:**
   - CLI: `drone-fly harness gates backprojection-eval --runs <dir>... --track <xml> [--trackdata <dir>]
     [--modes pnp,pnp_checked] [--tune-runs <dir>...] [--with-frames] [--device ...] [--out <dir>]`, wired in
     `cli/main.py` next to `_cmd_gates_diagnose` (L633, gsub parser L1136+).
   - **Truth/matching core (pure, unit-tested):**
     - Truth passages come from `load_catalogue_course` (detector_noise.py L1154) + `frame.origin`, de-duplicated by
       centre.
     - `match_by_bearing` returns (best, angle, ambiguous). A detection is ambiguous when the best angle exceeds the
       tolerance (default 5°), or when a second distinct passage is within the tolerance or within
       `ambiguity_margin_deg` (default 3°) of the best.
     - Ambiguous detections are excluded from the stats and counted per bin.
     - Errors: LOS radial / tan-h / tan-v (`sim.noise_model.los_frame`), world z, and full 3-D.
     - Bins: `detector_noise.RANGE_EDGES` + 40-60 + >60.
   - **Pass A (primary AC4 evidence, no GPU):** re-solves every logged full-keypoint detection (about 90-92 %) with
     `solve_planar_pnp` + `detection_from_pose`, under each mode.
     - Pose: telemetry.csv interpolated at the detection's source stamp + pose_offset_s. Position is lerped and the
       quaternion is slerped (`gate_cv._slerp`). Stamps outside the telemetry span are skipped and counted.
     - Detections are de-duplicated by (frame_index, order).
     - Partial detections are reported as not re-solvable.
     - An evaluator-only size-range comparison column is included. Size range is never a live mode.
   - **Pass B (tracker replay):** a replay subclass of `TrackingGateEstimator` overrides `detect_frame` to return the
     Pass-A detections for each tick.
     - It is driven with the logged state, stamps and the run's tracker config. Both modes get identical inputs.
     - Caveat: partial detections are missing from both streams.
     - Fidelity: the replayed `pnp` target is compared with the logged `gate.target` (share within 0.5 m, median
       deviation). If fidelity is low, Pass B is marked indicative only.
   - **Pass C (optional `--with-frames`):** re-runs the real detector on the saved frames (every 10 ticks, about 140
     per run) in both modes. Per-bin n is printed and thin bins are flagged.
   - **Logged-target report:** the logged `gate.target` vs truth gives the live gate-z error, used for the AC7 re-fly.
   - **Retention, per bin and mode:**
     - matched-unambiguous kept / rejected
     - unmatched kept / rejected
     - rejections per reason
   - **OneDrive hygiene:** the tool takes explicit run dirs and opens only config.yaml, ticks.jsonl, telemetry.csv and
     frames.csv, plus the `frames/<file>` paths listed in frames.csv. It never globs, rglobs, lists or walks a run dir,
     and the module docstring says so.
   - **Output:** `artifacts/harness/reports/uc91-backprojection/<UTC run-id>/report.md` + `report.json`, recording the
     exact CLI invocation.
   - **Thresholds and split:**
     - Tune on uc-89 attempts 1-3 (T180325, T180418, T180500). Report on held-out attempts 4-6 (T180602, T180649,
       T180758). A run in both sets is an error.
     - Physics starting points:
       - Ratio band: about ±2.5σ, from PnP σ_r≈0.128·r against size-range noise from 1 px keypoint error at the 768
         input. The upper bound also allows about 60° of foreshortening.
       - Tilt: 30°.
     - These are adjusted only on the tune set. The report prints the final values next to the physics starting
       points.
   - **Pre-registered AC4 rule (held-out; Pass A primary, Pass B secondary):**
     - Beyond 40 m (matched, unambiguous): p90 |z err| AND z sd both drop by at least 30 % vs `pnp`, with at least 90 %
       of matched-unambiguous detections kept in every bin with n ≥ 15.
     - Within 40 m: |median z| stays ≤ 2 m and retention stays ≥ 90 %.
     - If the rule is not met, the doc and PR say AC4's noise clause is unmet. There is no re-tuning on the held-out
       set.
6. **Doc `docs/harness/gate-backprojection.md`, linked from `docs/harness/perception-providers.md`:** the framing from
   section 0, then the AC1 path, the AC3 evidence plus the align.py result, the ground truth, the before/after tables
   (tune and held-out separately), the range_mode and knobs, the truncated-box fallback, and the CLI invocation. The
   40-60 m radial claim is stated only from the disambiguated numbers.
7. **AC7 re-fly (orchestrator):**
   - Config: `artifacts/harness/live-m14-w768.yaml` + `gates.learned.range_mode: pnp_checked`, **only if the held-out
     rule passed**. Otherwise use default `pnp`, noted in the report.
   - Everything else stays identical: width 768, align_spawn_yaw true, latency compensation off.
   - Passes are verified via telemetry / `completeness_from_pose`, never `gate_passed`.
   - The evaluator's logged-target pass runs on the new flights.

## Files Affected
**Production (developer):**
- `src/drone_fly/harness/geometry.py`
- `src/drone_fly/harness/liftoff/gate_cv.py`
- `src/drone_fly/harness/liftoff/gate_learned.py`
- new `src/drone_fly/harness/liftoff/gate_range.py`
- new `src/drone_fly/harness/offline/gate_backprojection_eval.py`
- `src/drone_fly/harness/offline/align.py`
- `src/drone_fly/cli/main.py`
- new `docs/harness/gate-backprojection.md` + one link in `docs/harness/perception-providers.md`

**Tests (qa), CPU-hermetic:** new `tests/test_uc91_gate_backprojection.py`.
- A synthetic arch projected through `CameraModel` (uptilt 20°, off-centre principal point, known attitude), then PnP +
  `detection_from_pose`, recovers the centre within tolerance, including z at 40/60 m.
- Flipping the uptilt sign shifts z by the expected amount.
- principal_point config, `scaled_to`, and align.py's `replace` preserve it.
- `size_range` matches truth for fronto-parallel and oblique gates.
- Truncation (edge, mask, small hull) is marked invalid and falls back.
- Ratio and tilt rejections, including the IPPE wrong-branch normal rejected and the right branch accepted.
- `range_mode` defaults to `pnp`; passthrough and validation errors.
- `match_by_bearing` ambiguity.
- Slerp interpolation and out-of-span skipping.
- Retention and per-reason counters.
- Pass A on a synthetic run folder, with partials counted and not solved.
- Pass B determinism and a synthetic fidelity reproduction.
- A run in both the tune and held-out sets is an error.
- No-walk guarantee: Path.glob/rglob/os.walk/os.listdir monkeypatched to raise.
- `tests/test_harness_gate_learned.py` is extended only if its stub-net fixture is the natural place.

## Risks
- **The UC premise is false.** The baseline already meets AC4's bias clause, and the re-fly will likely show no change
  in passes. This must be communicated.
- **Recall.** `pnp_checked` lowers recall, which is already about 30 %; that is why the retention floor exists. If it
  cannot meet the noise rule while keeping 90 % retention, the honest outcome is "noise clause unmet", with no tuning
  toward a pass.
- **Pass B fidelity** is limited by async detections, pose_offset_s and the missing partials. It is quantified, and
  Pass A stays primary.
- **The 40-60 m radial over-estimate** may be detector keypoint localisation at the 768 input (retraining is out of
  scope) or a matching artifact. The disambiguated numbers decide which.
- **The normal-tilt check** assumes vertical gate planes. It is configurable, and null disables it.
- **Stale replays** don't affect the tool, since it is telemetry-based.
- **`CameraModel` change:** a trailing default field, so there is no behaviour change for existing users.
- **AC2** is satisfied as "already geometry-based PnP", plus a size cross-check and the truncation fallback. The primary
  range source is not switched.

## Challenger non-blocking notes for the developer
(a) Pass B should reuse the tracker's own `_pose_at` / pose history, so pose_offset_s semantics match live.
(b) The report prints the final tune-set `range_check` values next to the physics starting points.
(c) Keep the re-fly conditional: `pnp_checked` only if the held-out rule passes.

## Challenger verdict
APPROVED after 2 rounds.
- v1 had 3 Majors plus Minors:
  - ambiguous mirrored-arch matching;
  - an AC4 noise reduction that could be achieved by deleting detections, with no retention metric and tuning on the
    test set;
  - a before/after that was not like-for-like on the tracked target, with saved frames only every 10 ticks.
- v2 resolved all of them.
