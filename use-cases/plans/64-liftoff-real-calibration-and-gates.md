---
plan_for: use-cases/64-liftoff-real-calibration-and-gates.md
work_branch: feat/uc-64-liftoff-real-calibration-and-gates
run: drone-fly-uc-64
approved: 2026-10-11
---

> UC-64 FINAL APPROVED PROPOSAL — challenger approved in round 2. Consolidates v1, orchestrator amendments,
> v2, v2.1 (QA verification), v2.2 (artifacts/harness/liftoff) and the challenger's implementation notes.
>
> Abbreviations: T = C:\dev\drone-fly-uc-64-liftoff-real-calibration-and-gates; H = T\src\drone_fly\harness;
> R = C:\dev\drone-fly\artifacts\harness (real data, read-only); A = T\artifacts\harness\liftoff (gitignored
> output root, organised by kind).

## Analysis

UC-62 built a composed harness (registry / builtins / runner / session / liftoff parts) and tested it only against fakes and synthetic data.

**Config**
- The config overlay rejects keys that aren't in the packaged default (config.py `overlay`) and replaces lists wholesale.
- So every new key must exist in `liftoff_5inch.yaml`, and a preset must be an overlay on it.
- `liftoff_5inch.yaml:12` already has `harness: liftoff`.

**Telemetry**
- `TelemetryFormat.parse` already supports `rpm_count_prefix`.
- The `GyroMap` defaults (telemetry.py:198-215) and the synthetic fixture generator both encode the falsified convention.
- test_harness_liftoff_telemetry.py:312 pins YAML == code defaults.

**Gate estimator**
- `OpenCvGateEstimator` (gate_cv.py) mixes red-HSV detection with the tracker (acquire / hold / lost / pass, next gate, encodings) that AC5 must keep.

**Offline tooling**
- offline\track_xml.py only parses flat `instanceID` lists. It already takes gate plane normals from the instance rotation (track_xml.py:232) and picks the sign by travel direction.
- offline\evaluate.py only reads harness run folders, not the manual capture format (telemetry_raw.csv hex 97-byte datagrams + frames.csv `t_wall_ns,file`).

**Real race and track data**
- Real race XML: out-of-order `checkPointPassages`. The walk is 8(Start) 5 7 6 8(Pass) 4 3 1 8(Finish), `requiredLaps` 4, and `checkPointID` = track `instanceID`.
- Arch 8 is crossed in opposite directions.
- Shipped races can branch (liftoff-flight-analyzer track_format.py:199-245).
- props.json arch: legs are capsules (r 0.5, x = ±3.5), the bar sits at y 3.958, chamfer capsules sit at (±2.937, 3.399) at 45°, and the trigger box is centred at (0, 1.74, 0) with half-size (3.09, 1.847). That gives an aperture of 6.18 × 3.69 m.

**Runner, CI and dependencies**
- Runner `latency_s` keys are pinned by tests.
- CI installs only `--extra dev`: no cv2, no transformers, no GPU, but torch is core.
- There's no torchvision, and .venv-cuda has a hand-installed CUDA torch, so the proposal adds **no new dependencies**.

**HUD**
- Top-left mode/battery, top-centre timer/lap, top-right ALT/SPD, right ghost list, bottom-centre stick widgets, bottom-left "Rec", centre crosshair.
- The racing line can't be masked.

## Proposed Solution

### 1. Preset mechanism and packaged preset (AC1)

**New preset file**
- New `H\configs\liftoff_micro_arena.yaml`: an overlay (`harness: liftoff`) on liftoff_5inch.yaml. Every value carries a provenance comment.
- Telemetry: `rpm_count_prefix: true`, `gyro_units: deg_s`, `gyro_map` wx = −roll / wy = +pitch / wz = −yaw.
- Camera: resolution [2560, 1080], `hfov_deg` 120, `uptilt_deg` 20, Arena HUD masks (§5).
- Controls: rates 100/0/70, `hover_stick` −0.06.
- `world_scale` 4.0 (provisional).
- `gate_types` table (§2); `parts.gate_estimator: learned`; `gates.learned.weights: artifacts/harness/liftoff/models/liftoff-gates.pt`.
- `depth.calibration` from §15e.
- `offline.laps: null` (the route gives the laps); item_table for barriers / flags / concrete as in R\liftoff-arena-infinity.yaml.

**Loading**
- `config.load_config(harness, user_config, overrides, preset=None)`. The chain is default ← packaged preset ← user `--config` ← overrides.
- The preset's `harness` is checked against the requested harness. An unknown preset raises an error that lists the available presets.
- session.py and the offline entry points accept `preset`. The CLI gets `--preset` on run / eval / label / train-gates / depth-fit / pad-check.

**5-inch default and code defaults**
- liftoff_5inch.yaml gets the corrected `gyro_map` and `rpm_count_prefix`. It also gains the keys the overlay mechanism needs: a generic `gate_types` entry reproducing today's 1.5 × 1.5 gate, `camera.masks: []`, a `gates.learned` block, a `focus` section and `parts.focus_probe`.
- Airframe values stay unchanged.
- In telemetry.py, the `GyroMap` dataclass defaults and the `from_config` defaults change to wx = −roll, wy = +pitch, wz = −yaw, and the docstring is updated.

### 2. Gate-type table (AC1, shared by offline and runtime)

New `H\gate_types.py` (pure numpy):
- `GateType` fields: `class`, `match` (itemID regex), `aperture [w, h]`, `outer [w, h]`, `center_offset` (Unity-local aperture centre), planar `keypoints` (Unity-local x right / y up; default = the 4 outer corners), optional `flip_pairs`.
- `GateTypeTable.from_config` (validated), `classify`, `object_points(cls)` (keypoints relative to the aperture centre, for PnP), `class_index`.

Infinity Loop entry:
- `inflatable_arch`, `^InflatableArch`, aperture 6.18 × 3.69, outer 8.0 × 4.46, `center_offset` (0, 1.74, 0), open at the bottom.
- 10 keypoints, all on the visible tube surface, taken from props.json colliders:
  - feet outer (±4.0, 0)
  - leg outer at mid-height (±4.0, 1.5)
  - leg inner at mid-height (±3.0, 1.5)
  - chamfer outer midpoints: centre (±2.937, 3.399) offset 0.5 m outward along its normal
  - bar top (0, 4.458)
  - bar underside (0, 3.458)
- The upper half alone gives 4 or more non-collinear points.
- `flip_pairs` pair L with R.

### 3. Race route parsing (AC3)

In track_xml.py, `parse_race` detects `checkPointPassages`:
- Index passages by `uniqueId`, start at `passageType` Start, and follow `nextPassageIDs` until `passageType` Finish.
- At a branch, take the first successor and record the branch.
- Errors: a revisited `uniqueId`, a missing Start, or a dangling id.
- Returns a `RaceRoute` with ordered checkpoint ids, passage types, directionality, laps (`requiredLaps`), a finished flag and the branches.
- Circuit: per-lap sequence = walk minus the final Finish. Point-to-point: the whole walk, 1 lap.
- The legacy flat list returns the same shape and its behaviour doesn't change.

`load_track_course(..., gate_types=None)`:
- The route decides which items are gates. Class and dimensions come from gate_types (fallback: item_table `kind: gate`). An unmatched referenced itemID raises an error naming it.
- One Gate per passage (repeats allowed); centre via `center_offset`.
- Yaw: the plane normal comes from the instance rotation (existing code), and the sign is chosen per passage by prev/next travel. The two passages of arch 8 end up 180° apart.
- Laps = `requiredLaps` unless `offline.laps` overrides.

### 4. Capture loader

New `H\offline\capture.py`, `load_capture(dir, telemetry_cfg, world_scale=1.0, lag_ms=0)`:
- Parses the hex datagrams through `TelemetryFormat` (`received_at` = `t_wall_ns`/1e9).
- Pairs each frame to the nearest datagram within 30 ms.
- Segments at wall-clock gaps > 1 s, sim-time resets and position jumps (DiscontinuityDetector).
- Drops frames that are outside a segment or within 0.5 s of a cut, recording the reason.
- Also reads harness run folders.

### 5. Frame masks (AC7)

New `H\liftoff\masks.py`, `FrameMasks`:
- Regions are `{name, rect: [x, y, w, h]}` in pixels at `camera.resolution` (a resolution is required when masks are set), scaled to the frame.
- Methods: `apply`, `contains`, `valid_mask`.

Where the masks apply:
- Gate tracker: the image is masked before detection, and detections whose centre lies in a mask are dropped.
- Depth estimator: masked pixels become +inf after calibration, so they produce no ray hit and no cluster.
- Auto-labeller: keypoints in a mask are marked not visible, and training images are masked the same way.

### 6. Tracker refactor and learned estimator (AC5)

**Tracker refactor (gate_cv.py)**
- Extract the tracker into a base `TrackingGateEstimator(GateEstimator)` with an abstract `detect(image, state)`. `GateDetection` gains `cls`, and per-detection aperture is used throughout, including a `default_aperture` for the acquire fallback.
- Association prefers candidates of the same class. Masks are applied in the base.
- A shared helper runs planar PnP for arbitrary object points (IPPE, 4 or more points).
- `OpenCvGateEstimator`'s API and behaviour stay unchanged; the existing tests stay unmodified.

**New `H\liftoff\gate_net.py`** (pure torch, lazy): `GateKeypointNet`, CenterNet-style.
- Backbone → light FPN at stride 8 → heads: per-class centre heatmap, 2K keypoint offsets, K visibility logits.
- Backbones: `resnet18-hf` (transformers `microsoft/resnet-18`, pretrained, pure torch) and `tiny-cnn` (pure torch, for CI and as a fallback).
- Includes the decode step (max-pool NMS) and the loss (focal + masked L1 + BCE).

**New `H\liftoff\gate_learned.py`**, `LearnedGateEstimator(TrackingGateEstimator)`:
- Registered in builtins.py as `learned`; no runner edits.
- `gates.learned` keys: weights, device, fp16, input_width 768, score_threshold, min_keypoints 4, max_detections. Relative weight paths resolve against the repo root.
- The weights file carries metadata: classes, keypoint counts, a gate-type hash, backbone, input size, held-out split, captures, git head. On load it validates against `gate_types`; a mismatch is a config error.
- Weights load and warm up in `from_config`, so a missing or invalid file refuses the run before arming, with an error naming train-gates.
- `detect`: mask → `torch.nn.functional.interpolate` resize (no cv2, so this path runs in CI) → net → decode → keep visible in-image keypoints (at least `min_keypoints`) → PnP with the class's object points (cv2) → `GateDetection` with the class aperture.
- Results are cached per `frame.index`, and `info["inference_s"]` is recorded.

**Training (new `H\offline\gate_train.py`)**
- CLI: `drone-fly harness train-gates --labels <dir>... [--out, default A\models\liftoff-gates.pt] [--preset] [--epochs] [--backbone] [--holdout-frac 0.2 | --holdout t0:t1]`.
- Masked, resized images with colour, blur and brightness jitter, plus flip via `flip_pairs`.
- Time split: the last 20 % of each capture's in-flight time by default, stored in the metadata.
- AdamW with cosine schedule; writes training_log.json.
- If `microsoft/resnet-18` can't be downloaded, it fails with a clear message naming the model id and the `tiny-cnn` fallback.

### 7. Auto-labelling (AC4)

New `H\offline\autolabel.py`. CLI: `drone-fly harness label --capture --track [--race] [--props] [--preset] [--out, default A\labels\<capture-name>] [--preview N] [--lag-ms]`.
- Each gate-type item is placed with its position, full Unity Euler rotation and scale.
- Keypoints go Unity-local → Unity world → contract, and are projected with the harness `CameraModel.world_to_camera` + `project` (Liftoff metres).
- Output: labels.jsonl, with per gate: instance, class, keypoint pixels, visibility (in front, in image, unmasked), centre pixel, range, bearing.
- Output: labels_meta.json with the config snapshot, counts and drop reasons.
- `--preview` writes overlays (cv2). `--props` cross-checks gate_types against collider-derived outer bounds and trigger box, warning on mismatch over 5 %.
- A test pins the harness projection against an independent restatement of the verified reference convention (project_arches.py: Unity frame, inverse body rotation, then R_x(−uptilt), pinhole), to sub-pixel equality.

### 8. Offline gate scoring (AC6)

`evaluate.py` + CLI: `drone-fly harness eval --capture <dir> --labels <dir> [--preset] [--gate-weights] [--device]`. The existing `--session` mode stays.
- Uses only the held-out frames from the weights metadata.
- A labelled gate counts if its true range is 4–25 m and its centre is in the image. It's matched by a same-class detection within max(1 m, 0.25 × range).
- Per type: n, detection rate, median position error (m and % of range), median yaw error (180° folded), false detections per frame, inference latency p50/p95 (CUDA-synced).
- Targets are printed, not asserted.
- The report states that the time split is optimistic (one 4-lap capture, so held-out laps revisit the same viewpoints).
- `evaluate_frames` gains median ray error and the within-0.5 m share.

### 9. Depth calibration and raycast scoring (AC9)

New `H\offline\depth_fit.py`. CLI: `drone-fly harness depth-fit --capture --track [--race] [--props] [--preset] [--every N] [--report, default A\reports\depth-fit\]`.

Ground-truth sample points:
- Arch keypoint polylines, densified, with the surface taken as the centre line moved 0.5 m toward the camera.
- Barrier face grids.
- Flag axis points, offset by their radius.
- Kept if in front, in image, unmasked and 2–40 m away, with a coarse z-buffer to drop occluded points.

Fit and scoring:
- Robust fit (Huber/RANSAC) of scale and shift between the model's raw patch median and true camera-z (inverse form if `depth.inverse`).
- Reports pairs, before/after median absolute relative error, RMSE and per-range-bin errors.
- Then scores the synchronous `MonocularDepthFreeSpaceEstimator`, using the fitted calibration, against contract `raycast()` on the XML obstacles at training scale: median ray error, share within 0.5 m, in-FOV and all rays, coverage.
- Prints the YAML snippet.
- `--props` cross-checks item_table sizes, warning only.
- The report notes that arches aren't raycast obstacles.

### 10. Focus fail-safe (AC10)

- `FocusProbe` ABC in interfaces.py, plus a registry kind `focus_probe`.
- New `H\liftoff\focus.py`, `win32-foreground`: GetForegroundWindow, with the process name cached per HWND, compared to `focus.process_name` "Liftoff.exe" (optional title regex). It raises at start on non-Windows.
- `Harness.focus_probe` is built by the liftoff factory from `parts.focus_probe` + `focus`, and skipped in dry runs.
- Runner: after `_wait_for_state`, `_wait_for_focus` waits up to `focus.wait_for_focus_s` (15 s) with the pad neutral. Each tick, right after the abort check, a loss of focus stops the run with new reason `STOP_FOCUS_LOST`.

### 11. Pad stick-direction check (AC8)

New `H\liftoff\pad_check.py`. CLI: `drone-fly harness pad-check [--preset] [--config] [--magnitude 0.15] [--pulse-s 0.2] [--min-alt 0.5] [--max-alt 6] [--out, default A\reports\pad-check\<stamp>]`.
- Composes the UDP state source, `VgamepadActionSink` (new public `send_sticks`), the hotkey and the focus probe, with an injectable clock.
- Waits for telemetry and focus, climbs, then holds altitude with P+D throttle around `hover_stick` from telemetry Z.
- Per axis: alternating ± short pulses; measures the mean `FlightState.omega` against the expected `RateProfile.rate_deg_s`.
- Reports sign agreement, magnitude ratio and a suggested `stick_signs` block; writes report.json.
- Goes neutral and cuts throttle on F12, stale telemetry, focus loss, `altitude_out_of_band`, completion or error, and records the reason.

### 12. Latency and run artifacts (AC11, AC12)

- TickRecord gets a new `perception_s` {gate, free_space}; the runner times the two calls separately. `latency_s` keys stay unchanged.
- The depth worker publishes `info["inference_s"]`; the learned estimator does the same.
- RunRecorder writes latency_summary.json on close: p50/p95/p99/max for state, gate detector, depth (worker + update), policy, actuation and total; effective rate; missed deadlines; estimate ages; GPU name in meta.
- CLI: `drone-fly harness latency-report <run>` prints a Markdown table, recomputing from ticks.jsonl if needed.

**Fail-safe evidence (challenger note).** `_finish` records the final tick *before* `_failsafe` runs, so the neutral evidence can't be appended to that record. Pick whichever is simpler, and keep existing recorder tests green:
- the recorder writes one closing record after the fail-safe (event `failsafe:neutral` + the sink's `neutral_sticks()`), or
- the neutral output is written into meta.json beside `failsafe_ok`.

### 13. ADR (AC14), developer, per the UC-62 precedent

New `T\docs\adr\0003-real-liftoff-calibration.md` covers:
- the inverted gyro signs (with evidence) and the RPM count prefix
- linked-list routes: branches, double crossings, per-passage yaw
- multi-type gates via XML/props auto-labelling
- HUD masking (layout-specific); pad focus behaviour and the focus fail-safe
- the preset mechanism and the artifacts layout
- the time-split caveat and the one-time `microsoft/resnet-18` download
- the "add a gate style" workflow: capture → extract with liftoff-flight-analyzer (`python <clone>/src tracks|props`) → gate_types entry → label → train-gates → eval
- the live procedure
- a CLEARLY MARKED placeholder for the AC12 GPU table, which main fills in after the live run

Also edit `docs/adr/0002-live-harness-layering.md` with the `learned` and `focus_probe` rows and a pointer to 0003.

### 14. Live procedure (orchestrator with the user, from the worktree checkout)

Prerequisites:
- .venv-cuda with the harness extra.
- Liftoff at 2560 × 1080 fullscreen, telemetry on, "Test small drone", Arena "04 - Infinity Loop", ghosts off if possible.
- The §15 outputs present.
- Runs use `--output-root artifacts/harness/liftoff/runs`.

**Check after every run:** meta.json `stop_reason` and `failsafe_ok: true`; the final tick has the `stop:<reason>` event; the neutral evidence (§12) shows rate sticks at 0 and throttle at −1.

**Step 0. Model check.**
- Confirm `training/07-proc-course-flight/model_readout.pt` is 30-dim (README / UC-62 comment) and pick the newest 70-dim checkpoint (12-tunnels or 11-count-curriculum).
- Do a 1 s `--dry-run --duration 1` for each. A width mismatch fails `load_policy` before arming.
- Expect `completed`, a matching `meta.extra_dim` and a recorded sha256.

**Step 1. Pad check.**
- Run `pad-check --preset liftoff_micro_arena` and click into Liftoff within 15 s.
- Expect report.json. If the suggested `stick_signs` differ, update them.

**Step 2. Stage (a), 30-dim.** Overlay: `free_space_estimator: none`, `assume_no_obstacles: true`.

| Run | Trigger | Expected stop |
|---|---|---|
| a1 nominal | Start facing the first arch, about 60 s | `gate_lost`, `completed` or `user_abort` acceptable; check latency_summary, missed-deadline rate, gate age |
| a2 | F12 | `user_abort` |
| a3 | Alt-tab | `focus_lost`; also record what Liftoff does with its last input while unfocused |
| a4 | Overlay `parts.focus_probe: none`. First record whether a paused Liftoff keeps streaming (telemetry.csv during the pause), then quit the session from Liftoff's Esc menu, or exit the game, so the UDP stream stops | `stale_telemetry` within 0.2 s |
| a5 | Start on the ground facing away from every arch, with clearance from the wall (the runner hovers at `hover_stick` during acquire, so the micro lifts off). Fallback: an overlay with a single full-frame mask | `gate_lost` after `acquire_timeout_s` (3 s) |

**Step 3. Stage (b), 70-dim (acceptance target).** Overlay: `free_space_estimator: depth`, `--extra-dim 70`.
- b1 nominal: as a1, plus depth inference p50/p95 and free-space age.
- b2–b5: as a2–a5.
- b6: overlay `runner.max_estimate_age_s: 0.001`. Expect `perception_stale`. It may trip during warm-up; either is fine, as long as the stop reason is `perception_stale` and the fail-safe is clean.

**Step 4. Main's post-live edit.** Run `latency-report <b1 run>` and paste the table into the ADR 0003 placeholder, adding the a3/a4 observations.

**Step 5. Main's copy.** Copy `T\artifacts\harness\liftoff\` to `C:\dev\drone-fly\artifacts\harness\liftoff\` before any worktree removal.

### 15. Offline real-data pipeline

**Owner and rules**
- The developer runs it after the code and CI are green.
- GPU-heavy steps (train-gates, GPU eval, depth-fit) need a checkpoint message to main and clearance first. Label and mask measurement run on the CPU and need none.
- Inputs are R, read-only: the capture `2026-10-11-infinity-manual`, `LiftoffArenaTrack04_0001.xml`, `LiftoffArenaRace04_0001.xml`, props.json.
- All outputs go under A, never committed. Only the depth calibration and the HUD mask rects are committed, in the preset with provenance comments. No agent writes to the main checkout.
- Every report goes to main as a short summary plus its file path.

**Steps**
- **(a) Masks.** Measure on 3 or more named frames, including `frames\1791717949360510000.jpg`. Preset comments name the frames. Mask overlays go to `A\reports\masks\`.
- **(b) Label.** Run `label --preset liftoff_micro_arena --capture … --track … --race … --props … --out A\labels\2026-10-11-infinity-manual --preview 12`. Previews go to `A\reports\label-preview\`. Report counts, drop reasons and the alignment spot-check verdict.
- **(c) train-gates (GPU).** Writes `A\models\liftoff-gates.pt` and training_log.json.
- **(d) eval (GPU), held-out only.** Writes `A\reports\gate-eval\eval_report.json`. Report the per-type numbers. If detection is under 80 % or median error is over 15 % of range, escalate to the user via main (more captures vs provisional thresholds); no silent pass.
- **(e) depth-fit (GPU).** Writes `A\reports\depth-fit\depth_fit_report.json`. The fitted values go into the preset. Report the fit error and ray scores.
- If any step fails (download, GPU, out of memory), report to main, and main takes over.

**QA verification**
- Review labels_meta.json and at least 3 previews plus the mask overlays.
- Re-run eval on the held-out split from the developer's weights. GPU only with clearance; otherwise `--device cpu`, with latency marked non-representative. Numbers must match within detection ±2 pts and median errors ±10 %.
- Check the preset values equal the depth-fit report and mask notes.
- Check git staging has nothing under `artifacts/` and no game data.
- Re-run outputs go to `A\reports\qa\`. Pass/fail goes to main with paths.

## Files Affected

**Production code (developer)**, `src/drone_fly/**`, plus docs per the UC-62 precedent:
- New:
  - H\configs\liftoff_micro_arena.yaml
  - H\gate_types.py
  - H\liftoff\masks.py, focus.py, gate_net.py, gate_learned.py, pad_check.py
  - H\offline\capture.py, autolabel.py, gate_train.py, depth_fit.py
  - T\docs\adr\0003-real-liftoff-calibration.md
- Modified:
  - H\configs\liftoff_5inch.yaml
  - H\config.py, session.py, interfaces.py, registry.py (KINDS += `focus_probe`), builtins.py, runner.py, types.py, recording.py
  - H\liftoff\__init__.py, telemetry.py, gate_cv.py, depth.py, gamepad.py
  - H\offline\track_xml.py, evaluate.py
  - T\src\drone_fly\cli\main.py
  - T\docs\adr\0002-live-harness-layering.md
- No pyproject.toml or uv.lock change.

**Test code (QA)**, `tests/**`:
- New fixtures:
  - tests\data\harness\liftoff_real_datagrams.csv: a few seconds of real 97-byte datagrams with rotation on all axes, trimmed from R
  - tests\data\harness\extract_real_datagrams.py, documenting the source window
  - a synthetic race_passages.xml plus track fixture: out-of-order passages, a double crossing in opposite directions, a branch
- Regenerated: liftoff_telemetry.csv, via the updated generator.
- New tests:

| Test | Covers |
|---|---|
| test_harness_real_telemetry.py | AC2: ok under the preset, \|corr\| ≥ 0.95, \|scale\| 1 ± 0.05, ok=False under old signs, 97 bytes / RPM count |
| test_harness_presets.py | AC1 + provenance comments + 5-inch fixes + unknown preset + harness mismatch |
| test_harness_race_route.py | AC3 + legacy + 180° yaw pair + branch + errors |
| test_harness_gate_types.py | gate-type table |
| test_harness_autolabel.py | AC4: reference-convention equality, visibility / mask / drop rules, --props warning |
| test_harness_gate_learned.py | AC5: tiny-cnn decode + interpolate path in CI, PnP via class dims (cv2 importorskip), tracker semantics, registry, weights/class validation, CPU train-gates smoke |
| test_harness_masks.py | AC7: fake `TrackingGateEstimator` subclass, CI; depth no-hit / no-cluster |
| test_harness_focus.py | AC10: wait timeout, mid-run loss → `focus_lost` + fail-safe + neutral evidence |
| test_harness_pad_check.py | AC8: fakes, sign suggestion, every abort reason including altitude band |
| test_harness_latency.py | AC12: `perception_s`, latency_summary.json, latency-report |

- Also new: CLI parser tests for the new subcommands; local opt-in real-data tests (env `DRONE_FLY_LIFTOFF_DATA` → R, skip when absent) for label, eval and depth-fit; a guard test that tests/data contains no Liftoff XML or game frames (AC13).
- Updated, deliberately: assertions encoding the falsified gyro convention in test_harness_liftoff_telemetry.py (`test_default_gyro_map_units_axes_and_signs`), test_harness_gyro_check.py (suggested signs) and test_harness_scale.py if affected. Recorder tests stay green.
- QA also verifies the §15 outputs.

## Risks & Considerations

- **AC15 exception.** Correcting the gyro defaults changes a few existing assertions that encode the wrong convention. This is called out in the PR; every other existing test stays unmodified.
- **Data volume and optimism.** About 380 in-range arch frames from one capture, one gate type, one track. AC6 may miss its targets, and the time split is optimistic. Likely request: 2–3 more captures. Thresholds are reported, not asserted, and escalated if missed.
- **Detector choice** (the UC's open question). CenterNet keypoints + PnP with a ResNet-18 via transformers: no new dependencies, and a new gate style is a table entry. Needs a one-time HF download.
- **Detection gaps.** Close or partial gates whose centre is off-image aren't detected (tracker hold and pass rules cover the approach). Label occlusion isn't modelled.
- **GPU budget.** Detector + depth + policy on an RTX 3050 under Liftoff may miss 50 Hz. Mitigations: fp16, input width, per-frame caching, the depth age limit. Measured and logged.
- **Focus.** While Liftoff is unfocused, the neutral command may not reach it; its behaviour is observed live (a3) and recorded in the ADR. Whether a paused Liftoff keeps streaming is observed in a4.
- **Depth truth.** Arches aren't raycast obstacles, so ray truth ignores arch legs. Ghost drones can't be masked.
- **Layout-specific masks.** HUD masks are specific to 2560 × 1080 and the Arena HUD.
- **Provisional values.** The 70-dim checkpoint is chosen at run time; the rate gap and `world_scale` 4 are provisional. Both are acceptable under the I/O-only criteria.
- **Out of scope.** Branching races take the first successor; resizable gate subtypes aren't handled.
- **Artifacts.** Lost with the worktree unless main runs step 5.
- **Untouched.** training/, gpu_prototype/, docs/INFERENCE_CONTRACT.md, training configs, PROJECT_BRIEF.md (stale `stack.frameworks` ignored).
- **Never committed.** Game data: track XMLs, props, frames.

## Challenger verdict

**Approve** (round 2). Every AC (1–15) maps to a section; every UC pitfall is acknowledged in Risks; scope creep minimal (`latency-report` and session-mode depth metrics tied to AC9/AC12); nothing under training/, gpu_prototype/ or docs/INFERENCE_CONTRACT.md changes; no new dependencies. Non-blocking notes: (i) fail-safe neutral evidence must be written after `_failsafe` (see §12); (ii) a5 needs wall clearance because the runner hovers during acquire; (iii) b6 may trip during depth warm-up — expected stop `perception_stale` either way.
