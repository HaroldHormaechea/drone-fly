# ADR 0003: Calibrating the live harness against real Liftoff

- **Status:** accepted (UC-64)
- **Date:** 2026-10-11
- **Builds on:** [ADR 0002](0002-live-harness-layering.md) (harness layering)
- **Code:** `src/drone_fly/harness/` (preset `configs/liftoff_micro_arena.yaml`, `gate_types.py`,
  `liftoff/{gate_learned,gate_net,masks,focus,pad_check}.py`,
  `offline/{capture,autolabel,gate_train,depth_fit}.py`)

## Context

UC-62 built the harness and tested it against fakes and synthetic recordings. UC-64 connects it to
a live Liftoff session. The reference setup is **Liftoff Arena, race "04 - Infinity Loop"**, flown
with the **micro "Test small drone"**: FT Gremlin frame, FX805 micro camera, lens "None",
ZetaFlight Acro, Betaflight rates 100/0/70, game at 2560x1080 fullscreen.

All the evidence comes from one hand-flown capture,
`artifacts/harness/captures/2026-10-11-infinity-manual`:

- 27,868 raw UDP datagrams;
- 2,043 wall-clock-stamped frames;
- 182 s of continuous flight, plus some shorter segments.

The track and race XMLs and the prop collider shapes were extracted with liftoff-flight-analyzer.
None of that data is committed: it is game data, gitignored under `artifacts/`.

## Findings

### Telemetry

- **The roll and pitch gyro signs were inverted.** `gyro_check` finite-differences the
  quaternion stream into body rates. Run over the 182 s segment (18k samples), it gives
  |correlation| 0.99 and |scale| 1.00 in deg/s, but only with `wx = -roll`, `wy = +pitch`,
  `wz = -yaw`. The UC-62 defaults (`+roll`, `-pitch`) scored a sign agreement of 0 on roll and
  pitch.
- **MotorRPM carries a uint8 count prefix.** Every datagram is 97 bytes: 23 floats, plus 1 count
  byte before the 4 RPM floats.
- Both findings describe Liftoff, not the airframe. They are now the defaults in `GyroMap` and in
  `liftoff_5inch.yaml`, and the preset also sets them explicitly.
- A committed fixture of real datagrams pins this convention in CI.

### Race routes are linked lists

Liftoff 1.4 race files list `checkPointPassages`. Each passage has:

- a `uniqueId`;
- a `checkPointID` (the track `instanceID`);
- a `passageType` (Start / Pass / Finish);
- a `directionality`;
- `nextPassageIDs`.

Passages are stored out of order. The Infinity Loop walk is 8 (Start), 5, 7, 6, 8 (Pass), 4, 3,
1, 8 (Finish), with `requiredLaps` 4, so the centre arch is crossed twice per lap.

`offline/track_xml.parse_race_route` follows the list from Start to Finish:

- It takes the first successor at a branch and records the branch. Shipped races can branch.
- It rejects loops, a missing Start and dangling ids.
- When Start and Finish share a checkpoint, the route is a circuit: the per-lap sequence is the
  walk minus the final Finish. A point-to-point route is one lap.

Each passage becomes one ground-truth gate. Its yaw sign follows the travel direction from the
previous passage to the next one, so a gate crossed both ways gets two yaws 180° apart.

On Infinity Loop both centre-arch crossings actually go the same way through the arch plane: the
two diagonals of the figure-eight. The telemetry confirms it, with all 7 crossings at about -y in
the contract frame. The two passages therefore get the same yaw there, even though their
`directionality` labels differ.

The legacy flat `instanceID` race format still works unchanged.

### Gate styles: multi-type learned detection, labelled automatically

Infinity Loop uses dark inflatable arches (`InflatableArchBrandless01`). They are open at the
bottom, so the UC-62 red-quad HSV detector cannot see them. Later courses mix up to about three
gate styles. Rather than writing a detector per style:

- **A gate-type table** (`gate_types`, in `harness/gate_types.py`) maps an `itemID` regex to a
  class, aperture and outer size, aperture centre and planar keypoints on the visible frame.
  For the arch these are 10 points on the tube surface, from the props.json colliders.
- **Auto-labelling** (`drone-fly harness label`) projects every gate of every configured type into
  each capture frame:
  1. The XML gives position, Euler rotation and scale.
  2. The point goes from Unity-local to the contract frame.
  3. The harness `CameraModel` projects it: pinhole, camera pitched up by `uptilt`, frames paired
     with telemetry by wall clock.

  On the reference capture the projected arches land within a few pixels. Frames in
  respawn/discontinuity segments are dropped.
- **Left/right keypoints are view-relative.** A symmetric gate looks the same from the front and
  from behind. The labeller swaps each type's `flip_pairs` when the gate is seen from behind, and
  flip augmentation swaps them too. PnP then recovers the mirror-equivalent pose: same centre,
  with the normal sign resolved along the travel direction by the tracker.
- **The network** (`liftoff/gate_net.py`) is a CenterNet-style keypoint net:
  - backbone (ResNet-18 through `transformers`, ImageNet-pretrained) → light FPN at stride 8;
  - heads for a per-class centre heatmap, keypoint offsets and keypoint visibility.

  It adds no new dependency; `tiny-cnn` is the offline / CI fallback.
- **The `learned` estimator** (`liftoff/gate_learned.py`) runs the net, keeps visible unmasked
  keypoints (at least 4), and solves planar PnP against the class's object points. It reuses the
  UC-62 tracker unchanged: acquire, hold, lost, pass, next gate, encodings. That tracker is now the
  shared `TrackingGateEstimator` base, so the OpenCV estimator behaves exactly as before.
- **Weights validation.** The weights file records classes, keypoints, a gate-type hash, the
  backbone config, the input size, the held-out split, the captures and the git head. A mismatch
  with the configured `gate_types` refuses the run before the pad is armed.

### HUD masking (layout-specific)

The Arena HUD covers parts of the frame:

- top-left: mode and battery;
- top-centre: timer and lap;
- top-right: ALT/SPD;
- right side: the ghost list;
- bottom-centre: the stick widgets;
- bottom-left: "Rec".

`camera.masks` lists pixel rectangles at `camera.resolution`, scaled to the frame. Masks are
applied in three places:

- Gate detection: the image is masked, and detections centred in a mask are dropped.
- Depth: masked pixels become `+inf`, so they produce no ray hit and no obstacle cluster.
- Labelling: masked keypoints are marked not visible, and training images are masked the same way.

The preset ships the measured rectangles. They are valid only for 2560x1080 and this HUD. Some
things deliberately stay unmasked:

- the centre crosshair, because masking it would drop every gate dead ahead;
- the racing line and AI ghost drones, which cannot be masked.

### Focus fail-safe and the pad stick-direction check

The ViGEm pad's input goes to whichever window has focus. The `win32-foreground` focus probe
(`parts.focus_probe`) compares the foreground window's process with `focus.process_name`
(`Liftoff.exe`):

- After arming, the runner waits up to `focus.wait_for_focus_s` (15 s) with the pad at neutral.
- After that, losing focus stops the run with `focus_lost`.
- Dry runs build no probe.

`drone-fly harness pad-check` sends small, known stick pulses while holding altitude with a P+D
throttle loop around `hover_stick`. It reports each axis's body-rate sign and magnitude against
the configured rate curve, and suggests `controls.stick_signs`. It goes neutral on F12, stale
telemetry, loss of focus, leaving the altitude band, completion or error.

The run folder's `meta.json` now records the sink's neutral output (`failsafe_output`) beside
`failsafe_ok`. That is the evidence that every stop leaves the rate sticks centred and the
throttle at -1.

> Observed live (filled in after the staged run): _what Liftoff does with its last input while
> unfocused (run a3), and whether a paused Liftoff keeps streaming telemetry (run a4)._

### Presets and the artifacts layout

A **preset** is a packaged YAML overlay on a harness default. The config chain is: default ←
`--preset` ← `--config` ← CLI overrides. `--preset liftoff_micro_arena` carries every calibrated
value, each with a comment naming its measurement source:

- telemetry fixes;
- 2560x1080, 120° HFOV, 20° uptilt;
- HUD masks;
- rates 100/0/70 and `hover_stick` -0.06;
- provisional `world_scale` 4;
- the arch gate type;
- `parts.gate_estimator: learned`;
- the fitted depth calibration;
- the obstacle item table.

An unknown preset, or a preset for another harness, is a config error.

Liftoff outputs live under `artifacts/harness/liftoff/` (gitignored), organised by kind:

| Folder | Contents |
|---|---|
| `labels/<capture>/` | `labels.jsonl` and `labels_meta.json` |
| `models/` | `liftoff-gates.pt` and `training_log.json` |
| `reports/{masks,label-preview,gate-eval,depth-fit,pad-check}/` | offline reports |
| `runs/` | live runs (`--output-root artifacts/harness/liftoff/runs`) |

Each live run folder also contains `latency_summary.json`. `drone-fly harness latency-report <run>`
prints it as a table.

### Depth calibration

`drone-fly harness depth-fit` fits `depth.calibration` (`metres = scale * raw + shift`) on the
capture, against XML-derived surfaces:

- arch tube centre lines moved by the tube radius toward the camera;
- barrier face grids;
- flag axes.

Points are kept if they are in front, in the image, unmasked, 2–40 m away and not occluded
according to a coarse z-buffer. The fit is robust (RANSAC, then Huber). depth-fit then scores the
reconstructed 40-ray scan against the contract `raycast()` on the XML obstacles at training scale.
Arches are gates, not raycast obstacles, so the ray truth ignores their legs.

## Training splits and model selection

`train-gates` cuts each capture's in-flight time into three disjoint slices, all recorded in the
weights metadata (`splits`):

| Slice | Default | Used for |
|---|---|---|
| held out | the last 20 % (or `--holdout t0:t1`) | `eval --capture` only; never loaded for training |
| validation | the 10 % immediately before the held-out slice | choosing the epoch to keep |
| train | the rest | gradient steps |

A 2 s guard gap separates neighbouring slices, to limit leakage between consecutive frames.

By default the epoch with the lowest validation loss is kept, with early stopping after 10 epochs
without improvement. `--select last` keeps the final epoch instead.

This is a plan amendment made during UC-64. The first run kept only the last epoch, and on the
single capture the held-out loss rose from epoch ~7 (about 3.3) to about 6 by epoch 20, while the
train loss kept falling: the model was overfitting. Choosing the epoch on the held-out frames
would have contaminated the evaluation, so model selection got its own validation slice.
`eval --capture` checks that the held-out range is disjoint from the train and validation ranges
before scoring.

**Known limitation: the selection signal.** The validation loss is a weighted sum, and the
heatmap focal term dominates it. On the reference capture it bottoms out at epoch 3, while the
keypoint offsets are still under-trained. The predicted keypoint spread on held-out arches is about
0.72x the true spread, so PnP places gates about 1.4x too far.

Detection on the held-out laps (faster, banked, motion-blurred) was 19.6 %:

- the network peaked on only about half of the in-range arches;
- most of the remaining misses are range errors outside the matching tolerance.

On training frames the same pipeline finds about 96 % of the arches, so the shortfall is
generalisation from one capture, not a pipeline bug. A keypoint- or detection-based selection
metric is a candidate follow-up; it was deliberately not tuned in this proof of concept.

**Evaluation model vs. deploy model.** There are two weight files:

| File | Trained on | Purpose |
|---|---|---|
| `models/liftoff-gates.pt` | the three-way split | the honest held-out measurement above; the preset's default weights |
| `models/liftoff-gates-deploy.pt` | every labelled frame (`--holdout-frac 0 --val-frac 0 --select last`) | the live proof-of-concept run, through an overlay |

The deploy model is trained for 30 epochs so that the keypoint offsets converge. Its metadata says
`purpose: deploy`, and `eval --capture` refuses it, because no frame of it is held out. It
memorises Infinity Loop by design. Its numbers are in-sample and cannot stand in for the AC6
measurement.

## Offline evaluation and its caveats

`drone-fly harness eval --capture … --labels …` scores the learned detector only on the held-out
frames named in the weights metadata. By default these are the last 20 % of each capture's
in-flight time. Per gate type, for gates 4–25 m ahead with their centre in the image, it reports:

- detection rate (target ≥ 80 %);
- median position error (target ≤ 15 % of range);
- median yaw error (sign-folded);
- false detections per frame;
- CUDA-synchronised inference latency.

The targets are printed, not asserted. For UC-64, which is a proof of concept of the end-to-end
chain (telemetry → senses → policy → pad), the owner decided on 2026-10-11 to treat the 80 % / 15 %
targets as provisional:

- A miss is reported but does not block.
- Memorising Infinity Loop is acceptable.
- More tracks will be captured later.

The eval report carries the same note.

**The time split is optimistic.** With one 4-lap capture, the held-out laps revisit the
viewpoints seen in training. About 380 in-range arch frames, from one gate type on one track, are
expected to be too little and too correlated to generalise. The likely next step is 2–3 more
hand-flown captures, including other tracks.

The `microsoft/resnet-18` backbone is downloaded once, at training time. Loaded weights carry the
backbone config, so the runtime never downloads anything.

## How to add a gate style

1. **Capture**: fly the course by hand while recording raw telemetry datagrams and wall-clock
   stamped frames, using the owner's `capture.py`. The layout is `telemetry_raw.csv` +
   `frames.csv` + `frames/`. A harness run folder also works.
2. **Extract track data** with liftoff-flight-analyzer, from its clone:
   `python <clone>/src tracks` writes the track/race XMLs, and `python <clone>/src props` writes
   `props.json` with collider shapes.
3. **Add a `gate_types` entry**:
   - the class and the `itemID` regex;
   - aperture and outer size, and the aperture centre (from the trigger box and colliders);
   - 4 or more planar keypoints on the visible frame, left/right pairs listed as `flip_pairs`;
   - optionally the tube `outline` and `tube_radius` for depth-fit.
4. **Label**:
   `drone-fly harness label --preset … --capture … --track … --race … --props … --preview 12`.
   Check the counts and drop reasons, the `--props` warnings and the previews (keypoints on the
   gate).
5. **Retrain**: `drone-fly harness train-gates --preset … --labels <all label folders>`.
   Changing `gate_types` changes the gate-type hash, so old weights and labels are refused until
   regenerated.
6. **Evaluate**: `drone-fly harness eval --preset … --capture … --labels …`. Gather more captures
   if a type misses its targets.

## Live procedure (summary)

Prerequisites:

- `.venv-cuda` with the harness extra;
- Liftoff at 2560x1080 fullscreen with telemetry on;
- the "Test small drone" on Arena "04 - Infinity Loop", ghosts off if possible;
- trained weights and the fitted preset.

Every run uses `--preset liftoff_micro_arena --output-root artifacts/harness/liftoff/runs`. After
each run, check:

- `stop_reason` and `failsafe_ok: true` in `meta.json`;
- the final tick has a `stop:<reason>` event;
- `failsafe_output` shows centred rate sticks and the throttle at -1.

Steps:

1. **Model check**: a 1 s `--dry-run --duration 1` of the 30-dim checkpoint and of the newest
   70-dim checkpoint. A width mismatch fails before arming.
2. **`pad-check --preset liftoff_micro_arena`**: click into Liftoff within 15 s, then apply the
   suggested `stick_signs` if they differ.
3. **Stage (a), 30-dim**, with `free_space_estimator: none` and `assume_no_obstacles: true`:

   | Run | Trigger | Expected stop |
   |---|---|---|
   | a1 nominal | none | `gate_lost`, `completed` or `user_abort` |
   | a2 | F12 | `user_abort` |
   | a3 | alt-tab | `focus_lost` |
   | a4 | quit the session (probe off) | `stale_telemetry` |
   | a5 | start facing away from every arch | `gate_lost` |

4. **Stage (b), 70-dim, the acceptance target**, with `free_space_estimator: depth`. Runs b1–b5
   repeat a1–a5, plus:

   | Run | Trigger | Expected stop |
   |---|---|---|
   | b6 | `runner.max_estimate_age_s: 0.001` | `perception_stale` |

5. Run `latency-report <b1 run>` and paste the table below.

## GPU budget (stage b, RTX 3050, Liftoff rendering)

> **PLACEHOLDER — filled in by the orchestrator after the live run.** Paste the output of
> `drone-fly harness latency-report <b1 run folder>` here. It covers per-stage p50/p95/p99/max for
> state, gate detector, depth worker and update, policy, actuation and total, plus the effective
> rate and the missed-deadline share.

## Consequences

- Calibrated values are packaged and traceable to their measurement. A new airframe or track is a
  new preset, and a new gate style is a table entry plus a retrain.
- The 5-inch default now carries the corrected Liftoff telemetry convention. Its airframe values
  stay unverified (`CALIBRATE`).
- `world_scale` 4 and the rate gap are provisional: the models are capped at about 230 °/s, while
  the pilot flies at about 2,600 °/s. The live run is judged on I/O and fail-safe behaviour, not
  on flight success.
- Detection depends on how much capture data exists. Thresholds are reported and escalated, never
  silently relaxed.
- Branching races follow the first successor only. Resizable gate subtypes are not handled.
