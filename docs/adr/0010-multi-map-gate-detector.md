# ADR 0010: A multi-map gate detector trained on catalogue-linked captures

- **Status:** in progress (UC-71). The CPU pipeline (P0b) is landing checkpoint by checkpoint; the
  live captures (P1, P2), the GPU training (P4, P5) and the `liftoff-gates-v2` release (P6) are
  pending (§ Results).
- **Date:** 2026-10-13
- **Builds on:** [ADR 0003](0003-real-liftoff-calibration.md) (the UC64 learned detector v1, the
  reference capture), [ADR 0007](0007-track-library-and-teacher.md) (the track catalogue and its
  map-level split), [ADR 0009](0009-student-distillation-and-detector-noise.md) (the detector-noise
  model, the held-out window)
- **Code:** `src/drone_fly/harness/` (`gate_types`, `config`, `configs/gate_types/liftoff.yaml`,
  `liftoff/gate_learned`, `offline/capture_index`, `offline/capture`, `offline/align`,
  `offline/autolabel`), `src/drone_fly/tracks/` (`gate_coverage`, `liftoff`, `cli`),
  `src/drone_fly/cli/main.py`

## Context

The v1 detector (`liftoff-gates-v1`) was trained on one hand-flown capture of one map (Infinity
Loop) and one gate type (the inflatable arch). On its held-out frames it detects 19.6 % of the gates
at 4-25 m and none below 8 m (ADR 0009). Flying random Liftoff courses needs a detector that works
across maps, lighting and gate styles, scored on maps it never saw.

## Decision

### 1. Two couplings are cut first

- **H1, the sim conversion.** The conversion fingerprint hashes the gate table. The table now lives
  in its own file (`configs/gate_types/liftoff.yaml`, referenced from the `liftoff` preset) and
  grows; `tracks.liftoff.harness_tables()` keeps only `CONVERSION_GATE_CLASSES`
  (`inflatable_arch`) projected to the 9 keys it had before UC71. The default conversion
  fingerprint stays `639beb3ef1bd8bec` and every converted course is byte-identical (a pin test,
  committed before the change). New gate types reach the sim only through a deliberate
  `CONVERSION_VERSION` bump.
- **H2, the weights.** Weights embed the gate table they were trained with (v1 does). The
  estimator decodes and solves PnP with that table, checked against the weights' own
  `gate_types_hash`; a class whose dimensions differ from the configured table only warns. A grown
  table therefore does not make v1 unloadable. Weights without an embedded table keep the strict
  check.
- **Releases.** `gate_learned.RELEASES = {tag: {file: sha256}}`; file names are unique across
  releases, so a configured path still resolves by its file name. The preset selects a release with
  `gates.learned.release` (`v1` until P6); an explicit `weights:` always wins.

### 2. Gate types and coverage

- One class per gate geometry; `match` covers the branding variants. `bottom: open|closed` is
  required in a gate-types file, `family` is optional; neither enters the table hash.
- Every item of a captured track is exactly one of: a visible gate class, a checkpoint volume
  (`Checkpoint*` / `TriggerBox*`, never labelled), a `not_gates` entry, or **unknown** (any route
  item, or any itemID that looks like a gate, that is none of the others). `tracks check
  --gate-coverage` lists unknown itemIDs and route passages with no visible gate (volume-only);
  `--require-gate-coverage` fails on either; `harness label` refuses a track with unknown items.

### 3. Captures carry a `meta.json`

Each capture names its catalogue entry (`catalogue_id`), the track/race hashes, how it was made
(`source`, `recording`, `pose_source`, `time`), its camera and extra masks, the drone profile,
completeness and the alignment verdict. The split is always re-read from the catalogue. New
captures live at `artifacts/harness/captures/<Map>/<session>/` with globally unique session names
`<map-kebab>-<YYYY-MM-DD>-<replay|manual>-<NN>`; the legacy reference capture
`2026-10-11-infinity-manual` keeps its name and place (v1 metadata and the r5 noise model key on
it) and gets a `meta.json` through `captures register`. Field reference:
`harness/offline/capture_index.py`; user steps: `docs/harness/liftoff-captures.md`.

The reference capture registers as `liftoff-arena-04` (LiftoffArena, train), completeness
`incomplete` (25 of 33 route crossings, judged from its telemetry with the env pass rule).

### 4. Projection alignment (`captures verify`)

The score is the image gradient **across** each projected gate edge (`|g . n|`, Sobel on a 640 px
wide, per-frame mean-normalised image), averaged over sampled frames with a gate within 25 m, divided
by the same score at seeded random pose displacements (2-4 m, 5-15 deg yaw). Frames come from the
paired, static-trimmed set: a parked drone says nothing about the clock. A grid search covers the
time offset (+-1.5 s, 20 ms) and optionally the uptilt (+-5 deg, 1 deg).

**Threshold, calibrated 2026-10-13 on the reference capture** (`captures verify --calibrate`,
114 frames):

| Pose | Margin |
|---|---|
| best (offset +0.06 s, uptilt 20 deg) | **1.40** |
| uptilt +3 deg | 1.15 |
| uptilt -3 deg | 1.09 |
| clock +0.5 s | 1.05 |
| clock -0.5 s | 1.03 |

The plan's provisional `MARGIN_MIN` of 1.5 is **not reached** by the correct pose of a known-good
capture, so it was lowered to **1.25**, midway between the correct pose and the nearest wrong one
(**deviation from the approved plan**). With `--search-uptilt` the best cell is uptilt 19 deg,
offset 0.0 s, margin 1.54 (preset: 20 deg; the drone file says 20.03 deg) - a 1 deg uptilt residual,
inside the search step. `verified` also requires >= 50 frames with a gate in view and a best cell
that is not on the edge of the search range. The reference capture is `verified: true` (margin
1.40, offset +0.06 s).

**Camera check (CP6b, a fix of the CP6 design).** QA showed on synthetic renders that the absolute
margin does not reject a wrong camera: with the correct camera at margin 2.2-2.6, an hfov error of
-10 deg still scored 1.53 (-20 deg: 1.37) and an uptilt error of -10 / +10 deg 1.38 / 1.60, all above
1.25 - the arch legs are vertical, so pitch and scale errors keep many edge samples on them, and
hfov was never searched. `verify` therefore also re-scores, at the best offset only (no lag x camera
grid), an uptilt +-6 deg (1 deg steps) x hfov +-10 deg (5 deg steps) grid around the camera under
test, and the capture is unverified when the peak is more than one step away on either axis, with
its own reason (`camera off: best hfov 130 vs meta 120`). The check runs only when the margin
passes (without signal the peak location is noise). With `--search-uptilt` the camera under test
carries the searched uptilt, which `verify` then writes to `meta.camera` (`source: align`). The
absolute margin and the edge rule stay; there is no score-ratio criterion (one rule, the peak
location). On the reference capture the peak is exactly the preset camera (uptilt 20, hfov 120):
still `verified: true`, margin 1.40. The verdict records each axis neighbour's score relative to
the camera under test (`alignment.camera_check.neighbours`); reference capture:

| Neighbour | uptilt -6 | -3 | +3 | +6 | hfov -10 | -5 | +5 | +10 |
|---|---|---|---|---|---|---|---|---|
| score / centre | 0.76 | 0.81 | 0.81 | 0.85 | 0.72 | 0.80 | 0.84 | 0.84 | Runtime: 65 extra re-scores,
about 11 s per `verify --search-lag` in total on the CPU (was about 12 s; frame loading dominates).

**P1 gate (mandatory).** The gap between the correct pose (1.40) and the nearest wrong one (1.15)
is thin. Before strategy B (recording-XML pose) is trusted, `captures verify --calibrate` is re-run
on the first P1 replay capture (replay HUD masks may shift the baseline); if the correct pose does
not clear 1.25 with the wrong poses clearly below, or the camera check fails, strategy B is not used
until the score is revised. The result is recorded here.

### 5. Dataset and training (`harness gates train`, `gates dataset-report`)

- **Split routing from the catalogue, one choke point.** `gate_dataset.assemble` routes train-map
  captures to training and val-map captures to model selection; a held-out-map capture raises
  `TrackLeakageError` (never silently filtered). Captures without a catalogue id, with unverified
  alignment, uncovered items, no labels or labels from another gate table are refused with the
  reason.
- **v1's held-out window stays held out.** Frames inside the window recorded by
  `--protect-holdout-of` (default the v1 release weights) are never loaded; the new weights record
  the same window (`splits` / `holdout`), so `detector_noise.heldout_window` and the Infinity Loop
  regression keep using it. Val-map captures go into `selection_captures`, not `captures`, so they
  count as never trained.
- **Selection by val-map detection rate** (`--select val-detection`, every `--val-every` 2 epochs:
  decode + PnP + the eval matching rule; ties to the lower median relative error), not val loss (the
  focal term dominated the v1 val loss, ADR 0003). Map x class balanced sampling; a static-trim drop
  reason; a `uint8` memmap frame cache keyed by the data manifest's sha.
- **G1** (`gates dataset-report`, exit 0 only on GO). On the reference capture alone: NO-GO on every
  data criterion (291 in-range arch gate-frames, 36 at 4-8 m, one map), protected window OK (324
  frames, 148 in-range arch gate-frames: exactly v1's window).

### 6. Evaluation and miss attribution (`harness eval --detector`, `gates diagnose`)

`harness eval --detector <weights|dir> --split held-out [--compare <v1>]` scores every frame of the
held-out-map captures (refusing any capture the weights trained or selected on), per map x type and
per type pooled, with range bins 4-8 / 8-15 / 15-25 m, the ADR 0003 metrics on matched-by-centre
detections (median signed range error, keypoint-spread ratio), yaw error, false detections per frame
and latency (CPU-measured latency is flagged). Targets 70 % / 20 % under the D-U1 rule (accepted,
§ User decisions); `--compare` scores both detectors on v1's protected Infinity Loop window and
passes when the new arch detection at 4-25 m is >= v1 - 5 pp (the 4-8 m bin is tracked, not gated).

**v1 diagnosis (P0b, CPU)**: `gates diagnose --weights liftoff-gates.pt --captures
2026-10-11-infinity-manual` on v1's held-out window (324 frames). It reproduces ADR 0009's 19.6 % and
ADR 0003's 0.72 spread ratio:

| range m | n | detected | no centre peak | class confusion | few keypoints | PnP range | signed range err | spread ratio |
|---|---|---|---|---|---|---|---|---|
| 4-8 | 15 | 0.0 % | 3 | 0 | 0 | 12 | +147.5 % | 0.40 |
| 8-15 | 38 | 26.3 % | 9 | 0 | 0 | 19 | +37.8 % | 0.72 |
| 15-25 | 95 | 20.0 % | 52 | 0 | 0 | 24 | +25.5 % | 0.80 |
| 4-25 | 148 | 19.6 % | 64 | 0 | 0 | 55 | +37.9 % | 0.72 |

Reading: below 15 m v1 finds the gate (a centre peak with enough keypoints) but squeezes the
keypoints (spread 0.40-0.72), so PnP places it 38-148 % too far and the world match fails (31 of 53
misses are `pnp_range`); beyond 15 m most misses are no centre peak at all (52 of 76). The R2 rung
(keypoint-heatmap refinement) targets the first, more and nearer data the second. No scene cache:
`occluded` is not available on this capture.

### 7. Replay capture (P1, measured live 2026-10-13)

- **No telemetry during replay** (0 datagrams in 46 s on 127.0.0.1:9001): strategy B - frames +
  the recording's XML pose, aligned by `captures verify` - is the capture method. The Play click
  is the frame clock's origin (verify found -0.02 s / 0.00 s).
- **The replay renders the recording's own drone camera** (uptilt from its `Blueprint/camera/angle`;
  FOV is the global setting, 120 deg). `meta.json` takes the recording's uptilt.
- **The P1 gate passed** on `04 - Infinity Loop - 01_16_606 - 2026-09-09` (preset camera):
  margin 1.52; clock +-0.5 s 1.09 / 1.01; uptilt +-3 deg 1.35 / 1.40 (above 1.25, rejected by the
  camera peak rule: the peak is within one step of the recorded uptilt). Strategy B is trusted with
  both rules.
- **Replay browser driving.** The browser's dropdown and list entries are text the driver cannot
  read (no OCR), so `captures replay --browser-pos ENV,RACE,ROW` takes the operator's indices; a
  wrong pick replays another flight and fails `verify`. Every step is checked on screen (browser
  panel, the paused playback bar, the bar changing after Play).
- **Other drones' recordings: train maps only (decision, team lead, 2026-10-13).** The 93 Borrum
  180 X recordings (uptilt 30) verify with their own camera (margin 1.51, peak uptilt 29). They are
  valid training data: the detector works from the image, the intrinsics are the same (global
  FOV), and labels and PnP use the per-capture camera, so they add map and view diversity. The
  deployed drone is the uptilt-20 small drone, so model selection and held-out evaluation stay on
  its camera: the inventory marks such recordings `train-only` on train maps and `no` elsewhere,
  `captures replay` refuses them on val / held-out maps, `gate_dataset.assemble` refuses a
  non-preset-uptilt capture on a val map, and `eval --split held-out` skips (or, named explicitly,
  refuses) one. They are captured after all preset-camera recordings, at most 4 per train map,
  favouring maps with few preset captures.
- **Gate classes.** A capture whose track has no class of the gate-type table (most non-arena
  maps today) cannot be verified until P3 grows the table; it is captured now and verified then.

## User decisions (2026-10-13)

The user answered the plan's open decisions:

- **D-U1, the AC7 pass rule: accepted as proposed.** The held-out targets (>= 70 % detection,
  median position error <= 20 % of range, 4-25 m) are judged per gate type pooled over held-out
  maps, only for types seen in training with n >= 30 in-range gate-frames. Types only on held-out
  maps are reported as `unseen type`, types with n < 30 as `insufficient n`, neither judged
  (`evaluate.D_U1_RULE`). This narrows AC7's wording.
- **D-U2, targets missed after R3:** publish v2 anyway if it beats v1 on the held-out maps (pooled
  detection and median relative error) and stays within 5 pp of v1 on the Infinity Loop window;
  record the miss here. Switching the preset default (`gates.learned.release`) to v2 still needs
  the user's explicit OK.
- **D-U3, AI-attempt recordings:** moot - none exist among the stored recordings.
- **D-U4, hand-flown maps:** done - the user flew AutumnFields race 04, which cleared G1 (§ Results).

## Results

### P1 replay captures (2026-10-13, 17:08-19:15, unattended)

34 replay captures plus the reference capture, all preset camera except the one Borrum test
capture (`liftoff-arena-2026-10-13-replay-02`, uptilt 30, train map):

| Split | Map | Captures | Verified now |
|---|---|---|---|
| held-out | ThePit | 5 | 3 (tracks 02, 05) |
| held-out | MinusTwo | 2 | 0 |
| held-out | HangarC03 | 2 | 0 |
| val | AutumnFields | 3 | 0 |
| val | Rustline | 3 | 0 |
| val | Surtur | 3 | 0 |
| train | LiftoffArena | 5 (+ reference) | 5 (+ reference) |
| train | MarinaBay, ShortCircuit, TheGreen | 1, 1, 2 | 1, 1, 1 |
| train | StrawBale, PineValley, Permafrost, Hannover, BandoCity | 2, 2, 1, 1, 1 | 0 |

Verified captures had margins 1.37-2.37 with the camera peak within one step (uptilt 19.03 vs
20.03, hfov 120). The 22 unverified ones are on tracks without an inflatable arch: `verify` needs
a gate class of the table on the track, so they stay `verified: false` (excluded by `assemble` and
G1) until P3 adds their gate classes and re-runs `captures verify`.

**Operational lessons.**
- A window that comes to the front during a replay (the Claude terminal did, once) puts foreign
  frames into the capture and sends the end-of-replay keys elsewhere; that capture and the next
  were discarded and redone. `captures replay` now watches the focus during the recording
  (refocusing, and refusing the capture when focus was lost) and refocuses before leaving.
- Dropdowns scroll no further than their end: an entry among the last 5 of a list longer than 5 is
  picked from the end (`--browser-pos ENV/N,RACE/N,ROW` gives the entry counts).
- Frame rate: 7.3-10 fps at the requested 10 (heavy maps render slower).
- The Borrum pass (other-camera recordings on train maps) was not run in this window: a
  follow-up.

### P3 gate types, labels, verification and G1 (2026-10-13, CPU)

**Gate types.** Ten classes were added for the captured tracks (`configs/gate_types/liftoff.yaml`),
one per geometry with a regex over the branding variants, dimensions from the props.json colliders
(the trigger via `opening()`): `airgate_big`, `airgate_small`, `generic_gate`,
`generic_double_gate`, `multigp_frame_7x6`, `multigp_frame_8x10`, `light_gate_300x220`,
`light_hoop`, `inflatable_gate`, `truss_gate_finish`. A new optional key `normal: x` handles props
built facing their `+x` axis (`TrussGateFinishGeneric01`). Calls made from the colliders alone and
marked `HEURISTIC` in the file: the open bottom of both MultiGP frames (no bottom-bar collider) and
the lower opening as the double gate's aperture. The conversion still sees only
`inflatable_arch`: the default fingerprint stays `639beb3ef1bd8bec`.

**Coverage.** Over the 21 catalogue entries with a capture there is no unknown itemID left
(`tracks check --require-gate-coverage` exits 0). Many route passages have no visible gate at all -
the race marks them with flags, tyres or light rigs around an invisible checkpoint volume
(autumn-fields-06, permafrost-02, rustline-01, straw-bale-05: 100 %; surtur-02 98 %; short-circuit-02
90 %; others 12-63 %). That is a property of the track, not a table gap, so
`--require-gate-coverage` now fails on unknown itemIDs only and the volume-only share is reported;
`--require-visible-passages` keeps the stricter rule (**deviation from AC5's wording**, which
failed on either).

**Verification.** The geometry of the new classes is confirmed by `captures verify`: 13 of the 22
pending captures verified with margins 2.01-3.60 and the camera peak within one step
(bando-city 3.60, hangar-c03 2.89 / 2.95, minus-two 2.45 / 2.65, the-pit-06 2.55 / 2.19, pine-valley
2.27 / 2.14, hannover 2.37, surtur-01 2.31, the-green-01 2.01). Unverifiable and excluded: the 9
captures on tracks with no visible gate (AutumnFields 06 x3, Rustline 01 x3, StrawBale 05 x2,
Permafrost 02) and surtur-02/03 (one finish gate: 20 / 15 frames with a gate in view < 50).

**Occlusion.** Scene caches were built for 7 captured maps (`tracks extract-scenes`, read-only on
the game files, into the worktree's `artifacts/harness/scenes/`; ThePit, MarinaBay, TheGreen and
Surtur found no matching bundle). `harness/offline/occlusion.py` tests the drone-to-gate-centre
sight line (every 0.25 m, the last 1 m at the gate and the first 0.3 m at the drone skipped)
against the caches' boxes, capsules and spheres. Mesh and terrain colliders are not in the caches,
so this is a **lower bound**. Occluded share of in-range gate-frames, verified captures:

| Map | Type | 4-8 m | 8-15 m | 15-25 m | 4-25 m |
|---|---|---|---|---|---|
| LiftoffArena | inflatable_arch | 0.0 % of 513 | 0.6 % of 1051 | 1.0 % of 1631 | 0.7 % of 3195 |
| Hannover | airgate_big | 0.0 % of 118 | 0.0 % of 244 | 3.2 % of 442 | 1.7 % of 804 |
| Hannover | inflatable_gate | 0.0 % of 20 | 6.2 % of 48 | 21.1 % of 71 | 12.9 % of 139 |
| BandoCity | light_gate_300x220 / light_hoop | 0 % | 0 % | 0 % | 0 % of 333 / 262 |
| HangarC03 | truss_gate_finish | 0 % | 0 % | 0 % | 0 % of 501 |
| MinusTwo | airgate_small | 0 % | 0 % | 0 % | 0 % of 6828 |
| PineValley | airgate_big | 0 % | 0 % | 0 % | 0 % of 1297 |
| ShortCircuit | inflatable_arch | 0 % | 0 % | 0 % | 0 % of 127 |

Overall 54 of 14,286 (0.4 %), concentrated in Hannover's hedged route beyond 15 m. Not material:
per the plan, labels stay unflagged and occlusion handling is not built in UC71; `diagnose` can
use the caches where they exist.

**G1 (`gates dataset-report`): NO-GO on one criterion, val maps (1 of 2).** In-range (4-25 m)
gate-frames, with the 4-8 / 8-15 / 15-25 m bins:

| Split | Map | Type | 4-25 | 4-8 | 8-15 | 15-25 |
|---|---|---|---|---|---|---|
| train | LiftoffArena | inflatable_arch | 3047 | 498 | 1013 | 1536 |
| train | PineValley | airgate_big | 1297 | 286 | 412 | 599 |
| train | Hannover | airgate_big | 804 | 118 | 244 | 442 |
| train | Hannover | inflatable_gate | 139 | 20 | 48 | 71 |
| train | TheGreen | inflatable_gate | 419 | 69 | 136 | 214 |
| train | TheGreen | inflatable_arch | 378 | 67 | 120 | 191 |
| train | BandoCity | light_gate_300x220 | 333 | 63 | 109 | 161 |
| train | BandoCity | light_hoop | 262 | 14 | 80 | 168 |
| train | MarinaBay | inflatable_arch | 294 | 55 | 97 | 142 |
| train | MarinaBay | multigp_frame_8x10 | 289 | 75 | 82 | 132 |
| train | MarinaBay | multigp_frame_7x6 | 56 | 11 | 21 | 24 |
| train | ShortCircuit | inflatable_arch | 127 | 14 | 37 | 76 |
| val | Surtur | airgate_big | 678 | 138 | 238 | 302 |
| held-out | MinusTwo | airgate_small | 6828 | 1181 | 1891 | 3756 |
| held-out | ThePit | inflatable_arch | 1237 | 221 | 510 | 506 |
| held-out | ThePit | generic_gate | 721 | 99 | 206 | 416 |
| held-out | HangarC03 | truss_gate_finish | 501 | 91 | 172 | 238 |

Train 7,445 gate-frames on 7 maps (1,290 at 4-8 m), held-out 3 maps with a seen type
(`inflatable_arch` 1,237), the protected v1 window intact. Val: only Surtur. AutumnFields' only
preset-camera recordings are on race 06, which has no visible gate; Rustline and DubaiLegends have
no visible gates on any track; the Borrum recordings may not be used on val maps. **The second val
map needs a hand-flown capture (P2):** AutumnFields 03 or 04 (13 / 12 inflatable arches, a known
class) or 02 (19 MultiGP 8x10 frames plus 5 MultiGP float squares, a new class). Another unattended
replay session cannot fill it.

Under D-U1, of the held-out types only `inflatable_arch` is seen in training; `airgate_small`,
`generic_gate` and `truss_gate_finish` are reported as `unseen type` (`airgate_small` is
`airgate_big`'s shape at 0.59 scale, but a class carries one size for PnP).

**G1 update (2026-10-13 evening): GO.** The user hand-flew AutumnFields race 04 (`04 - Mudlarking -
02_27_681 - 2026-10-13`, preset drone); its replay capture `autumn-fields-2026-10-13-replay-04`
verified (margin 1.73, camera peak uptilt 19.03 / hfov 120) and adds 707 in-range arch
gate-frames (130 at 4-8 m) on a second val map. Val: AutumnFields 707, Surtur 678. Every G1
criterion now passes.

### P4 training (started 2026-10-13 21:57)

R1 (`harness gates train --all-captures --select val-detection --val-every 2 --epochs 40`,
resnet18-hf as v1, balanced map x class sampling, v1's window protected) runs with
`--frame-stride 2`: consecutive 10 fps frames are near-duplicates, so every second frame halves
the epoch time and the frame cache (6.8 GB) while the counts still clear G1 (val AutumnFields ~353,
Surtur ~339 in-range gate-frames). The GPU was **contended**: the user played League of Legends on
it during the run, so timings and latencies of this run are not clean measurements.
`--all-captures` leaves held-out-map captures out of training (named ones still raise).

**Roll augmentation for R2 (target-matched).** The camera roll of a frame (the horizon tilt in the
image: the screen angle of world-up, `harness/offline/roll.py`) was measured on 2026-10-13
(|deg|):

| Source | n | p50 / p90 / p95 / max | 0-15 / 15-35 / 35-60 / 60-90 / 90-135 / >135 % |
|---|---|---|---|
| UC69 live, 10 finished Infinity Loop races (teacher it1374) | 38,946 ticks | 35 / 57 / 63 / 81 | 15.3 / 35.0 / 42.0 / 7.6 / 0 / 0 |
| live, within 8 m of the next gate | 7,142 | 33 / 57 / 62 / 78 | 18.2 / 34.6 / 40.3 / 6.9 / 0 / 0 |
| UC71 train frames (dataset 9560fa4b-w768) | 8,155 | 14 / 39 / 47 / 166 | 52.7 / 33.1 / 13.2 / 0.9 / 0.1 / 0 |
| UC71 val (AutumnFields + Surtur) | 1,519 | 10 / 31 / 39 / 61 | 62.7 / 29.6 / 7.6 / 0.1 / 0 / 0 |
| manual flight, moving | 12,300 | 11 / 28 / 41 / 180 | 62.3 / 31.5 / 3.1 / 0.8 / 0.8 / 1.5 (crashes) |

The training frames are far more level than live flight (median 14 vs 35 deg; the 60-90 deg band
~8x under-represented) and val is more level still, so plain `val_det` is optimistic about live
flight. `RollAug` therefore brings an augmented frame (p 0.6) to a **target roll** drawn from the
live mix - a random sign times a magnitude from 0-15 15 %, 15-35 35 %, 35-60 40 %, 60-90 7 %,
90-135 3 % (the last a deliberate crash-and-recovery tail, the user's call), clamped to +-135 deg -
by rotating it `target - frame_roll` about the principal point (each frame's roll comes from its
paired attitude; a horizontally flipped frame's roll changes sign). A frame without a known roll
gets a raw angle from 0-15 10 %, 15-35 25 %, 35-60 35 %, 60-90 25 %, 90-135 5 %. The buckets are
recorded in `params.roll_aug`. Rotation keeps keypoint identity (no `flip_pairs`). The learned
path has no "top above bottom" assumption (decode: per-slot offsets; PnP: IPPE by index; the
normal oriented by travel), so gates rolled past 90 deg are not rejected; the OpenCV quad
estimator's TL/TR/BR/BL corner order is the only such assumption and it is not on this path.

**`val_det_live_roll`** (`--live-roll-val`, report only; `val_det` stays the selection metric for
comparability with R1): every val frame is re-rolled once to a live-mix target (fixed seed 71,
the same every epoch) and scored with the attitude whose camera sees the rolled image
(`roll.rolled_quat`) and each gate's in-image test re-judged at its rolled centre, so the eval
rule is unchanged. **Limitation:** a rolled frame is not a real banked frame - the corners are
padded (mean colour) and the scene, lighting and motion blur are those of the level flight - so
it measures roll robustness, not live accuracy. `--rotated-val 0,30,...` reports image-space
detection at fixed angles alongside.
