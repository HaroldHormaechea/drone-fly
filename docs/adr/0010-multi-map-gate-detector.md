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
and latency (CPU-measured latency is flagged). Targets 70 % / 20 % under the D-U1 rule (pending the
user's acceptance); `--compare` scores both detectors on v1's protected Infinity Loop window and
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

## Results

Pending (P0b checkpoint 9, P1-P6).
