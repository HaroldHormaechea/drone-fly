---
plan_for: use-cases/71-multi-map-gate-detector.md
work_branch: feat/uc-71-multi-map-gate-detector
run: drone-fly-uc-71
approved: 2026-10-13
---

# APPROVED PLAN — UC71 proposal v2 (analyst-71), approved by challenger-71 after round 2

## Challenger verdict

**Approve.** Round 1 asked for 5 Major and 6 Minor fixes; v2 applies all of them. The orchestrator's three requirements are met:
1. **CPU work comes first.** P0a and P0b are all CPU. P0b is split into 9 ordered checkpoints, each committed with green tests; the first is a fingerprint pin written against the pre-change code.
2. **GPU and live steps are isolated and marked.**
   - P1 is live and unattended; it needs a hands-off window with Vanguard closed.
   - P2 is live with the user present.
   - P4 and P5 are GPU; nothing runs on the GPU before about 14:30.
   - P6 publishes only with user confirmation.
   - Nothing live starts before the user accepts the plan.
3. **Every acceptance criterion is tied to a measurable check.** There is an AC verification table, and the G1 data-sufficiency go/no-go comes before any GPU training.

Both UC70 findings are handled:
- **Detector weakness, in scope.** Every report gives range bins (4–8, 8–15, 15–25 m) plus the ADR 0003 signed range bias and keypoint-spread metrics. Each training-ladder step has a numeric trigger. Selection is by val-map detection rate, not val loss.
- **Live tracker gap, deferred to UC72 with a rationale.** The live tracker recognises 2 of 8 passes, with a next-gate mis-association of 0.57. UC71 hands over the v2 noise model and a refit `lost_timeout_s`.

Notes for the developer and QA (Minor, non-blocking):
- **Recheck the alignment margin.** The 1.5 margin was calibrated on a manual capture. Recheck it on the first P1 replay capture before trusting strategy B.
- **R2 latency.** Check it against the same 25 ms p50 budget as R3.
- **Protected window in G1.** G1 must assert the protected Infinity Loop window is fully present: 148 in-range arch gate-frames. A partial window must not pass. The check is that the window's frame set equals v1's recorded held-out window, not a lower bound; QA tests it.
- **Replay HUD masks.** These may shift the align baseline, so record the P1 recheck in ADR 0010.

Docs ownership (`docs/`, `tracks/README.md`) is outside both path globs. The orchestrator assigns it to the developer.

## Decisions for the user on return (answer when accepting the plan)

**D-U1. AC7 pass rule.**
- **Proposed:** the held-out targets (≥ 70 % detection, median position error ≤ 20 % of range, 4–25 m) are judged per gate type, pooled over held-out maps.
- A type is judged only if it was seen in training AND has n ≥ 30 in-range gate-frames.
- Types that appear only on held-out maps (e.g. TrussGateFinishGeneric01, AirgateSmall150H210B01) are reported as an `unseen type` stratum, not judged. Types with n < 30 are reported as `insufficient n`.
- This narrows AC7's wording, so the user must accept it.

**D-U2. Targets missed after R3.**
- **Proposed default:** publish v2 anyway if it beats v1 on held-out maps (pooled detection and median relative error) and passes the Infinity Loop check (≥ v1 − 5 pp). Record the miss in ADR 0010 and keep the preset on v1 unless the user says switch.
- **Alternative:** hold the release.

**D-U3. AI-attempt recordings.** Proposed: at most 3 per map (they are mostly short crashes on Infinity Loop), and prefer the user's own recordings.

**D-U4. P2 hand-flown maps.** The final list comes from the inventory. Expected minimum:
- held-out: ThePit and MinusTwo, which have the most visible gate types; ideally all four held-out maps;
- val: AutumnFields and Surtur, the only val maps with visible route gates (DubaiLegends and Rustline have none);
- train: whatever the replays leave uncovered.

## Analysis

**What exists today (UC64 pipeline):**
- `harness label` projects every placed item that matches `gate_types`, using the telemetry pose and the preset camera. Occlusion is not modelled.
- `harness train-gates` splits each capture by time slice.
- `harness eval --capture --labels` scores one capture at 4–25 m against the 80 % / 15 % targets.
- `gate_learned.py` auto-downloads `liftoff-gates-v1` using a flat file-name → sha256 table and one `RELEASE_URL`.
- The hand-flown recorder is the owner's external `capture.py`, which is not in the repo. Its layout is `telemetry_raw.csv` + `frames.csv` + `frames/`.
- `ui_driver.py` has verified menu primitives, measured at 2560×1080, but no replay steps.

**v1 results:**
- ADR 0009: 19.6 % detection at 4–25 m, 0 % below 8 m, 1.03 false positives per frame.
- ADR 0003 L244-246 gives the root causes:
  - the predicted keypoint spread is 0.72× the true spread, so PnP places gates about 1.4× too far, which causes range misses;
  - the val loss is dominated by the focal term.

**Catalogue:**
- Splits:
  - train: 12 maps, LiftoffArena included, so Infinity Loop (`liftoff-arena-04`) is a train map;
  - val: AutumnFields, DubaiLegends, Rustline, Surtur;
  - held-out: HangarC03, MinusTwo, ThePit, TheRussianWoodpecker.
- Most route items are invisible `Checkpoint*` volumes. There are about 40 visible gate itemIDs, and some types exist only on held-out maps.

**Coupling hazards:**
- **H1.** `ConversionConfig.from_dict` defaults `gate_types` through `harness_tables()`, which reads `data.get("gate_types")` from the raw preset YAML (`tracks/liftoff.py:156-160`). `fingerprint()` hashes those raw dicts, and `training/evaluate.py` refuses a fingerprint mismatch. Growing the table, adding a key to the arch entry, or moving the table to a file would change the sim conversion. That breaks the UC68 teacher eval and the UC70 courses.
- **H2.** `gate_learned.load_weights` requires the configured classes and hash to equal the weights' own. A grown table makes v1 unloadable.

## Proposed solution

### Folder structure (understandable, and scales to more drones and maps)

**Code** (paths under `src/drone_fly/harness/` unless noted).

New files:
- `liftoff/recorder.py`: a live recorder (raw UDP + wall-stamped JPEG via `ScreenFrameSource`), used for hand-flown and replay capture.
- `liftoff/ui_replay.py`: replay-menu steps on the `ui_driver` primitives. Every step is verified; a failure aborts with a screenshot.
- `offline/recordings.py`: recording XML parsing, completeness, inventory.
- `offline/capture_index.py`: the `meta.json` schema and validation, discovery, capture hash, catalogue linking, per-capture camera and mask overrides.
- `offline/align.py`: the projection-alignment score.
- `offline/gate_dataset.py`: assembly, split enforcement, manifest, memmap frame cache, map×class-balanced sampling, dataset report.
- `offline/gate_diagnose.py`: miss attribution.
- `src/drone_fly/tracks/gate_coverage.py`.
- `configs/gate_types/liftoff.yaml`: the gate table in its own packaged file.

Extended: `gate_types.py`, `config.py`, `configs/liftoff.yaml`, `offline/{capture,autolabel,gate_train,evaluate}.py`, `liftoff/gate_learned.py`, `liftoff/gate_net.py` (R2 only), `tracks/{liftoff,cli}.py`, `cli/main.py`.

**Artifacts** (gitignored, under `artifacts/`):

| Path | Contents |
|---|---|
| `artifacts/harness/captures/<Map>/<session>/` | `meta.json`, frames, telemetry or pose |
| `artifacts/harness/liftoff/labels/<Map>/<session>/` | labels |
| `artifacts/harness/liftoff/datasets/<manifest-sha8>/` | memmap frame cache |
| `artifacts/harness/liftoff/models/<detector-id>/` | `liftoff-gates-v2.pt` (full/eval), `liftoff-gates-v2-deploy.pt`, `*.training_log.json`, `data_manifest.json`, `eval/`; the UC70 noise-model folder sits beside the weights |

Rules:
- Session names are globally unique: `<map-kebab>-<YYYY-MM-DD>-<replay|manual>-<NN>`. That name is the capture name that the weights meta and `detector_noise.heldout_window` key on.
- The legacy reference capture `artifacts/harness/captures/2026-10-11-infinity-manual` is NOT moved, because v1 meta and the r5 noise model reference it by name. `captures register` adds a `meta.json` in place, and discovery accepts both folder depths.
- `meta.json` records the drone profile (`drones/<id>`) and the camera block, which is the hook for more drones.

### AC1. Replay telemetry check (`docs/harness/liftoff-captures.md`, new)

**AC1a (CPU, P0a).** Read-only listing and parsing of 3–5 files under `%USERPROFILE%\AppData\LocalLow\LuGus Studios\Liftoff\Recordings\…`; nothing is copied into the repo or artifacts. It documents:
- the schema;
- whether there is a per-sample pose + timestamp trajectory, and its rate;
- level, track and race references (name or uuid);
- drone and camera settings;
- checkpoint and lap events.

**AC1b (live, unattended, P1).**
- Run the recorder in telemetry-only mode during one replay playback, and measure the datagram count and rate.
- When AC1a found a pose, fit the time offset and rate between the telemetry and the XML pose.
- Measure the frame mapping with `align`.
- Document the camera mode and how to force FPV, the replay HUD masks, and whether FOV/uptilt follow the recording's drone or the current settings.

**Decision table** (drives AC3):
- **A:** telemetry streams during replay → record frames + UDP, the hand-flown layout.
- **B:** XML pose only → record frames + XML pose, aligned with `align`.
- **C:** neither → the replay command refuses and prints the manual-fallback instructions.

### AC2. `drone-fly harness captures inventory [--dir …] [--json out]`

For each recording it lists:
- name and mode folder;
- map, track and race as parsed;
- duration;
- source: `ai-attempt` when the name matches the liftoff-episodes `-model-…-attempt-N` pattern, otherwise `user`;
- the catalogue match: track/race uuid → `entries[].uuids`, otherwise map + display name, otherwise `unmatched` (listed but not capturable);
- completeness;
- camera parameters if present;
- `capturable`: yes, or no with a reason.

**Completeness sources, in priority order:**
1. Checkpoint or lap events in the XML, judged against the race XML route and laps (primary when present).
2. The XML pose trajectory, replayed through the existing pass rule (`course_senses.GateProgress` / `detector_noise.truth_progress` on `tracks.loader.load_course`).
3. The `meta.json` completeness of a capture already made from this recording (sha-matched). That capture's telemetry is judged with source 2 at capture time.

`unknown` is allowed only when none of the three exists, and this is documented.

**Incomplete races** are used up to the end of valid flight. Labels cover only gates projected into view.

**Static-pose trim.** Any stretch that moves less than 5 cm and 1° for more than 1 s keeps one frame. It is a new counted drop reason, `static`, in `capture.load_capture`. It is off by default, so existing results are unchanged, and `gate_dataset` turns it on.

### AC3. `drone-fly harness captures replay --recording <name> [--max-s N] [--fps 10]`

Steps (`ui_replay`):
1. Make sure Liftoff is running.
2. Open the replay list.
3. Select the recording.
4. Play it.
5. Force the FPV camera.
6. Verify each step with pixel probes and/or telemetry.
7. Record:
   - **Strategy A:** the recorder.
   - **Strategy B:** frames + XML pose resampled to the frame clock, written in the run-folder layout so `load_capture` reads it unchanged. That means `telemetry.csv` with `received_at`, velocity by finite difference and gyro 0, plus `frames.csv` with FRAME_COLUMNS.
8. Stop and return to the menu.

Any failure aborts with exit ≠ 0 and a screenshot path under `artifacts/harness/liftoff/reports/ui-failures/`.

`meta.json` contains:
- `schema_version`, `catalogue_id`, map;
- split, for information only (it is always re-read from the catalogue at use);
- track and race file names + sha256;
- `source`, recording name + sha256;
- completeness;
- `pose_source`: `udp` or `recording-xml`;
- time offset and rate;
- camera: {resolution, hfov, uptilt, source};
- extra masks;
- drone profile;
- the `verified` flag + margin;
- git head and creation time.

Under strategy C the command prints the manual fallback and exits as a config error.

### AC4. Manual fallback

- `drone-fly harness captures record --track <catalogue-id> [--max-s N] [--fps 10]` uses the in-repo recorder. It writes the manual layout plus `meta.json` (`source: manual`).
- `captures register <dir> --track <id>` back-fills `meta.json` for existing captures.
- `docs/harness/liftoff-captures.md` gives the step-by-step:
  1. load the catalogue track in Liftoff;
  2. run `captures record`;
  3. fly 3–4 laps, varying height, speed and line;
  4. stop;
  5. run `captures verify`.

### Projection alignment (`offline/align.py`, CLI `captures verify <dir> [--search-lag] [--search-uptilt]`)

**Score.** On sampled frames, the score is the mean Sobel gradient magnitude along each labelled gate's projected outline, divided by the same mean at a random pose offset (the baseline). Grid search covers a time offset of ±1.5 s in 20 ms steps and, optionally, uptilt ±5° in 1° steps.

**`verified: true` requires all of:**
- margin (best / baseline) ≥ 1.5;
- the best offset is not on the edge of the search range;
- at least 50 frames with a labelled gate in view.

**Threshold calibration (provisional).** The threshold is calibrated in P0b on the reference capture. The correct pose must score ≥ 1.5, and a 3° uptilt error or a 0.5 s lag error must score clearly lower. ADR 0010 records the value and the evidence.

**What it is used for:**
- time alignment, mandatory under strategy B;
- lag estimation, replacing the hand-set `--lag-ms`;
- verifying the FPV camera and camera parameters, including for older drone/camera setups.

Unverified captures are excluded by `gate_dataset`. It runs on CPU with OpenCV.

### AC5. Gate types

**Schema (`gate_types.py`):**
- `bottom: open|closed` is required in the liftoff table.
- `family` is optional, for reporting.
- `signature()` and `hash()` are unchanged, so the new keys don't alter the hash.

**Table location:**
- The table moves to its own file, `configs/gate_types/liftoff.yaml`, referenced from the `liftoff` preset.
- `load_config` resolves the reference into the usual `gate_types` list.
- `liftoff_5inch.yaml` `square_gate` is untouched.

**Coverage rule.** For every item in the track XMLs of catalogue entries that have a capture, the item is exactly one of:
1. a visible gate class in the table;
2. a checkpoint volume (`Checkpoint*` / `TriggerBox*`), which is never labelled;
3. explicitly not a gate, via a new `not_gates` regex list;
4. otherwise **unknown**. This covers any route item, and any item whose name looks like a gate: `(?i)gate|arch|hoop|frame|truss|airgate`.

**Classes:**
- One class per geometry, with a regex covering the branding variants.
- Dimensions are drafted from the props.json colliders (`autolabel.collider_bounds`, `check_gate_types_against_props`, the conversion's `opening()`).
- Default keypoints:
  - closed frames: 4 outer + 4 aperture corners;
  - open arches: feet + mid-legs + top;
  - flip pairs always.
- Every class is checked with label previews.
- `harness gates draft-types --props … --items …` prints candidate entries only.

**Coverage enforcement.**
- `tracks check` lists unknown itemIDs and route passages with no visible gate, per entry.
- `--require-gate-coverage` makes either case a non-zero exit.
- `harness label` and `gates train` refuse a capture whose track has unknown items.

**H1 fix.**
- `harness_tables()` resolves the preset's gate-types file reference.
- It keeps only `CONVERSION_GATE_CLASSES = {inflatable_arch}`, projected to today's key set, so the new `bottom`/`family` keys are stripped.
- The default `ConversionConfig.fingerprint()` and every converted course stay byte-identical.
- The pin test is the first P0b commit, written and passing against the pre-change code.
- New gate types do not reach the sim in UC71. Adding them later is a deliberate `conversion_version` bump.

**H2 fix.**
- `load_weights` builds the estimator's table from the weights' `meta.gate_types` when present (v1 has it).
- It verifies that table against `meta.gate_types_hash`, and only warns when physical dimensions differ from the configured table.
- Weights without embedded `gate_types` keep the strict check.

### AC6. `drone-fly harness gates train`

A new `gates` group: `train`, `dataset-report`, `draft-types`, `diagnose`. `harness train-gates` stays as an alias.

Options:
- `--captures <dir|glob>…` or `--all-captures`;
- `--split-source tracks/liftoff.yaml`;
- `--detector-id`, default `liftoff-gates-v2-<YYYYMMDD-HHMM>`;
- the existing hyper-parameters;
- `--input-width`, `--frame-stride`, `--max-frames-per-capture`;
- `--select val-detection|val-loss|last`, default `val-detection`;
- `--val-every 2`;
- `--protect-holdout-of <weights>`, default the v1 full weights.

**`gate_dataset.assemble`:**
- Resolves each capture's `catalogue_id` to its split from the catalogue at run time.
- Split routing:
  - train-map captures → train;
  - val-map captures → selection;
  - a held-out-map capture → raises `TrackLeakageError`, never silently filtered.
- Refused with a reason: no catalogue id, unverified alignment, or uncovered items.
- v1's held-out windows are never loaded.
- Applies per-capture camera and masks, the static trim, the frame stride and map×class-balanced sampling.

**Frame cache.** Masked and resized frames, as at runtime, go to a memmap `datasets/<manifest-sha8>/`. In RAM it would be about 0.75 MB per frame at 768 px, about 15 GB for 20k frames.

**Selection (`val-detection`).** Every 2 epochs, run decode + PnP + the eval matching rule on the val-map frames. Keep the best detection rate; ties go to the lower median relative error.

**Outputs** in `models/<detector-id>/`:
- weights and log;
- `data_manifest.json`:
  - per capture: name, catalogue id, map, split role, capture sha256, counts, protected windows;
  - plus the catalogue sha, gate-types hash, git head and hyper-parameters;
- the capture sha256 hashes `meta.json` + `frames.csv` + the telemetry/pose file + a digest of (frame file name, size);
- the weights meta `captures`/`splits` carry the split role, so `detector_noise.heldout_window` treats val and held-out captures as never trained.

**`--deploy`** retrains on train + val maps for the chosen epoch count (`select last`), producing `liftoff-gates-v2-deploy.pt` (`purpose: deploy`). Held-out maps are never included.

### Data-sufficiency gate G1 (`gates dataset-report`: a dry run of `assemble`; CPU; after P3, before P4)

The report counts in-range gate-frames by split × map × type, with the 4–8 m bin separate. In range means 4–25 m, centre in image, unmasked, and not flagged as occluded.

**GO requires all of:**
- **Train:** ≥ 3,000 in-range gate-frames from ≥ 5 maps, including ≥ 300 frames in the 4–8 m bin.
- **Val:** ≥ 2 val maps, each with ≥ 300 in-range gate-frames of visible gates.
- **Held-out:** ≥ 2 held-out maps, and ≥ 1 seen type with ≥ 30 in-range gate-frames.
- **Infinity Loop:** the protected v1 window is fully present, 148 in-range arch gate-frames as in UC70. A partial window does not pass.

**NO-GO** → more P2 capture on the maps the report names. P4 does not start.

### AC7. `drone-fly harness eval --detector <weights|dir> --split held-out [--captures …] [--compare <weights>]`

**Scope.** Every frame of held-out-map captures. It refuses any capture that the weights' manifest or meta lists as trained.

**Report.** Per map × type and per type pooled over held-out maps, with range bins 4–8, 8–15 and 15–25 m plus pooled 4–25 m:
- n and detection rate;
- median position error (m and % of range);
- **median signed range error (%)** and **median predicted/true keypoint-spread ratio**, on matched-by-centre detections (the ADR 0003 root-cause metrics);
- median yaw error (folded);
- false positives per frame;
- latency, CUDA-synchronised; latency measured on CPU is flagged.

**Targets.** 70 % / 20 % under the D-U1 rule. The single-capture UC64 eval keeps its 80 % / 15 % constants.

**Infinity Loop regression (`--compare <v1>`).**
- Both models are scored on the protected v1 window, with the same bins and metrics.
- Pass: v2 arch detection at 4–25 m ≥ v1 − 5 pp.
- The 4–8 m bin (v1 = 0 %) is a tracked outcome, reported as v2 vs v1, not a gate.

**`gates diagnose --weights … --captures …`** attributes each miss, per range bin, to one of:
- no centre peak;
- fewer than `min_keypoints` visible;
- PnP range error over tolerance, with the signed bias and spread ratio;
- class confusion;
- occluded-by-scene, only when the scene cache is ok, and for reporting only.

It runs on v1 in P0b (CPU, the reference held-out frames), and its table goes into ADR 0010.

**Training ladder.** All decisions use val maps only. Held-out is scored once per finalist.

| Rung | What | Gate to accept | Rough GPU time (RTX 3050) |
|---|---|---|---|
| R1 | v1 architecture, multi-map data, `val-detection`, balanced sampling | — | ≈ 2–3 h for ≤ 40 epochs, val decode every 2 epochs |
| R2 | Keypoint-heatmap refinement head (CenterNet-pose: per-slot heatmaps at stride 8; regressed keypoints snapped to the peaks); weights `format: 2`; `build_from_meta` still loads format 1 | R2 latency is checked against the same 25 ms p50 budget as R3 | ≈ 3–4 h |
| R3 | `input_width` 1024 and/or `resnet34-hf` | Accepted only if CUDA latency stays ≤ 25 ms p50 (40 Hz budget with async inference) | ≈ 4–5 h |
| Deploy | Retrain on train + val | — | ≈ 1.5–2.5 h |
| P5 | Eval + noise fit + latency | — | ≈ 0.5 h |

**Triggers:**
- **R2 runs** if, after R1, val detection at 4–25 m < 70 %, OR median signed range error > +15 %, OR spread ratio < 0.85.
- **R3 runs** if, after R2, val detection is still < 70 % AND diagnose attributes ≥ 30 % of misses to "no centre peak".
- **After R3:** D-U2 applies.

Never tune on held-out. All estimates are rough and scale with G1's frame counts.

### AC8. Release

**`gate_learned.py`:**
- `RELEASES = {tag: {file: sha256}}`. v1 keeps its two files; v2 adds `liftoff-gates-v2.pt` and `liftoff-gates-v2-deploy.pt`.
- File names are unique across releases, so `resolve_weights_path` still keys on the file name and builds the URL from the owning tag.
- sha256 is verified as today.

**Preset:**
- A new key, `gates.learned.release`, **defaults to v1 until P6**. With `weights` unset it resolves to the standard path for that release; an explicit `weights:` always wins.
- P6 switches it to v2 with a one-line change, plus the `test_harness_presets.py` and `test_harness_uc64_cli_docs.py` updates. That happens only if AC7 passes or the user decides so (D-U2).

**Release assets:**
- deploy weights, full weights;
- training log, `data_manifest.json`;
- the v2 noise model + report. These come from the unchanged UC70 tool, `train detector-noise --detector <v2 full> --captures <held-out-map captures>`, plus `--latency-only --device cuda` on the GPU.

**Publishing.** `gh release create liftoff-gates-v2` is outward-facing and needs explicit user confirmation. The shas are committed after the upload, and nothing from `artifacts/` is tracked.

### AC9. Tests (hermetic: synthetic only, no game data)

**New:**
- `tests/test_harness_recordings.py`: a synthetic recording XML following the AC1a schema, with invented values. It tests:
  - parsing;
  - catalogue linking (uuid, name, unmatched);
  - completeness from events, from the pose trajectory, and from a captured-meta merge;
  - `unknown` only when no source exists;
  - AI-attempt naming.
- `tests/test_harness_capture_index.py`: `meta.json` validation, both discovery depths, hash stability, the static trim (off by default).
- `tests/test_harness_align.py`: synthetic outlines; recovering a known lag and uptilt; rejecting a wrong camera; the threshold logic.
- `tests/test_harness_gate_dataset.py`:
  - train/val from a synthetic catalogue;
  - a held-out capture raises `TrackLeakageError`;
  - refusal of no-id, unverified and uncovered captures;
  - protected windows;
  - manifest contents;
  - balanced weights;
  - memmap round-trip;
  - `dataset-report` counts and the G1 verdict logic, including a partial protected window failing;
  - a tiny-cnn multi-capture train smoke on CPU with `val-detection`.
- `tests/test_harness_gate_coverage.py`:
  - table validation (bottom required, unique classes);
  - unknown-item and volume-without-visible-gate reports;
  - the `--require-gate-coverage` exit code;
  - `label` refusing unknown items.

**Extended:**
- `test_tracks_liftoff_convert.py`: the fingerprint pin (first commit); file-reference resolution; identical conversion with the grown table.
- `test_harness_gate_learned.py`: v1-style weights load with a grown table; v1/v2 name resolution; checksum failure; release default v1.
- `test_harness_capture_eval.py`: multi-capture held-out eval with bins, signed bias and spread ratio; refusal of trained captures; `--compare`; the unseen and insufficient strata.
- `test_harness_cli.py`: the new parsers and the `train-gates` alias.
- `tests/liftoff_capture_support.py`: synthetic recording XML and multi-map catalogue builders.

## AC verification table

| AC | Check | Pass | Phase | Who |
|---|---|---|---|---|
| 1 | `docs/harness/liftoff-captures.md` sections AC1a + AC1b | Doc exists. The decision resolves to A, B or C. It records the XML schema and pose rate, the measured datagram rate during replay (0 counts as a result), the offset/rate fit when A + pose, and the FPV/HUD/camera findings | P0a + P1 | CPU + live unattended |
| 2 | `captures inventory` on the user's folder + `test_harness_recordings.py` | Every recording listed with map/track, duration, catalogue match and completeness; `unknown` only where no source exists; tests green | P0b (tests), P0a/P3 (real run) | CPU |
| 3 | One end-to-end `captures replay` on a real catalogue-matched recording | Exit 0; capture under `captures/<Map>/<session>/` with a `meta.json` that has `catalogue_id`; `verified: true` with margin ≥ 1.5; `load_capture` pairs ≥ 80 % of frames. Under C, the command prints the fallback (test) | P1 | live unattended |
| 4 | `captures register` on the reference capture + at least 1 `captures record` capture | Both load through `load_capture`, label with 0 unknown items, and are `verified: true`; the doc steps are present | P0b (register), P2 (record) | CPU; live + user |
| 5 | `tracks check --require-gate-coverage` + `test_harness_gate_coverage.py` | Exit 0 over every entry with a capture; `label` refuses unknown items (test); fingerprint pin green | P3 | CPU |
| 6 | Real `gates train` run + `test_harness_gate_dataset.py` | Manifest lists every capture with sha256 + split role; no held-out-map capture; protected windows recorded; outputs in `models/<detector-id>/`; tests green | P4 | GPU |
| 7 | `harness eval --split held-out --compare <v1>` | Under the D-U1 rule: ≥ 70 % and ≤ 20 % of range per judged type; Infinity Loop arch ≥ v1 − 5 pp; bins, bias and spread reported | P5 | GPU (CPU possible) |
| 8 | Release + fresh-download test | All assets uploaded. With the local weights deleted, the estimator re-downloads them with sha256 verified. The preset selects v2 after the D-U2 / AC7 outcome; v1 stays selectable via `release: v1`. `git ls-files` contains no frames, XML or captures | P6 | user confirmation |
| 9 | `uv run --extra dev pytest` (venv python) | All new and extended tests green; no game data in `tests/` | each P0b checkpoint | CPU |
| — | G1 `gates dataset-report` | GO criteria above | after P3 | CPU |

## Phases and checkpoints

**Hard rules for every phase:**
- Nothing live starts before the user has accepted the plan.
- Live phases need an agreed hands-off window and the liftoff-episodes prerequisites: Vanguard/anti-cheat closed, Liftoff borderless at 2560×1080, telemetry enabled.
- No GPU work before about 14:30, or until the current training finishes.

**P0a (CPU).** Read-only listing and parsing of the user's Liftoff Recordings folder in AppData; nothing is copied. It answers AC1a and gives a first inventory sketch.

**P0b (CPU).** Ordered checkpoints, each committed with green tests:
1. H1 fingerprint pin, written against the pre-change code.
2. H2 self-describing weights + release table (default v1).
3. Gate-type schema, table file + `harness_tables()` resolution and projection, coverage + `tracks check`.
4. `capture_index` + `register`, run on the reference capture.
5. `recordings` + `inventory`.
6. `align`, with the threshold calibrated on the reference capture.
7. `gate_dataset` + `gates train` + `dataset-report`.
8. Multi-capture eval + `diagnose`, run on v1 with its table recorded.
9. Recorder + `ui_replay` with placeholder positions.

**Later phases:**

| Phase | Needs | Content |
|---|---|---|
| P1 | Live game, unattended | AC1b. Measure the replay-menu positions from `ui shot`, and the replay HUD masks. The AC3 end-to-end check. `captures replay` over the capturable recordings, within the D-U3 cap. Recheck the 1.5 alignment margin on the first replay capture before trusting strategy B |
| P2 | Live game, user present | `captures record` on the D-U4 maps that the inventory leaves uncovered |
| P3 | CPU | Label all captures. Run `captures verify` and review the previews. Finalise the gate types for exactly the captured tracks. `tracks check --require-gate-coverage` clean. Report the occluded share (scene caches). G1 |
| P4 | GPU | R1 → (R2) → (R3) on val; deploy retrain |
| P5 | GPU | Held-out eval + compare; v2 noise model, including `--latency-only --device cuda` |
| P6 | User confirmation | Publish `liftoff-gates-v2`, commit the shas, switch the preset (one line + tests) |

## Files affected

**Production (developer).**

New:
- `C:\dev\drone-fly-uc-71-multi-map-gate-detector\src\drone_fly\harness\liftoff\recorder.py`
- `…\harness\liftoff\ui_replay.py`
- `…\harness\offline\recordings.py`
- `…\harness\offline\capture_index.py`
- `…\harness\offline\align.py`
- `…\harness\offline\gate_dataset.py`
- `…\harness\offline\gate_diagnose.py`
- `…\harness\configs\gate_types\liftoff.yaml`
- `…\src\drone_fly\tracks\gate_coverage.py`

Modified:
- `…\harness\gate_types.py`
- `…\harness\config.py`
- `…\harness\configs\liftoff.yaml`
- `…\harness\offline\{capture,autolabel,gate_train,evaluate}.py`
- `…\harness\liftoff\gate_learned.py`
- `…\harness\liftoff\gate_net.py` (R2 only)
- `…\src\drone_fly\tracks\{liftoff,cli}.py`
- `…\src\drone_fly\cli\main.py`

**Docs (assigned to the developer by the orchestrator):**
- `…\docs\harness\liftoff-captures.md`
- `…\docs\adr\0010-multi-map-gate-detector.md`
- `…\docs\adr\0003-real-liftoff-calibration.md`
- `…\tracks\README.md`

**Test (qa).**
- New: `tests\test_harness_recordings.py`, `test_harness_capture_index.py`, `test_harness_align.py`, `test_harness_gate_dataset.py`, `test_harness_gate_coverage.py`.
- Extended: `test_tracks_liftoff_convert.py`, `test_harness_gate_learned.py`, `test_harness_capture_eval.py`, `test_harness_cli.py`, `test_harness_presets.py`, `test_harness_uc64_cli_docs.py`, `tests\liftoff_capture_support.py`.

## Risks & considerations

1. **The recording XML format is unknown** (UC pitfall: missing input). P0a runs before the parser and fixture are finalised. If the outcome is C, AC3 degrades to the documented refusal and P2 grows.
2. **Chase cam, overlays and other camera setups** (pitfalls 2 and 3). The alignment margin checks for FPV. Replay HUD masks are set per capture. Recordings whose camera can't be verified are excluded. FOV may be global while uptilt comes from the drone; AC1b decides.
3. **Unattended UI automation** (pitfalls 4 and 5). Every step is verified, and a failure produces a screenshot. Positions are measured in P1 at 2560×1080. Running it takes over the mouse and keyboard focus.
4. **Invisible checkpoints and val maps without visible gates.** These depend on the D-U1 and D-U4 decisions. `tracks check` reports the share of passages that are volume-only. For those passages, the student's UC70 `perception.uncovered` stays the answer.
5. **70 % target vs 19.6 % today.** G1, the ladder with numeric triggers, and `diagnose` give a measurable path. D-U2 covers a miss.
6. **H1 and H2.** Mitigated as described, with the pin test first. New gate types do not reach the sim in UC71.
7. **The live tracker gap is out of scope: it goes to UC72.**
   - The gap is 2 of 8 passes and a next-gate mis-association of 0.57. It is tracker logic, outside UC71.
   - It goes to a new UC only if UC72's scope is frozen. ADR 0009 already names it UC72's top follow-up.
   - UC71 provides the v2 noise model, including CUDA latency and FP/mis-association stats.
   - UC71 also provides a refit value for `gates.lost_timeout_s` (v1-fitted 1.3 s) from the v2 miss-burst p99, recorded in ADR 0010 but not applied to the preset in UC71.
   - ADR 0010 records this hand-off as a deviation.
8. **UC70 coupling.** The UC70 student was trained under v1's noise model. Once the preset is on v2, the live detector differs from that model, which UC70 only warns about. UC72 should fine-tune under the v2 model.
9. **Occlusion.** Optional and off; it is not built in UC71. Diagnose and P3 report the occluded share. Build it only if that share is material, as a plan amendment.
10. **Leakage.** One choke point raises an error. The v1 window is protected. Selection uses val maps only. Held-out is scored once per finalist.
11. **Scale.** The memmap cache bounds RAM. Captures are about 0.3 MB per frame at 10 fps, so about 3 GB per 20 min, all under the gitignored `artifacts/`. Val decode runs every 2 epochs.
12. **Outward release** needs user confirmation. No game data is committed; a hygiene test is extended.
