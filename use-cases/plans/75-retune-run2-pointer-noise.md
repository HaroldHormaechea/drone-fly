---
plan_for: use-cases/75-retune-run2-pointer-noise.md
work_branch: feat/uc-75-retune-run2-pointer-noise
run: drone-fly-uc-75
approved: 2026-10-15
---

# UC-75 implementation plan — run2b pointer-noise retune + off-axis trust guard

Challenger verdict: **Approve** (round 1). Its four minor notes are folded in and marked [C1]–[C4].

TARGET_DIR = C:\dev\drone-fly-uc-75-retune-run2-pointer-noise (base = main incl. UC-74/PR #81, so the "stacks on unmerged UC-74" pitfall is resolved). Brief: profiles: [] (none); paths.production = src/drone_fly/**, paths.test = tests/**; build: `uv run --extra dev pytest` / `uv run ruff check .` / `uv run ruff format .`

## Analysis
- **Run-2 config:** src/drone_fly/training/configs/student-liftoff-run2.yaml has perception.pointer {bias_sigma_rad 0.05, jitter_sigma_rad 0.03, dropout_p 0.1, bearing_tol_deg 20.0}, marked PROVISIONAL pending UC74.
- **Measured source:** docs/harness/uc74-arena-pointer-calibration.json has suggested_bias_sigma_rad 0.0073442, suggested_jitter_sigma_rad 0.0850286, suggested_pointer_bearing_tol_deg 16.3781 and detection_rate 0.52318 (395/755).
  - The jitter is a PER-AXIS std: pointer_eval.py:499-501 concatenates the az and el residuals.
  - That matches the sim model. The per-tick ptr_jit is a 2-wide standard normal (az, el) scaled by jitter_sigma_rad (perception.py:642, perception_ref.py:248), so the values drop in 1:1.
- **Training config loader** (training/config.py:202-225, 639):
  - `extends` is deep-merged by load_raw, which pops `extends` at every level (config.py:205), so the merged raw mapping carries no `extends` key [C2].
  - PointerSection (config.py:462-470) is strict and already has all four keys, so no schema change is needed.
  - `name` defaults to the file stem (config.py:204).
- **Noise injection** (this resolves the dropout_p pitfall against the code):
  - Pointer dropout is drawn per EPISODE only. ptr_drop lives in EPISODE_DRAWS (perception.py:100-105).
  - On reset, ptr_dropped = ptr_drop < pointer_dropout_p (perception.py:499; mirrored in student/perception_ref.py:134), and avail = target_valid & ~ptr_dropped (perception.py:647).
  - There is NO per-tick availability parameter. Adding one would need a new TICK_DRAWS slot (perception.py:83-97). That changes the fixed draw layout the NumPy reference replays for parity, which shifts the random stream for every existing config.
  - It would also need new fields in PerceptionConfig / PointerSection / distill.py:341-344 / checkpoint.py:101-104 / student_obs.py metadata / perception_ref. That is the sim pointer path, which AC5 forbids touching.
- **Student statefulness:**
  - K1Student (controller/student.py:154-165) rebuilds its propagated state from zeros on every forward, so it is feedforward per tick.
  - The pointer is not in the obs history EMA (student_obs.py:20-24).
  - What the student learns about pointer absence is therefore governed by the marginal share of training ticks with bit=0.
- **Live reader:** src/drone_fly/harness/liftoff/pointer_cv.py, HudPointerReader __init__ L126-156, from_config L157-191, read L320-330.
  - The az convention everywhere is atan2(by, bx): sim perception.py:643 and pointer_eval.az_el L71-74.
  - UC74's |az| bins are body-frame |az_gt| (pointer_eval.py:478-482).
- **Live FP suppression:** none. fp_suppress and bearing_tol exist only in the sim, perception_ref and checkpoint, not in harness/. In the sim, the pointer sub-gate is masked by ptr_avail (perception.py:662-663).
- **Preset tests** (test_harness_presets.py):
  - L291-296 require every liftoff_arena key to exist in the liftoff_5inch default (strict overlay).
  - L341-346 parametrise the source-comment checks.
  - L298-313 check the built reader.
- **Real CLI** (training/cli.py:831-878):
  - `train student` needs --teacher <run> --noise <noise_model.yaml>, and --resume takes <run folder>.
  - `train export <run> --out <file.pt>` requires --out and takes --checkpoint val|final|itN.
  - The use-case text's recipe omits these.

## Proposed Solution
1. **NEW src/drone_fly/training/configs/student-liftoff-run2b.yaml** (AC1, AC2, AC7)
   - Keys: `extends: student-liftoff-run2`, `name: student-liftoff-run2b`, and ONLY `perception.pointer`:
     - bias_sigma_rad: 0.0073
     - jitter_sigma_rad: 0.085
     - bearing_tol_deg: 16.4 (p95 13.1 deg x 1.25)
     - dropout_p: 0.5
   - Each overridden line carries an inline comment citing UC74 + docs/harness/uc74-arena-pointer-calibration.json + the key.
   - The word "provisional" appears nowhere in this file.
   - A header comment covers:
     - The purpose: A/B against the unchanged run2.
     - The dropout rationale (item 2).
     - The real recipe: `drone-fly train student --config student-liftoff-run2b --teacher <run> --noise <noise_model.yaml> --device cuda [--resume <run folder>]` → `drone-fly train export <run> --out <student-run2b.pt>` → `drone-fly harness run --harness liftoff --checkpoint <student-run2b.pt> --preset liftoff_arena`.
     - That the CUDA retrain and the re-eval vs run-2 are user-side GPU follow-ups.
   - student-liftoff-run2.yaml stays UNCHANGED.
2. **Dropout resolution:** keep the per-episode proxy at dropout_p = 0.5, with no per-tick parameter in this UC. The YAML comment records the rationale:
   - (a) The policy is feedforward per tick, so the marginal share of bit-0 ticks is what matters. That share is p; live it is 1 - 0.523 = 0.477, rounded to 0.5. The guard only raises the live unavailable share.
   - (b) 0.5 keeps half the episodes pointer-on, so the student still learns to use the pointer.
   - (c) A per-tick or state-conditioned model needs a new TICK_DRAWS slot plus sim/reference changes, which AC5 forbids, so it becomes a follow-up.
3. **MODIFY src/drone_fly/harness/liftoff/pointer_cv.py** (AC3)
   - HudPointerReader.__init__ gains the keyword `max_bearing_az_deg: float | None = None`, stored as float or None. None means off, which is the current behaviour.
   - from_config reads camera.pointer.max_bearing_az_deg:
     - Call require_number DIRECTLY with allow_none=True, min_value=0 exclusive, max_value=180.
     - Do NOT use the existing `num` lambda. It wraps the value in float(), so a null would crash [C1].
     - Absent or null gives None.
   - read(): after body_bearing and the existing all-zero check, if a threshold is set and degrees(|atan2(by, bx)|) is STRICTLY greater than it, return UNAVAILABLE.copy() (bit 0, bearing zeroed). Exactly equal stays available.
   - The guard goes in read(), not in detect_icon(), which stays a pure pixel detector.
   - pointer_cv must NOT import from offline/ (offline imports pointer_cv).
   - The module docstring gets a short UC75 paragraph:
     - It cites UC74's off-axis verdict.
     - It notes that a guarded block looks identical to "no icon" downstream.
     - It notes that the only pointer FP gate is sim-side and already masked by availability, so nothing composes inconsistently.
4. **MODIFY src/drone_fly/harness/configs/liftoff_5inch.yaml:** add `max_bearing_az_deg: null` under camera.pointer with a comment (UC75; off by default). The strict-overlay test needs the key there, and default-off leaves every existing config unchanged.
5. **MODIFY src/drone_fly/harness/configs/liftoff_arena.yaml:** under camera.pointer set `max_bearing_az_deg: 30.0`.
   - The comment cites UC74 real-frame captures: faithful within |az| ~20-25 deg (median ~4 deg); clamps toward the screen edge beyond ~30-40 deg (error 15-75 deg).
   - The comment must match the provenance regex fit|captures|real frames|CALIBRATE.
   - With the ROI [700,130,1200,760] at 122 deg hfov the reader can report up to ~43 deg, so 30 deg is a real cut.
6. **MODIFY docs/harness/uc74-arena-pointer.md** (developer; docs sit outside paths.production/test, as in UC-74)
   - Add a new "UC75 — run2b" subsection after "AC9 — inputs for a future run-2 noise re-tune". It covers:
     - The applied values.
     - The dropout rationale.
     - The guard and the arena's 30 deg.
     - The real retrain/export/fly recipe.
     - That the retrain and re-eval are deferred to the user's GPU.
     - EXPLICITLY, that existing run-2 exports flown with --preset liftoff_arena now get the |az| guard too [C4].
   - Update the "natural next refinement" sentence in "Conversion-model verdict" to point at the new key.
   - Add a note in "Reproduce the calibration": re-measuring through the arena preset now applies the guard, so set camera.pointer.max_bearing_az_deg to null to reproduce the committed JSON. pointer_eval computes its stats via reader.read() (pointer_eval.py:240/473).

NOT touched (AC5): sim/perception.py, sim/student_obs.py (VERSION 3 / DIM 65), student/perception_ref.py, student/checkpoint.py, training/distill.py, training/config.py, student-liftoff-run2.yaml.

## Files Affected
**Production (developer):**
- src/drone_fly/training/configs/student-liftoff-run2b.yaml (new)
- src/drone_fly/harness/liftoff/pointer_cv.py
- src/drone_fly/harness/configs/liftoff_5inch.yaml
- src/drone_fly/harness/configs/liftoff_arena.yaml
- docs/harness/uc74-arena-pointer.md (docs)

**Test (qa):**
- tests/test_training_student.py (beside test_run2_config_plumbs_the_perception_features, L129):
  - a) Provenance: load the committed JSON from the repo.
    - Check bias / jitter / tol with math.isclose(rel_tol=0.01) against the suggested_* keys.
    - Check |dropout_p - (1 - detection_rate)| <= 0.05.
  - b) Exact override: flatten load_raw for run2b and for run2.
    - The differing leaves must be exactly {name, perception.pointer.bias_sigma_rad, jitter_sigma_rad, dropout_p, bearing_tol_deg}.
    - load_raw strips `extends`, but exclude it defensively anyway [C2].
    - Assert run2 is still (0.05, 0.03, 0.1, 20.0).
  - c) Load:
    - student_from_mapping(load_raw("student-liftoff-run2b"), validate_tracks=False) succeeds.
    - distill.perception_config maps the four values (same pattern as L135-155).
    - run2b appears in packaged_names().
  - d) YAML text: the overridden lines cite UC74, and the file contains no "provisional".
- tests/test_harness_pointer_cv.py (fixtures in tests/pointer_fixtures.py):
  - e) Default: no kwarg, a from_config without the key, and an explicit null all give None. An off-axis icon reads available exactly as today.
  - f) Boundary:
    - Draw an off-centre icon, take its pixel from detect_icon, and compute its actual |az| from body_bearing.
    - Build BOTH boundary readers through the same construction path, with an otherwise identical camera/ROI, at thresholds az+0.5 deg and az-0.5 deg.
    - The az+0.5 reader must read available (bit 1, bearing unchanged); the az-0.5 reader must read UNAVAILABLE (bit 0, all zeros).
    - Cover both the kwarg path and the from_config path [C3].
  - g) Validation: from_config rejects <= 0 and > 180 with HarnessConfigError and accepts null.
- tests/test_harness_presets.py:
  - h) The liftoff_arena reader has max_bearing_az_deg == 30.0, and the default liftoff camera.pointer.max_bearing_az_deg is None. Add "max_bearing_az_deg" to the test_arena_calibrated_values_name_their_source parametrisation.
  - i) AC5 guard: student_obs VERSION == 3 and DIM == 65.

## Risks & Considerations
- **Dropout is a proxy.** It matches the live MARGINAL availability but not its correlation with state.
  - Live, the pointer drops at the start line, at gate passes and off-axis; sim pointer-on episodes keep a perfect pointer at every az.
  - A deterministic state-conditioned sim guard would close that gap, but it changes the sim pointer path and reference parity, so AC5 puts it out of scope.
  - Follow-up UC if the run2b re-eval shows pointer over-reliance.
- **The guard is live-only**, so the sim never produces a "guarded-unavailable" pointer. This is tolerated because the student now trains with bit-0 in 50% of episodes. Same follow-up.
- **Jitter is mildly conservative.** The calibration stats were measured on |az_gt| < 50 deg and inlier-trimmed to 80%. With the 30 deg guard live, the real residual is somewhat narrower than 0.085, and the real noise is Gaussian plus a fat tail. Accepted, because the UC mandates the measured numbers.
- **Bias comes from one capture.** suggested_bias_sigma_rad is the hypot of ONE capture's median bias, not a spread across episodes. It is used as the per-episode sigma per UC74/UC75.
- **Behaviour shift:** 2.8x jitter, 50% dropout and the guard mean the student leans less on the pointer. No flight-quality claim is made; the re-eval vs run2 is the deferred user-side experiment.
- **Reproducibility:** re-running UC74's calibration through the arena preset changes its stats, which the doc note covers. The provenance test reads the committed JSON, so it is unaffected.
- **Backward compatibility:** the guard is None by default everywhere and only liftoff_arena opts in; run2 is unchanged and asserted. Existing run-2 exports flown with the arena preset DO get the guard. That is intended and documented [C4].
- **The UC's recipe shorthand is inaccurate** (--resume takes a folder; --teacher/--noise are needed; export needs --out). The docs and the YAML header use the real CLI.
- **No GPU** or training is needed in-sandbox.

## Challenger recommendation (follow-up, out of scope)
A deterministic, state-conditioned sim pointer guard (|az| > threshold → unavailable) to close the train/deploy asymmetry, if the run2b re-eval shows pointer over-reliance.
