# Use Case 66: Drone profiles + system identification of Liftoff drones (micro + 5-inch)

## Summary
The live Liftoff runs show the policy saturating every stick from the first tick and tumbling. The root cause is a **sim-to-Liftoff dynamics gap**: training uses a hard-coded Meteor75 analog (m = 0.032 kg, T/W 2.5, max body rate 4 rad/s, hover ≈ 0.5 by construction) inside `gpu_prototype/`, while Liftoff flies its own airframes through Betaflight PIDs and rate curves. This use case adds two things. (1) A **drone profile** format: one committed folder per airframe under a new top-level `drones/<drone-id>/` (`drone.yaml` + `sysid.md`), loaded by a new `src/drone_fly/drones/` package. The Meteor75 nominal is ported into `drones/meteor75/drone.yaml` without changing its values. (2) A **system-identification** pipeline (`src/drone_fly/sysid/`, CLI `drone-fly sysid …`). It flies scripted test manoeuvres in Liftoff through the existing long-lived virtual pad, under a simple stabilizing controller driven by telemetry, records telemetry to `artifacts/sysid/<drone-id>/<session>/`, and fits the airframe parameters the simulator needs. Everything is expressed relative to hover (in g), because mass is not observable in Liftoff. Two Liftoff airframes are identified: the UC64 micro "Test small drone" (FT Gremlin, ZetaFlight Acro, Betaflight rates 100/0/70) as `liftoff-gremlin-micro`, and a 5-inch as `liftoff-5inch-<model>`. The UI driver is extended to **select the drone in Liftoff's menus**, so both sessions can run unattended. The fitted profiles are the input to UC67.

## Acceptance Criteria
1. **Folder structure:**
   - `drones/README.md` explains the layout: one folder per airframe, `drone.yaml` as the contract, `sysid.md` as the report, and raw data kept under `artifacts/sysid/` (gitignored).
   - Code lives in `src/drone_fly/drones/` (schema + loader) and `src/drone_fly/sysid/` (manoeuvres, recorder, fitter, report).
   - No new top-level folder other than `drones/` is introduced by this use case.
2. **Profile schema:**
   - `drone.yaml` has a `schema_version`, an `id`, a `source` (`analog` / `liftoff-sysid` / …), and a nominal block.
   - The nominal block holds:
     - hover throttle (as a stick fraction);
     - the thrust curve (throttle fraction → vertical acceleration in g, as points or polynomial coefficients);
     - per-axis rate response (Betaflight rate profile, first-order time constant, pure delay);
     - linear and quadratic drag (per body axis or isotropic);
     - motor lag;
     - control-input-to-gyro delay;
     - the camera (HFOV and uptilt) when known.
   - It also has a `randomization` block: a min/max per parameter, used by UC67.
   - Every measured value carries its provenance: session id and fit residual.
   - `drone_fly.drones.load(id)` validates the file and raises a clear error naming the field on any missing or out-of-range value.
3. **Meteor75 port:** `drones/meteor75/drone.yaml` reproduces the values in `src/drone_fly/adapter/meteor75.py` and `gpu_prototype/hover_gpu.py`. A test asserts they agree. Existing code paths are not yet switched over (that is UC67).
4. **Drone selection in Liftoff:**
   - `drone-fly harness ui select-drone --name "<Liftoff drone name>"` selects a saved drone through Liftoff's menus, using positions measured on the 2560×1080 reference like `ui_driver.POINTS`.
   - It is verified by a pixel or telemetry check, not a screenshot read.
   - It fails with a named error rather than flying the wrong airframe.
   - Unit tests cover the scaling and validation, as for the existing UI actions.
5. **Scripted manoeuvres:**
   - `drone-fly sysid liftoff --drone <id> [--session NAME]` assumes a level is loaded with the named drone (or selects it, see AC4).
   - It arms, takes off to a hold altitude under a stabilizing controller (attitude + altitude PID on telemetry), then runs a fixed, seeded schedule. Each experiment returns to a stabilized hover before the next. The schedule:
     - throttle steps / chirps;
     - roll, pitch and yaw rate doublets at several amplitudes, up to the profile's full-stick rate;
     - a short coast for drag (level forward flight, then neutral).
   - It records raw telemetry (~100 Hz), the commanded sticks and wall-clock timestamps to `artifacts/sysid/<drone-id>/<session>/` with a `meta.json`.
6. **Safety:**
   - The sysid run uses the same fail-safes as `harness run`: F12, stale telemetry, focus loss, disarm with RPM confirmation, and the pad watchdog.
   - It also enforces an altitude floor/ceiling and a tilt bound outside the active experiment window.
   - Any violation aborts that experiment, re-stabilizes, or disarms. It is recorded in `meta.json` and never crashes the run.
   - A crash (collision) ends the session cleanly, and the session can be resumed from the next experiment.
7. **Fitting:**
   - `drone-fly sysid fit <session-dir> --drone <id>` fits every nominal field from AC2 and writes `drones/<id>/drone.yaml` and `drones/<id>/sysid.md`.
   - `sysid.md` contains the fitted values with uncertainty, residual plots saved next to it as PNG, and the default randomization ranges (nominal ± max(3σ, 10%)).
   - The fit uses a time-split held-out portion of the session.
   - The report states the one-step and 1-second open-loop prediction error on held-out segments for attitude rate and vertical acceleration.
8. **Fit validation:** on held-out segments, the 1-second open-loop prediction of body rates has a median absolute error ≤ 15% of the commanded rate amplitude. Vertical acceleration is within ≤ 0.15 g. If not met, the report says so explicitly; the profile is still written but marked `fit_quality: poor`.
9. **Hermetic tests:**
   - A synthetic plant with known parameters (including delay and drag) generates a session.
   - The fitter recovers every parameter within 5% (delay within one telemetry tick).
   - Schedule generation is deterministic for a seed.
   - The safety logic triggers on synthetic bound violations.
   - No test needs Liftoff, a GPU or a gamepad.
10. **Live sessions (executed by the team, unattended allowed):**
    - One sysid session each for `liftoff-gremlin-micro` and one 5-inch, on a flat open map with room for the manoeuvres.
    - Both profiles are committed (YAML + report + PNGs; no raw telemetry or frames).
    - The report compares the measured micro hover throttle with UC64's −0.06 stick figure.
11. **Docs:** `docs/harness/liftoff-live-session.md` and the `liftoff-episodes` skill gain a short "measure a new drone" section: select the drone, run sysid, fit, commit the profile.

## Potential Pitfalls & Open Questions
- **Risk** — The stabilizing controller itself must work on an unknown airframe. Start it with conservative gains and an auto-tune or gain-scheduling step on the hover throttle, measured first by a slow throttle ramp at ground level.
- **Ambiguity** — Which map to use for sysid. Prefer a large open map with a flat floor; the team picks one and documents it. It must be selectable through "Last Session" or the new menu automation.
- **Missing input** — Which 5-inch: the team picks Liftoff's stock 5-inch freestyle/race build and records its exact Liftoff name and settings in `drone.yaml`.
- **Edge case** — Liftoff's per-drone rate profile may differ from the preset's `controls.rates`. The pad's inverse rate curve must use the drone profile's rates, which UC67 wires through.
- **Assumption** — Telemetry stays at ~100 Hz and motor RPM is available on both airframes.
- **Risk** — Battery sag in Liftoff changes thrust over a session. Record battery voltage and either fit it or keep sessions short; this is noted in the report.
- **Risk** — Unattended menu automation may misclick. Every step is verified by telemetry or pixel checks and aborts on mismatch (the same rule as `ui_driver`).

## Original Description
> Ok we'll have to go the next step. Despite having "memorized" the course as you mentioned in a previous UC, we really aren't flying well. The AI just tumbles right from the start, with throttle/yaw/roll/pitch input from the start.
>
> We do have the possibility of exporting the track geometry for further trainings by converting it to pybullet, but the thing is, I want to GENERALIZE this so it can tackle random courses in liftoff.
>
> Ideas?

(After the proposed plan — measure Liftoff's drone, match the sim with domain randomization, multi-track teacher, detector-noise student, multi-map detector, Liftoff evaluation:)

> Prepare everything as use cases. Once they are set-up, and we validate them together, wait for me to give you the go ahead (I'll be out of home for a couple hours so that's good time to use, but i'll leave after we have set this up). When doing the analysis, make sure whateve rwe create fits in a good understandable folder structure, and is scalable in case we need to train more drones and such.

## Clarifications
- Q: Where should drone and track definitions live?
  A: Top-level `drones/` + `tracks/`; code in `src/drone_fly/{drones,sysid,sim,tracks,training}`; outputs under `artifacts/{sysid,training,eval}/<drone>/…`.
- Q: Which Liftoff drone(s) should be identified?
  A: The micro and a 5-inch.
- Q: How should the 5-inch be measured while the user is away?
  A: Automate drone switching through Liftoff's menus.
- Q: May the dev-team run live Liftoff sessions unattended?
  A: Yes, Liftoff only, with League / Riot Vanguard closed.
