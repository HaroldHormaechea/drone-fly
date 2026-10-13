# ADR 0008: Flying the teacher in Liftoff with oracle gates

- **Status:** accepted (UC-69)
- **Date:** 2026-10-13
- **Builds on:** [ADR 0007](0007-track-library-and-teacher.md) (track library, teacher-obs v1,
  CourseFrame, the conversion fingerprint), [ADR 0002](0002-live-harness-layering.md) (harness
  parts and the runner), [ADR 0006](0006-profile-driven-batched-sim.md) (action contract v2)
- **Code:** `src/drone_fly/harness/` (`oracle`, `teacher`, `scoring`, `session`, `runner`,
  `offline/oracle_report`), `liftoff/gamepad.py` (`arm_remaining_s`), `cli/main.py`
  (`harness run --teacher`, `harness oracle-report`),
  `scripts/liftoff/Invoke-LiftoffAttempts.ps1` (`-Teacher`)

## Context

Before investing in perception (the UC70 student and the UC71 detector), we need to know whether the
**control** problem is solved in the real game. The UC68 teacher was trained with exact gate poses
and obstacle geometry. If it flies Liftoff well when it gets the same exact observation, the
remaining gap is perception. If it does not, the gap is sim-to-Liftoff dynamics or timing, and the
way it fails says which.

## Decision

1. **An oracle gate source** (`gate_estimator: oracle`) replaces the detector. It loads the
   catalogue entry exactly as the teacher was trained on it: `tracks.loader.load_course` with the
   checkpoint sidecar's conversion. It refuses a different conversion fingerprint, props sha256
   or map scene sha256. Each tick it maps the telemetry pose into the course frame and advances
   the route with the sim's own `gate_crossed` (slack 0) and `RouteProgress`.
2. **One frame.** The oracle runs at `world_scale` 1.0, so the preset's value (4.0 for the K1
   models) is ignored and recorded. The course's `CourseFrame` is the only place lengths change
   frame.
3. **The observation is the reference builder.** `TeacherObservationAssembler` calls
   `tracks.teacher_obs_ref.build_teacher_obs`, the NumPy reference the batched sim builder is
   tested against. Its inputs are:
   - the oracle's course-frame state and route progress;
   - the previous tick's RAW policy output;
   - the arming time left, from the gamepad sink: `max(arm_settle_s, arm_press_s)` counted down
     from the Arm press, which is 0.5 s at tick 0, as in the sim.

   The teacher needs the `oracle` block, so pairing it with a vision detector fails the coverage
   check (exit 4).
4. **Episode monitor.** The sim ends an episode on contact, but in Liftoff contact is physical, so
   the live attempt ends only by fixed rules:
   - time limit;
   - out of bounds or out of the corridor;
   - settled after a crash;
   - inverted near the ground;
   - respawn;
   - no lift-off.

   Contact under the sim's own rules is scored **offline**. That gives two completion figures:
   - `completion_raw`;
   - `completion_sim_rule`, which is the headline.

   The ends go through one new runner branch (`EstimatorStop`), which stops before anything is
   sent on that tick.
5. **Start check, before Arm.** At tick 0 the telemetry spawn must match the XML spawn: 1 m
   horizontally, 1 m vertically, 20° of yaw, and at rest. If it does not, the run exits 6 and the
   drone is never armed. So a wrong Last Session cannot produce scores.
6. **Leakage guard, fail-closed.** `BuildContext.purpose` defaults to `None`. The oracle accepts
   only `eval` and `diagnostic`, and `harness run` passes `eval`. No training module imports the
   oracle, and no training CLI exposes it.
7. **Concurrent-training guard.** The teacher and Liftoff share one GPU, and the teacher runs on
   the CPU. The session refuses with exit 2 while the teacher's run is still writing checkpoints
   or progress. `--allow-concurrent-training` overrides this, and the session is then labelled
   as not counting toward AC9.
8. **Sessions.** Attempts go to
   `artifacts/eval/<drone>/<run>/oracle/<track>/sessions/<UTC>-<sha8>/` (gitignored). The folder
   holds:
   - the frozen checkpoint, its sidecar and the run config;
   - `attempts/`;
   - `report.md` and `report.json`.

   A new checkpoint sha starts a new session. Only a summary report is committed, under
   `docs/harness/reports/`.
9. **Anti-cheat on every live Liftoff run.** `require_no_anticheat` now runs before arming on
   every live Liftoff run, the detector path included. Dry runs and recording-only sinks skip it.

## Consequences

- The verdict is mechanical:
  - **control transfers** when live sim-rule completion is within 20 pp of the sim's randomized
    completion on the same track and start;
  - **does not transfer** otherwise, naming the dominant failure mode;
  - **inconclusive** when the sim itself is below 50 %.
- A **geometry disagreement** flag marks sessions where raw and sim-rule completion differ by more
  than 20 pp. That points at the stale-build scene or at TRS approximations, not at control.
- One train track with N = 10 gives a wide Wilson interval; the report states it. Liftoff's own
  race logic (checkpoints, auto-restart) is not observable. The oracle's crossings are
  authoritative, and the live and offline crossings are compared.
- Telemetry latency (8–14 ms) is not corrected; the sim's 0–60 ms action-delay randomization
  covers it.
- The lift-off rule uses the sim's rule (above the resting height and climbing) with a 2 cm margin
  against telemetry jitter, or 0.3 m above the support, whichever comes first.

## Results (2026-10-13, AC9)

The session flew `teacher_it1920` (val-selected) on Infinity Loop for 10 attempts. The summary is
[docs/harness/reports/uc69-oracle-liftoff-arena-04.md](../harness/reports/uc69-oracle-liftoff-arena-04.md).

**What transferred:**
- take-off was clean;
- every attempt flew 11 of 33 crossings (a full lap plus three gates), with 91.7 % per-passage
  success;
- the loop ran at 50 Hz with 1 missed deadline;
- the live and offline crossings agreed exactly.

**What ended every attempt:**
- In all 10, about 18.3 s after spin-up, the pack hit Liftoff's 8.10 V low-voltage cutoff
  (74 % charge left), and the motors cut to idle under full stick.
- The teacher saturates the collective almost all the time. The simulator has no battery, so the
  sim reaches 100 %.
- At 24 m/s the measured specific force at full stick is only about 2 g median (3 g p95), against
  the profile's static 4.23 g. So Liftoff laps in 13.0 s against the sim's 10.6 s.

**Changes made after this run:**
- The classifier gained a `power_loss` mode (priority 1c): every motor below 3000 rpm while the
  throttle stick is above 0.9.
- A take-off tumble now requires that the attempt never reached the first gate.
- The report records the pack voltage and charge at the cut, the time from spin-up, and the
  full-stick specific force.
- It exports the per-attempt power time series for fitting a pack / thrust model.

**Conclusion:** control transfers with exact gates. The next gap is a power / aerodynamics model in
the simulator (pack sag and cutoff, thrust vs. airspeed), not perception.
