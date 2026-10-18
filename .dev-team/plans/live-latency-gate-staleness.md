---
plan_for: (free-form task)
work_branch: feat/live-latency-gate-staleness
run: drone-fly
approved: 2026-10-18
---

# Live gate pose at capture + Smith-predictor latency compensation — approved plan

lat-challenger approved this after one revision round. The orchestrator accepted it; the user is away and asked for autonomous work. The branch was cut at b286822 (spawn-yaw alignment, since merged to main as PR #98). Brief: profiles [].

## Analysis
1. **Gate pose staleness (confirmed in code).**
   - `GateEstimator.detection_from_pose` (gate_cv.py:242-282) builds r_wc from the current tick's state.quat and pos from state.pos.
   - The async learned detector's `_raw_detections_async` (gate_learned.py:376-392) returns the last completed detections, which are 6-9 frames old. `detect()` re-runs PnP on them every tick with the current state.
   - Resulting error: est = pos_now + R_now R_cap^T (G - pos_cap).
   - The capture stamp is already in `self._det_stamp`. Frame.stamp, FlightState.received_at and WallClock all use perf_counter.
2. **Control latency.**
   - The obs is assembled from the latest telemetry with no account of the ~75 ms command-to-measured-response lag or the in-flight commands.
   - The contract exposes plant_step(state_dict, raw_action) with DT=0.02.
   - The 07/11 gpu_prototype checkpoints are action-spec v1.
   - Training randomizes a 0-2 step action delay (gates_dr.py, 0-40 ms), so full compensation of D lands inside the trained distribution. Do not tune delay_s to "leave some" latency.
3. **Eyes ground hits (70-dim).** Real, but separate. Deferred.

## Proposed Solution

### A. Pose at capture (gate_cv.py; benefits opencv and learned)
- **Pose history:** a bounded deque of (received_at, pos, quat), maxlen ~64, cleared in reset(). The discontinuity reset at runner.py:558-561 therefore also clears it. `update()` appends the current state before detect_frame and skips a duplicate received_at.
- **`_pose_at(stamp)`:**
  - returns the current pose when the stamp is None or the history has fewer than 2 distinct stamps;
  - otherwise it clamps to [oldest, newest] and interpolates: linear for pos, normalized slerp for quat with the shortest-arc sign flip;
  - **clock-mismatch guard:** a stamp outside [oldest - 0.25 s, newest + 0.25 s] (POSE_STAMP_TOL_S) falls back to the current pose, increments `_pose_fallbacks`, and warns once.
- **`pose_offset_s`:** gate_estimator config key, default 0.0, validated within [-0.2, 0.2]. The lookup happens at det_stamp + pose_offset_s.
- **`detection_from_pose`** uses the capture pose for r_wc, center and d. Distance is measured from the capture position.
- **Unchanged:** tracking and the pass rule stay on the current pose. Loopback is untouched (no frame source).

### B. Smith predictor (new src/drone_fly/harness/latency_comp.py)
- **Pure helper `predict_from(state_dict, r, now, commands, delay_s, dt, plant_step)`:**
  - horizon H = (now - r) + D, giving n = round(H/DT) steps, capped at max_horizon_s = 0.2;
  - at step j it applies the newest command with sent_at <= r + j*DT - D; before the history starts it uses the pre-history command;
  - it returns the state unchanged when there are no commands or n == 0.
- **`LatencyCompensator`:**
  - record(sent_at, command);
  - predict(state, now) -> FlightState, with battery preserved;
  - reset(), called on discontinuity together with aligner.reset.
- **v1 only.** A v2 policy makes `auto` resolve off with a warning, and explicit `on` raises HarnessConfigError.
- **Config:**
  - `runner.latency_compensation`: off | auto | on, default auto;
  - `runner.latency_delay_s`: validated 0..0.3. Its default comes from the V1 real-data fit, rounded to 5 ms; 0.075 is a placeholder until then.
- **`auto` rules:**
  - on only for gpu_prototype contract checkpoints on the wall clock, and only if V1 passes; otherwise it resolves off for live too, and the report says so;
  - off on the sim clock (loopback stays byte-identical) and for teacher/student;
  - explicit `on` with teacher/student is refused.
  - The keys go into configs/liftoff_5inch.yaml and configs/loopback.yaml.
- **Runner wiring:**
  - perception (gate estimator, free-space, aligner.capture) keeps the measured state;
  - only `assembler.assemble` takes `latency_comp.predict(state, t2)`, then aligner.apply as today;
  - `record(clock.now(), command)` runs after a successful send;
  - TickRecord gains an optional predicted_state, serialized only when not None;
  - a one-time `latency_comp:<n>steps` event is recorded.
  - Note: TickRecord.state lacks received_at, so get r as sim_time - ages_s["telemetry"] (telemetry.py:392), or from the telemetry CSV.
- **Docs** (docs/harness/liftoff-live-session.md): pose at capture; the compensator and what delay_s means; the training-delay rationale (compensate fully); fitting delay_s and pose_offset_s from recordings; troubleshooting.

### C. Deferred
The eyes ground-plane filter and the 20 Hz effective tick rate.

## Verification (gating, offline only; scripts in C:\dev\drone-fly\artifacts\diag\ or $TEMP; only unit tests are committed)

**V1 (REQUIRED, gates any live flight): open-loop replay on real data.**
- Data: liftoff-20261018T110* ticks.jsonl.
- Method: run the real predict_from with the recorded commands, using send time ≈ sim_time + latency total. Compare against the measured state interpolated at r+H. Skip ticks past the end of the recording or across a discontinuity.
- Report: median and p90 of attitude (deg), omega, vel and pos error, prediction vs stale baseline. Sweep D over 0..0.15 in 25 ms steps, per run and pooled. Also report omega error growth with horizon, and the tick count.
- Acceptance: the prediction beats the baseline on attitude and vel at the fitted D. The default delay_s is set to the pooled fit. If it fails, `auto` is off for live, reported as a finding.

**V2: pose staleness on real data.**
- est_fixed comes from the real `_pose_at`; expect |est_fixed - G0| < ~0.1 while tilting.
- Sweep pose_offset_s over -0.1..0.1 and report the best value. The default stays 0 unless the fit is clear and consistent; the orchestrator decides.

**V3: closed-loop sim at harness timing.**
- Timing: 50 ms period, 75 ms delay, 160 ms gate lag, with the real compensator and `_pose_at`.
- Nominal acceptance: 5 s, no crash, <10% saturation.
- Robustness cases, each against the nominal predictor:
  - predictor delay 45 and 105 ms;
  - truth RATE_KP x0.5 and x2;
  - truth thrust x0.8 and x1.2.
- A single robustness failure is a finding. The check fails, and escalates, only if the nominal case fails or both cases of one mismatch pair crash.

**Unit tests (CPU-only, numpy plant_step):**
- capture-pose vs current-pose centre;
- single-entry history is identical to the old behaviour;
- clamp, slerp midpoint, reset clears the history;
- an out-of-tolerance stamp falls back to the current pose and the counter increments;
- the offset shifts the lookup;
- predict_from matches a manual plant_step roll;
- horizon and command selection, identity with no commands, the cap;
- v2 makes auto resolve off and refuses on; config validation;
- runner: assemble gets the predicted state while perception and the aligner get the measured one;
- existing loopback parity tests pass unchanged.

**No live flight in this run**; the orchestrator owns it.

## Files Affected
- **Production:** src/drone_fly/harness/liftoff/gate_cv.py; src/drone_fly/harness/latency_comp.py (new); runner.py, config.py, session.py, types.py; configs/liftoff_5inch.yaml (+ `pose_offset_s: 0.0`), configs/loopback.yaml.
- **Tests:** tests/** as above (QA).
- **Docs:** docs/harness/liftoff-live-session.md.

## Risks
- Model mismatch: plant_step is not Liftoff's 5-inch dynamics. V1 measures this directly, and the 0.2 s cap and off switch limit the damage.
- received_at includes UDP jitter.
- A render-to-capture bias may exist; pose_offset_s covers it.
- Eyes rays stay about one delay stale.
- V1 data is limited (short, crashing, aggressive runs), so the 110* runs must be pooled.
- Send times are reconstructed from the per-tick latency fields.
- Rebase: the branch is based on b286822, and main now has PR #98 merged (same commit plus tests). Rebase onto origin/main before opening the PR.
