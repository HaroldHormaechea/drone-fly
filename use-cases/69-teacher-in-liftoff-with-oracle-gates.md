# Use Case 69: Fly the teacher in Liftoff with exact gate poses (oracle perception) to test control apart from detection

## Summary
Before investing in perception (the UC70 student and the UC71 detector), check whether the **control** problem is solved in the real game. This use case adds an **oracle gate source** to the live harness. Instead of running the learned detector, it computes next-gate observations exactly from the loaded track's XML (the UC68 catalogue entry / `harness/offline/track_xml.py`) and the drone's telemetry pose, with the same XML-to-world alignment that UC64 auto-labelling uses. With it, `drone-fly harness run --harness liftoff` can fly the **UC68 privileged teacher** checkpoint directly: its observation is assembled from oracle gates plus telemetry, matching the teacher's versioned observation schema, including obstacle features from the XML where the schema has them. A scripted session flies a set of attempts on the Liftoff tracks reachable today (Infinity Loop as Last Session, plus any track the UC66 `select-drone --level` path can load). Each attempt is scored from telemetry against the track XML and compared with the teacher's simulator completion on the same track. The outcome is a short **diagnosis report** that decides what to work on next:
- if the teacher flies well with oracle gates, the remaining gap is perception (UC70/UC71);
- if it does not, the remaining gap is sim-to-Liftoff dynamics or timing, and the report shows how it fails (tumble at take-off, oscillation, overshoot, crash on gate frames, timing misses).

The oracle is a diagnostic: it is never used to train or to tune held-out results.

## Acceptance Criteria
1. **Oracle gate source:**
   - The harness gains a selectable gate source, `perception: oracle`, next to the existing detector source.
   - Each tick it produces the next N gate poses relative to the body (and obstacle features where the observation schema has them) from the track XML and the latest telemetry pose.
   - It uses the same frame conventions and XML-to-world alignment as UC64 auto-labelling and the UC68 course conversion.
   - Gate progression (which gate is "next") follows the race route. It advances on a telemetry-detected plane crossing inside the aperture, and repeated and multi-lap routes are supported.
   - The run meta records `perception: oracle` and the track id. Every report or overlay that shows such a run labels it as oracle.
2. **Teacher deployment:**
   - The harness loads a UC68 teacher checkpoint.
   - It checks the observation-schema version in the checkpoint against the assembler, and refuses a mismatch with a clear error.
   - The observation is assembled from oracle gates + telemetry exactly as in the simulator. A hermetic test feeds identical state through the simulator observation and the harness oracle assembler and asserts equal vectors, within float tolerance.
   - Action contract v2 (UC67) drives the gamepad sink through the active drone profile (`liftoff-gremlin-micro`).
3. **Start alignment check:**
   - Before the first attempt, the session verifies that the telemetry spawn matches the track XML start within tolerance (the UC66 spawn check / UC64 projection check).
   - On a mismatch it aborts with a named error, so a misaligned track never produces misleading scores.
4. **Scripted attempts:**
   - `scripts/liftoff/Invoke-LiftoffAttempts.ps1` (or an equivalent `drone-fly` command) runs N attempts per track with resets, using the oracle source and the teacher checkpoint.
   - It uses the existing exit-code semantics and all fail-safes: F12, focus loss, stale telemetry, RPM-confirmed disarm, pad watchdog, and the anti-cheat guard.
   - Default: Infinity Loop, ≥ 10 attempts. Other tracks run when they are loadable without new menu automation.
5. **Scoring and comparison:**
   - Each attempt is scored from telemetry + XML: gates passed in order (plane crossings inside the aperture), completion, lap time, stop reason, time to first gate, and the max tilt/rate in the first 2 s.
   - The scoring is a module UC72 can reuse.
   - A report `artifacts/eval/<drone-id>/<run-id>/oracle/<track-id>/report.md` (+ JSON) puts each track's Liftoff oracle completion next to the teacher's simulator completion on the same track and the same start mode.
6. **Diagnosis:**
   - The report classifies failed attempts into failure modes from telemetry: take-off tumble, attitude oscillation, altitude loss / floor, gate-frame collision, overshoot / missed turn, timeout.
   - It also includes the harness tick-rate and missed-deadline statistics for the run.
   - It ends with a one-paragraph conclusion: whether control transfers (oracle completion within 20 pp of sim) or not, and the dominant failure mode.
   - The conclusion is posted in the PR and in `docs/harness/liftoff-live-session.md`.
7. **No leakage:** the oracle source cannot be selected by any training or fine-tuning command (UC70, UC72). A test enforces this.
8. **Tests:** hermetic tests cover:
   - oracle gate observations on a synthetic track XML + synthetic telemetry trajectory: poses, gate advance on crossing, repeats and laps;
   - sim/harness observation parity;
   - the schema-version guard;
   - the scoring and failure-mode classifier on synthetic trajectories.
9. **Live acceptance (unattended allowed, Liftoff only, League / Riot Vanguard closed):**
   - At least 10 oracle attempts on Infinity Loop with the UC68 teacher on `liftoff-gremlin-micro`.
   - The report is generated and committed as a summary only; no telemetry or frames are committed.
   - Acceptance is the pipeline producing the diagnosis. Flight success is reported, not required.

## Potential Pitfalls & Open Questions
- **Dependency** — Needs a trained UC68 teacher checkpoint and the UC68 observation schema. It runs after UC68 and before UC70.
- **Risk** — The teacher's observation may include obstacle information that the XML gives only approximately (rotated props as AABBs). Use the same conversion as training so the oracle matches what the teacher saw in simulation.
- **Edge case** — Liftoff's own race logic (checkpoint triggers, auto-restart, end-of-race screens) can disagree with the plane-crossing gate advance. The oracle uses its own crossing logic; differences are logged.
- **Assumption** — Infinity Loop is reachable as Last Session (UC66 restores it) and its XML is in the local trackdata. More tracks need UC72's `load-track`, unless the UC66 Quick Setup level list already reaches them.
- **Risk** — Telemetry pose latency (about 8–14 ms) makes oracle gate poses slightly stale. This matches what the live harness sees and is not corrected.

## Original Description
> After the end of these UC's we have scheduled, will we be able to actually train something that flies half-decently in liftoff?

Claude suggested decoupling control from perception: fly the UC68 teacher in Liftoff with exact gate poses from the track XML + telemetry before building the student and the detector. The user replied:

> yes, create it and shift numbers of other uc's as required

## Clarifications
- Q: Where does it go in the sequence?
  A: Right after UC68, before the student. The former UC69–71 are renumbered to UC70–72.
