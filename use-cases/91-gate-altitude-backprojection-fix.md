# Use Case 91: Gate-altitude back-projection fix (perception/calibration)

## Summary
UC-90 traced the live climb-over-gate-0 to a **perception** bug: the harness's learned-gate pipeline
systematically underestimates gate altitude. In the best model-14 flight (uc-89-results, attempt 5) the
gate-0 z estimate has a median of ~1.6–2.7 m at every range. Beyond 40 m it is very noisy (stdev up to
24 m, spikes to +59.7 m). The true gate 0 sits ~10–18 m up. The policy is told the gate is near the
ground while it is ~12 m overhead, so it never descends to it. Investigate and fix the gate-centre 3D
back-projection in the harness perception, where a learned 2D gate detection becomes a world/contract
position. Questions: is range from monocular depth or from the gate's known physical aperture
(size-based range / PnP)? Are the camera extrinsics (uptilt / pitch) and intrinsics (FOV, principal
point) used in back-projection correct for the real Liftoff camera? Measure gate-z error against
ground truth (track geometry) as a function of range before and after the fix. Then re-fly model 14
(`training/14-grounded-launch`) live at detector width 768. Sim already passes 13.3 % with true gate z,
so a correct live gate z should transfer. No reward or training change: perception/calibration only.

## Acceptance Criteria
1. **Derivation documented.** A written account (in the PR / plan / a short doc under `docs/`) of exactly
   how the live gate-centre 3D position (especially z) is derived today: the code path from detector
   output → range → back-projection → world/contract frame. It names the files and functions, and
   whether range comes from monocular depth, bbox size, or PnP.
2. **Range from known gate geometry.** If range comes from learned/monocular depth (or another
   unstable source), the pipeline uses size-based range from the gate's known physical aperture (bbox
   size vs known gate dimensions, or PnP on the gate corners/keypoints when available) as the primary
   range source. A documented fallback covers a truncated or partially visible box (e.g. clipped at the
   image edge, where size-based range is invalid). Gate dimensions come from track data or a
   documented constant, not a magic number.
3. **Camera calibration verified.** The camera uptilt/pitch, FOV/focal length and principal point used in
   back-projection are checked against the actual Liftoff camera (drone preset settings / Liftoff
   config / evidence from recorded frames vs known track geometry). Any mismatch is corrected, with
   the sign convention of the uptilt rotation explicitly verified, since a sign error gives a systematic
   altitude bias that grows with range.
4. **Measured error vs range, before and after.** An offline evaluation on recorded live data (e.g. the
   uc-89-results flights: telemetry + Unity pose + track XML) reports gate-z (and ideally range and
   lateral) error against ground-truth gate positions, binned by range. It reports bias (median/mean) and
   noise (stdev / p90) for the old and new derivation. The fix must remove the systematic low bias:
   |median z error| ≲ 2 m within 40 m, and the noise beyond 40 m must drop substantially.
5. **Default-safe config.** The new derivation is selectable by config (e.g. a `gates.learned.range_mode`
   or equivalent). If the new mode becomes the default, existing behaviour stays reachable. Unit tests
   cover the back-projection math: a synthetic gate at a known pose projects and back-projects to
   within tolerance, including camera uptilt and off-centre principal point.
6. **Tests (CPU-hermetic).** Size/PnP range, uptilt sign, principal-point handling, the truncated-box
   fallback, and config passthrough are unit-tested without GPU or Liftoff.
7. **Live re-fly (orchestrator step).** Re-fly `training/14-grounded-launch` live on Infinity Loop at
   detector width 768 with the fix on, several attempts, Liftoff replay recorded. Report the gate-pass
   rate verified against ground truth (recording/telemetry vs track, never the harness's `gate_passed`
   heuristic), plus the live gate-z error on these flights.

## Potential Pitfalls & Open Questions
- **Assumption** — The bias is in back-projection (range and/or extrinsics), not in the detector's 2D box
  position. If the 2D boxes themselves are vertically biased, report it. Retraining the detector is out
  of scope.
- **Edge case** — Size-based range breaks for boxes clipped at the image border or partially occluded, and for
  gates seen strongly off-axis (aspect change). It needs a fallback and/or a validity gate.
- **Risk** — Fixing z while range is still wrong could move the error into the horizontal position. Evaluate full 3D
  error, not z alone.
- **Risk** — Ground-truth alignment: the track XML gate pose must be in the same frame as the telemetry
  (UNITY_TO_CONTRACT, course frame, spawn-yaw alignment). Mis-alignment would show up as fake bias.
- **Risk** — Liftoff can save stale replays (byte-identical to a previous attempt). Cross-check replays with md5, and
  fall back to telemetry-based gate crossings.
- **Risk** — The loop rate at 768 is ~26–31 Hz under GPU contention. A PnP/size path must not add measurable
  latency.

## Original Description
On the Liftoff box: UC-90 traced the climb-over-gate-0 to a perception bug — the gate detector
systematically underestimates gate altitude. From the best flight's telemetry (uc-89-results attempt 5),
the gate-0 z estimate is a median ~1.6–2.7m across all ranges and very noisy beyond 40m (stdev up to 24m,
spikes to +59.7m), while the true gate-0 is ~10–18m up. The detector reports the gate near the ground
when it's ~12m overhead, so the policy never descends to it.

Investigate the gate-center 3D back-projection in the harness perception (where the learned gate
detection becomes a world/contract position):
1. How is gate-z derived? Monocular depth, or the gate's known physical aperture size (size-based range /
   PnP)? Size-based range from the known gate dimensions is far more stable at distance — if it's using
   learned depth, switch to size/PnP for range.
2. Camera pitch/uptilt + principal-point calibration. A systematic low altitude bias strongly implies the
   camera extrinsics (uptilt angle) used in back-projection are off — a few degrees of pitch error at 38m
   is many metres of altitude error. Verify against the actual Liftoff camera.
3. Measure gate-z estimate error vs ground-truth altitude (track geometry) as a function of range, to
   confirm bias+noise and whether size-based range fixes it.
4. Then re-fly training/14-grounded-launch at detector width 768 and check whether it now flies through
   gate 0 (sim already does 13.3%, so a correct gate-z should transfer). Report the current gate-z
   derivation, the measured bias/noise, the fix, and the new gate-pass rate.

No reward/training change is needed — this is a perception/calibration fix.

## Clarifications
- Q: Live flights authorized? A: Yes (user, 2026-10-20): "you are authorized to stop any application and
  commit/push as required, and to run liftoff live tests. Run autonomously i won't be around".
