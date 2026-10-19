# Use Case 89: Grounded-launch behavior + placeholder-gate frame bug (live 768 crash)

## Summary
After UC-87/88 made the controller smooth and the live detector was switched from input width 512 to
**768** (which tripled matched-gate ticks — the gate is now seen from the start pad), live Liftoff
flights on Infinity Loop **still crash within ~2.4 s, in the start chute, never leaving the ground
meaningfully**. Live telemetry (OneDrive `uc-88-results/`, the 768 run folders + `NOTES-768-flights.md`)
pins this to **two distinct, confirmed-plausible code defects** — NOT controller smoothness and NOT the
detector itself:

1. **Placeholder-gate frame bug (harness).** When the learned detector returns no valid gate, the
   student assembler (`src/drone_fly/student/obs_ref.py` / `build_student_obs`, no-detection path keyed
   on `target_valid`) emits a *placeholder* current-gate, and `src/drone_fly/harness/yaw_align.py`
   rotates gate blocks into the training heading. A world-frame placeholder (reported as `[1, 0]`,
   i.e. yaw 0 / +x) rotated by the live **−90° spawn heading** becomes a **fake gate ~90° to the side**.
   Consequence: with no detection, the policy steers hard at a phantom gate. This corrupted *every*
   512 flight — the "33 s best flight" was mostly chasing a gate that wasn't there.

2. **Grounded-launch gap (training).** From a **grounded start with a real gate dead-ahead (~38 m)** —
   exactly what 768 now provides — the policy answers by **holding a full 4 rad/s roll for ~2 s, then
   rolls into the chute wall as it lifts off**. Training *does* start grounded 50% of the time
   (`StartConfig mode: mixed, ground_fraction 0.5`, `arm_delay_s 0.5`), so it is not "never grounded."
   The gap is narrower: whether those ground starts present a **gate dead-ahead in view** at the live
   geometry and model the Liftoff **arm → ~0.5 s idle → launch** sequence, so the policy learns to lift
   off *toward* a visible gate instead of slamming into full roll on the pad.

Fix both, consistently across harness (deploy) and sim (training), then retrain the warm-start recipe
and validate the grounded launch in sim. Final proof is a live 768 re-fly on the Liftoff box.

## Acceptance Criteria
1. **Placeholder-gate frame correctness (harness + sim parity).** When the detector returns no valid
   target, the assembled gate input must NOT present as a spurious off-axis gate under yaw alignment.
   Either hold the last-known estimate in the correct frame, or emit a neutral input that yaw-alignment
   maps to "no usable gate" — never a world-frame constant that rotates into a fake ~90° gate. The
   assembled `curr_gate`/`rel` on no-detection must be frame-correct for a −90° spawn *and* arbitrary
   spawn headings. The detected path (valid target) stays **byte-identical** — no behavior change when a
   gate IS seen.
2. **Sim/harness consistency.** The no-detection gate semantics are identical in training (`CourseEnv`)
   and the live harness, so trained behavior matches deployment (same contract the UC-88 memory-parity
   AC established).
3. **Grounded start presents a gate in view.** Training ground-start episodes must place a gate
   dead-ahead and in view at a realistic distance (live ~38 m; at `world_scale 12` ≈ 3.2 training-scale
   — verify `spawn_distance_m` / course geometry covers it), through the Liftoff-like arm→idle→launch
   transition, so the grounded-launch-toward-gate regime is actually in the training distribution.
4. **Retrain validates the launch.** Retrain the warm-start recipe (`uc87-deploy-robust-warmstart`) with
   the fixes. In a **grounded-start sim eval with a gate in view**, the resulting policy lifts off and
   heads toward the gate — no full-roll-into-ground pathology — and grounded-start completion is
   non-zero and not materially worse than airborne-start completion. (Retrain is an orchestrator step on
   the GPU box, like UC-87/88/89-exp; the dev-team delivers the code + CPU tests, not the GPU run.)
5. **Tests (CPU-hermetic).** No-detection placeholder frame correctness (−90° + arbitrary spawn);
   grounded-start-with-gate-in-view present in the training distribution; arm→launch sequence behavior;
   back-compat / default-off where a knob is introduced.
6. **No regression.** The shipped UC-87/88 smoothness and the existing airborne-start completion must
   not regress. The 0.5 s deploy slew cap and the 768 detector config remain in force.

## Potential Pitfalls & Open Questions
- **Don't touch the detected path.** The placeholder fix must be a no-op whenever a valid detection
  exists (byte-identical) — only the no-detection branch changes.
- **Arm-delay interaction.** During `arm_delay_s` the motors are idle, so pad commands do nothing until
  spin-up; the pathological "already commanding full roll at lift-off" must be addressed across that
  transition, not just at t=0.
- **Frame of the fix.** The bug is a *frame* error (world vs. training-heading). Fix it in the frame
  where the rotation is applied; add a test that would have caught a −90° spawn specifically.
- **Scale / geometry.** 38 m real ÷ `world_scale 12` ≈ 3.2 training-scale; ensure the grounded-start
  gate distance and the raycast/aperture scales line up with the live course.
- **Don't over-fit the Infinity-Loop chute.** Keep the grounded-launch fix general across courses.
- **Sim-to-real.** Sim can only show frame-correctness + a clean grounded launch; the real test is a
  live 768 re-fly (Liftoff box). Hold any "works" verdict for that.

## Original Description
(From the Liftoff-box agent, 2026-10-19, 768 re-fly campaign.) Switching the detector to 768 tripled
matched-gate ticks (7–15% → 28–35%) and the detector now finds a gate from the start pad, but all 8
flights at 768 crashed within 2.4 s in the start chute (0/33 gates). Root causes: (1) at 768 the policy
is given the correct gate (dead ahead, ~38 m) while grounded and answers with a full ~4 rad/s roll for
~2 s, rolling into the chute wall at lift-off; (2) at 512 there is no detection on the pad, so the
harness feeds a placeholder gate `[1, 0]` in world coordinates which the input-rotation step turns into
a fake gate ~90° to the side (the start faces −90°), so the 512 flights — including the 33 s "best" —
were steering at a gate that isn't there. Leads to check before retraining the detector: the placeholder
rotated in the wrong frame; whether training ever starts from the ground with a gate in view. Recommended
handing both to a develop run on drone-fly.

## Clarifications
- Q: Dev-team on both leads, or lean/self-iterate? A: **Dev-team on both (rigorous)** — user, 2026-10-19.
- Q: Detector retraining (lever A step 2)? A: Out of scope here; the 768 width switch is enough to expose
  the real blocker. Detector retraining on flight-like frames is a separate, later effort.
