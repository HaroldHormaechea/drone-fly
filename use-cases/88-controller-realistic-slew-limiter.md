# Use Case 88: Controller-realistic command slew-rate limiter (action filter)

## Summary
Add a **slew-rate limiter (action filter)** to the policy's command output so the commanded sticks
cannot move faster than a real transmitter/pilot can physically produce. UC-87 made the policy much
smoother in sim (live-like command slew 14.9 vs model 11's 26.7 rad/s·s⁻¹, >5 Hz power 6.3% vs 9.9%,
yaw-flips 1.79 vs 7.11), **but the live Liftoff recording is still erratic** — the user's observation:
"it moves the stick faster than usually possible with a controller." Root cause: there is **no
slew-rate limiter anywhere in the command path** (`policy.act → decode_action → sink.send`, raw), so
the memoryless/high-rate policy emits instantaneous setpoint jumps every tick with no cap on how fast
the command moves, and the sim's soft action-rate *penalty* does not hard-limit slew (and the sim
rate-loop model doesn't perfectly match real Betaflight). The original UC-87 analysis explicitly noted
model 11 had "no action-rate penalty **and no action filter**"; UC-87 added the penalty, this use case
adds the filter.

The limiter must be applied **both at deployment (harness) and in training (sim)** with identical
semantics, so the policy learns to fly within the cap instead of the cap lagging its intent.

## Acceptance Criteria
1. **Slew-rate limiter (shared implementation).** A stateful, per-axis limiter clamps the change in the
   applied command between consecutive policy steps to at most `max_slew · dt`, where `dt` is the
   policy-step period. It applies to all commanded axes (the 3 body-rate setpoints + collective).
   Starting default: **full command-range traversal in ~0.25 s** (configurable; tunable after live
   flight). The same code/semantics are used in the sim action path and the harness command path.
2. **Deployment (harness).** The limiter runs in the harness command path (after `decode_action`,
   before `sink.send`), stateful across the attempt, reset on attempt start / telemetry discontinuity
   (same boundaries as `SpawnYawAligner` / the memory ring).
3. **Training (sim).** The same limiter is applied in the sim so the policy trains within the cap
   (`CourseEnv` action application / plant input). The policy therefore learns control that respects
   the slew limit rather than fighting it.
4. **Memory parity.** The policy's `prev_action` memory input must reflect the **actually-applied
   (post-limiter) command**, not the raw pre-limiter output, in BOTH sim and harness — so the policy
   observes what the drone really received and the sim↔harness contract stays consistent.
5. **Config + logging.** The slew cap is a configurable field (per-axis or shared), recorded in the run
   config and the deliverable; default-off preserves byte-identical behaviour for existing configs so
   old checkpoints/configs are unaffected.
6. **Retrain.** Retrain the UC-87 warm-start recipe (`uc87-deploy-robust-warmstart`) with the limiter
   in-loop → a new deployable checkpoint. The smoothness eval shows the commanded slew is **hard-capped
   at the controller-realistic value** (no >cap excursions), high-frequency (>5 Hz) power is further
   reduced vs UC-87, and course completion is preserved (within a few points of UC-87's live-like).
7. **Tests (CPU-hermetic).** The limiter clamps slew correctly (including at the cap and at zero);
   sim↔harness parity of the limiter + the post-limiter `prev_action`; default-off is byte-identical;
   boundary reset behaves. 

## Potential Pitfalls & Open Questions
- **Completion vs restriction.** Too tight a cap can make the policy sluggish and cost completion;
  0.25 s is a starting point — report the completion impact and keep the cap configurable.
- **Where "full range" is defined.** The raw action is `tanh(a₁:₄)·max_rate_rad_s` (rates) +
  `collective`; define the slew in a consistent space (e.g. the applied body-rate setpoint span
  `2·max_rate_rad_s`, and the collective's g-range) and document it.
- **prev_action semantics (AC4) are the crux.** If the memory still sees the raw (pre-limiter) action,
  the policy can't learn to respect the cap and the live/sim contract diverges — the post-limiter value
  must be what's fed back and recorded.
- **Deployment-only fallback.** The limiter is useful on the existing UC-87 checkpoint at deployment
  even before retraining (a quick live fly-test), though training-with-limiter is the robust fix.
- **Sim-to-real.** The real test is a Liftoff flight; the sim can only show the slew is capped and
  completion holds. Live validation happens on the Liftoff box.

## Original Description
(Authored 2026-10-19 from the user.) After UC-87 shipped, the live Liftoff recording of the new policy
was still erratic. The user's diagnosis: "I think it moves the stick faster than usually possible with
a controller." Investigation confirmed no slew-rate limiter exists in the command path. Fix: a
controller-realistic slew-rate limiter (action filter) in both the harness and training, then retrain.

## Clarifications
- Q: Deploy-side only, or train with it too? A: Full dev-team — limiter in harness AND training, then
  retrain (user decision).
- Q: Starting slew cap? A: Full command-range traversal in ~0.25 s (gentle, clean lines); configurable
  and tunable after flying.
