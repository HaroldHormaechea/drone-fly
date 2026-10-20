# Use Case 90: Gate-approach altitude tracking (stop climbing over the gate)

## Summary
UC-89 solved the grounded launch — live (768 detector, 0.5 s cap), the deploy-robust policy now leaves
the start chute and flies 135–318 m toward gate 0 on every attempt (vs model 13 rolling into the chute
wall in < 2.4 s). **But it still passes 0/33 gates: it climbs OVER gate 0 instead of through it.**
Telemetry from the best live flight (uc-89-results, attempt 5) shows the drone flies a **median +11 m
above the perceived gate-0 altitude** on approach — the grounded-launch training taught it to climb
hard off the pad, but nothing teaches it to **level off at the current gate's altitude** as it arrives.
(The gate-0 altitude perception is also noisy — detector z median ~1.7 m, stdev 6.5 m, spikes to +60 m,
since monocular depth is poor at range — which compounds the problem but is secondary.)

Add a **gate-approach altitude-tracking reward** to `CourseEnv` so the policy aligns its altitude with
the current target gate's altitude as it approaches (reducing the vertical overshoot), then retrain.
Default-off ⇒ byte-identical; CPU-tested; the retrain + live 768 re-fly are the orchestrator's follow-up.

In sim with perfect gate z the grounded eval already completes ~13.3 %, so part of the live 0 % is the
noisy live z + real climb dynamics (sim-to-real); the altitude reward targets the dominant, controllable
part (the drone not settling to gate height), with an optional gate-z-noise robustness knob for the rest.

## Acceptance Criteria
1. **Approach altitude-tracking reward (shaping).** A new, configurable reward term in
   `src/drone_fly/sim/course.py` that rewards the drone for aligning its altitude (z) with the **current
   target gate's** altitude as it approaches — e.g. a penalty on the vertical error `|drone_z − gate_z|`
   gated by horizontal proximity to the gate (so it bites on the approach, not across the whole course).
   It must key on the **current target gate** (generalizes to all gates, not just gate 0), using the
   same gate the progress/pass logic uses. Configurable weight (and any proximity window) on
   `CourseConfig`; **default 0 ⇒ byte-identical** to current training.
2. **Vertical-alignment specific.** The term must reward *alignment to the gate's z*, not a generic
   low/high-altitude bias — being at the gate's height is rewarded; being above OR below it is penalized
   symmetrically (or justified if asymmetric). It must not merely duplicate the existing
   progress/clearance/floor shaping (no double-counting) — document how it composes with them.
3. **No regression.** The term, at its intended weight, must not regress UC-89's grounded-launch
   completion (~13.3 % grounded, sim) or the airborne completion, and must not reintroduce the cruise
   roughness the deploy work removed. The aim is to *raise* the gate-pass rate by cutting the vertical
   overshoot.
4. **(Optional, secondary) gate-z-noise robustness.** A default-off knob to perturb the *observed*
   gate z (domain randomization) during training so the policy tolerates the noisy live estimate. Mark
   clearly optional — the altitude reward is the primary deliverable; include this only if it is cheap
   and does not destabilize training. Default-off ⇒ byte-identical.
5. **Config + logging.** The new weight/knobs are configurable `CourseConfig` fields with validation
   and `from_dict` passthrough, recorded in the run config; default-off preserves byte-identical
   behaviour for existing configs and old checkpoints.
6. **Tests (CPU-hermetic).** The altitude reward is maximal when the drone is at the gate's z and
   decreases as vertical error grows; the proximity gating behaves (no/low contribution when far,
   active on approach); it keys on the current target gate (advances with the gate index); default-off
   is byte-identical (reward + `extra()` + episode return unchanged); if the DR knob is added, its
   default-off parity holds.
7. **Retrain (orchestrator GPU step).** Retrain the UC-89 grounded-launch recipe
   (`uc89-grounded-launch`) with the altitude-tracking reward on → a checkpoint whose sim approach
   shows the drone settling toward the gate's altitude (reduced +Δz overshoot) while keeping the
   grounded launch and airborne completion. Then a live 768 re-fly (the real proof). Dev-team delivers
   code + CPU tests only.

## Potential Pitfalls & Open Questions
- **Over-strong shaping distorts the trajectory.** Too heavy an altitude penalty can cause dive-bombing,
  stalls, or hurt completion — keep it proximity-gated and tunable; the retrain picks the weight.
- **Perfect-z in sim vs noisy-z live.** The reward uses the true gate z in sim; the altitude term alone
  won't fix the *perception* noise (AC4 / a later perception fix addresses that). Call this out.
- **Don't double-count** with the existing progress/clearance/floor rewards — compose cleanly and
  document.
- **Generalize across gates** — key on the current target gate, not a hardcoded gate 0.
- **Interaction with the grounded launch** — the launch *wants* an initial climb; the altitude term
  must not fight the launch off the pad (proximity gating to the gate should naturally avoid this, since
  gate 0 is ahead at ~3.2, not overhead). Verify the launch still works.
- **Sim-to-real.** Sim can only show reduced overshoot + held completion; the real proof is the live
  768 re-fly.

## Original Description
(Authored 2026-10-19 from the live UC-89 768 re-fly.) After UC-89's grounded-launch fix, the drone
flies to gate 0 but passes 7–13 m too high — it climbs over the gate. Telemetry: median +11 m above the
perceived gate-0 altitude on approach; the policy climbs off the pad but never levels to gate height.
Fix: a gate-approach altitude-tracking reward so it settles to the current gate's altitude, then retrain.

## Clarifications
- Q: Dev-team + retrain, or perception-first? A: **Dev-team + retrain** (user, 2026-10-19) — the
  dominant, controllable cause is the policy not settling to gate height; the noisy-z perception is
  secondary (optional AC4 / later).
- Q: Scope of the reward? A: An approach-altitude-tracking shaping term keyed on the current target
  gate, proximity-gated, default-off; optional gate-z-noise DR as a secondary robustness knob.
