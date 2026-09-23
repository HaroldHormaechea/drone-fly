# Use Case 56: Retune drone dynamics to a Meteor75 Pro analog + wide domain randomization

## Summary
drone-fly's nominal plant is the pybullet **CF2X** reference (~0.027 kg, T/W 2.25). This UC retargets the nominal to a **BetaFPV Meteor75 Pro analog** (75 mm-wheelbase 1S brushless whoop, 1102-class motors, 40 mm props, ~30–36 g AUW) — mass, thrust / T-W, rotational inertia, arm length, prop diameter — so acro training (UC-55) runs on a plant matching a real, ownable drone. It **simultaneously widens the domain-randomization envelope** (reusing the UC-48 **T/W-preserving** machinery) to span **whoop → 5" racer** (roughly ~0.03 kg / T-W ~2.5 up to ~0.6–0.7 kg / T-W ~10), so the trained policy is robust across a broad drone range and can later be **fine-tuned to other drones** (incl. Liftoff-class racers). Rotational inertia — absent from spec sheets — is **estimated from a motor-position point-mass model** (four motor masses at the 75 mm-wheelbase quad-X arm positions plus a central body mass), parameterized by arm length and the motor/body mass split so it scales correctly across the envelope. The pybullet CF2X body is reparameterized (mass / inertia / thrust band) rather than swapping URDFs, keeping UC-48 T/W-preservation and the UC-49 observability/regression guard intact but retuned. Exact Meteor75 Pro specs are sourced from an authoritative reference and cited. This changes the plant → a **fresh training run** is required, and it couples with UC-55 (PID gains scale with inertia/T-W) and UC-57 (control Hz).

## Acceptance Criteria
1. **Spec match:** with randomization off, the nominal reproduces sourced Meteor75 Pro-analog specs (mass, T/W, arm length ≈ 26.5 mm from a 75 mm wheelbase, prop diameter 40 mm) within a documented tolerance, with the spec source cited in code/docs.
2. **Deterministic nominal:** randomization-off is deterministic and equals the Meteor75 Pro nominal.
3. **Inertia from point-mass model:** rotational inertia is computed from four motor masses at the quad-X arm positions + a central body mass (parameterized by arm length and mass split), and a hermetic test asserts a known geometry → known inertia diagonal.
4. **Wide T/W-preserving randomization:** the UC-48 mechanism is retuned to a documented **whoop → 5" racer** envelope (ranges for mass, T/W, drag, arm length, …), configurable via the train YAML (UC-51 exposure pattern).
5. **T/W invariance across the envelope:** peak T/W stays consistent under mass randomization end-to-end (UC-48 property holds for the new nominal and across the wide range).
6. **Hover invariant:** throttle 0.5 ≈ hover at the new nominal T/W (or the change is explicitly documented).
7. **Regression guard updated:** the UC-49 end-to-end T/W guard is updated to the new nominal.
8. **Envelope documented:** the range of drones the randomization spans (for later fine-tuning) is written down in code/docs.
9. Hermetic tests only (spec-match, inertia-from-geometry, randomization bounds, T/W invariance, nominal determinism); behavioral verdict deferred to the owner's GPU retrain.

## Potential Pitfalls & Open Questions
- **Risk** — Fresh training run required (plant changed).
- **Risk** — A whoop → 5" envelope is **very wide**; the policy may struggle to master such a broad distribution in one run. Mitigation options (curriculum over the envelope, or start narrow then widen) are left to the training strategy, not this UC.
- **Missing input** — Authoritative Meteor75 Pro spec source; dev-team sources and cites it (exact AUW / thrust figures vary by battery + motor KV — "analog" tolerances accepted).
- **Edge case** — Drag and motor-response lag across a 20× mass range may need per-scale modeling, not a single constant; document what is and isn't modeled.
- **Coupling** — PID gains (UC-55) and control Hz (UC-57) depend on these dynamics; flagged, resolved there.

## Original Description
User request: create a use case to "adjust dynamics." Clarified intent: retune the simulated drone to a **Meteor75 Pro analog** and **also randomize** the dynamics so the policy can later be **fine-tuned to other drones**. Randomization envelope chosen **wide (whoop → 5" racer)**. Rotational inertia to be **estimated from motor positions**. Context: this session established that the current CF2X dynamics differ substantially from real FPV drones, which matters for the eventual Liftoff/real-drone transfer goal.

## Clarifications
- Q: Which Meteor75 variant is the target nominal?
  A: A Meteor75 Pro analog (1102 brushless 1S whoop class).
- Q: How wide should the domain-randomization envelope be?
  A: Wide — whoop → 5" racer (broad T/W + mass range), for robustness and later fine-tuning.
- Q: How should rotational inertia be modeled (not on spec sheets)?
  A: Estimate from motor positions (point masses at the quad-X arm positions + a central body mass).
