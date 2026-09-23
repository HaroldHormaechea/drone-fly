# Use Case 57: Raise & decouple control frequency (policy Hz vs inner-loop/physics Hz)

## Summary
The env runs the policy control loop at **20 Hz** (`dt=0.05`), physics at ≥240 Hz, and command latency fixed at **2 steps (=100 ms)**. 20 Hz is low for agile acro flight and for a stable inner rate loop (UC-55). This UC raises the policy control rate default to **50 Hz** and **decouples the connectome policy's decision rate from the inner rate-loop + physics rate**: the expensive 25k-neuron policy runs at 50 Hz while the UC-55 rate PID + physics run at the higher physics substep rate — matching both real drones (RC/setpoint rate < FC loop rate) and biology (descending commands slower than the haltere reflex). The task is held **invariant in real-time terms**: episode duration in **seconds** is preserved (max episode steps scales with the control rate) and command **latency is expressed in milliseconds** and converted to steps at the active rate (replacing the fixed 2-step value), so changing Hz never silently alters the task or latency semantics. Control rate, the physics/inner ratio, and latency-ms are exposed and validated via the train YAML (UC-51 pattern). Implies a **fresh run**; couples with UC-55 (rate-loop stability improves at the higher inner rate) and UC-56 (dynamics).

## Acceptance Criteria
1. **Configurable policy rate, default 50 Hz:** the control rate is configurable and defaults to 50 Hz; runs reproducibly at the configured rate.
2. **Decoupled rates:** the policy decision rate is independent of the inner-loop/physics rate; physics + the UC-55 rate PID run at ≥ the policy rate, and the ratio is documented and validated.
3. **Constant episode seconds:** max episode steps scales with the control rate so wall-clock seconds-per-episode is invariant to the Hz change (assert at 20 Hz vs 50 Hz).
4. **Hz-invariant latency:** command latency is specified in ms and converted to steps at the active rate, so effective real-time latency is unchanged across rates (a 100 ms default reproduces the current 2-step latency at 20 Hz).
5. **Curriculum/reward correctness:** timestep-based curricula (UC-51) and any per-step reward terms remain semantically correct at the new rate; any per-step scaling is documented.
6. **YAML exposure:** control rate, physics/inner ratio, and latency-ms are exposed and validated in the train config (UC-51 pattern).
7. Hermetic tests only (rate plumbing, max_steps scaling, latency-ms→steps conversion, decoupling ratio); behavioral verdict deferred to the owner's GPU retrain.

## Potential Pitfalls & Open Questions
- **Risk** — At 50 Hz there are 2.5× more env steps per sim-second, so a fixed `total_timesteps` covers less sim time → likely needs a larger budget (more compute); update the training guidance accordingly.
- **Edge case** — Connectome inference cost rises with policy Hz; decoupling (policy 50 Hz < physics rate) is the mitigation, but confirm the policy isn't the throughput bottleneck at 50 Hz.
- **Coupling** — UC-55 PID gains and UC-56 dynamics interact with the chosen rates; flagged, resolved there.

## Original Description
User request: create a use case for "probably train at more Hz?" — raise the control/simulation frequency above the current 20 Hz. Clarified in this session into raising AND decoupling the three rates in play (policy decision rate, inner rate-loop rate, physics rate), so the expensive connectome policy need not run at the full physics rate.

## Clarifications
- Q: What should the policy control-rate default become (currently 20 Hz)?
  A: 50 Hz.
- Q: Decouple the policy decision rate from the inner rate-loop + physics rate?
  A: Yes — policy at the (lower) control rate; rate PID + physics run faster.
- Q: Hold the task invariant when Hz changes (episode seconds + latency in ms)?
  A: Yes — scale max_steps to keep episode seconds constant, and express latency in ms so it's Hz-invariant.
