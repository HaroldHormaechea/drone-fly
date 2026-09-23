# Use Case 55: Inner-loop body-rate controller (acro flight controller)

## Summary
drone-fly's CTBR action `[throttle, roll, pitch, yaw]` (collective thrust + body-**rate** commands) is mapped to motors by a **static open-loop feedforward mix** (`ctbr_to_rpm` in `src/drone_fly/adapter/pybullet_adapter.py`) with **no gyro feedback** — commanded rates are never regulated toward achieved rates, so any command dumps fixed differential torque and the body tumbles (proven this session: a null command `[0.5,0,0,0]` hovers with zero horizontal drift under both native and randomized mass, but full-range noisy rate commands flip the drone 180° and crash it within ~1 s across every seed). This is "acro without a flight controller" — a rung below real acro, and the structural root cause behind ~15 failed-takeoff use-cases. This UC adds the missing **inner-loop rate PID** in a **new dedicated controller module** the adapter calls: it reads measured body angular velocity (`raw[13:16]`) and drives the four motors so *commanded rate ≈ achieved rate*, with the normalized command mapped (linearly, via a **parameterizable curve hook** left in place for future sim-transfer) to a body-rate setpoint clamped to `max_body_rate` (~3.4–4.0 rad/s, already a parameter). Throttle passes through untouched. PID gains are **documented defaults exposed as config overrides**. It deliberately does **not** add auto-level / attitude-angle hold — the policy keeps full acro agency (flips, inverted flight, arbitrary attitudes); the loop only stabilizes the plant against command noise, mirroring the haltere rate-reflex sitting beneath the descending connectome commands. The now-redundant **attitude-authority curriculum is retired** (removing the iter-81 anneal landmine). The loop applies to the **pybullet backend only**; the numpy `SimpleDroneAdapter` (CI backend) is left byte-identical so CI stays hermetic. Because this changes the control problem, a **fresh training run is required** (policies trained on the unstabilized plant will not transfer).

## Acceptance Criteria
1. **Rate tracking:** given a fixed roll/pitch/yaw setpoint, the measured body rate converges to within a small tolerance of the commanded rate within a bounded time (hermetic test against the pybullet sandbox or a unit-level plant model).
2. **No tumble under noise:** under sustained full-range noisy rate commands (the diffuse-policy regime), peak tilt stays bounded well below 180° and the drone does not crash within the probe horizon — contrast the pre-change ~180°/≈1 s baseline.
3. **Setpoint clamp:** a normalized command of ±1 maps to ±`max_body_rate`; values beyond ±1 are clipped and never exceed the clamp.
4. **Throttle untouched:** for a pure-throttle command (rates = 0) the motor outputs and resulting hover behavior are byte-identical to a documented reference.
5. **Null-command hover:** `[0.5,0,0,0]` still hovers with zero horizontal drift and zero attitude (regression guard for the property the probe established).
6. **Rate, not angle:** a sustained nonzero rate command keeps the attitude changing (the loop holds the *rate*, not an angle) — verifies acro mode, not self-level.
7. **Authority curriculum retired:** the `a[1:4] *= attitude_authority` scaling (`racing_env.py`) and its curriculum stage/config keys are removed; no remaining code path scales the rate channels by an authority factor.
8. **Config-overridable gains:** PID gains have documented defaults and are overridable via config without code edits.
9. **numpy backend unchanged:** `SimpleDroneAdapter` behavior is byte-identical (CI hermeticity preserved); the rate loop is pybullet-only.
10. **Parameterizable curve hook:** the command→setpoint mapping is linear by default but routed through a hook a future nonlinear (Betaflight/Liftoff-style) rates curve can replace without touching the policy.
11. All new tests are hermetic (no GPU, no training); the behavioral flight verdict is out of scope and deferred to the owner's GPU retrain.

## Potential Pitfalls & Open Questions
- **Edge case** — PID stability depends on the effective inner control rate (dt = 0.05 with `pyb_freq` substeps); this **couples with UC #3 (train at higher Hz)** — flagged here, resolved there, not absorbed into this UC.
- **Assumption** — Retiring the attitude-authority curriculum leaves its config keys vestigial; this UC removes/deprecates them cleanly rather than leaving dead config behind.

## Original Description
Add the missing inner-loop rate controller ("flight controller") between the connectome policy's CTBR axis commands and the pybullet motor mixer, so drone-fly flies true ACRO mode. Today the action is [throttle, roll, pitch, yaw] (collective thrust + body-RATE commands) but ctbr_to_rpm is a STATIC open-loop feedforward mix with NO gyro feedback — a commanded body rate is never regulated to the achieved rate, so any command noise dumps raw torque and the body tumbles (empirically proven this session: a null command hovers perfectly with zero drift, but full-strength noisy rate commands flip the drone 180° and crash it within ~1s across all seeds). This is "acro without a flight controller" — a rung BELOW real acro, and the root cause of 15+ use-cases of no-takeoff.

Scope: add a rate PID (inner loop) that reads measured body angular velocity (pybullet exposes it at raw[13:16]) and drives the four motors so commanded rate = achieved rate, with the command in [-1,1] mapping to a body-rate setpoint clamped to max_body_rate (~3.4-4.0 rad/s, already a parameter). Keep throttle passthrough (index 0 untouched). Do NOT add auto-level / attitude-angle hold — the policy must retain full acro agency (can flip, roll inverted, hold any attitude by commanding rates); the rate loop only damps the PLANT so noise doesn't tumble it, it never picks an attitude. Biologically this mirrors the haltere reflex (peripheral rate feedback) sitting below the descending connectome commands.

Interactions to decide in clarification: (a) the attitude-authority curriculum (racing_env.py:503 a[1:4] *= authority) is a crude stand-in for this missing loop — retire it or repurpose it to ramp the rate-SETPOINT range? (b) make the command->rate curve parameterizable (linear now, but leave a hook) so a later Liftoff/Betaflight-style nonlinear rates curve can be matched — designed-for-transfer. (c) PID gain source: hand-tuned constants vs derived from the CF2X inertia. (d) where the loop lives: inside PyBulletAdapter/ctbr_to_rpm vs a new controller module. Confirm this means a FRESH training run (a policy trained on the unstabilized plant won't transfer to the stabilized one). Hermetic tests only (rate-tracking convergence, no-tumble-under-noise, throttle-untouched byte-identity, setpoint clamp) — behavioral verdict is the user's GPU retrain.

## Clarifications
- Q: Fate of the attitude-authority curriculum once the rate loop exists?
  A: Retire it entirely (the loop makes it obsolete; removes the iter-81 anneal landmine).
- Q: Where should the rate controller live?
  A: A new dedicated controller module the adapter calls (separable, unit-testable, reusable by a future Liftoff adapter).
- Q: How should PID gains be sourced?
  A: Documented default gains exposed as config overrides (retunable without code edits).
- Q: Fresh-run acknowledgment + should the numpy CI backend get the loop too?
  A: Fresh run acknowledged; apply the rate loop to the pybullet backend only, leaving SimpleDroneAdapter byte-identical for CI hermeticity.
