# Use Case 67: Profile-driven batched simulator with dynamics/latency randomization, hover-relative throttle and ground starts

## Summary
Promote the batched GPU quadcopter simulator from `gpu_prototype/` (`BatchedDrone` in `hover_gpu.py`, plus the course logic in `gates_gpu.py` / `gates_dr.py` / `gates_proc_env.py` / `course_gen.py`) into a tested package, `src/drone_fly/sim/`. Its plant is parameterized by a UC66 **drone profile** instead of hard-coded Meteor75 constants. Every profile parameter is sampled per environment from the profile's `randomization` ranges. Control rate and observation-to-action latency are also randomized, covering the live harness's measured reality: about 20 Hz, jitter, and skipped ticks. The action interface changes so that **collective is expressed in g relative to hover** (1.0 = hover) rather than as a raw stick fraction. The live harness's `VgamepadActionSink` then converts it to stick positions using the active drone's measured thrust curve and Betaflight rates, so one policy action means the same thing on any airframe. Episodes can **start on the ground at rest**, including the arm delay, as Liftoff does. The plant is validated against Liftoff by replaying the UC66 sysid stick logs open-loop. PPO and the training entry points move to `src/drone_fly/training/` behind a `drone-fly train` CLI; `gpu_prototype/` stays as frozen history. This use case delivers the simulator and the interface, not new trained policies (that is UC68).

## Acceptance Criteria
1. **Structure:**
   - The simulator code is in `src/drone_fly/sim/` (`plant.py`, `randomization.py`, `timing.py`, `course.py` or equivalent).
   - The trainer is in `src/drone_fly/training/` (`ppo.py`, `cli.py`).
   - `drone-fly train --help` lists the subcommands.
   - Run outputs go to `artifacts/training/<drone-id>/<run-id>/`: config snapshot, profile snapshot, checkpoints, logs.
   - `gpu_prototype/` is not modified except for a README note pointing to the new home.
2. **Profile-driven plant:**
   - `sim.Plant(profile, n_envs, device, seed)` takes a `DroneProfile`.
   - It implements thrust via the profile's curve, the per-axis rate response (first-order + delay, with the Betaflight rate curve), drag and motor lag.
   - There are no Meteor75 literals left in `src/drone_fly/sim/`.
3. **Meteor75 equivalence:**
   - With `drones/meteor75` and randomization disabled, the new plant matches `gpu_prototype`'s `BatchedDrone` within 1e-4 per state component over 200 steps for identical seeds and actions.
   - The existing `training/07-proc-course-flight` checkpoint still completes its procedural course in the new simulator at the rate it did before (± 5 pp).
4. **Dynamics randomization:**
   - Each environment samples every randomizable parameter uniformly within the profile's `randomization` ranges at reset.
   - A training config can scale the ranges (0 = nominal, 1 = profile ranges).
   - The samples are logged per episode.
5. **Timing randomization:**
   - Per environment, the control period is sampled from a configurable set or range (default 20–50 Hz).
   - The action delay is sampled (default 0–60 ms), with per-tick jitter.
   - A configurable probability of a skipped tick holds the previous action.
   - The defaults cover the live harness's measured latency, which is cited from a UC65/UC66 run folder in the config comments.
6. **Hover-relative throttle:**
   - The canonical action becomes `[collective_g, roll_rate, pitch_rate, yaw_rate]`.
   - `collective_g` is in [0, max_g of the profile], normalized for the network as documented.
   - The harness `ContractCommand` gains the same meaning, and the inference contract version is bumped.
   - `VgamepadActionSink` maps `collective_g → throttle stick` with the active profile's inverse thrust curve, and maps the rates with the profile's Betaflight rates.
   - The `liftoff` preset names the active drone (`drone: liftoff-gremlin-micro`) instead of duplicating rates and hover stick.
   - Old checkpoints (contract v-previous) still load through an adapter that maps their raw throttle fraction, so existing runs keep working.
7. **Ground starts:**
   - The course/env can start episodes on the floor at rest, with motors idle for a configurable arm delay (default from the Liftoff live run: arm + 0.5 s settle).
   - Training configs choose a mix of ground and air starts.
   - A test asserts that the drone does not move before the delay ends, and lifts off under a > 1 g command.
8. **Liftoff replay validation:**
   - `drone-fly sim replay-sysid <session-dir> --drone <id>` feeds the logged sticks into the nominal plant from the same initial state.
   - It reports attitude and velocity error over 1 s and 2 s windows.
   - On the UC66 held-out segments, the median 1-second attitude error is ≤ 10° and the velocity error is ≤ 1 m/s for the micro profile. The result is written into `drones/<id>/sysid.md`.
9. **Tests:** hermetic CPU tests cover plant determinism per seed, randomization bounds, timing/delay semantics, the ground-start behaviour, the inverse thrust-curve sink mapping, and Meteor75 equivalence (a reduced env count is fine). A GPU is optional.
10. **Throughput:** the promoted simulator keeps ≥ 50% of `gpu_prototype`'s env-steps/s on the local GPU at 8192 envs. Measured and reported.

## Potential Pitfalls & Open Questions
- **Risk** — Changing the action meaning breaks every existing checkpoint and the live harness contract. The compatibility adapter (AC6) and the contract version bump must land together.
- **Assumption** — First-order rate response + delay is enough to mimic Betaflight's PID loop at policy rates (20–50 Hz). If the UC66 fit residuals say otherwise, add a second-order term. That is decided by the data, not up front.
- **Edge case** — Ground contact: the current batched simulator has a floor kill or penalty. Ground starts need a resting-contact model that doesn't terminate the episode at t = 0.
- **Ambiguity** — Which `gpu_prototype` files are canonical. The team inventories them and promotes only what UC68 needs (plant, gate course, procedural generator, PPO, the reservoir/readout actor used by the 07 model). The rest stays in place.
- **Risk** — Promoting the connectome/reservoir actor may pull in `gpu_prototype`-only dependencies. Keep the actor in `src/drone_fly/controller/` if it already lives there.

## Original Description
> Ok we'll have to go the next step. Despite having "memorized" the course as you mentioned in a previous UC, we really aren't flying well. The AI just tumbles right from the start, with throttle/yaw/roll/pitch input from the start.
>
> We do have the possibility of exporting the track geometry for further trainings by converting it to pybullet, but the thing is, I want to GENERALIZE this so it can tackle random courses in liftoff.
>
> Ideas?

> Prepare everything as use cases. Once they are set-up, and we validate them together, wait for me to give you the go ahead (I'll be out of home for a couple hours so that's good time to use, but i'll leave after we have set this up). When doing the analysis, make sure whateve rwe create fits in a good understandable folder structure, and is scalable in case we need to train more drones and such.

## Clarifications
- Q: The batched GPU simulator lives in `gpu_prototype/` as one-off scripts. Promote it?
  A: Promote it to `src/drone_fly/sim` (plus `src/drone_fly/training`, with a `drone-fly train` CLI); `gpu_prototype` stays as frozen history.
- Q: Where should drone and track definitions live?
  A: Top-level `drones/` + `tracks/`; outputs under `artifacts/{sysid,training,eval}/<drone>/…`.
