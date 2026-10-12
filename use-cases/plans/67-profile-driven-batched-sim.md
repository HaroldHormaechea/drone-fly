---
plan_for: use-cases/67-profile-driven-batched-sim.md
work_branch: feat/uc-67-profile-driven-batched-sim
run: drone-fly-uc-67
approved: 2026-10-12
---

# UC67 — approved implementation plan

Approved by challenger-67 on round 1 (v1): no Critical or Major issues. Its 11 Minor conditions are folded in and marked [C#].

TARGET_DIR = C:\dev\drone-fly-uc-67-profile-driven-batched-sim (written T\ below).
- Use case: T\use-cases\67-profile-driven-batched-sim.md.
- The brief lists `profiles: []`, so no profile skills apply.
- paths.production is src/drone_fly/**; paths.test is tests/**.
- Tooling: pytest, ruff (line length 100), Python 3.11. No new dependencies.

## Analysis

1. **Prototype lineage.**
   - hover_gpu.BatchedDrone has no lag, drag or latency. Each step:
     - omega += 18*(cmd-omega)*0.02;
     - a quaternion Euler step plus normalize, using the NEW omega;
     - thrust = clip((a0+1)/2,0,1)*TW*M*G;
     - then vel, then pos.
     Done envs are auto-reset through _perturb.
   - gates_dr adds motor lag (alpha = min(DT/tau, 1)), quadratic drag on the OLD velocity, a 0-2 step delay on all four channels, and per-episode domain randomization (DR).
   - gates_proc_env.BatchedProcCourse is the 07 env.
     - It inlines the BatchedDrone plant and builds a 1024-course course_gen pool sorted by difficulty.
     - Its PROC_* env-var flags are read at import.
     - Its "ground takeoff" is a spawn at z >= 0.12, with no contact model and a 20-step floor-kill grace.
     - extra() has 30 dims, or 70 with raycast.
2. **The 07 checkpoint.**
   - T\training\07-proc-course-flight\model_readout.pt is tracked in git.
   - It is a K1ReservoirAug(extra_dim=30) trained with PROC_TRACTABLE=1 and PROC_NO_OBS=1.
   - Recorded completion is 96% (185/192, verify_proc.py, AUDIT_SEED 1234), measured on another machine.
   - The actor wraps controller.actor.ConnectomeActorNetwork, which is already in src.
   - The real K1 is gitignored at C:\dev\drone-fly\artifacts\pruned\k1.
3. **Harness access to gpu_prototype.**
   - The harness reaches gpu_prototype only through harness\contract.py load_contract, which overrides the device and K1_PATH.
   - tests\conftest.py already provides:
     - harness_contract, a session-scoped fixture that loads gpu_prototype on CPU with the 322-neuron fixture connectome;
     - harness_checkpoint_30 and harness_checkpoint_70.
   - So the AC3 CPU reference comes from the real prototype code, and no copy is needed.
4. **Harness action path.**
   - policy.decode_action (contract §2) produces ContractCommand(raw, collective, thrust, rates).
   - The gamepad sink maps it through ThrottleMap (anchored at HOVER_THR 0.4 -> hover_stick) and the Betaflight inverse.
   - loopback applies plant_step to the raw action. The overlay reads cmd["collective"].
   - No contract version exists today.
   - Checkpoints are bare state dicts, and the strict=False loader forbids unexpected keys, so metadata cannot be embedded in the .pt.
5. **UC66 profiles are hover-relative.**
   - They carry a throttle->g curve, Betaflight rates plus tau_s and delay_s, drag as an acceleration, motor_lag_s, latency, and the RANDOMIZABLE keys.
   - drones\meteor75\drone.yaml is a value-exact port of BatchedDrone/gates_dr.
6. **sysid code.**
   - fit.load_session returns data in the contract frame, with held_out flags and done-only windows.
   - model.py defines the signal chain and DEFAULT_SIGNS.
   - synthetic.generate_session produces hermetic sessions.
   - Micro held-out errors are 0.6% on rate and 0.093 g on vertical acceleration. So first-order + delay is sufficient, and the second-order question is decided by the AC8 data.
   - The sessions are gitignored and live only in the UC66 worktree: ...\artifacts\sysid\liftoff-gremlin-micro\micro-full and ...\liftoff-5inch-vortex-pro-250\vortex-full.
7. **Latency evidence.**
   - Source: C:\dev\drone-fly\artifacts\harness\liftoff\runs\liftoff-20261011T233116352187Z\latency_summary.json (07 checkpoint, liftoff preset).
   - Rate: 50 Hz target, 36.9 Hz effective, 41/116 missed deadlines (35%).
   - Tick period: p50 20.3 ms, p95 40.4 ms.
   - Compute: p50 17.3 ms, p95 25.5 ms, max 31 ms.
   - Telemetry age: p50 7.7 ms, p95 14.4 ms. Profile input_s is 3 ms.
   - Observation to action: about 28 ms p50 and 43 ms p95.
8. **Smoke test.**
   - tests\test_smoke.py imports drone_fly.train and drone_fly.evaluate, which no longer exist, so it fails on main.
   - src\drone_fly\env is the pybullet-era config surface that record/ still uses. It is unrelated to the new sim.

## Proposed Solution

### A. Inventory
- **Promoted:**
  - hover_gpu: quaternion helpers, plant, PPO loop;
  - gates_dr: DR semantics, subsumed by profile randomization;
  - gates_proc_env + course_gen;
  - reservoir_aug + the R/HID constants from gates_k1_reservoir;
  - train_aug_course / eval_aug / verify_proc, as CLI subcommands.
- **Stays where it is:** everything else, including inference.py, which remains the harness's v1 contract.
- **gpu_prototype:** only a pointer section at the top of T\gpu_prototype\README.md changes. It says the directory is frozen history and points to the new locations.

### B. T\src\drone_fly\sim\

Torch only. No module-level device or tensors, no env vars, and **no imports of drone_fly.harness [C10]**.

- **__init__.py:** exports.
- **quat.py:** the batched wxyz helpers ported from hover_gpu, plus an exponential-map quaternion step.
- **plant.py:**
  - **Construction and state.** Plant(profile, n_envs, device, seed, config).
    - It has its own torch.Generator seeded from seed and never uses the global RNG.
    - Per-env parameter tensors come from the profile times the randomizer samples:
      - thrust knots, with a batched bucketize-based interp;
      - Betaflight rc/super/expo computed in torch, mirroring sysid.model.betaflight_rate_rad, without using DroneProfile.rate_profile (which imports the harness) [C10];
      - tau and delay;
      - per-body-axis drag;
      - motor lag;
      - input latency.
    - State: pos, vel, quat, omega, motor state in g, rate filters, the delay line, time, and on_ground.
  - **Inputs.**
    - step(collective_g, rates_rad_s, dt, action_delay_s) is the policy path.
    - step_sticks(throttle_frac, sticks, dt, signs) is the replay path: Betaflight curve for the axes, thrust curve for throttle.
    - Both follow the same order as sysid/model.py: input_s, then per-axis delay, then Betaflight, then first order [C7].
  - **Per-substep physics.** In order, matching BatchedDrone/gates_dr:
    1. Select the command from the delay line.
    2. Clamp collective to the nominal max_g, then multiply by thrust_scale.
    3. Clamp the rate setpoint to the nominal full-stick rate, then multiply by rate_scale. This models the sink inverting the nominal curves on a drone that differs.
    4. First-order motor lag.
    5. First-order per-axis rates.
    6. Attitude.
    7. Body drag on the old velocity, plus gravity.
    8. Semi-implicit vel/pos.
    9. The ground rule.
  - **PlantConfig:**
    - substeps (default set by the AC10 measurement, expected 1-2);
    - integrator;
    - dtype (float32, or float64 for tests);
    - ground_z and rest height;
    - gravity 9.81.
  - **Integrators.**
    - "exact" is the default:
      - exact ZOH alpha = 1-exp(-h/tau);
      - a single intra-substep switch at the delayed-command boundary;
      - an exponential-map attitude step using the substep-mean omega.
    - "euler-legacy":
      - alpha = min(h/tau, 1) (tau = 0 gives alpha = 1);
      - the Euler quaternion step with the new omega.
      - It reproduces BatchedDrone/gates_dr exactly.
  - **Delay semantics [C1].**
    - Throttle delay = timing action delay + latency.input_s.
    - Axis a delay = timing action delay + input_s + rates.a.delay_s.
    - If the profile randomizes latency.control_to_gyro_s, a common-mode offset (sampled c2g minus nominal c2g) is added to all four channels, throttle included.
    - Nominal c2g is never added on its own, because it is derived (input_s + mean delay_s).
    - Consequences:
      - on meteor75 this gives the intended 0-40 ms delay on all four channels;
      - on the micro it stacks with the per-axis delay_s randomization (about ±1 ms).
    - This double randomization is documented explicitly in plant.py and in the ADR.
  - **Delay line [C2].**
    - A ring buffer of K pending commands with effect times.
    - Effect times are made monotonic (effect_k = max(effect_k, effect_{k-1})). So the command in force is always the most recently issued one whose effect time has passed.
    - With zero delay the buffer has no effect.
  - **No airframe literals.**
- **randomization.py:** DynamicsRandomizer(profile, scale, generator).
  - At a masked reset it samples EVERY key in profile.randomization uniformly in the scaled range:
    - lo' = nom + s(lo-nom), hi' = nom + s(hi-nom);
    - s=0 is nominal, s=1 is the profile range;
    - multipliers have nominal 1.
  - Unlisted keys stay nominal.
  - Per-axis drag nominals sample each axis independently.
  - It exposes the sample tensors and the log columns.
- **timing.py:**
  - **TimingConfig:**
    - control_hz: a range or a set, default 20-50 Hz;
    - action_delay_s: default 0-0.060 s;
    - delay_jitter_s: per-tick std, default 0.005, clamped >= 0;
    - skip_prob: default range 0-0.35;
    - a "fixed" mode (one rate, zero delay, zero skip) for the legacy config.
    - The defaults carry comments citing the run folder above.
  - **ControlTiming:** samples per env per episode, then per tick (jitter, skip Bernoulli).
  - **Skip semantics [C3].**
    - On a skip, that env's plant advances by an extra period inside the SAME transition, holding the previous action.
    - This is a per-env dt multiplier of 2 for that step, with no separate policy step.
    - It mirrors the live failure: no new action is computed, and the held action spans about 2 periods.
    - The dt-scaled reward handles the variable dt.
    - The ADR states this.
- **action.py:**
  - **Version and canonical action.** CONTRACT_VERSION = 2. The canonical action is [collective_g, roll_rate, pitch_rate, yaw_rate].
  - **Network normalization** (documented here and in INFERENCE_CONTRACT.md):
    - collective_g = clamp(1 + collective_scale_g*a0, 0, max_g). Network output 0 is hover, and the readout bias initializes at 0.
    - rates = tanh(a1:4) * max_rate_rad_s.
  - **No defaults in sim [C4].**
    - collective_scale_g and max_rate_rad_s have NO defaults in src/drone_fly/sim. They come from the training config and the sidecar.
    - proc-course-dr.yaml sets collective_scale_g 1.0 and max_rate_rad_s 4.0. That equals the legacy semantics, keeps UC68 warm starts affine, and fits every profile envelope.
  - **v1 legacy decoder.**
    - collective_g = ref.thrust_g(clip((a0+1)/2, 0, 1)).
    - rates = tanh(a) * the reference full-stick rate.
    - ref = drones.load("meteor75") at runtime.
    - A torch decoder (for sim) and a numpy decoder (for the harness) are parity-tested.
  - **Sidecar.**
    - "<stem>.contract.json" sits beside the .pt, with {contract_version, action{collective_scale_g, max_rate_rad_s}, extra_dim, actor, drone, run_id}.
    - A missing sidecar means v1.
  - **Warm-start note.** On meteor75, v1 -> v2 is affine on a0 (g = 1.25 a0 + 1.25). This is documented for UC68 warm starts.
- **course_gen.py:** a verbatim numpy port of gpu_prototype\course_gen.py.
- **course.py:** CourseEnv(profile, n_envs, device, seed, config: CourseConfig, plant_config, timing, randomization_scale).
  - **Port.** A port of BatchedProcCourse.
    - Every PROC_* flag becomes a CourseConfig field: tier (tractable/full, tunnels, multilap, variety), no_obs, obs_terminal, shaping, gap, floor, raycast.
    - obs() (12 dims) and extra() (30/70) are unchanged. Rewards are unchanged.
  - **Rate invariance.**
    - Per-step dense terms are multiplied by dt/reward_reference_dt_s.
    - reward_reference_dt_s is a required CourseConfig field with no sim default; the legacy and DR configs set 0.02 [C4].
    - Episode length and floor grace are expressed as time.
  - **Host-sync-free step.** Respawn uses masks.
  - **StartConfig [AC7]:**
    - ground_fraction;
    - an air-height range;
    - arm_delay_s (the configs set 0.5 s = gamepad max(arm_settle_s, arm_press_s));
    - a "legacy" mode for 07 parity.
  - **Ground start.**
    - The drone spawns at rest on the floor at the course start, level.
    - During the arm delay the plant receives idle commands (collective 0, rates 0), whatever the policy outputs. The policy still observes.
  - **Resting contact [C9].**
    - While z <= rest height and the drone cannot lift:
      - z is clamped;
      - vel and omega are zero;
      - attitude is held;
      - the rate-filter and motor states are held at rest, so ground commands do not wind up into a jump at lift-off.
    - Lift-off is defined as z > rest height + eps with vz > 0.
    - The rest height is above the 0.08 floor-kill threshold.
    - Floor contact is non-terminal until the first lift-off; after that the legacy floor crash applies.
    - The ADR documents the simplifications: no tip-over, no sliding, no I-term windup.
  - **Air start.** Thrust state 1 g, level, z drawn from the configured range.
  - **Per-episode log.**
    - At reset: env index, episode id, every dynamics sample, every timing sample, and the start mode.
    - At episode end: return, length, completed, crash cause.
    - Rows go into a GPU buffer that is drained once per PPO iteration.
- **replay.py [AC8]:** replay_session(session_dir, profile), built on sysid.fit.load_session.
  - **Windows.** Every held-out, done window. Aborted windows are skipped.
  - **Plant setup.**
    - The nominal plant, with randomization off, n_envs = number of windows, on CPU in float64.
    - It is initialized from the measured state at window start:
      - vel, quat, omega;
      - motor state = thrust_g of the throttle in force;
      - rate filters = measured omega;
      - the delay line prefilled with the commands before the window.
  - **Replay.** The logged sticks are fed ZOH through step_sticks with DEFAULT_SIGNS, at the command period.
  - **Metrics.**
    - Attitude error is the geodesic angle in degrees; velocity error is |dv| in m/s.
    - Both are taken at the 1 s and 2 s endpoints, plus the mean over each window.
    - Medians are taken over windows.
    - Windows shorter than 2 s are excluded from the 2 s statistic, and the counts are reported [C7].
  - **Gate (micro only).** The median 1 s attitude error must be <= 10 deg and the median 1 s velocity error <= 1 m/s.
  - **Report.**
    - --write-report writes a "Plant replay (UC67)" section between HTML comment markers in drones\<id>\sysid.md, as an idempotent replace.
    - sysid\report.py write_report is changed to preserve that block.
    - It runs for both Liftoff profiles. Only the micro is gated; the vortex is reported as-is.
  - **Failure is a blocker [C7].**
    - If the micro misses the gate, the result is still written to sysid.md.
    - The developer then escalates to main/user before completion.
    - The response would be a second-order term, decided by the data.

### C. T\src\drone_fly\training\
- **config.py:** TrainConfig plus a YAML loader. Fields:
  - drone id, n_envs, budget;
  - lr, roll, epochs, minibatches, ent_coef, log-std schedule;
  - curriculum fractions;
  - randomization_scale, timing, starts, course, plant;
  - action spec;
  - actor + extra_dim;
  - warm_start {checkpoint, src_extra_dim, legacy_v1};
  - seed, k1_path.
- **ppo.py:**
  - The train_aug loop (reservoir feature caching, GAE, clip, annealed log-std, curricula) over a small actor protocol (features/act/value).
  - The plain-MLP port is dropped [C8].
  - It saves stripped readouts (no body.layer.*) that load through inference.load_policy, plus the contract sidecar.
- **run_dir.py:** writes artifacts\training\<drone-id>\<run-id>\ (artifacts/** is gitignored).
  - run-id = UTC timestamp + config name.
  - config.yaml: the resolved snapshot, including contract_version [C rec].
  - drone.yaml: a byte copy of the profile, plus its sha256.
  - checkpoints\: model_readout.pt + .contract.json, plus periodic checkpoints.
  - logs\: train.log, metrics.jsonl, episodes.csv.
- **cli.py:** `drone-fly train run <config.yaml>`, `train eval <checkpoint> ...`, and `train bench`.
  - **--compare-prototype [C6].**
    - Loads gpu_prototype through harness.contract.load_contract(k1_path=...).
    - PROC_TRACTABLE and PROC_NO_OBS are set before gates_proc_env is imported.
    - The comparison is paired on both sides: the same course seeds and ids (sim.course_gen gives identical courses), the deterministic actor mean, the same success definition, the same time limit and the same episode count.
    - It reports the new rate, the same-machine prototype rate, and the historical README 96%.
    - Optionally, for information only and not gated, it also reports 07 under the proc-course-dr defaults.
- **configs (package data):**
  - **T\src\drone_fly\training\configs\meteor75-legacy.yaml:**
    - nominal, randomization 0;
    - fixed 50 Hz, zero delay;
    - legacy start;
    - euler-legacy, substeps 1;
    - tractable + no_obs;
    - v1 decode;
    - reward_reference_dt_s 0.02.
  - **T\src\drone_fly\training\configs\proc-course-dr.yaml** (the template UC68 extends):
    - randomization scale 1;
    - timing defaults;
    - ground/air mix 0.5;
    - v2 action with collective_scale_g 1.0 and max_rate_rad_s 4.0;
    - reward_reference_dt_s 0.02.
- **T\src\drone_fly\cli\main.py:**
  - Adds `train` and `sim` groups with lazy imports, like `sysid`.
  - `drone-fly sim replay-sysid <session-dir> --drone <id> [--write-report]`.
  - Bench lives in exactly one place (train bench).

### D. T\src\drone_fly\controller\reservoir.py
- Contains K1ReservoirAug, with the same parameter and buffer names as the prototype, so model_readout.pt loads with the strict=False / body.layer.*-only rule.
- Also contains the R/HID constants, warm_start, and load_readout(path, extra_dim, k1_path, device).
- The K1 path is explicit, or resolved by harness.contract.resolve_k1_path. It is never hardcoded.
- The collective bias initializes per the action spec:
  - v1 = 2*hover-1, from the reference profile;
  - v2 = 0.

### E. Harness (lands together with the version bump)
- **types.py:**
  - ContractCommand gains collective_g and contract_version.
  - collective and thrust keep their current meaning: the reference-plant fraction and Newtons; for v2 they are derived from collective_g on meteor75.
  - So the overlay, recordings and parity are unchanged.
- **policy.py:**
  - ContractPolicyRunner.load reads the sidecar (none = v1) and logs the decoded version. The version is also recorded in the run meta.
  - decode_action delegates to sim.action's numpy decoders.
  - v1 stays bit-identical.
- **config.py + liftoff\gamepad.py:**
  - A new top-level `drone: <id>` key.
  - **With a drone:**
    - throttle stick = 2*throttle_for_g(collective_g) - 1, clamped, using the profile's inverse thrust curve;
    - RateProfiles come from the profile's rates;
    - rate_scale and stick_signs remain preset settings.
  - **Without a drone:** the legacy rates/hover_stick path, re-expressed on collective_g with the reference max_g. This is equivalent to today's mapping.
  - **Conflict:** `drone` together with controls.rates or hover_stick is a HarnessConfigError.
- **Presets:**
  - liftoff.yaml: `drone: liftoff-gremlin-micro`; drop controls.rates and hover_stick.
  - liftoff_5inch.yaml: `drone: liftoff-5inch-vortex-pro-250` replaces the CALIBRATE placeholders, with a comment that fit_quality is poor.
- **loopback.py:**
  - v1 keeps plant_step on the raw action.
  - v2 is stepped by a single-env sim.Plant (CPU, meteor75 or the config's drone, euler-legacy).
- **Docs:**
  - INFERENCE_CONTRACT.md: a "Contract version" section.
  - New ADR T\docs\adr\0006-profile-driven-batched-sim.md, covering:
    - the plant;
    - delay / common-mode semantics;
    - skip semantics;
    - ground-contact simplifications;
    - action v2;
    - the inventory;
    - the throughput breakdown (plant / timing / course / extra) and the substeps choice;
    - the AC3b, AC8 and AC10 results.
  - drones\README.md and the meteor75 drone.yaml header comment.
  - Harness docs: the `drone` key.

### F. Measurements (the developer runs them; results go in the ADR and the run folders)
- **AC3b.**
  - Command: `drone-fly train eval training\07-proc-course-flight\model_readout.pt --config meteor75-legacy --legacy-v1 --episodes 192 --seed 1234 --compare-prototype --k1-path C:\dev\drone-fly\artifacts\pruned\k1`, on the local GPU.
  - Pass = |new - same-machine prototype| <= 5 pp.
- **AC10.**
  - At 8192 envs, compare the prototype BatchedProcCourse (tractable + no_obs, random actions, step + extra()) with CourseEnv (meteor75, randomization 1, timing defaults, default substeps).
  - Report env-steps/s and the ratio. Target >= 50%.
- **AC8.**
  - Copy the two UC66 sessions into T\artifacts\sysid\<id>\<session>\ (gitignored).
  - Run replay-sysid --write-report for both drones.
  - The orchestrator also preserves them in C:\dev\drone-fly\artifacts\sysid before the UC66 worktree is cleaned up. (Orchestrator note: done on 2026-10-12.)
  - They are never committed.

## Files Affected

### Production (developer-67)
- New: T\src\drone_fly\sim\{__init__,quat,plant,randomization,timing,action,course_gen,course,replay}.py
- New: T\src\drone_fly\training\{__init__,config,ppo,run_dir,cli}.py
- New: T\src\drone_fly\training\configs\{meteor75-legacy,proc-course-dr}.yaml
- New: T\src\drone_fly\controller\reservoir.py
- Modified: T\src\drone_fly\cli\main.py
- Modified: T\src\drone_fly\harness\{types,policy,config,loopback}.py, liftoff\gamepad.py, configs\liftoff.yaml, configs\liftoff_5inch.yaml
- Modified: T\src\drone_fly\sysid\report.py
- Docs:
  - T\gpu_prototype\README.md (pointer only)
  - T\docs\INFERENCE_CONTRACT.md
  - T\docs\adr\0006-profile-driven-batched-sim.md
  - T\drones\README.md
  - T\drones\meteor75\drone.yaml (comment)
  - the generated replay sections in T\drones\liftoff-gremlin-micro\sysid.md and T\drones\liftoff-5inch-vortex-pro-250\sysid.md

### Tests (qa-67) — hermetic, CPU; a GPU is optional
- **test_sim_plant.py:**
  - determinism per seed;
  - zero delay is a no-op;
  - a step command appears exactly after its delay;
  - jitter;
  - monotonic effect times when jitter would reorder [C2];
  - skip = 2x dt holding the action [C3];
  - clamps and scales;
  - drag sign;
  - the stick path uses the Betaflight curve;
  - the common-mode offset is applied to all four channels [C1].
- **test_sim_meteor75_equivalence.py:**
  - Uses the harness_contract fixture.
  - BatchedDrone runs on CPU, with _perturb replaced by a no-op on the instance and its state copied from the plant.
  - Bounded random raw actions go through the v1 decoder for 200 steps.
  - Every component of pos/vel/quat/omega must agree within 1e-4.
  - If float64 is used, set_default_dtype is restored in a finally/fixture [C11].
- **test_sim_randomization.py:**
  - bounds;
  - scale 0 = nominal, scale 1 = range;
  - unlisted keys stay nominal;
  - all listed keys are sampled;
  - log columns.
- **test_sim_timing.py:** period range/set, delay range, jitter clamp, skip statistics, fixed mode.
- **test_sim_ground_start.py:**
  - no motion during arm_delay under a 2 g command;
  - no wind-up at lift-off [C9];
  - stays down under < 1 g;
  - lifts off under > 1 g after the delay;
  - no termination at t = 0;
  - the ground/air mix fraction.
- **test_sim_course.py:**
  - course_gen equals the prototype for several seeds (via import_reference).
  - CourseEnv vs BatchedProcCourse on CPU, from identical injected state and course ids, over a short horizon with no respawn: obs, extra, reward and done must be equal.
  - CourseConfig is built from the flag globals the fixture-imported gates_proc_env actually has, not hardcoded to tractable.
  - The test skips or fails clearly if PROC_* is set in the shell [C5].
- **test_sim_replay.py:**
  - a synthetic session replay gives small errors;
  - the 2 s window exclusion;
  - the report section is written and replaced idempotently, and survives report regeneration.
- **test_sim_action_contract.py, plus updates to test_harness_gamepad.py, test_harness_contract.py and test_harness_loopback_parity.py:**
  - v1 decode is bit-identical to today's; the torch and numpy decoders agree;
  - sidecar present / absent (= v1);
  - inverse thrust-curve sink: 1.0 g gives about -0.058 on the micro, and the mapping is monotone and clamps;
  - rates come from the profile;
  - drone + rates is a config error;
  - the liftoff preset loads.
- **test_training_cli.py:**
  - `drone-fly train --help` lists run/eval/bench.
  - A tiny CPU run (fixture connectome) writes the full run-folder layout, including the sidecar, episodes.csv, and contract_version in config.yaml.
- **test_sim_no_airframe_literals.py [C4]:**
  - A targeted scan of src\drone_fly\sim for the airframe constants M 0.032, TW 2.5, RATE_KP 18, HOVER_THR 0.4, MAX_BODY_RATE 4.0 and DT 0.02.
  - No ad-hoc allowlist. This is possible because those numbers live only in configs and sidecars.
- **test_smoke.py:**
  - The subpackage list becomes connectome, controller, env, sim, training, cli.
  - This fixes a pre-existing failure on main and is flagged as incidental.

## Risks & Considerations
1. **AC10 throughput.**
   - At 8192 envs the GPU is limited by kernel launches.
   - Mitigations: no host syncs, masked respawn, fused per-env params, substeps chosen by measurement, and no reliance on torch.compile (unreliable on Windows).
   - If the target is missed, report it honestly with the breakdown.
2. **AC3a float drift.**
   - Mitigated with bounded actions and float64.
   - Equivalence is claimed only for euler-legacy with substeps 1. The default "exact" integrator intentionally differs.
3. **AC3b.**
   - Needs the GPU and the gitignored K1.
   - The gate is the same-machine paired comparison; the historical 96% is reported alongside.
4. **AC8.**
   - The velocity budget is tight: the micro's held-out vertical error of 0.093 g is up to ~0.9 m/s per second if it is all bias.
   - Missing the gate is a blocker that escalates (second-order term, decided by the data).
   - Frames and signs are reused from sysid, not re-derived.
5. **Sidecar versioning.**
   - A v2 .pt copied without its .contract.json decodes as v1.
   - Mitigations: the decoded version is logged and recorded in the run meta and config.yaml, and the docs say to move the two files together.
6. **Live behaviour of the 07 model changes slightly away from hover.**
   - The measured inverse thrust curve replaces the piecewise map; the two agree at hover (-0.058 vs -0.06).
   - The measured micro rates (rc 0.85, super 0.71) replace the preset's 1.0/0.7. That is about 17% more stick per rad/s, and it is the correct inversion.
7. **Ground start.**
   - There is no "armed" observation.
   - Contact is deliberately simple (no tip-over, no sliding, no I-term windup). Documented in the ADR.
8. **Reward dt scaling.** Legacy configs are identical; returns change for non-50 Hz envs.
9. **Naming overlap.** src\drone_fly\env (pybullet-era) and src\drone_fly\sim coexist; the ADR says which is which.
10. **UC66 sessions.**
    - They live only in the UC66 worktree.
    - Preserve them in C:\dev\drone-fly\artifacts\sysid before cleanup (done), and never commit them.
11. **Out of scope.** Kept extensible but not built:
    - the UC68 multi-track teacher (CourseEnv takes a course source);
    - the UC70 observation noise (no config field added [C8]);
    - moving the harness's v1 path off gpu_prototype;
    - a plain-MLP actor [C8].

## Challenger verdict (challenger-67)
APPROVE (round 1). There are no Critical or Major issues. The Minor items below are binding conditions for implementation and QA, and they are folded in above as [C1]–[C11].

1. **Common-mode delay double-counts latency.** Document it in the ADR and in plant.py. Apply the offset to all four channels, throttle included. Never add nominal c2g.
2. **Monotonic effect times** in the delay line, with a test.
3. **Skip semantics.** A skip advances the plant an extra period while holding the action, inside the same transition (dt x2).
4. **No sim defaults for reward_reference_dt_s or max_rate_rad_s.** They come from the configs and the sidecar. The literal scan is targeted at airframe constants, with no allowlist.
5. **CourseEnv-vs-prototype CPU test.** Match CourseConfig to the module's actual PROC_* globals. Skip or fail clearly if PROC_* is set in the shell.
6. **AC3b is paired.** Use the same course seeds and ids, the deterministic mean, and the same success definition, time limit and episode count. Report the new rate, the same-machine prototype rate and the historical 96 %. proc-course-dr results are optional and for information only.
7. **AC8 micro failure is a blocker.** Write the result to sysid.md, then escalate to main. Exclude windows under 2 s from the 2 s statistic and report the counts. Use the same signal-chain order as sysid/model.py.
8. **Scope.** Drop the MLP actor port and the obs-noise field. Keep the liftoff_5inch preset change, the smoke-test fix (incidental) and the report.py change.
9. **Ground contact.** Hold the filter and motor states at rest while clamped. Lift-off is z > rest + eps with vz > 0. The rest height must be above the 0.08 floor-kill threshold. Document the simplifications.
10. **Dependency direction.** sim must not import harness; it computes the Betaflight curve itself in torch.
11. **Default dtype.** The AC3a test restores torch's default dtype in a finally block or fixture.

Recommendations:
- Record contract_version in the run config.yaml, the sidecar and the harness run meta.
- Record the substeps default and the throughput breakdown in the ADR.
- Preserve the UC66 sysid sessions in C:\dev\drone-fly\artifacts\sysid (done 2026-10-12).
