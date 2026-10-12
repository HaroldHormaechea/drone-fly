# ADR 0006: Profile-driven batched simulator, action contract v2, ground starts

- **Status:** accepted (UC-67)
- **Date:** 2026-10-12
- **Builds on:** [ADR 0005](0005-drone-profiles-and-sysid.md) (drone profiles, sysid),
  [ADR 0002](0002-live-harness-layering.md) (harness parts),
  [INFERENCE_CONTRACT.md](../INFERENCE_CONTRACT.md) (obs / action / file formats)
- **Code:** `src/drone_fly/sim/` (`plant`, `randomization`, `timing`, `action`, `course`,
  `course_gen`, `replay`, `quat`), `src/drone_fly/training/` (`config`, `ppo`, `run_dir`, `cli`,
  `configs/`), `src/drone_fly/controller/reservoir.py`, harness `types` / `policy` / `config` /
  `loopback` / `liftoff/gamepad`
- **Data:** `artifacts/training/<drone-id>/<run-id>/`, `artifacts/eval/` (gitignored)

## Context

The policies fly well in `gpu_prototype/` and tumble in Liftoff. The prototype's plant is a
hard-coded Meteor75 analog: thrust is a fraction of a fixed T/W, rates are linear up to 4 rad/s, the
response is instantaneous, and the action arrives every 20 ms with no delay. UC66 measured what
Liftoff's drones actually do (`drones/<id>/drone.yaml`), and the live harness measured its own
timing: about 37 Hz effective with 35 % missed deadlines, and 28-43 ms from observation to action.
Live episodes also start on the ground, which the prototype never does.

UC67 promotes the simulator into a tested package parameterised by a profile, randomizes the
dynamics and the timing, changes the action so it means the same thing on any airframe, and
validates the plant against Liftoff. Training new policies is UC68.

## Decision

### Inventory: what moved, what stayed

| prototype | now | notes |
|---|---|---|
| `hover_gpu.py` quaternion helpers, `BatchedDrone` plant | `sim/quat.py`, `sim/plant.py` | the plant is profile-driven |
| `gates_dr.py` (lag, drag, latency, DR) | `sim/plant.py` + `sim/randomization.py` | subsumed by the profile's `randomization` block |
| `gates_proc_env.BatchedProcCourse` | `sim/course.py` `CourseEnv` | `PROC_*` env vars -> `CourseConfig` fields |
| `course_gen.py` | `sim/course_gen.py` | verbatim (identical courses per seed) |
| `reservoir_aug.py` + `R`/`HID` of `gates_k1_reservoir.py` | `controller/reservoir.py` | same parameter names: every shipped readout loads |
| `train_aug_course.py` / `eval_aug.py` / `verify_proc.py` | `drone-fly train run / eval` | |
| `hover_gpu.train` + `reservoir_aug.train_aug` | `training/ppo.py` | the plain-MLP actor was not ported |

Everything else stays in `gpu_prototype/`, which is frozen history (its README points here).
`gpu_prototype/inference.py` remains the harness's v1 contract reference (`load_policy`,
`plant_step`, `raycast`); moving the harness off it is out of scope.

`src/drone_fly/sim` (this simulator) and `src/drone_fly/env` (the pybullet-era config surface still
used by `record/`) coexist; `env` is not the simulator.

### The plant

`sim.Plant(profile, n_envs, device, seed, config, randomization_scale)` is torch only. It has no
module-level device or tensors, no environment variables, no import of `drone_fly.harness`, and no
airframe constant: every number comes from the profile, the randomization samples or the config.
Each plant has its own seeded `torch.Generator`.

The signal chain is the order of `sysid/model.py`: input delay, then per-axis delay, then the
Betaflight curve (stick path only), then a first-order response. Each substep:

1. select the command in force from the delay line;
2. clamp the collective to the profile's NOMINAL `max_g`, then multiply by `thrust_scale`;
3. clamp the rate setpoint to the NOMINAL full-stick rate, then multiply by `rate_scale` (the live
   sink inverts the nominal curves, so a different airframe sees a scaled response);
4. first-order motor lag;
5. first-order per-axis rate response;
6. attitude;
7. body-axis drag on the old velocity, plus gravity;
8. semi-implicit velocity / position;
9. the ground rule.

Two integrators. **`exact`** is the default: it uses exact zero-order-hold first-order responses
(`1 - exp(-h/tau)`), one intra-substep switch at the delayed command's effect time (so a delay is
resolved exactly, not to the substep), and an exponential-map attitude step with the substep-mean
body rate and the mean thrust. **`euler-legacy`** uses `alpha = min(h/tau, 1)`, the command at the
substep start, the first-order quaternion step with the new rate and the new thrust. With one
substep, `euler-legacy` reproduces `BatchedDrone` / `gates_dr` exactly: the measured maximum
difference is 4e-14 over 200 steps, in float64.

**Delay semantics.** A command pushed at `t` takes effect on the throttle at
`t + action_delay + input_s` and on axis `a` at `t + action_delay + input_s + delay_s(a)`. If the
profile randomizes `latency.control_to_gyro_s`, the common-mode offset `sampled c2g - nominal c2g`
is added to **all four** channels, throttle included. The nominal c2g is never added on its own,
because it is derived (`input_s + mean(delay_s)`). On `meteor75` (all delays 0, c2g randomized
0-40 ms) this gives `gates_dr`'s 0-2 step latency on all four channels. On a sysid profile it stacks
with the per-axis `delay_s` randomization (about +-1 ms on the micro). This double randomization is
deliberate and small.

**Delay line.** A ring of the last `K` commands with per-channel effect times. Effect times are made
monotonic in push order (`effect_k = max(effect_k, effect_{k-1})`), so the command in force is
always the newest one whose time has passed: jitter never lets an old command override a newer
one. With zero delay the line has no effect. `K` is sized from the longest delay over the shortest
step.

### Randomization (AC4)

At every (masked) reset each env draws **every** key listed in the profile's `randomization`
uniformly from the range scaled about the nominal: `lo' = nom + s (lo - nom)`,
`hi' = nom + s (hi - nom)`. `s = 0` is nominal, `s = 1` is the profile range, and multipliers have
nominal 1. Unlisted keys stay nominal. A per-axis drag nominal draws each axis independently.

All 18 samples are one `(n, 18)` table drawn with one `rand`. The table is logged per episode
(`episodes.csv`) with the timing samples, the start mode and the outcome.

### Timing (AC5) and skip semantics

Per env and episode, `sim/timing.py` draws:

- a control rate: 20-50 Hz, or a set of rates;
- a base action delay: 0-60 ms;
- a skip probability: 0-35 %.

Per tick it draws the delay plus 5 ms Gaussian jitter (clamped at 0) and a Bernoulli skip. The
defaults cite `artifacts/harness/liftoff/runs/liftoff-20261011T233116352187Z/latency_summary.json`
(the 07 checkpoint on the `liftoff` preset):

- rate: 50 Hz target, 36.9 Hz effective;
- 41/116 missed deadlines;
- tick period: p50 20.3 ms, p95 40.4 ms;
- compute: p50 17.3 ms, p95 25.5 ms;
- telemetry age: p50 7.7 ms, p95 14.4 ms.

A config can also choose a `fixed` mode: one rate, no delay, no skip.

**A skipped tick** mirrors the live failure: no new action arrives in time, so the action in force
is held for about two periods. In the env that is one transition whose plant time is `2 * period`
(a per-env dt multiplier of 2), not an extra policy step.

### The course env

`CourseEnv` is a port of `BatchedProcCourse`: the same pool (sorted by difficulty), `obs()` (12),
`extra()` (30, or 70 with the raycast), pass rule, contact nudge, lookahead, raycast and reward
terms. On CPU, from identical injected state, it matches the prototype to float32 roundoff (3e-6)
on obs, extra (raycast included) and reward.

- **Rate invariance.** The dense per-step terms (upright, low spin, obstacle contact, gap steering)
  are multiplied by `dt / reward_reference_dt_s`. The progress and shaping potential differences,
  and the gate, completion and crash events, are not. `reward_reference_dt_s` has no sim default;
  the configs set 0.02 s. Episode length (28 s) and the floor-kill grace (0.4 s) are times: the
  prototype's 1400 and 20 steps.
- **No host synchronisation in `step()`.** Respawn, contact and the completion metric are masked
  device ops. The episode log is a device buffer drained once per PPO iteration.

### Ground starts (AC7) and their simplifications

`StartConfig.mode: mixed` draws a ground start with probability `ground_fraction`.

- **Ground start.** The drone spawns level and at rest on the floor at the course start, at
  `rest_height_m = 0.12` (above the 0.08 m floor kill). For `arm_delay_s` (0.5 s in the configs:
  the gamepad's `max(arm_settle_s, arm_press_s)`) the plant receives idle commands (0 g, 0 rad/s)
  whatever the policy outputs. The policy still observes.
- **Air start.** The drone spawns level, with motor state 1 g and z in `air_z_m`.
- **Legacy start.** `mode: legacy` reproduces the prototype spawn.

While a grounded drone cannot lift off:

- its height is clamped and its velocity and body rates are zero;
- its attitude is held;
- its rate response stays at rest;
- its motor state is capped at the thrust that would lift it.

So ground commands do not wind up into a jump. Lift-off is `z > rest + eps` with `vz > 0`. Floor
contact is non-terminal until the first lift-off; after it the legacy floor crash applies.

Deliberately simple: no tip-over, no sliding, no I-term wind-up, no "armed" observation.

### Action contract v2 (AC6)

The canonical action is `[collective_g, roll_rate, pitch_rate, yaw_rate]`. `collective_g` is body-z
specific force in g, so 1.0 is hover on any airframe. The network output `a` maps as
`collective_g = clamp(1 + collective_scale_g * a0, 0, max_g)` and
`rates = tanh(a1:4) * max_rate_rad_s`. Network output 0 is hover, and the readout's collective bias
starts at 0. `collective_scale_g` and `max_rate_rad_s` have no default in `sim`: they come from the
training config and are written to the checkpoint's sidecar `<stem>.contract.json`.
`proc-course-dr.yaml` uses 1.0 g and 4.0 rad/s, the legacy semantics on meteor75.

Old checkpoints (no sidecar) are **v1**. A v1 decoder maps the raw collective fraction through the
reference profile `drones/meteor75` (`collective_g = thrust_g(clip((a0+1)/2))`). The torch decoder
(sim) and the NumPy decoder (harness) are parity-tested, and the harness's v1 decoding is
bit-identical to the pre-UC67 one. On meteor75, v1 relates to v2 by `g = 1.25 a0 + 1.25`, an affine
map in `a0`. `warm_start(..., legacy_v1_to_scale_g=...)` uses it to start a v2 readout from a v1
one (UC68).

In the harness:

- `ContractCommand` gains `collective_g` and `contract_version`. `collective` / `thrust` keep their
  meaning: the reference-plant fraction and its Newtons, derived from `collective_g` for v2. So the
  overlay, the recordings and the loopback parity are unchanged.
- The runner logs the decoded version and records it in the run meta.
- A new top-level `drone: <id>` key makes the gamepad sink use the profile's **inverse thrust
  curve** (`stick = 2 * throttle_for_g(collective_g) - 1`) and its Betaflight rates.
  `rate_scale` and `stick_signs` stay preset settings.
- `drone` together with `controls.rates` or `controls.hover_stick` is a config error.
- The `liftoff` preset names `liftoff-gremlin-micro`. The 5-inch default names
  `liftoff-5inch-vortex-pro-250`, whose fit quality is poor.
- The loopback steps v1 actions with `plant_step` on the raw action. It steps v2 commands on a
  single-env `sim.Plant` (euler-legacy, at the contract DT).

Consequence: the 07 model now flies the micro with its **measured** inverse thrust curve and rates
(rc 0.85 / super 0.71) instead of the preset's hand-set 1.0 / 0.7 and its hover anchor. The two
agree at hover (-0.058 vs -0.06 stick). Away from hover, the measured rates give about 17 % more
stick per rad/s, which is the correct inversion.

**Risk.** A v2 `.pt` copied without its `.contract.json` decodes as v1. The decoded version is
logged and recorded in the run meta and `config.yaml`. Move the two files together.

### Training (AC1)

- `drone-fly train run <config>` trains a packaged config (`meteor75-legacy`, `proc-course-dr`) or
  a YAML file (`extends:` merges a parent). It writes `artifacts/training/<drone-id>/<run-id>/`:
  `config.yaml` (with `contract_version`), `drone.yaml` + `drone.sha256`, `checkpoints/` (readout +
  sidecar, plus periodic ones) and `logs/` (`train.log`, `metrics.jsonl`, `episodes.csv`).
- `train eval` runs the paired evaluation.
- `train bench` is the only throughput benchmark.

## Results

### AC3: Meteor75 equivalence and the 07 checkpoint

- **AC3a.** With `drones/meteor75`, no randomization, `euler-legacy` and one substep, the plant
  matches `BatchedDrone` within 4e-14 per state component over 200 steps (float64, bounded random
  actions through the v1 decoder).
- **AC3b.** Measured on 2026-10-12 on the local RTX 3050 (CUDA), with the real K1 connectome:

  ```
  drone-fly train eval training/07-proc-course-flight/model_readout.pt --config meteor75-legacy --legacy-v1 --episodes 192 --seed 1234 --compare-prototype --k1-path <k1>
  ```

  The comparison was paired: the same courses, ids and spawns, the deterministic mean, the same
  success definition, time limit and episode count.

  | env | completed | rate |
  |---|---|---|
  | new simulator | 187 / 192 | 97.4 % |
  | gpu_prototype, same machine | 187 / 192 | 97.4 % |
  | historical (README, other machine) | 185 / 192 | 96 % |

  The difference is 0.0 pp (limit 5 pp): mean episode lengths and crash counts are identical.
  For information only (not gated), under `proc-course-dr` (randomization 1, random timing, mixed
  starts, decoded v1) the same model completes 116 / 192 (60.4 %). That is the gap UC68 trains
  against.

### AC8: open-loop replay against Liftoff

`drone-fly sim replay-sysid` feeds the logged sticks of every held-out, completed experiment into
the nominal plant, from the measured initial state. The plant uses the exact integrator, 1 ms
steps, and every command at its own logged time. The results are written into each drone's
`sysid.md`.

| drone | windows (1 s / 2 s) | median attitude error 1 s | median velocity error 1 s | gate |
|---|---|---|---|---|
| liftoff-gremlin-micro | 30 / 8 | 0.14 deg | 0.41 m/s | **met** (<= 10 deg, <= 1 m/s) |
| liftoff-5inch-vortex-pro-250 | 31 / 8 | 0.60 deg | 0.65 m/s | not gated (fit_quality poor) |

First-order plus delay is enough at this resolution, so no second-order term was added. Most
experiment spans are shorter than 2 s, so only 8 windows enter the 2 s statistic.

### AC10: throughput and the substeps choice

`drone-fly train bench` measured 8192 envs on the local RTX 3050 on 2026-10-12 at about 19:40
local time. The GPU was idle (no other GPU load). The run used random actions and `step` +
`extra()`; the prototype was `BatchedProcCourse` (tractable + no_obs), and the new env was
`proc-course-dr` (meteor75, randomization 1, timing defaults, episode log on).

| env | env-steps/s | of prototype | plant | timing | course step | extra |
|---|---|---|---|---|---|---|
| gpu_prototype | 979,433 | 100 % | | | | |
| CourseEnv, substeps 1 | 691,413 | **71 %** | 2.73 ms | 0.20 ms | 9.56 ms | 2.00 ms |
| CourseEnv, substeps 2 | 656,086 | 67 % | 5.21 ms | 0.09 ms | 10.62 ms | 2.03 ms |

The plant, timing, course-step and extra columns are per-step times at 8192 envs. The course step
includes the plant and timing.

The target is >= 50 %, so both settings pass. The step is limited by kernel launches. The course
step (masked respawn of every env on every step, contact, rewards, the log) costs more than the
plant, so a second substep costs only about 4 %. The default stays **`substeps: 1`**: the exact
integrator already resolves delays within a substep, and one substep is the faster setting.

Two earlier fixes mattered: caching the randomizer bounds (which removed per-reset host-to-device
copies) and drawing the samples as one table. Together they took the course step from 13.0 ms (the
first bench, at 53 %) to 9.6 ms.

Two later re-runs (`artifacts/eval/bench/ac10-run*.log`) are **invalid**: other applications were
loading the GPU (90-100 % utilisation), and the prototype dropped to 31k-91k env-steps/s.

## Consequences

- Training targets a named airframe and its measured envelope. UC68 extends `proc-course-dr.yaml`
  (drone, budget, warm start).
- The action means the same thing on every airframe. The live sink owns the airframe-specific
  inversion.
- Run folders and sidecars make a checkpoint's contract and drone explicit.
- `gpu_prototype/` is no longer edited. The UC66 sysid sessions it was validated against live only
  in local `artifacts/` and are never committed.
