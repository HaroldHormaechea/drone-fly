# Inference contract — deploying a trained K1 flight model

This is the frozen interface for running a trained model **outside the training loop** (e.g. a liftoff /
deployment harness). It documents exactly what goes **in** (observation), what comes **out** (action),
the **plant** the action drives, and the **file/model formats**. Build against this and the harness stays
decoupled from the training code, which can keep evolving.

Reference implementation: [`gpu_prototype/inference.py`](../gpu_prototype/inference.py) — loads a model and
exposes `act(obs) -> action`, plus a pure reference `plant_step(...)` matching the simulator so a harness
can integrate flight without importing the training env.

---

## 0. Contract version (UC67)

The **action** contract has two versions. The observation (§1), the extra senses (§4) and the file
format (§5) are the same in both.

| version | raw action `a` means | used by |
|---|---|---|
| **1** (legacy) | `a0` -> the collective **fraction** `clip((a0+1)/2, 0, 1)` of the reference plant (`drones/meteor75`, §2-3); `tanh(a1:4) * MAX_BODY_RATE` rad/s | every checkpoint trained before UC67 (`training/01`-`12`, `gpu_prototype`) |
| **2** | `collective_g = clamp(1 + collective_scale_g * a0, 0, max_g)`: body-z specific force in **g**, 1.0 = hover on ANY airframe; `rates = tanh(a1:4) * max_rate_rad_s` | checkpoints trained with `drone-fly train` on `drone_fly.sim` |

The canonical (decoded) action of both versions is `[collective_g, roll_rate, pitch_rate,
yaw_rate]`. A v1 action decodes to it through the reference profile:
`collective_g = thrust_g(fraction)`, i.e. `2.5 * fraction` on meteor75. On meteor75, v1 and v2
are related by `collective_g = 1.25 * a0 + 1.25`, an affine map in `a0`.

**The version lives in a sidecar** beside the checkpoint, `<stem>.contract.json`:

```json
{"contract_version": 2,
 "action": {"collective_scale_g": 1.0, "max_rate_rad_s": 4.0, "max_g": 2.5},
 "extra_dim": 30, "actor": "k1-reservoir-aug", "drone": "meteor75", "run_id": "..."}
```

- A checkpoint **without a sidecar is v1**. Always move the `.pt` and its `.contract.json` together;
  a v2 model read as v1 flies wrong.
- `collective_scale_g` and `max_rate_rad_s` come from the training config (no defaults);
  `max_g` is the training drone's envelope.
- A v2 readout's collective bias starts at 0, which commands hover.

**Deploying v2.** Command `collective_g` and the rates to the target airframe. Its inverse thrust
curve gives the throttle: `stick = 2 * throttle_for_g(collective_g) - 1`. Its Betaflight rates give
the stick deflection for each body rate. The live harness does this with the `drone: <id>` config
key (`drones/<id>/drone.yaml`).

The harness reads the sidecar (`ContractPolicyRunner.load`), logs the decoded version and records
it in the run's `meta.json`. Its decoders are
`drone_fly.sim.action.decode_numpy` (harness) and `decode_torch` (training), which are
parity-tested. The v1 decoding is bit-identical to §2. `gpu_prototype/inference.py` (`plant_step`)
remains the v1 reference plant; the v2 reference plant is `drone_fly.sim.Plant` with the
checkpoint's drone profile. See [ADR 0006](adr/0006-profile-driven-batched-sim.md).

## 1. Observation (the core 12-dim vector — shared by ALL models)

`obs` is a **float32[12]**, built by `BaseEnv.obs()` / `BatchedProcCourse.obs()` as the concatenation:

| idx | name | dim | frame | meaning |
|----|------|-----|-------|---------|
| 0:3 | `rel` | 3 | world | `target_gate_center − drone_position` (vector to the current target) |
| 3:6 | `g_body` | 3 | body | gravity direction `(0,0,−1)` rotated into the body frame (attitude cue) |
| 6:9 | `vel` | 3 | world | linear velocity |
| 9:12 | `omega` | 3 | body | angular velocity (body rates) |

This 12-vector is the **only** input the frozen connectome brain sees (via `input_projection`). Everything
else is read by the *readout*, not the brain (§4).

## 2. Action (4-dim) and its exact mapping to the plant (contract v1)

The model outputs a **float32[4]** raw action, nominally in `[−1, 1]`, laid out `[collective, wx, wy, wz]`.
The simulator (`hover_gpu.Base.step`) maps it as:

```python
thrust   = clamp((a[0] + 1) * 0.5, 0, 1) * MAX_THRUST     # collective, Newtons. NOTE: linear-clamp, NOT sigmoid
rate_cmd = tanh(a[1:4]) * MAX_BODY_RATE                   # desired body rates (rad/s), [wx, wy, wz]
```

> The abstract `encoding.py` contract says "sigmoid throttle"; the **actual trained simulator uses the
> linear-clamp map above**. Replicate `hover_gpu`, not `encoding.py`. The policy's throttle output bias is
> set so a neutral action hovers: `pi.bias[0] = 2·HOVER_THR − 1`.

## 3. Plant (CTBR with an inner rate loop) — constants + integrator

Constants (`hover_gpu.py`):

| symbol | value | meaning |
|--------|-------|---------|
| `M` | 0.032 kg | mass |
| `G` | 9.81 | gravity |
| `TW` | 2.5 | thrust-to-weight |
| `MAX_THRUST` | `TW·M·G` ≈ 0.785 N | collective at full stick |
| `HOVER_THR` | `1/TW` = 0.4 | collective fraction that hovers |
| `MAX_BODY_RATE` | 4.0 rad/s | body rate at full stick |
| `RATE_KP` | 18.0 | inner rate-loop gain (rad/s² per rad/s error) |
| `DT` | 0.02 s | **50 Hz** control + integration |

Integrator per step (semi-implicit Euler, quaternion attitude):

```
omega_dot = RATE_KP * (rate_cmd - omega)
omega    += omega_dot * DT
quat     += 0.5 * quat_mul(quat, [0, omega]) * DT ;  quat /= ‖quat‖
thrust_world = R(quat) @ [0, 0, thrust]
acc  = thrust_world / M + [0, 0, -G]
vel += acc * DT
pos += vel * DT
```

`quat` is `[w, x, y, z]`. `tilt = acos(clamp(g_body_z_component))` is tracked to detect ballistic tumbling
(a model that "completes" while inverted is cheating — see `docs/FLIGHT_MODELS.md`).

## 4. Models and the extra-sense readout

Two actor families (both freeze the ~3.86M real MaleCNS edges; only the readout + input projections train):

- **Pure reservoir** — `gates_k1_reservoir.K1Reservoir`. Readout sees only the brain tap. `act(obs12)`.
  Used by the straight-gate model.
- **Augmented reservoir** — `reservoir_aug.K1ReservoirAug(extra_dim)`. Readout sees `[brain_tap ‖ extra]`.
  Used by oval / track / pad / obstacle / proc models. **`extra_dim` varies per model** — it must match
  the checkpoint or `load_state_dict` fails on a shape mismatch:

  | model | `extra_dim` | extra vector |
  |-------|-------------|--------------|
  | oval-lap / figure8-chicane | **12** | just the obs, re-fed to the readout |
  | pad-lap / obstacle-lap | 12 (+ task senses, see that scenario) | |
  | proc-course-flight | **30** | full course senses (below) |
  | proc + raycast (experimental) | **70** | course senses + 40 rays |

  The full course-sense **extra** vector (proc) is:

  ```
  extra = [ obs(12) ‖ curr_gate(3) ‖ next_lookahead(7) ‖ obstacle_vision(NVIS·4=8) ‖ raycast(40?) ]
          EXTRA_DIM = 12 + 3 + 7 + 8 (+40 if raycast) = 30  (or 70 with raycast)
  ```
  - `curr_gate(3)` = `[cos(gate_yaw), sin(gate_yaw), aperture]` of the current target gate.
  - `next_lookahead(7)` = `bearing(3)` + `orient(2 = cos,sin of next gate yaw)` + `dist(1)` + `vis(1 = 1−occluded)`.
  - `obstacle_vision(8)` = for the nearest `NVIS=2` obstacles: `rel(3) + size(1)`, zeroed if none.
  - `raycast(40)` = forward 90° cone, 40 free-space distances (normalized), only if the model was trained with it.
- **Brain-integrated (experimental)** — `reservoir_brain.K1ReservoirBrainRay`: rays injected *into* the
  connectome. `act(obs12, rays40, extra)`. This is the open-problem obstacle line, not a shipped model.

> For a **liftoff / basic-flight** harness, you only need §1–§3 plus a pure or augmented model with a simple
> target: set `rel` toward a waypoint, feed zeros for obstacle/lookahead/raycast senses (no obstacles), and
> the augmented models fly fine. Full course senses are only needed for gated course flying.

## 5. File formats (what's on disk)

- `training/<name>/model_readout.pt` (~0.6 MB) — a **stripped** state_dict: the trainable readout **and**
  the frozen-but-random `body.input_projection`. The ~3.86M-edge sparse layer (`body.layer.*`) is **removed**
  and **rebuilt from `artifacts/pruned/k1/` at construction**. Therefore **load with `strict=False`** and
  assert the only missing keys are `body.layer.*`:

  ```python
  missing, unexpected = actor.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
  assert all(k.startswith("body.layer.") for k in missing) and not unexpected
  ```
  `body.input_projection` **must** be in the file — it is random at construction and the readout was trained
  against that specific projection (do not reconstruct it).
- `artifacts/pruned/k1/` — the real K1 connectome (neuron ids, edge index/weights, sign mask). Loaded by
  `drone_fly.connectome.load_connectome`. This is the frozen "brain"; it is the same for every model.

## 6. Minimal harness loop (pseudocode)

```python
from inference import load_policy, plant_step, init_state   # see gpu_prototype/inference.py
policy = load_policy("training/03-oval-lap-flight/model_readout.pt", extra_dim=30)
s = init_state(pos=[0,0,0.12])                 # on the ground
for t in range(N):
    obs   = make_obs(s, target)                # §1 layout
    extra = make_extra(obs, course, s)         # §4, or zeros for obstacle-free basic flight
    action = policy.act(obs, extra)            # float32[4]
    s = plant_step(s, action)                  # §3 integrator (50 Hz)
```

## 7. Teacher checkpoints: `teacher-obs` v1 / v2, CourseFrame, sidecar keys (UC68)

Teacher checkpoints are written by `drone-fly train teacher` and use action contract **v2** (§0).
They take a different observation from the K1 models: the privileged **`teacher-obs`**, with
exact (noise-free) quantities. v1 has 148 float32 values; v2 adds the battery for 150. See
[ADR 0007](adr/0007-track-library-and-teacher.md). The actor is a plain MLP (`actor: mlp-teacher`)
that applies its own fixed per-value input scaling, so the observation is fed **raw**.

### Course frame

All positions are in the **course frame**. This is the contract frame (Z-up, right-handed, metres),
translated so that the spawn's support point is the origin, then divided by `world_scale`. It is
never rotated. `drone_fly.sim.pool.CourseFrame` is the one implementation:

```
to_course_point(p)  = (p - origin_contract) / world_scale
to_course_vector(v) = v / world_scale
```

The UC68 teacher is trained at `world_scale: 1.0` (Liftoff metres). A harness must take
`world_scale` from the sidecar, **not** the legacy `world_scale: 4.0` of the live-harness config.
The body frame is FLU.

### Layout

| block | slice | content |
|---|---|---|
| `harness30` | [0:30) | exactly `ContractObservationAssembler.assemble(..., extra_dim=30)`: the §1 `make_obs` 12, `curr_gate`, `next_lookahead` and `obstacle_vision` from `course_senses`, over the course's **harness obstacle list**, with no occlusion noise |
| `attitude6` | [30:36) | the first two columns of R(body → world) |
| `vel_body` | [36:39) | velocity in the body frame |
| `gates` | [39:78) | target, next and next-next passage × 13: centre relative to the drone (body, 3), travel-oriented normal (body, 3), up vector (body, 3), w, h, distance, valid. A row past the finish is all zeros. |
| `obstacles` | [78:142) | the 8 nearest surfaces of the full geometry × 8: nearest point relative to the drone (body, 3), distance, outward normal (body, 3), valid (0 beyond 12 m) |
| `altitude` | [142] | z minus the highest solid or floor surface straight below |
| `prev_action` | [143:147) | the raw policy output of the previous tick, before decoding, clipping or arm override; zeros at the start |
| `arm_remaining_s` | [147] | seconds until the motors accept commands; 0 once armed and for air starts |
| `battery` (v2 only) | [148:150) | `batt_charge` (0–1) and `batt_volts / cells` (V per cell), exactly as Liftoff's telemetry reports them; input scale 1 and 1/4.2 |

**Versions.** `teacher-obs` v2 (150 values) is the current version; v1 is its first 148 values.
The sidecar's `obs_schema.version` says which one a checkpoint takes. A harness builds the battery
block from live telemetry with `tracks.teacher_obs_ref.battery_from_telemetry(batt_charge,
batt_volts, cells)`, where `cells` comes from the drone profile's `power.cells` and is recorded in the
sidecar as `obs_schema.battery.cells` (with `power_model: true|false`). A v2 teacher was
trained with the plant's power model (low-voltage cutoff, thrust drop at speed); see
`drones/<id>/sysid.md`, "Power model".

**Harness obstacle list.** Placed track items only: never scene colliders, never route-gate frames.
Each is one AABB, kept if its max half-extent is at most 3.0 m and it lies inside the course
corridor. At most `O_h` items are kept (722 for the current catalogue), nearest the route first.
`course_senses` runs over the whole list. The rule text is pinned in `obs_schema.harness30.harness_list`
and `course_conversion`, so a live oracle (UC69) passes an identical list.

**Gate pass rule** (`sim.pool.gate_crossed`). An opening counts only when the segment from the
previous to the current position crosses its plane in the travel direction, inside the outline. A
volume counts when the segment enters it moving along the route. Doubling back never counts.
Evaluation uses zero slack.

References:
- `drone_fly.tracks.teacher_obs_ref.build_teacher_obs` is the NumPy single-drone reference. It
  calls the real harness assembler.
- `drone_fly.sim.teacher_obs.build_teacher_obs` is the batched training builder. Tests pin it to
  the reference.

### Sidecar keys

A teacher's `<stem>.contract.json` adds three keys to the §0 ones:

```json
{"contract_version": 2, "actor": "mlp-teacher", "drone": "liftoff-gremlin-micro", "...": "...",
 "obs_schema": {"name": "teacher-obs", "version": 1, "dim": 148, "blocks": ["..."],
                "harness30": {"...": "..."}, "gates": {"...": "..."}, "input_scale": ["..."]},
 "world_scale": 1.0,
 "course_conversion": {"version": 1, "fingerprint": "...", "parameters": {"...": "..."},
                       "rules": {"...": "..."}, "scenes": {"<Map>": {"sha256": "...", "bundle": "...",
                       "build_id": "...", "status": "..."}}, "props": {"sha256": "...", "build_id": "..."}}}
```

- `obs_schema` describes the observation exactly: its block names, widths and offsets, plus the
  pinned `course_senses` constants and harness-list rule.
- `course_conversion` records how the XMLs and scenes became courses: the rules, the parameters
  (corridor, cell size, candidate cap, observation range, harness cap and max half-extent), and the
  hashes of every scene cache and of `props.json`.
- `drone-fly train eval` refuses a checkpoint whose schema name or version, or whose conversion
  version, it does not know.
- A checkpoint without `obs_schema` is not a teacher.

## 8. Student checkpoints: `student-obs` v2 and the `drone-fly-student` export (UC70)

The deployable student is the K1 reservoir actor distilled from the teacher under a measured
detector-noise model ([ADR 0009](adr/0009-student-distillation-and-detector-noise.md)). It sees only
what the live harness can assemble.

### Units

**Student units**: Liftoff metres divided by the student's world scale `k`, exactly the harness's
`WorldScale(k)` (the checkpoint's `world_scale`; 10 for the first student). Rates, quaternions and
times are not scaled.

### Layout (61 float32, `drone_fly.sim.student_obs`, v2)

| Indices | Block | Contents |
|---|---|---|
| [0:30) | `harness30` | `ContractObservationAssembler.assemble(state_k, GateEstimate_k, FreeSpaceEstimate(obstacle_vision = zeros), 30)`: the contract `make_obs` (rel to the perceived target, gravity in body, velocity, body rates), `curr_gate` (perceived yaw, the detector class's nominal aperture / k), `next_lookahead`, `obstacle_vision` = zeros |
| [30:36) | `attitude6` | R(body -> world) columns 0 and 1 |
| [36:39) | `vel_body` | body-frame velocity / k |
| [39:43) | `prev_action` | the previous tick's raw output (before decoding, clipping or the arm override) |
| [43] | `arm_remaining_s` | the arming time left |
| [44:47) | `perception` | `min(target_age, T)/T`, target valid, `min(next_age, T)/T`, with `T = lost_timeout_s`; age = now - the capture stamp of the frame behind the last match |
| [47:59) | `history` | EMA (tau 0.25 s) of the body-frame target vector / k (3) and of `prev_action` (4); the world-frame target change since the previous tick, rotated into the body (3; zero unless matched this tick, zero on a hand-over); ticks since a match / 50, capped at 1 (1); the hand-over flag (1) |
| [59:61) | `battery` | `charge` (Liftoff `batt_charge`, 0..1) and volts per cell (`batt_volts / cells`, V; `cells` = the drone profile's `power.cells`, recorded in `obs_schema.battery.cells`) — exactly `teacher-obs` v2's battery block (`tracks.teacher_obs_ref.battery_from_telemetry`); in the sim, the plant's power model |

The history resets at episode / session start (its EMAs start from the first tick's values).

v2 (UC70, after UC68's power model) appended the `battery` block; v1 (59 values) is no longer
deployed and a v1 export is refused by the version guard. The live harness needs `battery` in the
telemetry `StreamFormat`; without it a student run is refused (exit 2).

`drone_fly.student.obs_ref.build_student_obs` builds it through the real assembler; the harness's
`StudentObservationAssembler` calls the same function, and the batched training builder is
tolerance-tested against it (1e-5).

The gate tracker's `GateEstimate.info` carries the ages: `target_age_s`, `next_age_s`,
`target_valid`, `hand_over` (and `matched`).

### Export format

`drone-fly train export <run> --out student.pt` writes ONE `torch.save` mapping plus a
`student.contract.json` sidecar with the same `meta`:

```json
{"format": "drone-fly-student", "format_version": 1,
 "state_dict": {"...": "the readout only (k1.body.layer.* is rebuilt from the K1 connectome)"},
 "meta": {"contract_version": 2, "action": {"collective_scale_g": 1.0, "max_rate_rad_s": 4.0, "max_g": 4.23},
          "actor": "k1-student", "actor_arch": {"encoder": true, "residual": false, "...": "..."},
          "obs_schema": {"name": "student-obs", "version": 2, "dim": 61, "world_scale": 10.0,
                         "tracker": {"lost_timeout_s": 1.3, "acquire_timeout_s": 3.0, "assoc_radius": 1.0},
                         "camera": {"resolution": [2560, 1080], "hfov_deg": 120.0, "uptilt_deg": 20.0},
                         "battery": {"features": ["charge", "volts_per_cell"], "cells": 3, "...": "..."},
                         "...": "..."},
          "world_scale": 10.0, "teacher": {"...": "..."}, "noise_model": {"sha256": "...", "detector_id": "..."},
          "perception": {"...": "..."}, "camera": {"...": "..."}, "timing": {"...": "..."},
          "course_conversion": {"...": "..."}, "k1": {"prune_provenance_sha256": "..."},
          "latency": {"p50_ms": 0.0, "p99_ms": 0.0, "budget_ms": 10, "pass": true},
          "deployable": true, "role": "deployable",
          "requires": {"gate_estimator": "learned", "detector_sha256": "...",
                       "free_space_estimator": "none", "assume_no_obstacles": true}}}
```

**Version guard** (`drone_fly.student.checkpoint.validate_meta`, run by `harness run` before
arming): the embedded `obs_schema` must deep-equal this build's `student-obs` schema for the same
parameters, the action contract must be v2 and the format version known; otherwise
`StudentSchemaError` names both versions and the run exits 2. `--extra-dim` is not accepted with a
student checkpoint.
