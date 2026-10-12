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
