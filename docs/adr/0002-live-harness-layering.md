# ADR 0002: Live-harness layering (Liftoff as the worked example)

- **Status:** accepted (UC-62); real-Liftoff calibration in
  [ADR 0003](0003-real-liftoff-calibration.md) (UC-64)
- **Date:** 2026-10-11
- **Code:** `src/drone_fly/harness/`, CLI `drone-fly harness …`

## Context

Trained K1 flight models (`training/NN-*/model_readout.pt`) run inside the GPU training env.
UC-62 runs them against **external, real-time flight systems**: first Liftoff (a closed FPV
game, reached only through UDP telemetry, the screen and a virtual gamepad), plus a
sim-loopback over the contract's reference plant that checks the harness reproduces the
reference inference path. More targets will follow (other sims, other games), so the harness
needs a structure where a new target or a new perception method is an addition, not an edit.

Two hard constraints shape it:

1. **The inference contract is frozen and owned elsewhere.** `docs/INFERENCE_CONTRACT.md` +
   `gpu_prototype/inference.py` define the model I/O. Another agent keeps changing the training
   code on `main`, so the harness must reach the contract by reference, never copy it, and never
   write to `gpu_prototype/` or `training/`.
2. **At runtime the model may only use what a pilot could perceive:** telemetry and the screen.
   Liftoff track/race XMLs are allowed **offline only**, as ground truth for scoring the vision
   estimators.

## Decision

### Layers

```
           +--------------------------- HarnessRunner (fixed-rate loop, fail-safe) ---------------------------+
 telemetry |  StateSource ──> FlightState ─┐                                                                  |
 screen    |  FrameSource ──> Frame ───────┼─> GateEstimator ──> target, curr_gate, next_lookahead ─┐         |
           |                               └─> FreeSpaceEstimator ──> obstacle_vision, raycast ─────┤         |
           |                                                                                        v         |
           |                          ObservationAssembler (contract make_obs + extra layout) ──> obs, extra  |
           |                                                    PolicyRunner (contract load_policy) ──> raw    |
           |                                                    decode_action (contract §2) ──> command        |
 gamepad   |  ActionSink <───────────────────────────────────────────────────────────────────────────┘         |
           +----------------------------------------------------------------------------------------------------+
```

| part | contract | Liftoff implementation | sim-loopback implementation |
|---|---|---|---|
| `StateSource` | telemetry -> `FlightState` (contract world frame, training scale) | `liftoff-udp` (`liftoff-csv` replays a recording) | `plant` (reads the reference plant) |
| `FrameSource` | screen/camera -> `Frame` | `screen` (dxcam, else mss) | none |
| `GateEstimator` | -> `target`, `curr_gate(3)`, `next_lookahead(7)` | `opencv` (HSV quads + PnP); `learned` (keypoint net per gate type + PnP, UC-64) | `course` (ground truth) |
| `FreeSpaceEstimator` | -> `obstacle_vision(8)`, `raycast(40)` | `none` (obstacle-free) / `depth` (monocular depth net) | `raycast` (contract `raycast()`) |
| `ObservationAssembler` | -> `obs(12)`, `extra(extra_dim)` | `ContractObservationAssembler` (shared) | same |
| `PolicyRunner` | `act(obs, extra)` -> raw[4] | `ContractPolicyRunner` (shared) | same |
| `ActionSink` | `ContractCommand` -> native input | `vgamepad` (ViGEm Xbox 360 pad) | `plant` (contract `plant_step`) |
| `AbortSignal` | user abort | `hotkey` (F12) | none |
| `FocusProbe` | target window focused? | `win32-foreground` (UC-64; skipped in dry runs) | none |

A `Harness` is a **composition object** built by a registered factory from YAML config — never a
subclass per target. Parts and harness factories are registered by name
(`drone_fly.harness.registry`); registration targets are lazy `"module:attr"` strings, so
registering a part imports nothing.

### Contract bridge rules

- `harness/contract.py` is the **only** module that touches `gpu_prototype/`, read-only. It
  re-exports `load_policy`, `make_obs`, `plant_step`, `raycast`, `init_state`, `quat_rotate`,
  `quat_rotate_inv`, `CONE`, `N_RAYS`, `RAY_RANGE`, `MAX_BODY_RATE`, `MAX_THRUST`, `HOVER_THR`,
  `DT` **by reference**. `import_reference()` is the only path to `gates_proc_env` and
  `course_gen` (reference maths and course sampling).
- Import traps it handles: `hover_gpu.dev` is overridden **before** anything binds it by value
  (CPU-only machines and CI); `reservoir_aug.K1_PATH` is set from `--k1-path`; modules must resolve
  inside the configured dir (sys.path collision guard); a second load with a different device
  raises (the device is process-global); the pure reservoir is refused (its K1 path is hardcoded).
- Checkpoints load exactly per contract §5 via the contract's `load_policy` (`strict=False`, only
  `body.layer.*` missing). `decode_action` writes out contract §2 (linear-clamp collective,
  `tanh * MAX_BODY_RATE`) because sinks need the physical command; the parity test pins it.
- The legacy `drone_fly.adapter` is **not** used: its throttle / Euler semantics contradict §2.

### Observation assembly and coverage

`extra = obs(12) ‖ curr_gate(3) ‖ next_lookahead(7) ‖ obstacle_vision(8) ‖ raycast(40)`,
truncated to `extra_dim` (12 / 30 / 70 are clean prefixes). Each estimator declares the blocks it
`provides`; before the model is loaded and before the sink is armed, the runner checks that the
configured perception covers every block the model needs and **refuses to start, naming the
missing blocks** (exit code 4). The `none` free-space estimator never provides `raycast` (a fake
"all clear" scan would silently blind an obstacle model) and only provides a zero
`obstacle_vision` when the config states `assume_no_obstacles: true` — so a 30-dim model can fly
an obstacle-free Liftoff course and a 70-dim model is refused until `depth` is configured.

`curr_gate`, `next_lookahead` and `obstacle_vision` exist upstream only as batched torch code in
`gates_proc_env.py`; `harness/course_senses.py` is a NumPy single-drone port, held to the
original by a drift test (through `import_reference`). Documented deviations:

1. Training adds bearing noise (σ = 0.15) when the next gate is occluded; the harness is
   deterministic and omits it.
2. An unseen / occluded next gate never gets a zero bearing: it points at the last-known next-gate
   position re-bearing'd from the current position (with known geometry that is exactly the
   noiseless training bearing); with no history, the unit vector to the current gate.
   `orient`, `dist`, `vis` are zero, as in training.
3. Rotated track-XML obstacles are reduced to axis-aligned boxes (training only has AABBs).

Curricula are fixed at evaluation values (difficulty 1: full-size on-path obstacles, no pass
slack).

### Frames and scale

- Contract world: right-handed, Z-up; body FLU; quaternions `[w, x, y, z]` body -> world.
- Unity (Liftoff): left-handed, Y-up, Z-forward. Positions/velocities map `X = z_u, Y = -x_u,
  Z = y_u`. That map is a reflection (det -1), so a quaternion's vector part maps through the same
  matrix **and is negated**; `w` is unchanged.
- Body rates: Liftoff's gyro convention is not documented, so `telemetry.gyro_map` is an explicit
  axis/sign/unit table, and `gyro_check` **measures** it by finite-differencing the converted
  attitude stream (sign agreement, scale, correlation, a suggested table).
- `WorldScale` is the only place lengths change scale: `training_m = target_m / world_scale` for
  positions, velocities, gate centres, apertures, obstacle geometry, `obstacle_vision`, lookahead
  distance, depth ranges and track-XML geometry. Rates and quaternions are never scaled; the
  gamepad `rate_scale` is a separate action-side knob.

### Runner and fail-safe semantics

- Fixed rate (default 50 Hz = the training `DT`), deadline scheduling on an injectable clock;
  missed deadlines are counted and skipped (no catch-up burst).
- Per tick: abort? → state (stale check; a telemetry discontinuity — `t` reset or position jump,
  i.e. a respawn — resets both estimators) → gate update → free-space latest → assemble → act →
  decode → `sink.send` → record (per-stage latency, estimate ages, events).
- **Latched fail-safe** on: telemetry age > `stale_after_s`; gate status `lost`; free-space estimate
  age > `max_estimate_age_s` (when the model needs it); any perception / policy / sink exception;
  the abort hotkey; Ctrl+C. Action: `sink.failsafe()` = neutral rates + throttle cut (retried once),
  then stop with a reason code (exit code 5).
- The sink receives only the decoded contract command. **No yaw shaping or rate override.**
- Every run records under `artifacts/harness/<harness>-<UTC>/` (refused under `training/`):
  config, meta (checkpoint sha256, device, git HEAD), raw telemetry, sampled frames, estimator
  outputs, obs/extra, actions, latencies.

### Liftoff worked example

- **Telemetry** (`liftoff/telemetry.py`): packed little-endian float32 blocks in the configured
  `StreamFormat` order (timestamp, position, attitude xyzw, velocity, gyro pitch/roll/yaw, inputs,
  battery, motor RPM); a stdlib UDP listener on a background thread, latest wins.
- **Gates** (`liftoff/gate_cv.py`): HSV threshold → convex quads → `solvePnP` against the
  configured *nominal* outer gate size → camera → body (uptilt) → world; the gate yaw comes from
  the plane normal oriented along the direction of travel. Tracking keeps a current and a next
  gate; a pass (proximity rule, plane crossing, or image area peaking then vanishing) advances.
  When the gate is lost, the last world estimate is held for `lost_timeout_s` (`rel` recomputed
  from live position), then the runner fails safe.
- **Actions** (`liftoff/gamepad.py`): body rate → deg/s (× `rate_scale`) → inverse of the
  Betaflight rate curve (rc rate / super rate / expo per axis) → stick; throttle is piecewise
  linear through `(0, -1)`, `(HOVER_THR, hover_stick)`, `(1, +1)`. Explicit FLU signs: `+wx` → roll
  right, `+wy` (nose down) → pitch forward, `+wz` (CCW) → yaw left.
- **Free space** (`liftoff/depth.py`, `depth`): a pluggable `DepthModel` (first backend:
  Depth-Anything-V2 metric via `transformers`, lazily loaded) runs on its own worker thread, latest
  frame wins, with an affine scale/shift calibration (optionally on inverse depth). Each tick, the
  contract's 40 rays (velocity-aligned, target fallback below 0.3 m/s, built by
  `geometry.ray_directions`, pinned to `raycast()`) are projected world → body → camera → pixel
  with the capture-time attitude. In-image rays take a robust patch minimum; z-depth → range
  (`z / cos`) → `WorldScale` → clamp at `RAY_RANGE × ray_range_scale` → normalise. Ground hits are
  rejected (training rays never hit the floor). Out-of-image rays use the fill policy
  `clear | nearest_in_fov (default) | hold_last`, and the in-FOV mask and coverage are recorded.
  `obstacle_vision` comes from near-depth connected components (or zeros). The estimate is stamped
  with the frame capture time, and the runner fails safe beyond `max_estimate_age_s`.
- **Offline evaluation** (`harness/offline/`, the only XML reader): track XML items are classified by
  a configurable item-id table, run through the Unity converter and `WorldScale`, and rotated
  boxes become AABBs. The race XML gives the gate order. `drone-fly harness eval` replays a recorded
  session (each sampled frame paired with the nearest telemetry sample; estimators re-run
  synchronously). It reports gate position/yaw error and detection rate, free-space per-ray MAE and
  near-weighted error against the contract `raycast()` on the XML obstacles, `obstacle_vision`
  error, the gyro check, and a suggested depth-scale fit.
- Per-game / airframe settings (telemetry port and format, FOV, uptilt, nominal gate size, HSV,
  rate profile, hover stick, world scale, depth calibration) live in YAML; `liftoff_5inch.yaml` is
  the default. Packaged presets (`--preset liftoff_micro_arena`) overlay it with measured values.
- UC-64 additions (see [ADR 0003](0003-real-liftoff-calibration.md)): the corrected gyro
  convention (`wx = -roll`, `wy = +pitch`, `wz = -yaw`) and MotorRPM count prefix, linked-list race
  routes, the `gate_types` table and the `learned` gate estimator (the tracker is now the shared
  `TrackingGateEstimator` base), `camera.masks`, the `focus_lost` fail-safe, per-estimator
  perception timing (`perception_s`) and `latency_summary.json`.

## How to add a harness or an estimator

1. Implement the interface (e.g. `class MyDepthEstimator(FreeSpaceEstimator)` with
   `provides = frozenset({"obstacle_vision", "raycast"})` and a `from_config(cfg, ctx)` classmethod).
2. Register it: `register("free_space_estimator", "my-depth", "my_pkg.mod:MyDepthEstimator",
   requires=("some_backend",))` — built-ins do this in `harness/builtins.py`.
3. Select it in YAML (`parts.free_space_estimator: my-depth`). A new *target* additionally
   registers a `harness` factory `(config, ctx) -> Harness` and a packaged default YAML.

No runner edit is needed; the coverage check and the fail-safe apply automatically.

## Known risks and follow-ups

- **Scale / plant mismatch (dominant).** Models were trained on a 32 g whoop (T/W 2.5,
  4 rad/s, 4 m rays); a 5-inch quad has T/W > 8 and much higher rates. `world_scale`,
  `rate_scale`, the rate profile and `hover_stick` are configurable, but Liftoff flight is a
  warm start, not a zero-shot guarantee. Contract-exact I/O and measured perception accuracy are
  the acceptance targets, not flight success.
- **Camera heading.** The policy is yaw-agnostic (trained with a velocity-aligned ray cone and
  world-frame `rel`), while an FPV camera only sees forward. If the drone yaws away from its travel
  direction, gates leave the image. This UC applies **no** yaw shaping; a yaw-alignment layer is a
  candidate follow-up use case.
- **Latency.** Perception may not sustain 50 Hz while Liftoff renders; estimates are consumed
  asynchronously with age checks, and per-stage latency is recorded for live calibration.
- **Respawns / menus.** Detected via telemetry discontinuities; tailing Unity `Player.log` is
  deferred.
- **Calibration inputs** from the owner: telemetry format/port, gyro convention (measured by
  `gyro_check`), rate profile, hover stick, gate size and colour, track-XML item table, and the
  in-game binding of the virtual pad.
