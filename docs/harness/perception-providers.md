# Perception providers (UC-86)

The live model ("brain") reads fixed contract blocks. A **perception provider** is the harness part
that turns the screen frame and the telemetry state into those blocks. Any "eyes" system
(Depth-Anything rays, the UC-78/83/85 vision eyes, a future stereo / optical-flow / segmentation
net) is a provider: implement the contract below, select it from config, run the conformance kit,
then score it offline. The brain's input layout never changes; providers convert into it.

Code: `src/drone_fly/harness/perception.py` (contract), `composite.py` (composition),
`perception_kit.py` (conformance kit), `offline/perception_eval.py` (evaluation),
`examples/brightness.py` (the example plugin).

## The contract (version 1)

`perception.CONTRACT_VERSION = 1`. There are two runner slots, and both roles derive from
`perception.PerceptionProvider`:

| slot (`parts.*`) | ABC | blocks it may provide |
|---|---|---|
| `gate_estimator` | `interfaces.GateEstimator` | `target`, `curr_gate`, `next_lookahead`, `oracle` (teacher only) |
| `free_space_estimator` | `interfaces.FreeSpaceEstimator` | `obstacle_vision`, `raycast` |

Blocks (`perception.BLOCKS`):

| block | width | dtype | range | frame / units |
|---|---:|---|---|---|
| `target` | 3 | float32 | finite | contract world frame (right-handed, Z-up), training-scale m; the assembler turns it into obs `rel` |
| `curr_gate` | 3 | float32 | finite | `[cos yaw, sin yaw, aperture]` (`course_senses.curr_gate_block`) |
| `next_lookahead` | 7 | float32 | finite | `course_senses.next_lookahead_block` |
| `obstacle_vision` | 8 | float32 | finite | the 2 nearest obstacles' offset + size, training scale |
| `raycast` | 40 | float32 | [0, 1] | contract `CONE` order; 1 = clear to `RAY_RANGE` |
| `oracle` | - | - | - | teacher-only shared state, not read from the estimate |

A provider declares, as class (or instance) attributes:

- `provides` -- the blocks it fills;
- `provenance` -- `screen` (pixels on screen), `telemetry` (the game's telemetry stream, or a
  constant that reads no environment data) or `metadata` (track files, labels, ground truth);
- `optional` -- `True` when providing blocks the model does not read is fine (the `none`
  estimator);
- `gpu_duty` -- the worst-case GPU duty of an async CUDA worker, if any;
- `contract_version` -- inherited (`1`).

`describe()` returns all of it as a dict; it is recorded in the run's `meta.json` under
`perception`.

### Stamp semantics

`stamp` is the capture time (`perf_counter`) of the newest input the estimate was built from.

- A frame-based free-space provider stamps **`-inf`** until it has a valid estimate, so the
  runner's `runner.max_estimate_age_s` check fails safe (`perception_stale`). An invalid output
  (startup, a history gap) never replaces the held estimate; a gap (no new frame) holds the stamp.
- Exceptions: a gate estimator stamps `now` on a tick without a frame (`frame.stamp` otherwise),
  and constant providers (`none`, the loopback `course` / `raycast` and the `oracle`) stamp `now`.
- `reset()` (a telemetry discontinuity) must never move the stamp forward; it may drop the held
  estimate (eyes) or keep it (depth).

### Lifecycle, threading and CUDA

`start()` (may start one worker thread) / `update()` (synchronous, never blocks on inference) /
`reset()` / `stop()` (joins the worker). An async provider reads frames and state from
`ctx.resources["frame_source"]` / `["state_source"]`; one thread owns the model, and a `reset()`
while the worker runs is applied on the worker thread. On CUDA, run the worker (and the warm-up) on
a dedicated CUDA stream so it never serialises behind the policy on the default stream; only host
NumPy results cross threads. Keep the GPU duty (forward time x rate) at or below 0.5 and the
estimate age below `runner.max_estimate_age_s`; declare `gpu_duty`. `liftoff/eyes.py` is the
reference implementation.

`info` may carry `frame_index` and `inference_s` (per-frame wall time): they feed the latency rows.

## Validation

- **At session build** (`perception.check_session`, after the coverage check): every provider's
  declaration (`check_spec`: known blocks, blocks of its role, contract version, a valid
  provenance). A provider **none of whose blocks the loaded model consumes** is refused unless it
  is `optional` (a provider whose blocks are only partly consumed is logged, not refused). A model
  that needs a block nobody provides is refused as before (`PerceptionCoverageError`).
- **Live, on the first estimates** (`perception.PerceptionRuntime`): every estimate of a provider is
  validated up to and including its first one with a finite stamp: each declared block is present,
  has the declared width, a floating dtype, finite values inside its range; `in_fov_mask` is
  bool-like with 40 values; the stamp is a real number (not NaN, not +inf). A violation stops the
  run with `perception_error` before anything is sent on that tick.

Errors are `PerceptionContractError` (a `HarnessConfigError`) and name the provider and the block,
e.g. `perception provider 'my_eyes' block 'raycast': value 1.3 above the range [0.0, 1.0]`.

## Screen-only guard (provenance)

```yaml
inputs:
  require_provenance: auto   # auto | screen | any
```

`auto` means `screen` on the live Liftoff harness and `any` elsewhere (sim-loopback, offline).
With `screen`, a provider whose provenance is `metadata` -- or undeclared (fail closed) -- is
refused, e.g. the `oracle` / `course` gate estimators or obstacle geometry from track XML.
Telemetry-derived providers are allowed. The UC69 oracle-teacher session (`harness run --teacher
... --perception oracle`) sets `require_provenance: any` itself (it is a diagnostic by design);
set it yourself for any other offline / diagnostic run that needs metadata perception. The eyes
estimator keeps its own refusal of `course` / `oracle` gates.

## Selecting a provider (config syntax)

`parts.gate_estimator` / `parts.free_space_estimator` accept:

1. a **registered name**, as before (`learned`, `opencv`, `oracle`, `course`, `none`, `depth`,
   `eyes`, `raycast`, `composite`);
2. an **import reference** `module.path:ClassName` -- nothing is added to `builtins.py`:

   ```yaml
   parts:
     free_space_estimator: drone_fly.harness.examples.brightness:BrightnessRaysProvider
   ```

3. an **entry-point name** from the `drone_fly.perception` group (lookup only; `harness list` keeps
   listing registered parts). In the plugin package's `pyproject.toml`:

   ```toml
   [project.entry-points."drone_fly.perception"]
   my-eyes = "my_eyes.provider:MyEyesProvider"
   ```

   then `parts: {free_space_estimator: my-eyes}`.

The class is built like any part: `Cls.from_config(cfg, ctx)` when it has that classmethod, else
`Cls(cfg, ctx)`, with the role's config section (`free_space` / `gates`) overlaid with
`part_options.<kind>`. An import-reference or entry-point part must be an instance of its role's
ABC. A missing package raises `HarnessDependencyError`; a wrong reference a `HarnessConfigError`
naming it. Plugin modules must keep heavy imports (torch, cv2) inside functions, like the
built-ins: the harness package imports none of them.

## Composition

Different blocks can come from different providers. Rays from one provider and gates from another
is the normal two-slot set-up (`parts.gate_estimator` + `parts.free_space_estimator`). To split the
free-space blocks themselves, use the `composite` free-space estimator:

```yaml
parts:
  free_space_estimator: composite
inputs:
  free_space:
    - {provider: eyes, blocks: [raycast]}
    - {provider: depth, blocks: [obstacle_vision]}
    # or two claimants of one block, with exactly one primary:
    # - {provider: eyes, primary: true}
    # - {provider: my_pkg.flow:FlowRays, blocks: [raycast], options: {threshold: 0.4}}
```

Each entry: `provider` (any of the three name forms), optional `blocks` (default: all it
provides), `primary`, and `options` overlaid on its config. A block claimed by one child goes to it;
a block claimed by several goes to the single `primary` one -- no primary or several primaries is
refused, naming the block and the claimants. A child that wins no block is refused. `raycast` and
`in_fov_mask` come from the raycast owner; the composite's stamp is the **oldest** owner stamp, so
the one `runner.max_estimate_age_s` covers every block. Gate blocks stay single-provider. Per-block
age limits are a follow-up (today: one limit, the oldest stamp).

## Latency rows

Each run's `latency_summary.json` gains, per provider, `latency_ms["provider:<name>"]` (the
`update()` wall time), `latency_ms["provider:<name>:frame"]` (the provider's per-frame
`inference_s`, counted once per new `frame_index`) and `ages_ms["provider:<name>"]` (now minus its
stamp). Composite children get their own rows. Runs without providers keep the old shape.

## Conformance kit

```python
from drone_fly.harness.perception_kit import check_provider
from my_eyes.provider import MyEyesProvider


def test_my_eyes_sync():
    check_provider(lambda frames, states: MyEyesProvider(), role="free_space_estimator",
                   name="my-eyes", frame_shape=(270, 640, 3))


def test_my_eyes_async():
    check_provider(lambda frames, states: MyEyesProvider(frame_source=frames, state_source=states),
                   role="free_space_estimator", name="my-eyes", async_mode=True)
```

`factory(frame_source, state_source)` builds the provider (`frame_source` is `None` in sync mode).
The kit checks the declaration, the stamp rules (`-inf` before the first estimate, a stamp no newer
than the newest frame, a held stamp in a gap, reset and recovery), every estimate's contract, the
async hammer (updates with interleaved resets while the worker runs) and that `stop()` joins its
threads. A failure raises `PerceptionContractError` naming the provider and the check. It runs on
CPU, without Liftoff or the network.

Staleness exceptions the kit accepts: constant providers stamp `now` (no `-inf` phase, no held
stamp), and a provider whose first `update()` after `reset()` raises `EstimatorStop` (a documented
"reset ends the attempt", e.g. the `oracle`'s respawn rule) passes the reset check -- the report
carries `reset_ends_attempt` and the post-reset frames are skipped. An `EstimatorStop` anywhere
else still fails the kit.

## Offline evaluation

```
drone-fly perception eval --provider eyes --config live-eyes.yaml \
    --dataset <eyes dataset> --split held-out [--max-frames 300] [--device cuda] \
    [--trackdata-root <Liftoff track XMLs>] [--out <dir>]
```

The provider is built like a live session would build it, in synchronous mode, and fed the
dataset's full-resolution capture frames + telemetry in time order (each scored row gets the
preceding 0.6 s of its segment as history; the provider is reset at captures, segments and stride
gaps). The gate target is the label route's passage centre (offline only). The report has the
`eyes eval` shape -- UC-78 metrics (per-ray miss, in-view miss, nearest-quarter 0-4 m miss, false
alarm on clear rays) -- plus `provider`, `provider_describe`, `latency_ms` (p50 / p95 / p99 of
`update()`, this device) and `skipped_frames` (rows without a valid estimate yet). It writes
`perception_eval_<split>.json` / `.md`; compare two providers like-for-like with

```
drone-fly eyes compare --v1 depth/perception_eval_held-out.json --v2 eyes/perception_eval_held-out.json
```

The train split is refused, and so are maps the provider trained on (when its `meta` carries
`split_maps`).

## The example plugin, end to end

`drone_fly.harness.examples.brightness:BrightnessRaysProvider` is a teaching example (never fly
with it): a ray whose pixel patch is darker than a threshold reads `0.5`, else clear.

1. **Implement** -- subclass `FreeSpaceEstimator`, `update()` returns a `FreeSpaceEstimate`.
2. **Declare** -- `provides = {"raycast", "obstacle_vision"}`, `provenance = "screen"`, stamp
   `frame.stamp`, `-inf` before the first frame.
3. **Select** -- `parts: {free_space_estimator: drone_fly.harness.examples.brightness:BrightnessRaysProvider}`.
4. **Conform** -- `check_provider(lambda f, s: BrightnessRaysProvider(camera, cone), role="free_space_estimator")`.
5. **Evaluate** -- `drone-fly perception eval --provider drone_fly.harness.examples.brightness:BrightnessRaysProvider --dataset ...`.
