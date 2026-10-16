# UC78 — vision "eyes" v1: camera estimator of the 40 obstacle rays

The 70-input connectome (lineage 10/11/12) consumes 40 obstacle rays as the last block of its extra
vector. In the training sim these come from `inference.raycast`: 40 Fibonacci-spiral rays in a
±45° cone around the velocity heading, each the free distance to the nearest obstacle surface,
clipped at `RAY_RANGE` (4 m) and normalised to [0, 1] (1 = clear). Live in Liftoff there is no
ground-truth geometry at inference, so the rays must come from the camera. The "eyes" are a small
learned network that reads the last few Liftoff frames plus telemetry ego-motion and outputs, per
ray, that normalised free distance and a confidence.

This slice delivers the FOV report, exact labels, the dataset, the model with its train / eval /
bench commands, and this document. Two later slices are separate use cases: the noise model plus a
student retrain under it (UC-73 AC7-8), and live validation in Liftoff (UC-73 AC9-10).

Code: `src/drone_fly/eyes/`. CLI: `drone-fly eyes …`.

## Inputs and outputs

**Inputs (deployable path, `EyesEstimator.update(frame, state, target)`):**

- **Frames.** The current frame and K−1 = 2 history frames. Each is masked (the preset's HUD rects),
  area-resized to 192×80 RGB and rounded back to uint8. The dataset cache and live inference both
  use the same `prepare_frame`, so a cached frame and a live frame are bit-comparable. The frames
  are stacked as 3K = 9 channels, current frame first.
- **History offsets.** The actual time offsets of the two history frames, in seconds
  (`hist_dt`, see the time stride section below).
- **Ego-motion (10 values, training scale).**
  - Gravity in the body frame, `R^T (0, 0, −1)`: the contract `make_obs` block. Level flight gives
    `(0, 0, −1)`.
  - Body velocity `R^T v` (3 values) and speed `|v|` (1 value).
  - Body rates from the attitude change `q(t − dt)^−1 q(t)` over the actual dt, using the stride-1
    history frame. The gyro is **not** used: replay captures carry `gyro = 0`, so a gyro input
    would differ between training and live.
- **Per ray.**
  - The unit camera-frame direction.
  - `(u, v)` in [−1, 1], the projection of the ray through the preset camera (2560×1080, hfov
    120°, uptilt 20°).
  - `in_view`: in front of the camera, inside the image and outside every HUD mask.
- **Target.** The gate estimator's current target, in the contract frame. It only orients the cone
  below 0.3 m/s, exactly as `inference.raycast` does.

**Outputs:** `free` (40 values in [0, 1], 1 = clear to the scaled range), `conf` (40 values in
[0, 1]), `in_view` (40), `history_ok`, and the stamp of the frame the estimate came from.

**Network (`EyesNet`, 0.62 M parameters, budget 1.5 M):**

- Five stride-2 stages (32-48-64-96-128 channels).
- For each ray, a bilinear `grid_sample` of the stride-8 and stride-16 maps at its `(u, v)`. The
  sampled features are **zeroed when the ray is out of view**.
- The global average of the last stage, plus the ray direction, `in_view`, the ego vector and
  `hist_dt`.
- A shared 256-128 MLP produces the free value (sigmoid) and a confidence logit.

## Velocity-aligned rays

The cone follows the **velocity**, the camera follows the **body**. The ray basis is
`harness.geometry.ray_directions`, a copy of `inference.raycast`'s six-line basis that a test pins
against it:

- the forward axis is the velocity when the speed is above 0.3 m/s, else the direction to the
  target, else +x;
- roll comes from world up.

The canonical cone comes from the weights meta, so inference never loads the contract. Which rays
the camera sees therefore depends on how far the body pitch diverges from the flight path.

## FOV coverage (AC1)

Run `drone-fly eyes fov-report --preset liftoff`. It sweeps body pitch 0-60° (nose down, 5° steps)
against a flight-path angle of ±15°, and adds the low-speed case where the cone points at the target.
Results for the `liftoff` preset with its HUD masks:

| body pitch | 0° | 10° | 20° | 30° | 40° | 50° | 60° |
|---|---:|---:|---:|---:|---:|---:|---:|
| rays in view (level path) | 27 | 31 | 34 | 32 | 27 | 23 | 18 |
| min–max over ±15° path | 18–34 | 25–34 | 29–34 | 26–34 | 21–33 | 15–30 | 9–26 |

Over the 10-40° racing band: 5 rays are always in view, 0 are never in view, and 35 depend on
attitude. Between 21 and 34 rays are in view per pose.

## Out-of-view handling

An out-of-view ray gets **no image features**: its sampled features are zeroed and `in_view = 0` is
an input. It is still predicted, from the global image context, its direction and the ego-motion, so
it acts as a learned prior. Its confidence is expected to be low, and the eval reports every figure
split in-view vs out-of-view.

When history is missing (startup, a telemetry discontinuity, a frame gap), the estimator outputs
`free = 1`, `conf = 0` and `history_ok = False`. It never duplicates frames to fill the stack.

An odometry-propagated obstacle map (memory for the rays below and beside the frame) is a follow-up.

## Labels: the training rule vs the diagnostic kinds (AC2)

Command: `drone-fly eyes label --capture <dir>`.

**Pose.** `load_capture` uses the preset's telemetry config at `world_scale` W (4.0), with the same
converter as the live state source. So every label pose is exactly the FlightState the estimator
gets live.

**Course and position.** The course is the capture's catalogue entry (UC71 `meta.json`
`catalogue_id`), converted with `ConversionConfig.world_scale == W`; any other value is refused. The
course frame is `(p − origin) / W`, a pure translation of the FlightState, so the raycast position is
`state.pos − origin / W`. Never apply `to_course_point` to a FlightState: it is already divided by W.

**Label (the training target).** This is `contract.raycast` against the harness obstacle list
(placed non-gate items as AABBs, the list the UC69 oracle uses). It covers **obstacles only**, never
gates or the floor, exactly as the 70 model's training sim; that was the orchestrator's ruling.
`ray_range_scale` s is honoured by dividing every length by s before the cast.

**Diagnostic `aux_free` / `aux_kind` (AC2's hit kind).** The same rays are cast against the full
geometry: oriented boxes, capsules, the obstacle AABBs, and the floor plane or heightfield (marched).
The kinds are `none / obstacle / gate / scene / item / floor`.

- When several surfaces are hit at the same distance, the kind takes this precedence: obstacle
  (the label's own hit), then floor, gate, scene and item.
- A placed-item hit counts as a `gate` when its hit point lies within a passage's `max(w, h) / 2`
  plus 1 m (Liftoff) of the passage centre.

`aux_*` is **never** a training target. It shows what the label leaves out.

**Example (LiftoffArena replay-01).** Of the rays where the label is clear, about 5,000 hit scene
geometry (mostly the arena ground and stands) and about 500 hit gate arches within range.

## Looming

Monocular depth from a single frame is weak, and parallax gives no depth at the focus of expansion,
which is exactly where the velocity-aligned cone points. Stacking K frames over a 0.4 s window lets
the network use **looming**: the growth of an obstacle's image between frames, i.e. its
time-to-contact. Range then follows from time-to-contact × speed, and the speed is an input.

## Time stride and derived rates

Captures run at about 6-12 fps and the live loop at up to 40 Hz, so history is picked by **time**.
For `j = 1 .. K−1` the history frame is the one nearest to `t − j·dt`, within `±tol`, strictly
earlier, in the same segment. A frame is never used twice.

The defaults follow the orchestrator's ruling:

- `dt = 0.2 s` and `tol = 50 %` of the stride (±100 ms).
- At 6 fps the manual Infinity Loop capture's frames sit 0.167 s and 0.333 s back, which still
  qualifies.
- The ~10 fps replays match exactly.

The actual offsets therefore vary, so they are fed to the net as `hist_dt` (K−1 values, seconds).
The body rates use the actual dt as well. `dt`, `tol` and `hist_dt_input` are recorded in the
weights meta (`input`), and the labels record `ego_dt_s` / `hist_tol_s`. A dataset refuses labels
whose ego stride differs from its own.

## Scale

`W = scaling.world_scale` (4.0 for `liftoff`) and `s = depth.ray_range_scale` (1.0). They give:

- a sensing range of `RAY_RANGE · s` = 4 training m = **16 Liftoff m**;
- `dist_m` (Liftoff metres) `= free · RAY_RANGE · s · W`.

Labels, the dataset manifest and the weights meta all record W, s and the label settings sha. The
dataset refuses mixed label hashes, and eval refuses weights trained on other label settings.

## Warning time (evidence for UC-73 AC10)

`warning_s = RAY_RANGE · s · W / speed`, with speed in Liftoff m/s. It is the time the sensing range
buys at the current speed. It is `inf` at or below 0.3 m/s, and those frames are excluded from the
stats. The labels, the manifest (per split and map) and the eval report it as p05/p50/p95 plus the
share of frames below 0.3 s and 0.5 s. The eval also reports the miss rate on frames below 0.5 s.

Measured on the 5 LiftoffArena replays (16 m range): p05 0.81 s, p50 1.08 s, p95 2.6 s. Under 0.5 %
of frames have less than 0.5 s.

So at these replays' speeds (about 17 m/s median) the 16 m range gives about 1 s of warning, more
than the 0.2-0.3 s the use case feared at full racing speed. The range change stays a follow-up
(UC-73 AC10), to be decided on the held-out eval.

## Dataset (AC3)

Command: `drone-fly eyes dataset --label-missing --build-cache`.

**Split.** The split comes from the catalogue's `maps.<map>.split`, never from a capture's meta.
`TrackLeakageError` is raised if a non-train map would land in `train`.

**Exclusions:**

- unverified alignment, in every split;
- a camera uptilt more than 2° from the preset's: kept on train maps only, flagged
  `uptilt-mismatch-train-only` (LiftoffArena replay-02 has a 30° uptilt);
- no history (`no-history`, a per-sample drop).

**Frame cache.** One resumable uint8 memmap per capture under `<data root>/cache/eyes/<key>/`, at
about 46 KB per frame. A disk guard refuses the build when the projected size plus 1 GB exceeds the
free space.

**Outputs.** `samples.npz` and `manifest.json` / `.md` under `<data root>/datasets/eyes/<id>/`. The
manifest records counts, hit shares, label vs aux kinds, drop reasons and warning-time stats per
split and per map.

## Commands

```
drone-fly eyes fov-report --preset liftoff [--capture <dir>]
drone-fly eyes label --capture <dir> [--preview 8] [--trackdata-root <dir>]
drone-fly eyes dataset --label-missing --build-cache [--captures-root <dir>]
drone-fly eyes train --dataset <dataset dir> [--epochs 40] [--batch 64] [--device cuda]
drone-fly eyes train --dataset <dataset dir> --out <run dir> --resume
drone-fly eyes eval --weights <run>/eyes.pt --dataset <dataset dir> --split held-out
drone-fly eyes bench --weights <run>/eyes.pt --with-gate-detector
```

`--preset liftoff` is the default. Outputs land under the data root (`DRONE_FLY_DATA_ROOT`, else
`artifacts/harness/liftoff/training-data`):

| output | location |
|---|---|
| labels | `labels/eyes/<capture>/` |
| frame cache | `cache/eyes/` |
| datasets | `datasets/eyes/<id>/` |
| models | `models/eyes/<stamp>-<dataset id>/` (`eyes.pt`, `eyes.meta.json`, `training_log.json`, `resume.pt`) |
| FOV report | `reports/eyes/fov/` |

Eval and bench reports are written next to the weights.

**Training details:**

- AdamW with cosine decay, AMP on CUDA, and photometric + blur augmentation with no flips.
- The loss is smooth-L1 on `free`, ×4 on hit rays, plus a confidence BCE against
  `|err| < 0.5` training m.
- Model selection is on the best validation loss.
- A `STOP` file in the run folder, or Ctrl-C, ends the run at the last finished epoch. `--resume`
  is bit-identical and refuses a different dataset.

## GPU and disk notes

- The gate detector, the eyes and the connectome share the RTX 3050's 8 GB. `eyes bench` reports
  the eyes' and eyes + detector peak CUDA memory (null on CPU). The connectome footprint is not
  measured (follow-up).
- The latency budget is one 40 Hz tick (25 ms). The full-resolution frame copy and area resize
  dominate on CPU (about 15 ms per frame). On CUDA the resize runs on the GPU.
- The frame cache is about 46 KB per frame: the 16 UC71 captures (about 13k frames) need about
  0.6 GB.

## Known limits

- Motion blur at racing speed, lighting differences across maps, and untextured walls or floors
  (no texture to loom).
- Thin poles occupy few pixels at 192×80. The eval reports thin (< 0.2 m) obstacles separately.
- **Label fidelity.**
  - The label is the training raycast over the harness obstacle list, so it ignores gates, scene
    geometry and the floor, exactly like the sim the student was trained in.
  - The aux channel and `--preview` overlays show the difference.
  - Liftoff geometry from XML plus the scene caches may miss decorative meshes or include
    invisible colliders.
- **Replay captures.** Replay poses are 10 Hz-interpolated, `gyro = 0`, with finite-difference
  velocity. The ego rates come from attitude deltas, so they do not depend on the gyro.
- **Capture rate.** The manual capture runs at about 6 fps, which motivated the 0.2 s stride.
- **Not produced here.** The 8-value `obstacle_vision` block is not produced (follow-up). The
  estimator is not yet a registered `free_space_estimator` part (UC-73).
- **Duplicated reader.** UC-71's capture tooling is not on this branch; `eyes.capture_meta` is a
  schema-compatible reader, to unify later.

## Results

_To be filled by the orchestrator after the GPU training run (`eyes train` on the RTX 3050, then
`eyes eval --split held-out` / `--split val` and `eyes bench --with-gate-detector`)._
