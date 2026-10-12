# ADR 0007: Track library and multi-track privileged teacher

- **Status:** accepted (UC-68)
- **Date:** 2026-10-12
- **Builds on:** [ADR 0006](0006-profile-driven-batched-sim.md) (batched simulator, action contract
  v2, ground starts), [ADR 0005](0005-drone-profiles-and-sysid.md) (drone profiles),
  [INFERENCE_CONTRACT.md](../INFERENCE_CONTRACT.md) §7 (teacher-obs v1, CourseFrame, sidecar keys)
- **Code:** `src/drone_fly/tracks/` (`catalogue`, `scan`, `scenes`, `liftoff`, `procedural`,
  `loader`, `teacher_obs_ref`, `cli`, `liftoff_extract/`), `src/drone_fly/sim/` (`pool`,
  `multi_course`, `teacher_obs`), `src/drone_fly/controller/mlp_teacher.py`,
  `src/drone_fly/training/` (`evaluate`, `config`, `ppo`, `run_dir`, `cli`,
  `configs/teacher-liftoff.yaml`), `harness/offline/track_xml.py` (`load_track_geometry`)
- **Data:** `tracks/liftoff.yaml`, `tracks/procedural.yaml` (committed; see
  [tracks/README.md](../../tracks/README.md)); `artifacts/harness/trackdata/` (XMLs, `index.json`,
  `props.json`, `scenes/`) and `artifacts/training/<drone>/<run>/` (gitignored)

## Context

The policies trained up to UC67 memorised one course and tumbled in Liftoff. UC68 makes "a course"
data-driven and trains a **privileged teacher** that generalises across courses. It sees exact gate
poses and obstacle geometry, which the student (UC70) will later learn to replace with perception.
The teacher must reach ≥ 95 % mean completion on the training tracks and ≥ 80 % on held-out tracks.
Held-out tracks are on maps the teacher never saw.

The local Liftoff data has 92 track/race pairs over 20 maps. Routes have 8–79 passages per lap and 1–4
laps. Gaps between passages range from 0.1 m to 212 m, and some maps put terrain hundreds of metres
up. About 1000 route passages go through invisible checkpoint volumes rather than visible gates.

## Decision

### D1. Teacher: an MLP actor-critic from scratch

- **Actor:** 256-256-128 ELU with fixed per-value input scaling inside the module, and a Gaussian
  head on the raw v2 action with a learned `log_std`.
- **Critic:** 512-512, asymmetric. It also sees critic-only features: the 18-value dynamics sample,
  the timing sample, `t / time_limit` and the route-progress fraction.
- There is no warm start: the connectome belongs to the UC70 student. `k1-reservoir-aug` remains
  selectable for ablations.

### D2. Observation `teacher-obs` v1

148 raw float32 values in the course frame. The layout, the CourseFrame and the sidecar keys are
specified in [INFERENCE_CONTRACT.md §7](../INFERENCE_CONTRACT.md).

- The first 30 values are exactly what the live harness assembles, so the student and the UC69
  oracle can reuse them.
- The other blocks are attitude, body velocity, three passages ahead, the 8 nearest surfaces,
  altitude, the previous action and the arm countdown.
- The schema is written into every checkpoint's sidecar.

### D3. Courses, route progress, pass rule, supporting surfaces

**Courses.**
- `SimCourse` (`sim/pool.py`, NumPy only) stores one lap. Revisits within the lap are unrolled.
- A circuit is `laps × n + 1` crossings: it ends by re-crossing passage 0, as Liftoff scores it. A
  point-to-point route is `n` crossings.
- A route with more than 128 passages per lap is `unconvertible`, never truncated.

**Pass rule.**
- An opening counts when the step segment crosses its plane in the travel direction, inside the
  outline plus a slack.
- A volume counts when the segment enters it moving along the route. Doubling back never counts.
- The slack anneals from 0.5 m to 0 over the first half of training, and is 0 in every evaluation.

**Ground starts on real surfaces.**
- At conversion, a ray from the spawn finds the supporting top face. Its top is the course origin,
  where the plant's `ground_z = 0` sits.
- While grounded, and until 0.3 s after lift-off, contact with the support is ignored.
- After that, being within 0.08 m above an upward face (or the floor plane or terrain) is `floor`,
  and any other contact within `drone_radius_m` 0.12 is `collision`.
- The plant rests the drone at 0.12 m, so a slow take-off that stays at rest height is not a floor
  crash. Only actually coming back down is.

**Environment.** `sim/multi_course.py` `MultiCourseEnv` holds a `CoursePool`.
- At each reset, an env draws a group by its mix weight, then a course inside the group (scene tracks
  weighted ×2).
- `CourseEnv` (UC67) is untouched, so its equivalence tests hold by construction.

### D4. Geometry sources and conversion

Each Liftoff course combines up to three geometry sources, all converted from Unity to the contract
frame by one shared function:
- the map's **scene cache**, when one is usable;
- **placed track items**: XML pose ∘ `props.json` prefab colliders;
- the **floor plane** `min(0, lowest opening bottom)`, or the terrain heightfield where one was
  extracted.

Primitives are oriented boxes, capsules and spheres. Trigger colliders, checkpoint volumes and
spawn/action blueprints are never solid. Prop meshes become flagged OBBs. Scene mesh colliders are
counted, not emitted.

**Gate frames that only approximate their shape (conversion v2).** Some gate props have a mesh as
their frame collider (`LightGate300x220cm*`), or ship a whole-gate `approx` box (`AirgateLiftoff*`,
`AirgateBig*`). Turned into one OBB, that box fills the gate's own opening: half-extents 1.81 × 1.41
× 0.27 m with the opening centre 0.27 m inside. This was found after the pilot. 54 passages on 12 of
the 63 train tracks were impassable, including Arena, Hannover and Bando tracks, which capped
completion and taught the policy to avoid gates.

Now, when such a solid's prefab-frame AABB contains the prop's own trigger opening, it becomes a
ring of up to 4 bars around the opening rectangle. The bars use the solid's own extent and
thickness, so the frame stays solid while the opening is clear. `CONVERSION_VERSION` went to 2,
which invalidates the conversion caches, and the catalogue was re-scanned with its splits
unchanged.

After the fix, 9 train passages still have a solid within 0.12 m of their centre. Eight are
checkpoint volumes with a real object inside, and a volume counts on entry anywhere, so they still
pass. The ninth is the centre divider of a double gate.

**Two obstacle lists:**
- The **full geometry** feeds contact and the `obstacles` block. Per course, it is indexed by a
  corridor-only cell table: spawn and route ±25 m, 4 m cells, up to 32 nearest primitives per cell.
- The **harness list** feeds `harness30`. It holds placed items only, never scene colliders or
  route-gate frames. Items must have a max half-extent ≤ 3.0 m, lie inside the corridor, and be
  among the O_h nearest the route.
  - O_h = max(64, ceil(p99 of the train tracks' list sizes)) = **722**.
  - This keeps the harness-compatible prefix meaningful: no 60 m floor box or gate frame can occlude
    a gate.

**Scene caches.**
- Four caches came from liftoff-flight-analyzer (HangarC03, LiftoffArena, PineValley, Rustline).
  They were built for game build 24687907, while the XMLs and props are build 25599583.
- A build mismatch is a warning plus the `stale-build` flag. Validity is decided geometrically,
  against a synthetic witness from the current XMLs (every opening centre ±0.5 m, spawns +0.5 m). A
  cache needs ≥ 70 % of its gates near the witness and ≤ 0.5 % of witness points inside solids. All
  four passed.
- **Extraction** (`tracks extract-scenes`) was run for the 16 maps without a cache:
  - 5 extracted and validated on the current build: BandoCity, DubaiLegends, Hannover,
    ParisDroneFestival, ShortCircuit;
  - MinusTwo came out ambiguous and was left without a scene;
  - the other 10 maps had no identifiable scene.
  - On the 4 maps with known caches, identification never picked a wrong bundle.
- The extractor is a port of about 1200 lines from liftoff-flight-analyzer commit d7250d5 (MIT; each
  file keeps the notice and its source path). UnityPy is an optional extra (`liftoff-extract`,
  1.25.4), imported lazily only inside the extract functions.

Today **9 of 20 maps have a usable scene**.

| split | tracks with a scene | tracks without |
|---|---|---|
| train | 30 | 33 |
| val | 4 | 10 |
| held-out | 2 | 13 |

Maps without a scene train as `no-scene`: placed props plus the floor plane. That is optimistic,
because there are no walls, buildings or terrain, and evaluation reports it as a separate stratum.

### D5. World scale 1.0

The plant is the measured gremlin micro in Liftoff metres. Any global scale factor would break the
consistency between geometry and dynamics, and no single factor could fix the per-map spread of
scales anyway. The scale is applied only through `CourseFrame` and recorded in the catalogue, config
and sidecar. The live harness's `world_scale: 4.0` is a hack from the 07 era; UC69 must use the
sidecar value.

### D6. Split by map

- Maps are ordered by `sha256("68:<map>")`. Held-out fills first, then val, each to
  `round(0.15·N)` tracks and at least 2 maps. The rest is train.
- The split does not depend on which scene caches exist.
- The scanned split:
  - **train:** 12 maps, 63 tracks;
  - **val:** AutumnFields, DubaiLegends, Rustline, Surtur (14 tracks);
  - **held-out:** HangarC03, MinusTwo, ThePit, TheRussianWoodpecker (15 tracks).
- **Infinity Loop** (`liftoff-arena-04`) is a **train** track with full scene geometry, so UC69's
  oracle comparison on it is a train-track comparison.
- Procedural specs are train only.

### D7. Leakage guard and parameter sources

`tracks.select(catalogue, selector, purpose="train")` is the single choke point used by `train
teacher` and by config validation. It raises `TrackLeakageError` for any `val` or `held-out` entry.
There is no in-training validation.

| parameter | value | source |
|---|---|---|
| procedural `scale` | 6.1–25.0 (leg spacing p10 14.6 m / p90 60.1 m ÷ course_gen's 2.4 m mean chord) | `tracks stats --split train` |
| procedural `aperture_m` | 2.2–10.0 (aperture p10 / p90) | `tracks stats --split train` |
| harness-list cap O_h | 722 (p99 721.88 over 63 train tracks) | train stats |
| `corridor_m`, cell size, C, `obs_range_m` | 25 m, 4 m, 32, 12 m | a priori |
| `harness_obstacle_max_half_m` | 3.0 | a priori |
| `scene_weight` | 2.0 | a priori |
| outlier thresholds | spacing > 100 m, altitude span > 50 m, extent > 500 m, passages > 64, aperture < 1 m | a priori |
| `v_ref_mps`, time limit | 4.0 m/s; 10 s + route / v_ref, at most 240 s | a priori |
| rewards | see D8 | a priori (rebalanced before the pilot) |
| route-start clearance | 0.3 m beyond `drone_radius_m` and above the floor | a priori (fixed after the pilot, see D9) |

The conversion rules necessarily saw every XML and prop: they decode geometry, they do not tune the
policy. Any later change is driven by val only.

### D8. Training config `teacher-liftoff`

- **Drone and actor:** `liftoff-gremlin-micro`, `mlp-teacher`, `teacher-obs-v1`.
- **Tracks:** a mix of 60 % `liftoff:train` (scene ×2) and 40 % `procedural:proc-liftoff`, with
  `allow-missing`.
- **Randomization:** dynamics randomization 1 and random timing (20–50 Hz, delays, skipped ticks).
- **Starts:** 50 % ground. 30 % of the air starts begin 2–6 m before a random route passage.
- **PPO:** 8192 envs, rollouts of 64, γ 0.995, λ 0.95, 4 epochs, 16 minibatches, lr 3e-4, `log_std`
  initialised at −0.5, budget 2e9.

**Rewards.** All set a priori:
- progress: 1.0 per metre toward the target passage;
- gate +10, finish +50, crash −10;
- dense terms, × dt / 0.02: upright, low spin, and action rate −0.01·‖Δa‖².

**Procedural obstacle sizes (fixed after the pilot).** `course_gen` obstacles are sized for its
0.45–1.5 m gates. Multiplying their sizes by the course `scale` (6–25) produced 35–66 m blocks, and
procedural episodes ended in collision 93.5 % of the time. Now positions and vertical sizes scale with
`scale`, and horizontal sizes scale with the course's mean aperture ratio (mapped / original), so
obstacles keep their size relative to the gates. The blocking obstacle that `course_gen` puts on the
straightest leg is intended: the policy must fly around it.

**Pre-pilot rebalance of the dense terms.** The approved plan had upright 0.3 and spin 0.2. These
were changed to **0.005 / 0.005** before any training run, so this is not val tuning.

| | before (0.3 / 0.2) | after (0.005 / 0.005) |
|---|---|---|
| hovering, or sitting upright on the pad | 25 /s | 0.5 /s |
| flying the route at 2 m/s: dense + 2 /s progress + gates at the train median spacing of 31 m (0.64 /s) | ≈ 27.6 /s | ≈ 3.1 /s |
| flying at 5 m/s | ≈ 30.6 /s | ≈ 7.1 /s |
| discounted dense stream forfeited by a crash (γ 0.995 at 20–50 Hz) | ≈ 100–250 | ≈ 1–2.5 |

- **Before:** flying earned only about 10 % more than hovering, and a crash forfeited far more than
  its −10 penalty. With time limits up to 240 s, "never take off" was a strong optimum.
- **After:** hovering earns about 6× less than slow flight. The crash penalty dominates what a crash
  forfeits, so there is no incentive to end episodes early either.
- A per-second time penalty and a penalty-form upright term were rejected. Both would pay a
  tumbling policy to crash early.

### D9. Budget, long runs and the AC8 protocol

**Long runs.**
- Checkpoints are written atomically every 600 s and about every 2.5e8 steps:
  - `teacher.pt` and `teacher_it<N>.pt`, each with its sidecar;
  - `resume.pt` (models, optimiser, counters, curriculum, RNG).
- `--resume <run>` continues a run, and a `STOP` file or Ctrl-C saves and exits.
- `logs/progress.json` holds steps, fps and ETA.

**AC8 protocol.**
- **Counting mode.** Only `randomized` counts: dynamics randomization 1, random timing, ground starts
  at the spawn, mean action, slack 0.
- **Metric.** The headline is the macro mean of per-track full-race completion, with train and
  held-out reported separately. Each is broken down by scene stratum.
- **Seeds.** N = 32 episodes per track with seed 1234; val selection uses seed 4321.
- **Checkpoint.** The one with the best val macro completion.
- **Effort.** At most 2 full runs: bench → 1e8 pilot → run 1 (≤ 2e9) → one val-driven change set and
  run 2 only if a target is missed → final train and held-out eval, reported whatever the result.

**Bench** (`train bench`, 8192 envs, RTX 3050 Laptop shared with other GPU load):
- 195,507 env-steps/s, env only;
- pool 167 MB on the GPU; peak 412 MB.

<!-- RESULTS:PILOT -->

<!-- RESULTS:RUN1 -->

### D10. Catalogue schema

The full schema is in [tracks/README.md](../../tracks/README.md). Both sources share one entry schema
(`id`, `source`, `track` / `race`, `map`, `gate_types`, `laps`, `split`, `status`, `stats`,
`flags`). Only names, hashes and derived stats are committed. Unconvertible pairs are kept with their
reason, and splits stay stable across re-scans.

## Consequences

- Any new local Liftoff track is one `tracks scan` away from training, with its split fixed by its
  map.
- Most tracks still lack scene geometry. Their completion numbers are optimistic, and the eval
  report labels them as such. Extracting more scenes improves fidelity without changing the split.
- Held-out has only 2 tracks with scene geometry (HangarC03), so the held-out result says little
  about flying among buildings.
- UC69 reuses `CourseFrame`, `tracks.teacher_obs_ref`, `gate_crossed` and the sidecar's
  `course_conversion` to run the teacher live with oracle gates.
- For UC70, the student should get `attitude6` and `vel_body` as well, because the live 30-value
  observation has no heading.
