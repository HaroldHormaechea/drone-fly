---
plan_for: use-cases/68-track-library-multi-track-teacher.md
work_branch: feat/uc-68-track-library-multi-track-teacher
run: drone-fly-uc-68
approved: 2026-10-12
---

# UC68 approved implementation plan

Proposal v5 from analyst-68, approved by challenger-68 in round 5 of 6.

## Challenger verdict

**Verdict:** Approve.

**Strengths:**
- v5 is self-contained and resolves all 25 review issues; see its reply table.
- The harness list separates the 30-value harness prefix (harness-faithful) from the shape-exact teacher obstacle block.
- Supporting-surface rules make ground starts viable.
- `stale-build` plus geometric validation keeps the four uc02 caches usable.
- `allow-missing` plus `scene_weight` keeps the split independent of which caches exist.
- The AC8 protocol is concrete, with bounded effort (at most 2 full runs).
- One leakage choke point, plus a train-only parameter-source table.
- MultiCourseEnv leaves UC67 CourseEnv parity untouched.
- The UC69 handoff is explicit: CourseFrame, `tracks/teacher_obs_ref.py`, `gate_crossed`, `course_conversion`, and Infinity Loop = `liftoff-arena-04`, a train track with scene geometry.

**Issues found.** These are non-blocking Minors; the developer MUST apply them.
1. **[Minor] Keep route-gate frames out of the harness list.** Exclude every route-gate item's colliders from the harness obstacle list. They stay in full geometry for contact and for the obstacles block.
   - Otherwise the current gate's frame bars sit within max_half+0.15 of the pos→next-gate segment, and the next gate reads as occluded on almost every approach.
   - The legacy harness never treats gates as obstacles.
   - Test: a lookahead through an arch frame is not occluded.
2. **[Minor] Floor-kill rule vs slow take-offs.** If the pilot shows early `floor` outcomes on slow take-offs, measure clearance from the support top, as the legacy z<0.08 kill does, rather than tightening the rule.

**Recommendations:**
- Put the actual scanned split in the ADR.
- State up front, in the AC8 headline, that held-out has only 2 tracks with scene geometry.

**Orchestrator addenda, for the developer:**
- Try `tracks extract-scenes` for the 16 maps without a cache. It needs UnityPy, installed via `uv pip install --python`.
  - Report which maps extract, which validate, and which come out ambiguous.
  - Extraction only reads the installed Liftoff files and never launches the game. Do not launch Liftoff.
  - If extraction fails or is ambiguous, keep `no-scene` and continue. It must not block the run.
- Copy the four uc02 caches from `C:\dev\liftoff-uc02-run\trackdata\scenes\` into the gitignored scenes root (done by the developer).

## Approved amendments (challenger-68, after the v5 approval)

These are the analyst's late round-4 refinements. The challenger accepted them, and they override v5 where they differ.

1. **Harness obstacle list cap.**
   - The cap is O_h = max(64, ceil(p99 of the TRAIN tracks' harness-list sizes)), taken from `tracks stats --split train`. It replaces v5's a-priori H = 256.
   - A track over O_h keeps its items nearest the route; the overflow is counted in its stats.
   - O_h is recorded in `course_conversion` and in the ADR parameter-source table, with source "train stats".
2. **Spawn support offset.** The per-track spawn support offset (support top − XML spawn z) is recorded in `stats.spawn_support_offset_m`.
3. **Arch-frame test.** QA adds a fixture with an arch gate frame. Test: a lookahead through the arch frame is not occluded, which covers challenger Minor 1.

---

# Approved proposal (analyst-68 v5, verbatim)

UC68 consolidated proposal v5 from analyst-68. It is self-contained, merges v1 through v4, and answers the challenger's issues 1–25 and the user's decisions. It supersedes every earlier version.

WT = C:\dev\drone-fly-uc-68-track-library-multi-track-teacher (main 876ab8a).

Brief frontmatter:
- paths.production is `src/drone_fly/**`; paths.test is `tests/**`;
- profiles [];
- test command `uv run --extra dev pytest`. Locally that is C:\dev\drone-fly\.venv-cuda\Scripts\python.exe with PYTHONPATH=src. Never `uv sync`.

Numbering:
- UC69 = oracle teacher in Liftoff;
- UC70 = student;
- UC71 = detector;
- UC72 = Liftoff eval and fine-tune.

User decisions folded in:
- the teacher is a plain MLP (confirmed);
- the teacher must use extractable Liftoff obstacle geometry;
- the existing scene caches are the initial source;
- extracting more maps is optional.

## Analysis

### Local data (read-only)

**Trackdata** (C:\dev\drone-fly\artifacts\harness\trackdata):
- 92 track/race pairs over 20 maps.
- `index.json` maps track uuid → race uuid, environment, laps and route.
- `props.json`: format 1, game build 25599583, 148 item kinds. Each item lists its prefab colliders (box / capsule / sphere / mesh) with `trig` flags.
  - This is NOT byte-identical to C:\dev\liftoff-uc02-run\trackdata\props.json, which has the same size but different content.
  - drone-fly's own copy matches the current build, so it is the one used.

Tracks per map:
- LiftoffArena 7;
- AutumnFields, BardwellsYard, StrawBale, TheGreen, ThePit: 6 each;
- BandoCity, Hall26, Hannover, MarinaBay, MinusTwo, ParisDroneFestival, Permafrost, PineValley: 5 each;
- Surtur 4;
- Rustline and ShortCircuit: 3 each;
- HangarC03 and TheRussianWoodpecker: 2 each;
- DubaiLegends 1.

Infinity Loop is `LiftoffArenaTrack04_0001.xml` / `LiftoffArenaRace04_0001.xml` ("04 - Infinity Loop"). Its catalogue id will be `liftoff-arena-04`. It has 8 passages per lap over 7 unique checkpoints, and 4 laps.

**Routes.** All races parse with `parse_race_route`:
- 8–79 passages per lap (MarinaBay04: 79 passages over 48 unique checkpoints);
- 15 routes revisit a checkpoint;
- 10 routes branch (the first successor is followed);
- 4 routes are point-to-point;
- laps range from 1 to 4.

**Scale.**
- Spacing between passages ranges 0.1–212 m.
- Extents reach 1129 m.
- Altitude spans reach 575 m; terrain puts absolute z at 150–750 m on some maps.

**Route checkpoints (about 52 kinds).**
- About 1000 passages go through invisible volumes: CheckpointBoxFlexible01 (sized by `<scale>`), CheckpointBox<W>mX<H>m, and Sphere/Cylinder checkpoints (no colliders).
- The rest are visible gate props. Their `trig` collider is the opening, and its thin axis (x OR z) is the normal.

**Spawns.** 87 of 92 tracks have a SpawnPointSingle item.

**Scene caches.** Located at C:\dev\liftoff-uc02-run\trackdata\scenes\. Built 2026-08-30 by liftoff-flight-analyzer `scene.py`, format 1, game build 24687907. Each was identified with a REAL witness replay.

| Map | box / capsule / sphere | scene mesh (not emitted) | terrain | boxes with a half-extent over 10 m |
|---|---|---|---|---|
| HangarC03 | 6490 / 2599 / 930 | 577 | 0 | 171 |
| LiftoffArena | 10788 / 4631 / 63 | 844 (about 5 %) | 0 | 656 |
| PineValley | 11956 / 1826 / 1227 | 461 | 1 | 51 |
| Rustline | 75316 / 27178 / 526 | 3095 | 16 | 23 (bounds about 1.5 km; 11 MB of JSON) |

The other 16 maps have no cache. Building them needs UnityPy (not installed) plus the analyzer's bundle identification.

### Code constraints

- `sim/course.py` CourseEnv is legacy and pinned by the UC67 equivalence tests. It hard-codes:
  - NG_MAX 20 gates and MO 10 obstacles;
  - an ARENA radius of 16 m and a 3.5 m ceiling;
  - a floor at z<0.08;
  - a 28 s episode;
  - a yaw-only gate with a sphere pass rule;
  - identity spawn attitude.
- `harness/offline/track_xml.load_track_course` drops gate roll and pitch, the triggers, the spawn and the colliders.
- The harness's 30-value observation (`ContractObservationAssembler`, built with `make_obs` + `course_senses`) has no heading. Its occlusion test and `obstacle_vision` use obstacle centres and the largest half-extent.
- The K1 actor's frozen body sees only obs12.
- `ppo.py` has no resume.
- The harness's `world_scale: 4.0` is a provisional hack from the 07 era.
- The gremlin micro profile uses real Liftoff metres: about 4.2 g, terminal speed about 35–40 m/s.

## Proposed Solution

### D1. Teacher: an MLP actor-critic trained from scratch (confirmed by the user)

- **Actor:** 256-256-128 ELU, with fixed per-block input scaling inside the module (recorded in the schema). Gaussian head on the raw v2 action.
- **Critic:** asymmetric, 512-512. It gets the teacher obs plus critic-only features that are never in the schema:
  - the 18-value dynamics sample;
  - the timing sample;
  - `t / time_limit`;
  - route progress fraction.
- There is no warm start from 07. The connectome lives in the UC70 student.
- `actor.kind: k1-reservoir-aug` stays selectable, for ablations only.

### D2. Observation schema `teacher-obs` v1: 148 float32, raw and exact

**Course frame.** The contract frame (Z-up, right-handed, metres), translated so the spawn support point is the origin, then divided by world_scale. It is never rotated.
- It is ONE transform, `CourseFrame` in `sim/pool.py`, holding `origin_contract` and `world_scale`:
  - `to_course_point(p) = (p − origin) / world_scale`;
  - `to_course_vector(v) = v / world_scale`;
  - plus the inverses.
- The UC69 oracle calls it.
- The body frame is FLU.

**Blocks:**
1. `harness30` [0:30) — exactly `ContractObservationAssembler.assemble(..., extra_dim=30)`.
   - obs12 comes from the contract's `make_obs`.
   - `curr_gate` and `next_lookahead` come from `course_senses` over `SimCourse.to_harness_course()`:
     - yaw is the heading of the travel-oriented normal;
     - aperture is min(w, h);
     - no occlusion noise.
   - Occlusion and `obstacle_vision` use ONLY the course's **harness obstacle list** (D4).
   - The schema pins:
     - the extra layout names and widths;
     - the contract obs source and version;
     - the `course_senses` constants (max half-extent + 0.15; t in (0.02, 0.98); float32);
     - the harness-list rule.
2. `attitude6` [30:36): the first two columns of R(body→world).
3. `vel_body` [36:39).
4. `gates` [39:78): 3 passages (target, next, next-next, per RouteProgress) × 13 values:
   - centre relative to the drone, in body frame (3);
   - travel-oriented normal, body frame (3);
   - up vector, body frame (3);
   - w and h in metres (2);
   - distance in metres (1);
   - valid (1), which is 0 past the finish.
   - For volumes, the normal is the route direction and w/h are the cross-section.
5. `obstacles` [78:142): the 8 nearest primitives by SURFACE distance, from the full shape-exact geometry (scene + props + floor/terrain; D4) × 8 values:
   - nearest surface point relative to the drone, body frame (3);
   - distance (1);
   - outward normal, body frame (3);
   - valid (1), which is 0 beyond `obs_range_m` = 12.
6. `altitude` [142]: z minus the support or floor height under the drone (D3).
7. `prev_action` [143:147): the raw policy output of the previous tick, BEFORE contract decoding, clipping or arm override. Identical in sim and harness. Zeros at episode start.
8. `arm_remaining_s` [147]: seconds until the motors accept commands.
   - In the sim: `max(0, arm_until − t)`.
   - In the harness: the remaining arm-confirm and settle time, `max(arm_settle_s, arm_press_s)` counted down from the arm press.
   - 0 once armed, and 0 for air starts.

Every block can be built live from the XML, the scene and props caches, telemetry and harness state.

The schema is defined in `sim/teacher_obs.py` and written to the sidecar under `obs_schema`.

### D3. Courses: one stored lap, route progress, pass rule, supporting surfaces

`sim/pool.py` is NumPy-only, with no torch at import.

**`SimCourse`:**
- `frame`;
- the per-lap route, with revisits unrolled WITHIN the lap;
- each passage:
  - centre, rotation, kind (`opening` or `volume`);
  - for openings, w/h and a convex outline of up to 16 vertices;
  - for volumes, shape and half-size;
  - class id;
- laps and a circuit flag;
- spawn pose in the course frame, plus the original contract-frame pose and a `synthetic` flag;
- floor plane z;
- an optional terrain heightfield;
- primitives: full geometry, plus the harness-list subset flag;
- cell table;
- bounds;
- time limit;
- meta (id, source, split, map, scene status, flags).

**`RouteProgress`** (NumPy reference, with a batched torch twin):
- A circuit is `laps × n + 1` crossings: it starts by crossing passage 0 and finishes by re-crossing passage 0 after the last lap, as Liftoff scores it.
- A point-to-point route is n crossings.
- The target is route[k mod n].
- Lookahead is lap-aware and invalid after the final crossing.
- G = the pool's largest per-lap passage count, hard cap 128. A larger track becomes `unconvertible: too many passages (<n>)` at scan time; it is never truncated.

**`gate_crossed`** (NumPy reference + torch twin):
- An `opening` counts only when the segment prev→pos crosses the plane from negative to positive along the travel normal, with the intersection inside outline + slack.
- A `volume` counts when the segment ENTERS it with positive displacement along the route direction.
- Doubling back never counts.
- Slack follows a curriculum during training and is **0 in every eval**.

**Supporting surfaces and ground start:**
- **(a) Spawn support.** At conversion, a ray is cast down from the spawn +0.5 m against the solid primitives.
  - The first top face within ±1.0 m of the XML spawn z becomes the support. Its top z is the origin z, so the plant's `ground_z = 0` sits exactly on the support's top face.
  - If no face is found, the XML spawn z is used and the track is flagged `spawn-unsupported`.
- **(b) Grounded phase.** While `grounded`, and until lift-off + 0.3 s, contacts with any primitive whose top face lies within 0.05 m of the support plane are ignored; the plant's ground rule owns them.
- **(c) After lift-off:**
  - Being within `FLOOR_KILL` 0.08 m above an upward-facing face (outward normal z > 0.7) of a solid, the terrain heightfield or the floor plane ends the episode with outcome **floor**. These are the legacy floor semantics, measured as vertical clearance, not `drone_radius`.
  - Any other contact within `drone_radius_m` (default 0.12) of a primitive ends it with outcome **collision**.
- **(d) Floor plane.** `floor_z = min(0, lowest opening bottom)` is the backstop on every map. The plant's `ground_z` stays 0, so a raised spawn platform can legitimately be flown off below 0. This is intended.

**`CoursePool`:**
- Pads to G per-lap passages and P primitives.
- Concatenates the cell tables with offsets.
- The group mix comes from `tracks.mix`, e.g. [{select: liftoff:train, weight: 0.6, scene_weight: 2.0}, {select: procedural:proc-liftoff, weight: 0.4}].
- At a masked reset, an env draws a group and then a course, all on device from the plant's generator:
  - uniformly inside the group;
  - tracks with a usable scene get weight × `scene_weight`.
- The EpisodeLog gains `group` and `course_id`.
- `tracks/courses.json` maps each id to its source, split, map and scene status.
- `metrics.jsonl` gets per-group completion.

**Environment.** A new `sim/multi_course.py` `MultiCourseEnv`. The legacy `CourseEnv` is NOT edited, so its equivalence holds by construction. MultiCourseEnv reuses Plant, ControlTiming, EpisodeLog and the constants, and adds:
- per-course bounds, floor and time limit;
- spawn yaw;
- route starts (`start.route_fraction`, training only);
- the D3 pass and contact rules;
- configurable rewards;
- a prev-action buffer;
- `teacher_obs()` and `critic_extra()`.

### D4. Geometry sources and conversion

**Scene caches.**
- The scenes root resolves as `--scenes-root` → `$DRONE_FLY_SCENES` → `<trackdata root>/scenes`. The default is `artifacts/harness/trackdata/scenes/`, which is gitignored.
- The recommended setup, documented in `tracks/README.md`, is to copy or symlink the four uc02 caches there.
- No absolute uc02 path appears in any committed file.
- The run snapshot records each cache's sha256 and its resolved source path.
- The format is the analyzer's (`colliders` in Unity world space; `counts`, `skipped`, `source`).

**Build-id mismatch.** All four caches are build 24687907, while the props and XML are 25599583.
- A mismatch is a WARNING plus a `stale-build` flag, and both ids are recorded.
- Validity is checked geometrically per map. The identification score is re-run against the current XMLs with a synthetic witness:
  - every opening centre ±0.5 m along its normal;
  - every spawn raised by 0.5 m.
- The pass condition is gates near ≥ 0.7 and witness points ≤ 0.5 % inside solids.
- Only a failed check excludes a scene, giving status `invalid`.

**Optional extraction (port, not depend).** The port goes in `src/drone_fly/tracks/liftoff_extract/`:
- `unityfs.py`: node sniff, pure-Python lz4, cstr;
- `transforms.py`: qmul/qrot, TRS compose, contains_point, path_inside;
- `scene.py`: discovery, `read_scene` and identification; raises `TrackDataError`;
- `props.py`: the props.json builder and the micro `opening()`;
- `game.py`: install lookup and build id.

It is about 700 lines, ported from liftoff-flight-analyzer commit d7250d5. Each file header keeps the MIT notice and the source path.

UnityPy:
- is an optional extra, `liftoff-extract = ["UnityPy==<pin>"]`;
- is imported lazily, only inside the extract functions;
- is installed with `uv pip install --python C:\dev\drone-fly\.venv-cuda\Scripts\python.exe UnityPy==<pin>`.

`drone-fly tracks extract-scenes [--map …] [--bundle M=<name>] [--witness M=<run dir>] [--force]`:
- runs only for maps without a valid cache;
- identifies scenes with a real `--witness` if given, else the synthetic witness;
- reads each candidate bundle once, scores every map, assigns uniquely, and marks a map `ambiguous` when the margin is small;
- writes to the scenes root.

`tracks extract-props` rebuilds props.json only if it is missing or stale.

Terrain: extraction attempts TerrainData heightmaps as a heightfield (2 m grid). This is best effort; on failure the map keeps the floor plane.

**Per-map scene status** (catalogue `maps.<Map>.scene`):
- `status`: `ok | stale-build | invalid | ambiguous | missing`;
- bundle; build_id; sha256;
- counts {box, capsule, sphere, mesh, terrain, trigger};
- flags:
  - `no-terrain`: terrain > 0 and no heightfield (PineValley and Rustline today);
  - `partial`: mesh share > 10 %;
  - `stale-build`.

`ok` and `stale-build` are usable.

**Maps without a usable scene** (`scene_policy: allow-missing`, the default) use:
- placed track items composed from props.json (real primitives, triggers non-solid);
- the `item_table` override;
- the floor plane.

They are flagged `no-scene`. `scene_policy: require` stays available.

**Primitives.**
- Types: box (OBB), capsule (centre, rotation, axis, half-length, radius), sphere.
- Unity → contract is ONE shared function. The rotation is conjugated by UNITY_TO_CONTRACT, which is safe for these mirror-symmetric primitives; the result then goes through CourseFrame.
- **Opening outlines and normals are NOT symmetric.** Their vertices are mapped as POINTS through the full mapping, including the handedness flip, then re-ordered convex. The normal is a vector, re-oriented along travel.

Primitive sources:
- (a) scene colliders;
- (b) placed items: XML pose (position, Euler Z-X-Y, `<scale>`) ∘ props colliders, composed as TRS;
- (c) the `item_table` override.

**Non-uniform scale under rotation.** The analyzer's TRS approximation is kept: component-wise scale, shear ignored, consistent with how the caches were built. Placed items whose `<scale>` is non-uniform AND whose collider is rotated in the prop frame get the OBB of the sheared corners. They are counted (`stats.sheared`).

**Never solid:**
- `trig` colliders: Checkpoint* volumes, TriggerBox_*, gate openings;
- Action and Spawnpoint blueprints.

Route-gate frames' non-trigger colliders ARE solid. A solid that intersects an opening is flagged, not dropped.

**Prop meshes** (decoded triangles) become an OBB in the prop frame, counted and flagged. Scene MeshColliders are only counted; they never become one big AABB.

**Openings.** The props `trig` collider goes through the ported `opening()`:
- thin → `opening` (normal axis x or z, w, h, centre, outline);
- thick → `volume`.

Checkpoint* volumes come from name × `<scale>`. Harness `gate_types` wins where it has an entry.

**Two obstacle lists per course:**
- **Full geometry:** every solid primitive plus floor/terrain. It feeds contact (D3) and the `obstacles` block.
- **Harness list:** feeds harness30 occlusion and `obstacle_vision`. It contains:
  - placed track items only (props-composed and item_table), never scene colliders;
  - converted to harness `Obstacle` (centre + AABB half-extents);
  - only items whose max half-extent ≤ `harness_obstacle_max_half_m` = 3.0;
  - only items inside the corridor;
  - capped at H = 256 by route distance, with the overflow counted.
- The harness list is evaluated over the WHOLE list with the real `course_senses` argsort semantics (a batched torch twin over H, parity-tested).
- The rule is recorded in `obs_schema` and `course_conversion`, so the UC69 oracle passes the identical list.
- This keeps the 30-value prefix meaningful: no 60 m floor box can occlude a gate.

**Cell tables (cost control).**
- Built only over the corridor: spawn + route ±25 m (`corridor_m`), in 4 m cells.
- Each cell lists up to C = 32 full-geometry primitives whose surface lies within `obs_range_m` (12 m) + the cell half-diagonal, nearest first.
- Primitives that no cell references are dropped from the pool.
- Overflow is counted.
- Per step: `searchsorted` to the cell, then exact signed distance / nearest surface over C.
- The heightfield is a per-course floor function sampled bilinearly. It is never a primitive, never in C, and never a giant box.
- Out-of-bounds means outside the course AABB + margin.

**Reference functions.** NumPy `obstacle_candidates`, `nearest_surface`, `floor_height` and `support_top` live in `sim/pool.py`. They are parity-tested against torch twins and reusable by UC69.

### D5. World scale = 1.0

- World scale is applied only through CourseFrame.
- It is recorded in the catalogue, the config and the sidecar.
- ADR rationale:
  - the plant is the measured gremlin in Liftoff metres, so any global factor breaks the geometry↔dynamics consistency;
  - a single factor can't fix the per-map spread anyway.
- Mitigations instead: per-track stats and flags, procedural courses sized from TRAIN stats, and the per-track eval.
- The harness's 4.0 is a 07 hack; UC69 must use the sidecar value.

### D6. Split (AC2): by map, filled by track share, independent of caches

- Maps are ordered by sha256("<seed>:<map>"), seed 68.
- Greedy fill: held-out until it has ≥ round(0.15·N) tracks, then val likewise, with ≥ 2 maps each. The rest is train.
- The split is a pure function of seed + maps. Scene-cache availability NEVER influences it.
- The header records method, seed, ratios and per-split map/track counts.
- On a re-scan:
  - existing maps keep their split;
  - new tracks inherit their map's split;
  - a new map is hash-bucketed by cumulative ratio.
- Simulated today (N = 92, target 14):
  - held-out = MinusTwo, TheRussianWoodpecker, ThePit, HangarC03 (15 tracks);
  - val = Surtur, DubaiLegends, Rustline, AutumnFields (14 tracks);
  - train = 12 maps (63 tracks), including LiftoffArena and PineValley.
- Tracks with a cached scene per split: train 12 (Arena 7 + PineValley 5), val 3 (Rustline), held-out 2 (HangarC03).
- So Infinity Loop (`liftoff-arena-04`) is a TRAIN track with full scene geometry. UC69's oracle comparison on it is therefore a train-track comparison, and the ADR and the UC69 handoff say so.
- Procedural specs are train only.

### D7. Leakage guard (AC9) and data-derived parameters

- Single choke point: `tracks.select(catalogue, selector, purpose)` + `tracks.loader.load_course`.
- `purpose="train"` raises `TrackLeakageError` for any non-train entry.
- `train teacher` and config validation use it. There is no in-training validation.

Parameter sources, in an ADR table (parameter → value → source):
- **Train stats only:** `drone-fly tracks stats --split train` aggregates over one split. The procedural `scale` / `aperture_m` ranges come from its spacing and aperture p10–p90.
- **Fixed a priori:**
  - `v_ref_mps` 4.0;
  - `corridor_m` 25, 4 m cells, C 32, `obs_range_m` 12;
  - `harness_obstacle_max_half_m` 3.0, H 256;
  - `scene_weight` 2.0;
  - outlier thresholds: spacing > 100 m, altitude span > 50 m, extent > 500 m, passages > 64, min aperture < 1 m;
  - reward weights (D8).
- **Later changes:** driven by val only.

The conversion rules necessarily saw every XML and prop. They decode geometry and are not policy tuning; the ADR says so.

### D8. `configs/teacher-liftoff.yaml` (extends proc-course-dr)

- drone `liftoff-gremlin-micro`; actor `mlp-teacher`; obs `teacher-obs-v1`.
- tracks:
  - mix: liftoff:train 0.6 (`scene_weight` 2.0) + procedural:proc-liftoff 0.4;
  - world_scale 1.0;
  - `scene_policy: allow-missing`.
- Randomization 1; timing random.
- start: mixed, ground_fraction 0.5, route_fraction 0.3, yaw from the course.
- Slack curriculum → 0 over the first half of the budget.
- Bounds: the course + 20 m.
- Time limit: route-based, v_ref 4 m/s, base 10 s, max 240 s.
- Collision and floor are terminal.
- Rewards (set a priori):
  - progress: 1.0 per metre toward the target passage;
  - gate +10;
  - finish +50;
  - crash −10;
  - upright 0.3 and spin 0.2, dense × dt/0.02;
  - action rate −0.01·‖Δa‖², dense.
- PPO: n_envs 8192, roll 64, gamma 0.995, lam 0.95, epochs 4, minibatches 16, lr 3e-4, learned log_std (init −0.5), budget 2e9.
- `checkpoint_every_s` 600.

### D9. Budget, long runs, AC8 protocol

**Throughput.** The UC67 env runs about 690k steps/s. Estimates for MultiCourseEnv:
- about 300–400k steps/s for the env;
- about 220–300k steps/s with PPO, idle, at 8192 envs, and about half that under contention;
- so 2e9 steps take about 2–2.5 h idle and 4–5 h contended.

`train bench --config teacher-liftoff --no-prototype` runs first, and its result goes in the ADR.

**Rustline memory.** The cell table covers the corridor only. For a route of about 2 km, a ±25 m corridor, an altitude span of about 60 m and 4 m cells:
- about 1×10⁵ occupied cells × 32 × int32 ≈ 12 MB per track, plus keys;
- primitives only where referenced, about 10⁴ × 20 float32 ≈ 1 MB;
- so ≤ 50 MB for Rustline's 3 tracks.

The bench reports the measured per-track pool memory.

**Long-run mechanics.**
- Checkpoints are written atomically every 600 s and every N iterations:
  - `teacher.pt` and `teacher_it<N>.pt`, each with its sidecar;
  - `resume.pt`: actor, critic, optimizer, counters, curriculum, and the torch and env RNG.
- `--resume <run>` reopens the run without rewriting config.yaml (n_envs and device may change), appends to the logs, and resets the envs.
- A `STOP` file or Ctrl-C saves and exits.
- `logs/progress.json` holds steps, fps and ETA.
- Logs are always CSV/JSONL, plus TensorBoard in `logs/tb/` when it can be imported.

**AC8 protocol.**
- **(a) Counting mode:** only `randomized` counts — randomization 1, timing random, ground_fraction 1, start at the spawn, mean action, slack 0. `nominal` is reported for information.
- **(b) Metric:** the headline is the macro mean, over all of a split's Liftoff tracks, of per-track full-race completion.
  - Train and held-out are reported separately; procedural courses are excluded.
  - Each split is also broken down by scene status: `scene` / `no-scene` / flagged (`no-terrain`, `partial`, `stale-build`).
  - `no-scene` completion is labelled optimistic (no walls or buildings).
  - Track and scene counts are printed beside each number.
- **(c) N and seeds:** N = 32 episodes per track, seed 1234. Val selection uses seed 4321.
- **(d) Checkpoint:** the one with the highest val macro completion in randomized mode, among the periodic checkpoints (about every 2.5e8 steps) and the final one.
- **(e) Stopping rule:**
  1. bench;
  2. pilot of 1e8 steps;
  3. run 1 (≤ 2e9);
  4. only if train < 95 % or val < 80 %: ONE val-driven change set, then run 2 (≤ 2e9);
  5. final train and held-out eval and report, whatever the result. At most 2 full runs.

**Report.** Per track:
- completion;
- gates passed (count and fraction);
- per-passage and per-lap success;
- lap and full times of completed runs;
- outcome shares: collision / floor / timeout / out-of-bounds;
- stats, flags and scene status.

If a target is missed, the report lists the blocking failure modes and whether they cluster on flagged or no-scene tracks. There is no held-out tuning.

### D10. Catalogue schema (AC1), uniform across sources

**Header:**
- `schema_version`, `source`;
- `root` (`--root` → `$DRONE_FLY_TRACKDATA` → header, relative to the repo root);
- `scenes_root` default;
- `world_scale`;
- `split{method, seed, ratios, counts}`;
- `maps{Map: {split, scene{status, bundle, build_id, sha256, counts, flags}}}`;
- `root_files{index.json, props.json: sha256 + build_id}`;
- `outlier_thresholds`.

**Entry:**
- `id` (`<map-kebab>-<NN>`);
- `source` (`liftoff-xml | procedural`);
- `track` / `race` (relative to the root; Liftoff only);
- `map`;
- `gate_types` (the route's sorted itemIDs/classes);
- `laps`;
- `split`;
- `status` (`ok | unconvertible: <reason>`);
- extras:
  - `name`, `uuids`, `sha256{track, race}`;
  - `stats`: passages, unique, branches, route_length_m, spacing min/median/max, extent, altitude span, aperture min/median, harness-list size and overflow, full-geometry primitives, cell overflow, sheared, mesh-flagged, and spawn real/synthetic/unsupported;
  - `flags`.
- `liftoff_menu`: reserved, optional, unvalidated, not implemented.

`tracks/procedural.yaml` uses the same entry schema:
- `source: procedural`, `split: train`, `status: ok`;
- no file references;
- `spec{seed_base, pool_size, ng, laps, tier, tunnels, variety, no_obs, scale, aperture_m}`.

`tracks list` and `tracks check` handle both sources.

What gets committed: names, hashes and derived stats only. Never coordinates, caches or absolute local paths.

### Components

**A. `harness/offline/track_xml.py`:**
- Adds `load_track_geometry` → `TrackGeometry`:
  - passages with contract-frame pose and scale;
  - placed items' pose, scale and xsi:type;
  - spawn items;
  - route meta.
- `load_track_course` is refactored on top of it with identical outputs.

**B. `src/drone_fly/tracks/`:**
- `catalogue.py`: schema, selectors, split, `select(..., purpose)`, `TrackLeakageError`.
- `scan.py`:
  - pairs tracks and races via index.json, falling back to `<Map>Track<NN>_<v>` ↔ `<Map>Race<NN>_<v>`;
  - converts each pair and computes stats, flags, scene status and validity;
  - writes `tracks/liftoff.yaml`, preserving splits and keeping unconvertible entries.
- `scenes.py`: scenes-root resolution, cache loading, build-id handling, geometric validation.
- `liftoff.py`: TrackGeometry + props + scene → SimCourse. It covers openings and volumes, primitives, the two obstacle lists, solid filters, support detection, terrain and plane, cell tables and CourseFrame.
- `procedural.py`: `sim.course_gen.sample_course` used verbatim, then scale / aperture_m applied.
- `loader.py`:
  - `load_course(entry, *, root, scenes_root, conversion)`, the single loader for both sources;
  - `build_pool(selectors, weights, purpose)`.
- `teacher_obs_ref.py`: the NumPy single-drone reference `build_teacher_obs(state, course, progress, prev_action, arm_remaining_s)`.
  - harness30 comes from the REAL `ContractObservationAssembler` + `course_senses` over `to_harness_course()` and the harness list.
  - The other blocks come from the `sim/pool.py` references.
- `liftoff_extract/`: as described in D4.
- `cli.py`:
  - `tracks list [--split] [--source]`;
  - `check [--root] [--scenes-root] [--require-scenes]`;
  - `scan --source liftoff [--root] [--scenes-root] [--seed]`;
  - `stats --split`;
  - `extract-scenes`, `extract-props`.

**C. `sim/pool.py`, `sim/multi_course.py`, `sim/teacher_obs.py`** (D2–D4). `sim/__init__.py` exports them. sim never imports harness or tracks.

**D. `controller/mlp_teacher.py`.** It implements the PPO actor protocol: `features` → (n, 0); `act`; `value`; a full state_dict.

**E. Training:**
- `config.py`:
  - a `tracks` section (mix, scene_weight, scene_policy, world_scale, roots, conversion parameters);
  - actor kinds; `obs`;
  - MultiCourseEnv sections;
  - `checkpoint_every_s`;
  - validation for purpose train.
- `ppo.py`:
  - env and actor factories (legacy unchanged);
  - teacher obs + critic extras;
  - resume / STOP / interrupt;
  - per-group metrics;
  - optional TensorBoard.
- `run_dir.py`:
  - reopen for resume;
  - a `tracks/` snapshot: catalogue.yaml, courses.json, and files.sha256 (XMLs, scene caches with resolved source paths, props) of train-loaded files only;
  - `resume.pt`, `eval/`.
- `sim/action.py` `write_sidecar` gains optional keys:
  - `obs_schema`;
  - `world_scale`;
  - `course_conversion`: version; item rules; opening, solid, support and harness-list rules; corridor/cell/C/obs_range/H/max-half; per-map scene sha256 + bundle + build_id + status; props sha256 + build_id.
- `evaluate.py`: the per-track engine and report (D9).
- `cli.py`:
  - `train teacher --drone --tracks --procedural [--config] [--resume] [--budget] [--n-envs] [--device] [--max-iterations] [--scenes-root]`;
  - `train eval <run|checkpoint> [--tracks] [--episodes] [--mode nominal|randomized|both] [--seed]`, which refuses an unknown schema or conversion. The legacy `--config` path is kept.
- `configs/teacher-liftoff.yaml` (D8).

**F. `cli/main.py`** registers `tracks`.

**G. Docs:**
- `docs/adr/0007-track-library-and-teacher.md` covers:
  - D1–D10;
  - port provenance;
  - the scene sources, build mismatch and validation;
  - the split, including where Arena and Infinity Loop fall;
  - the parameter-source table;
  - bench and memory;
  - results under the AC8 protocol.
- `tracks/README.md` covers:
  - every catalogue field for both sources;
  - selectors and the split;
  - roots and the cache setup (copy or symlink the uc02 caches);
  - extraction and the UnityPy install line;
  - flags and scene_policy;
  - "no game data committed".
- `docs/INFERENCE_CONTRACT.md` gets a section on teacher-obs v1, CourseFrame and the sidecar keys.

**Reference patterns:**
- `training/config.py:192-253`;
- `training/run_dir.py:41-105`;
- `sim/course.py:480-540`;
- `harness/assembler.py:58-88`;
- `harness/course_senses.py:52-148`;
- `harness/offline/track_xml.py:311-436`;
- `training/cli.py:370-404`;
- analyzer `sources/liftoff/scene.py:194-315`, `common/geometry.py:49-117` and `sources/liftoff_micro/props.py:160-196`.

## Files Affected

### Production (developer)

New:
- WT\src\drone_fly\tracks\__init__.py, catalogue.py, scan.py, scenes.py, liftoff.py, procedural.py, loader.py, teacher_obs_ref.py, cli.py
- WT\src\drone_fly\tracks\liftoff_extract\__init__.py, unityfs.py, transforms.py, scene.py, props.py, game.py
- WT\src\drone_fly\sim\pool.py, multi_course.py, teacher_obs.py
- WT\src\drone_fly\controller\mlp_teacher.py
- WT\src\drone_fly\training\evaluate.py, configs\teacher-liftoff.yaml

Modified:
- WT\src\drone_fly\sim\__init__.py, sim\action.py
- WT\src\drone_fly\harness\offline\track_xml.py
- WT\src\drone_fly\controller\__init__.py
- WT\src\drone_fly\training\config.py, ppo.py, run_dir.py, cli.py
- WT\src\drone_fly\cli\main.py
- WT\pyproject.toml (the optional extra only)

`sim/course.py` is untouched.

Developer-owned files outside paths.production:
- WT\tracks\liftoff.yaml, generated by `tracks scan` on the local data with the uc02 caches copied into the scenes root, then committed;
- WT\tracks\procedural.yaml and WT\tracks\README.md;
- WT\docs\adr\0007-track-library-and-teacher.md and WT\docs\INFERENCE_CONTRACT.md.

### Test (QA)

Fixtures in WT\tests\data\tracks\, synthetic only:
- 4 fake maps;
- XML pairs covering:
  - multi-lap + revisit, branch, point-to-point;
  - missing spawn, unknown checkpoint, a route over the passage cap;
  - named CheckpointBox, scaled Flexible;
  - gate props facing x and facing z;
- a mini index.json and props.json;
- `scenes/FakeMap*.json` containing:
  - a rotated box, capsules on each axis, a sphere, a trigger;
  - a 60 m floor box under a gate, and a raised platform under a spawn;
  - mesh and terrain counts, a small heightfield;
  - a stale build id.

Test files:
- `test_tracks_catalogue.py`: both sources; the README covers every field; selectors; list and check.
- `test_tracks_split.py`:
  - track-share quotas, ≥ 2 maps per split;
  - determinism and recorded counts;
  - stability, inheritance and new-map bucketing;
  - the split is unchanged when scene caches are added or removed.
- `test_tracks_scan.py`:
  - both pairings;
  - unconvertible reasons, including the passage cap;
  - splits preserved;
  - check catches missing or changed files;
  - scene status, including stale-build accepted and failed validation → invalid.
- `test_tracks_liftoff_convert.py`:
  - per-lap route;
  - opening/volume precedence;
  - Unity → contract primitives checked by point containment in both frames, plus an **asymmetric outline** checked the same way;
  - props placement with scale, and the sheared count;
  - the non-solid rules;
  - the harness list: no scene colliders, max-half cap, H cap;
  - support detection on a floor box and on a platform, and the unsupported flag;
  - the cell table is complete within range;
  - heightfield vs plane;
  - the `no-scene` fallback;
  - CourseFrame round trip;
  - conversion is reproducible from `course_conversion`.
- `test_tracks_liftoff_extract.py`:
  - TRS compose;
  - `read_scene` on a fake UnityPy stream (monkeypatched, never imported);
  - identification scoring, rejection, ambiguity and assignment;
  - the missing-UnityPy error;
  - `opening()`.
- `test_tracks_import_graph.py`: in a subprocess, imports `drone_fly.tracks.loader`, `drone_fly.sim.course`, `drone_fly.sim.multi_course`, `drone_fly.training.ppo`, `drone_fly.harness` and `drone_fly.tracks.teacher_obs_ref`, then asserts `"UnityPy" not in sys.modules`.
- `test_tracks_procedural.py`: determinism; equals course_gen × scale.
- `test_sim_course_pool.py`:
  - padding and masks;
  - mix and scene_weight frequencies;
  - course_id and group logging;
  - RouteProgress parity: laps, revisits, finish re-crossing, point-to-point;
  - gate_crossed parity: a fast crossing, outside the outline, backwards → no count, volume entry;
  - per-primitive distance parity; heightfield; bounds and time limit;
  - **a ground start on a floor box and on a raised platform does not collide; lift-off clears; later top-face contact → floor; side contact → collision.**
- `test_sim_teacher_obs.py`:
  - the layout;
  - batched == `tracks.teacher_obs_ref` on random states, including the route end and the lap wrap;
  - the harness30 prefix is built through the real assembler;
  - **a big floor box under a gate does not occlude the lookahead**;
  - prev_action and arm_remaining semantics.
- `test_training_teacher.py`: a 1–2 iteration CPU run with a tiny MLP; run-folder contents and sidecar keys; resume; STOP.
- `test_training_leakage.py`: a spy on `load_course` sees only train ids; val and held-out are rejected by the CLI and the config; the snapshot holds train files only.
- `test_training_eval_tracks.py`: JSON/MD shape, including per-passage, per-lap and the scene strata; outcome mapping; determinism; unknown schema or conversion refused.

These must stay green and untouched: test_sim_meteor75_equivalence, test_sim_course, test_sim_ground_start, test_harness_offline_eval, test_harness_race_route, test_sim_no_airframe_literals.

## Risks & Considerations
1. **AC8 feasibility.** A full race can require up to about 237 crossings, which needs a per-passage success rate of about 0.9998. The report shows per-passage and per-lap rates.
2. **Scene coverage.** Only 4 of 20 maps have scenes, covering 17 of 92 tracks.
   - Most tracks train as `no-scene` (placed props + floor plane), which is optimistic. This is stated and reported per stratum, and the eval report labels it.
   - Extracting more maps is optional and improves this.
3. **Stale scene builds.** All four caches predate the current game build; geometric validation guards against this.
4. **Terrain.** PineValley and Rustline have no extracted terrain (flag `no-terrain`, plane fallback).
5. **Scene identification without a real flight is unproven.** Mitigations: dense witnesses, unique assignment, the ambiguity margin and overrides. It only applies to optional new extractions.
6. **Throughput and memory.** Rustline is the stress case; the bench decides n_envs.
7. **Heading-blind live observation.** For UC70, adding attitude6 and vel_body to the student is recommended.
8. **Harness world_scale 4.0.** It is a 07 hack; teacher-lineage checkpoints need the sidecar value.
9. **Small held-out split.** About 15 tracks, of which only 2 have a scene, so results are noisy. They are reported per stratum.
10. **Branching routes** follow the first successor; counted in the stats.
11. **Committed catalogue.** Names, hashes and stats only; confirmed.
12. **Use-case pitfalls covered:**
    - Arena: present and has a scene.
    - Scale: D5.
    - Revisits and laps: D3.
    - The AABB assumption: superseded by D4.
    - 80 % held-out: Risk 1 and the stratified report.

## Reply table

| # | Issue | Resolution |
|---|---|---|
| 1 | AC8 protocol | D9 (a)–(e), plus scene strata |
| 2 | Data-derived parameters | D7: train-only stats command, a priori list, ADR source table; note on conversion rules |
| 3 | Split by track share | D6: 15 / 14 / 63, ≥ 2 maps per split, independent of caches |
| 4 | One stored lap; cap → unconvertible | D3: RouteProgress with finish re-crossing; G = per-lap max ≤ 128 |
| 5 | AC1 fields | D10, both sources, README test |
| 6 | harness30 pins and parity | D2 block 1, the D4 harness list, parity via the real assembler |
| 7 | Legacy identity | MultiCourseEnv by default; CourseEnv untouched |
| 8 | Pass rule | Direction rules; eval slack 0; ground_z/floor_z in D3(d) |
| 9 | World scale | D5 ADR rationale |
| 10 | Scene colliders (a)–(f) | D4: caches in gitignored artifacts; optional UnityPy + import test; never-solid rules; terrain heightfield or plane + flags; synthetic fixtures; scene sha/bundle/build/status in `course_conversion` |
| 11 | Reference placement | `tracks/teacher_obs_ref.py` |
| 12 | prev_action / arm definitions | D2 blocks 7–8 |
| 13 | Crossing direction | D3 gate_crossed |
| 14 | One course-frame transform | CourseFrame in `sim/pool.py`, called by the oracle |
| 15 | Degenerate harness30 | D4 harness list: track items only, max-half ≤ 3 m, H 256, full-list argsort; floor-box test |
| 16 | Supporting surfaces | D3 (a)–(d): support detection, grounded grace, top-face → floor, other contact → collision; tests |
| 17 | Denominator | Superseded by 22: allow-missing means no exclusion; strata are reported |
| 18 | Outline handedness | Outline points mapped through the full mapping; asymmetric-outline test |
| 19 | Non-uniform scale | The analyzer's TRS approximation; sheared OBBs counted |
| 20 | Import test | `test_tracks_import_graph.py` including harness; MIT headers |
| 21 | Build-id mismatch | Warning + `stale-build`; geometric validation decides; both ids recorded |
| 22 | Scene policy | allow-missing default with a props + plane fallback (`no-scene`); `scene_weight` affects sampling only; split unaffected; strata in eval; cached tracks per split 12 / 3 / 2 |
| 23 | Arena and Infinity Loop | `liftoff-arena-04` is a train track, converted WITH the LiftoffArena scene; UC69 compares on a train track (stated) |
| 24 | Cache root | `--scenes-root` → `$DRONE_FLY_SCENES` → `<root>/scenes`; copy/symlink setup documented; no absolute paths committed; the snapshot records sha + source path |
| 25 | Rustline scale | Corridor-only cells; estimated ≤ ~50 MB for its 3 tracks, measured in the bench; PineValley and Rustline flagged `no-terrain` |
