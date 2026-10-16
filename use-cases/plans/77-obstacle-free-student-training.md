---
plan_for: use-cases/77-obstacle-free-student-training.md
work_branch: feat/uc-77-obstacle-free-student-training
run: drone-fly-uc-77
approved: 2026-10-15
---

# UC-77 implementation plan: obstacle-free student training

Challenger (challenger-77) verdict: **Approve**, no Critical or Major issues. The orchestrator's v1.1 decisions are folded in as N4 and N5. The user directed an autonomous run, so there was no plan-approval gate.

TARGET_DIR = C:\dev\drone-fly-uc-77-obstacle-free-student-training

## Analysis
**Where obstacle geometry enters the sim.** It comes in two ways:
- `SimCourse.prims` + `cells` drive contact (`sim/multi_course.py:865-873`). They also feed the teacher's privileged `obstacles` block (8 nearest surfaces) and its `altitude` field.
- `SimCourse.harness` feeds the teacher's `harness30.obstacle_vision`.

So obstacle-free means filtering prims and emptying the harness at course-conversion time. No env or obs code changes.

**Where the gate/non-gate distinction lives.** In `tracks/liftoff.py:723-746` the scene prims are concatenated first, then placed items with `item_slices (a, b, is_gate)`. The gate flag exists ONLY there: `Primitives.source` cannot tell gate frames from other items. So the filter must be inside `convert_liftoff`.

**Procedural courses.** Gates have no prims, so every primitive is an obstacle and is also on the harness list.

**Why the flag can't go in ConversionConfig.** It must stay out, because the student enforces the teacher sidecar's conversion fingerprint (`training/evaluate.py:89-101`). The cache key gets an extra part only when the mode is not the default.

**Infinity Loop (`liftoff:liftoff-arena-04`)**, measured read-only:
- 6113 prims: 5773 scene, 221 item, 119 table.
- 247 harness items, which are all of the placed non-gate items.
- Floor and bounds:
  - floor_z is -0.865 (the gate aperture bottoms);
  - the real arena floor is a scene collider with top about 0;
  - bounds z runs from -22 to 24.

The 247 props:

| Item | Count | Distance from route | Top z |
|---|---|---|---|
| TlTrackLight01 | 110 | — | -0.67 (buried) |
| ConradCrowdControlBarrier01 | 94 | about 2.9 m | 0.16 |
| AirflagLiftoffSlalomDark01 | 24 | 13-16 m | 1.78 |
| ASLLightBeam04 | 8 | — | below the floor (buried) |
| AdvertBoard | 6 | — | 0.45 |
| LightConeRed01 | 4 | about 2 m | 2.16 |
| ConcreteBarrier01 | 1 | — | 0.0 |

There are also 7 gate arches (35 prims).

So the corridor is effectively open. Dropping scene colliders would delete the real ground, which is why scene colliders are kept.

## Obstacle definition
An obstacle is any placed track item that is not a route-gate frame: the harness-list population before its size, corridor and cap filters. On procedural courses, every primitive is an obstacle.

KEPT:
- scene colliders (floor, arena structures, stands);
- route-gate frames;
- heightfield, floor_z and bounds (the sim's only "ceiling" is bounds_hi.z);
- corridor, passages, spawn and time limit.

## Proposed Solution (by file)
1. **`src/drone_fly/tracks/liftoff.py`**
   - Add an OBSTACLE_MODES constant ("all", "none") and an `obstacles="all"` keyword on `convert_liftoff`. It is NOT a ConversionConfig field.
   - "none" mode:
     - compute spawn support and origin on the full prims_c (so spawn, origin, floor_z, passages, bounds and time limit are identical to the default);
     - then keep only the scene slice plus the is_gate item slices, before the course-frame transform;
     - set an empty HarnessList and build cells from the reduced prims;
     - add stats `obstacles: "none"` and `obstacles_removed` ONLY in "none" mode.
   - An unknown mode raises TrackDataError.
   - Update the docstring.
2. **`src/drone_fly/tracks/procedural.py`** — `convert_procedural(..., obstacles="all")`. "none" means no obstacle prims and an empty harness, built the same way as `no_obs`, so the gate and start draws are unchanged. Same stats as Liftoff.
3. **`src/drone_fly/tracks/loader.py`**
   - `load_course` and `build_pool` take `obstacles="all"` and forward it.
   - Both cache keys (line 162 for procedural, line 200 for Liftoff) get an `obstacles` part only when the mode is not "all".
   - `conversion_record` is unchanged.
4. **`src/drone_fly/training/config.py`**
   - `TracksConfig.obstacles: str = "all"`, validated in `__post_init__`; a ValueError becomes TrainConfigError.
   - One top-level flag; no per-mix-entry flag.
   - `DistillSection` gains `track_evals: tuple[str, ...] = ()` and `final_report: bool = False`.
5. **`src/drone_fly/training/ppo.py`** — `build_teacher_pool` (line 334) forwards `cfg.tracks.obstacles`.
6. **`src/drone_fly/training/distill.py`**
   - `_train_student` (line 1179) and `bench_student` (line 1915) forward the mode. Augmentation works unchanged.
   - `eval_pool` (line 615) takes an optional `obstacles` argument that defaults to the config's mode, so val/selection is obstacle-free too.
   - `student_sidecar` adds `course_obstacles` only when the mode is not "all".
   - `run_feasibility` forwards the mode to `evaluate_teacher` and records it in feasibility.json. Feasibility stays non-gating (AC4).
   - `_val_select` also scores each `track_evals` pool (noise mode, val episodes and seed). It writes them into the val_history.jsonl row under `tracks` and logs them. Checkpoint selection stays on liftoff:val only.
   - `final_report: true`: when training finishes without interruption, call `evaluate_student_run` once for liftoff:held-out and once per track_evals selector (AC5).
   - `evaluate_student_run` takes `obstacles: str | None` (None means the run config's mode) and records it in the report. An overridden mode adds an `-obstacles-<mode>` filename suffix. `student_markdown` prints the mode.
7. **`src/drone_fly/training/evaluate.py`**
   - `evaluate_teacher` takes `obstacles: str | None` (None means the teacher run config's mode).
   - `cmd_teacher_eval` forwards `--obstacles` and prints the student-noise outcome breakdown.
8. **`src/drone_fly/training/cli.py`** — the `train eval` parser gets `--obstacles {all,none}` (unset means the run config). `--tracks liftoff:liftoff-arena-04` already works through id selectors.
9. **New `src/drone_fly/training/configs/student-liftoff-open.yaml`**
   - `extends: student-liftoff-run2b`, `name: student-liftoff-open`;
   - `tracks: {obstacles: none}`;
   - `distill: {track_evals: ["liftoff:liftoff-arena-04"], final_report: true}`.

   It gets a run2b-style header comment. run2 and run2b are untouched.
10. **New `docs/experiments/uc77-obstacle-free-student.md`**:
    - why obstacle-free matches the blind student;
    - the obstacle definition, with the Infinity Loop numbers;
    - the accepted risk on obstacle tracks (UC73 "eyes" is the follow-up);
    - train, per-track eval, export and fly commands;
    - a Results section the orchestrator fills after AC7.

**AC3.** No changes to student_obs, student_env, teacher_obs, action or export. The schema stays v3 / DIM 65 and obstacle_vision stays zeros.

## Binding implementation notes
- **N1 (flags).** `loader.py:211-212` `c.meta.update(meta)` overwrites `flags` from the convert step. Add `obstacle-free` to the loader's own `meta["flags"]` in load_course, for both the Liftoff meta and the procedural base_meta, whenever the mode is not "all". Tests may also rely on `stats["obstacles"]` and `stats["obstacles_removed"]`. Tests must not assume a convert-time flag survives load_course.
- **N2 (spawn on a removed item).** In "none" mode, record `stats["spawn_support_removed"]: bool`, true when the spawn support primitive came from a removed (non-gate item) slice. On Infinity Loop the support set is 19 scene prims plus 1 table prim at top about 0, and the support offset is -0.002. Report the actual value for arena-04; the docs state it.
- **N3 (empty eval pools).** If a `track_evals` (or final_report) selector's eval_pool comes back empty, skip it with a warning. It must not crash `_val_select` or the final report.
- **N4 (snapshot).** When the mode is "all", OMIT `tracks.obstacles` from the run snapshot or config.yaml. Existing configs and default run folders stay byte-identical. A missing key loads as "all". "none" is written and round-trips via `student_raw_from_snapshot` and `load_student_run`.
- **N6 (distill snapshot defaults).** `StudentConfig.snapshot` (`config.py:615-620`) writes `asdict(distill)` whole. Omit `distill.track_evals` and `distill.final_report` from the snapshot when they hold their defaults, the same pattern as N4; a missing key loads as the dataclass default. Test (b'): the run2b student snapshot has none of `tracks.obstacles`, `distill.track_evals` or `distill.final_report`, and the open config's snapshot keeps all three and round-trips them.
- **N5 (preset).** Nothing in UC-77 depends on `liftoff_arena`. The open config header and the docs fly line use `--preset liftoff` plus a config overlay (the UC74 camera.pointer block incl. max_bearing_az_deg 30, pointer_roi, detector-v1 161f7a83 weights; UC-76 documents the overlay).

## Files Affected
**Production (developer):**
- `src/drone_fly/tracks/liftoff.py`
- `src/drone_fly/tracks/procedural.py`
- `src/drone_fly/tracks/loader.py`
- `src/drone_fly/training/config.py`
- `src/drone_fly/training/ppo.py`
- `src/drone_fly/training/distill.py`
- `src/drone_fly/training/evaluate.py`
- `src/drone_fly/training/cli.py`
- `src/drone_fly/training/configs/student-liftoff-open.yaml` (new)
- `docs/experiments/uc77-obstacle-free-student.md` (new)

**Tests (QA), all CPU and hermetic:**

New `tests/test_tracks_obstacle_free.py` (uses `tests/tracks_support.py` build_library / fake-arena-01):
- **(a) Liftoff:**
  - the harness list is empty;
  - the pillar (-3, 1.5, 16) and crate (3, 0.5, 6) Unity points are no longer solid;
  - the gate post (1.2, 1.5, 0) is still solid and the opening is clear;
  - the scene floor and platform remain;
  - spawn, origin, floor_z, bounds, passages and time limit equal the default conversion;
  - `obstacles_removed` is correct;
  - the obstacle-free flag survives load_course (N1);
  - `spawn_support_removed` is present (N2).
- **(a) Procedural:** zero prims, an empty harness, and identical passages and spawn.
- **(a) Teacher-obs side:** the env has no valid surface candidates on an obstacle-free procedural pool. The obstacles block holds only the floor, and harness obstacle_vision is "clear".
- **(a) Augmented copies** stay obstacle-free.
- **(b) Default parity:**
  - no kwarg equals `obstacles="all"` (prims, harness, cells, stats);
  - the default cache key parts are unchanged;
  - no new stats or flags appear;
  - an unknown mode raises;
  - plus the N4 snapshot tests: the run2b and teacher-liftoff-aug snapshots have no `tracks.obstacles` key; the open snapshot has `obstacles: none` and round-trips; an old teacher snapshot without the key loads as "all".

Student tests (extend `tests/test_training_student.py` or add `tests/test_training_student_open.py`):
- **(c)** student-liftoff-open is in `packaged_names()` and loads strictly. A flatten-diff over run2b shows exactly name + tracks.obstacles + distill.track_evals + distill.final_report. A bad mode raises TrainConfigError.
- **(d)** CPU smoke on the shrunk open config: procedural-only mix, 8 envs, 2 iterations, val_every_iters 1, tiny_val, and final_report off or monkeypatched. Assert:
  - the run completes;
  - the training courses are obstacle-free;
  - val_history has `tracks` entries;
  - the sidecar has `course_obstacles: none`.
- **(d')** final_report path, with evaluate_student_run monkeypatched: held-out plus each track_evals selector are reported. Also cover the N3 empty-pool skip.
- **(e)** VERSION 3 / DIM 65 and the sidecar obs_schema ("student-obs", 3, 65).
- **AC4** feasibility spy: the mode is forwarded and recorded; the check is non-gating.
- **AC5 CLI** (extend `tests/test_training_eval_student.py`): `--obstacles` is forwarded and recorded, and an override gets the filename suffix.

## Risks & Considerations
- **Scene colliders are kept on purpose.** That keeps the real ground. On tracks whose hazards are scene geometry, some collisions will remain; a "bare" mode is a possible follow-up.
- **Placed platforms and ramps are removed in "none" mode.** Ground starts are unaffected (the plant holds ground_z = 0 and the spawn is computed on the full geometry). N2 makes it auditable.
- **The blind student will crash on obstacle tracks (accepted).** Scope is open tracks; docs must say so.
- **This may not fix the live Infinity Loop crash.** Its props are mostly flush with or below the floor. The run2b out-of-bounds (about 25 %) and floor (about 20 %) failures are not obstacle-driven. The arena-04 per-track sim eval attributes the cause; reward and termination tuning stays out of scope.
- **Infinity Loop is in the train split.** Its score is an in-distribution check; held-out macro_completion is the generalisation metric.
- **Compute.** About 3-4.5 h on the RTX 3050. final_report adds a few GPU minutes. AC7 is orchestrator-run after merge.
- **Cache.** New obstacle-free cache entries are created under artifacts/tracks/cache.
