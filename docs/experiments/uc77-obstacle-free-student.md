# UC77 — obstacle-free student training (`student-liftoff-open`)

The deployable K1 student is **blind to obstacles**. It flies with `free_space_estimator: none` and
an all-zero `obstacle_vision` block. Run2b (UC75) still trained on Liftoff courses that keep every
placed prop, so it was being taught to dodge things it cannot see. The 2026-10-15 run2b run ended at
about 1 % completion once the teacher handed over (beta → 0): collisions in about 45 % of episodes,
out-of-bounds about 25 %, floor about 20 %. Live on Liftoff Arena "04 - Infinity Loop" it crashed
within about 10 s.

`student-liftoff-open` is run2b on **obstacle-free** courses, so the training courses match what the
student perceives. The target is open race tracks like Infinity Loop, which has nothing solid in its
race corridor.

## What "obstacle-free" removes

`tracks.obstacles: none` (default `all`) applies to every course the config builds: training,
validation, feasibility, the per-track evals and the final report.

- **Removed:** every placed track item that is not a route-gate frame. That is the population of the
  harness obstacle list before its size, corridor and cap filters: props, barriers, flags, cones,
  placed platforms and ramps. They leave contact and the teacher's privileged `obstacles` block, and
  the harness list (the teacher's `obstacle_vision`) is empty. On procedural courses every obstacle is
  removed.
- **Kept:** scene colliders (the real ground, arena structures, stands), route-gate frames, the
  placed item the spawn stands on (its support, so the sim start matches the live one), the
  heightfield, `floor_z` and the world bounds (`bounds_hi.z` is the sim's only ceiling).
- **Unchanged:** spawn, course origin, passages, bounds and time limit. The spawn support is computed
  on the full geometry first, so these equal the default conversion.

The mode is not a `ConversionConfig` field, so the conversion fingerprint the student checks against
the teacher sidecar does not change. Obstacle-free courses are cached under their own key in
`artifacts/tracks/cache/` and carry the `obstacle-free` course flag. Their stats add `obstacles`,
`obstacles_removed` (items), `spawn_support_kept` (true when the spawn stands on a placed item that
would otherwise be removed) and `spawn_support_item` (`<item_id>#<instance_id>`). Default courses, configs, run snapshots and
sidecars are byte-identical to before: `tracks.obstacles`, `distill.track_evals` and
`distill.final_report` are left out of `config.yaml` at their defaults.

### Infinity Loop (`liftoff:liftoff-arena-04`)

| | default | obstacle-free |
|---|---|---|
| primitives | 6113 (5773 scene, 221 item, 119 item-table) | 5809 (5773 scene, 35 gate-arch, 1 spawn support) |
| harness list | 247 | 0 |
| `obstacles_removed` | — | 246 items |
| `floor_z` | -0.865 (gate aperture bottoms) | -0.865 |
| spawn support offset | -0.002 m | -0.002 m |
| `spawn_support_kept` / `spawn_support_item` | — | true / `ConcreteBarrier01#146` |

The 247 harness-list props (246 removed; the spawn's barrier is kept) are mostly flush with or
below the arena floor:

| item | count | distance from route | top z |
|---|---|---|---|
| TlTrackLight01 | 110 | — | buried |
| ConradCrowdControlBarrier01 | 94 | about 2.9 m | 0.16 |
| AirflagLiftoffSlalomDark01 | 24 | 13-16 m | 1.78 |
| ASLLightBeam04 | 8 | — | buried |
| AdvertBoard | 6 | — | 0.45 |
| LightConeRed01 | 4 | about 2 m | 2.16 |
| ConcreteBarrier01 | 1 | — | 0.0 |

**Spawn support.** The Infinity Loop spawn sits on that one `ConcreteBarrier01`. Its top (contract
z 0.86) is the course origin; the scene floor under it is 0.76 m lower (contract z 0.10). The barrier
is real in Liftoff and the drone takes off from it, so obstacle-free mode **keeps** it. Removing it
would leave nothing under the start until the floor 0.76 m below, and the sim start would differ
from the live one. Only the spawn's own support item is kept, and it stays off the (empty) harness
list.

## Accepted risk: obstacle tracks

A student trained obstacle-free **will crash into real obstacles** on tracks that have them. That is
accepted. This variant is for open tracks only. Obstacle perception is the separate UC73 "eyes"
track, after which the connectome flies with learned camera rays.

Removing obstacles may also not be enough on its own. Run2b's out-of-bounds (about 25 %) and floor
(about 20 %) failures are not obstacle-driven, and the Infinity Loop props barely reach the corridor.
The per-track eval below attributes the remaining failures. Reward and termination tuning is out of
scope.

## Train

```
drone-fly train student --config student-liftoff-open --teacher <teacher_it1374 run> \
    --noise <noise_model.yaml> --device cuda [--resume <run folder>]
```

- The UC68 teacher (`teacher_it1374`) is used unchanged. Feasibility runs it on obstacle-free held-out
  and val courses, and records `obstacles: none` in `eval/feasibility.json`. It is logged, never
  gating.
- Every validation (`val_every_iters`) also scores `liftoff:liftoff-arena-04` (noise mode, the val
  episodes and seed). The result goes under `tracks` in `eval/val_history.jsonl` and the log.
  Checkpoint selection stays on `liftoff:val`.
- `distill.final_report: true`: after an uninterrupted run, the val checkpoint's `train eval` report
  is written for `liftoff:held-out` and for `liftoff:liftoff-arena-04`
  (`eval/student-held-out.{json,md}`, `eval/student-liftoff-arena-04.{json,md}`).
- The student sidecar records `course_obstacles: none`. The obs schema stays `student-obs` v3 / DIM 65.

Infinity Loop is in the **train** split (all seven LiftoffArena tracks are), so its score is an
in-distribution check. Held-out `macro_completion` stays the generalisation metric. Budget: about
3-4.5 h on the RTX 3050 at 2048 envs.

## Evaluate one track

```
drone-fly train eval <run> --tracks liftoff:liftoff-arena-04
drone-fly train eval <run> --tracks liftoff:liftoff-arena-04 --obstacles all
```

The run config's mode is the default. The report prints completion and the student-noise outcome
breakdown (collision, floor, out-of-bounds, gate_lost, timeout), and records `obstacles`.
`--obstacles` overrides the mode, for example to score the open student with the props back in, and
suffixes the report files `-obstacles-<mode>`. `--obstacles` works for teacher runs too.

## Export and fly

```
drone-fly train export <run> --out <student-open.pt>
drone-fly harness run --harness liftoff --preset liftoff --config <overlay.yaml> \
    --checkpoint <student-open.pt> --recordable
```

Load Infinity Loop via Quick Setup and verify the spawn before arming. `<overlay.yaml>` is the
`liftoff` preset overlay documented with UC-76:

- the UC74/UC75 `camera.pointer` block, including `max_bearing_az_deg: 30`;
- `camera.pointer_roi`;
- the detector-v1 `161f7a83` weights (`gates.learned.weights`).

The `liftoff_arena` preset is not used, because it refuses the gremlin-micro student.

## Results

Run `20261015T172229Z-student-liftoff-open`: RTX 3050, 2048 envs, 80M steps, 2026-10-15/16.
It resumed once after a reboot at it 160. The export is `student-open.pt` (latency p50 0.70 ms, p99 1.19 ms).

**Val selection (liftoff:val, noise).** The best checkpoint is it 611 at **1.8 %** macro
completion (run2b 0.9 %, run-2 1.8 %).

**Final report (32 episodes per track, seed 1234, obstacles `none`):**

| set | teacher exact | student exact | student noise | student outcomes (noise) |
|---|---|---|---|---|
| liftoff:held-out (15 tracks) | 17.1 % | 0.0 % | 0.0 % | out-of-bounds 42 %, collision 29 %, floor 17 %, gate_lost 10 % |
| liftoff:liftoff-arena-04 (Infinity Loop, train split) | **93.8 %** | **0.0 %** | 0.0 % | collision 50 %, out-of-bounds 44 %, floor 6 % |

**Reading.** Removing obstacles did not unlock the student.
- With **exact perception** the K1 student completes **0 %**, where the teacher completes 94 % on Infinity Loop and 17 % held-out.
- The gap is therefore not perception (the noise and exact scores are equal) and not obstacles. The distilled K1 student simply does not reproduce the teacher's control once beta reaches 0. Training-episode completion fell from ~55 % at beta 0.9 to ~2–5 % at beta 0.
- The remaining "collisions" on an obstacle-free Infinity Loop are scene and gate-frame contacts by an unstable trajectory, not props.
- Next lever: the student/distillation itself, i.e. the K1 capacity vs DAgger setup. The report's "K1 capacity (MLP exact − K1 exact)" attribution is unmeasured; an `--actor mlp-student` run of the same config measures it.

**Live Infinity Loop:** see the follow-up run notes (at most two recorded attempts, 2026-10-16).
