# Use Case 68: Track library + multi-track privileged teacher

## Summary
Make "a course" a data-driven, scalable concept, and train a **teacher** policy that generalizes across courses instead of memorizing one. A new top-level `tracks/` folder holds committed **catalogues**: one YAML per source, starting with `tracks/liftoff.yaml`. It lists every Liftoff track/race pair under the user's local `artifacts/harness/trackdata/` (game data stays gitignored; the catalogue only names files). Each entry records its map/environment, gate types, laps, whether it converts, and a **train / val / held-out split assigned by map**, so held-out tracks are unseen environments, not just unseen layouts. `src/drone_fly/tracks/` turns a catalogue entry or a procedural spec into the simulator's course representation, reusing `harness/offline/track_xml.py`. The UC67 simulator trains a privileged teacher on the Liftoff drone profile, with dynamics and timing randomization, ground starts, and a mix of Liftoff training tracks and procedural courses. The teacher observes exact next-gate poses (the same quantities the live harness estimates, without noise) plus whatever obstacle information the course has. `drone-fly train eval` reports per-track completion in the simulator. Targets: **≥ 95% on training tracks and ≥ 80% on held-out tracks**.

## Acceptance Criteria
1. **Catalogue format:**
   - `tracks/README.md` documents the schema. Each entry has: `id`, `source` (`liftoff-xml` / `procedural`), file references (track + race XML, relative to a configurable local root, default `artifacts/harness/trackdata/`), `map`, `gate_types`, `laps`, `split`, and `status` (`ok` / `unconvertible: <reason>`).
   - `drone-fly tracks list [--split held-out]` prints the catalogue.
   - `drone-fly tracks check` validates that the local files exist and convert.
2. **Catalogue population:**
   - `drone-fly tracks scan --source liftoff` scans the local trackdata folder, pairs the race files with their tracks, converts each pair, and writes or updates `tracks/liftoff.yaml`.
   - It keeps existing split assignments stable and assigns new ones deterministically by map: roughly 70/15/15 by map, with seeds recorded.
   - Unconvertible pairs are kept with their reason.
   - The committed catalogue covers all pairs currently present locally.
3. **Course conversion:**
   - `src/drone_fly/tracks/` exposes one loader that returns the simulator's course type for both sources.
   - Liftoff conversion reuses `track_xml.py`: gate order from the race route, gate type dimensions, obstacles as AABBs, world scale from config.
   - The procedural source wraps the existing generator with a seed and difficulty knobs.
   - Tests use small synthetic XML fixtures; no game data is committed.
4. **Batched multi-course training:**
   - The simulator can hold a pool of courses and assign one per environment at reset, according to a configured mix (e.g. `liftoff:train` 60% + procedural 40%).
   - Gate counts, layouts and obstacle sets may differ per environment.
   - Episode stats are logged per course id.
5. **Teacher training:**
   - `drone-fly train teacher --drone <id> --tracks liftoff:train --procedural 0.4 [--config …]` trains with the UC67 dynamics and timing randomization and the ground/air start mix.
   - It writes `artifacts/training/<drone-id>/<run-id>/` (checkpoints, config + profile + catalogue snapshot with file hashes, TensorBoard/CSV logs).
   - It can resume from a checkpoint.
6. **Teacher observation:** the observation spec is a documented superset of the live-harness gate observation, with exact values: next 1–2 gate poses relative to the body, own velocity, attitude and rates, plus obstacle features where present. A versioned observation schema is written into the checkpoint.
7. **Evaluation:**
   - `drone-fly train eval <run> --tracks liftoff:held-out --episodes N` runs deterministic and randomized-dynamics evaluations.
   - For each track it reports: completion rate, mean gates passed, lap time, and crash reasons (collision / floor / timeout / out-of-bounds).
   - It writes `artifacts/training/<drone-id>/<run-id>/eval/<split>.json` and a Markdown summary.
8. **Targets:** with the `liftoff-gremlin-micro` profile, randomization on, and ground starts, the teacher reaches **≥ 95% mean completion on train tracks and ≥ 80% on held-out tracks** (held-out by map). If the target is missed after reasonable effort, the run is reported with per-track breakdowns and the blocking failure modes, rather than tuned on held-out data.
9. **No held-out leakage:** a test asserts that the trainer never loads `split: held-out` or `val` entries. Hyperparameter selection uses `val` only.
10. **Tests:** hermetic tests cover the catalogue schema, deterministic split assignment, scan pairing (with fixtures), mixed-pool course assignment, and the leakage guard.

## Potential Pitfalls & Open Questions
- **Missing input** — The local trackdata may not contain the Liftoff Arena maps (Infinity Loop came from `liftoff-flight-analyzer` extraction). The scan reports which maps exist; Arena tracks are added if their XMLs are present.
- **Risk** — Some Liftoff track scales (huge outdoor maps) are far outside the procedural distribution. Normalize with the configured world scale, and keep the per-map stats so that outliers are visible.
- **Edge case** — Race routes that revisit a checkpoint or have multiple laps. `parse_race_route` already supports these, and the simulator's gate sequence must support repeats.
- **Assumption** — AABB obstacles are good enough for training collisions; rotated props become conservative boxes, as in `track_xml.py`.
- **Risk** — The 80% held-out target may be too high for maps whose gates are very different (dive gates, tight indoor spaces). The per-track report shows where.

## Original Description
> Ok we'll have to go the next step. Despite having "memorized" the course as you mentioned in a previous UC, we really aren't flying well. The AI just tumbles right from the start, with throttle/yaw/roll/pitch input from the start.
>
> We do have the possibility of exporting the track geometry for further trainings by converting it to pybullet, but the thing is, I want to GENERALIZE this so it can tackle random courses in liftoff.
>
> Ideas?

> Prepare everything as use cases. Once they are set-up, and we validate them together, wait for me to give you the go ahead (I'll be out of home for a couple hours so that's good time to use, but i'll leave after we have set this up). When doing the analysis, make sure whateve rwe create fits in a good understandable folder structure, and is scalable in case we need to train more drones and such.

## Clarifications
- Q: What simulator completion target should the teacher hit on held-out tracks?
  A: ≥ 80% held-out, ≥ 95% train.
- Q: Where should track definitions live?
  A: Top-level `tracks/` (catalogues), code in `src/drone_fly/tracks/`.
