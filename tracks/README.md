# Track catalogues

This folder holds the committed **track catalogues** (UC68, [ADR 0007](../docs/adr/0007-track-library-and-teacher.md)):
one YAML file per source.

| file | source | written by |
|---|---|---|
| `liftoff.yaml` | `liftoff-xml`: Liftoff track/race XML pairs | `drone-fly tracks scan --source liftoff` |
| `procedural.yaml` | `procedural`: seeded procedural course pools | by hand |

The code that reads them lives in `src/drone_fly/tracks/`. `tracks.loader.load_course` turns any
entry into the simulator's `SimCourse`.

**No game data is committed.** The catalogues hold names, relative file names, hashes and derived
statistics only: no coordinates, no geometry, no scene caches and no absolute local paths. The XMLs,
`index.json`, `props.json` and the scene caches stay in the gitignored trackdata folder on each
machine.

## Commands

```
drone-fly tracks list [--split train|val|held-out] [--source liftoff|procedural]
drone-fly tracks check [--source ...] [--root DIR] [--scenes-root DIR] [--require-scenes]
drone-fly tracks scan --source liftoff [--root DIR] [--scenes-root DIR] [--seed 68]
drone-fly tracks stats --split train [--root DIR]
drone-fly tracks extract-scenes [--map M ...] [--bundle M=NAME] [--witness M=POINTS] [--force]
                                [--root DIR] [--scenes-root DIR] [--game-dir DIR] [--max-bundle-mb N]
drone-fly tracks extract-props [--force] [--root DIR] [--game-dir DIR]
```

- `list` prints both catalogues, or one source or split.
- `check` verifies that every referenced file exists locally, that its sha256 matches the catalogue,
  and that the entry converts. With `--require-scenes`, a map without a usable scene fails.
- `scan` pairs, converts and hashes every local track/race pair, then writes `tracks/liftoff.yaml`.
  Existing split assignments are preserved (see [Split](#split)).
- `stats` aggregates spacing and aperture percentiles over **one** split. Data-derived parameters
  (the procedural ranges, the harness-list cap) come from `--split train` only.
- `extract-scenes` and `extract-props` read the installed Liftoff files with UnityPy. They **never
  launch the game**. See [Scene caches](#scene-caches).

## Roots

| what | resolution order (first wins) | default |
|---|---|---|
| trackdata root | `--root` → `$DRONE_FLY_TRACKDATA` → the catalogue header `root` (relative to the repo root) | `artifacts/harness/trackdata` |
| scenes root | `--scenes-root` → `$DRONE_FLY_SCENES` → `<trackdata root>/scenes` | `artifacts/harness/trackdata/scenes` |

The trackdata root holds the `*Track*.xml` / `*Race*.xml` files, `index.json` (track uuid → race
uuid, environment, laps, route) and `props.json` (per-item prefab colliders). Both roots are
gitignored.

## Catalogue schema (`schema_version: 1`)

### Header (`liftoff.yaml`)

| field | meaning |
|---|---|
| `schema_version` | `1` |
| `source` | `liftoff-xml` |
| `root` | default trackdata root, relative to the repo root |
| `scenes_root` | default scenes root, or `null` for `<root>/scenes` |
| `world_scale` | course-frame divisor (1.0: Liftoff metres, see ADR 0007 D5) |
| `conversion_version` | version of the conversion rules; eval refuses an unknown one |
| `scanned` | date of the last scan |
| `split` | `method`, `seed`, `ratios` (train / val / held-out) and per-split `counts` of maps and tracks |
| `maps` | per map: its `split` and `scene` (`status`, `bundle`, `build_id`, `sha256`, `counts`, `flags`) |
| `root_files` | `index.json` and `props.json`: `sha256` and game `build_id` |
| `harness_cap` | the harness obstacle-list cap O_h: `value`, `p99`, `tracks`, `source` (train stats only) |
| `outlier_thresholds` | `spacing_max_m`, `altitude_span_m`, `extent_m`, `passages`, `aperture_min_m` |
| `entries` | the tracks |

### Entry (both sources)

| field | meaning |
|---|---|
| `id` | `<map-kebab>-<NN>` for Liftoff (e.g. `liftoff-arena-04`, Infinity Loop); a free name for procedural pools |
| `source` | `liftoff-xml` or `procedural` |
| `track`, `race` | XML file names relative to the trackdata root (Liftoff only) |
| `map` | Liftoff environment, or `procedural` |
| `gate_types` | sorted item ids/classes of the route's checkpoints |
| `laps` | laps of the race (`null` for a procedural pool, whose spec draws them) |
| `split` | `train`, `val` or `held-out` (procedural: always `train`) |
| `status` | `ok`, or `unconvertible: <reason>` (e.g. `too many passages (<n>)`); such entries are kept, never dropped |
| `name` | the track's display name |
| `uuids` | `track` and `race` uuids |
| `sha256` | `track` and `race` file hashes; `check` compares them |
| `stats` | derived statistics, below |
| `flags` | per-track flags, below |
| `spec` | procedural only, below |
| `liftoff_menu` | reserved for a future in-game menu reference; optional, not validated, not implemented |

`stats` (Liftoff): `passages` (per lap), `unique`, `branches`, `laps`, `circuit`, `openings`,
`volumes`, `route_length_m` (whole race), `spacing_m` (`min` / `median` / `max`), `extent_m`,
`altitude_span_m`, `aperture_m` (`min` / `median`), `harness_list`, `harness_overflow`,
`harness_candidates`, `primitives` (full geometry), `cells`, `cell_overflow`, `sheared`,
`mesh_flagged`, `unknown_items`, `solid_in_opening`, `spawn` (`real` / `synthetic` /
`unsupported`), `spawn_support_offset_m` (support top minus XML spawn z), `time_limit_s`.

`spec` (procedural): `seed_base`, `pool_size` (courses in the pool, `seed_base + i`), `ng` (gate
count range), `laps`, `tier`, `tunnels`, `variety`, `no_obs`, `scale` (multiplies every position and vertical size; horizontal obstacle sizes follow the
aperture mapping instead),
`aperture_m` (opening size range). The `scale` and `aperture_m` ranges come from
`tracks stats --split train`.

### Flags

| flag | meaning |
|---|---|
| `no-scene` | no usable scene cache: the course has placed props plus the floor plane only (no walls, buildings or terrain). Results on it are optimistic. |
| `stale-build` | the scene cache was built for an older game build. It passed geometric validation, so it is used. |
| `no-terrain` | the map has terrain, but no heightfield was extracted; the floor plane stands in |
| `partial` | more than 10 % of the scene's colliders are meshes, which are not emitted |
| `spawn-unsupported` | no supporting top face within ±1 m of the spawn; the XML spawn z is the origin |
| `synthetic-spawn` | the track has no spawn item; one is synthesised before the first passage |
| `branching` | the route branches; the first successor is followed |
| `solid-in-opening` | a solid collider intersects a gate opening (kept, flagged) |
| `outlier-spacing` / `-altitude` / `-extent` / `-passages` / `-aperture` | a stat beyond `outlier_thresholds` |

## Selectors

Training configs and the CLI pick entries with `<source>:<term>`:

- `source` is `liftoff` or `procedural`;
- `term` is a split (`train`, `val`, `held-out`), `all`, `map=<Map>`, or an entry `id`.

Examples: `liftoff:train`, `liftoff:held-out`, `liftoff:map=LiftoffArena`, `liftoff:liftoff-arena-04`,
`procedural:proc-liftoff`.

**Leakage guard (AC9).** Every training load goes through `tracks.select(..., purpose="train")`. It
raises `TrackLeakageError` if a selector matches any `val` or `held-out` entry. Only
`drone-fly train eval` may load them. Hyperparameters are chosen on `val` only.

## Split

The split is **by map**, so held-out tracks are unseen environments, not just unseen layouts.

- Maps are ordered by `sha256("<seed>:<map>")`, with seed 68.
- They are filled greedily: held-out until it has at least `round(0.15·N)` tracks and 2 maps, then
  val the same way. The rest is train.
- It is a pure function of the seed and the map names. **Scene-cache availability never changes it.**
- On a re-scan, existing maps keep their split and new tracks inherit their map's split. A new map
  is hash-bucketed by cumulative ratio.

The current split (92 tracks, 20 maps):

| split | maps | tracks | tracks with a usable scene |
|---|---|---|---|
| train | 12 | 63 | 30 (BandoCity, Hannover, LiftoffArena, ParisDroneFestival, PineValley, ShortCircuit) |
| val | 4: AutumnFields, DubaiLegends, Rustline, Surtur | 14 | 4 (DubaiLegends, Rustline) |
| held-out | 4: HangarC03, MinusTwo, ThePit, TheRussianWoodpecker | 15 | 2 (HangarC03) |

## Scene caches

Without a scene cache, a Liftoff course has only its placed track items (from `props.json`) and a
floor plane. With one, it also gets the map's own colliders: buildings, walls, trees and so on.
`scene_policy: allow-missing` (the training default) uses `no-scene` maps as they are, while
`scene_policy: require` refuses them.

**Format.** One JSON file per map, `<scenes root>/<Map>.json`, in the liftoff-flight-analyzer
format (`colliders` in Unity world space; `counts`, `skipped`, `source`).

**Setup.** Copy or symlink existing caches into the scenes root. For example, from a
liftoff-flight-analyzer working copy:

```
cp <analyzer trackdata>/scenes/*.json artifacts/harness/trackdata/scenes/
```

**Extraction.** `tracks extract-scenes` builds caches for maps that have none. It needs the optional
UnityPy extra:

```
uv sync --extra liftoff-extract
# or, into an existing venv:
uv pip install --python <venv python> UnityPy==1.25.4
```

- It reads each candidate scene bundle once, scores every map against it, and assigns bundles to
  maps uniquely.
- The witness for scoring is a synthetic one: points just before and after every opening, plus the
  raised spawns. Pass `--witness M=<points.json>` to use real flown points instead.
- A map whose best match is not clearly better than the runner-up becomes `ambiguous` and stays
  without a scene. Use `--bundle M=<name>` to force a bundle.
- It never launches Liftoff.

**Validation.** Every cache is checked geometrically against the current XMLs. It must have at least
70 % of its gates near the witness and at most 0.5 % of witness points inside solids. A cache built
for an older game build is used with a warning (`stale-build`). A failed check marks the map
`invalid`, and it is then treated as `no-scene`.

Scene `status`: `ok` and `stale-build` are usable. `invalid`, `ambiguous` and `missing` are not.

The run snapshot (`artifacts/training/<drone>/<run>/tracks/files.sha256`) records each loaded
cache's sha256 and resolved source path.
