# drone-fly

Can the wiring diagram of a real fruit-fly brain (the **MaleCNS connectome**) seed a neural
controller that **reinforcement-learns** to fly a quadrotor through a timed race course in
simulation?

## Intent & purpose

drone-fly is a research prototype that wires three things together:

- a **real insect connectome** (MaleCNS) instantiated as a small, *trainable* sparse layer;
- an **RL training loop** (PPO, Stable-Baselines3) over a gymnasium quadrotor racing environment;
- a **playback viewer** that shows neuron activation as an MRI-style brain heatmap.

The controller's architecture is *seeded from biology* rather than being a generic MLP:
observations are projected into named sensory populations, propagated through the connectome's
real synapses, and read out from motor neurons into throttle / roll / pitch / yaw. It is **not** a
biophysical brain simulation — the wiring comes from the fly; the connection strengths are learned.

## Goals

1. Prove end-to-end that a connectome-seeded agent can learn to fly a waypoint course.
2. Stay **reproducible & hermetic** — a committed fixture connectome trains offline, in CI.
3. Keep the circuit **inspectable** — prune to a minimal sensory→motor slice and watch it "think".
4. Make observations **biologically bound & extensible** — modality→population mapping with a
   graft-ready block schema for adding future senses.

## How to create a slice

A *slice* is a pruned sensory→motor subcircuit of the connectome — smaller, faster to train.

1. Pick a prune config: `configs/prune/k0.yaml` (tightest) … `k2.yaml` (richest); edit its
   `out` and `prune_k` if needed.
2. Run: `uv run drone-fly prune --config configs/prune/k2.yaml`. **By default prune uses the full
   MaleCNS connectome:** if it isn't already in the default location it is **auto-downloaded**
   (~109MB, once) and reused thereafter — see [Connectome data](#connectome-data). Pruning the full
   matrix (~161k neurons / ~25M edges) takes minutes and real memory; that is expected, not a hang.
3. To prune a *specific* connectome instead (no download), set `connectome:` in the config — e.g.
   `connectome: tests/fixtures` for the committed offline slice, or `connectome: <a-prior-slice>`.
4. The slice (+ provenance) is written to the config's `out:` dir, e.g. `artifacts/pruned/k2`.
5. Reuse it by setting `connectome: artifacts/pruned/k2` in a train config.

## How to train a model

1. Install: `git clone <repo> && cd drone-fly && uv sync --extra dev`.
2. Connectome: the default `connectome: tests/fixtures` runs as-is; for a real run see
   [Connectome data](#connectome-data).
3. *(Optional)* Create a slice (above) and point the train config at it.
4. Edit `configs/train/example.yaml`: set `name`, `connectome`, and a real `timesteps`
   (e.g. `1_000_000`).
5. Train: `uv run drone-fly train --config configs/train/example.yaml`. Outputs land under
   `training/<name>/{checkpoints,logs,recordings}/`.
6. Evaluate: `uv run drone-fly evaluate --config configs/evaluate/example.yaml` (point its
   `checkpoint` at `training/<name>/checkpoints/ppo_racer_final.zip`).

Resume is automatic: `resume: auto` in the train config continues from the newest checkpoint if
one exists, else starts fresh.

**Live training dashboard (UC-22).** On an interactive terminal `train` shows a full-screen
Rich TUI: grouped TIME/TRAIN/ROLLOUT values, one sparkline + tendency (improving/worsening/flat)
row per tracked metric with an iterations progress bar, a right-hand pane tailing the raw
stdout/stderr (including pybullet's native prints), and a bottom health status bar. It is
default-on for a TTY and disabled with `--no-tui`; a non-TTY / piped / CI run auto-falls back to
the plain Stable-Baselines3 line logger, so CI and log files are unaffected. The CSV and
TensorBoard learning curves are written either way.

## Other useful commands

- **Sanity check** (offline, seconds): `uv run drone-fly smoke-train --connectome tests/fixtures`.
- **Watch it think**: open `viz/viewer.html` in a browser (no server/build) and load a recording
  produced with `record: true` in a train/evaluate config.
- **Start over**: `uv run drone-fly clean` — dry-run by default (lists what it *would* delete);
  add `--yes` to wipe `artifacts/` + `training/<name>/`, `--include-prunes` to also drop slices.

---

## Details

### Connectome data
- **Full matrix (prune default):** the whole-brain MaleCNS connectivity (CC-BY — cite MaleCNS /
  `connectome_data_prep`). `drone-fly prune` with no explicit `connectome:` uses it, **auto-downloading**
  it on first run into the default location (`DRONE_FLY_CONNECTOME_DIR`, else `data/connectome/`,
  gitignored) and reusing it after. Failures (no network, truncated/corrupt transfer) are reported
  as a single clean line (exit code `2`) and never leave a partial connectome behind.
  - Provision it ahead of time (or refresh it) with `uv run drone-fly fetch-connectome`
    (`--connectome-dir <dir>` for a custom location, `--force` to re-download). Once present, prune
    (and any train/evaluate config pointed at the default dir) reuses it with no download.
  - The download writes `mcns_inprop_all_neuron.npz` plus the meta saved **renamed** to
    `mcns_inprop_all_neuron_meta.csv` (the loader infers the sidecar from the `.npz` stem).
- **Fixture (explicit, offline):** a real ~300-neuron MaleCNS slice ships in `tests/fixtures/`
  (`mcns_fixture.npz` + `mcns_fixture_meta.csv`). It is used **only when named explicitly** —
  e.g. `smoke-train --connectome tests/fixtures`, or `connectome: tests/fixtures` in a config — so
  hermetic CI and quick offline checks stay network-free. Nothing defaults to it. Regenerate it
  with `uv run python scripts/build_test_fixture.py` (dev-time, needs network).
- `train` / `evaluate` defaults are unchanged: they still resolve `connectome` via the loader's
  order (explicit → `DRONE_FLY_CONNECTOME_DIR` → `data/connectome`) with no auto-download.

### Config-driven CLI
`train` / `evaluate` / `prune` / `prune-trained` take **only** `--config <yaml>` — every setting
lives in the YAML. Each key maps 1:1 to a former flag; an omitted key uses that flag's default, so
a config reproduces the equivalent run exactly. Unknown/missing/mistyped keys and malformed YAML
fail with a clear one-line error (exit code `2`, no stack trace). The example configs under
`configs/` are runnable as-is and document every key. `smoke-train` and `fetch-connectome` keep
their small flag surfaces.

### Training health & capacity guardrail (UC-23)
Every `train` run assesses training health with a headless, pure-logic engine and gates on the
seeded actor's capacity:

- **Pre-train capacity guardrail.** Before training starts, the resolved (post-prune) actor's
  **trainable-parameter count** is checked against a floor (default `3000`, calibrated on the
  committed fixture: the `prune_k: 0` minimal corridor yields 942 params — undersized — while the
  `prune_k: 2` default slice yields 7918, which passes). The verdict is always logged. When the
  actor is under-capacity the run **prompts to confirm** on an interactive terminal (declining
  aborts), and on a non-interactive/CI start (no TTY) it degrades to **warn-and-continue**. Set
  `strict_capacity: true` to abort instead in either mode — a clean one-line message and **exit
  code `3`** (config errors own `2`). Override the floor with `capacity_floor: <int>`. A
  sufficiently-capable start never prompts and changes nothing.
- **Runtime health assessment.** During training a callback snapshots the metrics each rollout and
  emits a verdict — `normal` / `warning` / `critical` with a human message and the contributing
  reasons — as a log line (only on a status change or every 10th update, so it never spams). It
  codifies the recurring diagnosis (a healthy critic with a flat, non-committing actor and ~0%
  success = an under-capacity slice) plus reward-stall, `approx_kl` runaway, `value_loss`
  divergence, and premature-entropy-collapse rules. The engine is pure and CI-unit-tested, and its
  verdict object is the interface a future status bar (UC-22) renders.

### Observation schema & retraining
The `schema` train key (UC-13) opts into a named **block observation schema**. `migrated_v1`
re-binds the 12-d observation into a *vision* block (target-relative → visual neurons) and a
*proprioception* block (self-motion → the mechanosensory/proprioceptive population).
`obstacle_vision_v2` (UC-15) extends `migrated_v1` with an *obstacle-vision* block (width 12 =
nearest-3 pillars × 4 features) bound to the same visual population; setting it widens the
observation to 24-d and makes the env emit the egocentric obstacle encoding (and, when course
randomization is on, sample pillars). `battery_hunger_v3` (UC-17) extends `obstacle_vision_v2`
with a width-1 *battery* block bound to the approximate `hunger` (internal-state / feeding)
population; setting it widens the observation to 25-d, turns on the battery drain + thrust-impact
model, and makes the env emit the battery dim (encoded as depletion = 1 − charge).
`damage_proprioception_v4` (UC-19) extends `battery_hunger_v3` with a width-1 *damage* block bound
to the `proprioceptive` population (a **second** proprioceptive block alongside `migrated_v1`'s
self-motion one — they coexist because the actor scatters blocks additively); setting it widens the
observation to 26-d, turns on the integrity damage + control-authority model, and makes the env
emit the damage dim (encoded as `1 − integrity`, so pristine = 0). Omitting `schema` keeps the
legacy single-projection behaviour. **Note:** a block schema changes the input→sensory wiring, so
checkpoints trained under a different wiring do not carry over — a fresh train is required. A
checkpoint can be *grafted* one step up the chain (`migrated_v1` → `obstacle_vision_v2` →
`battery_hunger_v3` → `damage_proprioception_v4`; the appended block is zero-initialised → identical
actions until fine-tuned), but VecNormalize obs-stats are tied to the old width and do **not**
carry to the wider observation, so fresh normalisation stats are part of that retrain.

### Obstacles (UC-15)
Courses may carry cylindrical **pillar obstacles** (floor-anchored: `center`, `radius`,
`height`). A drone↔pillar contact each step applies a severe, **non-terminating** penalty
(`RewardConfig.obstacle_penalty`, default 50, edge-triggered once per contact) — only
floor/ceiling/out-of-bounds crashes end an episode, so the drone can recover aerially and still
finish. The randomizer (under the course-randomize axis) samples solvability-guarded pillars,
and the 3D viewer draws them as wireframe cylinders. See `default_obstacle_course()` for the
fixed manually-placed default set.

### Recharge pads (UC-18)
A landing pad tagged `rechargeable` is a **recharge pad**: while the drone is *fully docked* on
it (the UC-16 docked state — a slow, upright, over-pad floor contact; hovering does **not**
count), the battery refills toward `1.0` at `BatteryConfig.recharge_rate` per second (clamped at
full). The rate exceeds the docked drain, so a dwell nets a gain. This reuses UC-17's battery
observation block — **no new observation dim, so checkpoints are not invalidated**. To make a
recharge worth the detour without a shaping reward, recharge is a **course-variation axis**
(`RandomizationConfig.enable_recharge`). **As of UC-24** the randomizer places **exactly one**
recharge pad per randomized course at an eligible gate anchor **regardless** of whether the course
is energy-constrained (feature presence, not a variable cover — before UC-24 pads were placed only
on over-budget courses, so under the shipped default battery *none* were ever produced). The
battery-aware solvability guard still holds: under the default battery a course is unconstrained and
the lone pad is a bonus; only under a cranked (test) battery must the single pad form a valid
one-pad cover. A per-rechargeable-pad step-budget allowance
(`EpisodeConfig.recharge_step_allowance`) keeps a legitimate recharge detour within the timeout.
Off by default (no recharge pads ⇒ byte-identical behaviour). The energy model is a conservative
generation-and-guard heuristic; it only guarantees model-level reachability.

### Damage & repair pads (UC-19)
With `DamageConfig.enabled` (turned on for you by the `damage_proprioception_v4` schema — see
above), the drone carries a scalar **integrity** `∈ [0, 1]` (`1.0` pristine). Each **UC-15
obstacle-contact edge event** sheds `damage_per_contact` integrity (once per contact, clamped ≥ 0);
floor/ceiling crashes are unaffected and still terminate. Damage degrades **control authority**
only — the effective `max_body_rate` is scaled by a linear `authority_factor(integrity)` floored at
`min_authority` — so a damaged drone is *sluggish but flyable* (a recoverable handicap); `max_thrust`
/ mass / drag are untouched, and integrity feeds **neither** termination **nor** reward. A landing
pad tagged `repairable` is a **repair pad**: while the drone is *fully docked* on it (the UC-16
docked state; hovering does **not** count), integrity is restored toward `1.0` at
`DamageConfig.repair_rate` per second (clamped at full) — the exact mirror of a recharge pad. A
per-repairable-pad step-budget allowance (`EpisodeConfig.repair_step_allowance`) keeps a legitimate
repair detour within the timeout. Whether repair is *needed* to finish is a **course-variation**
axis (agility-heavy courses become uncompletable once authority is degraded; `default_repair_course()`
is the damage-heavy fixture). Everything is **off by default** — at `integrity == 1.0` the enabled
path is byte-identical to the disabled one (the `min_authority < max_body_rate` invariant guarantees
it), and with no repairable pads/damage config an `EnvConfig()` is unchanged. Enabling the schema
adds the 26th observation dim, so **pre-UC-19 checkpoints are invalidated** (retrain, or graft one
step from `battery_hunger_v3` as above).

### Full-course randomization by default & placement toggles (UC-24)
By default a bare `randomize: true` train run now trains against the **full** course: it defaults to
the `damage_proprioception_v4` (26-d) schema **and** turns on obstacle, recharge-pad, and repair-pad
placement — gates + pillars + exactly one recharge pad + exactly one repair pad per course (each pad
is *feature presence*, not a variable cover). Three-state YAML toggles let you steer each placement
axis independently of the schema:

| Key | Default (when unset / `null`) |
|---|---|
| `randomize_obstacles` | on when `randomize` **and** the effective schema has an `obstacle_vision` block |
| `randomize_recharge_pads` | on when `randomize` **and** the schema has a `battery` block |
| `randomize_repair_pads` | on when `randomize` **and** the schema has a `damage` block |

Set any toggle to `true`/`false` to force it, overriding the schema-aware default. Because a recharge
pad is inert without battery physics and a repair pad without the damage block (auto-enabling either
would change the observation width), enabling `randomize_recharge_pads` without a battery-block
schema — or `randomize_repair_pads` without a damage-block schema — is a **fail-loud** config error
(exit 2), not a silent no-op. When both recharge and repair are on and the course has only one
eligible gate anchor (e.g. a 1-gate course), the two co-locate into a single dual-purpose pad. The
non-randomized path (`randomize: false`/unset) is byte-identical to before (no schema, no placement).

**Retrain note:** defaulting randomized runs to the 26-d schema changes the observation width, so
pre-UC-24 randomized-run checkpoints are invalidated and need a fresh retrain (same class of change
as the UC-13/15/17/19 schema extensions). See `configs/train/example.yaml` for the documented keys.

### Grounded / no-progress early termination (UC-25)
Training episodes used to waste almost the whole step budget with the drone lying motionless on the
floor: the numpy adapter's floor collision is `position[2] <= floor_z`, but a resting drone
asymptotes ~8 mm **above** the floor and never crosses it, so `collided`/`crash`/`terminated` never
fire and the episode only ended by truncation at the inflated `max_steps`. `EarlyTerminationConfig`
(the `EnvConfig.early_termination` block, **on by default**) adds two env-level detectors, evaluated
each `step()`: a **grounded** detector (the drone sits within `floor_epsilon` above `floor_z` at speed
`≤ rest_speed_epsilon`, and is not docked) and a **no-progress** detector (distance to the current
target gate — or to the finish on the last leg — fails to drop by more than `progress_epsilon`,
measured against the best distance reached so far). Either one persisting for `stuck_window`
consecutive steps ends the episode as a **crash** (`terminated=True`, the existing collision penalty,
and `info["collided"]=True`), with an additive `info["early_termination"]` key reporting `"grounded"`,
`"stuck"`, or `None`. The legitimate UC-16 docked/servicing state is **exempt** while service is
*productive* (battery or integrity strictly improving) — a drone that docks once then idles is still
cut once improvement stops. Because the counters start at 0 and need a full window of qualifying
steps, a normally-flying or promptly-crashing episode is byte-identical to before (no obs-schema,
width, or checkpoint change); set `early_termination.enabled = false` to restore the legacy behaviour.
All four thresholds are documented, tunable constants.

### Visualization & recording
Enable recording in a train/evaluate config with `record: true` (tune cadence via `record_every`);
frames land in that run's `training/<name>/recordings/`. Open `viz/viewer.html` in a browser
(dependency-free, `file://`-safe) and load a recording — no server or build step.

Two panels: an MRI/fMRI-style activation heatmap over a static, spatially-registered MaleCNS brain
outline (`viz/brain_outline.js`, `top-down` / `front` / `side` presets + a per-frame ↔ global
intensity toggle), and a **3D flight view** driven by the recorded `meta.course` geometry. 3D
controls: **drag** to **rotate**, mouse **wheel** to **zoom**, the same view presets, and a `0.25`×
slow-inspection speed. Regenerate the outline with `uv run python scripts/build_brain_outline.py`.

The 3D flight view renders the course's **landing pads** and **obstacles** as floor-anchored discs,
each with a legend entry. Pads are coloured by kind — **recharge** pads are green (`#00e676`),
**repair** pads deep orange (`#ff6d00`), and **plain** landing pads neutral grey (`#90a4ae`); a pad
that is *both* rechargeable and repairable renders with the **repair** colour (a display-only
precedence — the underlying pad still both recharges and repairs; env behaviour is unchanged).
Obstacles keep the existing purple (`#ba68c8`) pillar wireframe but now also draw a floor base ring +
filled disc so they read clearly against the floor grid. The pad kind is recorded additively in
`meta.course` (`pads[].kind`); legacy recordings without pads, obstacles, or the new kind field still
load and render unchanged (each new field read is guarded — graceful degradation).

Neuron coordinates use real MaleCNS **soma** positions; a neuron lacking one uses a deterministic
computed-layout **fallback** (clearly labelled). The committed fixture ships real anatomy
tokenlessly; arbitrary user slices may need a `NEUPRINT_TOKEN` (see `.env.example`), and
`uv run python scripts/fetch_soma_positions.py` refreshes the committed soma sidecar.

**Positions are provisioned at slice time, not per training run (UC-27).** When a connectome
artifact is created — by `prune` (the slice), `fetch-connectome` (the base download), or
`prune-trained` (the post-training subcircuit) — neuron positions are computed once and cached as a
`<stem>_positions.csv` sidecar beside the `.npz` (plus a `<stem>_soma.csv` sidecar of the real
anatomy). A recording-enabled training run then only **loads** that sidecar — no per-run neuPrint
fetch and no per-run spectral eigendecomposition. Key points:

- **Real anatomy is the tokenless default.** The primary anatomy source is the connectome's own meta
  CSV: the MaleCNS download already ships `mcns_all_neuron_meta.csv`, whose `somaLocation` column
  holds real neuPrint MaleCNS 8 nm-voxel soma coordinates (CC-BY). Provisioning reads it for the
  artifact's `bodyid`s **before any network call**, so a freshly fetched or pruned artifact gets real
  soma coordinates offline, with no token. Anatomy precedence:
  **meta `somaLocation` → `DRONE_FLY_SOMA_CSV` → `<stem>_soma.csv` → neuPrint → spectral.** neuPrint
  (a `NEUPRINT_TOKEN`) is only an optional supplement for anatomy absent from the meta; the dense
  spectral layout is a last resort for neurons with no `somaLocation` anywhere. Anatomically soma-less
  neurons (peripheral sensory afferents whose cell bodies sit outside the brain volume) are flagged
  missing and handled by the partial-anatomy fill — never faked.
- **Format.** One row per neuron: `bodyid,has_position,x,y,z,u,v,source,projection`. `x/y/z` are
  blank when a neuron has no 3-D anatomy; `has_position` is an independent flag. Floats are written
  at full `%.17g` precision, so a load reproduces the original compute exactly.
- **Node-set binding.** A sidecar is reused only when its neuron-id set and projection match the
  loaded connectome exactly; otherwise it is recomputed. A pruned subcircuit never picks up the full
  connectome's positions, and a stale/corrupt sidecar is never applied to the wrong neurons.
- **Self-heal.** An older artifact with no sidecar (sliced before this feature) still works: the
  recorder computes the layout once, warns, and persists the sidecar (best-effort — a read-only
  directory does not crash the run), so later runs take the fast load path.
- **Large connectomes.** Because the meta supplies real somas for the full connectome, `fetch-connectome`
  provisions real anatomy directly with **no** dense `eigh` over ~161k neurons (UC-27 AC-11). A dense
  spectral layout at that scale is infeasible, so if a compute would still need the fallback for more
  than `DRONE_FLY_SPECTRAL_MAX` neurons (default `50000`) **and** anatomy is missing/partial, the real
  soma sidecar is still written but the full-graph position layout is deferred with a warning: provide
  anatomy (`NEUPRINT_TOKEN` or `DRONE_FLY_SOMA_CSV`) or prune the connectome before recording.

### Requirements
- **Python 3.11** + [`uv`](https://docs.astral.sh/uv/); `uv sync --extra dev` installs everything
  for lint, tests, and `smoke-train`.
- The `simple` (numpy) adapter needs no native deps and runs anywhere. The `pybullet` adapter
  (full physics sim, for mastery training) needs a C/C++ toolchain and is verified on macOS +
  Xcode CLT; `./scripts/train.sh` bootstraps it and launches a config-driven run.
- CI gates: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`.

### Tested-vs-untested boundary
What CI actually exercises versus what needs the native sim, recorded from a real install attempt:

| Path | Status | Reality |
|------|--------|---------|
| `simple` numpy adapter, loader, prune, config, viz contract | tested (CI) | runs hermetically offline |
| `pybullet` full-physics adapter | untested in CI | needs a C/C++ toolchain; verified on macOS + Xcode CLT |
| `pybullet` via macOS conda (`setup-sim-macos.sh`) | untested in CI | no macOS/conda runner in CI; uses prebuilt conda-forge pybullet — validated on a managed macOS 26 arm64 box |

The gym-pybullet-drones pin is resolved with `git ls-remote` to a fixed commit SHA (never floating
`main`); `scripts/train.sh` and the adapter share that SHA and verify `import pybullet` before training.

**Mastery goal & dynamics.** The target is **80**% waypoint **mastery** (course completion). Mastery
runs use **fixed** environment dynamics (**no domain randomization**) for reproducibility; enable
randomization explicitly only for robustness experiments.

### macOS (Apple Silicon) real-physics sim setup (UC-20)
On modern macOS (26 / Tahoe, Apple Silicon) the `scripts/train.sh` from-source `pybullet` build
**cannot compile**: pybullet's vendored zlib (`zutil.h`'s `#define fdopen(fd,mode) NULL`) collides
with the macOS SDK's `_stdio.h` `fdopen` declaration, so no from-source pybullet (3.2.6, or the 3.2.7
that gym-pybullet-drones pins) builds — regardless of Xcode CLT / clang version. **From-source
pybullet is therefore unsupported on modern macOS SDKs.** Use the prebuilt path instead:

    ./scripts/setup-sim-macos.sh          # add --dry-run to print the plan without touching anything

It provisions the sim from **prebuilt binaries only**:

- **miniforge** — installed only after an explicit prompt (into `~/miniforge3`, user-space; never
  `sudo`, never silent; decline and it installs nothing and exits non-zero);
- a **`dronefly` conda env** (Python 3.12);
- **conda-forge prebuilt `pybullet` (3.2.5**, the newest arm64 build on conda-forge) — the C/C++
  compiler is never invoked for pybullet;
- **`gym-pybullet-drones`** at the exact commit pinned in `scripts/train.sh` (single source of
  truth — the SHA is read out of `train.sh`, never duplicated here), installed with
  **`pip install --no-deps`** so pip does not pull in and source-build a newer pybullet, plus its
  runtime deps installed explicitly;
- `pip install -e ".[dev]"` and the `drone-fly` console script.

**Why 3.2.5 + `--no-deps`:** gym-pybullet-drones pins `pybullet>3.2.7` conservatively, but the
`CtrlAviary` / `DroneModel` / `Physics` symbols this project uses run fine against conda-forge's 3.2.5;
`--no-deps` is what keeps pip from dragging in a source pybullet build. The explicit runtime-dep list
is hand-maintained to track the validated recipe, with the script's post-install import check
(`pybullet` + `gym-pybullet-drones` + `drone_fly`) as the backstop.

Afterwards `./scripts/train.sh` **auto-detects macOS** and runs training through the `dronefly` env
(it never attempts the doomed source build on macOS). If the prebuilt path cannot be satisfied, the
script **fails fast** and prints a ready-to-run **Linux-container** command (Apple `container` /
Docker) that installs the prebuilt manylinux `pybullet` wheel from PyPI — it never falls back to a
source build. The macOS/conda path has no CI runner, so (like the `pybullet` boundary above) it is an
**untested in CI** boundary; what CI *can* check is that the script is `shellcheck`-clean, parses under
`bash -n`, and is safely `--dry-run`-able.

### Project layout & history
Source under `src/drone_fly/` (connectome loader, controller/actor, adapter, env, train, evaluate,
record, clean); tests under `tests/`; the viewer under `viz/`; example configs under `configs/`.
Full project definition is in `PROJECT_BRIEF.md`; feature history and status live in
[`USE_CASES.md`](USE_CASES.md) and `use-cases/`.

## License

MIT (covers the drone-fly source only, not the upstream MaleCNS dataset).
