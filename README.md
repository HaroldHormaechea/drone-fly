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

**Windows support (UC-32).** The TUI now works on Windows rather than needing `--no-tui`:
parallel rollouts (`n_envs>1`) no longer crash with the TUI on (each spawned worker's native
output goes to a per-worker file under `<run>/logs/workers/` instead of a redirect that killed
Windows workers; a genuine worker failure now surfaces the *real* traceback, not an opaque
`EOFError`), the elapsed clock and liveness tick every second independent of PPO iteration
boundaries, the dashboard renders full-screen, Python logging is routed into the logs pane
(pybullet's native startup banner is redirected to `<run>/logs/native.log`), and the
`OSError [WinError 1]` logging spam is gone. On a terminal that cannot drive the full-screen
alternate-screen buffer (e.g. legacy `conhost.exe`) the TUI **fails fast** with an actionable
error telling you to use Windows Terminal or pass `--no-tui`. macOS/Linux behavior is unchanged.

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
finish. The randomizer (under the course-randomize axis) samples solvability-guarded pillars —
**as of UC-35** placed *between consecutive waypoints* so the drone is forced to evade them (see
"Course placement geometry" below) — and the 3D viewer draws them as wireframe cylinders. See
`default_obstacle_course()` for the fixed manually-placed default set.

### Recharge pads (UC-18)
A landing pad tagged `rechargeable` is a **recharge pad**: while the drone is *fully docked* on
it (the UC-16 docked state — a slow, upright, over-pad floor contact; hovering does **not**
count), the battery refills toward `1.0` at `BatteryConfig.recharge_rate` per second (clamped at
full). The rate exceeds the docked drain, so a dwell nets a gain. This reuses UC-17's battery
observation block — **no new observation dim, so checkpoints are not invalidated**. To make a
recharge worth the detour without a shaping reward, recharge is a **course-variation axis**
(`RandomizationConfig.enable_recharge`). **As of UC-24** the randomizer places **exactly one**
recharge pad per randomized course at an eligible anchor **regardless** of whether the course
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

### Course placement geometry (UC-35)
Two corrections to how randomized courses are laid out, both **placement-geometry only** — no change
to the observation schema, reward, connectome, or dynamics, and a non-randomized / disabled-axis run
still reproduces bit-for-bit (AC-5/6/7):

- **Pads off waypoints.** A recharge/repair pad is never placed on a gate column any more (docking on
  the point the drone already flies through was degenerate). The placer offsets each pad off its
  anchor gate — perpendicular to the local path first, then axial — to the first fixed candidate
  position that stays ≥ `pad_min_gate_distance` (`R_pad`) from **every** gate centre, inside the
  lateral corridor, and clear of every pillar's descend column. The search draws **no RNG**, so it
  only ever appends a pad and never shifts the sampled geometry.
- **Obstacles between waypoints, forced-but-evadable.** Pillars are sampled inside a perpendicular
  corridor around a waypoint→waypoint segment (including start→first-gate), close enough that the
  straight path passes within the pillar so the drone must deviate — while the solvability guard
  guarantees the course stays feasible (each pillar keeps gate-passability clearance from every
  waypoint, leaves an escape lane within `±lateral_bound`, and never forms an unevadable wall with a
  neighbour). Each pillar consumes a fixed number of draws whether placed or skipped, so the
  configured `obstacle_count_range` is an **upper bound** (a course may carry fewer, even zero on
  pathologically short courses).

New `RandomizationConfig` fields (all documented, sane defaults, tunable):

| Field | Default | Meaning |
|---|---|---|
| `pad_min_gate_distance` | `1.0` | `R_pad` — min horizontal distance a placed pad keeps from every gate centre |
| `drone_radius` | `0.15` | generation-time drone body buffer for clearance maths (not read by the sim collision test) |
| `obstacle_evasion_margin` | `0.2` | extra slack beyond `drone_radius` for squeezing past a pillar / off gates / off other pillars |
| `obstacle_corridor_half_width` | `0.75` | max perpendicular offset of a pillar axis from its segment centreline |
| `obstacle_gate_clearance` | `0.5` | sampler-side band kept from segment endpoints; eligible-segment half-length floor |
| `min_obstacle_separation` | `0.3` | extra pillar↔pillar spacing beyond `2·(drone_radius + evasion_margin)` |
| `obstacle_along_margin_frac` | `0.3` | fraction of each segment trimmed per end when drawing a pillar's along-position |

The pre-UC-35 `obstacle_lateral_offset_range` is retained (unshifted) but no longer read by the
sampler; `obstacle_clearance` is retained as the pad descend-column margin.

**Retrain note:** defaulting randomized runs to the 26-d schema changes the observation width, so
pre-UC-24 randomized-run checkpoints are invalidated and need a fresh retrain (same class of change
as the UC-13/15/17/19 schema extensions). See `configs/train/example.yaml` for the documented keys.

### Grounded / no-progress early termination (UC-25, UC-36)
Training episodes used to waste almost the whole step budget with the drone lying motionless on the
floor: the numpy adapter's floor collision is `position[2] <= floor_z`, but a resting drone
asymptotes ~8 mm **above** the floor and never crosses it, so `collided`/`crash`/`terminated` never
fire and the episode only ended by truncation at the inflated `max_steps`. `EarlyTerminationConfig`
(the `EnvConfig.early_termination` block, **on by default**) adds two env-level detectors, evaluated
each `step()`: a **grounded** detector (the drone sits within `floor_epsilon` above `floor_z` at speed
`≤ rest_speed_epsilon`, and is not docked) and a **no-progress** detector (distance to the current
target gate — or to the finish on the last leg — fails to drop by more than `progress_epsilon`,
measured against the best distance reached so far). Each detector has its **own** firing window: the
grounded detector fires after `grounded_window` consecutive grounded steps (default **10** = 0.5 s @
20 Hz — a floored drone is unambiguously dead, so its recording/eval episode is cut promptly instead
of dragging to the horizon, UC-36), while the no-progress detector keeps the more lenient
`stuck_window` (default **100** = 5 s @ 20 Hz) so a slow-but-recovering flight isn't cut prematurely.
Either one reaching its window ends the episode as a **crash** (`terminated=True`, the existing
collision penalty, and `info["collided"]=True`), with an additive `info["early_termination"]` key reporting `"grounded"`,
`"stuck"`, or `None`. The legitimate UC-16 docked/servicing state is **exempt** while service is
*productive* (battery or integrity strictly improving) — a drone that docks once then idles is still
cut once improvement stops. Because the counters start at 0 and need a full window of qualifying
steps, a normally-flying or promptly-crashing episode is byte-identical to before (no obs-schema,
width, or checkpoint change — the shorter `grounded_window` still can't fire in the never-grounded
golden fixtures); set `early_termination.enabled = false` to restore the legacy behaviour.
All five thresholds (`floor_epsilon`, `stuck_window`, `grounded_window`, `progress_epsilon`,
`rest_speed_epsilon`) are documented, tunable constants.

### Floor start & airborne survival reward (UC-37)
Early training used to "fall like a rock": the drone spawned **mid-air** (`start_z_range=(0.7,1.5)`)
and, with the policy's near-zero initial throttle, simply dropped from height, while the reward had
**no dense survival term** — so every episode ended at the −100 crash cliff, returns were flat
(~−101), and there was no gradient toward staying up. The task itself is easily solvable (the sim's
thrust-to-weight ratio is 2, so 50 % throttle hovers); the failure was a **bootstrap trap**, not a
physics limit. UC-37 fixes it with two paired changes:

- **Floor start (`EnvConfig.floor_start`, default `True`).** On reset the drone now spawns **resting
  on the floor** (`z ≈ course.floor_z`, ~zero velocity), the way a real drone begins — instead of
  the artificial mid-air start. The override is applied at the env layer *after* course sampling, so
  the RNG stream and determinism are untouched. Set `floor_start: false` to restore the legacy
  mid-air start. There is deliberately **no** hover-bias on the action: the neutral (zero) action
  still maps to ~zero throttle, so the policy must *learn* to command throttle and take off — a
  floor-start drone under zero action stays on the floor (it does not spontaneously lift).
- **Airborne survival reward (`RewardConfig.airborne_bonus`, default `0.1`).** A small per-step reward
  is paid **only while the drone is airborne** (above the floor band, `floor_z + floor_epsilon`) and
  is exactly **zero on/at the floor** — so the only path to reward is to throttle up and stay up. It
  is sized against two bounds: the net per-airborne-step reward (`airborne_bonus − time_penalty` =
  0.10 − 0.05 = **+0.05**) is strictly positive (a gradient toward takeoff), and the max survival
  reward over the default 3-gate episode (budget 800 steps → 0.1 × 800 = **80**) is below the
  `completion_bonus` (**100**), so loitering scores strictly worse than completing the course.

Two supporting rules keep this consistent with the earlier detectors: (1) a **pre-takeoff floor
contact is not a crash** — a grounded drone at zero throttle would otherwise insta-crash at step 1 —
suppressed only before the first takeoff and only within the floor band; and (2) the **grounded
early-termination detector arms only after the first takeoff** (a floor-start drone that never lifts
off is bounded instead by the no-progress/stuck detector and the episode timeout, so episodes never
run unbounded), while a drone that takes off and then floors is cut exactly as under UC-36. Any
episode that **starts airborne** (above the floor band — e.g. the scripted UC-25/UC-36 fixtures) is
considered "taken off" at step 0, so grounded-termination arms immediately and those paths stay
byte-for-byte unchanged.

**Honest large-N caveat.** The survival-vs-completion bound above is anchored to the **default**
800-step budget. For large randomized courses (N up to ~10, step budget up to ~2200) the *theoretical*
max survival reward (0.1 × 2200 = 220) exceeds `completion_bonus`; loiter-domination there does **not**
rest on the per-step arithmetic but on the no-progress/stuck detector (`stuck_window`, 100 steps)
cutting a non-progressing hover, plus the forgone per-gate and completion bonuses.

This UC intentionally changes default training dynamics, so the committed seed-42 golden rollout
(`tests/data/uc08_baseline_rollout.npz`, regenerated via `scripts/regen_uc08_baseline.py`) legitimately
reflects the new floor-start default; airborne-start reward/termination paths and determinism are
preserved.

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

**Complete coverage on real coordinates (UC-28).** The heatmap renders **every** neuron at real
coordinates with none silently dropped. Neurons with a real soma (the central "brain") render at
their true anatomy and are the visually dominant element. Genuinely soma-less peripheral afferents
(cell bodies outside the imaged brain) are placed in a **schematic fly body around the brain**,
grouped by their real categorical body-region label — `subclass` for limbs (`leg`, `haltere`,
`campaniform`) and `superclass` for the coarse groups (`vnc_sensory` → `vnc`, `sensory_ascending`
→ `ascending`), with any generic/unrecognised label falling to a neutral `torso` group (never a
specific limb). The label rules (`REGION_LABEL_RULES`, limb `subclass` wins over coarse
`superclass`) and cluster offsets (`REGION_CLUSTER_OFFSETS`) live in
`src/drone_fly/record/coordinates.py`. Placement is **deterministic** (a neuron's exact spot in
its cluster is a `zlib.crc32` function of its bodyid — bit-reproducible across processes, never
`hash()`) and **honest**: no fabricated precise xyz — `coords3d` stays real-or-null and the
schematic render coords live in a separate always-finite `display3d` field, flagged
`placement="schematic"`. A **brain-scale guardrail** keeps the brain dominant: body clusters are
offset by a bounded multiple of the brain's own bounding box (`MAX_BODY_OFFSET_FACTOR` +
`REGION_CLUSTER_RADIUS_FRAC`), so the brain stays ≥ `BRAIN_DOMINANCE_MIN_FRACTION` of the total
rendered extent. In the viewer, schematic (body) neurons are drawn fainter than real-anatomy
neurons and are spatially separated, so a schematic dot is never mistaken for a real soma.

**Modality toggle (UC-28).** The brain-map panel has a **modality** selector that overlays rings
on the UC-13 modality populations — `vision`, `proprioceptive`, and `hunger` — on top of the hot
activation colormap (it does not replace the colours). The populations are the real biological
labels tagged per-neuron in `meta.modality` at record time (fail-soft: a modality absent from the
slice is simply not tagged). `damage`/nociception is **unavailable** — MaleCNS ships no nociceptive
label and there is no modality rule for it — so it is documented as absent rather than faked.

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
- **Format.** One row per neuron:
  `bodyid,has_position,x,y,z,u,v,source,projection,placement,region,x3d,y3d,z3d`. `x/y/z` are
  blank when a neuron has no 3-D anatomy; `x3d/y3d/z3d` (the UC-28 full-coverage `display3d`
  render coords) are always finite; `has_position` is an independent flag. Floats are written at
  full `%.17g` precision, so a load reproduces the original compute exactly. A sidecar written
  before UC-28 lacks the last five columns, so it is treated as a miss and recomputed (self-heal).
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
- CI gates: `uv run ruff check .`, `uv run ruff format --check .`, and pytest. Run the tests
  locally with `uv run --extra dev pytest` — pytest ships in the `dev` extra, so the bare
  `uv run pytest` fails to collect unless the extra is already synced (CI runs `uv sync --extra
  dev` first, then `uv run pytest`).

### Tested-vs-untested boundary
What CI actually exercises versus what needs the native sim, recorded from a real install attempt:

| Path | Status | Reality |
|------|--------|---------|
| `simple` numpy adapter, loader, prune, config, viz contract | tested (CI) | runs hermetically offline |
| `pybullet` full-physics adapter | untested in CI | needs a C/C++ toolchain; verified on macOS + Xcode CLT |
| `pybullet` via macOS conda (`setup-sim-macos.sh`) | untested in CI | no macOS/conda runner in CI; uses prebuilt conda-forge pybullet — validated on a managed macOS 26 arm64 box |
| Windows CUDA via `setup-sim-windows.ps1` | untested in CI | no Windows/GPU runner in CI; opt-in CUDA 12.4 torch wheel index — the real GPU run is a documented manual step (verified on an RTX 3050, 8 GB, Ampere/sm_86) |

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

### Windows (NVIDIA CUDA) real-physics training (UC-31)
On a Windows PC with an NVIDIA GPU (e.g. an RTX 3050, 8 GB, Ampere/sm_86) you can train the large
connectome slice on the **GPU** instead of the CPU. Unlike the Mac's MPS backend, CUDA has real
sparse-tensor support, so the sparse connectome propagation (the bottleneck) runs on the GPU. The
runtime already targets CUDA — `resolve_device()` auto-selects `"cuda"` when `torch.cuda.is_available()`,
and `device: cuda` in the train config forces it — so this is environment setup + docs only; training
dynamics, the observation schema, the connectome, reward, and checkpoints are unchanged.

Provision it with the PowerShell mirror of the macOS script:

    ./scripts/setup-sim-windows.ps1              # add -DryRun to print the plan without touching anything
    ./scripts/setup-sim-windows.ps1 -Help        # usage

It runs entirely in **user space (no admin)** and:

- prompts before installing **uv** (into your user profile; decline and it installs nothing and exits
  non-zero);
- checks the NVIDIA driver via `nvidia-smi` and **fails with an actionable message if the reported CUDA
  version is older than 12.x** (the torch wheel targets CUDA 12.4);
- creates/reuses an isolated **`.venv-cuda`** venv (Python 3.12 — gym-pybullet-drones needs ≥ 3.12);
- installs a **pinned CUDA 12.4 PyTorch wheel** (`torch==2.6.0`) from the PyTorch CUDA wheel index
  (`--index-url https://download.pytorch.org/whl/cu124`), with the index **scoped to the torch line
  only** — it never becomes the global resolver;
- installs `pybullet==3.2.6` + `numpy<2` + the `drone-fly` project (`-e ".[dev]"`) in a single PyPI
  resolve (so the `numpy<2` ABI co-satisfies pybullet), then `gym-pybullet-drones` at the exact commit
  pinned in `scripts/train.sh` (single source of truth) with `--no-deps` plus its runtime deps
  explicitly — `--no-deps` keeps its `pybullet>3.2.7` pin from clobbering the pinned `3.2.6`;
- verifies `torch.cuda.is_available()`, prints the GPU name, imports `pybullet` / `gym_pybullet_drones`
  / `drone_fly`, and asserts `resolve_device() == "cuda"`.

Then train / benchmark on the GPU (device comes from the config, not a CLI flag):

    # in your train config: set `device: cuda` (or omit it for the auto-policy) and a small `timesteps`
    .venv-cuda\Scripts\drone-fly train --config configs\train\example.yaml

The startup log prints the resolved `device=cuda`, and Stable-Baselines3 emits per-iteration
`time_elapsed` / `fps` — compare one iteration to the ~821 s CPU and ~2.57 s-per-iter MPS baselines.

**8 GB VRAM tuning.** 8 GB is tight for the ~122k-neuron slice. On a **CUDA out-of-memory** error, lower
`n_envs` and/or `batch_size` in the train config (and/or train a smaller pruned slice); halve them until
the run fits, then tune back up. The same guidance (`CUDA_OOM_HINT`) is logged whenever CUDA is selected
and printed by the setup script's verification step.

**Isolation & CI.** The CUDA torch index and pin live **only** in `setup-sim-windows.ps1` — `pyproject.toml`
and `uv.lock` are untouched, so the default Linux/macOS install and the hermetic Linux CI resolve are
byte-for-byte unchanged (this is an opt-in Windows path). Like the macOS/conda boundary above, the
Windows/GPU path has no CI runner and the real GPU run is a **documented manual step**; what CI *can*
check is the monkeypatched CUDA device branch and (where PSScriptAnalyzer / `pwsh` are available) that the
script is analyzer-clean and parses. **Note:** `pybullet==3.2.6` ships no Windows/py3.12 wheel, so on
Windows it builds from source — if that step fails, install **Microsoft C++ Build Tools** ("Desktop
development with C++") and re-run; the script surfaces this hint on failure.

### Project layout & history
Source under `src/drone_fly/` (connectome loader, controller/actor, adapter, env, train, evaluate,
record, clean); tests under `tests/`; the viewer under `viz/`; example configs under `configs/`.
Full project definition is in `PROJECT_BRIEF.md`; feature history and status live in
[`USE_CASES.md`](USE_CASES.md) and `use-cases/`.

## License

MIT (covers the drone-fly source only, not the upstream MaleCNS dataset).
