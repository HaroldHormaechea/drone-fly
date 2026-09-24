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
Either one reaching its window ends the episode (`terminated=True`), with an additive
`info["early_termination"]` key reporting `"grounded"`, `"stuck"`, or `None` as the authoritative
reason. **UC-38 decoupled the penalty from the cut, and UC-58 made every early-termination cut
penalty-free** (see the UC-58 note below): ONLY a genuine floor/ceiling/out-of-bounds collision keeps
the `collision_penalty` and sets `info["collided"]=True`. The **grounded** cut (a previously-airborne
drone that dropped back onto the floor), the **no-progress ("stuck")** cut, and a pure `max_steps`
timeout all terminate **penalty-free** with `info["collided"]=False`. UC-58 flipped the grounded cut
from paying the penalty to penalty-free on purpose: with the redesigned reward there is no per-step
`time_penalty` to "escape", so a gentle grounded-rest after a failed takeoff must not cost −100 (that
would re-create the early-termination trap the reward redesign eliminates); the cut still BOUNDS the
episode, it just no longer punishes it. The legitimate UC-16 docked/servicing state is **exempt** while service is
*productive* (battery or integrity strictly improving) — a drone that docks once then idles is still
cut once improvement stops. Because the counters start at 0 and need a full window of qualifying
steps, a normally-flying or promptly-crashing episode is byte-identical to before (no obs-schema,
width, or checkpoint change — the shorter `grounded_window` still can't fire in the never-grounded
golden fixtures); set `early_termination.enabled = false` to restore the legacy behaviour.
All five thresholds (`floor_epsilon`, `stuck_window`, `grounded_window`, `progress_epsilon`,
`rest_speed_epsilon`) are documented, tunable constants.

### Floor start & airborne survival reward (UC-37)
> **Partly superseded by UC-58.** The **floor start** (`EnvConfig.floor_start`, default `True`) is
> KEPT. The **airborne survival reward** (`airborne_bonus`) described here was replaced by the single
> UC-58 Altitude reward and no longer exists as a separate term. The historical prose below still
> references `airborne_bonus` / `time_penalty` / `climb_target_height` — see the reward table and
> [Takeoff-oriented reward redesign (UC-58)](#takeoff-oriented-reward-redesign-uc-58) for the current
> reward.

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
  mid-air start. (Historical note: UC-37 originally shipped with *no* action bias — the neutral
  action mapped to ~zero throttle — but that left the policy unable to discover takeoff. UC-40 added
  a fresh-build hover-bias and **UC-44 supersedes it with a climb-bias** (`CLIMB_BIAS_THROTTLE` 0.6)
  so a fresh policy's default action now produces gentle net-positive lift; see the
  [UC-44 section](#airborne-start-reverse-curriculum--climb-biased-init-uc-44). A resumed checkpoint
  keeps its trained bias.)
- **Airborne survival reward (`RewardConfig.airborne_bonus`, default `0.2`).** A small per-step reward
  is paid **only while the drone is airborne** (above the floor band, `floor_z + floor_epsilon`) and
  is exactly **zero on/at the floor** — so the only path to reward is to throttle up and stay up. Since
  UC-42 it is **altitude-graded** (see the [UC-42 section](#takeoff-gradient-graded-airborne-survival-reward-uc-42)):
  the payout scales linearly with height toward `climb_target_height`, so holding a higher altitude
  pays strictly more. It is sized against two bounds: at the target the net per-airborne-step reward
  (`airborne_bonus − time_penalty` = 0.20 − 0.05 = **+0.15**) is strictly positive (a gradient toward
  takeoff), and the max survival reward over the default 3-gate episode (budget 800 steps → 0.2 × 800 =
  **160**, plus the climb-term bound 1.98 = **161.98**) is below the `completion_bonus` (**200**), so
  loitering scores strictly worse than completing the course.

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
max survival reward (0.2 × 2200 = 440) exceeds `completion_bonus`; loiter-domination there does **not**
rest on the per-step arithmetic but on the no-progress/stuck detector (`stuck_window`, 100 steps)
cutting a non-progressing hover, plus the forgone per-gate and completion bonuses.

This UC intentionally changes default training dynamics, so the committed seed-42 golden rollout
(`tests/data/uc08_baseline_rollout.npz`, regenerated via `scripts/regen_uc08_baseline.py`) legitimately
reflects the new floor-start default; airborne-start reward/termination paths and determinism are
preserved.

### Decoupled early-termination penalty (UC-38)
> **Updated by UC-58.** UC-38 kept the **grounded** cut paying `collision_penalty` (only the stuck /
> timeout cuts were made penalty-free). UC-58 went further and made the **grounded** cut penalty-free
> too — see the [early-termination section](#grounded--no-progress-early-termination-uc-25-uc-36) above.
> The bullet below describing the grounded cut as still paying the penalty is historical.

Even with UC-37's survival reward, training still converged to a non-flying policy (`ep_rew` pinned
near **−105**, 0 % success): a floor-sitting or non-progressing drone was cut by the no-progress
detector, and `racing_env.py` folded **every** early-termination cut into `crash=True`, so the cut
ate the full `collision_penalty` (100) — scoring ≈ −5 − 100, identical to a genuine floor crash. The
+airborne survival differential (≈ +10 over a 100-step hover) was swamped by the −100 terminal, so
takeoff never out-scored sitting. UC-38 **decouples the no-progress/timeout cut from the collision
penalty**:

- **No-progress ("stuck") cut and pure `max_steps` timeout are penalty-free.** The episode still
  ends (stuck → `terminated=True`; timeout → `truncated=True`), but the step reward carries **no**
  `collision_penalty` and `info["collided"]=False`. The reason is reported via
  `info["early_termination"]` (`"stuck"` or `None`) — `info["collided"]` no longer falsely claims a
  collision for a no-progress cut.
- **Genuine collisions and the grounded cut still pay the penalty.** A real floor/ceiling/OOB
  contact (raw `state.collided`, not a controlled dock) is byte-for-byte unchanged, and the
  **grounded** cut (a previously-airborne drone that dropped back onto the floor) keeps the penalty
  and `info["collided"]=True` — a post-takeoff drop to the floor is a failed flight. A genuine
  collision that coincides with a stuck cut on the same step still pays the penalty (the real
  collision dominates).

This restores a clean positive gradient — hovering 100 steps then being cut (≈ +5) now beats sitting
on the floor for the same window (≈ −5) — without weakening the crash / floor-shortcut ordering, and
without rewarding loitering (a valid completion's +`completion_bonus` still dominates the bounded
loiter return). Control flow is byte-for-byte as before UC-38; only reward magnitude and
`info["collided"]` on a stuck/timeout cut changed, so no golden fixture regeneration is needed (the
committed rollouts store observations/actions, not rewards, and the seed-42 baseline fires no cut).
Paired with a small exploration bump, `TrainConfig.ent_coef` default **0.0 → 0.01**, to sustain
exploration long enough for the policy to discover takeoff before entropy decays.

### Takeoff bootstrap: crash-cliff relief & climb reward (UC-39)
After UC-37/38 the "no learning signal" pathology was fixed and hovering was made net-positive, yet
training still converged to a non-flying policy (`ep_len` ≈ 101, 0 % success, `ep_rew` ≈ −5): the
drone spawned, sat, and was cut by the no-progress detector. The `sit`-vs-`hover` *terminal* ordering
was already correct (hover ≈ +5 > sit ≈ −5), so the blocker was **not** that sitting is too
comfortable — it was that **every path from sitting to hovering runs through a genuine floor/ceiling
collision** (`collision_penalty` = 100). PPO propagates that −100 (discounted by γ) back onto the
"throttle up" actions that begin any takeoff, so those correct actions receive **negative advantage**
and the policy learns "attempting flight leads to disaster — don't". UC-39 attacks that barrier
directly with two coupled levers.

**Reward table (single source of truth).** Every term the env applies, with its shipped
`RewardConfig` value and when it fires.

> **Reward redesigned from scratch in UC-58.** Once the plant became flight-capable (UC-55 rate loop
> + UC-56 Meteor75 T/W 2.5), the reward accreted across UC-37→50 was shown to *incentivize early
> termination* (reward-vs-episode-length correlation **−0.87** — shorter episodes scored better). The
> `time_penalty`, `airborne_bonus`, UC-39 `climb_*`, UC-43 `ground_break_*`, and UC-50 `altitude_hold_*`
> terms were **retired** and replaced by ONE sustained, level-based, saturating **Altitude** reward
> paid every step from `h=0`. The table below is the CURRENT reward; the UC-37/39/42/43/50 narrative
> in the following sections is kept as historical context (see
> [Takeoff-oriented reward redesign (UC-58)](#takeoff-oriented-reward-redesign-uc-58) for the rationale).

| Action | Reward | Condition |
|---|---|---|
| Progress | +1.0 × Δdist | per step, for closing distance to the current target waypoint (`progress_weight`) |
| Altitude (takeoff / hold) | +0.2 × min(h, target)/target | **per step from h=0** — sustained, level-based, saturating: 0 at the floor, ramping linearly to +0.2 at `altitude_target` (1.0 m) and flat above it. The takeoff/hold signal (`altitude_weight` / `altitude_target`, UC-58) |
| Gate passed | +10 / N | on a validly passed gate, normalised by gate count N (`gate_bonus`) |
| Course completed | +200 | on a valid all-gates-then-finish (`completion_bonus`) |
| Collision (floor/ceiling/OOB) | −100 | on a genuine crash; terminates the episode (`collision_penalty`) |
| Obstacle contact | −50 | edge-triggered once per distinct pillar contact; non-terminating (`obstacle_penalty`) |
| Grounded / stuck / timeout cut | 0 | **penalty-free** (UC-58/38) — a post-takeoff grounded-rest, a no-progress ("stuck") cut, or a pure `max_steps` timeout ends the episode but is never punished (only a *genuine* crash pays `collision_penalty`) |

The reward-column values are the env **defaults** (`RewardConfig` constants); keep this table in sync
with any future reward change. **Control-rate scaling (UC-57).** The single *per-step, time-extensive*
term — **Altitude** — is multiplied by `per_step_scale = dt / BASELINE_DT` (`BASELINE_DT = 0.05` = the
20 Hz baseline; `= 1.0` at 20 Hz, `= 0.4` at the 50 Hz run default) so its *per-episode integral* is
invariant to the control rate: at 50 Hz an episode has ~2.5× more steps, each paying 0.4×, for the
same total (this preserves the documented loiter < completion ordering across rates). All other terms
are left unscaled — Progress telescopes over distance (path-, not time-extensive), and the
gate/completion/collision/obstacle terms are per-event. **Default-on:** the redesigned reward ships on
for every fresh run; `altitude_weight` / `altitude_target` are exposed in the train and evaluate YAML
(defaults 0.2 / 1.0). There are no longer any potential-based shaping terms, so the old
`climb_gamma == training γ` coupling is gone.

> **Superseded by UC-58.** Both levers below (the `climb_*` potential and the training-time
> collision-penalty curriculum) were **retired** in UC-58 — the `climb_weight` / `climb_target_height`
> / `climb_gamma` fields, the `collision_penalty_start` / `_end` / `_hold_fraction` /
> `_warmup_fraction` / `_enabled` `TrainConfig` fields, the `set_collision_penalty` env method, and the
> `drone_fly.train.collision_curriculum` module no longer exist. They are kept below as historical
> context. See [Takeoff-oriented reward redesign (UC-58)](#takeoff-oriented-reward-redesign-uc-58).

- **Lever 1 — dense potential-based climb reward (`RewardConfig.climb_weight` = 2.0,
  `climb_target_height` = 1.0 m, `climb_gamma` = 0.99; default on).** A small per-step reward pays for
  upward progress from a floor start toward the hover target, on the very step altitude is gained —
  *before and independent of* any later crash — so the per-action advantage of the initial "throttle
  up" actions is positive even on a takeoff that later crashes. It is **potential-based** shaping (Ng
  et al. 1999): `Φ(h) = climb_weight · min(max(h, 0), climb_target_height)` and the per-step
  contribution is `F = climb_gamma · Φ(curr) − Φ(prev)`, where `h` is altitude above the floor. This
  form is deliberate: it **telescopes**, so a climb-then-descend round trip nets ≈ 0 (it cannot be
  farmed into a loiter optimum by bobbing); its per-episode total is bounded by ≈ `climb_gamma ·
  climb_weight · climb_target_height` = **1.98**, far below `completion_bonus` (200) and at/below a
  normalised `gate_bonus`; it is ≈ 0 on the floor; and it **caps at the target height**, so there is
  no incentive to climb into the ceiling. **Coupling note:** `climb_gamma` **must** equal the training
  discount γ (`TrainConfig.gamma`, 0.99) for the shaping to stay policy-invariant — if you change the
  training γ, update `climb_gamma` in lockstep.
- **Lever 2 — crash-cliff relief via a training-time collision-penalty curriculum (default on).**
  During training the genuine floor/ceiling/OOB collision penalty follows a **hold-then-ramp**
  schedule: it is **held at `TrainConfig.collision_penalty_start` (2) through the first
  `collision_curriculum_hold_fraction` (0.4) of `total_timesteps`** (0–40%, the whole fly-learning
  phase), then **ramped linearly up to `collision_penalty_end` (100)** over the next
  `collision_penalty_warmup_fraction` (0.1, the ramp width; the YAML key — the underlying
  `TrainConfig` field is `collision_curriculum_warmup_fraction`) of `total_timesteps` (40%→50%), then
  **held at the full value 100** for the remainder (50–100%). *(UC-51 restaggered the ramp width
  0.5→0.1 so the collision step reaches full strength at ~50% and is isolated from the other two
  curricula — see the [UC-51 section](#restaggered-curriculum-schedule--exposed-knobs-uc-51).)*
  Holding a low penalty through the fly-learning phase
  removes the crash barrier while the drone learns to fly — UC-41 found that the earlier from-t=0
  linear ramp re-erected the crash cliff (to ~33 by 16% of training) before the policy had learned
  to fly, so a *failed* takeoff (which trips the grounded cut that pays the collision penalty) stayed
  worse than the penalty-free do-nothing floor and the policy committed to do-nothing. Ramping the
  penalty back to full strength restores precision so the drone does not learn permanently-sloppy
  floor/ceiling-clipping flight. The schedule is implemented as an SB3 callback
  (`drone_fly.train.collision_curriculum`) that pushes the current value into the training envs each
  rollout via `env_method("set_collision_penalty", …)`; it is a pure function of `num_timesteps`, so
  a resumed run continues it correctly. Crucially, this is **training-only**: the env's *default*
  `RewardConfig.collision_penalty` stays **100** (the value in the table above), and termination is
  never affected — a genuine crash still terminates the episode and is still penalised, just at the
  active curriculum magnitude. Set `collision_curriculum_enabled = False` to train at the constant
  default. As a companion exploration guard, UC-41 raises `ent_coef` to **0.005** (still strictly
  positive) so the action std keeps exploring long enough for the relieved crash-cliff to make
  takeoff→progress the higher-advantage path.

Together these make the first increments of flight net-positive in expected advantage instead of
punished, without breaking any UC-03/16/37/38 reward ordering: completion still dominates loitering,
a floor-shortcut still loses to a valid completion (the end-value penalty stays 100), hover still
beats sit, and a full hovering episode still beats a takeoff-then-immediate-crash episode even at the
curriculum's lowest endpoint (there is no grounded penalty, so no suicide optimum).

### Takeoff gradient: graded airborne survival reward (UC-42)
> **Superseded by UC-58.** The graded `airborne_bonus` term (and the `completion_bonus` 100→200 bump
> made to preserve its bound) were folded into the single UC-58 Altitude reward; `airborne_bonus` no
> longer exists. `completion_bonus` stays 200. Kept below as historical context.

UC-41 removed the crash-cliff *punishment* and that half worked — a fresh pybullet run no longer
crashes, freezes, or trips the K0 verdict; the actor holds a calm hover. **But it still never took
off:** across three recordings the drone stayed pinned at the floor (z ≈ 0.0135 m), throttle sitting
right at hover, reward at the pure time-penalty floor. Removing the punishment did not create enough
*pull* toward sustained lift, so the policy settled into a **penalty-free hover-rest local optimum**.

The diagnosis (evidence-backed, all three candidate levers adjudicated):

- **Climb/airborne magnitude — the primary cause.** With the old **flat** `airborne_bonus` (0.1), the
  only durable altitude reward, net of the time penalty and the discounted climb-potential *leak*
  (`(γ−1)·climb_weight·climb_target = −0.02/step`), a sustained hold netted only **+0.03/step** over
  hover-at-floor — the same thin margin UC-38 already showed was too weak. Worse, because the bonus
  was **flat**, net-hold as a function of altitude `h` was `0.05 − 0.02h`: *decreasing* in `h`, so
  holding a higher altitude was actually slightly **worse**. The reward had the right sign but no real
  climb gradient.
- **Exploration (`ent_coef`) — ruled out.** The fresh run showed exploration alive (≈47 % of steps
  above 0.6 throttle, no std-collapse / freeze / K0), so the UC-40/41 guards hold; `ent_coef` stays
  **0.005**.
- **Thrust authority — ruled out on physics evidence.** A pybullet probe confirmed thrust-to-weight
  ≈ 2.25 (throttle 0.9 lifts z from 0.02 → 2.66 m in 14 steps). The drone *can* climb trivially; the
  blocker is purely that sustained above-hover throttle was never reinforced.

The fix is a single, minimal lever set (no change to `racing_env.py`, `ent_coef`, the collision
curriculum, or the UC-40 hover-bias):

- **Altitude-graded airborne reward.** The airborne survival term is now scaled by fractional height
  toward the target: `airborne_bonus · min(max(h, 0), climb_target_height) / climb_target_height`
  while airborne, else 0. This flips the perverse gradient into a **monotone climb-to-target pull**:
  net-hold as a function of altitude becomes `0.18h − 0.05` (break-even at `h ≈ 0.28 m`, **+0.13/step
  at the target** — ≈ 4.3× the old +0.03), and it **saturates flat above the target** so there is no
  ceiling-seeking. It is still exactly 0 on the floor (non-farmable) and the per-episode max is still
  `airborne_bonus × budget`.
- **`airborne_bonus` 0.1 → 0.2** to give that gradient enough magnitude to survive PPO advantage
  normalization.
- **`completion_bonus` 100 → 200** strictly as **forced bound-preservation**: with `airborne_bonus`
  at 0.2 the per-episode airborne max over the default 800-step budget is 160 (plus the climb bound
  1.98 = 161.98), which would exceed the old 100 and break the loiter < completion invariant. 200
  restores it tight (headroom ≈ 1.23). It is **not** a takeoff signal and is kept deliberately tight
  because reward VecNormalize is on — a larger completion spike would only inflate the return-std the
  normaliser divides by.

All UC-37/38/39 invariants are preserved: the telescoping climb term is untouched (round-trip nets
≈ 0, non-farmable), the graded survival reward is ≈ 0 at rest and capped at the target, `climb_gamma`
still equals the training γ, and the penalty-free stuck-cut / anti-suicide orderings are intact.
`climb_weight`, `climb_target`, `time_penalty`, `progress_weight`, `collision_penalty`, `gate_bonus`,
and `ent_coef` are all unchanged.

### Sub-threshold ground-breaking reward + reward-system audit (UC-43)
> **Superseded by UC-58.** The `ground_break_*` potential fixed a sub-threshold dead zone that the
> UC-58 Altitude reward eliminates structurally (it pays from `h=0`, so there is no dead zone to
> bootstrap over). `ground_break_weight` / `ground_break_height` no longer exist. Kept below as
> historical context; the "desired outcome" audit table further down is updated for the UC-58 reward.

UC-40/41/42 removed the freeze, the crash-cliff, and the perverse once-airborne gradient — yet a
fresh full-stack run (~99k steps) still **never breaks ground at all**. Every sampled recording pins
z at ~0.0135 m (resting height — no hop, no lift-and-drop) and `ep_rew_mean` is glued to exactly
**−5** = `−time_penalty (0.05) × stuck_window (100)`. That −5 is diagnostic: any airborne time at all
would add the airborne bonus and lift the mean, so the **entire episode population is planted**, not
just the sampled ones. The clipped throttle mean sits flat at hover (~0.48) and is not rising.

**Root cause — a sub-threshold dead zone (chicken-and-egg).** The airborne survival flag only trips
at `z > floor_z + floor_epsilon` (≈ 0.05 m; `racing_env.py:384`), so UC-42's graded airborne reward —
and *any* airborne credit — never activates while the drone rests at ~0.0135 m. The only reward
active below the threshold is the UC-39 climb potential, and it is (a) **return-invariant** — it
telescopes to ≈ 0 for the tiny transient hops a resting drone makes — and (b) too weak (≈ 0.027/step
at its strongest vs. the −0.05 time penalty). Net sub-threshold signal is negative, so PPO gets no
gradient favouring higher throttle → the throttle mean never rises → nothing ever gets airborne to
collect the airborne/climb reward. Breaking ground needs *sustained* consecutive above-hover throttle
that per-step exploration noise around a 0.48 mean effectively never produces on its own.

The `racing_env.py:384` airborne gate is **correct-by-design, not a bug** — it defines what "airborne"
means for the survival term. So the fix is purely additive in `reward.py` / `config.py`;
`racing_env.py` is unchanged.

**The fix — a second potential-based "ground-breaking" reward** (`RewardConfig.ground_break_weight`
= 0.5, `ground_break_height` = 0.05 m; default on) active in the sub-threshold band, saturating
exactly at the airborne threshold for a clean handoff to UC-42:

```
Φ_gb(h) = ground_break_weight · min(max(h, 0), ground_break_height) / ground_break_height
F_gb    = climb_gamma · Φ_gb(curr) − Φ_gb(prev)     # reuses climb_gamma (== training γ)
```

Properties (all covered by a deterministic reward-math test):

- **Dense sub-threshold gradient.** Slope `ground_break_weight / ground_break_height` = **10/m**
  (5× the climb slope), so a genuine break from rest pays strongly — a 0.0135 m hop earns ≈ 0.13 of
  ground-break shaping — while staying planted earns strictly less.
- **≈ 0 at rest** (`Φ_gb(0) = 0`) and **non-farmable**: it telescopes, so a bob (up then back down)
  nets `(γ−1)·ΣΦ ≤ 0`. There is no reward for hovering-in-place at the floor.
- **Continuous handoff, no double-count.** `ground_break_height` equals the airborne threshold
  `EarlyTerminationConfig.floor_epsilon` (an explicit, documented coupling — the UC-43 test asserts
  the equality against its source, not an independent literal), so Φ_gb saturates exactly where the
  airborne flag trips. Above the threshold Φ_gb is flat ⇒ `F_gb = (γ−1)·ground_break_weight =
  −0.005/step`, height-independent — a benign constant leak, not a second altitude reward stacked on
  UC-42's.
- **Bound preserved.** The per-episode ground-break contribution is bounded by `climb_gamma ·
  ground_break_weight` = **0.495**. Combined shaping is now 160 (airborne) + 1.98 (climb) + 0.495
  (ground-break) = **162.48 < `completion_bonus` 200**, so the loiter < completion invariant still
  holds. (The **honest large-N caveat** on the airborne bound above is unchanged — adding 0.495 does
  not change that story.)
- **Sizing.** `ground_break_weight` < 0.9 is the hard seam-monotonicity bound (the net-hold band
  slope `0.18 − 0.2·w` must stay > 0); 0.5 keeps a +0.08/m margin and a strong transient. It is **not**
  shrunk below that.

**Honest scope note (Risk 1).** Potential-based shaping is **return-invariant** by construction: a
steeper sub-threshold potential strengthens the per-step learning signal for the first centimetres of
lift, but it does **not** by itself change the episodic optimum for an un-sustained hop. So this fix
is **necessary but maybe not sufficient** — it makes the right actions pay per-step, which is what PPO
reinforces, but confirming *behavioral* takeoff-learning is the user's pybullet GPU retrain, not this
gate. If that retrain shows exploration still can't produce the *sustained* above-hover throttle,
the documented next levers are temporally-correlated exploration (OU / pink noise) and/or a higher
fresh-build throttle-bias init (extending the UC-40 hover-bias above 0.5); `ent_coef` is already 0.005.

**Reward-system audit.** Per the governing directive — *any* nudge toward *any* desired outcome, no
matter how small, must be rewarded, with no dead zone where genuine incremental progress earns zero
or goes negative — every desired outcome has a dense, non-farmable crediting gradient. **Updated for
the UC-58 reward** (the four accreted altitude terms audited here were replaced by the single
level-based Altitude reward):

| Desired outcome | Crediting gradient (UC-58) | Verdict |
|---|---|---|
| Break ground `[0, 0.05 m]` | the **Altitude** reward pays from `h=0` (strictly increasing, no dead zone) | ✔ (dead zone eliminated structurally) |
| Climb to target `[0.05, 1.0 m]` | the **Altitude** reward, linear to +0.2 at `altitude_target` (1.0 m) | dense, non-farmable ✔ |
| Hold altitude at/above target | the **Altitude** reward saturates flat at +0.2 — a sustained, level-based signal (not a telescoping potential that nets ≈0) | ✔ (sustained, anti-suicide) |
| Reduce distance to next gate | `progress` term (dense, telescoping) | ✔ |
| Pass a gate | `gate_bonus` event + the progress approach gradient | ✔ |
| Complete the course | `completion_bonus` + progress/gate gradient (dominant, hierarchy on top) | ✔ |
| Above target / into ceiling | intentionally saturated (no ceiling-seeking) | correct — not a desired outcome ✔ |

Per-term verdicts (fixing only the clearly-wrong, evidence-backed takeoff-blocker; documenting the
rest rather than speculatively rewriting):

- **`time_penalty`** — correct; kept. The fix adds the missing positive term, it does not remove the
  time cost.
- **`progress`** — healthy and telescoping; a weak, geometry-dependent vertical component is noted but
  is *not* a bootstrap and is left unchanged (a deferred follow-up, to avoid double-shaping).
- **climb potential (UC-39)** — correct but return-invariant and weak below threshold; kept
  byte-identical and *supplemented* below the threshold by the new term.
- **graded airborne (UC-42)** — correct above the threshold; the dead zone below it is remedied by the
  new term, so UC-42 itself is unchanged.
- **`collision_penalty` + UC-41 curriculum** — healthy: during the fly-learning phase (0–40 %) the
  penalty is held at 2.0, so a *failed* takeoff (~−2) already beats the planted stuck-cut (−5). No
  change.
- **`obstacle_penalty`, gate/completion bonuses, anti-suicide (pre-takeoff floor-contact
  suppression), stuck / grounded early-termination** — all healthy. No change.

**Preserved invariants.** `racing_env.py`, the UC-40 hover-bias, the UC-41 collision curriculum,
`ent_coef` (0.005), and every prior reward value are untouched. The only pinned-scalar movement is a
benign side-effect of the new term's `−0.005/step` constant leak above the threshold, which shifts
some exact-value assertions in the UC-42 tests (a legitimate contract change; the structural
properties — monotone-increasing, flat-above-target, ≈0-at-floor, telescoping round-trip — all hold).

**Validation is at the reward-math level only.** pybullet is unavailable in the sandbox and the numpy
`simple` adapter is over-optimistic about takeoff (UC-40 proved this), so the committed acceptance gate
is a **deterministic reward-function test** showing sustained climb out-rewards hover-rest by a
normalization-surviving, relational margin (not an absolute one — VecNormalize makes a global rescale a
no-op). A `simple`-adapter smoke-train is a wires/finite/no-collapse check only; **behavioral takeoff
confirmation is deferred to the user's pybullet GPU retrain and is not claimed here.**

### Airborne-start reverse curriculum & climb-biased init (UC-44)
Seven consecutive reward-shaping use cases (UC-37→43) produced **zero** altitude movement: on a
565k-step run the drone sat at z ≈ 0.0135 m from episode 50 through episode 900, `ep_rew_mean` glued
to exactly **−5** (the pure time-penalty floor), success 0 %. The reward-math tests correctly verify
the climb/airborne gradient is well-formed — so the bottleneck is **not** the reward shape but that
PPO's policy never outputs sustained above-hover throttle, so the drone never enters the airborne
region every shaping term targets. A correctly-shaped gradient the policy never experiences teaches
nothing. UC-44 stops tweaking reward magnitude (the reward function is **completely untouched** — all
UC-37→43 reward invariants and doc-contract tests stay green) and attacks the **discovery** problem
directly with two composed levers that change only the spawn **state** and the policy **init**.
Temporally-correlated exploration (OU / pink noise) is deliberately deferred to a later UC so the
effect of these two levers can be attributed cleanly.

- **Lever 1 — airborne-start reverse curriculum (training-time only; `TrainConfig`
  `airborne_curriculum_enabled` default `True`).** Instead of always spawning on the floor (UC-37's
  floored start), the training envs spawn the drone at an initial altitude that is **held at
  `climb_target_height` above the floor through a warmup/start-delay
  (`airborne_curriculum_warmup_fraction`, default **0.6** — UC-51), then anneals linearly down to
  `floor_z`** across the window `[warmup, anneal]` where `airborne_curriculum_anneal_fraction` is
  **1.0** (UC-51, up from 0.5), then holds it on the floor for the remainder. With these defaults the
  spawn stays fully airborne through ~60% of training and reaches the floor only at the very end
  (100%), making floor-takeoff the last, isolated stage (see the
  [UC-51 section](#restaggered-curriculum-schedule--exposed-knobs-uc-51)). **Semantic note:** the
  airborne `warmup_fraction` is a **hold / start-delay** (the spawn stays airborne through it), NOT a
  ramp width; it must compose with the anneal as `warmup <= anneal`. Early in training the policy
  experiences the rewarded airborne region from
  step 0 and only has to learn to **maintain** altitude — far easier than discovering takeoff — and
  as the spawn anneals to the floor it must learn takeoff itself, now bootstrapped from a
  hover-competent policy. The schedule (`drone_fly.train.airborne_curriculum.spawn_z_at`) is a pure
  function of `num_timesteps`: monotone non-increasing, clamped to `[floor_z, high_z]`, returns the
  high endpoint through the warmup and exactly `floor_z` at/after the anneal end — so a **resumed**
  run continues it correctly. Setting `airborne_curriculum_warmup_fraction = 0.0` reproduces the
  pre-UC-51 single-window anneal exactly. It is pushed into the envs each rollout by an SB3 callback
  via
  `env_method("set_spawn_z", …)`, via a per-rollout SB3 callback. Set
  `airborne_curriculum_enabled = False` to train at the constant floored spawn (byte-identical to
  UC-43).
- **Training-only scope (does not leak into the takeoff measurement).** The curriculum callback is
  attached to the **training** run only, and `RaceEnv.set_spawn_z` is a per-instance override that
  defaults to `None`. **Eval and standalone-recording envs** are separate instances that never
  receive the callback, so they keep spawning on the floor (`z ≈ floor_z`) exactly as before — we
  still measure **true** takeoff. Note that early-training **in-training** recordings (the
  `RecordingCallback` on env-0 of the *training* venv) will show the **raised spawn by design** —
  that recorder is illustrative of the training rollout, not the takeoff measurement; the floored
  takeoff measurement is the **eval-time** recorder (`evaluate … --record`) and the eval episodes.
- **Lever 2 — climb-biased throttle init (fresh-build only).** UC-40 initialized a fresh policy's
  `action_net.bias[THROTTLE_INDEX]` to `HOVER_THROTTLE` (0.5 — net-zero thrust, the drone merely
  floats). UC-44 **supersedes** that with a new `CLIMB_BIAS_THROTTLE` = **0.6** (in
  `controller/encoding.py`), slightly above hover, so a fresh policy's default action produces gentle
  net-positive **lift** and collects airborne/climb reward immediately, compounding with the reverse
  curriculum. There is exactly **one** throttle-bias initializer (`_apply_climb_bias` in
  `train/loop.py`) — the hover-bias path is replaced, not duplicated. Same guards as UC-40: it fires
  **only on a fresh build** (a resumed checkpoint keeps its trained bias) and fails loud if
  `model.policy.squash_output` is ever `True` (biasing a squashed mean would not land the action on
  the climb-bias point). `HOVER_THROTTLE` (0.5) is retained as the documented hover reference for the
  adapter dynamics.

> **⚠️ Fresh-run requirement (read before evaluating this change).** A **valid** test of UC-44
> requires a **brand-new model with old checkpoints cleared** — both levers are **defeated by
> resuming from a checkpoint**: the climb-bias init applies only to fresh builds, and a resumed
> pre-fix checkpoint carries an already-collapsed policy that no spawn schedule can un-collapse. The
> prior 565k-step run was almost certainly invalidated by resuming (hover-bias only ever applied on
> fresh builds). Before the retrain, **delete the old checkpoints** (e.g. clear
> `training/<name>/checkpoints/` or point at a fresh run name) and confirm `resume` does **not** pick
> up a stale checkpoint, so the climb-bias fires and no collapsed policy is inherited. This is a
> documentation guarantee, not a code-enforceable one.

**Validation (lab-only; the ~12h GPU retrain stays the user's and is not a CI gate).** Unit tests
cover the spawn schedule (endpoints, monotonicity, clamp, out-of-range `ValueError`, degenerate
cases), the training-only scope (eval/recording spawn on the floor), the fresh-only climb-bias init
and its squash guard, and early-termination compatibility with airborne spawns (a drone spawned
airborne is not cut by the grounded/no-progress detector within its warm-up window, while a drone
that then falls to the floor and rests is still grounded-cut). No `racing_env.py` early-termination
change was needed: the existing `_took_off` latch arms from the spawn state, so an airborne spawn is
"taken off" at step 0 and the grounded detector already behaves correctly, and the stuck/no-progress
counters start at 0 (a full window must accumulate before any cut). A short CPU smoke-train
demonstrates that with an airborne spawn the drone collects airborne/climb reward, `ep_rew` rises
above the −5 floor, and the action std does not collapse. **Behavioral takeoff confirmation is the
user's fresh GPU retrain and is not claimed here.**

### Attitude-authority curriculum (UC-46)
> **Retired in UC-55.** This curriculum was a crude stand-in for the missing inner-loop rate
> controller. UC-55 adds that controller (see [Inner-loop body-rate controller
> (UC-55)](#inner-loop-body-rate-controller-acro-flight-controller-uc-55)) and **removes** the
> attitude-authority machinery entirely — the `attitude_authority_*` config/YAML keys, the
> `set_attitude_authority` env method, and the curriculum module/callback no longer exist. The
> section below is kept as historical context for the takeoff-saga.

UC-45 proved the recordings are **byte-faithful**, so the drone's failure to fly is real behaviour,
not a logging artifact: it **tumbles**. Level control is fine (throttle 1.0 + level attitude climbs
0.9→9.36 m; 0.5 hovers at 0.900 m), but the open-loop CTBR→RPM mixer (`ctbr_to_rpm`; `rate_gain=0.15`
on the pybullet backend, `BASE_MAX_BODY_RATE=4.0` on the simple backend) has **no attitude
stabilization**, and early PPO — with an unbounded Gaussian head (`squash_output=False`) and action
std ~1.0 — commands wild roll/pitch/yaw **rate** commands. The drone flips (roll reached 171° in a
reproduction), vectors its thrust sideways/down, loses net lift, and falls. This is the same root
that defeated the seven reward/curriculum use cases before it. UC-46 adds a training-time
**attitude-authority curriculum** — the direct analogue of UC-44's airborne-start reverse curriculum,
applied to attitude instead of spawn altitude. **No reward-function change and no inner-loop attitude
controller** (both deliberately deferred so the result is attributable to authority-limiting alone);
the mixer, exploration/entropy, and UC-44/UC-45 are all untouched.

- **What it scales (training-time only; `TrainConfig` `attitude_authority_curriculum_enabled` default
  `True`).** The roll/pitch/yaw command channels (action indices **1, 2, 3**) are multiplied by an
  **authority factor** in `RaceEnv.step` before they reach the adapter. **Throttle (index 0) is never
  scaled** — lift authority is untouched, so the drone can always climb. The factor **starts at
  `attitude_authority_start` (default 0.25) and anneals linearly up to `1.0` (full authority)** over
  the first `attitude_authority_anneal_fraction` (default **0.25** — UC-51, down from 0.5) of
  `total_timesteps`, then holds full authority for the remainder. This makes attitude authority reach
  full **earliest** of the three curricula (~25%), the first isolated difficulty step (see the
  [UC-51 section](#restaggered-curriculum-schedule--exposed-knobs-uc-51)). Early on a noisy policy
  physically cannot flip the drone (its
  attitude commands are capped to a quarter of their range → it stays roughly level → net thrust stays
  up → it climbs and collects the existing airborne/climb reward); as the authority anneals to full
  the policy regains full maneuvering control, bootstrapped from an upright-and-climbing policy.
- **Clip-then-scale ordering (deliberate; a structural cap, not a statistical one).** Because
  `squash_output=False`, the raw Gaussian samples reaching `RaceEnv.step` are **unbounded** — the
  largest, most flip-inducing commands are exactly the ones outside the `[-1, 1]` box. The env
  therefore **clips first, then scales**: `sanitize_action(action)` clips into the canonical CTBR box,
  then the clipped attitude channels are multiplied by the factor. This guarantees effective
  `|rpy| ≤ factor` for **all** inputs including saturated ones. Scale-then-clip would leak at full-
  sized commands and invert the lever, so the ordering is load-bearing.
- **The schedule** (`drone_fly.train.attitude_curriculum.attitude_authority_at`) is a pure function of
  `num_timesteps`: monotone non-decreasing, clamped to `[attitude_authority_start, 1.0]`, returns the
  start value at step 0 and exactly `1.0` at/after the anneal end — so a **resumed** run continues it
  correctly. It is pushed into the envs each rollout by an SB3 callback via
  `env_method("set_attitude_authority", …)`, exactly like the UC-44 airborne curriculum. It raises
  `ValueError` on a misconfigured `attitude_authority_anneal_fraction` or `attitude_authority_start`
  outside `[0, 1]` (a negative authority would invert the channels; above `1.0` would amplify past the
  sanitized envelope). Set `attitude_authority_curriculum_enabled = False` to train at constant full
  authority (byte-identical to pre-UC-46).
- **Training-only scope (does not leak into the flight measurement).** The callback is attached to the
  **training** run only, and `RaceEnv.set_attitude_authority` is a per-instance override defaulting
  **1.0** (full authority). **Eval and standalone-recording envs** are separate instances that never
  receive the callback, so they run at **full authority (1.0) = the annealed endpoint** and measure
  true flight. The `!= 1.0` guard keeps the default / eval / recording / post-anneal / disabled step
  path byte-identical to pre-UC-46.

> **⚠️ Fresh-run requirement (read before evaluating this change).** As with UC-44, a **valid**
> evaluation requires a **brand-new model with old checkpoints cleared**. The UC-44 climb-bias init
> fires **only on a fresh build**, and a resumed pre-fix checkpoint carries an already-collapsed
> policy that no curriculum — spawn or attitude — can un-collapse. Before the retrain, **delete the
> old checkpoints** (clear `training/<name>/checkpoints/` or point at a fresh run name) and confirm
> `resume` does not pick up a stale checkpoint. This is a documentation guarantee, not a
> code-enforceable one.

**Validation (lab-only; the ~12h GPU retrain stays the user's and is not a CI gate).** Hermetic unit
tests (no pybullet) cover the schedule (start value at fraction 0, exactly 1.0 at/after the anneal
end, monotonicity, clamp, out-of-range `ValueError` on both fields, degenerate/disabled → constant
1.0, resume-correctness), the env in-box behaviour (only channels 1/2/3 scaled, throttle untouched,
default 1.0 byte-identical passthrough, and a saturated-input regression lock on the clip-then-scale
ordering), and the training-only scope (the callback pushes the scheduled factor through the wrapper
stack while an eval/recording env reports authority 1.0 regardless of schedule). A short CPU/pybullet
smoke sanity-checks the start value: under reduced early authority the drone stays markedly more
upright (lower max |roll|/|pitch|) and gains altitude versus full authority with the same
policy/noise. **Behavioral takeoff confirmation is the user's fresh GPU retrain and is not claimed
here.**

### T/W-preserving pybullet dynamics randomization (UC-48)
UC-47 traced the drone's refusal to leave the ground to a **mass/thrust-scale mismatch** on the
pybullet backend (not the reward, curricula, mixer, or attitude — those were never the wall). Domain
randomization samples an **absolute** body mass (~1 kg) sized for the `SimpleDroneAdapter` point-mass
model (`BASE_MAX_THRUST = 2·m·g`, so its T/W is preserved by construction). The pybullet adapter
applied that ~1 kg to the **CF2X** body — native mass **0.027 kg**, motor thrust `KF` sized for
0.027 kg — while leaving the mixer's RPM band native. Weight ≈ 10 N against ≈ 0.6 N of max thrust
gives **thrust-to-weight ≈ 0.24**: guaranteed free-fall at any throttle, on the mastery backend the
≥80 % bar is defined against. The simple backend is unaffected (its thrust tracks the sampled mass).

- **The fix (T/W-preserving; default ON via `EnvConfig.pybullet_tw_preserving`).** On the pybullet
  backend the sampled mass is reinterpreted as a **CF2X-relative multiplier**
  `mass_ratio = sampled_mass / BASE_MASS`. The applied body mass becomes `0.027 · mass_ratio` and the
  mixer's `hover_rpm` / `max_rpm` band is scaled by `√mass_ratio`. Because peak T/W = `(max_rpm /
  hover_rpm)²` and **both** RPMs scale by the same factor, peak T/W is **invariant** — a randomized-
  heavier drone keeps the native **~2.25** peak (hover at throttle 0.5, climb at 1.0) instead of
  collapsing to 0.24. Only the mixer's mass-dependent **constants** change; its **structure**
  (`ctbr_to_rpm`), the reward, the UC-44 airborne curriculum, the UC-46 attitude-authority curriculum,
  and the climb-bias init are all untouched. Best-effort rotational inertia is scaled from a native
  baseline captured once per body so it does not desync from the applied mass.
- **Where it lives.** A pure, hermetic helper `resolve_tw_preserving_dynamics` (+ `thrust_to_weight`)
  in `pybullet_adapter.py` does the math with **no pybullet import**, so CI asserts the
  T/W-preservation property (hover ≈ 1.0, peak invariant within 1e-6, sanity band [1.5, 2.5]) without
  a simulator. The adapter applies it in `_apply_dynamics` (not `reconfigure`) so both the training env
  and the UC-47 diagnostic harness pick up the fix. Set `pybullet_tw_preserving = False` to restore the
  pre-UC-48 degenerate absolute-mass behavior (kept for opt-out and a regression bug-lock test).
  Additive and observation-schema-neutral, so it is checkpoint/obs byte-compatible.

> **⚠️ Fresh-run requirement.** Like UC-44/UC-46, observing takeoff needs a **brand-new** model with
> old checkpoints cleared: this intentionally changes the training-dynamics distribution, so
> checkpoints and recordings made on the broken scale are invalid. Real-pybullet confirmation via the
> UC-47 harness is reported in the PR and is **non-gating** (CI is hermetic — no pybullet). See
> `docs/uc48-mass-thrust-tw-preserving.md`.

### Drone-dynamics observability & T/W regression guard (UC-49)
UC-47/UC-48 fixed the mass/thrust-scale bug, but it had lived **silently** in the sampler/wiring layer
for many training UCs because nothing surfaced the drone's real thrust-to-weight. UC-49 makes that
physics **observable and regression-guarded** so it can never regress unseen. One shared, pure
computation — `drone_dynamics_summary(...) -> DroneDynamicsSummary` in
`src/drone_fly/adapter/dynamics_summary.py` — derives the summary **once** from UC-48's resolved
dynamics (`resolve_tw_preserving_dynamics` / `thrust_to_weight`), always from the **resolved
`applied_mass`** (`native · mass_ratio`), never the raw ~1 kg sampled mass that hid the bug. It reports
applied mass, weight (`mass·g`), peak T/W, hover throttle, max body-rate, and the live curriculum knob
(spawn-z; the UC-46 attitude-authority knob was removed in UC-55). It imports **no pybullet** (pure
numpy/arithmetic), so it is hermetic
and CI-gating. The same summary is surfaced in three places, and because they all read the one function
a display, a recording, and CI can never disagree about the drone's T/W:

- **Training TUI top segment.** A new `build_drone_panel` renders a full-width segment **above** the
  TIME/TRAIN/ROLLOUT panel showing Weight, T/W, hover throttle, max body-rate, and spawn-z (UC-55
  removed the attitude-authority field), refreshed per iteration. A `None` summary (before the first
  rollout) renders placeholders and never crashes, like every other panel.
- **Recording provenance.** Every recording gains an additive, presence-guarded `meta.drone_dynamics`
  block (beside the UC-45 `git_sha` / `checkpoint` / `dynamics` provenance). It is omitted entirely
  when unset, so pre-UC-49 recordings and no-randomization runs stay byte-for-byte back-compatible;
  `SCHEMA_VERSION` is unchanged.
- **CI T/W regression guard.** A hermetic test drives the **real `EnvConfig()`** default and the real
  `sample_dynamics` mass range through the summary on the pybullet path and asserts peak T/W stays
  **in-band ([1.5, 2.5])** and **mass-invariant** — gating the exact `EnvConfig → sample_dynamics →
  resolve` seam where the UC-47 bug lived. A build-time `logging.warning` (the `FLYABLE_TW_FLOOR = 1.0`
  tripwire) also fires if a drone is ever configured below hover-capable T/W.

**Curriculum-value convention.** `spawn_z` is the UC-44 curriculum knob (UC-55 removed the UC-46
`attitude_authority` knob from this summary). On the **training** env the TUI and recording show the
*live scheduled* (mid-anneal) value; at **eval/record** time the curriculum callback never runs, so it
sits at its annealed endpoint (spawn-z = floor). This is observability/guard-only — it changes no
training behavior, the UC-48 fix, the simple backend, the reward, or the UC-44 curriculum.

### Restaggered curriculum schedule & exposed knobs (UC-51)
> **Partly superseded.** Two of the three curricula this section restaggered are gone: attitude
> authority (retired in UC-55) and the collision-penalty curriculum (retired in UC-58). Only the
> **airborne-spawn** curriculum (UC-44) and the exposed `ent_coef` / PPO knobs remain; the
> collision-penalty and attitude-authority knobs no longer exist. Kept below as historical context.

The three training curricula — attitude authority (UC-46), the collision penalty (UC-41), and the
airborne spawn (UC-44) — previously all annealed on the same `anneal_fraction ≈ 0.5` schedule, so at
the 50% mark the spawn dropped to the floor, attitude jumped to full, and the collision penalty
reached 100 **simultaneously**: one synchronized difficulty cliff the airborne-phase flight skill did
not survive. UC-51 does two things.

**1. Restaggered defaults — difficulty steps are now isolated, floor-takeoff last.** The default
schedule is reshaped so the anneals are ordered and separated:

| Curriculum | Knob(s) | Old | New (UC-51) | Reaches full / floor at |
|---|---|---|---|---|
| Attitude authority | `attitude_authority_anneal_fraction` | 0.5 | **0.25** | full authority by ~25% (first) — *retired in UC-55* |
| Collision penalty | `collision_curriculum_hold_fraction` / `collision_penalty_warmup_fraction` | 0.4 / 0.5 | 0.4 / **0.1** | full penalty (100) at 0.4+0.1 = ~50% (second) |
| Airborne spawn | `airborne_curriculum_warmup_fraction` / `airborne_curriculum_anneal_fraction` | — / 0.5 | **0.6** / **1.0** | held airborne through ~60%, floor by 100% (last) |

So the ordering is **attitude-full (~0.25) < collision-full (~0.5) < spawn-reaches-floor (1.0)**, and
floor-takeoff — the hardest stage — is isolated into the final, longest stretch. Default
`total_timesteps` is bumped **1M → 2M** so that isolated tail gets real budget. (UC-55 retired the
attitude-authority curriculum in favor of the inner-loop rate controller, so only the collision and
airborne curricula remain; the ordering note is kept for historical context.)

**New airborne warmup / start-delay.** `airborne_curriculum_warmup_fraction` (new) holds the spawn
fully airborne through the first fraction of training before it begins descending. **Watch the
semantic contrast:** the airborne `warmup_fraction` is a **hold/start-delay**, whereas the collision
`warmup_fraction` is a **ramp width** — opposite meanings on purpose (each mirrors its own curriculum's
existing shape). The airborne pair must compose as `warmup <= anneal`; setting
`airborne_curriculum_warmup_fraction = 0.0` reproduces the pre-UC-51 single-window anneal exactly.

**2. Exposed knobs — schedule experiments no longer need a code change.** Every curriculum knob plus
`ent_coef` and `timesteps` is now settable in the `drone-fly train --config` YAML (previously they
existed only as `TrainConfig` dataclass defaults behind a fixed whitelist). The full set
(**UC-55 removed the three `attitude_authority_*` keys**; UC-55 also added the `rate_kp` / `rate_ki` /
`rate_kd` / `rate_max_body_rate` rate-controller keys, documented in the [UC-55
section](#inner-loop-body-rate-controller-acro-flight-controller-uc-55)):
`ent_coef`; `airborne_curriculum_enabled` / `airborne_curriculum_warmup_fraction` /
`airborne_curriculum_anneal_fraction`; `collision_curriculum_enabled` /
`collision_penalty_start` / `collision_penalty_end` / `collision_curriculum_hold_fraction` /
`collision_penalty_warmup_fraction`. See `configs/train/example.yaml` for each with its default.

Each key is **optional** and validated at config load: fractions
must be in `[0, 1]`, `ent_coef >= 0`, the collision penalties `>= 0`, `timesteps >= 1`; the airborne
`warmup <= anneal` and collision `hold + warmup <= 1` compositions are checked too (resolving an
omitted partner against the dataclass default). Out-of-range, unknown, or incoherent values fail loud
with a clear `ConfigError` (exit 2). Setting a knob to its default value is **byte-identical to
omitting it** — the exposure plumbing is lossless; the restagger changes what "default" *means*, not
the plumbing.

**Wiring note (bug fixed).** The curriculum callbacks compute their schedule window against
`TrainConfig.total_timesteps`. Previously the CLI passed the YAML `timesteps` only as the `train()`
override and never set the dataclass field, so every schedule was pinned to the dataclass default
regardless of the configured budget. UC-51 sets `TrainConfig.total_timesteps` from the YAML
`timesteps`, so the schedule now tracks the actual run length.

**Scope.** Training-orchestration + config surface only — reward, env/`racing_env.py`, adapters, the
PPO policy assembly, and termination are untouched. Whether the restaggered schedule makes the drone
take off is a behavioral question answered by a fresh GPU retrain (out of scope here); the tests are
hermetic and assert schedule *shape*, YAML round-trip, and default-parity.

### Inner-loop body-rate controller (acro flight controller) (UC-55)
The CTBR action `[throttle, roll, pitch, yaw]` carries collective thrust plus body-**rate** commands,
but `ctbr_to_rpm` mapped them to motors with a **static open-loop feedforward mix and no gyro
feedback** — a commanded rate was never regulated toward the achieved rate, so command noise dumped
raw differential torque and the body tumbled (a null command `[0.5,0,0,0]` hovers with zero drift, but
full-range noisy rate commands flip the drone 180° within ~1 s). This was the structural root cause
behind ~15 no-takeoff use cases — "acro without a flight controller". UC-55 adds the missing
**inner-loop rate PID** and retires the UC-46 attitude-authority curriculum that stood in for it.

- **New controller module** (`src/drone_fly/adapter/rate_controller.py`, hermetic — no pybullet).
  `RateController.update(setpoint_rpy, measured_rate_rpy, dt)` runs a per-axis PID on
  `error = setpoint − measured` (integrator anti-windup-clamped to `±integral_limit`, output clipped to
  `[-1, 1]`) and returns a normalized **effort**. The pybullet adapter reads the achieved body rate
  (`raw[13:16]`, body-frame) **before** stepping, maps the normalized command to a body-rate setpoint
  via a swappable curve hook clamped to `max_body_rate`, runs the PID, and feeds the effort into the
  **same** quad-X mixer the open-loop command used. **Throttle (index 0) passes through untouched.**
- **Acro, not auto-level.** The loop holds a *rate*, never an *angle* — the policy keeps full acro
  agency (flips, inverted flight, arbitrary attitudes). It only damps the plant against command noise,
  mirroring the haltere rate reflex beneath the descending connectome commands.
- **Feedback polarity is inherited verbatim.** The effort→differential-RPM sign mapping comes straight
  from the existing mixer convention (no new sign assumption): the loop is correct iff the pre-existing
  open-loop convention was correct. The hermetic plant used in tests is self-consistent with the
  controller sign, so it validates the math but **not** the real per-axis pybullet polarity or
  closed-loop stability — those are part of the owner's fresh GPU/sandbox verdict, not the CI suite.
- **Config-overridable gains (AC8).** PID gains have documented defaults on
  `RateControllerConfig` (`kp`/`ki`/`kd`, `max_body_rate` single-sourced from `BASE_MAX_BODY_RATE`,
  `integral_limit`, and the `command_to_setpoint` curve hook — `linear_rate_curve` by default). They
  are overridable from the `drone-fly train`/`evaluate` YAML via `rate_kp` / `rate_ki` / `rate_kd`
  (+ optional `rate_max_body_rate`), threaded into `EnvConfig.rate_controller`. Each is optional,
  non-negative-validated, and **setting a key to its default is byte-identical to omitting it**.
- **Parameterizable curve hook (AC10).** The command→setpoint mapping is linear by default but routed
  through `RateControllerConfig.command_to_setpoint`, so a later nonlinear (Betaflight/Liftoff-style)
  rates curve can replace it **without touching the policy**.
- **pybullet-only; CI stays hermetic (AC9).** The rate loop is threaded into the pybullet backend
  alone (mirroring `pybullet_tw_preserving`); `SimpleDroneAdapter` is left byte-identical. Byte-identity
  holds for a pure-throttle command when the measured rate is zero (`error = 0 → effort = 0 →`
  base-only RPM == the pre-UC-55 `ctbr_to_rpm([throttle,0,0,0])`).
- **Curriculum retired.** The `a[1:4] *= attitude_authority` scaling in `racing_env.py`, its
  `set_attitude_authority` / `attitude_authority` accessors, the `attitude_curriculum.py` module and
  its training wiring, the `attitude_authority_*` `TrainConfig`/YAML keys, and the
  `attitude_authority` field on the drone-dynamics summary/TUI/recording are all removed (no code path
  scales the rate channels by an authority factor any more).

> **⚠️ Fresh-run requirement.** This changes the control problem: a policy trained on the unstabilized
> plant will not transfer to the stabilized one, so a valid verdict needs a **brand-new** model with
> old checkpoints cleared. All new tests are hermetic (no GPU, no training); the behavioral flight
> verdict is the owner's GPU retrain and is not a CI gate.

### Meteor75 Pro nominal & wide domain randomization (UC-56)
The nominal plant was the pybullet **CF2X** reference (~0.027 kg, peak T/W ≈ 2.25). UC-56 retargets it
to a **BetaFPV Meteor75 Pro analog** (75 mm 1S brushless whoop) so acro training (UC-55) runs on a real,
ownable drone, and **widens the domain-randomization envelope to span whoop → 5" racer** so the policy
is robust across a broad range and can later be fine-tuned to other drones. Source:
[BetaFPV Meteor75 Pro product page](https://betafpv.com/products/meteor75-pro-brushless-whoop-quadcopter)
(accessed 2026-09-23), assuming a 1S pack + 1102-class motors + 40 mm props. See
`docs/uc56-meteor75-dynamics.md` for full provenance and derivations.

- **Nominal (AC1/AC2)** — mass **0.032 kg** (30–36 g AUW band), peak **T/W 2.5**, wheelbase **75 mm**
  → arm coordinate `a = wheelbase/(2·√2) ≈ 26.5 mm`, prop **40 mm** (spec-only). Frozen in
  `src/drone_fly/adapter/meteor75.py` with a cited-source docstring. It is a reparameterization over the
  CF2X URDF (KF unchanged, no URDF swap): mass / hover-at-throttle-0.5 / peak T/W are **exact**, only the
  absolute RPM is a non-physical "analog". `PyBulletAdapter._build_env` bakes it onto the fresh body and
  does **not** touch `_pending_dynamics`, so a randomization-off run flies the exact nominal.
- **Inertia (AC3)** — a motor-position point-mass model (`meteor75.motor_position_inertia`): four motor
  masses at the quad-X arm positions `(±a, ±a, 0)` + a central point mass →
  `Ixx = Iyy = motor_fraction·M·a²`, `Izz = 2·motor_fraction·M·a²` (`motor_fraction = 0.5`, documented).
  It scales with mass **and** arm length, so it tracks the whole envelope; a hermetic test pins a known
  geometry → known diagonal.
- **Wide envelope (AC4/AC5), reusing UC-48** — three **pybullet-only** axes sampled independently per
  episode (draws appended **last** → seeded streams byte-identical): mass ratio `(1.0, 20.0)` × nominal
  (~32 g → ~640 g), peak T/W `(2.5, 10.0)`, arm `(0.0265, 0.078)` m. The resolver
  `resolve_tw_preserving_dynamics` gains an optional `target_tw`: given, `max_rpm = hover_rpm·√target_tw`
  so peak **T/W = target_tw exactly, independent of mass** (hover stays at throttle 0.5); `None` keeps
  the pre-UC-56 CF2X path so `test_uc48` stays green. The **simple/CI backend ignores these axes** — its
  ranges are untouched and it stays byte-identical / free-fall-proof.
- **YAML knobs (UC-51 pattern)** — `pybullet_mass_ratio_min/max`, `pybullet_tw_min/max`,
  `pybullet_arm_length_min/max` on the train **and** evaluate config; omit (or `null`) leaves the
  `RandomizationConfig` default for that side. Active only with `randomize_dynamics: true`. Validated at
  load: each bound `> 0`, `min <= max`, and T/W `min >= 1` (below-1 peak T/W cannot hover).
- **Observability (AC7)** — the shared `drone_dynamics_summary` resolves off the Meteor75 nominal and
  reports `arm_length`; the TUI, recording, and the UC-49 end-to-end T/W regression guard all report the
  identical retuned plant (nominal peak T/W 2.5).
- **Not modelled** — per-scale aerodynamic drag and motor-response lag across the ~20× mass range (a
  single linear-damping range is reused). With KF fixed, the UC-55 rate loop's differential authority
  varies across the envelope; retuning its gains is deferred to UC-55 (its `rate_*` knobs already allow
  it).

> **⚠️ Fresh-run requirement.** The plant changed (new nominal + a much wider envelope), so old CF2X-era
> checkpoints/recordings are invalid — observing behavior needs a **brand-new** GPU retrain. All UC-56
> tests are hermetic (spec-match, inertia-from-geometry, envelope bounds, T/W invariance, nominal
> determinism); the behavioral verdict is the owner's retrain and is not a CI gate. A whoop → 5"
> envelope is very wide — a curriculum over it (or start-narrow-then-widen via the YAML knobs) is a
> training-strategy choice, out of scope here.

### Raised & decoupled control frequency (UC-57)
The policy control loop ran at **20 Hz** (`dt = 0.05`), which is low for agile acro flight and for a
stable inner rate loop (UC-55). UC-57 raises the **run-layer default to 50 Hz** and **decouples** the
three rates in play — the expensive ~25k-neuron policy decides at `control_hz`, while the UC-55 rate
PID **and** physics run at `control_hz × physics_ratio` (matching real drones, where the FC loop rate
exceeds the RC/setpoint rate, and biology, where descending commands are slower than the haltere
reflex). The task is held **invariant in real-time terms**: episode duration in *seconds* is
preserved and command latency is expressed in *milliseconds*, so changing Hz never silently alters
the task or the latency semantics.

- **Byte-identity split (the load-bearing design).** All rate-dependent quantities derive from a
  single `BASELINE_DT = 0.05` in the new pure module **`drone_fly.env.timing`** (dt↔Hz,
  `scale_step_budget`, `resolve_latency_steps`, `inner_rate/inner_dt/iteration_count`,
  `pybullet_freqs`). The **dataclass defaults stay 20 Hz** (`EpisodeConfig.dt = BASELINE_DT`,
  `physics_ratio = 1`, `command_latency_ms = 0.0`) so `EnvConfig()` and the ~31-file byte-identity
  test culture are untouched; the **50 Hz default lives only at the run/CLI layer** (`control_hz=50`,
  `physics_ratio=10`, `command_latency_ms=0.0`, applied identically to **train and evaluate** so a
  50 Hz-trained policy is never evaluated on a 20 Hz plant). At `dt=0.05, ratio=1` every helper
  reduces to today's values exactly.
- **Constant episode seconds (AC3).** `RaceEnv` composes the whole baseline step budget
  (`max_steps + steps_per_gate·(N−1) + pad allowances`) then scales it **once** via
  `scale_step_budget` (`round(steps · BASELINE_DT/dt)`): identity at 20 Hz, ×2.5 at 50 Hz
  (400 → 1000), so `steps · dt` — the wall-clock seconds — is invariant.
- **Hz-invariant latency (AC4).** Command latency is specified in **ms** and converted to whole steps
  at the active rate in one round (`resolve_latency_steps`), summing the standing `command_latency_ms`
  with any domain-randomized latency (drawn in baseline steps). A 100 ms command reproduces the
  historical **2-step** latency at 20 Hz and **5 steps** at 50 Hz. The default stays **0.0 ms** (no
  forced 100 ms default — AC4 is a conversion *property*, not a runtime default). The env owns the
  conversion and hands both backends the resolved integer; a minimal FIFO (present on both the simple
  and pybullet backends) buffers the CTBR action upstream of the rate loop.
- **Decoupled inner loop (AC2, pybullet).** `PyBulletAdapter.step` runs `physics_ratio` inner ticks
  per policy step (read gyro → PID at `inner_dt = dt/physics_ratio` → mix → one physics tick), with
  collision OR-latched across ticks and returning the first collided tick's state. `pybullet_freqs`
  sizes the CtrlAviary so `pyb_freq` is always an integer multiple of `ctrl_freq` (fixes a latent
  `max(ctrl_freq·4, 240)` divisibility bug) while staying ≥ 240 and ≥ 4×inner. The **simple/CI backend
  integrates at the policy `dt`** (physics rate == policy rate) — a deliberate hermeticity scope call,
  consistent with the rate loop being pybullet-only.
- **Reward correctness (AC5).** The two per-step, time-extensive reward terms (`time_penalty`,
  graded `airborne_bonus`) are scaled by `dt/BASELINE_DT` so their per-episode integral is
  rate-invariant (see the reward-table note above); the potential-based shapers are already
  rate-correct. **γ coupling:** `gamma` is a per-step discount, so its real-time horizon shrinks as
  the rate rises — raise `gamma` toward 1 to keep the same horizon, and keep `RewardConfig.climb_gamma`
  **equal to** the training `gamma` (the shapers telescope only when they agree).
- **YAML knobs (UC-51 pattern).** `control_hz`, `physics_ratio`, `command_latency_ms` on the train
  **and** evaluate config (defaults 50 / 10 / 0). Validated at load: `control_hz > 0`,
  `physics_ratio` int ≥ 1, `command_latency_ms ≥ 0`.
- **Budget guidance.** `total_timesteps` counts env steps; at 50 Hz an episode spans ~2.5× more steps
  per sim-second, so a fixed budget covers ~2.5× less simulated flight time — scale `total_timesteps`
  up with `control_hz` to keep the same sim-time budget (e.g. 2M @20 Hz ≈ 5M @50 Hz).

> **⚠️ Fresh-run requirement.** The default plant rate changed (20 → 50 Hz, with a decoupled inner
> loop), so old 20 Hz checkpoints are not directly comparable — the behavioral verdict needs a
> **brand-new** GPU retrain. All UC-57 tests are hermetic (rate plumbing, seconds-invariance,
> latency-ms→steps conversion, decoupling ratio, `pyb_freq` multiples, ratio-1 byte-identity); the
> behavioral verdict is deferred to the owner's retrain and is not a CI gate.

### Takeoff-oriented reward redesign (UC-58)
With the plant finally flight-capable (UC-55 inner-loop rate controller + UC-56 Meteor75 nominal at
T/W 2.5), a fresh UC-55+UC-56 training run exposed that the reward — accreted across ~15 use cases
while the drone *could not* fly — now **incentivizes early termination rather than flight**. Recording
analysis (episodes 0–2450) showed cumulative reward *rising* as episode length *fell* (Pearson
**−0.87** between total reward and frame count — the two shortest 12-frame episodes scored best),
commanded throttle *declining* over training (mean 0.53 → 0.30), every airborne-start episode
descending to the floor (z-end ≈ 0.01 m from a 1.0 m start), and gate success flat at 0 %. The policy
was optimizing to **end episodes as fast as possible** — hitting the ground stops the per-step
`time_penalty` bleed — because it could earn no positive reward (no gate reachable, no effective climb
incentive).

**Root cause (three compounding factors in the pre-UC-58 default reward):**
1. **`time_penalty` (0.05/step)** — an unconditional per-step existence cost that made existence
   net-negative below a useful altitude, so diving to the floor to stop the bleed was optimal.
2. **The UC-50 progress hard-gate was live by default** — it withheld the positive progress reward
   whenever the drone was below a gate-referenced altitude band, penalizing a low floor-start policy
   even as it moved toward a gate.
3. **No sustained positive altitude term** — `airborne_bonus` was dead-zoned below the airborne
   threshold and net-negative low; the three potential-based terms (climb / ground-break /
   altitude-hold) telescope to ≈0 net by construction, giving only transient nudges clawed back on
   descent.

**The redesign (from scratch, one coherent reward).** The four overlapping altitude terms are replaced
by ONE **sustained, level-based, saturating altitude reward** paid from `h=0`:

```
altitude_reward(h) = altitude_weight · clamp(h, 0, altitude_target) / altitude_target   # × per_step_scale
```

with `altitude_weight = 0.2`, `altitude_target = 1.0 m`. The full step reward is now:

```
reward = progress + gate_bonus/N + completion_bonus + altitude_reward
         − collision_penalty(genuine crash) − obstacle_penalty
```

Why it satisfies the acceptance criteria:
- **Anti-suicide (AC1/AC2).** It is *level-based* (a function of the current height, not a telescoping
  potential), so it is a genuinely SUSTAINED positive signal for staying up. With `time_penalty` gone
  there is nothing to be "saved" by ending the episode, so a descent / early termination scores
  strictly below holding-or-climbing over the same-or-longer horizon — the −0.87 correlation is
  removed/inverted.
- **Takeoff gradient from the floor (AC3).** It is paid from `h=0` and is strictly increasing on
  `[0, altitude_target]`, so leaving the ground earns reward immediately — no dead zone (this is what
  the retired UC-43 ground-break potential was bolted on to patch; it is now structural).
- **Non-farmable (AC4).** It saturates flat at the target and is bounded by `altitude_weight` per step,
  so hovering/oscillating can never exceed genuine climb-to-target.
- **Hierarchy preserved (AC5).** The per-episode altitude ceiling over the default 800-step budget is
  `0.2 × 800 = 160 < completion_bonus 200`, so completing the course strictly dominates loitering.

**Retired (each removed, no dead/duplicate reward paths remain):** `time_penalty`; `airborne_bonus` +
its boolean airborne gate; the UC-39 `climb_weight` / `climb_target_height` / `climb_gamma` potential;
the UC-43 `ground_break_weight` / `ground_break_height` potential; the whole UC-50
`enable_altitude_decoupling` / `altitude_hold_weight` / `altitude_band` group (including its progress
hard-gate); and the UC-39/41 training-time collision-penalty curriculum (`collision_penalty_start` /
`_end` / `_hold_fraction` / `_warmup_fraction` / `_enabled` `TrainConfig` fields, the
`set_collision_penalty` env method, and the `drone_fly.train.collision_curriculum` module). The
fragile `climb_gamma == training γ` coupling is eliminated entirely — no potential-based shaping
remains.

**Grounded-rest is now penalty-free (the load-bearing subtlety).** `penalize_collision` changed from
`crash or (early_termination == "grounded")` to just `crash`. A gentle grounded-rest after a failed
takeoff no longer pays −`collision_penalty`; otherwise a failed takeoff attempt would cost −100 versus
a penalty-free do-nothing floor — exactly the early-termination trap the (now-retired) crash-cliff
curriculum papered over. The UC-25 grounded/stuck early-termination *detectors* are **kept** (they
still BOUND the episode, just penalty-free), and the UC-44/51 airborne-start spawn curriculum is
**kept** and reward-decoupled (its high spawn endpoint now derives geometrically from
`altitude_target`).

**Configurable + default-on.** `altitude_weight` / `altitude_target` are exposed and validated in the
train and evaluate YAML (weight ≥ 0, target > 0), folded into the run's `RewardConfig` by
`_apply_reward` exactly like the UC-57 control-rate knobs; the redesigned reward ships on for every
fresh run. **Validation is hermetic (AC10):** the reward is deterministic and unit-testable on
hand-built transitions; the **behavioral takeoff verdict is deferred to the owner's fresh GPU
retrain** — old checkpoints do not transfer (the reward changed for every run).

### Visualization & recording
Enable recording in a train/evaluate config with `record: true` (tune cadence via `record_every`);
frames land in that run's `training/<name>/recordings/`. Open `viz/viewer.html` in a browser
(dependency-free, `file://`-safe) and load a recording — no server or build step.

**Three-zone layout (UC-59, refined UC-60).** The top row holds two side-by-side panels — the
**anatomical brain map** (left, given the majority of the row) and the **flight actions** trace panel
(right) — and the **3D flight view** spans the full content width in a panel below them, as the
primary focus. Since UC-60 the two top panels are rendered at **equal height** (each stretches to the
taller of the two, via `align-items: stretch`), and the top row collapses to a single column at narrow
viewport widths. The brain map is an MRI/fMRI-style activation heatmap over a static,
spatially-registered MaleCNS brain outline (`viz/brain_outline.js`, `top-down` / `front` / `side`
presets + a per-frame ↔ global intensity toggle). The 3D flight view is driven by the recorded
`meta.course` geometry — controls: **drag** to **rotate**, mouse **wheel** to **zoom**, the same view
presets, and a `0.25`× slow-inspection speed. Regenerate the outline with
`uv run python scripts/build_brain_outline.py`.

Since UC-60 the brain canvas is **sized tight to the registered outline's projected shape** — its
aspect ratio is set per view from the outline's bounding box (the fly brain is much wider than tall,
so the image is **short**, not a forced square) — and the fixed voxel→canvas transform is sized to the
**outline bounding box alone** (no longer unioned with distant out-of-outline neurons), so the brain
fills its canvas without stretching. The aspect is written only when it changes, so the DPR-aware
backing store never feeds back into layout. The canvases are DPR-aware and resize with the layout (the
brain box is width-driven with a dynamic outline-derived aspect, and the boxes strip a fixed height,
so resizing never oscillates). To visually verify the layout
against an actual render, use the dev-only headless-browser screenshot helper (not shipped in the
viewer, not a runtime dependency, never imported by the test suite):

```
npm i -D playwright && npx playwright install chromium   # one-time, local dev only
node scripts/screenshot_viewer.mjs [out.png]             # renders viz/viewer.html with a synthetic recording
```

It renders the viewer with a small inline synthetic recording and saves a PNG. With no headless
Chromium available it prints a clear message and exits non-zero rather than pretending to succeed, so
CI stays hermetic — run it locally where Chromium is installed to produce the image.

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
rendered extent.

**Out-of-boundary neurons live in tagged boxes (UC-59, generalised UC-60).** The brain map itself
shows **only neurons inside the registered brain outline** (maximized to fill the panel). Since UC-60
the relocation rule is **geometric, not tag-based**: a neuron is relocated iff its projected position
falls **outside the brain outline polygon for the active view** — recomputed whenever you switch the
`top-down` / `front` / `side` preset, so the partition always matches what is drawn. (This replaces
UC-59's `placement="schematic"` rule; the point-in-polygon test runs in voxel space, independent of
the screen transform.) Relocated neurons move into a strip of **outlined, top-titled boxes** below the
map, grouped by their `meta.modality` tag: **vision (external)**, **proprioceptive**, **hunger**, and a
catch-all **other (untagged)** box that absorbs any afferent with no known modality tag (empty string,
unknown, or legacy). Every relocated neuron lands in exactly one box and none are dropped; empty boxes
are omitted and a recording with no out-of-boundary neurons shows a short muted note. The boxes sit
compactly just beneath the now-shorter brain. Each box is a mini heatmap that **animates its members'
live activation** over the timeline (play / scrub / speed), preserving the signal the old splat
carried, just reorganized. The partition is a single pure, DOM-free function (`partitionByBoundary`,
built on `pointInPolygon` / `pointInBBox`) shared by the box builder and the brain-map stamp; it is
computed once per `(view, recording)` and cached, never per animation frame. A view whose outline has
no registered polygon falls back to the outline bounding box, and a non-anatomical / no-outline
recording disables the test entirely (all neurons stay in-canvas, no boxes) — both documented degrades.
The descriptive legend text that previously sat under the brain map was removed in UC-60 to declutter
the shortened panel.

**Modality toggle (UC-28).** The brain-map panel has a **modality** selector that overlays rings
on the UC-13 modality populations — `vision`, `proprioceptive`, and `hunger` — on top of the hot
activation colormap (it does not replace the colours). The rings apply to the in-boundary neurons
shown on the brain map; the modality of relocated (out-of-boundary) neurons is conveyed by their
tagged boxes (above) instead. Toggling the modality selector does not re-partition the map (membership
is position-based) — it only draws or hides the overlay rings. The populations are the real biological labels tagged per-neuron
in `meta.modality` at record time (fail-soft: a modality absent from the slice is simply not tagged).
`damage`/nociception is **unavailable** — MaleCNS ships no nociceptive label and there is no modality
rule for it — so it is documented as absent rather than faked.

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
