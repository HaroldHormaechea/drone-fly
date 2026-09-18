# drone-fly

Can the wiring diagram of a real fruit-fly brain (the MaleCNS connectome) seed a neural
controller that reinforcement-learns to fly a quadrotor through a timed race course in
simulation?

## Quick start: train a fly to fly

Terse path from clone to a trained policy. Each step links to its full write-up below.

1. **Install** — `git clone <repo> && cd drone-fly && uv sync --extra dev`. See [Requirements](#requirements).
2. **Provision a connectome** — point at the committed fixture (`tests/fixtures`, works offline), or download the full MaleCNS matrix into `data/connectome`. See [Provisioning connectome data](#provisioning-connectome-data-required-before-training).
3. *(Optional)* **Prune once, reuse** — `uv run drone-fly prune --config configs/prune/k2.yaml` writes a reusable sensory→motor slice. See [Subcircuit pruning](#subcircuit-pruning-uc-04).
4. **Train** — `uv run drone-fly train --config configs/train/example.yaml` (edit the YAML's `name`, `connectome`, and `timesteps`). All settings — including the run name and connectome — live in the config; see [Run configuration](#run-configuration---config). Outputs land under `training/<name>/`.
5. **Resume** — set `resume: auto` (or `resume: latest`) in the train config; re-running then continues from the newest checkpoint under `training/<name>/checkpoints/` (see [Resume](#run-configuration---config)).
6. **Evaluate** — `uv run drone-fly evaluate --config configs/evaluate/example.yaml` (point its `checkpoint` at `training/<name>/checkpoints/ppo_racer_final.zip`).

> Hermetic sanity check (no network, seconds): `uv run drone-fly smoke-train --connectome tests/fixtures`. `smoke-train` keeps its small flag surface; the four settings-heavy commands (`train` / `evaluate` / `prune` / `prune-trained`) are config-driven.

## What it does

drone-fly is a research prototype that connects three things:

- **Connectome load** — reads MaleCNS connectivity (neurons and typed synaptic connections) from
  a cached, offline connectivity matrix on disk (the `connectome_data_prep` `.npz` + meta format).
  The offline load path needs no neuPrint token or network access; fetching a fresh matrix from
  neuPrint is a separate, later concern.
- **Connectome-seeded controller** — uses that connectivity to shape a trainable PyTorch policy
  network whose parameters are the connectome edges (sparse, connectome-derived — not a dense
  MLP). The wiring is seeded from the fly; the connection strengths are learned by RL. This is not
  a biophysically faithful brain simulation — the data does not provide neuron dynamics. A stub
  sensory-encoder / motor-decoder locks the I/O contract: a dummy observation in, a finite `(4,)`
  throttle/roll/pitch/yaw action out.
- **Reinforcement learning** — trains the controller (Stable-Baselines3 PPO) in an open
  Gymnasium quadrotor simulator to chase waypoints through gates, penalized for hitting the
  floor, ceiling, or obstacles and rewarded for the fastest gate-to-gate times.

drone-fly does **not** build a full biophysical brain simulator, and — since no installable
library actually does it (see the AxonWeave note below) — it also doesn't depend on a third-party
connectome runtime. Instead it uses a **small in-repo substrate** (`SparseConnectomeLayer`): the
real MaleCNS synapses as sparse *trainable* weights over a fixed sparsity pattern. The project's
real work is that substrate plus the *integration* (wiring it to an open drone sim) and the
*training* (RL for waypoint racing).

This repository is an early prototype. The **connectome plumbing** (UC-01), the **hardened
trainable substrate** (UC-02), and now the **start→gate→finish flight task** (UC-03 — racing
environment, reward, PPO training with checkpoint/resume, evaluation, and a one-command
bootstrap) are implemented and tested. See **[Flight training (UC-03)](#flight-training-uc-03)**
below.

> **AxonWeave note.** AxonWeave is **not a usable dependency** and is **not** what this project
> runs on. A source-verified check found it absent from PyPI and TestPyPI (HTTP 404 under every name
> variant), its GitHub repo (`dhakalnirajan/axonweave`) has zero releases/tags and a Rust-PyO3
> build, and its own README calls it "a production-oriented foundation, not a completed simulator."
> The connectome substrate is therefore the **in-repo `SparseConnectomeLayer`** — the real MaleCNS
> synapses as a sparse trainable PyTorch parameter over a fixed sparsity pattern. AxonWeave may be
> revisited only if it ever ships a real tagged wheel.

## Prior art / references

drone-fly reuses and follows existing open-source work rather than reinventing it:

- [AxonWeave](https://github.com/dhakalnirajan/axonweave) — an **unreleased/unusable** project (no
  PyPI wheel, zero releases/tags, README self-describes as an incomplete foundation). **Not used**;
  the substrate is the in-repo `SparseConnectomeLayer`. Listed only for provenance.
- [doomfly](https://github.com/nftechie/doomfly) — MaleCNS → ViZDoom with dopamine-cell
  reinforcement (interface template).
- [fly-craftax](https://github.com/liuzihe02/fly-craftax) — connectome + PPO (the
  substrate-plus-PPO shape this project needs).
- [flybody](https://github.com/TuragaLab/flybody) — a MuJoCo fly body with flight RL
  environments.
- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) — the open quadrotor
  simulator used behind a sim-agnostic control adapter.

See `PROJECT_BRIEF.md` (Technologies and Architecture) for the full list and the reuse rationale.

## Requirements

- Python 3.11 for the core project + CI (lint, tests, `smoke-train`)
- Python **≥ 3.12** for the optional PyBullet simulator (mastery training) — see the
  [Simulator bootstrap boundary](#simulator-bootstrap-tested-vs-untested-boundary)
- [uv](https://docs.astral.sh/uv/) for dependency management
- A free neuPrint auth token (from https://neuprint.janelia.org) to fetch the connectome
- `gym-pybullet-drones` + `pybullet` are installed from GitHub/optional extra, not required by
  CI (see below); a C/C++ toolchain is needed to build pybullet from source
- A GPU is optional but speeds up training; the prototype is CPU-runnable

## Quick start

```sh
git clone git@github.com:HaroldHormaechea/drone-fly.git
cd drone-fly

# Create the environment and install the project + dev tools.
uv sync --extra dev

# Run the checks that exist today (offline; the committed test fixture is used).
uv run ruff check .
uv run pytest

# Prove the flight loop wires together end-to-end, hermetically (no pybullet):
uv run drone-fly smoke-train --connectome tests/fixtures
```

For real (mastery) training you need the PyBullet simulator — install it via the one-command
bootstrap `./scripts/train.sh`, which pins an exact `gym-pybullet-drones` commit. See
[Flight training (UC-03)](#flight-training-uc-03).

The UC-01 tests run fully offline against a small, committed real-MaleCNS subgraph under
`tests/fixtures/`. A neuPrint token is only needed later, for fetching a fresh connectome — set
it via `cp .env.example .env` then edit `NEUPRINT_TOKEN` when that path is implemented.

### Provisioning connectome data (required before training)

Every run needs a connectome — for `train` / `evaluate` / `prune` / `prune-trained` this is the
config's `connectome:` key; for `smoke-train` it is the `--connectome` flag. It points at a
directory that holds **one** SciPy-sparse `.npz` connectivity matrix plus a sidecar CSV.
**Important naming rule:**
the loader looks for the CSV named after the matrix stem — a matrix `foo.npz` must sit next to
`foo_meta.csv` in the same directory. If the names don't match, the loader can't find the metadata.

You have three options, easiest first:

**Option A — use the committed fixture (zero download, recommended to start).** A real ~300-neuron
MaleCNS slice already ships in the repo under `tests/fixtures/` (`mcns_fixture.npz` +
`mcns_fixture_meta.csv`). It's small but real, and training works against it immediately:

```sh
# configs/train/example.yaml already sets `connectome: tests/fixtures` — runs as-is.
uv run drone-fly train --config configs/train/example.yaml
```

**Option B — download a fuller MaleCNS matrix.** Pull the canonical whole-brain MaleCNS
connectivity from the `connectome_data_prep` dataset into a folder, **renaming the meta CSV so its
stem matches the `.npz`**:

```sh
mkdir -p data/connectome
BASE=https://raw.githubusercontent.com/YijieYin/connectome_data_prep/main/data/maleCNS
curl -L "$BASE/mcns_inprop_all_neuron.npz" -o data/connectome/mcns_inprop_all_neuron.npz
# note the rename: the source is mcns_all_neuron_meta.csv, saved as <npz-stem>_meta.csv
curl -L "$BASE/mcns_all_neuron_meta.csv"   -o data/connectome/mcns_inprop_all_neuron_meta.csv

# In a train config, set `connectome: data/connectome` (and a real `timesteps`), then:
uv run drone-fly train --config configs/train/example.yaml
```

`data/connectome/` is the default location (`DRONE_FLY_CONNECTOME_DIR`, else `data/connectome`), so
once those two files are there you may omit the config's `connectome:` key entirely. `data/` contents are
gitignored — the download stays local. (The dataset is CC-BY; cite MaleCNS / `connectome_data_prep`.)

**Option C — regenerate the committed slice.** `uv run python scripts/build_test_fixture.py`
downloads the same source and writes a deterministic slice to `tests/fixtures/` (dev-time only,
needs network; provenance recorded in `tests/fixtures/FIXTURE_PROVENANCE.md`).

## Run configuration (`--config`)

The four settings-heavy commands take **all** of their settings from a single YAML file passed
with `--config <path>`; there are no per-setting flags on them any more (UC-11):

```sh
uv run drone-fly train         --config configs/train/example.yaml
uv run drone-fly evaluate      --config configs/evaluate/example.yaml
uv run drone-fly prune         --config configs/prune/k2.yaml
uv run drone-fly prune-trained --config configs/prune-trained/example.yaml
```

`smoke-train` (a CI/correctness helper) and `fetch-connectome` (a stub) keep their small flag
surface and are **not** config-driven.

**Shipped example configs** (under `configs/`, committed so the documented runs work as-is):

| File | Command | Runs as-is? |
|---|---|---|
| `configs/train/example.yaml` | `train` | ✅ smoke-sized run against `tests/fixtures` on the numpy backend |
| `configs/prune/k0.yaml`, `k1.yaml`, `k2.yaml` | `prune` | ✅ the k0 / k1 / k2 slices of `tests/fixtures` → `artifacts/pruned/` |
| `configs/evaluate/example.yaml` | `evaluate` | needs a real `checkpoint` (edit the path first) |
| `configs/prune-trained/example.yaml` | `prune-trained` | needs a real `checkpoint` (edit the path first) |

### Per-run output layout (`training/<name>/`)

A `train` config carries a **required `name`**, and all of that run's outputs go under
`training/<name>/`, split so differently-named runs never collide:

| Path | Contents |
|---|---|
| `training/<name>/checkpoints/` | step `ppo_racer_<steps>_steps.zip` + matching `..._vecnormalize_<steps>_steps.pkl`, `ppo_racer_final.zip`, and the canonical `vecnormalize.pkl`. **This is your trained agent.** |
| `training/<name>/logs/` | the learning curve: `progress.csv` + TensorBoard `events.out.tfevents.*` |
| `training/<name>/recordings/` | UC-05 activation playback files (when `record: true`) |

`name` must be non-empty and use only `[A-Za-z0-9._-]` (no path separators, no `.`/`..`) — there
is **no** flat `artifacts/models` fallback. `training/**` is gitignored. For `evaluate` /
`prune-trained`, `name` is optional; when set together with `record: true` (and no explicit
`record_dir`), recordings are routed to `training/<name>/recordings/` too.

> **Cleaning up runs (`clean`, UC-10).** To start from scratch, wipe the training *outputs*
> with `uv run drone-fly clean`. It is a **dry-run by default** — it lists what would be
> removed and deletes nothing; pass `--yes` (or `--force`) to actually delete. It removes the
> children of `artifacts/models/`, `artifacts/logs/`, `artifacts/activations/`, and every
> `training/<name>/` run folder (the directories themselves and any `.gitkeep` are kept), and
> never touches source, use cases, the brief, or the connectome under `data/`. Add
> `--include-prunes` to also clear the prepared prune slices (`pruned*` under `artifacts/` and
> `data/`). It operates relative to the current directory, so run it from the project root.

### Config schemas

Every key maps 1:1 to a former flag, and an **omitted key falls back to that flag's existing
default** — so a config reproduces exactly the run the old flags would have. Unknown keys,
missing required keys, wrong types, and malformed YAML all fail with a clear one-line error and
exit code `2` (never a stack trace).

**`train`** — `name` **(required)**; optional: `connectome`, `adapter` (`auto`|`simple`|`pybullet`,
default `auto`), `device` (`cpu`|`cuda`|`mps`), `timesteps`, `n_envs` (≥ 1), `resume`, `prune`,
`prune_k`, `record`, `record_every`, `record_dir`, `randomize`, `randomize_dynamics`.

**`evaluate`** — `checkpoint` **(required)**; optional: `vecnormalize`, `episodes`, `seed`
(default 0), `device`, `adapter` (default `auto`), `connectome`, `record`, `record_every`,
`record_dir`, `randomize`, `randomize_dynamics`, `name`, `prune`, `prune_k`.

**`prune`** — `connectome` **(required)**, `out` **(required)**; optional: `prune_k`, `prune_rule`.

**`prune-trained`** — `checkpoint` **(required)**, `out` **(required)**; optional: `connectome`,
`vecnormalize`, `prune`, `prune_k`, `metric` (`mean_abs`|`active_fraction`), `threshold`,
`threshold_mode` (`absolute`|`percentile`), `episodes`, `eps`, `finetune_steps`, `adapter`
(**default `simple`**, not `auto`), `device`, `seed`, `randomize`, `randomize_dynamics`, `name`.

### Resume (config-driven)

The train config's `resume:` key replaces the old `--resume` flag:

| `resume:` value | Behaviour |
|---|---|
| omitted / `null` | fresh run |
| `auto` | continue from the newest checkpoint under `training/<name>/checkpoints/` if one exists, else start fresh — **no error** (the idempotent bootstrap `scripts/train.sh` relies on) |
| `latest` | continue from the newest checkpoint; **hard error** if none exists (never a silent fresh start) |
| `<path/to/ckpt.zip>` | resume that exact checkpoint |

## Flight training (UC-03)

UC-03 delivers the first real flight task: a **start → gate → finish** course, a connectome-seeded
PPO policy, checkpoint/resume, evaluation, and a one-command bootstrap. Everything is driven
through the `drone-fly` CLI.

### The two simulator backends

The environment talks to the drone through a **sim-agnostic adapter** with two backends:

| Backend | Import | Physics | Used by |
|---|---|---|---|
| `SimpleDroneAdapter` | pure numpy, **no pybullet** | a fixed point-mass model | CI, `smoke-train`, every default test — fully hermetic |
| `PyBulletAdapter` | `gym-pybullet-drones` (guarded/lazy) | real rigid-body quadrotor | the owner's dev-time **mastery** run |

`adapter: auto` (the default for `train`/`evaluate`) uses PyBullet if it is importable, else the
numpy model, and **logs loudly which backend is active**. `smoke-train` always forces `simple`.
(Note: `prune-trained`'s adapter default is `simple`, not `auto`.)

> **`SimpleDroneAdapter` is not mastery physics.** The ≥ 80% completion bar (below) is defined
> against real PyBullet dynamics. The numpy model exists only so the env + policy + PPO loop are
> correctness-testable without the native sim. **CI-green ≠ mastery-achieved.**

### Commands

```sh
# Hermetic correctness run — a handful of steps on the numpy backend, no pybullet.
# This is exactly what CI runs; it proves the loop wires together and stays finite.
# (smoke-train keeps its flags; it is not config-driven.)
uv run drone-fly smoke-train --connectome tests/fixtures

# Full mastery training (owner's machine). Auto-detects the device; resumable.
# Everything — the run name, connectome, timesteps, device, resume policy — lives in the
# config. Edit configs/train/example.yaml (or copy it): set `name`, point `connectome:` at a
# provisioned dir (`tests/fixtures` works out of the box; swap in data/connectome for the
# fuller matrix), and raise `timesteps` for a real run.
uv run drone-fly train --config configs/train/example.yaml

# Resume policy is expressed in the config, not on the command line: set `resume: auto`
# (continue-or-fresh, no error) or `resume: latest` (require an existing checkpoint), or point
# `resume:` at an exact .zip. See "Resume (config-driven)" above.

# Evaluate a checkpoint over N episodes: completion rate + mean start→gate→finish time.
# Set `checkpoint:` (and optionally `vecnormalize:`, `episodes:`) in the evaluate config.
uv run drone-fly evaluate --config configs/evaluate/example.yaml
```

The config's `n_envs` (≥ 1) overrides `TrainConfig.n_envs` (parallel PPO rollout environments)
for a single `train` run: raise it to spread rollout collection across more CPU cores / increase
throughput. It defaults to **1** and omitting the key is byte-identical to today's behaviour; it
is also safe with a resume (Stable-Baselines3 rebuilds the rollout buffer to match the new env
count).

### Where training output is stored (and how to move it without git)

All output lands under a **`training/<name>/` folder created in the directory you run the command
from** — i.e. `<repo>/training/<name>/` when you launch from the repo root (as `scripts/train.sh`
does), where `<name>` is the required `name:` from the train config. Nothing is written anywhere
else, and **`training/` is gitignored**, so it is never committed or synced — the only copy of your
trained model is whatever is on that disk until *you* copy it. (See
[Per-run output layout](#per-run-output-layout-trainingname) for the full breakdown.)

| Path | Contents |
|---|---|
| `<repo>/training/<name>/checkpoints/` | Checkpoints `ppo_racer_<steps>_steps.zip` (every N steps), the matching `ppo_racer_vecnormalize_<steps>_steps.pkl`, and the canonical `vecnormalize.pkl`. **This is your trained agent.** |
| `<repo>/training/<name>/logs/` | The learning curve: `progress.csv` (open in any spreadsheet) and TensorBoard `events.out.tfevents.*`. |

To **back up or move** a run to another machine, just copy the whole `training/<name>/` folder
(Finder drag-and-drop, or `cp -R training/<name> /somewhere/`, or a USB drive) — no git needed. To
resume later, put `training/<name>/` back in the repo root and run `drone-fly train` with the same
config using `resume: auto` (or `resume: latest`), or point the config's `resume:` at an exact
`.zip` — or just re-run `scripts/train.sh`, which auto-resumes via `resume: auto`. To evaluate a
model you moved, set its `checkpoint:` and `vecnormalize:` paths in the evaluate config. An
interrupted run loses at most the steps since the last checkpoint (default every 25,000 steps —
tune `checkpoint_freq` in `src/drone_fly/train/config.py`).

### Mastery bar (a goal, not a CI gate)

The real-world goal is **reliable mastery**: the trained policy completes the course (passes the
gate **and** crosses the finish) in **≥ 80% of 20 evaluation episodes**. The threshold and episode
count are tunable constants in `src/drone_fly/train/config.py` (`mastery_threshold`,
`eval_episodes`). CI **never** gates on this bar — it only runs `smoke-train`. If PPO plateaus
below 80%, the tradeoff (more timesteps / a lower bar / an easier course) is surfaced rather than
running unbounded.

Evaluation metric semantics are pinned and disclosed: `completion_rate` is completed / **all** N
episodes; `mean_completion_time` is averaged over **completed episodes only** (reported as `n/a`
when zero completed); `completed_count` and `n_episodes` are always shown.

### Device auto-detection (Apple Silicon)

`train`/`evaluate` auto-select a torch device, or take an explicit `device:` (`cpu`|`cuda`|`mps`)
config key.

- An explicit `device:` always wins. `device: mps` also sets `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Otherwise: CUDA if available → `cuda`; **MPS is never auto-selected** → `cpu`; else `cpu`.

> **Apple-Silicon note.** MPS is opt-in, not automatic. PyTorch's MPS backend has incomplete
> sparse-tensor support and the connectome substrate propagates sparsely; PyBullet physics is
> CPU-bound and the policy net is tiny — so **CPU is the sane default on an M4**. Opt into MPS with
> `device: mps` in the config (which enables the CPU fallback for unsupported ops).

### Simulator bootstrap (one command)

`scripts/train.sh` creates a virtualenv, installs the project + the pinned simulator, verifies
`import pybullet`, and launches training with a YAML config. Idempotent resume comes from the
config's `resume: auto` (continue-or-fresh) — re-running the same config continues from the latest
checkpoint under `training/<name>/checkpoints/`, with no bash-side checkpoint probing:

```sh
./scripts/train.sh                              # uses configs/train/example.yaml
./scripts/train.sh configs/train/my-run.yaml    # first arg = the config to run
```

Device, timesteps, and every other setting live in the config (set `device: mps` there for Apple
Silicon).

The simulator is pinned to an **exact commit** (never floating `main`):
`gym-pybullet-drones @ 7ebad1e` (v2.2.0, the gymnasium-native line matching our gymnasium/SB3
pins) + `pybullet` (the `sim` extra). A failed sim install stops with an actionable message rather
than a cryptic mid-training crash.

#### Simulator bootstrap: tested-vs-untested boundary

The bootstrap's install steps were **really attempted in the Linux CI sandbox** (not just
eyeballed). What was verified here vs. deferred to the owner's macOS M4:

| Step | Verified in this Linux sandbox? | Outcome / note |
|---|---|---|
| Resolve the pin via `git ls-remote` | ✅ Verified | `main` → `7ebad1ecabd28a7000add2d05f888aa2e837c2cc` (v2.2.0); tags `v0.3.0…v1.0.0` also resolve. Pin is a fixed SHA. |
| `pybullet` install (prebuilt wheel) | ✅ Verified (Python 3.11) | `pybullet==3.2.6` installs from its wheel; `import pybullet` **succeeds**. Requires `numpy<2` at runtime (wheel built against the numpy 1.x ABI; numpy ≥ 2 → ABI `ImportError`). |
| `pybullet` build from source (Python 3.12) | ⚠️ Fails here (expected) | v2.2.0 pulls `pybullet==3.2.7`, which has no 3.12 wheel → source build → **`No such file or directory: 'c++'`**. The sandbox has **no C/C++ toolchain** (`gcc`/`g++`/`cc` all absent). On macOS with Xcode CLT this builds. |
| `gym-pybullet-drones @ v1.0.0` install | ⚠️ Needs a workaround | Default install fails: setuptools flat-layout *"Multiple top-level packages discovered"* (repo ships `ros2/`, `experiments/`, … beside `gym_pybullet_drones/`). Workaround verified: clone the ref, add a `setup.cfg` constraining `packages.find` to `gym_pybullet_drones*`. Also, v1.0.0 imports **legacy `gym`**, not gymnasium — which is why we pin **v2.2.0** instead. |
| `gym-pybullet-drones @ v2.2.0` (the pin) | ⚠️ Deferred to macOS | Clean src-layout, poetry build, **gymnasium-native**, aligned gymnasium 1.3 / SB3 2.9 pins. Its only blocker here is the pybullet 3.2.7 source build above (no compiler); it is expected to install on the owner's M4 (Xcode CLT + Python 3.12). |
| Full end-to-end training loop | ✅ Verified (numpy backend) | `smoke-train` + `train` + resume (step counter advances) + `evaluate` all run green on `SimpleDroneAdapter`, hermetically, under `--extra dev`. |

Bottom line: the **pin resolves**, `pybullet` itself installs and imports from a wheel, and the
whole training/eval loop is verified on the hermetic backend. The only step that cannot complete
in this sandbox is compiling `pybullet` from source for Python 3.12 — a missing-C++-toolchain gap,
not a code defect — which succeeds on the owner's macOS M4. `train.sh` hard-fails with an
actionable toolchain message if that build/import ever fails.

### Course geometry & fixed dynamics

The course (start, an ordered sequence of **N gates** — each a 3D centre with a spherical
capture aperture — a finish line, and the arena floor/ceiling) and the episode timing are
documented, tunable constants in `src/drone_fly/env/config.py`. The default course has **3
gates** (UC-09); `single_gate_course()` builds the one-gate course that reproduces the earlier
single-gate behaviour. Gates are passed **in order** by 3D proximity (coming within a gate's
aperture of its centre), and are strictly x-monotonic so the finish stays reachable. The
per-episode step budget scales with the gate count (`max_steps + steps_per_gate·(N−1)`), so N=1
keeps the 400-step floor. By default the course and drone dynamics are **fixed**; **per-episode
course and dynamics randomization are opt-in as of UC-08** (the course axis now also randomizes
the number of gates) — see [Domain randomization (UC-08)](#domain-randomization-uc-08). Obstacles
beyond floor/ceiling remain deferred to a later use case. A fixed seed yields a deterministic
evaluation for a given checkpoint (bit-exact reproducibility *across a resume boundary* is
best-effort, as on-policy PPO keeps no replay buffer).

### CLI stubs

`fetch-connectome` remains a documented stub (connectome provisioning is UC-01 / owner territory —
see [Provisioning connectome data](#provisioning-connectome-data)).

## Subcircuit pruning (UC-04)

Running the **full** MaleCNS connectome (~161k neurons / ~25M edges) as the live policy network
makes PPO intractable on a laptop (full-connectome smoke-training runs at ~0.85 steps/s ≈ weeks for
1M steps), because every forward/backward pass propagates the whole graph even though only the
directed **sensory → motor** subcircuit drives the 4 control outputs. UC-04 adds an opt-in,
deterministic, direction-aware **pruning step** that reduces a loaded connectome to that subcircuit
before the policy is built.

`prune_to_subcircuit(data, *, k=2, rule="path_slack")`
(`src/drone_fly/connectome/prune.py`) returns a **new** `ConnectomeData` — the input is never
mutated — keeping only the neurons and edges on the directed pathway from the sensory population
(`visual_projection`) to the motor population (`descending_neuron`), with `neuron_ids` /
`superclass` / `sign` / `top_nt` re-aligned to the pruned rows. It is **not** a random or
top-degree slice — it is the actual control circuit.

**The rule.** A single configurable **path-slack corridor** with parameter `k` (default `2`):

- Forward BFS from every sensory neuron over **successors** (respecting the `A[i,j] = j→i`
  convention), and backward BFS from every motor neuron over **predecessors**, give each node its
  distance from the sensory set (`d_fwd`) and to the motor set (`d_bwd`).
- Retained = all sensory neurons ∪ one reconstructed shortest sensory→motor path per reachable
  motor (so every retained motor stays reachable *within the pruned graph* for any `k`) ∪ the
  corridor `{u : d_fwd(u) + d_bwd(u) ≤ L + k}`, where `L` is the shortest sensory→motor path length.
- `k = 0` is the tight shortest-path corridor; larger `k` yields a monotone **superset** (richer
  k-hop neighbourhood). The default `k = 2` favours richness, since descending neurons integrate
  broadly. A motor neuron unreachable from any sensory neuron is dropped with a warning; an absent
  `superclass` column, a missing sensory/motor population, `k < 0`, an unknown rule, or a degenerate
  (empty) result all raise a clear error.

**Usage** (opt-in; omitting `prune` leaves UC-01/02/03 behaviour byte-identical):

```yaml
# In a train config: prune the loaded connectome on the fly before building the policy.
connectome: data/connectome
prune: true
prune_k: 0        # tightest corridor (omit for the default k=2)
```

```sh
# Same idea on the hermetic smoke run (smoke-train keeps its flags).
drone-fly smoke-train --connectome tests/fixtures --prune
```

`prune: true` is a **no-op on a resume** (skipped with a warning): a checkpoint already serialises
its own connectome graph, so re-pruning would desync it. `evaluate` rebuilds the pruned graph only
when it needs the connectome for recording (`prune`/`prune_k` must match training) — the pruned
graph is otherwise carried inside the checkpoint.

### Prune once, reuse (the `prune` export command)

Pruning the full 25M-edge matrix is a one-time cost you don't want to pay on every `train` run. The
`prune` subcommand runs the pruning once and **writes the pruned connectome to disk** in the same
on-disk format the loader reads, so you can point `train`/`smoke-train` at the saved slice with no
re-pruning (`prune: false` / no `--prune`) and no recompute:

```sh
# Prune the full matrix once and save the reusable slice. The shipped configs/prune/{k0,k1,k2}.yaml
# prune tests/fixtures; copy one and set `connectome: data/connectome`, `out: data/pruned`.
drone-fly prune --config configs/prune/k2.yaml    # default rule + k=2 (k0.yaml = tighter slice)

# Reuse it directly — in a train config, set `connectome: data/pruned` (no prune, no re-pruning):
drone-fly train --config configs/train/example.yaml
```

The output directory gets `connectome_pruned.npz` + `connectome_pruned_meta.csv` (carrying the
re-aligned `idx` / `bodyid` / `superclass` / `sign` / `top_nt` columns) plus a `PRUNE_PROVENANCE.md`
note (source, rule, `k`, input→pruned counts). The write **round-trips**: `load_connectome(<out>)`
reproduces the same pruned graph (identical neuron/edge counts and aligned meta), and is
deterministic. This also gives downstream tooling a stable, saved slice to work from.

**Reduction on the committed fixture** (the small 300-neuron / 10,600-edge real-MaleCNS hub-slice
under `tests/fixtures/`, measured by the tests):

| `k` | Neurons | Edges |
|---|---|---|
| 0 | 59 / 300 | 751 / 10,600 |
| 1 | 175 / 300 | 5,988 / 10,600 |
| 2 | 274 / 300 | 10,070 / 10,600 |

> **The fixture numbers are measured; the full-MaleCNS numbers are a *projection, not a
> measurement*.** The committed fixture is a dense, near-fully-connected hub-slice, so even `k = 2`
> barely reduces it — that is an artifact of the slice, not of the rule. On the full 161k-neuron /
> 25M-edge matrix the sensory→motor subcircuit is expected to be order-thousands of neurons (≪ 161k),
> but the multi-GB dataset is not present in this repo, so those figures are not asserted here. To
> **measure** the real reduction, provision the full matrix (see
> [Provisioning connectome data](#provisioning-connectome-data)) and run:
>
> ```yaml
> # train config
> connectome: <full-matrix-dir>
> prune: true
> ```
>
> The prune step logs the exact input→pruned neuron and edge counts.

## Post-training activation pruning (UC-07)

UC-04 prunes the connectome **structurally, before training**, so a trainable sensory→motor circuit
exists at all. UC-07 is the complement: **data-driven, post-training** pruning of a *trained*
policy. Once a policy has learned to fly, many interneurons sit near-zero across a rollout — fly-brain
functions (smell, memory, …) that training never recruited. This step **measures each neuron's
activation over a representative set of episodes**, removes the below-threshold interneurons, then
**fine-tunes** the survivors. The output is two things at once: a **smaller, faster policy** and the
**minimal functional flight circuit** — the sub-network the trained policy actually uses — saved as a
loadable, viewer-ready `ConnectomeData` slice. It is **opt-in** (a distinct `prune-trained`
subcommand); omitting it leaves UC-01…UC-06 byte-identical.

**The distribution caveat — read this first.** "Never activates" is only meaningful relative to an
input distribution. With no domain randomization, a trained policy memorizes one path, and measuring
on that path prunes the circuit down to that path — compressing the fly brain into a lookup table for
one trajectory. So measure over a **varied** distribution (`randomize` / `randomize_dynamics`,
UC-08). When neither axis is enabled the step **loudly warns** that the produced circuit is
**course-specific** (a demonstration of *this* run's used sub-network, not a general fly circuit) and
records that in the report and the slice provenance. In hermetic CI (fixed course) the warning always
fires by design.

**Importance metric.** Default `mean_abs` = per-neuron mean `|activation|` over all measured frames;
the secondary `active_fraction` (fraction of frames above `eps`) is always reported too. Both are
cheap magnitude statistics; a gradient/ablation metric would be more faithful but costlier and is
left as a documented future option. Pruning aggressiveness is always judged by **completion-rate
after**, never the proxy alone. `threshold` (default `1e-3`, absolute, calibrated against the
`tanh`-bounded `[-1, 1]` activation range) sets the cut; `threshold_mode: percentile` interprets it
as "drop the least-active X%" instead.

**What it guarantees.** The sensory (`visual_projection`) and motor (`descending_neuron`) endpoints
are **always retained** — pruning targets interneurons only, so the input/output boundary is never
silently severed. Sub-populations are **pinned by neuron identity** and persisted into the checkpoint,
so `PPO.load` rebuilds the same circuit rather than re-selecting a drifted sub-population. Trained
weights transfer **exactly** (retain-all ⇒ the pruned policy's action equals the original bit-for-bit
— the faithfulness anchor). After slicing, a forward-reachability check confirms each pinned motor is
still reachable from the sensory set on the pruned graph (partial ⇒ warn + `disconnected_motors` in
the report; zero ⇒ hard error). The input checkpoint and connectome are never mutated, and a fixed
`(seed, metric, threshold, episodes)` yields a deterministic pruned graph.

**Usage** (opt-in; the checkpoint does **not** store the connectome, so set the SAME
`connectome`/`prune`/`prune_k` used at training time). Start from
`configs/prune-trained/example.yaml`:

```yaml
# Measure over 20 randomized episodes, drop dead interneurons, fine-tune, save the minimal circuit.
checkpoint: training/baseline/checkpoints/ppo_racer_final.zip
connectome: data/pruned
vecnormalize: training/baseline/checkpoints/vecnormalize.pkl
metric: mean_abs
threshold: 0.001
episodes: 20
finetune_steps: 50000
randomize: true
out: artifacts/minimal_circuit
```

```yaml
# Hermetic demonstration on the fixture (fires the course-specific warning by design):
checkpoint: <ckpt>
connectome: tests/fixtures
prune: true
episodes: 3
finetune_steps: 0
out: /tmp/uc07_demo
```

```sh
drone-fly prune-trained --config configs/prune-trained/example.yaml
```

The `out` directory gets: `pruned_model.zip` (the fine-tuned smaller checkpoint), its
`vecnormalize.pkl`, `connectome_pruned.npz` + `_meta.csv` (the pruned slice — round-trips through
`load_connectome`, loadable by the UC-05/UC-06 viewer), a `PRUNE_TRAINED_PROVENANCE.md` note, and a
`PRUNE_TRAINED_REPORT.md` reporting neurons/edges before→after, the metric summary, the
**completion rate before / immediately-after-prune / after-fine-tune** (honest even when fine-tune
does not fully recover), `disconnected_motors`, and the course-specific warning when applicable.

> **Real reduction / completion numbers are a dev-time step on the owner's machine** with the full
> trained policy and a real fine-tune (documented boundary, mirroring UC-03/UC-04). The hermetic
> tests assert the plumbing — measurement, structured prune, endpoint retention, exact weight
> transfer, save→`PPO.load` round-trip, and the guards — not flight convergence. **Scope:**
> activation measurement + structured neuron-level pruning (reusing UC-04's `slice_connectome`) +
> short fine-tune + saving the checkpoint and slice + the report. Out of scope: any new training
> regime, domain randomization (UC-08 owns it), and viewer changes (UC-06 owns those).

## Activation recording & playback (UC-05)

Make the connectome controller *observable*: record what the fly "brain" is doing while it
flies, then play it back in a dependency-free browser viewer. Recording is **opt-in and off
by default** — with it off, train/eval numerics are byte-identical to UC-01…UC-04.

### Recording (config keys)

Recording attaches to the **evaluate** path (the tested, deterministic primary) and, best-effort,
to **train** (capture the brain mid-learning). It captures, for every Nth episode, the per-frame
neuron-activation vector over the pruned graph, the 4 control channels (throttle/roll/pitch/yaw),
the drone trajectory, and the episode outcome:

```yaml
# evaluate config — record every 5th episode. IMPORTANT: set the SAME connectome/prune/prune_k you
# trained with — the checkpoint does NOT store neuron_ids / superclass / soma positions, so the
# recorder re-loads the connectome and hard-asserts len(neuron_ids) == the actor's neuron count.
checkpoint: training/baseline/checkpoints/ppo_racer_final.zip
vecnormalize: training/baseline/checkpoints/vecnormalize.pkl
connectome: data/pruned
record: true
record_every: 5
name: baseline          # optional — routes recordings to training/baseline/recordings/
```

```yaml
# train config — training-time capture (documented best-effort; eval is the tested path):
name: baseline
connectome: data/full
prune: true
record: true
record_every: 50
```

Keys: `record: true` (enable), `record_every: N` (cadence, default 1), `record_dir: <dir>`
(override; defaults to `training/<name>/recordings/` for a train run, or when `evaluate` sets a
`name` with `record: true`, else `artifacts/activations/`). Each recorded episode is written to its
own self-contained file `episode_<n>.json`. Activations are `tanh`-bounded to `[-1, 1]` and
stored **quantised to `uint8`** (with `activation_scale`/`activation_offset` in the metadata for
exact dequantisation) — roughly 4× smaller than float32; the recorder logs each file's path and
size. Best paired with the **UC-04 pruned slice** (a few thousand neurons is legible; the full
161k is not).

Each recording also carries the **course geometry** under `meta.course`, so the viewer can
place the 3D floor and the start/gate/finish markers without any external config. It is an
**additive, back-compatible** block (recordings written before UC-06 simply lack it; the viewer
degrades gracefully). The values are read from `env/config.py`'s `CourseConfig` — coordinates
are in the env frame (z up, +x forward, right-handed). Since UC-09 a course holds **N ordered
gates**, so the block carries a `gates: [...]` array (one entry per gate, in pass order):

```json
"course": {
  "start":  [0.0, 0.0, 1.0],
  "gates": [
    { "center": [2.5,  0.0, 1.0], "aperture": 0.6, "plane": "yz" },
    { "center": [4.0,  0.6, 1.3], "aperture": 0.6, "plane": "yz" },
    { "center": [5.5, -0.5, 0.9], "aperture": 0.6, "plane": "yz" }
  ],
  "finish": { "x": 7.0 },
  "floor_z": 0.0,
  "ceiling_z": 2.5,
  "forward_axis": "x",
  "up_axis": "z"
}
```

These are semantic anchors only; the viewer derives the displayed floor/finish extent itself
(a padded bounding box over the trajectory and these anchors) so the floor always contains the
flown path. The viewer reads `course.gates`, falling back to a legacy singular `course.gate`
for pre-UC-09 recordings, and draws one ring per gate. When the file also carries a per-frame
`frames.target_gate` track (UC-09), the **current target gate** is highlighted (brighter, in a
distinct colour) while the others are dimmed; legacy files without the track render every gate
uniformly.

### Anatomical coordinates (real soma positions, tokenless by default)

The spatial brain map draws each neuron at its **real soma position**. These come from the
`somaLocation` column of the MaleCNS `connectome_data_prep` metadata — the *same value*
neuPrint's `fetch_neurons().somaLocation` returns, mirrored on public GitHub under CC-BY — so
the committed fixture demo is anatomical **offline, with no token**. The coordinate source order:

1. **Committed / local anatomical CSV** (tokenless, the default): a fixture sidecar
   `tests/fixtures/mcns_fixture_soma.csv` (produced by `scripts/fetch_soma_positions.py`), or a
   CSV named by `DRONE_FLY_SOMA_CSV`.
2. **neuPrint** via `neuprint-python` + `NEUPRINT_TOKEN` — a dev-time, networked step, only
   needed for arbitrary slices whose bodyids are not in a committed sidecar.
3. **Deterministic spectral layout** — a seed-free, sign-canonicalised spectral embedding of the
   pruned graph, used when no anatomy is reachable and **clearly labelled "computed (NOT
   anatomical)"** in the file and the viewer. Neurons missing a soma are flagged
   (`has_position=False`) and fallback-placed — never dropped, never fabricated.

Positions are stored full-3D plus a top-down projection (default dorsal plane; the viewer's
**front / side / top-down** view-preset selector switches planes, **top-down** default). The
8 nm voxel scale is assumed and irrelevant to a normalised top-down map.

Provision the fixture's real coordinates (dev-time, needs network — **tested boundary**: this
script is run and its 300/300 coverage asserted; it is not run in CI):

```sh
uv run python scripts/fetch_soma_positions.py   # writes tests/fixtures/mcns_fixture_soma.csv
```

### Viewer (`viz/`, no build step)

Open `viz/viewer.html` directly in a browser (no server, no npm) and pick a recorded file with
the file picker (works from `file://`). Three panels share one play/pause + scrubber timeline,
with speed presets **0.25× / 0.5× / 1× / 2× / 4×** (0.25× for slow, detailed inspection).
Pressing **Play** at the end restarts from the beginning.

- **Anatomical brain map** — neurons as dots at their projected soma positions, role-coloured,
  with a **front / side / top-down** view-preset selector (plain terms, never axis names;
  **top-down** default) and an anatomical-vs-computed label. Each neuron renders **~2px at
  rest** and **"beats"** — pulsing to ~9px on activation and easing back to rest over ~0.2s —
  so activation reads as a visible beat, not just a colour change. While paused or scrubbing,
  a neuron's size reflects *that frame's* activation exactly (no lingering animation).
- **Activation heatmap** — neurons × time, rows ordered sensory → interneuron → motor.
- **Flight panel** — the 4 action traces plus a genuinely **3D, orbitable flight scene**:
  a floor grid, **start** (green) / **gates** (one ring per gate, radius = aperture, with the
  **current target gate highlighted** and the rest dimmed when the recording carries a
  per-frame target-gate track) / **finish** (a translucent wireframe plane) markers, the 3D
  trajectory (flown bright, remaining dim), and a moving drone marker — all synced to the same
  playhead.
  - **Controls:** **drag to rotate** the camera, **wheel to zoom**.
  - **View presets:** a **front / side / top-down** selector (plain terms, never axis names);
    **top-down is the default**. The camera stays freely orbitable after picking a preset.
  - The 3D rendering is a **dependency-free** hand-rolled canvas-2D perspective projector (no
    three.js, no npm, no ES modules) so it stays `file://`-safe. Markers need the recording's
    course geometry (see below); older recordings without it still load — the scene just omits
    the markers it lacks (graceful degradation).

The scene uses the env's real coordinate frame (z up, +x forward, right-handed), so a right
bank reads as a right turn on screen rather than a mirrored one.

Gzipped recordings (`.json.gz`) are decompressed in-browser via `DecompressionStream`.

## Domain randomization (UC-08)

UC-03 trained on a **single fixed course**, so the policy could memorize one trajectory — which is
what the suspiciously perfect "100% completion / 1.00s" fixed-course number really was: one
replayed path, not flying ability. UC-08 makes the task **vary per episode**, so the only way to
earn reward is to actually fly to wherever the waypoints are. Two independent, **off-by-default**
axes:

- **Course randomization (`randomize`)** — the primary anti-memorization axis. Each `reset()`
  samples a new **number of gates** (default range `[1, 10]`, UC-09), a start, and per-gate 3D
  centre + aperture within configured ranges. The observation already carries the *relative*
  next-waypoint pose and the geometry/reward are course-parameterized (the per-gate bonus is
  normalized by gate count, so total gate reward stays comparable as N varies), so no observation-
  or reward-shape change is needed — only *which* course flows in per episode.
- **Dynamics randomization (`randomize_dynamics`)** — a secondary robustness / sim-to-sim axis.
  Each `reset()` samples per-episode mass, drag, thrust response, body-rate limit, and control
  latency. Mass and thrust are **independent** knobs (`thrust_acc = throttle · max_thrust / mass`),
  so a heavier drone genuinely flies differently rather than silently cancelling out.

Either axis can be enabled with the other off; both default off, so a `train` / `evaluate` config
that omits them is **byte-identical to UC-03** (a disabled axis makes no RNG draw, so it can't even
perturb the seeded stream).

```yaml
# train config — randomized courses (the honest, anti-memorization setup)
name: randomized
randomize: true
randomize_dynamics: true   # optional: add dynamics randomization for robustness
```

```yaml
# evaluate config — evaluate over N DIFFERENT sampled courses (seeded -> reproducible eval set)
checkpoint: training/randomized/checkpoints/ppo_racer_final.zip
randomize: true
```

**Ranges are a difficulty knob.** The per-parameter ranges live as documented, tunable constants in
`RandomizationConfig` (`src/drone_fly/env/config.py`), centred on the current `CourseConfig` /
dynamics defaults. Too narrow ⇒ still near-memorization; too wide ⇒ may be untrainable on a laptop
budget. The defaults meaningfully vary the course while staying solvable.

**Every sampled course is flyable (reject-then-fallback).** A pure sampler
(`src/drone_fly/env/randomization.py`) draws off the env's seeded RNG, walking x forward from the
start in incremental deltas (so gate ordering and spacing are structural), and a solvability guard
(`is_course_solvable`) enforces for **every** gate: apertures at least `aperture_min`, strictly
increasing x with adjacent gates at least `min_gate_spacing` apart in 3D, all gate/start heights
*strictly* inside the arena's vertical safety corridor (`floor_z + z_margin`, `ceiling_z − z_margin`),
lateral positions within bounds, the first gate a minimum gap ahead of the start, the finish beyond
the last gate, and the start outside gate 0's capture sphere. Degenerate draws are **rejected and
resampled** up to `max_resample_attempts`, then replaced by a deterministic, **zero-RNG fallback
course** that is solvable-by-construction for every N — so sampling is deterministic and can never
loop forever.

**Seeded and reproducible.** A given seed reproduces the same course *and* dynamics stream (the draw
order is pinned: course then dynamics). An `evaluate` config with `randomize: true` runs over N
different sampled courses that are themselves reproducible for a fixed seed, so two checkpoints are
compared on the *same* course set.

> **The honest-metric drop (read this).** The moment you switch to randomized training, the
> "100% / 1.00s" fixed-course figure **will fall** — because it was a memorized single path. This is
> the *point*, not a regression. `evaluate` with `randomize: true` reports a *distinct* number (tagged
> `mode=randomized` in the summary and `randomized=True` on `EvalMetrics`); that lower-but-honest
> completion rate over randomized courses is the **real baseline**. Do not read the drop as a step
> backward.

**Recorder stamps the per-episode course.** With randomization on, each recorded episode's
`meta.course` reflects **that episode's sampled course** (read back from the env's `active_course`),
so the UC-06 3D viewer draws the correct start / gate / finish / aperture markers per episode rather
than the static default.

**Scope guard.** UC-08 delivers course randomization + optional dynamics randomization + config/CLI
toggles + eval-over-randomized + recorder per-episode course stamping — **and nothing more**. It does
**not** add curriculum learning, obstacles / multi-gate courses, or any change to the training
algorithm; those remain deferred to future use cases.

## Project layout

```
src/drone_fly/
  connectome/   load cached MaleCNS connectivity from disk (offline loader.py) + sensory→motor pruning (prune.py)
  controller/   connectome-seeded PyTorch policy + encode/decode roundtrip (opt-in activation sink)
  adapter/      sim-agnostic drone backends (numpy SimpleDroneAdapter + guarded PyBulletAdapter)
  env/          Gymnasium start→gate→finish racing env (geometry, reward, RaceEnv)
  record/       UC-05 activation recording: recorder, rollout driver, soma-coordinate provisioning
  train/        PPO training loop, checkpoint/resume, device auto-detect
  evaluate/     run a trained agent, report completion rate + mean time
  config.py     YAML --config loading + per-command schema validation + run-layout helpers
  cli/          command-line entry points (train, evaluate, prune, prune-trained, smoke-train)
viz/            dependency-free static activation-playback viewer (HTML/JS/CSS, no build step)
configs/        example/reference YAML run-configs (train/, prune/, evaluate/, prune-trained/)
scripts/        dev-time utilities (build_test_fixture.py, fetch_soma_positions.py) + train.sh bootstrap
tests/          pytest suite (import + connectome-plumbing tests)
tests/fixtures/ committed small real-MaleCNS subgraph used by the offline tests
docs/adr/       architecture decision records
data/           runtime connectome cache (gitignored contents)
training/       per-run output: training/<name>/{checkpoints,logs,recordings}/ (gitignored contents)
artifacts/      pruned connectome slices + other byproducts (gitignored contents)
```

See `PROJECT_BRIEF.md` for the full project definition and the reasoning behind the scope
decisions.

## Known limitations

- Early prototype: UC-01 (connectome load), UC-02 (trainable substrate), UC-03 (flight task —
  env, PPO train/resume, evaluation, bootstrap), UC-04 (opt-in sensory→motor subcircuit
  pruning), UC-05 (opt-in activation recording + static playback viewer), and UC-08 (opt-in
  per-episode course + dynamics randomization) are implemented; later stages (obstacles,
  multi-gate courses, curriculum learning) are deferred.
- The activation viewer is a static, top-down playback tool (spatial map + heatmap + flight),
  not a live-streaming server or a 3D fly-through; training-time recording is best-effort
  (evaluation recording is the tested, deterministic path).
- Mastery (≥ 80% completion) is a dev-time goal on real PyBullet physics, verified on the owner's
  hardware — **not** something CI checks (CI runs only a hermetic numpy-backend smoke-train).
- Does not integrate with the Liftoff game (no public API); an open sim stands in for it. This
  is an explicit non-goal for now.
- Not a biophysically faithful simulation of the fly brain.
- No real (physical) drone hardware or sim-to-real transfer.
- Single-user local tool; no multi-user or production reliability guarantees.
- The MaleCNS dataset has its own license and citation terms; this repo redistributes only a
  small (~300-neuron) real slice as a test fixture, under the dataset's CC-BY attribution (see
  `tests/fixtures/FIXTURE_PROVENANCE.md`). Bulk connectome data is not committed — provision it
  yourself.
- The UC-01 sensory/motor neuron-index mapping is an arbitrary documented placeholder, not a
  biologically motivated selection; a later use case refines it.

## License

MIT (covers the drone-fly source code only, not the upstream MaleCNS dataset).
