# drone-fly

Can the wiring diagram of a real fruit-fly brain (the MaleCNS connectome) seed a neural
controller that reinforcement-learns to fly a quadrotor through a timed race course in
simulation?

## Quick start: train a fly to fly

Terse path from clone to a trained policy. Each step links to its full write-up below.

1. **Install** — `git clone <repo> && cd drone-fly && uv sync --extra dev`. See [Requirements](#requirements).
2. **Provision a connectome** — point at the committed fixture (`tests/fixtures`, works offline), or download the full MaleCNS matrix into `data/connectome`. See [Provisioning connectome data](#provisioning-connectome-data-required-before-training).
3. *(Optional)* **Prune once, reuse** — `uv run drone-fly prune --connectome data/connectome --out data/pruned` writes a reusable sensory→motor slice. See [Subcircuit pruning](#subcircuit-pruning-uc-04).
4. **Train** — `uv run drone-fly train --connectome <dir> --timesteps 1000000` (use `data/pruned` for the pruned slice, or add `--prune` to prune on the fly). See [Flight training](#flight-training-uc-03).
5. **Resume** — `uv run drone-fly train --resume artifacts/models/ppo_racer_<steps>_steps.zip`.
6. **Evaluate** — `uv run drone-fly evaluate --checkpoint artifacts/models/ppo_racer_final.zip --vecnormalize artifacts/models/vecnormalize.pkl --episodes 20`.

> Hermetic sanity check (no network, seconds): `uv run drone-fly smoke-train --connectome tests/fixtures`.

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

Every `train` / `smoke-train` / `evaluate` command needs a `--connectome <dir>` — a directory that
holds **one** SciPy-sparse `.npz` connectivity matrix plus a sidecar CSV. **Important naming rule:**
the loader looks for the CSV named after the matrix stem — a matrix `foo.npz` must sit next to
`foo_meta.csv` in the same directory. If the names don't match, the loader can't find the metadata.

You have three options, easiest first:

**Option A — use the committed fixture (zero download, recommended to start).** A real ~300-neuron
MaleCNS slice already ships in the repo under `tests/fixtures/` (`mcns_fixture.npz` +
`mcns_fixture_meta.csv`). It's small but real, and training works against it immediately:

```sh
uv run drone-fly train --connectome tests/fixtures --timesteps 1000000
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

uv run drone-fly train --connectome data/connectome --timesteps 1000000
```

`data/connectome/` is the default location (`DRONE_FLY_CONNECTOME_DIR`, else `data/connectome`), so
once those two files are there you may omit `--connectome` entirely. `data/` contents are
gitignored — the download stays local. (The dataset is CC-BY; cite MaleCNS / `connectome_data_prep`.)

**Option C — regenerate the committed slice.** `uv run python scripts/build_test_fixture.py`
downloads the same source and writes a deterministic slice to `tests/fixtures/` (dev-time only,
needs network; provenance recorded in `tests/fixtures/FIXTURE_PROVENANCE.md`).

## Flight training (UC-03)

UC-03 delivers the first real flight task: a **start → gate → finish** course, a connectome-seeded
PPO policy, checkpoint/resume, evaluation, and a one-command bootstrap. Everything is driven
through the `drone-fly` CLI (or `python -m drone_fly.cli`).

### The two simulator backends

The environment talks to the drone through a **sim-agnostic adapter** with two backends:

| Backend | Import | Physics | Used by |
|---|---|---|---|
| `SimpleDroneAdapter` | pure numpy, **no pybullet** | a fixed point-mass model | CI, `smoke-train`, every default test — fully hermetic |
| `PyBulletAdapter` | `gym-pybullet-drones` (guarded/lazy) | real rigid-body quadrotor | the owner's dev-time **mastery** run |

`--adapter auto` (default) uses PyBullet if it is importable, else the numpy model, and **logs
loudly which backend is active**. `smoke-train` always forces `simple`.

> **`SimpleDroneAdapter` is not mastery physics.** The ≥ 80% completion bar (below) is defined
> against real PyBullet dynamics. The numpy model exists only so the env + policy + PPO loop are
> correctness-testable without the native sim. **CI-green ≠ mastery-achieved.**

### Commands

```sh
# Hermetic correctness run — a handful of steps on the numpy backend, no pybullet.
# This is exactly what CI runs; it proves the loop wires together and stays finite.
uv run drone-fly smoke-train --connectome tests/fixtures

# Full mastery training (owner's machine). Auto-detects the device; resumable.
# --connectome points at a provisioned connectome dir — see "Provisioning connectome
# data" above. `tests/fixtures` works out of the box; swap in data/connectome for the
# fuller matrix.
uv run drone-fly train --connectome tests/fixtures --timesteps 1000000

# Resume an interrupted run from a checkpoint (step counter continues, not restarts).
uv run drone-fly train --resume artifacts/models/ppo_racer_120000_steps.zip

# Evaluate a checkpoint over 20 episodes: completion rate + mean start→gate→finish time.
uv run drone-fly evaluate \
  --checkpoint artifacts/models/ppo_racer_final.zip \
  --vecnormalize artifacts/models/vecnormalize.pkl --episodes 20
```

### Where training output is stored (and how to move it without git)

All output lands under an **`artifacts/` folder created in the directory you run the command from**
— i.e. `<repo>/artifacts/` when you launch from the repo root (as `scripts/train.sh` does). Nothing
is written anywhere else, and **`artifacts/` is gitignored**, so it is never committed or synced —
the only copy of your trained model is whatever is on that disk until *you* copy it.

| Path | Contents |
|---|---|
| `<repo>/artifacts/models/` | Checkpoints `ppo_racer_<steps>_steps.zip` (every N steps), the matching `ppo_racer_vecnormalize_<steps>_steps.pkl`, and the canonical `vecnormalize.pkl`. **This is your trained agent.** |
| `<repo>/artifacts/logs/` | The learning curve: `progress.csv` (open in any spreadsheet) and TensorBoard `events.out.tfevents.*`. |

To **back up or move** a run to another machine, just copy the whole `artifacts/` folder (Finder
drag-and-drop, or `cp -R artifacts /somewhere/`, or a USB drive) — no git needed. To resume later,
put `artifacts/models/` back in the repo root and run
`drone-fly train --resume artifacts/models/ppo_racer_<steps>_steps.zip` (or just re-run
`scripts/train.sh`, which auto-resumes from the latest checkpoint). To evaluate a model you moved,
pass its `--checkpoint` and `--vecnormalize` paths explicitly. An interrupted run loses at most the
steps since the last checkpoint (default every 25,000 steps — tune `checkpoint_freq` in
`src/drone_fly/train/config.py`).

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

`train`/`evaluate` auto-select a torch device, or take an explicit `--device {cpu,cuda,mps}`.

- Explicit `--device` always wins. `--device mps` also sets `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Otherwise: CUDA if available → `cuda`; **MPS is never auto-selected** → `cpu`; else `cpu`.

> **Apple-Silicon note.** MPS is opt-in, not automatic. PyTorch's MPS backend has incomplete
> sparse-tensor support and the connectome substrate propagates sparsely; PyBullet physics is
> CPU-bound and the policy net is tiny — so **CPU is the sane default on an M4**. Opt into MPS with
> `--device mps` (which enables the CPU fallback for unsupported ops).

### Simulator bootstrap (one command)

`scripts/train.sh` creates a virtualenv, installs the project + the pinned simulator, verifies
`import pybullet`, and launches (or resumes) training. It is **idempotent** — re-running resumes
from the latest checkpoint:

```sh
./scripts/train.sh                 # bootstrap + train/resume
./scripts/train.sh --device mps    # extra args pass through to `drone-fly train`
```

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
| Full end-to-end training loop | ✅ Verified (numpy backend) | `smoke-train` + `train` + `--resume` (step counter advances) + `evaluate` all run green on `SimpleDroneAdapter`, hermetically, under `--extra dev`. |

Bottom line: the **pin resolves**, `pybullet` itself installs and imports from a wheel, and the
whole training/eval loop is verified on the hermetic backend. The only step that cannot complete
in this sandbox is compiling `pybullet` from source for Python 3.12 — a missing-C++-toolchain gap,
not a code defect — which succeeds on the owner's macOS M4. `train.sh` hard-fails with an
actionable toolchain message if that build/import ever fails.

### Course geometry & fixed dynamics

The course (start, one gate with a circular aperture, finish line, arena floor/ceiling) and the
episode timing are documented, tunable constants in `src/drone_fly/env/config.py`. **Drone
dynamics are fixed for UC-03** — no domain randomization, no obstacles beyond floor/ceiling, no
multi-gate; those are deliberately deferred to a later use case. A fixed seed yields a
deterministic evaluation for a given checkpoint (bit-exact reproducibility *across a resume
boundary* is best-effort, as on-policy PPO keeps no replay buffer).

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

**Usage** (opt-in; omitting `--prune` leaves UC-01/02/03 behaviour byte-identical):

```sh
# Train on the pruned subcircuit of the full matrix (owner's machine — measures the real reduction).
drone-fly train --connectome data/connectome --prune            # default k=2
drone-fly train --connectome data/connectome --prune --prune-k 0  # tightest corridor

# Same flags on the hermetic smoke run.
drone-fly smoke-train --connectome tests/fixtures --prune
```

`--prune` is a **no-op on `--resume`** (skipped with a warning): a checkpoint already serialises its
own connectome graph, so re-pruning would desync it. There is no `--prune` flag on `evaluate` for
the same reason — the pruned graph is carried inside the checkpoint.

### Prune once, reuse (the `prune` export command)

Pruning the full 25M-edge matrix is a one-time cost you don't want to pay on every `train` run. The
`prune` subcommand runs the pruning once and **writes the pruned connectome to disk** in the same
on-disk format the loader reads, so you can point `train`/`smoke-train` at the saved slice with no
`--prune` flag and no recompute:

```sh
# Prune the full matrix once and save the reusable slice.
drone-fly prune --connectome data/connectome --out data/pruned            # default rule + k=2
drone-fly prune --connectome data/connectome --out data/pruned0 --prune-k 0  # tighter slice

# Reuse it directly — no --prune, no re-pruning.
drone-fly train --connectome data/pruned --timesteps 1000000
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
> ```sh
> drone-fly train --connectome <full-matrix-dir> --prune
> ```
>
> The prune step logs the exact input→pruned neuron and edge counts.

## Activation recording & playback (UC-05)

Make the connectome controller *observable*: record what the fly "brain" is doing while it
flies, then play it back in a dependency-free browser viewer. Recording is **opt-in and off
by default** — with it off, train/eval numerics are byte-identical to UC-01…UC-04.

### Recording (flags)

Recording attaches to the **evaluate** path (the tested, deterministic primary) and, best-effort,
to **train** (capture the brain mid-learning). It captures, for every Nth episode, the per-frame
neuron-activation vector over the pruned graph, the 4 control channels (throttle/roll/pitch/yaw),
the drone trajectory, and the episode outcome:

```sh
# Record every 5th eval episode. IMPORTANT: pass the SAME --connectome/--prune/--prune-k you
# trained with — the checkpoint does NOT store neuron_ids / superclass / soma positions, so the
# recorder re-loads the connectome and hard-asserts len(neuron_ids) == the actor's neuron count.
uv run drone-fly evaluate --checkpoint artifacts/models/ppo_racer_final.zip \
  --vecnormalize artifacts/models/vecnormalize.pkl \
  --connectome data/pruned --record --record-every 5

# Training-time capture (documented best-effort; eval is the tested path):
uv run drone-fly train --connectome data/full --prune --record --record-every 50
```

Flags: `--record` (enable), `--record-every N` (cadence, default 1), `--record-dir <dir>`
(default `artifacts/activations/`). Each recorded episode is written to its own self-contained
file `artifacts/activations/episode_<n>.json`. Activations are `tanh`-bounded to `[-1, 1]` and
stored **quantised to `uint8`** (with `activation_scale`/`activation_offset` in the metadata for
exact dequantisation) — roughly 4× smaller than float32; the recorder logs each file's path and
size. Best paired with the **UC-04 pruned slice** (a few thousand neurons is legible; the full
161k is not).

Each recording also carries the **course geometry** under `meta.course`, so the viewer can
place the 3D floor and the start/gate/finish markers without any external config. It is an
**additive, back-compatible** block (recordings written before UC-06 simply lack it; the viewer
degrades gracefully). The values are read from `env/config.py`'s `CourseConfig` — coordinates
are in the env frame (z up, +x forward, right-handed):

```json
"course": {
  "start":  [0.0, 0.0, 1.0],
  "gate":   { "center": [3.0, 0.0, 1.0], "aperture": 0.6, "plane": "yz" },
  "finish": { "x": 6.0 },
  "floor_z": 0.0,
  "ceiling_z": 2.5,
  "forward_axis": "x",
  "up_axis": "z"
}
```

These are semantic anchors only; the viewer derives the displayed floor/finish extent itself
(a padded bounding box over the trajectory and these anchors) so the floor always contains the
flown path.

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

Positions are stored full-3D plus a top-down projection (default dorsal **x–z** plane; the
viewer's axis selector switches planes). The 8 nm voxel scale is assumed and irrelevant to a
normalised top-down map.

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
  with an axis selector and an anatomical-vs-computed label. Each neuron renders **~2px at
  rest** and **"beats"** — pulsing to ~9px on activation and easing back to rest over ~0.2s —
  so activation reads as a visible beat, not just a colour change. While paused or scrubbing,
  a neuron's size reflects *that frame's* activation exactly (no lingering animation).
- **Activation heatmap** — neurons × time, rows ordered sensory → interneuron → motor.
- **Flight panel** — the 4 action traces plus a genuinely **3D, orbitable flight scene**:
  a floor grid, **start** (green) / **gate** (ring, radius = aperture) / **finish** (a
  translucent wireframe plane) markers, the 3D trajectory (flown bright, remaining dim), and a
  moving drone marker — all synced to the same playhead.
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
  cli/          command-line entry points (train, evaluate, smoke-train, prune)
viz/            dependency-free static activation-playback viewer (HTML/JS/CSS, no build step)
scripts/        dev-time utilities (build_test_fixture.py, fetch_soma_positions.py)
tests/          pytest suite (import + connectome-plumbing tests)
tests/fixtures/ committed small real-MaleCNS subgraph used by the offline tests
docs/adr/       architecture decision records
data/           runtime connectome cache (gitignored contents)
artifacts/      model checkpoints, TensorBoard logs, eval outputs, activation recordings (gitignored contents)
```

See `PROJECT_BRIEF.md` for the full project definition and the reasoning behind the scope
decisions.

## Known limitations

- Early prototype: UC-01 (connectome load), UC-02 (trainable substrate), UC-03 (flight task —
  env, PPO train/resume, evaluation, bootstrap), UC-04 (opt-in sensory→motor subcircuit
  pruning), and UC-05 (opt-in activation recording + static playback viewer) are implemented;
  later stages (domain randomization, obstacles, multi-gate courses) are deferred.
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
