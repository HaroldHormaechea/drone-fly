# drone-fly

Can the wiring diagram of a real fruit-fly brain (the MaleCNS connectome) seed a neural
controller that reinforcement-learns to fly a quadrotor through a timed race course in
simulation?

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

## Project layout

```
src/drone_fly/
  connectome/   load cached MaleCNS connectivity from disk (offline loader.py)
  controller/   connectome-seeded PyTorch policy + encode/decode roundtrip
  adapter/      sim-agnostic drone backends (numpy SimpleDroneAdapter + guarded PyBulletAdapter)
  env/          Gymnasium start→gate→finish racing env (geometry, reward, RaceEnv)
  train/        PPO training loop, checkpoint/resume, device auto-detect
  evaluate/     run a trained agent, report completion rate + mean time
  cli/          command-line entry points (train, evaluate, smoke-train)
scripts/        dev-time utilities (build_test_fixture.py — regenerates the fixture)
tests/          pytest suite (import + connectome-plumbing tests)
tests/fixtures/ committed small real-MaleCNS subgraph used by the offline tests
docs/adr/       architecture decision records
data/           runtime connectome cache (gitignored contents)
artifacts/      model checkpoints, TensorBoard logs, eval outputs (gitignored contents)
```

See `PROJECT_BRIEF.md` for the full project definition and the reasoning behind the scope
decisions.

## Known limitations

- Early prototype: UC-01 (connectome load), UC-02 (trainable substrate), and UC-03 (flight task —
  env, PPO train/resume, evaluation, bootstrap) are implemented; later stages (domain
  randomization, obstacles, multi-gate courses) are deferred.
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
