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

### Observation schema & retraining
The `schema` train key (UC-13) opts into a named **block observation schema**. `migrated_v1`
re-binds the 12-d observation into a *vision* block (target-relative → visual neurons) and a
*proprioception* block (self-motion → the mechanosensory/proprioceptive population). Omitting
`schema` keeps the legacy single-projection behaviour. **Note:** a block schema changes the
input→sensory wiring, so checkpoints trained under the old wiring do not carry over — a fresh
train is required.

### Visualization & recording
Enable recording in a train/evaluate config with `record: true` (tune cadence via `record_every`);
frames land in that run's `training/<name>/recordings/`. Open `viz/viewer.html` in a browser
(dependency-free, `file://`-safe) and load a recording — no server or build step.

Two panels: an MRI/fMRI-style activation heatmap over a static, spatially-registered MaleCNS brain
outline (`viz/brain_outline.js`, `top-down` / `front` / `side` presets + a per-frame ↔ global
intensity toggle), and a **3D flight view** driven by the recorded `meta.course` geometry. 3D
controls: **drag** to **rotate**, mouse **wheel** to **zoom**, the same view presets, and a `0.25`×
slow-inspection speed. Regenerate the outline with `uv run python scripts/build_brain_outline.py`.

Neuron coordinates use real MaleCNS **soma** positions; a neuron lacking one uses a deterministic
computed-layout **fallback** (clearly labelled). The committed fixture ships real anatomy
tokenlessly; arbitrary user slices may need a `NEUPRINT_TOKEN` (see `.env.example`), and
`uv run python scripts/fetch_soma_positions.py` refreshes the committed soma sidecar.

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

The gym-pybullet-drones pin is resolved with `git ls-remote` to a fixed commit SHA (never floating
`main`); `scripts/train.sh` and the adapter share that SHA and verify `import pybullet` before training.

**Mastery goal & dynamics.** The target is **80**% waypoint **mastery** (course completion). Mastery
runs use **fixed** environment dynamics (**no domain randomization**) for reproducibility; enable
randomization explicitly only for robustness experiments.

### Project layout & history
Source under `src/drone_fly/` (connectome loader, controller/actor, adapter, env, train, evaluate,
record, clean); tests under `tests/`; the viewer under `viz/`; example configs under `configs/`.
Full project definition is in `PROJECT_BRIEF.md`; feature history and status live in
[`USE_CASES.md`](USE_CASES.md) and `use-cases/`.

## License

MIT (covers the drone-fly source only, not the upstream MaleCNS dataset).
