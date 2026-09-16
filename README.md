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

The connectome runtime is **reused, not built from scratch**: running MaleCNS as a trainable
network is already solved and open-source. drone-fly's real work is the *integration* (wiring an
existing connectome substrate to an existing drone sim) and the *training* (RL for waypoint
racing).

This repository is an early prototype. The **connectome plumbing** (UC-01) is implemented and
tested — offline connectome load, a connectome-seeded policy, and a full encode → network →
decode roundtrip — while the flight environment, RL training, and evaluation stages are still
package skeletons.

> **AxonWeave note.** The intended connectome-substrate library, AxonWeave, is not installable in
> this environment (absent from PyPI; source not publicly reachable at UC-01 time). The controller
> uses it automatically if it ever becomes importable, and otherwise falls back to an in-repo
> `SparseConnectomeLayer` shim that realises the same connectome edges as a sparse trainable
> parameter. The plumbing contract and the roundtrip are identical either way.

## Prior art / references

drone-fly reuses and follows existing open-source work rather than reinventing it:

- [AxonWeave](https://github.com/dhakalnirajan/axonweave) — exposes MaleCNS as a sparse,
  trainable substrate for NumPy/PyTorch/TensorFlow (the reused connectome runtime).
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

- Python 3.11
- [uv](https://docs.astral.sh/uv/) for dependency management
- A free neuPrint auth token (from https://neuprint.janelia.org) to fetch the connectome
- `gym-pybullet-drones` is installed from GitHub, not PyPI (see below)
- A GPU is optional but speeds up training; the prototype is CPU-runnable

## Quick start

```sh
git clone git@github.com:HaroldHormaechea/drone-fly.git
cd drone-fly

# Create the environment and install the project + dev tools.
uv sync --extra dev

# The simulator is GitHub-only; install it separately when you need real training:
uv pip install "git+https://github.com/utiasDSL/gym-pybullet-drones"

# Run the checks that exist today (offline; the committed test fixture is used).
uv run ruff check .
uv run pytest
```

The UC-01 tests run fully offline against a small, committed real-MaleCNS subgraph under
`tests/fixtures/`. A neuPrint token is only needed later, for fetching a fresh connectome — set
it via `cp .env.example .env` then edit `NEUPRINT_TOKEN` when that path is implemented.

### Provisioning connectome data

- **Test fixture (committed):** a ~300-neuron real-MaleCNS slice lives under `tests/fixtures/`,
  regenerated deterministically by `uv run python scripts/build_test_fixture.py` (needs network;
  never run in CI). Its provenance and slice rule are recorded in
  `tests/fixtures/FIXTURE_PROVENANCE.md`.
- **Full matrix (optional):** point `DRONE_FLY_CONNECTOME_DIR` at a directory holding a full
  MaleCNS `.npz` + `*_meta.csv` to load the whole dataset instead of the fixture.

The `fetch-connectome`, `train`, and `evaluate` commands are planned CLI entry points; their
logic is not implemented yet, so they are not runnable from a fresh clone.

## Project layout

```
src/drone_fly/
  connectome/   load cached MaleCNS connectivity from disk (offline loader.py)
  controller/   connectome-seeded PyTorch policy + encode/decode roundtrip
  env/          Gymnasium quadrotor racing environment (gates, collision penalties)
  train/        RL training loop (Stable-Baselines3)
  evaluate/     run a trained agent, report gate/lap times vs. a baseline
  cli/          command-line entry points
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

- Structure-only scaffold: no pipeline logic is implemented yet — only package skeletons and
  import smoke tests.
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
