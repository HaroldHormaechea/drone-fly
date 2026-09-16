# drone-fly

Can the wiring diagram of a real fruit-fly brain (the MaleCNS connectome) seed a neural
controller that reinforcement-learns to fly a quadrotor through a timed race course in
simulation?

## What it does

drone-fly is a research prototype that connects three things:

- **Connectome fetch** — pulls MaleCNS connectivity (neurons and typed synaptic connections)
  from Janelia/Google's neuPrint service.
- **Connectome-seeded controller** — uses that connectivity to shape a trainable PyTorch policy
  network. The wiring is seeded from the fly; the connection strengths are learned by RL. This
  is not a biophysically faithful brain simulation — the data does not provide neuron dynamics.
- **Reinforcement learning** — trains the controller (Stable-Baselines3 PPO) in an open
  Gymnasium quadrotor simulator to chase waypoints through gates, penalized for hitting the
  floor, ceiling, or obstacles and rewarded for the fastest gate-to-gate times.

The connectome runtime is **reused, not built from scratch**: running MaleCNS as a trainable
network is already solved and open-source. drone-fly's real work is the *integration* (wiring an
existing connectome substrate to an existing drone sim) and the *training* (RL for waypoint
racing).

This repository is currently a structure-only scaffold: the pipeline stages exist as packages
with documented responsibilities, but the logic is not implemented yet.

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

# Configure your neuPrint token.
cp .env.example .env   # then edit .env and set NEUPRINT_TOKEN

# Run the checks that exist today.
uv run ruff check .
uv run pytest
```

The `fetch-connectome`, `train`, and `evaluate` commands are planned CLI entry points; their
logic is not implemented yet, so they are not runnable from a fresh clone.

## Project layout

```
src/drone_fly/
  connectome/   fetch MaleCNS connectivity from neuPrint, cache locally
  controller/   build the connectome-seeded PyTorch policy network
  env/          Gymnasium quadrotor racing environment (gates, collision penalties)
  train/        RL training loop (Stable-Baselines3)
  evaluate/     run a trained agent, report gate/lap times vs. a baseline
  cli/          command-line entry points
tests/          pytest suite (currently import smoke tests)
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
- The MaleCNS dataset has its own license and citation terms; this repo does not redistribute
  bulk connectome data — you fetch it at runtime with your own token.

## License

MIT (covers the drone-fly source code only, not the upstream MaleCNS dataset).
