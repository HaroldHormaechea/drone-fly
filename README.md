# drone-fly

Can the wiring diagram of a real fruit-fly brain (the **MaleCNS connectome**) seed a neural
controller that learns to fly a quadrotor through procedurally-random gate courses in simulation?

## What it is (current approach)

The controller is the real 25,627-neuron MaleCNS (**K1**) connectome (~3.86M synapses) wired as a
**frozen recurrent reservoir**, plus a small (~170k-parameter) trained **readout**. The connectome is
fixed and load-bearing; only the readout learns. Observations are projected into the frozen brain,
propagated through its real synapses, tapped from 1024 neurons, and read out into collective thrust +
body rates (CTBR). A **playback viewer** (`viz/`) shows neuron activation as an MRI-style heatmap.

Trained flight models live under `training/NN-<slug>/` (numbered in creation order) as a stripped
`model_readout.pt`. They are trained by the hand-rolled GPU PPO pipeline in **`gpu_prototype/`** and
deployed via **`gpu_prototype/inference.py`**.

> **New here?** Start with [docs/CONNECTOME_RESERVOIR.md](docs/CONNECTOME_RESERVOIR.md) (ground-up
> explainer). For the index of trained models and results see [docs/FLIGHT_MODELS.md](docs/FLIGHT_MODELS.md)
> and [training/EXPERIMENTS.md](training/EXPERIMENTS.md). To run/deploy a model outside the training
> loop (e.g. a live-flight harness), the frozen interface is [docs/INFERENCE_CONTRACT.md](docs/INFERENCE_CONTRACT.md).

## Repository layout

| path | what |
|------|------|
| `gpu_prototype/` | **the active lineage** — reservoir models, GPU PPO training, envs, audits, `inference.py` |
| `training/NN-<slug>/` | committed flight models (`model_readout.pt`) + viewer recordings, numbered by creation |
| `src/drone_fly/` | shared library used by the reservoir path: `connectome` (loads K1), `controller.actor` (the frozen sparse body), `env.config` (course dataclasses), `record.recorder` (viewer recordings) |
| `artifacts/pruned/k1/` | the frozen K1 connectome (neuron ids, edges, signs) — shared by every model |
| `viz/` | the browser playback viewer (`viewer.html` + `viewer.js`) |
| `docs/` | explainers, the inference contract, flight-model index |

## Run / deploy a trained model

Deployment needs only `torch`, NumPy, `artifacts/pruned/k1/`, and `drone_fly.connectome`:

```python
from inference import load_policy, init_state, plant_step, make_obs   # gpu_prototype/inference.py
policy = load_policy("training/03-oval-lap-flight/model_readout.pt", extra_dim=12)
s = init_state(pos=(0, 0, 0.12))
for _ in range(500):
    obs = make_obs(s, target=(4.0, 0.0, 1.2))
    s = plant_step(s, policy.act(obs, extra=None))
```

Obstacle models (`extra_dim=70`) additionally consume a 40-ray forward scan — see `inference.raycast()`
and [docs/INFERENCE_CONTRACT.md](docs/INFERENCE_CONTRACT.md).

## Audit a model (honesty-gated)

A completion % only counts if the flight is real (no barging through obstacles, no hovering, no
tumbling). From `gpu_prototype/`:

```
PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 \
  python verify_proc.py <model.pt> 1024     # deterministic completion + cheat checks
  python diagnose_proc.py <model.pt> 1024    # failure breakdown
```

`verify_committed.py` reloads each committed `model_readout.pt` into a fresh actor to prove the on-disk
artifact (not just the live model) flies.

## Connectome data

The committed slice `artifacts/pruned/k1/` is all the deployed models need. The full MaleCNS matrix
(CC-BY; cite MaleCNS / `connectome_data_prep`) can be fetched and pruned via `drone_fly.connectome`
(`fetch` auto-downloads into `DRONE_FLY_CONNECTOME_DIR`, else `data/connectome/`).

## History

An earlier lineage trained the connectome as a *trainable* layer via a Stable-Baselines3 PPO loop over a
gymnasium racing env, with a desktop app and pybullet backend. That approach was **superseded by the
frozen-reservoir method and removed in the 2026-10 cleanup** (SB3/gymnasium/pybullet code, the `app/`,
training `configs/`, and their tests). The reservoir pipeline in `gpu_prototype/` is the project today.
