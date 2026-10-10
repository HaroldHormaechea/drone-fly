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

## Live harness

`drone_fly.harness` runs a trained model against an external real-time flight system instead of
the training simulator, using the model I/O from [docs/INFERENCE_CONTRACT.md](docs/INFERENCE_CONTRACT.md)
exactly as written. A harness is built from separate parts: a state source (telemetry), a frame
source (screen capture), perception estimators (gate senses, free space / 40-ray scan), the contract
observation assembler, and an action sink. A fixed-rate runner (50 Hz by default) drives them, records
everything and fails safe on stale telemetry, a lost gate, stale perception, any exception or the
abort hotkey (F12; Ctrl+C always works): it centres the rate sticks and cuts the throttle. Layering,
frame conventions and how to add a harness or estimator:
[docs/adr/0002-live-harness-layering.md](docs/adr/0002-live-harness-layering.md).

```
drone-fly harness list                     # registered harnesses/parts and whether their backends are installed
drone-fly harness run --harness sim-loopback --checkpoint <model_readout.pt> --extra-dim 70
drone-fly harness run --harness liftoff --checkpoint training/07-proc-course-flight/model_readout.pt \
    --extra-dim 30 --config my_liftoff.yaml [--dry-run] [--duration 60]
drone-fly harness eval --session artifacts/harness/<run> --track <track.xml> --race <race.xml> --config my_liftoff.yaml
```

- **sim-loopback** flies the contract's reference `plant_step` through the harness interfaces, with
  ground-truth gates and the contract `raycast()`. It checks that the harness reproduces the
  reference inference path.
- **liftoff** (5-inch FPV) reads Liftoff's UDP telemetry, detects gates on screen with OpenCV
  (PnP against a configured nominal gate size), and drives a ViGEm virtual Xbox 360 pad through the
  inverse of your Liftoff rate profile. Settings for each game and airframe (telemetry port and
  format, FOV, uptilt, gate size and colour, rates, hover stick, `world_scale`) live in YAML. Your
  `--config` is overlaid on the packaged default `src/drone_fly/harness/configs/liftoff_5inch.yaml`;
  unknown keys are rejected. The values marked `CALIBRATE` there are assumptions to check against
  your setup.
- **Obstacle models** (`extra_dim=70`, e.g. `training/11-count-curriculum/model_readout.pt`, or the
  older `training/10-proc-obstacle-avoid/model_readout.pt`) need the 40-ray scan. Set
  `parts.free_space_estimator: depth`: a monocular depth network (Depth-Anything-V2 metric, downloaded
  on first use) is sampled along the contract's velocity-aligned ray cone. With the default `none`
  estimator the runner refuses to start a 70-dim model and names the missing `raycast` block. A
  12/30-dim model can fly an obstacle-free course with `free_space.assume_no_obstacles: true`.
- Liftoff track/race XMLs are **never** used at runtime. Only `harness eval` reads them, as ground
  truth for scoring the gate and depth estimators on a recorded run.
- The models were trained on a 32 g whoop, so flying a 5-inch Liftoff quad is a warm start, not a
  guaranteed zero-shot flight. The policy also does not keep the camera pointed where it flies (no yaw
  shaping); see the ADR.
- Runs are recorded under `artifacts/harness/<harness>-<UTC>/`: config, checkpoint sha256, raw
  telemetry, sampled frames, estimator outputs, obs/extra vectors, actions and per-stage latencies.
  Recording into `training/` is refused.
- Paths: `--gpu-prototype-dir` / `DRONE_FLY_GPU_PROTOTYPE` (default `gpu_prototype/`) and
  `--k1-path` / `DRONE_FLY_K1_PATH` (default `artifacts/pruned/k1/`).
- Exit codes: 0 ok, 2 config error, 3 missing optional dependency, 4 perception coverage refusal,
  5 fail-safe stop.

The core package and CI need none of the harness backends. Install them with
`uv sync --extra harness` (OpenCV, mss/dxcam, transformers, vgamepad). **On Windows this also launches
the ViGEmBus driver installer** (a UAC prompt), pulled in by `vgamepad`. The driver is required for
live Liftoff runs. Run the test suite with `uv run --extra dev pytest`.

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
