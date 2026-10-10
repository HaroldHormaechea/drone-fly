# Figure-8 / chicane — the K1 connectome flying a randomized both-direction track

Twelve deterministic recordings of the real K1 connectome (25,627 neurons, ~3.86M synapses, real
MaleCNS weights) flying a **randomized realistic lap** in acro (rate-mode) flight. Each episode is a
different track, sampled 50/50 from two families, so the policy must handle **turns in both
directions** — not a single-direction oval.

- **figure-8** — a Gerono lemniscate with a *height-separated* crossover (one pass high, one low → real
  3D flying): the drone banks one way on the first lobe, the other way on the second.
- **chicane** — a closed oval with alternating lateral jogs → the drone weaves left-right-left.

Both families are also parameter-randomized (size, crossover separation, jog amplitude) every episode.

## Result (deterministic eval, 256 drones)

| | completion | tilt | upright |
|---|---|---|---|
| overall | **247/256 (96%)**, gates 8/8 | mean 12°, max 44° | 100% |
| figure-8 | **139/139 (100%)** | | |
| chicane | **108/117 (92%)** | | |

Figure-8 is flown perfectly; the chicane's tight alternating turns are slightly harder (92%). These 12
recordings are a 6/6 mix of both families.

## Architecture

Same augmented frozen-K1 reservoir as `training/oval-lap-flight/`: the K1 body stays **frozen** and
processes every flight observation (a load-bearing feature generator, proven at 100% on the straight
course), and the small readout sees `[LayerNorm(brain-tap 1024) ‖ raw-obs 12]`, trained with
exploration annealing. Trainer `gpu_prototype/train_aug_course.py track`; env `gates_track.py` (per-env
gate layouts so different drones fly different tracks simultaneously). `model_readout.pt` (≈600 KB) is
the trainable readout; the frozen K1 body reconstructs from `artifacts/pruned/k1/` at load.

Each `episode_<n>.json` stores all 25,627 neuron activations per frame (per-episode visibility gain
applied, pattern real) + action + position + target gate + anatomical metadata, and the episode's own
randomized gate layout, so the app draws the exact track flown.
