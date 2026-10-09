# Oval lap — the K1 connectome flying a closed turning loop

Ten deterministic recordings of the real K1 connectome (25,627 neurons, ~3.86M synapses, real MaleCNS
weights) flying a **closed oval lap** in acro (rate-mode) flight: 6 gates around a horizontal ellipse
with same-direction (~60°) turns, closing back at the start. **99% deterministic completion** (127/128
in eval), gates 6/6, 99% upright, mean tilt ~25° (banking into the turns, controlled).

## Architecture: augmented reservoir (`[brain-tap ‖ flight-obs]`)

The straight-course policy (`training/k1-gate-flight/`) runs the K1 as a *pure* frozen reservoir —
readout sees only the brain. That saturates on a straight line but hits a **deterministic gap** on a
turning loop: the frozen brain features alone don't give the *mean* policy enough control for the hard
return-leg turns, so it relied on exploration noise (100% stochastic but ~0% deterministic; an MLP
does 100% deterministic, proving the loop is flyable).

Fix, used here and on all harder courses: the K1 body stays **frozen** and still processes every
observation (a load-bearing feature generator — proven at 100% on the straight course), but the small
readout *also* sees the raw 12-dim flight observation: readout input = `[LayerNorm(brain-tap 1024) ‖
obs 12]`. Combined with **exploration annealing** (force the Gaussian policy's std down over the back
40% of training so corrective control must live in the mean, not the sampling noise), the deterministic
policy jumps from ~0% to **99%**.

Trainer `gpu_prototype/train_aug_course.py lap` (actor+trainer in `reservoir_aug.py`), eval
`eval_aug.py`, recorder `record_aug.py`. The committed `model_readout.pt` (≈600 KB) holds only the
trainable readout; the frozen K1 body is reconstructed from `artifacts/pruned/k1/` at load time.

## Activations

As with the straight course, each `episode_<n>.json` stores all 25,627 neuron activations per frame
(viewer brain overlay) + action + position + target gate + anatomical neuron metadata. The frozen K1's
raw activation is near-zero (random input projection), so a per-episode scalar visibility gain (~38×)
is applied — pattern unchanged, only scaled for visibility.

## Course

Oval: 6 gates on an ellipse (center x=3, radii 3.0×2.5, height 1.2), aperture 1.0, lap closes at the
start point ~1 m below gate 0. Per-episode spawn noise varies the trajectory. ~6–7 s per lap.
