# proc-obstacle-avoid — first honest random-obstacle avoider

The K1 connectome flying **procedurally-random gate courses with on-path obstacles** and genuinely
weaving around them — the first model to beat the long-standing ~5% deterministic wall on this task.
See [`docs/FLIGHT_MODELS.md`](../../../docs/FLIGHT_MODELS.md) for the full diagnostic journey.

## Result (deterministic, 256 random courses)

- **33% completion, 0% barging** among completed (0.26 m min clearance, 2.3 m/s, 21° tilt — honest,
  aggressive flight; it steers around obstacles, it does not plow through or hover).
- Same policy, same gate courses, **obstacles off: 90%** — so the obstacle tax is ~57 pts and it captures
  ~37% of its own achievable. Courses are ~97% solvable (geometric audit), so the remaining gap is headroom,
  not unsolvable courses.
- The deterministic mean **climbed** through annealing (5% → 34% → 39% → 33%) rather than collapsing — the
  signature of a policy that actually solved the task, not one carried by exploration noise.

## Why it works (the recipe)

The earlier failures (raycasts into the readout/brain, position curriculum, GRU memory) all hit the same
wall because the real problem was the **reward**, not perception or architecture:
1. **Warm-start aggressive** from the 96% obstacle-free flier → starts at ~21° tilt, never falls into the
   timid near-hover basin the terminal crash penalty otherwise forces.
2. **Potential-based clearance shaping** (`PROC_SHAPING=1`): a dense, policy-invariant reward on the change
   in clearance to the nearest in-path obstacle — a gradient toward *steer around*, not just *don't crash*.
3. Terminal contact (barging impossible) + position curriculum (obstacles slide on) + mildly-stochastic
   anneal (logstd_final −1.0).

## Model / reproduce

`reservoir_aug.py` → `K1ReservoirAug(extra_dim=70)`: frozen K1 brain (3.86M real edges) + a readout over
`[brain tap 1024 || extra senses incl 40-ray cone]`. `model_readout.pt` is the stripped readout (the sparse
layer rebuilds from `artifacts/pruned/k1/`; load with `strict=False`).

```bash
cd gpu_prototype
# train:
PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 python train_shaping.py out.pt 14e6
# audit (deterministic, barging/hover honesty checks):
PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 python verify_proc.py \
  ../training/proc-obstacle-avoid/model_readout.pt 256
# re-record (gzip, viewer.js gunzips in-browser):
REC_STRIDE=2 PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 python record_aug.py \
  proc ../training/proc-obstacle-avoid/model_readout.pt proc-obstacle-avoid 14
```

Recordings are `episode_*.json.gz` (gzipped; open in the desktop app / `viz/viewer.js`).
