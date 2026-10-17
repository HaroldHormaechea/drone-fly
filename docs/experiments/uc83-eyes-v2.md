# UC-83 — eyes v2: cut the obstacle-ray miss rate

Sim-only (no Liftoff), in-sandbox follow-up to UC-82's negative result. UC-82 showed that feeding the
UC-78 "eyes v1" obstacle rays to the 70-input connectome made avoidance **worse** than blanking them
(collision 57 % vs 35 %), because eyes v1 misses ~96 % of obstacles (≈99 % of the nearest quarter,
0–4 m — the ones that matter) and collapses to "clear" under the ~5 % hit-ray imbalance. UC-82's
recommendation and UC-78's own "likely levers" both point at **the estimator, not more avoidance
training**. UC-83 builds eyes v2 with the two levers that are fully doable in the GPU sandbox on the
existing ~23 GB of UC-71 captures (no new capture flights): a **hit/no-hit classification head
trained with focal loss** and **denser, hit-weighted mining**. A depth backbone (UC-78 lever 3) is a
named follow-up, not built here.

## Design

### Opt-in, default-off byte-identical (AC1)

All v2 levers are opt-in; with every new flag unset, `eyes train` and the model are **byte-identical
to v1** (locked by `test_eyes_model` state-dict + forward-output equality, and the UC-78 eyes tests
still pass). `ARCH_VERSION` stays 2 — the base architecture is unchanged.

* **Classification head.** `EyesNet(hit_head=True)` adds a per-ray hit/no-hit logit from the same
  query features as the free/confidence head. It is constructed **last and only when enabled**, so
  every base-parameter RNG draw is identical to v1; a default-off net is bitwise-equal to a v1 net.
  `forward` returns `(free, conf)` with the head off and `(free, conf, hit_logit)` with it on. The
  weights meta carries an optional `arch.hit_head` (absent ⇒ false; v1 metas load unchanged).
* **Focal loss.** `eyes train --hit-head --focal` adds an α-balanced focal term (α=0.25, γ=2.0) on
  the hit logit, computed in float32 so it is safe under fp16 AMP. Focal down-weights easy "clear"
  rays by `(1 - p_t)^γ` and class-balances with α, so the head does not collapse to "always clear".

### Denser, hit-weighted mining (AC2)

The `--stride-s` lever is **de-scoped**: in `build_dataset` the sample count is fixed by the captured
frames, and a smaller history stride is relabel-coupled and only *reduces* samples. Instead:

* **Hit-weighted sampler** (`eyes train --hit-sampler`, the primary lever, no disk cost): each epoch
  resamples frame indices with replacement, weighting hit-bearing frames by `--hit-sample-w`
  (default 8). The epoch size is unchanged; resume stays deterministic (per-epoch reseed).
* **Hit-biased frame filter** (`eyes dataset --hit-frac F`, opt-in fallback, default off): keeps all
  hit-bearing train frames and subsamples clear ones so the **train split** reaches ~`F` hit-bearing
  frames (val / held-out untouched, so evaluation stays comparable). It never duplicates frames. The
  manifest's `mining` block records the target, seed, and the resulting train hit-ray fraction.

### Measurement and the AC4 bar

eyes v1-baseline (head off) and eyes v2 (head on + focal + hit-sampler) are trained on **one dense
dataset over the same train maps**, so the A/B isolates the v2 levers. Both are evaluated on the
**intact held-out maps** (HangarC03, MinusTwo, ThePit) at a pinned `tol_m` and range, producing the
UC-78 metrics (per-ray miss, miss-in-view, miss nearest-quarter 0–4 m, false-alarm on clear rays).

`eyes compare --v1 … --v2 …` makes the verdict mechanical. **AC4 gate (set + justified here):**

* **≥ 30 % relative reduction** in nearest-quarter (0–4 m) **in-view miss** vs the measured
  v1-baseline-on-identical-data (not UC-78's published numbers — UC-83 has only 5 train maps on disk
  vs UC-78's 7, so a fresh baseline is the fair control). UC-78's 0.99 nearest-quarter and the ≤0.69
  absolute milestone are reported as secondary anchors.
* **clear false-alarm ≤ 0.10** on v2 (pushing miss down must not blow up false alarms, which UC-82
  showed also mislead the connectome). Rebalancing is dialled back if the FA bound is breached.

The hit head's own recall / precision / F1 is reported as a **diagnostic** (`hit_head` section of the
eval report); the AC4 gate is measured on the `free` distance output for both v1 and v2 so the
comparison is like-for-like. The levers that move the free-distance miss are the hit-weighted sampler
+ hit loss weight + the focal-regularized shared trunk.

### Conditional: re-derive noise + re-run UC-82 (AC5)

Only if AC4 passes: re-derive the obstacle-ray noise model from v2's held-out error
(`eyes noise-model --from-eval <v2 run>/eval_held-out.json`) and re-run the UC-82 rays-vs-blank
comparison with the v2 noise on the same held-out proc courses + seed. This is the dominant compute
(a 5M-step retrain), so it is opt-in (`UC83_RUN_AC5=1`) and **reuses UC-82's blanked checkpoint**
(noise-independent) — only the rays arm retrains, via the `uc83-obstacle-rays-v2` config (identical
to `uc82-obstacle-rays` except it injects `artifacts/eyes/ray-noise-v2.yaml`). The headline is
whether better eyes now reduce collisions vs blanked, reversing UC-82's −62 %.

## Commands

The resumable driver `scripts/uc83_eyes_v2.sh [OUT_DIR]` runs the whole pipeline (per-step markers,
`STOP` file, `MAX_HOURS` budget). Equivalent manual steps:

```sh
# 1. one dense dataset over the train maps (+ frame cache); optional hit-biased filter
drone-fly eyes dataset --label-missing --build-cache --out artifacts/uc83/dataset        # [--hit-frac 0.25]

# 2. v1-baseline (default path, head off) and 3. eyes v2 on the SAME dataset
drone-fly eyes train --dataset artifacts/uc83/dataset --out artifacts/uc83/v1 --select last --tol-m 0.5
drone-fly eyes train --dataset artifacts/uc83/dataset --out artifacts/uc83/v2 --select last --tol-m 0.5 \
    --hit-head --focal --hit-sampler

# 4/5. eval both on the intact held-out split
drone-fly eyes eval --weights artifacts/uc83/v1/eyes.pt --dataset artifacts/uc83/dataset --split held-out --tol-m 0.5 --out artifacts/uc83/eval_v1
drone-fly eyes eval --weights artifacts/uc83/v2/eyes.pt --dataset artifacts/uc83/dataset --split held-out --tol-m 0.5 --out artifacts/uc83/eval_v2

# 6. AC4 verdict
drone-fly eyes compare --v1 artifacts/uc83/eval_v1/eval_held-out.json --v2 artifacts/uc83/eval_v2/eval_held-out.json --out artifacts/uc83/compare

# 7/8. conditional AC5 (only if AC4 passed): re-derive noise + re-run UC-82 rays-vs-blank
drone-fly eyes noise-model --from-eval artifacts/uc83/eval_v2/eval_held-out.json --out artifacts/eyes/ray-noise-v2.yaml
drone-fly train run uc83-obstacle-rays-v2 --output-root artifacts/uc83/ac5/rays-v2
drone-fly train proc-eval --checkpoint <rays-v2 ckpt> --config uc83-obstacle-rays-v2 \
    --ray-noise artifacts/eyes/ray-noise-v2.yaml --blank-checkpoint <reused UC-82 blank ckpt> \
    --blank-config uc82-obstacle-blank --episodes 256 --seed 1234 --out artifacts/uc83/08_ac5_eval.json
```

## Results

Run 2026-10-17, RTX A3000. v1-baseline and eyes v2 trained on one shared dense dataset over the 5
train maps on disk; both evaluated on the intact held-out split (HangarC03, MinusTwo, ThePit; tol
0.5 training m). Numbers from `artifacts/uc83/compare/compare.json` + `eval_v{1,2}/eval_held-out.json`.

### Ray accuracy: eyes v1-baseline vs eyes v2 (held-out, in-view, AC3/AC4)

| metric (held-out, in-view) | v1-baseline | eyes v2 | rel. change |
|---|---:|---:|---:|
| miss — nearest quarter (0–4 m) | 0.983 | 0.989 | **+0.6 %** (worse) |
| miss — aggregate | 0.942 | 0.928 | −1.6 % |
| false-alarm on clear rays | 0.019 | 0.055 | +0.036 abs |
| hit-head recall / F1 (diagnostic, in-view) | n/a | **recall 0.027 / F1 0.050** | — |

**AC4 verdict: NEGATIVE.** Nearest-quarter in-view miss did not fall ≥30 % (it rose 0.6 %); the
false-alarm bound held (0.055 ≤ 0.10). The aggregate in-view miss improved only marginally
(0.942 → 0.928, −1.6 %) at the cost of ~3× the false-alarm rate. AC5 was **not** run (gated on AC4).

## Result (the headline)

**The imbalance-correction levers are not the fix — the ceiling is data/sensing, not loss or
sampling.** The classification head + focal loss + hit-weighted mining did essentially nothing for
near-obstacle ranging (nearest-quarter miss stayed ~0.98–0.99, as in v1 and UC-78). Critically, the
hit head itself **also collapsed** — in-view recall 0.027 (overall 0.014): it almost never fires even
as a *detector*. So this is **not** the hoped-for detection-works / ranging-fails dissociation; **both
detection and ranging fail** at this data scale. Focal loss + oversampling can rebalance a classifier
only when the signal to separate the classes is present; here the model cannot tell hit from clear
from these frames at all, so correcting the class balance moves nothing. This is a stronger,
cleaner version of UC-78's "thin data" diagnosis: the bottleneck is the **information in the captures**
(5 train maps, monocular, few close-obstacle frames), not the training objective.

## Recommendation

- **Stop iterating on the estimator's loss/architecture against the current captures.** v2's levers
  (and, by extension, further loss/sampling tuning) are exhausted — they cannot manufacture a signal
  the data does not contain. The hit head's ~0.03 recall shows the model can't even *detect*
  obstacles here, let alone range them.
- **The lever is data/sensing, specifically close-range obstacle information:**
  1. **New capture flights close to obstacles** (UC-78 lever 1) — needs Liftoff, so it is the first
     out-of-sandbox follow-up: fly the 70-input model (or manual) deliberately near obstacles across
     more maps to get many near-hit frames, then rebuild the dense set and retrain v2's recipe (which
     is already in place and tested).
  2. **A depth-model backbone** (UC-78 lever 3) — pretrained monocular-depth features as the trunk,
     which inject the ranging prior the raw frames lack; a larger lift but the only in-principle way
     to range monocularly from limited data.
- **Do not re-run the UC-82 obstacle-avoidance comparison** with v2 — at ~0.99 near-miss the v2 rays
  are no better than v1's and would reproduce UC-82's negative (rays worse than blanked). Gate that
  re-run on an estimator that actually clears the AC4 near-miss bar.
- The UC-83 machinery (classification head, focal loss, hit-weighted sampler/mining, `eyes compare`,
  the resumable driver) is kept and tested — ready to apply the moment better capture data exists.
