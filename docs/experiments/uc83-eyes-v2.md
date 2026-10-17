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

> Filled by the orchestrator from the GPU run (`scripts/uc83_eyes_v2.sh`). The tables below are the
> shape to fill from `compare.json` / `eval_held-out.json` (and, if AC5 ran, `08_ac5_eval.json`).

### Ray accuracy: eyes v1-baseline vs eyes v2 (held-out, AC3/AC4)

| metric (held-out, in-view) | v1-baseline | eyes v2 | rel. change |
|---|---:|---:|---:|
| miss — nearest quarter (0–4 m) | _tbd_ | _tbd_ | _tbd_ |
| miss — aggregate | _tbd_ | _tbd_ | _tbd_ |
| false-alarm on clear rays | _tbd_ | _tbd_ | _tbd_ |
| hit-head recall / F1 (diagnostic) | n/a | _tbd_ | — |

Train hit-ray fraction (manifest `mining`): v1 ~_tbd_ (vs UC-78's ~5 %).

**AC4 verdict:** _PASS / NEGATIVE_ (≥30 % nearest-quarter in-view miss reduction and clear
false-alarm ≤0.10). _tbd_

### Collision share: rays-v2 vs blanked (AC5, only if AC4 passed)

| arm | collision share | completion | vs blanked |
|---|---:|---:|---:|
| rays (eyes-v2 noise) | _tbd_ | _tbd_ | _tbd_ |
| blanked (reused UC-82 control) | _tbd_ | _tbd_ | — |

## Recommendation

> Filled after the run. Expected branches:
>
> * **AC4 passes and AC5 shows rays-v2 < blanked:** better eyes help — adopt eyes v2 and the v2 noise
>   model for the obstacle-avoidance thread; UC-82's negative result is reversed.
> * **AC4 passes but AC5 shows no collision gain:** the estimator improved but the connectome still
>   does not benefit — next lever is a richer ray encoding or the depth backbone.
> * **AC4 fails:** the in-sandbox levers are not enough against the data ceiling — the named
>   follow-up is new close-to-obstacle capture flights (needs Liftoff) or a depth-model backbone
>   (UC-78 lever 3). Documented negative result; no AC5.
>   * **If the hit head reaches GOOD recall but the free-distance miss still does not clear the
>     bar, make that dissociation the headline:** it localizes the ceiling to monocular *ranging*
>     (distance) rather than *detection*. That sharpens the follow-up toward depth information
>     specifically — close-to-obstacle captures or a depth backbone — over just "collect more
>     data," since detection is already solved and more same-kind frames would not fix ranging.
