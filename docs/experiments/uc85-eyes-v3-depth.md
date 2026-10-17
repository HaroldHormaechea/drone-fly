# UC-85 — eyes v3: a pretrained monocular-depth backbone for the obstacle-ray estimator

Sim-only (no Liftoff), in-sandbox follow-up to UC-83's negative result. UC-83 showed the eyes
estimator cannot learn to range obstacles from the current 5-map captures even with a classification
head + focal loss + hit-weighted mining (near-obstacle miss stuck ~0.99, hit-head recall 0.027 —
**both** detection and ranging fail), and concluded the ceiling is **data/sensing, not the loss**.
UC-84 attacks that ceiling with *more data* (richer Liftoff captures). UC-85 attacks it with
*architecture*: replace the from-scratch CNN trunk with a **frozen pretrained monocular-depth
backbone**, so the ray head predicts distances from depth-aware features that carry a strong ranging
prior learned from millions of general images — the intuition a few thousand drone frames can't
teach. It is **opt-in / default-off** (the UC-78/83 `EyesNet` stays the default, byte-identical when
unset), is built and measured **on the current 5-map data now** (an independent read on whether a
depth prior helps even with thin data, parallel to the UC-84 capture flights, no Liftoff needed), and
is designed to re-run on UC-84's richer captures when they land. Because the eyes must run inside the
live control budget, UC-85 treats **latency as a first-class gate**: a backbone that improves ray
accuracy but blows the ~25 ms / 40 Hz tick (alongside the gate detector) is recorded as "accuracy-win
but not deployable," mirroring UC-79's diagnostic-not-deployable framing.

## Design

### Opt-in, default-off byte-identical (AC1)

Eyes v3 is a **sibling network**, `EyesDepthNet`, **not** a branch inside `EyesNet`. The default,
depth-off path is therefore byte-identical by construction (locked by `test_eyes_model` state-dict +
forward-output equality, and the UC-78/83 eyes tests still pass). `build_from_meta` dispatches to
`EyesDepthNet` only when `arch.depth_backbone` is set; the depth keys are written to `arch` **only
when enabled**, so a v1/v2 meta is byte-for-byte unchanged and loads as `EyesNet`. `ARCH_VERSION`
stays 2.

* **Backbone (the trunk).** The default is
  `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` — the same Depth-Anything-V2 (metric,
  small) model the live harness already ships (`drone_fly.harness.liftoff.depth`): a DINOv2 ViT-S,
  ~24.8 M params / **~99 MB**, **Apache-2.0**, comfortably under the ~150 MB / ~17 GB-free disk
  bound. Per frame it predicts a depth map; the **K** frames are stacked to **K channels** and fed to
  a small learnable adapter CNN.
* **Frozen + non-registered.** The backbone is held as a **non-registered attribute**
  (`DepthBackbone` is a plain class, not an `nn.Module`). This single decision yields every exclusion
  for free: `count_parameters` / `AdamW(net.parameters())` auto-exclude it (so `MAX_PARAMS = 1.5 M`
  is safe and **train.py's optimizer / budget logic is unchanged**), and `state_dict()` carries only
  the adapter + head — so checkpoints stay small (no ~99 MB bloat) and strict `load_state_dict`
  round-trips **without any backbone keys**. (Locked by a QA test asserting exactly this.)

### Depth features feed the ray head (AC3)

The adapter is the **same stride-2 stage stack** as `EyesNet` (channels 32-48-64-96-128), but with
`k_frames` input channels (the stacked depth maps) instead of `3·k_frames` RGB channels. It produces
the stride-8 / stride-16 maps + global vector that the **unchanged** per-ray `grid_sample` query and
256-128 MLP head consume, so the 6-arg forward IO contract (`frames, uv, rays_cam, in_view, ego,
hist_dt → (free, conf[, hit])`) and the K-frame + ego-motion inputs are all preserved. Looming (an
obstacle's depth map **growing** between the K frames) survives the depth transform: the per-frame
depth maps are standardised **jointly across the K frames** (not per frame), which normalises the
absolute scale the adapter sees while keeping cross-frame relative depth intact.

### Provisioning + offline fallback (AC2)

`drone-fly eyes provision-backbone [--model-id ID]` warms the Hugging Face cache once; later runs
reuse it, and with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` they load from the cache with no
network. `transformers` + `pillow` are imported **lazily** and live **only in the `harness` extra**
(never the default/dev deps), so CI stays network-free. A pure-torch **`tiny-depth`** sentinel
backbone (no transformers, no download) doubles as the offline fallback **and** the hermetic test
backbone.

### Measurement and the bars

eyes v1-baseline (depth off) and eyes v3-depth (`--depth-backbone`) are trained on **one shared
dense dataset** over the same train maps, so the A/B isolates the trunk. Both are evaluated on the
**intact held-out maps** (HangarC03, MinusTwo, ThePit) at a pinned `tol_m` and range, and compared
with `eyes compare` (unchanged — v3 is passed as `--v2`).

* **Accuracy (AC6):** ≥ **30 % relative reduction** in nearest-quarter (0–4 m) **in-view miss** vs
  the measured v1-baseline-on-identical-data, with **clear false-alarm ≤ 0.10**.
* **Latency (AC5):** `eyes bench` reports the v3 estimator's p50/p95/p99 vs the ~25 ms / 40 Hz tick
  (with the gate detector interleaved). **Both** bars must pass to "adopt."

## Commands

The resumable driver `scripts/uc85_eyes_v3_depth.sh [OUT_DIR]` runs the whole pipeline (per-step
markers, `STOP` file, `MAX_HOURS` budget; `UC85_BACKBONE_ID`, `UC85_DATASET`, `UC85_EPOCHS`, … knobs).
Equivalent manual steps:

```sh
# 0. provision the depth backbone once (offline-able afterwards; tiny-depth = no download)
drone-fly eyes provision-backbone --model-id depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf

# 1. one dense dataset over the train maps (+ frame cache)
drone-fly eyes dataset --label-missing --build-cache --out artifacts/uc85/dataset --disk-margin-gb 1

# 2. v1-baseline (default path, depth off) and 3. eyes v3-depth on the SAME dataset
drone-fly eyes train --dataset artifacts/uc85/dataset --out artifacts/uc85/v1 --select last --tol-m 0.5
drone-fly eyes train --dataset artifacts/uc85/dataset --out artifacts/uc85/v3 --select last --tol-m 0.5 \
    --depth-backbone --backbone-id depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf

# 4/5. eval both on the intact held-out split
drone-fly eyes eval --weights artifacts/uc85/v1/eyes.pt --dataset artifacts/uc85/dataset --split held-out --tol-m 0.5 --out artifacts/uc85/eval_v1
drone-fly eyes eval --weights artifacts/uc85/v3/eyes.pt --dataset artifacts/uc85/dataset --split held-out --tol-m 0.5 --out artifacts/uc85/eval_v3

# 6. accuracy verdict (AC6)
drone-fly eyes compare --v1 artifacts/uc85/eval_v1/eval_held-out.json --v2 artifacts/uc85/eval_v3/eval_held-out.json --out artifacts/uc85/compare

# 7. latency gate (AC5): p50/p95/p99 vs the live tick, with the gate detector interleaved
drone-fly eyes bench --weights artifacts/uc85/v3/eyes.pt --with-gate-detector --out artifacts/uc85/bench
```

## Results

Run of 2026-10-17, in-sandbox on the NVIDIA RTX A3000 Laptop GPU (6 GB), reusing the UC-83 dense
5-map dataset (`artifacts/uc83/dataset`); both nets trained 12 epochs, seed 78, tol 0.5 m, evaluated
on the intact held-out split (HangarC03 / MinusTwo / ThePit). Backbone:
`depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf` (DINOv2 ViT-S, 24.78 M frozen params,
~99 MB, Apache-2.0). Sources: `artifacts/uc85/compare/compare.json`, `eval_v{1,3}/eval_held-out.json`,
`artifacts/uc85/bench/bench.json`.

### Ray accuracy: eyes v1-baseline vs eyes v3-depth (held-out, in-view, AC6)

| metric (held-out, in-view) | v1-baseline | v3-depth | rel. change |
|---|---:|---:|---:|
| miss — nearest quarter (0–4 m) | 0.985 | 0.919 | −6.7 % |
| miss — aggregate | 0.943 | 0.927 | −1.7 % |
| false-alarm on clear rays (all) | 0.028 | 0.007 | −74 % |

**Accuracy verdict (AC6): NEGATIVE (miss gate FAIL, FA gate PASS).** The depth prior helps
*directionally* — nearest-quarter in-view miss drops 0.985 → 0.919 and clear false-alarm drops ~4×
(0.028 → 0.007, comfortably under the 0.10 bound) — but the 6.7 % relative nearest-quarter reduction
is far short of the ≥ 30 % bar. On the thin 5-map data the prior is not enough; near-obstacle ranging
is still essentially unsolved (miss ≈ 0.92).

### Latency: eyes v3-depth (AC5)

| stage | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|
| eyes total (incl. depth backbone) | 10.41 | 13.15 | 14.80 |
| + gate detector (tick) | 16.66 | 19.14 | 20.30 |

Budget: ~25 ms (40 Hz), measured on the A3000 GPU. **Within budget: YES** — `eyes_p95_within_budget`
= true (13.15 ms) and `combined_p95_within_budget` = true (tick p95 19.14 ms, incl. the gate
detector). Peak GPU memory: eyes 178 MB, eyes + detector 204 MB. **The expected main risk (a depth
backbone blowing the live tick) did not materialize on GPU** — the frozen Depth-Anything-V2-Small
trunk is comfortably deployable latency-wise; latency is *not* the blocker here.

## Recommendation (decision tree)

**This run landed in the Negative branch → defer to the UC-84 richer-data re-run.** The depth prior
is deployable (both latency bars pass) and directionally helpful (−6.7 % nearest-quarter miss, −74 %
false-alarm), but it does **not** clear the ≥ 30 % accuracy bar on the thin 5-map data. Do **not**
adopt it as the eyes default yet, and do **not** trigger the gated UC-82 re-run (it is gated on
clearing both bars). Notably, latency is *not* the obstacle — the "accuracy-win, not deployable"
branch and its mitigations are unused, because v3-depth already fits the 25 ms / 40 Hz tick on GPU.

Branches considered:

- **Adopt** — v3 clears **both** bars (≥ 30 % nearest-quarter miss reduction, FA ≤ 0.10, p95 within
  budget). Make it the eyes default for the obstacle path and gate the UC-82 obstacle-avoidance
  re-run on it (below). _Not taken: accuracy bar not cleared (6.7 % < 30 %)._
- **Accuracy-win, not deployable** — v3 clears accuracy but **not** latency. Record it as a
  non-deployable accuracy result and apply a **named mitigation**, then re-bench:
  - a smaller / distilled depth backbone (e.g. `Intel/dpt-swinv2-tiny-256`);
  - lower input resolution to the backbone;
  - a **features-only** adapter (tap an intermediate backbone stage instead of the full decoded depth
    map), cutting the decode cost.

  _Not taken: latency already passes (eyes p95 13.15 ms, tick p95 19.14 ms ≤ 25 ms); accuracy, not
  latency, is the gap._
- **Negative (depth prior insufficient on this data)** — v3 does **not** clear accuracy. This is an
  acceptable, informative outcome: the depth prior alone does not overcome the thin 5-map ceiling.
  **Defer to the UC-84 richer-data re-run** (below) rather than treating it as a failure. _← **This
  run.** The architecture lever is in place and cheap at inference; the binding constraint is data._

## Re-run hooks

- **On UC-84's richer captures.** When the UC-84 flights land and `eyes ingest` confirms a higher
  in-view hit-ray fraction, rebuild the dense dataset over the new captures and re-run the driver
  unchanged (`scripts/uc85_eyes_v3_depth.sh`) — the v1-vs-v3 A/B and both bars apply as-is. A depth
  prior **plus** more close-obstacle frames is the strongest combination available.
- **Gated UC-82 obstacle-avoidance re-run.** Only if v3 clears **both** bars: re-derive the
  obstacle-ray noise model from v3's held-out eval (`eyes noise-model --from-eval
  artifacts/uc85/eval_v3/eval_held-out.json`) and re-run the UC-82 rays-vs-blank comparison (reusing
  UC-82's blanked checkpoint, noise-independent) to test whether better eyes finally reduce collisions
  vs blanking — the question UC-82/UC-83 left open. Do **not** run this on a v3 that fails the
  accuracy bar: at ~0.99 near-miss the rays are no better than v1's and would reproduce UC-82's
  negative.

## Caveats

- **Low-resolution depth (192×80).** The dense cache is 192×80 (like-for-like with v1 for the AC6
  A/B); the backbone is resized up to its patch-multiple internally, but detail is bounded by the
  cache. A higher-resolution cache is a documented follow-up (it costs disk and re-labelling).
- **Cross-frame depth scale-inconsistency.** Monocular depth is only up-to-scale per frame; the K
  frames can carry slightly inconsistent absolute scales. Joint-across-K standardisation mitigates
  this for the looming cue, but residual scale drift is a known source of noise and a reason a depth
  prior may help ranging less than hoped on moving sequences.
- **Outdoor-metric model vs indoor maps.** The default backbone is the **outdoor** metric variant
  (the one already vendored by the harness); the Liftoff maps are mixed indoor/hangar. The absolute
  metric calibration may be off-domain — the adapter is trained to correct for it, but a matched
  (indoor) variant is a candidate if the result is negative.
