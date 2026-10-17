# UC-84 — Liftoff obstacle-capture: richer-data retrain result

Does tripling the eyes estimator's **in-view hit-ray fraction** (~5 % → ≥ 15 %) with obstacle-dense
Liftoff captures clear the AC4 near-miss bar that UC-83's training-time levers could not? This doc
records the achieved capture density and the v2-on-new-data vs UC-83 v1/v2-on-old-data comparison.

> **Status: skeleton — numbers pending the capture drop + retrain.** AC1/AC2 (protocol + tooling)
> and AC3/AC4/AC6 (ingest, driver, tests) are in place. The flights are user-side (Liftoff is
> Windows-only); once the captures are dropped in and validated, the sim-only retrain
> (`scripts/uc84_obstacle_capture.sh`) runs in-sandbox and the tables below are filled from its
> `artifacts/uc84/` outputs. Placeholders are marked _TBD_.

## Method

- **Capture protocol:** `docs/experiments/uc84-capture-protocol.md` — fly the 70-input
  `training/11-count-curriculum` model (teacher backup) through obstacle-dense train-map courses via
  the `liftoff-episodes` skill; assemble with the off-branch UC-71 tool; drop into the captures root.
  Val + held-out maps are frozen (no new captures) so the held-out comparison stays valid.
- **Density gate (AC3):** `drone-fly eyes ingest --label-missing` measures the in-view hit-ray
  fraction (`hit[in_view].mean()`, the shared `in_view_hit_counts` helper) per map and overall, and
  flags `below_target` until it clears 15 %.
- **Retrain/compare (AC4):** `scripts/uc84_obstacle_capture.sh artifacts/uc84` gates on the fraction,
  then delegates **byte-for-byte** to the tested UC-83 v2 recipe — one dense dataset over the
  enlarged train captures → train v1-baseline + v2 (`--hit-head --focal --hit-sampler`) on the same
  dataset → `eyes eval` both on the intact held-out split → `eyes compare`. tol 0.5 training m, same
  as UC-83, so the numbers are directly comparable.
- **Provenance:** the driver logs the frozen held-out/val map set + the catalogue sha256
  (`artifacts/uc84/00_provenance.txt`) so a shifted baseline is detectable.

## Capture density (AC3)

From `artifacts/uc84/ingest/ingest.json` after the drop:

| | in-view hit-ray fraction | maps | labelled frames |
|---|---:|---:|---:|
| UC-83 baseline (5 train maps) | ~0.05 | 5 | _TBD_ |
| UC-84 enlarged train set | _TBD_ | _TBD_ | _TBD_ |
| target | ≥ 0.15 | — | — |

## Results

Run _TBD (date, RTX A3000)_. v1-baseline and eyes v2 trained on one shared dense dataset over the
**enlarged** train maps; both evaluated on the **intact** held-out split (HangarC03, MinusTwo, ThePit;
tol 0.5 training m). Numbers from `artifacts/uc84/compare/compare.json` +
`artifacts/uc84/eval_v{1,2}/eval_held-out.json`, with the UC-83 columns from
`artifacts/uc83/` for reference.

### Ray accuracy: v2-on-new-data vs UC-83 (held-out, in-view, AC4)

| metric (held-out, in-view) | UC-83 v1 (old data) | UC-83 v2 (old data) | UC-84 v2 (new data) | AC4 bar |
|---|---:|---:|---:|---:|
| miss — nearest quarter (0–4 m) | 0.983 | 0.989 | _TBD_ | ≥ 30 % rel. reduction |
| miss — aggregate | 0.942 | 0.928 | _TBD_ | — |
| false-alarm on clear rays | 0.019 | 0.055 | _TBD_ | ≤ 0.10 |
| hit-head recall / F1 (diagnostic, in-view) | n/a | recall 0.027 / F1 0.050 | _TBD_ | — |

**AC4 verdict: _TBD_.** Pass = nearest-quarter in-view miss falls ≥ 30 % relative to the UC-84 v1
baseline **and** false-alarm-on-clear ≤ 0.10 (`eyes compare` writes `passed`).

## Next steps (gated on the AC4 verdict)

- **If AC4 clears:** the richer captures gave the estimator the close-obstacle signal it lacked.
  Re-run the UC-82 obstacle-avoidance rays-vs-blank comparison with the new eyes to confirm better
  rays actually improve avoidance — opt in with `UC83_RUN_AC5=1` on the driver (re-derives the
  ray-noise model from the v2 held-out eval and retrains only the rays arm vs the reused UC-82
  blanked control). Record the collision comparison here.
- **If AC4 does not clear:** record the negative — ≥ 15 % in-view hit density was still insufficient,
  which points at **monocular ranging genuinely needing depth features**, not more data. The named
  follow-up is **UC-85: a pretrained depth-model backbone** (UC-78 lever 3) as the estimator trunk.
  Do **not** re-run the UC-82 comparison with an estimator that hasn't cleared the near-miss bar.
