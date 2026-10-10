# 11 — count-curriculum (random obstacle avoider, 67% honest)

The best random-obstacle avoider to date: **67% honest deterministic completion** (seeded 1024 courses,
`AUDIT_SEED=1234`), **0% barging**, 2.51 m/s mean speed, 0.29–0.30 m min obstacle clearance. Nearly double
the previous best (`10-proc-obstacle-avoid`, 33–36%) and +28 points over its own warm-start baseline (39%).

## What changed vs. the previous avoider

An **obstacle-count curriculum**: active obstacles grow 1 → N over the first half of training
(`count_curr_frac=0.5`, env `_omask()` / `obs_count_curr`). Phase-0 diagnosis showed the wall was
*multi-obstacle threading* — the model handled a single in-path obstacle but fatally clipped one while
dodging another when 2+ were present. Teaching it one obstacle first, then progressively more, recovered
28 of the 51-point "obstacle tax" (gap to the 90% obstacle-off ceiling).

Everything else is the proven recipe: potential-based clearance shaping (`PROC_SHAPING`), terminal contact
(`PROC_OBS_TERMINAL` — barging is impossible, any touch = crash), warm aggressive start, mildly-stochastic
anneal. Trained via `gpu_prototype/exp11_count-curriculum.py`.

## Negative results (same family, did not beat this)

- **gap-steering reward** alone (`PROC_GAP`): 58% honest — helps, but less than count.
- **count + gap combined**: 64% — the gains do NOT stack (both target multi-obstacle threading).

## Files

- `model_readout.pt` — stripped deployable readout (170k params; keeps `body.input_projection`, `body.layer.*`
  rebuilt from `artifacts/pruned/k1/`). Load with `strict=False`, `extra_dim=70`. See `docs/INFERENCE_CONTRACT.md`.
- `recordings/` — gzipped viewer recordings (seed-labelled) for `viz/viewer.html`.

## Reproduce the audit

```
cd gpu_prototype
AUDIT_SEED=1234 PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 \
  python verify_proc.py ../training/11-count-curriculum/model_readout.pt 1024
```
