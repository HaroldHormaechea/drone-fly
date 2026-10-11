# 12 — tunnels (first fly-through-tunnel model)

The first model that navigates **tunnels / doorways** — openings walled on both sides and overhead that
the drone must thread (it cannot go around or over; terminal contact so it cannot clip through).

**Honest completion: 28%** on the tunnel course pool (seeded 1024, `AUDIT_SEED=1234`), **0% barging among
completed flights** (every successful run threads the gap cleanly), 2.43 m/s. For reference, the
non-tunnel avoider `11-count-curriculum` scores only **7%** on these same tunnel courses — so training on
tunnels transferred the skill (**4×**). Not yet mastered: 63% of episodes end in a fatal wall-clip, so a
tunnel-gap curriculum (wide → narrow) is the natural next step to push this higher.

## How it was trained

`gpu_prototype/exp16_tunnels.py`: resume `11-count-curriculum` (the 67% multi-obstacle avoider) and
fine-tune on the tunnel-augmented pool (`PROC_TUNNELS=1`). Each course gains a tunnel built from 3
axis-aligned boxes — two side walls + a ceiling tracking gate height — leaving a clear central lane. Floor
shaping on (threading at gate height makes ground-contact a real risk). Reuses the box collision/raycast
engine; `MO` raised 4→10. The inference contract / `extra_dim=70` is unchanged.

## Audit (needs the tunnel pool)

```
cd gpu_prototype
AUDIT_SEED=1234 PROC_TRACTABLE=1 PROC_TUNNELS=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 \
  python verify_proc.py ../training/12-tunnels/model_readout.pt 1024
```

## Files

- `model_readout.pt` — stripped readout (`extra_dim=70`, load `strict=False`; keeps `body.input_projection`).
- `recordings/` — gzipped viewer recordings. Note: the browser viewer currently approximates box obstacles
  as cylinders, so the tunnel walls render approximately; the trajectory (drone threading) is accurate.
