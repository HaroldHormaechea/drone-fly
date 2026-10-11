# 12 — tunnels (fly-through-tunnel model)

The first model that navigates **tunnels / doorways** — openings walled on both sides and overhead that
the drone must thread (it cannot go around or over; terminal contact so it cannot clip through).

**Honest completion: 31%** on the tunnel course pool (seeded 1024, `AUDIT_SEED=1234`), **0% barging among
completed flights** (every successful run threads the gap cleanly), 2.55 m/s. For reference, the
non-tunnel avoider `11-count-curriculum` scores only **7%** on these same tunnel courses — so training on
tunnels transferred the skill (**4.4×**). Not yet mastered (most failures are fatal wall-clips).

## How it was trained

Two stages. **exp16** (`gpu_prototype/exp16_tunnels.py`): resume `11-count-curriculum` (the 67% multi-
obstacle avoider) and fine-tune on the tunnel-augmented pool (`PROC_TUNNELS=1`) → 28%. **exp18**
(`exp18_tunnel_curr.py`): a tunnel **gap curriculum** — the walls carry an outward `ovec`, so the position
curriculum (`pos_curr_frac`) starts the gap wide/forgiving and narrows it to the designed tight gap over the
first half of training → **31%** (this model). Each tunnel is 3 axis-aligned boxes (two side walls + a
ceiling tracking gate height) leaving a clear central lane. Floor shaping on. Reuses the box collision/
raycast engine; `MO` raised 4→10. The inference contract / `extra_dim=70` is unchanged.

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
