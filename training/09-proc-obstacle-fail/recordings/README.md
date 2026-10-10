# proc-obstacle-fail — diagnostic recording of the random-obstacle avoidance FAILURE

**This is not a working model.** It is a recording of a *failed* attempt, committed so the flight
behaviour can be inspected for patterns. The open problem it illustrates is documented in
[`docs/FLIGHT_MODELS.md`](../../../docs/FLIGHT_MODELS.md) § "Work in progress".

## What this is

The **brain-integrated** K1 reservoir (`reservoir_brain.py` → `K1ReservoirBrainRay`) flying the
**procedurally-random gate courses with on-path obstacles** (tractable tier: 1 lap, 5–10 gates). In
this model the forward raycast cone (40 rays) is injected *into the connectome itself* through a wide,
dedicated 256-neuron sensory population (the "wider translator" experiment), on top of the frozen
12-dim flight-obs projection; only the readout + the two input projections are trained, the ~3.86M
real MaleCNS edges stay frozen.

## The result (why it's here)

- **Deterministic audit: ~5% completion** (`verify_brain.py`, 10/192) — honest, *not* cheating
  (2% barging, 0% among completed, speed 1.34 m/s: it flies cautiously and fails, it does not plow
  through obstacles or hover to game the metric).
- These 12 episodes are the deterministic policy; **1/12 completes**, consistent with the audit.

## What to look for (pattern-hunting)

- Does it **stall / slow down** as it approaches an on-path obstacle (cautious collapse) rather than
  committing to an avoidance arc?
- Does it **lose the course** after a near-miss (long, wandering episodes)?
- The obstacles are drawn at their true on-path positions; watch the drone path *relative* to them.

## Reproduce

```bash
cd gpu_prototype
# audit:
PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_OBS_TERMINAL=1 RAY_POP=256 python verify_brain.py proc_brain_wide.pt 192
# re-record (frames downsampled via REC_STRIDE to keep JSON under GitHub's per-file limit):
REC_STRIDE=2 PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_OBS_TERMINAL=1 RAY_POP=256 \
  python record_brain.py proc_brain_wide.pt proc-obstacle-fail 12
```

> Frames are downsampled (stride 2); `dt` in the recording is scaled so playback keeps real timing.
