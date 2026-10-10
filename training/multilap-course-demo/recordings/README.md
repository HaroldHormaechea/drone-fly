# multilap-course-demo — the full multi-lap circuit tier (visualization only)

**This is a visualization, not a trained result.** It exists so the **full multi-lap course tier** can
be *seen* being flown. The pilot here is the existing `proc-course-flight` readout, which was trained on
the easier *tractable* tier (1 lap, 5–10 gates), so it flies these harder courses imperfectly — it
completes the easy ones and fumbles the long multi-lap ones. Judge the **courses**, not the pilot.

## What the courses are (the 7-spec procedural generator, `course_gen.py`)

- **Circular, 2–3 laps.**
- **5–20 gates**, each **rotated** to the travel direction, with **varied heights** (from a half-gate
  under the floor up to ~2× gate-diameter above it).
- **Apertures** randomized 3×–10× drone width.
- **Gate spacing** 1–5 m; **at least one chicane** per course.
- **Obstacles are OFF** in this demo (`PROC_NO_OBS=1`) — this is the clean navigation tier, so you can
  see the circuit geometry without the (still-unsolved) on-path obstacle challenge. The obstacle tier is
  the open problem (see [`docs/FLIGHT_MODELS.md`](../../../docs/FLIGHT_MODELS.md) and the
  `proc-obstacle-fail` recording).

## Pilot / model

`reservoir_aug.py` → `K1ReservoirAug`: the frozen K1 reservoir (3.86M real edges) + a readout that also
reads the raw flight obs. `model_readout.pt` is from `training/proc-course-flight/`.

## Reproduce

```bash
cd gpu_prototype
# full multi-lap tier (no TRACTABLE cap), obstacles off, frames downsampled for commit size:
PROC_NO_OBS=1 REC_STRIDE=2 python record_aug.py proc \
  ../training/proc-course-flight/model_readout.pt multilap-course-demo 12
```

> Frames are downsampled (stride 2); `dt` in the recording is scaled so playback keeps real timing.
