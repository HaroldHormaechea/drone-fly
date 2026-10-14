# UC70 run-2 — connectome-student prototype & experiment loop

The run-2 connectome student (frozen K1 reservoir + trained readout) with the three run-2
perception features — **next-gate pointer obs**, **world-frame gate memory**, and
**false-positive suppression** — trained end-to-end in the batched simulator. This is a working
prototype to fly and iterate on, **not** a high-success model: the point is a complete pipeline you
can improve step by step.

## Result (first run, 2026-10-14)

Run: `artifacts/training/liftoff-gremlin-micro/20261014T190706Z-student-liftoff-run2/` (gitignored).
611 iterations, 80M steps, ~2.2 h at `n_envs 2048` on a 6 GB RTX A3000.

Held-out eval trajectory (maps the student never trained on: HangarC03, MinusTwo, ThePit):

| iter | held-out completion | collision | gate_lost |
|---|---|---|---|
| 200 | 0.0% | 0.15 | 0.26 |
| 400 | 0.0% | 0.71 | 0.05 |
| 600 | 0.9% | 0.71 | 0.12 |
| **611 (selected)** | **1.8%** | **0.64** | **0.12** |

**Verdict — diagnostic success, modest flight score.** Run-1 failed because the memoryless
student lost the gate off the teacher's path and never re-acquired (`gate_lost ≈ 0.94`, completion
collapsed to ~0 as the teacher's control share `beta → 0`). Run-2's pointer + gate memory **solved
that**: `gate_lost` fell to ~0.06 on the training distribution (~0.12 held-out) and **held through
`beta → 0`** — the student keeps the gate. The failure **moved** to **collision (~0.64)**: it now
tracks the right gate but crashes into obstacles/floor. This matches ADR 0009's prediction that,
once the information gap is closed, obstacle avoidance is the next bottleneck. The student sits near
the teacher's own held-out ceiling (10.8%).

**Next levers (in priority order):** obstacle/collision handling; the live pointer reader (UC-74
draft, unblocks the pointer gain in real Liftoff); more training maps (scene variety).

## Fly it / iterate (this Linux box, sim)

Helper: `artifacts/harness/liftoff/training-data/_tools/experiment-run2.sh` (creates the required
legacy-path symlinks itself; CUDA + `DRONE_FLY_DATA_ROOT` assumed).

```bash
# launch a new experimental training run (detached, resumable), default config/n_envs
artifacts/harness/liftoff/training-data/_tools/experiment-run2.sh train student-liftoff-run2 2048

# score a trained checkpoint on held-out maps (per-episode outcomes JSON)
artifacts/harness/liftoff/training-data/_tools/experiment-run2.sh eval \
  artifacts/training/liftoff-gremlin-micro/20261014T190706Z-student-liftoff-run2/checkpoints/student.pt
```

Raw CLI (what the helper wraps):
`DRONE_FLY_DATA_ROOT=<data-root> python -m drone_fly.cli.main train student --config student-liftoff-run2 --n-envs 2048 --device cuda`

### The experiment loop
1. Copy `src/drone_fly/training/configs/student-liftoff-run2.yaml` to a new variant and change
   one thing — a reward term (`env.rewards.*`), a perception toggle (`perception.pointer` /
   `gate_memory` / `fp_suppression`), or a pointer-noise sigma.
2. `experiment-run2.sh train <your-variant>` — detached, logs under
   `models/student/logs/exp-<variant>-<stamp>.log`.
3. Watch the iteration log (`it N steps … compl … outcomes {…}`): `compl` should rise and the
   dominant `outcomes` term tells you the failure mode to attack next (today: `collision`).
4. Compare the held-out `macro_completion` in `eval/val_history.jsonl` against this baseline (1.8%).

### Required legacy symlinks
The student trainer still reads the props/scene/label tables from the pre-data-root paths. The
helper recreates these each run; if running the raw CLI, ensure:
`artifacts/harness/trackdata → <data-root>/track-library/trackdata-uc68`,
`artifacts/harness/scenes → <data-root>/scenes`, `artifacts/harness/labels → <data-root>/labels`.
(A proper fix is to route these through `drone_fly.paths` — tracked as a known gap.)

## Watching it fly

- **Quantitative (works now, headless):** the training iteration log and `train eval` per-episode
  `outcomes` are the live signal — this is the loop above.
- **Visual 3-D (gap):** the `app/` + `viz/viewer.*` 3-D viewer is wired to the older
  `gpu_prototype/` reservoir pipeline (it replays episodes recorded by `gpu_prototype/record_aug.py`),
  so it does **not** render a `drone_fly.sim` run-2 rollout out of the box. Visualizing the run-2
  student needs a small bridge: a `drone_fly.sim` rollout recorder that emits the viewer's
  episode-JSON (`viz/viewer.js` loads a plain-JSON / gzip episode file). Not yet built.
- **Live Liftoff (Phase 2):** the real "watch it fly" surface is the `drone-fly harness run` path
  against live Liftoff on Windows, with the UC65 brain/debug overlay — this is where the UC-74
  pointer reader makes the pointer real. Tracked as Phase 2.
