# UC-79 — Student information-gap ablation

Which **missing teacher inputs** (not actor capacity, not perception noise) cause the distilled
student to complete 0 % with exact perception while the teacher scores 93.8 % on Infinity Loop and
17.1 % held-out? This is a diagnostic ablation: it appends ONE privileged teacher block at a time to
the MLP student's observation, trains each on the obstacle-free `student-liftoff-open` config under
an equal step budget, and compares held-out generalisation + the Infinity Loop check. It is pure GPU
training and evaluation in the batched simulator — **no Liftoff**.

## Hypothesis

UC-77 showed neither the K1 actor's capacity (an MLP student also fails) nor perception noise (exact
perception also fails) explains the collapse, and a 30 M-step PPO fine-tune did not help. The
leading hypothesis is an **information gap**: the deployable student obs (`student-obs` v3, 65
values) omits blocks the privileged teacher (`teacher-obs` v2, 150 values) acts on — the full
gate poses, the 8 nearest surfaces (incl. the floor), and the altitude above the surface below.
If appending one of these restores completion, that input matters; if even the full teacher obs
(`teacher_full`) collapses, the obs is not the bottleneck and the distillation setup is.

## The privileged blocks

Selected per run via `distill.student_privileged` (config field; default empty). Every block is
**always exact** (computed by the teacher's own code path on the env's truth state, so noise never
touches it), **sim-only**, **diagnostic**, and the resulting student is **never deployable** —
`train export` and `harness run` refuse it with "diagnostic privileged student, not deployable".

| Block | Width | What it adds | Variant obs dim |
|---|---|---|---|
| `altitude` | 1 | teacher's altitude-above-surface value | 66 |
| `floor` | 8 | floor/terrain surface only (teacher surface encoding) | 73 |
| `gate_next_next` | 13 | next-next gate pose (teacher gate encoding) | 78 |
| `gates_full` | 39 | target + next + next-next gate poses (teacher `gates` block) | 104 |
| `surfaces` | 64 | teacher's full 8-nearest-surfaces block | 129 |
| `teacher_full` | 150 (v2) / 148 (v1) | the complete teacher observation — **upper bound** | 215 / 213 |

**Obstacle-free caveat.** On open courses (`tracks.obstacles: none`) the `surfaces`/`floor` blocks
hold only scene colliders, route-gate frames and the floor — no placed props. That is exactly what
the open student meets in deployment, so the ablation stays faithful; it just means `surfaces` here
does not probe prop avoidance (that is the UC-73 "eyes" track).

## Running it (AC9) — artifacts to copy and the command

Runnable on any CUDA machine with **no Liftoff installed**. Copy onto the run machine:

1. **This repository**, installed so `drone-fly` is on `PATH` (`uv sync` or `pip install -e .`).
2. **The teacher run** `teacher_it1374` — the frozen teacher the student distills from. Pass its
   run folder (or run id under `<data root>/models/teacher/`) as `TEACHER`.
3. **The detector noise model**:
   `artifacts/models/detector-v1/liftoff-gates-161f7a83/noise_model.yaml` — pass its path as `NOISE`.
4. **The Liftoff track-conversion cache** under `artifacts/tracks/cache/` so the open courses resolve
   without Liftoff. If it is absent, rebuild it once from the trackdata (`drone-fly tracks …`) on a
   machine that has the trackdata; the cache is then portable.

Then run the resumable driver (trains baseline + 6 variants back-to-back, each writing the UC-77
final_report):

```bash
TEACHER=<teacher_it1374 run folder or id> \
NOISE=artifacts/models/detector-v1/liftoff-gates-161f7a83/noise_model.yaml \
DEVICE=cuda \
scripts/uc79_ablation.sh
```

It is **reboot-safe**: re-running skips completed variants (those with
`eval/student-held-out.json`), resumes an interrupted one from its run folder (the step budget is
cumulative), and only then continues — it never restarts the sequence. Each variant is ≈ 45 min on
an RTX 3050 at 2048 envs; all 7 share an identical step budget (the across-variant equalizer).

## Comparison table (AC10)

Rows: teacher, baseline (no privileged blocks), the 6 variants, and the UC-77 reference. Columns are
the AC8 metrics — held-out `macro_completion` (the **generalisation target**) and Infinity Loop
completion (`liftoff:liftoff-arena-04`, a train-split **check**), each with exact and noise
perception, plus the held-out outcome breakdown (floor / collision / out-of-bounds / gate_lost /
timeout), all at 32 episodes per track, seed 1234, obstacles `none`.

Fill the variant rows after the runs finish with:

```bash
python scripts/uc79_table.py artifacts/uc79-ablation
```

The teacher and UC-77 rows below are known; the baseline and 6 variant rows are placeholders until
the GPU run lands (the reader regenerates all rows except the static UC-77 reference).

| Run | Held-out (noise) | Held-out (exact) | Infinity (noise) | Infinity (exact) | Held-out outcomes noise (floor/coll/oob/lost/timeout) |
|---|---|---|---|---|---|
| teacher (exact) | — | 17.1% | — | 93.8% | — |
| baseline | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| altitude | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| floor | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| gate_next_next | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| gates_full | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| surfaces | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| teacher_full | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| UC-77 student (K1, ref) | 0.0% | 0.0% | 0.0% | 0.0% | — |
| UC-77 student (MLP, ref) | 1.0% | 0.0% | 0.0% | 0.0% | — |

## Recommendation rubric

Read the filled table against these rules (generalisation = held-out; Infinity Loop is a sanity
check in the train split):

- **A single block lifts held-out markedly over baseline** → that input is the gap. Follow up with a
  legitimate (telemetry/vision) equivalent of it, or an asymmetric-actor teacher that supplies it.
- **Only `gates_full`/`surfaces` (the broad blocks) help, not their narrow subsets**
  (`gate_next_next`, `floor`, `altitude`) → the student needs the full field, not a single value;
  prioritise the richer equivalent.
- **`teacher_full` lifts the student but no single block does** → the inputs matter only in
  combination; evaluate the smallest helpful subset next.
- **Even `teacher_full` collapses at beta→0** → the observation is **not** the problem. Retarget the
  distillation setup itself — the beta schedule, DAgger relabelling, or the loss — rather than adding
  inputs. This is the decisive negative result the UC is designed to surface.
- **Baseline and all variants read ~0 % at this budget** → inconclusive, not negative: re-run the top
  candidates at a longer step budget before concluding (the ~45 min budget is an assumption; see the
  configs' header).

All of these keep MLP-only scope; applying the winner(s) to the K1 connectome actor is the follow-up
UC.
