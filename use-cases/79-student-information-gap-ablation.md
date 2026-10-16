# Use Case 79: Student information-gap ablation (which privileged inputs the student needs)

## Summary
The UC-77 student completes 0 % with **exact** perception. The teacher (`teacher_it1374`) completes 93.8 % on Infinity Loop
(`liftoff:liftoff-arena-04`) and 17.1 % on `liftoff:held-out`. An MLP student under the same config also fails (held-out 1.0 % noise / 0.0 % exact, Infinity Loop 0 %), and a 30M-step PPO fine-tune of it did not help. So the bottleneck is neither the K1 actor's capacity nor perception noise. The leading hypothesis is an information gap. The student observation (`student-obs`, 61–65 values) lacks three blocks of the teacher observation (`teacher-obs` v2, 150 values):
- the full pose of the target, next and next-next gates (39 values);
- the 8 nearest surfaces of the full geometry, including the floor/terrain (64 values);
- the altitude above the surface below the drone (1 value).

This UC adds **opt-in, sim-only, diagnostic** privileged blocks to the student observation, one variant at a time, and trains each with the MLP-student actor on the `student-liftoff-open` config. The result is a comparison table showing which missing inputs matter. It is pure training and evaluation in the GPU simulator and needs no Liftoff. The default student obs schema, sidecars, exports and the deployed student stay unchanged. The output feeds a follow-up UC: legitimate telemetry/vision equivalents of the winning inputs, and/or an asymmetric-actor teacher.

## Acceptance Criteria
1. A new config field (for example `distill.student_privileged: [<block>, ...]`, default empty) selects privileged blocks appended to the student observation. The selectable blocks are:
   - `altitude`: the teacher's altitude-above-surface value;
   - `floor`: distance/direction to the floor/terrain only, the floor entry of the teacher's surface block;
   - `gate_next_next`: the next-next gate pose, in the teacher's encoding;
   - `gates_full`: the full poses of the target, next and next-next gates (the teacher's 39 values);
   - `surfaces`: the teacher's full 8-nearest-surfaces block (64 values);
   - `teacher_full`: the complete teacher observation, as an upper bound.
2. Each block is computed by the **same code path** the teacher uses for the same quantity, so it is numerically identical to the teacher's value for the same state. A test asserts this equality.
3. With the default (empty) setting, everything is byte-identical to before: the student obs dimension and schema version, `config.yaml` snapshots, student sidecars and exports. The existing tests still pass.
4. A student trained with any privileged block records the block list and the resulting obs dimension in its sidecar. `train export` and `harness run` **refuse** such a checkpoint with a clear message ("diagnostic privileged student, not deployable").
5. Exact vs noise perception: noise applies to the student's perceived inputs exactly as today. Privileged blocks are always exact, are documented as such, and the report labels them.
6. One diagnostic config per variant ships under `training/configs/` (e.g. `student-liftoff-open-priv-<block>.yaml`). Each is `student-liftoff-open` plus the privileged list, with `--actor mlp-student` as its actor, and uses a shortened budget of about 45 min per variant on an RTX 3050 at 2048 envs. The budget is documented and kept equal across variants and the baseline.
7. A **baseline** run (no privileged blocks, same shortened budget) is part of the ablation, so every variant is compared like-for-like.
8. Each run produces the UC-77 `final_report`: held-out `macro_completion` (the target metric) and Infinity Loop completion (a check, in the train split), each with exact and noise perception, plus the outcome breakdown (collision, floor, out-of-bounds, gate_lost, timeout). Same episodes and seed as UC-77 (32 per track, seed 1234, obstacles `none`).
9. A driver script or documented command sequence runs all 7 runs (the baseline plus 6 variants) back-to-back, unattended and resumable. A run interrupted by a reboot resumes rather than restarting the sequence. It is runnable on any CUDA machine with no Liftoff installed, needing only the repo, the teacher run and the noise model; the docs list exactly which artifacts must be copied over.
10. `docs/experiments/uc79-student-information-gap.md` contains one comparison table with the following rows: teacher, baseline, the six variants, and the UC-77 numbers as a reference. Its columns are the AC8 metrics. It ends with a **recommendation**: which inputs close the gap, and whether even `teacher_full` fails, which would point at the distillation setup itself (beta schedule, DAgger) rather than the obs.
11. Unit tests cover:
    - obs dimension per block combination;
    - the default being unchanged;
    - the export/harness refusal;
    - teacher-equality for each block;
    - a smoke train of one iteration with a privileged block on CPU.

## Potential Pitfalls & Open Questions
- **Risk** — Training with privileged blocks while the DAgger teacher still drives at high beta may hide the gap. Keep the beta schedule identical to `student-liftoff-open` so variants differ only in obs.
- **Risk** — `teacher_full` is the upper bound. If the MLP student with the full teacher obs still collapses at beta→0, the obs is not the problem, and the follow-up must target the distillation itself (DAgger relabelling, beta schedule, loss) instead.
- **Edge case** — Obstacle-free courses leave the `surfaces` block with only scene, gate-frame and floor entries. That is fine and matches what the open student will meet, but say so in the docs.
- **Assumption** — The ~45 min shortened budget is long enough to separate the variants. If the baseline and all variants are 0 %, re-run the top candidates at a longer budget before concluding.
- **Assumption** — MLP only. The K1 connectome is not run in this UC; a follow-up applies the winners to K1.
- **Missing input** — The other machine needs the teacher run (`teacher_it1374`), the noise model YAML and the Liftoff track conversion cache (or the trackdata to rebuild it). The docs must list the exact paths to copy.

## Original Description
UC-79: "Student information-gap ablation". The UC-77 student (K1 and MLP actors) completes 0% with exact perception while the teacher scores 93.8% on Infinity Loop and 17.1% held-out. Root-cause hypothesis: the student observation (student-obs, 61-65 values) lacks inputs the privileged teacher (teacher-obs v2, 150 values) acts on: (a) full pose of target/next/next-next gates (39 values), (b) 8 nearest surfaces incl. floor/terrain (64), (c) altitude above surface below (1). Goal: a diagnostic ablation, MLP-student actor, student-liftoff-open config, that appends ONE privileged block at a time to the student obs (sim-only, diagnostic, never deployable): altitude; floor-only surface; next-next gate pose; full gate poses; full 8 surfaces; and the full teacher obs as an upper bound. Each variant trains (~45 min on the RTX 3050) and reports held-out macro completion + Infinity Loop completion with exact and noise perception and the outcome breakdown (collision/floor/OOB/gate_lost/timeout), in one comparison table in docs/experiments. Must not change the default student obs schema, sidecars, or deployed student behaviour; variants are opt-in via config. Generalisation (held-out) is the target metric; Infinity Loop is a check. Output: a recommendation of which inputs matter, feeding a follow-up UC for legitimate (telemetry/vision) equivalents or an asymmetric-actor teacher.

(User, 2026-10-16: "#2 go ahead, check which inputs help".)

## Clarifications
- Q: How much GPU time can the ablation take?
  A: 6 variants, ~5 h.
- Q: Should the ablation also run the K1 connectome actor for the winning variant(s)?
  A: MLP only.
- Q: Can the GPU be used all day on this PC?
  A: "Does this require liftoff? if it does not, upload an UC with this and I'll run it in another computer". It does not, so the UC is registered for the user to run on another machine.
