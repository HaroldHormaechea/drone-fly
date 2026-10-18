# Use Case 77: Obstacle-free student training for open race tracks (Infinity Loop class)

## Summary
The deployable K1 student (UC70 run-2 / UC75 run2b) is **blind to obstacles**: it flies with `free_space_estimator: none` and an all-zero `obstacle_vision` block. Yet it is trained on Liftoff courses that keep their full obstacle geometry (containers, props, scene colliders). The 2026-10-15 run2b training ended at:
- about 1% completion once the teacher fully handed over (beta → 0);
- collisions about 45% of episodes, out-of-bounds about 25%, floor about 20%;
- gate loss only about 5%.

Live on Liftoff Arena "04 - Infinity Loop" it crashed within about 10 s. The student is being trained to dodge things it cannot see. This use case adds an **obstacle-free training mode** that matches the student's actual perception, for open race tracks like Infinity Loop, which has no real obstacles in the race corridor:
- training and validation courses keep the gates, floor, ceiling and world bounds;
- they drop non-gate obstacle geometry from contact and from the teacher's privileged obstacle block.

It ships as a new student config variant (`student-liftoff-open`, extending the UC75 `student-liftoff-run2b` noise model). It also adds a per-track sim evaluation on `liftoff-arena-04`, so "can it fly Infinity Loop in sim" is measured directly. The user then trains it on the local GPU and flies it live on Infinity Loop with a recorded, overlay-visible run.

## Acceptance Criteria
1. **Obstacle-free course option.** The training track mix supports an opt-in flag (e.g. `tracks.obstacles: none`, or a per-mix-entry flag). The flag removes every non-gate obstacle from the simulated course: from contact/collision and from the teacher's privileged obstacle observations. Gate frames, floor, ceiling and out-of-bounds limits stay. The default is unchanged (obstacles on), so every existing config trains bit-identically.
2. **Config variant.** A new packaged training config `student-liftoff-open.yaml` `extends: student-liftoff-run2b` and enables obstacle-free courses for training and validation. It loads strictly through the training config loader and appears in `packaged_names()`. `student-liftoff-run2b.yaml` and `student-liftoff-run2.yaml` are unchanged.
3. **Contract unchanged.** Student obs schema stays VERSION 3 / DIM 65, `obstacle_vision` stays zeros, and the student↔teacher interface and action contract are untouched. A checkpoint from the new config exports and flies through the same `train export` / `harness run --checkpoint` path.
4. **Teacher compatibility.** The existing UC68 teacher (`teacher_it1374`) is used unchanged. When its privileged obstacle block is emptied, it still produces valid actions, and the feasibility stage runs. If the teacher's feasibility on obstacle-free courses is reported, it is logged rather than gating the run.
5. **Per-track sim evaluation.** `drone-fly train eval` (or the student val loop) can score a checkpoint on a single selector, e.g. `liftoff:liftoff-arena-04`, in obstacle-free mode, reporting completion and an outcome breakdown. After training, the run reports both the standard held-out `macro_completion` and the Infinity Loop score.
6. **Tests.**
   - (a) Obstacle-free courses contain zero non-gate obstacles in contact and teacher obs, while gates, floor and bounds remain.
   - (b) Default-config courses are unchanged (an obstacle count or hash parity check against the current conversion).
   - (c) `student-liftoff-open` loads and overrides exactly the obstacle setting over run2b.
   - (d) A short CPU smoke run of the student loop with the new config, a few iterations at tiny `n_envs`, completes without error.
   - (e) Schema VERSION 3 / DIM 65 is asserted.
7. **Train + fly (user-authorized, run by the orchestrator after merge).**
   - Train `student-liftoff-open` on the local RTX 3050 to the config budget.
   - Export it.
   - Fly it live on Infinity Loop. The level is loaded via Quick Setup and the spawn is verified before arming, using the `liftoff` preset plus the pointer overlay. The overlay is recordable and the full screen is captured to MP4.
   - Report sim metrics vs run2b and the live outcome.
8. **Docs.** `docs/experiments/` gets a short section explaining why obstacle-free training matches the blind student, how to train and evaluate the variant, and the results.

## Potential Pitfalls & Open Questions
- **Risk (blind student on obstacle tracks)** — a student trained obstacle-free will crash into real obstacles on tracks that have them. That is accepted: it is scoped to open tracks, and obstacle perception is the separate UC73 / "eyes" track. The docs must say so.
- **Ambiguity (what counts as an obstacle)** — scene colliders near the corridor (arena walls, tyre stacks at gates) vs. props. The analyst must define it from the conversion code (`tracks/liftoff.py` two obstacle lists, `harness_obstacle_max_half_m`) so that Infinity Loop's actual corridor matches the obstacle-free sim.
- **Risk (out-of-bounds and floor still high)** — about 25% out-of-bounds and about 20% floor were also failure modes, and they are not obstacle-related. Removing obstacles may not be enough; the per-track eval will show it. Reward/termination tuning stays out of scope unless trivial.
- **Edge case (procedural share)** — procedural courses also carry obstacles (`tracks/procedural.py`); the flag must cover them too.
- **Assumption** — Infinity Loop (`liftoff-arena-04`) is in the **train** split (map LiftoffArena, all 7 arena tracks are train). The per-track eval on it is therefore an in-distribution check, not a held-out claim; the held-out `macro_completion` remains the generalisation metric.
- **Risk (compute)** — training is about 3–4.5 h on the RTX 3050 at 2048 envs, uncapped (`--max-hours`), and longer if the GPU is shared.

## Original Description
> Then we need to train for flights without real obstacles like this infinity loop. Go autonomously

## Clarifications
- The user asked Claude to proceed autonomously (2026-10-15): define, implement through the dev team, merge, train, and fly with a recorded overlay run, without stopping for plan approval.
