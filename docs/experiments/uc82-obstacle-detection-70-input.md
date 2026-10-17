# UC-82 — obstacle detection for the 70-input connectome (ray noise + avoidance retrain)

Sim-only (no Liftoff) closing of UC-73 AC7–AC8, building on UC-78's shipped "eyes" module. The
question: does the 70-input connectome actually *use* its 40 vision-derived obstacle rays to avoid
obstacles? We measure a ray-noise model from the eyes' documented accuracy, retrain the avoider
under that noise, and compare its collision share against the same model flown with its rays
blanked.

## Model identity (important)

The 70-input obstacle-avoider is **`training/11-count-curriculum/model_readout.pt`** (lineage
10/11/12; 67 % honest deterministic completion @ seed 1024, 0 % barging, stripped 674.8 K readout —
the body is rebuilt from the connectome at `artifacts/pruned/k1` via `reservoir.load_readout`,
`strict=False`). All UC-82 retrains warm-start from it (`src_extra_dim: 70`, `legacy_v1: false`).

It is **not** `uc70-diag-run1` (that is `in_dim: 61`, the deployable student-obs distill thread that
failed in UC-79/80/81) and **not** the 177 M `exp11_count.pt` (the full checkpoint). Warm-start load
is asserted in the tests (loaded readout extra width == 70).

## AC1 — the obstacle-ray noise model

`drone-fly eyes noise-model` writes `artifacts/eyes/ray-noise.yaml` (schema `ray-noise` v1,
`drone_fly.sim.ray_noise.RayNoiseModel`). No trained eyes estimator, dataset or frame cache is on
disk, and UC-78 showed even a reproduced eyes v1 is near-useless (95.8 % miss overall, 99 % on the
nearest obstacles), so the error source is **path (b): the UC-78 documented held-out eval
statistics** (`source: uc78-eval-documented-fallback`). `--from-eval eval_held-out.json` can instead
derive the model from an actual eyes eval report when one exists.

Field → source (miss dominates; bias/spread are documented second-order constants):

| field | value | source |
|---|---|---|
| `miss_prob.in_view` (0-1 / 1-2 / 2-3 / 3-4 m) | 0.990 / 0.913 / 0.680 / 0.450 | UC-78: nearest-quarter 0.990, in-view aggregate 0.913, 75-100 % bin ~0.45 (bin 2 interpolated) |
| `miss_prob.out_of_view` (all quarters) | 0.99 | out-of-view rays get no image features (in_view=0 → predict clear) → almost always missed |
| `false_alarm_clear.in_view` | 0.031 | UC-78 false-alarm rate on truth-clear rays |
| `false_alarm_clear.out_of_view` | 0.0 | zeroed features → predict clear → ~no false alarm |
| `radial_bias_m` | 0.0 | no documented systematic bias on the few detected rays |
| `sigma_m` | 0.5 | the eval confidence tolerance (tol_m, training m) |

Range: 4 training m = 16 Liftoff m (`world_scale` 4.0). The camera block (2560×1080, hfov 120°,
uptilt 20°, HUD masks) is read from the `liftoff` preset, so the simulator's in-view / out-of-view
split (`RayNoiseModel.camera_twin().in_view`, a batched-torch twin of
`eyes.camera_rays.project_rays`, pinned to the NumPy reference by a test) matches the eyes
projection.

**Follow-up (recorded):** a richer per-bin bias/σ grid once a trained eyes estimator exists; today
dropout + false alarm dominate, so bias/spread as single documented constants is defensible.

## AC2 — noise injection (and the blanked control)

`course.ray_noise` (a path to the model) injects the error into the 40 raycast rays of the 70-input
observation inside `CourseEnv.extra()`; `course.ray_blank` instead replaces them with **constant
clear (1.0)** — the AC8 control (NOT zeroed, so "no obstacle seen" rather than an out-of-distribution
zero). Both require `raycast` and are mutually exclusive. With both off the ray path is
**byte-identical** to before (no extra RNG is drawn; injection uses a dedicated generator so the
plant / timing streams are untouched) — test-locked.

## AC3/AC4 — the retrains

Two warm-started fine-tunes on obstacle-bearing tractable courses, same recipe as the 11-count
avoider (`raycast` + `shaping` + `obs_terminal`, obstacles on), differing only in the ray channel:

- `training/configs/uc82-obstacle-rays.yaml` — rays noise-injected (AC3).
- `training/configs/uc82-obstacle-blank.yaml` — rays blanked (AC4 control).

Both: 256 envs, 5 M steps, curricula off (full difficulty from step 0 — the model is already
trained). Driver: `scripts/uc82_obstacle_avoidance.sh` (steps 0→3, reboot-safe: per-step markers,
`STOP` file, ~10 h cap).

## AC5/AC7 — evaluation and the success bar

`drone-fly train proc-eval` flies one deterministic episode per env (mean action) on **held-out**
obstacle courses (a distinct pool base, disjoint from training), tallying the outcome breakdown from
the in-step `last_completed` / `cause_*` flags on `course.OUTCOMES` — obstacle contact is terminal,
so a **collision** is counted exactly once (this avoids the post-step-position tally bug noted in
EXPERIMENTS.md). `training/evaluate.py` is Liftoff-teacher-only, hence this new proc-course
evaluator.

**Success bar (AC7):** collision share with rays **< collision share blanked**, specifically a
**relative reduction ≥ 20 %** (`compare_collision`). The reduction is plausibly small or absent: at
~96 % miss the rays carry little signal, so the ray-noise student may converge to ≈ the blanked
control — the documented-negative branch below covers that outcome, and the deliverable is the first
measured evidence either way.

### Comparison (fill from `artifacts/uc82/03_eval.json`)

| model | collision share | completion | floor | out-of-bounds | ceiling | timeout |
|---|---:|---:|---:|---:|---:|---:|
| ray-noise (rays) | 0.570 | 0.148 | 0.102 | 0.000 | 0.000 | 0.180 |
| blanked control | 0.352 | 0.141 | 0.227 | 0.000 | 0.000 | 0.281 |
| 11-count reference (exact rays, obstacles-on) | 0.33* | 0.67 | — | — | — | — |

\* 11-count headline is 67 % honest completion @ seed 1024 with exact rays; its collision share on
this held-out proc set is a reference point, not the control.

Relative collision-share reduction: **−62.2 %** (bar 20 %). Verdict: **NEGATIVE** — the ray-noise
model collides *more* than the blanked control (0.570 vs 0.352), not less.

## Result (run 2026-10-17, RTX A3000, 256 held-out proc courses, seed 1234)

**Negative, and informative: feeding eyes-v1-grade vision rays makes obstacle avoidance worse, not
better.** Both arms fly about equally (completion 14.8 % rays vs 14.1 % blank — so the 70-input base
model works and the comparison is meaningful), but the ray-noise model hits obstacles **57.0 %** of the
time vs **35.2 %** for the blanked control. The outcome mix explains it: with noisy rays the model
*reacts* — fewer floor/timeout endings (floor 10.2 %, timeout 18.0 %) but far more obstacle
collisions, because at ~96 % miss and a 3.1 % false-alarm rate the rays mostly fail to warn of real
obstacles while occasionally inventing phantom ones, steering the connectome *into* trouble. The
blanked control, flying its learned path "blind but calm," hits fewer obstacles (35.2 %) and instead
floors/times-out more (floor 22.7 %, timeout 28.1 %).

So the connectome *does* respond to the ray inputs — but eyes-v1 rays carry more harm than signal. This
is the expected consequence of UC-78's own finding that the eyes v1 (95.8 % miss) are not yet good
enough to feed the connectome; UC-82 is the first *measured* confirmation of it at the connectome level.

## Recommendation

The lever is **ray accuracy, not avoidance training.** Do not iterate on the 70-input avoidance
retrain against these rays — garbage-in dominates. The next step is a materially better eyes estimator
before re-running this comparison:
- **More signal in the eyes:** more capture data and/or temporal memory (frame stack / recurrence) and
  the odometry-propagated obstacle map UC-73 lists as a stretch goal, to cut the ~96 % near-obstacle
  miss rate that makes the rays net-negative here.
- **Warning time:** at `RAY_RANGE` 4 m there is only ~0.2–0.3 s of warning at racing speed (UC-73
  AC10); a longer range (with a 70-model retrain at that range) may be needed even once the estimator
  improves.
- Only once the eyes clear a usefulness threshold (e.g. miss rate well below the point where false
  warnings dominate) should this rays-vs-blanked comparison be re-run; then live Liftoff validation
  (UC-73 AC9–10) follows. Re-running the avoidance retrain at current eyes fidelity is not worthwhile.
