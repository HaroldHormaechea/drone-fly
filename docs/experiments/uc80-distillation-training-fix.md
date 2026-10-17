# UC-80 — Fix the student distillation training

UC-79 established that the MLP student's ~0 % course completion is **not** an observation gap:
even `teacher_full` (the student handed the entire 150-value teacher observation) reached only
0.6 % held-out `macro_completion` (noise) / 1.0 % (exact) versus the teacher's 17.1 %, and no
single privileged block cleared ~0.2 %. The decisive negative result pointed at the **distillation
setup** itself — compounding error once the DAgger β anneals to 0 (the student drifts to states it
was never trained on) plus an action-matching loss that does not optimise trajectory/completion
success. This experiment retargets the distillation as an **ablation of three interventions**, each
measured independently against a baseline (the current UC-77/79 distillation), with the goal of
lifting held-out `macro_completion` to **≥ 5 %** (noise). It is pure GPU training and evaluation in
the batched simulator — **no Liftoff** — and runs in-sandbox on the RTX A3000 exactly as UC-79 did.

## Hypothesis

If the obs is not the bottleneck (UC-79), the two remaining suspects are **covariate shift** — the
student is trained on teacher-visited states but evaluated on the states it drives itself into once
β→0 — and the **objective** — matching the teacher's action per tick does not reward completing the
course. Three levers attack these:

1. **On-policy DAgger relabelling** (`distill.on_policy_relabel`) — execute the *student* action
   during rollouts (decoupled from β) so the frozen teacher labels the states the student actually
   visits. Directly attacks covariate shift; no change to the teacher or the loss.
2. **Gentler/longer β schedule** (`distill.beta_end` / `beta_over`) — keep teacher correction alive
   across the whole budget instead of annealing to 0 at the half-way point, so student-driven
   rollouts never lose the teacher. Attacks covariate shift from the teacher-execution side.
3. **PPO completion-reward fine-tune** (`finetune.steps`) — after imitation, optimise the distilled
   student with PPO against the env reward (course success), not action matching. Attacks the
   objective.

If a lever moves held-out completion markedly over the baseline it is the fix; if none reaches the
≥ 5 % bar, this doc records the negative result and names the next lever (denser reward, longer
budget, or **combined** interventions) rather than silently passing.

## The interventions

All three are **opt-in via config, default-off**; with every new field unset the distillation is
byte-identical to UC-77/79 (the `on_policy_relabel` default drops from the run snapshot, so
`config.yaml` is unchanged; the student obs dim/schema, sidecars and exports are untouched).

| Variant | Config | Lever | Default (baseline) |
|---|---|---|---|
| `baseline` | `student-liftoff-open-uc80-baseline` | — (current UC-77/79 distillation) | — |
| `onpolicy` | `student-liftoff-open-uc80-onpolicy` | `distill.on_policy_relabel: true` | `false` |
| `beta` | `student-liftoff-open-uc80-beta` | `distill.beta_end: 0.15`, `beta_over: 1.0` | `0.0`, `0.5` |
| `ppo` | `student-liftoff-open-uc80-ppo` | `budget: 1.2e7` + `finetune.steps: 6.0e+6` | PPO off (`steps: 0`) |

**On-policy relabelling.** In the DAgger rollout the teacher is already queried on every visited
state and its decoded action is added to the buffer as the label. Baseline *executes* a β-mix:
with probability β the teacher action, else the student's (plus exploration noise). On-policy
execution instead runs `act = s_mean + noise` **unconditionally**, so the states that fill the
buffer are the ones the student drives itself into — exactly the distribution it is evaluated on.
The teacher still labels every one of them, so there is **zero extra teacher throughput cost** (the
teacher was already queried every tick). There is no teacher-forcing warmup, so the run leans on the
warm-started actor; see the follow-up note below.

**β schedule.** `beta_at(frac)` anneals teacher-forcing from `beta_start` (1.0) to `beta_end` over
`beta_over` fraction of the budget. The baseline anneals to **0 over the first half** (so the whole
second half is unaided student execution). The `beta` variant sets `beta_end: 0.15`
(floors teacher correction at ~15 %) and `beta_over: 1.0` (stretches the anneal across the whole
budget), so student-driven rollouts keep receiving some teacher correction throughout. Schedule
stays `linear`; only the endpoint and horizon change. No code change — the schedule was already
parameterised.

**PPO fine-tune.** After DAgger the distilled student is optimised with PPO against the env reward —
the sim's `RewardConfig`: per-metre **progress** toward the target passage (1.0/m, dense), a **gate**
bonus (10) on each crossing, a **finish** bonus (50), a **crash** penalty (10), and small upright /
spin / action-rate regularizers. This optimises course success rather than matching the teacher's
action. No code change — the PPO phase (`ppo_finetune`) already exists; the variant only turns it on
and sizes it.

## Budget & equal-budget note (AC5/AC8)

All four runs share an **equal 1.8e7 total env-step budget**. Both training loops count env-steps
(DAgger `steps += rollout·n_envs`; PPO `steps += roll·n_envs`), so the totals are directly
comparable. The split:

- `baseline`, `onpolicy`, `beta` — the whole **1.8e7** on DAgger imitation.
- `ppo` — **1.2e7 DAgger + 6.0e6 PPO = 1.8e7** total.

Two confounds to read the table against:

- **Comparability confound.** The `ppo` arm gets fewer *imitation* steps (1.2e7 vs 1.8e7) than the
  other three, so a lower score could be less imitation rather than PPO not helping. Noted, not
  corrected (equal total steps is the chosen equalizer).
- **PPO undertraining risk.** 6.0e6 steps of PPO against a reward that is dense in progress but
  **sparse in the gate/finish bonuses that define completion** may not converge. If `ppo`
  underperforms, read it as *undertraining* (as UC-79 flagged for its 30 M-step PPO), not as the
  method failing — the follow-up is a denser completion-shaped reward or a longer PPO budget.

The `onpolicy` lever has **no teacher-forcing warmup** (execution is student-driven from step 0), so
it relies on the warm-started actor to bootstrap; pairing on-policy execution with the `beta` β-floor
(teacher correction that never zeroes) is the natural **combined** follow-up if either lever alone
moves completion.

## Running it (AC6) — artifacts to copy and the command

Runnable on any CUDA machine with **no Liftoff installed**. Copy onto the run machine:

1. **This repository**, installed so `drone-fly` is on `PATH` (`uv sync` or `pip install -e .`).
2. **The teacher run** `teacher_it1374` — the frozen teacher the student distills from. Pass its
   run folder (or run id under `<data root>/models/teacher/`) as `TEACHER`.
3. **The detector noise model**:
   `artifacts/models/detector-v1/liftoff-gates-161f7a83/noise_model.yaml` — pass its path as `NOISE`.
4. **The Liftoff track-conversion cache** under `artifacts/tracks/cache/` so the open courses resolve
   without Liftoff. If it is absent, rebuild it once from the trackdata (`drone-fly tracks …`) on a
   machine that has the trackdata; the cache is then portable.

Then run the resumable driver (trains baseline + 3 variants back-to-back, each writing the UC-77
final_report):

```bash
TEACHER=<teacher_it1374 run folder or id> \
NOISE=artifacts/models/detector-v1/liftoff-gates-161f7a83/noise_model.yaml \
DEVICE=cuda \
scripts/uc80_distillation_fix.sh
```

It is **reboot-safe**: re-running skips completed variants (those with `eval/student-held-out.json`),
resumes an interrupted one from its run folder (the step budget is cumulative), and only then
continues — it never restarts the sequence. All four share the equal total step budget (the
across-variant equalizer).

## Comparison table (AC7)

Rows: teacher, baseline (current distillation), the three variants, and the UC-77 + UC-79
(`teacher_full`) references. Columns are the AC5 metrics — held-out `macro_completion` (the
**generalisation target**) and Infinity Loop completion (`liftoff:liftoff-arena-04`, a train-split
**check**), each with exact and noise perception, plus the held-out outcome breakdown
(floor / collision / out-of-bounds / gate_lost / timeout), all at 32 episodes per track, seed 1234,
obstacles `none`.

Fill the four variant rows after the runs finish from each run's
`eval/student-held-out.json` and `eval/student-liftoff-arena-04.json` (`macro.student_noise` /
`macro.student_exact`; held-out outcomes from `outcomes.student_noise`). The teacher, UC-77 and
UC-79 reference rows below are known; the baseline and three variant rows are placeholders until the
GPU run lands.

| Run | Held-out (noise) | Held-out (exact) | Infinity (noise) | Infinity (exact) | Held-out outcomes noise (floor/coll/oob/lost/timeout) |
|---|---|---|---|---|---|
| teacher (exact) | — | 17.1% | — | 93.8% | — |
| baseline | 0.0% | 0.0% | 0.0% | 0.0% | 18.5% / 22.9% / 43.5% / 10.0% / 0.0% |
| onpolicy | 0.0% | 0.0% | 0.0% | 0.0% | 14.0% / 29.2% / 42.3% / 10.2% / 0.0% |
| beta | 0.0% | 0.0% | 0.0% | 0.0% | 48.8% / 10.6% / 19.4% / 19.2% / 0.0% |
| ppo | 0.0% | 0.0% | 0.0% | 0.0% | 18.1% / 26.7% / 40.2% / 8.8% / 0.0% |
| UC-77 student (MLP, ref) | 1.0% | 0.0% | 0.0% | 0.0% | — |
| UC-79 `teacher_full` (ref) | 0.6% | 1.0% | — | — | 20.6% / 51.2% / 2.9% / 8.3% / 4.0% |

## Recommendation

**Result (run 2026-10-16, RTX A3000, equal 1.8e7-step budget): negative — no single lever clears the
bar.** All four runs completed **0.0% held-out (noise and exact)** and **0.0% Infinity**, versus the
teacher's 17.1% / 93.8%. None reach the ≥5% success threshold (AC8 negative branch). Single-lever
fixes to the distillation — on-policy relabelling, a gentler β-floor, or a PPO completion-reward
fine-tune — do **not** by themselves move the student off zero at this shortened budget.

What the runs *do* show (the levers change behaviour, just not completion):
- **`onpolicy`** — indistinguishable from baseline (held-out outcomes ≈ 14/29/42/10). Making the
  teacher label student-visited states, on its own, did not help: the student still has no policy that
  survives its own trajectory.
- **`beta`** (end-β 0.15, over the whole budget) — the clearest behavioural shift: sustained teacher
  correction cut out-of-bounds (43.5% → 19.4%) and collisions (22.9% → 10.6%), but the drone now
  **sinks into the floor** (18.5% → 48.8% held-out; 81% on Infinity). It stays in-course near gates
  but cannot hold altitude — a control/stability failure, not a navigation one. Consistent with the
  UC-79 reading.
- **`ppo`** — the PPO fine-tune *was* learning its objective (reward/step climbed 0.237 → 0.288 and
  was still rising at the budget end) but produced no completions. This is **undertraining**, exactly
  the flagged risk of the 1.2e7 + 6e6 split — not evidence PPO fails. Its held-out mix (18/27/40/9) is
  near baseline because PPO only touched the last third of the budget.

**Recommendation / next lever.** The single-lever null result points at **combination + depth**, not a
different information source (UC-79 already ruled out observation):
1. **Combine `onpolicy` + `beta`'s β-floor** — student-driven execution that *still* receives teacher
   correction on the states it visits (neither alone sufficed; together they attack covariate shift
   and stability jointly).
2. **Give PPO a real budget and a denser completion-shaped reward** — the reward was still climbing;
   extend PPO well beyond 6e6 steps and/or shape altitude-hold / gate-approach explicitly, since the
   dominant residual failure under teacher correction is floor-sinking.
3. Then, if a combination clears ≥5% held-out, apply the winner to the K1 connectome actor (separate
   follow-up). Scope here stayed MLP-student on obstacle-free `student-liftoff-open`.

The budget is the shortened 1.8e7 equaliser; a longer-budget re-run of the combined fix is the cheap
confirmation before concluding the distillation can't be salvaged by these levers.
