# Experiments ledger

One unified, sequentially-numbered timeline for the whole campaign so the order — and which one is latest — is unambiguous. Numbers 01–10 are the committed model milestones (dirs renamed to `NN-<slug>` in creation order); 11+ are the ongoing experiment runs. When an experiment beats the current best honestly it is promoted to the next committed model dir (`training/NN-<slug>/`).

**Naming convention**
- Committed model dir:    `training/NN-<slug>/`  (numbered in creation order; next = 11)
- Experiment launcher:    `gpu_prototype/expNN_<slug>.py`
- Working checkpoint:      `gpu_prototype/expNN_<slug>.pt` (scratch; promoted into the model dir on a win)
- `Honest?` = did completed runs avoid barging (terminal-contact held)? A higher % that cheats does NOT count — audit with `verify_proc.py`.

**Current best (committed):** `10-proc-obstacle-avoid` — 33–36% honest deterministic, 0% barging. Warm-start ckpt `gpu_prototype/proc_shapeA_term.pt`.

## Committed models (01–10, creation order)

| # | dir | created | what it is | result |
|---|-----|---------|-----------|--------|
| 01 | 01-acro-gate-flight | 2026-10-09 13:27 | acro/CTBR dynamics, 3 straight gates | 100% |
| 02 | 02-k1-gate-flight | 2026-10-09 15:02 | frozen-K1 reservoir, 3 straight gates | 100% |
| 03 | 03-oval-lap-flight | 2026-10-09 17:35 | closed oval lap (same-direction turns) | 99–100% |
| 04 | 04-figure8-chicane-flight | 2026-10-09 18:48 | randomized figure-8 / chicane per episode | 96% |
| 05 | 05-pad-lap-flight | 2026-10-09 21:35 | oval lap + battery + charging pad | 99% |
| 06 | 06-obstacle-lap-flight | 2026-10-09 22:33 | oval lap + pillar obstacles (weave) | 99% |
| 07 | 07-proc-course-flight | 2026-10-10 02:18 | procedurally random gate courses | 96% |
| 08 | 08-multilap-course-demo | 2026-10-10 13:19 | proc readout flying full multi-lap tier (demo) | demo |
| 09 | 09-proc-obstacle-fail | 2026-10-10 13:19 | diagnostic recording of the obstacle failure | diag |
| 10 | 10-proc-obstacle-avoid | 2026-10-10 18:27 | proc courses + on-path obstacles (SOLVED) | 33% honest |

## Experiment runs (11+)

**Key diagnosis (Phase 0, 2026-10-10, corrected):** failures break down as **84% obstacle collision, 15% floor crash, 0% ceiling/arena, 0% timeout** — and **100% of obstacle-collision deaths occur on courses with 2+ obstacles**. (An earlier pass wrongly reported "100% crashes": `env.step()` respawns dead envs internally, so reading `env.pos` after `step()` gave post-respawn positions and mislabeled obstacle hits. Fixed by stashing exact in-step `cause_*` flags in the env.) Mean death tilt 69°. 8+ gate courses fail more (28%) than 5–7 (44%); wide gates easy (65%), tight/mid ~32–35%. **The wall is multi-obstacle avoidance**, not single obstacles, stability, or tight gates. Prioritize count curriculum + gap-steering.

| # | slug | date | hypothesis / change | launcher | ckpt | Result (det %) | Honest? | Outcome |
|---|------|------|---------------------|----------|------|----------------|---------|---------|
| — | (baseline) | 2026-10-09 | clearance shaping + terminal + warm aggressive start | train_shaping.py | proc_shapeA_term.pt | 33–36% | yes (0% barge) | committed as 10-proc-obstacle-avoid |
| 11 | count-curriculum | 2026-10-10 | obstacle-count curriculum 1→N (count_curr_frac=0.5), resume baseline | exp11_count-curriculum.py | exp01_count.pt | _running_ | — | — |
| 12 | gap-steering | 2026-10-10 | gap-steering reward PROC_GAP (fly along clearest gate-ward cone dir), resume baseline | exp12_gap-steering.py | exp02_gap.pt | _running_ | — | — |
