# Experiments ledger

Sequentially-numbered experiment runs for the proc-obstacle-avoidance campaign (and beyond).
Each training run gets the next `expNN` number so the order — and which one is latest — is unambiguous.

**Naming convention**
- Output recordings dir:  `training/expNN-<slug>/`
- Checkpoint:             `gpu_prototype/expNN_<slug>.pt`
- One row per run below. Fill `Result` after `verify_proc.py` + `diagnose_proc.py`.
- `Honest?` = did completed runs avoid barging (terminal-contact held)? A higher % that cheats does NOT count.
- Promote a winner to a descriptively-named committed dir (e.g. `proc-obstacle-avoid`) only when it beats the current best honestly.

**Current best (committed):** `proc-obstacle-avoid` — 33–36% honest deterministic, 0% barging. Warm-start ckpt `gpu_prototype/proc_shapeA_term.pt`.

**Key diagnosis (Phase 0, 2026-10-10, corrected):** failures break down as **84% obstacle collision, 15% floor crash, 0% ceiling/arena, 0% timeout** — and **100% of obstacle-collision deaths occur on courses with 2+ obstacles**. (An earlier pass wrongly reported "100% crashes": `env.step()` respawns dead envs internally, so reading `env.pos` after `step()` gave post-respawn positions and mislabeled obstacle hits. Fixed by stashing exact in-step `cause_*` flags in the env.) Mean death tilt 69°. 8+ gate courses fail more (28%) than 5–7 (44%); wide gates easy (65%), tight/mid ~32–35%. **The wall is multi-obstacle avoidance**, not single obstacles, stability, or tight gates. Prioritize count curriculum + gap-steering.

| exp | slug | date | hypothesis / change | base recipe | ckpt | Result (det %) | Honest? | Outcome |
|-----|------|------|---------------------|-------------|------|----------------|---------|---------|
| —   | (baseline) | 2026-10-09 | clearance shaping + terminal + warm aggressive start | train_shaping | proc_shapeA_term.pt | 33–36% | yes (0% barge) | committed as proc-obstacle-avoid |
| 01  | count-curr | 2026-10-10 | obstacle-count curriculum 1→N (count_curr_frac=0.5), resume baseline | exp01_count_curr.py | exp01_count.pt | _running_ | — | — |
| 02  | gap | 2026-10-10 | gap-steering reward PROC_GAP (fly along clearest gate-ward cone dir), resume baseline | exp02_gap.py | exp02_gap.pt | _running_ | — | — |
