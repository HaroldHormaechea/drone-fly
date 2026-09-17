---
plan_for: (free-form task)
work_branch: feat/add-n-envs-flag
team: drone-fly
approved: 2026-09-17
---

# Implementation Plan — Add `--n-envs` CLI flag to `train`

**Task:** Add a `--n-envs` CLI flag to the `train` command that overrides `TrainConfig.n_envs` (default 1, behavior unchanged unless passed), mirroring `--timesteps`. Update README/help. Must interoperate with `--resume`.

**Status:** APPROVED by challenger (round 1). Guard adopted.

### Analysis
`train` exposes `--timesteps` (overrides `cfg.total_timesteps`) but `n_envs` (parallel PPO rollout envs; default 1, `config.py:45`) is only settable via `TrainConfig`. Add `--n-envs` mirroring `--timesteps` so users can raise parallelism to use more CPU cores. Behavior must be byte-identical when the flag is omitted.

**Mirror pattern (`--timesteps`):** declared `cli/__init__.py:111` (`type=int, default=None`), threaded at `cli/__init__.py:284` (`total_timesteps=args.timesteps`), applied in `train()` as a **local, no cfg mutation** — `loop.py:145`: `steps = total_timesteps if total_timesteps is not None else cfg.total_timesteps`.

**`n_envs` consumption (full-train):** `loop.py:195` `build_vec_env(..., n_envs=cfg.n_envs)`; `loop.py:225` `save_freq=max(cfg.checkpoint_freq // max(cfg.n_envs,1),1)`. `loop.py:288` is `smoke_train` only — out of scope, untouched.

**Design decision (challenger-verified):** `--n-envs` flows as a `train()` keyword param, resolved to a local `resolved_n_envs = n_envs if n_envs is not None else cfg.n_envs` beside line 145 — the truest mirror of `steps`. Rejected: (a) mutating `cfg.n_envs` (silent side-effect on caller cfg, unlike `--timesteps`); (b) constructing a `TrainConfig` in the CLI (CLI passes `cfg=None` today; would break symmetry).

### Proposed Solution
1. **`src/drone_fly/cli/__init__.py`**
   - `_add_train_args` (after `--timesteps`, ~line 111): add `--n-envs`, `type=int, default=None`, help text: overrides `TrainConfig.n_envs` (default 1) to raise parallel rollout envs / use more CPU cores. argparse maps `--n-envs` → `args.n_envs`. **`train` parser only** (not smoke-train).
   - `main()` train dispatch (~278–290): add `n_envs=args.n_envs,` to the `train(...)` call, next to `total_timesteps=args.timesteps`.
   - **Validation guard (adopted from challenger):** in `main()`, before dispatch, reject `n_envs < 1` with a clean argparse error rather than a cryptic deep-SB3 vec-env failure. Recommended form: `if args.n_envs is not None and args.n_envs < 1: build_parser().error("--n-envs must be >= 1")` (or an equivalent `type=` positive-int callable). Keeps `save_freq` safe too (already `max(...,1)`-guarded).
2. **`src/drone_fly/train/loop.py`**
   - `train()` signature: add keyword-only `n_envs: int | None = None` after `total_timesteps`; add a docstring entry mirroring `total_timesteps`.
   - After line 145: `resolved_n_envs = n_envs if n_envs is not None else cfg.n_envs`.
   - Line 195: `n_envs=cfg.n_envs` → `n_envs=resolved_n_envs`.
   - Line 225: `cfg.n_envs` in the `save_freq` division → `resolved_n_envs`.
   - Recording backend (`venv.get_attr("backend")[0]`, ~242) reads env-0 only — unaffected, no change.
3. **`README.md`**
   - Add `--n-envs` to the train flag docs (train examples at lines 14, 127, 141, 185, 357). Add a one-line note in the Flight-training section: purpose (CPU parallelism / throughput), default 1 (unchanged unless passed), safe with `--resume`.

### Files Affected
**Production code (developer):**
- `src/drone_fly/cli/__init__.py`
- `src/drone_fly/train/loop.py`
- `README.md` (doc; developer-owned)

**Test code (qa):**
- `tests/test_cli.py` — parser test: `--n-envs 4` → `args.n_envs == 4`; omitted → `None` (mirror `test_parser_train_flags`, line 27). Dispatch test: monkeypatch `drone_fly.train.loop.train` (correct patch point — `main()` does a call-time local import) to capture kwargs, assert `n_envs` forwarded. **Guard test:** `--n-envs 0` → `SystemExit` (locks the validation contract).
- `tests/test_train_resume.py` — loop-level: passing `n_envs=N` to `train()` overrides `cfg.n_envs` at the consumption sites (inspect `build_vec_env` receives `N` and/or `save_freq` uses `N`). Optional `--n-envs 2` resume smoke on the `simple` adapter confirming resume tolerates a changed env count.

### Risks & Considerations (challenger-verified)
1. **Resume:** `PPO.load(resume, env=venv)` sets `data["n_envs"] = venv.num_envs` **before** `_setup_model` rebuilds the rollout buffer, so a checkpoint trained at a different `n_envs` resumes with a correctly-sized buffer; the following `set_env(venv)` is harmless. Batch validity holds — buffer `2048*n_envs` always divisible by `batch_size` 64. VecNormalize stats are per-feature (env-count independent). No real resume risk.
2. **`save_freq`** stays at ~`checkpoint_freq` total env-steps regardless of parallelism — intent preserved.
3. **Validation** now handled by the guard (issue closed).
4. **Scope:** just this flag — `train` parser + `train()` + README + tests. No smoke-train flag, no consumer refactor, no evaluate change.

## Challenger verdict (round 1): APPROVE
Mirrors `--timesteps` cleanly (local, no cfg mutation), default-unchanged guaranteed (None-guarded), resume interop verified, scope tight. Minor `n_envs>=1` guard recommendation adopted into the plan.
