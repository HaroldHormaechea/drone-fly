---
plan_for: use-cases/63-tie-desktop-app-to-reservoir-runs.md
work_branch: feat/uc-63-tie-desktop-app-to-reservoir-runs
run: drone-fly-uc-63
approved: 2026-10-11
---

# UC-63 implementation plan — restore `app/` and repoint it at the reservoir pipeline

Approved by the challenger in round 2 of the analyst↔challenger loop. `<T>` = the project root (`C:\dev\drone-fly-uc-63-tie-desktop-app-to-reservoir-runs` in this run's worktree).

Run-level user decisions (binding):
- **Write scope:** the developer may write `src/drone_fly/**` **and** `app/**` (including `app/README.md`) for this run. QA writes `tests/**`. `gpu_prototype/**`, `training/**` and `viz/**` are not modified.
- **AC9** (EXPERIMENTS.md registration + `save_readout.py` promotion) is **deferred**. It is documented as out of scope, not built.
- **Risk 1** (`pyproject.toml` / `uv.lock`) is resolved by the user at plan preview. See the "User decisions at plan preview" section at the end.

### Analysis
At `cc817f4~1`, `app/` contained:
- `__init__.py` and `__main__.py` (the pywebview launcher plus `_window_icon_path`).
- `server.py` (FastAPI).
- `runs.py` (`RunRegistry` with injected spawn/signaler/probe; process-group spawn with stdout written to `training/<name>/logs/app.log`).
- `configs_io.py` and `field_help.py` (an SB3 YAML schema built from `drone_fly.config`).
- `status.py` (SB3 `status.jsonl` / `progress.csv`).
- `logs.py` (bounded tail plus a noise filter, stdlib only).
- `recordings.py` (`episode_<n>.json[.gz]` listing plus server-side gunzip, stdlib only).
- `static/` (`index.html`, `app.js`, `forms.js`, `modal.js`, `viewer-embed.js`, `app.css`, `fly.ico`/`fly.png`/`fly.svg`).

The SB3 imports are in `server.py`, `runs.py`, `configs_io.py` and the old `test_app_runs.py`. `logs.py`, `recordings.py` and `__main__.py` are clean.

Reservoir facts (verified in `gpu_prototype`):
- **`train_aug` signature** (`reservoir_aug.py:102`): `N=512`, `budget=20e6`, `roll=32`, `epochs=4`, `logstd_final=-2.0`, `anneal_start=0.4`, `warm_from=None`, `warm_src_extra_dim=12`, `logstd_init=-0.5`, `pos_curr_frac`/`count_curr_frac=None`.
- **Progress lines:** every 20 iterations it prints `[aug] it N steps XM fps F ep_rew/step r tilt Tdeg <env.metric()>`. For `BatchedProcCourse` the metric is `compl X% lap Y`; it may be absent for other envs. The final line is `[aug] saved <path>`. All prints use `flush=True`.
- **Checkpoints:** `save_path` is overwritten every 200 iterations **and** at the end, so the file's presence alone cannot distinguish a partial checkpoint from a final one.
- **Launchers:** exp11/12/13 and `train_shaping` take argv `<save> [budget]` (default `14e6`). Every other knob is hardcoded, including `warm_from` as an absolute `/workspace/drone-fly/...` base. No existing script forwards generic `train_aug` kwargs.
- **Env flags:** `gates_proc_env` reads them as `bool(os.environ.get(X, ""))`, so `"0"` counts as **on**. They are read at import time, so `verify_proc`, `diagnose_proc` and `record_aug` need the run's same flags.
- **Audit scripts:** `verify_proc.py <ckpt> [eps]` and `diagnose_proc.py <ckpt> [eps]` seed from `AUDIT_SEED` (default 1234) and print fixed-format lines. `verify_proc` prints the `among COMPLETED runs: barged ...` line only when `completed > 0`.
- **`record_aug.py`** `<course> <ckpt> <out_name> [n_ep=10]` writes gzipped files to a hardcoded `/workspace/drone-fly/training/<out_name>/recordings`.
- **Hardcoded root:** `K1_PATH` in `reservoir_aug` is hardcoded to `/workspace/drone-fly` as well.
- **Honesty definition** (`EXPERIMENTS.md`): "did completed runs avoid barging". Honest runs show 0% completed-barging at 1.9–2.7 m/s and 14–26° tilt. With `PROC_OBS_TERMINAL` on, the overall barging % equals the collision-death rate and is not a cheat signal.

### Proposed Solution
Restore every `app/` file from `cc817f4~1`, then make these changes:
- Delete `app/field_help.py`.
- Remove the Slices surface (`drone-fly prune` no longer exists).
- Remove Pause/Resume (`train_aug` has no signal trap and no resume).

1. **`<T>\app\configs_io.py`** (rewritten; the name is kept because AC1 names it). Stdlib only.
   - `RECIPES` table: `exp11_count-curriculum`, `exp12_gap-steering`, `exp13_count-gap`, `train_shaping`, and `train_aug` (generic, which runs `app/reservoir_launch.py`).
     - Each entry records the launcher path, default flags taken from each launcher's docstring "Run:" line, and the default budget (`14e6` for named recipes, `20e6` for generic).
     - It also records the **fixed** hardcoded knob values, including the `warm_from` base, marked `editable=false`.
   - `KNOBS` descriptors: `N`, `budget`, `roll`, `epochs`, `logstd_init`, `logstd_final`, `anneal_start`, `count_curr_frac`, `pos_curr_frac`, `warm_from`, and `warm_src_extra_dim`.
     - `warm_src_extra_dim` is an optional int, blank by default, and enabled only when `warm_from` is set.
     - Generic placeholders equal `train_aug`'s signature defaults.
   - `FLAGS`: the five `PROC_*` booleans (`TRACTABLE`, `RAYCAST`, `SHAPING`, `OBS_TERMINAL`, `GAP`).
   - `describe_schema()` reports which knobs are editable per recipe. Named recipes allow only `budget`, `save_path` and flags; everything else renders read-only as "set by <launcher>". `warm_from` is editable only on the generic recipe.
   - `validate_run_name`: `^[A-Za-z0-9][A-Za-z0-9._-]*$`, with no `..`.
   - New exception **`RecipeError`**. It is deliberately not named `ConfigError`, so the AC1 grep stays clean.
   - Per-run metadata lives in `training/<name>/run.json`: `{name, recipe, flags, knobs (editable ones only), save_path, warm_from}`.
   - Explicit Save overwrites the file, with no history (UC-61 AC9 is preserved).
   - Validation rules:
     - recipe and knob names must be known; types are coerced; fractions must be in `(0, 1]`;
     - `save_path` must end in `.pt` and sit inside the project root;
     - `save_path` must not be inside a committed or external dir, and must not collide with another run's `save_path`.
   - The default `save_path` is `training/<name>/model.pt` (gitignored). The run layout is `training/<name>/` for `run.json`, logs and recordings, which matches `record_aug`'s `out_name`.

2. **`<T>\app\runs.py`** (rewritten `RunRegistry`). `spawn`, `signaler`, `probe`, `settings_reader`, `tracked_probe`, `gpu_probe` and `clock` are all injected.
   - **Interpreter:** Settings provides `micromamba_executable` (default `"micromamba"`, resolved via `shutil.which`) and `conda_env` (default `"dronefly"`). Both are re-read on every launch, audit or record.
   - **Probe:** `<mm> run -n <env> python -c 'import torch'`. The verdict is memoised per `(exe, env)`. If micromamba is missing or the probe fails, raise `RunError` with an actionable message and never fall through silently.
   - **Argv:**
     - Named recipes: `[mm, run, -n, env, python, <abs gpu_prototype/launcher>, <abs save_path>, <budget>]`.
     - Generic: `[..., python, <abs app/reservoir_launch.py>, --save-path X]`, plus `--<knob>` only for the knobs the user set.
     - CWD is `<root>/gpu_prototype`.
   - **Env:**
     - Start from a copy of the app's own environment.
     - **Strip** every inherited `PROC_*` and `AUDIT_SEED`.
     - Set only the selected flags to `"1"`, **never** `"0"`.
     - Add `PYTHONUNBUFFERED=1`.
     - The env is passed to the spawner so tests can see it.
   - **Default spawner:** same as UC-61 (new process group / `start_new_session`, stdout+stderr appended to `training/<name>/logs/app.log`), plus an env argument. The signaler is the same as UC-61 (`CTRL_BREAK`, or `killpg SIGTERM`).
   - **States:**
     - `running`, `stopping`, `stopped`.
     - `finished`: `app.log` contains `[aug] saved <save_path>`.
     - `failed`: the process exited non-zero, or exited 0 without the saved line.
     - `partial`: the `.pt` exists but there is no saved line. It is labelled "partial checkpoint (auditable, not resumable)".
     - `ready`: `run.json` exists but there is no `.pt`.
     - `committed`: a `training/<dir>` with no `run.json` that either matches `^\d{2}-` or is git-tracked. The git check uses `tracked_probe`, which defaults to `git ls-files --error-unmatch` run against the project root, returns `False` when git is unavailable, and is always faked in tests.
     - `external`: any other dir without `run.json`, such as a pre-app CLI run. It is read-only, the same as `committed`.
     - On backend restart, state is reconciled from disk using the saved-line rule.
   - **Lifecycle:**
     - `launch(name, overwrite=False)`: if the `.pt` already exists the call returns 409 unless `overwrite` is set, because `train_aug` restarts from scratch.
     - `stop(name)`.
     - `delete(name)`: the UC-61 guards stay (name validation, refuse a live run, realpath containment), and committed/external dirs are also refused. Only `training/<name>/` is removed; an outside `save_path` is reported, not deleted.
     - `record(name, episodes=10)`: spawns `record_aug.py proc <save_path> <name> <n>` with the run's flags.
   - Audit and record are refused (409) while the run is `running` or `stopping`, because the checkpoint is rewritten every 200 iterations. Record is also refused on committed/external dirs.
   - **`audit_eligible`:** requires `model_readout.pt` (committed/external) or `save_path` (app runs). If missing, the reason is "no model_readout.pt" (01, 08 and 09 today).
   - **GPU probe:** `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits`, best-effort, returning `None` on failure. It is cached for `GPU_PROBE_TTL_S=5` using the injected clock, and `None` results are cached too.

3. **`<T>\app\status.py`** (rewritten). A pure parser over `app.log`.
   - `parse_aug_line` returns `{it, msteps, fps, ep_rew_per_step, tilt_deg, compl_pct|None, lap|None}`, or `None` for malformed or partial lines. It never raises, and the compl/lap group is optional.
   - It also recognises `[aug] saved <path>` and the header lines (`K1ReservoirAug: feat_dim...`, `warm-started from ...`).
   - `progress_snapshot(root, name, since)` uses a line cursor over complete lines only (the torn-tail rule) with bounded reads, and returns `{latest, points, saved, next}`.

4. **`<T>\app\audits.py`** (new).
   - **Running:** `verify_proc.py` and `diagnose_proc.py` run through the registry spawner with the audit's flags (pre-filled from `run.json` and editable; required for committed models), `AUDIT_SEED` (Settings, default 1234) and `eps` (default 512).
   - **Output:** goes to `<root>/.dev-app/audits/<name>/<ts>-<verify|diagnose>.log`, plus a sidecar `<ts>-<kind>.json` recording `{flags, seed, eps, exit_code}`. It is never written into `training/`, so committed dirs are untouched. Results come from the filesystem and survive restarts.
   - **Parsers:**
     - `parse_verify`: completion n/eps/pct, laps required, gate passes, `barging_pct`, `avg_inside_steps`, `min_clearance_m`, `mean_speed`, `mean_tilt_deg`, `mean_steps_alive`, and `completed_only {barged_pct, avg_inside_steps, mean_speed}` or `None`.
     - `parse_diagnose`: overall completion, the obstacle-count, tightest-gate and gate-count tables, and the failure anatomy.
     - Both are tolerant: a missing line gives `None`, and they never raise.
   - **`honesty_verdict(parsed, audit_flags)`**, an **ordered** rule table:
     - (a) Any of the completion, BARGING or HOVERING lines missing → `unverified`, and stop.
     - (b) Hover rules, regardless of completion: `mean_speed < HOVER_SPEED_MS=0.5` → `dishonest` ("hovering"); `mean_tilt < HOVER_TILT_DEG=5` → `suspect`.
     - (c) If the audit flags lack `PROC_OBS_TERMINAL`: overall barging `> OVERALL_BARGE_NONTERMINAL_PCT=2` → `suspect`. With `OBS_TERMINAL` on, overall barging is shown only, with a tooltip.
     - (d) Only when `completed > 0`: the among-COMPLETED line must be present, otherwise `unverified`. Completed-barged `> COMPLETED_BARGE_DISHONEST_PCT=2` → `dishonest`; above 0 and at most 2 → `suspect`.
     - (e) `completed == 0` with no reasons from (b) or (c) → `nothing-completed`. A missing among-COMPLETED line is legitimate here.
     - (f) Otherwise → `honest`.
     - Reasons accumulate. The final verdict is the most severe: `dishonest > unverified > suspect > nothing-completed > honest`.
   - A non-zero audit exit gives state `failed` and verdict `unverified` with the reason "audit process exited <code>". A partial parse is shown only under the failed banner, together with the log tail.
   - The API always bundles completion together with the verdict and the cheat checks.

5. **`<T>\app\reservoir_launch.py`** (new). The generic `train_aug` launcher.
   - A pure argparse/resolver with no torch import.
   - `main()` inserts `gpu_prototype` on `sys.path`, then imports `BatchedProcCourse`, `EXTRA_DIM` and `train_aug`.
   - It calls `train_aug(BatchedProcCourse, EXTRA_DIM, save_path=..., **only the knobs the user set**)`.
   - If `warm_from` is set and `warm_src_extra_dim` is blank, it uses the runtime `EXTRA_DIM`.
   - It contains zero training, env or reward logic. The challenger accepted it under AC10.

6. **`<T>\app\server.py`** (rewritten; same `create_app(project_root, registry=...)` seam).
   - `/` and the static mount stay unchanged; `/viewer` serves `viz/` unchanged.
   - `GET/POST /api/settings`: `project_root`, `host`, `port`, `micromamba_executable`, `conda_env`, `audit_seed`, `audit_eps`.
   - `GET /api/preflight` returns warnings: micromamba found, env probe result, `project_root == /workspace/drone-fly`, and whether the named-recipe base `.pt` files exist.
   - `GET /api/schema`.
   - `GET /api/runs` and `GET /api/runs/{name}`.
   - `GET/POST /api/runs/{name}/config`.
   - `POST /api/runs/{name}/launch` (`?overwrite`) and `POST /api/runs/{name}/stop`.
   - `DELETE /api/runs/{name}`.
   - `GET /api/runs/{name}/status` (polling), plus a best-effort SSE `/status/stream`.
   - `GET /api/runs/{name}/logs` uses `logs.tail`.
   - `POST/GET /api/runs/{name}/audits`.
   - `POST /api/runs/{name}/record`.
   - Recordings routes stay as before.
   - Error mapping: `RunError` → 409, `RecipeError` → 400, `FileNotFoundError` → 404.

7. **Restored verbatim:** `app/__main__.py`, `logs.py`, `recordings.py`, and `static/index.html`, `modal.js`, `viewer-embed.js`, `app.css`, `fly.ico`, `fly.png`, `fly.svg`. `app/__init__.py` is restored with an updated docstring.

8. **`app/static/app.js` and `forms.js`** (restored, then adapted).
   - Remove Slices and the YAML omit-vs-default logic. Keep the hash router, nav, Settings, explicit Save, toast and modal.
   - `#/train`: the run list, including committed and external dirs with their badges.
   - `#/train/new` and `#/train/<name>/config`: recipe, flag checkboxes, typed knobs (editable or fixed, read-only), `save_path`; Save and Save&Launch, with an overwrite confirm.
   - `#/train/<name>/status`: it, Msteps, fps, ep_rew/step, tilt, completion EMA ("—" when absent), curves, a GPU line, the log tail, and Stop.
   - `#/train/<name>/audits`:
     - a verify/diagnose form;
     - completion is rendered **only** inside a verdict panel: honest is neutral/green, suspect is amber, and dishonest/unverified/failed get a red "Not a success: <reasons>" banner;
     - the panel shows barging, completed-only barging, clearance, speed and tilt, plus the diagnose tables.
   - `#/train/<name>/recordings`: the unchanged viewer embed, plus a "Record N episodes" button (default 10).
   - "Continue from this checkpoint": a pure-UI action that pre-fills `#/train/new` with recipe `train_aug`, `warm_from` = the `.pt`, the source run's flags, a suggested name, and `warm_src_extra_dim` left blank. It is offered on finished/partial app runs and on committed/external dirs that have `model_readout.pt`.
   - `#/settings` and `#/about`.

9. **`<T>\app\README.md`** (rewritten).
   - Install with `uv sync --extra app`; run `python -m app`, or `--no-window`.
   - Preflight requirements: micromamba, the `dronefly` env, and the `/workspace/drone-fly` root.
   - The reservoir workflow, the state meanings (partial = auditable, not resumable), and the honesty-verdict rules and thresholds.
   - Committed and external dirs are read-only.
   - An owner-eyeball checklist for the native window and front-end.
   - An explicit "Deferred / out of scope: AC9 (EXPERIMENTS.md registration + save_readout.py promotion)" section.
   - All SB3 instructions removed.

### Files Affected
**Production code (developer):**
- `<T>\app\__init__.py`, `__main__.py`, `logs.py`, `recordings.py` (restore).
- `<T>\app\configs_io.py`, `runs.py`, `status.py`, `server.py` (rewrite).
- `<T>\app\audits.py`, `reservoir_launch.py` (new).
- `<T>\app\static\{index.html, modal.js, viewer-embed.js, app.css, fly.ico, fly.png, fly.svg}` (restore).
- `<T>\app\static\{app.js, forms.js}` (restore and adapt).
- `<T>\app\README.md` (rewrite).
- `app/field_help.py` is **not** restored.
- Subject to the user decision on Risk 1: `<T>\pyproject.toml` and `<T>\uv.lock`.

**Test code (QA, `tests/**`):**
- Restore verbatim: `test_app_logs.py`, `test_app_recordings.py`, `test_app_window_icon.py`.
- Rewrite `test_app_configs_io.py`:
  - schema and editable/fixed knobs;
  - `run.json` round-trip;
  - validation (committed/external dirs, `save_path` collisions, names);
  - an AST drift guard that parses `exp11/12/13`/`train_shaping.py`, `train_aug`'s signature, and `reservoir_launch`'s forwarded-knob set without importing them.
- Rewrite `test_app_runs.py`:
  - argv and cwd per recipe;
  - env assembly (off flags absent, never `"0"`; inherited `PROC_*`/`AUDIT_SEED` scrubbed; `PYTHONUNBUFFERED` set);
  - micromamba missing or probe failure → `RunError`;
  - state transitions with fake exit codes, and restart reconciliation via the saved line;
  - the overwrite guard and delete guards;
  - committed/external classification with a fake `tracked_probe` (CI never shells out to git);
  - the `audit_eligible` reason;
  - 409 for audit/record on live runs;
  - record argv (default 10);
  - GPU TTL caching with a fake clock.
- Rewrite `test_app_status.py`: full line, no compl group, negative reward, malformed/partial lines, the saved line, the cursor/torn tail.
- Rewrite `test_app_server.py`: `TestClient` with an injected fake registry; every route; the 400/404/409 mapping; the audit payload always bundles completion with the verdict.
- New `test_app_audits.py`:
  - the full precedence table, one case per branch, plus a multi-reason severity-order case;
  - hovering with 0 completions → `dishonest`;
  - the `OBS_TERMINAL`-dependent overall rule;
  - non-zero exit with a traceback → `failed`/`unverified`;
  - argv, env and seed via a fake spawner;
  - output lands under `.dev-app/audits`.
- New `test_app_reservoir_launch.py`: the pure resolver only, no torch; unset knobs omitted; `warm_src` resolved only when `warm_from` is set.
- New `test_app_imports.py`:
  - AC1: a subprocess imports `app.server`, `app.runs` and `app.configs_io`, then asserts `torch`, `pybullet` and `stable_baselines3` are not in `sys.modules`;
  - a word-boundary grep over `app/` for `\bConfigError\b`, `\bTrainRunConfig\b`, `\bPruneRunConfig\b`, `drone_fly\.config\b`, `"drone-fly train"`, `"drone-fly prune"` and `vecnormalize` must come back empty, and the test also asserts `RecipeError` is present.
- Not restored: `test_app_status_help.py`.
- New fixtures under `tests/fixtures/app_audit/`, in exact script print formats:
  - `verify_{honest, barging, barging_overall, suspect_1pct, lowtilt, hovering, none_completed, truncated, completed_line_missing, traceback}.txt`;
  - `diagnose_sample.txt`;
  - `aug_log_sample.log`.

### Risks & Considerations
1. **User decision needed (`pyproject.toml` + `uv.lock` are out of scope but required).** CI runs `uv sync --extra dev` with pytest `pythonpath ["src"]`, so CI cannot `import app` and fastapi/httpx are not installed.
   - Recommended: authorize restoring UC-61's lines:
     - fastapi, uvicorn and httpx in the `dev` extra;
     - an `app` extra (fastapi, uvicorn, pywebview);
     - pytest `pythonpath ["src", "."]`;
     - optionally the `drone-fly-app` script and `"app"` in the hatch packages;
     - a regenerated `uv.lock`.
   - Fallback if the user refuses: `tests/conftest.py` adds the repo root to `sys.path`, and the server tests `importorskip` fastapi. CI would then silently skip the server tests, which weakens AC11. Both analyst and challenger advise against it.
2. **Fixed `/workspace/drone-fly` paths.** `gpu_prototype` hardcodes this root (`K1_PATH`, the launcher `BASE` paths, `record_aug`'s output dir), so training, audit and record only work there. This can't be fixed in scope (AC10). Mitigation: a preflight warning and the README; on another root, recordings would land outside the app's view.
3. **`reservoir_launch.py`** is new glue code in `app/`. It is pure arg-forwarding, and the challenger accepted it under AC10.
4. **Named recipes** ignore every knob except `budget` and `save_path`, so those knobs are shown as fixed and read-only, kept honest by the AST drift test. No launcher is modified.
5. **Flag truthiness and inherited-env leakage** are handled in env assembly and tested.
6. **Verdict thresholds** (2% completed-barging, 0.5 m/s, 5°, 2% overall when non-terminal) are judgement calls modelled on `EXPERIMENTS.md`. They are named constants, documented, and fail closed.
7. **No pause/resume:**
   - Stop kills the process group and leaves a partial checkpoint that can be audited; relaunching overwrites it (with a confirm).
   - A run that survives a backend restart can't be re-attached. It is reconciled from disk via the log only.
8. **Committed and external dirs** are protected: no launch, record or delete, and no `save_path` inside them. Audits are written to the gitignored `.dev-app/`. AC10 holds.
9. **AC9 is deferred** per the user's decision and only documented.
10. **Other flags and launchers left out:** `PROC_TUNNELS`/`MULTILAP`/`VARIETY`/`FLOOR`/`NO_OBS` and exp14/16–20 are not exposed, because the UC scopes the 5 flags and 4 recipes. Adding them later is a one-line table change. They are still scrubbed from the inherited env.
11. **Windows Stop:** `CTRL_BREAK` goes to micromamba's process group, and the grandchild python might survive on Windows. On POSIX, `killpg` covers the whole group. This is on the owner-eyeball checklist.
12. **CI lint/format:** CI runs `ruff check` and `ruff format --check` over `app/` as well, so restored files must be lint- and format-clean.

### Additional detail from the challenger's consolidated approved proposal
These details come from the challenger's consolidated version of the same approved proposal. Where they are more specific than the text above, they take precedence.
- **Full `train_aug` signature:** `train_aug(env_cls, extra_dim, N=512, budget=20e6, lr=3e-4, roll=32, epochs=4, mb=8, ent_coef=0.01, logstd_final=-2.0, anneal_start=0.4, save_path=None, warm_from=None, warm_src_extra_dim=12, logstd_init=-0.5, curriculum_frac=None, pos_curr_frac=None, count_curr_frac=None)`.
- **Progress line format:** `[aug] it %4d steps %6.1fM fps %6.0f ep_rew/step %6.3f tilt %5.1fdeg <extra>`, where `<extra>` is `compl %4.1f%% lap %.1f` or empty.
- **Named-launcher fixed values** (`editable=false`):
  - all named launchers: `N=384`, `roll=32`, `epochs=4`, `logstd_init=-0.5`, `logstd_final=-1.0`, `anneal_start=0.5`;
  - exp11/13: `count_curr_frac=0.5`;
  - `train_shaping`: `pos_curr_frac=0.4`, `warm_src_extra_dim=30`;
  - `warm_from` is the hardcoded absolute BASE.
- **Default flags per recipe:**
  - exp11 = `TRACTABLE+RAYCAST+SHAPING+OBS_TERMINAL`;
  - exp12/13 = exp11 + `GAP`;
  - `train_shaping` = `TRACTABLE+RAYCAST+SHAPING`.
- **Audit script defaults:** `verify_proc.py` defaults to `eps=256` and `diagnose_proc.py` to `eps=512`. The app passes `eps` explicitly (Settings `audit_eps`, default 512).
- **Generic argv:** always includes `--budget Y`, plus `--warm-from` when set.
- **Env scrubbing:** covers every inherited `PROC_*`, including `TUNNELS`, `MULTILAP`, `VARIETY`, `FLOOR` and `NO_OBS`.
- **`tracked_probe` and the GPU probe:**
  - `tracked_probe` runs with a timeout and uses the project root as its cwd;
  - every test injects `tracked_probe`, so CI never calls git;
  - the GPU probe uses a short timeout; its default clock is `time.monotonic`.
- **Exit code** comes from `popen.poll()`.
- **Audits page:** its controls are disabled, with the reason shown, when the run is not `audit_eligible` or is live.
- **Packaging context:** at `cc817f4`, `pyproject.toml` dropped fastapi/uvicorn/httpx from dev, the `app` extra, the `drone-fly-app` script, `"app"` from the hatch packages, and `"."` from the pytest pythonpath. `.gitignore` still has `.dev-app/`, `training/**` with negations for the committed dirs, and `*.pt`.
- **Challenger residual Minor notes:**
  - `tracked_probe` uses the project root and is injected in every test;
  - "Continue from checkpoint" leaves `warm_src_extra_dim` blank so it resolves at runtime.

### User decisions at plan preview (2026-10-11)
- **Risk 1 — authorized:** the developer may edit `pyproject.toml` and regenerate `uv.lock`. This restores UC-61's packaging: fastapi, uvicorn and httpx in the `dev` extra; an `app` extra (fastapi, uvicorn, pywebview); pytest `pythonpath ["src", "."]`; the `drone-fly-app` script; and `"app"` in the hatch packages. The conftest/importorskip fallback is **not** used.
- **Risk 3 — accepted:** `app/reservoir_launch.py` is in scope.
- **Commits:** commit and push at each milestone.

### Challenger verdict
**Approve** (round 2). The round-1 Major issue (the order of the honesty-verdict rules: hovering must be flagged regardless of completion) and all seven Minor items were resolved in round 2. The new `app/reservoir_launch.py` launcher is accepted under AC10. The challenger agreed with escalating Risk 1 to the user rather than skipping server tests in CI.
