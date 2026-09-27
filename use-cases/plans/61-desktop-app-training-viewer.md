---
plan_for: use-cases/61-desktop-app-training-viewer.md
work_branch: feat/uc-61-desktop-app-training-viewer
team: drone-fly-uc-61
approved: 2026-09-27
---

# UC-61 Desktop App — FINAL APPROVED IMPLEMENTATION PLAN (v2, challenger-approved)

Status: BUILD-APPROVED implementation run. Analyst↔challenger peer loop closed — challenger approved after one revision round (Major #1 + Minors #2–#4 resolved). Every load-bearing claim below was independently re-verified against the current worktree code.

---

## Analysis

**Problem / task.** Build a single-user, local desktop application that replaces the two things the owner touches today — the training TUI (`src/drone_fly/train/tui/**`) and the standalone HTML recording viewer (`viz/viewer.{html,js}`). One UI drives the whole loop: (1) define/generate a slice, (2) configure + launch multiple named trainings, (3) monitor a running training (live progress, pause, resume, access recordings), (4) play back recordings in the existing brain+inputs+course viewer. Layout is a left-side nav (top level: Generate slices / Train / Settings, with sub-options) driving a right-side main view; a terminal menu item swaps the main view. Configuration is committed only via an explicit Save button (no auto-save on blur/focus/nav), and no configuration version history is required.

**Owner-approved build decisions (override any conflicting older phrasing, incl. the analysis doc):**
1. Framework = **FastAPI + Uvicorn backend wrapped in a pywebview native window from day one**. Vanilla HTML/CSS/JS front-end (no SPA framework/build chain). `viz/viewer.js` reused **unchanged**. Recording gzip decompressed **server-side** (so it does not depend on the system webview's `DecompressionStream`). Electron/Tauri rejected.
2. Scope = **Phases 0–2** of the analysis: configure + launch + monitor trainings; full run management (multiple named runs, concurrent, stop, pause/resume); the slice-definition surface; and the Settings pane — i.e. the complete left-nav. Phase 3 (PyInstaller packaging, remote-GPU topology) is OUT of scope.
3. Backend glue **allowed** in `src/drone_fly/train`: a **JSONL status-emitter** callback (TUI-parity progress) + a **checkpoint-on-signal handler** (lossless pause). Pure observability/coordination glue — no training/env/reward logic changes.
4. Topology = **local-only**. Keep the client/server split clean so remote-GPU is a later config change, but do not build remote control now.

**Code verification (current tree, worktree `/workspace/drone-fly-uc-61-desktop-app-training-viewer`):**
- **CLI surface** — `src/drone_fly/cli/__init__.py`: `main()` dispatches `train --config [--no-tui]` → `_run_train`, plus `evaluate/prune/prune-trained --config`, `clean`, `fetch-connectome`, `smoke-train`. Console script `drone-fly = drone_fly.cli:main` (pyproject). All four heavy commands are single-`--config <yaml>`. A bad config exits 2 with a one-line message (`ConfigError`), no stack trace.
- **Run registry** — `config.py: run_layout(name)` → `training/<name>/{checkpoints,logs,recordings}/` (`TRAINING_ROOT="training"`; `name` validated `[A-Za-z0-9._-]`, not `.`/`..`). This filesystem layout IS the multi-named-run registry — enumerate `training/*/`, no DB.
- **PruneRunConfig** (config.py ~L663) — exactly four fields: `connectome` (str, optional → default full MaleCNS), `out` (str, required), `prune_k` (int, default `DEFAULT_PRUNE_K`), `prune_rule` (str, default `DEFAULT_PRUNE_RULE`).
- **TrainRunConfig** (config.py ~L296–550) — full key set verified via `from_mapping`. **DELTA vs analysis doc §5:** the doc's train-form list OMITTED two real keys — `prune` (bool, default False) and `prune_k` (int, default) — both are `_Spec(..., default=...)`. AC4 ("every key maps to a control") requires them.
- **Omit-vs-default semantics** — `_validate` treats an omitted key OR explicit `null` as "use default". Two classes:
  - Class (a) — keys with a concrete `_Spec.default`: `adapter` (auto/simple/pybullet, default auto), `prune`, `prune_k`, `record`, `randomize`, `randomize_dynamics`, `strict_capacity`, `control_hz` (50), `physics_ratio` (10), `command_latency_ms` (0), `altitude_weight`/`altitude_target` (RewardConfig defaults). These always resolve to a value.
  - Class (b) — `| None` "leave-the-dataclass-default" keys: `connectome`, `device` (cpu/cuda/mps), `timesteps`, `n_envs` (≥1), `resume`, `record_every`, `record_dir`, `schema` (NAMED_SCHEMAS), the three tri-state placement toggles (`randomize_obstacles`/`randomize_recharge_pads`/`randomize_repair_pads`), `capacity_floor`, `ent_coef`, `airborne_curriculum_enabled`/`_warmup_fraction`/`_anneal_fraction`, `rate_kp`/`rate_ki`/`rate_kd`/`rate_max_body_rate`, `n_epochs`/`batch_size`/`n_steps`/`learning_rate`, and the six `pybullet_{mass_ratio,tw,arm_length}_{min,max}`. The form MUST emit a class-(b) key ONLY when the user actually sets it. The three placement toggles need a real 3-way (unset / true / false), not a checkbox.
- **Progress source** — verified there is NO structured status emitter today. `loop.py: _make_logger` configures the SB3 logger with formats `["csv","tensorboard"]` regardless of TUI state → `training/<name>/logs/progress.csv` is always written. The TUI reads live in-process via `TuiCallback` (`ep_info_buffer` + `logger.name_to_value`); none of that is exposed to another process. Health verdict = `HealthVerdict` (`train/health.py`) via `HealthCallback`; drone dynamics = `drone_dynamics_summary()` (`adapter/dynamics_summary.py`).
- **Callback wiring point** — `loop.py: train()` builds `callbacks: list = [checkpoint_cb]` (~L552), conditionally appends AirborneStartCurriculum / Recording / Health / (TUI-only) TuiCallback, then calls `model.learn(callback=callbacks if len(callbacks)>1 else checkpoint_cb)` (~L671). This is exactly where the new StatusEmitter and CheckpointOnSignal callbacks attach — and unlike TuiCallback the emitter must attach on the `--no-tui` path.
- **Pause/resume** — verified there is NO live pause. `CheckpointCallback(save_freq=checkpoint_freq//n_envs, save_vecnormalize=True)`; `checkpoint_freq` default **25_000** (`train/config.py`). Resume via config `resume: auto` → `_resolve_config_resume` → `find_latest_checkpoint` (newest checkpoint, else fresh — no error). A clean resume needs the `_<steps>_steps.zip`, its matching `_vecnormalize_<steps>_steps.pkl`, and the same YAML.
- **Recording + viewer** — `record/recorder.py` writes `training/<name>/recordings/episode_<n>.json[.gz]` (`gzip.open` when `gzip_output`; numbering continues across resumes). `viz/viewer.js` exposes a top-level `loadDocument(doc, name)` reachable as `window.loadDocument` (exactly how `scripts/screenshot_viewer.mjs` injects a recording). In-browser gunzip relies on `DecompressionStream` (throws if absent) → gunzip server-side.
- **Write scopes** — target `CLAUDE.md` §"Dev-team write authorizations" authorizes the developer for `app/**` (the desktop app), `src/drone_fly/**` (the two glue files), `viz/**`, `configs/**`, `pyproject.toml`, `uv.lock`, root docs. Hand-written tests under `tests/**` are QA's exclusive scope.
- **Tests / CI** — pytest, flat `tests/test_*.py` (81 files), `pythonpath=["src"]`, CI installs `--extra dev` ONLY (no pybullet/`sim` extra), no GPU, no display.

---

## Proposed Solution

### A. Backend glue in `src/drone_fly/train` (the ONLY new production code; AC6/AC7/AC10)

**1. `src/drone_fly/train/status_emitter.py` — `StatusEmitterCallback(BaseCallback)` (AC6).**
On each `_on_rollout_end`, write one JSON line to a configured `status.jsonl` path containing: timestamp, `total_timesteps` + target timesteps, rollout metrics (`ep_rew_mean`, `ep_len_mean`, fps), train metrics (loss / approx_kl / entropy / explained_variance when present), the health verdict (status + summary from `HealthVerdict`), and the dynamics summary (from `drone_dynamics_summary`). It reuses the SAME data sources as TuiCallback/HealthCallback — no new assessment. **Defensive contract (explicit):** any write/format error disables the callback and is never propagated into `model.learn` (mirrors TuiCallback). Wiring: `train()` gains a param `status_path: str | None = None`; when set, the callback is appended unconditionally (independent of the `tui` flag, so it works on the `--no-tui` path). `_run_train` passes `status_path = os.path.join(TRAINING_ROOT, cfg.name, "status.jsonl")`, so EVERY CLI train run emits it and the app just runs the plain `drone-fly train --config <path> --no-tui`. Default `None` keeps `smoke_train` and all existing tests byte-identical.

**2. `src/drone_fly/train/checkpoint_signal.py` — `CheckpointOnSignalCallback(BaseCallback)` (lossless pause; AC7).**
- **The signal handler is I/O-free ONLY.** On SIGTERM/SIGINT it sets `self._stop_requested = True` and records intent — nothing else. No `model.save`, no `venv.save`, no file writes inside the async handler. (This avoids the UC-53 corruption class: a Python signal handler interrupts the main thread mid-bytecode, so a save fired from the handler could hit mid-optimization/mid-write → BadZipFile / corrupted checkpoint.)
- **The flush happens at the next `_on_step`** (a safe SB3 loop boundary, never mid-write): it sees the flag, performs the checkpoint flush, then returns `False` to stop `model.learn` cleanly; `train()` then falls through to its normal final-save + return.
- **The flush is ATOMIC** — write `model.save` + `venv.save` to temp files, then `os.replace` into the final names `ppo_racer_<num_timesteps>_steps.zip` + `ppo_racer_vecnormalize_<num_timesteps>_steps.pkl` (the exact naming `find_latest_checkpoint` / `_infer_vecnormalize_path` expect, so `resume: auto` picks it up), matching the UC-53 remediation. Dev/QA must confirm the atomic path mirrors the existing UC-53 `ProgressReportingPPO.save` / `_progress_sink` exclusion.
- Handlers are registered on `_on_training_start` and **restored on training end (finally-guard)** so no handler leaks into a caller's process. Gated by `checkpoint_on_signal: bool = False` (byte-identity when off).

**3. `loop.py` / `cli/__init__.py` edits are WIRING-ONLY (AC10).** `train()` gains the two params above plus a conditional append to the existing `callbacks` list; `_run_train` passes the derived `status_path` and `checkpoint_on_signal=True`. Zero training/env/reward logic touched; byte-identity preserved when both features are off (both default off → `smoke_train` and every existing test path unchanged). The only NEW `src/drone_fly` code is the two files above.

### B. The desktop app under `app/**` (outside `paths.production`; AC10)

Proposed sub-layout:
```
app/
  __init__.py
  __main__.py     # python -m app: start uvicorn (127.0.0.1:ephemeral) in a thread, open a pywebview window at that URL; graceful shutdown
  server.py       # create_app() -> FastAPI; mounts routers + static; headlessly testable via starlette TestClient (no window)
  runs.py         # RunRegistry: enumerate training/*/ ; in-proc {name: {pid, popen, state}} ; spawn/stop/pause/resume state machine
  configs_io.py   # (de)serialize TrainRunConfig/PruneRunConfig <-> YAML honoring omit-vs-default; load / save / overwrite-in-place
  status.py       # tail training/<name>/status.jsonl (COMPLETE newline-terminated lines only, json.loads-guarded) + progress.csv fallback reader
  recordings.py   # enumerate training/<name>/recordings/ ; read + server-side gunzip episode_<n>.json[.gz] -> parsed JSON
  static/
    index.html    # app shell (left-nav + main-view container)
    app.css
    app.js        # hash router (#/train/<name>, #/train/new, #/slices/new, #/settings) + nav + fetch glue
    forms.js      # train-config form (every TrainRunConfig key, tri-state controls, omit-vs-default), slice form, settings form
    viewer-embed.js  # loads the viewer region and calls window.loadDocument(parsed, name)
```

**Dependencies (developer maintains `pyproject.toml`).** Add an optional group `app = ["fastapi", "uvicorn", "pywebview"]` for runtime; add `fastapi`, `uvicorn`, `httpx` to the **dev** extra so CI (dev-only) can import the backend and drive `starlette` TestClient hermetically. **pywebview stays OUT of the dev/CI extra** (it needs a display; the window-open path is the documented owner-eyeball piece). Regenerate `uv.lock`. Optional convenience console script `drone-fly-app = app.__main__:main`.

**Front-end** = vanilla HTML/CSS/JS, no SPA framework / no build chain (decision 1). `viz/` is served on a static route and the viewer is **reused unchanged**, embedded via `window.loadDocument`.

**Backend endpoints (each reduces to a CLI call or a filesystem read — AC10):**
- `GET /api/runs` → enumerate `training/*/` + live state; `GET /api/runs/{name}` → detail.
- `GET/POST /api/train-configs/{name}` → GET loads the YAML into the form; POST (Save) serializes form→YAML honoring omit-vs-default and **overwrites `configs/train/{name}.yaml` in place** (AC9); the written config defaults `resume: auto` unless the user explicitly set something else.
- `POST /api/runs/{name}/launch` → spawn `drone-fly train --config <path> --no-tui` as a subprocess with CWD = project root; register `{pid, popen, state}` (AC5).
- `POST /api/runs/{name}/stop` → terminate the subprocess (Popen group; per-OS `terminate()` / Windows `taskkill` / CTRL_BREAK).
- `POST /api/runs/{name}/pause` → send SIGTERM → the checkpoint-on-signal handler flushes at the next boundary → state=paused. `POST /api/runs/{name}/resume` → **pure relaunch of the same already-saved YAML with `resume: auto`** — zero implicit writes to the user's YAML (transient override config in a temp dir only for the edge case where the user's YAML has no `resume` field) (AC7 × AC9).
- `GET /api/runs/{name}/status/stream` (SSE) or `GET /api/runs/{name}/status` (poll) → stream `status.jsonl` lines (CSV-tail fallback) to drive a progress bar + curves (AC6).
- `GET /api/runs/{name}/recordings` → list; `GET /api/runs/{name}/recordings/{ep}` → server-side-gunzipped parsed JSON for `window.loadDocument` (AC8).
- `POST /api/slices` → serialize `PruneRunConfig`→YAML and invoke `drone-fly prune --config` (AC3).
- `GET/POST /api/settings` → local tool config (project root / working dir, default connectome, server host/port) (AC2 Settings pane; single-user, no secrets/accounts).
- Static routes: `/` → app shell, `/viewer/*` → `viz/`, `/app/*` → `app/static/`.

**Navigation/routing (AC2):** front-end-owned hash routes (`#/train/<name>`, `#/train/new`, `#/slices/new`, `#/settings`); a terminal (leaf) nav item swaps the right-side main view; non-leaf items expand/collapse; back/forward work; no server round-trip on nav.

**Editing model (AC9):** every form holds pending edits in local JS state; nothing persists on blur / focus-change / navigation; a single explicit Save button is the only commit point (overwrites the YAML in place); Launch is a separate action from Save; an optional unsaved-changes guard on navigation is acceptable; no snapshots/history.

**Train form (AC4):** every `TrainRunConfig` key maps to a control — including `prune` (toggle) + `prune_k` (int) that the analysis doc's §5 list omitted. Class-(a) keys resolve to defaults; class-(b) `| None` keys are emitted only when the user sets them; the three placement toggles are 3-way (unset/true/false). Config validation is delegated to `config.py` (exit 2 on bad input); the UI surfaces that one-line error. Multiple named trainings = multiple named YAMLs → multiple `training/<name>/`.

### UI authoring approach (optional, dev-time only)
The team MAY use **Claude Design** (the `DesignSync` tool + `/design-sync` skill) to author/preview the vanilla UI components, bound by: (a) **dev-time only, zero runtime dependency** — the shipped app stays self-contained vanilla HTML/CSS/JS served by FastAPI, runs fully offline with NO dependency on claude.ai; any design output is vendored into `app/static/`; (b) **optional/best-effort** — it needs the owner's claude.ai login and may be unreachable from a spawned agent, in which case the developer hand-authors the front-end; it must not block the build; (c) **no new ACs, no SPA framework, no build chain** (decision 1 holds). If design tooling is used heavily, the root-session **team-lead** (which holds the owner login) is the reliable place to drive authoring/preview and then vendor the results into `app/static/`.

---

## Files Affected

**Production code (developer):**
- CREATE `src/drone_fly/train/status_emitter.py` — the JSONL `StatusEmitterCallback`.
- CREATE `src/drone_fly/train/checkpoint_signal.py` — the `CheckpointOnSignalCallback` (lossless pause).
- MODIFY `src/drone_fly/train/loop.py` — wiring-only: `train()` gains `status_path: str | None = None` + `checkpoint_on_signal: bool = False`; conditional append to the existing `callbacks` list.
- MODIFY `src/drone_fly/cli/__init__.py` — wiring-only: `_run_train` passes the derived `status_path` + `checkpoint_on_signal=True`.
- CREATE `app/**` — backend + vanilla front-end (tree above).
- MODIFY `pyproject.toml` (+ regenerate `uv.lock`) — add the `app` optional group; add `fastapi`/`uvicorn`/`httpx` to the `dev` extra; optional `drone-fly-app` console script.
- MODIFY `README.md` (and optionally add `app/README.md`) — how to launch and use the app.
- `viz/viewer.{html,js,css}` is served and reused **unchanged** — NO viewer edit.

**Test code (QA) — hermetic, run in CI without a GPU/display (AC11):**
- `tests/test_status_emitter.py` — emitter writes well-formed JSONL per rollout; defensive-on-error; off by default (byte-identity).
- `tests/test_checkpoint_signal.py` — handler is I/O-free; the `_on_step` flush writes correctly-named checkpoint + vecnormalize atomically and requests stop; no-op when disabled; handler restored on end.
- `tests/test_app_configs_io.py` — TrainRunConfig/PruneRunConfig YAML round-trip incl. omit-vs-default (unset class-(b) key absent; tri-state 3-way; class-(a) defaults) and in-place overwrite.
- `tests/test_app_runs.py` — run enumeration from `training/*/`; subprocess spawn/stop/pause/resume state machine (subprocess + signals mocked/faked).
- `tests/test_app_status.py` — status.jsonl tail (complete-line-only, partial-line-tolerant, json.loads-guarded) + progress.csv fallback parsing.
- `tests/test_app_recordings.py` — enumerate + server-side gunzip of `.json` and `.json.gz` → parsed JSON.
- `tests/test_app_server.py` — FastAPI endpoints via starlette TestClient (no window): routes; Save overwrites YAML; launch/stop/pause/resume call the right CLI/registry (subprocess faked); recording route returns parsed JSON.
- **Documented for owner eyeball (cannot run headless, as with UC-59/60):** the pywebview native window + live viewer rendering.
- **QA note:** every CLI train run now also writes `training/<name>/status.jsonl` — any existing test that asserts the exact file/dir set under `training/<name>/` must treat `status.jsonl` as an EXPECTED new artifact, not a regression.

---

## Risks & Considerations
- **Checkpoint-on-signal corruption class (UC-53).** Closed by the I/O-free handler + `_on_step` flush + atomic temp-then-`os.replace` write. Dev/QA must confirm the atomic path mirrors the existing UC-53 `ProgressReportingPPO.save` / `_progress_sink` exclusion.
- **Subprocess lifecycle.** Zombies / no SIGTERM on Windows → use Popen groups; `terminate()` elsewhere, `taskkill` / CTRL_BREAK on Windows; reconcile the `{name: state}` map from `training/*/` on backend restart. Local-only keeps this bounded.
- **Lossy vs lossless pause.** Checkpoint-on-signal makes pause lossless; if the handler ever fails the fallback is lossy up to `checkpoint_freq` (25_000 steps) — document it.
- **Dependency placement.** `fastapi`/`uvicorn`/`httpx` MUST be in the `dev` extra so CI (dev-only) can import the backend and run hermetic tests; `pywebview` MUST stay out of the dev/CI extra (needs a display). If the web deps land only in an `app` extra, hermetic backend tests can't import — explicit risk.
- **`viz/viewer.js` unchanged mandate.** Embed strictly via `window.loadDocument` + server-side gunzip; do not edit viewer.js. The `DecompressionStream` webview unreliability is sidestepped by gunzipping server-side.
- **CWD sensitivity.** All CLIs are CWD-relative (`run_layout`, recorder, `clean`); the backend MUST spawn subprocesses with CWD = project root and resolve config/recording paths against it.
- **Pause/resume × AC9.** Standardizing on `resume: auto` from first launch makes pause→resume a pure relaunch of the already-saved YAML with zero implicit writes; Save stays the only writer of the user's config.
- **Concurrency / GPU contention.** Multiple named runs = multiple subprocesses; GPU contention is the user's call. Bind `127.0.0.1` on an ephemeral port and show the URL.

---

## Orchestrator (team-lead) hand-off notes added at persist time
- **Verified independently by the team-lead (second pair of eyes), against the current code:** all 11 ACs map to concrete files/behaviors; all 4 owner decisions honored; scope excludes Phase-3 packaging/remote; the UC-53 corruption class is correctly avoided (I/O-free handler + `_on_step` atomic flush).
- **Concrete atomic-write hint:** the training model is a `ProgressReportingPPO` (`src/drone_fly/train/progress_ppo.py`) whose `save()` ALREADY writes to a temp file beside the target and `os.replace`s it, and whose `_excluded_save_params()` already appends `_progress_sink` (L112) so `save()` won't crash. The checkpoint-on-signal flush should REUSE this existing `model.save()` (plus the vecnormalize save, matching `CheckpointCallback`), not hand-roll atomic writes — this closes the UC-53 corruption class for free.
- **Behavior-change to DOCUMENT in README (developer):** because `_run_train` passes `checkpoint_on_signal=True`, every direct `drone-fly train` run (not just app-launched runs) now traps the FIRST SIGINT/SIGTERM to flush a checkpoint and exit cleanly instead of aborting immediately. A second Ctrl-C still force-kills. Document this in the README training section.

Ready for Phase 2 (developer + QA).
