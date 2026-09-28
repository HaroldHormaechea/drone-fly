---
plan_for: (free-form task) — UC-61 desktop app training-status view polish batch (6 items)
work_branch: feat/uc-61-status-view-polish
team: drone-fly
approved: 2026-09-28
---

# APPROVED proposal — UC-61 desktop status-view polish (6 items)

Challenger approved after one revision. TARGET_DIR = `/workspace/drone-fly-uc-61-status-view-polish` (branch feat/uc-61-status-view-polish). Brief `profiles: []` → no profile skills apply. App = vanilla JS/CSS front-end + FastAPI backend, fully offline. Developer write scope = `app/**` (per TARGET_DIR/CLAUDE.md); QA = `tests/**`. **No `src/drone_fly/**` change is required.**

## Analysis
Status view = `renderRunStatus(name)` in `app/static/app.js` (~296-408): one `.card` with controls, progress bar + `progLabel`, `.metrics` grid, "Process logs" head, `.log-panel` (`<pre>`). `tick()` polls `GET /api/runs/{name}/status` every 2.5s; `renderMetrics()` hand-appends 8 cards; `renderMetricsCsv()` is the CSV fallback; `tickLogs()` tails `GET /api/runs/{name}/logs?since=` and auto-scrolls only when pinned to bottom. Formatters `fmt`/`fmtInt` (88-102). Info modal = `openModal` (`app/static/modal.js`); reusable `infoButton(title,help,example,constraintsHTML)` in app.js (558-570, `window.AppUI`). Config-field help = `app/field_help.py` (TRAIN_HELP + TRAIN_SECTIONS/SECTION_ORDER + merge_help/merge_sections) → `configs_io.describe_train_fields()` → `GET /api/train-configs/schema` → `forms.js`.

Status record shape (confirmed from `src/drone_fly/train/status_emitter.py` + `adapter/dynamics_summary.py`): top-level `timesteps`(int) `target_timesteps`(int|null) `n_updates`(int) `elapsed_seconds`(float) `fps`(float|null); `rollout{ep_rew_mean,ep_len_mean,success_rate}`; `train{loss,value_loss,approx_kl,entropy_loss,explained_variance,std}`; `health{status,message}`|null; `dynamics{backend,applied_mass,weight,thrust_to_weight,hover_throttle,max_body_rate,spawn_z,arm_length}`|null. `/status` envelope = `{source,latest,lines,next}`.

**Item-1 root cause (differs from brief guess):** there is NO `user-select` anywhere in `app/`. Real cause = `app/__main__.py:98` `webview.create_window(...)` omits `text_select=`, and pywebview defaults `text_select=False`, disabling selection at the native-window level. Primary fix is `text_select=True`; that path is owner-eyeball only (never imported in CI), so a CSS layer backs it up.

## Proposed Solution (approved)

### Item 1 — selectable status data
- `app/__main__.py` (~line 98): add `text_select=True` to `create_window(...)`. Owner-eyeball only, not CI-testable.
- `app/static/app.css`: `user-select:text` on `.card, .metrics, .metric, .log-panel, .modal-body`; `user-select:none` on `nav.sidebar, .btn-row, button, .view-head-actions` (keep chrome native-feeling). Robust regardless of webview default.

### Items 3 + 5 + 6 — collapse into ONE grouped, backend-driven design
Single **status-field descriptor list** owned by backend, consumed by a rewritten grouped renderer.
- `app/field_help.py`: add `STATUS_HELP` (curated plain-language, SB3/PPO-accurate but non-expert), a `STATUS_FIELDS` descriptor list, and `STATUS_GROUP_ORDER = ["Progress","Rollout","Train","Health","Dynamics"]`.
  - Descriptor shapes (exactly one of):
    - Path-backed: `{key,label,group,path:[...],format:{type,...},help,example?}`
    - Computed: `{key:"eta",label,group:"Progress",computed:true,format:{type:"duration"},help,example?}`
  - Closed `format.type` vocabulary: `float`(+digits) | `int` | `seconds` | `duration` | `badge` | `text`.
  - Grouping:
    - **Progress**: timesteps(int), target_timesteps(int), eta(computed/duration), elapsed_seconds(seconds), fps(float,0), n_updates(int)
    - **Rollout**: ep_rew_mean(float,3), ep_len_mean(float,1), success_rate(float,3)
    - **Train**: loss, value_loss, approx_kl, entropy_loss, explained_variance, std (float, per-field digits)
    - **Health**: status(badge) — health.message surfaced in the info popup (see below)
    - **Dynamics**: applied_mass, weight, thrust_to_weight("T/W"), hover_throttle, max_body_rate, spawn_z, arm_length (float); backend(text)
- `app/configs_io.py`: add `describe_status_fields()` (help merged), mirroring `describe_train_fields()`.
- `app/server.py`: add `GET /api/status-fields` → `{"fields": configs_io.describe_status_fields()}`, mirroring the schema route.
- `app/static/app.js`: rewrite `renderRunStatus`/`renderMetrics` to fetch the descriptor list once (cache like `getTrainSchema`), render metric boxes **grouped** under topic headings in `STATUS_GROUP_ORDER`, each box = label + value + `infoButton` (reuse `window.AppUI.infoButton` → `openModal`). Value read via `path` (or computed for eta) and formatted via `format.type`.

**Renderer empty/null contract (the QA-defining part):**
- (a) Per-box: path unresolved (any ancestor null/absent) OR leaf null/undefined/NaN → render `"—"` (box stays; grid shape-stable across polls).
- (b) Whole-group suppression: hide heading+boxes when the backing block is null — **Health** hidden when `record.health==null`, **Dynamics** hidden when `record.dynamics==null`. Progress/Rollout/Train always shown (blocks always present). ⇒ just-launched run shows Progress+Rollout+Train with "—" leaves and NO empty Health/Dynamics headings.
- (c) `source==="csv"` → existing flat `renderMetricsCsv` (ungrouped, no help), untouched; falsy latest → existing "no progress yet". Grouped renderer runs only for `source==="jsonl"` with a `latest`.

### Item 2 — ETA (frontend derivation, no backend field)
In the grouped renderer: `rate = fps>0 ? fps : (elapsed_seconds>0 ? timesteps/elapsed_seconds : 0)`. Clamps: `target_timesteps==null` → "—"; `rate<=0` → "—"; `remaining=target-timesteps`, if `<=0` → "done"; else `fmtDuration(remaining/rate)`. Add a `fmtDuration(seconds)` helper ("1h 23m"/"45s"). eta gets an info popup via `STATUS_HELP["eta"]`.

### Item 3 health.message
Kept: Health status box = badge; `health.message` appended into the Health info-popup body when present (not a permanent box, to protect the grid).

### Item 4 — log panel fills remaining vertical height
- `app/static/app.js`: `renderRunStatus` adds `view-fill` class to `#view`; `route()` removes it at top of every dispatch so only the status view uses fill mode.
- `app/static/app.css`: `#view.view-fill{display:flex;flex-direction:column;overflow:hidden}`; status `.card` under fill → `display:flex;flex-direction:column;flex:1;min-height:0`; `.log-panel` under fill → `flex:1;min-height:120px` (drop fixed `max-height:320px`). Internal scroll + auto-tail-only-when-pinned (`tickLogs`) unchanged. `min-height` floor prevents a tall metrics grid crushing the log to zero. (Definite-height ancestor chain verified: html/body 100% → body flex → #view 100vh.)

## Files Affected
**Production code (developer, `app/**`):**
- `app/__main__.py` — item 1 (text_select=True; untestable, owner-eyeball)
- `app/field_help.py` — items 2/3/5/6: STATUS_HELP + STATUS_FIELDS + STATUS_GROUP_ORDER (+ eta help)
- `app/configs_io.py` — item 6: describe_status_fields()
- `app/server.py` — item 6: GET /api/status-fields
- `app/static/app.js` — items 2/3/4/5/6: grouped renderer, ETA + fmtDuration, descriptor fetch/cache, info buttons, view-fill toggle
- `app/static/app.css` — items 1/4/5: user-select rules, flex-fill log panel, grouped-box headings

**Test code (QA, `tests/**`):**
- `tests/test_app_server.py` — `GET /api/status-fields`: returns descriptors; every field non-empty `help` + `label` + `group ∈ STATUS_GROUP_ORDER`; groups ⊆ order set; expected keys present.
- `tests/test_app_configs_io.py` (or new `tests/test_app_status_help.py`) — `describe_status_fields()`: well-formedness accepts `path` XOR `computed:true`; `format.type` ∈ closed set (`float/int/seconds/duration/badge/text`); STATUS_HELP no empty strings; **completeness derived from a real emitted record** (reuse `tests/test_status_emitter.py` to emit one `status.jsonl`, read `latest`, assert every leaf key in top-level+rollout+train+dynamics has a descriptor `path`; excludes `line` cursor, `health.message`, and `health.status`→badge). Auto-catches emitter drift.
- **Owner-eyeball (no JS harness exists):** item-1 selection, item-3/5 grouped boxes, item-4 fill/scroll, item-6 modal wiring + health.message, ETA display.

## Risks & Considerations
1. Item-1 primary fix (pywebview) untestable — CSS layer mitigates and is eyeball-verifiable.
2. Descriptor path+format lives server-side (challenger-endorsed) to kill JS/Py drift; format-vocab guard prevents a backend type the JS can't render.
3. Dynamics/Health groups auto-hide when their block is null (first-paint polish).
4. CSV fallback intentionally ungrouped/help-less (different key namespace).
5. Grouped boxes make the metrics section taller — worth an eyeball at small window heights (log panel has a min-height floor).
6. STATUS_HELP accuracy: keep each metric to 1-2 plain, SB3-accurate sentences.

## Orchestrator (team-lead) second-eyes verification
Independently confirmed in the worktree before persisting: (1) no `user-select` anywhere under `app/` — item-1 root cause is the pywebview `text_select` default, not a CSS rule; `app/__main__.py:98` omits `text_select=`. (2) `#view { height: 100vh }` (app.css:95) — the definite-height ancestor the flex-fill log panel requires. (3) `infoButton` at `app.js:558`, exported via `window.AppUI` (:572) — the reuse target for per-metric popups. (4) `describe_train_fields()` (configs_io.py:129) + `GET /api/train-configs/schema` (server.py:118-120) — the mirror targets for `describe_status_fields()` + `/api/status-fields`. Plan is app-only; ETA clamps and null/empty renderer contract are sound. Approved for Phase 2.
