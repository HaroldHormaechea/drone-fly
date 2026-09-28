---
plan_for: (free-form task) — UC-61 desktop app training UX batch (6 items)
work_branch: feat/uc-61-training-ux-batch
team: drone-fly
approved: 2026-09-28
---

# APPROVED PROPOSAL — UC-61 training-UX follow-up batch (6 items)

Challenger approved (independently verified every high-risk claim against source); the three notes below are folded in as developer/QA guidance, not a revision round. TARGET_DIR = `/workspace/drone-fly-uc-61-training-ux-batch`. `profiles: []` (none). `app/**` + `viz/**` are developer-writable per `TARGET_DIR/CLAUDE.md`; `tests/**` is QA. This batch does NOT touch `src/drone_fly/**`.

## Analysis
Desktop app = FastAPI backend (`app/*.py`) + vanilla offline HTML/CSS/JS (`app/static/*`), reusing `viz/viewer.js`.
- Router `app/static/app.js:704 route()` → `clearPoll()` (`:129`), removes `view-fill` (`:715`), awaits a `render*()` that only clears `root.innerHTML` after its first await (item-5 gap); `refreshNav()` at end (`:736`) + 5s interval (`:745`).
- `renderRunStatus` (`:351`) owns the 2.5s `pollTimer` (`:498`); `tick()` (`:460`) reads `/api/runs/{name}` (→`run.running`) + `/status` (→`snap.latest`,`snap.source`), rebuilds boxes via `renderMetricsGrouped`/`statusBox` (`:395`/`:431`). Elapsed only moves on the poll (item 1). Progress width = 0 when `target_timesteps` null (`:468`) → dead-looking bar (item 2).
- Forms (`forms.js:62 buildForm`): a descriptor with `choices` already renders `<select>` (`:117`); `collect()` (`:150`) coerces by `base_type` + omit-vs-default (byte-identity).
- Descriptors: `configs_io.describe_train_fields` (`:150`); `_train_choices()` already gives choices for adapter/device/schema, `_prune_choices()` for prune_rule → those 4 are ALREADY selects. `resume` (`config.py:403`) has no choices; `_resolve_resume` (`loop.py:242`) → valid = None/auto/latest/explicit-path (semi-open). `connectome` (`config.py:398`) is an open path (pruned slice, or fixture, or omit for default MaleCNS).
- Registry `app/runs.py`: `_training_dir` (`:290`/`:305`), `_LIVE_STATES` (`:54`), `describe.running`, DI seams spawn/signaler/cli/probe; `validate_run_name` (`config.py:52`) blocks separators/`.`/`..`. Tests use fakes + real tmp root.
- Viewer `viz/viewer.js`: perspective projection (`:1019`), spherical orbit cam, scene up=+z (`:989`); world z-UP/+x-FWD/−y=RIGHT (`:56`). `VIEW_PRESETS` (`:76`) side={yaw:−π/2}=eye at −y=right side. Default today=`top` (cam init `:990`, flight `#view-select` default `viewer.html:102`; distinct from the brain `#map-view-select` `:49-52`). `setScene` (`:~1322`) fits the bounding SPHERE `(radius/tan(FOV/2))*1.6` — not aspect-aware, not a box-fit. `applyPreset` (`:1324`) sets only yaw/pitch (never re-fits distance). `resize()` (`:995`)+ResizeObserver track container; `onWheel` (`:1301`) = manual zoom. Embedded viewer loads the same `viewer.html?embed=1`; `#view-select` stays in DOM.

## Proposed Solution

**Item 1 — 1s elapsed tick (`app/static/app.js`).** In `renderRunStatus` add closure `elapsedBase`/`elapsedBaseAt`(`performance.now()`)/`runActive`. In `tick()` set `runActive=run.running` and, when `latest.elapsed_seconds` is finite, reconcile `elapsedBase`/`elapsedBaseAt` each poll. Tag the Elapsed `.v` node in `statusBox` (when `d.key==="elapsed_seconds"`) with class `js-elapsed`. A module-scope `elapsedTimer=setInterval(…,1000)` writes `fmtDuration(elapsedBase+(performance.now()-elapsedBaseAt)/1000)` into the re-queried `.js-elapsed` node (guard null → CSV/suppressed). Clear `elapsedTimer` inside `clearPoll()` so `route()` tears it down (no leak). ETA stays poll-cadence.

**Item 2 — launching indeterminate bar (`app.js`+`app.css`).** In `tick()` compute `live=run.running`. Determinate (`source==="jsonl" && target_timesteps && timesteps`) → remove `indeterminate` class, set width (existing). `live && !determinate` → add `indeterminate` to `progWrap`, label "starting…". `!live && !determinate` → static, "no progress yet". CSS `.progress.indeterminate>span` = `width:100%!important` + repeating-linear-gradient stripes + `@keyframes prog-marquee` (pure CSS, offline).

**Item 3 — delete run + all data (`runs.py`,`server.py`,`app.js`,`app.css`,`modal.js`).** Default: delete OUTPUTS only (`training/<name>/`, gitignored); keep committed `configs/train/<name>.yaml` unless opt-in `delete_config`. `RunRegistry.delete(name,*,delete_config=False)`: (1) `validate_run_name`; (2) live guard `_resolve_state(name) in _LIVE_STATES` → RunError (409); (3) realpath+`os.path.commonpath` containment of `_training_dir(name)` within `<root>/training` (symlink defense) → RunError otherwise; (4) if neither dir nor config exists → `FileNotFoundError` (404), else `shutil.rmtree`; (5) if `delete_config` remove the YAML; (6) pop `_procs`/`_last_state`; (7) return `{name,deleted_outputs,deleted_config}`. Real-FS (tmp in tests), no new seam. Route `DELETE /api/runs/{name}?delete_config=bool` — RunError→409 via `_guard`, FileNotFoundError→404 (mirror recordings route `:246`). UI: destructive Delete button per row in `renderTrainList` (`button.danger`, `app.css:155`); confirm via `modal.js` extended with optional footer `actions` + returned choice (keeps focus-trap/ESC/backdrop), hosting an "also delete the saved config" checkbox; on confirm `api(DELETE …)` → toast → `refreshNav()` + re-render; if viewing the deleted run, `location.hash="#/train"`.
  → **Challenger note 1 (fold in):** with the default (config kept), the row still appears (config-only state) because listing unions `list_train_config_names`. Confirm modal + success toast must state this explicitly ("outputs deleted; run definition kept — tick the box to also remove the saved config"). Copy only, no logic change.

**Item 4 — enum comboboxes (`field_help.py`,`configs_io.py`,`server.py`,`forms.js`,`app.js`).** Closed enums → strict `<select>` (adapter/device/schema/prune_rule ALREADY are — no change). Open sets (`connectome`,`resume`) → `<datalist>`-backed text input (suggestions + free text preserved), keeping `collect()` byte-identical. `field_help.FIELD_OPTIONS` + `merge_options(fields,map)` (non-destructive, by name) attach `options`(static)/`options_endpoint`(dynamic URL)/`open`(bool). Seed: `resume→{options:["auto","latest"],open:True}`, `connectome→{options_endpoint:"/api/connectomes",open:True}`; wire into `describe_train_fields`/`describe_prune_fields`. `configs_io.list_connectomes(root)`: prune configs' `out` dirs that exist + immediate subdirs of `artifacts/pruned/`, deduped/sorted (suggestion-only). Route `GET /api/connectomes`. `forms.js`: `options`/`choices`→select; `open`→text input + `list=<datalistId>` + sibling `<datalist>`; nullable "set" box + `collect()` unchanged; constraints list `options`. Dynamic fetch: view (`renderTrainNew`/`renderRunConfig`) prefetches unique `options_endpoint`s and passes `optionsByField` into (still-sync) `buildForm`; on fetch failure degrade to a plain input.
  → **Challenger note 2 (fold in):** developer confirms the resume mode strings against `_resolve_resume` (loop.py:242) at impl time rather than hardcoding blindly (low-risk since datalist preserves free text).

**Item 5 — skeleton loaders (`app.js`+`app.css`).** In `route()`, synchronously `setActiveNav(hash)` (toggle `.active`/`aria-current` on `[data-route]` anchors) before awaits. Threshold skeleton: `const skelT=setTimeout(()=>paintSkeleton(kind),120)`; `clearTimeout(skelT)` in a `finally` (fast views → no flash; slow → skeleton until renderer's `innerHTML=""` or the catch replaces it). `paintSkeleton(kind)` (kind from the existing hash parse) paints greyed shimmer blocks per rough layout (form/list/status/recordings-settings-about). CSS `.skeleton`/`.skel-line`/`.skel-block` + shimmer `@keyframes`, offline.

**Item 6 — flight 3D default = angled side + auto-fit-to-box 75% (`viz/viewer.js`,`viz/viewer.html`).** Orientation: new preset `angled:{yaw:-5*Math.PI/12, pitch:Math.PI/12}` (right side nearest, ~15° toward forward + ~15° elevation → three-quarter perspective; well clear of the gimbal clamp). Cam init (`:990`) and `applyPreset` fallback (`:1325`) → `VIEW_PRESETS.angled`; `viewer.html` flight `#view-select` (`:99-102`, NOT `map-view-select`) default option `<option value="angled" selected>3D</option>`, front/side/top kept.
  Auto-fit (aspect+orientation aware, FILL=0.75): add `fitToScene()` — build the camera basis for current yaw/pitch (same math as `project()` `:1005`); over the 8 `scene.bb` corners compute `halfW=max|dot(c-center,r)|`, `halfH=max|dot(c-center,u)|`; with `t=tan(FLIGHT_FOV/2)`, `aspect=cssW/cssH`: `distV=halfH/(0.75*t)`, `distH=halfW/(0.75*t*aspect)`, `cam.dist=clamp(max(distV,distH),0.05,1e6)`. Degenerate guard (bbox3 already sanitizes; if non-finite/~0 fall back to `(radius/t)*1.6` or 10). Call `fitToScene()` from `setScene` (replacing the sphere heuristic) AND `applyPreset` (reset = orientation + 75% fit), resetting a `userZoomed=false` flag; `resize()` re-fits only when `!userZoomed`; `onWheel` sets `userZoomed=true`. Identical standalone + embedded (`viewer-embed.js` unchanged).
  → **Challenger note 3 (fold in):** at eyeball time verify the +x path recedes (not crowding) with the forward-offset eye; keep the documented fallbacks — flip azimuth sign to yaw=−7π/12, or drop to elevation-only (yaw=−π/2, pitch=+π/12, pure right-side) — for the developer.

## Files Affected
**Production code (developer):**
- `app/static/app.js` — items 1,2,3,4,5.
- `app/static/forms.js` — item 4 (select for `options`; datalist for `open`; constraints).
- `app/static/modal.js` — item 3 (optional footer `actions` + returned choice).
- `app/static/app.css` — items 2 (indeterminate `@keyframes`), 3 (delete button), 5 (skeleton shimmer).
- `app/field_help.py` — item 4 (`FIELD_OPTIONS`+`merge_options`).
- `app/configs_io.py` — item 4 (merge options; `list_connectomes`).
- `app/server.py` — item 3 (`DELETE /api/runs/{name}`), item 4 (`GET /api/connectomes`).
- `app/runs.py` — item 3 (`RunRegistry.delete` + live guard + realpath containment).
- `viz/viewer.js`, `viz/viewer.html` — item 6 (angled preset + `fitToScene` + wiring; default option).

**Test code (QA — `tests/**`; no JS harness → items 1/2/5/6 + delete UI = owner-eyeball; QA covers Python):**
- `tests/test_app_runs.py` — `delete()`: removes tree; refuses when running (RunError); realpath/name path-safety; config preserved by default vs removed with `delete_config=True`; absent → FileNotFoundError.
- `tests/test_app_server.py` — `DELETE /api/runs/{name}`: happy/409-running/404-absent/`delete_config` query; `GET /api/connectomes` enumeration.
- `tests/test_app_configs_io.py` — `list_connectomes`; descriptors carry `options`/`options_endpoint`/`open` for resume/connectome; existing `choices` intact for adapter/device/schema; `merge_options` non-destructive. NOTE: any test asserting an exact descriptor key-set must be updated for the new optional keys.

## Risks & Considerations
- Item 3 destructive: traversal blocked (validate_run_name + realpath containment), symlink escape blocked (realpath), config-delete opt-in (no silent loss of committed source), mandatory confirm; TOCTOU acceptable for single-user local. Post-delete-default row persists (note 1 → clarify in copy).
- Item 4: datalist for open fields avoids a free-text regression; byte-identity preserved; fetch failure degrades to plain input; descriptor-key-set tests need updating.
- Item 1: no timer leak (clearPoll), drift-free reconcile, guarded when Elapsed box absent.
- Item 2: gated on `run.running`; determinate path restores width + removes class.
- Item 5: 120ms threshold + clearTimeout prevents flash; skeleton always replaced.
- Item 6: perspective already active; gimbal-safe; forward-offset & fill-fraction are owner-eyeball with documented fallbacks; center-depth fit approximation absorbed by the 25% margin; no status-emitter changes (status-help completeness test unaffected).

## Additional item-6 developer notes (folded in post-approval)
- **Note A (initial-layout race, self-healing):** if `setScene`/`fitToScene` runs before the embedded iframe's layout settles, `cssW/cssH` can be momentarily stale — but the `ResizeObserver → resize() → re-fit (when !userZoomed)` path re-frames once layout lands. The developer MUST actually wire the resize re-fit (not only fit in `setScene`); that same mechanism gives both the embed-layout-race recovery and reset-on-aspect-change.
- **Note B (reset lands on default every load):** `setScene`'s follow-on `applyPreset(el("view-select").value)` (viewer.js:214, called on every scene/recording load) must reset orientation→`angled` AND re-fit to 75% — i.e. `applyPreset` calls `fitToScene()` and clears `userZoomed` — so each new recording load lands on the angled + 75% default, not a preserved custom orbit from the previous episode.

## Orchestrator (team-lead) second-eyes verification
Independently confirmed in the worktree before persisting: (1) `validate_run_name` (config.py `_NAME_RE`) rejects `.`/`..`/separators/spaces; `_LIVE_STATES = {running,pausing,stopping}` (runs.py:54), `_resolve_state` (:336), `_training_dir` (:305), `_procs`/`_last_state` all present — the delete live-guard + realpath/commonpath containment is grounded. (2) Two distinct selects exist — brain `#map-view-select` (viewer.html:49) and flight `#view-select` (viewer.html:99, default `top`); item 6 correctly targets the flight one; `VIEW_PRESETS` (viewer.js:76), cam-init (:990), `applyPreset` fallback (:1325) are the right hooks. (3) `choices`→`<select>` already works (forms.js:117-123), so item 4's only new work is `resume`/`connectome` as open datalists. App+viz only; no `src/drone_fly/**`. Approved for Phase 2.
