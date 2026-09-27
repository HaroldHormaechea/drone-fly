---
plan_for: use-cases/61-desktop-app-training-viewer.md (follow-up adjustments, pre-merge to PR #68)
work_branch: feat/uc-61-desktop-app-training-viewer
team: drone-fly-uc-61
approved: 2026-09-27
---

FINAL APPROVED PLAN — UC-61 follow-up adjustments (challenger approved, 1 revision round; 2 Majors + all recommendations resolved). Anchors to the existing `app/**` on PR #68 branch. No original ACs removed — this refines views 2/3/8/9 + the launch path (AC5/AC7) and keeps the suite green.

> **OWNER DECISION FOLDED IN (supersedes the analyst's "keep #meta-bar" note below):** the embedded viewer is **fully bare** — `#meta-bar` (recording provenance/dynamics, UC-45/49) IS stripped in embed mode, alongside the 4 enumerated chrome elements. Item 2's `.embed` hide-set therefore includes `#meta-bar`. Standalone viewer default remains untouched.

## Analysis (verified against the worktree)
- **Nav** flat today (index.html L10-28); runs injected flat into `#nav-runs` by app.js `refreshNav()` (L67-79). Router `route()` (L355-377) handles `#/train`, `#/train/new`, `#/train/<name>`, `#/slices`, `#/settings`. `renderTrainDetail(name)` (L156-274) renders Status+Recordings+Config in ONE view and owns the 2.5s pollTimer + `loadRecordings()`.
- **Recordings** today: link-per-episode list (app.js L217-229) → `ViewerEmbed.embedRecording` (viewer-embed.js iframes `/viewer/viewer.html`, drives `contentWindow.loadDocument`).
- **Viewer** (viz/viewer.html): `<header>` = h1 title + `.sub` instructions + `.loader` (`#file-input`+`#load-status`) + `#meta-bar`; `<main id=app>` = `.controls`, `.top-zones` (brain‖actions), `.bottom-zone` (full-width flight-3D), `#outcome`; then `<footer>`. viewer.js `loadDocument()` (L167) sets `#load-status` to "NAME — N frames, M neurons" (the caption item 2 strips). resize() (L982-987) derives backing store from clientWidth+clientHeight. No embed handling exists.
- **Slices**: app writes `configs/prune/<slug>.yaml` (configs_io `prune_config_to_yaml` L216-230) + runs `drone-fly prune`. No `list_prune_config_names` yet.
- **Item-4 bug (confirmed):** runs.py `_resolve_cli()` (L38-50) resolves `drone-fly` next to `sys.executable` = the app's `.venv`, which lacks pybullet (pyproject: pybullet only in the `sim` extra, L76-78). Training subprocess dies on `import pybullet`. `RunRegistry` has DI seams spawn/signaler/cli (L101-113); both test suites pass `cli="drone-fly"` explicitly (test_app_runs.py:65, test_app_server.py:61) → cli-short-circuit keeps them green.
- **Item-6 help:** `_Spec` (config.py L134-143) has NO description field → curate help in app layer, don't touch core config.py.
- **Scopes** (target CLAUDE.md): developer owns app/**, viz/**, configs/**, src/drone_fly/**, pyproject.toml, root docs; QA owns tests/**. CI installs `--extra dev` only (no pybullet/GPU/display). No JS test harness — UI verified by owner-eyeball (as UC-61), backend by TestClient.

## Proposed Solution

**Item 1 — Per-training nav sub-items.** Add leaf routes `#/train/<name>/{config,recordings,status}` in `route()`; `#/train/<name>` (no leaf) redirects to `.../status`. Split `renderTrainDetail` into `renderRunStatus` (controls+progress+metrics, owns pollTimer), `renderRunRecordings` (item 2), `renderRunConfig` (Forms editor+Save). `refreshNav` renders each run as a keyboard-operable `<button aria-expanded>` disclosure with 3 indented leaf links; current-route run auto-expands; `aria-current="page"` on active leaf. **Persist a module-level expanded-set of run names and reapply after each 5s `refreshNav` rebuild** so a user-expanded non-current run doesn't collapse (current-route run always in the set).

**Item 2 — Recordings redesign + app-only embedded viewer.** Replace the link list with a compact `<select>` picker (label `episode N (gz)`) + empty state; `change` → `ViewerEmbed.embedRecording`. Embed mechanism = **`?embed=1` URL param** (viewer-embed.js sets `iframe.src=/viewer/viewer.html?embed=1`; viewer.js reads `URLSearchParams` at startup → `document.body.classList.add("embed")` — the ONE viewer JS edit, behind the flag; standalone default byte-behaviour-identical). **Chrome hidden surgically via `.embed`-scoped CSS in viz/viewer.css — hide `.embed h1`, `.embed .sub`, `.embed .loader` (file input + `#load-status` caption), AND `.embed #meta-bar` (OWNER DECISION: fully bare embed — strip provenance/dynamics too).** Embedded 2-col layout: `#app` → CSS grid `grid-template-columns:1fr 1fr; align-items:stretch`; `.top-zones` forced to a vertical stack in col 1 (brain top, actions below); `.bottom-zone` (flight-3D) in col 2, full height. `.controls` full-width above the grid; `#outcome` full-width below it. Right-column flight canvas fills the taller column via an unbroken `.embed`-scoped `height:100%` chain (grid item→`.bottom-zone`→`.panel`→`#flight-canvas`) over a definite grid-row height — **resize() is NOT touched.** Optional FOUC guard: inert-without-`?embed` inline `<head>` snippet in viewer.html that sets the embed class on `documentElement` only when the param is present. Every change `.embed`/`?embed=1`-gated; standalone untouched.

**Item 3 — "Slices" menu.** Rename nav head "Generate slices"→"Slices" as an expandable disclosure mirroring Train: "New slice" + list of existing slices enumerated from `configs/prune/*.yaml`. Add `configs_io.list_prune_config_names` (mirror of `list_train_config_names`), `GET /api/slice-configs`→{names}, and `GET/POST /api/slice-configs/{name}` (load / save-and-regenerate). Slice leaf route `#/slices/<name>` shows the saved prune config in a Forms editor + "Regenerate slice" button.

**Item 4 — Interpreter/pybullet launch bug (highest priority).** Resolve the training executable **on every launch()/resume()/run_prune() call** (never at construction), in order:
1. Explicit `cli=` (test seam) → use verbatim, skip detection.
2. Settings override `train_executable` (re-read fresh each call): normalize (console-script path | venv python | venv dir → `drone-fly[.exe]` in the derived bin dir); **if it doesn't resolve to a real drone-fly script → RunError naming the bad path (no silent fallthrough).**
3. Auto-detect, probe-gated, priority `.venv-cuda` > `.venv` > other `.venv-*` (sorted) > **`os.path.dirname(sys.executable)`** (lowest priority — preserves today's success path; app's own interpreter chosen only if pybullet-capable).
4. None capable → RunError (train/resume only): "No Python environment with pybullet found — set the training interpreter in Settings, or create one with `uv sync --extra sim`".
Cross-platform bin dir (`Scripts` on nt / `bin` on posix) + exe suffix. **Caching boundary:** chosen executable NEVER cached (re-resolved each call → Settings override takes effect next launch, no restart); ONLY memoize the pybullet-probe verdict keyed by absolute venv path (the slow `<python> -c "import pybullet"` subprocess); the override bypasses the probe cache and is validated fresh. **pybullet probe is an injected DI callable** (alongside spawn/signaler/cli) → hermetic tests via fake venv layout + fake probe. **run_prune** does NOT require pybullet: relaxed resolution accepts any env with a `drone-fly` script (preferring the pybullet one), erroring only if none exists — the pybullet gate is train/resume-only. Settings override wired via `_DEFAULT_SETTINGS` (server.py L34-39, default None) + a Settings-pane field; `create_app` injects a settings-aware resolver into `RunRegistry`. CWD stays `project_root`.

**Item 5 — Fly-outline app icon.** Hand-authored inline-SVG fly outline at `app/static/fly.svg` (body+wings+antennae, single-color, zero deps) → favicon in index.html + reused as the sidebar brand mark. `app/static/fly.png` (GTK/Qt) + `app/static/fly.ico` (Windows) for the pywebview window, wired best-effort/guarded in `__main__.py` (`webview.start(icon=...)`, honored on GTK/Qt, ignored elsewhere; __main__ is owner-eyeball only, never CI). PNG/ICO can be a scripts/ helper or hand-provided; runtime dep-free.

**Item 6 — Field info icons → accessible modal.** Merge `help`+`example` into each field descriptor via a NEW app-layer `app/field_help.py` ({field_name→{help,example}} for train+prune+settings); `configs_io.describe_*_fields` merge it in (config.py untouched). forms.js `buildForm` adds a `type=button` "ⓘ" inline-SVG button per field → opens modal composing purpose (curated) + example (curated) + constraints (derived from existing base_type/nullable/required/default/choices). Settings rows (app.js `renderSettings`) get the same button + settings help map. NEW `app/static/modal.js` `openModal({title,bodyHTML})`: focus trap, ESC, role=dialog aria-modal aria-labelledby, backdrop-click close, inline-SVG close, **restore focus** to invoker; single-instance; loaded before app.js.

**Item 7 — Menu icons + UX.** Inline-SVG icons for Slices/Train/Settings + all sub-items (define once, reuse; no icon font/CDN). aria-current on active, keyboard-operable disclosures, :focus-visible, responsive panels, clear empty states, consistent iconography. app.css gains nav-tree/disclosure, compact-picker, modal, info-icon, nav-icon, empty-state styles.

## Files Affected

### Production code [developer]
- `app/static/index.html` — nav restructure (Slices/Train/Settings + icons), favicon + brand SVG, load modal.js.
- `app/static/app.js` — leaf routes; nested nav tree + persisted expanded-set; split run detail into Status/Recordings/Config; compact recordings picker; Slices menu + slice detail; Settings `train_executable` field + info icons; nav icons; modal wiring.
- `app/static/forms.js` — per-field "ⓘ" info button → modal.
- `app/static/modal.js` — NEW accessible modal.
- `app/static/app.css` — nav-tree/disclosure, picker, modal, info-icon, nav-icon, empty-state styles.
- `app/static/fly.svg`, `app/static/fly.png`, `app/static/fly.ico` — NEW icon assets.
- `app/static/viewer-embed.js` — iframe src → `?embed=1`.
- `app/field_help.py` — NEW curated help/example (train+prune+settings).
- `app/configs_io.py` — `list_prune_config_names`; load/save prune-by-name; merge help/example.
- `app/runs.py` — per-call interpreter resolver (explicit cli → settings override → auto-detect w/ sys.executable fallback), injected pybullet probe, probe-verdict cache keyed by venv path, clear RunError, run_prune relaxed; `cli=` backward-compat.
- `app/server.py` — `/api/slice-configs` (+ `{name}` load/save); `train_executable` default; settings-aware resolver injected into RunRegistry.
- `app/__main__.py` — pywebview window icon (best-effort, guarded).
- `viz/viewer.js` — read `?embed=1` → `body.embed` (the ONE viewer JS edit, behind flag).
- `viz/viewer.css` — `.embed`-scoped surgical chrome-hide (INCLUDING `#meta-bar` — fully bare embed, per owner) + 2-col embedded layout + height:100% fill chain.
- `pyproject.toml` — expected no-op (hatch `packages=["src/drone_fly","app"]`); **VERIFY the new .svg/.png/.ico under app/static/ are actually shipped in the wheel — hatchling may not include non-.py files by default; add an artifacts/force-include rule if needed** (harmless for run-from-source, but confirm).
- `app/README.md` — document new nav, embedded viewer, training-interpreter Setting.

### Test code [qa] (tests/**)
- `test_app_runs.py` — interpreter resolution matrix: priority `.venv-cuda`>`.venv`>`.venv-*`>sys.executable; override-wins; override-invalid→RunError; sys.executable fallback chosen when no `.venv*` capable; no-candidate→RunError (train/resume); run_prune accepts non-pybullet env; probe-verdict cached once per venv path; existing explicit-`cli=` tests still green.
- `test_app_configs_io.py` — `list_prune_config_names`; help/example present in descriptors.
- `test_app_server.py` — `/api/slice-configs` GET/POST; settings includes `train_executable`; RunError→409 for no-interpreter.
- Front-end (app.js/forms.js/modal.js + viewer embed): no JS harness → owner-eyeball (as UC-61) + optional static-grep tests asserting index.html/app.js reference the new leaf routes/assets and viewer.js gates on the embed param.

## Risks & Considerations
- Embedded flight canvas fill relies on an unbroken `.embed` height:100% chain over a definite grid-row height; resize() already uses clientW/H so no viewer-logic edit — but dev must keep the ancestor chain intact.
- Standalone viewer must stay identical: every embed change is `?embed=1`/`.embed`-gated; chrome hidden not removed (no null-ref regressions).
- Interpreter probe cost mitigated by per-venv-path verdict cache; override bypasses it and is validated fresh.
- prune must not require pybullet (relaxed resolution) or it breaks on non-sim envs.
- Windows: Scripts vs bin, .exe suffix, owner's `.venv-cuda\Scripts\drone-fly` pattern.
- Help-text drift (app-layer, keyed by field name) — acceptable; constraints still auto-derive.
- Modal focus-trap restore + single-instance.
- No JS tests → backend (resolver, routes, enumeration, help merge) is the load-bearing tested surface; UI is owner-eyeball.

## Orchestrator (team-lead) hand-off notes added at persist time
- **Independently verified (second pair of eyes):** all 7 owner items map to concrete files; the pybullet/interpreter fix (item 4) is robust (per-call re-resolution, validated override→RunError, probe-gated auto-detect with sys.executable fallback, prune relaxed, hermetic via injected probe DI) and correctly root-caused (`_resolve_cli` picking the app venv). No regression to the 11 original ACs. Constraints held: vanilla/no-SPA/no-build-chain, offline/no-CDN (inline SVG), standalone viewer untouched (embed-gated), app under app/**.
- **Owner decision applied:** embedded viewer is FULLY BARE — `#meta-bar` stripped in embed mode (item 2 hide-set + viewer.css line updated above).
- **Developer TODO to confirm:** the wheel-packaging of the new binary icon assets (see pyproject.toml note) — verify or add a force-include; not a blocker for run-from-source.

Ready for Phase 2 (developer + QA).
