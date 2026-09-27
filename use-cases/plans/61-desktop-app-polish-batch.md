---
plan_for: use-cases/61-desktop-app-training-viewer.md (training-view polish batch, pre-merge to PR #68)
work_branch: feat/uc-61-desktop-app-training-viewer
team: drone-fly-uc-61
approved: 2026-09-27
---

# APPROVED implementation proposal — UC-61 desktop-app training-view polish (7 items)

Challenger approved (v2). Target `/workspace/drone-fly-uc-61-desktop-app-training-viewer`, branch `feat/uc-61-desktop-app-training-viewer` (PR #68). Owner confirmed the item-5 log deny-list (Q1) — see the "OWNER-CONFIRMED" note in item 5.

## Proposed solution (per item)

**1. Settings at top nav level.** Root cause: Settings renders via `leaf()` (`.nav-leaf`, `padding-left:26px`) so it indents like a Train child, while Slices/Train are `disclosure()` heads (`.nav-disclosure`, 14px). Add a `topLink(href,iconName,label)` helper → `<a class="nav-leaf nav-top">` (no chevron, keeps active-route/keyboard handling). Render Settings with it at tree top level. CSS `.nav-leaf.nav-top { padding-left:14px; color:var(--fg); }` aligns it with the group heads.

**2. Train-config → titled sections in a responsive grid + Save top-right.**
- **Backend-driven grouping.** Add `TRAIN_SECTIONS` (field→section) + `SECTION_ORDER` to `app/field_help.py`; `describe_train_fields()` merges a `section` key into each descriptor. Full 43-field assignment (contiguous in dataclass order — verified):
  Core: name, connectome, adapter, device, timesteps, n_envs, resume · Pruning: prune, prune_k · Recording: record, record_every, record_dir · Task randomization: randomize, randomize_dynamics, schema, randomize_obstacles, randomize_recharge_pads, randomize_repair_pads · Capacity guard: strict_capacity, capacity_floor · Curriculum: ent_coef, airborne_curriculum_enabled, airborne_curriculum_warmup_fraction, airborne_curriculum_anneal_fraction · Rate controller: rate_kp, rate_ki, rate_kd, rate_max_body_rate · PPO: n_epochs, batch_size, n_steps, learning_rate · Dynamics envelope: pybullet_mass_ratio_min/max, pybullet_tw_min/max, pybullet_arm_length_min/max · Control rate: control_hz, physics_ratio, command_latency_ms · Reward: altitude_weight, altitude_target.
- **forms.js:** iterate `fields` in original descriptor order, push to `rows` in that order; grouping only relocates the DOM node into a lazily-created section container appended to a `.form-sections` responsive grid (`repeat(auto-fill,minmax(340px,1fr))`) in first-seen order. **`collect()` untouched → omit-vs-default byte-identity is structural, not incidental.** Prune form (no `section`) renders flat exactly as today. Narrow sections stack `.field` label-above-control.
- **Save top-right convention:** extend `headBar`/`runHead` with a right-aligned `.view-head-actions` slot. Move each screen's primary commit button there: Settings Save, Run-Config Save, New-training Save, slice Generate/Regenerate. **Launch stays distinct** — New-training keeps `Save` and `Save & launch` as two separate buttons in the slot. renderTrainList's "New training" link unchanged.

**3. Integer number formatting.** `fmt`: in the `digits===undefined` branch → `Number.isInteger(v) ? v.toLocaleString("en-US") : String(v)` (thousands separators, no decimals). Add `fmtInt(v)`. `renderMetricsCsv` → `Number.isInteger(v) ? fmtInt(v) : fmt(v,3)` (kills the `100000.000`; genuine decimals keep 3dp). Existing `fmt(x,N)` callers unchanged → precision preserved.

**4. Remove "source" indicator.** Drop `" · source: jsonl"` suffix (keep `step X / Y · update N · F fps`) and the `"source: progress.csv (fallback)"` label (neutral label instead). Keep internal `snap.source` branching and the `"no progress yet"` empty state.

**5. Live-logs panel.**
- **Capture (runs.py):** spawn seam → `spawn(argv, cwd, *, log_path=None)`. `_default_spawn` **`os.makedirs(dirname(log_path), exist_ok=True)` then `open(log_path,"a")`** (fixes first-launch crash — `launch()` doesn't pre-create `logs/`), passes `stdout=fh, stderr=STDOUT`, parent closes fh after Popen (child keeps dup'd fd — no leak, no pump thread). launch/resume pass `training/<name>/logs/app.log`; run_prune passes none. Signals still hit the process group (state machine untouched).
- **Filter+read (new `app/logs.py`, stdlib-only):** capture RAW, filter at read time (raw log retained for debugging). Bounded byte-offset tail: cap to last 64 KB/poll, advance cursor to first complete line, `log()` a truncation marker on skip; drop torn tail after last `\n`; guards `since>filesize→0`, file-absent→`{lines:[],next:since}`. Deny-list drops known pybullet/physics noise (anchored/scoped — no bare `semaphore`/`OpenGL` substrings), keeps everything else (SB3 rollout tables, checkpoint saves, health verdicts, our lines, warnings/tracebacks).
- **Endpoint (server.py):** `GET /api/runs/{name}/logs?since=<offset>` → `{lines:[filtered], next:<offset>}` (mirrors the `since:int=0` status route).
- **Frontend (app.js/css):** scrollable `<pre class="log-panel">` below the metric boxes in `renderRunStatus`; tail folded into the existing 2.5s `tick` (closure `logCursor`, **adopt returned `next` verbatim** — don't assume `prev+len`, per challenger note); auto-scroll to bottom only when already at bottom.
- **OWNER-CONFIRMED deny-list (Q1, 2026-09-27):** drop-known-noise / keep-everything-else, raw log retained on disk. Case-insensitive, anchored where noted: `^pybullet build time:` · `^argv\[0]=` · `^b3Printf` · `^b3Warning` · `^b3.*semaphore` · `^ven ?=` · `^GL_(VENDOR|RENDERER|VERSION)` · `^(Vendor|Renderer|Version) ?=` · `^(startThreads|stopThreads|numActiveThreads|Thread with id)` · `^(EGL|GLX|MesaGL)` · whole-line GL-init/OpenGL-driver lines (e.g. `^Workaround for some crash in the Intel OpenGL driver`) · `^ExampleBrowserThreadFunc` · blank lines. [Confirm/adjust folded in from the owner's answer.]

**6+7. About page.** `renderAbout()` + route `#/about`; nav `topLink("#/about","info","About")` **below** Settings at top level. Static, offline: app name/purpose + version (`0.1.0`) + "Built by Harold Hormaechea".

## Files Affected
**Production [developer]:** `app/static/app.js` (1,3,4,5,6,7), `app/static/forms.js` (2), `app/static/app.css` (1,2,5,6,7), `app/configs_io.py` (2), `app/field_help.py` (2), `app/runs.py` (5), `app/logs.py` NEW (5), `app/server.py` (5), `app/README.md` (docs).
**Test [qa]:** `tests/test_app_configs_io.py` (each section a contiguous ascending run over descriptor order; field set/defaults/always_resolves unchanged), `tests/test_app_runs.py` (launch creates+opens `logs/app.log` when `logs/` absent; passes `log_path`; fakes at :56 & :286 get `log_path=None`), `tests/test_app_server.py` (`/logs` tail/since fixture; fakes at :54 & :228; config-save byte-identity stays green), `tests/test_app_logs.py` NEW (deny-filter noise-dropped/signal-kept incl. negative "semaphore"/"OpenGL"-kept cases; bounded byte-offset cursor).
**No JS test harness** → items 1/3/4/6/7 + item-2 DOM are owner-eyeball; backend metadata + item-5 endpoint/filter/capture carry the automated coverage.

## Risks & Considerations
- Spawn-seam signature change → 4 fake-spawn sites updated (enumerated). `_FakePopen.__init__` unchanged.
- Byte-offset cursor safe (append-only; guards for `since>filesize` and absent file); bounded 64 KB read prevents unbounded poll payloads.
- Deny-list false pos/neg mitigated by anchoring + retained raw log + owner confirmation (Q1).
- Item-2 byte-identity is now structural (collect() iterates descriptor order regardless of visual grouping).

## Orchestrator (team-lead) verification note
Independently reviewed: 7 items map to concrete files; AC4 omit-vs-default preserved structurally (challenger verified section runs are contiguous in dataclass order + guard test); item-5 log capture is a plain fd redirect (no pump thread) so it does NOT touch the pause/resume/stop signal state machine and keeps the DI spawn/signaler fakes hermetic; deny-list anchored + raw log retained. No `src/drone_fly` change (all app-side). Owner confirms the deny-list (Q1). After green: merge PR #68 to main (owner authorized).
