# drone-fly desktop app (UC-61)

A lightweight, local, single-user desktop application that replaces the training TUI + the
standalone HTML recording viewer. It is a thin FastAPI/Uvicorn front-end over the existing
`drone-fly` CLIs (it shells out to them) plus the reused, dependency-free brain/inputs/course
recording viewer (`viz/viewer.{html,js,css}`).

See the repository `README.md` → **Desktop app (UC-61)** for install and usage. This file documents
the module layout for maintainers.

## Layout

| Module | Responsibility |
|---|---|
| `server.py` | `create_app()` → FastAPI app: JSON API routers + static mounts (`/app`, `/viewer`). Head-less testable via starlette `TestClient`. |
| `__main__.py` | `python -m app`: run Uvicorn on a loopback ephemeral port in a thread, open a pywebview native window. The only display-dependent path (owner-eyeball; never CI). |
| `runs.py` | `RunRegistry`: enumerate `training/*` + `configs/train/*`, derive run state, and drive the launch / pause / resume / stop subprocess lifecycle. Spawner, signaler **and the pybullet probe** are injectable for tests. Resolves the training interpreter fresh per call (see below). |
| `configs_io.py` | Introspect the real `TrainRunConfig`/`PruneRunConfig` dataclasses for the form schema; load/save YAML honouring omit-vs-default; validate through the real `from_mapping`. Enumerate + load saved slice (prune) configs. Merges curated help/example from `field_help.py` into descriptors. Also builds the **live-status descriptor list** (`describe_status_fields()`). |
| `field_help.py` | Curated per-field `help` + `example` prose (train / prune / settings), keyed by field name. Also owns the train-config **section map** (`TRAIN_SECTIONS` + `SECTION_ORDER`) and the **live-status descriptor list** (`STATUS_FIELDS` + `STATUS_HELP` + `STATUS_GROUP_ORDER`) that drives the grouped Status view. Merged into the form/status descriptors — the core config dataclasses stay untouched. |
| `status.py` | Tail `training/<name>/status.jsonl` (partial-write safe), with a `progress.csv` fallback. |
| `logs.py` | Bounded, filtered byte-offset tail of `training/<name>/logs/app.log` (the redirected run stdout+stderr). Caps each poll to 64 KB, advances only over complete lines, and applies a read-time deny-filter — the raw log stays on disk. |
| `recordings.py` | Enumerate `episode_<n>.json[.gz]`, read + gunzip **server-side**, return parsed JSON. |
| `static/` | Vanilla front-end: `index.html` (shell) + `app.css`, `app.js` (hash router + disclosure nav-tree + views), `forms.js` (schema-driven controls + info buttons), `modal.js` (accessible dialog), `viewer-embed.js` (iframe → `window.loadDocument`), `fly.svg`/`fly.png`/`fly.ico` (app icons). No build step. |

## Navigation & views

The left nav is a keyboard-operable **disclosure tree** (rendered by `app.js`, not hard-coded in
`index.html`) with inline-SVG icons:

- **Slices** — `New slice` + one entry per saved `configs/prune/*.yaml`. A slice entry opens
  `#/slices/<name>`: its prune config in a form editor with a **Regenerate slice** button.
- **Train** — `New training` + one **collapsible** entry per run. Each run expands to three leaf
  views: **Status** (`#/train/<name>/status`, live progress + lifecycle controls + process logs),
  **Recordings** (`#/train/<name>/recordings`, an episode picker feeding the embedded viewer), and
  **Config** (`#/train/<name>/config`, the per-run config editor). `#/train/<name>` redirects to
  `.../status`. Expanded runs stay open across the 5-second nav refresh. The run list also carries a
  per-row **Delete** button — see *Deleting a run* below.
- **Settings** — `#/settings`. A **top-level** nav link (aligned with the Slices/Train group heads,
  not indented like a run child).
- **About** — `#/about`. A second top-level link below Settings; a static, offline page with the
  app name/purpose, version, and author.

Every config/settings field carries an **ⓘ info button** that opens an accessible modal (focus
trap, ESC/backdrop close, focus restored to the button) describing the field's purpose, an example,
and its derived constraints (type / required / default / choices).

### Config form sections & the top-right Save convention

The train-config form (New training / per-run Config) groups its fields into **titled section
cards** (Core, Pruning, Recording, Task randomization, Capacity guard, Curriculum, Rate controller,
PPO, Dynamics envelope, Control rate, Reward) laid out in a responsive grid. Grouping is
backend-driven (`field_help.TRAIN_SECTIONS` / `SECTION_ORDER`, merged into each descriptor as a
`section` key) and is **purely visual**: fields render in their original descriptor order and
`collect()` is untouched, so omit-vs-default byte-identity is preserved. The prune (slice) form has
no sections and renders flat, exactly as before.

Each screen's **primary commit button** (Settings Save, per-run Config Save, New-training Save,
slice Generate/Regenerate) lives in a **right-aligned header slot** (`.view-head-actions`). Launch
stays distinct — the New-training screen keeps **Save** and **Save & launch** as two separate
buttons; launching is never folded into a plain save.

**Enum & path comboboxes.** Fields with a *closed* set of valid values (adapter, device, schema,
prune rule) render as a strict `<select>` from the config dataclass's own `choices`. Fields whose
valid space is *open* (`resume`, `connectome`) render as a text input backed by a `<datalist>` of
suggestions — `resume` offers `auto` / `latest`, `connectome` offers the discovered pruned slices
from `GET /api/connectomes` (prune-config `out` dirs that exist + immediate `artifacts/pruned/`
subdirs). Suggestions are **advisory only**: free text is always accepted (a checkpoint `.zip`, a
fixture path, or leaving it blank), and the emitted YAML is byte-identical to before — the datalist
only autocompletes. Suggestion metadata (`options` / `options_endpoint` / `open`) is backend-owned
(`field_help.FIELD_OPTIONS`, merged into the descriptors); a failed dynamic fetch degrades silently
to a plain input.

### Deleting a run

Each row in the run list has a destructive **Delete** button opening a confirm dialog (`modal.js`
footer actions). By default it deletes only the **generated outputs** — checkpoints, recordings and
logs under the gitignored `training/<name>/` — via `DELETE /api/runs/{name}`. The committed run
definition (`configs/train/<name>.yaml`) is **kept**, so the run stays listed as a config-only
`ready` entry you can relaunch (the confirm copy and the success toast both say so). Tick the
dialog's checkbox to also remove the saved config (`?delete_config=true`), permanently dropping the
definition. Guards (in `runs.RunRegistry.delete`): `validate_run_name`, a **409** refusal for a run
in a live state (stop it first), and a realpath + `os.path.commonpath` containment check so a
symlinked `training/<name>` can't escape the training root. Deleting a run with nothing to remove is
a **404**.

### Loading feedback (skeletons + progress)

Navigation shows an instant nav highlight and, for any view that takes longer than ~120 ms to load,
a greyed **skeleton** placeholder (a fast view never flashes one — the threshold timer is cleared).
On the Status view the **Elapsed** field advances once a second between the 2.5 s status polls (a
drift-free client clock reconciled to each server reading; ETA stays on the poll cadence), and the
progress bar switches to an **indeterminate striped marquee** while a run is launching or reports no
measurable progress yet (no target set), instead of showing a dead 0-width bar.

### Live status metrics (Status view)

Below the progress bar the Status view shows the live training metrics as **grouped metric boxes**,
laid out under topic headings (**Progress**, **Rollout**, **Train**, **Health**, **Dynamics**). The
grouping, labels, value formatting, and per-box help are entirely **backend-driven**: `app.js`
fetches a descriptor list once from `GET /api/status-fields`
(`configs_io.describe_status_fields()` → `field_help.STATUS_FIELDS` + `STATUS_HELP` +
`STATUS_GROUP_ORDER`) and renders it, so the JS never hard-codes which fields exist or how to format
them — Python and JS cannot drift on the record shape. Each box reads its value by descriptor `path`
(or is `computed`, like ETA) and formats it via a closed `format.type` vocabulary (`float` +
`digits` / `int` / `seconds` / `duration` / `badge` / `text`). A completeness test derives its
expectation from a real emitted `status.jsonl` record, so a new emitter leaf without a descriptor
fails CI.

Empty/null contract: an unresolved path or a null/NaN leaf renders `—` (the box stays, so the grid
is shape-stable across polls); the optional **Health** and **Dynamics** groups are hidden entirely
when their block is null (a just-launched run shows Progress/Rollout/Train with `—` leaves and no
empty Health/Dynamics headings). Each box carries an **ⓘ info button** opening the same accessible
modal used by config fields; the Health box surfaces `health.message` in its popup. **ETA** is
derived front-end from throughput (`fps`, or steps ÷ elapsed) and the steps remaining to the target,
showing `—` until a rate is known and `done` at the target. The **CSV fallback** (`progress.csv`,
when no `status.jsonl` exists) still renders as a flat, ungrouped, help-less grid.

The status view runs in **fill mode** (`#view.view-fill`): the card and its log panel flex to the
bottom of the window so the log gets the leftover height. Status data, metric boxes, log lines, and
modal copy are **text-selectable** (a CSS `user-select` layer backing the native pywebview
`text_select=True`); the nav and buttons stay unselectable to keep a native feel.

### Live process logs (Status view)

The Status view shows a scrollable **process-log panel** below the metric boxes. Training launches
redirect the child's stdout+stderr to `training/<name>/logs/app.log` (a plain fd redirect set up in
`runs._default_spawn` — no pump thread, so the pause/resume/stop signal path is unaffected). The
front-end tails it via `GET /api/runs/<name>/logs?since=<byte-offset>`, folded into the same 2.5 s
status poll; the client adopts the server's returned `next` offset verbatim and auto-scrolls only
when already pinned to the bottom.

The **raw log is retained in full on disk** for debugging; filtering happens only at read time
(`logs.tail`). A conservative, anchored, case-insensitive **deny-list** drops known pybullet /
OpenGL / thread-init boot noise (never a bare `semaphore` / `OpenGL` substring) and keeps everything
else — SB3 rollout tables, checkpoint saves, health verdicts, warnings, and tracebacks. Each poll is
bounded to the last 64 KB; when that skips earlier output a single truncation-marker line is shown.

### Embedded recording viewer

The Recordings view embeds `viz/viewer.html?embed=1` in an iframe. The `?embed=1` param is the one
embed-aware line in `viewer.js` (it adds `class="embed"` to `<body>`); all layout changes are
`.embed`-scoped CSS in `viz/viewer.css` — the standalone viewer (opened without `?embed=1`) is
byte-behaviour-identical. In embed mode the viewer is **fully bare**: the title, instructions, file
loader and provenance meta-bar are hidden, and the panels reflow to a compact 2-column layout
(brain + flight-actions stacked on the left, the 3D flight map filling the right).

The **3D flight** panel now defaults to an **angled three-quarter view** (`VIEW_PRESETS.angled`,
labelled *3D* in the flight view picker) instead of top-down, and **auto-fits the whole course to
~75% of the panel** — an aspect- and orientation-aware fit (`fitToScene()`, with a perspective
refinement pass so a long course never clips) that replaces the old bounding-sphere heuristic. It
re-fits on resize (so the embedded iframe's initial layout race self-heals and aspect changes
re-frame) and on every scene/recording load (each episode lands on the default view), unless the
user has manually zoomed. front / side / top-down remain available in the picker. This is a
viewer-only change; `viewer-embed.js` is unchanged.

### Training interpreter resolution (Settings → *Training interpreter*)

The app runs in its own venv, which may lack `pybullet` (a training-only dependency). `RunRegistry`
therefore re-resolves the `drone-fly` executable on **every** launch / resume / prune (never cached,
so a Settings change takes effect on the next launch — no restart), in order:

1. an explicit test-seam `cli=` (unit tests only);
2. the Settings **`train_executable`** override — a venv dir, its python, or a `drone-fly` console
   script; an unresolvable value is a hard error (no silent fallback);
3. auto-detect, gated on an `import pybullet` probe, over `.venv-cuda` > `.venv` > other `.venv-*` >
   the app's own interpreter;
4. otherwise an actionable error (`… set the training interpreter in Settings, or create one with
   uv sync --extra sim`).

Only the per-venv probe verdict is memoised (keyed by absolute venv path). Pruning does not need
pybullet, so it uses relaxed resolution (prefers a capable env, accepts any env with a `drone-fly`
script).

## Design constraints

- **CLI-only under the hood.** Every operation reduces to a `drone-fly` subcommand or a filesystem
  read. The only new `src/drone_fly` code is `train/status_emitter.py` (JSONL progress, AC6) and
  `train/checkpoint_signal.py` (lossless-pause checkpoint, AC7).
- **viewer reused unchanged.** `viz/viewer.js` is served as-is and driven via `window.loadDocument`;
  gzip is decompressed server-side so playback never depends on the webview's `DecompressionStream`.
- **Hermetic tests.** The backend is exercised via `TestClient` with a fake subprocess spawner — no
  GPU, no display, no real training. `pywebview` is intentionally outside the `dev`/CI extra.
- **Local-only.** Binds `127.0.0.1`. The client/server split is kept clean so remote-GPU could be a
  later config change, but remote control is not built here.
