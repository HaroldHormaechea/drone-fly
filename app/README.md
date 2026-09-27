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
| `configs_io.py` | Introspect the real `TrainRunConfig`/`PruneRunConfig` dataclasses for the form schema; load/save YAML honouring omit-vs-default; validate through the real `from_mapping`. Enumerate + load saved slice (prune) configs. Merges curated help/example from `field_help.py` into descriptors. |
| `field_help.py` | Curated per-field `help` + `example` prose (train / prune / settings), keyed by field name. Merged into the form descriptors for the info-modal — the core config dataclasses stay untouched. |
| `status.py` | Tail `training/<name>/status.jsonl` (partial-write safe), with a `progress.csv` fallback. |
| `recordings.py` | Enumerate `episode_<n>.json[.gz]`, read + gunzip **server-side**, return parsed JSON. |
| `static/` | Vanilla front-end: `index.html` (shell) + `app.css`, `app.js` (hash router + disclosure nav-tree + views), `forms.js` (schema-driven controls + info buttons), `modal.js` (accessible dialog), `viewer-embed.js` (iframe → `window.loadDocument`), `fly.svg`/`fly.png`/`fly.ico` (app icons). No build step. |

## Navigation & views

The left nav is a keyboard-operable **disclosure tree** (rendered by `app.js`, not hard-coded in
`index.html`) with inline-SVG icons:

- **Slices** — `New slice` + one entry per saved `configs/prune/*.yaml`. A slice entry opens
  `#/slices/<name>`: its prune config in a form editor with a **Regenerate slice** button.
- **Train** — `New training` + one **collapsible** entry per run. Each run expands to three leaf
  views: **Status** (`#/train/<name>/status`, live progress + lifecycle controls), **Recordings**
  (`#/train/<name>/recordings`, an episode picker feeding the embedded viewer), and **Config**
  (`#/train/<name>/config`, the per-run config editor). `#/train/<name>` redirects to `.../status`.
  Expanded runs stay open across the 5-second nav refresh.
- **Settings** — `#/settings`.

Every config/settings field carries an **ⓘ info button** that opens an accessible modal (focus
trap, ESC/backdrop close, focus restored to the button) describing the field's purpose, an example,
and its derived constraints (type / required / default / choices).

### Embedded recording viewer

The Recordings view embeds `viz/viewer.html?embed=1` in an iframe. The `?embed=1` param is the one
embed-aware line in `viewer.js` (it adds `class="embed"` to `<body>`); all layout changes are
`.embed`-scoped CSS in `viz/viewer.css` — the standalone viewer (opened without `?embed=1`) is
byte-behaviour-identical. In embed mode the viewer is **fully bare**: the title, instructions, file
loader and provenance meta-bar are hidden, and the panels reflow to a compact 2-column layout
(brain + flight-actions stacked on the left, the 3D flight map filling the right).

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
