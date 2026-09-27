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
| `runs.py` | `RunRegistry`: enumerate `training/*` + `configs/train/*`, derive run state, and drive the launch / pause / resume / stop subprocess lifecycle. Spawner + signaler are injectable for tests. |
| `configs_io.py` | Introspect the real `TrainRunConfig`/`PruneRunConfig` dataclasses for the form schema; load/save YAML honouring omit-vs-default; validate through the real `from_mapping`. |
| `status.py` | Tail `training/<name>/status.jsonl` (partial-write safe), with a `progress.csv` fallback. |
| `recordings.py` | Enumerate `episode_<n>.json[.gz]`, read + gunzip **server-side**, return parsed JSON. |
| `static/` | Vanilla front-end: `index.html` (shell) + `app.css`, `app.js` (hash router + views), `forms.js` (schema-driven controls), `viewer-embed.js` (iframe → `window.loadDocument`). No build step. |

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
