# UC-61 — Desktop App for Slices / Trainings / Recordings — Design & Architecture Analysis

> **Status: ANALYSIS ONLY.** Owner directive is to analyze and push to `main` for review, not to build. No implementation is proposed here beyond naming the thin glue a build would need. See `use-cases/61-desktop-app-training-viewer.md`. Produced by the dev-team (analyst↔challenger, challenger-approved after one revision round; every load-bearing repo claim independently verified against the code).

## 1. Goal & scope recap
A single-user, **local developer tool** that replaces two things the owner touches today: the training **TUI** (`src/drone_fly/train/tui/**`) and the standalone **HTML recording viewer** (`viz/viewer.{html,js}`). One UI drives the whole loop: define a slice, configure + launch named trainings, monitor a run (progress / pause / resume), and play back its recordings in the existing brain+inputs+course viewer.

Hard design constraints, all confirmed against the code:
- **Thin front-end over existing Python CLIs.** Every operation already exists as a `drone-fly <subcommand> --config <yaml>` call (§3). The app shells out; it adds no training/env/runtime logic to `src/drone_fly`.
- **Reuse `viz/viewer.js` verbatim** — it is dependency-free, `file://`-safe, and already exposes a programmatic load hook (§8).
- **Lightest framework that fits the Python+JS stack** (§2).
- Single-user, no multi-user/hosted requirements (consistent with brief non-goals).

## 2. Framework tradeoff & recommendation (AC1)

| Option | New runtime dep | Bundle | Per-OS packaging | Fit with this stack | Remote-GPU topology |
|---|---|---|---|---|---|
| **Electron** | Node + bundled Chromium | ~120–180 MB | electron-builder per OS | Poor — a whole JS runtime bolted onto a Python project | Needs custom IPC |
| **Tauri** | **Rust toolchain** + system webview | ~3–10 MB | Rust cross-compile per OS (incl. Windows GPU box, macOS) | Poor — introduces Rust into a Python+JS repo; biggest build-chain tax | Needs custom IPC |
| **pywebview** | Python + system webview | tiny (~ MBs) | PyInstaller per OS | **Good** — native Python, calls CLIs in the same venv; but per-OS webview quirks (esp. DecompressionStream, §8) | In-process only; no natural split |
| **Local FastAPI/Flask server + browser** | one Python web framework (into the *existing* uv env) | ~0 new binary | none — `uv run` launches it | **Best** — same interpreter/venv as the CLIs; serves the dependency-free `viewer.js` as a static route; subprocess mgmt in Python | **Native** — server on the GPU box, browser anywhere |

**Recommendation: a local FastAPI + Uvicorn backend, opened in the system browser, serving a small front-end that reuses `viewer.js` unchanged.** Rationale: it adds only a Python web framework to an environment that is *already* Python (the CLIs), bundles **no** Chromium/Rust/native-webview runtime, reuses `viewer.js` via one static route, runs the CLIs as subprocesses in the same venv with no marshaling, and — uniquely — its client/server split is the only option that natively answers the "training runs on a remote/Windows GPU box, UI runs elsewhere" pitfall. **Optional later:** wrap the same server in a **pywebview** window to give a genuine "desktop app" frame without changing the backend (pywebview just points at `http://127.0.0.1:<port>`). So the lightest viable core is FastAPI; the native-window feel is an additive, deferrable shell. Electron is explicitly rejected as the heaviest and worst-fitting; Tauri is rejected because a Rust build chain is the largest packaging tax for the least stack fit.

**Front-end tech:** deliberately **vanilla HTML/CSS/JS** (optionally a single tiny helper like Alpine.js or lit, ~KBs) — **no React/Svelte/Vue SPA framework**. The forms are static field sets and the nav is a handful of hash routes, so a heavy SPA build chain would directly undercut the "as lightweight as possible" mandate and add an npm/bundler step this Python repo otherwise avoids. This also keeps `viewer.js` (already dependency-free vanilla JS) a native peer, served from the same static route with zero integration friction.

## 2b. App shell, navigation & editing model (AC2, AC9)
**Navigation (AC2).** A persistent **left-nav** with three top-level groups — **Generate slices / Train / Settings** — each expandable to sub-options (e.g. Train → the named runs + "New training"; Generate slices → "New slice" + existing slices under `artifacts/pruned/*`). Clicking a **terminal (leaf)** item swaps the **right-side main view** to that surface; non-leaf items only expand/collapse. The route model is a simple hash route (`#/train/<name>`, `#/train/new`, `#/slices/new`, `#/settings`) so each main view is deep-linkable and back/forward work; no server round-trip on nav (the front-end owns routing, the backend only serves data/CLI endpoints).

**Settings pane content.** Holds only local tool config (not training params): project root / working directory (where CLIs are invoked), default connectome path, server host/port, and a "gunzip recordings server-side" toggle (§9). No secrets, no accounts (single-user). (Some of these could instead be launch flags — flagged as an open question if the owner prefers zero settings surface for MVP.)

**Editing model — explicit Save, no auto-save, no history (AC9).** Every form (train config, slice config, settings) holds pending edits in **local UI state only**. Nothing is persisted on blur / focus-change / navigation — the exact behavior the owner called out. A single explicit **Save** button is the *only* commit point: it serializes the form to its YAML and writes it, **overwriting** `configs/train/<name>.yaml` (or the slice/settings file) in place. **No version history / no snapshots** are kept — a Save is a plain overwrite, and leaving a form with unsaved edits discards them (with an optional "unsaved changes" guard prompt). Launch is a separate explicit action from Save, so the user can edit → Save → then Launch.

## 3. CLI surface every op reduces to (grounding for AC3/AC4/AC5/AC10)
Console script `drone-fly` (run via `uv run drone-fly …`), dispatched in `src/drone_fly/cli/__init__.py`. All four heavy commands are **single-`--config <yaml>`**:
- `train --config <yaml> [--no-tui]` — the run form target.
- `evaluate --config <yaml>` — replay/scoring a checkpoint.
- `prune --config <yaml>` — **this is "define/generate a slice"** (§4).
- `prune-trained --config <yaml>` — post-training activation prune.
- `clean [--yes] [--include-prunes]`, `fetch-connectome`, `smoke-train` — helpers.
All config schemas + validation live in `src/drone_fly/config.py`; a bad config exits 2 with a one-line message (no stack trace). **Every UI form serializes to one of these YAMLs and the app invokes the matching subcommand — nothing more.**

### Per-run output layout (the run registry, AC5)
`config.py: run_layout(name)` routes every train run under **`training/<name>/{checkpoints,logs,recordings}/`**, derived solely from the required `name` field; differently-named runs never collide. **This filesystem layout *is* the multi-named-run registry** — the app enumerates `training/*/` to list runs and find each run's checkpoints/logs/recordings, no database needed.

## 4. Slice-definition surface (AC3)
Wraps `drone-fly prune --config`. `PruneRunConfig` (config.py) has exactly four fields the form exposes:
- `connectome` (str, optional → defaults to auto-downloaded full MaleCNS; or point at `tests/fixtures` / a prior slice)
- `out` (str, required — e.g. `artifacts/pruned/k2`)
- `prune_k` (int, default `DEFAULT_PRUNE_K`)
- `prune_rule` (str, default rule)
Form = 2 text inputs + 1 number + 1 select → write YAML → `drone-fly prune --config`. No manual file editing.

## 5. Train-configuration surface (AC4)
`TrainRunConfig.from_mapping` (config.py) is the full key set a "train config form" edits; each maps 1:1 to a control. Grouped:
- **Identity/core:** `name`* (text, required — also the run dir), `connectome` (text/file), `adapter` (select: auto/…), `device` (select), `timesteps` (int), `n_envs` (int ≥1), `resume` (select/text: auto|latest|path).
- **Recording:** `record` (toggle), `record_every` (int), `record_dir` (text).
- **Task randomization:** `randomize`, `randomize_dynamics` (toggles); `schema` (select of NAMED_SCHEMAS); three tri-state placement toggles `randomize_obstacles|recharge_pads|repair_pads` (unset/true/false — a form needs a 3-way, not a checkbox).
- **Capacity guard:** `strict_capacity` (toggle), `capacity_floor` (int).
- **Curriculum:** `ent_coef`, `airborne_curriculum_enabled`, `airborne_curriculum_warmup_fraction`, `…_anneal_fraction`.
- **Rate controller (UC-55):** `rate_kp/ki/kd`, `rate_max_body_rate` (sliders, ≥0).
- **PPO hyperparams (UC-54):** `n_epochs`, `batch_size`, `n_steps`, `learning_rate`.
- **Dynamics envelope (UC-56):** `pybullet_mass_ratio_min/max`, `pybullet_tw_min/max`, `pybullet_arm_length_min/max`.
- **Control rate (UC-57):** `control_hz` (default 50), `physics_ratio` (default 10), `command_latency_ms` (default 0).
- **Reward (UC-58):** `altitude_weight`, `altitude_target`.
Important serialization subtlety the form must respect: many keys are **"omit == leave dataclass default"** (null vs explicit value are semantically different, e.g. tri-state toggles, curriculum knobs). So the form must distinguish "unset" from "set to default value" — emit the key only when the user actually sets it (or expose an explicit unset state). Validation is delegated to `config.py` (exit 2 on bad input); the UI surfaces that one-line error. **Multiple named trainings** = multiple named YAMLs → multiple `training/<name>/`.

## 6. Launch & manage trainings (AC5)
- Backend spawns `uv run drone-fly train --config <path> --no-tui` as a **subprocess**, CWD = **project root** (all CLIs are CWD-relative — `run_layout`, `clean`, recorder all resolve against CWD). `--no-tui` is belt-and-suspenders: the loop already auto-disables the TUI on a non-TTY (`sys.stdout.isatty()`), and CSV/TensorBoard are written either way.
- **Tracking:** a small in-backend map `name → {pid, popen, state}` for live control, backed by the durable `training/<name>/` filesystem for enumeration across restarts. Concurrent named runs = multiple subprocesses (GPU contention is the user's call).
- Stop = terminate the subprocess (see §11 for the Windows caveat).

## 7. Progress plumbing (AC6) — the single most-likely new glue
**Finding, grounded:** there is **no structured status emitter today.** The TUI reads live *in-process* from SB3 callbacks (`train/tui/callback.py` snapshots `ep_info_buffer` + `logger.name_to_value` each rollout) — none of that is exposed to a separate process. The **only** always-written structured artifact is SB3's **`progress.csv`** (+ TensorBoard) in `training/<name>/logs/` — written by `_make_logger` with formats `["csv","tensorboard"]` regardless of TUI state.

Two options:
1. **Zero-glue (works today):** the app **tails `training/<name>/logs/progress.csv`** — columns include `time/total_timesteps`, `rollout/ep_rew_mean`, `rollout/ep_len_mean`, `train/*`. Enough for a progress bar (`total_timesteps` vs configured `timesteps`) and the reward/loss curves. Cadence = one row per rollout (per `dump_logs`), so it's coarse and lacks the TUI's intra-rollout heartbeat and health/dynamics-summary lines.
2. **Recommended thin glue:** add a small **JSONL status emitter callback** in `src/drone_fly/train` (e.g. `training/<name>/status.jsonl`, one line per rollout with timesteps, metrics, health verdict, dynamics summary). This is the *one* piece of new backend code worth adding; it is pure observability (no training/env logic) and gives the app parity with the TUI. The backend then pushes updates to the UI via **SSE or WebSocket** (or the UI polls `/runs/<name>/status`).

**Recommendation:** ship MVP on CSV-tail (no `src/drone_fly` change), add the JSONL emitter in a later phase for TUI-parity richness. Flag to owner: the emitter is the main new glue in `src/drone_fly`.

## 8. Pause / resume (AC7) — real feasibility
**There is no live pause.** SB3 + this loop support only **checkpoint → stop → resume**:
- Checkpointing is periodic via `CheckpointCallback(save_freq, save_vecnormalize=True)` → `ppo_racer_<steps>_steps.zip` + `ppo_racer_vecnormalize_<steps>_steps.pkl` in `training/<name>/checkpoints/`.
- Resume path (`loop.py`): `resume: auto|latest|<dir>` → `find_latest_checkpoint` picks the highest `_<steps>_steps.zip` → `PPO.load(ckpt, env=venv)` + `VecNormalize.load(pkl)`.
- A clean resume needs three things co-located: the checkpoint `.zip`, its **matching** `vecnormalize_<steps>_steps.pkl`, and the **same config** (the app re-launches with the same YAML + `resume: auto`).

So the honest mechanism:
- **"Pause" = terminate the subprocess.** Progress since the last periodic checkpoint is **lost** (up to `checkpoint_freq` steps).
- **"Resume" = relaunch `train` with `resume: auto`** on the same YAML → continues from newest checkpoint.
- **Optional glue for lossless pause:** a "checkpoint-now-then-exit" signal handler so pause flushes a checkpoint at the current step before terminating. Small, but it *is* new code in `src/drone_fly` — call it out. MVP can ship lossy pause (terminate + resume:auto) and document the caveat.

## 9. Recording access + playback (AC8)
- Recordings are written to **`training/<name>/recordings/episode_<n>.json[.gz]`** (`record/recorder.py`, gzip optional; numbering continues across resumes). The app enumerates that dir per run.
- **Embedding the viewer needs no viewer changes.** `viz/viewer.js` defines a top-level `loadDocument(doc, name)` that (as a classic `<script>`) is reachable as **`window.loadDocument`** — exactly how `scripts/screenshot_viewer.mjs` injects a recording headlessly (`page.evaluate((d) => window.loadDocument(d, "…"), doc)`). So the app: serves `viewer.html/js/css` on a static route in the webview/browser, reads the chosen `episode_<n>.json[.gz]`, gunzips if needed, and calls `window.loadDocument(parsed, name)`. (The file-picker path via FileReader+DecompressionStream still works too, but the programmatic hook is cleaner for an app.)
- **Gzip caveat:** `viewer.js`'s in-browser gunzip relies on `DecompressionStream`. Present in modern Chromium/Edge WebView2, but not guaranteed in every system webview (pywebview on some Linux). **Mitigation:** have the FastAPI backend gunzip server-side and hand plain JSON to `loadDocument`, sidestepping webview capability entirely.

## 10. CLI-only-under-the-hood confirmation (AC10)
Confirmed: every operation reduces to an existing CLI + existing assets:
- slice → `prune --config`; train → `train --config`; monitor → read `training/<name>/logs/`; recordings → read `training/<name>/recordings/` + `viewer.js`; evaluate/prune-trained → their `--config` CLIs.

**New code required, named precisely:**
1. The app itself (backend + front-end) — lives **outside** `src/drone_fly` (e.g. `app/` or `tools/desktop/`); not training/env/runtime logic.
2. *(recommended, optional)* a JSONL **status emitter** callback in `src/drone_fly/train` for TUI-parity progress (§7).
3. *(optional)* a "checkpoint-now-on-signal" handler for lossless pause (§8).
4. The run registry can be **pure filesystem** (`training/*/`) — no `src/drone_fly` change.

Nothing else in `src/drone_fly` needs to change.

## 11. Phased delivery, risks, effort (AC11)
**Phase 0 — MVP (thin vertical slice):** configure + launch + monitor **one** training and open its recordings. Train form (common-knob subset) → write `configs/train/<name>.yaml` → spawn `train --no-tui` → progress via **CSV-tail** → list `recordings/` → play in embedded `viewer.js`. Local-only, single run. *~1–1.5 wk.*
**Phase 1 — Run management:** multiple named runs (list from `training/*/`, concurrent), stop + lossy pause/resume, full train-config form (all §5 keys incl. tri-state + omit-vs-default handling), one-line config-error surfacing. *~1 wk.*
**Phase 2 — Slice + settings + eval:** prune (slice) form (§4), Settings surface, evaluate/prune-trained forms; left-nav complete (Generate slices / Train / Settings). *~1 wk.*
**Phase 3 — Richness/packaging:** JSONL progress emitter (TUI parity) + SSE/WebSocket, lossless-pause glue, optional **pywebview** native window + PyInstaller packaging, remote-GPU topology. *~0.5–1 wk.*

**Risks & mitigations:**
- **Cross-platform packaging** (owner's Windows GPU box + macOS sim): FastAPI+browser needs no bundling (mitigates most of it); defer pywebview/PyInstaller to Phase 3.
- **Subprocess lifecycle:** zombies / no `SIGTERM` on Windows — use `Popen` groups + `taskkill`/`CTRL_BREAK` on Windows, `terminate()` elsewhere; reconcile state from `training/*/` on backend restart.
- **Progress plumbing:** CSV cadence is coarse and lacks health/heartbeat — mitigate with the JSONL emitter (Phase 3).
- **Pause/resume:** lossy up to `checkpoint_freq` — document it; add checkpoint-on-signal glue if unacceptable.
- **GPU/host coupling & topology:** assume **local-only first**; the client/server split keeps remote-GPU a config change (server on GPU box), not a rewrite.
- **Viewer gzip in webview:** gunzip server-side (§9).
- **Port conflicts / lifecycle:** bind `127.0.0.1` on an ephemeral port, show the URL.

## 12. Open questions for the owner
1. **Topology:** local-only first (assumed), or is remote control of the Windows/GPU box in scope for an early phase?
2. **Framework final call:** approve **FastAPI + browser** (optionally + pywebview window later)? Or is a genuine native-window (pywebview from day one) required, or Electron mandated despite the weight?
3. **Progress glue:** is adding the thin **JSONL status emitter** to `src/drone_fly/train` acceptable (for TUI-parity progress), or must the app stay CSV-tail-only?
4. **Pause semantics:** is **lossy** pause (terminate + `resume: auto`, losing ≤ `checkpoint_freq` steps) acceptable for MVP, or is the **checkpoint-on-signal** glue required up front?
5. **Settings surface:** is a Settings pane wanted for MVP (project root, connectome path, port, gzip toggle), or should those be launch flags/env for now and Settings deferred?
