# drone-fly desktop app

A local, single-user desktop app over the frozen-K1 reservoir pipeline in `gpu_prototype/`. From
one window you can:

- launch a reservoir run,
- watch its `[aug]` progress live,
- run the honesty-gated audits (`verify_proc.py` / `diagnose_proc.py`),
- record episodes (`record_aug.py`) and replay them in the embedded brain + actions + 3D-course
  viewer (`viz/viewer.{html,js,css}`, reused unchanged).

The app is a thin front-end. Every action spawns an existing `gpu_prototype/` script as
`micromamba run -n <env> python <script> ...`, or reads a file those scripts already write. The
only app-side code is coordination glue: a run registry, log and audit parsers, and
`app/reservoir_launch.py`, which forwards arguments to `train_aug`. It contains no training, env,
reward, or audit logic.

## Install and run

```bash
uv sync --extra app          # fastapi + uvicorn + pywebview
python -m app                # local server on an ephemeral loopback port + native window
python -m app --no-window    # server only; open the printed URL in a browser
```

`drone-fly-app` is the same entry point as a console script. `--project-root`, `--host` and
`--port` override the defaults.

## Preflight requirements

The Runs page and Settings show a **Preflight** banner when any of these is missing:

- **micromamba** must be on `PATH`, or set to a full path in Settings (`micromamba_executable`).
- **The `dronefly` env** (Settings `conda_env`) must have torch and the `gpu_prototype`
  dependencies. The app checks it once with `micromamba run -n dronefly python -c 'import torch'`.
  If micromamba or the env is missing, launch, audit and record fail with an actionable message.
  They never fall through to another interpreter.
- **The project root must be `/workspace/drone-fly`.** `gpu_prototype` hardcodes this path in
  `reservoir_aug.K1_PATH` (the pruned connectome), in the named launchers' warm-start bases, and
  in `record_aug.py`'s output dir. On any other root, training and audits fail, and recordings land
  where the app cannot see them.
- **The named-recipe warm-start bases** must exist: `gpu_prototype/proc_shapeA_term.pt` and
  `training/07-proc-course-flight/model_readout.pt`.

## Reservoir workflow

### 1. New run (`#/train/new`)

Choose a run name, a recipe, env flags, knobs, and a `save_path`, then click **Save** or
**Save & launch**. Nothing is saved automatically, and Save overwrites
`training/<name>/run.json` with no history.

| Recipe | Runs | Editable |
|---|---|---|
| `exp11_count-curriculum` | `gpu_prototype/exp11_count-curriculum.py <save> <budget>` | budget, save_path, flags |
| `exp12_gap-steering` | `gpu_prototype/exp12_gap-steering.py <save> <budget>` | budget, save_path, flags |
| `exp13_count-gap` | `gpu_prototype/exp13_count-gap.py <save> <budget>` | budget, save_path, flags |
| `train_shaping` | `gpu_prototype/train_shaping.py <save> <budget>` | budget, save_path, flags |
| `train_aug` (generic) | `app/reservoir_launch.py --save-path X --budget Y [--<knob> v ...]` | every knob |

- **Named launchers.** These hardcode every knob except the budget, including `warm_from`. The
  form shows those knobs read-only as "set by `<launcher>`", with the fixed value. Their default
  flags and budget (`14e6`) come from each launcher's docstring `Run:` line.
- **Generic recipe.** The `train_aug` recipe forwards only the knobs you fill in. A blank knob keeps
  `train_aug`'s own default, which the placeholder shows. `warm_src_extra_dim` is used only when
  `warm_from` is set; if you leave it blank, the runtime `EXTRA_DIM` is used.
- **Env flags.** The five flags are `PROC_TRACTABLE`, `PROC_RAYCAST`, `PROC_SHAPING`,
  `PROC_OBS_TERMINAL` and `PROC_GAP`. Each ticked flag is set to `"1"` on the child process. Unticked
  flags are left unset, never `"0"`, because `gates_proc_env` treats any non-empty value as on. Every
  `PROC_*` and `AUDIT_SEED` inherited from the app's own environment is stripped first.
- **`save_path`.** The default is `training/<name>/model.pt`, which is gitignored. The path must end
  in `.pt` and sit inside the project root. It must not be inside a committed or external dir, and
  it must not be used by another run.

The child runs with CWD `gpu_prototype/` and `PYTHONUNBUFFERED=1`. Its stdout and stderr are
appended to `training/<name>/logs/app.log`, after an `[app] launch ...` marker line that separates
one attempt from the next.

### 2. Status (`#/train/<name>/status`)

- **Metrics.** The page parses `train_aug`'s periodic line
  `[aug] it N steps XM fps F ep_rew/step r tilt Tdeg [compl C% lap L]` into iteration, Msteps, fps,
  ep_rew/step, tilt, completion EMA and lap. Completion shows "—" when the env prints no metric.
  Malformed or partial lines are skipped.
- **Other panels.** Curves for reward, completion and tilt; a progress bar against the budget; best-effort GPU usage
  from `nvidia-smi`; and a tail of the raw log.
- **Stop.** Stop terminates the run's process group. There is no pause or resume, because
  `train_aug` has neither a signal trap nor a resume path.

### 3. Audits (`#/train/<name>/audits`)

These run `verify_proc.py <ckpt> <eps>` or `diagnose_proc.py <ckpt> <eps>`.

- **Inputs.** The audit's flags are pre-filled from `run.json`. For committed or external models
  you must enter them, because `gates_proc_env` reads flags at import time and they must match
  training. `AUDIT_SEED` defaults to 1234, so models are compared on identical courses, and the
  episode count defaults to 512. Both defaults can be changed in Settings.
- **Output.** Results go to `.dev-app/audits/<name>/<id>.log`, with a JSON sidecar recording the
  flags, seed, episode count and exit code. They are never written into `training/`, and they
  survive an app restart.
- **When audits are refused.** Audits (and recording) are disabled, with the reason shown, while a
  run is running or stopping (`train_aug` rewrites the checkpoint every 200 iterations). They are
  also disabled when there is no checkpoint. For example, committed `01`, `08` and `09` show
  "no model_readout.pt".

### 4. Recordings (`#/train/<name>/recordings`)

- **Record.** "Record episodes" (default 10) runs `record_aug.py proc <save_path> <name> <n>`
  with the run's flags.
- **Play.** Pick an episode to play `training/<name>/recordings/episode_<n>.json[.gz]` in the
  embedded viewer. The server gunzips `.gz` files before sending them.

### 5. Continue from this checkpoint

This button appears on finished or partial app runs, and on committed or external dirs that have
a `model_readout.pt`. It opens a pre-filled **New run** using the generic `train_aug` recipe:
`warm_from` is set to that checkpoint, the source run's flags are copied, a name is suggested, and
`warm_src_extra_dim` is left blank.

## Run states

| State | Meaning |
|---|---|
| `ready` | `run.json` saved, no checkpoint yet |
| `running` / `stopping` | process live (stopping = Stop sent) |
| `finished` | this attempt's log ends with `[aug] saved <save_path>` |
| `stopped` | stopped by the user (this session) |
| `failed` | exited non-zero, or exited 0 without the saved line (this session) |
| `partial` | a `.pt` exists without the saved line: **partial checkpoint (auditable, not resumable)** |
| `committed` | `training/<dir>` without `run.json` whose name matches `^\d{2}-` or that is git-tracked |
| `external` | any other `training/<dir>` without `run.json` (e.g. a pre-app CLI run) |

**State after a restart.** When the app restarts, a run that was still training cannot be
re-attached. Its state is rebuilt from disk using the saved-line rule (`finished`, `partial` or
`ready`).

**Overwrite.** Launching a run whose `.pt` already exists asks for confirmation first, because
`train_aug` restarts from scratch and overwrites the checkpoint.

**Committed and external dirs are read-only.** The app never launches, records, or deletes them,
and never accepts a `save_path` inside them. Audits of these dirs write only to `.dev-app/`.

**Delete.** Deleting an app run removes only `training/<name>/`. A `save_path` outside that dir is
kept and reported back.

## Honesty verdict

A high completion % that barges through obstacles or hovers is **not** a success
(`training/EXPERIMENTS.md`: "a higher % that cheats does NOT count"). Completion is only shown
inside the verdict panel, next to the cheat checks.

The verdict is computed from `verify_proc` output with these ordered rules (`app/audits.py`):

1. **Missing lines.** If the completion, BARGING or HOVERING line is missing, the verdict is
   `unverified`. No further rules apply.
2. **Hovering.** These apply whether or not anything completed:
   - mean speed below **0.5 m/s** is `dishonest`;
   - mean tilt below **5°** is `suspect`.
3. **Overall barging.** Without `PROC_OBS_TERMINAL`, overall barging above **2%** is `suspect`.
   With `PROC_OBS_TERMINAL` on, contact ends the episode, so overall barging equals the
   collision-death rate. It is shown, but it is not a cheat signal.
4. **Completed-run barging.** If any episode completed, the "among COMPLETED runs" line must be
   present; otherwise the verdict is `unverified`. Completed-run barging above **2%** is
   `dishonest`, and above **0%** is `suspect`.
5. **Nothing completed.** If nothing completed and rules 2–3 found nothing, the verdict is
   `nothing-completed`.
6. **Honest.** Otherwise the verdict is `honest`.

When more than one rule fires, all reasons are listed and the most severe verdict wins:
`dishonest > unverified > suspect > nothing-completed > honest`.

- **Failed or interrupted audits.** If an audit exits non-zero, or never reports an exit code, the
  verdict is `unverified` and a red "Not a success" banner is shown. Any partial output appears
  only under that banner, with the log tail.
- **Diagnose audits.** `diagnose_proc` has no cheat checks. Its panel shows the verdict of the
  newest finished verify audit of the same run, or `unverified` if there is none.

## Deferred / out of scope

- **AC9 (EXPERIMENTS.md registration + `save_readout.py` promotion) is deferred.** The app does
  not append rows to `training/EXPERIMENTS.md` and does not promote checkpoints into committed
  `training/NN-<slug>/` dirs. Do both by hand, as before.
- **Not exposed.** Other env flags (`PROC_TUNNELS`, `PROC_MULTILAP`, `PROC_VARIETY`, `PROC_FLOOR`,
  `PROC_NO_OBS`) and launchers (exp14, exp16–20). They are still stripped from the inherited
  environment.

## Owner-eyeball checklist (not covered by CI)

CI drives the backend headless through `TestClient` with fake spawners and probes. Check these by
hand on a machine with a display, micromamba and a GPU:

- [ ] `python -m app` opens the native window with the fly icon; text in the status, log and
      audit panels is selectable.
- [ ] Nav: Train lists app runs plus committed/external dirs (with a "read-only" badge);
      each run expands to Status / Audits / Recordings (/ Config for app runs).
- [ ] New run: switching recipe resets the default flags. Named recipes show fixed knobs
      read-only. Generic `warm_src_extra_dim` is disabled until `warm_from` is filled. Leaving
      with unsaved edits asks for confirmation.
- [ ] Save & launch on a run whose `.pt` exists asks to overwrite.
- [ ] The first `[aug]` line appears after 20 iterations. Metrics, curves, the GPU line and the
      log tail update, and completion shows "—" for an env without a metric.
- [ ] Stop ends the run. **On Windows, confirm the grandchild `python` dies too:**
      `CTRL_BREAK` goes to micromamba's process group. The run then shows `stopped` with the
      partial-checkpoint note.
- [ ] Audits: verify and diagnose run, and their results persist across an app restart. A
      dishonest, unverified or failed result shows the red "Not a success" banner with
      completion struck through. Controls are disabled with the reason while a run is live.
- [ ] Record episodes writes `training/<name>/recordings/`, and the embedded viewer plays them.
- [ ] Continue from this checkpoint pre-fills the generic recipe from a finished run and from
      `training/11-count-curriculum`.
