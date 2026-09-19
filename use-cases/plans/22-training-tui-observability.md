---
plan_for: use-cases/22-training-tui-observability.md
work_branch: feat/uc-22-training-tui-observability
team: drone-fly-uc-22
approved: 2026-09-19
---

# UC-22 — Full-screen live training TUI (APPROVED plan)

Approved by challenger after one revision round. `TARGET_DIR=/workspace/drone-fly-uc-22-training-tui-observability`; brief frontmatter → `paths.production=["src/drone_fly/**"]`, `paths.test=["tests/**"]`, `profiles: []`, build=uv, tests `uv run pytest`.

## Analysis (grounded in the worktree)
- `src/drone_fly/env/racing_env.py::build_vec_env` (def ~:550): `DummyVecEnv → (seed) → VecNormalize` (fresh or `VecNormalize.load` on resume). **No Monitor** → `ep_info_buffer`/`ep_success_buffer` never populated → `rollout/ep_rew_mean|ep_len_mean|success_rate` never collected. Kwargs-only signature: `config, adapter, n_envs, seed, training, norm_reward, vecnormalize_path`.
- `src/drone_fly/train/loop.py::train()` (def ~:229): builds venv (`training=True`), `_make_logger(cfg.logs_dir)` = `configure(logs_dir, ["stdout","csv","tensorboard"])` (~:66), callbacks `[checkpoint_cb, (record_cb?), HealthCallback(thresholds)]`, then `model.learn(total_timesteps=steps, callback=..., progress_bar=False)`. `model.set_logger(...)` runs on both fresh and resume paths. Scheduled iterations = `steps // (cfg.n_steps * resolved_n_envs)` (wireframe "488").
- **UC-23 is live**: `train/health.py` (`assess_training_health`, `HealthVerdict{status,message,reasons}`) + `train/health_callback.py::HealthCallback` already exposes injected `on_verdict: Callable[[HealthVerdict],None]` and `latest_verdict`, and is already appended in loop.py. TUI consumes this seam — **no stub, no second assessment**.
- **SB3 2.9.0 callback timing (verified in `on_policy_algorithm.py`):** `collect_rollouts` fires `on_rollout_end` (:266) and updates `ep_info_buffer`/`ep_success_buffer` (:227) — both BEFORE `dump_logs` (:335, records `rollout/*` then `logger.dump()` clears `name_to_value`) and BEFORE `self.train()` (:337, records `train/*`). ⇒ at `on_rollout_end`, `rollout/*` is NOT in `name_to_value`; the buffers ARE fresh.
- CLI: `main → _run_train(args.config)` (`cli/__init__.py:294`, `_run_train` :338) → `train(...)`. `train` subparser at ~:203 takes only `--config`.
- Rich absent from `pyproject.toml` (only vendored under pip). SB3 exposes `VecMonitor` at `stable_baselines3.common.vec_env`. Rich 15.0.0 available in-env.

## Proposed Solution — strict data⟂render split, new subpackage `src/drone_fly/train/tui/`

**1. Data layer (pure, headless-tested) — `metrics.py`**
- `MetricDirection` enum: `HIGHER_BETTER` (success_rate, ep_rew_mean, explained_var), `LOWER_BETTER` (value_loss; entropy & std where shrinking is expected), `BAND` (approx_kl — flat/near-target = healthy).
- `TRACKED_METRICS`: ordered registry of the **7 rows** exactly per wireframe: `success_rate, ep_rew_mean, entropy, std, value_loss, approx_kl, explained_var` — each with label, source key, direction, optional concern predicate.
- `MetricHistory`: `deque(maxlen≈100)` rolling window per metric.
- Pure fns: `classify_trend(values, direction)→"improving"|"worsening"|"flat"` (direction-aware; band handled separately), `sparkline(values, width)→str` (rebucket window to fixed width via `▁▂▃▄▅▆▇█`, min–max normalized, flat→mid-block), `progress_fraction(current, scheduled)`, `eta(elapsed, fraction)`, `tendency_icon→▲/▼/►`, per-row `concern` flag.
- `DashboardModel`: grouped raw values (TIME iters/elapsed/eta; TRAIN ent_loss/value_loss/expl_var; ROLLOUT ep_rew/ep_len/success), the 7 `TrendRow`s, iteration progress (current/scheduled/pct), `latest_verdict: HealthVerdict|None`. `update(...)` ingests one rollout snapshot; `set_verdict(v)` stores the verdict. **entropy = −`train/entropy_loss`** (continuous case: `entropy_loss = −mean(entropy)`), **std = `train/std`** (UC-23 commitment proxy).
- **Timing contract (None/nan-tolerant by design):** any ROLLOUT value that is `nan` or any TRAIN value that is `None`/missing is NOT appended to history and renders as a placeholder (`—`), never `0`, never a crash — covers first-rollout warm-up and an absent `train/std`. Cross-source **off-by-one** (fresh ROLLOUT vs lagged TRAIN) is cosmetic, matches HealthCallback, documented in-code.

**2. Log capture (fd-level) — `capture.py`**
- `FdLogCapture` context manager: `os.pipe()`, save `os.dup(1)/os.dup(2)`, `os.dup2(write_fd, 1/2)` to redirect **C-level** stdout+stderr (catches pybullet native prints); daemon reader thread drains the read end (non-blocking / large buffer, no pipe-full deadlock) into a bounded `deque(maxlen=N)` scrollback. `__exit__` restores fds via saved originals, flushes, joins — on normal exit, exceptions, AND `KeyboardInterrupt` (terminal left usable, no lost final output).

**3. Render layer (Rich; untested boundary) — `render.py`**
- `build_layout(model, log_lines)→RenderableType`: Rich `Layout` for the 4 regions — left ~70% split: top grouped TIME/TRAIN/ROLLOUT value panel, middle 7 trend rows (`label value sparkline icon [⚠]`) + `iterations` progress bar (current/scheduled/pct); right ~30% raw-log `Panel` (scrollback tail); full-width bottom status `Panel` rendering the verdict with severity styling (normal/green, warning/yellow, critical/red). Pure `model→renderable` (constructible headlessly); only the live draw loop is untested.

**4. Orchestration — `dashboard.py`**
- `TrainingDashboard` owns the `DashboardModel`, an `FdLogCapture`, a Rich `Live`. `live_session()` ctx-mgr: enter fd-capture + `Live(screen=True)`, yield, guarantee teardown (Live stop → fd restore) even on Ctrl-C. `redraw()` rebuilds `build_layout` from model+scrollback. `set_verdict` proxies to the model. **Graceful degradation (AC8):** too-small terminal / `NO_COLOR` / dumb term → shrink/omit panes or fall back to line logging, never crash.

**5. SB3 bridge — `callback.py`**
- `TuiCallback(BaseCallback)`, on `_on_rollout_end`:
  - **ROLLOUT from the model buffers (fresh, VecMonitor-dependent), NOT the logger:** `ep_rew = safe_mean([e["r"] for e in self.model.ep_info_buffer])`, `ep_len = safe_mean([e["l"] for e in self.model.ep_info_buffer])`, `success = safe_mean(self.model.ep_success_buffer)`. `safe_mean` (import from `stable_baselines3.common.utils`) of an empty buffer → `nan` → data layer treats as "not observed".
  - **TRAIN + 5 train-derived trend rows from `self.model.logger.name_to_value.get(key)` (lagged-by-one, `None`-tolerant):** `train/entropy_loss`(→entropy), `train/std`, `train/value_loss`, `train/approx_kl`, `train/explained_variance`.
  - Then `dashboard.model.update(...)` (fresh ROLLOUT + possibly-`None` TRAIN + `n_updates` callback-owned counter + elapsed) → `dashboard.redraw()`.
  - Fully defensive: any error disables the callback and logs, never crashes training.

**6. Enabling data-layer change — `build_vec_env`**
- Insert `VecMonitor(venv, info_keywords=("is_success",))` right after `DummyVecEnv` (post-seed), **before** both the `.load` and fresh `VecNormalize(...)` branches ⇒ Monitor wraps raw episodes, normalization on top (raw ep stats — AC1 & wrapper-order risk). **Gate on `training=True`** so eval/non-training is byte-identical (AC1). Seed stays on the raw env (no RNG change).

**7. Logger + wiring — `loop.py`**
- `_make_logger(logs_dir, include_stdout=True)`: TUI active → `["csv","tensorboard"]` only (drop SB3 `stdout` HumanOutputFormat so Rich/SB3 don't fight the screen; CSV/TensorBoard preserved, AC7); TUI off → unchanged. Gate covers both fresh and resume `set_logger` paths.
- `train(..., tui: bool | None = None)`: `tui_enabled = (tui is not False) and sys.stdout.isatty()`. Enabled → no-stdout logger; construct `HealthCallback(thresholds=..., on_verdict=dashboard.set_verdict)` (reuse the existing one — inject the seam); append `TuiCallback(dashboard)` **after** HealthCallback (verdict fresh before redraw); run `model.learn(...)` **inside** `dashboard.live_session()`. Disabled → current path exactly. `live_session()` wraps only `model.learn`, so the capacity-guard interactive prompt runs before fd-capture engages (no conflict).

**8. CLI — `cli/__init__.py`**
- `train_p.add_argument("--no-tui", action="store_true", ...)`; thread `main → _run_train(args.config, no_tui=args.no_tui) → train(..., tui=not no_tui)`.

**9. Dependency — `pyproject.toml`**
- Add pinned Rich to **runtime** `dependencies` (default-on). Recommend `rich==13.9.4`; dev may bump to whatever `uv` locks (15.0.0 available in-env) provided it's an exact pin.

## Files Affected

**Production code → developer**
- `pyproject.toml` — pinned `rich` in runtime deps.
- `src/drone_fly/env/racing_env.py` — training-only `VecMonitor(info_keywords=("is_success",))` inside `VecNormalize` in `build_vec_env`.
- `src/drone_fly/train/loop.py` — `_make_logger(include_stdout=…)`; `tui` param + TUI gate/wiring around `model.learn`; inject `on_verdict` into existing `HealthCallback`.
- `src/drone_fly/cli/__init__.py` — `--no-tui` on `train` subparser; thread through `_run_train`→`train`.
- `src/drone_fly/train/tui/__init__.py` (new) — exports.
- `src/drone_fly/train/tui/metrics.py` (new) — data layer (pure).
- `src/drone_fly/train/tui/capture.py` (new) — fd log capture.
- `src/drone_fly/train/tui/render.py` (new) — Rich render.
- `src/drone_fly/train/tui/dashboard.py` (new) — orchestration/Live.
- `src/drone_fly/train/tui/callback.py` (new) — `TuiCallback`.

**Test code → qa**
- `tests/test_tui_metrics.py` (new) — trend classification across all directions (higher/lower-better + entropy/std shrinking) and approx_kl **band**; sparkline fixed width & bucketing incl. flat/degenerate series; rolling window ~100 truncation; ETA & progress math. Plus the **first-rollout snapshot** case (ROLLOUT present/empty→nan with all `train/*`=`None`) → `update()` succeeds, TRAIN + 5 train-derived rows show placeholders (no crash, no spurious `0`/history), success_rate/ep_rew_mean populate from buffers. Guards timing issues 1–3 together. Pure, no terminal.
- `tests/test_tui_capture.py` (new) — write to real fd 1/2 → captured into scrollback; teardown restores fds; bounded scrollback cap; large-volume write no deadlock. Headless-safe.
- `tests/test_tui_render.py` (new) — `build_layout(model)` returns a Rich renderable without error for a representative model (headless construction only; **no pty, no Live**).
- `tests/test_cli.py` / new `tests/test_tui_fallback.py` — `--no-tui` parses & propagates; non-TTY run auto-falls back to the stdout logger; TUI not engaged when stdout isn't a TTY.
- Extend `build_vec_env` tests — `VecMonitor` present when `training=True`, **absent** when `training=False`; wrapper order (Monitor inside Normalize) yields raw (un-normalized) episode stats. Confirm resume path (`test_train_resume.py`) stays green.

## Risks & Considerations
- **Wrapper order (AC1):** VecMonitor MUST sit inside VecNormalize; insert after DummyVecEnv+seed, before both VecNormalize branches.
- **Resume compat:** old `vecnormalize.pkl` saved without VecMonitor; `VecNormalize.load` only restores stat arrays + rebinds `.venv`, doesn't validate inner structure ⇒ added inner VecMonitor is safe. QA covers resume stays green.
- **Callback ordering:** TuiCallback after HealthCallback; also inject `on_verdict` (order-independent for the value) — belt and suspenders.
- **fd capture only when TUI on:** dup2 engaged only inside `live_session()`; TUI-off leaves SB3 stdout untouched.
- **Ctrl-C/exception teardown (AC8):** fds restored in `finally`; no swallowed final output, no garbled terminal.
- **Untested boundary:** live Rich draw not exercised in CI (no TTY), like viewer/pybullet; coverage = data + capture + render-construction + fallback. No pty test expecting pixels.
- **HealthCallback intact:** keeps self-accounting (harmless duplication); not simplified, to protect merged UC-23 tests.
- **Rich pin:** exact, resolvable in `uv.lock`; dev confirms via `uv add`.
- **Scope guard (AC7):** no change to env dynamics/obs/reward/connectome/checkpoints beyond the additive VecMonitor wrap; no retrain.

## Challenger verdict: APPROVED (round 2)
One Major fixed round 1→2: read `model.ep_info_buffer`/`ep_success_buffer` directly at `_on_rollout_end` (not `rollout/*` from name_to_value, which is recorded after the callback and cleared by the prior dump). Two Minors folded in: `train/*` one-iteration-lagged & absent on the first rollout; `train/std` continuous-only → explicit None/nan→placeholder tolerance contract with a first-rollout unit test. All 7 enforcement points satisfied (hermetic CI, non-TTY byte-identity, no env/obs/checkpoint change, consumes UC-23 seam, fd restore on Ctrl-C, pinned Rich runtime dep, no scope creep).
