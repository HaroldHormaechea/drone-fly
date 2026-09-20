---
plan_for: use-cases/32-windows-tui-parallel-training.md
work_branch: feat/uc-32-windows-tui-parallel-training
team: drone-fly-uc-32
approved: 2026-09-20
---

# Approved implementation plan — UC-32 Windows full-screen live TUI + crash-free parallel training

Challenger APPROVED (revision 2). No code snippets — patterns + paths only. All paths absolute inside `/workspace/drone-fly`.

## Analysis
Three Windows-only defects, all under the TUI/worker-output path (all vanish with `--no-tui`), verified against the code:
1. **Worker crash at n_envs>1** — `train/loop.py:413` passes `suppress_worker_output=tui_enabled` into `build_vec_env`; `env/racing_env.py` `_subproc_factory` (lines 668–676) does `os.dup2(devnull_fd,1/2)` inside each spawned worker. Under Windows `spawn` this kills the worker; parent gets opaque `EOFError`/`WinError 109` at first communication (SB3 `SubprocVecEnv.__init__` does `remotes[0].recv()`), real traceback swallowed into devnull.
2. **`WinError 1` log spam** — root `StreamHandler` from `logging.basicConfig` (`cli/__init__.py:289`) keeps writing to the console the TUI commandeered; Windows emit throws `OSError [WinError 1]`. macOS/Linux route the same Python logs through `FdLogCapture`'s fd-1/2 dup2 into the pipe→pane, so this is Windows-specific.
3. **Freeze / not full-screen / logs above panels** — redraw driven only by `TuiCallback._on_step`/`_on_rollout_end` (`tui/callback.py`) so the clock freezes during `PPO.train()`; a WinError during redraw trips `except → self._enabled=False` / `dashboard.redraw`'s `self._live=None`, permanently killing liveness; `dashboard.py:_start` builds `Console(file=display_stdout)` on a plain fdopen stream so Rich's `screen=True` alt-buffer doesn't engage full-screen, worst on legacy conhost.

Overarching constraint applied everywhere: **AC-9 (macOS/Linux byte-for-behavior unchanged) and AC-10 (`--no-tui` success path byte-identical)** → every new branch is platform-guarded to `sys.platform=="win32"` or is error-path-only. `cfg.logs_dir` = `training/<name>/logs` (`cli/__init__.py:369`) is the run logs root; per-worker files under `<logs_dir>/workers/`.

## Proposed Solution
**A. Crash-free parallel envs + fail-loud (AC-1/2/3)** — `env/racing_env.py` + new `src/drone_fly/env/worker_output.py`.
- Replace the single shared `_subproc_factory` with **per-index factories**, each given a distinct per-worker log path. `build_vec_env` gains `worker_log_dir: str | None`; `loop.py` passes `<cfg.logs_dir>/workers` when `tui_enabled`, else None (today's behavior). Keep `suppress_worker_output` as the gate.
- `worker_output.py`: `worker_log_path(dir, idx)`; `redirect_worker_fds(path)` — Windows-safe: `os.open(path, O_WRONLY|O_CREAT|O_TRUNC)`, `os.dup2` onto fd 1 & 2, rebind `sys.stdout/err`, defensive but does NOT silently swallow a redirect failure; `read_worker_errors(paths, max_lines)`; `WorkerStartupError(RuntimeError)`.
- **Prune/clear the `workers/` dir at build** (not just per-file O_TRUNC) so stale files from a higher previous `n_envs` don't linger (AC-2).
- **Fail-loud (AC-3):** wrap the `SubprocVecEnv(...)` construction **and first `reset`** in `build_vec_env` to catch `EOFError`/`BrokenPipeError`/`ConnectionError` → re-raise `WorkerStartupError` enriched with `read_worker_errors(...)`. TUI on → reads per-worker files; `--no-tui` (no redirect) → workers still write tracebacks to console (success path byte-identical) and the wrapper raises a clear message instead of opaque `EOFError`. A minimal `SubprocVecEnv` subclass overriding `step_wait` for mid-run death is optional (not a blocker).

**B. No WinError-1 + Python logs in the pane (AC-4/7)** — new `src/drone_fly/train/tui/logbridge.py`, Windows-guarded.
- `DashboardLogHandler(logging.Handler)` → formats records into a bounded deque the dashboard exposes.
- On live-session enter (Windows only): attach the handler to the root logger and **sweep + remove/disable every handler on the root logger (and ancestors) whose stream is `sys.stdout`/`sys.stderr`**; restore exactly on teardown. No console writes during TUI → no WinError 1 (AC-4); Python records land in the pane (AC-7).
- `dashboard._log_lines()` source select: Windows → `DashboardLogHandler` buffer; macOS/Linux → `FdLogCapture.lines()` (unchanged, AC-9).
- **pybullet native banner → file (AC-7), PRIMARY mechanism = direct os-level redirect** (not a pipe tee): on Windows at session enter, save the real terminal fd via `os.dup(1)` (and a saved dup for fd 2) for Rich's display, then open `<logs_dir>/native.log` and `os.dup2` it onto fd 1 & 2 directly, before pybullet connect. Rich still draws to the saved terminal fd. macOS/Linux keep `FdLogCapture` pipe→scrollback→pane unchanged.

**C. ~1 Hz refresh independent of PPO boundaries (AC-5)** — `tui/dashboard.py` + `tui/metrics.py`, **Windows-guarded**.
- Background daemon timer (injectable now/sleep/stop-Event), engages only on `win32`, every ~1 s advances elapsed + `redraw` so clock/liveness tick even during `PPO.train()`. Pure `_refresh_tick(now)`-style method for hermetic testing; the thread wrapper is the documented untested boundary.
- `DashboardModel` gains a clock-only updater (`tick_clock(elapsed_seconds)`) updating elapsed+ETA without disturbing rollout counters.
- **Single `threading.Lock` in `TrainingDashboard`** guards every model-mutation+redraw section (callback `tick`/`update`, `set_verdict`, timer clock-tick) → no cross-thread `build_layout` race. Uncontended (no-op) on macOS/Linux.

**D. Full-screen alt-buffer + capability fail-fast (AC-6/8)** — `tui/dashboard.py` + new `src/drone_fly/train/tui/capability.py`, **Windows-guarded**.
- Rich `Console` for `Live` built with `force_terminal=True` + terminal size from `os.get_terminal_size` so `screen=True` engages the alt-buffer full-screen even drawing to the display fd (AC-6).
- `assert_tui_capable()` — **primary gate = Rich's reliable signal** (`Console.legacy_windows` / VT-enable probe); hard-error ONLY when that says incapable. `WT_SESSION`/`TERM_PROGRAM` are corroboration only, can never *cause* a hard-error (no false-negatives). Actionable message ("use Windows Terminal or pass --no-tui"). **Runs BEFORE any `Console`/`Live` enter** so `force_terminal` never engages an alt-buffer on incapable conhost (AC-8 ordering).

**E. Bounded redraw failure** — `tui/dashboard.py`: replace one-strike-`_live=None` with an **N=3 consecutive-failure counter** (reset on success) + **~30 s rate-limited warning**; only after N failures disable live updates. Transient blips self-heal; persistent errors degrade quietly without re-flooding AC-4.

**F. Wiring** — `train/loop.py`: pass `worker_log_dir` when `tui_enabled`; call `assert_tui_capable()` then install/uninstall logbridge + start/stop timer inside `dashboard.live_session()`; all under the existing `tui_enabled = (tui is not False) and sys.stdout.isatty()` gate (loop.py:333).

## Files Affected
**Production code (developer):**
- `src/drone_fly/env/racing_env.py` — per-index factories, `worker_log_dir` param, fail-loud construction+reset wrap, optional `step_wait` subclass, `workers/` prune.
- `src/drone_fly/env/worker_output.py` *(new)* — `redirect_worker_fds`, `worker_log_path`, `read_worker_errors`, `WorkerStartupError`.
- `src/drone_fly/train/loop.py` — wiring (F), platform guards.
- `src/drone_fly/train/tui/dashboard.py` — Windows refresh timer, shared lock, logbridge install, capability call, `Console(force_terminal)`, native-fd redirect (save fd 1 AND fd 2), log-source select, bounded-failure counter.
- `src/drone_fly/train/tui/logbridge.py` *(new)* — `DashboardLogHandler` + full handler sweep/restore.
- `src/drone_fly/train/tui/capability.py` *(new)* — `assert_tui_capable`.
- `src/drone_fly/train/tui/metrics.py` — `tick_clock` clock-only updater.
- `src/drone_fly/train/tui/capture.py` — Windows role = direct file redirect (save both fds), not a pipe tee; macOS/Linux unchanged.
- `src/drone_fly/train/tui/render.py` — minor if any (pane renders passed lines).

**Test code (qa)** — all hermetic, `uv run --extra dev pytest`, monkeypatch `sys.platform`/`isatty`/suppression/clock; no real Windows/GPU/terminal (AC-11):
- `tests/test_worker_isolation.py` — update `_GateFactory`/assertions from devnull → per-worker logfile; no-leak-to-parent-stdout still holds.
- `tests/test_worker_output.py` *(new)* — `redirect_worker_fds` writes to file; `read_worker_errors` tails; `WorkerStartupError` raised & enriched on simulated `EOFError` (mock `SubprocVecEnv`), TUI and `--no-tui` framings.
- `tests/test_tui_logbridge.py` *(new)* — handler routes records to buffer; every stdout/stderr handler removed then restored; monkeypatch `sys.platform`.
- `tests/test_tui_capability.py` *(new)* — capable (WT/VS Code AND bare-but-capable) vs incapable (legacy) → actionable raise only on incapable; no false-negative.
- `tests/test_tui_refresh_clock.py` *(new)* — injected clock advances `elapsed_seconds` ~1 Hz with zero `_on_step`; asserts timer is Windows-guarded (no thread/tick on non-win32).
- `tests/test_tui_callback.py` — adjust for heartbeat/lock; assert lock serializes.
- dashboard tests — bounded-failure counter (N=3) + rate-limited warning.
- `tests/test_build_vec_env.py` / `tests/test_vec_parallel.py` — `worker_log_dir` param + files created.
- `tests/test_cli.py` — `--no-tui` path unchanged.

## Risks & Considerations
- **[Primary residual, needs real-Windows check by user]** The exact Windows mechanism by which devnull-dup2 kills the worker (defect 1) and whether Windows fd `dup2`-to-file reliably captures native C-level writes (AC-7 banner) cannot be reproduced in hermetic CI. Design uses direct dup2-to-a-real-file (most robust option) + fail-loud so any failure surfaces the real error instead of an opaque EOF. Flagged honestly rather than hidden behind the capture layer.
- AC-9/AC-10 protected by strict Windows-guarding of B/C/D and error-path-only fail-loud; `--no-tui` success path untouched.
- AC-8 false-negative avoided by keying the hard-error on Rich's reliable signal, env vars corroboration-only.

## Challenger's non-blocking implementation note (for the developer)
On the Windows native-redirect teardown, ensure **both fd 1 and fd 2 are restored** to the original terminal — save a dup for each (or restore both from the one saved terminal fd). `FdLogCapture._restore` in `tui/capture.py` already saves/restores both as the reference pattern.

## Challenger verdict
**Approve** (revision 2). Round 1 flagged 3 Majors — (1) cross-platform 1 Hz timer would regress macOS AC-9 + cross-thread redraw race; (2) AC-7 native-banner-to-file relied on the same Windows fd-capture the plan distrusts; (3) relaxing one-strike redraw would re-flood logs — all resolved in rev 2 (win32-guarded timer + single lock; direct dup2 of native.log; N=3 consecutive-failure counter + rate-limited warning). Residual Windows-fd risk cannot be reproduced in hermetic CI → needs a real-Windows smoke check by the user before UC-32 is considered fully verified.
