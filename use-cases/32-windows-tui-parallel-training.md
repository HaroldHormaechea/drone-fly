# Use Case 32: Windows — full-screen live TUI with per-second updates, logs pane, and crash-free parallel training

## Summary
The live training dashboard (TUI) and its rollout path are not Windows-safe, and its refresh/layout are unsatisfactory even when it runs. On Windows (PowerShell, uv CPython 3.12, RTX 3050) three defects occur that never appear on macOS/Linux, all traceable to the TUI/worker-output path (all vanish under `--no-tui`): (1) `SubprocVecEnv` workers die at `n_envs>1` because the TUI enables `suppress_worker_output`, whose per-worker `os.dup2(devnull)` kills spawned Windows workers, surfacing only as an opaque `BrokenPipeError [WinError 109]`/`EOFError` at the first `env.reset()` with the real worker error swallowed; (2) the root logging handler writes to a console the TUI has commandeered, throwing repeated `OSError [WinError 1]` "logging error" blocks; (3) the dashboard redraws only at PPO iteration boundaries, so with ~100 s single-env rollouts it looks frozen, isn't full-screen, and prints INFO logs to the scrollback above the panels instead of into its own logs pane. The fix makes the TUI **work on Windows** (not disable it): crash-free parallel envs, no logging crashes, and a full-screen dashboard that ticks every second with Python logs rendered in a logs pane. macOS/Linux behavior is preserved exactly and `--no-tui` stays byte-identical. This use case **supersedes/extends UC-30 (tui-intra-rollout-heartbeat)** as the authoritative TUI-liveness + Windows-safety work.

## Acceptance Criteria
1. On Windows, training with `n_envs>1` and the TUI **on** completes rollouts without the worker crash (no `WinError 109`/`EOFError` at `env.reset()`); throughput scales with `n_envs`.
2. Each spawned worker's native stdout/stderr is redirected to a **per-worker log file** (Windows-safe) instead of devnull; the files are inspectable after a run and live under the run's logs location (not leaked across runs).
3. When a rollout worker genuinely fails, the **real worker exception** is surfaced to the user (fail loud) regardless of TUI/`--no-tui` — never collapsed into an opaque `EOFError`.
4. With the TUI active on Windows, **no `OSError [WinError 1]`** "logging error" blocks appear; log output is not corrupted by the dashboard.
5. The TUI's **elapsed timer advances every second**; the UI refreshes at ~1 Hz independent of PPO iteration boundaries (per-iteration metrics may stay stale between iterations, but the clock and liveness must tick).
6. The TUI renders **full-screen** (alternate-screen buffer), filling the terminal on Windows as it does on macOS/Linux.
7. **Python logging** lines render **inside the logs pane**, not in the scrollback above the panels; pybullet's native startup banner is **redirected to a file** so it no longer pollutes the screen.
8. On a Windows terminal that cannot support the alternate-screen buffer / 1 Hz redraw (e.g. legacy `conhost.exe`), the TUI **fails fast with an actionable error** (use Windows Terminal, or pass `--no-tui`) rather than degrading or crashing.
9. **macOS/Linux TUI behavior is unchanged** (no visual/behavioral regression) — platform-guarded differences only.
10. `--no-tui` remains the byte-identical plain-SB3-logger path.
11. New tests are **hermetic and platform-mockable** (monkeypatch `sys.platform` / `stdout.isatty` / the worker-output-suppression decision / the refresh clock) — no real Windows, GPU, or terminal — and pass under `uv run --extra dev pytest`.

## Potential Pitfalls & Open Questions
- **Risk** — Terminal-capability detection for AC-8 must be reliable enough not to false-positive on modern terminals (Windows Terminal, VS Code integrated terminal) and wrongly hard-error a capable setup.
- **Risk** — Per-worker log files (AC-2) accumulate on disk; they need a sensible location (e.g. under the run's `logs/`) and must not leak or collide across runs.
- **Edge case** — The 1 Hz refresh (AC-5) must not fight SB3's own stdout writes or re-introduce the `WinError 1` on the log handler.
- **Assumption** — UC-30's heartbeat code already exists (UC-30 is `done`) but does not hold on Windows; the dev-team may revisit/replace it as part of this work.

## Original Description
From a live Windows debugging session (2026-09-20). The user, training on Windows (PowerShell, uv-managed CPython 3.12, RTX 3050, 8 GB) after the UC-31 CUDA setup, hit a series of TUI/rollout failures and asked for a fix with these explicit UI requirements: "make it so the elapsed time keeps advancing every second (so update UI every second), make it full screen, and that the logs show in the logs box and not on the top as now."

The live training TUI / rollout path is not Windows-safe. Three evidence-backed symptoms, all reproduced on real Windows; none reproduce on macOS/Linux; `--no-tui` is the current workaround for all three (which pins the cause to the TUI/worker-output path):

1. SubprocVecEnv workers die at n_envs>1 on Windows. With the TUI on (default), `build_vec_env(..., suppress_worker_output=tui_enabled)` (src/drone_fly/train/loop.py:413) makes each spawned worker `os.dup2(devnull_fd, 1)`/`dup2(...,2)` at startup (src/drone_fly/env/racing_env.py `_subproc_factory`). On Windows spawn this kills the worker; the parent then gets an opaque `BrokenPipeError [WinError 109]` / `EOFError` at the first `env.reset()`. The worker's REAL exception is swallowed by the devnull redirect. Confirmed at n_envs=2 and n_envs=8 on a tiny model (k1, ~4M params, 25,627 neurons), before any GPU work — so it is NOT OOM. With `--no-tui` (suppress_worker_output=False), n_envs>1 works.

2. OSError [WinError 1] log spam. The root log handler (logging.basicConfig, src/drone_fly/cli/__init__.py:289) writes to a console the TUI has commandeered, so logging's emit throws WinError 1 ("Función incorrecta") on stream.write, printing repeated "--- Logging error ---" blocks (e.g. from src/drone_fly/train/health_callback.py). Non-fatal but floods output. Gone with `--no-tui`.

3. Perceived "freeze". The TUI only redraws at PPO iteration boundaries; at n_envs=1 a 2048-step PyBullet rollout takes ~100s (observed iter1 @ elapsed 11s → iter2 @ 1m51s), so the gap reads as a hang. Not an actual hang, but poor UX with no per-second liveness, not full-screen, and INFO logs printed above the panels rather than in the logs pane.

Constraints: preserve current macOS/Linux TUI behavior exactly (no regression), prefer platform-guarded changes; keep `--no-tui` byte-identical; production code under src/drone_fly/**, tests under tests/** and hermetic/platform-mockable (no real Windows/GPU/terminal in CI); `uv run pytest` fails to collect (pytest is in the `dev` extra) — use `uv run --extra dev pytest`. Related prior work: UC-22 (live TUI), UC-26 (SubprocVecEnv parallel rollout backend), UC-30 (TUI intra-rollout heartbeat), UC-31 (Windows CUDA setup script — separate, fixed in PR #35).

## Clarifications
- Q: What goes in the logs pane (pybullet's native C-level stdout needs fd redirection; Python logging is easy to reroute)?
  A: Python logs only; route Python logging into the logs pane and redirect pybullet's native startup banner to a file so it stops polluting the screen.
- Q: If a Windows terminal can't support full-screen/alt-buffer or a 1 Hz redraw (e.g. legacy conhost.exe), what should happen?
  A: Fail fast with an actionable error (tell the user to use Windows Terminal or pass `--no-tui`) rather than degrading or crashing.
- Q: To keep BOTH parallel envs (n_envs>1) AND the TUI on Windows, where should each worker's native stdout/stderr go instead of the devnull dup2 that currently crashes workers?
  A: Redirect each worker's native output to a per-worker log file (Windows-safe, inspectable on failure).
- Q: How should UC-32 relate to the existing UC-30 (tui-intra-rollout-heartbeat)?
  A: UC-32 supersedes/extends UC-30 as the authoritative TUI-liveness + Windows-safety use case; the dev-team may revisit UC-30's heartbeat code.
