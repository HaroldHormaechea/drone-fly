# Use Case 22: Full-screen live training TUI (observability dashboard)

## Summary
Replace the training run's stock Stable-Baselines3 stdout table with a full-screen, live-updating terminal dashboard (tmux/htop/k9s style) built on **Rich** (`Live` + `Layout`), redrawn from a SB3 callback on each PPO update. The layout is a left column (~70%) over a full-width bottom status bar, with a right column (~30%) of raw logs: **top-left** shows current raw values grouped `TIME` / `TRAIN` / `ROLLOUT`; **middle-left** shows one trend row per tracked metric with a sparkline and an improving/worsening/flat tendency icon (success_rate first), plus an iterations progress bar (current vs scheduled); **right** tails raw stdout/stderr (the pybullet C-level spew) captured via `os.dup2` fd redirection into a pipe; **bottom** is a full-width status bar rendering a health verdict. The dashboard is **default-on when stdout is an interactive TTY** and disabled with `--no-tui`; on non-TTY/piped/CI runs it falls back to today's SB3 line/table logger so hermetic CI is unaffected. A required enabling change wraps the training vec env in **`VecMonitor(info_keywords=("is_success",))`** — today `build_vec_env` (`src/drone_fly/env/racing_env.py:445`) goes `RaceEnv → DummyVecEnv → VecNormalize` with no Monitor, so SB3's `ep_info_buffer` is never populated and `rollout/ep_rew_mean`, `rollout/ep_len_mean`, and `rollout/success_rate` are never collected (this is why `ep_rew_mean` is absent from the CSV); the dashboard needs that data. The **data layer** (metrics capture, per-metric rolling history + trend computation, sparkline bucketing, ETA/progress math) is separated from the **render layer** (Rich widgets) so the data half is fully unit-testable headlessly; only the live render is an untested boundary. The status-bar verdict comes from UC-23's `assess_training_health()` engine (injected dependency); a trivial "Training progressing normally" stub is acceptable if UC-23 isn't yet implemented. Strictly observability/output — no change to env dynamics, observation schema, reward, connectome, or checkpoints; existing CSV/TensorBoard outputs stay intact.

## Layout contract (wireframe)

The render must reproduce these four regions and their placement:

```
┌─ drone-fly · training ───────────────────────────────┬─ logs ──────────────┐
│ TIME              TRAIN             ROLLOUT           │ pybullet build...   │
│  iters  81/488    ent_loss  -2.94   ep_rew   -1204    │ startThreads...     │
│  elapsed 2h15m    value_loss 51.2   ep_len    210     │ Version = 3.2.5     │
│  eta    1h04m     expl_var   0.63   success    0%     │ (raw stdout/stderr  │
├───────────────────────────────────────────────────────┤  from sim + SB3,    │
│ TRENDS                                                │  scrolling, ~30% w) │
│  success_rate    0%     ▁▁▁▁▁▂▁▁    ▼ worsening   ⚠   │                     │
│  ep_rew_mean    -1204   ▁▁▂▂▃▃▄▄    ▲ improving       │                     │
│  entropy         2.94   ────────    ► flat        ⚠   │                     │
│  std             0.502  ▇▇▆▆▆▅▅▅    ▼ shrinking (slow)│                     │
│  value_loss      51.2   ▇▆▆▅▅▄▄▃    ▲ improving       │                     │
│  approx_kl       0.007  ▂▃▂▃▂▃▂▃    ► flat            │                     │
│  explained_var   0.63   ▁▂▃▄▅▆▆▇    ▲ improving       │                     │
│  iterations   [██████░░░░░░░░] 81/488  (17%)         │                     │
├───────────────────────────────────────────────────────┴─────────────────────┤
│ ⚠ WARNING: actor entropy flat & 0% success — connectome slice likely too small│
└───────────────────────────────────────────────────────────────────────────────┘
```

**Trend rows (7), in this order:** `success_rate`, `ep_rew_mean`, `entropy` (from `entropy_loss`), `std`, `value_loss`, `approx_kl`, `explained_var` — followed by the `iterations` progress bar (current vs scheduled). Each trend row = value + sparkline + tendency icon (▲ improving / ▼ worsening / ► flat), with an optional ⚠ marker when the metric is in a concerning state.

## Acceptance Criteria
1. **VecMonitor data layer:** the training vec-env build wraps the env in `VecMonitor` with `info_keywords=("is_success",)`, so `rollout/ep_rew_mean`, `rollout/ep_len_mean`, and `rollout/success_rate` are populated and appear in the existing CSV/TensorBoard outputs. Eval/non-training construction is unchanged in behavior. `VecMonitor` wraps the raw episodes (before `VecNormalize`) so reported episode reward/length are un-normalized.
2. **TTY-gated default-on + `--no-tui`:** with an interactive TTY the TUI renders by default; `--no-tui` disables it; a non-TTY/piped/CI run auto-falls back to the current SB3 logger. The fallback path reproduces the current behavior (CI unaffected).
3. **Layout matches the wireframe contract:** the render produces the four regions — grouped TIME/TRAIN/ROLLOUT value panel (top-left), the 7 per-metric trend rows with sparkline + tendency icon (success_rate first), the iterations progress bar (current vs scheduled), a right-hand raw-log pane (~30% width), and a full-width bottom status bar.
4. **Trend/tendency data layer (unit-tested, headless):** given a sequence of per-update metric values, the data layer computes for each tracked metric a tendency classification (improving / worsening / flat) and a fixed-width sparkline over a **rolling window of the last ~100 updates**, plus ETA (from elapsed + progress fraction) and progress fraction — all pure functions tested without a terminal. Tendency direction correctly accounts for "lower-is-better" metrics (`value_loss`, and `entropy`/`std` where shrinking is expected) vs "higher-is-better" (`success_rate`, `ep_rew_mean`, `explained_var`); `approx_kl` is treated as a band (flat/near-target is healthy, not "improving").
5. **C-level log capture:** the right pane displays lines emitted to the process's real stdout/stderr file descriptors (pybullet's native prints), not only Python `logging` records, via `os.dup2` fd redirection that is torn down cleanly on exit (no lost/duplicated final output, no deadlock if the pipe fills), with a bounded scrollback.
6. **Health verdict rendering:** the bottom status bar renders the `assess_training_health()` result (message + severity styling: normal / warning / critical). The engine is injected; when absent, a stub verdict ("Training progressing normally") is used so UC-22 stands alone.
7. **Scope containment / no regression:** no change under `src/drone_fly/{env,controller,adapter,connectome}` behavior beyond the additive `VecMonitor` wrap; observation schema, reward, and checkpoints are untouched (no retrain). Full existing suite stays green on Linux CI (`ruff check .`, `ruff format --check .`, `uv run pytest`). The new Rich runtime dependency is declared and pinned.
8. **Graceful degradation:** a terminal too small for the layout, or a `NO_COLOR`/dumb-terminal environment, does not crash the run — it degrades (shrinks/omits panes or falls back to line logging) and training completes normally; Ctrl-C mid-run restores the terminal (fds restored, no garbled prompt).

## Potential Pitfalls & Open Questions
- **Assumption** — Rich is added to the **runtime** (not dev-only) dependencies since the TUI is default-on for real runs; pinned to a currently-available version.
- **Edge case** — fd-level `os.dup2` redirection must not swallow the SB3 logger's own stdout when the TUI is off, and must restore fds on exceptions/`KeyboardInterrupt` (Ctrl-C mid-run leaves the terminal usable).
- **Edge case** — SB3's `Logger` and Rich's `Live` both write to stdout; when the TUI is on, SB3's `stdout` HumanOutputFormat must be dropped from the logger (CSV/TensorBoard kept) so they don't fight over the screen.
- **Boundary** — the live Rich render is **not** exercised in Linux CI (no interactive TTY), exactly like the viewer and pybullet sim. Coverage is the headless data layer + a `--no-tui`/non-TTY fallback assertion; the drawing is a documented untested boundary. Do not add a test that spawns a pty expecting pixels.
- **Dependency** — the status-bar verdict is owned by UC-23; UC-22 codes against an interface and stubs it if UC-23 isn't merged yet.
- **Risk** — vec-env wrapper order: `VecMonitor` must sit **inside** `VecNormalize` (Monitor wraps the raw env, normalization on top) so episode stats are raw, not normalized.

## Original Description
From the user (gathered in the root-session design conversation): "I want this to become a full-screen CLI, something like tmux. Right side we could have logs (e.g. all the crap output by what I guess is pybullet) occupying about 30% of the width. Left side we should have a panel on top with these metrics for time, train, rollout, and under it a list of the tendency of every result. E.g. the first row being success_rate and an icon showing if the tendency is improving or worsening, another with a progress bar of the number of iterations vs scheduled iterations, and the other values that can signify issues. On the bottom, a small status segment/bar, full-width, showing 'Training progressing normally', 'WARNING: Connectome too small', or whatever depending on the values/progress we are seeing."

Layout contract agreed (embedded above as the wireframe). Preceding context: the current output is SB3's stock `HumanOutputFormat` table, and the training env is **not** Monitor-wrapped, so `ep_rew_mean`/`ep_len_mean`/`success_rate` are never collected — hence adding `VecMonitor` is folded into this UC as the enabling data-layer change.

## Clarifications
- Q: How should the TUI turn on?
  A: Default-on whenever stdout is an interactive TTY; `--no-tui` disables it; non-TTY/piped/CI falls back to the current SB3 line/table logger (CI stays green).
- Q: Which TUI library backs it?
  A: Rich (`Live` + `Layout`), redrawn from a SB3 callback each PPO update. No async framework.
- Q: How deep should the log pane capture go?
  A: Capture at the C-level file descriptors (`os.dup2` stdout/stderr into a pipe) so pybullet's native prints are captured, not only Python logging.
- Q: Where does the status-bar health logic live?
  A: UC-23 owns a headless, CI-tested `assess_training_health()` engine; UC-22's status bar only renders its verdict. UC-22 stubs the verdict if UC-23 isn't yet built.
- Q: Which metrics get a trend row?
  A: 7 rows — success_rate, ep_rew_mean, entropy, std, value_loss, approx_kl, explained_var — plus the iterations progress bar.
- Q: Sparkline / trend window length?
  A: Rolling window of the last ~100 updates (rebucketed to fixed sparkline width).
- Q: Data/render separation?
  A: Required — data layer (metrics, trend, sparkline, ETA, progress) is pure and unit-tested headlessly; only the live Rich draw is an untested boundary.
- Q: Runtime/observation/checkpoint impact?
  A: None — observability/output only; adds VecMonitor + TUI + Rich dep; no env/obs/reward/connectome/checkpoint change, no retrain.
