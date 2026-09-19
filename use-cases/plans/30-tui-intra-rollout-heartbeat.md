---
plan_for: use-cases/30-tui-intra-rollout-heartbeat.md
work_branch: feat/uc-30-tui-intra-rollout-heartbeat
team: drone-fly-uc-30
approved: 2026-09-19
---

# UC-30 — Intra-rollout heartbeat: FINAL APPROVED PROPOSAL
Analyst↔challenger agreement reached (challenger approved rev 2). Ready for developer + QA.

## Analysis
The UC-22 live TUI sits at all-zeros (elapsed included) during a slow first rollout because `TuiCallback._on_step` is a bare `return True` and `elapsed_seconds` is computed only in `_on_rollout_end` (callback.py:51-52, 73). Rich `Live` (refresh_per_second=4) only re-renders the last renderable `redraw()` built, so it stays pinned to the initial all-zero model. The metrics/render layer updates correctly when fed — the only gap is feed cadence. Fix = a throttled intra-rollout heartbeat on `_on_step`.

## Proposed Solution (4 files)

### (a) `src/drone_fly/train/tui/callback.py` — heartbeat + clock seam
- **Clock seam:** add `now: Callable[[], float] = time.monotonic` to `TuiCallback.__init__`, store `self._now`; replace both `time.monotonic()` uses with `self._now()`.
- **Throttle:** module const `_HEARTBEAT_MIN_INTERVAL = 0.25` (4×/sec, matches `dashboard._REFRESH_PER_SECOND`). New `self._last_heartbeat: float | None = None`. In `_on_step`: if `not self._enabled` → `return True`; `now = self._now()`; if `_last_heartbeat is not None and (now - _last_heartbeat) < _HEARTBEAT_MIN_INTERVAL` → `return True`; else set `_last_heartbeat = now` and run the heartbeat body; always `return True`.
- **`_on_rollout_start()` (new hook):** capture `self._rollout_start_timesteps = self.num_timesteps` and reset `self._last_heartbeat = None` (first step of each rollout redraws immediately → AC-1 instant liveness).
- **Heartbeat body:**
  - `current = max(0, self.num_timesteps - self._rollout_start_timesteps)`
  - `n_steps = int(getattr(self.model, "n_steps", 0) or 0)`; `n_envs = int(getattr(self.training_env, "num_envs", 0) or getattr(self.model, "n_envs", 1) or 1)`; `target = n_steps * n_envs` (Dummy/Subproc-safe, mirrors health_callback.py:96).
  - `elapsed = self._now() - self._start_time` (guard `_start_time is None` → 0.0).
  - `self.dashboard.model.tick(elapsed_seconds=elapsed, rollout_steps=current, rollout_target=target)` then `self.dashboard.redraw()`.
- **Defensive:** wrap the body in `try/except Exception` mirroring `_on_rollout_end` (callback.py:88-90): on error log + `self._enabled = False`; never raise.
- **`_on_rollout_end` left byte-for-byte unchanged.**

### (b) `src/drone_fly/train/tui/metrics.py` — dedicated `tick()` (NOT via `update()`)
Critical correctness point (found by challenger, verified): `update()`'s `raw[...]` assignments (metrics.py:379-384) are **unconditional**, so routing the heartbeat through `update()` would blank all six values-panel entries to `—` after the first rollout. Therefore:
- Add fields in `__init__`: `self.rollout_steps: int = 0`, `self.rollout_target: int = 0`.
- Add:
```python
def tick(self, *, elapsed_seconds, rollout_steps, rollout_target) -> None:
    self.elapsed_seconds = float(elapsed_seconds)
    self.rollout_steps = int(rollout_steps)
    self.rollout_target = int(rollout_target)
```
  Touches neither `history` nor `raw` → the last rollout-end snapshot persists through the next collection.
- Add `rollout_progress(self) -> tuple[int,int,float]` → `(rollout_steps, rollout_target, progress_fraction(rollout_steps, rollout_target))` (reuses existing helper; clamps [0,1], 0 when target unknown).
- **`update()` and `_on_rollout_end` are NOT modified.**

### (c) `src/drone_fly/train/tui/render.py` — collecting bar
In `build_trends_panel`, **below** the existing iterations progress bar, add a distinct "collecting" row: `ProgressBar(total=max(target,1), completed=min(current,target))` + label `f"collecting {cur}/{tgt} ({format_pct(frac)})"`, sourced from `model.rollout_progress()`. Distinct label ("collecting" vs "iterations") keeps the UC-22 iterations bar intact. Placed in the trends panel (`ratio=1`, flexible), NOT the values panel (fixed `size=7`, already full → would clip).

### (d) `dashboard.py` / `loop.py` — NO change
`redraw()` already rebuilds from the model each call (so log pane refreshes live → AC-6). Clock seam defaults to `time.monotonic`, and n_steps/n_envs are read at runtime, so the `TuiCallback(dashboard)` call site (loop.py:528) is unchanged.

## Files Affected
**Production (developer):** `src/drone_fly/train/tui/callback.py`, `src/drone_fly/train/tui/metrics.py`, `src/drone_fly/train/tui/render.py`. (dashboard.py/loop.py unchanged.)
**Test (qa):** `tests/test_tui_callback.py`, `tests/test_tui_metrics.py`, `tests/test_tui_render.py`.

## Tests (pure model/render + synthetic callback harness; injected clock; NO live Rich, NO pty, NO sleeps — matches test_tui_callback.py:44-57)
- **AC-1:** `_on_training_start`→`_on_rollout_start`→`_on_step` with injected clock; assert `tick` gave non-zero `elapsed_seconds` + `rollout_steps>0`, zero `_on_rollout_end` calls, redraw fired, `raw` still all-None pre-rollout.
- **AC-2:** many rapid `_on_step` calls under a controlled clock → redraw count bounded (~4/sec).
- **AC-3:** `current = num_timesteps - rollout_start` vs `n_steps*n_envs`; resets on a second `_on_rollout_start`.
- **AC-4 (the key regression test):** full `_on_rollout_end` snapshot (populated `raw`) → several heartbeats → assert `model.raw` unchanged AND values-panel render still shows those values, not `—`. Plus existing rollout-end tests kept.
- **AC-5:** boom in heartbeat → `_enabled=False`, no raise.
- **AC-6:** heartbeat `redraw()` refreshes `_log_lines()`.
- **AC-7:** collecting bar renders headlessly; UC-22 "iterations" bar still present; full suite green + offline; viewer untouched.
- **metrics:** `tick()` leaves `history`+`raw` untouched (assert before/after); `rollout_progress()` math (reset, clamp, 0-when-target-unknown).

## Risks & Considerations
- 0.25s time gate → per-step fast-path cost is a subtract+compare; fast slices unaffected.
- Injected clock removes all wall-clock/sleep reliance in tests.
- `target=0` (pre-tick or missing attrs) degrades to a 0-fraction bar; `progress_fraction` guards `scheduled<=0` (no div-by-zero).
- Heartbeat never touches training dynamics/obs/connectome/reward/checkpoints; viewer untouched (no JS).

## AC → where satisfied
- AC-1 → callback heartbeat via `tick`; test_tui_callback.
- AC-2 → `_HEARTBEAT_MIN_INTERVAL` + injected clock; redraw-bound test.
- AC-3 → `num_timesteps - _rollout_start_timesteps` vs `n_steps*n_envs`, reset on rollout start.
- AC-4 → `update()`/`_on_rollout_end` untouched + `tick()` isolation; post-rollout regression test.
- AC-5 → try/except disable; boom test.
- AC-6 → `redraw()` rebuilds with `_log_lines()`.
- AC-7 → additive collecting bar; UC-22/23/26 preserved; offline CI; viewer untouched.
