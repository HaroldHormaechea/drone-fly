---
plan_for: use-cases/52-optimize-phase-tui-progress.md
work_branch: feat/uc-52
team: drone-fly-uc-52
approved: 2026-09-23
---

# UC-52 — Optimize-phase progress + collect/optimize timing in the training TUI

APPROVED by challenger (first round; all load-bearing claims re-verified against the `feat/uc-52` checkout + installed SB3 2.9.0). One Minor note (single-owner begin/end) folded in.

## Approach (challenger-approved)
A minimal `ProgressReportingPPO(PPO)` subclass instruments the hook-less optimize loop via a counting-proxy over `rollout_buffer.get()` — zero duplication of gradient/loss math. An optional sink is inert when unattached (headless/`--no-tui`/smoke/resume-before-attach) so those paths stay byte-identical to plain PPO. All redraws go through the dashboard's existing single RLock (no new thread). Optimize redraw throttled to **1 Hz** via a dedicated constant, separate from the 4 Hz collection heartbeat.

## Verified seams (current checkout + installed SB3 2.9.0)
- `PPO.train()`: `for epoch in range(n_epochs): for rollout_data in self.rollout_buffer.get(self.batch_size)` — one `get()` per epoch, one yield per minibatch; `target_kl` break is inside the minibatch loop + `if not continue_training: break` after each epoch (only ever *reduces* counts → "exceed N ⇒ disable" is inherently safe).
- `RolloutBuffer.get` (buffers.py:481): `while start_idx < buffer_size*n_envs: yield…; start_idx += batch_size`, `buffer_size == n_steps` ⇒ **M = ceil(n_steps*n_envs/batch_size)**, partial final minibatch included.
- `on_policy_algorithm.py`: `on_rollout_start`(192) → `on_rollout_end`(266) → `self.train()`(337) — collect duration known before optimize.
- `BaseAlgorithm.load` (base_class.py:730) `model = cls(...)` ⇒ `ProgressReportingPPO.load(...)` returns the subclass (resume path AC-8 covered by the subclass reference alone).
- loop.py: import ~372; resume `PPO.load` ~493; fresh `PPO("MlpPolicy", venv, ...)` ~496; `_apply_climb_bias` ~514; dashboard build ~519.
- callback.py `_HEARTBEAT_MIN_INTERVAL = 0.25` (untouched); dashboard.py single RLock + injectable `self._now` (line 92); metrics.py `tick()`/`tick_clock()` non-clobber of `raw`/`history`; render.py `values` region `size=8` (render.py:275), `build_trends_panel` collecting bar, `build_values_panel` TIME column.
- Hermetic test seam: `build_vec_env(adapter="simple", n_envs=1, seed=0, training=True)` + CPU PPO on the committed connectome fixture (per `test_uc40_hover_bias.py`) — no production signature change for AC-10(c).

## Changes

### 1. NEW `src/drone_fly/train/progress_ppo.py` → `ProgressReportingPPO(PPO)`
- **No `__init__` override.** Read the sink via `getattr(self, "_progress_sink", None)`; `attach_progress_sink(sink)` sets it (attach-None supported). Keeps fresh `cls(...)` and resume `load→cls(..., _init_setup_model=False)` byte-identical to plain PPO.
- **Pure module-level helper `counting_get_proxy(real_get, n_epochs, total_minibatches, on_tick)`** (no PPO/torch import to test): returns a `get`-signature callable; each call = one epoch (epoch counter++), each yielded minibatch (minibatch counter++) invokes `on_tick(epoch, minibatch)` and forwards the sample untouched. The proxy takes ONLY `on_tick` (plus N, M for anomaly detection) — it does **not** own begin/end.
- **`train()` override (no gradient-body copy):**
  - `sink = getattr(self, "_progress_sink", None)`; if `None` → `return super().train()` immediately (byte-identical, zero overhead).
  - Else `N = self.n_epochs`, `M = ceil(self.n_steps * self.n_envs / self.batch_size)`. Save `original_get = self.rollout_buffer.get`.
  - **`train()` is the sole owner of begin/end** (challenger Minor #1): call `sink.begin_optimize(N, M)` before `super().train()`, install the proxy (`self.rollout_buffer.get = counting_get_proxy(original_get, N, M, sink.tick_optimize)`), time `super().train()`, then after it returns call `sink.set_optimize_duration(elapsed)` (explicit setter, not overloading end()) and `sink.end_optimize()`.
  - **`finally`:** always restore `self.rollout_buffer.get = original_get`; ensure `end_optimize` fires exactly once even on early return/exception (guarded so begin without end can't leave `is_optimizing` stuck).
  - **Disable-on-anomaly:** if observed epochs exceed `N`, or an unexpected yield/buffer shape appears, set an internal `_disabled` flag, stop calling the sink, let `super().train()` finish untouched. Top-level `try/except` around the whole instrumentation degrades to plain `super().train()`; individual sink calls guarded so a sink error never perturbs training.
- **Sink protocol (clean):** `begin_optimize(N, M)` / `tick_optimize(epoch, minibatch)` / `end_optimize()` / `set_optimize_duration(seconds)` / `set_collect_duration(seconds)`.

### 2. `src/drone_fly/train/loop.py`
- Import `from drone_fly.train.progress_ppo import ProgressReportingPPO`.
- Resume (~493): `ProgressReportingPPO.load(resume, env=venv, device=resolved_device)`.
- Fresh (~496): `ProgressReportingPPO("MlpPolicy", venv, ...)` — kwargs unchanged.
- After the dashboard block (~528, dashboard built after the model + after `_apply_climb_bias`): `if tui_enabled and dashboard is not None: model.attach_progress_sink(dashboard)`. Unattached on TUI-off/`--no-tui`/non-TTY/`smoke_train`/resume-before-dashboard (AC-4, AC-8).

### 3. `src/drone_fly/train/tui/metrics.py` (`DashboardModel`)
- `__init__` state (None/nan-tolerant): `optimize_epoch=0, optimize_total_epochs=0, optimize_minibatch=0, optimize_total_minibatches=0, collect_seconds=None, optimize_seconds=None, is_optimizing=False`.
- Methods (all leave `raw`/`history` untouched, mirroring `tick()`): `begin_optimize(n_epochs, total_minibatches)` (totals + counters=0 + `is_optimizing=True`); `tick_optimize(epoch, minibatch)`; `end_optimize()` (`is_optimizing=False`, keep last counts for a clean 100% frame); `set_collect_duration(seconds)`; `set_optimize_duration(seconds)`.
- Getters: `optimize_progress() -> (cur, total, frac)` with `cur = optimize_epoch*M + optimize_minibatch`, `total = N*M`, `frac = progress_fraction(cur, total)`; `phase_split() -> (collect_s, optimize_s, optimize_frac)` with `optimize_frac = optimize_s/(collect_s+optimize_s)` guarded for None/zero. Text clamps `min(cur, total)` (AC-7).

### 4. `src/drone_fly/train/tui/dashboard.py`
- Dedicated module constant `_OPTIMIZE_MIN_INTERVAL = 1.0` (separate from `_REFRESH_PER_SECOND`/`_TIMER_INTERVAL` and callback `_HEARTBEAT_MIN_INTERVAL`).
- `_last_optimize_redraw: float | None = None` in `__init__`.
- Lock-guarded passthroughs (each `with self._lock:`, all defensive):
  - `begin_optimize(N, M)`: mutate model, reset `_last_optimize_redraw = None`, redraw unconditionally.
  - `tick_optimize(epoch, minibatch)`: mutate counters unconditionally; gate the Rich rebuild behind `_OPTIMIZE_MIN_INTERVAL` via injectable `self._now`; first tick after begin (gate None) always redraws.
  - `end_optimize()`: mutate model, reset gate, redraw unconditionally.
  - `set_collect_duration(seconds)` / `set_optimize_duration(seconds)`: mutate model only, no redraw (mirror `set_verdict`).
- All redraws via existing `_redraw_locked()` / single RLock — no new thread (AC-9).

### 5. `src/drone_fly/train/tui/callback.py`
- `_on_rollout_start`: record `collect_start = self._now()`.
- `_on_rollout_end`: `collect_seconds = self._now() - collect_start`; feed `self.dashboard.set_collect_duration(collect_seconds)`. Inside the existing defensive try/except; heartbeat/`_HEARTBEAT_MIN_INTERVAL` untouched (AC-3).
- (Optimize duration is measured inside `train()` and fed via `set_optimize_duration`.)

### 6. `src/drone_fly/train/tui/render.py`
- `build_trends_panel`: add `optimizing epoch e/N · minibatch m/M (X%)` ProgressBar row directly below the `collecting` bar, from `optimize_progress()`; clamp `min(epoch,N)`/`min(minibatch,M)`; degrade to empty bar + placeholder when not optimizing / pre-first-tick.
- `build_values_panel` TIME column: split line e.g. `split c 12s · o 46s (79%)` from `phase_split()` (`format_duration`/`format_pct`; `—` when None).
- `build_layout`: bump `values` region `size` 8→9 (confirmed render.py:275). Trends panel is `ratio=1`, no change.

## Files Affected
**Production (developer):** NEW `src/drone_fly/train/progress_ppo.py`; `src/drone_fly/train/loop.py`; `src/drone_fly/train/tui/metrics.py`; `src/drone_fly/train/tui/dashboard.py`; `src/drone_fly/train/tui/render.py`; `src/drone_fly/train/tui/callback.py` — all match `paths.production=["src/drone_fly/**"]`.
**Test (qa):** NEW `tests/test_uc52_optimize_progress.py` — match `paths.test=["tests/**"]`.

## AC → change map (all 11)
1. optimizing bar below collecting, cumulative `(epoch*M+minibatch)/(N*M)` → render §6 + `optimize_progress()` §3.
2. collect-vs-optimize split in TIME column → callback §5 (collect) + `train()` optimize timing + `set_optimize_duration` §1 + `phase_split()` §3 + render §6.
3. dedicated `_OPTIMIZE_MIN_INTERVAL=1.0` distinct from `_HEARTBEAT_MIN_INTERVAL`; begin/end redraw immediately → dashboard §4.
4. sink None ⇒ `train()` returns `super().train()` immediately, byte-identical, zero overhead → progress_ppo §1 + attach-only-when-TUI §2.
5. counting-proxy over `rollout_buffer.get`, no gradient-body copy → progress_ppo §1.
6. disable-on-anomaly + `finally` restore of `rollout_buffer.get` → progress_ppo §1.
7. `M = ceil(n_steps*n_envs/batch_size)`, `min(cur,total)` clamp → §1/§3/§6.
8. `ProgressReportingPPO` on fresh + resume; sink attached only when TUI active → loop §2.
9. all redraws via existing single RLock, no new thread → dashboard §4.
10. hermetic tests (a–e) → test file.
11. scope confined to the six files + tests; reward/env/adapter/policy untouched → whole proposal.

## Test plan (AC-10, hermetic — no GPU/pybullet)
- (a) render assertions: optimizing bar + split line across synthetic begin/tick/end (0%, mid, 100%, pre-first-tick placeholder), headless layout build.
- (b) `counting_get_proxy` pure helper with fake iterables + recording `on_tick` → exactly `N×M` ticks with correct `(epoch, minibatch)` indices; partial-final-minibatch case; **train() emits exactly one begin / one end** (single-owner, challenger Minor #1).
- (c) hermetic smoke run: `ProgressReportingPPO("MlpPolicy", build_vec_env(adapter="simple", ...), n_epochs, batch_size, device="cpu", policy_kwargs=build_policy_kwargs(connectome, cfg))` + recording sink + `learn(total_timesteps=256)` → per iteration `begin_optimize → tick_optimize(≥1) → end_optimize` with correct N/M, on the committed fixture.
- (d) 1 Hz throttle via `_FakeClock` injected as `now=` (per `test_tui_refresh_clock.py`) + fake Live sink → redraw throttled to ≈1/sec; begin/end force immediate redraw regardless of gate.
- (e) unattached-sink byte-identity: `train()` with no sink calls `super().train()` and never touches `rollout_buffer.get` (spy).
- non-clobber: begin/tick/end_optimize leave `raw`/`history` untouched.

## Risks & Considerations
- Instance-level rebind of `self.rollout_buffer.get` for the duration of `super().train()`; restored in `finally`. Single-threaded, buffer not shared. Top-level guard degrades to plain `super().train()` if `rollout_buffer` is unexpectedly absent.
- `target_kl` early-stop reduces observed counts (bar just never hits 100% that iter) — tolerated; only *exceeding* N trips disable. drone-fly config has `target_kl=None`.
- 1 Hz redraw overhead ≈0.25–0.75% of optimize wall-clock (`backward()`/`optimizer.step()` release the GIL); zero on unattached path.
- Layout 8→9 bump low-risk (trends panel flexes at `ratio=1`).
- SB3 drift bounded by hard `==2.9.0` pin (confirmed in pyproject.toml); proxy reads only the stable one-`get()`-per-epoch contract.

## Challenger verdict
APPROVE. All 11 ACs map to concrete changes; scope confined to the six `src/drone_fly/**` files + `tests/**`. Minor note folded in: single ownership of begin/end_optimize lives in `train()`, not the counting-proxy, so AC-10(b)'s "exactly one begin / one end" holds; AC-10 tests enforce it.
