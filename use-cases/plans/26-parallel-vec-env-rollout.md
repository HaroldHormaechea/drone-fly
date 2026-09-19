---
plan_for: use-cases/26-parallel-vec-env-rollout.md
work_branch: feat/uc-26-parallel-vec-env-rollout
team: drone-fly-uc-26
approved: 2026-09-19
---

# UC-26 — Parallel vec-env rollout backend — FINAL APPROVED IMPLEMENTATION PLAN
Approved by the challenger. TARGET_DIR = `/workspace/drone-fly-uc-26-parallel-vec-env-rollout`. Self-contained source of truth for developer + QA. Covers the 11 use-case ACs plus the two user gates (AC-12 recording, AC-13 TUI); honors all six fixed design decisions.

---
## Analysis
PPO rollout is single-core. `build_vec_env` (`.../src/drone_fly/env/racing_env.py:549-602`) hardcodes `DummyVecEnv([_factory for _ in range(max(1,n_envs))])`, which steps every sub-env sequentially in the main process — so raising `n_envs` (CLI default unset→1) only enlarges the rollout batch and per-iteration wall time; net samples/sec stays flat. Fix: add a `SubprocVecEnv` backend (one worker process per env → real cross-core parallelism), auto-selected, wrapped identically to the current stack.

Two config layers feed n_envs and BOTH must be preserved:
- `TrainRunConfig.n_envs: int | None` (`.../src/drone_fly/config.py:223`) — CLI value, `None` = unset. Flows to `train(n_envs=...)`.
- `TrainConfig.n_envs: int = 1` (`.../src/drone_fly/train/config.py:45`) — the fallback used today at `.../src/drone_fly/train/loop.py:321` (`resolved_n_envs = n_envs if n_envs is not None else cfg.n_envs`).

**Load-bearing finding — the "spawn re-seeds the connectome" risk dissolves.** The connectome is NOT in the env. `RaceEnv.__init__` (`racing_env.py:64`) takes only `EnvConfig` + an adapter string and builds `make_adapter(...)` + pure geometry/reward — no connectome. The connectome seeds the *policy* (`ConnectomeFeaturesExtractor` via `build_policy_kwargs` in `loop.py`), which lives in the main-process PPO model and is never pickled to workers. Each spawned worker therefore rebuilds only `RaceEnv(EnvConfig, adapter)` — cheap (import package + one pybullet client + numpy course), which is exactly the parallel work we want. Verified empirically in the worktree: `pickle.dumps(EnvConfig())` round-trips; the factory closure round-trips through SB3 2.9.0's `CloudpickleWrapper` and rebuilds a working env; `SubprocVecEnv.__init__` accepts `start_method`. No cached connectome-artifact path in config is needed.

---
## Proposed Solution

### §1 — Pure resolver + named constants (`.../src/drone_fly/env/racing_env.py`)
Add beside `build_vec_env` (import `pybullet_available` off the existing `from drone_fly.adapter import ...`; no new import cycle):
- Constants: `DEFAULT_PARALLEL_N_ENVS = 8`; `VEC_ENV_START_METHOD = "spawn"`.
- `resolve_vec_env(adapter, n_envs_arg, cfg_n_envs) -> (resolved_adapter, resolved_n_envs, vec_backend)` — the **3-branch resolver (LOCKED)**:
  - `resolved_adapter = adapter if adapter != "auto" else ("pybullet" if pybullet_available() else "simple")`
  - `parallel_capable = (resolved_adapter == "pybullet")`
  - `resolved_n_envs`: `n_envs_arg` if it is not None; **elif** `parallel_capable` → `DEFAULT_PARALLEL_N_ENVS` (8); **else** → `cfg_n_envs` (preserves today's fallback — the branch that keeps the existing tests green).
  - `vec_backend = "subproc" if (parallel_capable and resolved_n_envs > 1) else "dummy"`.
  - User-facing warning: only when `n_envs_arg and n_envs_arg > 1` and not `parallel_capable` (user asked for >1 envs but the simple adapter stays serial DummyVecEnv). This is the subproc-on-simple pitfall surface.
  Pure and independently unit-testable.

### §2 — `build_vec_env` gains two params (`.../src/drone_fly/env/racing_env.py:549`)
New keywords: `vec_backend: str = "auto"` (`"auto"|"dummy"|"subproc"`) and `suppress_worker_output: bool = False`.
- `"auto"` → internally calls `resolve_vec_env(adapter, n_envs, cfg_n_envs=n_envs)` and uses the returned backend (concrete-n_envs callers pass a value, so this is the identity for them).
- `"subproc"` → `SubprocVecEnv([subproc_factory ...], start_method=VEC_ENV_START_METHOD)`; otherwise `DummyVecEnv([_factory ...])`.
- **Two distinct factories (LOCKED, CRITICAL):** the dummy branch keeps the current `_factory` (`return make_env(cfg, adapter=adapter)`) **byte-identical** — it runs in the MAIN process. The subproc branch uses a *separate worker-only* factory that, **only when `suppress_worker_output` is True**, redirects the worker's OWN fd 1/2 to `os.devnull` before building the env. The `os.dup2` executes only inside spawned workers, never the main process (a main-process dup2 would kill main-process logging + the TUI).
- The `VecMonitor(info_keywords=("is_success",))`-inside-`VecNormalize` wrapping (`racing_env.py:585-600`) and `venv.seed(seed)` sit OUTSIDE/independent of the base-env choice → identical for both backends. `norm_reward`/`vecnormalize_path`/`training` branches unchanged; VecNormalize stats remain backend-independent.
- `vec_backend`/`suppress_worker_output` are INTERNAL test hooks — not wired to any CLI/config field; forcing subproc-on-simple through them intentionally does NOT warn (keeps QA output clean).
- Backward-compat: existing callers (`evaluate/evaluator.py:106` with `n_envs=1`; `prune_trained/workflow.py` ×3 + `prune_trained/measure.py`; existing tests) pass concrete `n_envs=1` and no `vec_backend` → `"auto"`→dummy → byte-identical.

### §3 — `train()` wiring (`.../src/drone_fly/train/loop.py`)
- Replace the `resolved_n_envs = n_envs if n_envs is not None else cfg.n_envs` line (~:321) with `resolved_adapter, resolved_n_envs, vec_backend = resolve_vec_env(adapter, n_envs, cfg.n_envs)`.
- Use `resolved_n_envs` for the existing scheduled-iters/save-freq math (`loop.py:437-447` and `:559`) — formulas unchanged.
- Call `build_vec_env(config=env_config, adapter=resolved_adapter, n_envs=resolved_n_envs, vec_backend=vec_backend, suppress_worker_output=tui_enabled, seed=cfg.seed, training=True, vecnormalize_path=stats_path)`.
- Emit the AC-11 startup log once: `logger.info("rollout: n_envs=%d backend=%s (adapter=%s)", resolved_n_envs, vec_backend, resolved_adapter)` (covers the TUI-off requirement).
- `smoke_train` untouched — forces `adapter="simple"` → dummy, default 1 → CI stays green.

### §4 — AC-11 TUI (pure layer only; no live `Live`)
- `.../src/drone_fly/train/tui/metrics.py`: `DashboardModel.__init__` gains `n_envs: int = 1` and `backend: str = "dummy"` (stored as resolved values).
- `.../src/drone_fly/train/tui/render.py`: `build_values_panel` renders them (e.g. a ` envs  8 (subproc)` line in the TIME column of the "drone-fly · training" panel) — must show RESOLVED values.
- `.../src/drone_fly/train/tui/dashboard.py`: `TrainingDashboard.__init__` gains `n_envs`/`backend`, forwarded into `DashboardModel`.
- `loop.py`: build `TrainingDashboard(scheduled_iters=..., n_envs=resolved_n_envs, backend=vec_backend)`. `TuiCallback` unchanged (values are static per run).

### §5 — Docs (`.../configs/train/example.yaml`)
Expand the `n_envs` comment to document: the auto rule (subproc iff pybullet & n_envs>1), default-8 for parallel runs, the `spawn` start method, P-core-vs-E-core sweet-spot guidance (8 = M4 Pro's 8 performance cores; E-cores oversubscribe), and that the `simple` adapter stays serial. Mirror in README if it documents training knobs.

---
## Files Affected

### Production code (developer)
- `/workspace/drone-fly-uc-26-parallel-vec-env-rollout/src/drone_fly/env/racing_env.py` — add `DEFAULT_PARALLEL_N_ENVS`, `VEC_ENV_START_METHOD`, `resolve_vec_env`; `build_vec_env` gains `vec_backend` + `suppress_worker_output`, the SubprocVecEnv branch, and the subproc-only worker factory (devnull redirect gated on `suppress_worker_output`).
- `/workspace/drone-fly-uc-26-parallel-vec-env-rollout/src/drone_fly/train/loop.py` — call the resolver; thread resolved adapter/n_envs/vec_backend + `suppress_worker_output=tui_enabled` into `build_vec_env`; AC-11 startup log; pass n_envs/backend to `TrainingDashboard`.
- `/workspace/drone-fly-uc-26-parallel-vec-env-rollout/src/drone_fly/train/tui/metrics.py` — `DashboardModel` n_envs/backend fields.
- `/workspace/drone-fly-uc-26-parallel-vec-env-rollout/src/drone_fly/train/tui/render.py` — render resolved n_envs/backend in `build_values_panel`.
- `/workspace/drone-fly-uc-26-parallel-vec-env-rollout/src/drone_fly/train/tui/dashboard.py` — `TrainingDashboard` n_envs/backend params.
- `/workspace/drone-fly-uc-26-parallel-vec-env-rollout/configs/train/example.yaml` — parallel-behavior docs.
- (optional) `/workspace/drone-fly-uc-26-parallel-vec-env-rollout/src/drone_fly/env/__init__.py` — export resolver/constants if convenient.

### Test code (QA)
- `/workspace/drone-fly-uc-26-parallel-vec-env-rollout/tests/test_build_vec_env.py` — resolver truth table (pybullet+n>1→subproc; simple→dummy; n_envs==1→dummy; arg-None → 8 if parallel-capable else `cfg_n_envs`) [AC-1/AC-5]; wrapper order identical for a forced `vec_backend="subproc"` build (VecNormalize→VecMonitor→SubprocVecEnv) [AC-2]; forced subproc-on-simple reset+step completes with finite/correct-shape obs [AC-4]; `.close()` after an induced worker error does not hang [worker-crash pitfall]. Monkeypatch `pybullet_available` to exercise the pybullet branches without pybullet installed.
- `/workspace/drone-fly-uc-26-parallel-vec-env-rollout/tests/test_train_resume.py` — the three existing n_envs tests MUST stay green: `test_train_n_envs_defaults_to_cfg_when_omitted` (arg None, simple, cfg=2 → 2), `test_train_n_envs_save_freq_uses_resolved_count` (arg 4, simple → 4, 64//4=16), `test_resume_smoke_tolerates_changed_n_envs` (arg 2, simple → 2).
- **AC-3** (new, e.g. in `tests/test_build_vec_env.py` or a new `tests/test_vec_parallel.py`) — a module-level pid-returning env wrapped directly in `SubprocVecEnv(start_method="spawn")` asserts distinct worker PIDs (proves cross-process + the start-method constant); the real factory's cross-core functionality is proven by AC-4's forced-subproc step (SB3 exposes no PID getter on our env).
- **AC-12** (new `tests/test_record_parallel.py`) — force `build_vec_env(vec_backend="subproc", adapter="simple", n_envs=2, training=True, seed=0)`, build a small PPO on the committed fixture connectome + attach `RecordingCallback` + `ActivationRecorder(tmp_path)`, run a TINY `model.learn` (smoke/tiny_cfg budget — keep it fast/non-flaky under real `spawn`); assert episode files exist, parse as valid JSON, have monotonically-increasing indices, and single-env frame counts. Plus eval-path invariance: an `evaluate_checkpoint(record=True)` recording (n_envs=1) is uncorrupted and its stack contains no SubprocVecEnv.
- **AC-13** (new, TUI) — worker-isolation: force `build_vec_env(vec_backend="subproc", adapter="simple", n_envs=2, suppress_worker_output=True)` with a module-level probe env that writes to stdout in `reset`/`step` → assert NO leakage to the parent's stdout; companion with `suppress_worker_output=False` (default) → assert output is NOT suppressed (proves the gate both ways); assert `rich`/`drone_fly.train.tui` absent from a worker. Pure-layer (in `tests/test_tui_metrics.py`/`tests/test_tui_render.py`): `DashboardModel`/`build_values_panel` show resolved n_envs+backend for both `8/subproc` and `1/dummy` — no live `Live`.
- `/workspace/drone-fly-uc-26-parallel-vec-env-rollout/tests/test_config.py` — existing n_envs-None passthrough stands; add resolver-default assertions if not colocated in test_build_vec_env.

---
## Acceptance-criteria mapping (AC-1 .. AC-13)
- **AC-1 (auto backend, wired end-to-end):** §1 `resolve_vec_env` + §2 `build_vec_env` branch + §3 `train()` call. Test: resolver truth table.
- **AC-2 (identical wrapper stack):** §2 — VecMonitor-inside-VecNormalize + seed are backend-independent. Test: forced-subproc wrapper-order assertion.
- **AC-3 (real cross-core parallelism):** §2 SubprocVecEnv with `spawn`. Test: pid-distinct probe env + AC-4 functional step.
- **AC-4 (spawn correctness / picklable factory):** §1 `VEC_ENV_START_METHOD="spawn"`; connectome-not-in-env finding + verified cloudpickle round-trip. Test: forced subproc-on-simple reset+step.
- **AC-5 (n_envs default 8 for parallel):** §1 3-branch resolver (arg-None + parallel_capable → 8; else `cfg_n_envs`; explicit wins). Test: resolver defaults + the three test_train_resume cases.
- **AC-6 (hermetic CI green):** §3 smoke_train untouched (simple→dummy); no new heavy dep (SB3 ships SubprocVecEnv). Whole suite stays green.
- **AC-7 (checkpoint/resume + step math at larger n_envs):** §2 VecNormalize `.load` unchanged/backend-independent; §3 scheduled-iters/save-freq math uses `resolved_n_envs` (formulas at loop.py:437-447/559 unchanged). Test: AC-12 uses PPO.load path; resolver feeds the math.
- **AC-8 (policy stays on CPU):** no device change anywhere; `resolve_device` untouched.
- **AC-9 (no obs/connectome/checkpoint change):** rollout plumbing only; existing callers byte-identical; base (dummy) path unchanged.
- **AC-10 (documented named config):** §1 constants + §5 example.yaml docs.
- **AC-11 (parallelism surfaced in UC-22 TUI):** §4 resolved n_envs+backend through the pure DashboardModel/render layer; §3 startup log for TUI-off. Test: pure render assertion (resolved values, both backends).
- **AC-12 (recording safety — user gate #1):** single main-process writer, env-0 slice, no worker writes; eval/playback single-env. Test: forced-subproc record run + eval invariance.
- **AC-13 (TUI under parallel — user gate #2):** metrics/health backend-agnostic (main-process VecMonitor→buffers→TuiCallback); Live/fd-capture main-process singletons; workers never import the TUI; log-pane devnull redirect gated on `tui_enabled`. Test: worker-isolation + pure-layer.

Every `## Potential Pitfalls` item is covered: spawn re-import cost (dissolved — connectome not in env); simple+subproc (auto keeps simple on dummy; warns if user sets n_envs>1 on simple); deterministic tests vs multiprocess RNG (determinism tests stay on dummy/simple; equivalence at learns/updates level); worker crash/cleanup (SubprocVecEnv exception pipe + `.close()` test); oversubscription (documented per fixed decision #3); stock SubprocVecEnv used as-is (no custom shared memory).

---
## Risks & Considerations
- Multiprocess rollout ordering is not byte-reproducible → determinism-sensitive tests stay on dummy/simple; equivalence asserted at the learns/updates level (fixed decision #5).
- 8 workers + 1 learner on an 8P+4E M4 Pro slightly oversubscribes P-cores — accepted per fixed decision #3, documented in example.yaml.
- Policy stays on CPU (fixed decision #4); no device changes.
- No observation/connectome/reward/checkpoint change — rollout infrastructure only (fixed decision #5); VecNormalize stats backend-independent.
- `profiles: []` in the brief → no profile skills apply (Python project; no java-* profiles).

---
## Grounded verdicts (as the user requested)
- **Recording saves are unaffected because** eval and UC-05/06 playback recording run single-env (`build_vec_env(n_envs=1)` → DummyVecEnv, or a single raw `gymnasium.Env`) and never enter the parallel backend; and training-time recording's sole writer — the main-process `ActivationRecorder`/`RecordingCallback` — captures only env-0's slice from the batched arrays SubprocVecEnv returns, so the spawn workers (which only step envs and never touch the recorder or its output dir) cannot collide with, overwrite, interleave, or corrupt any save.
- **UC-22 TUI reporting is unaffected/handled because** its data panels (metrics, 7 trends, iterations bar, health status) all source from the main-process `VecMonitor`→`ep_info_buffer`→`TuiCallback` (backend-independent); Rich `Live` + fd-capture are main-process singletons and workers never import the TUI (empirically verified); and the only behavior change — the log pane showing main-process logs rather than per-worker pybullet native spew under subproc — is a deliberate, documented choice with worker fds redirected to devnull **only when the live display is active (`tui_enabled`)** so nothing leaks onto/corrupts the display, while non-TUI subproc runs keep worker output and worker crashes still surface via SubprocVecEnv's exception pipe.

---
## Locked decisions
1. **3-branch `resolve_vec_env(adapter, n_envs_arg, cfg_n_envs)`** — explicit arg wins; else parallel-capable→8; else `cfg_n_envs`.
2. **`VEC_ENV_START_METHOD = "spawn"`** for SubprocVecEnv.
3. **Log-pane handling** — subproc-factory-only `os.devnull` fd 1/2 redirect, gated on `suppress_worker_output=tui_enabled`; dummy factory byte-identical; non-TUI subproc runs leave worker output intact.
4. `DEFAULT_PARALLEL_N_ENVS = 8`. `vec_backend`/`suppress_worker_output` are internal test hooks (no CLI surface, no warn). AC-11 via the pure DashboardModel/render layer.

**All 13 ACs and every pitfall covered; no fixed design decision violated. Approved — ready for developer/QA.**
