---
plan_for: use-cases/23-training-health-and-capacity-guardrail.md
work_branch: feat/uc-23-training-health-and-capacity-guardrail
team: drone-fly-uc-23
approved: 2026-09-19
---

# UC-23 — Training-health assessment engine + pre-train capacity guardrail (APPROVED plan)

Approved by challenger after one revision round. `TARGET_DIR=/workspace/drone-fly-uc-23-training-health-and-capacity-guardrail`; brief frontmatter → `paths.production=["src/drone_fly/**"]`, `paths.test=["tests/**"]`, `profiles: []`, build=uv, tests `uv run pytest`. Note: **UC-22 does not exist in the repo yet** — greenfield; design the injectable verdict interface UC-22 will later consume (no stub to replace today).

## Analysis
Training runs via `src/drone_fly/train/loop.py::train`: load connectome → optional `prune_to_subcircuit(k)` (`DEFAULT_PRUNE_K=2`) → build `PPO("MlpPolicy", venv, policy_kwargs=build_policy_kwargs(...))` (feature backbone `ConnectomeFeaturesExtractor` wrapping `ConnectomeActorNetwork`) → `model.learn(callback=[CheckpointCallback, optional RecordingCallback])`.

Two verified findings that shape the design:
- **No `Monitor` in `build_vec_env`** (DummyVecEnv→VecNormalize only), so SB3 does NOT log `rollout/ep_rew_mean`/`rollout/success_rate`. The runtime callback therefore **self-accounts** `ep_rew_mean` and `success_rate` from step infos (env sets `info["is_success"]`/`info["completed"]` at racing_env.py:510/531), reusing the RecordingCallback pattern — this keeps the env stack untouched (AC7). Available `train/*` keys: `explained_variance`, `approx_kl`, `value_loss`, `std` (=`exp(log_std).mean()`, the entropy/commitment proxy), `loss`, `policy_gradient_loss`.
- **Capacity signal** = trainable params of the resolved actor `model.policy.features_extractor.actor` (reach via existing `actor_from_model(model)` in `controller/sb3.py`): `sum(p.numel() for p in actor.parameters() if p.requires_grad)` = sparse `edge_weight` (one per pruned-connectome edge, dominant) + `input_projection` Linear(12→32) + `readout` Linear(16→4). Edge count dominates → cleanly discriminates k=0 (undersized) from k=2 (default) slices.

## Proposed Solution

**NEW `src/drone_fly/train/health.py` — pure headless engine (no I/O, fully unit-testable).**
- Types: `HealthStatus = Literal["normal","warning","critical"]`; frozen dataclasses `HealthReason(rule_id, text, status)` and `HealthVerdict(status, message, reasons: list[HealthReason])`. Overall `status` = worst reason (critical>warning>normal); `message` = short human summary. This stable shape IS the interface UC-22's status bar will render.
- `TrainingSignals`: rolling-window snapshot — current value + recent history for `ep_rew_mean`, `success_rate`, `explained_variance`, `entropy_std`, `approx_kl`, `value_loss`, plus `n_updates`.
- `CapacityFacts`: `trainable_params: int`, `obs_dim: int`, `action_dim: int` (+ optional `neuron_count`/`edge_count` for the message).
- `HealthThresholds` frozen dataclass with **named, documented defaults** (no magic numbers): `min_updates` (warm-up), `success_eps` (~0 band), `entropy_flat_tol`, `ev_healthy`, `kl_target_band`, `value_loss_divergence_factor`, `entropy_collapse_frac`, `reward_stall_tol`, `capacity_floor: int`. Module constants `DEFAULT_THRESHOLDS` and `DEFAULT_CAPACITY_FLOOR` (floor set from MEASURED fixture values — see Tests).
- **Single public entry** (matches AC1/AC4 literal interface): `assess_training_health(signals: TrainingSignals | None = None, capacity: CapacityFacts | None = None, thresholds: HealthThresholds = DEFAULT_THRESHOLDS) -> HealthVerdict`. `signals is None` → capacity-only pre-train path; `signals` present → runtime assessment (capacity, if passed, folded in as an extra reason). Private helpers `_assess_capacity(capacity, thresholds)` and `_assess_runtime(signals, thresholds)` for clean tests.
- `_assess_runtime` checks the **warm-up guard FIRST** — `signals.n_updates < thresholds.min_updates` → returns `normal`/"insufficient history" BEFORE inspecting any metric series. Then five independent rules, each emitting a `HealthReason` when it fires:
  - **under_capacity_signature** — EV healthy/rising AND entropy_std flat & high AND success_rate ~0 over window → warning/critical "likely-undersized connectome slice (K0 signature)". Does NOT fire if entropy_std falling OR success/reward climbing.
  - **reward_stalled** — `ep_rew_mean` not improving / declining across window.
  - **approx_kl_runaway** — KL sustained above target band.
  - **value_loss_divergence** — value_loss growing unbounded and/or EV collapsing.
  - **premature_entropy_collapse** — entropy_std crashing near zero early, before any success/reward gain.
- `_assess_capacity`: `trainable_params < capacity_floor` → warning/critical under-capacity verdict; else `normal`.

**NEW `src/drone_fly/train/capacity_guard.py` — impure pre-train orchestration.**
- `count_actor_trainable_params(model) -> int` via `actor_from_model`.
- `CapacityAbort(RuntimeError)` — signals a clean non-zero-exit abort.
- `enforce_capacity(model, *, strict: bool, thresholds, is_interactive=None) -> HealthVerdict`: build `CapacityFacts`, call `assess_training_health(capacity=...)`, **always log the verdict line** (AC4, runs on every start). If under-capacity: `strict` → raise `CapacityAbort` (both modes); elif interactive TTY (`sys.stdin.isatty() and sys.stdout.isatty()`) → `input()` confirm, decline → raise `CapacityAbort`; else (no TTY) → log warn-and-continue. At/above floor → no prompt, no behavior change.

**NEW `src/drone_fly/train/health_callback.py` — SB3 `BaseCallback` runtime wiring.**
- Owns a rollout counter (`n_updates`, incremented in `_on_rollout_end`) INDEPENDENT of `name_to_value`.
- Self-accounts per-env episode reward + success from `self.locals` infos/dones/rewards; records success on the `dones[i]` step via `info.get("is_success", False)`; reward accumulators reset on done. Rolling deques.
- Reads `train/*` via `self.model.logger.name_to_value.get(key, default)` (default `None`, filtered out) so first-rollout absence never KeyErrors.
- On `_on_rollout_end`: build `TrainingSignals`, call `assess_training_health(signals=...)`, store `self.latest_verdict`, invoke optional injected `on_verdict: Callable[[HealthVerdict], None]` (the seam UC-22 plugs into). Emit a log line only on **status change or every N updates** (AC6 no-spam cadence). Fully defensive — never crashes training (RecordingCallback pattern).

**MOD `src/drone_fly/train/loop.py`** — add `strict_capacity: bool = False`, `capacity_floor: int | None = None` params to `train()`. After model build (fresh AND resume paths), before `model.learn`: call `enforce_capacity(model, strict=strict_capacity, thresholds=...)` (floor override applied when `capacity_floor` given) and append `HealthCallback` to `callbacks`. Wrap the `enforce_capacity`+`learn` region so `venv.close()` runs on `CapacityAbort` (finally/except) — no leaked envs on in-process abort. `smoke_train` stays warn-only, never strict (no config, non-interactive CI).

**MOD `src/drone_fly/config.py`** — add `_Spec("strict_capacity", (bool,), default=False)` and `_Spec("capacity_floor", (int,))` to `TrainRunConfig` specs + the two dataclass fields. (Config keys ONLY — the `train` subparser is config-YAML-only; NO argparse flag is added. AC5's "flag / config key" satisfied by the config key.)

**MOD `src/drone_fly/cli/__init__.py`** — `_run_train` threads `strict_capacity=cfg.strict_capacity, capacity_floor=cfg.capacity_floor` into `train()`; `main()` catches `CapacityAbort` → one-line guidance + exit code 3 (2 is ConfigError), mirroring the ConfigError convention (no stack trace).

## Files Affected
**Production code (developer):**
- `src/drone_fly/train/health.py` (new)
- `src/drone_fly/train/capacity_guard.py` (new)
- `src/drone_fly/train/health_callback.py` (new)
- `src/drone_fly/train/loop.py` (mod)
- `src/drone_fly/config.py` (mod)
- `src/drone_fly/cli/__init__.py` (mod)

**Test code (qa):**
- `tests/test_training_health.py` (new) — pure engine: every rule with a triggering AND non-triggering synthetic history; under-capacity signature positive + negative (falling entropy / rising success suppress it); warm-up guard; verdict shape; status precedence critical>warning>normal. (AC1/AC2/AC3)
- `tests/test_capacity_guard.py` (new) — floor boundary; **calibration on committed fixture**: load `tests/fixtures/mcns_fixture.npz`, prune k=0 and k=2, build actor for each, assert `k0_params < DEFAULT_CAPACITY_FLOOR <= k2_params` with `DEFAULT_PRUNE_K == 2`; TTY confirm/decline via monkeypatched isatty+input; non-interactive warn-continue; strict abort in both modes; `capacity_floor` override; venv closed on abort. (AC4/AC5)
- `tests/test_health_callback.py` (new) — callback self-accounts success/reward; **first-rollout/missing-`train/*`-keys → `normal`/"insufficient history", no crash**; ≥2-rollout populated verdict; invokes injected `on_verdict`; log cadence; never crashes; `smoke_train` stays green non-interactive. (AC6/AC7)

## Risks & Considerations
- **Floor calibration** — `DEFAULT_CAPACITY_FLOOR` must be set from the MEASURED k=0/k=2 fixture actor param counts (developer measures during implementation), not a guessed number; the calibration test enforces the ordering.
- **success_rate source** — self-accounted from `info["is_success"]`, NOT a Monitor wrapper (keeps env stack untouched; AC7). Rejected the Monitor alternative deliberately.
- **entropy proxy** — `train/std` (Gaussian action std) stands in for policy entropy (SB3 logs no raw entropy for continuous actions); rule language matches the UC's "std barely shrinking / entropy pinned high".
- **SB3 loop ordering** — `_on_rollout_end` fires inside `collect_rollouts`, BEFORE that iteration's `PPO.train()`, so first-rollout `train/*` keys are absent and later reads carry a stable one-iteration lag; handled by warm-up-first + `.get` defaults.
- **Prompt vs future UC-22 Live** — guard runs before `model.learn` (before any TUI takes the screen), safe now; documented for UC-22.
- **No new dependency; no change to env dynamics, observation schema, reward, connectome graph, or checkpoints (no retrain).** All new reads are read-only. (AC7)

## Challenger verdict: APPROVED (one revision round)
Verified against loop.py, config.py, cli/__init__.py, sb3.py, prune.py, record_callback.py, racing_env.py, and test fixtures. Key correctness points confirmed: no Monitor → self-accounting success/reward required and AC7-safe; SB3 fires `_on_rollout_end` before `PPO.train()` so warm-up guard short-circuits before reading possibly-absent `train/*`; `n_updates` from a callback-owned counter; capacity floor calibrated against committed `mcns_fixture.npz` at k=0/k=2 with `DEFAULT_PRUNE_K==2`. No new deps; no env/obs/reward/connectome/checkpoint changes.
