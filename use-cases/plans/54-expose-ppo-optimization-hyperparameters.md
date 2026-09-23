---
plan_for: use-cases/54-expose-ppo-optimization-hyperparameters.md
work_branch: feat/uc-54
team: drone-fly-uc-54
approved: 2026-09-23
---

# UC-54 — Expose 4 PPO optimization hyperparameters in the train YAML

APPROVED by challenger (first round; all 7 ACs verified against the worktree + SB3). Challenger's 4 non-blocking notes folded in.

## Summary
Expose `n_epochs`, `batch_size`, `n_steps` (int) and `learning_rate` (float) in the validated train YAML, mirroring UC-51's exposure pattern. Defaults UNCHANGED (10/64/2048/3e-4); omitting a key is byte-identical to today. Scope confined to the config surface + tests — no training-math/curriculum/reward/env/TUI changes.

## Analysis (verified against `/workspace/drone-fly-uc-54`)
- `TrainConfig` already declares all four as dataclass fields with the target defaults (`src/drone_fly/train/config.py:52-62`: `learning_rate=3e-4`, `n_steps=2048`, `batch_size=64`, `n_epochs=10`). This file is NOT edited (AC7).
- `TrainRunConfig` frozen dataclass + its `_Spec` whitelist in `from_mapping` omit all four → unreachable from `--config` today. This is the gap.
- PPO construction consumes `cfg.learning_rate/.n_steps/.batch_size/.n_epochs` on the real `train()` path (`loop.py:500-503`, `train()` at :293), which `_run_train` routes through (never `smoke_train`).
- `_run_train` (`cli/__init__.py:372-397`) builds `curriculum_overrides = {…}`, filters to non-None → `overrides`, then `TrainConfig(models_dir=…, logs_dir=…, **overrides)`. The four map 1:1 to `TrainConfig` field names (no rename) → clean extension of that same dict.
- `_validate`/`_check_type` already handle unknown-key rejection, explicit-`null`==omitted, and int-for-float acceptance — no helper changes needed.

## Proposed Solution (prose, no code)
1. **`src/drone_fly/config.py` — `TrainRunConfig`:** add four fields (`n_epochs`/`batch_size`/`n_steps`: `int | None`; `learning_rate`: `float | None`) with a short UC-54 comment (None = "leave TrainConfig default untouched"); add four `_Spec` entries in `from_mapping` with NO default (None sentinel): `_Spec("n_epochs",(int,))`, `_Spec("batch_size",(int,))`, `_Spec("n_steps",(int,))`, `_Spec("learning_rate",(float,))`; add four `is not None`-guarded range checks beside the existing `n_envs >= 1` check (post-`_validate`, pre-`_validate_curriculum`), each raising `ConfigError` naming the key: `n_epochs>=1`, `batch_size>=1`, `n_steps>=1`, `learning_rate>0`. NO divisibility check.
2. **`src/drone_fly/cli/__init__.py` — `_run_train`:** add the four keys (1:1 names) to the existing `curriculum_overrides` dict, so the existing non-None filter + `TrainConfig(**overrides)` carry them unchanged.
3. **`configs/train/example.yaml`:** document the four keys in the Optional-keys area (a small "PPO optimization" sub-block), each commented-out with its default + a one-line note: larger `batch_size` = more VRAM (watch 8 GB OOM), lower `n_epochs` = faster but fewer passes / less sample-efficient, `n_steps` changes rollout-buffer size (and iteration count), and `batch_size` need not divide `n_steps·n_envs` (SB3 uses a partial final minibatch — no hard check).

## Files Affected
**Production:** `src/drone_fly/config.py` (4 fields + 4 `_Spec` + 4 range checks); `src/drone_fly/cli/__init__.py` (4 keys into overrides dict).
**Config/docs (in-scope):** `configs/train/example.yaml`.
**Test:** `tests/test_config.py` (accept-all-four; None when omitted; explicit-null==omit; YAML round-trip; range rejection for ints at 0/negative and lr at 0/negative; type rejection bool-for-int and string-for-number — mirror the `_UC51_KNOBS` cluster). `tests/test_cli.py` (hermetic wiring via monkeypatch `drone_fly.train.loop.train`, assert four reach `TrainConfig`; set-to-default==omit **dataclass-equality** byte-identity; default-parity bare config → 10/64/2048/3e-4 — mirror `test_uc51_*`).

## AC mapping
- AC1 → `_Spec` + range checks (config.py); test_config.py accept/type/range
- AC2 → None-sentinel, no default (config.py); test_config.py None/null
- AC3 → overrides dict (cli); test_cli.py wiring into TrainConfig
- AC4 → non-None filter + dataclass `==` (set-to-default forwards the value, but `TrainConfig(n_epochs=10)==TrainConfig()`); test_cli.py byte-identity via dataclass equality
- AC5 → test_cli.py bare config → 10/64/2048/3e-4
- AC6 → configs/train/example.yaml docs + notes
- AC7 → scope confined (config.py, cli/__init__.py, example.yaml, tests/**); no train/config.py default / loop.py / curriculum / reward / env / TUI changes

## Risks & Considerations
- No behavioral change (defaults untouched; omit==today) — guarded by AC4/AC5 tests.
- No divisibility check for `batch_size` vs `n_steps·n_envs` (SB3 only warns; partial final minibatch expected — UC-52 `M=ceil(...)`).
- `smoke_train`'s `n_steps`/`batch_size` clamp (`loop.py:747-748`) is a non-issue — `--config` never reaches the smoke path; tests must not route through it.
- `learning_rate>0` strict (a 0 LR is a degenerate no-op); ints `>=1`.
- int-for-float `learning_rate: 3` accepted as-is (consistent with `ent_coef`); no coercion — out of scope.

## Challenger verdict
APPROVE — all 7 ACs verified against the worktree. Defaults preserved (train/config.py untouched); wiring target real (loop.py:500-503 on the train() path); reuses UC-51 None-sentinel + non-None overrides; 1:1 names; range checks beside n_envs; no divisibility check; scope confined; hermetic tests via the UC-51 monkeypatch pattern. AC4 byte-identity test asserts dataclass equality of the two constructed TrainConfigs (set-to-default IS forwarded, but equals the default).
