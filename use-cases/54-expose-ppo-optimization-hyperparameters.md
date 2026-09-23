# Use Case 54: Expose PPO optimization hyperparameters in the train YAML

## Summary
The PPO optimization hyperparameters `n_epochs` (default 10), `batch_size` (64), `n_steps` (2048), and `learning_rate` (3e-4) exist only as hardcoded `TrainConfig` dataclass defaults (`src/drone_fly/train/config.py`) and are **not reachable from `drone-fly train --config`** — the YAML resolver whitelist in `TrainRunConfig.from_mapping` (`src/drone_fly/config.py`) omits them. UC-52's collect/optimize timing split revealed the optimize phase is ~91% of each iteration's wall-clock (e.g. `o 4m45s` vs `c 28s`), driven by `n_epochs × ceil(n_steps·n_envs / batch_size)` gradient steps on the ~25.6k-neuron actor with a GPU-starving `batch_size=64`. The two highest-leverage speed levers (`n_epochs` down; `batch_size` up) therefore currently require a code edit to tune. This use case exposes those four hyperparameters in the validated train YAML and wires them through the CLI into `TrainConfig`, reusing the exact None-sentinel overrides mechanism UC-51 added (`_run_train`, `cli/__init__.py:372-397`). **Defaults are unchanged** — omitting a key is byte-identical to today (10 / 64 / 2048 / 3e-4). Scope is confined to the config surface + tests; no training-math, curriculum, reward, env, or TUI changes.

## Acceptance Criteria
1. The train YAML accepts optional keys `n_epochs`, `batch_size`, `n_steps` (integers) and `learning_rate` (float), validated by `TrainRunConfig.from_mapping` with type + range checks (`n_epochs`, `batch_size`, `n_steps` each `>= 1`; `learning_rate > 0`), each rejected with a clear `ConfigError` on a bad value.
2. Each key is optional (None-sentinel, no `_Spec` default) — omitting it leaves the corresponding `TrainConfig` field at its dataclass default.
3. Each key is wired end-to-end: a value set in the YAML reaches the matching `TrainConfig` field consumed by PPO construction in `loop.py` (verified without a GPU/sim). The four map 1:1 to `TrainConfig.n_epochs` / `.batch_size` / `.n_steps` / `.learning_rate`.
4. Set-to-default == omit: constructing the run with any of these keys set to their current defaults (10 / 64 / 2048 / 3e-4) yields a `TrainConfig` byte-identical to omitting them.
5. Default-parity: a bare config (none of the four set) reproduces the existing defaults `n_epochs=10, batch_size=64, n_steps=2048, learning_rate=3e-4`.
6. `configs/train/example.yaml` documents the four new keys with their defaults in the "Optional keys" block, with a one-line note that they trade training speed vs. sample efficiency / VRAM (larger `batch_size` uses more VRAM; lower `n_epochs` is faster but fewer passes) and that `batch_size` need not divide `n_steps·n_envs` (SB3 uses a partial final minibatch).
7. Scope is confined to `src/drone_fly/config.py`, `src/drone_fly/cli/__init__.py`, `configs/train/example.yaml`, and `tests/**`. No changes to `train/config.py` **defaults** (only its fields are already there), `loop.py`, curriculum modules, reward, env, adapters, or the TUI. (If the CLI wiring needs the overrides dict extended, that's within `cli/__init__.py`.)

## Potential Pitfalls & Open Questions
- **1:1 mapping (no rename):** unlike UC-51's `collision_penalty_warmup_fraction → collision_curriculum_warmup_fraction`, all four names match their `TrainConfig` fields exactly — add them straight into the existing non-None overrides dict in `_run_train`.
- **`batch_size` vs buffer size:** SB3 PPO warns (does not error) when `n_steps·n_envs` isn't divisible by `batch_size`; do NOT add a hard divisibility check — a partial final minibatch is expected and already handled (UC-52's `M = ceil(...)`).
- **`n_steps` interactions (note, not blockers):** `n_steps` changes the rollout buffer size (`n_steps·n_envs`) and therefore the iteration count (`total_timesteps / (n_steps·n_envs)`) and the UC-52 minibatch total; the UC-51 curriculum schedule is a fraction of `total_timesteps`, so it is unaffected. Worth a doc line but no special handling.
- **Validation placement:** mirror UC-51 — add `_Spec` entries (types) plus explicit `is not None`-guarded range checks raising `ConfigError`, consistent with the existing `n_envs >= 1` check.
- **Defaults unchanged:** this UC must NOT change any default value; it only surfaces the knobs. A behavioral change (e.g. shipping `n_epochs=5`) would be a separate decision.

## Original Description
Expose n_epochs, batch_size, n_steps, learning_rate in the train YAML so the optimize-phase speed/quality trade-off (surfaced by UC-52's collect/optimize split — optimize is ~91% of wall-clock, batch_size=64 starves the GPU) can be tuned from --config without code edits. Same exposure pattern as UC-51 (None-sentinel _Spec entries + validation + non-None overrides dict → TrainConfig). Keep defaults untouched (10/64/2048/3e-4) so it's a no-op until set; then the user can try n_epochs: 5 (free ~45% speedup) and cautiously batch_size: 128/256 (watch 8 GB VRAM OOM). Hermetic tests only.

## Clarifications
- Q: Change any defaults?
  A: No — expose only; defaults stay 10/64/2048/3e-4 (omit == today). Tuning is the user's per-run choice.
- Q: Team or direct?
  A: Dev-team, autonomous through merge (same as UC-51/52/53).
