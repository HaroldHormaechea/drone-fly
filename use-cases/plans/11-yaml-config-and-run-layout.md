---
plan_for: use-cases/11-yaml-config-and-run-layout.md
work_branch: feat/uc-11-yaml-config-and-run-layout
team: drone-fly-uc-11
approved: 2026-09-18
---

# UC-11 — YAML `--config` for the CLIs + per-run `training/<name>/` output layout

Challenger approved after one revision round. All paths inside TARGET_DIR `/workspace/drone-fly-uc-11-yaml-config-and-run-layout`.

## Analysis
The `drone-fly` CLI (`src/drone_fly/cli/__init__.py`) is a single argparse parser with subcommands train / smoke-train / evaluate / prune / prune-trained / fetch-connectome, each carrying per-setting flags. Output paths are flat today: `TrainConfig.models_dir="artifacts/models"`, `logs_dir="artifacts/logs"` (`src/drone_fly/train/config.py`), recorder default `artifacts/activations` (`src/drone_fly/record/recorder.py:48`). Resume (PR #12) lives in `_resolve_resume`/`find_latest_checkpoint` in `train/loop.py` and resolves against `cfg.models_dir`. `.gitignore` ignores `artifacts/**`. **No YAML library is installed** (pyyaml absent from pyproject + uv.lock). **UC-10's `clean` command does not exist yet** (UC-10 pending), so AC8's clean clause is a documented forward dependency, not actionable now.

## Proposed Solution
**1. New `src/drone_fly/config.py`** — `ConfigError(ValueError)`; `load_yaml(path)` (clear error on missing/unreadable file; `yaml.safe_load` with `YAMLError` wrapped into `ConfigError` naming the file; requires a top-level mapping — AC6); four per-command dataclasses `TrainRunConfig`/`EvaluateRunConfig`/`PruneRunConfig`/`PruneTrainedRunConfig`, each with `from_mapping()` that validates required keys, rejects unknown keys, and type-checks, raising `ConfigError` naming command+key (AC6); `run_layout(name)` → checkpoints/logs/recordings dirs under `training/<name>/`; `validate_run_name(name)` (required, non-empty, regex `^[A-Za-z0-9._-]+$`, not `.`/`..` — prevents empty name and path escape; AC3 "no artifacts/models fallback").

Per-command config keys (faithful 1:1 flag→key migration, AC1):
- train: `name`(REQUIRED), `connectome`, `adapter`, `device`, `timesteps`, `n_envs`, `resume`, `prune`, `prune_k`, `record`, `record_every`, `record_dir`, `randomize`, `randomize_dynamics`.
- evaluate: `checkpoint`(REQUIRED), `vecnormalize`, `episodes`, `seed`, `device`, `adapter`, `connectome`, `record`, `record_every`, `record_dir`, `randomize`, `randomize_dynamics`, optional `name`.
- prune: `connectome`(REQUIRED), `out`(REQUIRED), `prune_k`, `prune_rule`.
- prune-trained: `checkpoint`(REQUIRED), `out`(REQUIRED), `connectome`, `vecnormalize`, `prune`, `prune_k`, `metric`, `threshold`, `threshold_mode`, `episodes`, `eps`, `finetune_steps`, `adapter`, `device`, `seed`, `randomize`, `randomize_dynamics`, optional `name`.

**2. Per-command default parity (AC7).** Omitted keys inherit the EXISTING per-command argparse/constant default — never a shared default set. Locked traps: `adapter` = `"auto"` for train/evaluate but **`"simple"` for prune-trained**; `seed`=0 (evaluate, prune-trained); prune-trained numerics imported from `prune_trained.{importance,measure,workflow}` (`DEFAULT_METRIC`/`DEFAULT_THRESHOLD`/`ABSOLUTE_MODE`/`DEFAULT_EPISODES`/`DEFAULT_EPS`/`DEFAULT_FINETUNE_STEPS`); prune from `connectome.prune` (`DEFAULT_PRUNE_K`/`DEFAULT_PRUNE_RULE`); train `record_every` omitted→`None` (keeps the "set without record" warning detectable), resolved to 1 at dispatch as `main()` does today; `n_envs` omitted→`None` sentinel.

**3. Run-layout wiring (AC2).** For train, the CLI builds `TrainConfig(models_dir="training/<name>/checkpoints", logs_dir="training/<name>/logs")` and passes `record_dir="training/<name>/recordings"` (unless config sets explicit `record_dir`). `train()`/`_resolve_resume`/`find_latest_checkpoint` are UNCHANGED — they already key off `cfg.models_dir`, so `resume: latest` auto-resolves under `training/<name>/checkpoints/` for free (AC5), preserving behaviour parity (AC7). `TrainConfig` defaults stay `artifacts/…` so smoke-train is byte-identical. For evaluate/prune-trained: when optional `name` present + record on + no explicit `record_dir`, recordings default to `training/<name>/recordings/`; else keep `artifacts/activations` (parity).

**4. Resume in config (AC5).** `resume`: omitted/`null`→fresh; `latest`→PR#12 auto-latest (newest under checkpoints dir, hard error if none); `auto`→newest-if-exists-else-fresh, no error (idempotent bootstrap for train.sh); explicit `<path.zip>`→passthrough. One unit test per value.

**5. CLI refactor (`cli/__init__.py`).** train/evaluate/prune/prune-trained take only `--config <path>`; all per-setting flags removed (AC1). `main()` loads+validates config and dispatches to the existing `train()`/`evaluate_checkpoint()`/`_run_prune_export`/`prune_trained()` unchanged. `ConfigError`→clean one-line message, exit code 2 (no stack trace, AC6). Preserve both guards: `n_envs<1`→ConfigError ("must be >= 1"); `record_every` set without `record`→existing warning (logger `drone_fly.cli`). smoke-train + fetch-connectome keep current surface (not in AC1's four commands).

**6. YAML dep.** Add `pyyaml` to `[project].dependencies` (CORE, since CI installs `--extra dev` only and the CLI imports yaml unconditionally); regenerate + commit `uv.lock` via `uv lock`.

**7. Example YAMLs (AC4), committed under `configs/` (outside gitignored `training/`):**
- `configs/train/example.yaml` — runnable as-is: `name: baseline`, `timesteps: 2000` (smoke value; comment to raise for a real run), `adapter: simple`, `connectome: tests/fixtures`, `resume: auto`.
- `configs/prune/k0.yaml`, `k1.yaml`, `k2.yaml` — `connectome: tests/fixtures`, `prune_k: 0|1|2`, `out: artifacts/pruned/k0|k1|k2` (gitignored under `artifacts/**`, so running them never dirties the tree).
- `configs/evaluate/example.yaml`, `configs/prune-trained/example.yaml` — for the docs.

**8. `.gitignore`** — add `training/**` (+ `!training/**/.gitkeep`), mirroring the `artifacts/**` pattern.

**9. `scripts/train.sh`** — config-driven: `CONFIG="${1:-configs/train/example.yaml}"`, drop the bash-side `find_latest_checkpoint` probe and `--resume` injection (idempotent resume now comes from `resume: auto`), `exec … drone-fly train --config "$CONFIG"`; keep the venv/sim bootstrap intact.

**10. `README.md`** — migrate every train/evaluate/prune/prune-trained example to `--config`; document the four schemas, the `training/<name>/{checkpoints,logs,recordings}/` layout (replacing the `artifacts/models|logs|activations` docs), the shipped example YAMLs, and an explicit note that UC-10's `clean` must target `training/<name>/` when built (AC8 forward dependency).

## Files Affected
**Production code (developer):**
- `src/drone_fly/config.py` — NEW (schema dataclasses, loader, validation, run-layout + name helpers, ConfigError).
- `src/drone_fly/cli/__init__.py` — MODIFY (flags → `--config`; config-driven dispatch; ConfigError→exit 2; preserve both guards).
- `pyproject.toml` — MODIFY (add pyyaml to core deps) + regenerate `uv.lock`.
- `.gitignore` — MODIFY (`training/**`).
- `scripts/train.sh` — MODIFY (config-driven; drop bash resume probe).
- `configs/train/example.yaml`, `configs/prune/k0.yaml`, `configs/prune/k1.yaml`, `configs/prune/k2.yaml`, `configs/evaluate/example.yaml`, `configs/prune-trained/example.yaml` — NEW.
- `README.md` — MODIFY (config workflow + layout + example YAMLs + AC8 deferral note).
- `src/drone_fly/train/config.py` — NO change (defaults stay `artifacts/…` for smoke-train parity; CLI overrides per run).

**Test code (qa):**
- `tests/test_cli.py` — MODIFY: removed-flag parsing/dispatch tests (train flags, prune flags, randomize flags, n-envs, resume-flag parsing) will break; rewrite to `--config` parsing + dispatch + validation-error cases.
- `tests/test_record_cli.py` — MODIFY: `--record*` flag tests + `evaluate --record` dispatch will break; migrate to config-driven recording.
- `tests/test_config.py` — NEW: required/empty/invalid `name` (AC3); malformed YAML / missing-required / unknown-key (AC6); each `resume` value (AC5); run-layout paths (AC2); per-command parity defaults incl. the adapter `simple`/`auto` split (AC7); both preserved guards.
- Audit `tests/test_train_resume.py`, `tests/test_evaluate.py`, `tests/test_prune_trained.py`, `tests/test_record_backcompat.py` — call stage functions directly, expected unaffected; confirm no CLI-flag/`artifacts/models` coupling.

## Risks & Considerations
- **AC8/UC-10 is unbuilt** — no `clean` command exists; the new `training/<name>/` layout is built now and left for UC-10 to consume, documented in README + this plan as a named forward dependency (not a silent gap).
- **Breaking change (AC-wide)** — old flag invocations stop working immediately; mitigated by shipping example configs + train.sh + README + all migrated tests in the same change (AC8).
- **pyyaml is a new core runtime dependency** — pure-Python, low risk, but requires committing a regenerated `uv.lock`; touches CI hermeticity.
- **`resume: auto`** extends AC5's listed values ("e.g.") to preserve train.sh's idempotent bootstrap without bash-parsing YAML.
- **Example configs' hermeticity** relies on `tests/fixtures` being a loadable connectome (it is, used throughout the hermetic tests); prune examples write only into gitignored `artifacts/pruned/`.

## Challenger verdict
**APPROVED** (round 2). Round 1 raised two Major items — (1) per-command default parity for AC7 (the prune-trained `adapter="simple"` vs train/evaluate `"auto"` trap; seed/prune-trained numeric defaults from existing constants), and (2) `hyperparameters:` scope creep (cut) — plus four minors. All six accepted; fixes verified against the actual code. All 8 ACs covered. Ready for developer/QA.
