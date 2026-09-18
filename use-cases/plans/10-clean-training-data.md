---
plan_for: use-cases/10-clean-training-data.md
work_branch: feat/uc-10-clean-training-data
team: drone-fly-uc-10
approved: 2026-09-18
---

# UC-10 — `drone-fly clean` (wipe training outputs, dry-run default)

Challenger approved (first pass, two minor notes folded in). TARGET_DIR `/workspace/drone-fly-uc-10-clean-training-data`. Prose only.

## Analysis
UC-11 merged, so: (a) the CLI is config-based (`src/drone_fly/cli/__init__.py`) — `build_parser()` registers subcommands via `sub.add_parser`, `main(argv)` dispatches on `args.command`, `ConfigError`→one-line log + exit 2; `clean` needs no YAML so it follows the `smoke-train`/`fetch-connectome` inline-flag pattern. (b) The per-run layout `training/<name>/{checkpoints,logs,recordings}` exists via `run_layout` in `src/drone_fly/config.py`.

**Source-of-truth output locations (import, never hardcode):** `artifacts/models`=`TrainConfig.models_dir`, `artifacts/logs`=`TrainConfig.logs_dir` (`src/drone_fly/train/config.py`), `artifacts/activations`=`DEFAULT_RECORD_DIR` (`src/drone_fly/record/recorder.py`), `training/` root=literal in `run_layout` — extract a new `TRAINING_ROOT="training"` constant in `config.py` so `clean` shares it.

**Prune-slice reconciliation:** AC4 text says `data/pruned*` but the real `drone-fly prune` writes to a user-specified `out`, and every committed example (`configs/prune/k{0,1,2}.yaml`) uses `out: artifacts/pruned/k*`. Resolution (challenger-approved): `--include-prunes` removes `pruned*` globs under BOTH `artifacts/` and `data/`. Raw connectome `data/connectome` (`DEFAULT_CONNECTOME_DIR`) never matches `pruned*` → AC5 holds. A code comment will note the AC-text-vs-real-convention reconciliation.

## Proposed Solution
1. **`src/drone_fly/config.py`** (MODIFY, developer): add module-level `TRAINING_ROOT = "training"`; `run_layout` uses `os.path.join(TRAINING_ROOT, name)`. Byte-identical join → existing `test_config.py`/`test_cli.py` assertions unaffected.

2. **`src/drone_fly/clean/__init__.py`** (CREATE, developer): new stage subpackage mirroring `train/`, `evaluate/`, `record/`.
   - Imports source-of-truth constants (no hardcoded output strings). **Exposes its target roots as importable values** so QA can assert they equal `TrainConfig().models_dir` / `.logs_dir` / `DEFAULT_RECORD_DIR` / `TRAINING_ROOT` (guards against future re-hardcoding).
   - Frozen `CleanReport` dataclass: list of removed / would-be-removed paths, dry-run flag, human summary string.
   - Discovery function `(root, include_prunes)` → ordered target list: the **children** of `artifacts/models`, `artifacts/logs`, `artifacts/activations`, and `training/` (each root dir itself preserved; `.gitkeep` children skipped); when `include_prunes`, also glob-matches `artifacts/pruned*` + `data/pruned*`. Non-existent roots contribute nothing (AC7).
   - **Confinement guard (AC6):** every candidate resolved and asserted under `root.resolve()`; escapes skipped with a warning. `root` is always CWD; targets are literal relative joins — no user-supplied paths.
   - Execute function `(delete: bool)`: dry-run removes nothing, returns would-remove report; delete mode removes each target and returns removed report; always succeeds (exit 0), empty→"nothing to clean". **Per-target removal branch (folded from challenger #2):** `os.path.islink(p) or p.is_file()` → `unlink`; real directory → `shutil.rmtree`; already-missing tolerated. This avoids `rmtree`-on-symlink `OSError` that would break the exit-0/partial-delete guarantee.

3. **`src/drone_fly/cli/__init__.py`** (MODIFY, developer): add `clean` subparser with `--yes`, `--force` (either triggers deletion), `--dry-run`, `--include-prunes`; add `_run_clean(args)` branch in `main`. **Effective delete = `(args.yes or args.force) and not args.dry_run`** (AC8: dry-run wins). `_run_clean` uses `root = Path.cwd()`, runs discovery + execute, prints per-path lines + summary, returns 0. Update module docstring: list `clean`, and note (folded from challenger #3) that `clean` operates relative to CWD like the other commands and that the dry-run default protects a wrong-CWD invocation.

## Files Affected
**Production code (developer):**
- `/workspace/drone-fly-uc-10-clean-training-data/src/drone_fly/config.py` (MODIFY — extract `TRAINING_ROOT`)
- `/workspace/drone-fly-uc-10-clean-training-data/src/drone_fly/clean/__init__.py` (CREATE — clean stage logic)
- `/workspace/drone-fly-uc-10-clean-training-data/src/drone_fly/cli/__init__.py` (MODIFY — `clean` subparser + `_run_clean`)

**Test code (qa):**
- `/workspace/drone-fly-uc-10-clean-training-data/tests/test_clean.py` (CREATE) — unit tests with `tmp_path` root: AC1 dry-run default lists, deletes nothing; AC2 delete removes + report/counts; AC3 default set = children of models/logs/activations + `training/<name>/` folders; AC4 `--include-prunes` removes `artifacts/pruned*` + `data/pruned*`, default leaves them + other `data/` intact; AC5 `src/`,`viz/`,`tests/`,`use-cases/`,`scripts/`,`PROJECT_BRIEF.md`,`USE_CASES.md`,`data/connectome/` survive a delete; AC6 out-of-root target refused; AC7 missing dirs → "nothing to clean", no error; `.gitkeep` preserved; **symlink child removed cleanly without OSError** (challenger #2); source-of-truth assertion (roots equal the imported constants).
- `/workspace/drone-fly-uc-10-clean-training-data/tests/test_cli.py` (MODIFY) — parser accepts `clean [--yes|--force|--dry-run|--include-prunes]`; `main(["clean"])` in chdir'd tmp tree = dry-run (exit 0, nothing deleted); `main(["clean","--yes"])` deletes (exit 0); `main(["clean","--yes","--dry-run"])` deletes nothing (AC8); exit 0 nothing-to-clean (AC7).

## Risks & Considerations
- **AC4 both-locations** (`artifacts/pruned*` + `data/pruned*`): approved reading of intent; add reconciliation comment.
- **Symlink children:** handled via islink/is_file→unlink vs rmtree branch (challenger #2) — must be built + tested.
- **CWD-as-root:** consistent with train/evaluate output resolution; confinement guard prevents escape; dry-run default mitigates wrong-CWD; documented in help/docstring (challenger #3).
- **Delete granularity:** child-level (preserve root dir + `.gitkeep`), whole `training/<name>/` folders.
- **No ConfigError path** for `clean` → never exit 2; always 0 per AC1/AC2/AC7.

## Challenger verdict
**APPROVED** (first pass). Safety verified on every flagged point: dry-run default (AC1), protected-path survival + `data/connectome` can't match `pruned*` (AC5), tree confinement via resolve+assert-under-root (AC6), `--include-prunes` scoped to `pruned*` only (AC4), UC-11 `training/<name>/` layout wiped from source-of-truth not hardcoded (AC3), `--yes --dry-run`→dry-run wins (AC8), graceful nothing-to-clean exit 0 (AC2/AC7). Two minor notes folded into the plan (unlink symlinked children vs rmtree; CWD-as-root documented). No scope creep.
