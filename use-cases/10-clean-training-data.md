# Use Case 10: Clean all training data & models (start-from-scratch)

## Summary
Add a `drone-fly clean` CLI subcommand that safely wipes training *outputs* so the user can restart from scratch. By default it does a **dry-run** — lists exactly what would be removed and deletes nothing; an explicit `--yes` (or `--force`) performs the deletion. It removes all training outputs: `artifacts/models/` (checkpoint `.zip`s + VecNormalize `.pkl`s), `artifacts/logs/`, `artifacts/activations/`, and any per-run `training/<name>/` trees (the UC-11 layout). A `--include-prunes` flag additionally clears the prepared prune slices under `data/pruned*` for a total reset; without it, those slices (and all other inputs) are preserved. It is a **full wipe** (no per-run/per-category targeting). It never touches `src/`, `viz/`, `tests/`, `use-cases/`, `scripts/`, `PROJECT_BRIEF.md`, `USE_CASES.md`, or the raw connectome source, and never operates outside `TARGET_DIR`. It prints a summary of what was (or would be) removed and exits 0 even when there's nothing to clean.

## Acceptance Criteria
1. `drone-fly clean` with no flags performs a **dry-run**: it lists every path that would be removed and deletes nothing (exit 0).
2. `drone-fly clean --yes` (or `--force`) actually deletes the training outputs and prints a summary (paths/counts removed, or "nothing to clean").
3. The default target set is all training outputs: everything under `artifacts/models/`, `artifacts/logs/`, `artifacts/activations/`, and any `training/<name>/` run folders.
4. `--include-prunes` additionally removes the prepared prune slices (`data/pruned*`); **without** it, `data/pruned*` and all other `data/` contents are left intact.
5. It never deletes or modifies `src/`, `viz/`, `tests/`, `use-cases/`, `scripts/`, `PROJECT_BRIEF.md`, `USE_CASES.md`, or the raw/base connectome under `data/` — verifiable by asserting those paths survive a `--yes` run.
6. All operations are confined within `TARGET_DIR`; it never acts on absolute or out-of-tree paths.
7. Missing output locations are handled gracefully (e.g. no `training/` yet) — it reports "nothing to clean" rather than erroring.
8. `--yes` combined with `--dry-run` (if both are accepted) resolves safely — dry-run wins, nothing is deleted; documented behaviour, not a crash.

## Potential Pitfalls & Open Questions
- **Edge case** — the `training/<name>/` layout doesn't exist until UC-11; the cleaner must tolerate its absence now and pick it up once it lands.
- **Assumption** — output locations are the brief's defaults (`artifacts/models`, `artifacts/logs`, `artifacts/activations`); if UC-11 makes these configurable, `clean` should read the same config/source of truth rather than hardcoding them.
- **Assumption** — `--include-prunes` targets only `data/pruned*` (the generated slices), not the raw connectome download; the raw source is preserved so a re-prune doesn't require re-fetching.

## Original Description
Create a way to clean ALL training data and prepared models so the user can start training from scratch. It should remove the training artifacts — checkpoint/step `.zip` files, the matching VecNormalize `.pkl` stats, training logs, activation recordings (`artifacts/activations/` and any per-run recording folders), and any per-run `training/<name>/` output folders — WITHOUT touching source code, the connectome data, prepared prune slices, use cases, or the brief. It should be safe: confirm before deleting (or support a dry-run / --yes), report what it removed, and be scoped only to the artifact/training-output locations. Likely a script under scripts/ and/or a CLI subcommand.

## Clarifications
- Q: What should "clean prepared models" include — how aggressive is the wipe?
  A: Both, via a flag — default removes training outputs only; `--include-prunes` also clears the pruned connectome slices (`data/pruned*`) for a total reset.
- Q: How should the cleanup be invoked?
  A: A `drone-fly clean` CLI subcommand (consistent with train/evaluate/prune).
- Q: What's the default safety behaviour when run with no flags?
  A: Dry-run by default — lists what would be removed, deletes nothing; requires an explicit `--yes`/`--force` to delete.
- Q: Selective cleaning or full wipe?
  A: Full wipe only (no per-run `--name` or per-category targeting).
