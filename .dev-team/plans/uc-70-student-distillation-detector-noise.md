---
plan_for: "(free-form task) UC70 test-suite follow-up"
work_branch: feat/uc-70-student-distillation-detector-noise
team: drone-fly
approved: 2026-10-14
---

# FINAL APPROVED PROPOSAL — UC70 test-suite follow-up (test-only)

Branch: `feat/uc-70-student-distillation-detector-noise`. Both changes land under `tests/**` (QA write-scope per brief frontmatter `paths.test: ["tests/**"]`). Zero `src/` changes.

## Analysis

**Part 1 — root cause CONFIRMED by reproduction.** The two fast failing tests fail with `FileNotFoundError` on `repo/artifacts/sysid/.../meta.json`. The sysid CLI writes the session under `paths.sysid_root()` (`src/drone_fly/sysid/session.py:959`: `root = Path(out_root) if out_root is not None else paths.sysid_root()`). `sysid_root()` = `data_root()/sysid` (`src/drone_fly/paths.py:70-71`). `data_root()` precedence (`paths.py:32-43`): `$DRONE_FLY_DATA_ROOT` (stripped) → `find_repo_root()/artifacts/harness/liftoff/training-data`. The `roots` fixture (`tests/test_sysid_cli.py:17-25`) only monkeypatches `find_repo_root → repo` and sets `DRONE_FLY_DRONES_DIR`; it does **not** set `DRONE_FLY_DATA_ROOT`. So the session lands at `repo/artifacts/harness/liftoff/training-data/sysid/...`, not the asserted `repo/artifacts/sysid/...`. Three tests assert the old path: `test_drone_mismatch_exits_5_and_is_resumable` (line 80), `test_existing_session_without_resume_is_exit_2` (second `main()` call, line 88), `test_synthetic_session_then_fit_end_to_end` (line 118, `@pytest.mark.slow`).

**Part 2 — helpers verified in `src/drone_fly/training/distill.py:94-158`.** All read `os.environ` at call time and do a lazy `from drone_fly import paths`, so `monkeypatch`+`tmp_path` works exactly like `tests/test_paths.py`:
- `default_teacher(explicit)` (110-117): explicit → `$DRONE_FLY_TEACHER` (`TEACHER_ENV`) → `models_root("teacher")/DEFAULT_TEACHER_RUN` only if `.is_dir()`, else None.
- `default_noise_model(explicit, configured)` (120-129): explicit → configured → `$DRONE_FLY_NOISE_MODEL` (`NOISE_MODEL_ENV`) → `models_root("detector-v1")/liftoff-gates-161f7a83/noise_model.yaml` only if `.is_file()`, else None.
- `student_tracks_root(explicit)` (132-140): explicit → if `$DRONE_FLY_TRACKDATA` (`drone_fly.tracks.catalogue.TRACKDATA_ENV`) set ⇒ returns None → `track_library_root()/trackdata-uc68` if `.is_dir()` → None.
- `resolve_teacher_path(recorded)` (143-158): recorded exists ⇒ `str(recorded)` as-is; missing + a `checkpoints` segment ⇒ remap `models_root("teacher")/<run-folder>/<checkpoints/...>` (run-folder = part before `checkpoints`) if it exists else original; missing + no `checkpoints` ⇒ bare-name `models_root("teacher")/p.name` if exists else original.

**k1 resolver — correction to the task brief.** There is **no k1 default in `distill.py`** (`cfg.k1_path` defaults to None, `training/config.py:123`; distill only threads the param). The real resolver is `drone_fly.harness.contract.resolve_k1_path(explicit=None)` (`src/drone_fly/harness/contract.py:130-154`). Precedence: explicit/config → `$DRONE_FLY_K1_PATH` (`K1_PATH_ENV="DRONE_FLY_K1_PATH"`, contract.py:40; **must exist or raises `HarnessConfigError`**) → `paths.pruned_root()/"k1"` (= `data_root()/pruned/k1`) if exists → legacy `find_repo_root()/artifacts/pruned/k1` if exists → None.

**Existing coverage:** none for these distill path helpers. `tests/test_paths.py` is the style reference (env-only data root + `delenv(..., raising=False)`).

## Proposed Solution

**File 1 (MODIFY) — `/workspace/drone-fly/tests/test_sysid_cli.py`**, `roots` fixture (lines 17-25). Add one line after `repo.mkdir()`:
`monkeypatch.setenv("DRONE_FLY_DATA_ROOT", str(repo / "artifacts"))`
Makes `sysid_root() == repo/artifacts/sysid`, satisfying all three failing assertions with **zero assertion edits**, and deterministically overrides any leaking ambient var (auto-restored at teardown — no `delenv` needed). Keep the existing `find_repo_root` patch (session.py and the drones path still consult it). Fixture return `(repo, droot)` unchanged.

**File 2 (CREATE) — `/workspace/drone-fly/tests/test_distill_paths.py`**. Pure-pathlib precedence tests mirroring `test_paths.py`. Shared fixture: `monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path/"data"))`, `monkeypatch.setattr(config, "find_repo_root", …)` to neutralize the legacy k1 branch, and `delenv` of `DRONE_FLY_TEACHER`, `DRONE_FLY_NOISE_MODEL`, `DRONE_FLY_TRACKDATA`, `DRONE_FLY_K1_PATH` (`raising=False`). Create dirs/files under the tmp data root per rung. Tests:
- `default_teacher`: explicit wins (no existence needed) → env over default → default returns path only when `models/teacher/<run>` dir created, else None.
- `default_noise_model`: explicit → configured → env → default file created ⇒ returned, else None.
- `student_tracks_root`: explicit wins → `$DRONE_FLY_TRACKDATA` set ⇒ None → default `track-library/trackdata-uc68` dir present ⇒ its path, else None.
- `resolve_teacher_path`: (a) recorded exists ⇒ as-is; (b) missing + `checkpoints` segment + remap target created ⇒ remapped; (c) **[challenger Minor #1]** missing + `checkpoints` segment + remap target NOT created ⇒ original recorded string (covers `else str(p)` at line 154); (d) missing + no `checkpoints` + bare-name target created ⇒ remapped; (e) missing + nothing ⇒ original. Build recorded paths as `.../teacher/<run>/checkpoints/<file>` so `parts[i-1]` indexes the real run folder.
- `resolve_k1_path` (import from `drone_fly.harness.contract`): default resolves `data_root()/pruned/k1` when that dir created (assert `== (tmp/"data"/"pruned"/"k1").resolve()`, `.resolve()` both sides); None when nothing exists; **[challenger Minor #2]** `$DRONE_FLY_K1_PATH` pointing at a nonexistent path ⇒ asserts `HarnessConfigError`. Comment noting it's a harness helper surfaced here as the distillation K1 input.

## Files Affected
- **Test code (for QA):**
  - MODIFY `/workspace/drone-fly/tests/test_sysid_cli.py` (one line in the `roots` fixture)
  - CREATE `/workspace/drone-fly/tests/test_distill_paths.py`
- **Production code (for developer):** none — test-only task.

## Risks & Considerations
- `test_synthetic_session_then_fit_end_to_end` is `@pytest.mark.slow` — covered by the fixture fix, but run it explicitly (it may be deselected by default) to confirm the path change holds end-to-end.
- Env isolation via `delenv` prevents a developer's ambient vars from making "returns default/None" assertions flaky; `monkeypatch` auto-restores, so no cross-test bleed.
- k1 equality uses `.resolve()` on both sides (symlink-safe; low risk on this Linux sandbox).
- `resolve_teacher_path` remap builds from `Path(...).parts`; construct recorded paths with a real `checkpoints` segment to exercise the run-folder indexing, not just the bare-name fallback.
- No production changes ⇒ no `allowed-commands.yaml` additions. Verify with `/workspace/drone-fly/.venv/bin/python -m pytest` (uv-synced, cv2 available), running files one at a time.

## Challenger's Final Verdict: **APPROVE** (no revision round)
Independently verified every code claim — all correct. Excellent k1 catch — the task's guess was wrong; real resolver is `harness.contract.resolve_k1_path`. Two Minor suggestions folded in (resolve_teacher_path "remap target absent → original" branch; resolve_k1_path must-exist-or-raise). Scope clean (both files under `tests/**`). Proceed to QA.
