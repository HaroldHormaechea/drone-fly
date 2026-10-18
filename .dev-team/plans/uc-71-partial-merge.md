---
plan_for: use-cases/71-multi-map-gate-detector.md (partial delivery: merge AC1-6, AC9; AC7/AC8 open)
work_branch: feat/uc-71-multi-map-gate-detector
run: drone-fly-uc-71
approved: 2026-10-18
---

# UC-71 partial merge into main — approved plan

uc71-challenger approved analyst v1 in round 1, with 4 Minor notes (m1–m4) folded in.

## Analysis (verified read-only)
- `git merge-tree --write-tree origin/main HEAD` shows exactly 3 conflicts: USE_CASES.md, harness/contract.py, tests/test_harness_presets.py. The branch is 54 ahead / 153 behind. The real UC-71 delta is 49 files.
- **Hidden issue A:** eyes/bench.py:43-46 reads `gates.learned.weights` raw. After the merge, the liftoff preset uses `release: v1` with no `weights`, so `eyes bench` would silently skip the detector.
- **Hidden issue B:** liftoff/recorder.py:376/392/435-437 import `drone_fly.harness.offline.*`, which breaks the layering rule. test_runtime_modules_never_import_offline_or_heavy_at_module_level already fails on the branch.
- **Live default unchanged:** the preset's `release: v1` resolves to <data root>/models/detector-v1/liftoff-gates.pt, sha256 161f7a83… (= RELEASED_WEIGHTS). RELEASE_ALIASES has only v1, so v2 cannot be selected. The default and liftoff_arena presets stay on opencv.

## Plan
1. **Merge** `git merge origin/main` (not a rebase). Conflict resolutions:
   - contract.py: take main's version (same PR #80 code; docstring only);
   - test_harness_presets.py: union of the imports, isort-ordered;
   - USE_CASES.md: main's rows, with row 71 → in-progress, 2026-10-18.
   Commit the merge.
2. **Fix A:** eyes/bench.py `_detector` resolves through `gate_learned.configured_weights(lc)`.
   - An explicit gate_weights argument still wins.
   - No download. A missing file keeps the "not found" skip.
   - Help text at eyes/cli.py:406 and cli/main.py:1040 becomes "gates.learned.weights / release".
3. **Fix B:** a new harness/offline/capture_session.py.
   - Move record_manual, find_recording, record_replay, MANUAL_FALLBACK, REPLAY_STRATEGY and REPLAY_HUD_MASKS there.
   - Import CaptureRecorder and write_pose_capture at module level, so the monkeypatch seam works.
   - CaptureRecorder, RecorderStats, _screen_grabber, pose_rows and write_pose_capture stay in liftoff/recorder.py.
   - cli/main.py keeps its function-local imports, now from capture_session.
   - No shim, and the test is not relaxed.
   - Update docs/harness/liftoff-captures.md and ADR 0010.
4. **Docs:**
   - Append "## Delivery status (partial, 2026-10-18)" to use-cases/71-multi-map-gate-detector.md (keep its CRLF line endings):
     - AC1–6 and AC9 delivered;
     - AC7 open (best held-out val_det 0.503, target ≥0.70);
     - AC8 open (no v2 release; the preset stays v1);
     - follow-ups: unify eyes/capture_meta with offline/capture_index (would change capture_sha and invalidate label caches); resume training.
   - Add "Status: partial — AC7/AC8 open" in the ADR 0010 Results section. AC text stays unchanged.
5. **QA:**
   - Retarget tests/test_harness_capture_recorder.py (imports, REPLAY_HUD_MASKS, monkeypatches → capture_session.CaptureRecorder).
   - Add a bench test: liftoff preset with release v1 resolves the v1 path or skips with "not found", never "not configured". Use a tmp data root and no download.
   - Run the full suite plus ruff.
   - Read-only smokes, outputs to $TEMP: `harness captures inventory` and `eyes ingest`.
   - Confirm the v1 sha resolution and that the default and liftoff_arena presets are still opencv.
6. **Orchestrator:** ledger row 71 → in-progress; open the PR titled as a partial delivery, then merge.

No weights downloads and no training in this run.
