# Use Case 53: Fix checkpoint-save crash — exclude the progress sink from PPO serialization

## Summary
UC-52's `ProgressReportingPPO` (`src/drone_fly/train/progress_ppo.py`) attaches the live `TrainingDashboard` as an instance attribute `self._progress_sink` (via `attach_progress_sink`) when the TUI is active. The dashboard holds a `threading.RLock`. SB3's `BaseAlgorithm.save()` serializes the model's `__dict__` (minus `_excluded_save_params()`) with cloudpickle, and `_progress_sink` is **not** in the exclusion list — so every checkpoint save while the TUI is attached crashes with `TypeError: cannot pickle '_thread.RLock' object`. Worse, the crash happens mid-write, leaving a truncated/corrupt checkpoint `.zip` that then fails to resume with `BadZipFile: File is not a zip file`. This is a regression introduced by UC-52 (PR #58, `452b7bb`): the hermetic tests exercised `model.save()` only with a simple picklable recording sink, never with a sink holding an unpicklable object like the real dashboard, so the gap slipped through.

This use case does two things:
1. **The fix:** override `_excluded_save_params()` on `ProgressReportingPPO` to exclude `_progress_sink` (and any other transient instrumentation attribute that must not be pickled), so `save()` succeeds with the dashboard attached; on load the sink is simply absent and is re-attached by `loop.py` when the TUI is active (the code already reads it via `getattr(..., None)`).
2. **The hardening:** make checkpoint writes **atomic** so that *any* future mid-write failure (not just this pickle error) can never corrupt an existing or target checkpoint. Override `ProgressReportingPPO.save()` to serialize to a temporary file in the same directory as the target, then atomically `os.replace()` it into place — so a partial/failed write leaves the previous checkpoint intact and never produces a half-written `.zip`. This is what turns a crash into a recoverable stop instead of a corrupted-checkpoint dead end.

Scope is confined to `progress_ppo.py` + tests. No behavior change to training math or the optimize-progress feature.

## Acceptance Criteria
1. `ProgressReportingPPO.save(path)` succeeds when a progress sink holding an unpicklable object (e.g. a `threading.RLock`, or the real `TrainingDashboard`) is attached — no `TypeError: cannot pickle` is raised.
2. `_progress_sink` is excluded from the saved data via an override of SB3's `_excluded_save_params()` that returns the base list plus `"_progress_sink"` (do not drop or reorder the base entries).
3. A full round trip works: `save()` → `ProgressReportingPPO.load()` → the loaded model has no `_progress_sink` (a subsequent `getattr(self, "_progress_sink", None)` is `None`), and `train()`/`save()` continue to work on the loaded model (re-attaching a sink afterwards still functions).
4. With no sink attached, `save()`/`load()` remain byte-identical to plain PPO behavior (the exclusion is a no-op when the attribute is absent).
5. The optimize-progress feature itself is unchanged: with a sink attached, `train()` still emits `begin/tick/end_optimize` and the 1 Hz throttle behavior is unaffected (existing UC-52 tests still pass).
6. Scope is confined to `src/drone_fly/train/progress_ppo.py` and `tests/**`. No changes to `loop.py`, the TUI modules, reward/env/adapter, or policy/connectome math.
7. **Atomic save:** `ProgressReportingPPO.save()` writes to a temporary file in the **same directory** as the target path and then atomically `os.replace()`s it into place, so an interrupted/failed serialization never leaves a truncated `.zip` at the target and never destroys a pre-existing checkpoint at that path. (`os.replace` is atomic same-volume on both POSIX and Windows.)
8. **Path/extension fidelity:** the atomic override preserves SB3's existing `save()` contract — the final file lands at the same resolved path SB3 would use (including SB3's `.zip` extension normalization), the temp file is created in the same directory (same filesystem, so `os.replace` is truly atomic), and the temp file is cleaned up on failure. When the save target is a file-like/buffer object rather than a filesystem path, delegate straight to `super().save()` (atomic rename is meaningless for a buffer).
9. **Failure leaves prior checkpoint intact:** a simulated mid-save failure (serialization raises) must leave any previously-existing checkpoint at the target path unchanged and readable, and must not leave a stray temp file.

## Potential Pitfalls & Open Questions
- **Exclusion completeness:** `_progress_sink` is the only instance attribute added by `ProgressReportingPPO` that can hold an unpicklable object (other locals in `train()` are not persisted on `self`). Confirm no other added attribute (e.g. any disable flag) needs exclusion; if it's a plain bool it's picklable and can stay.
- **Base-list composition:** `_excluded_save_params()` must call `super()._excluded_save_params()` and append — never hardcode SB3's list (it varies by version; SB3 is pinned 2.9.0 but still).
- **Atomic-save composition:** the atomic `save()` override and the `_excluded_save_params()` exclusion compose — `save()` writes to a temp path via `super().save(temp)` (which already honors the exclusion), then `os.replace(temp, final)`. Get the final resolved path right: SB3 normalizes a missing extension to `.zip`, so either resolve the final name the same way and create the temp beside it, or let SB3 write the temp (`super().save(temp_with_zip)`) and rename to the caller's resolved target. The temp must share the target's directory/volume for `os.replace` to be atomic.
- **Windows specifics (user runs Windows/CUDA):** `os.replace` maps to an atomic replace on same-volume NTFS; a temp file in a different dir (e.g. `%TEMP%`) would fall back to a non-atomic copy — so the temp MUST be created in the target directory.
- **Regression coverage:** the new test must attach a sink that is genuinely unpicklable (holds an `RLock`) — a plain recording sink would not reproduce the bug. Ideally also assert the exact prior failure mode is gone (save with such a sink no longer raises).

## Original Description
User's training run crashed at a checkpoint save with `TypeError: cannot pickle '_thread.RLock' object` (SB3 `model.save()` → cloudpickle), then could not resume because the interrupted save left a corrupt zip (`BadZipFile: File is not a zip file`). Root cause: UC-52 attaches the RLock-holding TrainingDashboard as `model._progress_sink`, which SB3's `save()` tries to pickle. Fix: exclude `_progress_sink` from `_excluded_save_params()`. Immediate user workaround is `--no-tui` (sink never attached). This UC is the permanent fix + the regression test that was missing (UC-52 only tested save with a picklable recording sink).

## Clarifications
- Q: Team or direct hotfix?
  A: Dev-team, autonomous through merge — the missing regression test (save with an unpicklable sink) is exactly what let the bug ship, so QA must add it.
