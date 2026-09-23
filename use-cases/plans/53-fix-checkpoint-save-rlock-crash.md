---
plan_for: use-cases/53-fix-checkpoint-save-rlock-crash.md
work_branch: feat/uc-53
team: drone-fly-uc-53
approved: 2026-09-23
---

# UC-53 — checkpoint-save RLock crash fix + atomic-save hardening

APPROVED by challenger (9/9 ACs, revised after scope expansion; verified against installed SB3 2.9.0 source). Two challenger Minor impl notes + test-hardening carry-overs folded in.

## Analysis (verified against source)

**Exclusion bug.** `ProgressReportingPPO` (`src/drone_fly/train/progress_ppo.py`) has no `__init__`; the sole persistent instance attribute it adds is `self._progress_sink` (set by `attach_progress_sink`, line ~94; read in `train()` via `getattr(self,"_progress_sink",None)`). The `counting_get_proxy` anomaly state is a local dict, and `_safe_sink_call` is a `@staticmethod` — neither adds instance state. SB3 2.9.0 `_excluded_save_params()` (base_class.py:303) returns an 8-item list that omits `_progress_sink`, so when the TUI is attached the RLock-bearing `TrainingDashboard` reaches cloudpickle → `TypeError: cannot pickle '_thread.RLock'`. The crash mid-write truncates the checkpoint `.zip` → later resume fails with `BadZipFile`. Attach happens only under `tui_enabled` (loop.py:543, outside the resuming/else branch); resume `load()` (loop.py:494) yields a model with no sink and loop.py re-attaches — so no call-site changes are needed.

**Save internals.** `BaseAlgorithm.save(self, path, exclude=None, include=None)` (base_class.py:819) computes `exclude = set(exclude).union(self._excluded_save_params())` then `data.pop(name, None)` over `data = self.__dict__.copy()` — exclusion is by-name against the instance dict, so it's a guaranteed no-op when the attr is absent. It ends at `save_to_zip_file(path,...)` → `open_path(path,"w",suffix="zip")`; `open_path_pathlib` (save_util.py:275) appends `.zip` **only when `path.suffix == ""`** (any existing suffix kept verbatim), auto-creates a missing parent dir (catches `FileNotFoundError` → `path.parent.mkdir(parents=True, exist_ok=True)`, save_util.py:283-284), and accepts `io.BufferedIOBase` buffers written directly.

## Proposed Solution (no code)

Two composing overrides on `ProgressReportingPPO`, both in `src/drone_fly/train/progress_ppo.py`:

**(A) `_excluded_save_params(self) -> list[str]`** → return `super()._excluded_save_params() + ["_progress_sink"]`. Call super and append; never hardcode SB3's list.

**(B) `save(self, path, exclude=None, include=None) -> None`** — atomic write:
1. **Buffer passthrough:** if `path` is NOT a `str`/`os.PathLike` (use `isinstance(path, (str, os.PathLike))`), `return super().save(path, exclude=exclude, include=include)` — atomic rename is meaningless for a buffer/file-like target.
2. **Resolve final path identically to SB3:** `final = Path(path)`; if `final.suffix == ""`, `final = Path(f"{final}.zip")`. (`ckpt`→`ckpt.zip`; `ckpt.zip`→`ckpt.zip`; `ckpt.foo` kept.)
3. **Parent-dir fidelity (challenger Minor #1):** `final.parent.mkdir(parents=True, exist_ok=True)` before creating the temp, so the override matches SB3's auto-create-parent behavior instead of raising `FileNotFoundError` where plain SB3 would succeed.
4. **Temp beside the target:** create a uniquely-named temp in `final.parent` (same dir ⇒ same volume ⇒ `os.replace` truly atomic, incl. Windows/NTFS — the temp MUST NOT go to `%TEMP%`/another volume). The temp name must carry a NON-EMPTY suffix (e.g. `.tmp`) so SB3's `open_path` won't re-append `.zip` to it. **Close the temp fd immediately (challenger Minor #2):** if using `tempfile.mkstemp(...)`, `os.close(fd)` right away (or use `NamedTemporaryFile(delete=False)` and close it) — on Windows a lingering open handle can block `super().save()`'s own `open("wb")` and the subsequent `os.replace`.
5. **Serialize to temp via super:** `super().save(str(temp), exclude=exclude, include=include)` — honors override (A), so `_progress_sink` is dropped here.
6. **Atomic publish:** `os.replace(temp, final)`.
7. **Failure handling:** wrap 5–6 in try/except; on ANY exception remove `temp` if it exists (best-effort) and re-raise. `final` is only ever touched by the terminal `os.replace`, so a pre-existing checkpoint at `final` stays byte-intact and readable, and no stray temp remains.

No change to `train()`, `attach_progress_sink`, `counting_get_proxy`, `_safe_sink_call`, or to `loop.py`/TUI/env/reward/adapter/policy/connectome math.

## Files Affected
- **Production:** `src/drone_fly/train/progress_ppo.py` — add methods (A) and (B). New imports likely `os`, `pathlib`, `tempfile`.
- **Test (QA owns):** new `tests/test_uc53_checkpoint_save_sink_exclusion.py`:
  - **AC1:** attach a sink holding a real `threading.RLock` (genuinely unpicklable — a plain recording sink like UC-52's would NOT reproduce the bug); precondition-assert `pytest.raises(TypeError, pickle.dumps, sink)` so the regression can't rot into a no-op; then assert `model.save(tmp)` does not raise.
  - **AC2:** `_excluded_save_params()` == base list + `"_progress_sink"`, base entries preserved.
  - **AC3:** save → `ProgressReportingPPO.load()` → `getattr(loaded,"_progress_sink",None) is None`; re-attach a sink and confirm functional.
  - **AC4:** no-sink save/load path unaffected.
  - **AC7:** save produces the checkpoint at the SB3-resolved path via temp+rename; assert no stray temp remains and the final `.zip` is a valid loadable archive; assert the temp path SB3 wrote equals the one renamed from (double-extension guard).
  - **AC8:** `save("ckpt")` and `save("ckpt.zip")` land at the same file; save to an `io.BytesIO` buffer delegates to super, round-trips, and attempts no rename.
  - **AC9 (key):** simulate a mid-save failure by monkeypatching at a point AFTER the temp is created but BEFORE `os.replace` (patch `save_util.save_to_zip_file` or `PPO.save` to raise), with a valid pre-existing checkpoint already at the target; assert the old checkpoint is byte-unchanged and still `load()`-able, `save()` raised, and no temp file was left in the dir.
  - Use a FULLY-constructed tiny `ProgressReportingPPO` (small MlpPolicy on a tiny gym env, minimal `n_steps`) for the real save/load round trips — NOT the `__new__(...)` inert-path shortcut UC-52 uses. No connectome actor needed.
  - **AC5 guard:** existing `tests/test_uc52_optimize_progress.py` must still pass unchanged.

## Risks & Considerations
- **Temp double-extension trap:** temp must have a non-empty suffix or SB3 appends `.zip` and we'd rename the wrong file — mitigated in step 4; add the equality assertion in the AC7 test.
- **Windows fd/handle (Minor #2):** close the temp fd before super/replace.
- **Parent-dir fidelity (Minor #1):** mkdir the parent before the temp.
- **Windows atomicity:** `os.replace` is atomic same-volume on NTFS; same-directory temp guarantees same volume; `%TEMP%` would degrade to a non-atomic cross-volume copy — explicitly avoided.
- **Exclusion completeness:** re-verified — `_progress_sink` is the only unpicklable-capable added attribute.
- **Composition:** (B) calls `super().save(temp, exclude, include)` which applies (A); forwarding `exclude`/`include` preserves SB3's public contract.
- **Scope:** AC-6 confines everything to `progress_ppo.py` + `tests/**`; the atomic override lives in `progress_ppo.py`, so no scope conflict.

## Challenger verdict
APPROVE — all 9 ACs covered. Design re-verified against SB3 2.9.0: composition with `_excluded_save_params()` holds, path resolution matches `open_path_pathlib`, temp double-extension trap avoided with a non-empty temp suffix, AC-9 "prior checkpoint intact" holds because `final` is only touched by the terminal `os.replace`. Two Minor impl notes (mkdir parent; close temp fd before replace) + test-hardening recs folded in above.
