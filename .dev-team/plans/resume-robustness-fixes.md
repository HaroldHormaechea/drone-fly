---
plan_for: (free-form task)
work_branch: feat/resume-robustness-fixes
team: drone-fly
approved: 2026-09-18
---

# Implementation Plan — Resume-robustness fixes (auto-latest checkpoint + recording no-clobber)

**Status:** APPROVED by challenger (round 1, no revision). 3 Minor items folded in.

### Analysis
Two resume-robustness fixes + one warning:
- **Fix 1** — `--resume` today requires an exact `.zip` (`cli/__init__.py::_add_train_args` ~line 104, `default=None`); a user silently rewound training. Add directory / `latest` / bare-flag resolution to the newest checkpoint via the existing `find_latest_checkpoint(models_dir)` (`train/loop.py` ~line 65). Explicit `.zip` must stay byte-identical.
- **Fix 2** — `RecordingCallback._on_training_start` (`train/record_callback.py` ~line 70) resets `self._episode = 0`, so a resumed run overwrites the prior run's `episode_<n>.json[.gz]` in the same `record_dir`. Continue past existing recordings instead.
- **Fix 2b** — `--record-every` passed without `--record` is a silent no-op; warn.

### Proposed Solution (no code)

**A. `src/drone_fly/cli/__init__.py`**
1. `_add_train_args`: change `--resume` to `nargs="?"`, `default=None` (omitted → fresh), `const="latest"` (bare `--resume` → sentinel `"latest"`). Explicit `.zip` and directory forms flow through the value. Update help text.
2. `_add_record_args`: change `--record-every` `default` from `1` → `None` (so "explicitly passed" is detectable). **[challenger #2 — load-bearing]** In BOTH the `train` and `evaluate` dispatch branches, coalesce `record_every = args.record_every if args.record_every is not None else 1` before calling the callee — `RecordingCallback.__init__` does `max(int(record_every),1)` and `evaluate_checkpoint` does `max(record_every,1)`, so a leaked `None` throws. The developer must not drop either branch.
3. Add `_warn_record_every_without_record(args)` logging a warning when `args.record_every is not None and not args.record`; call it in both the `train` and `evaluate` branches. Reference pattern: the `--n-envs < 1` guard in `main()`'s train branch (~line 258).

**B. `src/drone_fly/train/loop.py`**
1. Add module-level `_resolve_resume(resume, models_dir)`, checks ordered **`None` → `"latest"` sentinel → `os.path.isdir(...)` → passthrough** (sentinel before isdir): `None→None`; `"latest"→find_latest_checkpoint(models_dir)`; a directory → `find_latest_checkpoint(that_dir)`; else return the explicit path unchanged. If the sentinel/directory branch resolves to `None` (no `*_steps.zip`), raise `FileNotFoundError` naming the searched dir — keep this hard error (silent train-from-scratch is the data-loss this task kills).
2. In `train()`, call it at the top of the resuming logic and use the **resolved** concrete path everywhere `resume` is used today (`_infer_vecnormalize_path`, `PPO.load`, the resume log line). `reset_num_timesteps=not resuming` unchanged. Confirmed: the resolved latest `*_steps.zip` basename still matches `_(\d+)_steps\.zip$`, so sibling-`.pkl` VecNormalize lookup works unchanged.

**C. `src/drone_fly/train/record_callback.py`**
1. Add module-level `_highest_episode_index(record_dir) -> int`: glob `episode_*.json` AND `episode_*.json.gz`, parse with end-anchored `episode_(\d+)\.json(\.gz)?$`, return max or `-1` for absent/empty dir.
2. In `_on_training_start`, replace `self._episode = 0` with `self._episode = _highest_episode_index(self.recorder.out_dir) + 1`. Empty dir → `0` (fresh-run parity); resumed → `max+1` (no clobber; absolute-index `% record_every` cadence preserved). **[challenger #3]** Add a one-line comment noting the counter is derived from files on disk, so with `record_every>1` it won't equal the prior run's cumulative episode count — that's intentional (guarantees no clobber + preserves cadence); don't "correct" it.

### Files Affected
**Production code (developer):**
- `src/drone_fly/cli/__init__.py`
- `src/drone_fly/train/loop.py`
- `src/drone_fly/train/record_callback.py`

**Test code (qa):**
- `tests/test_train_resume.py` — `--resume` directory → newest ckpt; `"latest"` → newest in `cfg.models_dir`; explicit `.zip` unchanged; empty dir/`latest` with no checkpoints → `FileNotFoundError`.
- `tests/test_cli.py` — parser: bare `--resume` → `"latest"`; `--resume dir/` and `--resume x.zip` parse (existing `assert args.resume=="x.zip"` still holds); warning fires for `--record-every` without `--record` and not with `--record` (caplog).
- **[challenger #1 — corrected pointer]** `tests/test_record_cli.py` — the assertion asserting the `record_every` default lives here (`test_record_off_by_default` ~lines 60-63: `assert args.record_every == 1`), NOT in `test_cli.py`. Update it to `is None`, and add a dispatch-boundary test that the coalesced value reaching the callee is `1`.
- `tests/test_record_rollout.py` — recording-callback counter continuation: seed a `record_dir` with `episode_0/1.json` (+ a `.json.gz` variant) and assert a fresh callback starts numbering at `max+1`; empty dir → 0. (`test_record_backcompat.py` covers `.json`/`.json.gz` duality if QA prefers to split the suffix case.)

### Risks & Considerations
- `nargs="?"` bare-flag swallow risk absent — `train` has zero positionals.
- `--record-every` default 1→None is internal only; coalesce keeps runtime identical.
- FileNotFoundError only triggers for the new directory/`latest` forms; explicit-path behaviour unchanged.
- Out of scope (challenger noted, do NOT add): `--record-dir` without `--record` is the same class of no-op, but the task only asks about `--record-every`.
- Backward-compat: fresh run (empty record_dir) → episode 0; explicit `.zip` resume unchanged.

## Challenger verdict (round 1): APPROVE
Verified all code claims against source; design correct, complete, scoped, backward-compatible. Edge cases resolve cleanly: bare `--resume` (no positional swallow), no-checkpoints-found (loud FileNotFoundError, not silent fresh-start), vecnormalize sibling lookup (regex matches resolved path), both `.json`/`.json.gz` scanned, fresh-run parity preserved. 3 Minor items folded into the plan above.
