---
plan_for: (free-form task)
work_branch: feat/fix-windows-cuda-torch-clobber
team: drone-fly
approved: 2026-09-20
---

# Approved implementation plan — cu124 torch-clobber fix (setup-sim-windows.ps1)

## Analysis
`scripts/setup-sim-windows.ps1` step 4 installs `torch==2.6.0+cu124` via `--index-url $TorchIndex` (`https://download.pytorch.org/whl/cu124`). Steps 5 and 7 run `& uv pip install --python $Venv …` against **PyPI only**. With `torch` unconstrained in `pyproject.toml` and `stable-baselines3==2.9.0` depending on torch, uv re-resolves torch from PyPI (Windows = CPU-only wheels), silently replacing the cu124 build (observed `2.14.0+cpu`), so step 8's `torch.cuda.is_available()` assertion fails. Step 6 is `--no-deps`, so it does not touch torch. `--reinstall-package torch` is the load-bearing swap mechanism: without it uv treats an installed `+cpu` build as satisfying `torch==2.6.0` and skips the swap. Confirmed by the manual remedy.

## Proposed Solution
**File:** `scripts/setup-sim-windows.ps1`
1. **Add helper `Restore-CudaTorch`** near `Write-SetupLog`/`Write-Fail`/`Show-Usage`: logs via `Write-SetupLog`; runs `& uv pip install --python $Venv $TorchPin --index-url $TorchIndex --reinstall-package torch` (reuses existing vars — no second literal); checks `$LASTEXITCODE`; `Write-Fail` with an actionable message on failure.
2. **Step 4 unchanged** — standalone initial install, keeps its fresh-install error message.
3. **Call `Restore-CudaTorch` after step 5** (defensive) **and after step 7** (load-bearing), each immediately after that step's existing `$LASTEXITCODE` check, with intent comments:
   - after step 7 = load-bearing (step 8 verifies right after; nothing installs between) — must not be removed;
   - after step 5 = purely defensive (step 7 re-clobbers regardless) — kept for the "torch is always cu124" invariant / fail-fast, not because step 7 depends on it.
4. **Update `-DryRun` plan text** — fix the misleading step-5 "torch already installed" line; add lines noting the cu124 re-pin after steps 5 and 7; keep the AC-7 "pyproject.toml / uv.lock never touched" note.

## Files Affected
- **Production code (developer):** `scripts/setup-sim-windows.ps1` — add `Restore-CudaTorch`; call after steps 5 and 7 (with intent comments); update `-DryRun` lines.
- **Test code (qa):** `tests/test_setup_windows_docs.py` — add a hermetic regression test (no CUDA / no pwsh execution):
  - order-independent presence checks: `--reinstall-package torch` appears; `Restore-CudaTorch` is both defined and called;
  - ordering guard: a `Restore-CudaTorch` **call** (function definition excluded) appears *after* the `-e '.[dev]'` PyPI resolve line. **Do NOT** key the ordering on the `--index-url`/cu12x literal (it lives only in the helper definition near the top, so that check would false-fail correct code).
  - existing assertions stay green (single cu12x index literal, no second SHA literal, AC-7 no-leak into pyproject/uv.lock).

## Risks & Considerations
- **Path scoping.** `scripts/**` is in neither `paths.production` (`src/drone_fly/**`) nor `paths.test` (`tests/**`). `scripts/**` is scoped to the **developer** for this run (precedent: sibling setup scripts authored by the developer under UC-20/UC-31). QA owns `tests/**` as usual.
- **AC-7:** `pyproject.toml` / `uv.lock` untouched; index/pin confined to the ps1.
- **No second literal:** helper reuses `$TorchPin`/`$TorchIndex`.
- **Stale CPU-torch deps** after step 5/7: torch 2.6's deps are a subset (incl. sympy→1.13.1, correct for torch 2.6, not a regression); reinstall replaces torch; step 8's import + `cuda.is_available()` is the backstop.
- **PSScriptAnalyzer (AC-6):** helper uses `Write-SetupLog`/`Write-Fail` + the `&`/`$LASTEXITCODE` idiom; approved verb `Restore-`; no new suppressions needed.
- **CI cannot execute this** (Linux / no GPU / no pwsh); real verification is the documented manual Windows run per UC-31. The QA test is static/structural only.

## Challenger verdict
**Approve** (after 2 rounds) — "correct, minimal, and implementable. Ship it." Round 1 requested revision on one Major (the QA ordering assertion originally keyed on the `--index-url cu12x` literal, which lives above all step logic and would false-fail correct code; fixed to key on the helper call).
