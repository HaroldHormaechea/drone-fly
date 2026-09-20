---
plan_for: use-cases/31-windows-cuda-training-setup.md
work_branch: feat/uc-31-windows-cuda-training-setup
team: drone-fly-uc-31
approved: 2026-09-20
---

# ORCHESTRATOR WRITE-SCOPE GRANT (this run only)
`paths.production` is `src/drone_fly/**`. For UC-31 the developer is **explicitly authorized** to also write these out-of-glob files (precedent: UC-20 scripts, UC-28 viz/README):
- `scripts/setup-sim-windows.ps1` (NEW)
- `README.md`
- `.claude/allowed-commands.yaml`
- `.gitignore`

**MUST NOT change** `pyproject.toml` or `uv.lock` (keeps the CUDA index/pin opt-in and the Linux CI resolve hermetic). QA stays in `tests/**`. `PLAN_FILE`/ledger are orchestrator-written.

---

# FINAL APPROVED PROPOSAL — UC-31 Automated Windows + NVIDIA CUDA training setup
(Analyst + Challenger agreed. Challenger verdict: APPROVE, v2.)

## Summary
Give drone-fly a reproducible Windows+CUDA training path (RTX 3050, 8 GB, Ampere/sm_86) mirroring the macOS UC-20 automation: a PowerShell setup script + verification/benchmark + VRAM guidance + README, with only CI-testable slices exercised in the Linux/no-GPU sandbox. The runtime already selects CUDA (`resolve_device()`), so this is setup-tooling + docs + one small in-scope device.py change.

## Verified ground truth
- `src/drone_fly/train/device.py:60` `resolve_device()` returns `"cuda"` when `_cuda_available()` (rule 2) and honors `device="cuda"` override; `VALID_DEVICES=("cpu","cuda","mps")`. Callers: `train/loop.py:327`, `evaluate/evaluator.py:103`. No runtime change needed for selection.
- CUDA branch already unit-tested headless in `tests/test_device.py` (`test_auto_prefers_cuda_when_available`, `test_override_cuda_wins_even_without_hardware`).
- UC-20 mirror template: `scripts/setup-sim-macos.sh` + its CI contract in `tests/test_bootstrap_docs.py` (exists/executable, parse, lint **skip-if-absent**, README doc, no second SHA) + README boundary row (`README.md:344`).
- `pyproject.toml`: `torch` unpinned (uv.lock→2.14.0); `sim` extra pins `pybullet==3.2.6` (needs numpy<2). `scripts/train.sh` single-sources `PYBULLET_DRONES_REPO/REF` (`7ebad1e…`, v2.2.0), `SIM_PYTHON=3.12` (gym-pybullet-drones needs py≥3.12). CI = `uv sync --extra dev` + ruff + `ruff format --check` + pytest on py3.11; no shellcheck job.
- Sandbox: `pwsh`/GPU/Windows all absent → all PowerShell/PSScriptAnalyzer/CUDA assertions skip-if-absent.

## Proposed Solution

**1. `scripts/setup-sim-windows.ps1` (NEW).** PowerShell mirror of `setup-sim-macos.sh`: header/usage, `-DryRun`/`-Help`, env overrides (`DRONE_FLY_SIM_DRYRUN`, `DRONE_FLY_SIM_PYTHON` default `3.12`, `DRONE_FLY_VENV` default `.venv-cuda`), colored `Write-Log`/`Fail`, hermetic `-DryRun`, prompt-before-install (user-space `irm https://astral.sh/uv/install.ps1|iex`, decline→exit non-zero), idempotent venv reuse, clear success/fail summary.

**Install ordering & index/dep discipline (must implement exactly — CI can't catch this):**
  1. **Torch first, CUDA index scoped to that line ONLY:** `uv pip install --python <venv> torch==<cu12x-pin> --index-url https://download.pytorch.org/whl/cu124`. `--index-url` (replaces PyPI) appears on the torch line only, never global/exported.
  2. **Single unified PyPI resolve for the rest:** `uv pip install --python <venv> "pybullet==3.2.6" "numpy<2" "git+<repo>@<ref>" -e ".[dev]"` — one solver call so numpy<2 ABI + torch + gym-pybullet-drones co-satisfy (prevents silent numpy 2.x clobber of pybullet). No `--index-url` here → PyPI only; torch already satisfied.
  3. **gym-pybullet-drones = `--no-deps` + explicit runtime deps**, pin single-sourced from `train.sh` via `Select-String` (no second 40-hex SHA literal). Fallback if `--no-deps` drops a Windows-only dep: full-deps + explicit `pybullet==3.2.6`/`numpy<2` pins.
  4. **Driver check:** parse `nvidia-smi` reported `CUDA Version:` ≥ 12.x; if missing/too old print an actionable upgrade message. Post-install `torch.cuda.is_available()` verify is the backstop.
  5. **Post-install verify AFTER `-e .`:** assert `torch.cuda.is_available()`, print `torch.cuda.get_device_name(0)`, import `pybullet`/`gym_pybullet_drones`/`drone_fly`, assert `resolve_device()=="cuda"`, surface `CUDA_OOM_HINT`.

**2. Benchmark (AC-2) — documented, no new production code.** README documents `drone-fly train --device cuda` with tiny timesteps. `loop.py:357-362` already logs `device=%s`; SB3's logger emits per-iteration `time_elapsed`/`fps` — compare to the UC's CPU 821 s / MPS baselines. (Fallback: a timing snippet in the ps1 verify block if a cleaner single number is wanted.)

**3. `src/drone_fly/train/device.py` (EDIT — only in-`paths.production` touch).** Add documented `CUDA_OOM_HINT` constant (mirrors `APPLE_SILICON_NOTE` at `device.py:37`) advising lower `n_envs`/`batch_size` (and/or smaller slice) on OOM. **Log it via `logger.info` in `resolve_device()` on both cuda paths** (auto after `device.py:82`; `override=="cuda"`) — mirrors existing note logging. `train/loop.py` untouched → zero training-dynamics change (AC-7 safe). Optional re-export via `train/__init__.py`.

**4. `README.md` (EDIT).** New "Windows (NVIDIA CUDA) real-physics training (UC-31)" section mirroring the macOS section (`README.md:353`): uv + CUDA wheel index-url setup, device auto-select, 8 GB VRAM/OOM tuning, benchmark command; + boundary-table row `Windows/CUDA via setup-sim-windows.ps1 | untested in CI | no Windows/GPU runner`.

**5. `.claude/allowed-commands.yaml` (EDIT).** Append `pwsh` (skip-if-absent lint/parse), mirroring the `shellcheck` entry.

**6. `.gitignore` (EDIT).** Add `.venv*/` — current `.gitignore:10-11` only ignores `.venv/`+`venv/`, not `.venv-cuda/`.

**7. AC-7 isolation (do NOT change `pyproject.toml`/`uv.lock`).** CUDA index+pin confined to the ps1 → default Linux/macOS resolve and hermetic CI (`uv sync --extra dev`) byte-unchanged.

## Files Affected

**Production (developer):**
| File | Change | Scope |
|---|---|---|
| `scripts/setup-sim-windows.ps1` | NEW | **Outside `paths.production` → WRITE-SCOPE GRANT** (precedent UC-20) |
| `README.md` | EDIT | **Outside `paths.production` → GRANT** (precedent UC-28) |
| `.claude/allowed-commands.yaml` | EDIT (add `pwsh`) | **Outside `paths.production` → GRANT** |
| `.gitignore` | EDIT (add `.venv*/`) | **Outside `paths.production` → GRANT** |
| `src/drone_fly/train/device.py` | EDIT (`CUDA_OOM_HINT` + logging) | In-scope |
| `src/drone_fly/train/__init__.py` | EDIT (optional re-export) | In-scope |
| **MUST NOT change** | `pyproject.toml`, `uv.lock` | keeps CUDA opt-in + CI hermetic |

**Test (qa, `tests/**`):**
- `tests/test_setup_windows_docs.py` (NEW, mirrors `test_bootstrap_docs.py`): ps1 exists; references CUDA wheel index (`download.pytorch.org/whl/cu`) + a pinned cu12x torch; single-sources the gym-pybullet-drones pin from `train.sh` with no second 40-hex SHA literal; references `uv`; PSScriptAnalyzer-clean **skip-if-absent**; PowerShell syntax parse via `pwsh -NoProfile` **skip-if-absent**; README has Windows/CUDA section + boundary row; **neither `pyproject.toml` nor `uv.lock` contains a `download.pytorch.org/whl/cu` URL** (AC-7 guard).
- `tests/test_device.py` (EDIT): label existing cuda tests `# UC-31`; add ONLY new assertions — `CUDA_OOM_HINT` info log fires on both cuda paths (caplog) + content mentions `n_envs`/`batch_size`. No duplicate copies.

## Risks & Considerations
- **torch pin drift:** ps1 cu12x torch is an isolated-venv pin, may differ from lock's 2.14.0; only requirement = compatible with `stable-baselines3==2.9.0` + `gymnasium==1.3.0`. Developer confirms (a) a cu12x torch wheel and (b) a cp312 Windows `pybullet==3.2.6` wheel exist.
- **numpy ABI** is the sharpest footgun — the single unified resolve (point 2) is mandatory, not optional.
- Everything Windows/GPU is untested-in-CI (mirrors shellcheck). Hermetic gate = existence/pin/index/doc assertions + monkeypatched CUDA device branch + pyproject/uv.lock no-index guard.

## AC-1..AC-7 → where satisfied
- **AC-1** → `scripts/setup-sim-windows.ps1` (uv env, pinned cu12x torch via index-url, pybullet+gym-pybullet-drones per pins, project install, idempotent/user-space, success/fail summary, driver-too-old msg); gated by `test_setup_windows_docs.py`.
- **AC-2** → ps1 post-install verify + documented `drone-fly train --device cuda` (device log `loop.py:357` + SB3 per-iter `time_elapsed`/`fps`).
- **AC-3** → `tests/test_device.py` monkeypatched auto-cuda + `device="cuda"` override (existing, UC-31 labeled), headless on Linux.
- **AC-4** → `CUDA_OOM_HINT` logged at runtime in `resolve_device()` + README + ps1 output; tested via caplog + content.
- **AC-5** → README Windows/CUDA section + boundary row; tested in `test_setup_windows_docs.py`.
- **AC-6** → PSScriptAnalyzer + syntax-parse tests skip-if-absent; `pwsh` in `allowed-commands.yaml`.
- **AC-7** → CUDA index/pin confined to ps1; `pyproject.toml`/`uv.lock` untouched + no-index guard test; full ruff/ruff-format/pytest suite green + offline.

**Ready for the developer.**
