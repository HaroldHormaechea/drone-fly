---
plan_for: use-cases/20-macos-sim-setup-automation.md
work_branch: feat/uc-20-macos-sim-setup
team: drone-fly-uc-20
approved: 2026-09-18
---

# UC-20 — Automated macOS real-physics sim setup (prebuilt pybullet, no source build)

Challenger-APPROVED (first pass; 3 minor folds baked in). Tooling + docs only; **nothing under `src/`**, no `pyproject.toml` change. Challenger verified: SHA match (train.sh:31 == pybullet_adapter.py:28), `.[dev]` cannot drag a pybullet source build (`pybullet==3.2.6` lives ONLY in the `sim` extra), AC4 verify import paths match the adapter exactly.

## Approved Solution

**1. NEW `scripts/setup-sim-macos.sh`** — executable, `#!/usr/bin/env bash`, `set -euo pipefail`, bash-3.2-safe (no assoc arrays / `mapfile` / `${x^^}`; only `$(...)`, plain loops, `sed`/`grep`). Flow:
- Resolve `SCRIPT_DIR`/`REPO_ROOT` from `BASH_SOURCE`; define `log`/`fail` mirroring train.sh (lines 44–47).
- Dry-run mode (`--dry-run` flag and/or `DRONE_FLY_SIM_DRYRUN=1`) that prints the plan and performs no network/conda/prompt actions (QA lever). Plus `-h/--help`.
- **Single-source the pin:** read `PYBULLET_DRONES_REPO` + `PYBULLET_DRONES_REF` values out of `scripts/train.sh` with portable `sed` BRE `\(...\)`; `fail` if either is empty. **Never** write a second SHA literal into this file.
- **Detect conda** (`command -v conda`, else `$CONDA_EXE`, else `~/miniforge3/bin/conda`) → `CONDA_BIN`.
- **AC2 prompt-before-install miniforge:** if no conda, explain + `read -r` explicit yes/no; anything but explicit yes → `fail`, install nothing, non-zero exit, never `sudo`. On yes: `uname -m` → `Miniforge3-MacOSX-arm64.sh` (or x86_64), `curl -fsSLo` installer, `bash <installer> -b -p "$HOME/miniforge3"` (batch, user-space); bind `CONDA_BIN="$HOME/miniforge3/bin/conda"`. All conda ops via `"$CONDA_BIN"` + `conda run -n <env>` (never `conda activate`).
- **AC3 prebuilt-only:** env `dronefly` (override `DRONE_FLY_CONDA_ENV`), Python 3.12 (override `DRONE_FLY_SIM_PYTHON`); create if absent else reuse. `conda install -y -n dronefly -c conda-forge pybullet`. `conda run -n dronefly pip install --no-deps "git+${REPO}@${REF}"`. Then explicit runtime deps: `gymnasium>=1.3,<2 stable-baselines3>=2.9,<3 transforms3d>=0.4,<0.5 control>=0.10.2,<0.11 matplotlib pillow scipy pytest`. Then `conda run -n dronefly pip install -e ".[dev]"` from `REPO_ROOT`.
- **AC4 verify:** `conda run -n dronefly python -c` importing `pybullet`, `from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary`, `from gym_pybullet_drones.utils.enums import DroneModel, Physics`, `drone_fly`; print `pybullet.getAPIVersion()` + the active backend (parity with `__init__.py`'s "Using PyBullet backend" log); non-zero exit on any import failure.
- **AC5 fail-fast + steer-to-container:** `steer_to_container()` helper called if the conda-forge pybullet install fails or verify fails — prints ready-to-run Apple `container run …` (macOS 26) AND `docker run …` fallback mounting `REPO_ROOT`, using the prebuilt manylinux `pybullet` wheel from PyPI + `pip install -e .[dev]`; exits non-zero. **Never a source build.**

**2. MODIFY `scripts/train.sh` — guarded Darwin dispatch (AC6).** **Insert at line 45** — immediately after the `fail()` definition (line 44) and **BEFORE the `command -v uv` check (line 46)** [fold #1: exact line 45; if it lands after 46, macOS hosts without uv hit a spurious "uv is required" failure and AC6's no-uv-on-mac intent breaks]. Block: `if [ "$(uname -s)" = "Darwin" ]; then …; fi`. Inside: resolve `CONDA_BIN` + env name; if the `dronefly` env exists → verify `$CONFIG` exists (reuse `fail`) then `exec "$CONDA_BIN" run --no-capture-output -n <env> drone-fly train --config "$CONFIG"`; else `fail` telling the user to run `./scripts/setup-sim-macos.sh` (from-source pybullet can't build on modern macOS SDKs — UC-20). Linux/CI never enters the branch → the existing source-build path (pin defs 30–31, uv install, `import pybullet` verify heredoc, launch) is byte-for-byte unchanged; all four train.sh doc-contract assertions keep passing.

**3. MODIFY `README.md` (AC7)** — additive edits near Requirements (line 181) / Tested-vs-untested boundary (line 189): macOS subsection (miniforge → conda-forge pybullet 3.2.5 → `gym-pybullet-drones --no-deps` → `drone-fly` console script via `setup-sim-macos.sh`), state from-source pybullet is unsupported on modern macOS SDKs and why (vendored zlib `zutil.h fdopen` macro vs SDK `_stdio.h`), the 3.2.5/`--no-deps` rationale, the fail-fast/container escape hatch, plus a note that the explicit dep list tracks the validated recipe with the AC4 verify as backstop [fold #3]. Add a boundary-table row for the macOS conda path as **untested in CI**. Preserve every existing required token verbatim (`tested-vs-untested`, `git ls-remote`, `pybullet`, `80`+`mastery`, `fixed`+`domain random`, and the `test_readme_documents_*`/`test_viz_contract` tokens) — edit additively, never replace.

**4. MODIFY `.claude/allowed-commands.yaml`** — append `shellcheck` (AC1 gate runs where shellcheck is installed; currently absent from this sandbox).

## CRITICAL implementation precisions (challenger-mandated)
- **train.sh insertion = line 45** (after `fail()`, before the uv check). Unambiguous.
- **`set -e` + probe pipelines [fold #2]:** under `set -euo pipefail`, every probe that can legitimately return non-zero (`command -v conda`, `conda env list | grep -q dronefly`, etc.) MUST be wrapped in `if …; then` or suffixed `|| true` — a bare probe aborts the script when an env is simply absent (normal control flow, not an error).
- **Explicit dep list is hand-maintained [fold #3]**, mirroring the validated recipe; a missing gym-pybullet-drones dep surfaces only on a Mac via the AC4 verify (fail-fast, never a source build). Accepted; noted in README/plan.

## Files Affected
**Production / tooling+docs (developer):** `scripts/setup-sim-macos.sh` (NEW, executable), `scripts/train.sh` (MODIFY — guarded Darwin dispatch only), `README.md` (MODIFY — additive), `.claude/allowed-commands.yaml` (MODIFY — add `shellcheck`).
- Scoping note: `paths.production = src/drone_fly/**`, `paths.test = tests/**`. These deliverables are tooling/docs **outside** both globs — developer writes `scripts/`, `README.md`, `.claude/allowed-commands.yaml`; QA writes `tests/`. No `src/` change (AC8), no `pyproject.toml` change.

**Test (QA) — `tests/test_bootstrap_docs.py` (MODIFY, additive):** `setup-sim-macos.sh` exists + executable; parses under `bash -n`; references `gym-pybullet-drones` and reads the pin from train.sh; **no second 40-hex SHA literal** (single-source enforcement); `shellcheck`-clean (skipif shellcheck absent — mirror the existing `bash`/`node` skipif idiom); README documents the macOS path + `--no-deps` + `miniforge`/`conda`. Keep ALL existing assertions intact. **Do NOT** add any assertion that pretends to test the conda/macOS exec path — that boundary is untestable in Linux CI.

## Gates (Linux CI, unchanged): `ruff check .`, `ruff format --check .`, `uv run pytest`. AC8: nothing under `src/` changes; full suite stays green. `.sh` files aren't ruff-scoped but must be shellcheck-clean.

## Risks & Considerations
- **Untested macOS/conda boundary** — cannot run in Linux CI; mitigated by dry-run + `bash -n` + shellcheck-where-present + guarded train.sh branch; documented in README. Same class as the existing pybullet boundary.
- **shellcheck absent in this sandbox** — AC1 verified only where shellcheck exists; QA test skips otherwise (CI green). Developer must write the script shellcheck-clean by construction (quote all expansions, no unused vars).
- **Single-source SHA via sed** — fail-on-empty + QA "no second SHA literal" guard against drift.
- **conda-forge pybullet 3.2.5 vs the `>3.2.7` pin** — conservative; validated working; AC5 container path is the escape hatch.

## Challenger verdict
**APPROVE — no Critical/Major.** Verified: source-build impossible by construction (conda prebuilt + `--no-deps` + `.[dev]` not `.[dev,sim]`); SHA single-sourced via sed (QA enforces no second literal); miniforge prompt-before-install/never-sudo; fail-fast steer-to-container; guarded Darwin branch keeps Linux byte-unchanged; README doc-contract preserved; no `src/` change. 3 minor folds (train.sh line-45 insertion; wrap `set -e` probes; hand-maintained dep list backstopped by AC4) — all baked in above.
