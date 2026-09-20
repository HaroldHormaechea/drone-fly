# Use Case 31: Automated Windows + NVIDIA CUDA training setup

## Summary
Provide a reproducible Windows + NVIDIA CUDA training path for drone-fly so the large connectome slice trains on a GPU (user's RTX 3050, 8 GB, Ampere/sm_86) instead of the M4 Pro CPU, where one PPO iteration of the 122,816-neuron slice takes ~13m41s. CUDA — unlike the Mac's MPS — has real sparse-tensor support, so the sparse connectome propagation (the bottleneck) runs on the GPU rather than falling back to CPU. The runtime already targets CUDA (`src/drone_fly/train/device.py` `resolve_device()` auto-selects `cuda` when `torch.cuda.is_available()`; `device="cuda"` is a valid override); what's missing is the environment setup, verification, and docs. This UC mirrors the macOS setup automation (UC-20) but for Windows/CUDA using **uv + the PyTorch CUDA wheel index**, delivered as a **PowerShell script** `scripts/setup-sim-windows.ps1` that provisions the env and installs a **pinned CUDA 12.x** PyTorch build + pybullet + gym-pybullet-drones (existing pins) + the project; a user-run verification/benchmark that confirms `torch.cuda.is_available()`, prints the GPU, confirms `resolve_device()=="cuda"`, and times one iteration; 8 GB VRAM tuning guidance; and a README section. Because the CI/dev sandbox is Linux with no GPU/Windows, the actual CUDA run is a **documented manual step**; the CI-testable slices are the device-selection CUDA branch (monkeypatched) and PowerShell script lint. Scope is setup tooling + docs + the device test only — no change to training dynamics, observation schema, connectome, reward, checkpoints, or the env/adapter; existing Linux/macOS installs and the hermetic Linux CI stay green.

## Acceptance Criteria
1. **PowerShell setup script** `scripts/setup-sim-windows.ps1`: provisions a Python env via **uv**, installs a **pinned CUDA 12.x** PyTorch build (via the PyTorch CUDA wheel `--index-url`, compatible with the RTX 3050 + a recent driver) + pybullet + gym-pybullet-drones (per existing pins) + the drone-fly project; idempotent/reusable, user-space (no admin), prints a clear success/failure summary; surfaces a clear message if the NVIDIA driver/CUDA runtime is too old.
2. **Verification/benchmark** (documented command the user runs on Windows): asserts `torch.cuda.is_available()` is True, prints the GPU name, confirms `resolve_device()` returns `"cuda"`, and runs a one-iteration training smoke printing the device used + iteration wall-clock (comparable to the 821s CPU / 2.57s-per-iter MPS baselines).
3. **CUDA device-selection CI test** (headless, no GPU): monkeypatch `torch.cuda.is_available()` → True and assert `resolve_device()` returns `"cuda"`; also assert the explicit `device="cuda"` override path. Runs in the Linux CI without a GPU.
4. **VRAM guidance:** docs state that on CUDA OOM (8 GB is tight for 122k) the user lowers `n_envs`/`batch_size` (and/or uses a smaller slice); a clear OOM hint is surfaced where practical.
5. **Docs:** a Windows/CUDA training section in README (mirroring the macOS section) — uv+CUDA-index setup, device auto-select, VRAM tuning, the benchmark command.
6. **Script lint:** `setup-sim-windows.ps1` passes PSScriptAnalyzer when available, else a syntax check; gated/skipped in CI when the tool is absent (UC-20 precedent with shellcheck).
7. **Scope + CI:** no change to training/obs/connectome/reward/checkpoints/env-adapter; the CUDA torch index/pin is an **opt-in Windows path** that does NOT alter the default Linux/macOS install or CI; full hermetic Linux suite (`ruff check .`, `ruff format --check .`, `uv run pytest`) green + offline; CUDA/Windows parts guarded/skipped when unavailable.

## Potential Pitfalls & Open Questions
- **Edge case (dev-team can't run CUDA/Windows):** CI is Linux, no GPU — the GPU run is a documented manual verification; the dev-team must NOT install CUDA torch or run pybullet-CUDA in CI.
- **Risk (dependency isolation):** the CUDA torch wheel/index must not change the default install or CI — keep it a separate, opt-in Windows path/extra so Linux/macOS resolves unchanged.
- **Edge case (8 GB VRAM):** 122k may OOM; guidance + `n_envs`/`batch_size` knobs required (AC-4).
- **Assumption:** the RTX 3050 (sm_86) + a recent driver supports the pinned CUDA 12.x runtime; the script surfaces a clear message if the driver/toolkit is too old.
- **Note (MPS already viable):** after this UC was drafted, the user measured MPS at ~2.57s/iter (ETA 2h57m) — so CUDA is now an *optimization / bigger-VRAM* path, not the only route to tractable training. Value: dedicated 8 GB VRAM (bigger slices/batches) + robust CUDA sparse. Deliberate; documented.

## Original Description
The user has a Windows PC with an NVIDIA GeForce RTX 3050 (8 GB, Ampere/sm_86) and wants a reproducible way to train the large connectome slice on it, because CUDA (unlike the Mac's MPS) has real sparse-tensor support so the sparse connectome propagation runs on the GPU. The runtime already auto-selects CUDA (`train/device.py`); what's missing is the Windows environment setup + verification + docs. Mirror the macOS setup automation (UC-20) but for Windows/CUDA. CI is Linux with no GPU/Windows, so the GPU run is a documented manual step; CI covers the device-selection branch (monkeypatched) and script lint. The user requested this after seeing 13m41s/iter on the M4 Pro CPU.

## Clarifications
- Q: Environment manager on Windows?
  A: **uv + the PyTorch CUDA wheel index** — consistent with the project's existing uv tooling; pybullet and CUDA-build torch both ship Windows wheels.
- Q: Script form?
  A: **PowerShell script** `scripts/setup-sim-windows.ps1` — native Windows, mirrors UC-20's `setup-sim-macos.sh`.
- Q: CUDA wheel pinning?
  A: **Pin a specific CUDA 12.x PyTorch build** matched to the project's torch version + the RTX 3050, documented — reproducible.
