#!/usr/bin/env pwsh
#
# setup-sim-windows.ps1 — UC-31: provision the drone-fly real-physics ("mastery") training
# stack on Windows with an NVIDIA CUDA GPU (e.g. an RTX 3050, 8 GB, Ampere/sm_86).
#
# Why this exists: scripts/train.sh's Linux uv path source-builds pybullet and installs a
# default (CPU/PyPI) torch — neither gives Windows a CUDA GPU build. This script is the
# Windows mirror of scripts/setup-sim-macos.sh (UC-20): it provisions an isolated uv venv,
# installs a PINNED CUDA 12.x PyTorch wheel from the PyTorch CUDA wheel index, then pybullet
# + gym-pybullet-drones (at the exact commit pinned in scripts/train.sh — single source of
# truth) + the drone-fly project, and finally verifies torch.cuda + the sim actually import.
#
# The CUDA wheel --index-url is confined to the torch line ONLY, and this file NEVER touches
# pyproject.toml / uv.lock — so the default Linux/macOS install and the hermetic Linux CI
# resolve stay byte-for-byte unchanged (UC-31 AC-7). It is an opt-in Windows path.
#
# Because CI is Linux with no GPU/Windows, the real CUDA run is a DOCUMENTED MANUAL step; the
# CI-testable slices are the device-selection CUDA branch (monkeypatched) and this script's
# lint/parse. See the README "Windows (NVIDIA CUDA) real-physics training" section.
#
# Usage:
#   ./scripts/setup-sim-windows.ps1              provision the env (prompts before installing uv)
#   ./scripts/setup-sim-windows.ps1 -DryRun      print the plan; touch no network / install / prompt
#   ./scripts/setup-sim-windows.ps1 -Help        show this help
#
# Env overrides:
#   DRONE_FLY_SIM_DRYRUN=1   same as -DryRun
#   DRONE_FLY_SIM_PYTHON     env Python version (default: 3.12; gym-pybullet-drones needs >=3.12)
#   DRONE_FLY_VENV           venv directory (default: .venv-cuda)
#
# Requires: PowerShell 5.1+ (or PowerShell 7 / pwsh). Runs entirely in user space (no admin).

[CmdletBinding()]
param(
    [switch]$DryRun,
    [Alias('h')]
    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Pinned, isolated CUDA torch build (UC-31). ------------------------------------------
# CUDA 12.4 wheels cover the RTX 3050 (Ampere/sm_86) and a recent driver. This pin lives
# ONLY here (never pyproject.toml / uv.lock), keeping the CUDA path opt-in. The requirement
# is compatibility with the project's stable-baselines3 (2.9.x) + gymnasium (1.3.x) pins;
# torch 2.6.0+cu124 ships a cp312 win_amd64 wheel and satisfies both.
$TorchPin   = 'torch==2.6.0'
$TorchIndex = 'https://download.pytorch.org/whl/cu124'

function Write-Log {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '',
        Justification = 'User-facing colored setup output, mirroring setup-sim-macos.sh.')]
    param([Parameter(Mandatory)][string]$Message)
    Write-Host "[setup-sim-windows.ps1] $Message" -ForegroundColor Blue
}

function Write-Fail {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '',
        Justification = 'User-facing colored error output, mirroring setup-sim-macos.sh.')]
    param([Parameter(Mandatory)][string]$Message)
    Write-Host "[setup-sim-windows.ps1] ERROR: $Message" -ForegroundColor Red
    exit 1
}

function Show-Usage {
    Write-Output @'
setup-sim-windows.ps1 — provision the drone-fly real-physics + CUDA training stack on Windows.

  ./scripts/setup-sim-windows.ps1              provision the env (prompts before installing uv)
  ./scripts/setup-sim-windows.ps1 -DryRun      print the plan; touch no network / install / prompt
  ./scripts/setup-sim-windows.ps1 -Help        show this help

Env overrides:
  DRONE_FLY_SIM_DRYRUN=1   same as -DryRun
  DRONE_FLY_SIM_PYTHON     env Python version (default: 3.12)
  DRONE_FLY_VENV           venv directory (default: .venv-cuda)
'@
}

if ($Help) { Show-Usage; exit 0 }

# --- Config + env overrides. ------------------------------------------------------------
$SimPython = if ($env:DRONE_FLY_SIM_PYTHON) { $env:DRONE_FLY_SIM_PYTHON } else { '3.12' }
$Venv      = if ($env:DRONE_FLY_VENV) { $env:DRONE_FLY_VENV } else { '.venv-cuda' }
if ($env:DRONE_FLY_SIM_DRYRUN -eq '1') { $DryRun = $true }

$ScriptDir = $PSScriptRoot
$RepoRoot  = Split-Path -Parent $ScriptDir
$TrainSh   = Join-Path $ScriptDir 'train.sh'

# --- Single-source the gym-pybullet-drones pin from scripts/train.sh. --------------------
# NEVER a second 40-hex SHA literal here — parse it out of train.sh (the one source of truth).
if (-not (Test-Path -LiteralPath $TrainSh)) {
    Write-Fail "Cannot find scripts/train.sh at $TrainSh — the sim pin is single-sourced from it."
}
$repoMatch = Select-String -LiteralPath $TrainSh -Pattern '^PYBULLET_DRONES_REPO="([^"]*)"' |
    Select-Object -First 1
$refMatch = Select-String -LiteralPath $TrainSh -Pattern '^PYBULLET_DRONES_REF="([^"]*)"' |
    Select-Object -First 1
if (-not $repoMatch) { Write-Fail "Could not read PYBULLET_DRONES_REPO from $TrainSh (single source of the pin)." }
if (-not $refMatch)  { Write-Fail "Could not read PYBULLET_DRONES_REF from $TrainSh (single source of the pin)." }
$DronesRepo = $repoMatch.Matches[0].Groups[1].Value
$DronesRef  = $refMatch.Matches[0].Groups[1].Value
$DronesGit  = "git+$DronesRepo@$DronesRef"

# gym-pybullet-drones runtime deps installed EXPLICITLY (because --no-deps skips them, and
# because its own metadata pins pybullet>3.2.7 — which would clobber our pinned 3.2.6). The
# post-install import check (AC-2) is the backstop. Mirrors setup-sim-macos.sh's RUNTIME_DEPS.
$RuntimeDeps = @(
    'gymnasium>=1.3,<2',
    'stable-baselines3>=2.9,<3',
    'transforms3d>=0.4,<0.5',
    'control>=0.10.2,<0.11',
    'matplotlib',
    'pillow',
    'scipy',
    'numpy<2'
)

# --- Dry run: print the plan; NO network / install / prompt work. ------------------------
if ($DryRun) {
    Write-Log 'DRY RUN — no network, install, or prompts. Planned actions:'
    Write-Log "  repo root:              $RepoRoot"
    Write-Log "  venv:                   $Venv (Python $SimPython)"
    Write-Log "  torch pin (CUDA):       $TorchPin  via --index-url $TorchIndex"
    Write-Log "  drones pin (train.sh):  $DronesRepo @ $DronesRef"
    Write-Log '  1. ensure uv (prompt before installing into user space via astral.sh; never admin)'
    Write-Log '  2. check the NVIDIA driver via nvidia-smi (reported CUDA Version must be >= 12)'
    Write-Log "  3. create/reuse the uv venv '$Venv' (Python $SimPython)"
    Write-Log "  4. uv pip install $TorchPin --index-url $TorchIndex   (CUDA index scoped to torch ONLY)"
    Write-Log '  5. uv pip install "pybullet==3.2.6" "numpy<2" -e ".[dev]"   (one PyPI resolve; numpy<2 ABI co-satisfied; torch already installed)'
    Write-Log "  6. uv pip install --no-deps $DronesGit   (so its pybullet>3.2.7 pin never clobbers 3.2.6)"
    Write-Log '  7. uv pip install <gym-pybullet-drones runtime deps>   (explicit; numpy<2 pinned)'
    Write-Log '  8. verify: torch.cuda.is_available(), GPU name, import pybullet/gym_pybullet_drones/drone_fly, resolve_device()=="cuda"'
    Write-Log '  NOTE: pyproject.toml / uv.lock are never touched — the CUDA path stays opt-in (AC-7).'
    Write-Log '  NOTE: pybullet 3.2.6 ships no Windows/py3.12 wheel, so it builds from source; install Microsoft C++ Build Tools if it fails.'
    exit 0
}

Set-Location -LiteralPath $RepoRoot

# --- 1. Ensure uv (prompt before installing; user space; never admin). -------------------
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Log 'uv is not installed. It is required to create the venv and install packages.'
    Write-Log 'This installs uv into your user profile (no admin, no system-wide changes).'
    $reply = Read-Host '[setup-sim-windows.ps1] Install uv now? Type "yes" to proceed'
    if ($reply -ne 'yes') {
        Write-Fail 'Declined uv install — nothing was installed. Install uv yourself (https://docs.astral.sh/uv/) and re-run, or re-run and type "yes".'
    }
    $installer = Join-Path ([System.IO.Path]::GetTempPath()) 'uv-install.ps1'
    Write-Log 'Downloading the uv installer from https://astral.sh/uv/install.ps1 ...'
    Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1' -OutFile $installer
    Write-Log 'Running the uv installer (user space) ...'
    & $installer
    # The installer adds uv to PATH for new sessions; make it usable in THIS session too.
    $uvBin = Join-Path $env:USERPROFILE '.local\bin'
    if (Test-Path -LiteralPath (Join-Path $uvBin 'uv.exe')) { $env:PATH = "$uvBin;$env:PATH" }
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCmd) {
        Write-Fail "uv installed but is not on PATH in this session. Open a new terminal (so PATH refreshes) and re-run ./scripts/setup-sim-windows.ps1."
    }
}
Write-Log "Using uv: $($uvCmd.Source)"

# --- 2. Driver check: nvidia-smi must report a CUDA runtime >= 12.x. ---------------------
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    Write-Fail 'nvidia-smi not found. Install a current NVIDIA GPU driver (which provides nvidia-smi) for your RTX card, then re-run. The driver bundles the CUDA runtime the torch wheel needs.'
}
$smiOut = & nvidia-smi
if ($LASTEXITCODE -ne 0) {
    Write-Fail 'nvidia-smi failed to run. Check that the NVIDIA driver is installed and the GPU is visible, then re-run.'
}
$cudaLine = $smiOut | Select-String -Pattern 'CUDA Version:\s*([0-9]+)\.([0-9]+)' | Select-Object -First 1
if (-not $cudaLine) {
    Write-Log 'WARNING: could not parse a "CUDA Version" from nvidia-smi output; continuing. The post-install torch.cuda check is the backstop.'
} else {
    $cudaMajor = [int]$cudaLine.Matches[0].Groups[1].Value
    $cudaMinor = [int]$cudaLine.Matches[0].Groups[2].Value
    if ($cudaMajor -lt 12) {
        Write-Fail "Driver reports CUDA $cudaMajor.$cudaMinor, but this setup needs a CUDA 12.x-capable driver (the torch build targets cu124). Update your NVIDIA driver to a recent (CUDA 12.x) release and re-run."
    }
    Write-Log "NVIDIA driver reports CUDA $cudaMajor.$cudaMinor (>= 12.x) — OK."
}

# --- 3. Create/reuse the uv venv. -------------------------------------------------------
$PythonExe = Join-Path $Venv 'Scripts\python.exe'
if (Test-Path -LiteralPath $PythonExe) {
    Write-Log "Reusing existing venv '$Venv'."
} else {
    Write-Log "Creating venv '$Venv' (Python $SimPython) ..."
    & uv venv --python $SimPython $Venv
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Could not create a Python $SimPython venv. Install it first: uv python install $SimPython"
    }
}

# --- 4. torch FIRST — CUDA wheel index scoped to this line ONLY. -------------------------
Write-Log "Installing $TorchPin from the CUDA wheel index ($TorchIndex) ..."
& uv pip install --python $Venv $TorchPin --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Failed to install $TorchPin from $TorchIndex. Check network access to download.pytorch.org, or that the pinned CUDA torch build still exists for your Python ($SimPython)."
}

# --- 5. Single unified PyPI resolve for the rest (numpy<2 ABI co-satisfies pybullet). ----
# No --index-url here (PyPI only); torch is already installed so it stays the cu124 build.
# The gym-pybullet-drones git URL is deliberately NOT in this resolve: its metadata pins
# pybullet>3.2.7, which would conflict with our pinned pybullet==3.2.6. It goes in step 6
# with --no-deps instead.
Write-Log 'Installing pybullet==3.2.6 + numpy<2 + drone-fly (editable, [dev]) in one PyPI resolve ...'
& uv pip install --python $Venv 'pybullet==3.2.6' 'numpy<2' -e '.[dev]'
if ($LASTEXITCODE -ne 0) {
    Write-Fail 'Failed the unified PyPI install. pybullet 3.2.6 has no Windows/py3.12 wheel, so it builds from source: install "Microsoft C++ Build Tools" (Desktop development with C++) and re-run. See the README Windows/CUDA section.'
}

# --- 6. gym-pybullet-drones with --no-deps (keeps our pinned pybullet 3.2.6). ------------
Write-Log "Installing gym-pybullet-drones @ $DronesRef with --no-deps (so its pybullet>3.2.7 pin never clobbers 3.2.6) ..."
& uv pip install --python $Venv --no-deps $DronesGit
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Failed to install gym-pybullet-drones @ $DronesRef (--no-deps)."
}

# --- 7. Its runtime deps, explicitly (because --no-deps skipped them). -------------------
Write-Log 'Installing gym-pybullet-drones runtime deps explicitly (hand-maintained; the import check is the backstop) ...'
& uv pip install --python $Venv @RuntimeDeps
if ($LASTEXITCODE -ne 0) {
    Write-Fail 'Failed to install the explicit gym-pybullet-drones runtime deps.'
}

# --- 8. Verify torch.cuda + the sim actually import (AC-2 backstop). ---------------------
Write-Log 'Verifying torch.cuda + sim imports (torch.cuda.is_available, GPU name, pybullet, gym-pybullet-drones, drone_fly, resolve_device) ...'
$verify = @'
import torch

assert torch.cuda.is_available(), (
    "torch.cuda.is_available() is False. The GPU is not visible to this torch build. "
    "Check your NVIDIA driver (nvidia-smi) and that the CUDA torch wheel installed correctly."
)
print("torch:", torch.__version__)
print("CUDA device:", torch.cuda.get_device_name(0))

import pybullet
import gym_pybullet_drones  # noqa: F401
import drone_fly  # noqa: F401
from drone_fly.train.device import CUDA_OOM_HINT, resolve_device

dev = resolve_device()
assert dev == "cuda", f"resolve_device() returned {dev!r}, expected 'cuda'."
print("resolve_device() ->", dev)
print("pybullet API:", pybullet.getAPIVersion())
print(CUDA_OOM_HINT)
print("verify OK")
'@
$verify | & $PythonExe -
if ($LASTEXITCODE -ne 0) {
    Write-Fail 'Post-install verification failed. torch.cuda or a sim import did not come up cleanly — see the error above. Do NOT train against a broken stack.'
}

Write-Output ''
Write-Log "Done. The '$Venv' venv has CUDA torch ($TorchPin) + the real-physics sim."
Write-Log "Set 'device: cuda' in your train config (or omit it — resolve_device() auto-selects CUDA), then:"
Write-Log "  $Venv\Scripts\drone-fly train --config configs\train\example.yaml"
Write-Log 'Benchmark one iteration (watch the device log + SB3 time_elapsed / fps lines; compare to the ~821 s CPU / ~2.57 s-per-iter MPS baselines):'
Write-Log "  $Venv\Scripts\drone-fly train --config configs\train\example.yaml   # config: device: cuda, small timesteps"
