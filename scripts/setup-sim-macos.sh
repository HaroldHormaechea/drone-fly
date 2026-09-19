#!/usr/bin/env bash
#
# setup-sim-macos.sh — UC-20: provision the PyBullet real-physics ("mastery") sim on modern
# macOS (26 / Tahoe, Apple Silicon) from PREBUILT binaries only.
#
# Why this exists: scripts/train.sh installs pybullet FROM SOURCE into a uv venv. On the
# macOS 26 SDK, pybullet's vendored zlib (`zutil.h`'s `#define fdopen(fd,mode) NULL`) clashes
# with the SDK's `_stdio.h` fdopen declaration, so NO from-source pybullet (3.2.6, or the
# 3.2.7 that gym-pybullet-drones pins) can compile — regardless of Xcode CLT/clang version.
# This script sidesteps the compiler entirely: miniforge -> conda-forge PREBUILT `pybullet`
# -> gym-pybullet-drones via `pip install --no-deps` -> the `drone-fly` console script.
#
# It NEVER attempts a source build. If the prebuilt path cannot be satisfied it fails fast
# and prints a ready-to-run Linux-container command (Apple `container` / Docker) instead.
#
# The gym-pybullet-drones repo + pinned commit are read out of scripts/train.sh (single
# source of truth) — this file never duplicates the pin.
#
# Usage:
#   ./scripts/setup-sim-macos.sh            provision the sim (prompts before installing miniforge)
#   ./scripts/setup-sim-macos.sh --dry-run  print the plan; touch no network / conda / prompt
#   ./scripts/setup-sim-macos.sh -h|--help  show this help
#
# Env overrides:
#   DRONE_FLY_SIM_DRYRUN=1   same as --dry-run
#   DRONE_FLY_CONDA_ENV      conda env name (default: dronefly)
#   DRONE_FLY_SIM_PYTHON     env Python version (default: 3.12)
#
# bash 3.2-safe (macOS /bin/bash): no associative arrays, ${x^^}, or mapfile.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_SH="$SCRIPT_DIR/train.sh"

log() { printf '\033[1;34m[setup-sim-macos.sh]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[setup-sim-macos.sh] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
setup-sim-macos.sh — provision the PyBullet real-physics sim on modern macOS from PREBUILT binaries.

  ./scripts/setup-sim-macos.sh            provision the sim (prompts before installing miniforge)
  ./scripts/setup-sim-macos.sh --dry-run  print the plan; touch no network / conda / prompt
  ./scripts/setup-sim-macos.sh -h|--help  show this help

Env overrides:
  DRONE_FLY_SIM_DRYRUN=1   same as --dry-run
  DRONE_FLY_CONDA_ENV      conda env name (default: dronefly)
  DRONE_FLY_SIM_PYTHON     env Python version (default: 3.12)
EOF
}

CONDA_ENV="${DRONE_FLY_CONDA_ENV:-dronefly}"
SIM_PYTHON="${DRONE_FLY_SIM_PYTHON:-3.12}"
DRYRUN="${DRONE_FLY_SIM_DRYRUN:-0}"

# --- Args -------------------------------------------------------------------------------
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRYRUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1 (try --help)." ;;
  esac
  shift
done

# --- Single-source the pin from scripts/train.sh (never a second SHA literal here). ------
[ -f "$TRAIN_SH" ] \
  || fail "Cannot find scripts/train.sh at $TRAIN_SH — the sim pin is single-sourced from it."
# Portable BRE: capture the double-quoted value, ignore any trailing comment.
REPO="$(sed -n 's/^PYBULLET_DRONES_REPO="\([^"]*\)".*/\1/p' "$TRAIN_SH")"
REF="$(sed -n 's/^PYBULLET_DRONES_REF="\([^"]*\)".*/\1/p' "$TRAIN_SH")"
[ -n "$REPO" ] || fail "Could not read PYBULLET_DRONES_REPO from $TRAIN_SH (single source of the pin)."
[ -n "$REF" ]  || fail "Could not read PYBULLET_DRONES_REF from $TRAIN_SH (single source of the pin)."
DRONES_GIT="git+${REPO}@${REF}"

# gym-pybullet-drones runtime deps, installed EXPLICITLY (because --no-deps skips them).
# Hand-maintained to mirror the validated recipe; the AC4 import check is the backstop.
RUNTIME_DEPS=(
  gymnasium'>=1.3,<2'
  stable-baselines3'>=2.9,<3'
  transforms3d'>=0.4,<0.5'
  control'>=0.10.2,<0.11'
  matplotlib
  pillow
  scipy
  pytest
)

# --- Fail-fast escape hatch: steer to a Linux container; NEVER a source build. -----------
steer_to_container() {
  local reason="$1"
  log "Native prebuilt path failed: $reason"
  log "NOT attempting a source build (impossible on modern macOS SDKs — see UC-20)."
  log "Run the sim in a Linux container instead (prebuilt manylinux pybullet wheel from PyPI):"
  cat >&2 <<EOF

  # Apple 'container' (macOS 26):
  container run --rm -it -v "$REPO_ROOT":/work -w /work python:3.12 \\
    bash -lc "pip install pybullet '$DRONES_GIT' && pip install -e .[dev] && drone-fly train --config configs/train/example.yaml"

  # Docker fallback:
  docker run --rm -it -v "$REPO_ROOT":/work -w /work python:3.12 \\
    bash -lc "pip install pybullet '$DRONES_GIT' && pip install -e .[dev] && drone-fly train --config configs/train/example.yaml"

EOF
  fail "macOS native sim setup could not complete; use the container command above."
}

# --- Dry run: print the plan; NO network / conda / prompt work. --------------------------
if [ "$DRYRUN" = "1" ]; then
  log "DRY RUN — no network, conda, or prompts. Planned actions:"
  log "  repo root:            $REPO_ROOT"
  log "  conda env:            $CONDA_ENV (Python $SIM_PYTHON)"
  log "  pin (from train.sh):  $REPO @ $REF"
  log "  1. ensure conda (prompt before installing miniforge into ~/miniforge3; never sudo)"
  log "  2. create/reuse conda env '$CONDA_ENV' (Python $SIM_PYTHON)"
  log "  3. conda install -c conda-forge pybullet   (PREBUILT — the compiler is never invoked)"
  log "  4. pip install --no-deps $DRONES_GIT"
  log "  5. pip install <runtime deps> then pip install -e \".[dev]\"   (NOT .[dev,sim])"
  log "  6. verify imports: pybullet, gym_pybullet_drones (CtrlAviary/DroneModel/Physics), drone_fly"
  log "  On any prebuilt/verify failure: fail fast + print the Linux-container command (never a source build)."
  exit 0
fi

cd "$REPO_ROOT"

# --- 1. Detect conda (command -v, then $CONDA_EXE, then ~/miniforge3). -------------------
CONDA_BIN=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE:-}" ]; then
  CONDA_BIN="$CONDA_EXE"
elif [ -x "$HOME/miniforge3/bin/conda" ]; then
  CONDA_BIN="$HOME/miniforge3/bin/conda"
fi

# --- 2. AC2 — prompt before installing miniforge; never silent, never sudo. --------------
if [ -z "$CONDA_BIN" ]; then
  log "No conda / miniforge found on this machine."
  log "The prebuilt sim path needs a conda env with conda-forge's prebuilt pybullet."
  log "This will download miniforge into $HOME/miniforge3 (user-space; no sudo, no system changes)."
  printf '[setup-sim-macos.sh] Install miniforge now? Type "yes" to proceed: '
  reply=""
  read -r reply || reply=""
  if [ "$reply" != "yes" ]; then
    fail "Declined miniforge install — nothing was installed. Install conda/miniforge yourself, or re-run and type 'yes'."
  fi
  arch="$(uname -m)"
  case "$arch" in
    arm64)  installer="Miniforge3-MacOSX-arm64.sh" ;;
    x86_64) installer="Miniforge3-MacOSX-x86_64.sh" ;;
    *) fail "Unsupported macOS arch: $arch (expected arm64 or x86_64)." ;;
  esac
  url="https://github.com/conda-forge/miniforge/releases/latest/download/$installer"
  tmp="${TMPDIR:-/tmp}/$installer"
  log "Downloading $installer ..."
  curl -fsSLo "$tmp" "$url" || fail "Failed to download miniforge from $url"
  log "Installing miniforge (batch, user-space) into $HOME/miniforge3 ..."
  bash "$tmp" -b -p "$HOME/miniforge3" || fail "miniforge install failed."
  CONDA_BIN="$HOME/miniforge3/bin/conda"
  [ -x "$CONDA_BIN" ] || fail "miniforge installed but $CONDA_BIN is missing."
fi
log "Using conda: $CONDA_BIN"

# --- 3. AC3 — create/reuse the env, install PREBUILT pybullet + gym-pybullet-drones. -----
env_exists=0
if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  env_exists=1
fi
if [ "$env_exists" = "1" ]; then
  log "Reusing existing conda env '$CONDA_ENV'."
else
  log "Creating conda env '$CONDA_ENV' (Python $SIM_PYTHON) ..."
  "$CONDA_BIN" create -y -n "$CONDA_ENV" "python=$SIM_PYTHON" \
    || fail "Failed to create conda env '$CONDA_ENV'."
fi

log "Installing PREBUILT pybullet from conda-forge (the compiler is never invoked) ..."
"$CONDA_BIN" install -y -n "$CONDA_ENV" -c conda-forge pybullet \
  || steer_to_container "conda-forge pybullet install failed"

log "Installing gym-pybullet-drones @ $REF with --no-deps (so pip never source-builds pybullet) ..."
"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" pip install --no-deps "$DRONES_GIT" \
  || fail "Failed to install gym-pybullet-drones @ $REF (--no-deps)."

log "Installing gym-pybullet-drones runtime deps explicitly (hand-maintained; AC4 verify is the backstop) ..."
"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" pip install "${RUNTIME_DEPS[@]}" \
  || fail "Failed to install the explicit runtime deps."

log "Installing drone-fly (editable) + dev extras from $REPO_ROOT ..."
"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" pip install -e ".[dev]" \
  || fail "Failed to install drone-fly (editable) + dev extras."

# --- 4. AC4 — verify the sim actually imports; steer to container on failure. ------------
log "Verifying the sim imports (pybullet, gym-pybullet-drones, drone_fly) ..."
"$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python - <<'PY' \
  || steer_to_container "sim import verification failed"
import pybullet
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary  # noqa: F401
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: F401
import drone_fly  # noqa: F401

print("sim import OK: pybullet API", pybullet.getAPIVersion())
print("Using PyBullet backend (mastery physics).")
PY

printf '\n'
log "Done. The '$CONDA_ENV' conda env has the real-physics sim (prebuilt pybullet)."
log "Train with it:"
log "  ./scripts/train.sh            (auto-detects macOS and uses this env)"
log "or manually:"
log "  $CONDA_BIN run --no-capture-output -n $CONDA_ENV drone-fly train --config configs/train/example.yaml"
