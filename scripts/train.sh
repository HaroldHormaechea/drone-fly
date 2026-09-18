#!/usr/bin/env bash
#
# One-command bootstrap for the UC-03 mastery training run (AC7).
#
# Creates an isolated virtualenv, installs the project + the optional simulator stack
# (pybullet + the GitHub-only gym-pybullet-drones, pinned to an exact commit), verifies the
# native pybullet extension imports, then launches (or RESUMES) PPO training.
#
# Usage:
#   ./scripts/train.sh [CONFIG]
# CONFIG defaults to configs/train/example.yaml. All run settings — including the run name,
# connectome, timesteps, and resume policy — live in that YAML (UC-11 config-driven CLI).
#
# Idempotent: the config's `resume: auto` continues from the latest checkpoint under
# training/<name>/checkpoints/ if one exists, otherwise starts fresh — so re-running this
# script resumes rather than restarting, with no bash-side checkpoint probing. A failed sim
# install stops with an actionable message — never a cryptic mid-training crash.
#
# This script targets the owner's macOS Apple-Silicon (M4 Pro) dev machine, which has a
# C/C++ toolchain (Xcode Command Line Tools). See the README "Simulator bootstrap:
# tested-vs-untested boundary" table for exactly what was verified in the Linux CI sandbox
# vs deferred to macOS.
#
set -euo pipefail

# --- Pinned, reproducible simulator ref (NEVER floating `main`). --------------------------
# gym-pybullet-drones v2.2.0 (gymnasium-native line, matching our gymnasium/SB3 pins).
# Resolved with `git ls-remote https://github.com/utiasDSL/gym-pybullet-drones main` at
# authoring time; re-verified below at runtime before install.
PYBULLET_DRONES_REPO="https://github.com/utiasDSL/gym-pybullet-drones"
PYBULLET_DRONES_REF="7ebad1ecabd28a7000add2d05f888aa2e837c2cc"  # v2.2.0

# Sim needs Python >= 3.12 (gym-pybullet-drones v2.2.0 requires-python ^3.12).
SIM_PYTHON="${DRONE_FLY_SIM_PYTHON:-3.12}"
VENV_DIR="${DRONE_FLY_VENV:-.venv-sim}"

# Config-driven run (UC-11): everything the run needs lives in this YAML.
CONFIG="${1:-configs/train/example.yaml}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '\033[1;34m[train.sh]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[train.sh] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || fail \
  "uv is required. Install it: https://docs.astral.sh/uv/  (curl -LsSf https://astral.sh/uv/install.sh | sh)"

# --- 1. Re-verify the pin actually resolves before we rely on it. -------------------------
log "Resolving pinned gym-pybullet-drones ref via git ls-remote ..."
if ! git ls-remote "$PYBULLET_DRONES_REPO" | grep -q "$PYBULLET_DRONES_REF"; then
  # A pinned commit SHA is not always advertised by ls-remote (only refs are). Fall back to
  # confirming the repo is reachable; the exact SHA is enforced by the git+ URL at install.
  git ls-remote "$PYBULLET_DRONES_REPO" >/dev/null 2>&1 \
    || fail "Cannot reach $PYBULLET_DRONES_REPO — check your network. Pin was $PYBULLET_DRONES_REF."
  log "Repo reachable; exact SHA $PYBULLET_DRONES_REF enforced by the git+ URL at install."
else
  log "Pin $PYBULLET_DRONES_REF resolved."
fi

# --- 2. Create the venv + install project and simulator stack. ---------------------------
if [ ! -d "$VENV_DIR" ]; then
  log "Creating virtualenv ($VENV_DIR) with Python $SIM_PYTHON ..."
  uv venv --python "$SIM_PYTHON" "$VENV_DIR" \
    || fail "Could not create a Python $SIM_PYTHON venv. Install it (e.g. 'uv python install $SIM_PYTHON')."
fi

log "Installing project + dev + sim (pybullet) into $VENV_DIR ..."
uv pip install --python "$VENV_DIR" -e ".[dev,sim]" \
  || fail "Failed to install the project + pybullet. pybullet has a native extension: on a \
source build you need a C/C++ toolchain (macOS: 'xcode-select --install'; Debian/Ubuntu: \
'apt-get install build-essential')."

log "Installing gym-pybullet-drones @ $PYBULLET_DRONES_REF (GitHub-only, pinned) ..."
uv pip install --python "$VENV_DIR" \
  "git+${PYBULLET_DRONES_REPO}@${PYBULLET_DRONES_REF}" \
  || fail "Failed to install gym-pybullet-drones @ $PYBULLET_DRONES_REF. See the README \
'Simulator bootstrap' boundary table for the known flat-layout/toolchain pitfalls."

# --- 3. Verify the native sim actually imports (AC7 hard gate). ---------------------------
log "Verifying pybullet + gym-pybullet-drones import ..."
"$VENV_DIR/bin/python" - <<'PY' || fail "pybullet/gym-pybullet-drones failed to import. Do NOT train against a broken sim. \
Fix the install (C/C++ toolchain for pybullet's native extension) and re-run ./scripts/train.sh."
import pybullet          # native extension must load
import gym_pybullet_drones  # noqa: F401
print("sim import OK:", pybullet.getAPIVersion())
PY

# --- 4. Verify the config exists before launching. ----------------------------------------
[ -f "$CONFIG" ] || fail "Config file not found: $CONFIG. Pass one as the first argument, or \
copy configs/train/example.yaml and edit it. Idempotent resume comes from the config's \
'resume: auto' key (no bash-side checkpoint probing)."

# --- 5. Launch training. ------------------------------------------------------------------
# All settings (name, connectome, timesteps, device, resume policy, ...) come from the YAML.
# `resume: auto` in the config gives the idempotent continue-or-fresh behaviour this script
# used to implement in bash.
log "Launching training with config: $CONFIG ..."
exec "$VENV_DIR/bin/drone-fly" train --config "$CONFIG"
