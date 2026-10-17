#!/usr/bin/env bash
# UC-79 student information-gap ablation — unattended, resumable driver (AC9).
#
# Trains the 7 diagnostic runs back-to-back on ONE CUDA machine: the baseline (no privileged
# blocks) + the 6 privileged variants. Each run is an MLP student distilled from the frozen
# teacher on the obstacle-free `student-liftoff-open` config with an opt-in privileged obs block;
# every run then writes the UC-77 final_report (held-out + Infinity Loop, exact + noise). No
# Liftoff is needed — this is pure GPU training + evaluation in the batched simulator.
#
# ── What you must copy onto the run machine (nothing else) ────────────────────────────────────
#   1. This repository (installed: `uv sync` / `pip install -e .`, so `drone-fly` is on PATH).
#   2. The teacher run folder `teacher_it1374` (the frozen teacher the student distills from).
#   3. The detector noise model YAML, e.g.
#        artifacts/models/detector-v1/liftoff-gates-161f7a83/noise_model.yaml
#   4. The Liftoff track-conversion cache under `artifacts/tracks/cache/` (so the open courses
#      resolve without Liftoff). If absent, rebuild it once from the trackdata with
#      `drone-fly tracks ...` (see docs/experiments/uc79-student-information-gap.md).
#   (No Liftoff install, no GPU-vision stack — only the repo + these three artifacts.)
#
# ── Usage ─────────────────────────────────────────────────────────────────────────────────────
#   TEACHER=<teacher run folder or run id> \
#   NOISE=<path to noise_model.yaml> \
#   [DEVICE=cuda] [STATE_ROOT=artifacts/uc79-ablation] \
#   scripts/uc79_ablation.sh
#
# ── Resumability (AC9) ──────────────────────────────────────────────────────────────────────
# State lives entirely on the filesystem under $STATE_ROOT/<variant>/, so the sequence is
# reboot-safe: re-run this exact command after an interruption and it
#   * SKIPS any variant that already has eval/student-held-out.json (its final_report ran), and
#   * RESUMES an interrupted variant from its run folder (cumulative step budget → resume-safe),
#   * only then starts the next, never restarting the whole sequence.
# The equal STEP budget (not wall-clock) is the across-variant equalizer — see the configs.
set -euo pipefail

VARIANTS=(baseline altitude floor gate_next_next gates_full surfaces teacher_full)

: "${TEACHER:?set TEACHER to the teacher run folder or run id (e.g. teacher_it1374)}"
: "${NOISE:?set NOISE to the detector noise_model.yaml path}"
DEVICE="${DEVICE:-cuda}"
STATE_ROOT="${STATE_ROOT:-artifacts/uc79-ablation}"
DRONE_FLY="${DRONE_FLY:-drone-fly}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Newest run folder (a directory containing config.yaml) under a variant's state root, or empty.
find_run() {
  local vroot="$1"
  [ -d "$vroot" ] || return 0
  find "$vroot" -type f -name config.yaml -printf '%h\n' 2>/dev/null | sort | tail -n 1
}

echo "[uc79] repo=$REPO_ROOT teacher=$TEACHER noise=$NOISE device=$DEVICE state=$STATE_ROOT"

for v in "${VARIANTS[@]}"; do
  cfg="student-liftoff-open-priv-$v"
  vroot="$STATE_ROOT/$v"
  run="$(find_run "$vroot")"

  if [ -n "$run" ] && [ -f "$run/eval/student-held-out.json" ]; then
    echo "[uc79] $v: COMPLETE ($run) — skip"
    continue
  fi

  if [ -n "$run" ]; then
    echo "[uc79] $v: RESUME $run"
    "$DRONE_FLY" train student --config "$cfg" --teacher "$TEACHER" \
      --noise "$NOISE" --device "$DEVICE" --resume "$run"
  else
    echo "[uc79] $v: FRESH -> $vroot"
    "$DRONE_FLY" train student --config "$cfg" --teacher "$TEACHER" \
      --noise "$NOISE" --device "$DEVICE" --output-root "$vroot"
  fi
  echo "[uc79] $v: done"
done

echo "[uc79] all 7 variants complete. Build the comparison table with:"
echo "       python scripts/uc79_table.py $STATE_ROOT"
