#!/usr/bin/env bash
# UC-80 distillation-training-fix ablation — unattended, resumable driver (AC6).
#
# Trains the 4 runs back-to-back on ONE CUDA machine: the baseline (current UC-77/79 distillation)
# + the three intervention variants (on-policy DAgger relabelling / gentler-longer β schedule /
# PPO completion-reward fine-tune). Each run is an MLP student distilled from the frozen teacher on
# the obstacle-free `student-liftoff-open` config under an EQUAL 1.8e7 total-env-step budget; every
# run then writes the UC-77 final_report (held-out + Infinity Loop, exact + noise, 32 ep, seed
# 1234). No Liftoff is needed — this is pure GPU training + evaluation in the batched simulator.
#
# ── What you must copy onto the run machine (nothing else) ────────────────────────────────────
#   1. This repository (installed: `uv sync` / `pip install -e .`, so `drone-fly` is on PATH).
#   2. The teacher run folder `teacher_it1374` (the frozen teacher the student distills from).
#   3. The detector noise model YAML, e.g.
#        artifacts/models/detector-v1/liftoff-gates-161f7a83/noise_model.yaml
#   4. The Liftoff track-conversion cache under `artifacts/tracks/cache/` (so the open courses
#      resolve without Liftoff). If absent, rebuild it once from the trackdata with
#      `drone-fly tracks ...` (see docs/experiments/uc80-distillation-training-fix.md).
#   (No Liftoff install, no GPU-vision stack — only the repo + these three artifacts.)
#
# ── Usage ─────────────────────────────────────────────────────────────────────────────────────
#   TEACHER=<teacher run folder or run id> \
#   NOISE=<path to noise_model.yaml> \
#   [DEVICE=cuda] [STATE_ROOT=artifacts/uc80-ablation] \
#   scripts/uc80_distillation_fix.sh
#
# ── Resumability (AC6) ──────────────────────────────────────────────────────────────────────
# State lives entirely on the filesystem under $STATE_ROOT/<variant>/, so the sequence is
# reboot-safe: re-run this exact command after an interruption and it
#   * SKIPS any variant that already has eval/student-held-out.json (its final_report ran), and
#   * RESUMES an interrupted variant from its run folder (the cumulative STEP budget is the
#     authoritative equalizer across resumes), and
#   * only then starts the next, never restarting the whole sequence.
# The equal total STEP budget (not wall-clock) is the across-variant equalizer — see the configs.
# For the `ppo` variant that total is split 1.2e7 DAgger + 6e6 PPO; the others spend 1.8e7 on DAgger.
set -euo pipefail

VARIANTS=(baseline onpolicy beta ppo)

: "${TEACHER:?set TEACHER to the teacher run folder or run id (e.g. teacher_it1374)}"
: "${NOISE:?set NOISE to the detector noise_model.yaml path}"
DEVICE="${DEVICE:-cuda}"
STATE_ROOT="${STATE_ROOT:-artifacts/uc80-ablation}"
DRONE_FLY="${DRONE_FLY:-drone-fly}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Newest run folder (a directory containing config.yaml) under a variant's state root, or empty.
find_run() {
  local vroot="$1"
  [ -d "$vroot" ] || return 0
  find "$vroot" -type f -name config.yaml -printf '%h\n' 2>/dev/null | sort | tail -n 1
}

echo "[uc80] repo=$REPO_ROOT teacher=$TEACHER noise=$NOISE device=$DEVICE state=$STATE_ROOT"

for v in "${VARIANTS[@]}"; do
  cfg="student-liftoff-open-uc80-$v"
  vroot="$STATE_ROOT/$v"
  run="$(find_run "$vroot")"

  if [ -n "$run" ] && [ -f "$run/eval/student-held-out.json" ]; then
    echo "[uc80] $v: COMPLETE ($run) — skip"
    continue
  fi

  if [ -n "$run" ]; then
    echo "[uc80] $v: RESUME $run"
    "$DRONE_FLY" train student --config "$cfg" --teacher "$TEACHER" \
      --noise "$NOISE" --device "$DEVICE" --resume "$run"
  else
    echo "[uc80] $v: FRESH -> $vroot"
    "$DRONE_FLY" train student --config "$cfg" --teacher "$TEACHER" \
      --noise "$NOISE" --device "$DEVICE" --output-root "$vroot"
  fi
  echo "[uc80] $v: done"
done

echo "[uc80] all 4 runs complete. Each variant's metrics are in its run folder under"
echo "       $STATE_ROOT/<variant>/<profile>/<run_id>/eval/:"
echo "         student-held-out.json        (held-out macro_completion — the generalisation target)"
echo "         student-liftoff-arena-04.json (Infinity Loop — train-split check)"
echo "       Fill the comparison table in docs/experiments/uc80-distillation-training-fix.md from"
echo "       the macro.student_noise / macro.student_exact fields and outcomes.student_noise."
