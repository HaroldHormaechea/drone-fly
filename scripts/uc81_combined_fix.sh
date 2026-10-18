#!/usr/bin/env bash
# UC-81 combined-distillation-fix ablation — unattended, resumable driver (AC5).
#
# Trains the 3 runs back-to-back on ONE CUDA machine: the baseline (current UC-77/79/80
# distillation) + the two UC-81 interventions:
#   * combined     — on-policy relabelling + the β-floor together (both levers active), the
#                    UC-80 doc's named follow-up; 1.8e7 DAgger.
#   * combined-ppo — the combined imitation setup, THEN an extended PPO fine-tune (finetune.steps
#                    3e7) against the env reward augmented with the opt-in altitude-hold shaping
#                    term (env.rewards.floor_clearance 0.03). Total ≈ 1.8e7 + 3e7 = 4.8e7.
# Each run is an MLP student distilled from the frozen teacher on the obstacle-free
# `student-liftoff-open` config; every run then writes the UC-77 final_report (held-out + Infinity
# Loop, exact + noise, 32 ep, seed 1234). No Liftoff is needed — pure GPU training + evaluation in
# the batched simulator.
#
# NOTE ON BUDGET (AC4 — equal-budget deliberately RELAXED): baseline and combined both spend 1.8e7
# on DAgger; combined-ppo adds a 3e7 PPO phase on top (~4.8e7 total), because UC-80 proved the
# short equal-budget PPO arm undertrains. The combined-ppo result therefore bundles TWO changes
# (extended PPO budget AND the floor-clearance reward) — a documented confound; see the experiment
# doc docs/experiments/uc81-combined-distillation-fix.md.
#
# ── What you must copy onto the run machine (nothing else) ────────────────────────────────────
#   1. This repository (installed: `uv sync` / `pip install -e .`, so `drone-fly` is on PATH).
#   2. The teacher run folder `teacher_it1374` (the frozen teacher the student distills from).
#   3. The detector noise model YAML, e.g.
#        artifacts/models/detector-v1/liftoff-gates-161f7a83/noise_model.yaml
#   4. The Liftoff track-conversion cache under `artifacts/tracks/cache/` (so the open courses
#      resolve without Liftoff). If absent, rebuild it once from the trackdata with
#      `drone-fly tracks ...` (see docs/experiments/uc81-combined-distillation-fix.md).
#   (No Liftoff install, no GPU-vision stack — only the repo + these three artifacts.)
#
# ── Usage ─────────────────────────────────────────────────────────────────────────────────────
#   TEACHER=<teacher run folder or run id> \
#   NOISE=<path to noise_model.yaml> \
#   [DEVICE=cuda] [STATE_ROOT=artifacts/uc81-ablation] \
#   scripts/uc81_combined_fix.sh
#
# ── Resumability (AC5) ──────────────────────────────────────────────────────────────────────
# State lives entirely on the filesystem under $STATE_ROOT/<variant>/, so the sequence is
# reboot-safe: re-run this exact command after an interruption and it
#   * SKIPS any variant that already has eval/student-held-out.json (its final_report ran), and
#   * RESUMES an interrupted variant from its run folder (the cumulative STEP budget — DAgger then
#     PPO — is the authoritative progress marker across resumes), and
#   * only then starts the next, never restarting the whole sequence.
set -euo pipefail

VARIANTS=(baseline combined combined-ppo)

: "${TEACHER:?set TEACHER to the teacher run folder or run id (e.g. teacher_it1374)}"
: "${NOISE:?set NOISE to the detector noise_model.yaml path}"
DEVICE="${DEVICE:-cuda}"
STATE_ROOT="${STATE_ROOT:-artifacts/uc81-ablation}"
DRONE_FLY="${DRONE_FLY:-drone-fly}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Newest run folder (a directory containing config.yaml) under a variant's state root, or empty.
find_run() {
  local vroot="$1"
  [ -d "$vroot" ] || return 0
  find "$vroot" -type f -name config.yaml -printf '%h\n' 2>/dev/null | sort | tail -n 1
}

echo "[uc81] repo=$REPO_ROOT teacher=$TEACHER noise=$NOISE device=$DEVICE state=$STATE_ROOT"

for v in "${VARIANTS[@]}"; do
  cfg="student-liftoff-open-uc81-$v"
  vroot="$STATE_ROOT/$v"
  run="$(find_run "$vroot")"

  if [ -n "$run" ] && [ -f "$run/eval/student-held-out.json" ]; then
    echo "[uc81] $v: COMPLETE ($run) — skip"
    continue
  fi

  if [ -n "$run" ]; then
    echo "[uc81] $v: RESUME $run"
    "$DRONE_FLY" train student --config "$cfg" --teacher "$TEACHER" \
      --noise "$NOISE" --device "$DEVICE" --resume "$run"
  else
    echo "[uc81] $v: FRESH -> $vroot"
    "$DRONE_FLY" train student --config "$cfg" --teacher "$TEACHER" \
      --noise "$NOISE" --device "$DEVICE" --output-root "$vroot"
  fi
  echo "[uc81] $v: done"
done

echo "[uc81] all 3 runs complete. Each variant's metrics are in its run folder under"
echo "       $STATE_ROOT/<variant>/<profile>/<run_id>/eval/:"
echo "         student-held-out.json         (held-out macro_completion — the generalisation target)"
echo "         student-liftoff-arena-04.json (Infinity Loop — train-split check)"
echo "       Fill the comparison table in docs/experiments/uc81-combined-distillation-fix.md from"
echo "       the macro.student_noise / macro.student_exact fields and outcomes.student_noise."
echo "       For combined-ppo also report the PPO reward/step curve (logs/) as a hover-hack /"
echo "       undertraining sanity check — see the experiment doc."
