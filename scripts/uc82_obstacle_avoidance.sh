#!/usr/bin/env bash
# UC-82 — obstacle-avoidance for the 70-input connectome, end to end (sim-only, no Liftoff).
#
# Steps (each skipped if its output already exists, so a reboot / re-run resumes):
#   0  build the obstacle-ray noise model      -> artifacts/eyes/ray-noise.yaml
#   1  retrain the 70-input model UNDER ray noise (AC3)        -> rays checkpoint
#   2  retrain the SAME model with rays BLANKED (AC4 control)  -> blank checkpoint
#   3  evaluate both on the SAME held-out obstacle courses + collision comparison (AC5/AC7)
#
# Reboot-safe: marker files under $OUT_DIR record each step's output; a `STOP` file in $OUT_DIR
# makes the driver exit cleanly before the next step; the driver also stops before a step once it
# has run longer than MAX_HOURS (default 10 h) so an overnight window / cron can resume it.
#
# NOTE: `drone-fly train run` has no mid-run resume, so a step interrupted part-way restarts from
# scratch; each step (~2-4 h) fits inside one window. Copy nothing in: the 70-input warm-start model
# (training/11-count-curriculum/model_readout.pt), the K1 connectome (artifacts/pruned/k1) and the
# derived noise model are all produced in-repo.
#
# Usage: scripts/uc82_obstacle_avoidance.sh [OUT_DIR]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-artifacts/uc82}"
DRONE_FLY="${DRONE_FLY:-uv run drone-fly}"
MAX_HOURS="${MAX_HOURS:-10}"
EPISODES="${UC82_EPISODES:-256}"
SEED="${UC82_SEED:-1234}"
NOISE_YAML="artifacts/eyes/ray-noise.yaml"
RAYS_CONFIG="uc82-obstacle-rays"
BLANK_CONFIG="uc82-obstacle-blank"

mkdir -p "$OUT_DIR"
START_EPOCH="$(date +%s)"
MAX_SECONDS=$(awk "BEGIN{printf \"%d\", $MAX_HOURS*3600}")

log() { printf '[uc82 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

check_stop() {
  if [ -f "$OUT_DIR/STOP" ]; then
    log "STOP file present ($OUT_DIR/STOP): exiting cleanly; re-run to resume."
    exit 0
  fi
  local now elapsed
  now="$(date +%s)"
  elapsed=$((now - START_EPOCH))
  if [ "$elapsed" -ge "$MAX_SECONDS" ]; then
    log "time budget reached (${MAX_HOURS} h): exiting cleanly; re-run to resume."
    exit 0
  fi
}

# Run `drone-fly train run <config>` and record the final checkpoint path into $2.
train_run() {
  local config="$1" marker="$2" run_root="$3" logf="$4"
  if [ -f "$marker" ]; then
    log "skip train $config (checkpoint recorded: $(cat "$marker"))"
    return 0
  fi
  check_stop
  log "train $config -> $run_root"
  mkdir -p "$run_root"
  # shellcheck disable=SC2086  # $DRONE_FLY is an intentional multi-word launcher
  $DRONE_FLY train run "$config" --output-root "$run_root" | tee "$logf"
  local ckpt
  ckpt="$(grep -E '^checkpoint: ' "$logf" | tail -1 | sed 's/^checkpoint: //')"
  if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
    log "ERROR: no checkpoint produced for $config (see $logf)"
    exit 1
  fi
  printf '%s\n' "$ckpt" > "$marker"
  log "done train $config: $ckpt"
}

# Step 0 — the obstacle-ray noise model (AC1).
if [ -f "$NOISE_YAML" ]; then
  log "skip noise-model ($NOISE_YAML exists)"
else
  check_stop
  log "build noise-model -> $NOISE_YAML"
  # shellcheck disable=SC2086
  $DRONE_FLY eyes noise-model --out "$NOISE_YAML"
fi

# Step 1 — retrain under ray noise (AC3).
train_run "$RAYS_CONFIG" "$OUT_DIR/01_rays_ckpt.txt" "$OUT_DIR/rays" "$OUT_DIR/01_rays.log"

# Step 2 — retrain with rays blanked (AC4 control).
train_run "$BLANK_CONFIG" "$OUT_DIR/02_blank_ckpt.txt" "$OUT_DIR/blank" "$OUT_DIR/02_blank.log"

# Step 3 — evaluate both on the same held-out courses + the collision comparison (AC5/AC7).
EVAL_JSON="$OUT_DIR/03_eval.json"
if [ -f "$EVAL_JSON" ]; then
  log "skip eval ($EVAL_JSON exists)"
else
  check_stop
  RAYS_CKPT="$(cat "$OUT_DIR/01_rays_ckpt.txt")"
  BLANK_CKPT="$(cat "$OUT_DIR/02_blank_ckpt.txt")"
  log "eval rays vs blank on $EPISODES held-out courses (seed $SEED)"
  # shellcheck disable=SC2086
  $DRONE_FLY train proc-eval \
    --checkpoint "$RAYS_CKPT" --config "$RAYS_CONFIG" --ray-noise "$NOISE_YAML" \
    --blank-checkpoint "$BLANK_CKPT" --blank-config "$BLANK_CONFIG" \
    --episodes "$EPISODES" --seed "$SEED" --out "$EVAL_JSON"
fi

log "UC-82 complete. Eval report: $EVAL_JSON"
log "Fill the comparison table in docs/experiments/uc82-obstacle-detection-70-input.md from it."
