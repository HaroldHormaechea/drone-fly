#!/usr/bin/env bash
# UC-83 — eyes v2: cut the obstacle-ray miss rate (classification head + focal loss + denser mining),
# measure v1-vs-v2 on the intact held-out maps, and — only if v2 clears the AC4 bar — re-derive the
# obstacle-ray noise model and re-run the UC-82 rays-vs-blank comparison. Sim-only, in-sandbox
# (RTX A3000), no Liftoff.
#
# Steps (each skipped if its marker/output already exists, so a reboot / re-run resumes):
#   1  build ONE dense dataset over the train maps (+ frame cache, --label-missing)   -> dataset dir
#   2  train the v1-baseline (default path: head off) on that dataset                 -> v1 eyes.pt
#   3  train eyes v2  (--hit-head --focal --hit-sampler) on the SAME dataset          -> v2 eyes.pt
#   4  eval v1 on the intact held-out split                                           -> v1 eval json
#   5  eval v2 on the SAME held-out split                                             -> v2 eval json
#   6  compare (AC4 verdict: nearest-quarter in-view miss reduction + FA bound)       -> compare json
#   --- conditional on the AC4 verdict AND UC83_RUN_AC5=1 (heavy: a 5M-step retrain) ---
#   7  re-derive the obstacle-ray noise model from the v2 held-out eval              -> ray-noise-v2
#   8  retrain ONLY the rays arm under the v2 noise; proc-eval vs the REUSED UC-82    -> collision cmp
#      blanked checkpoint on the same held-out proc courses + seed
#
# Reboot-safe: marker files under $OUT_DIR record each step's output; a `STOP` file in $OUT_DIR makes
# the driver exit cleanly before the next step; it also stops before a step once it has run longer
# than MAX_HOURS (default 10 h) so an overnight window / cron can resume it.
#
# NOTE: eyes train HAS mid-run resume (resume.pt) but this driver treats a train step as atomic (it
# re-runs a step whose marker is missing). The AC1-AC6 steps are minutes of GPU; only AC5 (steps
# 7-8) is heavy and it is opt-in + reuses the UC-82 blanked checkpoint (noise-independent).
#
# Usage: scripts/uc83_eyes_v2.sh [OUT_DIR]
#   UC83_RUN_AC5=1            also run the conditional AC5 re-run when AC4 passes
#   UC82_BLANK_CKPT=path      the reused UC-82 blanked checkpoint (else auto-discovered)
#   UC83_EPOCHS, UC83_SEED, UC83_BATCH, UC83_HIT_FRAC, UC83_EPISODES, UC83_PROC_SEED, MAX_HOURS
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-artifacts/uc83}"
DRONE_FLY="${DRONE_FLY:-uv run drone-fly}"
MAX_HOURS="${MAX_HOURS:-10}"
EPOCHS="${UC83_EPOCHS:-12}"
SEED="${UC83_SEED:-78}"
BATCH="${UC83_BATCH:-64}"
SPLIT="held-out"
TOL_M="0.5"                       # pinned so v1 and v2 miss/FA are measured at the same tolerance
# AC2: the hit-weighted sampler is the primary lever (train-time, no disk cost). The hit-biased
# dataset filter is an opt-in fallback; set UC83_HIT_FRAC to e.g. 0.25 to also bias the train set.
HIT_FRAC="${UC83_HIT_FRAC:-}"
# AC5 (conditional) knobs, mirrored from scripts/uc82_obstacle_avoidance.sh.
EPISODES="${UC83_EPISODES:-256}"
PROC_SEED="${UC83_PROC_SEED:-1234}"
RAYS_V2_CONFIG="uc83-obstacle-rays-v2"
BLANK_CONFIG="uc82-obstacle-blank"
NOISE_V2_YAML="artifacts/eyes/ray-noise-v2.yaml"

mkdir -p "$OUT_DIR"
START_EPOCH="$(date +%s)"
MAX_SECONDS=$(awk "BEGIN{printf \"%d\", $MAX_HOURS*3600}")

log() { printf '[uc83 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

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

# ---- Step 1: one dense dataset over the train maps (shared by v1-baseline and v2) -------------
DATASET_MARKER="$OUT_DIR/01_dataset.txt"
if [ -f "$DATASET_MARKER" ]; then
  log "skip dataset (recorded: $(cat "$DATASET_MARKER"))"
else
  check_stop
  log "build dense dataset (+cache, label-missing)"
  DS_OUT="$OUT_DIR/dataset"
  HF_ARGS=()
  if [ -n "$HIT_FRAC" ]; then
    HF_ARGS=(--hit-frac "$HIT_FRAC")
    log "hit-biased dataset filter ON: --hit-frac $HIT_FRAC"
  fi
  # shellcheck disable=SC2086
  $DRONE_FLY eyes dataset --label-missing --build-cache --out "$DS_OUT" \
    "${HF_ARGS[@]}" | tee "$OUT_DIR/01_dataset.log"
  if [ ! -f "$DS_OUT/manifest.json" ]; then
    log "ERROR: dataset build produced no manifest (see 01_dataset.log)"; exit 1
  fi
  printf '%s\n' "$DS_OUT" > "$DATASET_MARKER"
  log "dataset -> $DS_OUT"
fi
DATASET="$(cat "$DATASET_MARKER")"

# ---- train helper: $DRONE_FLY eyes train ... --out <run>; record <run>/eyes.pt into a marker ----
train_eyes() {
  local marker="$1" run="$2" logf="$3"; shift 3
  if [ -f "$marker" ]; then
    log "skip train (recorded: $(cat "$marker"))"
    return 0
  fi
  check_stop
  mkdir -p "$run"
  log "train -> $run ($*)"
  # shellcheck disable=SC2086
  $DRONE_FLY eyes train --dataset "$DATASET" --out "$run" --epochs "$EPOCHS" --batch "$BATCH" \
    --seed "$SEED" --select last --tol-m "$TOL_M" "$@" | tee "$logf"
  if [ ! -f "$run/eyes.pt" ]; then
    log "ERROR: no eyes.pt produced in $run (see $logf)"; exit 1
  fi
  printf '%s\n' "$run/eyes.pt" > "$marker"
  log "done train: $run/eyes.pt"
}

# ---- Step 2: v1-baseline (default path, head off) -------------------------------------------
train_eyes "$OUT_DIR/02_v1_ckpt.txt" "$OUT_DIR/v1" "$OUT_DIR/02_v1.log"

# ---- Step 3: eyes v2 (hit head + focal + hit-weighted sampler) ------------------------------
train_eyes "$OUT_DIR/03_v2_ckpt.txt" "$OUT_DIR/v2" "$OUT_DIR/03_v2.log" \
  --hit-head --focal --hit-sampler

V1_CKPT="$(cat "$OUT_DIR/02_v1_ckpt.txt")"
V2_CKPT="$(cat "$OUT_DIR/03_v2_ckpt.txt")"

# ---- eval helper: eyes eval --weights W --dataset D --split held-out --out <dir> ------------
eval_eyes() {
  local marker="$1" ckpt="$2" evdir="$3" logf="$4"
  if [ -f "$marker" ]; then
    log "skip eval (recorded: $(cat "$marker"))"
    return 0
  fi
  check_stop
  mkdir -p "$evdir"
  log "eval $ckpt on $SPLIT -> $evdir"
  # shellcheck disable=SC2086
  $DRONE_FLY eyes eval --weights "$ckpt" --dataset "$DATASET" --split "$SPLIT" \
    --tol-m "$TOL_M" --out "$evdir" | tee "$logf"
  local ej="$evdir/eval_${SPLIT}.json"
  if [ ! -f "$ej" ]; then
    log "ERROR: no $ej produced (see $logf)"; exit 1
  fi
  printf '%s\n' "$ej" > "$marker"
  log "done eval: $ej"
}

# ---- Step 4 / 5: eval v1 and v2 on the intact held-out split --------------------------------
eval_eyes "$OUT_DIR/04_v1_eval.txt" "$V1_CKPT" "$OUT_DIR/eval_v1" "$OUT_DIR/04_v1_eval.log"
eval_eyes "$OUT_DIR/05_v2_eval.txt" "$V2_CKPT" "$OUT_DIR/eval_v2" "$OUT_DIR/05_v2_eval.log"
V1_EVAL="$(cat "$OUT_DIR/04_v1_eval.txt")"
V2_EVAL="$(cat "$OUT_DIR/05_v2_eval.txt")"

# ---- Step 6: AC4 verdict (v1-vs-v2 ray accuracy) -------------------------------------------
COMPARE_JSON="$OUT_DIR/compare/compare.json"
if [ -f "$COMPARE_JSON" ]; then
  log "skip compare ($COMPARE_JSON exists)"
else
  check_stop
  log "compare v1 vs v2 (AC4)"
  # shellcheck disable=SC2086
  $DRONE_FLY eyes compare --v1 "$V1_EVAL" --v2 "$V2_EVAL" \
    --out "$OUT_DIR/compare" | tee "$OUT_DIR/06_compare.log"
fi
if [ ! -f "$COMPARE_JSON" ]; then
  log "ERROR: compare produced no $COMPARE_JSON"; exit 1
fi
read_passed() { python3 -c "import json;print(json.load(open('$COMPARE_JSON'))['passed'])" 2>/dev/null \
  || uv run python -c "import json;print(json.load(open('$COMPARE_JSON'))['passed'])"; }
AC4_PASSED="$(read_passed)"
log "AC4 verdict: passed=$AC4_PASSED (see $COMPARE_JSON)"

log "AC1-AC6 complete. v1 vs v2: $COMPARE_JSON"
log "Fill the comparison table in docs/experiments/uc83-eyes-v2.md from the eval + compare JSON."

# ---- Steps 7-8: conditional AC5 (only if AC4 passed AND opted in) ---------------------------
if [ "$AC4_PASSED" != "True" ]; then
  log "AC4 did not pass -> AC5 skipped (documented negative result; next lever: close-to-obstacle "
  log "Liftoff captures or a depth backbone). Done."
  exit 0
fi
if [ "${UC83_RUN_AC5:-0}" != "1" ]; then
  log "AC4 PASSED. AC5 is heavy (a 5M-step retrain) and opt-in: re-run with UC83_RUN_AC5=1 to run it."
  exit 0
fi

# Step 7 — re-derive the obstacle-ray noise model from the v2 held-out eval.
if [ -f "$NOISE_V2_YAML" ]; then
  log "skip noise-model ($NOISE_V2_YAML exists)"
else
  check_stop
  log "re-derive ray-noise from v2 eval -> $NOISE_V2_YAML"
  # shellcheck disable=SC2086
  $DRONE_FLY eyes noise-model --from-eval "$V2_EVAL" --out "$NOISE_V2_YAML"
fi

# Reuse the UC-82 blanked checkpoint (noise-independent): only the rays arm retrains.
BLANK_CKPT="${UC82_BLANK_CKPT:-}"
if [ -z "$BLANK_CKPT" ] && [ -f "artifacts/uc82/02_blank_ckpt.txt" ]; then
  BLANK_CKPT="$(cat artifacts/uc82/02_blank_ckpt.txt)"
fi
if [ -z "$BLANK_CKPT" ]; then
  BLANK_CKPT="$(find artifacts/uc82/blank -name 'model_readout.pt' 2>/dev/null | sort | tail -1 || true)"
fi
if [ -z "$BLANK_CKPT" ] || [ ! -f "$BLANK_CKPT" ]; then
  log "ERROR: no UC-82 blanked checkpoint found; set UC82_BLANK_CKPT=<path to model_readout.pt>"
  exit 1
fi
log "reusing UC-82 blanked checkpoint: $BLANK_CKPT"

# Step 8 — retrain ONLY the rays arm under the v2 noise, then proc-eval vs the blanked control.
RAYS_V2_MARKER="$OUT_DIR/07_rays_v2_ckpt.txt"
if [ -f "$RAYS_V2_MARKER" ]; then
  log "skip rays-v2 train (recorded: $(cat "$RAYS_V2_MARKER"))"
else
  check_stop
  RAYS_V2_RUN="$OUT_DIR/ac5/rays-v2"
  mkdir -p "$RAYS_V2_RUN"
  log "retrain rays arm under v2 noise ($RAYS_V2_CONFIG) -> $RAYS_V2_RUN"
  # shellcheck disable=SC2086
  $DRONE_FLY train run "$RAYS_V2_CONFIG" --output-root "$RAYS_V2_RUN" | tee "$OUT_DIR/07_rays_v2.log"
  RV2="$(grep -E '^checkpoint: ' "$OUT_DIR/07_rays_v2.log" | tail -1 | sed 's/^checkpoint: //')"
  if [ -z "$RV2" ] || [ ! -f "$RV2" ]; then
    log "ERROR: no rays-v2 checkpoint produced (see 07_rays_v2.log)"; exit 1
  fi
  printf '%s\n' "$RV2" > "$RAYS_V2_MARKER"
fi
RAYS_V2_CKPT="$(cat "$RAYS_V2_MARKER")"

AC5_EVAL="$OUT_DIR/08_ac5_eval.json"
if [ -f "$AC5_EVAL" ]; then
  log "skip AC5 proc-eval ($AC5_EVAL exists)"
else
  check_stop
  log "proc-eval rays-v2 vs REUSED blanked control on $EPISODES held-out courses (seed $PROC_SEED)"
  # shellcheck disable=SC2086
  $DRONE_FLY train proc-eval \
    --checkpoint "$RAYS_V2_CKPT" --config "$RAYS_V2_CONFIG" --ray-noise "$NOISE_V2_YAML" \
    --blank-checkpoint "$BLANK_CKPT" --blank-config "$BLANK_CONFIG" \
    --episodes "$EPISODES" --seed "$PROC_SEED" --out "$AC5_EVAL"
fi

log "UC-83 complete (incl. AC5). Ray accuracy: $COMPARE_JSON ; collision comparison: $AC5_EVAL"
log "Fill both tables in docs/experiments/uc83-eyes-v2.md."
