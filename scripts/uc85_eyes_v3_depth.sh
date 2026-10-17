#!/usr/bin/env bash
# UC-85 — eyes v3: swap the from-scratch CNN trunk for a FROZEN pretrained monocular-depth backbone
# and measure v3-depth vs a v1-baseline on the SAME current 5-map data (like UC-83), with latency as
# a hard deployability gate. Sim-only, in-sandbox (RTX A3000 / 3050), no Liftoff.
#
# Steps (each skipped if its marker/output already exists, so a reboot / re-run resumes):
#   0  provision the depth backbone into the HF cache (once; tiny-depth needs no download)  -> marker
#   1  build / reuse ONE dense dataset over the train maps (+ frame cache, --label-missing)  -> ds dir
#   2  train the v1-baseline (default path: depth off) on that dataset                       -> v1 pt
#   3  train eyes v3-depth (--depth-backbone --backbone-id) on the SAME dataset              -> v3 pt
#   4  eval v1 on the intact held-out split (HangarC03 / MinusTwo / ThePit, pinned tol)      -> v1 json
#   5  eval v3 on the SAME held-out split                                                    -> v3 json
#   6  compare (AC6 verdict: nearest-quarter in-view miss reduction + FA bound)              -> compare
#   7  bench v3-depth latency (p50/p95/p99 vs the 25 ms / 40 Hz tick, +gate detector)        -> bench
#
# Reboot-safe: marker files under $OUT_DIR record each step's output; a `STOP` file in $OUT_DIR makes
# the driver exit cleanly before the next step; it also stops before a step once it has run longer
# than MAX_HOURS (default 10 h) so an overnight window / cron can resume it.
#
# After step 7, fill the v1-vs-v3 table + the latency line + the recommendation in
# docs/experiments/uc85-eyes-v3-depth.md from the eval / compare / bench JSON.
#
# Usage: scripts/uc85_eyes_v3_depth.sh [OUT_DIR]
#   UC85_BACKBONE_ID=depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf  (default; the plan's)
#   UC85_BACKBONE_ID=tiny-depth   use the pure-torch stub (fast, no download; a wiring smoke only)
#   UC85_DATASET=path             reuse an existing dataset dir (e.g. artifacts/uc83/dataset) as-is
#   UC85_EPOCHS, UC85_SEED, UC85_BATCH, UC85_TOL_M, UC85_DISK_MARGIN_GB, MAX_HOURS
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${1:-artifacts/uc85}"
DRONE_FLY="${DRONE_FLY:-uv run drone-fly}"
MAX_HOURS="${MAX_HOURS:-10}"
EPOCHS="${UC85_EPOCHS:-12}"
SEED="${UC85_SEED:-78}"
BATCH="${UC85_BATCH:-64}"
SPLIT="held-out"
TOL_M="${UC85_TOL_M:-0.5}"          # pinned so v1 and v3 miss/FA are measured at the same tolerance
BACKBONE_ID="${UC85_BACKBONE_ID:-depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf}"
DISK_MARGIN_GB="${UC85_DISK_MARGIN_GB:-1}"

mkdir -p "$OUT_DIR"
START_EPOCH="$(date +%s)"
MAX_SECONDS=$(awk "BEGIN{printf \"%d\", $MAX_HOURS*3600}")

log() { printf '[uc85 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

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

# ---- Step 0: provision the depth backbone into the HF cache (offline-able afterwards) ----------
PROVISION_MARKER="$OUT_DIR/00_backbone.txt"
if [ -f "$PROVISION_MARKER" ]; then
  log "skip provision (recorded: $(cat "$PROVISION_MARKER"))"
else
  check_stop
  log "provision depth backbone: $BACKBONE_ID"
  # shellcheck disable=SC2086
  $DRONE_FLY eyes provision-backbone --model-id "$BACKBONE_ID" | tee "$OUT_DIR/00_backbone.log"
  printf '%s\n' "$BACKBONE_ID" > "$PROVISION_MARKER"
  log "backbone cached: $BACKBONE_ID"
fi

# ---- Step 1: one dense dataset over the train maps (shared by v1-baseline and v3-depth) --------
DATASET_MARKER="$OUT_DIR/01_dataset.txt"
if [ -n "${UC85_DATASET:-}" ]; then
  if [ ! -f "$UC85_DATASET/manifest.json" ]; then
    log "ERROR: UC85_DATASET=$UC85_DATASET has no manifest.json"; exit 1
  fi
  printf '%s\n' "$UC85_DATASET" > "$DATASET_MARKER"
  log "reuse dataset: $UC85_DATASET"
elif [ -f "$DATASET_MARKER" ]; then
  log "skip dataset (recorded: $(cat "$DATASET_MARKER"))"
else
  check_stop
  log "build dense dataset (+cache, label-missing)"
  DS_OUT="$OUT_DIR/dataset"
  # shellcheck disable=SC2086
  $DRONE_FLY eyes dataset --label-missing --build-cache --out "$DS_OUT" \
    --disk-margin-gb "$DISK_MARGIN_GB" | tee "$OUT_DIR/01_dataset.log"
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

# ---- Step 2: v1-baseline (default path, depth off) ------------------------------------------
train_eyes "$OUT_DIR/02_v1_ckpt.txt" "$OUT_DIR/v1" "$OUT_DIR/02_v1.log"

# ---- Step 3: eyes v3-depth (frozen pretrained monocular-depth trunk) ------------------------
train_eyes "$OUT_DIR/03_v3_ckpt.txt" "$OUT_DIR/v3" "$OUT_DIR/03_v3.log" \
  --depth-backbone --backbone-id "$BACKBONE_ID"

V1_CKPT="$(cat "$OUT_DIR/02_v1_ckpt.txt")"
V3_CKPT="$(cat "$OUT_DIR/03_v3_ckpt.txt")"

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

# ---- Step 4 / 5: eval v1 and v3 on the intact held-out split --------------------------------
eval_eyes "$OUT_DIR/04_v1_eval.txt" "$V1_CKPT" "$OUT_DIR/eval_v1" "$OUT_DIR/04_v1_eval.log"
eval_eyes "$OUT_DIR/05_v3_eval.txt" "$V3_CKPT" "$OUT_DIR/eval_v3" "$OUT_DIR/05_v3_eval.log"
V1_EVAL="$(cat "$OUT_DIR/04_v1_eval.txt")"
V3_EVAL="$(cat "$OUT_DIR/05_v3_eval.txt")"

# ---- Step 6: AC6 accuracy verdict (v1-vs-v3 ray accuracy; v3 passed as --v2) ----------------
COMPARE_JSON="$OUT_DIR/compare/compare.json"
if [ -f "$COMPARE_JSON" ]; then
  log "skip compare ($COMPARE_JSON exists)"
else
  check_stop
  log "compare v1 vs v3 (AC6 accuracy gate)"
  # shellcheck disable=SC2086
  $DRONE_FLY eyes compare --v1 "$V1_EVAL" --v2 "$V3_EVAL" \
    --out "$OUT_DIR/compare" | tee "$OUT_DIR/06_compare.log"
fi
if [ ! -f "$COMPARE_JSON" ]; then
  log "ERROR: compare produced no $COMPARE_JSON"; exit 1
fi
read_passed() { python3 -c "import json;print(json.load(open('$COMPARE_JSON'))['passed'])" 2>/dev/null \
  || uv run python -c "import json;print(json.load(open('$COMPARE_JSON'))['passed'])"; }
ACC_PASSED="$(read_passed)"
log "accuracy verdict: passed=$ACC_PASSED (see $COMPARE_JSON)"

# ---- Step 7: AC5 latency gate (p50/p95/p99 vs the live tick, with the gate detector) --------
BENCH_JSON="$OUT_DIR/bench/bench.json"
if [ -f "$BENCH_JSON" ]; then
  log "skip bench ($BENCH_JSON exists)"
else
  check_stop
  log "bench v3-depth latency (+gate detector) -> $OUT_DIR/bench"
  # shellcheck disable=SC2086
  $DRONE_FLY eyes bench --weights "$V3_CKPT" --with-gate-detector \
    --out "$OUT_DIR/bench" | tee "$OUT_DIR/07_bench.log"
fi
if [ ! -f "$BENCH_JSON" ]; then
  log "ERROR: bench produced no $BENCH_JSON"; exit 1
fi

log "UC-85 complete."
log "  accuracy (AC6): $COMPARE_JSON (passed=$ACC_PASSED)"
log "  latency  (AC5): $BENCH_JSON"
log "Fill docs/experiments/uc85-eyes-v3-depth.md: v1-vs-v3 table (from the eval + compare JSON),"
log "the latency line (from bench.json: eyes_ms.total.p95 vs budget_ms), and the recommendation"
log "(adopt / accuracy-win-not-deployable + mitigation / negative -> defer to UC-84)."
