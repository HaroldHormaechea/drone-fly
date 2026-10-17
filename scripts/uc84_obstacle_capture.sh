#!/usr/bin/env bash
# UC-84 — obstacle-capture retrain/compare driver. A THIN WRAPPER over scripts/uc83_eyes_v2.sh:
# it adds one in-sandbox gate (step 0) in front of the already-tested UC-83 v2 recipe and then
# delegates the whole dataset -> v1-baseline -> v2 -> eval -> eval -> compare pipeline to it,
# unchanged. Sim-only (RTX A3000), no Liftoff — capture execution/assembly is user-side.
#
# Step 0 (this script)  ingest gate: validate the dropped-in captures and measure the in-view
#                       hit-ray fraction; ABORT if the overall fraction is below the 15 % target
#                       (so you fly more / use the degraded-flying fallback before spending GPU),
#                       unless --force. `eyes ingest` accepts the frozen val / held-out eval
#                       captures listed (name + capture sha) in the committed manifest
#                       tracks/liftoff-frozen-eval-captures.json, and hard-fails (exit 2) on any
#                       new or modified non-train capture, so the held-out split can never leak in.
# Steps 1-6 (delegated) scripts/uc83_eyes_v2.sh "$OUT_DIR": one dense dataset over the enlarged
#                       train captures (--label-missing picks up the new drops with no code change),
#                       train v1-baseline + v2, eval both on the intact held-out split, compare
#                       (AC4 verdict). The UC-83 recipe is reused byte-for-byte (not edited).
#
# The heavy AC5 / UC-82 re-run (steps 7-8 of uc83_eyes_v2.sh) stays gated behind UC83_RUN_AC5=1 —
# this wrapper does not set it, so it is OFF unless you opt in (then it is passed straight through).
#
# Resumable: step 0 records a marker ($OUT_DIR/00_ingest.txt); the delegated recipe has its own
# per-step markers + STOP-file ($OUT_DIR/STOP) + MAX_HOURS budget, so a reboot / re-run resumes.
# Remove $OUT_DIR/00_ingest.txt to re-run the gate after dropping in more captures.
#
# Usage: scripts/uc84_obstacle_capture.sh [OUT_DIR] [--force]
#   --force / UC84_FORCE=1   proceed to the retrain even if the fraction is below target
#   UC84_CAPTURES_ROOT=path  captures root to ingest (default: the harness captures root)
#   UC84_TARGET=0.15         in-view hit-ray target the gate enforces
#   UC84_FROZEN_MANIFEST=p   frozen-eval manifest (default tracks/liftoff-frozen-eval-captures.json)
#   plus everything scripts/uc83_eyes_v2.sh honours (UC83_EPOCHS, MAX_HOURS, UC83_RUN_AC5, ...)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="artifacts/uc84"
FORCE="${UC84_FORCE:-0}"
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -*) printf 'unknown flag: %s\n' "$arg" >&2; exit 2 ;;
    *) OUT_DIR="$arg" ;;
  esac
done

DRONE_FLY="${DRONE_FLY:-uv run drone-fly}"
TARGET="${UC84_TARGET:-0.15}"
INGEST_DIR="$OUT_DIR/ingest"
INGEST_JSON="$INGEST_DIR/ingest.json"
INGEST_MARKER="$OUT_DIR/00_ingest.txt"
PROVENANCE="$OUT_DIR/00_provenance.txt"

mkdir -p "$OUT_DIR"
log() { printf '[uc84 %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

check_stop() {
  if [ -f "$OUT_DIR/STOP" ]; then
    log "STOP file present ($OUT_DIR/STOP): exiting cleanly; re-run to resume."
    exit 0
  fi
}

# ---- provenance: freeze the held-out map set + catalogue sha so a shifted baseline is detectable.
record_provenance() {
  local py="import sys
from drone_fly.tracks import catalogue as c
from drone_fly.tracks.catalogue import sha256_file
cat = c.load_named('liftoff')
ms = {m: (v or {}).get('split') for m, v in cat.maps().items()}
held = sorted(m for m, s in ms.items() if s in ('val', 'held-out'))
sha = sha256_file(cat.path) if cat.path is not None else 'none'
print('catalogue: ' + str(cat.path))
print('catalogue_sha256: ' + sha)
print('held_out_frozen (no new captures): ' + ', '.join(held))
print('train_maps: ' + ', '.join(sorted(m for m, s in ms.items() if s == 'train')))
import os
from pathlib import Path
from drone_fly.eyes.ingest import frozen_manifest_default, load_frozen_manifest
fm = Path(os.environ.get('UC84_FROZEN_MANIFEST') or frozen_manifest_default())
print('frozen_manifest: ' + str(fm))
print('frozen_manifest_sha256: ' + (sha256_file(fm) if fm.is_file() else 'missing'))
print('frozen_manifest_entries: ' + str(len(load_frozen_manifest(fm))))"
  python3 -c "$py" 2>/dev/null || uv run python -c "$py"
}
log "recording provenance -> $PROVENANCE"
record_provenance | tee "$PROVENANCE"

# ---- Step 0: ingest gate -----------------------------------------------------------------------
if [ -f "$INGEST_MARKER" ]; then
  log "skip ingest gate (recorded: $(cat "$INGEST_MARKER")) — remove $INGEST_MARKER to re-gate"
else
  check_stop
  CR_ARGS=()
  if [ -n "${UC84_CAPTURES_ROOT:-}" ]; then
    CR_ARGS=(--captures-root "$UC84_CAPTURES_ROOT")
    log "ingesting captures root: $UC84_CAPTURES_ROOT"
  fi
  if [ -n "${UC84_FROZEN_MANIFEST:-}" ]; then
    CR_ARGS+=(--frozen-manifest "$UC84_FROZEN_MANIFEST")
    log "frozen-eval manifest: $UC84_FROZEN_MANIFEST"
  fi
  log "ingest gate: validate captures + measure in-view hit-ray fraction (target >= $TARGET)"
  # eyes ingest exits 2 on any new / modified val or held-out capture (not a matching entry of the
  # frozen-eval manifest) => set -e + pipefail abort here (held-out guard). The frozen eval
  # captures themselves are reported as frozen-eval and not counted.
  # shellcheck disable=SC2086
  $DRONE_FLY eyes ingest --label-missing --target "$TARGET" --out "$INGEST_DIR" \
    "${CR_ARGS[@]}" | tee "$OUT_DIR/00_ingest.log"
  if [ ! -f "$INGEST_JSON" ]; then
    log "ERROR: ingest produced no $INGEST_JSON (see 00_ingest.log)"; exit 1
  fi
  read_below() { python3 -c "import json;print(json.load(open('$INGEST_JSON'))['below_target'])" \
    2>/dev/null || uv run python -c "import json;print(json.load(open('$INGEST_JSON'))['below_target'])"; }
  read_share() { python3 -c "import json;print(json.load(open('$INGEST_JSON'))['overall']['in_view_hit_ray_share'])" \
    2>/dev/null || uv run python -c "import json;print(json.load(open('$INGEST_JSON'))['overall']['in_view_hit_ray_share'])"; }
  BELOW="$(read_below)"; SHARE="$(read_share)"
  log "overall in-view hit-ray fraction: $SHARE (target >= $TARGET; below_target=$BELOW)"
  if [ "$BELOW" = "True" ]; then
    if [ "$FORCE" != "1" ]; then
      log "ABORT: fraction below the $TARGET target. Fly more obstacle-dense courses or use the"
      log "       degraded/perturbed-flying fallback (docs/experiments/uc84-capture-protocol.md,"
      log "       Section f), then re-run. Pass --force (or UC84_FORCE=1) to retrain anyway."
      exit 1
    fi
    log "WARNING: fraction below target but --force set — proceeding to the retrain anyway."
  fi
  printf 'gate passed: in_view_hit_ray_share=%s target=%s below_target=%s force=%s\n' \
    "$SHARE" "$TARGET" "$BELOW" "$FORCE" > "$INGEST_MARKER"
fi

# ---- Steps 1-6 (+ opt-in AC5): delegate to the tested UC-83 v2 recipe, unchanged ---------------
log "delegating dataset -> v1 -> v2 -> eval -> eval -> compare to scripts/uc83_eyes_v2.sh $OUT_DIR"
log "(AC5/UC-82 re-run runs only when UC83_RUN_AC5=1 — currently: ${UC83_RUN_AC5:-unset})"
exec scripts/uc83_eyes_v2.sh "$OUT_DIR"
