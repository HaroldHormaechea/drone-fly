---
plan_for: (free-form task)
work_branch: feat/fix-uc84-ingest-frozen-eval
run: drone-fly
approved: 2026-10-17
---

# Fix: `eyes ingest` / UC-84 driver step 0 must not fail on the frozen val/held-out captures

Challenger verdict: **APPROVE** (round 1). Minors to fold in during implementation:
1. Docs give one exact copy-pasteable re-baseline command for the helper (no CLI); QA asserts the doc names it.
2. Provenance hashes the manifest; add `tracks/*.json text eol=lf` to `.gitattributes` (or hash canonicalised entries) so CRLF checkouts don't change the sha. The frozen check itself is unaffected.
3. The frozen check is purely name + capture_sha; the manifest's map/split/catalogue_id fields are informational, not extra failure conditions.
4. `frozen_eval_missing` stays warning-only (one markdown line).
5. Driver test: forward the stub to the real CLI (`python -m drone_fly.cli.main` or the DRONE_FLY override); pre-write labels for the synthetic train capture so it doesn't depend on trackdata.

Expected after the fix: real step 0 still ABORTS with exit 1 (overall 0.0436 < 0.15) until new train captures land, unless `--force`. Correct behaviour.
The developer is authorized to edit `tracks/`, `scripts/`, `docs/` and `.gitattributes` (outside `paths.production`) for this fix.

## Analysis
Reproduced on the real data root (`C:\dev\drone-fly`, `.venv-cuda\Scripts\drone-fly.exe eyes ingest --out <tmp>`):
- 36 captures discovered: 14 train included; 3 train `alignment-unverified` (permafrost-…-01, straw-bale-…-01/02), which is expected; 19 `wrong-split`: 10 val (AutumnFields x4, Rustline x3, Surtur x3) and 9 held-out (HangarC03 x2, MinusTwo x2, ThePit x5), all named `<map>-2026-10-13-replay-NN`.
- Exit code is already correct: the CLI exits 2 (`EXIT_CONFIG`; cli/main.py:941-943 catches the `HarnessConfigError` raised by `run_ingest` at eyes/ingest.py:182-187). The driver aborts correctly because scripts/uc84_obstacle_capture.sh:29 sets `set -euo pipefail` (rc=2 verified). The protocol doc (line 211) says only "exits non-zero"; it should say exit 2.
- On the failure path nothing is printed to stdout: `cmd_ingest` prints its JSON only after `run_ingest` returns.
- With eval captures set aside, the overall in-view hit-ray fraction is 0.0436 (20692/475073, 17665 frames, 14 captures), so after the fix step 0 still correctly ABORTs with exit 1 on below_target unless `--force`.
- Root cause: `_ingest_one` (ingest.py:238-244) treats any val/held-out capture as a violation, but these 19 are the frozen eval baseline that `eyes dataset`/`eyes eval` consume from the same root. The invariant is "no NEW captures on val/held-out maps".
- `capture_sha()` (eyes/capture_meta.py:122-133) already exists: sha256 over meta.json + frames.csv + telemetry, the identity used for label staleness (ingest.py:259). Hashing all 19 takes about 0.6 s.

## Proposed Solution
Rule: a committed frozen-eval capture manifest keyed by (capture folder name, capture_sha). A val/held-out capture is `frozen-eval` (reported, excluded from the fraction, not a failure) iff its name is listed AND its current capture_sha equals the recorded sha. Anything else hard-fails as today. Keying on name + sha means a copy under a new name is still caught.

1. **New committed manifest `tracks/liftoff-frozen-eval-captures.json`**: format 1, a "why" note, the catalogue sha at freeze time (informational), and one entry per capture (`name`, `map`, `split`, `catalogue_id`, `capture_sha256`) for the 19. Generate it from the real root (read-only, `DRONE_FLY_DATA_ROOT=C:\dev\drone-fly\artifacts\harness\liftoff\training-data`) via the helper in step 2. Spot-check the sha prefixes: the-pit-…-01 → 5d12115bb6b8, autumn-fields-…-01 → 14551db98675, rustline-…-01 → 36fe094df2bd.
2. **`src/drone_fly/eyes/ingest.py`**:
   - Default manifest path constant via `drone_fly.tracks.catalogue.catalogue_dir()`.
   - Loader returning `{name: entry}`. Missing file → empty mapping (safe default); malformed → `HarnessConfigError`.
   - Builder helper that writes manifest entries for every capture resolving to val/held-out under a captures root (discover_captures, read_capture_meta, catalogue, capture_sha). Documented as the deliberate re-baseline procedure; no CLI flag.
   - `run_ingest`: new keywords `frozen_manifest` (path; defaults to the committed file) and `strict: bool = True`; the loaded mapping is passed into `_ingest_one`.
   - `_ingest_one`, non-train branch: listed + sha match → `frozen-eval`; listed + sha differs → `frozen-eval-modified` (violation); else `wrong-split` (violation). Record the sha in the rec. Add both reasons to REASONS and the docstring.
   - Aggregation: frozen captures stay `included: False`. New report keys `frozen_eval` (names), `frozen_eval_missing` (warning only, never fails), `frozen_manifest` (path), `overall.captures_frozen_eval`. The `wrong_split` key name is kept but now lists all violations (wrong-split + frozen-eval-modified).
   - Move the raise into a public `check_held_out_invariant(report)`; `run_ingest` calls it when strict. The message says new or modified captures on frozen val/held-out maps are not in <manifest>, and how to re-baseline.
   - `_markdown`: status `frozen-eval (not counted)`, a frozen-count line, the violation banner kept for real violations, and a warning line when `frozen_eval_missing` is non-empty.
3. **`src/drone_fly/eyes/cli.py` `cmd_ingest`**: call `run_ingest(..., strict=False)`. Always print a human-readable summary first: overall fraction with hits/rays, target, baseline, BELOW TARGET / meets target, one per-map line with share and capture count, counts of included / frozen-eval / excluded / violations, the report dir. Then the existing JSON block, adding `captures_frozen_eval` and `violations`. Then `check_held_out_invariant(report)`, so a violation still exits 2 with the error on stderr. New optional `--frozen-manifest` flag.
4. **`scripts/uc84_obstacle_capture.sh`**: no structural change to step 0. Update the header and line-90 comments to the new rule. Honour an optional `UC84_FROZEN_MANIFEST` → `--frozen-manifest` (same pattern as UC84_CAPTURES_ROOT/CR_ARGS). `record_provenance` also logs the manifest path, its sha256 and its entry count.
5. **Docs**:
   - `docs/experiments/uc84-capture-protocol.md`: rewrite the *Held-out-split invariant* section, keeping the heading (test_uc84_docs.py:50). Frozen = the 19 committed captures; new non-train or modified frozen captures hard-fail with exit 2. Explain re-baselining. Update the reasons list (lines 209-210) with frozen-eval / frozen-eval-modified, say "exits 2", and say the fraction is printed on stdout. Fix the wording at lines 41-44 and the stale "8 captures" at lines 13-14 (36 now).
   - `docs/experiments/uc84-liftoff-obstacle-capture.md` lines 18 and 27-28: one-line tweaks (frozen = committed manifest; provenance logs the manifest sha).

## Files Affected
- **Production (developer):** `src/drone_fly/eyes/ingest.py`, `src/drone_fly/eyes/cli.py`. Also, authorized though outside `paths.production`: `tracks/liftoff-frozen-eval-captures.json` (new), `scripts/uc84_obstacle_capture.sh`, `docs/experiments/uc84-capture-protocol.md`, `docs/experiments/uc84-liftoff-obstacle-capture.md`, `.gitattributes`.
- **Tests (QA)**, CPU-only with synthetic captures (`tests/eyes_support.write_capture` + `fake_catalogue`; MapA train / MapB val / MapC held-out) and a tmp fixture manifest:
  - `tests/test_uc84_ingest.py`:
    - frozen val + held-out with matching sha → `frozen-eval`, not included, no raise, fraction from train only, `frozen_eval` listed
    - a new unlisted val capture alongside frozen ones → raises; `wrong_split` names only the new one
    - a listed name with changed content → `frozen-eval-modified`, raises
    - missing manifest → the val capture fails
    - a manifest entry absent from the root → `frozen_eval_missing` warning, no raise
    - `strict=False` returns the report instead of raising
    - markdown shows the frozen status
    - builder round-trip: build → load → ingest passes
    - existing tests unchanged
  - CLI via `drone_fly.cli.main.main([...])`:
    - exit 0 with frozen + train, and the summary (overall fraction, per-map line, BELOW TARGET) on stdout
    - exit 2 with a new val capture, the summary still printed before the stderr error
  - `tests/test_uc84_driver.py`: keep the stub tests. Add a step-0 test where the stub forwards `eyes ingest` to the real CLI (dataset/train/eval/compare stubbed), with DRONE_FLY_DATA_ROOT = tmp, UC84_FROZEN_MANIFEST = a tmp fixture, and real catalogue ids (train `bando-city-01`, val `rustline-01` in the fixture manifest):
    - (a) frozen + train with UC84_FORCE=1 → rc 0, `00_ingest.txt` written, recipe delegated
    - (b) plus an unlisted val capture → rc 2, no marker, no `eyes dataset`
    - assert provenance contains the manifest sha line
    - pre-write labels instead of relying on `--label-missing`

## Risks & Considerations
- Re-baselining eval data requires regenerating and committing the manifest; this is deliberate. A frozen capture whose meta.json is re-edited becomes `frozen-eval-modified` and fails, and the message says how to re-baseline.
- A catalogue split change (val/held-out → train) makes old captures count as train. Out of scope; visible via provenance.
- `frozen_eval_missing` is warning-only, so removing eval captures from the real root is reported, not enforced.
- `run_ingest` stays strict by default; only the CLI uses strict=False.
- Real driver step 0 aborts with exit 1 (0.044 < 0.15) until new train captures arrive. Correct.
- `--label-missing` may relabel train captures into the data root's labels folder if label settings change; they are currently up to date.
