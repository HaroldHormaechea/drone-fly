---
plan_for: use-cases/84-liftoff-obstacle-capture-protocol.md
work_branch: feat/uc-84-liftoff-obstacle-capture
team: drone-fly-uc-84
approved: 2026-10-17
---

# UC-84 — Liftoff obstacle-capture protocol — Approved implementation plan

Challenger APPROVED (round 2; 2 Majors on AC2 writer + held-out split resolved, re-verified against code). In-place `/workspace/drone-fly`. Depth backbone = UC-85, out of scope.

## Analysis
UC-84 attacks the data ceiling UC-83 proved: eyes v2 left near-obstacle miss ~0.99 and the hit head collapsed (in-view recall 0.027) because the captures carry almost no close-obstacle signal (5 train maps, monocular, ~5% in-view hit-ray fraction). UC-84 does NOT touch the estimator; it defines + tools an obstacle-dense Liftoff capture protocol to push the in-view hit-ray fraction to **≥15%**, then rebuilds the dataset and re-runs UC-83's already-tested v2 recipe to see if richer data clears the AC4 bar (≥30% rel nearest-quarter in-view miss reduction, FA-on-clear ≤0.10).

**Hard Liftoff/sandbox boundary (grounded in code):** no in-repo writer of the `captures/<Map>/<cap>/` layout exists (only readers: `eyes/capture_meta.py`, `harness/offline/capture.py`). The 8 on-disk captures were assembled off-branch by UC-71 tooling from saved Liftoff race recordings — confirmed on a real `meta.json` (`pose_source:"recording-xml"`, `camera.source:"recording"`, `alignment.camera_check{...}`, `race.file/sha256`, `recording` block). So capture execution + assembly run on the user's Windows/Liftoff machine; the dev-team builds only the in-sandbox pieces (AC3 ingestion/validation, AC4 driver, AC6 tests), and the sim-only retrain runs in-sandbox once the user drops captures in.

**New metric:** in-view hit-ray fraction = `hit[in_view].mean()` (share of in-view rays striking an obstacle). Labels carry per-ray `hit` + `in_view` arrays (`eyes/labels.py`; obstacles = forward-cone `raycast_obstacles()`, so obstacles-ahead frames count even when later avoided). Not currently surfaced — `_group_stats` reports `rays_hit_share` (all rays) + `in_view_share` separately, not the conditional.

## Proposed Solution (per AC)

**AC1 — Capture protocol doc** → `docs/experiments/uc84-capture-protocol.md` (new). (a) Map list: keep the 5 captured train maps, add obstacle-dense catalogue train maps already carrying track/race XML (BardwellsYard, Hall26, ParisDroneFestival, Permafrost, ShortCircuit, StrawBale, TheGreen). (b) Course selection: races with many placed non-gate items in the forward cone. (c) Driving method: the 70-input `training/11-count-curriculum` model (reproducible, obstacle-aware); teacher as backup. (d) Per-map frame target sized to reach ≥15% overall (planning figure, enforced by AC3). (e) ≥15% rationale: ~3× the ~5% baseline UC-83's rebalancing couldn't overcome. (f) Degraded/perturbed fallback (gate-triggered by AC3's <15% flag): action-noise/perturbed flying → slower speeds + tighter courses → sample colliding runs → manual near-obstacle passes. **Held-out-split invariant:** new maps catalogued `split: train` only; HangarC03/MinusTwo/ThePit + val maps frozen (no new captures) so the cross-run comparison stays valid.

**AC2 — Capture tooling** → documented in the AC1 doc (runbook, no in-sandbox code): **Fly** (`liftoff-episodes` skill drives the 70-input model, saves in-game recordings every K attempts) → **Assemble** (off-branch UC-71 capture tool converts each recording to a capture folder + runs the alignment/camera-check that sets `alignment.verified`) → **Schema contract** (authoritative, in the doc: `frames/` + `frames.csv` + telemetry (`telemetry_raw.csv` or `telemetry.csv`) + `meta.json{schema_version:1, catalogue_id, map, camera{resolution,hfov_deg,uptilt_deg}, camera.source, alignment{verified:true, lag_ms}, alignment.camera_check.ok, split, pose_source:"recording-xml"}`) → **Drop** into `.../training-data/captures/<Map>/`. AC3 is the backstop. Developer nit: mirror the EXACT on-disk field names — top-level `pose_source`, `camera.source`, `alignment.verified`/`alignment.camera_check.ok`; there is no top-level `source:"replay"`.

**AC3 — Ingestion + validation** (sim-only, in-sandbox): new `drone-fly eyes ingest` (`eyes/cli.py` + new `src/drone_fly/eyes/ingest.py`). Discovers captures (reuse `discover_captures`); validates each — meta schema/`catalogue_id`/`map`/`camera`, requires `alignment.verified==true`, presence of frames+telemetry, pairing accounting via `load_capture`'s `pair_tol_s`/segmenting; **asserts `split==train`, failing loudly on any held-out/val map**; labels via `run_label`; computes per-capture, per-map, overall in-view hit-ray fraction from the labels npz, reports vs ~5% baseline, **flags <15%**. Emits JSON + markdown with actionable per-capture reasons (`no-meta`, `alignment-unverified`, `no-catalogue`, `no-trackdata`/`no-scene`, `no-labels`, `stale-labels`, `wrong-split`). The fraction is a single shared helper also wired into `_group_stats` as `in_view_hit_ray_share` (one path, one source of truth). Scope: conformance + pairing accounting, NOT independent alignment re-verification.

**AC4 — Resumable retrain/compare driver** → new `scripts/uc84_obstacle_capture.sh`, a thin wrapper REUSING `scripts/uc83_eyes_v2.sh` (not editing it). Step 0 (own marker): `eyes ingest` gate — abort if overall <15% (`--force` to override). Steps 1-6: delegate to `uc83_eyes_v2.sh artifacts/uc84` (dataset→v1-baseline→v2→eval→eval→`eyes compare`, per-step markers + STOP-file + MAX_HOURS resume; AC5/UC-82 re-run stays gated behind unset `UC83_RUN_AC5`). Enlarged captures land in the default captures root, so `eyes dataset --label-missing` picks them up with no code change. Driver logs the held-out map set + catalogue sha so a shifted baseline is detectable.

**AC5 — Result doc** → `docs/experiments/uc84-liftoff-obstacle-capture.md` (new): method + comparison-table structure (achieved fraction; v2-on-new vs UC-83 v1/v2-on-old: nearest-quarter + aggregate in-view miss, FA-on-clear, hit-head recall) + gated next-steps (AC4 clears → re-run UC-82 rays-vs-blank with new eyes; else → record negative, name UC-85 depth backbone). Numbers filled when the sim-only retrain runs after the capture drop. Links to the AC1 doc for the run command.

**AC6 — CI-hermetic tests** (CPU-only, no Liftoff/real captures): `tests/test_uc84_ingest.py` (reuse `tests/liftoff_capture_support.py` + `tests/eyes_support.py`: format rejects incl. `alignment.verified:false`, pairing-drop accounting, the fraction on a hand-built case, <15% flag on/off, `split!=train` fail, per-reason coverage, positive coverage of `in_view_hit_ray_share`); `tests/test_uc84_driver.py` (stubbed `DRONE_FLY`: step/skip/resume markers, ingest-gate abort, STOP-file early-exit); doc-structure test mirroring `test_eyes_docs.py` for both docs.

## Files Affected
- **Production (developer, `src/drone_fly/**`):** `eyes/ingest.py` (new); `eyes/cli.py` (register `eyes ingest`); `eyes/dataset.py` (shared metric helper + `in_view_hit_ray_share` in `_group_stats`).
- **Scripts (developer-authorized, `scripts/**`):** `scripts/uc84_obstacle_capture.sh` (new wrapper).
- **Docs (developer-authorized, `docs/experiments/**`):** `uc84-capture-protocol.md` (AC1/AC2), `uc84-liftoff-obstacle-capture.md` (AC5).
- **Test (qa, `tests/**`):** `test_uc84_ingest.py`, `test_uc84_driver.py`, doc-structure assertions.

## Risks & Considerations
- Liftoff/sandbox boundary: AC1/AC2 docs only; AC5 numbers not producible by the dev-team now (skeleton + method). No over-claiming.
- Clean-pilot pitfall: mitigated by obstacle-dense courses + geometry-ahead metric + <15% gate + fallback (AC1 §f).
- Comparison validity: held-out split pinned (invariant + AC3 assertion + driver logging).
- New-map prerequisite: needs catalogue entry + track/race XML + scene cache, else AC3 drops with a named reason.
- ≥15% may still not clear AC4 (monocular may need depth) → UC-85; AC5 records the negative.
- Disk: ~18 GB free; reuse `eyes dataset` disk-margin guard + prunable intermediates.
- Reuse integrity: wrapping (not editing) `uc83_eyes_v2.sh` keeps the tested recipe byte-identical; the only tested-path change is the additive stats field (safe — verified vs `test_eyes_dataset.py`).

## Challenger verdict (APPROVE, round 2)
Round-1 Majors (AC2 had no honest capture-writer path; held-out-split integrity) resolved: AC2 now the real Fly→Assemble→Drop runbook grounded in the on-disk meta.json (liftoff-episodes + off-branch UC-71 tool), AC3 asserts new captures resolve to split==train. Verified: `in_view_hit_ray_share` is additive (no exact-key test breaks), all reuse targets exist (`uc83_eyes_v2.sh`, `run_label`/raycast, `load_capture`, test helpers), held-out split is catalogue-driven so the invariant is enforceable. Boundary clean (AC1/AC2 + AC5 numbers user-side; AC3/AC4/AC6 + sim-only retrain in-sandbox). No UC-85 scope creep. Non-blocking nit: mirror exact on-disk meta.json field names in the AC1 schema contract.
