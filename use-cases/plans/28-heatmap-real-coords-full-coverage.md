---
plan_for: use-cases/28-heatmap-real-coords-full-coverage.md
work_branch: feat/uc-28-heatmap-real-coords-full-coverage
team: drone-fly-uc-28
approved: 2026-09-19
---

# ORCHESTRATOR WRITE-SCOPE GRANT (this run only)
`paths.production` in the brief is `src/drone_fly/**`. For UC-28 the developer is **explicitly authorized** to also write these load-bearing out-of-glob files (precedent: UC-12/UC-15 edited the viewer):
- `viz/viewer.js`
- `viz/viewer.html`
- `viz/viewer.css`
- `README.md`

No other out-of-glob writes. QA remains scoped to `tests/**`. `PLAN_FILE` and the ledger are orchestrator-written (exempt).

---

# UC-28 — FINAL APPROVED PROPOSAL (analyst ↔ challenger agreement, Revision 2)

Heatmap on real coordinates + complete body-schematic neuron coverage.

## Analysis (surfaces pinned)
- **Coordinate provenance**: `src/drone_fly/record/coordinates.py::provision_positions()/resolve_positions()` builds the positions dict `{source, projection, coords3d, coords2d, has_position}`. `coords3d[i]`=real soma xyz or `None`; `has_position[i]`=True iff real soma; `coords2d[i]` always finite. Soma-less afferents currently get `coords3d=None`, `has_position=False`, `coords2d`=spectral min-max-scaled INTO the brain bbox (`_scale_into_box` + `missing_idx` branch, coordinates.py:528-542) — overlapping the brain. **This is the surface UC-28 replaces.**
- **Recorder** (`record/recorder.py`) serializes the whole positions dict verbatim into `meta.positions` (+ `meta.superclass`/`roles`/`neuron_ids`), all index-aligned → any new positions-dict key flows to the viewer JSON with no recorder change.
- **Viewer** (`viz/viewer.js`): `projectedPoints()` (:274-287) consumes positions; `stampFrame` (:408) joins activation→neuron **purely by array index `i`** (`act[i]` at `sx[i],sy[i]`). ⇒ AC-8 "bodyid join stays exact" = **preserve index alignment; never reorder** any positions array. `ensureMapCache` (:369) keys the fixed-mode transform off `BRAIN_OUTLINE.bbox3d`.
- **Fixture reality** (verified): `tests/fixtures/mcns_fixture_meta.csv` = **322 neurons, 272 real soma, 50 soma-less**; soma from committed `mcns_fixture_soma.csv`. `ConnectomeData` carries `superclass/neuron_class/subclass/cell_type`. All 50 soma-less carry: `subclass` {campaniform 41, haltere 8, leg 1} + `superclass` {sensory_ascending 25, vnc_sensory 25}. **Region label lives in `subclass` (limbs) + `superclass` (vnc/ascending), NOT `cell_type`** (opaque codes). **No blank-label soma-less exists** in the fixture → AC-4/vnc/ascending branches need a synthetic fixture.

## Proposed Solution
**Compute schematic body placement in Python** (`coordinates.py`), not JS — makes AC-3/4/5/8 pytest-testable (the viewer is only ever checked statically + `node --check`; visuals are documented MANUAL checks per UC-12/15 precedent). Region labels are already on `ConnectomeData` at every `resolve_positions` call site (recorder, cli fetch/prune, prune_trained) — no new threading. Viewer stays thin, consuming new fields.

### New positions-dict fields (additive; honesty + back-compat preserved)
1. `placement`: list[str] ∈ {`anatomical`,`schematic`,`computed`} — real soma / soma-less body-cluster / whole-slice spectral fallback. Drives AC-6 glyph distinction + per-neuron honesty.
2. `region`: list[str] — body region for schematic neurons (`leg`/`haltere`/`campaniform`/`vnc`/`ascending`/`torso`); `""` for anatomical & computed.
3. `display3d`: list[[x,y,z]] — **full-coverage 3D render coords**, always finite (real soma / schematic-body 3D / spectral) → lets the viewer render every neuron on every plane (AC-2) and enforce the guardrail in 3D (AC-5).

**Unchanged (honesty + AC-8 + UC-27):** `coords3d` stays real-xyz-or-`None` (sidecar `x/y/z` blank iff `None`); `has_position`/`source`/`projection` unchanged. `coords2d` = projection of `display3d`.

### coordinates.py (production)
- `resolve_body_region(data) -> list[str]` (index-aligned), ordered constant `REGION_LABEL_RULES` with documented precedence (limb subclass wins over coarse vnc/ascending): `subclass=="leg"`→leg; `subclass=="haltere"`→haltere; `subclass` startswith "campaniform"→campaniform; `superclass=="vnc_sensory"`→vnc; `superclass=="sensory_ascending"`→ascending; else→`torso` (`REGION_TORSO`).
- `_schematic_body_coords(...)`: brain bbox from real `coords3d`; cluster center = `brain_center + REGION_CLUSTER_OFFSETS[region]·brain_extent` (bounded by `MAX_BODY_OFFSET_FACTOR`); within-cluster spread via **`zlib.crc32(str(bodyid).encode("utf-8"))`** → deterministic point in a sphere of `REGION_CLUSTER_RADIUS_FRAC·brain_extent`. **Builtin `hash()` FORBIDDEN** (PYTHONHASHSEED-randomized; ids can be str) — documented in a code comment. No float RNG; bit-reproducible cross-process.
- Rework `provision_positions` branches: anatomical→`placement="anatomical"`; soma-less `missing_idx` branch → **replace `_scale_into_box` with schematic body placement**, `placement="schematic"`, `region=…`; whole-no-anatomy branch → `placement="computed"`, spectral (unchanged — no brain to anchor), `region=""`.
- Extend `POSITIONS_SIDECAR_COLUMNS` + `_write/_load_positions_sidecar` to round-trip `placement`, `region`, `display3d` (x3d/y3d/z3d). Old sidecars → MISS → self-heal recompute (no committed `_positions.csv`, so nothing breaks).
- Named constants (documented): `REGION_CLUSTER_OFFSETS`, `MAX_BODY_OFFSET_FACTOR`, `REGION_CLUSTER_RADIUS_FRAC`, `BRAIN_DOMINANCE_MIN_FRACTION`. **Guardrail math**: total extent ≤ `brain_extent·(1+2·(MAX+RADIUS))` ⇒ pick constants so `1/(1+2·(MAX+RADIUS)) ≥ BRAIN_DOMINANCE_MIN_FRACTION` with margin (e.g. MAX≈0.42, RADIUS≈0.08 → ≈0.5). Document the relationship so test & constants can't drift.

### recorder.py (production)
- Add `meta.modality`: per-neuron tag (index-aligned, `""` when none), computed once at `__init__` via `controller.modality.select_modality` for the **exact fixed set `{vision, proprioceptive, hunger}`** (the only rules that exist + resolve here). **Fail-soft**: catch base `ModalityError` (covers all 3 subclasses) → skip an absent modality, never crash. **`damage`/nociception is documented as unavailable** (no rule, no MaleCNS label) in code + README + viewer legend — no silent gap (D8). No circular import (verified: `controller.modality` imports only `connectome.loader`); read-only w.r.t. training/checkpoints (AC-9).

### viz/viewer.js + viewer.html + viewer.css (production — under the write-scope grant above)
- `projectedPoints()`: prefer `display3d[i]` projected on the current plane → every neuron on every plane; legacy fallback to coords3d/coords2d.
- **Fixed-mode transform fix**: widen `makeTransform` inputs in `ensureMapCache` to the **union** of `BRAIN_OUTLINE.bbox3d` and the `display3d` extent, else body clusters clip off-canvas.
- AC-6 distinction via `placement` (reduced splat weight/opacity + spatial separation); activation coloring applies to both.
- AC-7: new `viewer.html` toggle (`modality-tag-select`) → modality-marker overlay drawn ON TOP of the heatmap, does not touch the hot colormap; legend lists only `{vision, proprioceptive, hunger}`.

### README.md (docs)
- Document real-coord rendering, body-schematic placement + region mapping + constants, modality toggle, damage-unavailable note. (MEMORY: update README at end of every UC.)

## Files Affected
**Production code (developer):** `src/drone_fly/record/coordinates.py`, `src/drone_fly/record/recorder.py`, `viz/viewer.js`, `viz/viewer.html`, `viz/viewer.css`, `README.md` (last four under the write-scope grant above).
**Test code (qa):**
- `tests/test_record_coordinates.py` — **update** existing soma-less fallback assertions (they assert spectral-into-box coords2d for the 50 afferents — UC-28 legitimately changes this) + AC-1/2/8 cases.
- `tests/test_viz_contract.py` — extend positions-schema contract (`placement`/`region`/`display3d`), `meta.modality`, static viewer.js assertions (display3d, placement, modality toggle), `node --check`.
- New `tests/test_record_body_schematic.py` — AC-3 cluster placement + **cross-process determinism**; **synthetic fixture exercising all six `REGION_LABEL_RULES` branches** (leg/haltere/campaniform/vnc/ascending/torso), AC-4 torso; AC-5 guardrail on **3D bbox AND projected default plane**; AC-8 mixed real+schematic+computed no-NaN + index alignment; `meta.modality` tag set exactly `{vision, proprioceptive, hunger}∪{""}`.

## AC → where satisfied
- **AC-1** real coords: real soma→`coords3d`/`display3d`; viewer projects. Test: fixture real-soma neuron at known coords.
- **AC-2** complete coverage: `display3d` finite for all N; viewer returns non-null for all → rendered==N. Test: `len(display3d)==n` all-finite, `placement` covers N.
- **AC-3** deterministic body placement: `resolve_body_region` + crc32 spread. Test: limbs land in cluster; cross-process identical.
- **AC-4** generic→torso: synthetic blank-label soma-less → `region="torso"`, torso cluster not a limb.
- **AC-5** guardrail: bounded offsets + constants; brain bbox ≥ `BRAIN_DOMINANCE_MIN_FRACTION` × total (3D + default plane).
- **AC-6** distinction: `placement` + viewer glyph/opacity + spatial separation; activation coloring on both.
- **AC-7** modality toggle: `meta.modality`{vision,proprioceptive,hunger} + viewer overlay toggle not overriding hot colormap.
- **AC-8** graceful/alignment: `display3d` always finite; arrays index-aligned, never reordered; positional join intact.
- **AC-9** scope+CI: no training/obs/reward/checkpoint change; viewer dependency-free; `node --check` + `ruff check` + `ruff format --check` + `uv run pytest` green offline; constants named/documented.

## Risks & Considerations
1. **viz/+README outside paths.production** → orchestrator write-scope grant recorded at top of this file.
2. Region label in `subclass`+`superclass`, not `cell_type` — multi-column precedence (verified).
3. Fixture has no blank-label soma-less → AC-4/vnc/ascending via synthetic fixture (full 6-branch coverage).
4. Fixed-mode transform must widen to outline∪display3d or body clips off-canvas.
5. Schematic offsets relative to the real-soma bbox present in-file (guardrail uses same bbox) → body scale tracks the actual in-file brain.
6. Changing soma-less coords2d breaks current test_record_coordinates.py expectations — intended; QA updates.
7. Sidecar schema bump → external `_positions.csv` recompute (self-heal); none committed → safe.
8. Determinism must be crc32 (never builtin `hash()`); cross-process QA assertion.
9. `meta.modality` fail-soft on absent modality; damage documented unavailable.
10. Guardrail constants ↔ asserted fraction kept consistent + documented.

**Status: APPROVED by challenger (Revision 2). Cleared to implementation.**
