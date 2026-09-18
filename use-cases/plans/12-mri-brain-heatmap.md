---
plan_for: use-cases/12-mri-brain-heatmap.md
work_branch: feat/uc-12-mri-brain-heatmap
team: drone-fly-uc-12
approved: 2026-09-18
---

# UC-12 — MRI-style full-brain activation heatmap

Challenger approved; the 5 binding conditions are folded in (marked ⟨C1⟩–⟨C5⟩). Viewer-only change; **no recorder/schema change**. All paths under `/workspace/drone-fly-uc-12-mri-brain-heatmap/`.

**Write scope (orchestrator-confirmed):** the developer may write `viz/**`, `scripts/**`, `README.md` (all authorized in the target `CLAUDE.md`); QA writes `tests/**`. `python`/`uv` granted for the one-time `build_brain_outline.py` run.

## Analysis
The anatomical panel today (`viewer.js` `drawBrainMap`/`bounds`/`projectedPoints`) renders ~10k per-neuron dots and **auto-fits to the bbox of whichever neurons are in the current recording** (non-uniform stretch `sx`/`sy`). That auto-fit is fundamentally incompatible with a fixed full-brain outline (a pruned slice's bbox is a sub-region of the brain). The real architectural change is replacing per-recording auto-fit with a **fixed full-brain→canvas transform** shared by both the outline and the activation splats. Soma coords are raw neuPrint MaleCNS 8 nm voxel space (e.g. `[46716,21029,15714]`), sourced tokenless from `connectome_data_prep`'s public CC-BY `mcns_all_neuron_meta.csv` `somaLocation` column.

## Proposed Solution

**`viz/brain_outline.js`** (NEW, committed static asset) — `const BRAIN_OUTLINE = { voxel_space:"neuPrint MaleCNS 8nm", attribution:"<bare string, NO http(s):// literal ⟨rec⟩>", region:"<brain|whole-CNS, documented ⟨C4⟩>", bbox3d:{min:[x,y,z],max:[x,y,z]}, planes:{ top:{axes:[0,2],polygon:[[u,v]...]}, front:{axes:[0,1],...}, side:{axes:[1,2],...} } }`. Raw voxel units so it co-registers with `coords3d` by construction. Loaded via a classic `<script>` (file://-safe, no fetch/module).

**`scripts/build_brain_outline.py`** (NEW, dev-time, tokenless) — mirrors `fetch_soma_positions.py` (argparse/urllib/pandas). Downloads the public meta CSV, parses all ~161k `somaLocation` points, projects to the three planes (top=xz, front=xy, side=yz per `MAP_VIEW_PRESETS`/`PROJECTIONS`), computes a compact concave silhouette per plane (alpha-shape or density-grid marching-squares; ~80–150 verts), the full-brain 3D bbox, and emits `viz/brain_outline.js`. ⟨C4⟩ Region ambiguity: the full meta may include VNC somas → the hull could be whole-CNS. The script MUST inspect the CSV for a usable region/soma-side field and, if present, filter to the brain region so the typical population isn't a tiny speck; if absent, ship the whole-CNS hull and record `region:"whole-CNS"` in the asset + README. Registration is exact either way (the "only used regions light up" behavior is intended). Output committed like `mcns_fixture_soma.csv`.

**`viz/viewer.js`** (MODIFY):
- **Fixed full-brain transform helper.** Project `BRAIN_OUTLINE.bbox3d` to the active plane's two axes → canvas with **uniform scale + centering** (preserve aspect; replaces stretch-fit for anatomical recordings) and the **same `H - y` Y-flip** the splats use. ⟨rec⟩ Outline polygon and splats MUST go through the identical transform incl. the flip, or the outline mirrors.
- **Remove dots + beat entirely** (AC1): delete `BEAT_REST/BEAT_PEAK/BEAT_SPAN/BEAT_RELEASE_TAU`, `state.beat`, `seedBeat()`, `updateBeat()`, and the dot loop in `drawBrainMap`. (Beat lived only in the brain map; flight panel has none — verified.)
- **Splat heatmap `drawBrainMap`.** Precompute **once per view** (load/view-change only) each neuron's screen (x,y) via the fixed transform (perf pitfall). Per frame: accumulate `frames.activations[frame][i]`-weighted Gaussian stamps additively into an offscreen Float32 accumulation buffer, normalize, map through an MRI/fMRI "hot" colormap (black→red→orange→yellow→white), `putImageData`→`drawImage` (smoothed for softness), then stroke the outline polygon (+ faint fill) on top.
- **Normalization toggle (AC5).** `state.mapNorm` ∈ {`frame`,`global`}; default `frame` (punchy "lights up"). ⟨C3⟩ **Decision: exact global max via a one-time fine-buffer pre-pass** at load and on view-change — replay every frame's accumulation once into a scratch buffer, track the true global peak. This is exact (no coarse-cell approximation, no clipping). Per-frame mode normalizes by that frame's own peak. Normalized values clamped to [0,1] regardless.
- ⟨C1⟩ **Missing-asset guard (AC7/AC8).** Guard `typeof BRAIN_OUTLINE !== "undefined"`; if absent, degrade to auto-fit splats with no outline — never a `ReferenceError`. Same code path as the non-anatomical degrade.
- ⟨C2⟩ **`tick()` cleanup.** With the beat gone the splat changes only when `state.frame` advances → redraw the brain map **only on `frameChanged`**; drop the per-rAF `else drawBrainMap()` branch (no re-splatting 10k gaussians 60×/s while idle).
- **Graceful degradation (AC8).** When `positions.source` is not anatomical (computed spectral layout, values ~[-0.1,0.1], not voxel space) or `has_position` all-false: skip the fixed voxel frame, auto-fit splats within the canvas, suppress the outline, keep the "computed (NOT anatomical)" badge. Neurons with null `coords3d` on a non-default plane don't contribute there (as today). No crash for legacy files.
- Keep `MAP_VIEW_PRESETS` + the `map-view-select` handler; add a `map-norm-select` change handler.

**`viz/viewer.html`** (MODIFY): add `<script src="brain_outline.js"></script>` on the line **immediately before** the byte-identical `<script src="viewer.js"></script>` ⟨rec⟩; add a norm toggle `<select id="map-norm-select">` (options `per-frame`/`global`, per-frame default) into the anatomical panel `.panel-tools` beside `map-view-select`; replace the role-dot legend with an MRI intensity-ramp legend. Keep `map-view-select` values `front`/`side`/`top` with `top` default (pinned by a test).

**`viz/viewer.css`** (MODIFY): minor styles for the intensity-ramp legend + norm toggle.

**`README.md`** (MODIFY): rewrite the "Anatomical brain map" bullet (dots+beat → MRI heatmap over static registered outline); document the normalization toggle, the outline asset + `build_brain_outline.py` regen + region/attribution ⟨C4⟩. Keeps the README contract honest and unblocks the README test reconciliation.

## Files Affected
**Production code (developer):** `viz/brain_outline.js` (new), `viz/viewer.js`, `viz/viewer.html`, `viz/viewer.css`, `scripts/build_brain_outline.py` (new), `README.md`.
**Test code (qa):** `tests/test_viz_contract.py`.

## QA scope — test evolution (challenger-confirmed legitimate; UC-06-removed-UC-05-flat-path precedent)
Exactly two existing tests break and both are authorized (AC1 mandates dot removal; the pitfall note says the beat goes with the dots):
- `test_viewer_js_has_neuron_beat_envelope` — remove/replace (beat is gone).
- `test_readme_documents_uc06_3d_controls_and_beat` — reconcile the `"beat"` assertion with the rewritten README.
**Add UC-12 assertions (static grep tokens — CI is headless):** dots/beat removed from anatomical panel; `brain_outline.js` exists and is referenced by `viewer.html`+`viewer.js`; `BRAIN_OUTLINE` consumed; additive splat present (`globalCompositeOperation`/`"lighter"`); a colormap present; `map-norm-select` exists with `per-frame`/`global` values + per-frame default (mirror the `map-view-select` `_extract_select`/`_options` pattern); `brain_outline.js` carries no `http(s)://`; `node --check viz/viewer.js` and `node --check viz/brain_outline.js` pass.
⟨C5⟩ **MANUAL browser checks (document; NOT automatable — headless CI, per the "a right turn looks right" precedent):** AC3 heatmap animates across frames; AC4 outline+splats visually register (heat sits inside the outline, aligned) across all three presets; AC5 toggling per-frame↔global visibly changes contrast/comparability; AC6 outline projects consistently per plane. QA must list these as manual and NOT claim automated coverage for them.

## Risks & Considerations
- **R (perf):** ~10k neurons/frame — mitigated by per-view precomputed positions + accumulation-buffer/colormap (cheap array math + one drawImage), not 10k gradient fills. The global-norm pre-pass ⟨C3⟩ is O(nFrames×neurons×stamp) once per load/view-change; fine for typical recordings, but flag if a recording has very many frames.
- **R (aspect-ratio change):** anatomical rendering moves from stretch-fit to uniform-scale — existing non-UC-12 recordings' maps will look different. Justified by registration.
- **R (outline region ⟨C4⟩):** documented above; readability, not registration.
- **R (network for asset build):** `build_brain_outline.py` downloads the public CSV; if network is unavailable in the sandbox and the CSV isn't already cached locally, the developer must flag it rather than commit a partial/empty asset.

## Challenger verdict
**APPROVED** (no revision loop; 5 minor conditions folded in). Verified: AC4 registration exact by construction (outline from the same soma cloud recordings use; neuPrint mesh correctly rejected); AC7 dependency-free/no-build/canvas-2D/file:// intact (committed const via classic script, network only in dev-time build script); AC1 dots+beat fully removed (beat is brain-map-only); the two breaking tests are legitimate evolution (UC-06→UC-05 precedent).
