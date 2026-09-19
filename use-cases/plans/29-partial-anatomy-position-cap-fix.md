---
plan_for: use-cases/29-partial-anatomy-position-cap-fix.md
work_branch: feat/uc-29-partial-anatomy-position-cap-fix
team: drone-fly-uc-29
approved: 2026-09-19
---

# UC-29 — Partial-anatomy connectomes above the spectral cap must still provision positions
**FINAL proposal — approved by challenger (first round).**

## Analysis

**Bug.** Recording-enabled training on a large pruned slice (real hit at n=122,816) raises from `ActivationRecorder.__init__` (`src/drone_fly/record/recorder.py:150-162`): "Neuron positions could not be provisioned … exceeds the spectral-layout cap and has no/partial anatomy." The slice HAS real anatomy (central neurons carry `somaLocation`), and UC-28 places soma-less afferents via cheap crc32 schematic clusters (no eigendecomposition). Provisioning needs ZERO spectral compute — yet it is wrongly refused.

**Root cause (verified in worktree).** The large-connectome guard in `resolve_positions` (`coordinates.py:1205`):
```
if n > _spectral_cap() and _anatomy_coverage(compute_data, override=meta_map) < n:
```
The `< n` defers on ANY partial anatomy — stale from UC-27, when the soma-less remainder needed the expensive spectral `np.linalg.eigh` fill. UC-28 replaced that with schematic placement (`provision_positions`, `:705-812`): with ≥1 real soma, `_schematic_body_coords` (`:640-673`) anchors soma-less afferents to the real-soma brain bbox — no eigh. The spectral path (`_computed_positions` → `_spectral_embedding`, `:676-704`) now runs ONLY when there is zero real anatomy to anchor.

**Guard↔provision consistency is provably exact (verified + confirmed by challenger).** `_anatomy_coverage(compute_data, override=meta_map)` (`:1102-1117`) and `provision_positions` both call the SAME `_load_anatomical(data, override=meta_map)` on the same `compute_data`. Biconditional:
- coverage>0 ⟺ ≥1 connectome id intersects the mapping ⟺ `real.shape[0]>0` in `_schematic_body_coords` ⟺ cheap real+schematic path (no eigh), at any n.
- coverage==0 ⟺ `_load_anatomical` is None OR mapping intersects zero ids → `_schematic_body_coords` None → `_computed_positions` (spectral).

So `== 0` refuses **exactly** the cases that would hit eigh and admits every case UC-28 can place cheaply — no 200GB-eigh regression possible. The pitfall's "foreign-id/disjoint mapping" is covered: `_resolve_meta_soma` (`:386-449`) subsets the meta map to `data.neuron_ids` (`:445-448`), so meta_map never carries foreign ids; a disjoint mapping arriving via a lower tier still yields coverage 0 by the intersection count → guard defers.

## Proposed Solution

**1. Narrow the guard** (`coordinates.py:1205`): `_anatomy_coverage(compute_data, override=meta_map) < n` → `... == 0`. Rest of `resolve_positions` untouched.

**2. Fix deferral WARN wording** (`coordinates.py:1206-1214`): "No/partial anatomy for a %d-neuron connectome" → describe the true remaining condition, **zero** real anatomy at scale (keep the `%d`/`%d`/NEUPRINT_TOKEN/SOMA_CSV_ENV args). Drop "partial".

**3. Fix recorder raise + comment** (`recorder.py:155-162`): "no/partial anatomy" → "**has zero anatomy** … Prune the connectome before recording, or provide anatomy via NEUPRINT_TOKEN or DRONE_FLY_SOMA_CSV." Keyed on zero anatomy, consistent with the guard.

**4. Update docstrings** (no behavior): `coordinates.py` module "Large-connectome guard" paragraph (`:80-86`) "anatomy is absent/partial" → "there is zero real anatomy"; guard inline comment (`:1201-1203`) if it says partial.

No change to training dynamics, observation schema/width, connectome graph, reward, or checkpoints. `DRONE_FLY_SPECTRAL_MAX` (`_spectral_cap`, `:838-865`) still works; becomes unnecessary for partial-anatomy connectomes. All UC-27/UC-28 behavior preserved.

## Files Affected

**Production code (developer):**
- `src/drone_fly/record/coordinates.py` — guard `< n` → `== 0` (:1205); deferral WARN wording (:1206-1214); module-docstring guard paragraph (:80-86) + guard inline comment (:1201-1203).
- `src/drone_fly/record/recorder.py` — RuntimeError message + preceding comment (:155-162).

**Test code (qa):**
- `tests/test_record_provisioning.py`:
  - **Invert** `test_ac11_partial_anatomy_above_cap_writes_soma_but_defers_positions` (:459-478) → "partial anatomy above cap now PROVISIONS": cap below n, `_build_meta_connectome` (6 neurons, 4 soma-bearing), assert `result is not None`, spy `_spectral_embedding` == [] (AC-1 zero spectral), `has_position == [T,T,T,T,F,F]`, all `display3d` finite, `_positions.csv` written, real soma sidecar written. **Also positively assert** `placement == ["anatomical"×4, "schematic"×2]` and `region` non-empty for the two soma-less slots — proves the cheap schematic body was actually placed, not merely "not None" (challenger rec). Rename.
  - **Keep** `test_ac11_no_anatomy_above_cap_defers_without_eigh` (:446-457) — zero anatomy above cap still defers, no eigh, no positions.csv (AC-3). **Add** a `caplog` assertion that the coordinates.py deferral WARN contains "zero" and NOT "partial" (challenger note 2 — AC-6 also covers the WARN, not just the RuntimeError).
  - **Keep** `test_ac11_complete_anatomy_above_cap_provisions_without_eigh` (:481-501) — full anatomy above cap (AC-1/AC-4).
  - **Add** disjoint-id edge case (pitfall). Per challenger note 1: use a **bare synthetic connectome with NO meta somaLocation** (so `meta_map == {}` — NOT `_build_meta_connectome`, whose meta would make `_load_anatomical` return the override outright and never consult the env CSV). Point `DRONE_FLY_SOMA_CSV` at a soma CSV whose bodyids are **disjoint** from the connectome ids, cap below n. Assert: `_load_anatomical(compute_data, override={})` **is not None** (a real mapping WAS loaded — so the test exercises the intended "non-empty mapping, zero effective coverage" path, not an accidental None path), `result is None`, spectral spy == []. Proves the guard keys on *effective* coverage, not mapping-non-empty.
  - **Add/confirm** small-connectome regression (AC-4): n ≤ cap in each anatomy state behaves as before (if not already covered elsewhere).
  - Update module docstring AC-11 line (:20-22): "zero anatomy above the cap defers; partial/complete provisions".
- `tests/test_record_recorder.py`:
  - **Add** AC-2: constructing `ActivationRecorder` on a partial-anatomy above-cap connectome no longer raises.
  - **Add** AC-6 wording: on a zero-anatomy above-cap connectome the recorder still raises, and the message contains "zero" and NOT "partial".

## Risks & Considerations
- **Message/guard drift** (pitfall): mitigated by AC-6 tests asserting BOTH the recorder RuntimeError and the coordinates WARN are keyed on "zero", not "partial".
- **Foreign-id mapping via non-meta tiers**: `_anatomy_coverage`'s intersection count returns 0 → guard defers; the disjoint-id `DRONE_FLY_SOMA_CSV` test (with `_load_anatomical`-not-None assertion) exercises exactly this.
- **No new spectral risk**: the change only *removes* deferrals for cases already destined for the cheap schematic path; never routes a new case into spectral.
- **CI hermetic/offline**: synthetic connectomes + committed `uc27_soma_meta.csv`; no network/token. `node --check viz/viewer.js` unaffected.

## AC → where satisfied
- **AC-1** → guard `== 0`; inverted partial test (spectral spy == [], placement/region asserts) + complete-anatomy test.
- **AC-2** → guard fix; new recorder no-raise test.
- **AC-3** → zero branch preserved; kept no-anatomy test (+ WARN wording caplog) + disjoint-id edge test + recorder raise test.
- **AC-4** → guard only fires when `n > _spectral_cap()`; small-connectome regression test.
- **AC-5** → `_spectral_cap` untouched; env sets cap in tests.
- **AC-6** → WARN + RuntimeError reword; wording assertions on BOTH messages.
- **AC-7** → changes limited to guard + wording + tests + docs; hermetic offline suite; Risks section.

**Challenger verdict: Approve** (guard `== 0` verified provably correct in code; two minor test notes folded in). Ready for the developer and QA.
