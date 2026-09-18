---
plan_for: use-cases/13-modality-mapped-populations.md
work_branch: feat/uc-13-modality-mapped-populations
team: drone-fly-uc-13
approved: 2026-09-18
---

# UC-13 — Modality-mapped neuron populations + graft-ready observation schema (Option B)

Challenger-APPROVED (peer loop converged ~round 5/6). This is the analyst↔challenger agreed plan in full.

## Recorded decisions (must appear prominently in the PR)
1. **User chose Option (B):** the committed test fixture is regenerated to carry real `class`/`subclass` from the full MaleCNS meta AND re-selected to contain a real proprioceptive/mechanosensory population + vision. AC4's re-bind and AC6's fail-loud run against genuine biological labels on the CI fixture — no synthetic stand-in, no `ascending_neuron` proxy. Fixture regeneration + its generator are in-scope for this UC.
2. **AC4 checkpoint invalidation:** re-binding the observation deliberately changes input→sensory wiring; the pre-existing 250k checkpoint is NOT carried over — a retrain under the new schema is required. Call this out loudly in the PR.
3. **UC-05 soma coverage-contract change (consequence of B):** real sensory afferents have no brain-volume somata, so the regenerated fixture has partial soma coverage by construction. The existing 100%-coverage assertion is updated to an exact partition (below). Nothing is fabricated; missing somata are flagged, not invented. Agreed in-scope for (B), no new user decision needed.

## Orchestrator role-boundary ruling (this run)
- **Developer** owns: `src/drone_fly/**`, `scripts/build_test_fixture.py`, and **regenerating the committed fixture artifacts** `tests/fixtures/{mcns_fixture.npz,mcns_fixture_meta.csv,mcns_fixture_soma.csv,FIXTURE_PROVENANCE.md}` by RUNNING that build script (generated data, prerequisite for the developer's own actor/smoke-train verification). Developer MUST NOT edit any `tests/*.py`.
- **QA** owns: all `tests/*.py` — the NEW test files, the fixture-derived-constant recompute-sweep, and the exact-partition rewrite. QA reads the regenerated fixture the developer produced; QA does NOT re-run the fixture generator or edit `src/**`/`scripts/**`.
- Authorization for the developer to write `tests/fixtures/**` is added to the target `CLAUDE.md` for this run.

## Feasibility (verified offline against the cached full MaleCNS — no network)
Cached: `~/.cache/drone-fly/connectome_src/mcns_inprop_all_neuron.npz` + `mcns_all_neuron_meta.csv`. The meta carries `class`+`subclass`. Real populations: `mechanosensory_proprioceptive`=1383 (mechanosensory*=5512), plus visual/olfactory/gustatory/thermosensory/hygrosensory. Proprioceptive→descending is genuinely wired (all 1310 descending reachable ≤6 hops). Regeneration is deterministic and offline.

## Analysis
Current pipeline: loader (`src/drone_fly/connectome/loader.py`) surfaces only superclass/sign/top_nt via the `_OPTIONAL_META_COLUMNS` tuple; `ConnectomeData` has no class/subclass. Actor (`src/drone_fly/controller/actor.py:121`) uses one `Linear(OBS_DIM→sensory_size)` scattered via `index_add` into a single `visual_projection` sensory population; motor readout gathers `descending_neuron`. Obs (12-d) is built in `env/racing_env.py::RaceEnv._observation` = target-relative(3)+attitude(3)+linvel(3)+angvel(3); `OBS_DIM=12` in `controller/encoding.py`. Prune (`connectome/prune.py`) builds the directed visual_projection→descending_neuron corridor. Graft precedent: `prune_trained/transfer.py` copies input_projection/readout 1:1 (key+shape). smoke_train (`train/loop.py`) defaults `prune=False` → runs on the raw fixture.

## Proposed Solution
- **AC1 (loader class/subclass):** in `loader.py`, replace the `_OPTIONAL_META_COLUMNS` tuple with a CSV-column→attribute map, adding `class→neuron_class` (reserved-word alias) and `subclass→subclass`; add `neuron_class`/`subclass` fields to `ConnectomeData` (default None, same degrade pattern). Update `_load_neuron_attrs`, and `save_connectome` to round-trip both back as CSV `class`/`subclass`. Update `connectome/prune.py::slice_connectome` to carry `neuron_class`/`subclass` through the induced slice (mirror the superclass/sign/top_nt `[kept]` handling). Purely additive; existing superclass selection untouched.
- **AC2 (modality selector):** new `controller/modality.py` — a documented registry mapping each modality to a selection rule over a `ConnectomeData`: vision→superclass `visual_projection`; mechanosensory/proprioceptive→`class` (`mechanosensory_proprioceptive`/`mechanosensory`*); gustatory/olfactory/thermo/hygro→`class`; hunger/motion→curated cell_type/type name lists flagged `approximate=True`. Returns indices (into that data's row order → works on full or sliced) + approximate flag. Label strings VERIFIED against real data.
- **AC6 (fail-loud):** the selector is a DISTINCT path that RAISES on zero-match; it does NOT reuse `populations._select_by_superclass` (whose legacy degrade returns PLACEHOLDER and never raises). Two real cases: proprioceptive present in full fixture but dropped by the visual→descending prune → raise; gustatory/olfactory/thermo/hygro absent from fixture → raise.
- **AC3 (versioned block schema):** new `controller/obs_schema.py` — `ObsSchema` = ordered named blocks (name, width, bound population) + integer `version`; total width = sum of block widths; serializable so it rides in the checkpoint.
- **AC4 (decompose + re-bind + smoke-train):** additive `obs_schema` param on `ConnectomeActorNetwork`. `None` → current single-projection path, byte-identical (keeps UC-01..12 + UC-07 transfer.py green). Given → a `ModuleList` of per-block `Linear(block_width→len(bound_pop))`, each `index_add`-scattered into its bound population; motor readout + connectome graph unchanged. Migrated schema v1: vision block width 3 (target-relative)→visual_projection; proprioceptive block width 9 (attitude+linvel+angvel)→`mechanosensory_proprioceptive` (real, via the class selector). 3+9=12 → OBS_DIM and env `Box(12)` unchanged (env not edited). smoke_train(connectome=regenerated fixture, prune=False) with this schema asserts a finite completed update (trainability). Checkpoint not carried.
- **AC5 (zero-init graft + parity):** graft builds a NEW actor from the extended (wider) schema, then transfers weights transfer.py-style (copy existing block projections, readout, `layer.edge_weight`, and buffers by key+shape; zero-init ONLY the new block's `Linear` weight AND bias). NEVER `load_state_dict(strict=True)` of the pre-graft checkpoint into the wider actor. In schema mode `forward()` validates `x.shape[-1] == sum(block.width)`; the OBS_DIM/`Box(12)` check governs only the legacy/live path (AC7). Parity test at actor level with synthetic wider obs.
- **Config (UC-11 pairing):** add one validated `schema` key to `TrainRunConfig` via the existing `_Spec`/`from_mapping`, importing the default from `obs_schema` (never re-declared). AC7 parity: omitting it → legacy single-projection run exactly (default None → legacy path).
- **AC7:** env unchanged, no new live input; only parity + smoke sanity tests.

## Fixture regeneration (in-scope; deterministic; offline) — DEVELOPER
Change `scripts/build_test_fixture.py` selection to: **soma-filtered global-top-degree CORE** (central neurons — visual_projection, descending, hub intrinsics — all soma-populated) **∪ a top-degree `mechanosensory_proprioceptive` quota selected WITHOUT the soma filter** (real afferents, intentionally soma-less), then induced subgraph; deterministic `(-degree, idx)`, no randomness. Add `class`,`subclass` to `META_KEEP_COLUMNS`. Verified this preserves visual→descending prune non-degeneracy (16/16) and yields a genuinely-wired proprioceptive set. Regenerate `tests/fixtures/mcns_fixture.npz`, `mcns_fixture_meta.csv`, `FIXTURE_PROVENANCE.md` (record new rule + counts + CC-BY), and `mcns_fixture_soma.csv` in sync (offline from the meta's `somaLocation`; only the soma-populated core gets rows).

## Blast-radius recompute-sweep (explicit step — recompute, never guess) — QA
After the deterministic regen, grep `tests/` for every fixture-derived constant and recompute against the new fixture: `FIXTURE_EXPECTED_SCALE` (loader.py — note: developer verifies this src constant); `tests/test_prune.py` FIXTURE_PRUNE_SCALE {0:(59,751),1,2:(274,10070)} + doc table + lines 138-139/318-319/346-347/433/451-452; `tests/test_prune_trained.py:73` `UC04_FIXTURE_PRUNE_SCALE={0:(59,751),1:(175,5988),2:(274,10070)}`; `tests/test_cli.py:278` (59/751); `tests/test_encoding.py:29` (N=300); `tests/test_record_cli.py:168`+pruned 274; `tests/test_record_rollout.py:117`; `tests/test_policy.py:16` density comment; `tests/test_uc01_smoke.py:19`. Also record the new core_size/quota for the soma partition test.

## Soma coverage — EXACT partition test (per challenger refinement) — QA
Rewrite `tests/test_record_coordinates.py:66/74` from "300/300" to an exact partition (as strong, not relaxed): `sum(has_position) == core_size` (every core neuron real+finite+deterministic coords); every proprioceptive-quota neuron flagged `has_position=False`; and the identity `has_position[i] == (soma present for neuron i)` for all neurons. Reuses the existing flag-not-drop path (`test_record_coordinates.py:101`). Never fabricate coords; STOP+flag if `somaLocation` is unusable for a selected core bodyid.

## Files Affected
**Production / dev-time (developer):**
- `src/drone_fly/connectome/loader.py` (AC1 + FIXTURE_EXPECTED_SCALE recompute)
- `src/drone_fly/connectome/prune.py` (slice_connectome carries neuron_class/subclass)
- `src/drone_fly/controller/modality.py` — NEW (AC2/AC6)
- `src/drone_fly/controller/obs_schema.py` — NEW (AC3/AC5)
- `src/drone_fly/controller/actor.py` (additive obs_schema block path; schema-width check)
- `src/drone_fly/controller/sb3.py` (thread obs_schema through features_extractor_kwargs → pickled into checkpoint)
- `src/drone_fly/train/config.py` (+ `config.py` if needed) — validated `schema` key, AC7 parity
- `scripts/build_test_fixture.py` (modality-quota + soma-core selection; class/subclass columns)
- Regenerated `tests/fixtures/{mcns_fixture.npz,mcns_fixture_meta.csv,mcns_fixture_soma.csv,FIXTURE_PROVENANCE.md}` (developer runs the build script — authorized for `tests/fixtures/**` this run)
- `src/drone_fly/controller/__init__.py` (exports)

**Test (qa):**
- NEW `tests/test_modality.py` (class/subclass selection + approximate flags + BOTH fail-loud cases on real data)
- NEW `tests/test_obs_schema.py` (ordering/widths/version/total width)
- NEW `tests/test_graft.py` (zero-init graft parity: `torch.equal`, zero weight+bias, single+batched, existing params byte-unchanged, negative case where a non-zeroed block DOES change the action)
- `tests/test_connectome_loader.py` (class/subclass surfaced + degrade-to-None + save/load round-trip via crafted meta; FIXTURE_EXPECTED_SCALE)
- `tests/test_actor.py` (legacy path unchanged; schema path shapes)
- smoke-train test asserting the migrated schema trains finitely on the fixture (AC4)
- Recompute-sweep updates: `test_prune.py`, `test_prune_trained.py`, `test_cli.py`, `test_encoding.py`, `test_record_cli.py`, `test_record_rollout.py`, `test_policy.py`, `test_uc01_smoke.py`
- Exact-partition rewrite: `tests/test_record_coordinates.py`

## Risks & Considerations
1. **Checkpoint invalidation (AC4)** — intentional; surfaced loudly (recorded decision #2).
2. **Fixture regeneration blast radius** — many fixture-derived + prune-derived counts must be recomputed in lockstep (explicit sweep step above). Main cost of (B). Mitigated: reselection preserves visual→descending prune (verified) and all currently-used populations.
3. **Soma coverage** — real sensory afferents lack brain somata (0% across all sensory classes; central ~96–99%). Handled as a design invariant (soma-complete core + flagged-missing afferents) + exact-partition test (recorded decision #3). Backstop: never fabricate; STOP+flag if a selected CORE bodyid lacks `somaLocation`.
4. **Determinism/provenance** — regen must stay deterministic; provenance records the new rule + counts + CC-BY.
5. `class`/`subclass` label strings VERIFIED against real data (prior uncertainty retired).
6. **Additive obs_schema default** — legacy single-projection preserved so UC-07 transfer.py + existing tests stay byte-identical; a hard cutover would force a transfer.py rewrite (out of scope).

Approved by challenger. Ready to forward to developer + QA.
