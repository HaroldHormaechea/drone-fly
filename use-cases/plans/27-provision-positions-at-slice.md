---
plan_for: use-cases/27-provision-positions-at-slice.md
work_branch: feat/uc-27-provision-positions-at-slice
team: drone-fly-uc-27
approved: 2026-09-19
---

# UC-27 — Provision neuron positions at slice time — FINAL APPROVED IMPLEMENTATION PLAN

*Approved by challenger (rev 2). TARGET_DIR = `/workspace/drone-fly-uc-27-provision-positions-at-slice`. All paths below are absolute inside TARGET_DIR.*

---

## Analysis

**Current behavior (verified from source).** `ActivationRecorder.__init__` (`src/drone_fly/record/recorder.py:133`) calls `provision_positions(data, projection=...)` from `src/drone_fly/record/coordinates.py` on **every** recording-enabled run. The recorder is constructed in `src/drone_fly/train/loop.py:481`, only when `record=True` (`record` defaults `False`, `src/drone_fly/config.py:258`). It runs **once in the main process** (not per SubprocVecEnv worker → UC-26 parallel backend unaffected).

`provision_positions` resolves anatomy via `_load_anatomical`: tier-1 `DRONE_FLY_SOMA_CSV` env CSV → tier-2 `<connectome-stem>_soma.csv` sidecar (`_sidecar_path` derives it from `data.source`) → tier-3 neuPrint (`_load_from_neuprint`, guarded by `NEUPRINT_TOKEN`). For any neuron lacking a soma it computes a dense `np.linalg.eigh` spectral layout (`_spectral_embedding`, coordinates.py; warns for n>6000). It **caches/writes nothing**, so anatomy fetch + eigendecomposition are redone every run.

**Three distinct output branches of `provision_positions` (this drives the sidecar schema):**
1. **anatomical present** → `has_position=True`, `coords3d[i]=[x,y,z]` (real soma).
2. **partial-anatomy fallback** (some neurons missing soma) → `has_position=False`, `coords3d[i]=None`, `coords2d[i]` = spectral projection scaled into the anatomical extent via `_scale_into_box`.
3. **no anatomy at all** → returns `coords3d = embedding.tolist()` (**non-null for every neuron**) with `has_position=[False]*n` and `source=SOURCE_COMPUTED`. ← the null-ness of `coords3d` is therefore NOT derivable from `has_position`.

**Artifact/data facts.**
- `load_connectome` (`src/drone_fly/connectome/loader.py`) sets `data.source = str(npz_path)`, so at **train** time `_sidecar_path(data)` resolves sidecars correctly beside the artifact.
- `save_connectome(data, out, stem)` (loader.py) returns `(npz_path, meta_path)`, writes `bodyid` = `data.neuron_ids` with contiguous `idx`; `load_connectome` sorts by `idx` (no-op) → row order preserved. `bodyid` is the stable node-set key.
- At **slice/write** time the pruned `ConnectomeData.source` is the slice-source string, **not** the just-saved npz — so the write side must key sidecars off the explicit `npz_path` returned by `save_connectome`, never `data.source`.
- Three artifact-producing hooks, each already holding a `ConnectomeData` + a known output npz:
  1. `_run_prune_export` — `save_connectome(pruned, cfg.out)` at `src/drone_fly/cli/__init__.py:481` → `<out>/connectome_pruned.npz`. **Primary.**
  2. `_run_fetch_connectome` → `ensure_full_connectome()` (`src/drone_fly/connectome/fetch.py`) → `<dest>/mcns_inprop_all_neuron.npz` (+ `_meta.csv`); reuse-guarded, atomic; returns the dest dir.
  3. `prune_trained` workflow — `save_connectome(pruned, out, stem="connectome_pruned")` at `src/drone_fly/prune_trained/workflow.py:428`.
- The committed fixture has **partial** soma coverage by construction (per `tests/test_record_coordinates.py` header) — ideal for the partial-anatomy test.

**FIXED DESIGN DECISIONS honored** — all 7, plus the two sidecars: `<stem>_soma.csv` (real anatomy) and `<stem>_positions.csv` (computed layout). The only divergences from "unchanged vs today" (the large-connectome refuse cap and AC-2 being conditional on anatomy availability) are exactly FIXED DECISION #7 and are approved, not deviations.

---

## Proposed Solution

### (a) `<stem>_positions.csv` schema + node-set binding

**Columns (final):** `bodyid, has_position, x, y, z, u, v, source, projection`. One row per neuron. The file is a verbatim serialization of the `provision_positions()` dict, so a load is a pure deserialize → AC-8 holds trivially. `source` and `projection` are constant across rows.

**`coords3d` null-ness serialized DIRECTLY, independent of `has_position` (the critical correctness point):**
- **Write:** for each neuron, if `coords3d[i] is None` → write `x,y,z` as **empty cells**; else write the three values at **`%.17g`** (bit-exact float64). `has_position[i]` is written to its own column, independently. `u,v` (= `coords2d[i]`) are always written at `%.17g`.
- **Reconstruct:** `coords3d[i] = None if (x,y,z cells empty) else [float(x),float(y),float(z)]`; `has_position[i]` restored from its own column and coerced to Python `bool`; `coords2d[i] = [float(u), float(v)]` always.
- This round-trips all three branches exactly:
  - anatomical → `has_position=True`, `x,y,z` filled → `coords3d=[x,y,z]`
  - partial-anatomy fallback → `has_position=False`, `x,y,z` **empty** → `coords3d=None`
  - no-anatomy spectral → `has_position=False`, `x,y,z` filled → `coords3d=[x,y,z]` (non-null)

**Node-set binding / HIT rule (AC-7):** a positions sidecar is a **HIT** only if, after read, its `bodyid` set **equals exactly** the loaded connectome's `neuron_ids` set (same cardinality + members) **AND** its `projection` equals the requested projection. Coerce **both** id collections to a single canonical int type before the set-equality check and before the realign lookup (build `{int(bodyid): row}` vs `{int(x) for x in data.neuron_ids}`); guard non-int-coercible ids (fixtures may carry string/range ids) with the existing `_is_int_like` pattern and fall back to a string-normalized comparison so a valid sidecar still HITs. On a HIT, realign sidecar rows to connectome row order by bodyid lookup. Any of {unreadable file, missing columns, wrong length, id-set mismatch (pruned subcircuit vs full, stale artifact), projection mismatch} → **MISS** → recompute path (never partially applied, never applied to the wrong neurons). This single rule covers AC-7 + the stale/corrupt-sidecar pitfall.

**`<stem>_soma.csv`** (existing tier-2 sidecar): columns `bodyid,x,y,z` — exactly what `_load_soma_csv` / `_mapping_from_frame` already read. Written when neuPrint anatomy is fetched, so later loads hit tier-2 offline (AC-6).

### (b) New `coordinates.py` surface + hook wiring

In `src/drone_fly/record/coordinates.py`:
- `_sidecar_path(...)` — extend to accept an optional explicit npz base (default: derive from `data.source` — read-path behavior unchanged). Add `_positions_sidecar_path(...)` mirroring it → `<stem>_positions.csv`.
- `_load_positions_sidecar(pos_path, data, projection) -> dict | None` — read + validate (node-set + projection) + realign; reconstruct the positions dict per (a); return `None` on any miss.
- `_write_positions_sidecar(pos_path, data, positions, *, strict: bool)` — serialize the dict at `%.17g`; when `strict=False`, wrap the write so any `OSError` warns-and-continues instead of raising.
- Add `SPECTRAL_MAX_NEURONS` constant (env-overridable via `DRONE_FLY_SPECTRAL_MAX`, default ≥50000) + the refuse guard (see (d)). Keep the existing n>6000 warning.
- **`resolve_positions(data, *, projection=DEFAULT_PROJECTION, artifact_npz=None, persist=True, persist_strict=True) -> dict`** — single orchestrator:
  1. Compute `pos_path` / `soma_path` from `artifact_npz` (write side) else `data.source` (read side).
  2. **Fast path:** `cached = _load_positions_sidecar(...)`; if not `None` → **return it with zero calls to `_spectral_embedding` / `_load_from_neuprint` / `_load_anatomical`** (AC-4).
  3. **neuPrint fetch+persist:** if `NEUPRINT_TOKEN` set, no `DRONE_FLY_SOMA_CSV`, and `soma_path` absent → call `_load_from_neuprint(data.neuron_ids)`; if it returns a mapping, write `<stem>_soma.csv` (`bodyid,x,y,z`). neuPrint preferred over spectral, persisted once (AC-6).
  4. **Large-connectome refuse guard** (see (d)) — may WARN + defer here instead of computing.
  5. **Compute** via the existing `provision_positions(data, projection)` (now it finds the soma sidecar as tier-2) — output identical to today (AC-8).
  6. If `persist` → `_write_positions_sidecar(pos_path, data, result, strict=persist_strict)`.
  7. Return the dict.
  Keep `provision_positions` itself **output-identical** — it stays the compute core.

**Read path** — `src/drone_fly/record/recorder.py` (~:133): replace `provision_positions(data, projection=projection)` with `resolve_positions(data, projection=projection, persist=True, persist_strict=False)`. HIT → fast load (AC-4). MISS → compute-once + WARN ("provisioning positions on the fly") + best-effort persist → next run fast (AC-5 self-heal). `data.source` = npz path at train time, so sidecars resolve beside the artifact.

**Write hooks (strict persist, explicit `artifact_npz`):**
1. `src/drone_fly/cli/__init__.py` — in `_run_prune_export`, after `save_connectome` at line 481, call `resolve_positions(pruned, projection=DEFAULT_PROJECTION, artifact_npz=npz_path, persist=True)` (**AC-1**; writes `_positions.csv`, plus `_soma.csv` when anatomy fetched).
2. `src/drone_fly/cli/__init__.py` — in `_run_fetch_connectome`, after `ensure_full_connectome()`, load the just-fetched connectome and call `resolve_positions(..., artifact_npz=<mcns npz path>, persist=True)`, subject to the (d) large-connectome guard (**AC-2**).
3. `src/drone_fly/prune_trained/workflow.py` — after `save_connectome` at line 428, call `resolve_positions(pruned, artifact_npz=npz_path, persist=True)` for the subcircuit (**AC-3**; own node-set-keyed sidecars).
The orchestrator/root remains the sole artifact writer at slice time — this is a coordination-artifact write consistent with the existing `PRUNE_PROVENANCE.md` / report writes; these sidecars are exempt from developer/QA `paths.production`/`paths.test` scopes.

### (c) Partial-anatomy fill rule
Numerics unchanged — delegates to today's `provision_positions`: anatomical neurons → `has_position=True`, real `coords3d`; missing neurons → `has_position=False`, `coords3d=None`, `coords2d` = spectral projection scaled into the anatomical extent (`_scale_into_box`). The sidecar records **which strategy covered which neuron** via the per-neuron `has_position` column together with empty `x,y,z` cells for fallback rows. Output is always length-N with finite `u,v` → no length mismatch (satisfies the partial-anatomy pitfall + AC-7).

### (d) Full-connectome `fetch-connectome`, no/partial anatomy — WARN + defer (refuse cap)
A dense `eigh` on the full ~161k MaleCNS connectome is infeasible (a 161k×161k dense matrix ≈ 200 GB), so spectral must **never** run at that scale. `resolve_positions`'s guard: when a compute would require the spectral fallback for `n > SPECTRAL_MAX_NEURONS` neurons —
- **complete real anatomy available** (token→neuPrint fully covers the node set, or a soma CSV) → provision anatomical sidecars, **no spectral** (satisfies AC-2 when anatomy exists);
- **no OR partial anatomy** → do **not** attempt the giant eigendecomposition: emit a clear WARNING ("no/partial anatomy for the full connectome; positions not provisioned — set `NEUPRINT_TOKEN` or `DRONE_FLY_SOMA_CSV`, or prune before recording"), write `_soma.csv` for any covered subset, **skip `_positions.csv`**, and **defer** to train time (train-time self-heal on the full connectome also refuses at that scale — correct, since recording the full 161k graph is not a supported path; pruning first is). The **no-OR-partial** coverage is required because today's `_spectral_embedding` embeds the whole graph regardless of how many somas are missing, so even partial anatomy on a huge connectome would trigger a full-graph `eigh`.
AC-2 is thus conditional on anatomy availability (documented). Hermetic CI exercises `fetch-connectome` with a small mocked/fixture connectome (below the cap) with mocked/fixture anatomy, so positions ARE provisioned offline and AC-2 is testable without network.

---

## Files Affected

### Production code (developer)
- `src/drone_fly/record/coordinates.py` — add `_positions_sidecar_path`, `_load_positions_sidecar`, `_write_positions_sidecar`, `resolve_positions`; add `SPECTRAL_MAX_NEURONS` + `DRONE_FLY_SPECTRAL_MAX` + refuse guard; make `_sidecar_path` accept an explicit npz base; persist neuPrint anatomy to `<stem>_soma.csv`. Keep `provision_positions` output-identical (compute core).
- `src/drone_fly/record/recorder.py` (~:133) — call `resolve_positions(..., persist=True, persist_strict=False)` instead of `provision_positions`.
- `src/drone_fly/cli/__init__.py` — `_run_prune_export` (after :481) and `_run_fetch_connectome` (after fetch) provision + persist sidecars via `resolve_positions(..., artifact_npz=npz)` (fetch subject to the (d) guard).
- `src/drone_fly/prune_trained/workflow.py` (after :428) — same provisioning call for the subcircuit artifact.
- Docs: `README.md` + relevant docstrings — sidecar path/format, node-set binding, self-heal, `SPECTRAL_MAX_NEURONS`/`DRONE_FLY_SPECTRAL_MAX` (default + env override), and the large-connectome defer behavior (AC-9).

### Test code (qa)
- `tests/test_record_coordinates.py` — fast-path HIT = zero `_spectral_embedding`/`_load_from_neuprint` calls (spy/mock, realistic int64 bodyids) (AC-4); AC-8 loaded==fresh **exact** equality across all three branches; corrupt / wrong-length / id-set-mismatch / projection-mismatch → miss+recompute (AC-7 + stale pitfall); partial-anatomy `has_position` split persisted **and fallback rows reload as `None`** (AC-7 / partial-anatomy pitfall).
- `tests/test_record_recorder.py` (or a new `tests/test_record_selfheal.py`) — self-heal: no sidecar → compute+warn+write, second run fast-loads (AC-5); best-effort write on a read-only dir warns and does not raise.
- `tests/test_prune.py` and/or `tests/test_cli.py` — prune export writes `_positions.csv` covering the pruned id-set (+ `_soma.csv` when anatomy) (AC-1).
- `tests/test_fetch.py` — fetch-connectome provisions sidecars for a small mocked connectome; large-connectome no-anatomy → warn + no `_positions.csv` (AC-2, d).
- `tests/test_prune_trained.py` — subcircuit gets its own node-set-keyed sidecars (AC-3).
- Scope-containment: `record: false` provisions nothing; obs schema/width, reward, checkpoints untouched (AC-9). Full `uv run ruff check .` + `uv run ruff format --check .` + `uv run pytest` green and offline (AC-9).

---

## AC → where-satisfied
- **AC-1** (slice provisions, primary) → `_run_prune_export` `resolve_positions(pruned, artifact_npz=npz_path)`; `tests/test_prune.py` / `tests/test_cli.py`.
- **AC-2** (fetch-connectome provisions) → `_run_fetch_connectome` provision, anatomy-conditional per (d); `tests/test_fetch.py`.
- **AC-3** (prune-trained provisions) → `prune_trained` workflow post-:428 provision; `tests/test_prune_trained.py`.
- **AC-4** (training only loads, fast path) → `resolve_positions` fast path returns with zero `_spectral_embedding`/`_load_from_neuprint`; `tests/test_record_coordinates.py` spy.
- **AC-5** (self-heal) → recorder `resolve_positions(persist=True, persist_strict=False)` compute+warn+persist; self-heal test (second run fast-loads).
- **AC-6** (real anatomy fetched once + persisted, preferred) → step 3 of `resolve_positions` writes `<stem>_soma.csv`, reused via tier-2; test.
- **AC-7** (node-set binding, no stale reuse) → `_load_positions_sidecar` exact int-coerced id-set + projection validation → MISS on mismatch; `tests/test_record_coordinates.py`.
- **AC-8** (positions unchanged vs today) → sidecar is a verbatim serialization of `provision_positions`; direct null-ness; `%.17g`; loaded==fresh exact-equality test.
- **AC-9** (scope containment + CI) → no train/obs/reward/checkpoint change; `record:false` unaffected; docs updated; hermetic `ruff`+`pytest` green + offline.

---

## Risks & Considerations
- **Float round-trip (AC-8):** mitigated by `%.17g`; the one correctness-sensitive detail. AC-8 test asserts **exact** equality (not `allclose`) and confirms `pd.read_csv` re-parses to bit-identical float64.
- **Empty-cell reconstruction (challenger non-blocking note):** `pd.read_csv` reads blank `x,y,z` back as **`NaN`** in a float column, so the "cells empty" check in `_load_positions_sidecar` must use `pd.isna(...)` (not `== ""`). Because `provision_positions` only ever stores finite coords (its parsers guard finiteness), `NaN` unambiguously means `coords3d[i] is None` — no ambiguity. The AC-7 partial-anatomy test must explicitly assert the fallback rows **reload as `None`**. Also coerce `has_position` back to Python `bool` (not `numpy.bool_`) so the AC-8 exact comparison matches on type.
- **Fixture pollution:** self-heal writes beside the connectome; tests must operate on **tmp-copied** connectomes (`tmp_path`) so committed `tests/fixtures/` is never dirtied by a run.
- **Write-side source mismatch:** handled by threading `artifact_npz` explicitly at all three hooks — never rely on the pruned `data.source` at write time.
- **Full-connectome scale:** the refuse cap prevents an accidental 161k `eigh`; documented WARN/defer path.
- **Concurrency:** worktree-isolated run; provisioning is main-process only and sidecar writes are the last, small, single-writer step — no contention with UC-26 parallelism.

---

# AMENDMENT (challenger-approved 2026-09-19) — real tokenless anatomy from the connectome meta CSV

Folds **real, tokenless anatomy (connectome meta CSV `somaLocation`)** into the plan above as the PRIMARY anatomy tier, wired into all three write hooks. Everything above stands (orchestrator `resolve_positions`, positions/soma sidecars, node-set HIT rule, self-heal, `%.17g`, refuse cap). Only the anatomy SOURCE + precedence change. Adds AC-10 (real anatomy from meta CSV, primary, tokenless) and AC-11 (full connectome gets real anatomy without the giant eigendecomposition).

## Decisive verified facts
- `save_connectome` (loader.py) writes only `idx,bodyid,superclass,class,subclass,cell_type,top_nt,sign` — **NO `somaLocation`/xyz**; `ConnectomeData` has no soma field ⇒ a pruned artifact's own meta can never carry anatomy → prune/prune-trained must read the **SOURCE** anatomy and subset by pruned bodyids.
- fetch-connectome downloads `mcns_all_neuron_meta.csv` → saved as `<npz-stem>_meta.csv`, which **does** carry `somaLocation` → fetch reads its own sibling meta.
- `_mapping_from_frame`/`_parse_soma_location` already parse `somaLocation` OR `x,y,z` keyed on bodyid — no new parser.
- Committed `mcns_fixture_meta.csv` has **no** `somaLocation` column → AC-10 test needs a NEW tiny DISTINCT meta fixture WITH a `somaLocation` column.

## Anatomy precedence (documented + enforced at compute time)
**meta `somaLocation` (tier-0) → `DRONE_FLY_SOMA_CSV` → `<stem>_soma.csv` → neuPrint → spectral.**

## Design (for the developer)
1. **`_resolve_meta_soma(data, source_data) -> dict[int, xyz]`** (tier-0 mapping, may be partial/empty):
   - `base_npz = _base_npz_path(source_data)` if `source_data` given (prune/prune-trained), else `_base_npz_path(data)` (fetch/read). `_base_npz_path` strips `" [pruned:…]"`/`" [activation-pruned …]"` tags → recovers the real on-disk source npz.
   - read `<base-stem>_meta.csv` with `usecols=["bodyid","somaLocation"]` (catch `ValueError` when the column is absent → fall through) → `_mapping_from_frame`. Meta-`somaLocation` is tier-0 on BOTH read and write sides.
   - **soma-sidecar fallback scoped to the WRITE side only** (`source_data` given): fall to `<source-stem>_soma.csv` via `_load_soma_csv`. On the read side (`source_data=None`) do NOT fall back to the artifact's own `<stem>_soma.csv` here — let it resolve at its normal tier-2 in `_load_anatomical` (documented order holds).
   - subset to `data.neuron_ids` via `_canonical_id`/`_is_int_like` (same canonicalization as the node-set HIT rule). Return int-keyed subset.
2. Thread the mapping into the compute as contained optional kwargs (default `None` → output-identical, AC-8-safe):
   - `_load_anatomical(data, *, override=None)` — non-empty override is tier-0, ahead of env-CSV/sidecar/neuPrint. Partial override → remaining soma-less neurons flow to the existing partial-anatomy fill.
   - `provision_positions(data, *, projection, anatomy_override=None)` — passes override to `_load_anatomical`. Compute core otherwise untouched.
   - `_anatomy_coverage(data, *, override=None)` — counts coverage from `_load_anatomical(data, override=override)`.
3. **`resolve_positions`** gains `source_data` param; new step order:
   1) fast path; 2a) `meta_map = _resolve_meta_soma(compute_data, source_data)`; 2b) if `meta_map` non-empty, (over)write `<stem>_soma.csv` from it (authoritative real anatomy; prevents a stale sidecar on later self-heal); 2c) `_maybe_persist_neuprint_soma` (only when `meta_map` empty + token); 3) refuse guard `if n>_spectral_cap() and _anatomy_coverage(compute_data, override=meta_map)<n: WARN+defer` (WARN reworded: real soma sidecar WAS written, full-graph layout deferred — set token / prune first); 4) `provision_positions(compute_data, projection, anatomy_override=meta_map)`; 5) persist positions.

## Hook wiring
- `_run_prune_export` → `resolve_positions(pruned, artifact_npz=npz_path, source_data=data, persist=True)` (`data` = SOURCE, pre-prune).
- `prune_trained/workflow.py` (~:428) → `resolve_positions(pruned, artifact_npz=npz_path, source_data=base, persist=True)`.
- `_run_fetch_connectome` — unchanged call; tier-0 auto-derives from the artifact's own sibling meta (has somaLocation).

## AC-10 / AC-11 / AC-8
- **AC-10:** central neurons get real somas from the meta; soma-less afferents flagged missing → partial-anatomy fill (never faked). Real `<stem>_soma.csv` written; zero neuPrint + zero spectral for soma-bearing.
- **AC-11:** at fetch, step 2b writes real somas as a pure CSV subset (no eigh); refuse guard defers the full-graph positions layout; residual spectral guarded. Recording the full graph stays unsupported (prune first).
- **AC-8:** numerical-identity holds (kwargs default None → identical); the change to which neurons get real vs fallback coords is the intended, documented consequence.

## Test additions (qa)
- NEW tiny meta fixture WITH `somaLocation`, DISTINCT from `mcns_fixture_meta.csv`.
- `test_prune.py`/`test_cli.py`: prune → pruned `<stem>_soma.csv` has real subset coords for soma-bearing slice bodyids; zero neuPrint; zero spectral for soma-bearing (spy) [AC-1/AC-10].
- `test_fetch.py`: small fetched connectome w/ somaLocation meta → real soma+positions sidecars, tokenless, no neuPrint/spectral for soma-bearing (AC-2/AC-10); large no-full-anatomy → real soma sidecar written + positions deferred + WARN (AC-11/d).
- Also fix the two existing tests the baseline fetch-hook broke: `test_cli.py::test_fetch_connectome_downloads` and `::test_fetch_connectome_forwards_flags` (mock `load_connectome` + `resolve_positions`, or use a real fixture dir).
- All other approved tests (AC-3..AC-9) stand. Docs: README/docstrings state the 5-level precedence, tokenless-by-default, soma-less afferents, full-connectome defer.
