---
plan_for: use-cases/01-connectome-plumbing-poc.md
work_branch: feat/uc-01-connectome-plumbing-poc
team: drone-fly-uc-1
approved: 2026-09-16
---

# UC-01 Approved Implementation Plan — Connectome plumbing POC

Analyst↔challenger reached agreement (challenger APPROVED, round 2). Grounded on the real
AxonWeave v0.2.0 and connectome_data_prep repos (source-traced, not README-guessed).

## Task
UC-01 (`use-cases/01-connectome-plumbing-poc.md`): a non-flight POC proving MaleCNS is a
runnable, trainable substrate before any flight/RL work. Load an OFFLINE cached connectivity
matrix (connectome_data_prep format), instantiate it via AxonWeave as a sparse trainable
PyTorch policy, prove a full encode→network→decode roundtrip returning a deterministic finite
`(4,)` action (THROTTLE/ROLL/PITCH/YAW), covered by a fast CI pytest smoke test. No physics,
no RL, no training.

## Grounded dependency facts (source-verified)
- **AxonWeave v0.2.0 is on PyPI** (small wheel; the multi-GB MaleCNS substrate is provisioned
  separately via `axonweave substrate install male-cns:v1.0` and is NOT needed for CI). The
  brief's "GitHub-only" note is stale → recommend a later `revise-brief` (non-blocking).
- **AxonWeave can build a real layer from an arbitrary in-memory sparse matrix, offline:**
  `axonweave.core.graph.ConnectomeGraph(weights: scipy.sparse.csr_matrix, body_ids: np.ndarray)`
  → `.n_neurons`/`.n_edges`/`.save()`/`.load()`; `axonweave.core.brain.BiologicalBrain(graph)`
  → `.torch_layer(trainable_edges=True)`. This is the same facade `axonweave.load()` returns,
  without any named multi-GB substrate or network.
- **connectome_data_prep** ships real, committed, in-repo scipy-sparse `.npz` + meta-`.csv`
  matrices (exact target format) at
  `https://raw.githubusercontent.com/YijieYin/connectome_data_prep/main/data/<file>` — e.g.
  `adult_type_inprop.npz` (14.9MB)+`adult_type_meta.csv`,
  `adult_inprop_cb_neuron_no_CX_axonic_postsynapses.npz` (33.7MB)+its meta. Strict-MaleCNS
  build path is the committed notebook `maleCNS/maleCNS_prepare_all_neuron.ipynb` (+`mcns_ad_split`).

## Core design — two-tier data path (real matrix primary, offline/hermetic CI)
1. **Real full path (documented, network-gated, NOT in CI):** provision the real MaleCNS matrix
   into gitignored `data/connectome/` (via `axonweave substrate install` or downloading the
   connectome_data_prep matrix). Loader reads it when present.
2. **Offline CI path (committed, real-format, small):** a small real fly-connectome subgraph
   committed under `tests/fixtures/` (NOT gitignored — confirmed `*.zip`≠`.npz`). Real data,
   deterministically sliced — never `numpy.random`.

## Fixture-generator provenance strategy (ordered; developer follows in order)
1. **Preferred:** a genuinely MaleCNS-derived matrix — a `data/*.npz` confirmed MaleCNS, or run
   `maleCNS_prepare_all_neuron.ipynb` at dev-time (networked dev box, not CI).
2. **Acceptable fallback:** a real committed connectome_data_prep matrix even if central-brain/
   FAFB-adult, **documented honestly** as the fixture's provenance and flagged as a known
   strict-MaleCNS deviation for UC-02 to tighten (still exercises real connectome topology,
   which is what the plumbing POC de-risks).
3. **HARD RULE:** if no real fly-connectome matrix is obtainable in the dev environment, the
   developer STOPS and escalates to the team lead — synthetic/`numpy.random` fabrication is
   forbidden (pitfall #1).
Deterministic, sparsity-preserving slice (few hundred neurons); provenance (source file, URL,
slice rule, resulting counts) recorded in the generator script + a fixture README so the
reduction is logged, not silent (pitfall #2).

## Files to create/modify (all inside `/workspace/drone-fly-uc-01-connectome-plumbing-poc`)
**Production code (developer):**
- `src/drone_fly/connectome/loader.py` (new) — `ConnectomeData` (adjacency = scipy CSR float32
  N×N, `neuron_ids`, `neuron_count`=N, `edge_count`/`synapse_count`=nnz, `source` tag).
  `load_connectome(path=None)` resolves artifact dir: explicit arg → env var
  `DRONE_FLY_CONNECTOME_DIR` → default `data/connectome/`; reads `.npz`+meta; performs NO
  network I/O and does NOT import `neuprint`; raises a clear actionable error when the artifact
  is absent (never silently hits the network). Defines TWO documented constants:
  `FIXTURE_EXPECTED_SCALE` (exact fixture neuron & edge counts) and `MALECNS_V1_EXPECTED_SCALE`
  (full-dataset scale).
- `src/drone_fly/connectome/__init__.py` — export `load_connectome`, `ConnectomeData`, both constants.
- `src/drone_fly/controller/policy.py` (new) — `ConnectomePolicy(nn.Module)` from `ConnectomeData`.
  **Primary path:** build `ConnectomeGraph`→`BiologicalBrain`→`.torch_layer(trainable_edges=True)`
  (real AxonWeave, offline). **Guarded fallback:** in-repo `SparseConnectomeLayer` shim (real
  fixture edges as a sparse trainable parameter) used ONLY if the step-0 spike finds AxonWeave's
  `core.*` internals unavailable/renamed on the pinned version — no silent substitution. Exposes
  `n_neurons` and `n_connections`. `forward(state)->state`.
- `src/drone_fly/controller/encoding.py` (new) — locked I/O contract (AC4/AC6): `OBS_DIM`,
  `ACTION_DIM=4`, `ACTION_LAYOUT=(THROTTLE,ROLL,PITCH,YAW)`, documented ranges (THROTTLE∈[0,1]
  via sigmoid; ROLL/PITCH/YAW∈[-1,1] via tanh), fixed reproducible `SENSORY_NEURON_INDICES`/
  `MOTOR_NEURON_INDICES` (documented placeholder per pitfall #3). `encode_observation(obs)->
  neuron_input[N]` (scatter into sensory indices), `decode_action(state)->action[4]` (read 4
  motor indices → documented range). Rich docstring so UC-02 depends on it without re-deriving.
- `src/drone_fly/controller/roundtrip.py` (new, or fold into policy) — `run_roundtrip(obs, seed)
  ->action(4,)`: sets `torch.manual_seed(seed)` BEFORE policy instantiation (so randomly-
  initialized trainable edges are reproducible), then encode→propagate→decode; returns a finite
  `(4,)` tensor.
- `src/drone_fly/controller/__init__.py` — export policy, encode/decode, contract constants, roundtrip.
- `pyproject.toml` (modify) — add `scipy` to core deps; add `axonweave` (torch support) to
  INSTALLED deps (core deps or `[dev]` so `-e ".[dev]"` pulls it — NOT an uninstalled optional
  extra) so CI exercises real AxonWeave; **pin the exact axonweave version the step-0 spike validated.**
- Committed fixture-generator script (developer's choice of location, e.g.
  `scripts/build_test_fixture.py`) — encodes the ordered provenance strategy + hard escalation
  rule; network/full-data required so NOT run in CI.

**Test code (qa):**
- `tests/fixtures/<name>.npz` + `tests/fixtures/<name>_meta.csv` (new) — the small real-subset
  committed fixture, with CC-BY attribution.
- `tests/conftest.py` (new) — fixtures: fixture path; an offline-enforcement fixture that
  monkeypatches `socket.socket` to raise during load/roundtrip tests (proves AC2 "networking
  unavailable").
- `tests/test_connectome_loader.py` (new) — AC1/AC2: load fixture; assert counts **exactly
  equal** `FIXTURE_EXPECTED_SCALE`; assert `MALECNS_V1_EXPECTED_SCALE` defined/positive; assert
  no network/neuprint import under socket-disabled fixture.
- `tests/test_policy.py` (new) — AC3: instantiate policy; assert `n_connections == edge_count`
  AND sparsity ratio `nnz/N² < ` a documented threshold (not just `< N²`).
- `tests/test_roundtrip.py` (new) — AC4/AC5: encode→forward→decode; assert `(4,)`, all finite,
  within documented ranges; byte-identical across two fixed-seed runs.
- `tests/test_uc01_smoke.py` (new) — AC7: fast end-to-end load+roundtrip, well under ~60s on the
  small fixture. Keep existing `tests/test_smoke.py` (import checks) unchanged.

## Developer step 0 (carry-forward from challenger, do first)
Short spike against the installed AxonWeave: confirm `ConnectomeGraph`/`BiologicalBrain`/
`torch_layer` build a layer from an arbitrary sparse matrix. **Pin the exact validated axonweave
version in `pyproject.toml`.** If those `core.*` internals are unavailable/renamed on the pinned
version → take the documented shim-fallback path; do NOT improvise around a private API.

## Files Affected (roll-up)
- **Production (developer):** `src/drone_fly/connectome/{loader.py,__init__.py}`,
  `src/drone_fly/controller/{policy.py,encoding.py,roundtrip.py,__init__.py}`, `pyproject.toml`,
  fixture-generator script.
- **Test (qa):** `tests/fixtures/<name>.npz`, `tests/fixtures/<name>_meta.csv`,
  `tests/conftest.py`, `tests/test_connectome_loader.py`, `tests/test_policy.py`,
  `tests/test_roundtrip.py`, `tests/test_uc01_smoke.py`.

## Risks & Considerations
1. **AC1 full-scale vs offline/<60s:** reconciled — exact-match against `FIXTURE_EXPECTED_SCALE`
   always-on (offline); full-scale assertion against `MALECNS_V1_EXPECTED_SCALE` in an
   auto-skipped test that runs only when the multi-GB artifact is present.
2. **AxonWeave `core.*` API stability:** internals, so version-pin from the spike is mandatory;
   guarded shim fallback if unavailable.
3. **Fixture provenance/licensing:** MaleCNS is CC-BY (redistribution with attribution OK); keep
   the committed fixture small (not "bulk"), attributed. Tier-2 honest-deviation documented if
   strict-MaleCNS isn't obtainable.
4. **Sensory/motor mapping is an arbitrary documented placeholder** (pitfall #3) — UC-01 locks
   the contract shape, not biological correctness; UC-02 refines.
5. **Determinism:** seed before instantiation; CPU torch; assert finite range + byte-identical.
6. **CI weight:** scipy + axonweave(small wheel) + torch(already declared) install adds time, but
   that's install, separate from the <60s test-execution budget. Ruff `select=["E","F","I","UP","B"]`
   + `ruff format --check` gate style — code must be formatted.
7. **Stale brief note:** AxonWeave is on PyPI, not GitHub-only → later `revise-brief` (non-blocking).

## Challenger final verdict
APPROVE after one revision round. Offline/CI-friendliness, real-AxonWeave exercise, testable
sparsity/roundtrip, and no-scope-creep all satisfied. Implementation watch item: the developer
must pin the validated axonweave version and not improvise around the private `core.*` API —
use the documented shim fallback if the internals are unavailable.
