---
plan_for: use-cases/04-connectome-subgraph-pruning.md
work_branch: feat/uc-04-connectome-subgraph-pruning
team: drone-fly-uc-4
approved: 2026-09-17
---

# UC-04 Approved Implementation Plan — Function-targeted connectome subgraph pruning

Analyst↔challenger agreement (v2, approved; challenger independently reproduced all numbers).
TARGET_DIR = /workspace/drone-fly-uc-04-connectome-subgraph-pruning.

## Analysis
Running the full MaleCNS (161,429 neurons / 25,083,972 edges) as the live policy net makes PPO
intractable — the actor propagates the whole graph every step though only the directed
`visual_projection`→`descending_neuron` subcircuit drives the 4 control outputs. UC-04 adds a
deterministic, direction-aware pruning step reducing a loaded `ConnectomeData` to that subcircuit,
behind an opt-in flag so UC-01/02/03 are byte-identical when off.

Key facts confirmed in code: `adjacency[i,j]` = weight **j→i**; per-edge E/I sign comes from the
**presynaptic (column) neuron** (`policy.py`: `edge_sign = sign_arr[coo.col]`), so re-slicing
per-neuron `sign[kept]` preserves it; the connectome is serialized into the SB3 checkpoint via
`features_extractor_kwargs`, so evaluate/resume reconstruct the same graph (no eval flag needed).
The committed fixture (300n/10600e) is a dense, fully-strongly-connected hub-slice, so a pure
reachability corridor = whole graph; the path-slack rule below reduces it.

## Resolved design question — pruning rule
**Path-slack corridor**, one configurable rule with parameter `k` (default `DEFAULT_PRUNE_K = 2`),
unifying both use-case candidates: `k=0` = shortest-path corridor (tight), larger `k` = richer
k-hop neighbourhood (recommended default). Direction-correct, O(V+E), monotone superset in `k`.
Default k=2 favours richness (descending neurons integrate broadly).

**Retained-set construction (Option A — path inclusion, guarantees AC2 by construction):**
1. Forward BFS from full sensory set over successors (CSC columns), tracking **parent pointers** →
   `d_fwd`, `parent`.
2. Backward BFS from full motor set over predecessors (CSR rows) → `d_bwd`; `L = min d_bwd(sensory)`.
3. Retained = all sensory endpoints ∪ (for each sensory-reachable motor `m`: nodes of one
   reconstructed shortest sensory→m path) ∪ corridor `{u : d_fwd(u)+d_bwd(u) ≤ L+k}`.
4. Motor not sensory-reachable → dropped with warning; empty/degenerate result → error (AC9).

Every retained motor is reachable **within the pruned subgraph** for any k, since each
`parent[v]→v` edge survives `A[kept][:,kept]`.

**Verified numbers (fixture, path-inclusion):** k=0 → 59/300 neurons, 751/10600 edges;
k=1 → 175/5988; k=2 → 274/10070. 18/18 motor present, 0 unreachable in the pruned graph at every k.

## Proposed Solution
- **NEW `src/drone_fly/connectome/prune.py`**: constants `PRUNE_RULE_PATH_SLACK="path_slack"`,
  `DEFAULT_PRUNE_K=2`, `DEFAULT_PRUNE_RULE`; `prune_to_subcircuit(data, *, k=DEFAULT_PRUNE_K,
  rule=DEFAULT_PRUNE_RULE) -> ConnectomeData` (AC1 signature). Builds retained set as above;
  submatrix `adjacency[kept][:,kept]`; re-slices `neuron_ids`/`superclass`/`sign`/`top_nt` by
  `kept`; new `ConnectomeData` (input unmutated); `source` gets provenance suffix; logs input→pruned
  counts (AC7). Errors: `superclass is None`, missing population, `k<0`/unknown rule, degenerate result.
- **MODIFY `src/drone_fly/connectome/__init__.py`**: export the new symbols.
- **MODIFY `src/drone_fly/train/loop.py`**: add `prune: bool=False`, `prune_k: int=DEFAULT_PRUNE_K`
  to `train()` (thread through `smoke_train`); apply prune between load and `build_policy_kwargs`,
  logging reduction. **Skip prune when `resume` is set, with a warning** (checkpoint carries its own
  graph). Default off ⇒ UC-01/02/03 unchanged (AC8).
- **MODIFY `src/drone_fly/cli/__init__.py`**: add `--prune` (store_true) + `--prune-k INT` to
  `train`/`smoke-train`; pass through. No flag on `evaluate` (by design).
- **MODIFY `README.md`**: document flags + rule + fixture reduction table; state the full-MaleCNS
  figure is a **projection, not a measurement**, and give the exact
  `drone-fly train --connectome <full> --prune` command to measure it. (README IS in the developer's
  write scope per the target CLAUDE.md § "Dev-team write authorizations" — no exemption needed.)

## Files Affected
**Production (developer)** — `paths.production = src/drone_fly/**` + README (authorized):
- `src/drone_fly/connectome/prune.py` (new)
- `src/drone_fly/connectome/__init__.py` (modify — exports)
- `src/drone_fly/train/loop.py` (modify — opt-in prune + resume guard)
- `src/drone_fly/cli/__init__.py` (modify — `--prune`/`--prune-k`)
- `README.md` (modify — docs; authorized via CLAUDE.md § Dev-team write authorizations)

**Test (qa)** — `paths.test = tests/**`:
- `tests/test_prune.py` (new): no-mutation; direction-correctness; meta re-alignment
  (`superclass`/`sign`/`top_nt`/`neuron_ids` length M, correct mapping); per-edge sign preserved;
  determinism (identical `indices`/`indptr`/`data`+`neuron_ids` across runs); monotone superset
  kept(k)⊆kept(k+1); **reachability in the pruned subgraph** (0 unreachable motor); population-compat
  (`select_populations` still finds descending/visual on pruned graph); reduction reporting;
  **REQUIRED synthetic long-chain case** (motor at d_fwd=5, L=1, k=2 → far motor reachable; isolated
  / pure-upstream / reverse-only nodes pruned); degrade/error cases (superclass None, missing
  population, degenerate); pipeline smoke (`train(..., prune=True)` on fixture).
- `tests/test_cli.py` (modify): `--prune`/`--prune-k` parse + thread-through.

## Risks & Considerations
1. README is authorized for the developer (CLAUDE.md § Dev-team write authorizations) — resolved.
2. Full-matrix numbers are projections (multi-GB dataset not present here); only fixture numbers are
   measured — README labels this explicitly.
3. On the dense fixture, k=2 barely reduces (274/300); the AC10 "smaller" test pins an explicit small
   k (k=0 → 59/300) for an unambiguous shrink.
4. `--prune` is a no-op on `--resume` (guarded + warned).

## Scope addition (post-approval, user-directed) — export the pruned slice + README quick guide

Added by the user after approval (small, additive; does not change the pruning algorithm):

1. **`drone-fly prune` export subcommand.** New CLI subcommand
   `drone-fly prune --connectome <input> --out <dir> [--prune-k K] [--prune-rule R]` that loads a
   connectome, prunes it via `prune_to_subcircuit`, and **writes the pruned connectome to `<dir>`**
   as a `.npz` matrix + `<stem>_meta.csv` in the exact on-disk format `load_connectome` reads (mirror
   the writer in `scripts/build_test_fixture.py`; preserve `idx/bodyid/superclass/sign/top_nt`
   columns). Also write a small provenance note (source, rule, k, input→pruned counts). Purpose:
   **prune once, reuse** via `train --connectome <dir>` (no re-pruning), and give UC-05 a stable saved
   slice to visualize. Must **round-trip**: `load_connectome(<dir>)` reproduces the pruned
   `ConnectomeData` (same counts, meta aligned). Deterministic; clear errors. This is AC11.
2. **README quick guide.** Add a concise **numbered, step-by-step "Quick start: train a fly to fly"**
   section at the TOP of the README (right after the intro) — terse numbered bullets only (clone/
   install → provision connectome → optionally prune → train → resume → evaluate), with the detailed
   paragraphs kept below it. Also document the new `prune` export command in the pruning section.

QA additions: round-trip test for the `prune` export (prune → write tmp dir → `load_connectome` →
assert equivalence of counts + meta) and a `test_cli.py` case for the `prune` subcommand parse/dispatch.

## Challenger final verdict
APPROVE (v2). Fixed the v1 Major AC2 reachability gap via Option A shortest-path inclusion (motor
endpoints stay connected in the pruned subgraph for any k — a pure corridor would have stranded them
on the full 161k matrix, invisible to the L=1 fixture). Independently reproduced k=0/1/2 numbers.
Minors resolved (--prune+--resume skip-with-warning; README labels full-matrix reduction as
projection). Direction-correctness, meta/sign integrity, efficiency (O(V+E) sparse BFS), and
back-compat all verified.
