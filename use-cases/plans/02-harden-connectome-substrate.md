---
plan_for: use-cases/02-harden-connectome-substrate.md
work_branch: feat/uc-02-harden-connectome-substrate
team: drone-fly-uc-2
approved: 2026-09-16
---

# UC-02 Approved Implementation Plan — Harden the connectome substrate for RL

Analyst↔challenger agreement (challenger APPROVED after one revision round). TARGET_DIR =
/workspace/drone-fly-uc-02-harden-connectome-substrate (worktree on feat/uc-02-harden-connectome-substrate).
profiles empty; maturity prototype; coverage target 50%; build via `uv run pytest` / `uv run ruff check .`.
Deps torch/gymnasium/stable-baselines3/numpy/scipy/pandas all declared & installed.

## KEY FINDING (resolves the use case's "biggest open question")
The committed 300-neuron fixture `tests/fixtures/mcns_fixture_meta.csv` ALREADY carries real
per-neuron metadata — `superclass`, `top_nt`, `sign` — with ZERO nulls (verified via `uv run`).
So a biologically-principled motor population AND a real E/I sign mask are obtainable FULLY OFFLINE
in CI — no neuprint/connectome_data_prep network fetch. Strictly better than the framed (a) network
/ (b) placeholder either/or. Confirmed:
- `sign`: 158 inhibitory (-1) / 142 excitatory (+1), reconciling exactly with `top_nt`
  (gaba+glutamate=158; ach+octopamine+dopamine+serotonin+unclear=142).
- Stored `.npz` weights are ALL POSITIVE (min 8.3e-6, max 0.122) — they carry NO sign. The meta
  `sign` column is the sole, additive source of E/I biology; 45% of edges have inhibitory presyn →
  the mask is non-trivial.
- `superclass` includes descending_neuron(18) — the fly's sole brain→VNC motor-command pathway
  (true leg/wing motor neurons live in the VNC, outside this brain volume, so descending neurons
  are the correct premotor readout) — and visual_projection(35) as sensory afferents.
- Reachability confirmed: all 18 descending (motor) neurons are reachable from the 35
  visual_projection (sensory) neurons within 2 hops → with default n_steps≥2, readout gradients
  reach the sparse weights non-trivially (AC5/AC6 robust on this fixture).
- Adjacency convention `A[i,j] = weight from j→i` (loader docstring) → per-edge Dale's-law sign =
  `sign[presynaptic col]`.

DECISION on the open question: Option (a) principled selection, ZERO network dependency. Placeholder
path kept only as a documented, tested degrade fallback (never silently shipped).

## SOLUTION — Production files (developer), all under src/drone_fly/**
1. connectome/loader.py — Extend `ConnectomeData` with three OPTIONAL aligned arrays (default None →
   back-compat, no existing test breaks): `superclass`, `sign` (int8 ±1), `top_nt`. Populate in
   `load_connectome` when meta columns present, aligned to matrix row order exactly like neuron_ids.
   Full multi-GB path unaffected.
2. controller/populations.py (NEW) — Documented, configurable size constants MOTOR_POP_SIZE /
   SENSORY_POP_SIZE. `select_motor_population(data)` → `superclass=="descending_neuron"`
   (deterministic: by descending total degree, tie-break ascending idx; capped).
   `select_sensory_population(data)` → `superclass=="visual_projection"`. Each returns
   (indices, mode∈{"biological","placeholder"}). Degrade path: if `superclass` is None or yields <1
   usable neuron, fall back to UC-01 encoding placeholder index fns, mode="placeholder", documented
   limitation. Sensory/motor sets guaranteed disjoint.
3. controller/policy.py — Harden `SparseConnectomeLayer` IN PLACE (AC7 replaces one-hop):
   (a) per-edge `sign_mask` registered buffer = `sign[col]`; effective weight = `sign_mask *
   raw_weight.abs()` with `raw_weight` init = `coo.data` → biological magnitudes preserved EXACTLY at
   init (|positive|==identity; softplus rejected because softplus(1e-3)≈0.693 would collapse all
   magnitudes — challenger's Major catch), and E/I sign immutable under training (AC1+AC3);
   (b) `n_steps` recurrent unroll (default >1, configurable): state_{t+1}=tanh(Weff@state_t), bounded
   nonlinearity keeps activity finite (AC2); (c) BATCHED state (B,N) support for SB3;
   (d) `propagation_mode∈{"scatter","sparse"}`, default "scatter" (index_add/gather, autograd-stable)
   as the AC6 fallback, `torch.sparse.mm` as alternate. Edge topology stays a buffer; only weights
   carry grad. `ConnectomePolicy` keeps returning (N,) state + `force_shim`/`backend` UNCHANGED so
   UC-01 roundtrip tests stay green; `_try_build_axonweave_layer` left as a commented, permanently-dead
   no-op — `axonweave` MUST NOT enter pyproject.
4. controller/actor.py (NEW) — `ConnectomeActorNetwork(nn.Module)`: input projection
   Linear(OBS_DIM→|sensory_pop|) scattered into sensory neurons' state → recurrent hardened
   propagation (n_steps) → gather ONLY motor-pop entries → readout Linear(|motor_pop|→4) → UC-01
   channel squashing (sigmoid THROTTLE [0,1], tanh attitudes [-1,1]), preserving ACTION_LAYOUT (AC7).
   forward((B,OBS_DIM))→(B,4); full N-state NEVER returned (AC4). Trainable = sparse edge weights +
   both Linear heads; grads reach the sparse weights (AC5/AC6).
5. controller/sb3.py (NEW) — `ConnectomeFeaturesExtractor(BaseFeaturesExtractor)` wrapping the actor,
   the AC5 SB3 seam for UC-03's PPO wiring.
6. controller/__init__.py — re-export populations fns + constants, actor, sb3 extractor.

## SOLUTION — Test files (qa), all under tests/**
- test_populations.py (AC4): descending_neuron/visual_projection rules, configurable sizes,
  disjointness, determinism, degrade fallback on a synthetic metadata-less ConnectomeData.
- test_sign_mask.py (AC3): mask from `sign`; training step changes magnitudes but never flips a
  per-synapse sign; degrade-to-default when `sign` absent; PLUS init-magnitude guard: effective init
  magnitude ≈ |coo.data| per edge (locks the reparam against silent regression).
- test_recurrent_propagation.py (AC2): impulse/reachability — unit impulse at one sensory neuron; a
  target 2-hops-away-but-not-1-hop is zero after n_steps=1 and non-zero after n_steps=2 (small impulse
  so tanh stays ~linear; proves genuine multi-hop, not a nonlinearity artifact); activity stays finite
  over many steps.
- test_trainable_weights.py (AC1): training step updates weights, edge_index buffer unchanged, only
  weights carry grad.
- test_actor.py (AC4): (B,4) in ranges/layout; only motor-pop feeds head; full state never output;
  input projection trainable.
- test_sb3_compat.py (AC5): instantiate feature extractor; one forward + one backward; grads reach
  sparse weights; CPU-deterministic for fixed seed.
- test_sparse_backward.py (AC6): both propagation modes → finite non-trivial grads on sparse weights,
  incl. BATCHED (N,B) autograd through sparse_coo_tensor.coalesce()→torch.sparse.mm; if sparse mode
  proves unstable, xfail/skip-with-reason while scatter default carries the guarantee (no
  silently-green flake).
- Update test_connectome_loader.py for the new surfaced metadata arrays.
- Reconcile UC-01 test_policy.py + test_roundtrip.py for hardened-layer semantics (seed before
  instantiating new Linear heads; softplus→abs keeps one trainable param/edge so the param-count
  assertion still holds).

## SCOPE GUARD
NO sim, NO PPO training loop, NO flight, NO reward. Deliverable stops at a trainable,
SB3-compatible, unit-tested connectome policy. All 7 acceptance criteria addressed above.

## Challenger final verdict
APPROVE after one revision round. Independently verified fixture meta (real superclass/sign/top_nt,
zero nulls, on the CI load path), adjacency convention (A[i,j]=j→i → sign from presynaptic column),
positive-only .npz weights (sign is genuine additive biology), and sensory→motor 2-hop reachability.
Forced fix: softplus reparam would collapse connectome magnitudes to ≈0.693 at init → replaced with
`sign_mask * raw.abs()`, raw init = coo.data, plus a magnitude guard test. Scope confirmed clean
(no sim/PPO/reward/flight; AxonWeave not reintroduced; UC-01 loader + (4,) contract reused).
