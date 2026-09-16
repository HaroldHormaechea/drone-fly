# Use Case 01: Connectome plumbing POC

## Summary
A minimal, non-flight proof-of-concept that confirms the MaleCNS connectome is usable as a
runnable, trainable substrate in code before any flight or RL work depends on it. The POC
loads MaleCNS connectivity from an **offline, cached connectivity matrix** (produced with
`connectome_data_prep`, committed/cached as an artifact so no live neuPrint token or network
access is required), instantiates it through AxonWeave as a sparse trainable PyTorch policy
network, and proves a **full input→output roundtrip**: a stub sensory-encoder maps a dummy
observation vector into the network's sensory neurons, activity propagates through the
connectome-structured network, and a stub motor-decoder reads designated motor neurons out to
a 4-channel action vector (throttle/roll/pitch/yaw). No physics sim, no RL, no training loop —
this de-risks the "connectome → runnable controller" plumbing and fixes the exact I/O
interface shape that UC-02 will consume. Lives under `src/drone_fly/connectome/` (data load)
and `src/drone_fly/controller/` (policy + encode/decode), exercised by a fast `pytest` smoke
test that runs in CI.

## Acceptance Criteria
1. A loader function reads the cached MaleCNS connectivity matrix and returns a structure
   exposing neuron count and synapse/edge count; a test asserts both are > 0 and match the
   expected scale of the cached MaleCNS dataset (documented constant).
2. The load path requires **no live neuPrint token and no network access** — it reads only
   the committed/cached artifact, and the smoke test passes with networking unavailable.
3. A `Policy` (`torch.nn.Module`) is instantiated from the loaded connectivity; its parameter
   count / connection sparsity reflects the connectome topology (sparse, connectome-derived)
   rather than a dense MLP, and a test asserts the sparsity/shape is connectome-consistent.
4. A stub **sensory-encoder** maps a dummy observation tensor into the network's sensory-neuron
   inputs, and a stub **motor-decoder** reads designated motor neurons out to a finite `(4,)`
   action vector in a documented range — the canonical throttle/roll/pitch/yaw layout UC-02 uses.
5. A single forward pass over the full encode → network → decode roundtrip returns a finite
   `(4,)` action for a dummy observation, asserted deterministic for a fixed seed.
6. The encode/decode interface (observation shape in, `(4,)` action out, neuron-index mapping
   contract) is documented in code so UC-02 can depend on it without re-deriving it.
7. A `pytest` smoke test covers the load + full roundtrip, runs in under ~60s in CI, and passes.

## Potential Pitfalls & Open Questions
- **Assumption** — AxonWeave can ingest a `connectome_data_prep`-format connectivity matrix
  directly (or with a thin adapter). If the formats don't line up, a small conversion shim is
  in scope for this POC; falling back to a synthetic sparse graph is explicitly *not* chosen
  (the user wants the real cached matrix).
- **Assumption** — The full 166k-neuron MaleCNS graph is tractable for a single CPU forward
  pass in CI within ~60s. If not, the POC may load the full graph but exercise the roundtrip on
  a documented subgraph/subset while still asserting the full graph loaded — the developer
  decides, but any reduction must be logged, not silent.
- **Ambiguity** — Which neurons count as "sensory inputs" and "motor outputs" is a modeling
  choice. For this POC the mapping may be a documented placeholder (e.g. a fixed, arbitrary but
  reproducible neuron-index selection); UC-02 / later work refines it to biologically motivated
  populations. The contract (not the biological correctness) is what UC-01 locks.
- **Edge case** — Cached-artifact size vs. git: a large matrix may need Git LFS or a
  download-on-first-run cache rather than a committed blob. The `.gitignore` already excludes
  `data/**` and `artifacts/**`, so the developer must decide cache-fetch vs. commit and document it.

## Original Description
Small POC that confirms the connectome plumbing works — load MaleCNS via AxonWeave and
instantiate a policy network — so the connectome plumbing is proven before adding flight.

## Clarifications
- Q: How should the POC obtain the MaleCNS connectivity data?
  A: Offline cached matrix (via `connectome_data_prep`), so CI can run it without a neuPrint
     token or network access.
- Q: What is the success bar for the plumbing POC?
  A: Full I/O roundtrip — stub sensory-encode → connectome network → motor-decode to a
     4-channel action, not just a bare forward pass.
