# Use Case 04: Function-targeted connectome subgraph pruning

## Summary
Running the full MaleCNS connectome (~161k neurons / ~25M edges) as the live policy network makes
PPO training intractable on a laptop (~days–weeks per run), because every forward/backward
propagates the whole graph even though only the **sensory→motor subcircuit** influences the 4
control outputs. This use case adds a **function-targeted pruning step** that reduces a loaded
`ConnectomeData` to the neurons and edges that actually lie on the pathway from the **sensory
population** (`visual_projection`) to the **motor population** (`descending_neuron`), discarding the
~160k neurons irrelevant to flight control. The result is a smaller, biologically-principled
`ConnectomeData` (meta and populations preserved and re-aligned) that the existing UC-02 substrate
and UC-03 training consume unchanged — turning "train on the real full connectome" from a multi-week
run into something laptop-tractable, while being *more* principled than the current top-degree
generic slice (not random, not hub-based — the actual control circuit). Pruning is deterministic,
respects edge direction, and is wired into the pipeline behind an opt-in flag so existing behavior
is unchanged. CI/tests run hermetically on the committed fixture; the real payoff (pruning the full
matrix) is a dev-time step on the owner's machine.

## Open design question (for the analyst/challenger to resolve in the plan)
How aggressive should the pruning rule be? Two candidates (recommend implementing a **configurable
rule with a sensible default**, not hardcoding one):
- **Shortest-path corridor** — keep only neurons on (near-)shortest directed sensory→motor paths.
  Smallest and fastest, but may drop useful lateral/recurrent circuitry.
- **k-hop neighborhood (recommended default)** — keep neurons within `k` synapses on a directed
  path between the sensory and motor sets (configurable `k`). Larger, richer, still ≪ 161k.
The plan should pick a default, expose the rule + parameter as documented constants, and report the
actual pruned neuron/edge counts achieved on the full MaleCNS.

## Acceptance Criteria
1. A pure function (e.g. `prune_to_subcircuit(data, ...)` in `src/drone_fly/connectome/`) returns a
   **new** `ConnectomeData` containing only the neurons/edges on the sensory→motor pathway under the
   chosen rule; the input `ConnectomeData` is not mutated.
2. **Population retention + reachability:** the selected sensory (`visual_projection`) and motor
   (`descending_neuron`) neurons are all present in the pruned graph, and every retained motor
   neuron remains reachable from ≥1 sensory neuron along directed edges in the pruned graph
   (unit-tested on the fixture).
3. **Direction-correct:** paths respect the adjacency convention (`A[i,j]` = weight from j→i), so the
   retained subgraph is the *directed* sensory→…→motor circuit, not an undirected blob. Tested.
4. **Meta/index integrity:** `neuron_ids`, `superclass`, `sign`, `top_nt` are re-aligned to the
   pruned matrix rows; per-edge signs and the sign-mask semantics are preserved; UC-02's
   population selection still finds the descending/visual populations in the pruned graph.
5. **Configurable rule:** the pruning rule and its parameter (e.g. `k`) are documented, configurable
   constants; a larger parameter yields a superset (monotonic growth), tested.
6. **Determinism:** same input + parameters → byte-identical pruned graph (neuron set, index order,
   edges), tested with a fixed tie-break.
7. **Reduction reporting:** the step logs/returns the input→pruned counts (neurons and edges). The
   README documents the actual reduction measured on the full MaleCNS (order thousands of neurons,
   not 161k) so the tractability gain is concrete, not asserted.
8. **Pipeline wiring:** an opt-in CLI/config option applies pruning before the policy is built, so
   `drone-fly train --connectome <full-matrix-dir> --prune[...]` trains on the pruned graph;
   omitting it leaves UC-01/02/03 behavior unchanged (back-compatible default).
9. **Degrade / error handling:** if the connectome lacks `superclass` meta (populations can't be
   identified), pruning stops with a clear, actionable message (never silently returns garbage); a
   rule that would yield an empty/degenerate graph errors clearly.
10. **Hermetic tests:** the full test suite runs on the committed fixture with no network; pruning
    the fixture yields a valid, smaller, still-connected subcircuit. UC-01/UC-02/UC-03 suites stay
    green.

## Potential Pitfalls & Open Questions
- **Rule aggressiveness (the design fork above).** Too strict → drops circuitry the policy needs;
  too loose → barely reduces the graph. The default must materially cut ~160k neurons while keeping
  the pruned graph connected sensory→motor. Report real numbers.
- **The pathway is diffuse.** Descending neurons integrate from much of the central brain, so even a
  correct sensory→motor subgraph may be thousands of neurons. That's acceptable (≪ 161k) but set
  expectations; do not promise a tiny circuit.
- **Pruning-step performance.** Computing reachability/paths over a ~25M-edge graph must use an
  efficient sparse BFS/traversal (not a Python O(N²) blowup) so the prune step itself doesn't take
  forever on the full matrix. It is a one-time dev-time cost, but must be minutes, not hours.
- **Consistency with downstream.** After pruning + index remap, UC-02's `select_motor_population` /
  `select_sensory_population` and the sign mask must still work on the pruned `ConnectomeData`
  without special-casing.
- **CI hermeticity.** Full-matrix pruning is dev-time only (needs the downloaded full matrix); the
  hermetic tests exercise the algorithm on the committed ~300-neuron fixture.
- **Interaction with the existing top-degree fixture.** The fixture is already a small generic slice;
  pruning it further is mainly to test the algorithm. The intended production use is prune-the-full-
  matrix. Document that distinction (like UC-03's SimpleDroneAdapter vs PyBullet split).

## Original Description
User request: instead of a random/generic slice of the connectome, choose the parts of the brain
that actually handle sensory input and motor output (and the wiring between them) and disregard the
rest — a function-targeted subgraph — so the real full MaleCNS connectome becomes tractable to train
on a laptop. Motivated by observing that full-connectome smoke-training runs at ~0.85 steps/s
(≈ weeks for 1M steps), dominated by propagating the whole 161k-neuron / 25M-edge graph.

## Clarifications
- Q: Random slice, or the functionally-relevant sensory/motor parts?
  A: Function-targeted — keep the sensory (visual_projection) and motor (descending_neuron) neurons
     and the neurons/edges on the directed paths between them; discard the rest. Not random, not the
     current top-degree hub slice.
- Q: How aggressive is the pruning?
  A: Deferred to the analyst/challenger — implement a configurable rule (k-hop neighborhood
     recommended as the default, shortest-path corridor as the tighter alternative) and report the
     actual reduction achieved on the full MaleCNS. Surface the choice in the plan preview.
