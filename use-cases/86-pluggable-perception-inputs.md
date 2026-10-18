# Use Case 86: Pluggable perception inputs — a contract between "eyes" and the brain

## Summary
Make the live model's environment-aware inputs pluggable. Today the 70-dim connectome ("brain") consumes fixed contract blocks:
- `raycast` (40 obstacle rays in CONE order, [0, 1]) and `obstacle_vision` (8 values: the 2 nearest obstacles' offset plus size), which come from the `free_space_estimator` part;
- `curr_gate`, `next_lookahead` and `target`, which come from the `gate_estimator` part;
- telemetry state.

These are produced by parts registered by hand in `src/drone_fly/harness/builtins.py` (`free_space_estimator`: none / depth / eyes; `gate_estimator`: learned / opencv / oracle / course).

UC-86 formalises a **perception-provider interface**: a documented, versioned contract stating which input blocks a provider supplies, their shapes, ranges, frames and units, the timestamp/staleness semantics, and its provenance (screen-derived vs metadata). Any "eyes" system (Depth-Anything raycast, the UC-78/83/85 vision eyes, a future stereo/optical-flow/segmentation net) can then be added as a plugin and selected from config, without editing harness code or the brain.

The deliverable has six parts:
1. the interface and its runtime validation;
2. config-driven plugin loading (a `module:Class` reference and/or Python entry points), on top of the existing registry;
3. a screen-only provenance guard for live flights;
4. a provider conformance test kit;
5. an offline evaluation command that scores any obstacle-vision provider against the eyes labels on the held-out split with the existing UC-78 metrics;
6. docs with a minimal example plugin.

The existing parts (none, depth, eyes, learned, opencv) are migrated onto the interface with byte-identical behaviour by default.

## Acceptance Criteria
1. **Interface.**
   - A perception-provider protocol/ABC in `src/drone_fly/harness/` (e.g. `perception.py`) declares:
     - the blocks provided (names from the contract's block registry);
     - each block's dtype and shape, value range and coordinate frame;
     - a `stamp` (capture time of the newest input used; −inf when no estimate);
     - `provenance` (`screen` | `telemetry` | `metadata`);
     - lifecycle (`start` / `stop` / `reset` / `update`) consistent with the current part contract.
   - The interface carries a contract version number. The existing `FreeSpaceEstimator` and `GateEstimator` roles are expressed through it, with no change in what the runner consumes.
2. **Runtime validation.** When a provider's output does not match its declared contract, the harness raises a clear error naming the provider and the block, at start and on the first estimate. Errors cover a wrong shape or dtype, values out of range, non-finite values, or a missing declared block. A provider that declares blocks the loaded model does not consume (or misses blocks it needs for the configured `extra_dim`) is refused at session build with an actionable message.
3. **Pluggable loading.**
   - A provider can be selected from config by registered name (as today) or by an import reference (`module.path:ClassName`).
   - Optionally, providers are discoverable through a Python entry-point group (e.g. `drone_fly.perception`).
   - No edit to `builtins.py` is needed to add a provider.
   - Lazy-import rules still hold: no torch at harness import time, and the hermetic import test stays green.
4. **Composition.** Different blocks can come from different providers: obstacle rays from one, gate blocks from another, with an explicit config mapping. Overlapping claims on the same block are refused unless one is marked as the primary.
5. **Screen-only guard.** A live session can be configured `inputs.require_provenance: screen` (the default for live Liftoff flights of environment-aware blocks). It then refuses any provider whose environment-aware blocks have `metadata` provenance, e.g. the `oracle` / `course` gate estimators, or obstacle geometry from track XML. Telemetry-derived state is allowed. Offline/sim harnesses are unaffected.
6. **Migration, byte-identical.** `none`, `depth`, `eyes`, `learned`, `opencv`, `oracle` and `course` run on the interface. Default configs produce the same observations as before: a recorded-run replay or a deterministic fixture shows identical assembled 70-dim / 30-dim observation vectors. All existing harness tests pass unchanged, except for updated registry snapshots.
7. **Conformance kit.**
   - A reusable test helper (e.g. `tests/perception_conformance.py`, or a public `drone_fly.harness.perception.check_provider`) exercises any provider on synthetic frames and telemetry. It checks the declared contract, staleness (−inf before the first estimate, a held stamp during a gap, reset), thread safety of async providers, and stop/join.
   - The kit is applied to every built-in provider and to the example plugin in CI (CPU only, no Liftoff, no network).
8. **Offline provider evaluation.**
   - `drone-fly perception eval --provider <name|module:Class> --config <yaml> --dataset <eyes dataset> --split held-out` runs an obstacle-vision provider over the dataset's frames and telemetry.
   - It reports the UC-78 metrics (per-ray miss, in-view miss, nearest-quarter 0–4 m miss, false alarm on clear rays) plus latency p50/p95 on the current device. It writes JSON and markdown comparable with `eyes compare`.
   - It works at least for `depth` and `eyes`, so they can be compared like-for-like.
9. **Example plugin + docs.**
   - A minimal example provider (e.g. a "constant clear" or "brightness-threshold" obstacle-vision plugin, outside `builtins.py`) shows the full path: implement → declare contract → select from config → pass the conformance kit → evaluate.
   - `docs/harness/perception-providers.md` documents the contract, provenance rules, config syntax, the composition rules and the evaluation command.
10. **Latency visibility.** Each provider's per-frame wall time and estimate age go into the run's `latency_summary.json` under its own provider name, with no change to recorder internals beyond a generic per-provider row.

## Potential Pitfalls & Open Questions
- **Assumption** — The interface sits at the harness-part level, which produces the contract blocks the brain consumes. The brain's input layout (contract blocks, `extra_dim`) is not changed.
- **Risk** — Migrating the existing parts could subtly change behaviour (ordering, reset timing). The byte-identical observation check (AC6) is the guard; keep the migration mechanical.
- **Edge case** — Async providers on CUDA must keep the stream-isolation and duty-cycle rules introduced for `eyes`. The interface should carry them as guidance, plus an optional GPU-duty declaration.
- **Edge case** — Gate blocks and obstacle blocks have different staleness needs. Per-block age limits may be needed; the default stays `runner.max_estimate_age_s`.
- **Risk** — Entry-point discovery adds packaging surface. It is optional, and import-reference loading alone satisfies AC3.
- **Ambiguity** — The exact labels a provider is scored against in `perception eval`: the eyes labels (track geometry raycast) are ground truth only for obstacle rays. Gate-provider evaluation is out of scope here (UC-71 has its own).

## Original Description
(Authored by the orchestrator on 2026-10-18 at the user's request, autonomous session.) User's words: "make 'input' pluggable, e.g. make an interface to the AI model so we can have pluggable eyes using different systems that convert to the input the 'brain' expects; define as a UC and implement it autonomously too." Context:
- Live environment-aware inputs must come only from what is visible on screen (no track metadata); telemetry is fine.
- The in-flight eyes-as-live-estimator PR (feat/eyes-live-free-space-estimator) introduced the first written obstacle-vision provider contract in its module docstring. UC-86 generalises it.

## Clarifications
(Answered by the orchestrator from the user's stated intent; the user was away and asked for autonomous work.)
- Q: Interface level: harness parts producing contract blocks, or a new brain-side adapter that changes the model input? A: Harness-part level. The brain's input contract stays fixed; providers convert into it.
- Q: Plugin mechanism? A: Config `module:Class` import references (required), plus an optional entry-point group. The registry stays the backbone.
- Q: Should "screen-only" be enforced? A: Yes, as a provenance flag with `require_provenance: screen`, defaulting on for live Liftoff environment-aware blocks.
- Q: Should existing parts be migrated? A: Yes, byte-identical by default, so there's one way to add perception.
- Q: How do we judge a new eyes system? A: An offline `perception eval` reusing the UC-78 ray metrics plus latency, on the intact held-out split.
