---
plan_for: use-cases/07-post-training-activation-pruning.md
work_branch: feat/uc-07-post-training-activation-pruning
team: drone-fly-uc-7
approved: 2026-09-17
---

# UC-07 Approved Implementation Plan — Post-training activation pruning → minimal functional flight circuit

Analyst↔challenger APPROVED (round 2). TARGET_DIR = /workspace/drone-fly-uc-07-post-training-activation-pruning.
User unavailable; all choices resolved with justified defaults.

### Analysis
Post-training, data-driven, structured (neuron-level) pruning of a *trained* connectome policy:
measure per-neuron activation over a representative episode set, remove low-activation interneurons,
fine-tune survivors to recover completion rate. Reuses UC-01 loader/`save_connectome`, UC-04 slice
machinery, the UC-05 actor sink (exact same activation quantity), UC-08 randomization, and the
evaluator's checkpoint-alignment contract. Opt-in and off by default → UC-01..06 byte-identical.

### Proposed Solution
**Importance metric (resolves the open question):** default `mean_abs` = per-neuron mean |activation|
over all frames; secondary reported stat `active_fraction` (fraction of frames |act|>eps). Drop
interneurons with metric ≤ `threshold` (absolute default ~1e-3, calibrated against post-tanh [-1,1],
documented + configurable; a percentile/relative `--threshold-mode` is the documented secondary).
Gradient/ablation importance noted as a costlier future option, NOT default. Aggressiveness judged by
completion-after, never the proxy. Deterministic tie-break: `kept` = sorted neuron indices.

**Measurement (AC1, non-invasive):** reuse `actor.sink` via a pull-based accumulator mirroring the
recorder (`sink()` stashes latest, `commit()` folds one sample/env step → running sums of |act|,
count>eps, frames). Two entry points mirroring the raw-vs-checkpoint split:
`measure_importance(actor, env, *, n_episodes, seed, metric, eps)` (raw actor + SimpleDroneAdapter env,
hermetic) and a checkpoint wrapper (`actor_from_model` + `build_vec_env` + `neuron_count==actor.n_neurons`
alignment assertion). Same loop counts `info["completed"]` → completion-before, stamps whether
randomization was on.

**Distribution guard (AC2, mandatory):** `course_specific` warning fires **precisely when neither
randomization axis is enabled** (multiple seeds on one fixed course still trips it).
`--randomize`/`--randomize-dynamics` flow into the measurement env via `env_config.randomization`.
In hermetic CI (fixture + SimpleDroneAdapter, no randomization) the warning ALWAYS fires by design;
that CI slice is explicitly not a robust circuit. Recorded in report + slice provenance.

**Endpoint retention (AC4):** retain every `superclass ∈ {visual_projection, descending_neuron}`
(full sets, matching UC-04's `_endpoints`); prune only below-threshold interneurons.

**Structured prune (AC3):** `kept = sorted(endpoints ∪ {interneurons above threshold})` → new public
helper `slice_connectome(data, kept, *, source)` (extracted from `prune_to_subcircuit`'s inline
`adjacency[kept][:,kept]` + `meta[kept]`; UC-04 refactored to call it, behaviour byte-identical) →
new smaller `ConnectomeData`, meta re-aligned, indices remapped, **input unmutated**. sign-mask
integrity is free: `SparseConnectomeLayer` rebuilds `sign_mask = sign[coo.col]` from pruned adjacency
+ `sign[kept]`.

**Post-prune connectivity check (the real gap vs UC-04):** after slicing, BFS forward (reuse
`prune._bfs`) from the **pinned** `sensory_index` over CSC successors on the *pruned* graph; assert
each **pinned** `motor_index` neuron is reached. Partial disconnection → loud WARNING + structured
`disconnected_motors` in report/provenance. **Zero pinned motors reachable → hard error** (can't fly,
fine-tune can't recover; aligns with AC10). (Plain reachability is conservative; propagation only
unrolls n_steps=2 hops — noted caveat; completion-after backstops it.)

**Weight transfer + fine-tune (AC5/AC6) — the crux:**
- Sparse `edge_weight` transfer is exact: pruned edges ⊆ original; map each pruned edge (r,c) →
  original (kept[r],kept[c]) **by (row,col) identity** and copy the trained magnitude.
- **Pin sensory/motor sub-populations by neuron identity** to defeat degree-reranking drift: add an
  optional `sensory_index`/`motor_index` ctor param to `ConnectomeActorNetwork` (threaded through
  `ConnectomeFeaturesExtractor`). When supplied it **bypasses `select_populations` entirely** (not a
  post-hoc buffer set); default None → current path (byte-identical UC-01..06). Endpoint retention
  keeps sub-pop sizes constant (32/16) → `input_projection`/`readout` transfer 1:1.
- **Persist the pinned indices through `features_extractor_kwargs`** (`{"data": pruned,
  "sensory_index": …, "motor_index": …}`) so the constructed actor uses them and they pickle into the
  checkpoint — guarantees `PPO.load` (which passes no connectome, re-runs the extractor ctor) can
  never re-select a misaligned sub-pop.
- Build fresh PPO on the pruned graph with pinned indices, load the transferred state, run a short
  `learn(reset_num_timesteps=False)` continue-pass (`--finetune-steps`, carrying `--vecnormalize`;
  warn + best-effort fresh stats if absent, like train-resume).
- Report completion **before / immediately-after-prune / after-finetune**.

**Faithfulness anchor:** with a threshold retaining ALL neurons, the pruned actor's output must equal
the original within float tolerance — unit-tested.

**Outputs (AC6/AC9):** (a) fine-tuned pruned checkpoint (`model.save`), (b) pruned slice via
`save_connectome` (round-trips `load_connectome`, viewer-loadable), (c) `PRUNE_TRAINED_REPORT.md` +
slice provenance: neurons/edges before→after, fraction removed, metric summary (min/median/max,
kept/dropped), metric+threshold+episode-distribution provenance, completion
before/after/after-finetune, `disconnected_motors`, course-specific warning when applicable.

**Determinism + immutability (AC7):** fixed (seed, metric, threshold, episode set) → deterministic
stats → deterministic `kept` → deterministic pruned graph. Input checkpoint file and input
`ConnectomeData` unmutated. Fine-tune seeded but numerics platform-variable; only the pruning DECISION
asserted deterministic.

**Opt-in CLI (AC8):** new `drone-fly prune-trained`: `--checkpoint --connectome [--vecnormalize]
[--prune --prune-k] [--metric mean_abs|active_fraction] [--threshold] [--threshold-mode] [--episodes N]
[--finetune-steps] [--randomize] [--randomize-dynamics] [--adapter] [--device] [--seed] --out`.
`--prune/--prune-k` rebuild the SAME base graph the checkpoint was trained on (composes UC-04 then
UC-07). Omitting the subcommand → UC-01..06 byte-for-byte unchanged.

**Errors/degrade (AC10):** missing `superclass` → error; threshold pruning to empty/degenerate →
error; no episodes measured → error; zero pinned motors reachable → error. When override supplied:
`sensory_mode`/`motor_mode` report `"pinned"`; assert `set(sensory_index) ∪ set(motor_index) ⊆
set(kept)` (size + order preserved) before transfer.

### Files Affected
**Production (`src/drone_fly/**`):**
- `connectome/prune.py` — extract public `slice_connectome(data, kept, *, source)`; refactor
  `prune_to_subcircuit` to call it (UC-04 unchanged). Export from `connectome/__init__.py`.
- `controller/actor.py` — optional `sensory_index`/`motor_index` ctor param bypassing
  `select_populations` when supplied (default None → current path); `"pinned"` mode tag.
- `controller/sb3.py` — thread the optional indices through `ConnectomeFeaturesExtractor` (+ its
  `features_extractor_kwargs` pickling path).
- NEW `src/drone_fly/prune_trained/`: `measure.py` (accumulator sink + measurement loop, raw +
  checkpoint), `importance.py` (metric + kept-set selection + reachability via `prune._bfs`),
  `transfer.py` (build pruned actor + exact identity weight transfer + wrap into PPO), `workflow.py`
  (orchestrate measure→prune→connectivity-check→fine-tune→save report/checkpoint/slice), `__init__.py`.
- `cli/__init__.py` — add `prune-trained` subcommand (additive, opt-in).

**Test (`tests/**`):** NEW `tests/test_prune_trained.py`, all hermetic on the committed fixture (no
network/browser/pybullet): measure importance (finite, deterministic across 2 runs); retain-all
threshold ⇒ pruned-actor output == original within tol (faithfulness anchor); real threshold ⇒
smaller/valid/endpoint-preserving graph, round-trips `load_connectome`, deterministic, finite `(4,)`;
**save → PPO.load (no connectome passed) → identical-action round-trip**; checkpoint+fine-tune
plumbing (tiny smoke_train checkpoint → prune-trained with trivially-short finetune → loads + finite
`(4,)`; slice round-trips; report written); guards (missing superclass / empty-prune / no-episodes /
zero-motors error; partial disconnection warns + records disconnected_motors; distribution warning
fires on fixed course); back-compat (feature off ⇒ UC-01..06 unchanged; default actor construction
byte-identical).

### Risks
- Weight-transfer faithfulness — mitigated by retain-all==identical test.
- Degree-reranking population drift — eliminated by pinned indices persisted through
  `features_extractor_kwargs`.
- Activation selection can sever obs→motor paths — caught by post-prune reachability (warn partial /
  error if zero).
- Fine-tune nondeterminism / re-overfit to fixed course — documented; only pruning decision asserted
  deterministic; report before/after/after-finetune; recommend `--randomize`.
- Real completion numbers for README (AC9) — full trained policy + fine-tune is a dev-time step on the
  owner's machine (documented boundary, mirroring UC-03/04); hermetic tests assert plumbing not convergence.
- VecNormalize for fine-tune — carried via `--vecnormalize`; warn + best-effort fresh if absent.
- **Scope guard:** activation-pruning + fine-tune + save + report ONLY. No new training regime, no
  domain randomization (reuse UC-08), no viewer changes.
