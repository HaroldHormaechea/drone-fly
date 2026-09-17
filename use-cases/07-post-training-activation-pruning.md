# Use Case 07: Post-training activation pruning → minimal functional flight circuit

## Summary
UC-04 prunes the connectome **structurally, before training** — it keeps the neurons and edges on
the directed sensory→motor pathway so PPO is tractable on a laptop. This use case is the
**complement**: **data-driven, post-training, structured (neuron-level) pruning** of the *trained*
policy. Once a policy has learned to fly, many neurons in the pruned substrate sit near-zero across
a rollout — often interneurons that inherited fly-brain functions irrelevant to flight (smell,
memory, gustation, courtship, …) and that training simply never learned to use. This use case
measures each neuron's **activation / contribution over a representative set of episodes** and
**removes the neurons that never (or rarely) activate or contribute nothing**, then **fine-tunes**
the survivors to recover any lost completion rate. The output is two things at once: (a) a
**smaller, faster policy** (efficiency / model compression), and (b) the **minimal functional
flight circuit** — the sub-network the trained fly-brain policy actually uses to fly, as a saved,
loadable, visualizable `ConnectomeData` slice (interpretability / a scientific result).

This is a standard ML technique — network pruning / model compression — and it follows the canonical
**train → prune → fine-tune** workflow. It is **opt-in and off by default** (a new CLI subcommand /
eval-time analysis flag), so UC-01…UC-06 behaviour is unchanged. It reuses UC-04's structured-pruning
+ `save_connectome` machinery so the removed neurons take their incident edges with them, meta
(`sign` / `neuron_ids` / `superclass` / `top_nt`) is re-aligned and indices remapped, and the result
round-trips through `load_connectome` and is visualizable by the UC-05/UC-06 viewer. The step is
deterministic, leaves the input policy/connectome unmutated, and logs the reduction (neurons/edges
before→after), the activation statistics, and the completion rate before vs after (including after
fine-tune). CI stays hermetic: activations are measured on a short fixture rollout, pruning is
asserted valid/loadable/deterministic, and no network or browser is involved.

**The key caveat — distribution dependence (read this before anything else).** "Never activates" is
only meaningful *relative to an input distribution*. The project today has **no domain randomization**:
training converges onto a single memorized path through one fixed course. If activations are measured
on that narrow, memorized run, the step will prune the policy down to *just that memorized path* and
destroy any latent robustness the wider circuit might have held — you would be "compressing" the fly
brain into a lookup table for one trajectory. Therefore this use case MUST measure activations over a
**varied / representative** distribution, and explicitly recommends running it **after** domain
randomization exists (a separate prerequisite UC it depends on for a *robust* result). If only a
fixed course is available, the step MUST loudly WARN that the produced circuit is **course-specific**
(a demonstration of the technique and of *this* run's used sub-network — not a general minimal fly
flight circuit).

## Relationship to other use cases (not a replacement for UC-04)
- **UC-04 (structural, pre-training):** topological pruning of the connectome graph so a trainable
  sensory→motor circuit exists at all. Answers *"what could plausibly matter for flight?"* — decided
  from graph structure, before any training.
- **UC-07 (activation/contribution, post-training):** removes what *turned out* to be dead weight for
  the learned task. Answers *"what did the trained policy actually use?"* — decided from measured
  behaviour, after training. They **compose**: UC-04 gives a trainable circuit; UC-07 trims it to the
  minimal functional slice. UC-07 never replaces UC-04 and does not re-run UC-04's structural rule.
- **UC-05/UC-06 (viewer):** the saved pruned slice from this UC is a smaller, legible input to the
  activation-playback / 3D viewer — "here is the circuit the fly actually flies with." This UC does
  **not** change the viewer (UC-06 owns viewer work); it only produces a slice the viewer can load.

## Open design question (for the analyst/challenger to resolve in the plan)
**Which importance metric?** Leave the choice to the plan; note both and pick a default:
- **Activation-magnitude (simple):** per-neuron mean `|activation|` and/or the fraction of frames the
  neuron is above a threshold, over the representative episodes. Cheap, intuitive, but a low-activation
  neuron can still be *important* (e.g. a rarely-firing gate).
- **Contribution / gradient-based (more principled):** importance ≈ effect on the policy output /
  loss when the neuron is ablated or scaled to zero (saliency / Taylor-expansion / ablation deltas).
  More faithful to "does removing it change flight?", costs more compute.
Recommend implementing a **configurable metric with a documented default and threshold**, and always
reporting completion-rate before vs after (and after fine-tune) so the pruning aggressiveness is
judged by behaviour, not by a proxy number alone.

## Acceptance Criteria
1. **Activation/importance measurement over a representative distribution.** A step runs the trained
   policy over a configurable set of episodes and records per-neuron importance (per the chosen
   metric) across all frames. The **episode distribution is a first-class input** (seeds / course
   variation / — when it exists — domain randomization); the number of episodes and the metric +
   threshold are documented, configurable constants. Measurement is **non-invasive**: it reuses
   UC-05's activation-capture path (detached / no-grad) and does not alter policy numerics.
2. **Distribution-dependence guard (mandatory).** If the measurement runs over a fixed/undiversified
   course (no domain randomization available), the step emits a prominent **WARNING** that the
   resulting circuit is course-specific and records that fact in the run report and in the saved
   slice's provenance note. The docs state that a robust minimal circuit requires a varied
   distribution (the future domain-randomization UC).
3. **Structured neuron-level pruning.** Pruning removes **whole neurons together with their incident
   edges** (not weight-zeroing / masking), producing a **new, smaller `ConnectomeData`**. `sign`,
   `neuron_ids`, `superclass`, `top_nt` are re-aligned to the pruned rows; per-edge signs and the
   sign-mask semantics are preserved; indices are remapped so UC-02's population selection still
   works on the result. Reuses UC-04's pruning + meta-realignment machinery rather than a parallel
   implementation.
4. **Endpoint retention.** The sensory (`visual_projection`) and motor (`descending_neuron`)
   populations are **retained by default** (or a single, documented rule governs the taps) — pruning
   targets **interneurons**; the input/output boundary is not silently severed. A retained-endpoints
   invariant is unit-tested on the fixture.
5. **Fine-tune to recover.** After pruning, a short **PPO continue-training** pass on the pruned
   policy is run (configurable, opt-in as part of the workflow) to recover completion rate. The run
   reports **completion rate before pruning, immediately after pruning, and after fine-tune** so the
   train→prune→fine-tune effect is visible and honest (including cases where fine-tune does not fully
   recover).
6. **Saved outputs.** The step produces: (a) a **pruned policy checkpoint** (the fine-tuned smaller
   policy), and (b) a **saved pruned connectome slice** written via UC-04's `save_connectome` format
   (`.npz` matrix + `<stem>_meta.csv`) so it **round-trips through `load_connectome`** and is
   loadable by the UC-05/UC-06 viewer, plus (c) a **report** (neurons/edges before→after, per-neuron
   activation stats / metric summary, completion before/after/after-fine-tune, metric + threshold +
   episode-distribution provenance, and the course-specific warning when applicable).
7. **Input immutability + determinism.** The input policy checkpoint and input `ConnectomeData` are
   **not mutated**. For a fixed seed, metric, threshold, and episode set, the measurement and the
   resulting pruned graph (neuron set, index order, edges) are **deterministic** (fixed tie-break),
   asserted on the fixture. (Fine-tune, being PPO, is seeded but its numerics may vary by platform;
   the *pruning* decision must be deterministic.)
8. **Opt-in wiring, back-compatible.** A new opt-in surface (e.g. `drone-fly prune-trained
   --checkpoint <ckpt> --connectome <dir> [--metric …] [--threshold …] [--episodes N] [--finetune-steps …]
   --out <dir>`, or an equivalent eval-time analysis flag) drives the workflow. Omitting it leaves
   UC-01…UC-06 behaviour byte-for-byte unchanged (asserted).
9. **Reduction + stats reporting.** The step logs and returns the input→pruned counts (neurons and
   edges), the fraction of neurons removed, and the activation/importance summary; the README
   documents the actual reduction and completion-before/after measured on a real trained policy so the
   efficiency and minimal-circuit claims are concrete, not asserted.
10. **Degrade / error handling.** If the connectome lacks the meta needed to identify endpoint
    populations, or a threshold/metric would prune to an empty/degenerate graph (or would remove an
    endpoint population), the step stops with a **clear, actionable message** — never silently
    returns garbage. Attempting to prune with **no representative episodes** measured is an error.
11. **Hermetic tests.** The full suite runs on the committed fixture with no network/browser:
    smoke-measure activations on a short fixture rollout (via the SimpleDroneAdapter), prune, and
    assert — the pruned graph is smaller, valid, endpoint-preserving, round-trips through
    `load_connectome`, and is deterministic; the pruned checkpoint loads and produces a finite `(4,)`
    action; back-compat when the feature is off. A real full fine-tune is a **dev-time** step on the
    owner's machine (documented boundary, mirroring UC-03/UC-04); the hermetic test may run a
    trivially-short fine-tune or assert the plumbing without requiring convergence. UC-01…UC-06
    suites stay green.

## Potential Pitfalls & Open Questions
- **Risk — distribution dependence is the whole ballgame.** Measuring "never activates" on the
  current no-domain-randomization, single-memorized-path policy will prune to that path and *look*
  like a great compression while silently destroying robustness. The representative-distribution
  requirement + the course-specific warning (AC-1, AC-2) exist precisely to stop a misleading result.
  Do not present a fixed-course circuit as "the minimal fly flight circuit."
- **Missing input — domain randomization does not exist yet.** A *robust* minimal circuit depends on
  a varied input distribution, which is a **separate prerequisite UC** not yet built. This UC is
  still valuable now (efficiency + demonstrating the used sub-network for one course), but the plan
  must be explicit that the general/robust result is gated on that future work.
- **Ambiguity — the importance metric (design fork above).** Activation-magnitude is simple but can
  drop a rarely-firing but important neuron; contribution/gradient-based is more faithful but costlier.
  Make it configurable; judge aggressiveness by completion-rate-after, not by the proxy alone.
- **Edge case — endpoints near zero.** A sensory or motor neuron might itself rarely activate on a
  given course; retention rules (AC-4) must protect the taps even when their measured activation is
  low, or document precisely when an endpoint may be dropped.
- **Assumption — reuse UC-04, don't fork it.** The neuron-removal + edge-drop + meta-realignment +
  `save_connectome` path already exists from UC-04; this UC selects a *different neuron set* (by
  measured activation, not graph structure) and feeds it through the same machinery. A parallel
  pruning implementation is out of scope and a maintenance hazard.
- **Risk — fine-tune masks or undoes the point.** Too much continue-training could re-recruit
  pruned-away capacity's role into survivors (fine) or overfit further to the single course (bad).
  Keep fine-tune short and documented; always report before/after/after-fine-tune so the effect is
  visible rather than hidden.
- **Edge case — activation definition.** Reuse UC-05's exact per-frame activation definition (the
  post-propagation neuron state used to produce each step's action) so importance is measured on the
  same quantity the viewer shows; document it, don't re-derive a second notion of "activation."
- **Scope guard.** In scope: post-training activation/contribution measurement over a representative
  distribution + structured neuron-level pruning (reusing UC-04) + a short PPO fine-tune + saving the
  pruned policy checkpoint and pruned connectome slice + a report, all opt-in and off by default.
  **Out of scope:** a new training regime, domain randomization (a separate prerequisite UC), and any
  viewer changes (UC-06 owns those — this UC only produces a slice the viewer can load).

## Original Description
The user asked whether pruning a connectome policy **after** training is useful — specifically to
remove neurons that never activate, since a fly brain contains large populations doing non-flight
functions (smell, memory, etc.) that a flight policy would never exercise. The discussion established
that this is a standard ML technique (network pruning / model compression, the canonical
train → prune → fine-tune workflow), and that it is doubly useful here: it makes the trained policy
smaller/faster **and** it reveals the **minimal functional flight circuit** — the sub-network the
trained fly-brain policy actually uses to fly, which is an interpretability / scientific result on top
of the efficiency win. The important caveat surfaced in the discussion: whether a neuron "never
activates" is only meaningful over a **varied / representative** input distribution — measuring on the
project's current single memorized path (no domain randomization) would prune the policy down to that
one path and throw away any latent robustness, so activation must be measured over a representative
distribution (ideally after domain randomization exists), and a fixed-course result must be flagged as
course-specific.

## Clarifications
- Q: Weight-zeroing / masking, or actually remove neurons?
  A: **Structured, neuron-level** pruning — remove whole neurons and their incident edges, producing a
     genuinely smaller `ConnectomeData` (not a mask over the full-size network).
- Q: What distribution should activations be measured over?
  A: A **representative / varied** distribution — ideally after domain randomization exists. On the
     current fixed course the step must WARN that the resulting circuit is course-specific, because
     measuring on the memorized single path would prune to that path and destroy latent robustness.
- Q: Do we just prune, or also recover?
  A: **Train → prune → fine-tune.** Run a short PPO continue-training pass on the pruned policy and
     report completion rate before pruning, after pruning, and after fine-tune.
- Q: What about the sensory and motor neurons?
  A: **Retain the endpoints** (`visual_projection` sensory, `descending_neuron` motor) by default —
     prune interneurons, be careful at the taps (or document a precise rule if an endpoint may go).
- Q: How is the pruned result saved?
  A: Reuse **UC-04's `save_connectome` format** (`.npz` + `_meta.csv`) so it round-trips through
     `load_connectome` and is loadable by the UC-05/UC-06 viewer; also save the fine-tuned pruned
     policy checkpoint and a report.
- Q: Does this replace UC-04's pruning?
  A: No — it **complements** it. UC-04 is structural/pre-training (tractability); UC-07 is
     activation/contribution-based/post-training (efficiency + interpretability). They compose.
- Q: How does it relate to the viewer?
  A: It **pairs** with the UC-05/UC-06 viewer — the smaller saved slice is a legible input to visualize
     the minimal circuit — but this UC makes **no viewer changes** (UC-06 owns those).
