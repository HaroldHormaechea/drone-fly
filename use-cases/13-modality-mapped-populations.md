# Use Case 13: Modality-mapped neuron populations + graft-ready observation schema

## Summary
Build the foundational machinery for biologically-bound, extensible observations. **(1)** Extend the loader to surface `class`/`subclass` (not just `superclass`), so populations are selectable by sensory modality. **(2)** A modality→population selector returning the neuron indices of a biological class in a given connectome/slice — cleanly-labeled modalities (vision=`visual_projection`, proprioceptive/`mechanosensory_*`, gustatory, olfactory, thermo, hygro) selectable, approximate ones (hunger, motion) exposed but flagged as curated-name-list-based. **(3)** A versioned, block-structured observation schema (obs = ordered named blocks, each declaring width + bound population) with a **full zero-init graft path**: adding a block appends zero-initialized input-projection columns so a schema-trained checkpoint loads under the wider obs and behaves identically until fine-tuned. As part of this, today's flight observation is **decomposed and re-bound** into biologically-mapped blocks — target-relative features → vision (`visual_projection`), self-motion features (attitude/linear+angular velocity) → a proprioceptive/mechanosensory population. This is a **deliberate re-architecture**: it changes the input→sensory wiring, so a policy must be retrained under the new schema (the existing checkpoint is not carried over). No new flight behaviour or live input is added; this is enabling machinery so later use cases (obstacles, damage, battery) add a block bound to the right population and warm-start.

## Acceptance Criteria
1. The loader surfaces `class`/`subclass` alongside `superclass`, without breaking existing `superclass`-based selection.
2. A modality→population selector maps a modality name to neuron indices of that class in a given connectome/slice; cleanly-labeled modalities are selectable and approximate ones (hunger, motion) are exposed but explicitly flagged as name-list-based.
3. The observation is an ordered set of **named, versioned blocks**; each block records its width and bound population.
4. Today's flight observation is decomposed into a vision block (target-relative) and a proprioceptive block (self-motion) and re-bound to those populations. This deliberately changes input wiring; a policy **retrained** under the new schema learns to fly (a smoke-train sanity check confirms trainability). The pre-existing checkpoint is explicitly **not** preserved.
5. **Graft:** adding a block to a schema-trained checkpoint appends zero-initialized input-projection columns; the grafted model's actions are identical to the pre-graft model on the old inputs with the new block zeroed (verifiable parity test) — motor readout and connectome graph unchanged.
6. The selector/binding fails loudly if a requested modality population is **absent from the active slice** (e.g. proprioceptive neurons not present in a vision→motor prune), rather than silently binding to nothing.
7. No new flight behaviour or live input is introduced; parity/sanity tests only.

## Potential Pitfalls & Open Questions
- **Risk** — re-binding invalidates the current 250k checkpoint; a retrain under the new schema is required. Accepted per the design choice, but must be called out in the change so it isn't a surprise.
- **Risk (feasibility)** — the proprioceptive/mechanosensory population may not exist in the k0 (`visual_projection→descending`) slice; re-binding self-motion there needs those neurons present. May force redefining the prune (include proprioceptive→motor paths) or training on the full connectome — a direct interaction with UC-04. The analyst must resolve slice coverage before the re-bind is buildable.
- **Ambiguity** — the exact split of the current 12-d obs across the vision vs proprioceptive blocks (which dims go where).
- **Assumption** — block composition / active modalities are config-driven (pairs with UC-11).
- **Assumption** — the graft path handles the connectome specifics (input `Linear(OBS_DIM→sensory_pop)` + pinned sensory/motor indices from UC-07), not a vanilla MLP widen.

## Original Description
After the other three use cases, prepare for what was discussed: map the neuron groups based on the stuff I expect to be training in the future (e.g. proprioception for damage, hunger feeling for battery, vision for obstacle and waypoint detection, motion for moving-target following), with the possibility of grafting more behaviours later — tying each new behaviour to the proper neurons. (Grounded by the labeling research: vision and proprioceptive/mechanosensory are cleanly labeled; gustatory/olfactory/thermo/hygro exist as `class`; hunger and motion are only approximate via curated cell-type name lists.)

## Clarifications
- Q: What's the scope line for this foundational use case?
  A: Machinery + migrate the current observation into the new schema (loader class column + modality selector + versioned block schema + graft path). No new modality/behaviour.
- Q: The current flight inputs are abstract state injected into visual_projection neurons — keep or re-bind?
  A: Re-bind to biologically-appropriate populations (interpreted as decomposing flight state into a vision block and a self-motion/proprioceptive block). This invalidates the current checkpoint and requires a retrain.
- Q: How far should the graft mechanism be built in this UC?
  A: The full zero-init graft path — the actual checkpoint-widening code, tested end-to-end.
- Q: Which modalities should the selector expose as bindable now?
  A: Clean labels (vision, proprioceptive/mechanosensory, gustatory, olfactory, thermo, hygro) plus the approximate ones (hunger, motion) clearly flagged as name-list-based.
