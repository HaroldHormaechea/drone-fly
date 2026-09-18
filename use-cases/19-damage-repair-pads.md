# Use Case 19: Damage/integrity + repair pads + proprioceptive(damage) observation block

## Summary
Give the drone a scalar **integrity** state (`1.0` pristine → `0.0` wrecked) that accrues **damage from UC-15 obstacle contacts** (the only non-terminating contact source in the env — floor/ceiling still crash), degrades **control authority**, is **restored by repair pads while fully landed**, and is **sensed** via a new observation block. Damage is **bundled with repair** because damage alone is inert (with no repair, the optimal policy is simply "avoid all contact," which UC-15's severe penalty already trains). Mechanics: **(a)** each obstacle-contact event (edge-triggered, sharing UC-15's contact signal — once per contact, not per overlapping frame) subtracts a tunable `damage_per_contact`; **(b)** integrity degrades **`max_body_rate`** (agility/steering) via `max_body_rate * f(integrity)` floored at `min_authority > 0` — deliberately *not* `max_thrust`, so a damaged drone is sluggish-but-flyable (a recoverable, learnable handicap, not an unrecoverable altitude loss); **(c)** **repair pads** (docking pads tagged `repair`) restore integrity while the drone is **fully landed** (UC-16 docked state) — hovering does not repair; **(d)** a width-1 **damage block** encoding `1 − integrity` (so "pristine" = the zeroed graft baseline) bound to the **`mechanosensory_proprioceptive`** population — a *second* proprioceptive block alongside UC-13's self-motion block, which coexist because the actor scatters blocks additively (`index_add`, overlap-safe). The new schema `extends()` the last-merged schema → **checkpoint invalidation (retrain)**. Repair-necessity is a **course-variation axis** (some courses damage-heavy enough that repair is needed to finish, others not). Off by default; hermetic on the numpy adapter.

## Acceptance Criteria
1. `integrity ∈ [0,1]` reset to `1.0`; off by default → pinned at `1.0`, no RNG perturbation, byte-identical to the pre-UC path.
2. Damage accrues from the **UC-15 obstacle-contact event** (edge-triggered, once per contact — NOT floor/ceiling `collided`, which still terminates); N distinct contacts → N decrements; continuous overlap across M steps → 1 decrement. Clamped ≥0.
3. Damage degrades the effective `max_body_rate` by a documented monotone `f(integrity)` floored at `min_authority > 0`; for a fixed action sequence a damaged instance shows a smaller attitude/angular-velocity response than pristine; at `integrity==1.0` dynamics are **byte-identical** to current; `max_thrust`/`mass`/`drag` untouched.
4. A pad tagged `repair` restores integrity (toward `1.0` at a documented rate, or on dock) **only while fully landed/docked** on it (UC-16) — not airborne, not on a non-repair pad; hovering does not repair; clamps at `1.0`.
5. A width-1 damage block encoding `1 − integrity`, `population="proprioceptive"`, appended in a new versioned schema that `extends()` its predecessor; `graft_actor` zero-inits it → **bit-identical actions** when the damage dim is fed `0` (pristine), reusing the UC-13 path.
6. Two proprioceptive blocks (UC-13 self-motion + this damage block) coexist: `forward` scatters both additively into the shared neurons; `select_modality("proprioceptive")` returns a non-empty population on the active slice or the build **fails loud** (`ModalityAbsentError`).
7. **Course variation:** repair-necessity is a documented axis — some courses are damage-heavy enough (obstacle density) that repair is needed to finish, others are completable without repairing.
8. Off-by-default byte-identity; hermetic (numpy adapter); smoke-train under the new schema finite; checkpoint invalidation documented in the PR.

## Potential Pitfalls & Open Questions
- **Dependency** — requires UC-15 (obstacle-contact = the damage source), UC-16 (docked state for repair). Damage-from-obstacles is the only coherent non-terminating source given floor/ceiling still terminate; accepted coupling to UC-15.
- **Risk (tuning band)** — `f(integrity)`, `min_authority`, `repair_rate`/`damage_per_contact` must sit in the narrow band where repair is *sometimes* worth the detour (too high `min_authority` → damage irrelevant, repair never worth it; too low → damaged flight uncompletable). Flagged as tunable, not asserted to "just work."
- **Open** — repair gradual vs instant; integrity encoding fixed at `1 − integrity` so the zeroed graft baseline = pristine (AC5 needs this or parity is subtly unsatisfiable).
- **Note** — two blocks on one population is a *learning-interference* consideration (the net must disentangle self-motion from damage on a shared substrate), biologically defensible and mechanically sound; PR note, not a blocker.
- **Ordering** — the schema predecessor is the last-merged schema (UC-16 adds no block, so this extends UC-17/UC-18's battery schema); resolve `extends()` base + version at implementation.
- **Risk** — checkpoint invalidation (accepted, as UC-13/15/17).

## Original Description
From the user's track-creator request ("pads to repair") plus the full-landing constraint and the reward-gate decision (some courses require repair, others don't, for variation). Fourth of the 4-UC line. Damage is bundled with repair because damage without a repair path is inert (optimal policy = avoid contact, already incentivised by UC-15).

## Clarifications / Decisions
- Q: Bundle damage with repair, or split?
  A: **Bundled** — damage alone adds observability cost with no new learnable behaviour; it only earns its keep with a repair path.
- Q: What does damage degrade?
  A: **Control authority (`max_body_rate`)**, floored at `min_authority` — sluggish but flyable/recoverable — not `max_thrust` (which risks unrecoverable altitude loss).
- Q: Damage source?
  A: **UC-15 obstacle contacts** (edge-triggered, shared signal) — the only coherent non-terminating source.
- Q: Does repair require a full landing?
  A: Yes — repair accrues only while fully landed/docked on a repair pad; hovering is insufficient.
- Q: Reward incentive?
  A: **Course-variation** — some courses damage-heavy enough to require repair, others not. No shaping-reward term.
