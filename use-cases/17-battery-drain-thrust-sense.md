# Use Case 17: Battery drain + thrust impact + battery ("hunger") observation block

## Summary
Add a first-class **battery/energy state** to the drone and wire its consequences through the dynamics, reward, and observation schema — **without pads** (recharge is UC-18). Today there is no energy model: thrust is `throttle * MAX_THRUST` with a fixed ceiling, and UC-08's `thrust_factor_range` only randomizes that ceiling once per episode (constant within it — not depletion). This UC introduces genuine intra-episode state: **(a)** a normalized `battery ∈ [0,1]` reset to full and drained each step by a base idle rate plus a **throttle-proportional** term over `dt`; **(b)** an **effective thrust ceiling that falls as charge depletes** (`effective_max_thrust = base * thrust_factor * ceiling_factor(battery)`), so a low battery cannot sustain hover; **(c)** a documented **depletion consequence** (soft: ceiling drops below hover so the drone sinks to the floor via the existing crash path; or hard: explicit termination — pin one); and **(d)** a width-1 **battery observation block** (normalized charge) bound to the approximate **`hunger`** population via UC-13's `graft_actor` path — a new schema that `extends()` UC-15's `obstacle_vision_v2` (schema width grows again), which **re-binds input→sensory wiring and therefore invalidates existing checkpoints (retrain required)**, exactly as UC-13/UC-15. Off by default (byte-identical when disabled); the default course is completable **without** recharge; hermetic on the numpy adapter.

## Acceptance Criteria
1. Adapter carries `battery ∈ [0,1]` reset to `1.0`; each step drains `idle_rate + throttle_rate*throttle` over `dt`, clamped ≥0. Monotone non-increasing; a higher-throttle trajectory drains strictly faster than a lower-throttle one over equal steps (tested).
2. Effective thrust = `throttle * effective_max_thrust(battery)` with `effective_max_thrust` a documented monotone function of charge (full at 1). Holding action/dynamics fixed, achievable vertical acceleration is strictly lower at low battery than full (tested).
3. Depletion consequence at `battery==0` is a single documented, tested choice (soft ceiling-below-hover → floor path, or hard terminate).
4. A width-1 battery block bound to `population="hunger"` is appended in a new versioned schema that `extends()` its predecessor (the last-merged schema, i.e. `obstacle_vision_v2`); `graft_actor` zero-inits the block (weight+bias) → **bit-identical actions** on old inputs with the battery dim at its baseline (charge encoding chosen so the zeroed graft input equals the trained baseline — documented).
5. **Off-by-default byte-identity:** disabled → `thrust == throttle*_max_thrust` exactly, no `np_random` draw, fixed-seed episode reproduces the pre-UC-17 trajectory bit-for-bit.
6. Composes with UC-08 `thrust_factor` via a documented product order (randomization sets the full-charge ceiling; battery scales within the episode); disabled path draws nothing.
7. Hermetic (numpy adapter, no pybullet/network); smoke-train under the new schema completes a finite update; the **default course is completable without recharging**.
8. Checkpoint invalidation is documented in the PR (schema extends → fresh retrain; graft-from-predecessor is the warm-start path; note VecNormalize obs-stats do not carry to the wider obs).

## Potential Pitfalls & Open Questions
- **Risk (training stability)** — a thrust ceiling that sags meaningfully mid-episode is a non-stationary control plant; combined with UC-08's ceiling randomization it's two multiplicative perturbations. Mitigate with a **near-empty knee** + **slow default drain** so most episodes finish with margin; both tunable constants.
- **Risk** — the `hunger` population is **approximate** (curated `subclass` name-list, flagged non-authoritative in `modality.py`/UC-13) and `select_modality` fails loud (`ModalityAbsentError`) if absent from the active slice. UC-14 (default → full connectome) likely provides it; the analyst must confirm slice coverage or the graft is unbuildable.
- **Open** — depletion soft vs hard; `ceiling_factor(battery)` shape (linear vs knee).
- **Edge case** — warm-up latency buffer applies a throttle; decide whether warm-up steps drain the battery for a consistent ledger.
- **Ordering** — the schema predecessor is whatever merged last on the chain (`obstacle_vision_v2` from UC-15); resolve `extends()` base + version at implementation against mainline.
- **Risk** — checkpoint invalidation (accepted, as UC-13/15).

## Original Description
Follows the user's observation that battery drain and its impact on thrust are not simulated today ("if not, that's another UC"). Second of the 4-UC line: battery **physics + observation** are split ahead of the recharge pads (UC-18) because a finite energy budget is valuable and independently testable, and building the physics first de-risks the training-stability question before pads are added.

## Clarifications / Decisions
- Q: Split battery physics from recharge pads?
  A: Yes — UC-17 is battery drain + thrust impact + battery obs block (no pads); recharge pads are UC-18.
- Q: Which population does the battery sense bind to?
  A: The approximate `hunger` population via the UC-13 graft path (flagged non-authoritative; verify slice coverage).
- Q: Reward incentive for managing battery?
  A: Handled at the pad level (UC-18): some courses require recharge, others don't. UC-17 itself only requires the default course be completable without recharge.
