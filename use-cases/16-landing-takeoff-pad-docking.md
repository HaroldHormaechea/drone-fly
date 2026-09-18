# Use Case 16: Controlled landing + takeoff (pad docking) foundation

## Summary
Rework floor contact so a **controlled full landing** on a designated pad is a valid **docked** state rather than a crash, and let the drone dwell docked and then take off again — while a hard, fast, off-pad, or over-tilted floor contact stays a terminating crash exactly as today. This is the shared foundation the recharge-pad (UC-18) and repair-pad (UC-19) use cases build on; it delivers **only** the land/dwell/takeoff mechanic, a generic floor-anchored `PadSpec` course element (center + horizontal radius, analogous to `GateSpec`), docking thresholds (max descent speed, max tilt, over-pad tolerance) in config, and a docked-state flag surfaced via `info` — **no batteries, no damage, and deliberately no observation-schema change** (so it does **not** invalidate checkpoints). A **full landing is required** — settled on the pad, low descent speed, roughly upright; hovering over a pad is not docking. Pads are **manual-only** this UC (no randomizer change, so the RNG stream is untouched). Off by default (no pads → byte-identical to UC-01..15). Fully hermetic on the numpy `SimpleDroneAdapter`.

## Acceptance Criteria
1. `CourseConfig` gains a floor-anchored `PadSpec` (center + horizontal radius) collection, placeable as a fixed manual default set; the default course has **zero pads** so `EnvConfig()`/`CourseConfig()` stay byte-identical to UC-15.
2. A floor contact that is (a) horizontally within a pad radius, (b) descent speed `|vz| <= max_dock_descent_speed`, and (c) roughly upright (`|roll|,|pitch| <= max_dock_tilt`) sets a `docked` state and does **not** terminate; the episode continues. Independently testable on hand-driven adapter states.
3. A floor contact failing **any** docking condition — too fast, too tilted, or off-pad — still terminates (crash) with the existing `collision_penalty`; each failing condition tested in isolation.
4. **Dwell:** a docked drone stays alive across ≥1 (and several) docked steps without crashing or completing; `docked` persists across them.
5. **Takeoff:** throttling up so the drone leaves the pad (`z > floor_z` after a step) clears `docked` and returns to normal flight; a subsequent controlled re-touchdown re-docks (dock↔fly repeatable within one episode).
6. The docked state is surfaced via `info` **only** — the observation vector is **byte-identical** (no obs-schema block, no graft, **no checkpoint invalidation** in this UC). Downstream UCs that need the policy to react to being docked add the block.
7. Descent-speed classification is derived without breaking the adapter contract (infer from `prev→curr z` across the step, or a documented adapter contact signal); docking applies to **floor** contact only — ceiling contact never docks.
8. **Off-by-default byte-identity:** with no pads, a fixed-seed episode is bit-identical to the pre-UC-16 baseline (same observation, reward stream, termination step, and RNG consumption — no `np_random` draw for pads).
9. Hermetic on the numpy adapter: a scripted dock → dwell → takeoff trajectory is tested offline (no pybullet, no network); a smoke run completes a finite docking trajectory. (Docking is validated by scripted trajectories, not learned behaviour — see pitfalls.)

## Potential Pitfalls & Open Questions
- **Assumption / load-bearing decision** — the numpy adapter clamps `z=floor_z` and zeroes `vz` on contact, so the env cannot read impact velocity from the returned state; recover it by inferring `(prev_pos.z − curr_pos.z)/dt` across the step (lower-risk, no adapter-contract change) rather than extending `DroneState`. The analyst pins this.
- **Edge case** — `DroneState.collided` is also set on ceiling contact; the re-classification must re-derive floor-vs-ceiling and only ever dock on floor.
- **Edge case** — a pad sits on the floor, so "over a pad" and "floor contact" coincide; define the rule for lateral drift off a pad edge while docked (re-crash vs benign).
- **Risk (reward)** — docked steps still cost `time_penalty` and earn nothing in this foundation UC (no dock reward). That's intended (no incentive to dock yet, so racing behaviour is unperturbed) — but it means docking is only *validated by scripted trajectories*, not learned; an RL smoke check proves the mechanic doesn't break training, not that a policy learns to dock.
- **Edge case** — dwell consumes the episode step budget; a long dwell can force truncation. Acceptable here (short scripted dwell); a recharge/repair UC needing long dwells may need budget relief (out of scope).
- **Assumption** — thresholds (`max_dock_descent_speed`, `max_dock_tilt`, pad radius) are documented tunable constants.
- **Assumption** — pads are manual-only this UC; randomized pads are deferred to keep the RNG stream byte-identical.

## Original Description
From the user, building the track/course creator: recharge and repair "may require landing and taking off," and later confirmed "it will require a full landing." A controlled landing/takeoff (pad docking) mechanic is therefore the shared foundation both the recharge-pad and repair-pad use cases depend on. Scoped (per the assessment of three drafted candidate UCs) as the first of a 4-UC line: **16 landing/dock → 17 battery+thrust+obs → 18 recharge pads → 19 damage+repair**.

## Clarifications / Decisions
- Q: Must recharge/repair require a full landing, or does hovering over the pad suffice?
  A: **Full landing required** — controlled touchdown + dwell + takeoff; hovering is not sufficient. Docked = settled on pad, low descent speed, roughly upright.
- Q: Does the docked state need to be in the observation now?
  A: **No** — surface via `info` only in this foundation UC, so it does not change the schema or invalidate checkpoints. The UCs that need the policy to react add the block.
- Q: How is this decomposed?
  A: **4-UC line**; this is UC-16, the foundation, built after UC-15 merges (it shares the new-course-element + recorder/viewer `meta.course` stamping plumbing).
