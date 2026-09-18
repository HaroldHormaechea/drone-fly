# Use Case 15: Obstacles in the course + vision sense (detection & avoidance)

## Summary
Add cylindrical **obstacles** (pillars: center + radius + height) to the course, placeable manually and by the randomizer, and give the drone a biologically-bound **vision** sense of them so it learns to avoid them. Each step the env detects drone↔pillar collision and applies a **severe negative penalty without terminating the episode** — the drone may recover aerially and continue flying; only the existing floor/ceiling/out-of-bounds crashes end an episode. The observation gains a new **vision block encoding the nearest-3 obstacles as egocentric relative vectors (zero-padded)**, bound to the `visual_projection` population via UC-13's `obs_schema` + `graft_actor` path — a new versioned schema `obstacle_vision_v2` that extends `migrated_v1`. Because this extends/re-binds the input→sensory wiring, **existing checkpoints are invalidated (a retrain under the new schema is required)**, exactly as in UC-13. The UC-08 randomizer's solvability guard is extended so a collision-free path start→gates→finish always exists and no gate/finish sits inside a pillar. The recorder stamps obstacles into `meta.course` and the dependency-free 3D viewer draws them. This is the first of three sequential course-element UCs; recharge/battery (UC-16) and repair/damage (UC-17) are out of scope here. Collision runs on the hermetic numpy `SimpleDroneAdapter` so CI stays offline.

## Acceptance Criteria
1. `CourseConfig` supports cylindrical obstacles (position + radius + height); they can be placed manually (a fixed default set) and by the randomizer.
2. The env detects drone↔pillar collision each step and, on contact, applies a **severe negative penalty but does NOT terminate** the episode (aerial recovery allowed); only floor/ceiling/out-of-bounds crashes terminate. Documented + tested.
3. The randomizer's solvability guard guarantees ≥1 collision-free path from start through all gates to finish, and never places a gate or the finish inside a pillar; degenerate draws are rejected/resampled then fall back to a collision-free-by-construction course (extends the UC-08 guard).
4. The observation gains a vision block bound to `visual_projection` via a new versioned schema (`obstacle_vision_v2`, extending `migrated_v1`); the block width and the nearest-3 egocentric encoding are documented.
5. Obstacle encoding is deterministic and egocentric (nearest-3 relative vectors, zero-padded when fewer than 3 obstacles); a test asserts the block's contents for a known obstacle layout, including the zero-pad case.
6. Graft parity: a `migrated_v1`-trained checkpoint grafts to `obstacle_vision_v2` with the obstacle block zero-initialized → identical actions until fine-tuned (reuses the UC-13 graft path); the checkpoint invalidation for a fresh v2 run is documented.
7. The recorder stamps obstacles into `meta.course` (additive/back-compatible), and the 3D flight viewer draws the cylinders; recordings without the field degrade gracefully.
8. Hermetic CI: collision detection, the obstacle env, and the schema/graft all have offline tests on the numpy adapter; a smoke-train under `obstacle_vision_v2` completes a finite update.
9. The "severe" penalty magnitude is a documented, tunable constant; a glancing contact penalizes but still allows the drone to complete the course (tested: contact does not end the episode).

## Potential Pitfalls & Open Questions
- **Open** — penalty accounting: whether the severe penalty is applied **per-step while overlapping** a pillar or **once per contact event**. Grazing a pillar must not silently stack an unbounded per-frame penalty that destabilizes training; the analyst pins this down (e.g. once-per-contact, or a capped per-step penalty) and documents it.
- **Assumption** — cylinders are floor-anchored (base at `floor_z`, extending up to `height`); collision = horizontal distance from the pillar axis `< radius` AND drone `z` within `[floor_z, floor_z + height]`. The analyst confirms the exact collision model against `RaceEnv`.
- **Assumption** — obstacles are **static** for this UC (no per-step motion); moving obstacles are a later UC.
- **Risk** — checkpoint invalidation: extending the observation schema re-binds input wiring, so pre-UC-15 checkpoints do not carry over (retrain required). Accepted, as in UC-13; must be surfaced in the PR.
- **Edge case** — nearest-3 with a course that has >3 pillars: only the closest 3 are visible to the policy — an intentional partial-observability choice; the analyst notes it. K (=3) is a documented constant.
- **Assumption** — collision uses the numpy `SimpleDroneAdapter` point position vs. the pillar volume (hermetic); real PyBullet rigid-body contact is deferred to the owner's dev-time sim, consistent with prior UCs.

## Original Description
Now, we should make our track creator be able to put pads to recharge, to repair, and then obstacles, shouldn't we?

(Scoped: this use case, UC-15, is the **obstacles** slice only, with full modality integration — a vision block bound to the visual_projection population. Recharge pads + battery are UC-16; repair pads + damage are UC-17.)

Collision refinement (verbatim): "It should penalyze but it should be able to reroute it it didn't crash with the floor - e.g. allow for aerial recovery. Penalization should be SEVERE though."

## Clarifications
- Q: What obstacle geometry should the course use?
  A: Cylinders (pillars) — center + radius + height.
- Q: What happens when the drone hits an obstacle?
  A: Severe penalty but NON-terminating — the drone can reroute / recover aerially as long as it didn't crash into the floor. Penalization should be SEVERE.
- Q: How should obstacles be encoded into the vision block?
  A: Nearest-K relative vectors (K≈3), egocentric, zero-padded.
- Q: Obstacle count/placement and motion?
  A: Static; a fixed default set plus a randomizable count/positions range under the UC-08 randomize axis (solvability-guarded).
- Q: Should the playback viewer draw obstacles now? (folded into the recap with a default)
  A: Default accepted — stamp obstacles into `meta.course` and draw the cylinders in the 3D flight view.
