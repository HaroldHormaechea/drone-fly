# Use Case 35: Course randomization placement — pads off waypoints, obstacles between waypoints

## Summary
When per-episode course randomization is enabled (UC-08/15/18/19/24 lineage), two placement problems make the randomized course wrong: **(1)** repair/recharge pads are placed **under / overlapping waypoints (gates)** — coinciding with the point the drone already flies through, so docking is trivial/degenerate; pads should be positioned away from every waypoint. **(2)** obstacles are scattered without regard to the flight path, so they frequently sit off-route where the drone never has to deal with them; obstacles should be placed **between consecutive waypoints**, within a corridor around the straight-line segment connecting them, so the drone is **forced to evade** them. Both fixes live in the env course-randomization / placement code. The placement must stay **deterministic per seed** and must keep the course **feasible** (evadable — never an unavoidable collision), and must not change the observation schema, reward, or dynamics beyond the placement geometry itself.

## Acceptance Criteria
### Pads off waypoints
1. With randomization on, no recharge or repair pad is placed within a configurable minimum distance `R_pad` of any waypoint/gate center; a test asserts this constraint holds for generated pad positions across many seeds.
2. Pad placement remains valid otherwise (still within course bounds, still respects the existing recharge/repair feature toggles and the "exactly one recharge + one repair pad" rule where applicable).

### Obstacles between waypoints (forced evasion)
3. Each placed obstacle lies **between two consecutive waypoints**: within the along-segment span of some waypoint→waypoint segment (including start→first-gate) AND within a configurable perpendicular corridor of that segment — i.e. near the nominal path — so it threatens the route. A test asserts every generated obstacle maps to an inter-waypoint segment within the corridor across many seeds.
4. The course stays **feasible/evadable**: obstacles leave clearance (≥ drone radius + margin) so the drone can pass; obstacles do not fully block a corridor and do not overlap waypoints or other obstacles beyond a threshold. A test checks the minimum-clearance invariant.

### Cross-cutting
5. **Determinism**: same seed → identical pad and obstacle placement (reproducible), preserving the existing seeding contract.
6. Non-randomized behavior is unchanged; the `randomize_obstacles` / `randomize_recharge_pads` / `randomize_repair_pads` toggles still gate whether each feature is placed.
7. No change to the observation schema, reward function, connectome, or dynamics beyond the placement geometry; existing configs that don't randomize reproduce bit-for-bit.
8. The radii/corridor parameters are sane defaults and configurable, and are documented.

## Potential Pitfalls & Open Questions
- **Ambiguity** — exact `R_pad`, obstacle corridor width, and clearance margin are implementation choices; they must be sane, configurable, and documented (AC-8). The challenger should confirm the chosen values keep courses solvable.
- **Risk** — forcing obstacles onto the path can make a course **infeasible** (unavoidable collision); AC-4's clearance invariant is the guardrail and must be enforced, not just hoped for.
- **Edge case** — very short segments between close waypoints: corridor placement must handle them (skip, or place proportionally) without stacking obstacles on a gate.
- **Edge case** — pad-exclusion vs obstacle-inclusion can conflict (a pad pushed off a waypoint could land in an obstacle corridor, or vice-versa); define a clear placement order/priority and a re-sample cap with a sane fallback.
- **Assumption** — "between waypoints" means along the ordered waypoint sequence used for the course, including the start→first-gate segment.

## Original Description
User, reviewing randomized courses on 2026-09-20:
- "When randomizing positions of repair/recharge pads, they seem to always be under a waypoint. They shouldn't be in a waypoint."
- "Obstacles should also be placed in-between waypoints, so the drone is FORCED to evade."
