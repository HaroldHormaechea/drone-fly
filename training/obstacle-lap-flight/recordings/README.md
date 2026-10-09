# Obstacle lap — the K1 connectome weaving around pillars

Ten deterministic recordings of the real K1 connectome (25,627 neurons, ~3.86M synapses, real MaleCNS
weights) flying the oval lap while **avoiding pillar obstacles**. **99% deterministic completion** with
**genuine evasion** — verified, not just "completion":

| Metric (256-drone deterministic eval) | Result |
|---|---|
| Completion | **99%** |
| Never entered a pillar | **100%** |
| Min clearance to pillar border | 0.27 m |

Contact is a **non-terminal nudge** (a bump knocks the drone off course, recoverable — not a crash), so
"completion" alone wouldn't prove avoidance; the clearance/contact check confirms the drone actually
weaves *around* the pillars (hence the ~44° banking) rather than barreling through.

## How it was trained — a curriculum (the key)

Naive obstacle training collapses: with pillars on the flight path from step 0, forward motion looks
risky, so the policy learns to **hover** (and a tilt penalty is no help — penalizing tilt removes the
only way to translate in acro flight). The fix is a **difficulty curriculum**:

1. Warm-start from the proven `oval-lap-flight` flier (it already flies the loop).
2. Pillars **start at the loop centre** (off the flight path — ignored), then **slide onto the path**
   over the first half of training (`gates_obstacles.difficulty` 0→1, driven by `curriculum_frac`).
3. The policy meets the obstacle gradually and learns to deviate incrementally — never a sudden wall.

Obstacle vision (relative position to the nearest 2 pillars) is fed to the **readout**; the K1 body
stays frozen on the 12-dim flight obs. Trainer `train_aug_course.py obstacles`; env `gates_obstacles.py`.
`model_readout.pt` (~610 KB) reload-verified at 100% completion / 100% clear.

## Course

Oval lap (6 gates) + 2 pillars (radius 0.25) near fixed segments, offset to leave a clear lane. Pillars
are drawn in the app via `CourseConfig.obstacles`. Each `episode_<n>.json` stores all 25,627 neuron
activations per frame + action + position + target gate + anatomical metadata (visibility gain applied).
