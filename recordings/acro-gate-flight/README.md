# Acro gate-flight — training-session recordings

Ten neuron-activation recordings sampled across a single training run (`PIHEAD_climb2`), showing the
connectome-seeded policy learning to fly the full 3-gate course in **acro (rate-mode) flight** — from
not completing the course to **deterministic 100% completion**.

Each `step_<N>/episode_0.json` is one deterministic episode flown by the checkpoint at training step
`N`, captured by the `ActivationRecorder` at the real **50 Hz** control rate. See
[`docs/TRAINING_CURRICULUM.md`](../../docs/TRAINING_CURRICULUM.md) for the full recipe and the
measurement note that made deterministic flight reproducible.

## Progression (deterministic, 50 Hz, wide straight bootstrap course)

| Training step | Completed course | Flight time | Episode reward | Frames |
|---:|:---:|---:|---:|---:|
| 149,976   | ✗ | — | 8.4   | 113 |
| 499,920   | ✗ | — | 9.2   | 133 |
| 999,840   | ✗ | — | 18.4  | 126 |
| 1,499,760 | ✗ | — | 9.5   | 147 |
| 1,999,680 | ✗ | — | 2.5   | 145 |
| 2,499,600 | ✗ | — | 7.6   | 127 |
| 2,999,520 | ✗ | — | 8.2   | 146 |
| **3,474,444** | **✓** | **1.78 s** | **225.3** | 89 |
| **3,549,432** | **✓** | **1.64 s** | **225.4** | 82 |
| **3,999,360** | **✓** | **1.86 s** | **224.9** | 93 |

The jump in episode reward (~8 → ~225) is the one-off `completion_bonus` (200) landing once the policy
flies all three gates and crosses the finish. Frame count drops (~130 → ~85) as flailing-until-timeout
becomes an efficient ~1.7 s traversal. The converged checkpoints complete **100% deterministically**
(independently confirmed by `drone-fly evaluate`: `completion_rate=100.0%`, MASTERY @ 80%), and
robustly — 50/50 completions across five random seeds.

## File format (`schema_version: 1`)

```
{
  "schema_version": 1,
  "meta":   { "neuron_ids", "superclass", "roles", "modality", "positions",
              "episode_index", "seed", "checkpoint" },
  "frames": { "activations"    : [per-frame × 247-neuron activation vectors],
              "actions"        : [per-frame collective-thrust + body-rate commands],
              "drone_position" : [per-frame xyz],
              "target_gate"    : [per-frame index of the gate currently being homed on] },
  "outcome": { "completed", "completion_time", "total_reward", "steps" }
}
```

`activations` is the load-bearing signal: the per-frame firing of each of the 247 connectome neurons
while the actor flies, aligned by `meta.neuron_ids` / `meta.roles` / `meta.positions` so activity can
be mapped back onto connectome anatomy.

## Reproduce / record more

```
DRONE_FLY_OBS_GRAVITY=1 drone-fly evaluate --config <cfg>
```
where `<cfg>` points `checkpoint` + `vecnormalize` at a `training/PIHEAD_climb2/checkpoints/` step,
sets `record: true` + `record_dir`, `adapter: pybullet`, `connectome: tests/fixtures`, `prune: true`,
`prune_k: 2`, and the run's reward knobs (`altitude_weight 0.4`, `hover_stability_weight 0.0`,
`upright_weight 0.15`, `spin_stability_weight 0.1`, `progress_weight 2.5`). The evaluator runs at 50 Hz
by default — the rate the policy was trained at.
