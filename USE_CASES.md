# Use Cases

Status ledger for use cases under `use-cases/`. Machine-maintained — the `define-use-case` skill appends rows; the dev-team orchestrator updates the `Status` and `Updated` columns as it works. Do not hand-edit those two columns unless you know why; edit the use-case file or re-run the skill instead.

Statuses:
- `pending` — saved but not yet picked up by the dev-team
- `in-progress` — the dev-team has started analysis
- `done` — implementation and tests completed
- `blocked` — the dev-team escalated (6-round cap hit, user abort, or infeasibility)

| # | File | Title | Status | Updated |
|---|------|-------|--------|---------|
| 01 | [use-cases/01-connectome-plumbing-poc.md](use-cases/01-connectome-plumbing-poc.md) | Connectome plumbing POC | done | 2026-09-16 |
| 02 | [use-cases/02-harden-connectome-substrate.md](use-cases/02-harden-connectome-substrate.md) | Harden the connectome substrate for RL | done | 2026-09-16 |
| 03 | [use-cases/03-start-gate-finish-flight.md](use-cases/03-start-gate-finish-flight.md) | Start→gate→finish flight training | done | 2026-09-16 |
| 04 | [use-cases/04-connectome-subgraph-pruning.md](use-cases/04-connectome-subgraph-pruning.md) | Function-targeted connectome subgraph pruning | done | 2026-09-17 |
| 05 | [use-cases/05-activation-record-playback.md](use-cases/05-activation-record-playback.md) | Neuron-activation recording + playback visualization | done | 2026-09-17 |
| 06 | [use-cases/06-3d-flight-viewer-neuron-beat.md](use-cases/06-3d-flight-viewer-neuron-beat.md) | 3D flight view + neuron "beat" in the playback viewer | done | 2026-09-17 |
| 07 | [use-cases/07-post-training-activation-pruning.md](use-cases/07-post-training-activation-pruning.md) | Post-training activation pruning → minimal functional flight circuit | done | 2026-09-17 |
| 08 | [use-cases/08-domain-randomization.md](use-cases/08-domain-randomization.md) | Domain randomization — per-episode course + optional dynamics | done | 2026-09-17 |
| 09 | [use-cases/09-configurable-gate-count.md](use-cases/09-configurable-gate-count.md) | Configurable number of gates (N free-3D waypoints) | done | 2026-09-17 |
| 10 | [use-cases/10-clean-training-data.md](use-cases/10-clean-training-data.md) | Clean all training data & models (start-from-scratch) | done | 2026-09-18 |
| 11 | [use-cases/11-yaml-config-and-run-layout.md](use-cases/11-yaml-config-and-run-layout.md) | YAML --config for the CLIs + per-run output layout | done | 2026-09-18 |
| 12 | [use-cases/12-mri-brain-heatmap.md](use-cases/12-mri-brain-heatmap.md) | MRI-style full-brain activation heatmap | done | 2026-09-18 |
| 13 | [use-cases/13-modality-mapped-populations.md](use-cases/13-modality-mapped-populations.md) | Modality-mapped neuron populations + graft-ready observation schema | done | 2026-09-18 |
| 14 | [use-cases/14-default-prune-full-connectome.md](use-cases/14-default-prune-full-connectome.md) | Default prune to the full auto-downloaded MaleCNS connectome | done | 2026-09-18 |
| 15 | [use-cases/15-obstacles-vision-avoidance.md](use-cases/15-obstacles-vision-avoidance.md) | Obstacles in the course + vision sense (detection & avoidance) | done | 2026-09-18 |
| 16 | [use-cases/16-landing-takeoff-pad-docking.md](use-cases/16-landing-takeoff-pad-docking.md) | Controlled landing + takeoff (pad docking) foundation | in-progress | 2026-09-18 |
| 17 | [use-cases/17-battery-drain-thrust-sense.md](use-cases/17-battery-drain-thrust-sense.md) | Battery drain + thrust impact + battery ("hunger") observation block | pending | 2026-09-18 |
| 18 | [use-cases/18-recharge-pads.md](use-cases/18-recharge-pads.md) | Recharge pads (dock-to-recharge, energy-constrained course variation) | pending | 2026-09-18 |
| 19 | [use-cases/19-damage-repair-pads.md](use-cases/19-damage-repair-pads.md) | Damage/integrity + repair pads + proprioceptive(damage) observation block | pending | 2026-09-18 |
