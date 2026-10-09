# Pad lap — the K1 connectome managing a battery + charging pad

Ten deterministic recordings of the real K1 connectome (25,627 neurons, ~3.86M synapses, real MaleCNS
weights) flying the oval lap **under a battery constraint**: thrust drains the battery, and a **charging
pad** off the flight path recharges it on fly-over. **99% deterministic completion** (127/128), gates
6/6 — the drone flies the lap, manages its charge, and diverts to the pad to recharge.

## The task

- **Battery** drains with thrust (`DRAIN`/s · thrust fraction); if it hits zero, thrust collapses
  (`EMPTY_THRUST`) and the drone can't hold altitude. A full lap can't be flown on the starting charge,
  so the drone must top up.
- **Charging pad** at a fixed position off the oval; flying within its radius recharges the battery.
- The readout sees battery level + relative pad position + an over-pad flag (fed as extra senses; the
  brain stays frozen on the 12-dim flight obs).

Structurally this *avoids the hover trap that defeated obstacles*: hovering drains the battery too, so
"do nothing" leads to death — flying is the only viable policy.

## Flight style (honest caveat)

This policy completes 99% of laps but flies **aggressively**: mean tilt ~46°, max ~67° (it never
inverts, so it is controlled, but it is not the clean upright flight of the oval ~25° or track ~12°).
The battery pressure incentivizes **speed** — laps here take ~3.2 s vs the oval's ~6 s — so the drone
banks hard and rushes, topping up at the pad as needed. A calmer-flight variant would need reward
tuning; note a direct tilt penalty is NOT an option (penalizing tilt collapses acro flight to a
stationary hover, since tilting is the only way to translate).

## Architecture

Augmented frozen-K1 reservoir (`reservoir_aug.py`), warm-started from `oval-lap-flight`, exploration
annealing. Trainer `train_aug_course.py pads`; env `gates_pads.py`. `model_readout.pt` (~610 KB) holds
the trainable readout + the frozen random `input_projection`; the sparse layer rebuilds from
`artifacts/pruned/k1/`. Reload verified at 98%.

Each `episode_<n>.json` stores all 25,627 neuron activations per frame + action + position + target
gate + anatomical metadata + the charging pad (drawn in-app), with a per-episode visibility gain on the
activations (pattern real).
