# 14 — grounded-launch (UC-89, EXPERIMENTAL — for live diagnostic, not a validated ship)

The 98-dim deploy-robust policy retrained with the UC-89 grounded-launch knobs on, so it learns to
**lift off from a grounded start toward a gate dead-ahead** — the live failure mode the 768 re-fly
exposed (the drone held full roll on the pad and crashed into the start chute within ~2.4 s).

This is the **v1** grounded-launch retrain (`uc89-grounded-launch`, `ground_gate_ahead_frac 0.8`,
`spawn_distance_m [2.5,4.0]`, `pre_acquire_s 0.3`, warm-started from model 11, 16 M steps). It is
shipped here **for a live diagnostic re-fly only** — it is NOT a validated replacement for
`training/13-deploy-robust/` (see the trade-off below). Weights file:
`model_readout.pt` (sha256 `50e53335…`), obs-contract v2, action-contract v2, extra_dim 98. The
sidecar carries `action_filter: {traversal_time_s: 0.5}` so the harness applies the same 0.5 s slew
cap as model 13.

## Why this exists

After UC-87/88 (smoothness) and the detector width fix (512→768, which made the gate visible from the
pad), live flights still crashed in the start chute: with a real gate dead-ahead from a grounded
start, the policy answered with a full roll. UC-89 (PR #106) added the frame-correct no-detection
neutral and the grounded-launch training knobs; this checkpoint is the retrain that exercises them.

## Sim results (128-ep deterministic eval, A3000)

| checkpoint | grounded-launch completion¹ | airborne completion² | cruise mean\|Δ\|² |
|---|---|---|---|
| `13-deploy-robust` (model 13) | 2.3 % | **28.9 %** | **0.74** |
| **this (uc89-grounded-launch v1)** | **13.3 %** | 18.0 % | 1.27 |

¹ all-grounded, gate-0 dead-ahead at ~3.2 training-scale, 0.5 s cap (`/tmp/uc89-grounded-eval.yaml`).
² standard live-like preset (`uc87-deploy-robust-warmstart`).

So it **fixes the grounded launch (~6×)** but at the cost of airborne completion and cruise
smoothness — lowering the knob fraction (v2, `frac 0.4`) collapsed the policy entirely, so the
knob is non-monotonic and a cleaner recipe (a dedicated launch-phase reward) is the likely follow-up.
The purpose of this deliverable is to answer the one question sim cannot: **does the launch fix
actually clear the start-chute crash live, and is the rougher cruise acceptable under the 0.5 s cap?**

## Live re-fly (Liftoff box)

Fly on Infinity Loop at detector **input width 768**, `align_spawn_yaw: true`, latency compensation
off. Verify gate completion with `completeness_from_pose` (not the overflight counter). The key
observations: (1) does it leave the start chute / climb and head to gate-0 instead of rolling into
the wall? (2) completion vs. model 13? (3) is the cruise acceptable with the 0.5 s cap, or too rough?
