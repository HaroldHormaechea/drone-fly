# Use Case 18: Recharge pads (dock-to-recharge, energy-constrained course variation)

## Summary
Add **recharge pads** that refill the battery while the drone is **fully landed** (docked) on them — building on the pad-docking foundation (UC-16) and the battery model (UC-17). A recharge pad is a docking pad tagged `recharge`; while the drone is docked on one, `battery` rises toward `1.0` at a documented rate (or snaps to full on dock — pin one), clamped at 1. Crucially, recharging **requires a full landing** — hovering over the pad does not recharge. To make pads meaningful despite the racing objective (which otherwise never rewards a detour), recharge is made a **course-variation axis** (per the user): **some courses are energy-constrained** — the finish is unreachable on a single charge, so the agent *must* land and recharge to complete — **while others are completable on one charge**, adding variation. The randomizer's solvability guard guarantees that any energy-constrained course has a reachable recharge pad within the energy budget. This is an **env-only** change (no new observation block — it reuses UC-17's battery block), so it does **not** invalidate checkpoints. Off by default; hermetic on the numpy adapter.

## Acceptance Criteria
1. A pad tagged `recharge` refills `battery` (toward `1.0` at a documented rate, or instantly on dock) **only while the drone is fully landed/docked on it** (the UC-16 docked state) — not while airborne, not on a non-recharge pad; charge clamps at `1.0`.
2. **Full landing required:** hovering over a recharge pad does **not** recharge (tested — recharge accrues only in the docked state).
3. **Course variation:** recharge-necessity is a documented axis — some courses are energy-constrained (finish provably unreachable on one charge → a recharge is required to complete), others are completable without recharging. Both cases are exercised.
4. **Solvability:** the guard guarantees every energy-constrained course has ≥1 recharge pad reachable within the energy budget (so it is completable *with* recharging); non-constrained courses stay completable without one. Tested (extends the UC-08 solvability machinery).
5. Off-by-default byte-identity (no recharge pads → no behaviour/RNG change); disabled axis draws nothing.
6. Hermetic (numpy adapter, no pybullet/network): a scripted "drain → land on recharge pad → refill → take off → finish" trajectory **completes a course that is unreachable on a single charge**, and the same course fails to complete without the recharge — demonstrating the pad is load-bearing, not scenery.

## Potential Pitfalls & Open Questions
- **Edge case** — while docked, drain must be net-negative (recharge rate > any docked drain) or the pad never fills; define docked-step energy accounting.
- **Open** — how the solvability guard *estimates* "reachable within the energy budget" (a conservative energy model over the reference polyline, mirroring UC-08's geometric guard); over-conservative is acceptable given the fallback.
- **Edge case** — recharge dwell consumes the episode step budget (UC-16 pitfall); an energy-constrained course may need a budget allowance so a legitimate recharge detour can still finish in time. The analyst pins this.
- **Assumption** — no new observation block; the policy senses charge via UC-17's battery block already present, so no checkpoint invalidation here.
- **Dependency** — requires UC-16 (docked state) and UC-17 (battery) merged.

## Original Description
From the user's track-creator request ("pads to recharge") plus the confirmed constraint that recharge requires a full landing, and the reward-gate decision that "some courses should require recharge, others should not, to add variation." Third of the 4-UC line.

## Clarifications / Decisions
- Q: Does recharging require a full landing?
  A: Yes — recharge accrues only while fully landed/docked; hovering is insufficient.
- Q: How is recharge made worth doing despite the time-pressured racing reward?
  A: **Course-variation approach** — some courses are energy-constrained so completion itself requires a recharge (the completion bonus pays for the detour); others are completable on one charge. No new shaping reward term (avoids reward-hacking).
- Q: Does this add an observation block?
  A: No — reuses UC-17's battery block; env-only change, no checkpoint invalidation.
