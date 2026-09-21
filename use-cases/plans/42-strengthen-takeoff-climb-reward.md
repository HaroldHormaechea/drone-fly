---
plan_for: use-cases/42-strengthen-takeoff-climb-reward.md
work_branch: feat/uc-42-strengthen-takeoff-climb-reward
team: drone-fly-uc-42
approved: 2026-09-21
---

# UC-42 — FINAL APPROVED proposal (analyst↔challenger converged, challenger APPROVED round 2)

TARGET_DIR `/workspace/drone-fly-uc-42-strengthen-takeoff-climb-reward`. All reward math verified against the pure `compute_reward`.

## Diagnosis (AC1 — evidence-backed; all three levers adjudicated)
Reward assembly `reward.py:100-117`; defaults `env/config.py:558-611`; airborne flag binary at `z>floor_z+floor_epsilon(0.05)` `racing_env.py:384`.
1. **Climb/airborne MAGNITUDE — PRIMARY, confirmed.** The only durable altitude reward is `airborne_bonus`; net of `time_penalty` and the discounted climb-potential **leak** `(γ−1)·w·target=−0.02/step`, sustained hold nets only **+0.03/step** vs floor-rest −0.05 (+0.08 differential, +10 over the ~101-step stall window) — *exactly the UC-38 margin that already failed*. Right sign, insufficient magnitude. Worse, with the flat bonus net-hold(h)=0.05−0.02h is **decreasing in h** (perverse: holding higher is slightly worse).
2. **ent_coef/exploration — RULED OUT as primary.** Fresh run: exploration alive (47% steps >0.6, no K0/freeze/std-collapse — UC-40/41 hold). Kept unchanged (0.005, >0).
3. **Thrust-authority — RULED OUT on HARD EVIDENCE** (orchestrator pybullet probe: TWR=2.25, throttle 0.9 → z 0.021→2.657 m in 14 steps). Sustained above-hover lifts trivially; the blocker is purely that the policy isn't reinforced into sustained above-hover.

## Fix (AC2/3/4 — minimal, single lever set; all invariants preserved)
1. **`airborne_bonus` 0.1 → 0.2** and **make it altitude-graded** (`env/reward.py`): airborne payout = `airborne_bonus·min(max(height_above_floor_curr,0),climb_target_height)/climb_target_height` while airborne, else 0. This flips the perverse gradient into a **monotone "climb to target" pull**: net-hold(h)=0.18h−0.05 (break-even h≈0.278, **+0.13/step at target** = 4.3× the old +0.03), flat above target ⇒ no ceiling-seeking. No signature change (h already threaded); **NO `racing_env.py` change**.
2. **`completion_bonus` 100 → 200** (`env/config.py`) strictly as **forced bound-preservation**: with AB=0.2 the per-episode airborne max over the 800-step budget is 160, +climb_bound 1.98 = 161.98 > 100 → the default-budget loiter<completion invariant breaks unless completion rises. 200 restores it tight (161.98<200, headroom 1.23 ≈ original 1.22). NOT a takeoff signal, and kept tight because `norm_reward=True` makes a bigger completion spike counterproductive (inflates the return-std VecNormalize divides by).
3. **UNCHANGED:** climb_weight (2.0), climb_gamma (0.99=γ), climb_target (1.0), time_penalty, progress_weight, collision_penalty, gate_bonus, obstacle_penalty, ent_coef. No crash-cliff/curriculum touch, no hover-bias touch.
4. **`README.md`:** reward table (airborne row 0.1→0.2 + grading note; completed row 100→200; climb row unchanged), re-derive UC-37/39 bound prose, add UC-42 subsection.

**Invariants preserved (verified):** telescoping climb term untouched (round-trip nets ≈0, non-farmable); graded survival ≈0 at rest (floor non-farmable, −0.05/step), capped at target, per-episode max still airborne_bonus·budget; `climb_bound=1.98 < completion(200)` AND `≤ gate_bonus/3 (3.333)`; `climb_gamma==γ`; penalty-free stuck-cut & anti-suicide intact.

## Validation (AC2/AC5/AC6) — reward-math test is the SOLE committed acceptance gate
CI stays hermetic on the numpy `simple` adapter. The pybullet probe is **orchestrator-run, non-committed, confirmation-only — NOT evidence-of-record**; acceptance gates solely on the deterministic reward-math test. Smoke-train = wires/finite/no-collapse only (never a takeoff claim).

## Files Affected
**Production (developer):** `src/drone_fly/env/config.py` (airborne 0.1→0.2, completion 100→200 + docstring bound re-derivations 80→160, 81.98→161.98), `src/drone_fly/env/reward.py` (graded airborne term + docstring; **contract change: airborne reward now height-graded, not bool-only**), `README.md`. NO change to `racing_env.py` or `train/config.py`.
**Test (qa):**
- NEW `tests/test_uc42_takeoff_reward.py` — gates on (a) durable net-hold differential (+0.03→+0.13/step, assert ≥~0.12 & > old +0.03), (b) monotonicity (hold-return strictly increasing in h), (c) explicit assert time_penalty/progress_weight/collision_penalty/gate_bonus UNCHANGED (proves relative restructuring, not global rescale under norm_reward). Absolute margin illustrative only. Plus invariant re-proofs.
- UPDATE `tests/test_reward.py`: `test_airborne_step_yields_the_bonus...` (thread h=target; graded amount), `test_net_per_airborne_step_is_strictly_positive` (net pin 0.05→0.15), `test_max_episode_survival_reward_is_below_completion_bonus` (80→160, 81.98→161.98, completion 100→200, keep combined<completion), `test_uc38_stuck_or_timeout_cut...` (thread real h), **and `test_uc39_hover_at_target_nets_above_sitting_on_floor:499` (assert 0.03→0.13, docstring `0.2−0.05−0.02`)** — the challenger's one Minor fixup.
- VERIFY green (relational, numeric comments shift): `tests/test_uc41_reward_gradient.py` (failed_takeoff>do_nothing + AC4 std-no-collapse), `test_uc39_hover_episode_beats_takeoff_then_immediate_crash` (AC8 anti-suicide), `test_env_contract.py:2478` (airborne>sit), `test_env_config.py` `test_uc39_climb_reward_defaults` (unchanged values). UPDATE `test_env_config.py` `test_uc37_..._airborne_bonus_defaults` (0.1→0.2).
- `test_reward_docs.py` (auto-keyed → satisfied by README table update), `test_bootstrap_docs.py`, `test_viz_contract.py` stay green. Full suite `uv run --extra dev pytest` (AC5).

## Risks
- **norm_reward=True** (`racing_env.py:812/888`): global rescale is a no-op → fix is relative/structural; completion kept tight to avoid inflating return-std once completions begin.
- **Airborne contract change** (bool→height-graded) touches the enumerated UC-37/38/39 asserts; behavior at h=target is backward-compatible (graded amount = airborne_bonus).
- No regression of prior fixes (crash-cliff/std-collapse/freeze). climb_weight ceiling (~3.37 for gate_bonus/3, ≤3 for hold-net-positive) recorded for any future change.

## Challenger's final verdict (round 2) — APPROVE
Load-bearing claims verified against source. Committed minimal lever set: airborne 0.1→0.2 + altitude-grading (core structural fix flipping the perverse gradient into a strictly-increasing climb-to-target pull), completion 100→200 as forced bound-preservation (tuned tight, headroom 1.23; not higher because reward VecNormalize is ON), climb_weight/ent_coef/collision-curriculum/hover-bias UNCHANGED. Validation trap guarded: deterministic reward-math test is the SOLE committed CI gate, gating on relative/structural properties (not an absolute margin — VecNormalize makes global rescales a no-op, the UC-38 trap); smoke-train wires-only; real-pybullet probe orchestrator-run/non-committed. All UC-37/38/39 invariants preserved; no UC-40/41 regression. One Minor fixup: `tests/test_reward.py:499` 0.03→0.13 + docstring re-derivation (caught by AC5 full suite; `test_env_contract.py:2478` relational, stays green).
