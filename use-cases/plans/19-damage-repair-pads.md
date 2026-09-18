---
plan_for: use-cases/19-damage-repair-pads.md
work_branch: feat/uc-19-damage-repair-pads
team: drone-fly-uc-19
approved: 2026-09-18
---

# UC-19 — Damage/integrity + repair pads + proprioceptive(damage) observation block

Challenger-APPROVED (first pass, no revisions; 3 minor folds included). All paths absolute in `/workspace/drone-fly-uc-19-damage-repair-pads/`. Autonomous run; forks resolved against binding decisions + code.

## Approach (one-line)
Model **integrity** as an adapter-resident scalar mirroring UC-17's battery one-for-one: degrade `max_body_rate` (not thrust), accrue damage from the reused UC-15 edge-triggered obstacle-contact event, repair via the UC-18 recharge-on-dock idiom gated to a `repairable` pad, sense `1-integrity` through a new v4 schema block bound to `proprioceptive`. One `DamageConfig.enabled` switch gates all of it (damage bundled with repair). No changes to `modality.py`, `actor.py`, `docking.py`, or `graft_actor` — all reused as-is.

## Production changes (developer)
1. **`src/drone_fly/env/config.py`**
   - New frozen `DamageConfig` (mirror `BatteryConfig`): `enabled=False`; `damage_per_contact` (~0.34, tunable); `min_authority` (~1.0 rad/s, `>0`); `repair_rate` (~0.5/s); method `authority_factor(integrity)->float`, monotone non-decreasing, **exactly `1.0` at integrity==1.0** (linear `f=integrity`). Docstring MUST state the invariant `min_authority < BASE_MAX_BODY_RATE`.
   - `PadSpec.repairable: bool = False` appended **last** (after `rechargeable`).
   - `EnvConfig.damage: DamageConfig` appended **last** (after `battery`).
   - `EpisodeConfig.repair_step_allowance: int` (mirror `recharge_step_allowance`); added in reset() **only when ≥1 repairable pad present**.
   - `default_repair_course()` (damage-heavy: dense on-corridor obstacles forcing multiple contacts + one `repairable` pad) and `single_repair_pad_course(...)` minimal fixture.
2. **`src/drone_fly/adapter/base.py`** — `DroneState.integrity: float = 1.0` appended **last** (after `battery`); no-op hooks `damage(amount)->float` and `repair(delta)->float` returning `1.0` (mirror `recharge`).
3. **`src/drone_fly/adapter/simple.py`** — `damage=None` ctor kwarg → `_damage_cfg`/`_damage_enabled`/`_integrity=1.0`; reset integrity to 1.0; in `step()` the ONLY change is `rates = [...] * (max(min_authority, _max_body_rate*authority_factor(_integrity)) if _damage_enabled else _max_body_rate)` using **start-of-step** integrity; override `damage`/`repair` (clamp ≥0 / ≤1); `_state` passes `integrity`.
4. **`src/drone_fly/adapter/__init__.py`** — `make_adapter(damage=None)` forwarded to both backends (mirror `battery`).
5. **`src/drone_fly/env/racing_env.py`** — `_damage_enabled`; pass `damage=...` to `make_adapter`; `obs_dim += 1` after battery; `_observation` appends `[1.0 - state.integrity]` after battery block, before `nan_to_num`; in `step()` after the recharge block: **damage first** (`if _damage_enabled and obstacle_contact: state = replace(state, integrity=adapter.damage(damage_per_contact))`), **then repair** (`if _damage_enabled and docked: pad = pad_under(...); if pad and pad.repairable: state = replace(state, integrity=adapter.repair(repair_rate*dt))`) — both via `dataclasses.replace` BEFORE `_observation`; damage feeds neither `crash`/`terminated` nor reward; reset() adds `repair_step_allowance` per repairable pad.
6. **`src/drone_fly/controller/obs_schema.py`** — `DAMAGE_PROPRIOCEPTION_V4 = ObsSchema((*BATTERY_HUNGER_V3.blocks, ObsBlock("damage", 1, "proprioceptive")), version=4)` (total 26); add `"damage_proprioception_v4"` to `NAMED_SCHEMAS`. Predecessor = `battery_hunger_v3` (v3, width 25).
7. **`src/drone_fly/train/loop.py`** — `_reconcile_damage(env_config, obs_schema)` mirroring `_reconcile_battery` (forces `damage.enabled=True` when schema has a `"damage"` block); chain `_reconcile_damage(_reconcile_battery(_reconcile_obstacle_vision(...)))`. v4 carries the battery block too, so both must be forced on to reach width 26; the existing `env.obs_width == total_width` assertion validates.

## Test coverage (qa)
- `test_config.py`/`test_env_config.py` — DamageConfig defaults, `authority_factor` monotone + `f(1.0)==1.0` exact, **hard assert `min_authority < BASE_MAX_BODY_RATE`**, PadSpec.repairable & EnvConfig() byte-identity.
- `test_adapter.py` — integrity reset/decrement(clamp≥0)/repair(clamp≤1); fixed-action-sequence damaged vs pristine → smaller attitude/angular response; **integrity==1.0 enabled-path byte-identical to disabled (run hermetic, no dynamics randomization)**; thrust/mass/drag untouched.
- `test_docking.py`/`test_env_contract.py` — repair only while docked on a `repairable` pad; hover no repair; non-repair pad no repair; clamp at 1.0; damage edge-trigger (N contacts→N decrements, M-step continuous overlap→1); off-by-default full-step + obs byte-identity.
- `test_obs_schema.py` — v4 extends v3, total 26, NAMED_SCHEMAS/resolve.
- `test_graft.py` — graft v3→v4 zero-inits damage block (weight+bias), action parity via `torch.equal` (single+batched), negative control.
- `test_modality.py`/`test_actor.py` — two proprioceptive blocks coexist (additive index_add, non-empty select_modality; ModalityAbsentError on a pruned slice).
- `test_schema_train.py` — smoke_train under v4 on `default_repair_course()` finite; obs_width==26.
- `test_env_contract.py` — **AC7 course variation (highest-risk deliverable):** replicate UC-18's mechanism — a scripted hermetic numpy trajectory that (a) **fails degraded** on `EnvConfig(damage=DamageConfig(enabled=True,...), course=default_repair_course())` and (b) **completes after a pad dwell/repair**, proving the pad is load-bearing; PLUS a light course completable without repair. Construct the damage config explicitly (mechanics gated by `damage.enabled`, not by the course). If no tuning band demonstrating both sides can be found, that is an **escalation trigger**, not a silent weakening of the test.

## Risks / notes for the PR
- **Tuning band** (`min_authority`/`damage_per_contact`/`repair_rate`) must sit where repair is *sometimes* worth the detour — flagged tunable, discharged empirically (UC-18 posture).
- **Course variation scoped to a fixed `default_repair_course()` + `PadSpec.repairable` + two-sided empirical test — RNG `enable_repair` placement axis deliberately DEFERRED** (scope creep otherwise).
- **Checkpoint invalidation:** v4 rebinds input→sensory wiring → retrain required; v3 checkpoints warm-start with the damage block zeroed. Document in PR.
- **Learning interference** (two blocks on one proprioceptive substrate) — PR note, not a blocker.
- **Same-step ordering** damage-then-repair, deterministic; document in the step() comment.

## Challenger verdict
**APPROVE — no revision required.** Verified against source: UC-15 obstacle_contact edge event reused verbatim (no second overlap check; never feeds terminated; floor/ceiling still crash); degrades `max_body_rate` at the sole `rates = [...] * _max_body_rate` line (thrust/mass/drag untouched; integrity==1 byte-identical given min_authority<max_body_rate); repair mirrors UC-18 recharge-on-dock (docked + pad.repairable; hover≠docked and non-repair pad no-op free); `1-integrity` encoding; schema extends battery_hunger_v3 (v3, 25→26); two proprioceptive blocks overlap-safe (additive index_add; select_modality fail-loud; fixture carries 50 mechanosensory_proprioceptive neurons → AC5/AC6 satisfiable); graft_actor zero-inits appended block → parity free; width coupling via _reconcile chain + obs_width==total_width assertion; off-by-default byte-identity. 3 minor folds: assert min_authority<max_body_rate + hermetic AC3; AC7 two-sided empirical load-bearing test = top execution risk + escalation trigger; explicit AC7 fixture wiring. RNG enable_repair placement axis deferred (concur). All 8 ACs addressed; none infeasible.
