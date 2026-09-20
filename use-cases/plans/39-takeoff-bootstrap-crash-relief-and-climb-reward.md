---
plan_for: use-cases/39-takeoff-bootstrap-crash-relief-and-climb-reward.md
work_branch: feat/uc-39-takeoff-bootstrap-crash-relief-and-climb-reward
team: drone-fly-uc-39
approved: 2026-09-20
---

# UC-39 — Approved Implementation Plan (takeoff bootstrap: crash-cliff relief + dense climb reward)

Challenger approved; all 7 binding minor notes folded in. All paths under TARGET_DIR = `/workspace/drone-fly-uc-39-takeoff-bootstrap-crash-relief-and-climb-reward`.

## Analysis
Diagnosis confirmed in source. `compute_reward` (`src/drone_fly/env/reward.py`) is a pure, keyword-only scalar fn taking `cfg: RewardConfig`. The env (`racing_env.py::step`) computes `airborne = position[2] > floor_z + floor_epsilon` and `penalize_collision = crash or (early_termination=="grounded")` (UC-38, line ~546), then calls `compute_reward`. Training γ=0.99, gae_lambda=0.95, ent_coef=0.01 (`train/config.py`). Env already reads `self._prev_pos[2]`, `state.position[2]`, `course.floor_z` — no new sensing.

**Credit-assignment justification (governing quantity = per-action GAE advantage, not episode-return-vs-sit).** Under γλ=0.94, the throttle-up action's advantage A₀ ≈ Σ(γλ)ˡδₗ. The dense potential-based climb reward pays a positive δ (~+0.59 per substantial from-floor climb step) on every climbing step, accumulating to ~+7 over ~20 steps, which dominates the back-propagated crash TD error ((γλ)²⁰·CP ≈ −2.9 at CP=10) ⇒ A₀ ≈ +4 > 0 even on a takeoff that later crashes — so PPO stops punishing throttle-up. The `G_sit ≈ −3.17` return-level figure is supporting context only; the curriculum endpoints (start=10 / end=100 / warmup=0.5) are an empirically-tunable starting schedule to confirm with a training run, NOT a proven return inequality.

## Proposed Solution (two coupled levers)

### Lever 1 — Dense potential-based climb reward (default on)
- **`RewardConfig` (`src/drone_fly/env/config.py`):** append AFTER `airborne_bonus` (preserves positional construction): `climb_weight=2.0`, `climb_target_height=1.0`, `climb_gamma=0.99`.
- **`compute_reward` (`reward.py`):** add keyword params `height_above_floor_prev: float = 0.0`, `height_above_floor_curr: float = 0.0` (defaults ⇒ byte-identical for every existing caller). Add term: Φ(h) = climb_weight·min(max(h,0), climb_target_height); F = climb_gamma·Φ(curr) − Φ(prev); `reward += F`.
- **Env wiring (`racing_env.py::step`):** pass `height_above_floor_prev = self._prev_pos[2] − course.floor_z`, `height_above_floor_curr = state.position[2] − course.floor_z`.
- **Note 4 — PBRS invariance coupling:** `climb_gamma` MUST equal training γ or policy-invariance (Ng et al. 1999) breaks. Document the coupling in `RewardConfig`; prefer sourcing/validating it against the train γ so a future γ change can't silently desync.
- **Magnitudes (pinned):** per-episode max total climb ≈ γ·w·target = 1.98 (< normalized gate_bonus 3.33 ≪ completion 100); telescopes ⇒ non-farmable; round-trip nets (γ−1)·w·peak ≈ −0.02 (≈0, ≤0); standing tax at target = (1−γ)·w·target = 0.02/step < airborne net 0.05 ⇒ hover net +0.03/step; capped at target ⇒ no ceiling-seeking; target 1.0 m aligns with first-gate height (z≈0.9–1.3), safely below ceiling 2.5.

### Lever 2 — Crash-cliff relief: training-time curriculum (env default stays 100)
- **`RaceEnv` (`racing_env.py`):** add `self._collision_penalty_override: float | None = None` + public `set_collision_penalty(value)`. In `step`, before `compute_reward`: `reward_cfg = self.config.reward`; if override not None → `reward_cfg = dataclasses.replace(reward_cfg, collision_penalty=override)`. Default None ⇒ byte-identical; env `RewardConfig` default stays 100.
- **Schedule (`train/config.py` or new `train/collision_curriculum.py`):** pure `collision_penalty_at(num_timesteps, cfg)` — linear ramp `collision_penalty_start`→`collision_penalty_end` over `warmup_fraction·total_timesteps`, clamped, monotonic non-decreasing.
- **Callback (`train/loop.py`):** SB3 callback (`_on_rollout_start`) computes CP for `self.num_timesteps` and calls `self.training_env.env_method("set_collision_penalty", cp)`. **Note 6:** verify at implementation time that `env_method` propagates through the Monitor/VecNormalize/VecEnv wrapper stack to the base `RaceEnv`. Stateless-of-num_timesteps ⇒ resume-correct. Wire into `train()` fresh + resume, default-on; smoke_train sits near start value (proves wiring).
- **`TrainConfig` new fields (pinned):** `collision_penalty_start=10.0`, `collision_penalty_end=100.0`, `collision_curriculum_warmup_fraction=0.5`, `collision_curriculum_enabled=True`. Keep `ent_coef=0.01`.
- **No suicide optimum:** no grounded penalty is added; hover net is positive (+0.03/step) and any crash is negative (CP≥10 even at start) ⇒ AC8 holds structurally at every curriculum value.

## Files Affected

**Production code (developer):**
- `src/drone_fly/env/reward.py` — climb shaping term + two height params.
- `src/drone_fly/env/config.py` — `RewardConfig`: append `climb_weight`, `climb_target_height`, `climb_gamma` (+ climb_gamma↔train-γ coupling doc, Note 4).
- `src/drone_fly/env/racing_env.py` — pass heights; add `_collision_penalty_override` + `set_collision_penalty`; apply override via `dataclasses.replace`.
- `src/drone_fly/train/config.py` — curriculum constants + pure `collision_penalty_at` (or new `src/drone_fly/train/collision_curriculum.py`).
- `src/drone_fly/train/loop.py` — build + append curriculum callback (fresh + resume), default-on.
- `README.md` — reward table (AC10) + prose describing climb reward & the 10→100 collision curriculum (dev-team-authorized).

**Test code (qa):**
- `tests/test_env_config.py` — update `test_reward_config_field_order` (was `fields[-1]=="airborne_bonus"`, ~line 181): now the 3 climb fields follow airborne_bonus; assert new tail order.
- `tests/test_reward.py` — ADD: AC1 (positive from-floor/substantial climb; ≈0 on floor; ≤0 above target), AC2 (round-trip ≈0 with tolerance/≤0 + per-episode bound < completion and ≤ normalized gate_bonus), AC5 combined bound (Note 3: airborne 80 + max climb 1.98 = 81.98 < 100 — extend/replace the airborne-only `test_max_episode_survival_reward_is_below_completion_bonus`), AC6 (shortcut < completion), AC7 (hover-at-target net incl. climb tax > sit), AC8 (hover-episode return > takeoff-then-immediate-crash, at worst-case low endpoint CP=10).
- `tests/test_env_contract.py` — AC4: genuine floor/ceiling crash → `terminated=True`, `info["collided"]=True`, penalty at active CP; stuck/timeout still penalty-free (UC-38 preserved).
- NEW `tests/test_collision_curriculum.py` — AC3: schedule endpoints (start@t=0, end@≥warmup) + monotonic; callback pushes CP into base env through the wrapper stack; env override alters the reward.
- NEW README doc-test (mirror `tests/test_bootstrap_docs.py`) — AC10.

## AC10 reward table (README) — MUST list EVERY term
Columns `Action | Reward | Condition`. Rows required (values = the RewardConfig constants so the doc-test's constant-match passes):
- **Progress** | +1.0 × Δdist | per step, closing distance to current target (do NOT drop `progress_weight`)
- Hover / airborne | +0.1 | per step above the floor band
- Climb | potential-based, weight 2.0, target 1.0 m | per step of upward progress (F = γΦ′−Φ)
- Time penalty | −0.05 | every step
- Gate passed | +10 / N | on a validly passed gate
- Course completed | +100 | on valid all-gates-then-finish
- Collision (floor/ceiling/OOB) | −100 (constant in the cell; describe the 10→100 training curriculum in PROSE) | on genuine crash; terminates
- Obstacle contact | −50 | edge-triggered, non-terminating
- No-progress / timeout cut | 0 | penalty-free (UC-38)

## Risks & Considerations
- γ<1 PBRS artifacts: standing tax 0.02/step (< airborne net, accounted) and a sub-1% high-altitude knife-edge ⇒ AC1's positive test uses a from-floor/substantial climb. climb_gamma=0.99 honors the policy-invariance requirement.
- **Note 7 — AC9 fixture regen:** enumerate airborne reward-bearing golden fixtures that legitimately change and regenerate deliberately with the change explained; near-floor fixtures (height≈0 ⇒ Φ≈0 ⇒ F≈0, e.g. UC-08 baseline/reproducibility/dynamics) stay byte-identical. Check `scripts/regen_uc08_baseline.py`.
- AC5 large-N caveat preserved (loiter-domination for large randomized courses rests on the stuck detector; +2 climb doesn't change that) — anchor the new bound to the default 800-step budget as UC-37 did.
- Curriculum endpoints are empirically-tunable; confirm takeoff emerges with a training run.
- All three CI gates must be green: `uv run ruff check .`, `uv run ruff format --check .`, `uv run --extra dev pytest`.

## Challenger verdict: APPROVE (no Critical/Major; 7 Minor notes folded above)
Credit-assignment crux passes on GAE advantage (A₀≈+4>0 at CP=10). Anti-suicide holds structurally + tested at CP=10. Ordering chain AC5/6/7 re-derived. PBRS telescopes/caps. Test blast radius: curriculum keeps env default 100 → only `test_reward_config_field_order` needs updating for the new fields. All 10 ACs map to concrete code/test changes.
