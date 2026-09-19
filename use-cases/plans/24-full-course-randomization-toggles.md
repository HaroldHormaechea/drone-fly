---
plan_for: use-cases/24-full-course-randomization-toggles.md
work_branch: feat/uc-24-full-course-randomization-toggles
team: drone-fly-uc-24
approved: 2026-09-19
---

# UC-24 — Full-course randomization by default + first-class placement toggles (APPROVED plan)

Approved by challenger after one revision round. Prose only. `TARGET_DIR=/workspace/drone-fly-uc-24-full-course-randomization-toggles`; brief frontmatter → `paths.production=["src/drone_fly/**"]`, `paths.test=["tests/**"]`, `profiles: []`, build=uv, tests `uv run pytest`.

## Summary
Make `drone-fly train` course randomization place the FULL feature set by default and expose independent, end-to-end-wired YAML toggles for obstacle / recharge-pad / repair-pad placement; make recharge-pad placement reachable (it is unreachable today) and add repair-pad placement (never existed). A bare `randomize: true` defaults to schema `damage_proprioception_v4` (26-d) with all three placement axes ON. Non-randomized path stays byte-identical.

## Approved design

**1. Config — TrainRunConfig (config.py ~215/253) — developer.** Add three flat YAML keys, each `bool | None` default `None` (three-state, like `schema`): `randomize_obstacles`, `randomize_recharge_pads`, `randomize_repair_pads`. Add matching `_Spec("...", (bool,))` entries WITHOUT `default=False` (so omitted/null ⇒ None).

**2. RandomizationConfig (env/config.py ~461) — developer.** Add `enable_repair: bool = False` and `repair_pad_radius: float = 0.5`, appended LAST after the recharge fields (field-order/positional compat; `RandomizationConfig()` byte-identical).

**3. CLI resolver (cli/__init__.py) — developer.** Keep `_env_config` UNCHANGED (still used by `_run_evaluate` cli:322 and `_run_prune_trained` cli:442). ADD `_resolve_train_randomization(cfg) -> (env_config, obs_schema)` and wire it ONLY into `_run_train` (which stops calling `_env_config`/`resolve_schema` directly):
  - Effective schema = `cfg.schema` if explicitly set, else (`"damage_proprioception_v4"` if `cfg.randomize` else None).
  - `obs_schema = resolve_schema(effective_schema)`.
  - Each placement toggle = its explicit value if set, else schema-aware default = (`cfg.randomize` AND effective obs_schema carries the corresponding block): obstacle_vision→obstacles, battery→recharge, damage→repair. (dpv4 has all three ⇒ bare `randomize:true` turns all three ON; a lighter explicit schema defaults placement to only what it can sense; explicit toggle always wins.)
  - env_config = None when `not (cfg.randomize or cfg.randomize_dynamics)` (byte-identical parity); else `EnvConfig(randomization=RandomizationConfig(enable_course=cfg.randomize, enable_dynamics=cfg.randomize_dynamics, enable_obstacles=…, enable_recharge=…, enable_repair=…))`.

**4. Coherence rule = fail-loud ConfigError (AC4) — developer.** In `_resolve_train_randomization`, UNCONDITIONALLY before building env_config: if effective recharge toggle ON but effective schema has no `battery` block → raise `ConfigError`; if repair ON but no `damage` block → raise `ConfigError` (exit 2, one-line). Rationale: auto-enabling battery/damage physics would add their width-1 obs dims and break the `env.obs_width == obs_schema.total_width` assertion (loop.py ~363); warn+skip reproduces the silent zero-pad bug UC-24 kills. Schema-aware defaults make this error unreachable except on explicit misconfig. This is the SOLE backstop preventing an inert repair pad (sample_course gets no damage config). Obstacles need NO coherence error (placement doesn't change obs width).

**5. Decouple obstacle placement (loop.py:100-101) — developer.** REMOVE the `enable_obstacles=True` side-effect inside `_reconcile_obstacle_vision`. Placement is now solely resolver/toggle-driven; the reconcile keeps only enabling `ObstacleVisionConfig` (obs width). Required so an explicit `randomize_obstacles: false` under the full schema is honored. (evaluate/prune-trained never hit the reconcile fns, so no regression there.)

**6. Recharge placement redefinition — exactly one pad, feature-presence (randomization.py) — developer.** Replace the greedy multi-pad cover `_place_recharge_pads` with `_place_single_recharge_pad(course, rcfg, battery)`:
  - Place EXACTLY ONE `rechargeable=True` pad at a deterministically-chosen eligible gate anchor, REGARDLESS of whether the course is energy-constrained (the deliberate change; UC-18 placed pads only on over-budget courses ⇒ zero under default battery ⇒ the bug).
  - Eligible = `_descend_column_clear` (reuse). Anchor choice zero-RNG/deterministic (preserve seeded stream + "only appends pads, never shifts geometry"): among clear gates pick the one nearest the path-energy midpoint.
  - Solvability: keep `is_course_solvable` recharge branch. Default battery (non-constrained, shipped case) ⇒ branch skipped ⇒ pad is a valid bonus ⇒ always solvable. Cranked test battery ⇒ single pad must form a valid one-pad cover (`_recharge_covering_valid` works for any pad set); else reject-resample→fallback.
  - No clear anchor ⇒ return None ⇒ `sample_course` resamples; `_fallback_course` (~530) is obstacle-free so always has a clear anchor (no deadlock); it uses the same single-pad placer.
  - DOCSTRING MUST document the one-pad narrowing of UC-18's guarantee: one pad covers only courses solvable with a single recharge; the fallback battery-aware `assert` (randomization.py:578) is a fail-loud precondition that can fire ONLY under a non-default constrained battery (unreachable under shipped default: ~133 m needed vs ~20–30 m realistic path).
  - Keep helpers `_recharge_covering_valid`, `_recharge_gate_indices`, `_path_energy`, `_reference_path`.

**7. Repair placement — new, exactly one pad (randomization.py) — developer.** `_place_single_repair_pad(course, rcfg)`: when `enable_repair` (+ damage active via the CLI coherence guard), place EXACTLY ONE `repairable=True` pad at an eligible (`_descend_column_clear`) gate anchor. No energy model (damage doesn't gate reachability) ⇒ pure zero-RNG feature placement. When BOTH recharge & repair ON: distinct anchors if ≥2 eligible gates; if only ONE eligible gate (e.g. 1-gate course/fallback) emit a SINGLE dual-purpose pad (`rechargeable=True, repairable=True`) — documented co-location exception, still "one of each" (counts 1/1), avoids pad_under shadowing. Wire both placers into `sample_course` and `_fallback_course`; pads appended last (recharge then repair), all zero-RNG.

**8. Docs — configs/train/example.yaml — developer.** Add the three keys as commented defaults; document full-by-default behavior for bare `randomize: true`; state the retrain/checkpoint-invalidation implication (26-d obs, same class as UC-13/15/17/19); describe recharge/repair as single feature-presence pads (load-bearing only when one pad suffices), NOT a variable cover. Keep keys commented so example.yaml still parses.

## Files Affected
Production → developer: `src/drone_fly/config.py`, `src/drone_fly/cli/__init__.py`, `src/drone_fly/env/config.py`, `src/drone_fly/env/randomization.py`, `src/drone_fly/train/loop.py`, `configs/train/example.yaml`.
Test → qa: `tests/test_config.py`, `tests/test_cli.py`, `tests/test_randomization.py`, `tests/test_schema_train.py`, possibly `tests/test_env_config.py` and any test asserting `_reconcile_obstacle_vision` sets `enable_obstacles`.

## Test plan (qa)
- test_config: 3 keys parse & type-check (bool); three-state default None; coherence ConfigError cases (recharge without battery-block schema; repair without damage-block schema).
- test_cli: resolver — full-schema default (randomize + nothing set ⇒ schema=dpv4, all three on); explicit schema override; each explicit toggle overrides independently; randomize:false ⇒ (None env_config, None schema) parity.
- test_randomization: enable_recharge ⇒ exactly one rechargeable pad under DEFAULT battery (AC2 fix); enable_repair ⇒ exactly one repairable pad; both ⇒ one-of-each distinct anchors OR single dual pad on a 1-gate course; off ⇒ none; disabled axes keep gate/dynamics stream byte-identical (zero-RNG); descend-column-clear respected; no-clear-anchor ⇒ resample→fallback (no deadlock). UPDATE UC-18 recharge tests (`test_non_constrained_course_gets_no_recharge_pads`, `test_every_tuned_course...`, `test_constrained_course...`, `test_fallback_under_recharge...`) to the one-pad model — pick a tuned battery + course size where one pad suffices for a meaningful constrained test.
- test_schema_train (AC7): `smoke_train` bypasses the resolver, so construct `env_config=EnvConfig(randomization=RandomizationConfig(enable_course=True, enable_obstacles=True, enable_recharge=True, enable_repair=True))` + `obs_schema=resolve_schema("damage_proprioception_v4")` EXPLICITLY (obstacles won't self-enable now); assert finite update completes, sampled course carries exactly one rechargeable + one repairable pad, `env.obs_width == 26`.

## Challenger verdict: APPROVE (round 2)
Verified against cli/__init__.py, train/loop.py, env/randomization.py, env/config.py, config.py, controller/obs_schema.py, env/racing_env.py. Round-1 issues (all accepted): M1 keep `_env_config` for evaluate/prune-trained + add `_resolve_train_randomization` only in `_run_train`; M2 document the one-recharge-pad narrowing of UC-18's covering guarantee; m3 coherence checks run unconditionally pre-dispatch → ConfigError; m4 AC7 smoke-train test sets placement flags explicitly + dpv4, asserts one rechargeable + one repairable pad and obs_width==26. Enforcement confirmed: randomize:false byte-identical; hermetic smoke-train green with full schema + all placement on, width coupling holds; one-pad-vs-energy-cover tension reconciled (single feature-presence pad, unconstrained under default battery); repair reuses descend-column-clear geometry; no scope creep.
