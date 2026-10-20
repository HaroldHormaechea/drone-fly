---
plan_for: (free-form task) — altitude-robust courses via vertically-spread gates
work_branch: feat/altitude-robust-gates
team: drone-fly
approved: 2026-10-20
---

# Approved plan — altitude-robust courses via vertically-spread gates

**Run order (team-lead ruling, authoritative — these exact filenames):**
- **PRIMARY — RUN FIRST:** `uc92-altitude-robust-warmstart-m13.yaml` (warm_start model 13, `src_extra_dim: 98`, ramp ONLY `vspread_frac`).
- **FALLBACK — only if primary misses:** `uc92-altitude-robust-warmstart.yaml` (warm_start model 11, `src_extra_dim: 70`, ramp `vspread_frac` alongside the uc87 DR ramps).
- The descent eval `uc92-altitude-descent-eval.yaml` extends the PRIMARY (`-m13`).

TARGET_DIR = /workspace/drone-fly-altitude-robust-gates. All paths absolute. No code in plan (prose only).

## Analysis
**Problem.** `src/drone_fly/sim/course_gen.py::_gen_centers()` (~L44-52) hardcodes gate heights to a flat 0–2 training-unit slab: `cz = 1.0 + 0.98·sin(k·ang+phase) + N(0,0.08)`, `k∈{1,2}`, clipped to `[GATE_Z_MIN=0.0, GATE_Z_MAX=2.0]` (module consts L29-30). The sinusoid center (1.0) and amplitude (0.98) are fixed and unparameterized; every loop centers at 1.0. The policy has never seen a gate above ~2 units. World scale 2.0; live, the deployed policy climbs ~6.5 training units over gate 0 (= 13 live m, above the UC-87 5.0 ceiling) and overflies every gate (0/33 live). The gap is policy-side (perception is correct per UC-91 PnP). A naive high-spawn-onto-flat-gates attempt FAILED: no altitude reward gradient from above + catastrophic forgetting (airborne 28.9%→7%). The fix must put gates at MANY heights with a reward gradient at every altitude, introduced gradually to protect normal-flight competence.
**Mechanism context.** `sample_course()` (L55-96) rejection-samples a loop (40 tries) requiring every consecutive 3D spacing ∈ `[MIN_SPACING=1.0, MAX_SPACING=5.0]`; `target_chord~U(1.8,3.0)` and `k`/`phase` are redrawn per try. The crash ceiling is `CourseConfig.ceiling_z` (course.py:161; default 3.5, UC-87 cfg 5.0) applied at `hit_ceil = pos.z > ceil_z` (course.py:1279). The curriculum (ppo.py L194-207) sets `env` attributes each PPO iteration: 0→1 ramps via `_ramp` (difficulty, obs_pos_curr, obs_count_curr) and staged `[start,end]` pairs via `_ramp_between` (deploy_dr_scale, input_dr_scale, penalty_scale; `_ramp_between(frac,None)=1.0`). The env reads these scalars at runtime — e.g. the obstacle-POSITION curriculum `_ocen()` (L725-727) stores obstacles on-path and slides them off by `ovec·PARK·(1-obs_pos_curr)`. `CourseEnv._gate(idx)` (L698-705) is the SINGLE gate accessor feeding `_target_pos()`, `obs()` rel, gate-pass, progress and all rewards; the only other `P["cen"]` reads are in `_spawn` (L800, L815) and use xy only.

## Proposed Solution
Design principle: keep the pool's `cen` z = today's model-11 heights (byte-identical), add a separate per-gate TALL target z, and lerp gate z UP (base→tall) at runtime via a curriculum scale — the obstacle-PARK pattern applied to altitude. This makes default-off byte-identity trivial (`cen` never changes) and confines the transform to `_gate()`.

1. **`course_gen.py` — parameterize the vertical band.** New optional keyword params on `sample_course`/`_gen_centers`, all defaulting to today's values: `gate_z_center_range: tuple[float,float]|None=None` (per-course vertical center `U(lo,hi)` for the TALL band; None⇒fixed 1.0), `gate_z_amp: float=0.98`, `gate_z_max: float=2.0`, `gate_z_min: float=0.0`. "Spread active" ≡ any differs from default. Keep today's draw order (`k`, `phase`, `znoise=r.normal(0,0.08,ng)`). Always compute `cz_base = clip(1.0 + 0.98·sin(k·ang+phase) + znoise, 0, 2)` EXACTLY as today (this stays `cen`'s z). When spread active, additionally draw a GATED per-course `center=U(lo,hi)` (no draw when off ⇒ default RNG stream untouched) and compute `cz_tall = clip(center + gate_z_amp·sin(k·ang+phase) + znoise, gate_z_min, gate_z_max)` from the SAME k/phase/znoise. Rejection accepts a try iff every consecutive spacing of BOTH the base loop AND the tall loop ∈ [1,5]. Emit a NEW course-dict key `"gate_z_tall": [ng floats]` ONLY when spread active (off ⇒ no key).
   *Geometry/feasibility:* horizontal chord is identical for base & tall; vertical spread only increases 3D distance, and 3D spacing `sqrt(chord_xy² + Δz(s)²)` is the norm of an affine function of scale ⇒ convex ⇒ its MAX is at an endpoint (so validating the tall endpoint bounds MAX across the whole ramp). The MIN bound is guaranteed per edge by `chord_xy ≥ MIN_SPACING` (3D ≥ chord_xy ≥ 1 at every scale). Worst adjacent |Δz| = `amp·2·sin(kπ/ng)`, maximal at k=2/ng=5 → `amp·1.90`; with chord up to ~3.5, `amp≲1.88` keeps 3D≤5 WITHOUT widening MAX_SPACING. The per-course center draw adds ZERO adjacent-spacing cost (constant loop shift) and is the main lever for "a target at every altitude." (If a future config needs amp>~1.8 it must widen MAX_SPACING or scale amp by ng — not done now.)

2. **`course.py` — CourseConfig, pool, env lerp.** CourseConfig gains the four fields after the UC-90 block (UC-87/88/89/90 default-OFF convention) + `__post_init__` validation (center_range `0≤lo≤hi`; `amp≥0`; `gate_z_min≤gate_z_max`). `from_dict` MUST tuple-coerce `gate_z_center_range` when present (a YAML list is unhashable and crashes `build_pool`'s cache key `tuple(sorted(ckw.items()))` — mirror the `spawn_distance_m` coercion ~L306). `pool_kwargs` forwards the knobs into `ckw` when active (so they reach `sample_course` and the cache key). `build_pool` (L322-389) packs a new `czt` array `(P, NG_MAX)` float32 from each course's `"gate_z_tall"` ONLY when present, sorted by the same `order` (default pool keeps today's key set + byte-identical `cen`). `CourseEnv.__init__`: `self.vspread_scale=1.0` (default full, mirrors deploy_dr_scale), `self._vspread_on = "czt" in pool` (the generic pool→tensor loop L506-511 picks up `czt` with correct device/dtype automatically). `_gate` (L698-705): when `_vspread_on`, replace the z channel with `czb_z + clamp(vspread_scale,0,1)·(czt − czb_z)` (czb_z = cen[...,2]; czt = `self.P["czt"][self.cid,idx]`) BEFORE the gstart rotation (rotation about world-z preserves z ⇒ commutes). Guarded so the default path is untouched/byte-identical.

3. **`training/config.py` — curriculum fraction.** `CurriculumConfig.vspread_frac: tuple[float,float]|None=None` (staged `[start,end]`, same family as timing_dr_frac). Add it to the existing `__post_init__` validation loop (`0≤start≤end≤1` + tuple coercion). None ⇒ off ⇒ scale pinned 1.0.

4. **`training/ppo.py` — apply the ramp.** In the curriculum block (~L203-207): `env.vspread_scale = _ramp_between(frac, cur.vspread_frac)`.

5. **Training config — `uc92-altitude-robust-warmstart-m13.yaml` (NEW, PRIMARY — RUN FIRST).** warm_start `training/13-deploy-robust/model_readout.pt`, `src_extra_dim: 98` (model 13 already 98-dim ⇒ no memory-column zero-init), `legacy_v1: false`. Curriculum MUST explicitly pin the inherited uc87 DR/penalty ramps to `null` and ramp ONLY `vspread_frac` — because `_merge` (config.py:214) deep-merges the `curriculum` section key-by-key, so merely setting `vspread_frac` would INHERIT the parent `timing_dr_frac:[0.15,0.45]`/`input_dr_frac:[0.45,0.75]`/`penalty_frac:[0.25,0.65]`, which start at scale 0 and would switch model 13's DR/perception OFF at step 0 and ramp them back up (perturbing model 13 off its trained full-DR distribution — the opposite of minimal-perturbation). Explicit block:
   ```yaml
   curriculum:
     timing_dr_frac: null
     input_dr_frac: null
     penalty_frac: null        # penalty_scale = init + 1.0·(1−init) = 1.0 (full)
     vspread_frac: [0.1, 0.7]
   ```
   With the three DR/penalty fracs null, `_ramp_between(frac,None)=1.0` ⇒ FULL DR + full penalty from step 0 ⇒ model 13 flies exactly its trained distribution, perturbed ONLY by the altitude ramp. (`difficulty_frac`/`obs_pos_frac`/`obs_count_frac` are already null in the parent.) course knobs: `gate_z_center_range: [1.0, 3.8]`, `gate_z_amp: 1.1`, `gate_z_max: 5.0`, `gate_z_min: 0.0`, `ceiling_z: 6.5`. `budget: 16.0e6`, `n_envs: 256`. Comments: obstacles-not-lerped (gate-centric UC, intentional); training-headroom (gates to 5.0 under 6.5 ceiling = 1.5 units; during the 10–70% ramp an unskilled policy may incur ceiling crashes approaching tall gates — a legitimate "don't overfly" signal, not a bug). Rationale: "preserve ~28.9% airborne" IS model 13's own number, so warm-starting model 13 + adding only altitude is minimal-perturbation.

6. **Training config — `uc92-altitude-robust-warmstart.yaml` (NEW, FALLBACK — only if primary misses).** `extends: uc87-deploy-robust-warmstart` (warm_start model 11, `src_extra_dim: 70`, zero-init 28 memory cols 70→98). Same course knobs + ceiling as the primary; ramp `vspread_frac: [0.1, 0.7]` ALONGSIDE the inherited uc87 DR/penalty ramps (this run must re-learn deploy-robustness AND altitude together).

7. **Eval config — `uc92-altitude-descent-eval.yaml` (NEW).** `extends: uc92-altitude-robust-warmstart-m13` (the PRIMARY). Override `start: {mode: mixed, ground_fraction: 0.2, air_z_m: [5.0, 9.0]}` (high air spawns that must DESCEND; 20% ground starts also exercise climb), `ceiling_z: 12.0` (eval-only headroom isolating descent from ceiling crashes — a drone spawned at rest at z∈(6.5,9.0] cannot clear a 6.5 ceiling within the 0.4 s grace), `vspread_frac: null` (scale 1.0 ⇒ full tall gates at eval).

## Files Affected
**Production code (developer):**
- `src/drone_fly/sim/course_gen.py` — vertical-band params + dual-endpoint rejection + conditional `gate_z_tall` output.
- `src/drone_fly/sim/course.py` — CourseConfig fields/validation/from_dict tuple-coercion/pool_kwargs; build_pool `czt` array; CourseEnv `vspread_scale`/`_vspread_on`/`_gate` lerp.
- `src/drone_fly/training/config.py` — `CurriculumConfig.vspread_frac` + validation.
- `src/drone_fly/training/ppo.py` — apply `env.vspread_scale`.
- `src/drone_fly/training/configs/uc92-altitude-robust-warmstart-m13.yaml` — NEW, PRIMARY (model 13, run first).
- `src/drone_fly/training/configs/uc92-altitude-robust-warmstart.yaml` — NEW, FALLBACK (model 11).
- `src/drone_fly/training/configs/uc92-altitude-descent-eval.yaml` — NEW, descent eval.

**Test code (qa):**
- `tests/test_uc92_altitude_gates.py` (NEW, CPU-hermetic, mirrors tests/test_uc90_gate_altitude.py): worktree guard (`hasattr(CourseConfig,"gate_z_center_range")`); default-off byte-identity (defaults present; `sample_course(seed)` with no spread kwargs emits NO new key, values unchanged); **feasibility over the FULL shipped pool** — iterate all `base..base+size-1` (1024) uc92 courses via `build_pool`+uc92 `pool_kwargs`, assert every edge's spacing ∈ [1,5] at scales {0,.25,.5,.75,1} AND `chord_xy ≥ 1` per edge AND all gate z ∈ [gate_z_min,gate_z_max] (proves no degenerate fallback ships); pool has `czt` with spread on / absent off, `cen` z == base; `_gate` lerp endpoints+monotonicity (0=base, 1=czt, .5 strictly between; default-off env unaffected); `CurriculumConfig(vspread_frac=[0.1,0.7])` validates, bad pairs raise, None⇒1.0; `extra_dim`/`memory_dim` unchanged.
- **Empirical (QA on the in-sandbox RTX A3000, SEQUENTIAL — single 6 GB card):** (1) retrain the PRIMARY `uc92-altitude-robust-warmstart-m13` (~16M steps, 256 envs) FIRST; eval descent via `drone-fly train eval --config uc92-altitude-descent-eval` → must MEANINGFULLY beat model 13 (`training/13-deploy-robust/model_readout.pt`); airborne/normal-flight completion within noise of model 13 (~28.9%) on model-13's eval distribution; smoothness (`mean_abs_delta`, `yaw_flips`) via `smooth_eval.evaluate_one` at `live-like` not materially worse than model 13. (2) ONLY if the primary misses: run the FALLBACK `uc92-altitude-robust-warmstart` (model 11) and re-eval. Report numbers.

## Risks & Considerations
1. **Warm-start source — RESOLVED by team-lead ruling:** model-13 config (`-m13`) is PRIMARY/run-first (minimal perturbation on existing deploy-robustness); model-11 config (bare name) is FALLBACK, spent only if the primary misses.
2. **Obstacles not lerped:** tunnel/fly-over/variety obstacle z come from the BASE centers and are NOT lerped, so at high vspread a tall loop's obstacles sit at model-11 heights. Intentional for this gate-centric UC (obstacles have their own curricula); noted in the config.
3. **Hard constraints met:** UC-90 `gate_alt_weight` left 0.0 (not reused — it collapses completion); all new fields default-OFF/byte-identical; old checkpoints/configs decode unchanged (additive dataclass defaults; from_dict unknown-key guard still passes).
4. **Feasibility fallback:** amp 1.1 ≪ the ~1.8 cap gives wide margin; the full-pool test proves no degenerate course ships; k is redrawn per try so a feasible draw is almost always found.
5. **Performance:** the `_gate` lerp is a couple of tensor ops gated by `_vspread_on`; negligible and fully skipped on the default/old-model path.

## Challenger's final verdict
APPROVE. Both Majors fully resolved. Fix #1 (eval `ceiling_z: 12.0`) cleanly isolates descent skill from ceiling collisions. Fix #2 (per-edge `chord_xy ≥ MIN_SPACING` rigorous MIN proof + interior-scale samples + full 1024-course pool iteration) closes the degenerate-fallback Critical. Independently verified: `_gate` is the correct seam (`_spawn` reads are xy-only), byte-identity holds (`_assert_same` key-exact; gated RNG), pool→tensor conversion picks up `czt` automatically, MAX_SPACING bound by the convex endpoint, amp-vs-ng math correct. Model-13 config adopted as primary per team-lead ruling.
