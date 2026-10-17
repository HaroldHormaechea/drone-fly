---
plan_for: use-cases/82-obstacle-detection-70-input-connectome.md
work_branch: feat/uc-82-obstacle-detection-70-input
team: drone-fly-uc-82
approved: 2026-10-16
---

# UC-82 — Obstacle detection for the 70-input connectome — Approved implementation plan

Challenger APPROVED (round 2). SIM-ONLY (no Liftoff). UC-73 AC7–AC8, building on UC-78's shipped eyes module.

## Key decisions (must also land in the experiment doc)
- **Model identity correction:** `uc70-diag-run1` is NOT the 70-input model (its `ablate.json` → `in_dim:61`, the deployable student-obs distill thread that failed in UC-79/80/81). The true 70-input obstacle-avoider is **`training/11-count-curriculum/model_readout.pt`** (lineage 10/11/12; committed headline **67% honest deterministic @ seed 1024, 0% barging**). All retrains warm-start from it.
- **AC1 noise-model source = path (b): derive from UC-78's documented held-out eval stats** ("documented measured fallback"), NOT train eyes. Rationale: no eyes checkpoint/dataset/cache on disk; even a reproduced eyes v1 is near-useless (95.8% miss, 99% nearest-quarter) per `docs/experiments/uc78-vision-eyes.md`, which itself says the eyes aren't ready to feed the connectome. Skipping eyes training also makes the overnight budget fit.
- **Likely-negative AC7 surfaced up front:** at ~96% miss the rays carry near-zero signal → the ray-noise student may converge to ≈ the blanked control. AC7's documented-negative branch covers this; deliverable is "first measured evidence" either way.

## AC → solution (all in src production scope; no gpu_prototype edits)
- **AC1** NEW `eyes` noise-model builder + `eyes noise-model` CLI subcommand → emits `ray-noise.yaml` (gate-`n.yaml`-shaped) with `source: uc78-eval-documented-fallback` + per-field notes. Field→source: miss_prob binned by distance (0–1/1–2/2–3/3–4m) × in-view/out-of-view (in-view 0–1m=0.990, in-view far ramp 0.913→~0.45, out-of-view ~0.99), FA on clear=0.031, `radial_bias=0.0` + single σ=0.5 training-m (eval tol) as documented constants (dropout+FA dominate; bias/spread second-order on rare detected rays).
- **AC2** NEW `src/drone_fly/sim/ray_noise.py` (`RayNoiseModel`, batched-torch apply) + hook in `course.py` `extra()`/`_raycast()`; default-off → **byte-identical** (test-locked). In/out-of-view via a batched torch twin of `eyes.camera_rays.project_rays`, test-pinned to the numpy ref.
- **AC3/AC4** NEW `configs/uc82-obstacle-rays.yaml` (raycast=true, no_obs=false, extra_dim=70, warm=11-count, src_extra_dim=70); blanked control via `ray_blank=true` → 40 rays = **constant clear(1.0)** (not zeroed). Plumb through `config.py` + `ppo.py build_env`.
- **AC5/AC7** NEW proc-course deterministic evaluator (existing `training/evaluate.py` is Liftoff-only) → tallies `course.py OUTCOMES` from in-step `last_outcome`/`cause_*` (NOT post-step positions — EXPERIMENTS.md bug). Mapping (sums to 100%): obstacle→collision, arena→out-of-bounds, floor, ceiling, timeout, completed→completion; no gate_lost on proc courses. **Success bar: ≥20% relative collision-share reduction** rays<blanked; else documented negative.
- **AC6** NEW `scripts/uc82_obstacle_avoidance.sh` (resumable/reboot-safe: STOP-after-~10h + STOP-file + skip-if-output-exists + RunDir resume) steps 0→4; NEW `docs/experiments/uc82-obstacle-detection-70-input.md`.
- **AC8** CPU-hermetic tests (see QA list).

## Files Affected
**Production (developer — `src/drone_fly/**`):** NEW `sim/ray_noise.py`; EDIT `sim/course.py` (CourseConfig fields + injection/blank hook + in-view torch twin); EDIT `training/config.py` (plumb ray_noise/ray_blank); EDIT `training/ppo.py` (build_env wiring); NEW proc-course evaluator (in/beside `training/evaluate.py`); NEW `eyes/` noise-model builder + `eyes/cli.py` subcommand; NEW `training/configs/uc82-obstacle-rays.yaml`.
**Developer-authorized:** `scripts/uc82_obstacle_avoidance.sh`, `docs/experiments/uc82-obstacle-detection-70-input.md`.
**Test (qa — `tests/**`):** `tests/sim/test_ray_noise.py` (load+inject; no-injection byte-identity; blank==1.0 constant; in-view twin vs camera_rays numpy ref); `tests/training/test_uc82_*.py` (collision-metric math; tiny CPU smoke retrain raycast+ray_noise asserting it runs+checkpoints).

## Must-verify-before-long-run (developer)
1-step warm-start load test: `controller/reservoir.py` loads the stripped 674.8K `model_readout.pt` `strict=False` (body rebuilt from connectome; do NOT use the 177M `exp11_count.pt`) — **assert loaded readout extra width == 70**.

## Overnight budget
256 envs (A3000 6GB), ~5M steps/run warm-started fine-tune (~2–4h each), 2 retrains + 2 evals + noise build + sanity = fits one overnight window with margin.

## Challenger verdict (APPROVE, round 2)
Verified ray dims, OUTCOMES, config validation, UC-78 numbers, uc70-diag-run1's in_dim:61, warm-start loader contract, Liftoff-only evaluate.py. Round-1 Major on AC1 (plan implied a per-bin bias/sigma grid the UC-78 doc doesn't contain) resolved: explicit field→source table, bias/spread as documented constants with dropout-dominates justification, source tag + per-field notes, follow-up recorded for the richer grid. All 8 ACs map; no scope creep (proc-course evaluator required because evaluate.py is Liftoff-only). Budget concrete and realistic. Carry-forward: warm-start from stripped model_readout.pt (assert extra-width==70); blank=constant-clear(1.0); collision share = obstacle outcome from in-step last_outcome/cause_* (never post-step positions); AC7 ≥20% relative reduction with documented-negative branch (negative plausible given eyes v1 fidelity).
