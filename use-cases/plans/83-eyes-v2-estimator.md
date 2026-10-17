---
plan_for: use-cases/83-eyes-v2-estimator.md
work_branch: feat/uc-83-eyes-v2-estimator
team: drone-fly-uc-83
approved: 2026-10-17
---

# UC-83 — Eyes v2 — Approved implementation plan

Challenger APPROVED (round 2). In-place `/workspace/drone-fly`, sim-only, no Liftoff. Follow-up to UC-82's negative result.

## Analysis
Root cause of UC-82 (eyes-v1 rays made avoidance worse, collision 57% vs 35% blanked): the estimator misses ~96% of obstacles (~99% nearest-quarter) and collapses to "clear" under ~5% hit imbalance. Fix via UC-78's two in-sandbox levers — classification head + focal loss + rebalancing, and hit-weighted mining.

**Verified disk state (reshapes the plan):**
- 23 GB captures, 8 maps, 19 captures, 26,696 raw frames. **Train maps present: only 5** (BandoCity, Hannover, LiftoffArena, MarinaBay, PineValley) ≈12.2k train frames — UC-78 used 7 + a manual capture, none of the extras on disk.
- **No val maps on disk** (AutumnFields/Surtur absent) → `train_eyes` forces `select=last`.
- **Held-out intact** (HangarC03, MinusTwo, ThePit ≈14.5k frames) → like-for-like eval fully possible.
- **No eyes v1 ckpt / dataset / cache on disk** — all rebuilt from scratch.
- **Disk footprint:** ≈46 KB/frame → ≈1.23 GB all / ≈0.56 GB train split vs 18 GB free; `ensure_cache` guards `free < projected + disk-margin-gb`. Non-issue.

## Proposed Solution
**AC1 — opt-in hit/no-hit classification head + focal loss (default-off byte-identical).** `eyes/model.py`: `EyesNet` gains `hit_head: bool=False`; existing trunk+`self.head` (2 outputs) untouched; when on, construct one extra `self.hit_head` module **last** (base-param RNG draw unchanged → default-off byte-identical) → hit logit. `forward` returns `(free, conf)` off, `(free, conf, hit_logit)` on. `default_meta`/`build_from_meta` carry optional `arch.hit_head` (absent⇒false; v1 metas load). `ARCH_VERSION` stays 2. `eyes/train.py`: `losses()` gains an opt-in α-balanced focal term (γ=2.0) with hit-weighted rebalancing, computed in float (AMP-safe); off ⇒ byte-identical.

**AC2 — denser, hit-weighted mining (disk-aware).** Stride lever DE-SCOPED (in `build_dataset` sample count is fixed by captured frames; `--stride-s` only sets history spacing and is relabel-coupled, so a smaller stride can only *reduce* samples). Delivered: (a) hit-weighted sampler at train time (primary, no disk cost); (b) opt-in `--hit-frac` hit-biased frame filter in the dataset build (default OFF; manifest records resulting hit-ray fraction). Disk guard present.

**AC3/AC4 — train v2 + v1-baseline, eval on intact held-out.** One dense dataset over the 5 train maps; train v1-baseline (head off = default path) and v2 (head on + focal + hit-sampler) on identical data → clean A/B isolating the AC1 lever. Eval both on held-out with pinned explicit `tol_m`/range. **Primary AC4 gate: ≥30% relative reduction in nearest-quarter (0–4 m) in-view miss vs the measured v1-baseline-on-identical-data**, absolute ≤0.69 and UC-78's 0.99 as reported secondary anchors; **FA on clear rays ≤0.10**. `eyes compare` helper makes the verdict mechanical. Compute: `select=last`, fixed ~8–12 epochs + weight decay + augmentation → minutes.

**Lever hierarchy:** sampler+focal primary/on; hit-biased filter default-OFF/opt-in fallback; all rebalancing tuned against the FA ≤0.10 bound (dial back miss-chasing if FA breaches) — protects the FA bound and the free-space regression.

**AC5 — conditional (only if AC4 passes).** Re-derive ray-noise via existing `eyes noise-model --from-eval <run>/eval_held-out.json`; re-run UC-82 rays-vs-blank (`scripts/uc82_obstacle_avoidance.sh` recipe) with the v2 noise, same held-out proc courses + seed 1234. Headline: collision share rays-v2 < blanked (reverse UC-82's −62%?). Heavy compute (two 5M-step 70-input retrains); **reuse UC-82's blanked checkpoint** (noise-independent) so only the rays arm retrains; driver resumable.

**AC6 — driver + doc.** `scripts/uc83_eyes_v2.sh` (resumable, per-step markers like uc82/uc79): label-missing → dataset → train v1-baseline → train v2 → eval both → (conditional) noise-model + uc82 re-run. `docs/experiments/uc83-eyes-v2.md`: v1-vs-v2 ray-accuracy table (+ rays-v2-vs-blank collision table if AC5) + recommendation.

**AC7 — CI-hermetic tests** (CPU-only): focal-loss math; default-off byte-identity (seeded state_dict equality + forward-output identity: arity + bitwise-equal free/conf + existing UC-78 eyes tests pass); hit-biased sampling/manifest on tiny synthetic captures; model IO with hit head on; CPU train smoke on v2 path; v1-vs-v2 comparison math.

## Files Affected
**Production (developer, `src/drone_fly/**`):** `eyes/model.py` (opt-in hit_head + meta), `eyes/train.py` (focal + rebalance + hit-sampler; off-path unchanged), `eyes/dataset.py` (hit-biased filter + manifest hit-fraction), `eyes/evaluate.py` (pure-NumPy v1-vs-v2 compare + md), `eyes/cli.py` (train/dataset flags + optional `eyes compare`); `eyes/ray_noise_model.py` already supports AC5 (no change expected).
**Developer-authorized outside src (per uc79/uc82 precedent, already in CLAUDE.md):** `scripts/uc83_eyes_v2.sh`, `docs/experiments/uc83-eyes-v2.md`.
**Test (qa, `tests/**`):** `test_eyes_model.py`, `test_eyes_train_smoke.py`, `test_eyes_dataset.py`, `test_eyes_eval_metrics.py`, `test_eyes_cli.py`.

## Risks & Considerations
1. No val maps → `select=last`; overfit discipline via fixed epoch budget + weight decay + augmentation (per-capture val carve-out out of scope — catalogue split forbids).
2. AC2 stride de-scoped (relabel-coupled, reduces count); hit-sampler + hit-biased filter delivered instead.
3. **Data ceiling** — monocular ranging from these captures may cap v2 below the usefulness bar; honest "better-but-not-enough" outcome is plausible → AC4 negative branch names the follow-up (new close-to-obstacle Liftoff captures / depth backbone).
4. Fair control is a v1-baseline on current data, not UC-78's published numbers (5 vs 7 maps).
5. AC5 is the dominant compute (hours), conditional + resumable, blanked arm reused from UC-82.
6. Byte-identity hazard — extra head constructed last/only-when-enabled; test-locked (state_dict + forward output).
7. Focal in float for fp16/AMP safety.

## Challenger verdict (APPROVE, round 2)
Verified all load-bearing claims against model.py/train.py/dataset.py/evaluate.py/ray_noise_model.py. Round-1 Majors resolved: AC4 bar reframed to ≥30% relative reduction vs measured v1-baseline-on-identical-data (not UC-78's published numbers), FA-clear ≤0.10; AC2 stride misframing de-scoped with code rationale → hit-weighted sampler (primary) + opt-in hit-biased filter (default-off). AC5 gated on AC4 + reuses UC-82 blanked checkpoint. AC1 byte-identity via hit_head constructed last/only-when-enabled, ARCH_VERSION stays 2, test-locked (state_dict + forward output). Disk a guarded non-issue (~0.56 GB vs 18 GB free). AC1–AC4 = minutes of GPU; AC5 = heavy conditional.
