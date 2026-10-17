---
plan_for: use-cases/85-eyes-v3-depth-backbone.md
work_branch: feat/uc-85-depth-backbone
team: drone-fly-uc-85
approved: 2026-10-17
---

# UC-85 — Eyes v3 depth backbone — approved implementation plan

Challenger verdict: **Approved** (after 2 review rounds). TARGET_DIR=/workspace/drone-fly, no profiles active.

## Analysis
UC-85 adds an **opt-in, default-off** pretrained monocular-depth backbone as the eyes estimator's trunk, feeding the existing per-ray free/conf (+UC-83 hit) head, to test whether a depth prior helps obstacle ranging on the thin 5-map data where UC-83's from-scratch CNN stalled (nearest-quarter miss ~0.99). Measured like-for-like vs a v1-baseline on the same held-out maps, with latency as a hard deployability gate.

Grounding (codebase read): the project already ships `DepthAnythingV2Model` (`harness/liftoff/depth.py`, `transformers` `AutoModelForDepthEstimation`, `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`, DINOv2 ViT-S ~24.8M params / ~99MB / Apache-2.0) and a pretrained-HF-backbone + pure-torch-CI-fallback pattern (`gate_net.py`). `transformers`+`pillow` live only in the `harness` extra (NOT `dev`) — verified absent from the dev/CI venv, which is load-bearing for AC8. `compare_reports` already encodes the AC6 gate (≥30% rel nearest-quarter in-view miss reduction, FA≤0.10); `bench.run_bench` already emits p50/p95/p99 + `eyes_p95_within_budget` vs 25ms/40Hz. Disk guard (`eyes dataset --disk-margin-gb`) and the uc83 resumable driver are reused.

## Proposed Solution (approved)
**Sibling opt-in trunk; EyesNet left byte-identical.**
1. **`src/drone_fly/eyes/model.py`** — new `EyesDepthNet` sibling (NOT a branch in EyesNet → default-path RNG/state_dict/byte-identity untouched by construction). Reuses the exact per-ray grid_sample query + 256-128 MLP head + optional hit-head. Trunk: a **frozen depth backbone** → per-frame **predicted depth map**, K maps stacked to **K channels** → a small learnable adapter CNN (few stride-2 convs) → stride-8/16 maps + global vector → the unchanged query/MLP. Preserves the K-frame (looming via depth-map growth), ego-motion, and IO contract (same 6-arg forward → `(free,conf[,hit])`). `default_meta` gains `arch.depth_backbone`, `arch.backbone_id`, `arch.backbone_input`; `build_from_meta` dispatches to `EyesDepthNet` when set, else `EyesNet` (absent/false ⇒ v1/v2 metas load unchanged).
   - **Frozen backbone held as a NON-REGISTERED attribute** (not an `nn.Module` submodule). This one decision yields all four exclusions for free — `count_parameters`/`AdamW(net.parameters())` auto-exclude it (MAX_PARAMS=1.5M safe, **no train.py optimizer/budget change**), and `state_dict()` carries only adapter+head so strict `load_state_dict` round-trips without backbone keys and checkpoints stay small (no ~99MB bloat under 17GB disk). The backbone is (re)instantiated from the HF cache by `arch.backbone_id` **inside `EyesDepthNet.__init__`/`build_from_meta`**, which covers all three construct sites (train, `evaluate.py:321`, infer) — the eval path is covered by construction itself. Developer MUST NOT register the backbone as a submodule.
2. **`src/drone_fly/eyes/depth_backbone.py` (NEW)** — deployable frozen-backbone wrapper + factory (no `harness` import): lazy-`transformers` `from_pretrained(backbone_id)` default Depth-Anything-V2-Small, HF-cache reuse; a pure-torch **`tiny-depth`** sentinel stub (no transformers) doubling as offline fallback AND the AC8 hermetic stub; a `provision()` that warms the cache once; offline via `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`.
3. **`src/drone_fly/eyes/train.py`** — `--depth-backbone`/`--backbone-id` plumbing + meta record only. (No optimizer/budget change — non-registration handles it.)
4. **`src/drone_fly/eyes/infer.py`** — `EyesEstimator` holds the frozen backbone for live/bench; forward contract unchanged.
5. **`src/drone_fly/eyes/cli.py`** — `--depth-backbone`/`--backbone-id` on `eyes train` (eval/bench inherit via meta); new `eyes provision-backbone [--model-id]`.
6. **`src/drone_fly/eyes/bench.py`** — no logic change (times the depth path via EyesEstimator; minimal annotation only).
7. **`scripts/uc85_eyes_v3_depth.sh` (NEW)** — resumable driver mirroring uc83: provision → dataset (reuse UC-83 dense set, `--disk-margin-gb`) → train v1-baseline (default-off) + v3-depth (`--depth-backbone`) on the SAME dataset → eval both on HangarC03/MinusTwo/ThePit (pinned tol/range) → `eyes compare` + `eyes bench`. Markers/STOP/MAX_HOURS.
8. **`docs/experiments/uc85-eyes-v3-depth.md` (NEW)** — design, commands, v1-vs-v3 table, latency line, recommendation (adopt / accuracy-win-not-deployable+mitigation / negative→defer UC-84), UC-84 richer-data re-run hook + gated UC-82 re-run path, plus caveats: 192×80 low-res depth, cross-frame depth scale-inconsistency, outdoor-metric-vs-indoor-maps.
9. **`pyproject.toml`** — comment only; reuse the `harness` extra (`uv sync --extra harness`); no new default/dev dep (keeps CI network-free).

### AC coverage
AC1 EyesDepthNet dispatch + byte-identical default + named backbone (DINOv2 ViT-S ~99MB Apache-2.0). AC2 HF-cache provisioning + `provision-backbone` + offline fallback + cached/no-download stub test. AC3 depth-map→K-channel adapter + unchanged query; forward-IO test. AC4 driver trains both on shared dataset, eval same held-out maps, `eyes compare` (unchanged). AC5 `eyes bench` p50/p95/p99 vs 25ms; doc verdict; mitigation = `Intel/dpt-swinv2-tiny-256`/lower res/features-only. AC6 compare gate + bench → doc verdict incl. negative→UC-84. AC7 driver + doc. AC8 all tests via `tiny-depth` stub (CPU, no transformers/network/GPU/captures).

## Files Affected
**Production (developer):** `src/drone_fly/eyes/model.py`, `src/drone_fly/eyes/depth_backbone.py` (NEW), `src/drone_fly/eyes/train.py`, `src/drone_fly/eyes/infer.py`, `src/drone_fly/eyes/cli.py`, `src/drone_fly/eyes/bench.py` (minimal), `pyproject.toml` (comment), `scripts/uc85_eyes_v3_depth.sh` (NEW), `docs/experiments/uc85-eyes-v3-depth.md` (NEW).
**Test (qa):** `tests/test_eyes_depth_model.py` (NEW), `tests/test_eyes_depth_train_smoke.py` (NEW), `tests/test_eyes_compare.py` (extend, v1-vs-v3), `tests/test_eyes_bench.py` (extend, within-budget math), `tests/test_eyes_docs.py` or `tests/test_eyes_v3_docs.py` (uc85 doc), `tests/eyes_support.py` (extend, `tiny-depth` stub + `tiny_run(..., depth_backbone=True)`).

## Risks & Considerations
1. **Latency (main risk, AC5)** — DINOv2 ViT-S on CPU likely >25ms; accepted as "accuracy-win, not deployable" + named mitigations. 2. **Input res** — run on 192×80 cache (like-for-like for AC4; higher-res cache is a documented follow-up). 3. **Frozen-backbone non-registration** — must stay a non-registered attribute (else ~99MB checkpoint bloat / strict-load break). 4. **CI hermeticity** — transformers absent in CI; all depth tests route through `tiny-depth` + lazy imports. 5. **Cross-frame depth scale-inconsistency** and 6. **outdoor-metric vs indoor maps** — doc caveats; fall under AC6 negative branch. 7. **Negative result acceptable** — informative, defer to UC-84.

**Carry-forward for the developer/QA (challenger note):** add an explicit test asserting the checkpoint round-trip — a depth-net `state_dict()` contains **no** backbone keys, and `build_from_meta` + strict `load_state_dict` succeeds — so the non-registration guarantee is locked by a test, not just by construction.

## Challenger final verdict (verbatim summary)
Approved after 2 review rounds. Verified against the repo: `transformers` genuinely absent from `--extra dev` venv (AC8 hermeticity real); `compare_reports` encodes the AC6 gate and compare CLI is `--v1/--v2`; `bench.run_bench` times `EyesEstimator.update` with p50/p95/p99 + `eyes_p95_within_budget` vs 25ms/40Hz; default-off byte-identity guaranteed by construction (sibling `EyesDepthNet`, EyesNet untouched). Round 1 Major (frozen-backbone exclusion mechanism unspecified / wrongly justified as mirroring gate_net) fixed in round 2 via the non-registered-attribute approach. Three minors folded into the doc (adapter default depth-map→K channels; cross-frame depth-consistency caveat; outdoor-metric domain caveat). All 8 ACs mapped, all pitfalls addressed, no scope creep.
