# Use Case 78: Vision "eyes" v1 — camera ray estimator for the 70-input connectome (dataset, labels, model, accuracy)

## Summary
This is the first slice of the UC-73 draft. It builds the fly's **"eyes"**: a small learned perception network that reads Liftoff camera frames plus telemetry ego-motion and outputs the 40 obstacle-ray values the user's **70-input connectome** (lineage 10/11/12, `sim/course.py` `ray_cone`) consumes:
- the rays form a ±45° forward cone (Fibonacci spiral) with `RAY_RANGE` 4 m, scaled for Liftoff by `ray_range_scale`;
- each ray also gets a confidence.

The slice delivers five things:
1. A **field-of-view coverage report**: which of the 40 rays the Liftoff camera can see at its 120° FOV and 20° uptilt.
2. An offline **ray-label generator**. It raycasts the track geometry (track XML + scene caches, the UC64/UC71 auto-label path) at each captured frame's telemetry pose, with the same hit rules as the 70 model's training sim, gates included. The labels are exact and cost nothing to produce.
3. A **dataset builder** over the existing UC71 capture folders (frames + telemetry).
4. A **small estimator network** with a train / eval CLI. It takes the last 2–3 frames plus ego-motion (attitude, rates, velocity) and outputs 40 distances plus confidences. Forward rays should lean on looming / time-to-contact, because parallax gives no depth at the focus of expansion.
5. An **accuracy and latency report** on held-out maps.

The deployable path is **vision-only**: no track geometry or oracle input can reach it at inference. The two later slices are left as follow-up use cases:
- the noise model plus retraining the 70-input student under it (UC-73 AC7–8);
- live validation in Liftoff (UC-73 AC9–10).

Out-of-view rays (below or beside the frame) are handled in this slice by a reported "unseen" flag or prior. An odometry-propagated obstacle map is a stretch goal, not required.

## Acceptance Criteria
1. **FOV coverage report.** A command (e.g. `drone-fly eyes fov-report --preset liftoff`) maps each of the 40 ray directions to in-view or out-of-view. It uses the preset's camera (resolution, hfov, uptilt) and the student's body↔camera transform, and writes JSON + Markdown. This is the first deliverable and is unit-tested.
2. **Ray labels.** `drone-fly eyes label --capture <dir>` writes, for every frame with a valid telemetry pose, the 40 ground-truth free distances (normalised as the 70 model expects, clipped at range) plus a "hit kind" (gate / obstacle / floor / none).
   - It reuses the course conversion and the sim's raycast code, so hit rules are identical to the training sim.
   - A test with a synthetic scene checks hand-computed distances within 5 cm.
   - Frames whose map has no track geometry are skipped with a reason.
3. **Dataset.** `drone-fly eyes dataset` builds a split by map (train / val / held-out, honouring the track catalogue split, no map leakage) over a captures root. A manifest records counts per map and split and a hash of the label settings.
4. **Estimator model and training.** `drone-fly eyes train` trains a small CNN, recurrent or stacked-frame, small enough for the RTX 3050 (8 GB). It takes the last K frames (configurable, default 3, downscaled), the ego-motion vector (attitude, body rates, body velocity) and the ray directions. It outputs 40 distances plus 40 confidences.
   - Checkpoints are resumable, as the student trainer's are.
   - Training must work on the GPU and be smoke-testable on the CPU with tiny fixtures.
5. **Vision-only inference.** At inference the estimator consumes only frames and telemetry ego-motion. A test proves that no track geometry, labels or oracle data are reachable from the deployable path (import graph and signature check).
6. **Accuracy report.** `drone-fly eyes eval` on held-out maps reports, per ray and aggregated:
   - median absolute error;
   - miss rate (a real hit within range reported as clear beyond a tolerance);
   - false-alarm rate.

   Figures are split into in-view and out-of-view rays and into distance bins, and confidence is calibrated per bin. The report is written to JSON + Markdown. Targets are recorded, not gating: the use case delivers a measured baseline.
7. **Latency.** The estimator runs at p95 within the live control budget on the RTX 3050, alongside the gate detector at the 40 Hz runner (≤ 25 ms tick). It is measured by an `eyes bench` command and reported.
8. **CI-hermetic tests.** In CPU-only Linux CI, with no Liftoff and no real captures:
   - FOV mapping;
   - label raycasts on synthetic scenes;
   - dataset split and leakage guard;
   - model shape and IO contract;
   - a tiny training smoke run;
   - the vision-only guard;
   - the eval metric math.
9. **Docs.** `docs/experiments/uc78-vision-eyes.md` covers the design (inputs and outputs, looming rationale, out-of-view handling), the commands, and a Results section the orchestrator fills after the GPU training run.

## Potential Pitfalls & Open Questions
- **Missing input (captures)** — only `uc74-flight-01` (Infinity Loop, 304 MB) is on this machine. The 16 UC71 map captures (36 GB) live in the user's OneDrive share, and disk free is about 41 GB. The GPU training run is blocked on the user providing a captures folder link. Code and tests are not blocked.
- **Risk (rays out of view)** — with a 20° uptilt and a ±45° cone, part of the cone, especially downward and sideways, is never in the frame. The FOV report (AC1) quantifies this. v1 reports and flags those rays; an obstacle map with memory is a follow-up.
- **Risk (4 m range vs speed)** — only about 0.2–0.3 s of warning at racing speed. Measure it, and leave the range change to a follow-up (UC-73 AC10).
- **Risk (labels vs reality)** — Liftoff geometry from XML plus scene caches may miss decorative meshes or include invisible colliders. Spot-check against frames; the hit-kind breakdown shows what dominates.
- **Edge case** — motion blur, lighting across maps, thin poles, untextured walls.
- **Assumption** — telemetry pose timestamps align with frames well enough for labels. The capture format already stamps frames and telemetry (UC64); an alignment-tolerance metric is reported.
- **Risk (GPU budget)** — the gate detector, the eyes and the connectome share 8 GB, and the eyes must stay small.
- **Risk (70-lineage flying quality)** — the 70-input connectome's own Liftoff flying quality is unestablished. It may need its own teacher distillation (the UC70 pattern) in a later use case.
- **Context (why now)** — the UC-75/77 students are blind to obstacles. On held-out maps their failures split roughly evenly between collisions, floor hits and out-of-bounds. The user chose to start the vision track tonight as the path that can address collisions and floor hits together, consistent with the end goal of the deployable model relying only on what it sees.

## Original Description
> Anyway it may be interesting to begin training the vision-aware one, which may require using Liftoff frame-by-frame captures to have a visual small AI model that is capable of detecting obstacles and their distance, somewhat, in every direction, so rays can hit that or something like that. Akin to eyes, you knoiw?

(Builds on the UC-73 draft and its clarifications: alongside the 07 student, not replacing it; telemetry ego-motion is allowed at inference; gates count as obstacles exactly as in the training env; range extension is allowed if the evidence supports it.)

## Clarifications
- The user asked Claude to work autonomously tonight (2026-10-15/16) and waived plan approval. This is the first of the 2–3 slices UC-73 asked for. Noise model + student retrain and live validation follow as their own use cases.
