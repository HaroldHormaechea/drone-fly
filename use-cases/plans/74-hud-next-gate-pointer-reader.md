---
plan_for: use-cases/74-hud-next-gate-pointer-reader.md
work_branch: feat/uc-74-hud-next-gate-pointer-reader
team: drone-fly-uc-74
approved: 2026-10-14
base_commit: a74932f (main with feat/uc-70 merged)
---

# UC-74 — Live HUD next-gate pointer reader (classic OpenCV, no ML) — Approved Implementation Plan

> Analyst↔challenger approved proposal. Base = HEAD a74932f (UC-70 merged); full scope cleared. Source of truth for the developer and QA.

## Analysis
Live HUD next-gate pointer reader, classic OpenCV only (zero ML). The UC-70 run-2 student already consumes a `("pointer",4)` obs tail (body-frame unit bearing 3 + availability bit 1); live it is always `None`→zeros/bit-0 because no reader populates `est.info["pointer"]`. This builds that reader, mirroring how gates become model inputs, and wires it into the single integration seam so the already-trained student consumes it unchanged.

Contract (verified present): `info["pointer"]` = `None|float32[4]` = `[cos(el)cos(az), cos(el)sin(az), sin(el), avail]`, body FLU. Sim truth: `rel_b=R.T@(center−pos)`, normalize, `az=atan2(y,x)`, `el=asin(z)`; `avail=target_valid AND NOT dropped`, zeros+bit0 when unavailable. Seam: `build_student_obs(..., pointer=(gate.info or {}).get("pointer"))` at `harness/student.py:130`. Schema VERSION=3/DIM=65 (`student_obs.py:49,63`) — untouched.

**Conversion (key simplification):** for a camera-locked icon at the gate's true projection, the body bearing needs only the camera uptilt model, NOT live attitude (it cancels): `body_dir = body_to_camera().T @ pixel_rays(icon_px)`, normalize, assemble. Attitude-aware hook kept in reserve if captures show Liftoff clamps/horizon-stabilizes the icon (bounds AC3's tight end).

## Proposed Solution
1. **NEW `src/drone_fly/harness/liftoff/pointer_cv.py`** — `HudPointerReader` (lazy `_cv2()` like gate_cv; no torch / no model load / no gate_net/gate_learned import → AC1). Over a resolved ROI of the **unmasked** frame: detect fixed icon via HSV colour-threshold + contour-shape (area/circularity/solidity), optional multi-scale template fallback; best-unique-match reject (ambiguous→unavailable). Icon px → camera ray → body bearing → `float32[4]` bit=1, else `zeros(3)+0.0` (AC4). `PointerRoi` helper: hybrid ROI — explicit `camera.pointer_roi` override wins, else derive from HUD `camera.masks`; scale with resolution via `FrameMasks.rects()`-style math + `CameraModel.scaled_to` (AC7).
2. **MOD `gate_cv.py` `TrackingGateEstimator`** — add optional `pointer_reader` to `__init__`; in `update()` set `info["pointer"]` right after `info.update(self.extra_info())` (line 441), before `return GateEstimate(...)` (442), using raw `frame.image` when `frame is not None` and a reader is set. Base class → both OpenCv & Learned (live) estimators inherit. `detect_frame` keeps masking its own copy → AC7 non-interference.
3. **MOD `gate_cv.py` + `gate_learned.py` `from_config`** — build the reader from the `camera` section (masks + optional `pointer_roi` + icon HSV/template config). `camera.pointer_roi` is config-level only → obs schema untouched (AC6-safe).
4. **NEW `src/drone_fly/harness/offline/pointer_eval.py`** — offline eval + calibration: labelled-frame manifest → precision/recall/≤5px localization (AC2); reconstructable ground-truth → az/el/combined angular error median & p95 (AC3); calibration report (bias, per-tick jitter std, availability/FP rates) framed to replace provisional `bias_sigma_rad`/`jitter_sigma_rad`/`pointer_bearing_tol_deg` (AC9). Synthetic in-sandbox; real frames on Windows.
5. **Cleanup (UC72→UC74):** `harness/student.py:128-129`, `student/obs_ref.py:88`, `sim/student_obs.py:171`, `training/configs/student-liftoff-run2.yaml:8,19`, and provisional-sigma lines `yaml:25-26`. **EXCLUDE `harness/scoring.py:1`** ("reusable by UC72") — legit forward-ref to UC72 Liftoff-eval, must NOT be changed.

Reference patterns: `OpenCvGateEstimator.detect_quads` (HSV→morph→contour); `TrackingGateEstimator.update` seam (gate_cv.py:320-442); `CameraModel.pixel_rays`/`body_to_camera` (geometry.py); `FrameMasks.rects` (masks.py:68) for scaling; `tests/liftoff_capture_support.py` for synthetic-fixture conventions.

## Files Affected
**Production code (developer):**
- NEW `src/drone_fly/harness/liftoff/pointer_cv.py`
- MOD `src/drone_fly/harness/liftoff/gate_cv.py`
- MOD `src/drone_fly/harness/liftoff/gate_learned.py`
- NEW `src/drone_fly/harness/offline/pointer_eval.py`
- MOD `src/drone_fly/harness/student.py`, `src/drone_fly/sim/student_obs.py`, `src/drone_fly/student/obs_ref.py`, `src/drone_fly/training/configs/student-liftoff-run2.yaml` (comment cleanup only)

**Test code (qa):**
- NEW `tests/test_harness_pointer_cv.py` — synthetic detection; hand-computed bearing unit test (principal-point icon → body optical axis `[cos uptilt,0,sin uptilt]`; offset icon → expected az/el) (AC3 in-sandbox); availability zeros/bit0 (AC4); hybrid ROI derive-vs-override + mask non-interference (AC7); no-ML assertion (AC1); offline latency guardrail over ROI, no regression to combined cost (AC5 offline).
- NEW `tests/test_harness_pointer_parity.py` — AC6: `build_student_obs` with reader-produced estimate → `("pointer",4)` tail non-zero present / zeros+bit0 absent; VERSION=3/DIM=65 unchanged; no retrain.
- NEW `tests/test_harness_pointer_eval.py` — `pointer_eval` on synthetic fixtures (AC2/AC3/AC9 plumbing).
- NEW `tests/pointer_fixtures.py` (or extend `liftoff_capture_support.py`) — render known icon at known pixel.

## Risks & Considerations
1. **Scope split (honor exactly).** In-sandbox gate: AC1, AC3-hand-computed, AC4, AC5-offline-guardrail, AC6-parity, AC7, + eval harness/synthetic fixtures proving AC2/AC3/AC9 plumbing. **Windows/real-footage follow-ups, NOT executed here:** AC2 real precision/recall, AC3/AC9 real bearing error + calibration numbers, AC8 live Liftoff, AC5 live p95. Code/harness designed to run there against `recording.py` raw (pre-mask) captures.
2. **Conversion-model ambiguity (bounds AC3):** pixel→ray→body holds only if icon = true gate projection; a clamped/edge-eased icon breaks it. Characterize from captures before trusting median≤3°/p95≤8°. Attitude-aware hook reserved.
3. **Icon variant/scale/colour brittleness + FP vs other HUD glyphs** — mitigated by ROI restriction + shape constraints + best-unique reject; validate across maps/themes on real captures.
4. **ROI derivation needs a HUD mask bounding the pointer;** if absent, the explicit `camera.pointer_roi` override is the robust path (wins when present).
5. **`fp_suppress_pointer` live guard** — enabling the pointer FP-suppression sub-gate live may need a guard when live bearing is noisier than sim assumed; acknowledged downstream (tied to AC9 re-tune), not on this run's happy path.
6. **Recording provenance** — empirical ACs need live-pilot HUD-visible raw captures (not `.xml` replays); fresh Windows capture pass may be needed. Out of in-sandbox scope.

Build tool: `uv` — test `uv run --extra dev pytest`, lint `uv run ruff check .`, format `uv run ruff format .`. Profiles: none.

## Challenger verdict — APPROVED
Full-scope UC-74 proposal passes review on the rebuilt tree (HEAD a74932f). Integration seam independently reconfirmed: `TrackingGateEstimator.update()` (`gate_cv.py:320`) has raw pre-mask `frame.image` in scope at the info-assembly point; `FrameMasks.apply` copies the image and `detect_frame` masks only its local copy, so the pointer reader (raw frame over its ROI) and gate masking don't collide → AC7 holds. Pointer contract present and matches design (`student_obs.py` VERSION=3/DIM=65/`("pointer",4)`, `student.py:130` seam, `obs_ref` fallback). Scope correct: full build-now set with real-footage/live ACs (AC2-real, AC3-real, AC8, AC5-live) deferred to the user's Windows machine; schema untouched (no retrain) → AC6. Cleanup correctly excludes `scoring.py:1` (legitimate forward-ref to UC72 Liftoff-eval). No Critical/Major issues outstanding.
