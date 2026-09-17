---
plan_for: use-cases/09-configurable-gate-count.md
work_branch: feat/uc-09-configurable-gate-count
team: drone-fly-uc-9
approved: 2026-09-17
---

# UC-09 Implementation Plan — Configurable number of gates (N free-3D waypoints)

**Status:** APPROVED by challenger (round 2). Target `/workspace/drone-fly-uc-09-configurable-gate-count`.

# Analysis
Today the course is a single scalar gate (`CourseConfig.gate_x/gate_center_y/gate_center_z/gate_aperture` + `finish_x`, `floor_z`, `ceiling_z`, `start_position`). `env/geometry.py` is a 3-state string machine `TO_GATE→TO_FINISH→DONE` on **forward x-plane crossing**. Reward adds `gate_bonus` once + `completion_bonus`. `racing_env.py` holds `self._phase`, builds a 12-d obs (`current_target − pos ++ attitude ++ vel ++ angvel`), truncates at `max_steps=400`. `env/randomization.py` samples ONE gate. `record/recorder.py::_course_meta` writes a single `gate` block; `viz/viewer.js::drawMarkers` draws one yz-ring. Only config/geometry/randomization/recorder/racing_env consume the scalar gate API (blast radius contained). Obs stays 12-d so checkpoints load (AC3). `info["phase"]` has no external readers.

# Proposed Solution

## 1. `src/drone_fly/env/config.py`
- New frozen `GateSpec`: `center: tuple[float,float,float]`, `aperture: float`, `.position` np property (aperture doubles as 3D capture radius — no separate tolerance).
- `CourseConfig`: drop scalar gate fields, add `gates: tuple[GateSpec, ...]` + `num_gates` property. **Default = 3-gate course** (AC1): start (0,0,1); gate0 (2.5,0,1.0) r0.6; gate1 (4.0,0.6,1.3) r0.6; gate2 (5.5,−0.5,0.9) r0.6; finish_x 7.0; floor 0.0; ceiling 2.5 (verified solvable). Add factory `single_gate_course(...)` for tests / N=1 path.
- **Invariant:** gates are **strictly x-monotonic** (ordered by increasing center.x); "free 3D" = free (y,z), monotonic x.
- `RandomizationConfig`: add `num_gates_range: tuple[int,int] = (1,10)` (AC5), `min_gate_spacing: float`; replace absolute `gate_x_range`/`finish_x_range` with forward-delta ranges `first_gate_gap_range` (lo ≥ `min_start_gate_gap`), `gate_gap_range` (lo ≥ `min_gate_spacing`), `finish_gap_range` (lo ≥ `min_gate_finish_gap`). Keep `gate_center_y_range`, `gate_center_z_range`, `gate_aperture_range`, `aperture_min`, `z_margin`, `lateral_bound`, `max_resample_attempts`.
- `EpisodeConfig`: add `steps_per_gate: int = 200` (keep `max_steps: int = 400` as base/floor).

## 2. `src/drone_fly/env/geometry.py` — proximity + indexed walk
- `gate_reached(prev_pos, curr_pos, gate) -> bool` = `dist(closest point on segment prev→curr, gate.center) ≤ gate.aperture` (**segment-to-point closest distance** — hard tunneling guard, degrades to point-proximity when near-stationary; faithful to AC2). Keep `finish_crossed(prev,curr,course)` unchanged (forward x-plane).
- Index model, state = `gates_passed: int (0..N)` + `done: bool`:
  - `current_target(course, gates_passed)`: `gates[gates_passed].position` if `< N`, else finish point `[finish_x, gates[-1].center_y, gates[-1].center_z]` (inherits sequence-last gate's y,z).
  - `advance(course, gates_passed, done, prev_pos, curr_pos) -> (gates_passed, done, event)`: if `gates_passed<N` and `gate_reached(...)` → `(+1, done, "gate")`; elif `gates_passed==N` and not done and `finish_crossed(...)` → `(N, True, "finish")`; else unchanged. Only the current-index gate is tested ⇒ out-of-order rejection is intrinsic (AC2). **Add a one-line comment** noting a single step must not span two consecutive gates — `min_gate_spacing` (≫ per-step travel) prevents it (challenger's non-blocking note).
- Remove `TO_GATE/TO_FINISH/DONE` + old `gate_passed`/`target_position` signatures; update `env/__init__.py` `__all__`.

## 3. `src/drone_fly/env/reward.py`
- `compute_reward(*, ..., num_gates: int = 1)` keyword-only; `event=="gate"` branch → `reward += cfg.gate_bonus / max(1, num_gates)` (AC4; div-by-zero guarded; N=1 == today exactly). Progress term needs **no** change (it telescopes across gate transitions, N-independent — challenger-confirmed). Completion/time/collision unchanged.

## 4. `src/drone_fly/env/racing_env.py`
- Track `self._gates_passed`/`self._done` (reset to 0/False). Obs & `_dist_to_target` use `current_target(course, self._gates_passed)` (shape unchanged).
- `step()`: `advance(...)`, `completed = event=="finish"`, pass `num_gates=course.num_gates` to `compute_reward`; add `info["target_gate"] = min(self._gates_passed, N)`; keep `info["phase"]` as a derived string.
- At `reset()` compute **`self._max_steps = episode.max_steps + episode.steps_per_gate*(active_course.num_gates − 1)`** from the active (possibly randomized) course; truncate on it. Invariant N=1⇒400. Budgets: N=3→800, N=10→2200.

## 5. `src/drone_fly/env/randomization.py` — N-gate sampler
- `is_course_solvable(course, rcfg)` for N gates: every gate z∈`(floor_z+z_margin, ceiling_z−z_margin)`, `|y|≤lateral_bound`, `aperture≥aperture_min`; x strictly increasing; adjacent 3D spacing ≥ `min_gate_spacing`; `finish_x ≥ max(gate.x)+min_gate_finish_gap`; start_z strictly inside corridor; **start outside gate0's sphere** (`norm(start−gates[0].center) > gates[0].aperture`).
- `sample_course`: draw `num_gates` first, then start, then incremental-x walk (`x_0 = start_x + draw(first_gate_gap_range)`, `x_i = x_{i−1}+draw(gate_gap_range)`, per-gate y/z/aperture, `finish_x = x_{N−1}+draw(finish_gap_range)`) — ordering/spacing/finish structural, fixed draw count per attempt given N (reproducible, AC6). Reject-resample to `max_resample_attempts`.
- On exhaustion, `_fallback_course(N, base_course, rcfg)`: centreline (y=0), corridor mid-z, `x_i = start_x + first_gap + i·gate_gap` (`first_gap = max(min_start_gate_gap, aperture+margin)`, `gate_gap ≥ min_gate_spacing`), fixed safe aperture ≥ `aperture_min`, finish beyond last. **Zero RNG draws**, solvable-by-construction ∀N∈[1,10]; guarded by `assert is_course_solvable(...)`. Pinned draw order (course then dynamics, no draw for disabled axis) preserved.

## 6. `src/drone_fly/record/recorder.py`
- `_course_meta` emits `gates: [ {center:[x,y,z], aperture, plane:"yz"}, ... ]` (+ `start`, `finish:{x}`, `floor_z`, `ceiling_z`, `forward_axis`, `up_axis`); drop singular `gate`.
- `capture_frame(..., target_gate: int | None = None)`: append to `self._target_gates`, serialise `frames.target_gate: [...]` when supplied (additive; omitted otherwise).

## 7. Recorder callers — pass `target_gate=info.get("target_gate")`
- `evaluate/evaluator.py` (~164), `record/rollout.py` (~94), `train/record_callback.py` (~85): one-line additive change each.

## 8. `viz/viewer.js`
- `buildFlightScene`: anchor over **all** gate centres. Read `const gates = course.gates || (course.gate ? [course.gate] : [])` (legacy fallback, AC7). `drawMarkers`: one yz-ring per gate; **current target** (from `frames.target_gate[frame]` at playhead) drawn brighter/thicker in a distinct colour, others dimmed; legacy/no-target-gate files → uniform (no highlight). Keep `floor/gate/finish/start` substrings; add `gates`.

## 9. Baseline fixture — regenerate (AC6)
- Regenerate `tests/data/uc08_baseline_rollout.npz`: roll new default 3-gate `EnvConfig()` @ seed 42 through the stored `actions` (unchanged), re-save `env_trace`. Recommend a committed `scripts/regen_uc08_baseline.py`. Retarget the two guard tests to "current default env @ seed 42 reproduces the committed baseline". **Commit message MUST explicitly call out the regeneration** (pitfall).

# Files Affected

**Production code — for the developer**
- `src/drone_fly/env/config.py` — GateSpec, gates+num_gates+3-gate default+single_gate_course, RandomizationConfig deltas, EpisodeConfig.steps_per_gate.
- `src/drone_fly/env/geometry.py` — segment proximity, current_target, advance; drop string-phase API.
- `src/drone_fly/env/reward.py` — num_gates normalisation.
- `src/drone_fly/env/racing_env.py` — index state, effective step budget, info["target_gate"], reward call.
- `src/drone_fly/env/randomization.py` — N-gate sampler + solvability + zero-RNG fallback.
- `src/drone_fly/env/__init__.py` — geometry exports.
- `src/drone_fly/record/recorder.py` — gates[] meta + per-frame target_gate.
- `src/drone_fly/evaluate/evaluator.py`, `src/drone_fly/record/rollout.py`, `src/drone_fly/train/record_callback.py` — pass target_gate.
- `viz/viewer.js` — N rings, current-target highlight, all-gate anchors, array-with-fallback.
- `scripts/regen_uc08_baseline.py` (new) — deterministic baseline regeneration.

**Test code — for the QA**
- `tests/test_waypoint_logic.py` — rewrite for segment-proximity + indexed advance/current_target (old plane tests obsolete).
- `tests/test_env_contract.py` — multi-gate ordered completion; out-of-order rejection; N=1 equivalence; effective-budget test (N=1⇒400, N=10⇒2200); timeout test via small custom EpisodeConfig; retarget the two baseline-guard tests.
- `tests/test_reward.py` — normalised per-gate bonus count+magnitude; N=1 == gate_bonus.
- `tests/test_randomization.py` — num_gates across [1,10]; per-gate aperture range spans tight+wide (tightest solvable); solvability + reproducibility ∀N∈[1,10]; fallback solvable ∀N∈[1,10]; start-not-in-gate0-sphere.
- `tests/test_viz_contract.py` + `tests/test_record_recorder.py` — meta.course `gates[]`; per-frame target_gate; static assert viewer fallback (`course.gates` + `course.gate ?`); adjust hardcoded singular-`gate` expectations.
- `tests/test_record_backcompat.py` — course-less + N-gate recordings stay schema-valid.
- `tests/data/uc08_baseline_rollout.npz` — regenerated (committed).

# Risks & Considerations
- **Baseline regen (AC6):** unavoidable; guard becomes "same seed → identical rollout"; call it out in the commit.
- **Tunneling:** closed by construction via segment-to-point distance; `min_gate_spacing` kept ≫ per-step travel so a step can't span two gates at high N (challenger note — one-line comment near `advance`).
- **Finish geometry:** forward x-plane retained; monotonic-x invariant + `finish_x ≥ max(gate.x)+gap` keeps courses completable; early finish ignored until all gates passed.
- **Default course pins the new baseline** — must stay solvable + regenerate deterministically.
- **Checkpoints:** obs/action shapes unchanged → load architecturally (may warrant retraining, AC3).

## Challenger verdict (round 2): APPROVE
Round 1 requested revision on 3 Major items; round 2 resolved all three (step budget pinned; gate ordering/finish invariant explicit; deterministic zero-RNG fallback). Additional v2 strengthenings: incremental-x delta sampler (ordering/spacing structural), segment-to-point proximity (tunneling closed). All AC1–AC8 mapped; deferred partial-observability scope confirmed absent. One non-blocking note: comment the two-gate-span skip-credit near `advance`.
