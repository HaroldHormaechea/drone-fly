# Use Case 08: Domain randomization — per-episode course (and optional dynamics) randomization

## Summary
UC-03 trains on a **single fixed course**: the geometry lives as frozen constants in
`env/config.py` (`CourseConfig` — `start_position=(0,0,1)`, `gate_x=3.0`, `gate_aperture=0.6`,
`finish_x=6.0`, `floor_z=0.0`, `ceiling_z=2.5`) and `RacingDroneEnv.reset()` rebuilds the *same*
course every episode. With a deterministic policy on one fixed course, the only thing the agent has
to learn is **one trajectory** — the project's current "100% completion / 1.00s" is a single
memorized run replayed, not general flying ability. There is no signal that forces the policy to
respond to *where* the gate and finish actually are.

This use case makes the task vary each episode so the only way to earn reward is to actually fly to
wherever the waypoints are. The **primary axis is per-episode COURSE randomization**: at each
`env.reset()`, sample a new course (start position, gate position + aperture, finish position) within
configurable ranges, respecting arena floor/ceiling bounds and keeping the course solvable. The
observation already carries the **relative next-waypoint pose** and the reward/geometry are already
position-parameterized (`target_position(phase, course)`, `advance_phase(...)`), so the policy *can*
generalize — it currently just never has to. A high completion rate over *randomized* courses then
means "it can fly," not "it memorized one path."

A **secondary, separately-toggleable axis is DYNAMICS randomization**: per-episode randomize mass,
drag, motor/thrust response, and control latency within ranges. This is the classic "domain
randomization" for robustness and sim-to-sim (eventually sim-to-real / Liftoff) transfer that
`PROJECT_BRIEF.md`'s architecture section already anticipated (portable control + domain
randomization). It is secondary to course randomization for the immediate anti-memorization goal but
is delivered as a documented, independently-enabled axis.

Everything is **configurable and OFF BY DEFAULT** (a `RandomizationConfig` with per-axis ranges and
enable flags; `--randomize` / `--randomize-dynamics` toggles, or config, on `train` / `evaluate`).
With randomization off, UC-01..06 behaviour is byte-identical — the fixed-course tests stay green.
Sampling is seeded and reproducible, degenerate samples are rejected/clamped by a solvability guard,
and the recorder stamps each episode's *actual* sampled course into `meta.course` so the UC-06 3D
viewer shows the right markers per episode. `evaluate --randomize` reports completion rate + mean
time over N **different** sampled courses — the honest robustness metric, distinct from the inflated
fixed-course number.

## Acceptance Criteria
1. **`RandomizationConfig` (course axis):** a documented, frozen config dataclass (alongside
   `EnvConfig` in `env/config.py` or a sibling module) carries per-parameter ranges for the sampled
   course — start position, gate position (`gate_x`, `gate_center_y`, `gate_center_z`),
   `gate_aperture`, and `finish_x` — plus an `enable_course` flag. Ranges have sensible defaults
   centred on the current `CourseConfig` values, and a minimum aperture (`aperture_min`). Off by
   default.
2. **Per-episode course sampling on `reset()`:** when course randomization is enabled,
   `RacingDroneEnv.reset(seed=...)` samples a **new** `CourseConfig` within the configured ranges and
   uses it for that episode (target geometry, phase advancement, floor/ceiling bounds all follow the
   sampled course). When disabled, `reset()` reproduces the current fixed-course behaviour exactly.
3. **Solvability guard (tested):** every sampled course is flyable — the gate lies between start and
   finish along the course axis, all points sit within arena `[floor_z, ceiling_z]` and lateral
   bounds, and `gate_aperture >= aperture_min`. Degenerate samples are rejected-and-resampled or
   clamped (documented which); a pure `sample_course(rng, cfg)` sampler is unit-tested for bounds and
   solvability across many draws.
4. **Seeded reproducibility:** a given seed reproduces the **same sampled sequence of courses** — the
   sampler is driven off the env's seeded RNG, so `reset(seed=S)` then repeated resets yield a
   deterministic course stream. Unit-tested (same seed → identical course sequence; different seed →
   different sequence).
5. **Optional dynamics axis (`enable_dynamics`):** an independently-toggleable second axis randomizes
   per-episode mass, drag, motor/thrust response, and control latency within configured ranges,
   applied to the adapter at `reset()`. Documented as robustness / sim-to-sim transfer, secondary to
   the course axis, and controllable *separately* from course randomization (either axis can be on
   with the other off).
6. **CLI / config toggles:** `train` and `evaluate` accept `--randomize` (course) and
   `--randomize-dynamics` (dynamics) — or the equivalent config — wiring the `RandomizationConfig`
   into the env. Both default off, so an unflagged invocation is identical to UC-03.
7. **Back-compat (byte-identical when off):** with both axes disabled, the env, recorder, training,
   and evaluation behave exactly as UC-01..06 — the existing fixed-course unit/integration tests pass
   unchanged, and a fixed-seed fixed-course episode is bit-identical to before this UC.
8. **Evaluation over randomized courses:** `evaluate --randomize` reports course-completion rate and
   mean start→gate→finish time over N **different** sampled courses (seeded, so the randomized eval
   set is itself reproducible for a fixed seed). This is reported as a *distinct* metric from the
   fixed-course eval; the README documents that switching to randomized training will **drop** the
   previously-inflated fixed-course number, and that this lower-but-honest number is the real
   baseline.
9. **Recorder stamps the per-episode course:** with randomization on, each recorded episode's
   `meta.course` block reflects **that episode's sampled course** (not the config default), so the
   UC-06 3D viewer places the correct start/gate/finish/aperture/floor/ceiling markers per episode.
   The recorder reads the actual course used for the episode rather than a static `CourseConfig`.
10. **Hermetic tests on the fixture:** the full suite runs with no network/browser — sampler
    determinism/bounds/solvability, env resets to a randomized course when enabled and to the fixed
    course when disabled, recorder-per-episode-course stamping, and back-compat. UC-01..06 suites stay
    green.
11. **Scope guard (asserted/documented):** this UC delivers env course randomization + optional
    dynamics randomization + config/CLI toggles + eval-over-randomized + recorder per-episode course
    stamping — and **nothing more**. It does NOT add curriculum learning, obstacles / multi-gate
    courses, or any change to the training algorithm; those are explicitly out of scope (future UCs).

## Potential Pitfalls & Open Questions
- **Risk (the inflated number will drop):** the moment training moves to randomized courses, the
  "100% / 1.00s" fixed-course figure will fall — because it was a memorized single path. This is the
  *point*, not a regression. Document the drop up front so the honest randomized-course completion
  rate becomes the new baseline and nobody reads the lower number as a step backward.
- **Edge case (degenerate samples):** naive uniform sampling can produce unflyable courses — gate
  behind the start, finish inside the gate, aperture smaller than the drone, waypoints outside
  `[floor_z, ceiling_z]`. The solvability guard (AC3) must reject/clamp these; decide and document
  reject-and-resample vs. clamp-to-bounds, and cap resample attempts so sampling can't loop forever.
- **Ambiguity (range widths are a difficulty knob):** how wide the ranges are directly sets task
  difficulty. Too narrow → still near-memorization; too wide → may be untrainable within the owner's
  laptop budget. Pick defaults that meaningfully vary the course while staying solvable, expose them
  as documented constants, and note they are tunable (mirrors UC-03's course-geometry-as-constants
  choice).
- **Dependency (recorder must stamp the real course):** UC-06's viewer reads `meta.course`
  (`_course_meta(course)` in `record/recorder.py`). With randomization the recorder must be handed
  the **actual per-episode sampled course**, not the static default, or the viewer will draw the
  wrong markers. This is a concrete wiring dependency, not just a config value.
- **Assumption (obs/reward are already position-parameterized):** the observation carries the
  *relative* next-waypoint pose and `target_position` / `advance_phase` already take a `CourseConfig`,
  so no observation-space or reward-shape change is required — only *which* course object flows in per
  episode. If any code path holds a course captured once at `__init__` (e.g. `self._prev_pos`,
  cached bounds/adapter config), it must be refreshed per `reset()`; audit for such captures.
- **Edge case (dynamics × control interface):** dynamics randomization touches the
  `gym-pybullet-drones` adapter (mass/drag/thrust/latency). The knobs must be applied through the
  adapter cleanly at `reset()` without breaking the CTBR→RPM mapping from UC-03; the `SimpleDrone`
  fixture adapter must accept (or safely no-op) the same knobs so hermetic tests don't need PyBullet.
- **Risk (seeded reproducibility across two RNG consumers):** the course sampler and dynamics sampler
  both draw from the env RNG; interleaving order must be fixed so a given seed reproduces the same
  course *and* dynamics stream. Pin the draw order and test it.
- **Assumption (eval-set reproducibility):** `evaluate --randomize` must itself be seeded so the N
  randomized courses are the same set across runs/checkpoints — otherwise two checkpoints aren't
  compared on the same courses. Fix the eval seed and document it.

## Original Description
The owner noticed that training always runs on the **same fixed track**, so the agent can just
**memorize one path** — which explains the suspicious "100% completion / 1.00s" result (it's one
replayed trajectory, not flying ability). They asked to "randomize the tracks or something" so the
policy has to actually learn to fly to wherever the gate and finish are, rather than overfitting a
single course.

## Clarifications
- Q: What's the primary fix for the memorization problem?
  A: **Per-episode course randomization** — sample a new course (start / gate + aperture / finish,
     within arena bounds and kept solvable) at each `reset()`. This is the anti-memorization axis and
     the main deliverable.
- Q: Is dynamics randomization in scope too?
  A: Yes, as an **optional, separately-toggleable second axis** (mass, drag, motor/thrust response,
     control latency) for robustness and sim-to-sim / eventual sim-to-real (Liftoff) transfer — the
     domain randomization the brief's architecture already anticipated. It is secondary to course
     randomization for the immediate goal.
- Q: Default behaviour and back-compat?
  A: Both axes are **off by default** and fully configurable (`RandomizationConfig` + `--randomize` /
     `--randomize-dynamics`). With randomization off, UC-01..06 behaviour is byte-identical and the
     existing fixed-course tests stay green.
- Q: How is progress measured honestly?
  A: `evaluate --randomize` reports completion rate + mean time over **N different sampled courses**
     (seeded/reproducible) — the real robustness metric. The fixed-course number will drop when
     moving to randomized training; that lower-but-honest number is the true baseline and is
     documented as such.
- Q: Anything downstream that must change?
  A: The **recorder must stamp each episode's actual sampled course** into `meta.course` so the UC-06
     3D viewer shows the right per-episode markers.
- Q: How does this relate to the roadmap?
  A: It is the prerequisite for a *meaningful* UC-07 (post-training activation pruning — prune after
     randomized training, not on a memorized path) and for a future connectome-vs-random ablation.
     Curriculum learning, obstacles / multi-gate courses, and training-algorithm changes are
     explicitly **out of scope** here (separate future UCs).
