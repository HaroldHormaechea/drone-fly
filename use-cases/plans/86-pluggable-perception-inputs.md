---
plan_for: use-cases/86-pluggable-perception-inputs.md
work_branch: feat/uc-86-pluggable-perception-inputs
run: drone-fly-uc-86
approved: 2026-10-18
---

# UC-86 — Pluggable perception inputs — Approved implementation plan

The challenger approved this in round 1 (no Critical/Major issues); its 9 Minor notes are folded in as [C1]..[C9]. The orchestrator reviewed and accepted it; the user is away and asked for autonomous work. `docs/harness/**` is pre-authorized. TARGET_DIR is this worktree; src paths are under src/drone_fly/harness unless stated otherwise.

## Analysis

**Two runner slots consume perception:**
- `Harness.gate_estimator` (GateEstimator ABC, interfaces.py:85): target/curr_gate/next_lookahead.
- `Harness.free_space_estimator` (FreeSpaceEstimator ABC, interfaces.py:100): obstacle_vision/raycast.

**Coverage and assemblers:**
- `provides` feeds `Harness.provided_blocks()` (interfaces.py:268), then `assembler.check_coverage` in `HarnessRunner.prepare` (runner.py:296). Layout: assembler.py:19 `EXTRA_LAYOUT`.
- The teacher assembler requires ("oracle",) (teacher.py:573).
- The student assembler requires target/curr_gate/next_lookahead (student.py:90) and forces `none` + `assume_no_obstacles: true` (student.py:204).

**Part construction:**
- Parts are built by name via `registry.build_part` (registry.py:148), called from liftoff/__init__.py:30 `make` and from loopback.py:215. The lambdas at loopback.py:234-235 bypass build_part.
- Only session.py:460 constructs HarnessRunner in production.

**Staleness:**
- Eyes and depth stamp -inf before a valid estimate (eyes.py:500, depth.py:476), with placeholder zeros(8)/ones(40).
- The runner enforces max_estimate_age_s only on free space, when it is needed (runner.py:583-586).
- Gate estimators stamp frame.stamp or now (gate_cv.py:458, oracle.py:534, loopback.py:165).
- depth.reset() (depth.py:291) keeps the held result while eyes reset drops it, so the kit uses a weaker generic reset rule to stay AC6-safe.
- Before detection, the gate target is pos (gate_cv.py:404-407).

**Latency recording:**
- recording.py:55 has fixed `LATENCY_STAGES`.
- `_latency_view` (recording.py:273) forwards the whole perception_s/ages_s.
- latency_markdown (recording.py:140) iterates generically.

**Eyes dataset** (C:\dev\drone-fly\artifacts\uc84\dataset):
- samples.npz holds per-sample capture, row, free(40), in_view(40), hit_min_half_m, warning_s, and more. The 192x80 frame cache is not provider input.
- Full-resolution frames and telemetry come via the manifest captures and harness/offline/capture.py:133 `load_capture`, plus the labels npz (eyes/labels.py:620-628: frame_file, stamp, segment, target_index, pos_c).
- The label cone target is the route passage centre (course frame). The state-frame target is passage.center + course.frame.origin / W (labels.py:607-612, 427), with the course from `load_label_course` (labels.py:584).
- UC-78 metrics come from eyes/evaluate.py:156 `evaluate_arrays`. `compare_reports` reads aggregate/distance_bins.

**Config overrides** must hit existing keys (config.py:167 set_dotted), so a new `inputs:` section must be in the packaged default YAMLs. Layering is default <- preset <- user, so liftoff_5inch.yaml + loopback.yaml cover the presets.

## Proposed Solution

### 1. Interface: NEW perception.py

numpy + stdlib only; no module-level import of interfaces, lazy imports inside functions where ABCs are needed.

- **Constants:** `CONTRACT_VERSION = 1`; `PROVENANCES = ("screen","telemetry","metadata")`.
- **Blocks:** a frozen `BlockSpec(name, width, dtype="float32", lo, hi, frame, units, description, in_estimate=True)`. `BLOCKS`:
  - target (3, finite, contract world frame Z-up, training-scale m);
  - curr_gate (3) and next_lookahead (7), frames/units per INFERENCE_CONTRACT.md / course_senses.py, finite;
  - obstacle_vision (8, finite);
  - raycast (40, [0,1], CONE order, 1 = clear to RAY_RANGE);
  - oracle (teacher-only, in_estimate=False).
- Move GATE_BLOCKS/FREE_SPACE_BLOCKS into perception.py and re-export them from interfaces.py. `ROLE_BLOCKS = {"gate_estimator": GATE_BLOCKS|{"oracle"}, "free_space_estimator": FREE_SPACE_BLOCKS}`.
- **`PerceptionProvider` mixin:**
  - class attrs `provides=frozenset()`, `contract_version=CONTRACT_VERSION`, `provenance: str|None=None` (undeclared counts as metadata for the guard, fail-closed), `optional=False`, `gpu_duty: float|None=None`;
  - no-op start/stop/reset;
  - `describe()` returns a small dict: version, provenance, blocks + BlockSpec fields, optional, gpu_duty.
  - The docstring generalises the eyes.py:6-25 contract (stamp semantics, CUDA stream isolation, duty guidance) and states the stamp exceptions [C6]: gate estimators stamp `now` without a frame, and constant providers (`none`, loopback/oracle) stamp `now`. The "-inf before first estimate" rule applies to frame-based free-space providers.
- **`check_spec(provider, role, name)`** raises `PerceptionContractError` (NEW in errors.py, subclass of HarnessConfigError; the message names the provider and block) for: an unknown block, a block outside the role, a version mismatch, an invalid provenance value.
- **`validate_estimate(name, provider, estimate, role)`** checks each declared in_estimate block:
  - present;
  - length == width;
  - floating dtype (int/bool/object rejected; the assembler casts);
  - all finite;
  - within [lo-1e-6, hi+1e-6].
  - `in_fov_mask`, if not None, is bool-like with width 40.
  - The stamp [C3] may be any real number (int/float/np.floating, not bool), not NaN, not +inf. -inf is allowed.
- **`InputsConfig.from_config(cfg)`:**
  - `inputs.require_provenance`: auto|screen|any (auto becomes screen when cfg.harness == "liftoff", else any);
  - `inputs.free_space`: a list.
- **`check_session(harness, assembler, extra_dim, cfg)`** iterates the gate, free-space and composite children:
  - runs check_spec on each;
  - **unused refusal:** a non-empty provides with an empty intersection with `assembler.required_blocks(extra_dim)`, and not `optional`, is refused with an actionable message. Partial non-consumption is allowed (logged and put in meta);
  - **screen guard:** in screen mode, metadata or undeclared provenance is refused. The message names the provider, its blocks, and `inputs.require_provenance: any` as the offline/diagnostic opt-out. Telemetry is allowed.
- **`PerceptionRuntime(harness)`** holds the provider names. `after_update(record, gate, fs, gate_wall_s, fs_wall_s, now)`:
  - [C2] validates every estimate of a provider up to and including its first estimate with a finite stamp. Providers that only stamp `now` are validated on the first call, so the -inf placeholders never end validation;
  - writes `record.perception_s["provider:<name>"]` = update wall time;
  - when info carries `inference_s` with a new `frame_index` (tracked per provider), also writes `record.perception_s["provider:<name>:frame"]`;
  - writes `record.ages_s["provider:<name>"]` = now - stamp;
  - handles composite children from `fs.info["providers"]`.

### 2. Migration (AC1/AC6)

- interfaces.py: `GateEstimator(PerceptionProvider, ABC)` and `FreeSpaceEstimator(PerceptionProvider, ABC)`. The getattr start/stop at runner.py:317/337 now finds a no-op, which has the same effect.
- Attributes only:
  - estimators.py NoObstacleFreeSpaceEstimator: provenance "telemetry" (a constant assertion that reads no environment data; documented) + optional=True;
  - liftoff/depth.py and liftoff/eyes.py: provenance "screen" (eyes also gpu_duty=MAX_GPU_DUTY);
  - liftoff/gate_cv.py TrackingGateEstimator (opencv + learned): "screen";
  - oracle.py OracleGateSource and loopback.py CourseGateEstimator / RaycastFreeSpaceEstimator: "metadata".

### 3. Pluggable loading (AC3)

[C7] registry.entry, its "unknown … registered: …" message, probe and the cli/main.py part listing stay unchanged. Only `build_part` (or a small helper it calls) gains a fallback chain:
1. A registered name works as today.
2. Otherwise a name containing ":" is imported as `module.path:Attr`. A missing package raises HarnessDependencyError; anything else raises HarnessConfigError naming the ref.
3. Otherwise, for perception kinds, look the name up in `importlib.metadata.entry_points(group="drone_fly.perception")`.
4. Otherwise re-raise the original unknown-part error.

Rules:
- Import-ref and entry-point perception parts must be instances of the role ABC (lazy import), else HarnessConfigError.
- After building a gate_estimator/free_space_estimator part, set `part.provider_name = name` (ignore AttributeError).
- Update the module docstring. No pyproject change. New modules import no torch/cv2.

### 4. Composition (AC4)

NEW composite.py `CompositeFreeSpaceEstimator(FreeSpaceEstimator)`, registered in builtins.py as free_space_estimator `composite`.

- **from_config** [C4] reads `inputs.free_space` from `ctx.config` (not the passed cfg). Entries are `{provider, blocks?, primary?, options?}`. Children are built like the liftoff `make`: `build_part("free_space_estimator", provider, {**free_space section, **part_options.free_space_estimator, **options}, ctx)`.
- **Block ownership:** a single claimant wins. With multiple claimants exactly one primary wins, otherwise HarnessConfigError naming the block and claimants. A child that wins nothing is refused.
- provides = the union. `children()` exists for check_session. start/stop/reset are forwarded; stop runs on all children even if one raises.
- **update():**
  - calls each child once;
  - takes obstacle_vision from its owner, and raycast + in_fov_mask from the raycast owner;
  - stamp = min of the owners' stamps;
  - info = {"providers": {name: {wall_s, stamp, inference_s, frame_index, info}}}, plus top-level frame_index/inference_s from the raycast owner.
- Gate blocks stay single-provider (documented). Rays from one provider and gates from another is the two-slot setup. Cross-role overlap is impossible via check_spec.

### 5. Screen guard (AC5)

- configs/liftoff_5inch.yaml and configs/loopback.yaml gain `inputs: {require_provenance: auto, free_space: []}` (commented).
- The session.py oracle-teacher overrides (session.py:369-377) add `"inputs.require_provenance": "any"` plus a log line.
- Offline/sim is unaffected. eyes.py keeps its own refusal.

### 6. Session wiring

- **session.py**, inside the try preceding `runner.prepare(arm=False)` (~session.py:474), so that a refusal shuts the harness down:
  - stamp missing provider names from cfg.parts;
  - [C1] run the existing coverage check first, so a missing block still raises PerceptionCoverageError first (e.g. test_harness_recording.py:315), then call `perception.check_session(h, assembler, runner.extra_dim, cfg)`. Use runner.extra_dim = policy.extra_dim, since run_harness gets extra_dim=None on the teacher/student paths;
  - pass `perception=PerceptionRuntime(h)` to HarnessRunner (construct the runtime first and attach it via a ctor kwarg or attribute).
- **runner.py:**
  - optional kwarg `perception=None` (None keeps current behaviour; existing runner tests are untouched);
  - after the perception try block (runner.py:562-575), and only when it did not already _finish [C2], call `after_update(...)` with the measured wall times;
  - a PerceptionContractError becomes `self._finish(record, STOP_PERCEPTION_ERROR, str(exc))`.
- Record a small `perception` key (the providers' describe()) in run meta via the existing meta dict; no extra recorder plumbing.

### 7. Latency (AC10)

recording.py summarize_ticks aggregates perception_s keys prefixed `provider:` into latency_ms[<key>], and ages_s keys prefixed `provider:` into ages_ms[<key>], only when present. Nothing else changes in the recorder.

### 8. Conformance kit (AC7)

NEW perception_kit.py, public `check_provider(factory, *, role, name="provider", frames=None, n_frames=12, frame_shape=(1080,2560,3), dt_s=0.05, async_mode=False, timeout_s=5.0)`.

- `factory(frame_source, state_source)` builds the provider; frame_source is None in sync mode. Synthetic numpy FrameSource/StateSource: textured frames, forward flight, target 5 m ahead.
- **Checks:**
  1. check_spec, then start().
  2. **Free-space role:**
     - before any frame the stamp is -inf, or == now for constant providers, and never newer than the newest input;
     - within n_frames a finite-stamp estimate appears, with stamp == (sync) / <= (async) the newest frame stamp;
     - every estimate passes validate_estimate;
     - a gap (same frame / None) holds the stamp;
     - after reset() the next stamp is <= the pre-reset stamp, and a new frame recovers.
  3. **Gate role:** the stamp is finite and <= now, and == frame.stamp when a frame is given to a frame-stamped provider.
  4. **Async:** update() is hammered with interleaved reset() while the worker runs. No exception, and outputs conform.
  5. stop() joins: no provider thread is alive after stop within timeout_s (threading.enumerate before/after).
- Returns a report. Failures raise PerceptionContractError naming the provider and check.

### 9. Offline eval (AC8)

NEW offline/perception_eval.py: `run_perception_eval(provider, dataset, *, config=None, preset="liftoff", harness="liftoff", split="held-out", device=None, max_frames=None, out=None, tol_m=None, target_fn=None)`.

**Setup:**
- Refuse the train split.
- load_config + contract.
- Build the provider via build_part("free_space_estimator", ref, cfg.section("free_space"), BuildContext(config=cfg, contract=contract, resources={})), in sync mode. Require raycast.
- Refuse if the provider meta's split_maps.train overlaps the split maps (mirrors evaluate.py:307-312).

**Per split capture:**
- Load the labels npz + meta lag_ms, then load_capture(capture, cfg.section("telemetry"), W, lag_ms), mapping frame files to CaptureFrame.
- Feed all frames in time order (history for eyes); reset at capture start and on segment change.
- Score the rows in the split samples.
- target = route passage centre + origin/W via eyes.labels.load_label_course. This is the default target_fn; it is injectable for tests.

**Scoring and report:**
- pred = estimate.raycast; latency = update() wall time; truth = samples free/in_view/hit_min_half_m/warning_s/maps; body = evaluate_arrays(pred, None, …).
- The report has the eyes eval shape (format 1, dataset, split, maps, tol_m, range_train_m, range_liftoff_m, world_scale, aggregate, distance_bins, per_ray, …), plus provider, provider_describe, device, latency_ms {p50,p95,p99,mean,n} and skipped_frames.
- It writes perception_eval_<split>.json + .md (eval_markdown + a latency section) to out (default <data root>/perception-eval/<provider-slug>-<UTC stamp>/), compatible with `eyes compare`.

**CLI:** `drone-fly perception eval --provider --config --preset --dataset --split {held-out,val} --device --max-frames --tol-m --out`, via add_parser wired in src/drone_fly/cli/main.py next to the eyes parser (cli/main.py:885 pattern).

**Local evidence (not CI):**
- eyes v2 (eyes.weights C:/dev/drone-fly/artifacts/uc84/v2/eyes.pt) on the uc84 held-out split with --max-frames ~300;
- depth on the same split, only if the Depth-Anything weights are cached (no download; report the gap otherwise).

### 10. Example + docs (AC9)

**NEW examples/__init__.py + examples/brightness.py:** `BrightnessRaysProvider(FreeSpaceEstimator)`.
- Provides raycast + obstacle_vision; provenance screen; sync; numpy only.
- Projects the 40 contract rays (harness.geometry ray_directions + CameraModel from the camera section, as depth.measure_rays) into the frame.
- An in-view ray's free = blocked_free (default 0.5) when the patch's mean grey is < threshold, else 1.0. Out-of-view rays are 1.0 with in_fov False.
- obstacle_vision is zeros. stamp = frame.stamp, -inf before the first frame.
- The docstring says it is a teaching example, never for flight.
- Selected as `parts.free_space_estimator: drone_fly.harness.examples.brightness:BrightnessRaysProvider` (not in builtins.py).

**NEW docs/harness/perception-providers.md** covers:
- the contract + version, and the block table;
- stamp semantics + the [C6] exceptions;
- provenance, the guard, and the oracle opt-out;
- config syntax: name / module:Class / entry-point group, with a pyproject snippet;
- composition + primary;
- validation errors;
- CUDA stream / duty guidance;
- latency rows;
- kit usage (a pytest example);
- perception eval + compare;
- the example plugin end to end.

Plus one link line in docs/harness/liftoff-live-session.md.

### AC map
- AC1 → 1, 2
- AC2 → 1, 6
- AC3 → 3
- AC4 → 4
- AC5 → 1, 5
- AC6 → 2 + golden test
- AC7 → 8
- AC8 → 9
- AC9 → 10
- AC10 → 1, 7

### Pitfalls
- The brain layout is untouched.
- Migration is attributes-only, guarded by the golden test.
- CUDA guidance + gpu_duty declaration.
- Per-block age limits are deferred (runner.max_estimate_age_s; the composite uses the oldest owner stamp; follow-up noted in docs).
- Entry points are lookup-only.
- Gate-provider eval is out of scope (UC-71).

## Files Affected

**Production (developer, src/drone_fly/**):**
- NEW: src/drone_fly/harness/perception.py, composite.py, perception_kit.py, offline/perception_eval.py, examples/__init__.py, examples/brightness.py
- EDIT: src/drone_fly/harness/interfaces.py, errors.py, registry.py, builtins.py, runner.py, session.py, recording.py, estimators.py, oracle.py, loopback.py, liftoff/depth.py, liftoff/eyes.py, liftoff/gate_cv.py, configs/liftoff_5inch.yaml, configs/loopback.yaml; src/drone_fly/cli/main.py

**Docs (pre-authorized docs/harness/**):** NEW docs/harness/perception-providers.md; EDIT docs/harness/liftoff-live-session.md (one link).

**Test (qa, tests/**):**
- NEW tests/test_perception_contract.py:
  - BLOCKS/describe, check_spec;
  - validate_estimate errors (shape/dtype/range/non-finite/missing; names provider+block; stamp types);
  - runtime validation in a session-mode runner, including validation continuing past -inf placeholders;
  - unused refusal (+ `none` optional, student path);
  - screen guard (oracle/course/undeclared refused on liftoff auto; telemetry allowed; loopback unaffected; oracle teacher override);
  - import-ref + entry-point loading (monkeypatched), role isinstance refusal, provider_name, registry.entry message unchanged.
- NEW tests/test_perception_composite.py: mapping, overlap refused, primary wins, empty child refused, stamp = min, lifecycle, config read from ctx.config + part_options.
- NEW tests/test_perception_conformance.py: the kit over
  - none (both modes);
  - depth (fake DepthModel, sync+async);
  - eyes (fake estimator, sync+async);
  - opencv (synthetic frame);
  - learned (full kit via a stub like tests/test_harness_gate_learned.py:457 ScriptedEstimator) [C5];
  - oracle (as tests/test_oracle_leakage.py);
  - loopback course/raycast, composite, the example plugin;
  - plus a self-test with a deliberately broken provider.
- NEW tests/test_perception_eval.py: synthetic dataset/captures (tests/eyes_support.py builders), injected target_fn, report keys, JSON+MD, compare_reports accepts it, train refused, CLI parse.
- NEW tests/test_perception_latency.py: provider rows present/absent, frame dedupe.
- NEW tests/test_perception_migration.py + tests/fixtures/uc86/observations_golden.npz + generator tests/uc86_golden.py:
  - [C8] the generator uses only base-commit APIs;
  - the golden is produced from 156432e (`git archive 156432e src | tar -x -C $TEMP/uc86-base`, PYTHONPATH=$TEMP/uc86-base/src; no repo mutation);
  - scenarios: loopback 70, liftoff none 30, depth-fake 70, eyes-fake 70 (opencv optional);
  - the test asserts np.array_equal of every obs/extra with PerceptionRuntime enabled.
- EDIT tests/test_harness_registry_imports.py: hermetic list extended with the new modules; registry snapshot if `composite` requires it.

## Risks & Considerations
- AC2 "not consumed" means wholly unconsumed (partial is allowed), to keep AC6. The challenger agreed.
- `none` provenance is "telemetry", because AC fixes three values. Documented.
- `inputs:` in the default YAMLs is additive to the recorded config; the developer checks any full-config snapshot tests. [C9] No existing run_harness test was found with 12-dim + raycast/depth/eyes, but the developer must run the full suite.
- Runtime validation covers estimates up to the first finite-stamp one; later bad outputs fall to the assembler's existing errors.
- The eval target uses label route geometry offline only; live stays screen-only.
- Depth real-data eval needs cached Depth-Anything weights; no download.
- The ~1 h budget is tight. Trim order if needed: entry-point lookup, then real-data eval evidence. Do not drop learned from the kit.
