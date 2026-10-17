---
plan_for: use-cases/79-student-information-gap-ablation.md
work_branch: feat/uc-79-student-information-gap-ablation
team: drone-fly-uc-79
approved: 2026-10-16
---

# UC-79 — Student information-gap ablation — Approved implementation plan

UC-79 (student information-gap ablation) — FINAL proposal, approved by the challenger. All paths under `/workspace/drone-fly` (in place, branch `feat/uc-79-student-information-gap-ablation`). Brief lists no profiles → none applied. Prose only; the developer writes code.

## Analysis
UC-77 trains the student via DAgger against the frozen teacher, both on the SAME env state. Key facts verified in the tree:
- `StudentCourseEnv` (`sim/student_env.py`) wraps the teacher's `MultiCourseEnv` and already exposes `teacher_obs()` → `build_teacher_obs(env, env.obs_version)` on the exact state the student obs is built from — the lever for AC2.
- `build_teacher_obs` (`sim/teacher_obs.py`) → 150 (v2)/148 (v1) values. Blocks: harness30[0:30), attitude6[30:36), vel_body[36:39), **gates[39:78)**, **obstacles[78:142)** (8 nearest surfaces incl. floor), **altitude[142]**, prev_action[143:147), arm_remaining_s[147], battery[148:150). `block_slice(name)` exists. The floor surface is folded into the distance-sorted obstacles block, so it is NOT a fixed slice (drives the AC2 floor design below).
- `build_student_obs` (`sim/student_obs.py`): `student-obs` v3, DIM=65. The actor obs width is read DYNAMICALLY downstream — `distill.py:1249/1617` do `obs_dim = senv.student_obs().shape[1]` for the DAgger buffer and rollout — so appending privileged columns in the env auto-sizes them.
- `MLPStudent` (`controller/student.py`) hardcodes `_mlp(DIM,…)` + `input_scale()` — the only build-time place needing the width threaded. UC-79 is MLP-only (AC6 `--actor mlp-student`); K1 stays untouched.
- Snapshot default-drop is `_DISTILL_UC77_DEFAULTS` popped in `StudentConfig.snapshot()` → the AC3 mechanism.
- `student_sidecar()` (`distill.py:947`) builds `obs_schema` via `student_obs.schema(...)`, sets `deployable`/`role`.
- Refusals: `export_student` (`training/export.py`) already refuses mlp unless `--baseline-export`; `plan_student` (`harness/student.py:166`) runs `validate_meta` then gates `deployable`/`allow_baseline`.
- Report: UC-77's `final_report` + `evaluate_student_run` ALREADY emit exactly AC8's metrics (held-out + each `track_evals` selector; student noise+exact; teacher exact; macro_completion; full outcome breakdown; `REPORT_EPISODES=32`, `REPORT_SEED=1234`). Infinity Loop = `liftoff:liftoff-arena-04` already a `track_evals` entry. AC8 needs NO new eval code — only the right config.
- `train student --resume <run>` resumes from `resume.pt`; the step `budget` is cumulative across resume, `max_hours`/wall resets per process.

## Proposed Solution

**1. Config field (AC1, AC3)** — `training/config.py`
- Add `student_privileged: tuple[str, ...] = ()` to `DistillSection`; add `"student_privileged": []` to `_DISTILL_UC77_DEFAULTS` (empty → popped from snapshot → config.yaml byte-identical).
- In `student_from_mapping`, coerce the YAML list to a validated string tuple (mirror `track_evals`).
- In `StudentConfig.__post_init__`, validate members ⊆ `{altitude, floor, gate_next_next, gates_full, surfaces, teacher_full}`; reject unknown/duplicate; reject any privileged list when `student.actor != "mlp-student"` (diagnostic is MLP-only → keeps K1 byte-identical).

**2. Privileged extraction via the teacher's exact code path (AC2)** — `sim/teacher_obs.py`
- Define ONE shared per-surface encoder `surface_feature(R, pos, pt, dist, nrm, valid) -> (n,8)` = `valid * cat([_body(R,pt-pos), nan_to_num(dist,posinf=0), _body(R,nrm), 1.0])` — exactly today's post-sort encoding. Refactor `build_teacher_obs` to call it on the full sorted surface tensors. **Must be a strict op-for-op factoring** (same shapes, same reduction order) so the `test_sim_teacher_obs.py` byte-identical pin can't drift on float op reordering — obstacles block stays byte-identical (AC3).
- Add `PRIVILEGED_WIDTHS = {altitude:1, floor:8, gate_next_next:13, gates_full:39, surfaces:64, teacher_full: dim(version)}` and `teacher_privileged_blocks(env, names, version)`: compute `t = build_teacher_obs(env, version)` once; return clean slices for altitude (`block_slice("altitude")`), gates_full (`block_slice("gates")`), surfaces (`block_slice("obstacles")`), gate_next_next (3rd 13-wide gate row, offset 39+26..39+39), teacher_full (all of `t`); for **floor**, apply the SAME `surface_feature` to the floor's raw row (`pt=fpt`, `dist=pos.z−floor_height(pos)`, `nrm=(0,0,1)`, `valid=dist≤OBS_RANGE_M & finite`). Append in fixed canonical (AC1-declaration) order, each once.

**3. Append privileged to student obs (AC1, AC5)** — `sim/student_env.py`
- `StudentCourseEnv.__init__` gains `privileged: tuple[str,...] = ()`; `_build_obs` concatenates `teacher_privileged_blocks(self.env, self.privileged, self.env.obs_version)` when non-empty. Privileged comes from env truth → always exact in both noise/exact modes (AC5, doc-labelled). Default empty → tensor unchanged (AC3).

**4. Thread obs width into the MLP actor (AC1)** — `controller/student.py`
- `MLPStudent.__init__` gains `obs_dim: int = DIM` + optional `input_scale_vec`; builds `_mlp(obs_dim,…)` and an input-scale buffer of length `obs_dim` (privileged columns use the teacher's per-value scales sliced from `teacher_obs.input_scale(version)` by block offset; floor uses the obstacle-feature pattern `[1/OBS_RANGE_M]*4+[1.0]*4`). `arch()` records `obs_dim` (+ privileged list) for round-trip.
- `build_student(..., obs_dim=None)` passes through (default `student_obs.DIM`; K1 path unchanged); `student_from_state` reads `arch["obs_dim"]` (default DIM) to rebuild at the right width.

**5. Sidecar + schema (AC4 marker)** — `sim/student_obs.py` + `distill.py`
- `schema(..., privileged=())`: when non-empty add a top-level `"privileged"` entry (names+widths+"always exact") and set `dim = DIM + Σ widths`, appending block layout entries; when empty add NOTHING (byte-identical, AC3).
- `student_sidecar(...)` passes `privileged=cfg.distill.student_privileged`; when non-empty set `role: "diagnostic"`, `deployable: false`, add top-level `privileged` list + `obs_dim`. Empty → unchanged.
- `build_student_env` **derives the privileged list from `cfg.distill.student_privileged` itself** and passes it into `StudentCourseEnv` (single choke point used by BOTH training at :1206 and eval/report via `evaluate_student_run`/`evaluate_policy`→build_student_env at :549/:2022 → identical obs width in train+eval, no AC8 mismatch). `build_actor` computes `obs_dim = student_obs.DIM + Σ PRIVILEGED_WIDTHS[b]` from the same field.

**6. Refusals (AC4)**
- `training/export.py`: in `export_student`, right after `side = read_sidecar(ck)` and BEFORE the `mlp-student and not baseline_export` gate (:130), raise `ExportError("diagnostic privileged student, not deployable")` unconditionally (fires even with `--baseline-export`), naming the block list.
- `harness/student.py`: in `plan_student`, immediately after `load_export` (:179) and BEFORE `validate_meta` (:180), raise `StudentSchemaError("diagnostic privileged student, not deployable")` unconditionally (fires even with `--allow-baseline`; precedes the generic dim-mismatch error).

**7. Diagnostic configs — baseline + 6 variants (AC6, AC7)** — `training/configs/`
- `student-liftoff-open-priv-{baseline,altitude,floor,gate_next_next,gates_full,surfaces,teacher_full}.yaml`, each `extends: student-liftoff-open`, `student.actor: mlp-student`, `tracks.obstacles: none` + `track_evals:[liftoff:liftoff-arena-04]` + `final_report: true` (inherited). baseline sets `student_privileged: []`; each variant sets the one block.
- **Authoritative equalizer = identical step `budget` (`base.budget`)** across all 7, sized to ~45 min on RTX 3050 @ 2048 envs (documented in each header with the bench assumption). `max_hours` OMITTED (it resets per process on resume → would break AC6 equality under the AC9 reboot scenario; the step budget is cumulative/resume-safe).

**8. Driver (AC9)** — `scripts/uc79_ablation.sh`
- Runs the 7 `drone-fly train student --config <variant> --teacher <run> --noise <model> --device cuda` back-to-back. Per variant: completed run (final ckpt + `eval/student-held-out.json`) → SKIP; partial → `--resume <run>`; else fresh. State = filesystem, so a reboot resumes the interrupted variant and continues (no sequence restart). Flags/env for teacher run, noise model, output root. Header: needs only the repo + the three copied artifacts, no Liftoff.

**9. Experiment doc (AC9 artifact list, AC10)** — `docs/experiments/uc79-student-information-gap.md`
- Hypothesis; privileged blocks labelled **always exact / sim-only / diagnostic / never deployable**; obstacle-free caveat (`surfaces` holds only scene/gate-frame/floor entries on open courses). Exact artifact-copy list: teacher run `teacher_it1374`, the noise-model YAML (`…/detector-v1/liftoff-gates-161f7a83/noise_model.yaml`), the Liftoff track-conversion cache under `artifacts/tracks/cache/` (or trackdata to rebuild), with repo-relative paths; plus the driver command. ONE comparison table: rows = teacher, baseline, 6 variants, UC-77 reference; columns = AC8 metrics; ships with teacher (93.8% Infinity / 17.1% held-out) + UC-77 student (0%) filled and variant rows as placeholders (numbers come from the user's GPU run). Ends with the recommendation rubric incl. the explicit "even `teacher_full` collapsing at beta→0 ⇒ obs isn't the problem, retarget the distillation setup (beta schedule/DAgger/loss)".
- Table-fill helper: a small `scripts/` reader over the 7 runs' `eval/*.json` emitting the Markdown rows (no new CLI subcommand).

**11. Tests (AC11)** — new `tests/test_privileged_ablation.py` + additions:
- obs dim per block combination (schema dim + built width == DIM + Σ widths);
- default unchanged (schema()/built obs byte-identical at `privileged=()`; snapshot drops the field — extend `test_obs_schema.py`);
- export + harness refusal with the "diagnostic privileged student" message (extend `test_export_no_swap.py`, `test_student_checkpoint.py`);
- teacher-equality per block — the five slice blocks vs their `build_teacher_obs` slices; **floor non-circular**: open-course state with no candidate scene surfaces → floor sorts to row 0 in-range → assert priv `floor` == `block_slice("obstacles")[0:8]` of REAL `build_teacher_obs` output;
- smoke train 1 iteration with `student_privileged:[altitude]`, mlp-student, CPU (reuse `tests/student_support.py`); assert it trains and the sidecar records privileged + the larger dim.

## Files Affected
**Production (developer):** `training/config.py`, `sim/teacher_obs.py`, `sim/student_env.py`, `controller/student.py`, `sim/student_obs.py`, `training/distill.py`, `training/export.py`, `harness/student.py`, 7× `training/configs/student-liftoff-open-priv-*.yaml`, `scripts/uc79_ablation.sh`, `scripts/` table-fill reader, `docs/experiments/uc79-student-information-gap.md`.
**Test (qa):** `tests/test_privileged_ablation.py` (new) + additions to `test_obs_schema.py`, `test_export_no_swap.py`, `test_student_checkpoint.py`, `test_training_student.py`/`student_support.py`.

## Risks & Considerations
- AC2 floor/obstacles byte-identical hinges on `surface_feature` being a strict op-for-op refactor — guarded by the `test_sim_teacher_obs.py` NumPy pin (challenger's reminder).
- AC3 byte-identical rests on three "add nothing at default" points (config drop, `schema()` no `privileged` key, `_build_obs` unchanged) — each test-locked.
- `teacher_full` width = `dim(teacher_version)` (v1 148 / v2 150); all other blocks share offsets across versions (battery is the v2 tail). Extractor uses `env.obs_version`.
- AC6 equal budget under reboot: equalized via cumulative step `budget`, not wall-clock. If baseline AND all variants read 0%, re-run top candidates at a longer budget before concluding (pitfall, documented).
- Beta schedule identical via `extends: student-liftoff-open` (variants differ only in obs). teacher_full-as-upper-bound handled by the recommendation rubric.
- K1 / deployable harness path (StudentObservationAssembler, `student/obs_ref.py`) untouched — diagnostic checkpoints never reach the harness (refused).

This is the approved plan; ready for the developer.

---

## Challenger verdict (round 2 of 6 — APPROVED)

Round 1 requested revision on 2 Majors + 3 Minors, all on the ACs flagged for scrutiny:
- **AC2** — the `floor` privileged block can't be a slice (it's distance-sorted among all surfaces); the original helper description was internally inconsistent. Fixed via a single shared per-surface encoder (`surface_feature`) used by both `build_teacher_obs` and the floor extractor, with a non-circular equality test (open course → floor sorts to row 0 → assert against real `build_teacher_obs` output).
- **AC6↔AC9** — the budget equalizer was inverted; `max_hours` resets on resume (not equal-budget-safe), only the step `budget` is cumulative. Fixed: identical step budget authoritative, `max_hours` omitted.
- Minors: harness refusal function name/placement (`plan_student`, before `validate_meta`, independent of `--allow-baseline`); export refusal unconditional before the mlp-student gate; `build_student_env` sources privileged from `cfg` at the single env choke point so eval/report inherit it.

Round 2 revision addressed all five correctly; the two Major fixes were verified against the code. Approved plan is sound and ready for the developer. All paths under `/workspace/drone-fly` (in place), scoped to `src/drone_fly/**` production + `tests/**`, plus `training/configs/`, `scripts/`, `docs/experiments/`.
