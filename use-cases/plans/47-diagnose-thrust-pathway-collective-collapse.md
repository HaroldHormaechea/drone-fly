---
plan_for: use-cases/47-diagnose-thrust-pathway-collective-collapse.md
work_branch: feat/uc-47-diagnose-thrust-pathway-collective-collapse
team: drone-fly-uc-47
approved: 2026-09-22
---

# UC-47 — Implementation Plan (approved)

## Orchestrator decision on the docs/ scope question (issue 6)
**Option (a) chosen.** The developer is authorized to write exactly two repo-root findings
artifacts — `docs/uc47-thrust-pathway-findings.md` and `docs/uc47-thrust-pathway-surface.json`
— as a PLAN_FILE-style coordination-artifact exception, even though they fall outside the
declared production scope `paths.production=["src/drone_fly/**"]`. This exception is limited to
those two paths; all other writes remain bound to `src/drone_fly/**` (developer) and `tests/**`
(QA). The OS-level TARGET_DIR grant already permits the write; this records the role-boundary
authorization so implementation does not stall mid-run.

## Goal
Build a reproducible characterization harness that maps achieved vertical thrust / T-W across command inputs through the open-loop CTBR→RPM mixer, reproduce the recorded free-fall through RaceEnv, and deliver a written CONFIRMED/REFUTED verdict on the collective-desaturation hypothesis + a data-grounded recommendation for the follow-up fix UC. Diagnostic only.

## Key code facts established (drive the design)
- **Mixer** `ctbr_to_rpm` (src/drone_fly/adapter/pybullet_adapter.py:35): per-rotor `rpm = base ± droll ± dpitch ± dyaw` then `np.clip(0, max_rpm)`. `base = hover_rpm + (throttle-0.5)*2*(max_rpm-hover_rpm)`. Differential terms sum to zero, so collective is preserved *only until a rotor clips*; near throttle extremes + commanded rates, over-max rotors clip down uncompensated → asymmetric collective bleed. At rpy=±1.0, rate_gain=0.15: ~21% thrust loss (three rotors 0.85·max, one clipped). For CF2X (HOVER≈14.5k, MAX≈21.7k → baseline T/W≈2.25 at throttle 1.0/rpy 0) that lands ~1.78 — nowhere near the recorded 0.2. Verdict genuinely open; MEASURE it.
- **pybullet adapter** sets RPM directly each control step (no motor-lag / spin-up model); `reconfigure` applies only start + best-effort mass/drag, and IGNORES max_thrust/max_body_rate/latency_steps. So on pybullet, domain-randomized thrust/body-rate/latency do NOT perturb the run; AC4's "models RPM lag?" answer is structurally NO.
- **Secondary suspects** (latency_steps, UC-17 battery ceiling_factor, UC-19 damage authority_factor) are modeled ONLY on SimpleDroneAdapter → AC5 sweeps are simple-backend (also hermetic/CI-gating).
- **Authority lever** `RaceEnv.set_attitude_authority(factor)` (racing_env.py:238) scales rpy channels via clip-then-scale in step (racing_env.py:455); throttle (index 0) never scaled. Reused READ-ONLY as the AC3 measurement knob and the AC6 replay knob — no curriculum change.
- **Recording format** (recorder.py:309): `frames.actions` (per-step raw [throttle,roll,pitch,yaw]), `frames.drone_position`, `meta.course.start` (spawn), `meta.dt`, `meta.dynamics` (mass/drag/thrust/body-rate/latency, present only if randomization was on), `meta.backend/git_sha/checkpoint`. NO authority stored.
- **CRITICAL (issue-2 finding):** both record paths log the RAW pre-authority-scale action (rollout.py:88-95 `capture_frame(action_np)`; evaluator.py:157-164 `capture_frame(action[0])`). The env scales rpy internally afterward. UC-46 committed schedule (start=0.25, anneal_fraction=0.5, total=1e6, n_envs=1) puts episode_350's applied authority at ≈0.26 (≈7k steps into a 500k anneal), consistent with the recorded drift≈0. So replaying at authority=1.0 would be WRONG and would manufacture a false mixer bleed (UC-45 misattribution class).

## Acceptance-criteria mapping
- **AC1** — committed harness runnable on real pybullet via `PYTHONPATH=<WORKDIR>/src /workspace/drone-fly/.venv/bin/python -m drone_fly.diagnostics.thrust_pathway --backend pybullet --out <path>`; randomization pinned OFF + fixed seed + fixed dynamics; emits a machine-readable JSON table of steady-state vertical accel + derived T/W vs command.
- **AC2** baseline sweep — throttle ∈ [0,1], rpy=0; report T/W per throttle; check 0.5 hovers / 1.0 climbs.
- **AC3** rate-coupled sweep (central question) — same throttle grid, rpy=±1.0 at authority factors {0.25,0.5,1.0} via `set_attitude_authority`; quantify T/W(throttle=1, rpy=±1) vs rpy=0 and collective-loss-per-unit-rate. **Verdict rests ONLY on real-pybullet + pure-ctbr_to_rpm analysis — NO simple mirror** (simple has no collective desaturation; its rate→lift loss is cos-tilt, a different mechanism; any simple cell shown must be labeled as such and is not evidence for/against the mixer hypothesis).
- **AC4** oscillation — bang-bang throttle 0↔1 vs steady equal-mean; report sign+magnitude honestly, noting the convexity prediction (throttle=0 → base≈7.2k rpm not clipped to 0; thrust∝RPM² convex ⇒ bang-bang likely *increases* mean thrust); state RPM-is-set-directly / no motor-lag.
- **AC5** secondary suspects — DRIVE (don't assert): latency_steps∈{0,1,2} on simple shows exact step-shift; battery/damage over first ~20 steps show ~0 delta (full charge → ceiling_factor≈1.0) PLUS a forced-low-charge cell to characterize the mechanism; note pybullet doesn't model these.
- **AC6** telemetry tie-back — replay episode_350's recorded RAW actions through RaceEnv (pybullet), spawn+mass/drag from meta, randomization OFF. Determine applied authority from EVIDENCE: (i) reconstruct schedule value (~0.26) as primary estimate; (ii) self-validate by sweeping authority over the anneal range and reporting the factor that best reproduces the recorded **3-D trajectory** (position RMSE all axes). SUCCESS = reproduces BOTH the ~7–8 m/s² descent AND near-zero lateral drift within a stated tolerance; a non-match is itself a reported finding. Never a bare-adapter probe. Re-fetch the recording via curl/HTTP from the Drive folder if needed — never the Drive MCP.
- **AC7** hermetic CI guard — simple-backend + pure-mixer assertions only (no pybullet import): mixer collective bleed direction+magnitude (Σrpm²(rpy=±1) < Σrpm²(rpy=0) strictly, ~15–25% band at rate_gain=0.15); simple baseline hover@0.5 / climb@1.0; latency step-shift; ctbr_to_rpm is pure/stateless (no motor state). Real-pybullet results reported but non-gating.
- **AC8** written verdict — CONFIRMED or REFUTED on collective-desaturation, the measured T/W surface, and a concrete fix-UC recommendation FOLLOWING the data (likely the deferred inner-loop attitude/thrust stabilizer, e.g. gym-pybullet-drones DSLPIDControl — only if the surface supports it). **Expected posture: mixer bleed likely REFUTED as the dominant cause; harness must surface the true mechanism** (off-vertical thrust vectoring from integrated attitude even at low authority / throttle not truly sustained at 1.0 / CF2X thrust mapping).
- **AC9** diagnostic-only invariant — diff adds only harness + tests + docs; no reward/curriculum/init/mixer/training change; full hermetic gate (ruff/mypy/pytest) stays green. Any concrete mixer defect found → deferred to a SEPARATE follow-up fix UC (UC-40→41/42 precedent).

## Files Affected
**Production code (developer):**
- `src/drone_fly/diagnostics/__init__.py` — NEW (package marker)
- `src/drone_fly/diagnostics/thrust_pathway.py` — NEW: rollout helper (returns steady-state vertical accel + T/W + per-cell attitude drift; backend-neutral); baseline/rate-coupled/oscillation/secondary sweeps; `replay_recording(path)`; JSON surface emit; `main(argv)` + `__main__` guard; pybullet import lazy (only under `--backend pybullet`) so the module is hermetic-safe.
- `docs/uc47-thrust-pathway-findings.md` — NEW: written verdict, populated with REAL pybullet numbers after the developer runs the harness via the main venv. **[authorized: orchestrator scope exception, see top of plan]**
- `docs/uc47-thrust-pathway-surface.json` — NEW: committed real-pybullet surface table (travels with the PR). **[authorized: orchestrator scope exception, see top of plan]**

**Test code (QA):**
- `tests/test_uc47_thrust_pathway.py` — NEW: hermetic AC7 assertions (no pybullet import; must not import the module's pybullet path).

Path classification per brief frontmatter: production=`src/drone_fly/**`, test=`tests/**`; docs/* are repo-root non-code artifacts (authorized above).

## Risks
- Verdict may be REFUTED — harness is designed to expose all candidate mechanisms, not just the leading one.
- Real-pybullet non-gating; CI hermetic. Lazy pybullet import mandatory; test module must never import it.
- Determinism: randomization OFF + fixed seed/dynamics for the clean surface; drive latency/battery/damage explicitly for AC5.
- Sweeps need a tall arena / high airborne spawn so floor/ceiling termination doesn't truncate the measurement window; free-fall cases read the first ~10–20 steps.
- episode_350 replay strictly through RaceEnv; authority reconstructed+self-validated, never assumed 1.0.

## Challenger final verdict
APPROVED after one revision round. Diagnostic-only invariant respected. Round-1 Major blockers fixed: (1) backend-mechanism conflation removed from AC3 (verdict rests on pybullet + pure-mixer only); (2) episode_350 replay fidelity — recordings log RAW pre-authority-scale action, so replay reconstructs applied authority (~0.26) and self-validates against the full 3-D trajectory rather than assuming authority=1.0. Verdict reframed to expect the collective-desaturation hypothesis REFUTED as the dominant cause; harness designed to surface the true mechanism. Clear to implement.
