---
plan_for: use-cases/88-controller-realistic-slew-limiter.md
work_branch: feat/uc-88-controller-realistic-slew-limiter
team: drone-fly-uc-88
approved: 2026-10-19
---

# UC-88 controller-realistic slew-rate limiter — APPROVED plan (analyst↔challenger v2)

## Summary
Add a stateful per-axis slew-rate limiter (action filter) shared by the sim action path and the harness command path, so the commanded sticks cannot move faster than a real transmitter. Default-OFF (byte-identical to UC-87). Retrain is the orchestrator's post-merge GPU job (AC6) — dev-team ships code + config + CPU-hermetic tests only.

## Design (agreed)

**Limiter space.** Clamp the PHYSICAL applied command: `collective_g` (g) and the 3 body `rates` (rad/s). Per-axis full range: rates `2·max_rate_rad_s`, collective `max_g`. "Full command-range traversal in T" (default T=0.25s, configurable; disabled when unset) ⇒ per-axis per-step cap = `full_range_axis · dt / T`. `dt`: sim = per-step `timing.tick()` dt; harness = `1/rate_hz` period.

**Shared impl in `src/drone_fly/sim/action.py`** (already the torch+numpy parity-tested contract module):
- `SlewSpec` (frozen): `traversal_time_s: float | None` (None ⇒ disabled); optional per-axis override later.
- `slew_clamp(prev_cmd, new_cmd, dt, spec, action_spec)` — mirror torch (sim) + numpy (harness).
- `SlewLimiter` — stateful wrapper holding the command-space carry state + `reset()`, zero-seeded.
- `raw_of_command(cmd, action_spec)` — effective-raw inverse: collective `a0' = (cg'−1)/collective_scale_g` (linear exact); rates `a_i' = atanh(clip(rate_i'/max_rate_rad_s, −1+ε, 1−ε))` (symmetric ε-guard; `rate'` always in range).

**AC4 — three distinct state pieces (the crux):**
- (a) `prev_applied_command = (cg,rates)` — the limiter's command-space carry state (enabled only).
- (b) `prev_applied_raw = a' = raw_of_command(clamped cmd)` — NEW, drives ONLY the policy memory block. ≡ raw `a` when OFF (bypass) → byte-identical; = `a'` when ON → AC4 ("policy observes what was actually applied").
- (c) `prev_action = raw intent a` — UNCHANGED from UC-87, drives the action-rate + saturation penalty ONLY. Reward path provably untouched on and off. This keeps the intent→intent penalty gradient that teaches the policy to stay under the cap instead of letting the limiter clip it.

## Files Affected

**Production (developer; `src/drone_fly/**`):**
- `sim/action.py` — `SlewSpec`, `slew_clamp` (torch+numpy), `SlewLimiter`, `raw_of_command`; sidecar read/write of an `action_filter` key.
- `sim/course.py` — new `slew_time_s: float|None=None` config field + `__post_init__` validation; add `prev_applied_raw` + limiter command-space state (keep `prev_action` as-is, L479); `_memory_block` (L927) first element → `prev_applied_raw`; in `step_command` apply the clamp AFTER arming (L970) BEFORE `plant.step` (L976), compute `a'`, extend `_update_memory` to set `prev_applied_raw`; penalty block (L1033-44) UNCHANGED; `_spawn` (L713) zero the two new states.
- `harness/runner.py` — reorder L662-679: act → `decode_action` → `slew.step(command, period)` → `mem.observe(obs, a')` (a' = raw if off else `raw_of_command(command)`) → `sink.send(command)`; `slew.reset()` beside `mem.reset()` at L519 + L595; new ctor arg; optional `record.command_raw` (None when off).
- `harness/session.py` — build `SlewLimiter` from the checkpoint sidecar `action_filter` spec + optional `RunnerSettings` override; wire into `HarnessRunner`.
- `harness/policy.py` — expose the sidecar `action_filter` spec from `ContractPolicyRunner.load` (like `action_spec`).
- `harness/config.py` — `RunnerSettings` optional slew-override field + `from_config` (~L197-255).
- `training/run_dir.py` + `training/distill.py` — pass the `SlewSpec` into `write_sidecar(extra=…)` at retrain so deploy==training by construction.
- (maybe) `harness/types.py` — optional `command_raw` on the record.

**Test (QA; `tests/**`):**
- Limiter clamp: at cap, at zero Δ, monotone ramp reaches target in ~T, per-axis caps (rates vs collective).
- Boundary reset: zero-seed, first-armed-step ramps from 0, both sim + harness.
- Sim↔harness parity: identical post-arming command sequence ⇒ identical post-limiter command AND identical post-limiter `prev_applied_raw` — with FIXED dt (sim jitter/DR off so sim dt == harness period) in the ARMED regime.
- `raw_of_command` inverse round-trip (torch + numpy, near-saturation edge, decode parity tolerance).
- Default-off byte-identical: sim env step + harness decode/observe unchanged; old checkpoint without `action_filter`; old `CourseConfig` loads disabled; `command_raw` None-path.
- Config validation (bad `traversal_time_s`).

## Notes for the deliverable
- Deploy-only fallback (cap on the existing UC-87 checkpoint) is capped-but-OOD — the policy sees `a'` memory it never trained on when the cap binds; robust fix is retrain (UC pitfall). Flag for the user, don't "fix."
- Slew kept OUT of the fixed `EPISODE_TAIL` log tensor (preserves log-reader parity); surface via `metric()`/info when enabled.
- Completion-vs-cap tradeoff (too-tight cap → sluggish): cap is configurable; the orchestrator reports the completion impact post-retrain (AC6).

## AC coverage
AC1-5,7 in-scope and addressed; AC6 (empirical retrain slew-cap / HF-power / completion) left to the orchestrator.

## Challenger verdict
APPROVED v2 (round 2). The penalty/memory decoupling (three state pieces) resolves the AC4 crux: memory sees the applied command (so the policy observes reality) while the penalty keeps the intent→intent gradient (so it learns to stay under the cap rather than let the limiter clip). Shared action.py impl → sim↔harness parity (mandatory parity test, fixed dt, armed regime). Default-off provably byte-identical (test-covered). Command-space clamp matches "full range in 0.25s". Scope clean: code+config+tests; retrain left to the orchestrator; UC-87 infra reused.
