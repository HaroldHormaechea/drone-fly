---
plan_for: use-cases/03-start-gate-finish-flight.md
work_branch: feat/uc-03-start-gate-finish-flight
team: drone-fly-uc-3
approved: 2026-09-16
---

# UC-03 Approved Implementation Plan — start→gate→finish flight training

Analyst↔challenger agreement (challenger APPROVED after one revision round; all 4 Majors resolved).
TARGET_DIR = /workspace/drone-fly-uc-03-start-gate-finish-flight. Prose only (developer implements).

# Analysis
UC-03 is the first real flight task, built on merged UC-01 (offline connectome load; locked 12-d obs
/ 4-ch action contract) and UC-02 (hardened `SparseConnectomeLayer` → `ConnectomeActorNetwork` →
`ConnectomeFeaturesExtractor` SB3 seam). The network already exists — UC-03 **wires it into PPO,
does not re-implement it**.

Load-bearing codebase facts (verified against the worktree):
- `controller/encoding.py:34-46` — `OBS_DIM=12`, `ACTION_DIM=4`, `ACTION_LAYOUT=(THROTTLE,ROLL,PITCH,YAW)`,
  throttle sigmoid[0,1], attitude tanh[-1,1]. Shapes locked — do not change.
- `controller/sb3.py:31-70` — `ConnectomeFeaturesExtractor(BaseFeaturesExtractor)`,
  `features_dim=ACTION_DIM=4`; docstring gives the exact `policy_kwargs` wiring.
- `controller/actor.py` — `ConnectomeActorNetwork.forward` `(B,OBS_DIM)→(B,ACTION_DIM)`; default
  `propagation_mode="scatter"` (autograd-stable, MPS-friendly).
- `controller/populations.py:326,329` — `MOTOR_POP_SIZE=16`, `SENSORY_POP_SIZE=32`. Committed fixture
  has 18 `descending_neuron` + 35 `visual_projection` + full `sign` column → actor runs in biological
  mode with a real E/I mask on the fixture; smoke/resume tests exercise the real substrate path.
- `controller/policy.py:263-283` — guarded-fallback precedent (dead axonweave hook → in-repo shim);
  the adapter reuses this philosophy.
- `connectome/loader.py:492-497` + `tests/conftest.py` `no_network` — auto-skip/offline idiom to reuse.
- `.github/workflows/ci.yml` — CI runs only `uv sync --extra dev` then ruff + `uv run pytest`; NO
  pybullet. CI + the entire default suite MUST run without pybullet importable. Most load-bearing constraint.
- Empty stubs `env/`, `train/`, `evaluate/`, `cli/`; `scripts/` has only `build_test_fixture.py`;
  `pyproject.toml` declares `drone-fly = "drone_fly.cli:main"` (so `cli.main` must exist). Sandbox is
  uv-only (`uv` at ~/.local/bin; python via `uv python`).

# Proposed Solution

## Core design: sim-agnostic adapter with a pybullet-free fallback (resolves AC9)
A thin sim-agnostic `adapter` (canonical CTBR action + standardized obs ↔ native sim API), two impls:
- **`PyBulletAdapter`** — wraps `gym-pybullet-drones`, guarded/lazy import (like the axonweave hook).
  Canonical CTBR → native (CTBR passthrough or CTBR→RPM, whichever the pinned sim exposes; unit-tested).
  For the owner's dev-time mastery run.
- **`SimpleDroneAdapter`** — lightweight fixed-dynamics analytic point-mass/rate model in pure
  numpy/torch, no pybullet, deterministic + seedable, floor/ceiling contact. What CI smoke-train and
  every default test uses → fully hermetic/offline.

`make_env(..., adapter="auto")` picks PyBullet if importable else Simple, logging loudly which backend
is active. `smoke-train` forces `adapter="simple"`; mastery `train` defaults to `"auto"`. Documented:
the ≥80% mastery bar (AC10) is defined against real PyBullet dynamics; `SimpleDroneAdapter` exists only
for hermetic correctness/CI — never presented as mastery physics. CI-green ≠ mastery-achieved.

## Files to create/modify (prose; no code)

**Adapter package (`src/drone_fly/adapter/`, new):**
- `__init__.py` — exports base + implementations + `make_adapter`.
- `base.py` — abstract `DroneAdapter`: `reset(seed)`, `step(canonical_action)->state`, accessors for
  position/velocity/attitude/angular-velocity + a collision flag. Documents the canonical 4-ch CTBR
  action (throttle[0,1], roll/pitch/yaw body-rates[-1,1]) and the standardized state → 12-d obs.
- `simple.py` — `SimpleDroneAdapter`: fixed-dynamics integrator (mass, gravity, drag, rate→attitude
  response as documented constants). Deterministic; floor/ceiling detection.
- `pybullet_adapter.py` — `PyBulletAdapter`: guarded `gym_pybullet_drones` import; single-drone env;
  canonical CTBR ↔ native mapping (CTBR→RPM if native is RPM); sim-state → standardized state.
  Actionable error if pybullet/sim absent (never a cryptic mid-run crash).

**Env package (`src/drone_fly/env/`, new):**
- `config.py` — documented tunable course/episode constants: start/gate/finish positions, gate
  aperture, arena bounds (floor z, ceiling z), max steps (timeout), sim dt.
- `geometry.py` — pure waypoint logic (AC2): `to_gate → to_finish → done` phase machine;
  `gate_passed(prev,curr)` and `finish_crossed(prev,curr)` as segment/plane-crossing tests, gate
  honoring aperture. Gate-before-finish enforced (finish while still `to_gate` does not complete).
  Pure, hand-transition testable.
- `reward.py` — pure reward (AC3): per-step time penalty (faster→higher return), completion bonus,
  explicit strong floor/ceiling penalties. Receives the geometry phase / valid-completion signal as an
  explicit pure-fn arg, so the completion bonus can only fire on a valid gate-then-finish completion,
  never on finish-before-gate. Testable incl. "shortcut through floor" scoring worse than a valid completion.
- `racing_env.py` — `RaceEnv(gymnasium.Env)`: obs `Box(shape=(OBS_DIM,))` (relative next-waypoint pose
  (3) + attitude (3) + linear vel (3) + angular vel (3)); action `Box(low=[0,-1,-1,-1],high=[1,1,1,1])`.
  `reset(seed)`/`step` drive the chosen adapter, call `geometry` + `reward`, terminate on completion /
  floor-or-ceiling collision / timeout. Fixed dynamics asserted/documented (AC11); deterministic for
  fixed seed (AC12). `make_env(...)` factory + VecNormalize-wrapped VecEnv builder.

**Train package (`src/drone_fly/train/`, new):**
- `device.py` — `resolve_device(override=None)` (AC8). Rule (MPS never auto-selected on Apple Silicon):
  explicit override wins; else CUDA → `cuda`; else MPS available but no opt-in → `cpu` (deliberate:
  sparse-op MPS gaps + CPU-bound PyBullet); MPS only via `override="mps"`, which also sets
  `PYTORCH_ENABLE_MPS_FALLBACK=1`; final fallback `cpu`. Documented Apple-Silicon note. Unit-testable
  without a GPU (monkeypatch `cuda.is_available`/`mps.is_available`).
- `config.py` — `TrainConfig` dataclass + tunable constants: total timesteps, checkpoint freq N, PPO
  hyperparams, `MASTERY_THRESHOLD=0.80`, `EVAL_EPISODES=20`, seed. (AC10 threshold + episode count are
  configurable constants here.)
- `loop.py` — PPO entrypoint (AC4/AC6): VecNormalize-wrapped vec env; `PPO` wiring
  `ConnectomeFeaturesExtractor` via `policy_kwargs` with `net_arch=dict(pi=[], vf=[...])` (documented
  decision keeping the 4-ch connectome features load-bearing; developer may tune `vf`);
  `CheckpointCallback(save_freq=N, save_vecnormalize=True)` → `artifacts/models/`; TensorBoard AND CSV
  → `artifacts/logs/`. `--resume <ckpt>`: `PPO.load(ckpt, env=venv)` + `VecNormalize.load(<stats>.pkl,
  venv)` + `set_env` → `learn(reset_num_timesteps=False)` so `num_timesteps` continues from the
  checkpoint. `smoke_train(...)`: a few steps, `adapter="simple"` (the CI path). Reconcile action bounds
  vs PPO's Gaussian head via env action-space + adapter clipping.
- `evaluate/evaluator.py` (new) — `evaluate_checkpoint(ckpt, episodes=EVAL_EPISODES, seed=...)`
  (AC5/AC10/AC12): load model + VecNormalize stats (`training=False`, `norm_reward=False`); N
  deterministic episodes (`predict(deterministic=True)`). Metrics dataclass with pinned semantics:
  `completion_rate` = completed/total over all N; `mean_completion_time` = mean start→gate→finish over
  completed episodes only; `completed_count` + `n_episodes` disclosed; `mean_completion_time` None/flagged
  when `completed_count==0`. Timed-out/crashed = non-completions, excluded from mean time. Deterministic
  for fixed seed + given checkpoint (best-effort across resume, documented).

**CLI (`src/drone_fly/cli/__init__.py`, edit):** `main()` argparse dispatch: `train`
(`--resume`,`--device`,`--timesteps`), `evaluate` (`--checkpoint`,`--episodes`), `smoke-train`. Thin.
(`fetch-connectome` may stay a documented stub.)

**Bootstrap (`scripts/train.sh`, new) — AC7, concrete verification required:**
- (a) Resolve the pin: `git ls-remote https://github.com/utiasDSL/gym-pybullet-drones <ref>` must return
  a SHA before hardcoding the exact `git+…@<commit-or-tag>` ref — no floating `main`.
- (b) Attempt the real install once, here: `uv venv` + `uv pip install "git+…@<ref>"` + `pybullet` to
  the extent the sandbox allows; capture success or the specific failure (e.g. native-build toolchain gap).
- (c) Tested-vs-untested boundary table in BOTH PLAN_FILE and README: which bootstrap steps were
  executed/verified in this Linux sandbox vs deferred to the owner's macOS M4, and why. `train.sh`
  hard-fails with an actionable message if `import pybullet` fails; idempotent (re-run resumes from
  latest checkpoint under `artifacts/models/`). "Looks correct" is not acceptable — the AC7 QA gate is
  that the boundary doc reflects a real install attempt.

**Docs/config edits:**
- `README.md` — document train/evaluate/bootstrap commands, device auto-detect + Apple-Silicon note,
  the mastery command + ≥80%/20-eps bar (goal, not CI), fixed-dynamics deferral, SimpleDroneAdapter-vs-
  PyBullet distinction, the tested-vs-untested bootstrap boundary, and smoke-≠-mastery scoping. (Refresh
  README at the end of every use case.)
- `pyproject.toml` — add an optional `sim` extra (e.g. `pybullet`) + document the git-only
  `gym-pybullet-drones` install. CI stays on `--extra dev` only; smoke-train must not need `sim`.

# Files Affected
**Production (developer) — `src/drone_fly/**`, `scripts/**`, docs/config:**
- `src/drone_fly/adapter/{__init__,base,simple,pybullet_adapter}.py` (new)
- `src/drone_fly/env/{config,geometry,reward,racing_env}.py` (new)
- `src/drone_fly/train/{device,config,loop}.py` (new)
- `src/drone_fly/evaluate/evaluator.py` (new)
- `src/drone_fly/cli/__init__.py` (edit)
- `scripts/train.sh` (new)
- `README.md`, `pyproject.toml` (edit)

**Test (qa) — `tests/**`:**
- `test_waypoint_logic.py` — AC2 (gate-before-finish; pass/cross detection on hand-built transitions)
- `test_reward.py` — AC3 (faster→higher return; collision penalized; shortcut-through-floor penalized;
  bonus only on valid completion)
- `test_adapter.py` — CTBR→RPM mapping (guarded/skipped if pybullet absent) + `SimpleDroneAdapter`
  determinism & floor/ceiling detection
- `test_env_contract.py` — AC1/AC11/AC12 (spaces & obs shape; termination; fixed dynamics; seed
  determinism) via `SimpleDroneAdapter`
- `test_device.py` — AC8: cuda→cuda; mps-available + no override → cpu; override="mps" → mps AND
  `PYTORCH_ENABLE_MPS_FALLBACK=="1"`; all-false → cpu (anti-naive-ordering regression)
- `test_train_resume.py` — AC6/AC9 real integration (no mocks/no stubbed policy): real
  `ConnectomeFeaturesExtractor` + `PPO` + `SimpleDroneAdapter`; assert `num_timesteps ==
  checkpoint_steps + Δ` (strictly >); reloaded `obs_rms.mean`/`.var` equal saved element-wise via
  `np.testing.assert_allclose`; finite throughout; hermetic under `--extra dev`
- `test_evaluate.py` — AC5/AC12 (completion_rate over all N; mean time over completed only;
  completed_count disclosed; deterministic for fixed seed)
- `test_cli.py` — smoke dispatch of train/evaluate/smoke-train (optional)
- `tests/conftest.py` (edit) — optional shared tiny-env / short-smoke fixtures reusing `connectome`

# Risks & Considerations
1. **AC9 hermeticity (top risk).** CI is `--extra dev` only; no evidence pybullet builds in this uv-only
   sandbox. `SimpleDroneAdapter` makes CI + default tests pass without pybullet; pybullet tests auto-skip.
   Bootstrap correctness (AC7) is decoupled from CI-green.
2. **SB3 wiring / connectome load-bearing.** `net_arch=dict(pi=[], vf=[...])` is the documented decision;
   SB3 still adds a final linear action net + log_std regardless.
3. **Action bounds vs PPO Gaussian head** — env action-space + adapter clipping (developer finalizes).
4. **Mastery reachability (AC10).** If PPO plateaus <80%, surface the budget-vs-bar tradeoff
   (threshold/timesteps/course-difficulty are `train/config.py` constants) — never run unbounded. CI never
   gates on it.
5. **Reproducibility (AC12).** Fixed seed → deterministic eval for a given checkpoint on CPU;
   bit-reproducibility across a resume boundary is best-effort (on-policy PPO, no replay buffer). Documented.
6. **Scope guard (AC11).** No domain randomization, no obstacles beyond floor/ceiling, no multi-gate —
   deferred, asserted/documented.

# Challenger carry-forward notes for developer/QA (advisory)
- If the sandbox pybullet install fails on a native-toolchain gap (likely), that is an acceptable AC7
  outcome PROVIDED the boundary doc records the exact failure AND the `git ls-remote` ref resolution still
  succeeded. Do not let a native-build failure degrade into an unpinned/unverified script.
- The resume test's `obs_rms` equality must use `np.testing.assert_allclose` (not bit-exact `==`) to
  survive pickle round-trip float representation.
