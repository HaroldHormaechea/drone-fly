---
schema_version: 1
project:
  name: drone-fly
  maturity_target: prototype
stack:
  languages: [python]
  frameworks: [pytorch, gymnasium, stable-baselines3, gym-pybullet-drones]
  runtimes: [server]
  versions: {python: "3.11", gymnasium: "1.3.0", stable_baselines3: "2.9.0", neuprint_python: "0.6.3"}
  data_stores: ["local filesystem (cached connectome, checkpoints, logs)", "neuprint (remote graph API)"]
build:
  tool: uv
  commands: {test: "uv run pytest", lint: "uv run ruff check .", format: "uv run ruff format ."}
paths:
  production: ["src/drone_fly/**"]
  test: ["tests/**"]
  api_boundary: []
test:
  framework: pytest
  levels: [unit, integration]
  coverage_target: "50%"
profiles: []
deployment:
  provider: self-hosted
  iac: none
  environments: [local]
vcs:
  enabled: true
  already_initialized: true
  default_branch: "main"
  remote: "git@github.com:HaroldHormaechea/drone-fly.git"
use_cases:
  index: "USE_CASES.md"
  folder: "use-cases/"
---

# Project Brief

## Overview

> **Note on domain decisions.** Interactive multi-choice prompting is unavailable in this
> scaffolding pass, so the domain choices below were set to the **recommended default path**
> and recorded explicitly here for visibility. Any of them can be changed later with the
> `revise-brief` skill. The recommendations follow directly from the domain realities of
> MaleCNS, Liftoff, and reinforcement learning (see the three scope decisions).

**Name:** drone-fly

**Problem statement.** Can the wiring diagram of a real fruit-fly brain (the MaleCNS
connectome) be used to build — and reinforcement-learn — a neural controller that flies a
quadrotor drone through a timed waypoint race course in simulation, avoiding floors,
ceilings, and obstacles?

**Primary users.**
- The project owner / researcher — exploring connectome-seeded control as a learning project,
  with limited prior ML/RL experience (so the pipeline and docs must be teaching-oriented).
- ML / computational-neuroscience hobbyists who want a reproducible example of turning a
  connectome into a trainable controller.
- (Aspirational) FPV-drone-racing enthusiasts curious about bio-inspired autopilots.

**Value proposition.** Most RL drone agents use generic neural networks. drone-fly is
differentiated on being **more bio-inspired / more novel**: it seeds the controller's
architecture from an actual insect connectome, making it a concrete, runnable bridge between
a public neuroscience dataset and a working RL flight task — something that otherwise requires
stitching several specialized fields together by hand.

**Maturity target:** **prototype** (recommended). Goal is to prove the idea end-to-end — a
connectome-seeded agent learns to fly a waypoint course in an open sim. Single-user,
research-grade, no reliability or multi-user guarantees.

### Scope decision 1 — Role of the connectome (brain framing)

**Chosen (recommended): connectome-seeded / connectome-constrained neural-net controller.**
MaleCNS is a *static synaptic wiring diagram* — neurons and typed synaptic connections,
queryable via neuPrint. It gives topology/connectivity, **not** tuned neuron dynamics or
synaptic weights. So drone-fly uses the connectome to **shape or constrain** a trainable
neural-network controller (connectivity structure seeded from MaleCNS; connection strengths
learned by RL). We explicitly do **not** promise a biophysically faithful simulated fly brain
— the data does not provide the dynamics that would require, and that is a much harder,
higher-risk research problem.

### Scope decision 2 — Training / "emulation" target (sim)

**Chosen (recommended): an open quadrotor flight-sim environment.** "Liftoff" is a
closed-source commercial FPV racing game with no official public API or mod/sim interface, so
driving it directly would require brittle screen-scraping and input injection. Instead,
drone-fly trains in an **open, Gymnasium-style FPV quadrotor simulator** (e.g.
`gym-pybullet-drones` or a Flightmare-style sim) that models a timed gate/waypoint course.
Achieving Liftoff-level fidelity or actually flying inside Liftoff is recorded as a
**stretch goal / non-goal** for now.

### Scope decision 3 — Training objective

**Chosen (recommended): reinforcement learning for timed waypoint racing.** A quadrotor agent
chases a sequence of waypoints/gates; it is **penalized** for colliding with floor, ceiling,
or obstacles and **rewarded** for minimal door-to-door (gate-to-gate) time. "Racing mode" is a
timed waypoint course. A mainstream RL stack is used (Python + Gymnasium + a PPO/SAC
implementation such as Stable-Baselines3).

**In scope (must deliver).**
1. A neuPrint-backed pipeline to fetch MaleCNS connectivity and derive a connectome-seeded
   controller network topology.
2. An open, Gymnasium-style quadrotor sim environment with a timed waypoint/gate racing task
   and floor/ceiling/obstacle collision penalties.
3. An RL training loop (PPO or SAC via Stable-Baselines3) that trains the connectome-seeded
   controller to fly the course.
4. A reward function optimizing fastest door-to-door (gate-to-gate) times with collision
   avoidance.
5. Teaching-oriented documentation and reproducible run scripts, given the owner's limited
   RL/connectome background.

**Non-goals (explicitly out of scope).**
1. Driving or integrating with the real Liftoff game (no public API) — stretch goal only.
2. A biophysically faithful, dynamics-level simulation of the fly brain.
3. Real (physical) drone hardware / sim-to-real transfer.
4. Multi-user, production-grade reliability or a hosted service.
5. Anything beyond racing/waypoint-following behavior (e.g. freestyle, payload, swarm).

**Success criteria.**
- The connectome-fetch pipeline runs and produces a controller topology derived from MaleCNS
  connectivity (reproducibly).
- The sim environment runs a timed multi-gate course with working collision penalties.
- A training run completes and the trained agent measurably improves over an untrained/random
  baseline — it completes the course and reduces door-to-door time across training.
- The whole pipeline (fetch → build controller → train → evaluate) is runnable from documented
  scripts by someone following the README.

## Monetization

> Recorded as the recommended default for a research/hobby prototype; changeable via
> `revise-brief`.

**Commercial intent:** No. drone-fly is a **non-commercial research / learning artifact**.

**Model:** none (open-source research project).

**License:** **MIT** (recommended default) — permissive, minimal friction for a reproducible
research example and for others to run/fork the pipeline. Can be revised to Apache-2.0 if
explicit patent-grant language is later wanted.

**Target market.**
- The project owner (learning / research use).
- ML and computational-neuroscience hobbyists and students.
- Open-source community interested in bio-inspired RL.

**Tiers:** none (not applicable — no paid packaging).

**Constraints.**
- **Data licensing:** MaleCNS connectome data is provided by Janelia/FlyEM (Google) via
  neuPrint under its own terms; this repo's MIT license covers **our code only**, not the
  upstream dataset. The dataset's citation/usage terms must be honored — do **not** redistribute
  bulk connectome data in the repo; fetch it at runtime and attribute the source.
- No ads, no data resale, no user accounts (single-user research tool).
- "Liftoff" is a third-party trademark; the project neither bundles nor redistributes any
  Liftoff assets and only references it as an aspirational fidelity target.

## Technologies

> Stack set to mainstream, well-supported defaults for a Python ML/RL research prototype.
> Versions pinned from the latest stable releases at scaffold time; changeable via
> `revise-brief`. Plain-language rationale is included because the owner is new to this area.

**Hard constraints.** Python-centric (this is an ML/RL + scientific-computing project). No Java
anywhere — so **no `java-*` profiles apply**. Must be able to fetch MaleCNS connectivity from
neuPrint and train in an open quadrotor sim.

**Primary runtime:** server / local workstation (Python process; ideally a machine with a GPU
for faster RL training, but CPU-runnable for the prototype).

**Languages:** Python (3.11). Single language for the whole pipeline — the neuroscience
(neuPrint), scientific computing (numpy/pandas), sim, and RL ecosystems are all Python-first,
which keeps the project approachable.

**Frameworks / core libraries:**
- **PyTorch** — the neural-network backend. Stable-Baselines3's RL algorithms are implemented
  in PyTorch, and the connectome-seeded controller network is defined here. (Widest ecosystem,
  best learning resources for a newcomer.)
- **Gymnasium (1.3.0)** — the standard RL environment interface. The drone sim is exposed as a
  Gymnasium environment so any RL library can train against it. (Successor to OpenAI Gym; the
  current community standard.)
- **Stable-Baselines3 (2.9.0)** — provides reliable, well-documented PPO and SAC
  implementations. Recommended default algorithm: **PPO** (robust, forgiving of
  hyperparameters, a good first choice for continuous-control tasks like flight); **SAC** kept
  as an alternative if sample-efficiency becomes a concern.
- **gym-pybullet-drones** — the recommended open quadrotor simulator (PyBullet physics,
  Gymnasium-compatible quadrotor dynamics + gates). **Note:** not published on PyPI — installed
  from GitHub (`pip install git+https://github.com/utiasDSL/gym-pybullet-drones`). A
  Flightmare-style sim is the fallback if this proves unsuitable. Recorded as a dependency to
  revisit during architecture.
- **neuprint-python (0.6.3)** — official client for querying the MaleCNS connectome from the
  neuPrint server.
- **numpy / pandas** — array math and tabular handling for connectivity data and rollouts.
- **matplotlib / tensorboard** — training-curve and trajectory visualization (teaching aid).

**Data stores.**
- No database engine. Connectome data is **fetched at runtime from neuPrint** (a hosted Neo4j
  graph exposed via HTTP) and cached to local files.
- Local filesystem is the only "store": cached connectivity (Parquet/CSV), trained model
  checkpoints, and TensorBoard logs under an `artifacts/`-style directory.

**Auth strategy:** none in-app (single-user research tool). The only credential is a
**neuPrint API auth token**, supplied via environment variable / `.env` (never committed).

**External services / APIs:**
- **neuPrint** (Janelia FlyEM / Google) — the MaleCNS connectome query API. Requires a free
  auth token.
- (Aspirational / non-goal) Liftoff — no API; not integrated.

**AI / ML dependency:** Local, self-trained models only — no hosted LLM/model provider. The RL
policy (connectome-seeded PyTorch network) is trained from scratch in-repo via
Stable-Baselines3. No external inference API.

**Build tool:** `uv` (fast, modern Python packaging/venv manager) with a `pyproject.toml`.
Chosen over bare `pip`/`venv` for reproducible, lockfile-backed installs — helpful for a
research repo others should be able to reproduce. `pip` remains usable as a fallback.

## Architecture

> Recorded as the recommended default shape for a single-machine Python research pipeline;
> changeable via `revise-brief`.

**Platforms:** server-only / **CLI**. No web, mobile, or desktop GUI. The project is a set of
command-line entry points plus an importable Python package. Runs on a Linux/macOS workstation
(GPU optional).

**Service shape:** **modular monolith** — one Python package (`drone_fly`) organized into
clear pipeline-stage modules. No network services, no microservices; the "boundaries" are
Python module boundaries, which is right for a single-user research prototype (simplest to
evolve, no distributed-system cost).

**Components (modules under `src/drone_fly/`).**
- `connectome` — queries neuPrint for MaleCNS connectivity and caches it locally. Responsible
  for the only inbound external data.
- `controller` — builds the connectome-seeded neural-network topology (PyTorch) from cached
  connectivity; exposes it as a policy network for the RL agent.
- `env` — the Gymnasium quadrotor racing environment (waypoint/gate course, collision
  penalties, door-to-door timing reward). Wraps `gym-pybullet-drones`.
- `train` — the RL training loop (Stable-Baselines3 PPO/SAC) wiring the controller policy to
  the env; writes checkpoints and TensorBoard logs.
- `evaluate` — loads a checkpoint, runs the agent on a course, reports lap/gate times vs. a
  random baseline, and renders trajectories.
- `cli` — thin command-line entry points (`fetch-connectome`, `train`, `evaluate`) that
  orchestrate the above.

**Communication (all in-process function calls):**
- `cli` → `connectome`, `train`, `evaluate` (Python calls)
- `connectome` → neuPrint (**HTTPS**, via `neuprint-python`)
- `controller` → `connectome` (reads cached connectivity)
- `train` → `env` + `controller` + Stable-Baselines3 (Python calls)
- `evaluate` → `env` + checkpoint files (Python calls)

**Async workloads:** none in the distributed sense. RL training is a **long-running,
synchronous CPU/GPU job** invoked from the CLI (potentially hours); no queues, schedulers, or
background workers. Checkpointing makes long runs resumable.

**Integrations:**
- **neuPrint** (read-only, inbound): MaleCNS connectivity, over HTTPS with an auth token.
- **gym-pybullet-drones / PyBullet** (in-process): the simulator providing quadrotor dynamics
  and gate geometry.
- (Non-goal) Liftoff: no integration.

**Data flow narrative.** `fetch-connectome` calls neuPrint, pulls MaleCNS neuron/synapse
connectivity, and caches it to `data/connectome/` (Parquet/CSV). `controller` reads that cache
to construct the seeded policy-network topology. `train` instantiates the Gymnasium racing
`env` and the seeded policy, runs PPO/SAC, and writes model checkpoints to `artifacts/models/`
and metrics to `artifacts/logs/` (TensorBoard). `evaluate` loads a checkpoint, flies the course
in the env, and emits timing metrics and trajectory plots to `artifacts/eval/`. Everything
stays on the local filesystem.

**Trust boundaries.** The only untrusted/external input is data pulled from neuPrint over the
network; it is treated as read-only reference data and cached locally. The one secret is the
**neuPrint auth token**, supplied via environment variable / uncommitted `.env` — it must never
be committed or logged. There is no user-supplied runtime input, no PII, and nothing sensitive
that must be prevented from leaving the system beyond the token itself. Bulk connectome data is
kept out of version control for licensing reasons.

**Multi-tenancy:** not applicable (single-user local tool).

## Quality & Standards

> Prototype-appropriate guardrails: enough to keep the code reproducible and honest without
> over-engineering a research project. Changeable via `revise-brief`.

**Style guide:** Python language defaults enforced by tooling (PEP 8 via ruff). No bespoke
company style.

**Linters / formatters (Python):**
- **ruff** — both linting (`ruff check`) and formatting (`ruff format`). One fast tool covers
  style, imports, and common bug patterns; the easiest single guardrail for a newcomer.
- **mypy** (optional, lightweight) — type-checking on the core `drone_fly` package; kept
  non-blocking at prototype stage.

**Testing strategy.**
- **Levels:** unit + a small amount of integration. Given the RL/sim nature, most tests are
  fast unit tests around deterministic logic (connectome parsing, controller topology
  construction, reward function, env step contracts). A couple of integration "smoke" tests
  assert the pipeline wires together (e.g., env resets/steps, a few training steps run) — these
  do **not** assert learning performance, which is stochastic and slow.
- **Coverage target:** ~50% (prototype). Focused on the deterministic, testable core; not a
  hard gate. RL training quality is validated empirically via evaluation, not unit coverage.
- **Framework:** **pytest**.
- External calls (neuPrint) are mocked/fixtured in tests — no live network in the test suite.

**Security baseline.**
- **Secrets:** the neuPrint auth token lives only in an uncommitted `.env` / environment
  variable. `.gitignore` excludes `.env`; the token must never be logged or committed.
- **Dependency scanning:** `pip-audit` (or `uv`'s audit) run periodically; Dependabot/Renovate
  optional later.
- No SAST, no auth testing (no in-app auth), **no threat-model document** — disproportionate
  for a single-user local research tool.

**Accessibility target:** not applicable (no UI; CLI + generated plots only).

**Performance budgets.** No hard latency/bundle budgets (offline research tool). Soft goals
only: connectome fetch caches so it runs once; training should be resumable via checkpoints;
prefer GPU when available but remain CPU-runnable for the prototype.

**Documentation:** README plus lightweight Markdown ADRs (`docs/adr/`) for the few consequential
choices (connectome-seeding approach, sim selection, RL algorithm). Teaching-oriented prose,
given the owner's stated limited background. No full docs site.

**Observability:** logs to stdout + **TensorBoard** for training metrics (reward curves,
episode length, gate times). No metrics/tracing stack — overkill for a local job.

## Profiles

None

## Deployment

> Local-first research prototype: there is no hosted production service. "Deployment" here means
> how a run is executed and how CI validates the code. Changeable via `revise-brief`.

### Production

- **Hosting target:** self-hosted / local. drone-fly is not a service — it runs as a
  command-line job on the researcher's own workstation, or optionally on a **rented GPU box**
  (e.g. a cloud VM or Colab-style notebook) for faster training. No always-on deployment.
- **Cloud provider:** none required. (If a GPU VM is later rented, provider choice is
  incidental and out of scope; no `profile-aws-deployment` is attached.)
- **IaC:** none (manual) — appropriate for a single-machine prototype.
- **CI/CD:** **GitHub Actions** — a lint + test workflow on push/PR (`ruff check`, `ruff
  format --check`, `pytest`). No deploy stage (nothing to deploy). Heavy RL training is **not**
  run in CI.
- **Environments:** a single logical environment (`local`). No staging/preview.
- **Secrets:** `.env` file (gitignored) holding the neuPrint auth token, loaded into the
  process environment. No secrets manager needed at this scale.
- **Observability:** stdout logs + TensorBoard logs written under `artifacts/logs/`. No hosted
  sink.
- **Backup / DR:** none for infrastructure. Reproducibility is the safety net — connectome
  cache and model checkpoints are regenerable from the pipeline; important checkpoints can be
  archived manually. Bulk connectome data and large artifacts are gitignored.

### Development

- **Local dev environment:** native toolchain via **uv** (`uv sync` to create the venv from
  `pyproject.toml` + lockfile). Fastest inner loop; reproducible via the lockfile.
- **Containerization:** **optional**. A `Dockerfile` is provided for a reproducible
  CPU-training image (helpful for the rented-GPU-box path and for contributors who don't want
  to install PyTorch/PyBullet natively), but native uv is the default path. Not required.
- **Hot reload:** not applicable (batch jobs, not a server). Fast feedback comes from the unit
  test suite and short "smoke" training runs.
- **Seed data:** **synthetic generator / fixtures**. Tests use small synthetic connectivity
  fixtures and a tiny toy course so they run without hitting neuPrint or doing real training. A
  documented `fetch-connectome` command pulls the real MaleCNS data for actual runs.
- **Database migrations:** not applicable (no database).

## Scaffolding Plan

> Structure-only. No dependencies installed, no builds run, no training executed. Git is
> already initialized (`main`) and `origin` already wired to
> `git@github.com:HaroldHormaechea/drone-fly.git`, so this step does **not** run `git init`,
> add a remote, or push — it only writes tracked files and a `.gitignore`. A local commit will
> be made; the root session owns the push.

### Directories (each holds a `.gitkeep` where otherwise empty)

- `src/drone_fly/` — the package root (production code; `paths.production`).
- `src/drone_fly/connectome/` — neuPrint fetch + local connectivity cache logic.
- `src/drone_fly/controller/` — connectome-seeded PyTorch policy-network construction.
- `src/drone_fly/env/` — Gymnasium quadrotor racing environment.
- `src/drone_fly/train/` — RL training loop (Stable-Baselines3).
- `src/drone_fly/evaluate/` — checkpoint evaluation + trajectory reporting.
- `src/drone_fly/cli/` — command-line entry points.
- `tests/` — pytest suite (`paths.test`).
- `tests/fixtures/` — small synthetic connectivity + toy-course fixtures.
- `docs/adr/` — lightweight Markdown architecture decision records.
- `data/connectome/` — runtime connectome cache (gitignored contents; `.gitkeep` tracked).
- `artifacts/models/` — training checkpoints (gitignored contents; `.gitkeep` tracked).
- `artifacts/logs/` — TensorBoard logs (gitignored contents; `.gitkeep` tracked).
- `artifacts/eval/` — evaluation outputs/plots (gitignored contents; `.gitkeep` tracked).

### Files

- `README.md` — project overview + honest quick-start (generated via `write-readme`).
- `LICENSE` — MIT license text (owner: Harold Hormaechea).
- `pyproject.toml` — project metadata, deps (pytorch, gymnasium, stable-baselines3,
  neuprint-python, numpy, pandas, matplotlib, tensorboard; dev: pytest, ruff, mypy, pip-audit),
  `uv` build backend, ruff/pytest config. Dependencies are declared, **not installed**.
- `.gitignore` — Python + ML ignores (`__pycache__/`, `.venv/`, `*.egg-info/`, `.env`,
  `data/**` except `.gitkeep`, `artifacts/**` except `.gitkeep`, checkpoints, TensorBoard runs,
  `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`).
- `.env.example` — template showing the required `NEUPRINT_TOKEN` / `NEUPRINT_SERVER` vars
  (real `.env` stays uncommitted).
- `src/drone_fly/__init__.py` — package marker with version.
- `src/drone_fly/<subpkg>/__init__.py` — one per subpackage (connectome, controller, env,
  train, evaluate, cli) so imports resolve. Each carries a one-line module docstring describing
  its responsibility (structure-only; no logic).
- `tests/__init__.py` and `tests/test_smoke.py` — a trivial import smoke test so the suite is
  runnable from day one.
- `docs/adr/0001-record-architecture-decisions.md` — seed ADR documenting the ADR practice and
  the three scope decisions (connectome-seeding, open sim, RL/PPO).
- `.github/workflows/ci.yml` — GitHub Actions: `ruff check`, `ruff format --check`, `pytest` on
  push/PR. No deploy job.

### Commands (in order)

1. `mkdir -p` (single call) for the directory tree above.
2. Write all files listed (via the Write tool).
3. Drop `.gitkeep` into otherwise-empty tracked dirs (`data/*`, `artifacts/*`, `tests/fixtures`).
4. `git -C /workspace/drone-fly add -A`
5. `git -C /workspace/drone-fly commit -m "…"` (local only).
6. **No** `git init`, **no** `git remote add`, **no** `git push` — VCS already set up; root
   session pushes.
