# Use Case 76: Turnkey "release-for-liftoff" run bundle (download, run one script, fly)

## Summary
Add a sim-scoped release generator, **`release-for-liftoff`** (tag track `liftoff-v*`). It packages everything an end user needs to fly a chosen drone-fly student model in Liftoff, with **no source checkout and no training data**. The bundle holds:
- the student export;
- the K1 connectome slice the export depends on (the export omits the `k1.body.layer.*` weights and rebuilds them from `k1_path`);
- the matching learned gate detector;
- a frozen, pre-validated harness config: the `liftoff` preset, plus the UC74/UC75 pointer block, plus the detector path. This is the overlay that had to be hand-built on 2026-10-15, because `liftoff_arena` refuses the gremlin-micro student;
- a pinned run-only install of drone-fly and its runtime deps;
- **one PowerShell entry script**.

The entry script:
- checks prerequisites: anti-cheat, the Liftoff telemetry config, ViGEmBus, a CUDA GPU and the display mode;
- starts the long-lived pad server and waits for it to be ready;
- launches Liftoff, selects the drone and level, and **confirms the drone is at that level's spawn before arming** (on 2026-10-15 a flight ran on BandoCity because "Last Session" was trusted);
- flies N attempts with the recordable overlay and saves Liftoff's recordings;
- cleans up.

Model/config coherence is checked **when the bundle is built**, using the same checks the live harness uses (`plan_student` + `check_student_config`). A bundle that would be refused at arming can therefore never be published. Release notes come from `generate-release` on the `liftoff-v*` track. The generator has a sim-agnostic core with a Liftoff adapter, so another sim can add its own `release-for-<sim>` track later. All of it is testable in CI without Liftoff, Windows or a GPU.

## Acceptance Criteria
1. **Generator command.** `drone-fly release build --sim liftoff --checkpoint <student export> [--level infinity-loop] --out <dir>` produces a versioned bundle `drone-fly-liftoff-<version>/`. A `release-for-liftoff` alias may be added. An unknown `--sim` exits 2 and lists the registered sims.
2. **Manifest.** The bundle has a `manifest.json` listing every file with its path, size, sha256, role (model / connectome / detector / config / runtime / script) and provenance. Provenance covers:
   - the student sha256 and run id;
   - the detector id (e.g. `161f7a83`);
   - the K1 prune provenance;
   - the git commit and drone-fly version.

   The entry script checks every sha256 before it flies and refuses on any mismatch.
3. **Build-time coherence gate.** Packaging runs `plan_student` and `check_student_config` against the frozen config. It refuses (non-zero exit, nothing written) on any of these:
   - a baseline export;
   - a camera, drone or battery-cell mismatch, which is exactly what `--preset liftoff_arena` triggers today with the gremlin-micro student;
   - a non-learned gate estimator;
   - a detector whose sha differs from the one the student's noise model describes. Here the live `noise_model_detector_mismatch` warning becomes an error.

   The resolved config is written into the bundle as `config.yaml`.
4. **Self-contained paths.** The bundled run resolves the K1, detector and config relative to the bundle root, without `find_repo_root()`. A test runs the CLI from an empty temp cwd outside any checkout and confirms all three resolve. No release flow depends on `artifacts/` or `gpu_prototype/`.
5. **No source checkout.** The first-run install, `Install-DroneFly.ps1` (or the entry script's first step), works like this:
   - a bundled `uv.exe` installs a pinned Python 3.11;
   - it installs the bundled drone-fly wheel plus a lock of the run-only deps into a bundle-local `runtime\`;
   - it installs CUDA torch from the PyTorch cu124 index.

   It is idempotent (a re-run is a no-op), never replaces a CUDA torch with a CPU one, and verifies `torch.cuda.is_available()`.
6. **One entry script.** `Run-DroneFly.ps1 [-Attempts N] [-RecordEvery K] [-Level <name>] [-Quit]` runs these steps in order, each failing with an actionable message and a documented exit code:
   - (a) anti-cheat guard;
   - (b) checks that `TelemetryConfiguration.json` has the endpoint and a StreamFormat that includes Battery, offering to write it with the user's consent;
   - (c) ViGEmBus present, guiding the one-time UAC install otherwise;
   - (d) CUDA available, else refuse (CPU misses the 25 ms tick);
   - (e) display checks: borderless 2560x1080 where detectable, otherwise a printed checklist (Arm=A/Disarm=B, deadzone 0);
   - (f) starts `harness pad` and waits until it is ready, with a bounded timeout and retry;
   - (g) launches Liftoff and selects the drone and level via Quick Setup, not "Last Session";
   - (h) flies the attempts with the recordable overlay and saves recordings;
   - (i) disarms, stops the overlay and pad, and quits Liftoff with `-Quit`.
7. **Level verification before arming.** Before the first command the runner confirms the drone is at the level's `QUICK_SETUP_LEVELS` spawn within `SPAWN_TOLERANCE_M`. On a mismatch it refuses with a distinct exit code and never arms. A unit test feeds wrong-track telemetry (a BandoCity spawn) and expects the refusal.
8. **Clean shutdown.** After the last attempt and `--save-recording`, the run process exits within a bounded time; the recordable overlay never keeps it alive (the hang observed on 2026-10-15). A regression test covers the shutdown path.
9. **Distribution.**
   - A GitHub Actions workflow is triggered by a `liftoff-v*` tag or `workflow_dispatch`. It attaches the bundle as release assets, each under 2 GiB (torch is fetched at install, not shipped).
   - It creates the release as a **draft**; publishing is a manual maintainer step.
   - It runs `generate-release` scoped to the release paths for the notes.
10. **Extensibility seam.** A `SimReleaseTarget` registry defines prerequisites, artifacts, config freezing, level verification and the entry-script template; Liftoff is the only entry. A test registers a fake sim and builds a bundle with it.
11. **CI-hermetic tests.** In Linux CI, with no Liftoff, GPU or network, tests cover:
    - manifest/sha generation and verification;
    - the build-time refusals;
    - bundle-relative path resolution;
    - prerequisite-check logic (mocked process lists and files);
    - level-verify logic;
    - a syntax or PSScriptAnalyzer check of the `.ps1` scripts (skipped when the tool is absent).
12. **Docs.** An end-user `README-RUN.md` inside the bundle (requirements, the one-time ViGEm UAC prompt, Liftoff settings) and a maintainer section under `docs/harness/`.

## Potential Pitfalls & Open Questions
- **Missing input (redistribution licence)** — whether the K1 slice (24 MB, derived from MaleCNS / neuPrint) and the detector weights (47 MB, trained on Liftoff frames) may be published openly is unconfirmed. Releases are created as **drafts** (AC9) until the maintainer confirms the licence and attribution. The detectors already ship publicly via `liftoff-gates-v1`.
- **Risk (no model worth shipping yet)** — the current students are weak: run-2 has 1.8% held-out completion, run2b 0.9%, and both crash within about 10 s live. The generator takes a required `--checkpoint` with no default; whether to add a minimum-completion gate is left to the implementation.
- **Risk** — the ViGEmBus installer needs admin/UAC, so it cannot run silently.
- **Risk** — UC64's display assumptions (2560x1080 borderless, monitor 1) and the HUD/pointer ROI do not carry over to other resolutions. The script refuses or warns when the screen does not match.
- **Edge case** — the pad server is slow on first connect (seen 2026-10-15). Readiness must be awaited explicitly (AC6f).
- **Edge case** — a connected physical controller may be read by Liftoff instead of the virtual pad.
- **Assumption** — the noise-model folder and the `transformers` / `liftoff-extract` deps are not needed at run time (free_space is `none` for students), so they stay out of the bundle.
- **Risk** — `scripts/setup-sim-windows.ps1`, which the docs reference, was deleted in `cc817f4`. Its cu124 re-pin logic (`87e4b73`) has to be rebuilt into the installer.
- **Assumption** — the level is a `-Level` parameter limited to `QUICK_SETUP_LEVELS`, defaulting to `infinity-loop` (the only level the UC74 pointer ROI is calibrated for). Uncalibrated levels warn.
- **Ambiguity** — the version scheme for `liftoff-v*` (semver vs. model-run id) and whether one release may hold several models. Default: semver, one model per release.

## Original Description
> Use a subagent to create an UC to make a release-generator that packages all the required stuff to be able to RUN the model in liftoff (and the release process should be something like release-for-liftoff, in case we add more sims later). So uesrs can just download that release, not the code, and be able to execute a script to run everything, including overlay, controls, etc

## Clarifications
The user asked Claude to proceed autonomously (2026-10-15). The open decisions were resolved with conservative defaults instead of a question round:
- Q: Which model ships?
  A: None by default. `--checkpoint` is required, because no current student flies well enough to be a default.
- Q: How does CUDA torch reach the user?
  A: It is installed at first run from the PyTorch cu124 index (bundle assets stay under 2 GiB). CPU is rejected because the K1 student takes about 220 ms per tick on CPU against the 25 ms tick.
- Q: Python runtime?
  A: A bundled `uv.exe` installs a pinned Python 3.11 plus the bundled wheel. PyInstaller was rejected as fragile with torch, PySide6 and vgamepad.
- Q: May the assets be published publicly?
  A: Unconfirmed. Releases are drafts; publishing is a manual step.
- Q: Which level runs?
  A: `-Level`, limited to `QUICK_SETUP_LEVELS`, default `infinity-loop`, with a warning for uncalibrated levels.
