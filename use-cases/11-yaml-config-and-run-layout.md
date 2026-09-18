# Use Case 11: YAML `--config` for the CLIs + per-run output layout

## Summary
Replace the per-setting flag soup on the `drone-fly` commands with a single **`--config <file>`** YAML. The `train`, `evaluate`, `prune`, and `prune-trained` commands each load **all** their settings from a config file — the previous per-setting flags (`--connectome`, `--timesteps`, `--n-envs`, `--randomize`, `--record*`, `--prune-k`, `--seed`, `--device`, `--resume`, etc.) are **removed** in favour of config keys. Every training config carries a **required `name`**, and all of that run's outputs are written under `training/<name>/` split into `checkpoints/` (step `.zip`s + VecNormalize `.pkl`s), `logs/`, and `recordings/` (activation playback), so runs never collide. Example YAMLs ship and run as-is: prune configs for the **k0, k1, k2** slices and at least one named training config. Resume is expressed in the config (e.g. `resume: latest`), and the existing auto-latest resume logic (merged via PR #12) resolves under `training/<name>/checkpoints/`. This is a deliberate breaking change to the CLI surface — `scripts/train.sh` and the README are migrated to the config workflow in the same change. It composes with UC-10 (the cleaner wipes `training/<name>/`).

## Acceptance Criteria
1. `train`, `evaluate`, `prune`, and `prune-trained` each accept `--config <path.yaml>` and load all settings from it; the previous per-setting flags are no longer accepted on those commands.
2. A `train` config with `name: foo` writes outputs under `training/foo/{checkpoints,logs,recordings}/`, with no collision between differently-named runs.
3. `name` is **required** in a training config; a missing/empty `name` fails with a clear validation error (no flat `artifacts/models` fallback).
4. Example YAMLs are committed and runnable as-is: prune configs producing the **k0, k1, k2** slices, and at least one named training config.
5. Resume is config-driven (e.g. `resume: latest | <path> | null`) and the existing auto-latest resume logic (PR #12) resolves the newest checkpoint under `training/<name>/checkpoints/`.
6. Malformed configs / missing required keys produce clear errors, not opaque stack traces.
7. Behaviour parity: a config reproduces the same run the equivalent old flags would have (faithful migration, not a behaviour change).
8. `scripts/train.sh` and the README are updated to the config workflow; the docs show the example YAMLs. UC-10's `clean` targets the new `training/<name>/` layout.

## Potential Pitfalls & Open Questions
- **Assumption** — `resume` becomes a config key (config-only leaves no `--resume` flag). Confirm at implementation that no per-invocation flag survives except `--config` itself (and `clean`'s operational `--yes/--dry-run/--include-prunes`, which is not a settings command).
- **Assumption** — required inputs for the other commands become config keys: `evaluate` (checkpoint path, episodes), `prune`/`prune-trained` (connectome, out dir, k, checkpoint).
- **Risk** — breaking change: old flag invocations stop working; example configs + `scripts/train.sh` + README must land together so nothing is left dangling.
- **Edge case** — `evaluate`/`prune-trained` outputs (e.g. recordings during eval) should also honour the `training/<name>/` layout when a name applies, to stay consistent with training runs.

## Original Description
Modify the python scripts so they are based on a prepared YAML passed with a `--config` parameter instead of the parameter soup we have now, and have example YAMLs already prepared for the stuff I want to make (e.g. prunes with k0 to k2, train that can be passed a NAME). Training should create a folder with the name put in the yaml attribute where we'd store the steps files and such, plus the recordings (so we don't mix stuff). E.g. `training/<name>/recordings/` and `training/<name>/progress/` or whatever.

## Clarifications
- Q: How should `--config` and the existing CLI flags relate?
  A: Config-only — the per-setting flags are removed; settings live only in the YAML.
- Q: Which commands should get a YAML config schema?
  A: All major commands — train, evaluate, prune, prune-trained.
- Q: What per-run output layout do you want?
  A: `training/<name>/{checkpoints,logs,recordings}/` (checkpoints, logs, and recordings each in their own subdir).
- Q: Is the run `name` mandatory, and what about the old flat default?
  A: Name required — outputs always go to `training/<name>/`; no flat `artifacts/models` fallback.
