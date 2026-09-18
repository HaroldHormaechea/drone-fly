# Use Case 14: Default `prune` to the full auto-downloaded MaleCNS connectome

## Summary
Change the default connectome for the pruning workflow from the committed ~300-neuron fixture to the **full MaleCNS connectome**. When the full connectome isn't present in the default location (`DRONE_FLY_CONNECTOME_DIR`, else `data/connectome`), `drone-fly prune` **auto-downloads** it from the public CC-BY `connectome_data_prep` source (implementing what is today only a `fetch-connectome` stub), then reuses it on later runs. The committed `tests/fixtures` connectome is **retained but only used when explicitly named as the target** — no command silently defaults to it; CI and `smoke-train` use it precisely because they pass it explicitly. The shipped example prune configs move to the new default; prune keeps hermetic CI coverage via a fixture-targeted test. Scope is the prune ("creating pruned versions") workflow only — `train`/`evaluate` defaults are unchanged.

## Acceptance Criteria
1. `drone-fly prune` with no explicit connectome resolves to the **full MaleCNS connectome** (the `DRONE_FLY_CONNECTOME_DIR` / `data/connectome` default), never `tests/fixtures`.
2. If the full connectome is absent from the default location, `prune` **auto-downloads** it (writing the `.npz` + correctly-named `<npz-stem>_meta.csv`) before pruning; if already present, it is reused with no re-download.
3. The download is actually implemented (the `fetch-connectome` stub becomes a real, independently-invokable download of the full connectome).
4. `tests/fixtures` is used **only when explicitly targeted** — via a config `connectome: tests/fixtures` or `smoke-train --connectome tests/fixtures`. No command resolves to it by default.
5. Hermetic CI stays green offline: the pytest suite + `smoke-train` explicitly target the fixture, and a **fixture-targeted prune test** preserves prune-logic coverage without network.
6. The example prune configs (`configs/prune/{k0,k1,k2}.yaml`) reflect the new default (full connectome).
7. Download failure / no network produces a clear, actionable error (no stack trace) and never leaves a partial or corrupt connectome behind (atomic write / temp-then-move; integrity check before success).
8. README/docs are updated so the documented prune workflow matches the new default; `train`/`evaluate` defaults are unchanged.

## Potential Pitfalls & Open Questions
- **Assumption** — the download destination is the `DRONE_FLY_CONNECTOME_DIR` / `data/connectome` default; the download-if-absent-else-reuse decision keys on the expected files already existing there.
- **Assumption** — the source is the public `connectome_data_prep` MaleCNS URLs the README already documents; the analyst confirms the exact URLs and the required `<npz-stem>_meta.csv` rename, and may use the machine's existing `~/.cache/drone-fly/connectome_src/` copy as a fallback source.
- **Edge case** — a partial/interrupted download must be cleaned up (download to a temp path, then atomic move) so a retry is not poisoned; verify non-zero/expected size before declaring success.
- **Edge case (performance)** — pruning the full ~161k-neuron / ~25M-edge matrix is far heavier than the fixture (minutes + memory); acceptable (it is the point), but the analyst should note the runtime/memory expectation so it isn't mistaken for a hang.

## Original Description
Ok, new change: Make it so we don't use the 300 neurons connectome at all; by default we should download the full one when creating pruned versions

(Refinement) In the UC, okay, keep the 300 nodes one, but make its usage conditional so only CI uses it (or anyone defining it as a target somehow).

## Clarifications
- Q: Which commands should default to the full connectome?
  A: `prune` only — train/evaluate keep their current behavior.
- Q: How should the full connectome get downloaded?
  A: Auto-download if absent — prune transparently downloads it when missing, then proceeds.
- Q: What happens to the committed 300-neuron fixture?
  A: Keep it, but make its usage conditional — nothing defaults to it; it is used only when explicitly named as the target (CI, smoke-train, or an explicit `connectome: tests/fixtures`).
- Q: The example prune configs (k0/k1/k2) currently run in CI against the fixture — how to reconcile with the new default?
  A: Point the example configs at the full connectome (default dir), and add/keep a separate fixture-targeted prune test so prune logic stays covered hermetically in CI.
