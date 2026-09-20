# Use Case 33: TUI status-box autosize + accurate OOM & test-command guidance

## Summary
Three small "papercut" fixes flagged during a 2026-09-20 Windows training session, bundled into one use case because each is a self-contained accuracy/usability correction with no impact on training dynamics.

**(1) TUI status-box dynamic height.** The full-width bottom status bar in the live training TUI (the health-verdict panel) is laid out with a fixed height — `build_layout` in `src/drone_fly/train/tui/render.py` splits the status region at `size=3` (a Rich `Panel` border plus a single content row). Multi-line health warnings therefore clip to one visible row and the rest is lost. The status region should size its height to the number of lines the current verdict message occupies (plus the panel's border/padding overhead), capped at a sane maximum so it never consumes the whole screen. This is the same class of fixed-height issue as the recent values-panel height bump (7→8 for the steps line). The fix must render identically on macOS/Linux and Windows.

**(2) Misleading CUDA-OOM guidance.** On CUDA selection, `src/drone_fly/train/device.py` logs `CUDA_OOM_HINT`, which tells users to "lower n_envs and/or batch_size in your train config". But `batch_size` (and `n_steps`) are **not** YAML train-config keys: they are `TrainConfig` field defaults in `src/drone_fly/train/config.py` (`n_steps: int = 2048`, `batch_size: int = 64`) and are absent from the `_Spec` validation list in `src/drone_fly/config.py`, so setting them in a train YAML would be rejected as unknown keys. The hint (and any related OOM messaging) points at a knob that does not exist. Two ways to resolve, for the dev-team/challenger to choose between:
- **(a) Expose the knobs:** add `batch_size` and `n_steps` as real, validated train-config keys (new `_Spec` entries in `src/drone_fly/config.py`), plumbed through into the PPO constructor, so the guidance becomes literally true.
- **(b) Correct the message:** rewrite `CUDA_OOM_HINT` to reference only levers that already exist — `n_envs`, a smaller pruned slice (`prune` / `prune_k`), and `device: cpu`.

**(3) Brief drift on the test command.** `PROJECT_BRIEF.md` frontmatter declares `build.commands.test: "uv run pytest"`, but pytest ships in the `dev` extra, so the bare command fails to collect (pytest is not present in the base environment). The working invocation is `uv run --extra dev pytest`. CI already sidesteps this (`.github/workflows/ci.yml` runs `uv sync --extra dev` before `uv run pytest`, so the dev extra is already synced), but the brief and any docs referencing the bare command are the source of drift. Fix so the declared/canonical test command actually collects and runs the suite — either update the brief (and docs) to `uv run --extra dev pytest`, or restructure packaging so bare `uv run pytest` works.

Scope is UI-layout + messaging + brief/doc accuracy only. No change to training dynamics, observation schema, connectome, reward, checkpoints, or the env/adapter. Default training behavior for all existing configs must be byte-for-byte unchanged.

## Acceptance Criteria

### Item 1 — TUI status-box autosize
1. The bottom status region's height is derived from the number of lines in the current health-verdict message (accounting for the panel border/padding), rather than a hardcoded `size=3`, so a multi-line warning renders all of its lines without clipping.
2. The status-region height is capped at a documented maximum (a small constant) so a pathologically long message cannot expand the panel to fill or overflow the terminal; content beyond the cap is truncated gracefully (e.g. still shows the first N lines).
3. A single-line verdict renders at the same visual height as today (no visual regression for the common case).
4. The autosize logic is unit-tested headlessly: given a verdict whose message spans K lines, the assembled layout allocates the expected height (min(K + overhead, MAX)); a 1-line verdict yields the baseline height; an oversized message clamps to the max. Tests do not require a live terminal draw.
5. Behavior is platform-independent (identical layout allocation on macOS/Linux and Windows).

### Item 2 — Accurate OOM / config guidance
6. The CUDA OOM guidance (`CUDA_OOM_HINT` in `src/drone_fly/train/device.py`, and any other user-facing OOM message/doc) references **only actionable levers** — i.e. every knob named in the message is either a validated YAML train-config key or a documented CLI/config mechanism that currently exists.
7. **If option (a) is chosen:** `batch_size` and `n_steps` are added as validated train-config keys (new `_Spec` entries with correct types) in `src/drone_fly/config.py`, are plumbed into the PPO constructor, and take effect at training time; a config that sets them is accepted (no "unknown key" rejection), and a config that omits them uses the existing `TrainConfig` defaults (`n_steps=2048`, `batch_size=64`) unchanged. A test asserts a YAML setting each key validates and reaches the model constructor.
8. **If option (b) is chosen:** the message names only `n_envs`, a smaller pruned slice (`prune`/`prune_k`), and/or `device: cpu`; a test (or grep-style assertion) confirms the message does not mention `batch_size` or `n_steps` while they remain unexposed.
9. Whichever option is chosen, the default training run (a config that touches none of these knobs) produces identical behavior to before this UC — no change to PPO hyperparameters, rollout size, or seeding for existing configs.

### Item 3 — Test command accuracy
10. The command declared in `PROJECT_BRIEF.md` frontmatter `build.commands.test` runs the pytest suite and collects tests without a collection/import error when run from a clean environment (i.e. the canonical documented command actually works).
11. Any docs/READMEs that quote the bare test command are updated to the same working invocation (no doc still tells a user to run a command that fails to collect).
12. CI remains green and its test invocation is unchanged or stays consistent with the canonical command (CI already syncs the `dev` extra before invoking pytest).

### Cross-cutting
13. The full hermetic suite passes offline: `uv run ruff check .`, `uv run ruff format --check .`, and the canonical test command (per AC-10) all green.
14. No change to training dynamics, observation schema, connectome, reward, checkpoints, or the env/adapter beyond the (optional) additive `batch_size`/`n_steps` plumbing in Item 2 option (a).

## Potential Pitfalls & Open Questions
- **Ambiguity** — Item 2 deliberately leaves the (a)-expose vs (b)-correct choice to the dev-team/challenger; the acceptance criteria are written to accept whichever is picked (AC-7 for a, AC-8 for b, AC-6/AC-9 for both). The challenger should pick and record the decision in the plan.
- **Risk** — if option (a) is chosen, exposing `batch_size`/`n_steps` must not change default training dynamics: configs that omit them must resolve to the current `TrainConfig` defaults (`n_steps=2048`, `batch_size=64`), so existing runs reproduce bit-for-bit (AC-7, AC-9).
- **Assumption** — the status-box needs a documented maximum height (a small constant, e.g. a handful of rows) to avoid consuming the whole screen; the exact cap is an implementation choice as long as it is bounded and documented (AC-2).
- **Edge case** — a very long single warning (one logical line that wraps, or one very long line) must clamp to the max height and truncate gracefully rather than overflow the terminal or crash the layout (AC-2).
- **Edge case** — the values-panel height was referenced in the task as recently bumped 7→8; on `main` at authoring time it reads `size=7`. Item 1 targets the *status* region regardless of the exact current values-panel constant; the dev-team should confirm the live value in `render.py` before editing and not assume a specific number.
- **Missing input** — it is unclear whether any non-CI docs (README, setup scripts, contributor notes) also quote the bare `uv run pytest`; the dev-team must grep the repo for the bare command and update every human-facing occurrence, not just the brief (AC-11).
- **Assumption** — CI is treated as already-correct (it syncs the `dev` extra first); the drift to fix is in the brief/docs, not the CI job (AC-12).

## Original Description
Three small deferred papercuts noticed during a 2026-09-20 Windows training session:

1. **TUI status/health box dynamic height.** The bottom status box in the live training TUI (the health-verdict/warnings panel) has a fixed height and clips multi-line warnings to one row. It should size its height to the number of lines its current message occupies (within a sane max), so multi-line health warnings render fully. Relevant code: `src/drone_fly/train/tui/render.py` builds the panels (the values-panel height was recently bumped 7→8 for the steps line — same class of fixed-height issue). Keep macOS/Linux and Windows consistent.

2. **Misleading CUDA-OOM guidance.** On a CUDA OOM, `src/drone_fly/train/device.py`'s `resolve_device` log (and/or the OOM hint) tells users to "lower n_envs and/or batch_size in your train config" — but `batch_size` and `n_steps` are NOT exposed as YAML train-config keys (they are `TrainConfig` defaults in `src/drone_fly/train/config.py`, not in the `_Spec` validation list in `src/drone_fly/config.py`). The guidance points at a knob that doesn't exist. Fix by EITHER (a) exposing `batch_size`/`n_steps` as real train-config keys (validated + plumbed into the PPO constructor), OR (b) correcting the message to point at levers that DO exist (`n_envs`, a smaller `prune_k`/connectome slice, `device: cpu`). Present both options and let the dev-team/challenger pick; acceptance criteria accommodate whichever is chosen.

3. **Brief drift on the test command.** `PROJECT_BRIEF.md` frontmatter `build.commands.test` = `uv run pytest`, but pytest lives in the `dev` extra, so the bare command fails to collect; the working command is `uv run --extra dev pytest`. Fix so the declared/canonical test command actually runs — either update the brief (and any docs/CI referencing the bare command) to `uv run --extra dev pytest`, or restructure packaging so `uv run pytest` works. Note CI (`.github/workflows/ci.yml`) already uses a working invocation; the drift is in the brief/docs.
