# Use Case 23: Training-health assessment engine + capacity guardrail

## Summary
Add a **headless, pure-logic `assess_training_health()` engine** that inspects training signals (and, before training, the seeded actor's capacity) and returns a structured verdict — `status` (normal / warning / critical), a human message, and the contributing reasons — which UC-22's status bar renders and which is also emitted as a plain log line when the TUI is off. It codifies the diagnosis this project keeps hitting: a **healthy critic with a flat, non-committing actor and 0% success is the signature of an under-capacity connectome slice** (e.g. the k=0 minimal corridor), plus other well-known PPO failure modes (reward stalled over a long window, `approx_kl` runaway, `value_loss` divergence, premature entropy collapse). It also adds a **pre-train capacity guardrail**: when the pruned/seeded actor's **trainable-parameter** count is below a sane floor, the run flags "connectome slice likely too small — actor may not learn" *before* hours are spent, rather than only diagnosing after the fact. On an interactive start the guardrail **prompts the user to confirm** before proceeding; on a non-interactive/autonomous/CI start (no TTY) it cannot block on input, so it degrades to warn-and-continue unless an opt-in `--strict-capacity` is set (which aborts). The engine is a set of pure, threshold-driven rules unit-tested headlessly (given metric histories / capacity facts → expected verdict); it reads but never mutates training state, the connectome, the observation schema, reward, or checkpoints. UC-22 consumes it via an injected interface (verdict object); implementing UC-23 replaces UC-22's stub. No new heavy dependency.

## Acceptance Criteria
1. **Pure verdict engine:** `assess_training_health(...)` takes a snapshot of training signals (per-metric current value + recent history over a rolling window) and optional capacity facts, and returns a structured verdict: `status ∈ {normal, warning, critical}`, a short `message`, and a list of `reasons` (rule id + human text). No I/O, no global/process state — fully unit-testable.
2. **Under-capacity signature rule:** given a healthy/rising `explained_variance`, a flat high `entropy` (actor not committing), and `success_rate` pinned at ~0 over the window, the engine returns a warning/critical verdict whose message identifies a likely-undersized connectome slice (the documented K0 signature). The rule does **not** fire when entropy is falling and/or success/reward are climbing.
3. **Other runtime failure-mode rules (each an independent reason, each unit-tested with a triggering and a non-triggering synthetic history):**
   - **Reward stalled/declining** — `ep_rew_mean` not improving (or dropping) over the window despite ongoing updates.
   - **`approx_kl` runaway** — sustained KL above a target band (updates too aggressive / diverging).
   - **`value_loss` divergence** — `value_loss` growing without bound and/or `explained_variance` collapsing (critic breaking down).
   - **Premature entropy collapse** — entropy/`std` crashing to near-zero very early, before any success/reward gain (policy prematurely deterministic).
4. **Pre-train capacity guardrail (trainable parameters):** before training starts, given the resolved (post-prune) actor's **trainable-parameter count** and the observation/action dimensionality, the engine produces a pre-train verdict; when the parameter count is below the configured floor it flags under-capacity. This check runs on **every** training start and is surfaced as a log line at minimum.
5. **Prompt-to-confirm (interactive) with a defined non-interactive fallback + opt-in strictness:** when an under-capacity start is detected **and** stdin/stdout is an interactive TTY, the run prompts the user to confirm before proceeding (decline → clean non-zero exit, no training). When **not** interactive (no TTY / autonomous / CI), it cannot block on input, so it degrades to **warn-and-continue**. An opt-in `--strict-capacity` (flag / config key) makes an under-capacity start **abort** with a non-zero exit and guidance in either mode. A sufficiently-capable start (at/above floor) never prompts and changes no behavior.
6. **Consumed by UC-22, standalone otherwise:** the verdict object matches the interface UC-22's status bar expects (so UC-22's stub is replaced). When run without the TUI, the runtime verdict is emitted at a sensible cadence as a log line (not spamming every update).
7. **Scope containment / no regression:** pure logic + a pre-train hook + logging/prompt only. No change to env dynamics, observation schema, reward, connectome graph, or checkpoints (no retrain). Full existing suite stays green on Linux CI (`ruff check .`, `ruff format --check .`, `uv run pytest`). No new heavy dependency.

## Potential Pitfalls & Open Questions
- **Assumption** — thresholds (entropy "flat" tolerance, success ~0 window length, KL target band, value_loss divergence criterion, trainable-parameter floor) are defined as named, documented constants/config with sensible defaults, not magic numbers scattered in code.
- **Edge case** — early-run noise: rules must not fire spuriously in the first few updates before enough history exists (a warm-up guard / minimum-samples requirement).
- **Edge case** — smoke runs / tiny fixtures legitimately have low capacity and few steps; the guardrail must **warn-only** there (non-interactive) and never turn CI-style smoke training into a false alarm that breaks the run.
- **Edge case** — the interactive prompt must not deadlock or fight the UC-22 Rich `Live` display if both are active at startup; resolve prompt ordering (prompt before the TUI takes over the screen).
- **Assumption** — the trainable-parameter floor's default value is calibrated so the documented k=2 default slice passes and a k=0 minimal slice trips it (validated against the fixture/connectome sizes), and is overridable via config.

## Original Description
From the user: a bottom status bar (UC-22) showing "Training progressing normally", "WARNING: Connectome too small", "or whatever depending on the values/progress we are seeing" — and, separately in the design conversation, a **capacity guardrail** so a k=0 (minimal-capacity) slice flags "likely under-capacity" up front instead of costing hours, motivated by an 81-iteration / 663k-timestep run that showed 0% success with a healthy critic (explained_variance rising) but a flat, non-committing actor (entropy pinned high, std barely shrinking, ent_coef=0). The health logic was split out of UC-22 into this engine so it is headless and CI-testable; UC-22 only renders its verdict.

## Clarifications
- Q: How should the pre-train capacity floor be measured?
  A: By the actor's **trainable-parameter count** below a configurable floor.
- Q: When an under-capacity slice is detected pre-train, what's the default behavior?
  A: **Prompt the user to confirm** before proceeding (interactive). Reconciled: on a non-interactive/autonomous/CI start with no TTY there is nobody to prompt, so it degrades to warn-and-continue; `--strict-capacity` forces an abort in either mode.
- Q: Which runtime failure-mode rules beyond the under-capacity signature?
  A: All four — reward stalled/declining, `approx_kl` runaway, `value_loss` divergence, and premature entropy collapse.
- Q: Relationship to UC-22?
  A: UC-22's status bar renders this engine's verdict; UC-22 stubs the verdict until this UC is implemented. This engine is headless and CI-unit-tested.
- Q: Runtime/observation/checkpoint impact?
  A: None — pure logic + a pre-train hook + logging/prompt; no env/obs/reward/connectome/checkpoint change, no retrain.
