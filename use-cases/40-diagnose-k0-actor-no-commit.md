# Use Case 40: Diagnose why the actor won't commit before accepting the K0 "undersized slice" verdict

## Summary
A 1M-step training run repeatedly fires `health_callback`'s **K0 signature** — `WARNING` at update 11, escalating to `CRITICAL` by update 20 (25%, ~3h in) — concluding the connectome slice is "likely undersized." Its reasoning: the **critic looks healthy** (explained_var trend rising) while the **actor never commits** — `entropy ~5.72` and action `std ~1.014` stay flat and high, `success_rate` pinned at 0%, `ep_rew_mean` glued to −5 on 101-frame episodes. But absolute `explained_var` is only **~0.03** (near-zero), contradicting the callback's own "high/rising" claim, so the K0 verdict may be mis-triggering on a mislabeled critic-health signal. UC-40 correctly diagnoses the real cause *with evidence* before acting on the expensive re-slice conclusion, ranking three suspects: (1) `ent_coef`/entropy schedule over-rewarding randomness, (2) a flat reward gradient starving the actor of signal, (3) a classification bug in the callback itself. It then **delivers a fix** — config and/or reward shaping as the evidence warrants, plus a correction to the callback's `explained_var` criterion — and, **if the evidence shows the slice genuinely is undersized, bumps the slice-size/capacity config** and confirms it loads and trains. It touches `src/drone_fly/train/health.py`, `health_callback.py`, the reward/entropy configuration, and (conditionally) the slice/capacity config.

## Acceptance Criteria
1. The true root cause of the pinned actor is identified **with evidence** (specific code/config lines, metric traces, or a minimal reproduction); each of the three suspects is explicitly confirmed or ruled out.
2. The `health_callback` `explained_var` criterion is **audited and corrected**: it is demonstrated whether "high/rising" can fire at an absolute value of ~0.03, and the threshold/normalization is fixed so the K0 signature only fires when the critic is genuinely healthy. Unit tests cover the corrected logic.
3. A concrete fix (training config and/or reward shaping, as the evidence warrants) is implemented such that the actor can commit — under it, `entropy`/`std` trend **down** and `success_rate` is able to leave 0 — with rationale tied to the evidence from criterion 1.
4. The fix is **lab-validated in-sandbox**: unit tests plus a short CPU smoke-train (a few hundred/thousand steps) demonstrating the actor's entropy/std actually respond to the change. (The full ~12h GPU retrain remains the user's to run on Windows; it is not a gate for this UC.)
5. **If the evidence shows the slice genuinely is undersized**, the slice-size/capacity config is bumped and confirmed to load and train (smoke-train boots and steps without error); the change is documented.
6. If reward shaping changes, the README reward-table / doc contracts are updated in the same change and stay green (`tests/test_bootstrap_docs.py`, `tests/test_viz_contract.py`); the UC-37/39 potential-based reward invariants are preserved.
7. The full suite stays green: `uv run --extra dev pytest`.

## Potential Pitfalls & Open Questions
- **Risk** — reward shaping is protected work (UC-37/39: non-farmable potential-based terms, anti-suicide test). Any change must preserve those invariants and their tests, not just append.
- **Assumption** — a CPU smoke-train of a few hundred/thousand steps is enough to show entropy/std *responding* to a fix, even though it can't reach task success. Treated as sufficient signal per the "lab-validated" deliverable choice.
- **Missing input** — whether `ent_coef` / the entropy schedule / slice-size are real config keys or hardcoded is still unverified in code (memory notes `batch_size`/`n_steps` are *not* real YAML keys — the same trap may apply). The analyst confirms the actual config surface before proposing config edits.

## Original Description
Captured from the live training TUI the user shared plus the ensuing analysis. A K0-signature training-health warning is firing on a drone-fly run:

- update 11: `WARNING` — "Likely-undersized connectome slice (K0 signature): critic is healthy (explained_variance high/rising) but the actor is not committing (action std flat and high) and success_rate is pinned at ~0. The slice may lack the capacity to learn."
- update 20 (25%, 3h07m elapsed, 245,760 / 1,000,000 steps, 6 subproc envs): escalated to `CRITICAL` with the same message.

Observed metrics at update 20: `ent_loss -5.72`, `value_loss 0.5` (trend worsening), `expl_var 0.03` (trend improving), `entropy 5.72` flat ⚠, `std 1.014` flat, `approx_kl 0.006` rising, `ep_rew_mean -5` flat, `ep_len 101`, `success 0%` flat. Episodes recorded at 25627 neurons.

The ask: investigate and correctly diagnose why the actor is not committing BEFORE accepting/acting on the slice-capacity conclusion. Prime suspects in priority order: (1) `ent_coef` / entropy schedule over-rewarding randomness; (2) flat reward gradient (no learning signal for the actor); (3) the `health_callback`'s own logic — it labels `explained_var` "high/rising" while the absolute value is ~0.03 (near zero), so the K0 diagnosis may be mis-triggering. Files of interest: `src/drone_fly/train/health.py`, `src/drone_fly/train/health_callback.py`, and the reward/entropy configuration. Respect drone-fly's README/reward-table test contracts if reward shaping changes.

## Clarifications
- Q: What should UC-40 actually deliver as 'done'?
  A: Diagnosis + fix, lab-validated — root-cause report + code/config fix validated by unit tests and a tiny CPU smoke-train proving the actor's entropy responds; the full ~12h GPU retrain stays the user's to run on Windows.
- Q: Is the health_callback's K0 detector itself in scope to change?
  A: Yes — audit & fix it; treat the explained_var "high/rising" claim at abs ~0.03 as a likely threshold/normalization bug and correct it so K0 only fires on a genuinely healthy critic.
- Q: How far can the fix reach into training behavior?
  A: Config + reward if warranted — `ent_coef`/entropy schedule AND reward shaping are both fair game if the evidence points there, with the UC-37/39 doc/test contracts respected.
- Q: If evidence shows the slice genuinely IS undersized (K0 is real), what should UC-40 do?
  A: Include a slice bump — change the slice-size/capacity config as part of this UC and validate it loads/trains.
