# Flight models — the K1 connectome flying courses

Index of the trained models in which the **real K1 connectome** (25,627 neurons, ~3.86M synapses, real
MaleCNS weights) flies a quadcopter in acro (rate-mode / CTBR) flight on the GPU-parallel simulator.
See [CONNECTOME_RESERVOIR.md](CONNECTOME_RESERVOIR.md) for how the frozen-reservoir approach works.

## Working models (deterministic eval, committed & reproducible)

| Scenario (`training/<name>/`) | Course | Deterministic completion | Mean tilt | Recordings |
|---|---|---|---|---|
| **k1-gate-flight** | 3 straight gates | **100%** | 7° | 10 |
| **oval-lap-flight** | closed oval lap (same-direction turns) | **99–100%** | 25° | 10 |
| **figure8-chicane-flight** | randomized figure-8 *or* chicane per episode (both turn directions) | **96%** (figure-8 100% / chicane 92%) | 12° | 12 |
| **pad-lap-flight** | oval lap + battery + charging pad (recharge mid-lap) | **99%** | 46° (aggressive — see note) | 10 |
| **obstacle-lap-flight** | oval lap + pillar obstacles (weave around) | **99%** (100% never enter a pillar, 0.27 m clearance) | 44° (weaving) | 10 |
| **proc-course-flight** | **procedurally random** gate courses (different every episode: 5–10 rotated gates, varied heights, apertures) | **96%** across the random pool (flight-audited: no hover, speed 2.75 m/s) | 21° | 12 |

The navigation courses (straight/oval/track) are **upright, controlled flight** (~7–25° tilt). The
**pad** model solves the battery/charging task at 99% but flies **aggressively** (~46° tilt, never
inverts): the battery pressure rewards speed, so it rushes and banks hard. Tilt is tracked precisely
because an early version of this project was briefly fooled by ballistic tumbling scoring "completion".

### What's committed per scenario
- `recordings/episode_*.json` — app-readable recordings (all 25,627 neuron activations per frame +
  action + position + target gate + anatomical neuron metadata). Open the scenario in the desktop app.
- `recordings/README.md` — per-scenario notes.
- `model_readout.pt` (~610 KB) — the trainable model. The frozen K1 **sparse layer** (~3.86M edges,
  the bulk of the 177 MB full checkpoint) is rebuilt from `artifacts/pruned/k1/` at load; everything
  else (the readout **and** the frozen-but-randomly-initialized `body.input_projection`) is in this file.
  > Note: `body.input_projection` must be committed — it is random at construction, *not* reproducible
  > from artifacts, and the readout is trained against that specific projection.

## Reproduce / verify

```bash
cd gpu_prototype
# reload each committed model_readout.pt into a fresh actor and deterministically eval:
python verify_committed.py
# train from scratch (augmented reservoir = [brain-tap 1024 || obs 12], exploration annealing):
python train_aug_course.py lap        # or: track
# deterministic eval + app recordings:
python eval_aug.py <ckpt> <lap|track> 128
python record_aug.py <lap|track> <ckpt> <scenario-name> 10
```

The straight course uses the pure reservoir (`gates_k1_reservoir.py`); the turning courses use the
augmented reservoir (`reservoir_aug.py`) — the K1 body also feeds the raw flight obs to the readout,
which (with exploration annealing) closes a deterministic-control gap on the hard return-leg turns.

## Work in progress

- **proc-course-flight — obstacles tier (open problem)** — the committed proc model above is the
  **obstacle-free** tier (random gate courses), which generalizes cleanly at 96%. Adding **obstacles to
  random courses** is not solved, and the reason is instructive — two failure modes, both audited:
  - **non-terminal nudge** (contact bumps, recoverable): deterministic completion looks high (76–78%) but
    it's **fake — ~52–54% of completed runs *barge* straight through obstacles** (plowing is cheaper than a
    detour that must generalize across random placements). A size curriculum (obstacles grow 0→full) did
    not change this.
  - **terminal contact** (`PROC_OBS_TERMINAL=1`, hit = crash → barging impossible): **honest but low —
    ~6–10% completion** (it crashes into randomly-placed on-path obstacles rather than reliably avoiding).
  So: non-terminal → cheats, terminal → can't. Clean high-completion evasion worked on `obstacle-lap-flight`
  only because that course was *fixed* with a *clear lane* + slide-in curriculum; generalizing avoidance
  across *random* on-path obstacles is the open hard problem.
  - **Raycast perception tried** (`PROC_RAYCAST=1`: a forward 90° cone of 40 free-space rays fed to the
    readout — `gates_proc_env._raycast()`). It lifted the *stochastic* policy ~10–13 pts at matched obstacle
    size, but under annealing the **deterministic mean still collapsed to ~5%** (same as without raycasts).
    So perception was **not** the bottleneck — the small frozen-reservoir *readout* can't turn free-space
    sensing into a placement-invariant avoidance policy that survives annealing.
  - **Next levers (future):** feed the raycasts into the **connectome itself** (a second sensory input
    projection — use the brain's capacity, not just the ~168k readout); and/or gentler annealing (deploy a
    mildly-stochastic policy, which flew clean at ~50%); and/or partial plasticity. Infra ready
    (`gates_proc_env.py`, `course_gen.py`, `verify_proc.py`; env vars `PROC_TRACTABLE`/`PROC_NO_OBS`/`PROC_OBS_TERMINAL`/`PROC_RAYCAST`).
- **pad-lap-flight (landing version)** — the committed `pad-lap-flight` above recharges on a *fly-over*
  (a flaw). A redesigned `gates_pads.py` requires an actual **controlled landing + full stop** on the
  pad, with a **timed gradual recharge**, **battery-relative urgency** (>65% ignore pad / ~40% divert /
  <25% override the course), and **battery sag** (non-linear thrust — a low battery can't fly
  aggressively). Retraining to replace the fly-over model.

Obstacle avoidance (above) was previously the hard case — naive training collapsed to a stationary
hover (any proximity/terminal penalty makes flying riskier than hovering) or, with nudge-only, to
unstable high-tilt flight; a direct tilt penalty is a non-starter (penalizing tilt removes the only way
to translate in acro). The **curriculum** (pillars start off-path and slide on, warm-started from the
oval) resolved it.
