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
| **proc-obstacle-avoid** | procedurally random gate courses **+ on-path obstacles** (weave around; terminal contact = crash) | **33%** honest (0% barging among completed, 0.26 m clearance; 90% on the same courses with obstacles off; ~97% solvable ceiling) | 21° | 14 |

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

- **proc-course-flight — obstacles tier — SOLVED at 33% honest deterministic** (`proc-obstacle-avoid`,
  clearance shaping + terminal + warm aggressive start). Random on-path obstacle avoidance that survives
  determinization is cracked; see the result + recipe at the end of this list. The long diagnostic journey
  that got there (every approach that *failed*, and why) is kept below because it pinpoints the real cause.

  The committed proc model is the **obstacle-free** tier (random gate courses), which generalizes cleanly at
  96%. Adding **obstacles to random courses** was the open hard problem — two early failure modes, both audited:
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
  - **Rays INTO the connectome tried (`reservoir_brain.py` → `K1ReservoirBrainRay`).** A trainable
    `ray_proj` injects the 40-ray cone into the brain's sensory neurons (frozen edges). First attempt
    collided obs + rays in the same 32 neurons; fixed with a **dedicated, wider 256-neuron ray
    population** (disjoint from the 12-dim obs neurons). It lifted the *stochastic* policy to ~18% peak but
    the **deterministic audit still collapsed to 5%** (`verify_brain.py`, 10/192) — *honest* (2% barging,
    0% among completed, 1.34 m/s), just can't complete. **Perception bandwidth is conclusively not the
    bottleneck.** A recording of this failure is at `training/09-proc-obstacle-fail/` for inspection.
  - **Position curriculum tried (`gates_proc_env._ocen`, `obs_pos_curr`) — FAILED, same wall.** Full-size
    obstacles start parked 3 m off-path and slide onto the path over the first 40% of training (then anneal
    from 0.55), mirroring what made `obstacle-lap-flight` evade cleanly. Completion rode ~80% while obstacles
    were off-path, fell to ~15% once fully on-path, and the deterministic audit landed at **5%** (9/192) —
    identical to the wide-translator run, honest (0% barging among completed, 1.21 m/s).
    - **Key diagnostic (obstacle tax):** the *same* policy on the *same* gate courses with obstacles **off**
      completes only **38%** (72/192) — far below the 96% a dedicated clean model reaches. So the
      terminal-crash-penalty + annealing regime produced **global timidity** (tilt fell 20°→13° through
      annealing), degrading even plain gate-flying — the collapse is not obstacle-specific, it's the
      deterministic mean going cautious. That reframes the fix: the problem is as much the
      determinization/caution dynamic as the avoidance itself.
  - **Recurrent (GRU) readout tried (`reservoir_gru.py`) — FAILED, and it was the decisive clue.** A GRU
    head gives the controller working memory to hold a multi-step avoidance arc through occlusion. Trained
    from scratch it learned to fly but then **collapsed to timid near-hover** (tilt 8.5°→3.5°, 0% det) the
    moment on-path obstacles + terminal crash dominated. Lesson: **memory was not the missing piece — the
    reward was.** With a terminal crash penalty on random obstacles, timid caution is always the safe local
    optimum, and no architecture escapes it.
  - **Courses confirmed solvable (~97%).** A geometric audit of 1000 tractable courses: every gate is
    reachable (the "blocking" obstacle sits at the *midpoint between* gates, never on a gate centre); the only
    defect was ~3% spawning the drone inside the fly-over box at the start — now fixed in `course_gen.py`
    (reject/nudge any obstacle whose footprint covers the start). So the achievable ceiling is ~97%, and the
    deterministic gap was always a policy/training problem, not unsolvable courses.

  **THE SOLUTION — `proc-obstacle-avoid` (clearance shaping + terminal + warm aggressive start).** Two fixes,
  targeting the two real causes (timidity + no steer-around gradient):
  1. **Warm-start aggressive** from the 96% obstacle-free flier → the policy begins at ~21° tilt, not timid.
  2. **Potential-based clearance shaping** (`gates_proc_env`, `PROC_SHAPING=1`): a dense, policy-invariant
     reward on the *change* in clearance to the nearest in-path obstacle — a real gradient toward "steer
     around", not just "don't crash". Plus terminal contact (forbids barging) + position curriculum +
     mildly-stochastic anneal (logstd_final −1.0).

  **Result (deterministic, 256 courses):** **33% completion, 0% barging** (0.26 m min clearance, 2.3 m/s,
  21° tilt — honest, aggressive). Same policy on the same gates with obstacles **off**: **90%** (so the
  obstacle tax is ~57 pts, and it captures ~37% of achievable). This is up from the **5% wall** every prior
  method hit. The deterministic mean *climbed* through annealing (5%→34%→39%→33%) instead of collapsing —
  the signature of a mean that genuinely solved the task. Recordings in `training/10-proc-obstacle-avoid/`.
  - **Counter-example (not committed):** the *non-terminal* + shaping variant scored 92% but **29% of
    completions barge** — free contact still makes plowing cheaper than detouring. Non-terminal → cheats.
  - **Next levers to push past 33%:** combine the shaping recipe with the GRU (now that timidity is handled,
    memory may add the arc-holding capacity); stronger shaping weight; longer training; partial plasticity.
    Reproduce: `PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 python train_shaping.py <out.pt>`.

### Deploying a model + visualizing the full course tier
- **Inference contract + standalone harness:** [`INFERENCE_CONTRACT.md`](INFERENCE_CONTRACT.md) documents
  the obs/action/plant/file formats; [`gpu_prototype/inference.py`](../gpu_prototype/inference.py) loads a
  `model_readout.pt` and flies it with a pure-NumPy reference plant (no training-env import) — for a
  liftoff/deployment harness.
- **`training/08-multilap-course-demo/`** — the existing proc readout flying the **full multi-lap tier**
  (2–3 laps, 5–20 gates, varied heights, no obstacles) so the big circuits can be *seen* (5/12 completed;
  the pilot was trained on the easier tractable tier).
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
