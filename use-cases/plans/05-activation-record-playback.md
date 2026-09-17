---
plan_for: use-cases/05-activation-record-playback.md
work_branch: feat/uc-05-activation-record-playback
team: drone-fly-uc-5
approved: 2026-09-17
---

# UC-05 Approved Implementation Plan — Neuron-activation recording + playback visualization

Analyst↔challenger APPROVED (round 2; challenger independently verified the coordinate source + code).
TARGET_DIR = /workspace/drone-fly-uc-05-activation-record-playback. Prose only, no code.

## Coordinate source — RESOLVED (the #1 open item)
Real anatomical soma positions are obtainable **tokenlessly**. The `somaLocation` column in
`mcns_all_neuron_meta.csv` (YijieYin/connectome_data_prep, raw GitHub, CC-BY — same source UC-01
uses) gives `[x y z]` in **neuPrint MaleCNS voxel space (8nm isotropic)**, covering **300/300 fixture
bodyids** (verified by pandas join; challenger confirmed). The `coords` column is a different,
near-empty annotation — NOT used. `somaLocation` equals `neuprint-python`'s
`fetch_neurons().somaLocation`, so the tokenless mirror = the token path. ⇒ the anatomical brain map
is the full, real, OFFLINE/CI **default**, not a fallback. NEUPRINT_TOKEN is needed only for
arbitrary user slices whose bodyids aren't in the committed sidecar; a deterministic spectral-layout
fallback covers any neuron lacking a soma. Projection: store full 3D, default dorsal **(x,z)** plane,
axes selectable in the viewer, documented + honest caveat (8nm assumed; absolute scale irrelevant to
a normalized top-down).

## The activation to record
`actor.py::ConnectomeActorNetwork.forward`: `propagated = self.layer(state)` — the full pruned-N
post-propagation state (B,N), computed before the motor `index_select`/readout. Bounded to [-1,1]
(`policy.py`: `state=tanh(W_eff@state)`), so uint8 quantization over [-1,1] won't clip. The recorder
surfaces a **copied, detached** view of this per env step.

## Production work (developer — src/drone_fly/**, scripts/**, viz/**, README, .env.example)
- NEW `src/drone_fly/record/`: `recorder.py` (ActivationRecorder; per-episode
  `artifacts/activations/episode_<n>.json`; documented `schema_version` schema — below; uint8+scale/
  offset quantization; optional gzip; logs path+size — AC3/AC10), `rollout.py`
  (`record_rollout(model_or_actor, env, recorder, connectome_data, n_episodes, seed, record_every)` —
  eval-style loop, clear-then-set the actor sink each step, read copied activation, collect action +
  `info["position"]` + outcome, emit every Nth episode; works with a raw actor OR loaded PPO so it is
  unit-testable without a checkpoint — AC1/AC4), `coordinates.py` (`provision_positions(connectome_data)`
  → source order: committed/local anatomical CSV (fixture sidecar, or `DRONE_FLY_SOMA_CSV` pointing at
  the connectome_data_prep meta — tokenless) → neuPrint+NEUPRINT_TOKEN (dev-time) → deterministic
  spectral fallback with **sign-canonicalized** eigenvectors, labelled "computed (NOT anatomical)";
  roles: sensory=`visual_projection`, motor=`descending_neuron`, else interneuron — AC6/AC9).
- MODIFY `controller/actor.py` — opt-in sink attribute (default None); after `propagated`,
  `if sink: sink(propagated.detach().cpu().clone().numpy())`. **Copy (clone) mandatory** so an
  in-place op on the recorded array can never corrupt the live forward output. No-op & bit-identical
  when off; detached (no grad perturbation) when on (AC2).
- MODIFY `controller/sb3.py` — passthrough to reach/toggle the actor via
  `model.policy.features_extractor.actor`; pin capture to the pi/actor path.
- MODIFY `env/racing_env.py` — add `info["position"]=state.position` in step() (back-compatible;
  existing tests use membership not exact key-set).
- MODIFY `evaluate/evaluator.py` — optional `recorder`/`record_every` params wired into the existing
  loop; **loads the same (pruned) ConnectomeData** (via `--connectome`/`--prune`/`--prune-k` or the
  persisted UC-04 slice) and hands it to the recorder for neuron_ids/superclass/adjacency (checkpoint
  does NOT retain these). **Hard alignment assertion**:
  `len(neuron_ids)==actor.n_neurons==activation.shape[-1]` → raise on mismatch. When record opts None,
  behaviour unchanged.
- MODIFY `cli/__init__.py` — `--record`/`--record-every N`/`--record-dir` on `evaluate` (and `train`);
  MODIFY `train/loop.py` — same flags, documented best-effort (batched rollout; eval is the tested
  primary).
- NEW `scripts/fetch_soma_positions.py` (dev-time, tokenless): downloads the connectome_data_prep
  meta, extracts `somaLocation` for the fixture bodyids, writes `tests/fixtures/mcns_fixture_soma.csv`
  (bodyid,x,y,z). Real acquisition, documents the tested-vs-untested boundary (AC7). **Developer owns
  this script but does NOT commit the tests/** output — QA runs it and commits the CSV (UC-01 precedent).**
- NEW `viz/viewer.html`+`viewer.js`+`viewer.css` — vanilla JS, no build; file-picker (FileReader)
  loads JSON from `file://`; three panels on one play/pause+scrubber playhead: (a) top-down anatomical
  brain map (canvas dots at projected soma, brightness=activation, role-colored, axis-selector,
  anatomical-vs-computed label), (b) time heatmap neurons×frames ordered sensory→inter→motor with
  playhead, (c) flight panel — 4 action traces + drone path synced (AC5/AC8/AC9). Default output plain
  JSON; gzip optional via DecompressionStream (documented).
- MODIFY README (record flags, artifacts path, viewer usage, UC-04 pruned-slice pairing, coordinate
  provisioning incl. tokenless default + token dev path + fallback, projection doc) and `.env.example`
  (NEUPRINT_TOKEN + dataset).

**Schema (self-contained per episode):** `meta{neuron_ids[], superclass[], roles[],
positions{source, projection:"xz", coords3d[[x,y,z]|null], coords2d[[u,v]], has_position[bool]},
episode_index, seed, checkpoint, backend, n_frames, n_neurons,
action_layout:["throttle","roll","pitch","yaw"], dt, activation_scale, activation_offset}`,
`frames{activations[F][N] (uint8), actions[F][4], drone_position[F][3]}`,
`outcome{completed, completion_time, total_reward, steps}`.

## Files Affected
**Production (developer)** — `src/drone_fly/**`, `scripts/**`, `viz/**`, README, .env.example:
- NEW `src/drone_fly/record/{recorder,rollout,coordinates,__init__}.py`
- MODIFY `src/drone_fly/controller/actor.py`, `controller/sb3.py`, `env/racing_env.py`,
  `evaluate/evaluator.py`, `train/loop.py`, `cli/__init__.py`
- NEW `scripts/fetch_soma_positions.py` (generator; developer does NOT commit its tests/** output)
- NEW `viz/{viewer.html,viewer.js,viewer.css}`
- MODIFY `README.md`, `.env.example`

**Test (qa)** — `tests/**`:
- NEW `tests/test_record_recorder.py`, `test_record_rollout.py`, `test_record_coordinates.py`,
  `test_record_backcompat.py`, `test_viz_contract.py`
- NEW committed data `tests/fixtures/mcns_fixture_soma.csv` (QA runs `scripts/fetch_soma_positions.py`
  and commits the output — QA-owned asset, UC-01 precedent)
- Possibly MODIFY `tests/conftest.py` (recorder/tmp-artifacts fixture). UC-01..04 suites stay green.

## Risks & Considerations
1. Coordinate source resolved tokenlessly (somaLocation, 300/300); token only for arbitrary
   off-sidecar slices; spectral fallback for any missing soma, labelled non-anatomical.
2. Non-invasive capture: detached+cloned sink; back-compat asserted bit-identical ON and OFF.
3. Checkpoint does not retain neuron_ids/superclass/adjacency → record path re-loads the pruned
   ConnectomeData and enforces the alignment assertion.
4. CI hermetic: no browser, no npm, no network, no token; recorder/coords tested on the fixture;
   fetch script is dev-time only.
5. Activation bounded [-1,1] → uint8 quantization safe; files compact (uint8 + optional gzip), size logged.
6. Projection axes (x,z default dorsal) documented + selectable; 8nm voxel scale assumed (irrelevant
   to normalized top-down).
7. Scope: recording + coordinate provisioning + static viewer only. No live server, no training
   changes, top-down projection only (no 3D fly-through).

## Challenger final verdict
APPROVE (round 2). Verified tokenless somaLocation (300/300 fixture coverage), the non-invasive hook
(detached+clone, bit-identical on/off), the alignment assertion (checkpoint drops data), and the
activation definition (final propagation state). 3 Majors fixed (coverage 300/300 not 293; re-load
ConnectomeData for neuron_ids; mandatory clone), 4 Minors incorporated (spectral determinism, SB3
double-capture guard, sink lifecycle, projection units).
