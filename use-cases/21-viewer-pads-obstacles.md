# Use Case 21: Viewer renders recharge/repair pads and obstacles

## Summary
The 3D flight/playback viewer (`viz/viewer.js` + `viewer.html`, the UC-06 path) does **not** visibly render the course's **landing pads** (plain docking, recharge, repair) or its **obstacles**, so runs that exercise UC-15 obstacles and UC-16/18/19 pads look empty of those elements. This UC makes all three visible and visually distinct in the viewer, and completes the data path so the viewer can tell pad kinds apart. The recorder (`src/drone_fly/record/recorder.py` `_course_meta`) already stamps an additive `meta.course.obstacles` block (center/radius/height) and a `meta.course.pads` block (center/radius) — but **pads do not carry their kind** (a pad's `rechargeable` / `repairable` flags from `CourseConfig.PadSpec`), and the viewer's `drawPads` was explicitly deferred ("deferred to a later UC"). This UC: (a) extends the recorder's pad serialization **additively** to include each pad's kind (recharge / repair / plain); (b) implements pad rendering in `viewer.js` — floor-anchored discs colored by kind with a legend entry each; (c) ensures **obstacles actually render** (a floor-anchored pillar/cylinder marker; the `obstacle` color already exists in the palette) and are legended. It is **viewer + recorder-serialization + docs/tests only** — no change to the RL runtime, env dynamics, observation schema, reward, or checkpoints; every change is additive and back-compatible (old recordings without the new fields still render via graceful degradation, exactly like the existing UC-15/16 presence-guards).

## Acceptance Criteria
1. **Recorder pad kind (additive):** `_course_meta` emits, for each pad in `meta.course.pads`, a `kind` (or equivalent `rechargeable`/`repairable` booleans) derived from the actual `PadSpec` flags — recharge, repair, or plain. Presence-guarded exactly like today (the `pads` key is emitted only when the course has pads); no-pad runs and every pre-UC-21 recording stay byte-for-byte unchanged. No other `meta` field changes.
2. **Pads render in the viewer:** `viewer.js` draws each pad from `meta.course.pads` as a floor-anchored disc at its true `center`/`radius` on `floor_z`. Absent `pads` → nothing drawn (graceful degradation, no error), matching the existing course-anchor handling.
3. **Pad kinds are visually distinct:** recharge, repair, and plain pads use three distinct, documented colors (my defaults: recharge = green, repair = orange/red, plain = neutral grey), each with its own legend entry in `viewer.html`.
4. **Obstacles render:** each `meta.course.obstacles` entry is drawn as a floor-anchored pillar/cylinder marker at its `center`/`radius`/`height`, in the existing `obstacle` palette color, with a legend entry. Absent `obstacles` → nothing drawn (graceful degradation).
5. **Back-compatibility / no regression:** a legacy recording with no `pads`/`obstacles`/`kind` fields loads and renders with no console error and no change to gate/finish/floor/trajectory rendering. No change to the RL runtime, env, obs schema, reward, or checkpoints (nothing under `src/drone_fly/{controller,env,adapter,train,connectome}` behavior changes beyond the additive recorder field).
6. **Docs + contract tests:** the `viz`/README doc-contract is extended additively — the new legend/keys (pad kinds + obstacle) are asserted by the viz-contract test suite, and every existing `test_viz_contract.py` / `test_bootstrap_docs.py` token stays intact and passing. The recorder pad-kind serialization is unit-tested (a course with recharge+repair+plain pads round-trips the correct kinds into `meta.course.pads`; a no-pad course omits the key).

## Potential Pitfalls & Open Questions
- **Assumption (design defaults, chosen since the user is away):** recharge = green, repair = orange/red, plain pad = neutral grey, obstacle = existing palette purple. The analyst may refine exact hex values for contrast against the floor grid, but must keep three distinguishable pad colors + a distinct obstacle color, each legended.
- **Edge case** — a pad flagged BOTH rechargeable and repairable (if the config allows it): pick a documented precedence or a combined marker; the analyst pins this against `PadSpec`.
- **Edge case** — obstacles were reported not visible despite an `obstacle` color existing in the palette; the analyst must determine whether `drawObstacles` is implemented+called at all (vs only the color defined) and fix so they actually render, not just recolor.
- **Boundary** — the HTML/JS viewer is not executed in Linux CI (no browser); like the existing viz contract, coverage is **static/contract-level** (token/legend assertions on `viewer.js`/`viewer.html` + the recorder unit test), and browser rendering is a documented untested boundary. Do not add a test that spins a browser.
- **Back-compat** — all viewer reads of the new fields must be optional-chained/guarded so older recordings (and no-pad/no-obstacle courses) render unchanged.
- **Scope** — viewer (`viz/**`) + the single additive recorder field (`src/drone_fly/record/recorder.py`) + docs/tests only. No env/obs/reward/checkpoint change.

## Original Description
From the user: "the landing pads for recharging and repairing aren't shown in the viewer — they should be. As should obstacles." Requested as the next auto-implemented UC while the user is away, so design calls are made by the orchestrator and documented here.

## Clarifications / Decisions
- Q: Which elements must the viewer show?
  A: Recharge pads (UC-18), repair pads (UC-19), and obstacles (UC-15) — each visually distinct with a legend entry.
- Q: Does the recorder already carry what the viewer needs?
  A: Obstacles yes; pads yes for geometry but NOT their recharge/repair kind — add the kind additively (presence-guarded, back-compatible).
- Q: Any runtime/observation/checkpoint impact?
  A: None — viewer + recorder-serialization + docs/tests only; all additive and back-compatible.
- Q: Colors? (user away → orchestrator default)
  A: recharge = green, repair = orange/red, plain pad = neutral grey, obstacle = existing palette color; refine for contrast but keep them distinct + legended.
