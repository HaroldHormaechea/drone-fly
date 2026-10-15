# UC74 — Live HUD next-gate pointer: Liftoff Arena calibration & launch recipe

The UC74 reader detects Liftoff's on-screen **next-gate pointer** (a white HUD ring) with classic
OpenCV — no ML — and converts its screen position into the body-frame unit bearing + availability
bit the UC70 run-2 student already consumes (`est.info["pointer"]`, schema v3). This doc records
the real-frame calibration for **Arena "04 - Infinity Loop" @ 2560×1080** and the turnkey recipe
to fly the run-2 student there with the pointer live.

Nothing here retrains the student: the pointer is a body-frame unit vector in the obs tail, so the
reader fills a signal that is otherwise zeros/bit-0 live (AC6).

## What the pointer looks like (Arena, ACRO)

- The next-gate pointer is a **clean white ring** (a see-through hole shows the background) that is
  billboarded at the next gate, so it **roams** the central play area as the gate moves.
- The fixed **ACRO aiming reticle** is an *identical-looking* white ring at the exact image centre,
  with four tick marks. It is a permanent same-size confuser, so the reader separates the pointer
  from the reticle **by position**: `camera.pointer.center_exclude_radius_frac` drops any detection
  within ~35 px of the principal point.
- Both are near-white (low S, high V); colour can't separate them. The pointer is keyed on white +
  shape (circularity/solidity) inside `camera.pointer_roi`, with the centre-exclusion removing the
  reticle and `ambiguity_ratio` rejecting frames with two equally-good blobs.

## Calibration (from `uc74-captures/uc74-liftoff-arena-04`, 755 HUD-visible frames)

Ground truth is the body-frame bearing to the current target gate, reconstructed from the paired
telemetry pose + the track/race XML route (the UC64/UC70 auto-label machinery), compared to the
reader's bearing on frames where the target gate is plausibly in view. Reproduce with
`drone_fly.harness.offline.pointer_eval.calibrate_from_capture` (see "Reproduce" below).

| Quantity | Value |
|---|---|
| Camera fit (hfov / uptilt) | **122° / 27.5°** (generic 120° / 25° defaults were within noise) |
| Detection rate (all frames) | 52% (395 / 755) — the pointer is genuinely absent at the start line, gate passes, and hard off-axis turns |
| Combined bearing error (inlier) | **median 3.7°, p95 13.1°** |
| Azimuth / elevation error (inlier median) | 2.4° / 2.4° |
| Systematic bias | (−0.4°, −0.1°) ⇒ ~0.007 rad |
| Per-tick jitter (std) | **4.9° ⇒ 0.085 rad** |

Machine-readable report: [`uc74-arena-pointer-calibration.json`](uc74-arena-pointer-calibration.json).

### AC9 — inputs for a future run-2 noise re-tune

The provisional `student-liftoff-run2.yaml` pointer sigmas vs. the real measurement:

| Config key | Provisional | Measured (suggested) | Takeaway |
|---|---|---|---|
| `pointer.bias_sigma_rad` | 0.05 (~2.9°) | **~0.007** | sim **over**-modelled bias |
| `pointer.jitter_sigma_rad` | 0.03 (~1.7°) | **~0.085** | sim **under**-modelled jitter (~2.8×) — the real HUD reader is noisier tick-to-tick than assumed |
| `pointer.bearing_tol_deg` | 20.0 | **~16** | provisional tolerance was slightly conservative |

Re-running run-2 training with these values is a **downstream experiment**, out of UC74's scope —
this only produces the numbers. The tail (p95 13°) is driven by motion blur and the pointer's
billboard offset from the gate centre, so a fat-tailed jitter model may fit better than a Gaussian.

### UC75 — run2b: the re-tuned noise model + off-axis guard

UC75 applies these numbers as a **new variant**,
`src/drone_fly/training/configs/student-liftoff-run2b.yaml` (`extends: student-liftoff-run2`,
overriding only `perception.pointer`); `student-liftoff-run2.yaml` is unchanged as the A/B baseline.
The student obs schema is unchanged (v3, DIM 65).

| Config key | run2 (provisional) | run2b | Source (`uc74-arena-pointer-calibration.json`) |
|---|---|---|---|
| `pointer.bias_sigma_rad` | 0.05 | **0.0073** | `suggested_bias_sigma_rad` 0.00734 |
| `pointer.jitter_sigma_rad` | 0.03 | **0.085** | `suggested_jitter_sigma_rad` 0.08503 (per-axis, as the sim draws it) |
| `pointer.bearing_tol_deg` | 20.0 | **16.4** | `suggested_pointer_bearing_tol_deg` 16.38 |
| `pointer.dropout_p` | 0.1 | **0.5** | `1 - detection_rate` = 1 - 0.523 |

**Dropout.** `dropout_p` is drawn per *episode*, while the live pointer is missing on ~48 % of
*frames*. The student policy is feedforward per tick and the pointer is not in its obs-history EMA,
so what it learns about pointer absence is set by the marginal share of bit-0 ticks, which is
`dropout_p`. 0.5 matches the live share (the off-axis guard below only raises it) and still leaves
half the episodes pointer-on. A per-tick or state-conditioned availability model would need sim and
reference-perception changes; it is a follow-up if the run2b re-eval shows pointer over-reliance.

**Off-axis guard.** `camera.pointer.max_bearing_az_deg` (default `null` = off) makes the live
reader report the pointer unavailable (bit 0, zero bearing) when the detection's body-frame
`|az| = |atan2(by, bx)|` is strictly greater than it. The `liftoff_arena` preset sets **30°**, per
the verdict below (this ROI reaches ~44° at 122° hfov, so it is a real cut). Note that this applies
to **every** student flown with `--preset liftoff_arena`, including existing run-2 exports; to fly without it,
pass a user `--config` that sets `camera.pointer.max_bearing_az_deg: null`.

**Retrain / export / fly** (the CUDA retrain and the re-evaluation vs the run-2 baseline are
user-side GPU follow-ups, not done in-sandbox):

```
drone-fly train student --config student-liftoff-run2b --teacher <run> --noise <noise_model.yaml>     --device cuda [--resume <run folder>]
drone-fly train export <run> --out <student-run2b.pt>
drone-fly harness run --harness liftoff --checkpoint <student-run2b.pt> --preset liftoff_arena
```

## Conversion-model verdict (bounds AC3)

Is the icon a faithful camera-frame projection of the gate (so pixel→ray→body holds), or a
clamped/horizon-stabilized edge indicator? Binning the combined error by the gate's off-axis angle
`|az_gt|` answers it (all detections, no inlier trim):

| `|az_gt|` | n | median err | p90 |
|---|---|---|---|
| 0–10° | 129 | **4.0°** | 18.8° |
| 10–20° | 117 | **3.8°** | 21.1° |
| 20–30° | 60 | 8.5° | 32.4° |
| 30–40° | 33 | 8.8° | 27.2° |
| 40–50° | 16 | 14.8° | 58.7° |
| 50–90° | 31 | 42.4° | 72.4° |
| 90–180° | 9 | 74.6° | 116.6° |

**Verdict:** the pointer is a *faithful* gate projection in the central cone (`|az| ≲ 20–25°`),
where the no-attitude `pixel→ray→body` model holds to ~4° median — and the fit landing on the
physical camera (122°/27.5°) confirms it. So the **attitude-aware hook is not needed for the
on-axis regime** that matters when flying toward a gate. Beyond ~±30–40° the error climbs steeply:
the icon **clamps toward the screen edge** (a direction indicator, no longer a true projection), so
those readings are unreliable. The `pointer_roi` already excludes the screen edges, and the student
trained with pointer dropout tolerates the resulting unavailability. UC75 adds the bearing-magnitude
guard: `camera.pointer.max_bearing_az_deg` (the `liftoff_arena` preset sets 30°) reads `|az| > 30°`
as unavailable — see "UC75 — run2b" above.

## Turnkey launch recipe (Windows + Liftoff)

The camera/pointer calibration lives in the **`liftoff_arena` preset**
(`src/drone_fly/harness/configs/liftoff_arena.yaml`), overlaid on the packaged `liftoff_5inch`
default.

```
drone-fly harness run \
  --harness liftoff \
  --checkpoint <path/to/student-liftoff-run2 export.pt> \
  --preset liftoff_arena \
  --overlay            # optional: UC65 debug overlay to watch the pointer bearing live (AC8)
```

**Is the pointer on by default with this preset? Yes — no extra flag.** The run-2 student *always*
consumes the `("pointer", 4)` obs block: it is baked into the trained network's input dimension
(schema v3, DIM 65), so there is nothing to "enable" at inference. What varies is only whether that
block is *live* or *degraded*:

- The live assembler always passes `pointer=(gate.info or {}).get("pointer")`
  (`harness/student.py`).
- `info["pointer"]` is populated iff the gate estimator was built with a pointer reader, which
  happens iff `camera.pointer.enabled: true` (and the block is configured). **The `liftoff_arena`
  preset sets `camera.pointer.enabled: true`**, so the reader is built and the pointer is fed from
  the live HUD — automatically, for both the OpenCV and the learned gate estimators.
- Without the preset (or with `camera.pointer.enabled: false`), `info["pointer"]` stays unset →
  the obs block falls back to zeros / availability bit 0 (the pre-UC74 degraded behaviour the
  student was trained to tolerate). No crash, just no pointer gain.

So flying with `--preset liftoff_arena` is sufficient to turn the live pointer on. To fly the same
student *without* it (A/B), drop the preset or override `camera.pointer.enabled=false`.

Before flying, confirm:

1. **Gate detector = learned.** The run-2 student is validated against the learned gate detector
   (`parts.gate_estimator: learned`) and its weights (`gates.learned.weights`; the released weights
   auto-download, sha256-verified). Set these in a thin user `--config` if the default doesn't
   already point at them.
2. **Camera match.** `check_student_config` refuses to fly if the live camera
   (resolution / hfov / uptilt) differs from what the student's noise model assumed. The preset is
   2560×1080 @ 122°/27.5°; if the student was exported for a different camera, re-export/record at a
   matching resolution, or adjust the preset to the student's camera (the pointer bearing itself is
   resolution-independent — a body-frame unit vector — but the gate obs is not).
3. **HUD on, ACRO mode.** The pointer is only present with the in-game next-gate HUD visible. The
   centre-exclusion assumes the ACRO reticle at screen centre.
4. **Telemetry StreamFormat** matches `telemetry.stream_format` (the student also observes the
   battery — UC69/70).

The pointer availability bit drops to 0 at the start line, at gate passes, and when the gate is far
off-axis; the student was trained with per-episode pointer dropout for exactly this (AC4), so it
keeps flying on the degraded signal.

## Re-tuning for another track / resolution

The masks, `pointer_roi`, HSV band and `center_exclude_radius_frac` are **layout-specific**. For a
new track or resolution: copy `liftoff_arena.yaml`, set `camera.resolution`, re-draw `camera.masks`
+ `camera.pointer_roi` from a capture, confirm the white band on your HUD theme, and re-run the
calibration below to refit hfov/uptilt and re-measure the bearing error.

## Reproduce the calibration

```python
from drone_fly.harness.config import load_config
from drone_fly.harness.liftoff.gate_cv import camera_from_config
from drone_fly.harness.liftoff.masks import FrameMasks
from drone_fly.harness.liftoff.pointer_cv import pointer_reader_from_config
from drone_fly.harness.offline.pointer_eval import calibrate_from_capture

cap = "uc74-captures/uc74-liftoff-arena-04"
# UC75: the preset now enables the |az| guard, which would change these stats; turn it off to
# reproduce the committed JSON (pointer_eval measures through reader.read()).
cfg = load_config(
    "liftoff", preset="liftoff_arena", overrides={"camera.pointer.max_bearing_az_deg": None}
)
cam = cfg.section("camera")
reader = pointer_reader_from_config(camera_from_config(cam), cam, FrameMasks.from_config(cam))
rep = calibrate_from_capture(
    reader, cap, f"{cap}/track.xml", race=f"{cap}/race.xml",
    telemetry_cfg=cfg.section("telemetry"), gate_aperture_m=10.0,
    report_path="docs/harness/uc74-arena-pointer-calibration.json",
)
print(rep.summary())
```

`gate_aperture_m` sets the route-progression pass radius used to pick the current target gate; make
it comfortably larger than the real gate if the default under-advances (the Arena arches need ~10 m
for the route to walk cleanly). Needs the `harness` extra (`opencv-python`); use
`opencv-python-headless` on a server without X.
```
