# Gate back-projection: how a learned detection becomes a 3-D gate (UC-91)

## Main finding: the UC-91 premise is wrong

UC-91 assumed that the live perception puts gate 0 near the ground while it is "10-18 m up". The
data says otherwise:

- **Gate 0 is not 10-18 m up.** All seven `InflatableArchBrandless01` arches on
  `LiftoffArenaTrack04_0001.xml` (Infinity Loop) stand at Unity y = 0.1, so the aperture centre sits
  at **1.84 m**: centre_offset 1.74 m, contract z = course z 0.98 + frame origin z 0.86. The
  "10-18 m" figure was the *drone's* height above the gate. The UC-89 notes say "passing 7-13 m too
  high".
- **The live estimate is correct.** Pass A below re-solves 2,800 logged detections from the six
  uc-89 model-14 runs. Within 40 m the z-error median is +0.01 m. Before take-off, z is off by
  +0.004 m at 25-40 m.
- **The policy was told the truth.** The overflight is a policy problem: model 14 predates the
  UC-90 altitude-reward work. A re-fly with any range mode should therefore show about the same gate
  passes. Neither better nor worse results should be credited to perception.

Back-projection does show two real effects, both smaller than the use case assumed:

1. **40-60 m radial over-estimate.** The median estimate lands +4.5 m too far away (about 10 %).
   The drone flies above the gate, so the line of sight slants down, and the too-far estimate also
   sits too low: z median −0.9 m.
   - The *angle* is right: the elevation error is +0.02 deg.
   - The over-estimate remains after ambiguous matches are removed, so it is not a matching
     artifact. It comes from range, which points to detector keypoint placement at the 768 input.
     Retraining the detector is out of scope.
2. **About 2 % false or wild detections.** Out of 2,800 re-solved detections, 63 match no arch and
   13 are ambiguous. Once acquired, a few of these were held for up to 10 s by
   `lost_policy: reacquire` / `lost_hard_stop_s: 10`. They explain the logged target spikes.

## AC1: the current derivation (code path)

No monocular depth is involved: `harness/liftoff/depth.py` feeds only the free-space eyes. Range
already comes from the gate's known geometry.

1. **Detect.** `gate_learned.LearnedGateEstimator.detect` runs `infer` (`gate_net.prepare_input` /
   `decode`; keypoints are rescaled to frame pixels).
   - It keeps keypoints with visibility ≥ 0.5 that lie inside the image and outside every mask.
   - It needs at least `min_keypoints` (4).
   - In async mode the detections come from an older frame, and that frame's capture stamp is kept.
2. **PnP.** `gate_cv.solve_planar_pnp` runs OpenCV IPPE against `GateType.object_points()`
   (`gate_types.py`): aperture-centre-relative, x right, y down, z = 0. For `inflatable_arch` that
   is 10 keypoints, outer 8.0 × 4.46 m, aperture 6.18 × 3.69 m, centre_offset 1.74 m. The result is
   the gate centre `t_cam` and normal `n_cam` in the camera frame, in Liftoff metres.
3. **Intrinsics.** `gate_cv.camera_from_config` builds a `geometry.CameraModel`: hfov pinhole,
   square pixels, principal point at the image centre unless `camera.principal_point` is set.
   `_camera_for` rescales it to the capture size.
4. **World.** `TrackingGateEstimator.detection_from_pose` takes the pose at
   `det_stamp + gates.pose_offset_s`, interpolated from the pose history by `_pose_at`. It then
   computes:
   - `r_wc = quat_to_matrix(q) @ body_to_camera().T`
   - `center = pos + r_wc @ (t_cam / world_scale)`
5. **Track.** `TrackingGateEstimator.update` produces `GateEstimate.target`. The runner's
   observation is `target - pos`, rotated by the spawn-yaw aligner. z is untouched.

## AC3: camera calibration

- **Config.** 2560 × 1080, hfov 120°, uptilt 20.0° (`configs/liftoff.yaml`; the drone file says
  20.0315°). The principal point is the image centre: Liftoff renders a centred pinhole camera, and
  the data below confirms it.
- **Sign, analytically.** `CameraModel.body_to_camera` puts the optical axis at
  (cos t, 0, sin t) in body FLU, so a positive uptilt pitches the camera up, as Liftoff's camera
  angle does.
- **Sign and value, empirically.** Pass A was re-run with the uptilt overridden; all six runs,
  matched detections. "Elev err" is the vertical angular error seen from the drone. A camera pitch
  error shows up as a constant elevation error at every range.

  | uptilt | 25-40 m z med | 25-40 m elev err | > 40 m z med | > 40 m elev err | matched < 40 m |
  |---|---|---|---|---|---|
  | **20 (config)** | **+0.01 m** | **−0.00°** | **−0.00 m** | **+0.00°** | 683 |
  | 19 | −0.65 m | −1.00° | −1.13 m | −0.74° | 681 |
  | 21 | +0.67 m | +0.99° | +0.66 m | +0.89° | 683 |
  | 22 | +1.33 m | +1.92° | +1.32 m | +1.73° | 684 |
  | −20 (sign flipped) | −26.8 m | −3.9° | +1.41 m | +0.23° | **11** (nothing lines up) |

  On the pad, the elevation error is +0.003° at 25-40 m and +0.001° beyond 60 m. In flight it is
  −0.5…+0.24° per bin. The configured uptilt is right to about 0.1°, and the sign is correct.

  The radial (PnP range) error is −0.1 m at 25-40 m and beyond 60 m. Range scales with the focal
  length, so hfov 120° is right too.
- **`offline/align.py` CP6b camera check** on the uc-89 runs: 
  - **Run used.** The camera check needs ≥ 50 frames with an arch within 25 m. These runs saved a
    frame every 10 ticks and spent most of their time far from the arches, so each run has only
    2-11 such frames: **CP6b is inconclusive on this data.** The run folders' `meta.json` is a run
    meta, not a capture meta, so `align_capture`'s steps were run directly on each run folder.
  - **Per run.** The best uptilt was 21-26°, often "camera off" by the one-step rule, on 2-11
    frames each.
  - **Pooled.** Over the six runs (30 frames), the peak is uptilt 22°, hfov 120°. The uptilt
    profile is flat between 20° and 23° (score 1.00 / 1.03 / 1.08 / 1.01 relative to 20°), with no
    hfov signal beyond ±5°.
  - **Weighing it.** The UC-84 reference-capture alignment found 19.03° with 120°. Pass A's
    2,800-detection elevation profile pins 20° far more tightly than 30 edge-score frames.
  - **Verdict.** No calibration change. The result is in
    `artifacts/harness/reports/uc91-backprojection/cp6b_camera_check.json`.

`camera.principal_point: [cx, cy]` was added for completeness (a non-centred render or a cropped
capture). It is validated to lie inside the image, scaled with the capture size
(`CameraModel.scaled_to`), and kept by `align.py`'s camera rebuilds. Liftoff leaves it unset.

## Ground truth and matching

- **Truth.** The route passages of the track's catalogue course
  (`detector_noise.load_catalogue_course`) in the contract frame, Liftoff metres. The telemetry is
  in the same frame: `unity_to_contract_position`, no origin shift; tick states × `world_scale`
  match `telemetry.csv` to 1e-6 m. The circuit repeats gate 0, so passages are de-duplicated to
  seven.
- **Matching.** Each estimate is matched by **bearing** from the drone, within 5°.
  - It is *ambiguous* when another arch lies within 5° or within 3° of the best match. Ambiguous
    estimates are excluded from the statistics and counted per bin. This is the mirrored-arch case:
    from the pad, several arches line up.
  - It is *unmatched* beyond 5°: a false positive or a wild PnP bearing.
- **Error components.** Estimate − truth, as world z; line-of-sight radial / tangential-horizontal /
  tangential-vertical (`sim.noise_model.los_frame`); and 3-D.

## Range modes (`gates.learned.range_mode`)

| mode | what it does |
|---|---|
| `pnp` (default; unchanged behaviour) | PnP range from the gate-type geometry; every solved detection is kept. |
| `pnp_checked` | Same PnP, but a detection is dropped when it fails `gate_range.check_detection`. `info.range_rejections` (ticks.jsonl) counts drops per reason, once per tick a cached detection is re-checked. |

The check (`src/drone_fly/harness/liftoff/gate_range.py`):

- **Size range.**
  - `size_range` = sqrt(A_obj / A_tangent): the hull area of the kept keypoints on the gate (m²,
    from the gate-type table) against the area of their camera rays on the plane perpendicular to
    their mean ray at unit distance.
  - Using rays rather than raw pixels makes it exact at the edge of the 120° lens, where pixels
    stretch by up to 1/cos³. It is also independent of the principal point.
  - An oblique gate reads farther by 1/sqrt(cos φ), so the ratio `r_pnp / r_size` is ≤ 1 for a
    correct pose. The band is asymmetric for that reason.
- **(a) Size ratio.** `size_ratio_min ≤ r_pnp / r_size ≤ size_ratio_max`.
- **(b) Normal tilt.** The world gate-plane normal tilts at most `max_normal_tilt_deg` out of the
  horizontal. Gates stand upright; this mostly catches the IPPE planar-ambiguity flip.
- **Truncated-box fallback.** The size range is *invalid* in three cases, and then only (b)
  applies:
  - a kept keypoint lies within `edge_margin_px` of the image border or a mask (a box clipped at
    the edge or by the HUD);
  - the kept hull covers less than `min_hull_frac` of the class's full hull (a partial detection);
  - fewer than 4 keypoints are kept.

  `max_normal_tilt_deg: null` disables (b).
- **Cost.** The check is pure NumPy and takes microseconds per detection, so it adds no measurable
  latency at the 26-31 Hz live loop.

| `range_check` knob | physics starting point | final (tuned on tune runs) |
|---|---|---|
| `size_ratio_min` | 0.47 = sqrt(cos 60°) × (1 − 2.5 sd) | 0.47 |
| `size_ratio_max` | 1.33 = 1 + 2.5 sd, sd 0.131 = PnP 12.8 % ⊕ size range 3 % at 50 m (1 px at the 768 input) | 1.33 |
| `max_normal_tilt_deg` | 30 | **45** |
| `edge_margin_px` | 8 | 8 |
| `min_hull_frac` | 0.5 | 0.5 |

## AC4: measured error vs range, before (`pnp`) and after (`pnp_checked`)

- **Data.**
  - Tune runs: uc-89 attempts 1-3 (T180325, T180418, T180500).
  - Held-out runs: attempts 4-6 (T180602, T180649, T180758).
  - Local copies are in `artifacts/harness/liftoff/runs/`; they are byte-identical to the
    uc-89-results copies.
- **How the knobs were tuned.**
  - The grid covered `size_ratio_min` 0.35-0.70, `size_ratio_max` 1.15-2.0 and tilt
    15 / 20 / 30 / 45 / off.
  - The selection rule was pre-registered in the code.
  - No grid cell passes the rule on the tune runs. The chosen cell gives the largest noise drop
    with retention held: only the tilt moved, 30 → 45.
  - The largest noise drop anywhere on the grid, ignoring retention, is **20 %**, and it keeps only
    35 % of the detections.
- **Error convention.** All values are in metres, estimate − truth, for matched, unambiguous
  detections.
- **Pad detections.** Static frames on the pad dominate the > 60 m bin (the far arches seen from
  the start), and those frames are almost error-free.

### Held-out, Pass A (primary: re-solved logged full-keypoint detections, telemetry pose)

| range | mode | n | z median | z sd | p90 \|z\| | radial median | kept |
|---|---|---|---|---|---|---|---|
| 18-25 m | pnp | 26 | +0.61 | 1.49 | 2.35 | −1.34 | — |
| | pnp_checked | 26 | +0.61 | 1.49 | 2.35 | −1.34 | 100 % |
| 25-40 m | pnp | 324 | +0.01 | 3.90 | 2.89 | −0.10 | — |
| | pnp_checked | 320 | +0.01 | 3.92 | 2.91 | −0.11 | 99 % |
| 40-60 m | pnp | 378 | −0.91 | 2.00 | 3.68 | +4.50 | — |
| | pnp_checked | 366 | −0.85 | 1.90 | 3.64 | +4.41 | 97 % |
| > 60 m | pnp | 653 | +0.00 | 0.84 | 0.89 | −0.12 | — |
| | pnp_checked | 646 | +0.00 | 0.81 | 0.86 | −0.12 | 99 % |
| **> 40 m pooled** | pnp | 1031 | −0.00 | 1.50 | 2.41 | +0.02 | — |
| | pnp_checked | 1012 | −0.00 | 1.43 | 2.24 | +0.01 | 98 % |

- Of the held-out detections, 61 were partial (keypoint identity is unknown, so they cannot be
  re-solved).
- Of the 1,424 re-solved held-out detections, 33 match no arch and 10 are ambiguous.

### Tune runs, Pass A

| range | mode | n | z median | z sd | p90 \|z\| | radial median | kept |
|---|---|---|---|---|---|---|---|
| 25-40 m | pnp | 299 | +0.01 | 2.57 | 2.98 | −0.06 | — |
| | pnp_checked | 292 | +0.01 | 2.57 | 2.97 | −0.06 | 98 % |
| 40-60 m | pnp | 435 | −1.12 | 2.38 | 4.18 | +4.88 | — |
| | pnp_checked | 416 | −1.10 | 2.23 | 4.15 | +4.67 | 96 % |
| > 40 m pooled | pnp | 1086 | −0.00 | 1.78 | 2.95 | +0.11 | — |
| | pnp_checked | 1057 | −0.00 | 1.67 | 2.87 | +0.10 | 97 % |

### Pre-registered AC4 rule (held-out, Pass A): **FAIL (noise clause unmet)**

- **Bias clause: met** in both modes, and already met by the baseline.
  - Within 40 m, |median z| ≤ 0.61 m in every bin with n ≥ 15.
  - Retention is ≥ 97 % in every bin.
- **Noise clause: not met.** Beyond 40 m, `pnp_checked` lowers p90 |z| from 2.41 to 2.24 m
  (**−7 %**) and the z sd from 1.50 to 1.43 m (**−5 %**). The rule needs −30 % on both.
- **Why it fails.** The far-range spread is not made of a few outliers that a range/tilt check can
  catch. It is the distributed 40-60 m radial over-estimate. Even giving up 65 % of the detections
  only buys −20 % on the tune runs.
- The held-out runs were not re-tuned.

### Pass B (tracker replay; secondary)

- **Fidelity.** The replayed `pnp` target matches the logged `gate.target` within 0.5 m on **82 %**
  of the held-out ticks where both hold a gate. Per run: 94 / 80 / 70 %. Median deviation 0.00 m,
  p90 4.5 m; the held state agrees on 95 % of ticks.
- **Tune runs.** Fidelity is 74 %, so Pass B is indicative only there.
- **Where it diverges.** The divergence comes mainly from partial detections, which are missing from the
  replay.
- **Held-out effect beyond 40 m.** `pnp_checked` changes the tracked target by 0 % (p90 |z|) and
  0.1 % (sd).

### Pass C (detector re-run on the saved frames, CUDA)

The saved frames are one every 10 ticks, so the bins are thinner than Pass A's. Partials are
included here. On the held-out runs, `pnp` → `pnp_checked`:

- **25-40 m:** z median +0.01 → +0.01 m, sd 1.18 → 1.18 m (n 46 → 45).
- **40-60 m:** z median −0.68 → −0.56 m, sd 1.87 → 1.90 m, radial +4.1 → +3.8 m (n 49 → 45).
- **> 40 m pooled:** p90 |z| 2.25 → 2.23 m.

Pass C agrees with Pass A.

### Logged target (what the policy saw), held-out, per tick while a gate is held

| range | n | z median | z sd | p90 \|z\| |
|---|---|---|---|---|
| 18-25 m | 117 | +0.57 | 2.47 | 5.33 |
| 25-40 m | 542 | +0.01 | 1.83 | 3.04 |
| 40-60 m | 173 | +0.63 | 2.15 | 3.26 |
| > 60 m | 71 (+24 ambiguous) | +4.55 | 2.10 | 4.55 |

The > 60 m row is most likely held targets matched to a far arch on the same bearing, not live far
detections; the Pass A > 60 m z median is 0.00. Within 60 m, the live gate z the policy saw was
right to well under a metre in median.

## Running the evaluation

```
drone-fly harness gates backprojection-eval \
  --tune-runs <run T180325> <run T180418> <run T180500> \
  --runs <run T180602> <run T180649> <run T180758> \
  --track artifacts/harness/trackdata/LiftoffArenaTrack04_0001.xml \
  --trackdata artifacts/harness/trackdata [--with-frames --device cuda]
```

- **Output.** The tool writes `report.md` and `report.json` under
  `artifacts/harness/reports/uc91-backprojection/<UTC stamp>/`, recording the invocation.
- **Pass C.** It needs CUDA torch. Run the CLI with `.venv-cuda\Scripts\drone-fly.exe`; the default
  `uv` env is CPU-only.
- **`--tune-runs`.** With it, tuning starts from the physics points. Without it, the configured
  `range_check` is evaluated as-is.
- **OneDrive hygiene.** The tool opens only `config.yaml`, `ticks.jsonl`, `telemetry.csv` and
  `frames.csv` in each run folder, plus the `frames/<file>` paths `frames.csv` lists. It never
  lists or walks a folder, so pointing it at OneDrive run folders is safe.

## Recommendation for the re-fly (AC7)

The AC4 noise rule failed, so re-fly with the **default `range_mode: pnp`**: use
`artifacts/harness/live-m14-w768.yaml` unchanged (detector width 768, `align_spawn_yaw: true`,
latency compensation off). Expect about the same gate passes as UC-89 (0/33): perception gives the
right gate z. Verify passes from telemetry / the recording against the track, never with
`gate_passed`. Then run this tool's logged-target pass on the new flights for the live gate-z error.
