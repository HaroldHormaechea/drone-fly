# Use Case 73: Camera-based obstacle "eyes" for the 70-input connectome

> **Status: draft. Needs refinement before development.** Expect to split this into 2–3 use cases once UC70–72 are done. Don't start a `develop` run on this file as it stands.

## Summary
Give the user's 70-input connectome (lineage 10/11/12) a vision-only source for its 40 obstacle-ray inputs, so it can fly in Liftoff without geometry or oracle data.

A small learned perception network plays the role of the fly's eyes and optic lobes:
- **Inputs:** the last 2–3 camera frames, plus ego-motion from telemetry (attitude, rotation rates and velocity, standing in for a real drone's gyro and visual odometry).
- **Outputs:** the 40 normalised free-distance values the model expects, plus a confidence for each ray. The rays form a 90° forward cone (±45°, Fibonacci spiral, `sim/course.py` `ray_cone`) with RAY_RANGE 4 m, scaled for Liftoff by `ray_range_scale`.
- **Forward rays** rely mainly on time-to-contact (looming) cues, because motion parallax carries no depth at the focus of expansion.
- **Obstacle map:** an obstacle map with memory, propagated by odometry, fills rays outside the camera's current view.

Training labels are free and exact. They are computed offline from track XML geometry and recorded poses, using the same approach as UC71's auto-labelling. Rays hit exactly what the 70 model's training simulator raycasts hit, gates included, so the inputs mean the same thing.

The estimator's error is then measured on held-out maps and turned into a noise model. The 70-input connectome is distilled and retrained under that noise (the UC70 pattern) and validated live in Liftoff. It is deployed alongside the 07 student, and the two are compared; it does not replace the 07 student.

The 4 m range gives only about 0.2–0.3 s of warning at racing speed. Extending it, which means retraining the 70 model with longer rays, is allowed if collision evidence shows too little warning.

End-to-end fine-tuning of the eyes and the connectome together is a later refinement, not part of this use case.

## Acceptance Criteria
1. **Field-of-view coverage report.** It maps each of the 40 ray directions to "in view" or "out of view" under the Liftoff camera's 20° uptilt and field of view. This is the first deliverable.
2. **Ray labels.** Labels are generated offline from track XML and recorded poses for every captured frame. They use the same hit rules as the 70 model's training simulator, gates included. Spot checks against hand-measured scenes agree within 5 cm.
3. **Inference inputs.** At inference the estimator consumes only camera frames and telemetry ego-motion (attitude, rates, velocity). A test proves that no track geometry or oracle input is reachable from the deployable path.
4. **Accuracy on held-out maps.** For each ray, report median absolute error and miss rate (a real obstacle within range reported as clear). Split the figures into in-view and out-of-view rays, and into distance bins. Targets are set during refinement.
5. **Latency.** Estimator latency p95 stays within the live control budget on the RTX 3050 (8 GB), together with the gate detector.
6. **Calibrated confidence.** For each confidence bin, the measured error rate matches the confidence within a stated tolerance.
7. **Noise model.** A measured noise model is produced, following the pattern of UC70's detector-noise model, and the simulator can inject it.
8. **Collision reduction in simulation.** The 70-input connectome student is retrained under the noise model. On unseen validation tracks it shows a lower collision share than the same model with its obstacle inputs blanked.
9. **Live validation.** It is validated live in Liftoff on at least one track with obstacles. Its collision share and completion are reported against the 07 student from UC70–72.
10. **Range decision.** If collisions show too little warning at 4 m, a longer RAY_RANGE is proposed with evidence. The 70 model is retrained at the new range before any live comparison.

## Potential Pitfalls & Open Questions
- **Missing input:** numeric targets for ray accuracy (AC4) and collision reduction (AC8) are still to be set during refinement.
- **Risk:** rays below the camera's view. With a 20° uptilt and a wide, short frame, the lower part of the ±45° cone may never be visible. The map must fill those rays, or a floor-plane prior is needed.
- **Risk:** at 4 m range there is very little warning at speed. This may force a range change and a retrain (AC10).
- **Edge case:** thin poles and nets, untextured walls, motion blur at racing speed, and lighting differences across maps.
- **Risk:** the GPU budget on the 8 GB card. The detector, estimator and connectome must fit together in memory and in per-frame latency.
- **Risk:** the 70-input lineage's own flying quality in Liftoff hasn't been established. It may need its own teacher distillation first, as UC70 does for 07.
- **Assumption:** odometry drift over the obstacle map's memory window is small enough at racing speed. It needs measuring.

## Original Description
> Later when we introduce vision, I'd want that us to have something "seeing" the differences between frames and with that building a set of "obstacles" and their estimated distance, which would then in turn be fed to the "vision neurons" in the 70 model...
>
> Yes, It'd be flying the 70-input connectome. And using a smaller AI for depth analysis would be analogous to having.. eyes for the fly that feed into the vision neurons, wouldn't it?

## Clarifications
- Q: Should the 70-input connectome with vision rays replace the 07 student or exist alongside it?
  A: Alongside it, with the two compared.
- Q: The rays only sense 4 m. Can the sensing range be extended, which means retraining the 70 model with longer rays?
  A: Allowed if the evidence says so.
- Q: May the eyes use telemetry ego-motion at inference?
  A: Yes: attitude, rates and velocity.
- Q: Should gates count as obstacles in the rays?
  A: The same as in the training environment.
- Note: the user asked for this to be captured as a draft that needs refinement.
