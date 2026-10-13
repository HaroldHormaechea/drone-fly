# Use Case 70: Deployable student policy distilled from the teacher under a measured detector-noise model

## Summary
The UC68 teacher sees exact gate poses; the live harness only has the learned gate detector's estimates. These are noisy, biased with range and angle, sometimes missing, and late. This use case closes that gap in two steps. First, it **measures** the detector's real error statistics in Liftoff from auto-labelled captures, where XML geometry is projected through telemetry pose (the UC64 machinery). From these it builds a **noise model**, stored next to the detector weights it describes, so each detector version carries its own noise model. Second, it trains a **student** whose observation is exactly what the live harness assembles, including the measured perception age. The student learns from the teacher by DAgger-style distillation in the UC67 simulator, with the noise model applied to the teacher's exact observations, and is optionally fine-tuned with PPO afterwards. The student is exported as a checkpoint that `drone-fly harness run --harness liftoff` loads directly, with its observation-schema and action-contract versions embedded. Success is measured in the simulator under noise on held-out tracks; flying it in Liftoff is UC72.

## Acceptance Criteria
1. **Noise measurement:**
   - `drone-fly train detector-noise --detector <weights> --captures <dir>…` runs the detector over auto-labelled capture frames.
   - Per gate type, binned by range and bearing, it computes: bias and covariance of the position, yaw and size estimates; detection probability; false-positive rate; and inference latency on the local GPU.
   - It writes `noise_model.yaml` + `noise_report.md` (with plots) next to the detector, e.g. `artifacts/harness/liftoff/models/<detector-id>/`, and is included in the next detector release (UC71).
2. **Noise model application:**
   - `sim` applies a noise model to exact gate observations on the GPU, in batch: biased/correlated position and yaw error, dropout following the detection probability, hold-last-estimate behaviour matching the harness, and perception age/latency.
   - It is unit-tested for statistics within tolerance of the model, over ≥ 10⁵ samples.
3. **Student observation parity:**
   - The student's observation is assembled by the same code path as the live harness (`ContractObservationAssembler`) or a tested equivalent.
   - A test feeds identical inputs through both and asserts bit-identical vectors.
   - Observation-schema and action-contract versions are written into the checkpoint, and `harness run` refuses a mismatch with a clear error.
4. **Student architecture (user's goal):**
   - The deployable student is the user's own connectome model lineage: the K1 reservoir actor (`controller/reservoir.py`, `K1ReservoirAug`), warm-started from the trained `training/07-proc-course-flight` readout via the UC67 v1→v2 conversion.
   - Its frozen K1 connectome body is kept; distillation trains the readout and any added input layers.
   - A plain-MLP student may be trained only as a comparison baseline and is reported next to it.
   - The report states the connectome student's held-out completion against the baseline's.
5. **Distillation:**
   - `drone-fly train student --teacher <run> --noise <noise_model.yaml> --drone <id> --tracks liftoff:train --procedural 0.4` trains by DAgger: the student acts, the teacher labels.
   - The teacher-forcing ratio follows a configurable decay schedule.
   - An optional PPO fine-tune phase follows.
   - Outputs go to `artifacts/training/<drone-id>/<run-id>/` with the same structure as the teacher.
6. **Deployable export:** `drone-fly train export <run> --out <path>` writes a checkpoint loadable by `harness run --checkpoint`. A test loads it through the harness policy loader with fakes and runs one tick.
7. **Targets:** in the simulator with the noise model, dynamics and timing randomization, and ground starts, the student reaches **≥ 70% completion on held-out tracks**, and at least 85% of the teacher's completion on the same tracks. The report includes per-track numbers and a noise ablation: student without noise vs. with noise.
8. **Latency fit:** the student's control rate during training matches what the live harness achieves (from the UC67 timing defaults). The live harness's tick-deadline settings are aligned with it in the `liftoff` preset.
9. **Tests:** hermetic tests cover noise-model fitting on synthetic detector errors (recovered within tolerance), batch application statistics, observation parity, the checkpoint version guard, and the export round-trip.

## Potential Pitfalls & Open Questions
- **Missing input** — Labelled captures exist only for Infinity Loop (UC64). The noise model is first fitted on Infinity Loop and refitted when UC71 adds more maps; the report states which maps it covers.
- **Risk** — Detector errors on unseen maps may be much worse than on its training map. Fit the noise model on **held-out** capture segments only, never on the detector's own training frames.
- **Assumption** — The 30-dim observation (gates + state, no obstacle rays) is the deployment target. The 70-dim depth variant is optional and follows the same path if depth noise is also modelled.
- **Edge case** — Gate-loss stops (`gate_lost`) in the harness: the student must learn to fly through detection dropouts, and the harness's gate-loss timeout must be consistent with the dropout lengths seen in training.

## Original Description
> Ok we'll have to go the next step. Despite having "memorized" the course as you mentioned in a previous UC, we really aren't flying well. The AI just tumbles right from the start, with throttle/yaw/roll/pitch input from the start.
>
> We do have the possibility of exporting the track geometry for further trainings by converting it to pybullet, but the thing is, I want to GENERALIZE this so it can tackle random courses in liftoff.
>
> Ideas?

> Prepare everything as use cases. Once they are set-up, and we validate them together, wait for me to give you the go ahead (I'll be out of home for a couple hours so that's good time to use, but i'll leave after we have set this up). When doing the analysis, make sure whateve rwe create fits in a good understandable folder structure, and is scalable in case we need to train more drones and such.

## Clarifications
- Q: Which model is the end goal?
  A: The user's own trained connectome model (the 07 K1-reservoir lineage) must end up flying in Liftoff, after fine-tuning or whatever else it takes. The student is therefore the K1 reservoir actor warm-started from 07, distilled from the teacher (added 2026-10-13).
