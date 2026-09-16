# 1. Record architecture decisions

- Status: accepted
- Date: 2026-09-16

## Context

drone-fly is a research prototype exploring whether the MaleCNS fruit-fly connectome can seed a
reinforcement-learning controller that flies a quadrotor through a timed race course in
simulation. The project owner is new to this area, so the consequential decisions should be
recorded in plain language and kept easy to revisit.

## Decision

We keep lightweight Markdown Architecture Decision Records (ADRs) in `docs/adr/`, one file per
decision, numbered incrementally. This first record also captures the three foundational scope
decisions made during project definition (see `PROJECT_BRIEF.md` for the full reasoning):

1. **Connectome role — connectome-seeded controller, not a fly-brain simulation.** MaleCNS is a
   static synaptic wiring diagram (topology only, no neuron dynamics or weights). We use it to
   shape/constrain a trainable PyTorch network whose connection strengths are learned by RL. We
   do not attempt a biophysically faithful simulation of the fly brain.

2. **Training target — an open quadrotor sim, not Liftoff.** Liftoff is a closed commercial game
   with no public API, so we train in an open Gymnasium-style quadrotor simulator
   (gym-pybullet-drones) modeling a timed gate course. Liftoff fidelity is a stretch/non-goal.

3. **Objective — reinforcement learning (PPO by default).** A quadrotor agent chases waypoints
   through gates, penalized for hitting floor/ceiling/obstacles and rewarded for minimal
   door-to-door time. We use Stable-Baselines3; PPO is the default (robust, forgiving), with SAC
   as an alternative if sample efficiency becomes a concern.

## Consequences

- The project is honest about what the connectome does and does not provide.
- Work is feasible on a single workstation without reverse-engineering a closed game.
- Future decisions (e.g., switching sim, changing the seeding scheme, adding SAC) get their own
  ADRs and can revise these without rewriting the whole brief.
