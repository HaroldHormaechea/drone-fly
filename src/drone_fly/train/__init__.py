"""Training stage: the reinforcement-learning training loop.

Wires the connectome-seeded controller policy to the racing environment and trains
it with Stable-Baselines3 (PPO by default; SAC as an alternative). Writes checkpoints
to artifacts/models/ and TensorBoard metrics to artifacts/logs/.
"""
