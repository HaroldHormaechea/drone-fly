"""Controller stage: build the connectome-seeded neural-network policy.

Uses cached MaleCNS connectivity to shape/constrain a trainable PyTorch policy
network (connectivity structure seeded from the connectome; connection strengths
learned by RL). This is NOT a biophysically faithful fly-brain simulation.
"""
