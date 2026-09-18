"""AC3 — the policy is a connectome-seeded sparse ``torch.nn.Module``.

Its connection count equals the loaded connectome edge count and its density is far
below a dense ``N x N`` MLP, i.e. the parameterisation reflects connectome topology,
not a fully-connected layer.
"""

from __future__ import annotations

import torch
from torch import nn

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller import ConnectomePolicy, SparseConnectomeLayer

# The committed fixture density is ~0.092 (8288 / 300**2). A dense MLP would be 1.0.
# Assert well below this documented threshold to prove it is genuinely sparse.
SPARSITY_THRESHOLD = 0.5


def test_policy_is_torch_module(connectome: ConnectomeData) -> None:
    policy = ConnectomePolicy(connectome, force_shim=True)
    assert isinstance(policy, nn.Module)


def test_connection_count_matches_edge_count(connectome: ConnectomeData) -> None:
    policy = ConnectomePolicy(connectome, force_shim=True)
    assert policy.n_connections == connectome.edge_count
    assert policy.n_neurons == connectome.neuron_count


def test_policy_is_sparse_not_dense(connectome: ConnectomeData) -> None:
    policy = ConnectomePolicy(connectome, force_shim=True)
    n = policy.n_neurons
    density = policy.n_connections / (n * n)
    assert 0 < density < SPARSITY_THRESHOLD, (
        f"density {density:.4f} not connectome-sparse (< {SPARSITY_THRESHOLD})"
    )
    # A dense MLP layer would have N**2 weights; the connectome policy has far fewer.
    assert policy.n_connections < n * n


def test_trainable_parameter_count_equals_edges(connectome: ConnectomeData) -> None:
    """Trainable params come from connectome edges, not a dense weight matrix."""
    policy = ConnectomePolicy(connectome, force_shim=True)
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    # The shim's only trainable tensor is the per-edge weight vector.
    assert trainable == connectome.edge_count


def test_backend_recorded_as_shim_when_forced(connectome: ConnectomeData) -> None:
    """force_shim must select — and honestly record — the shim backend (no silent swap)."""
    policy = ConnectomePolicy(connectome, force_shim=True)
    assert policy.backend == "shim"
    assert isinstance(policy.layer, SparseConnectomeLayer)


def test_forward_pass_shape_and_finite(connectome: ConnectomeData) -> None:
    policy = ConnectomePolicy(connectome, force_shim=True)
    state = torch.zeros(policy.n_neurons, dtype=torch.float32)
    out = policy(state)
    assert out.shape == (policy.n_neurons,)
    assert torch.isfinite(out).all()
