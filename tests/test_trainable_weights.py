"""AC1 — trainable sparse weights over a fixed sparsity pattern.

The connectome layer holds one trainable scalar per real synaptic edge; the edge
topology (which edges exist) is a non-trainable buffer. Verified:

* A training step updates ``edge_weight`` but leaves ``edge_index`` (the sparsity
  pattern) byte-for-byte unchanged.
* Only ``edge_weight`` carries gradients; ``edge_index`` / ``sign_mask`` do not.
* Parameter count equals the connectome edge count (genuinely sparse, not ``N**2``).
"""

from __future__ import annotations

import torch

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.policy import SparseConnectomeLayer


def test_only_edge_weight_is_trainable(connectome: ConnectomeData) -> None:
    layer = SparseConnectomeLayer(connectome.adjacency, sign=connectome.sign)
    trainable = [name for name, p in layer.named_parameters() if p.requires_grad]
    assert trainable == ["edge_weight"]
    # edge_index is a buffer, not a parameter, and does not require grad.
    assert not layer.edge_index.requires_grad
    assert layer.sign_mask is None or not layer.sign_mask.requires_grad


def test_parameter_count_equals_edge_count(connectome: ConnectomeData) -> None:
    layer = SparseConnectomeLayer(connectome.adjacency, sign=connectome.sign)
    total = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    assert total == connectome.edge_count
    assert layer.n_connections == connectome.edge_count


def test_training_updates_weights_but_not_topology(connectome: ConnectomeData) -> None:
    torch.manual_seed(0)
    layer = SparseConnectomeLayer(connectome.adjacency, sign=connectome.sign, n_steps=2)
    weight_before = layer.edge_weight.detach().clone()
    edge_index_before = layer.edge_index.detach().clone()

    opt = torch.optim.SGD(layer.parameters(), lr=0.5)
    state = torch.randn(4, layer.n_neurons)
    opt.zero_grad()
    loss = layer(state).pow(2).sum()
    loss.backward()

    # Gradient exists on the weights and is finite.
    assert layer.edge_weight.grad is not None
    assert torch.isfinite(layer.edge_weight.grad).all()
    assert (layer.edge_weight.grad != 0).any()

    opt.step()

    # Weights moved; the sparsity pattern (which edges exist) did not.
    assert not torch.equal(weight_before, layer.edge_weight.detach())
    assert torch.equal(edge_index_before, layer.edge_index)
    # Edge count is unchanged: no edge was added or removed.
    assert layer.n_connections == connectome.edge_count
