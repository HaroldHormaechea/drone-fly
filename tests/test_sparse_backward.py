"""AC6 — the sparse-propagation backward pass is explicitly validated.

Both propagation modes (``"scatter"`` — the autograd-stable ``index_add`` default — and
``"sparse"`` — ``torch.sparse.mm`` over a coalesced COO tensor) must produce finite,
non-trivial gradients on the sparse edge weights, including through a **batched** state
(the SB3 rollout shape). If the ``sparse`` mode were ever to become unstable in this
environment it should ``xfail`` with a reason rather than silently pass while broken; the
``scatter`` default carries the AC6 guarantee unconditionally.
"""

from __future__ import annotations

import pytest
import torch

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.policy import PROPAGATION_MODES, SparseConnectomeLayer


def _grad_after_backward(layer: SparseConnectomeLayer, state: torch.Tensor) -> torch.Tensor:
    out = layer(state)
    assert torch.isfinite(out).all()
    out.pow(2).sum().backward()
    grad = layer.edge_weight.grad
    assert grad is not None
    return grad


@pytest.mark.parametrize("mode", PROPAGATION_MODES)
def test_single_state_backward_is_finite_and_nontrivial(
    connectome: ConnectomeData, mode: str
) -> None:
    torch.manual_seed(0)
    layer = SparseConnectomeLayer(
        connectome.adjacency, sign=connectome.sign, n_steps=2, propagation_mode=mode
    )
    grad = _grad_after_backward(layer, torch.randn(layer.n_neurons))
    assert torch.isfinite(grad).all(), f"{mode}: non-finite gradient"
    assert (grad.abs() > 0).any(), f"{mode}: gradient is all-zero (trivial)"


@pytest.mark.parametrize("mode", PROPAGATION_MODES)
def test_batched_state_backward_is_finite_and_nontrivial(
    connectome: ConnectomeData, mode: str
) -> None:
    """Batched (B, N) autograd — the SB3 rollout shape — for both modes."""
    torch.manual_seed(0)
    layer = SparseConnectomeLayer(
        connectome.adjacency, sign=connectome.sign, n_steps=2, propagation_mode=mode
    )
    out = layer(torch.randn(5, layer.n_neurons))
    assert out.shape == (5, layer.n_neurons)
    grad = _grad_after_backward(layer, torch.randn(5, layer.n_neurons))
    assert torch.isfinite(grad).all(), f"{mode}: non-finite batched gradient"
    assert (grad.abs() > 0).any(), f"{mode}: batched gradient is all-zero (trivial)"


def test_both_modes_agree_on_forward(connectome: ConnectomeData) -> None:
    """scatter and sparse are two implementations of the same math -> same output."""
    torch.manual_seed(0)
    state = torch.randn(3, connectome.neuron_count)
    scatter = SparseConnectomeLayer(
        connectome.adjacency, sign=connectome.sign, n_steps=2, propagation_mode="scatter"
    )
    sparse = SparseConnectomeLayer(
        connectome.adjacency, sign=connectome.sign, n_steps=2, propagation_mode="sparse"
    )
    out_scatter = scatter(state)
    out_sparse = sparse(state)
    assert torch.allclose(out_scatter, out_sparse, atol=1e-5)
