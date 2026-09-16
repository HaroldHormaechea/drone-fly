"""AC2 — propagation is genuinely multi-step (multi-hop), not a single matmul.

With ``n_steps`` recurrent unroll steps, activity injected at one neuron should reach a
target that is two hops away but not one hop away only after ``n_steps >= 2``. We derive
such a (source, target) pair directly from the layer's effective weight matrix ``W``
(where ``out = W @ state``): a target ``t`` with ``W[t, s] == 0`` (not a 1-hop neighbour)
but ``(W @ W)[t, s] != 0`` (reachable in exactly 2 hops). A small impulse keeps ``tanh``
in its near-linear regime, so a non-zero reading at ``t`` reflects real 2-hop
connectivity rather than a nonlinearity artifact. Also checks activity stays finite over
many steps (the bounded nonlinearity prevents blow-up).
"""

from __future__ import annotations

import numpy as np
import torch

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.policy import SparseConnectomeLayer


def _effective_matrix(layer: SparseConnectomeLayer) -> np.ndarray:
    """Dense effective weight matrix ``W`` such that one hop is ``out = W @ state``."""
    ei = layer.edge_index.numpy()
    w = layer.effective_weight().detach().numpy()
    n = layer.n_neurons
    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[ei[0], ei[1]] = w  # row = postsynaptic, col = presynaptic
    return matrix


def _find_two_hop_pair(matrix: np.ndarray) -> tuple[int, int]:
    """Find (source, target) with target 2-hops-away-but-not-1-hop from source."""
    two_hop = matrix @ matrix
    n = matrix.shape[0]
    for s in range(n):
        one = matrix[:, s]
        two = two_hop[:, s]
        for t in range(n):
            if abs(one[t]) == 0.0 and abs(two[t]) > 1e-6:
                return s, t
    raise AssertionError("no 2-hop-not-1-hop pair found in the fixture")


def test_activity_appears_only_after_two_hops(connectome: ConnectomeData) -> None:
    matrix = _effective_matrix(SparseConnectomeLayer(connectome.adjacency, sign=connectome.sign))
    source, target = _find_two_hop_pair(matrix)

    impulse = torch.zeros(connectome.neuron_count, dtype=torch.float32)
    impulse[source] = 1e-3  # small -> tanh stays ~linear

    one_step = SparseConnectomeLayer(connectome.adjacency, sign=connectome.sign, n_steps=1)
    two_step = SparseConnectomeLayer(connectome.adjacency, sign=connectome.sign, n_steps=2)

    out1 = one_step(impulse)
    out2 = two_step(impulse)

    # Not a 1-hop neighbour: still (numerically) zero after a single step.
    assert abs(out1[target].item()) < 1e-9
    # Reached after two steps: genuinely multi-hop.
    assert abs(out2[target].item()) > 1e-9


def test_multi_hop_depends_on_step_count(connectome: ConnectomeData) -> None:
    """Different ``n_steps`` produce different states -> propagation truly unrolls."""
    impulse = torch.zeros(connectome.neuron_count, dtype=torch.float32)
    impulse[connectome.neuron_count // 3] = 1e-3
    out1 = SparseConnectomeLayer(connectome.adjacency, sign=connectome.sign, n_steps=1)(impulse)
    out3 = SparseConnectomeLayer(connectome.adjacency, sign=connectome.sign, n_steps=3)(impulse)
    assert not torch.allclose(out1, out3)


def test_activity_stays_finite_over_many_steps(connectome: ConnectomeData) -> None:
    """The bounded tanh nonlinearity keeps activity finite (no explode/vanish to NaN)."""
    layer = SparseConnectomeLayer(connectome.adjacency, sign=connectome.sign, n_steps=50)
    torch.manual_seed(0)
    state = torch.randn(connectome.neuron_count) * 5.0  # large drive
    out = layer(state)
    assert torch.isfinite(out).all()
    # tanh bounds every entry to [-1, 1].
    assert out.abs().max().item() <= 1.0 + 1e-6
