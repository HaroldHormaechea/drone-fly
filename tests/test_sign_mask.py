"""AC3 — fixed E/I sign mask preserves excitatory/inhibitory biology under training.

The hardened :class:`SparseConnectomeLayer` derives a per-edge sign from the
*presynaptic* (column) neuron's E/I sign (Dale's law) and uses an effective weight of
``sign_mask * raw_weight.abs()``. Consequences verified here:

* The sign mask equals ``sign[presynaptic_col]`` for every edge.
* Gradient training changes weight *magnitudes* but never flips a synapse's sign.
* At init the effective magnitude equals ``|coo.data|`` exactly (guards the reparam
  against a silent regression to a magnitude-collapsing transform like softplus).
* When ``sign`` is absent the mask degrades to a documented default (raw weight used
  directly; ``has_sign_mask is False``).
"""

from __future__ import annotations

import numpy as np
import torch

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.policy import SparseConnectomeLayer


def test_sign_mask_taken_from_presynaptic_neuron(connectome: ConnectomeData) -> None:
    layer = SparseConnectomeLayer(connectome.adjacency, sign=connectome.sign)
    assert layer.has_sign_mask
    coo = connectome.adjacency.tocoo()
    expected = np.asarray(connectome.sign)[coo.col].astype(np.float32)
    np.testing.assert_array_equal(layer.sign_mask.numpy(), expected)
    # Non-trivial mask: the fixture carries both excitatory and inhibitory presynapses.
    assert set(np.unique(layer.sign_mask.numpy()).tolist()) == {-1.0, 1.0}


def test_init_effective_magnitude_equals_connectome_magnitudes(
    connectome: ConnectomeData,
) -> None:
    """Reparam guard: at init |effective_weight| == |coo.data| edge-for-edge."""
    layer = SparseConnectomeLayer(connectome.adjacency, sign=connectome.sign)
    coo = connectome.adjacency.tocoo()
    eff = layer.effective_weight().detach().numpy()
    np.testing.assert_allclose(np.abs(eff), np.abs(coo.data), atol=1e-6)
    # Effective sign is exactly the fixed mask at init.
    np.testing.assert_array_equal(np.sign(eff), layer.sign_mask.numpy())


def test_training_changes_magnitude_but_never_flips_sign(
    connectome: ConnectomeData,
) -> None:
    torch.manual_seed(0)
    layer = SparseConnectomeLayer(connectome.adjacency, sign=connectome.sign, n_steps=2)
    sign_mask = layer.sign_mask.clone()
    mag_before = layer.effective_weight().detach().abs().clone()

    opt = torch.optim.SGD(layer.parameters(), lr=0.5)
    # A real loss with non-trivial gradients on the sparse weights.
    state = torch.randn(4, layer.n_neurons)
    for _ in range(5):
        opt.zero_grad()
        out = layer(state)
        loss = out.pow(2).sum()
        loss.backward()
        opt.step()

    eff_after = layer.effective_weight().detach()
    mag_after = eff_after.abs()

    # Magnitudes genuinely moved (training is doing something).
    assert not torch.allclose(mag_before, mag_after)
    # ...but no synapse's sign flipped: effective sign is still the fixed mask
    # everywhere the magnitude is non-zero (sign() is 0 only at an exact-zero weight).
    eff_sign = torch.sign(eff_after)
    unchanged = (eff_sign == sign_mask) | (eff_after == 0)
    assert bool(unchanged.all()), "an E/I sign flipped under training"
    # And no entry flipped to the opposite polarity.
    assert not bool((eff_sign == -sign_mask).any())


def test_degrade_to_default_when_sign_absent(connectome: ConnectomeData) -> None:
    """No sign data -> no mask imposed; effective weight is the raw trainable weight."""
    layer = SparseConnectomeLayer(connectome.adjacency, sign=None)
    assert layer.has_sign_mask is False
    assert layer.sign_mask is None
    np.testing.assert_array_equal(
        layer.effective_weight().detach().numpy(),
        layer.edge_weight.detach().numpy(),
    )


def test_synthetic_connectome_has_no_sign_mask(
    synthetic_connectome: ConnectomeData,
) -> None:
    assert synthetic_connectome.sign is None
    layer = SparseConnectomeLayer(synthetic_connectome.adjacency, sign=synthetic_connectome.sign)
    assert layer.has_sign_mask is False
