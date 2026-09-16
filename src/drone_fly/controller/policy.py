"""Connectome-derived trainable PyTorch propagation layer.

The layer's connectivity is *seeded by the connectome*: its trainable parameters are
the connectome edges (a sparse ``N x N`` weight matrix whose non-zeros are exactly the
real synaptic connections), not a dense MLP. A forward pass propagates a neuron-state
vector through that weighted connectivity for a configurable number of recurrent
unroll steps, applying a bounded (``tanh``) nonlinearity after each hop.

UC-02 hardening (replaces UC-01's one-hop shim)
------------------------------------------------
:class:`SparseConnectomeLayer` is hardened **in place** into an RL-ready substrate:

* **Fixed E/I sign mask.** When per-neuron sign is available, each edge carries a fixed
  ``+1`` / ``-1`` sign taken from its *presynaptic* neuron (Dale's law). The effective
  weight is ``sign_mask * raw_weight.abs()``: training adjusts synapse *magnitudes* but
  can never flip a synapse's excitatory/inhibitory sign. ``raw_weight`` is initialised
  to the connectome magnitudes exactly (``coo.data``), so at init the effective
  magnitude per edge equals ``|coo.data|`` (no ``softplus``-style magnitude collapse).
  When sign data is absent the mask degrades to a documented default: no sign is
  imposed and the raw weight is used directly (UC-01 behaviour), the limitation being
  that E/I biology is then not enforced.
* **Multi-step recurrent propagation.** ``state_{t+1} = tanh(W_eff @ state_t)`` unrolled
  ``n_steps`` (> 1 by default) so activity flows across multiple hops of the graph, not
  a single matmul. The bounded nonlinearity keeps activity finite over many steps.
* **Batched state.** ``forward`` accepts either a ``(N,)`` vector (returns ``(N,)``, the
  UC-01 contract) or a batched ``(B, N)`` tensor (returns ``(B, N)``) for SB3 rollouts.
* **Two propagation modes.** ``propagation_mode="scatter"`` (default) uses an
  autograd-stable ``index_add`` gather/scatter; ``"sparse"`` uses ``torch.sparse.mm``.
  The scatter path is the AC6 fallback if sparse autograd proves unstable.

Only the per-edge weight vector is an :class:`~torch.nn.Parameter`; the edge topology
and the sign mask are registered buffers (non-trainable). Parameter count therefore
equals the connectome edge count exactly — one trainable scalar per synapse.

Backends
--------
* **Guarded fallback — in-repo shim** (``backend == "shim"``):
  :class:`SparseConnectomeLayer`, described above. This is what runs in CI.
* **Primary — AxonWeave** (``backend == "axonweave"``): kept as a permanently-dead,
  commented no-op. The ``axonweave`` distribution is not installable in this environment
  (absent from PyPI; source repo not publicly reachable), so it is intentionally left
  out of ``pyproject.toml`` and never imported. The hook is retained only to document
  the intended integration point should a real distribution ever become reachable.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from drone_fly.connectome.loader import ConnectomeData

#: Default number of recurrent unroll steps. > 1 so propagation is genuinely multi-hop
#: (AC2). The committed fixture has all descending (motor) neurons reachable from the
#: visual (sensory) neurons within 2 hops, so 2 steps make readout gradients reach the
#: sparse weights non-trivially. Configurable per instance.
DEFAULT_N_STEPS = 2

#: Supported sparse-propagation strategies. ``"scatter"`` (autograd-stable index_add) is
#: the default and the AC6 fallback; ``"sparse"`` uses ``torch.sparse.mm``.
PROPAGATION_MODES = ("scatter", "sparse")


class SparseConnectomeLayer(nn.Module):
    """A sparse, connectome-seeded trainable recurrent propagation layer.

    The connectome edges become a trainable parameter vector (one weight per real
    synaptic connection). ``forward`` unrolls ``state_{t+1} = tanh(W_eff @ state_t)`` for
    ``n_steps`` steps, where ``W_eff`` is the sparse ``N x N`` matrix reconstructed from
    the fixed edge indices, the trainable magnitudes, and the fixed E/I sign mask.
    Parameter count equals the number of connectome edges — genuinely sparse (``nnz``
    params, not ``N**2``).

    Parameters
    ----------
    adjacency:
        ``N x N`` CSR matrix. ``adjacency[i, j]`` is the weight from neuron ``j`` onto
        neuron ``i`` (so a directed edge ``j -> i`` has ``row = i``, ``col = j``).
    sign:
        Optional length-``N`` ``±1`` array of per-neuron E/I sign. When given, each
        edge's fixed sign is taken from its *presynaptic* (column) neuron. ``None``
        degrades to a documented default: no sign mask (raw weights used directly).
    n_steps:
        Number of recurrent unroll steps (must be ``>= 1``; default
        :data:`DEFAULT_N_STEPS`).
    propagation_mode:
        ``"scatter"`` (default) or ``"sparse"`` — see :data:`PROPAGATION_MODES`.
    """

    def __init__(
        self,
        adjacency: sp.csr_matrix,
        *,
        sign: np.ndarray | None = None,
        n_steps: int = DEFAULT_N_STEPS,
        propagation_mode: str = "scatter",
    ) -> None:
        super().__init__()
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
        if propagation_mode not in PROPAGATION_MODES:
            raise ValueError(
                f"propagation_mode must be one of {PROPAGATION_MODES}, got {propagation_mode!r}."
            )

        coo = adjacency.tocoo()
        self.n_neurons = int(adjacency.shape[0])
        self.n_steps = int(n_steps)
        self.propagation_mode = propagation_mode

        # Edge topology is fixed (registered buffers); only the magnitudes are trainable.
        edge_index = np.vstack([coo.row, coo.col]).astype(np.int64)
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))
        # raw_weight init = connectome magnitudes EXACTLY (coo.data). The effective weight
        # applies sign and abs() (below); at init the magnitude is preserved bit-for-bit.
        self.edge_weight = nn.Parameter(torch.as_tensor(coo.data, dtype=torch.float32))

        # Fixed per-edge E/I sign buffer = sign of the presynaptic (column) neuron.
        # `has_sign_mask` records whether biology was available; the sign buffer is
        # non-trainable, so E/I polarity is immutable under gradient descent.
        if sign is not None:
            sign_arr = np.asarray(sign).reshape(-1)
            if sign_arr.shape[0] != self.n_neurons:
                raise ValueError(
                    f"sign array length ({sign_arr.shape[0]}) must equal neuron count "
                    f"({self.n_neurons})."
                )
            edge_sign = sign_arr[coo.col].astype(np.float32)
            self.register_buffer("sign_mask", torch.as_tensor(edge_sign, dtype=torch.float32))
            self.has_sign_mask = True
        else:
            self.register_buffer("sign_mask", None)
            self.has_sign_mask = False

    @property
    def n_connections(self) -> int:
        """Number of trainable connections (== connectome edge count)."""
        return int(self.edge_weight.shape[0])

    def effective_weight(self) -> torch.Tensor:
        """Return the per-edge effective weight actually used in propagation.

        With an E/I sign mask: ``sign_mask * raw_weight.abs()`` — the sign is fixed and
        only the (non-negative) magnitude is learned. Without one (documented degrade):
        the raw trainable weight, unchanged (UC-01 behaviour).
        """
        if self.has_sign_mask:
            sign_mask = cast(torch.Tensor, self.sign_mask)
            return sign_mask * self.edge_weight.abs()
        return self.edge_weight

    def _propagate(self, state: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """One hop: ``(W_eff @ state^T)^T`` for a batched ``(B, N)`` state.

        ``scatter`` gathers the presynaptic activations and scatter-adds the weighted
        messages onto the postsynaptic rows (autograd-stable). ``sparse`` builds a
        coalesced sparse COO matrix and uses ``torch.sparse.mm``.
        """
        edge_index = cast(torch.Tensor, self.edge_index)
        row, col = edge_index[0], edge_index[1]
        if self.propagation_mode == "scatter":
            messages = state.index_select(1, col) * weight.unsqueeze(0)  # (B, E)
            out = torch.zeros_like(state)
            return out.index_add(1, row, messages)
        # sparse mode: (N, N) @ (N, B) -> (N, B) -> (B, N)
        w = torch.sparse_coo_tensor(
            edge_index,
            weight,
            size=(self.n_neurons, self.n_neurons),
            check_invariants=False,
        ).coalesce()
        return torch.sparse.mm(w, state.t()).t()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Unroll ``tanh(W_eff @ state)`` for ``n_steps`` and return the final state.

        Accepts a ``(N,)`` vector (returns ``(N,)`` — the UC-01 contract) or a batched
        ``(B, N)`` tensor (returns ``(B, N)``).
        """
        single = state.dim() == 1
        x = state.to(torch.float32)
        if single:
            x = x.unsqueeze(0)
        elif x.dim() != 2:
            raise ValueError(
                f"state must be 1-D (N,) or 2-D (B, N); got shape {tuple(state.shape)}."
            )
        if x.shape[-1] != self.n_neurons:
            raise ValueError(
                f"state last dim ({x.shape[-1]}) must equal neuron count ({self.n_neurons})."
            )

        weight = self.effective_weight()
        for _ in range(self.n_steps):
            x = torch.tanh(self._propagate(x, weight))
        return x.squeeze(0) if single else x


class ConnectomePolicy(nn.Module):
    """A ``torch.nn.Module`` wrapping the connectome-seeded propagation layer.

    Instantiates :class:`SparseConnectomeLayer` from loaded connectome data, forwarding
    the per-neuron E/I sign when present. The active backend is recorded in
    :pyattr:`backend` (``"shim"`` in this environment) — no silent substitution. The
    ``(N,)``-in / ``(N,)``-out contract and the ``force_shim`` / ``backend`` surface are
    preserved from UC-01; the hardened multi-step propagation replaces the one-hop pass.

    Attributes
    ----------
    n_neurons:
        Number of neurons (``N``).
    n_connections:
        Number of connectome connections (sparse; equals the loaded edge count).
    backend:
        Which propagation backend is active.
    """

    def __init__(
        self,
        data: ConnectomeData,
        *,
        force_shim: bool = False,
        n_steps: int = DEFAULT_N_STEPS,
        propagation_mode: str = "scatter",
    ) -> None:
        super().__init__()
        self.n_neurons = data.neuron_count
        self._edge_count = data.edge_count
        self.backend = "shim"
        self.layer: nn.Module

        layer = None
        if not force_shim:
            layer = _try_build_axonweave_layer(data.adjacency)
        if layer is not None:
            self.layer = layer
            self.backend = "axonweave"
        else:
            self.layer = SparseConnectomeLayer(
                data.adjacency,
                sign=data.sign,
                n_steps=n_steps,
                propagation_mode=propagation_mode,
            )
            self.backend = "shim"

    @property
    def n_connections(self) -> int:
        """Number of connectome connections realised by the active backend."""
        n = getattr(self.layer, "n_connections", None)
        return int(n) if n is not None else int(self._edge_count)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Propagate a neuron-state vector through the connectome layer."""
        return self.layer(state)


def _try_build_axonweave_layer(adjacency: sp.csr_matrix) -> nn.Module | None:
    """Permanently-dead AxonWeave hook — always returns ``None`` (documented no-op).

    The ``axonweave`` distribution is not installable in this environment (absent from
    PyPI; source repo not publicly reachable), so it is intentionally kept out of
    ``pyproject.toml`` and never imported. The intended integration — build a real
    AxonWeave torch layer from the sparse matrix via the documented facade
    (``core.graph.ConnectomeGraph`` -> ``core.brain.BiologicalBrain`` ->
    ``.torch_layer(trainable_edges=True)``) — is retained only as the commented sketch
    below so the wiring point is documented should a real distribution ever ship. Until
    then the in-repo :class:`SparseConnectomeLayer` shim is the sole backend.
    """
    # import numpy as _np
    # from axonweave.core.brain import BiologicalBrain
    # from axonweave.core.graph import ConnectomeGraph
    #
    # body_ids = _np.arange(adjacency.shape[0])
    # graph = ConnectomeGraph(weights=adjacency.tocsr(), body_ids=body_ids)
    # brain = BiologicalBrain(graph)
    # return brain.torch_layer(trainable_edges=True)
    return None
