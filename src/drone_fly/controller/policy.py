"""Connectome-derived trainable PyTorch policy.

The policy's connectivity is *seeded by the connectome*: its trainable parameters are
the connectome edges (a sparse ``N x N`` weight matrix whose non-zeros are exactly the
real synaptic connections), not a dense MLP. A forward pass propagates a neuron-state
vector one hop through that weighted connectivity and applies a bounded nonlinearity.

Two backends
------------
* **Primary — AxonWeave** (``backend == "axonweave"``): build a real AxonWeave layer
  from the in-memory sparse matrix
  (``ConnectomeGraph`` -> ``BiologicalBrain`` -> ``.torch_layer(trainable_edges=True)``).
* **Guarded fallback — in-repo shim** (``backend == "shim"``):
  :class:`SparseConnectomeLayer`, a ``torch.nn.Module`` holding the real connectome
  edges as a sparse trainable parameter. Used **only** when AxonWeave (or its
  ``core.*`` internals) is unavailable on the installed version — never as a silent
  substitution: the chosen backend is recorded on the policy.

**Environment note (UC-01):** at implementation time the ``axonweave`` distribution was
not installable in this environment (absent from PyPI; source repo not publicly
reachable), so the shim backend is what actually runs in CI. The primary path is kept
intact and is selected automatically if/when AxonWeave becomes importable. See the
developer's implementation notes and the plan's step-0.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from drone_fly.connectome.loader import ConnectomeData


class SparseConnectomeLayer(nn.Module):
    """A sparse, connectome-seeded trainable linear propagation layer.

    The connectome edges become a trainable parameter vector (one weight per real
    synaptic connection). ``forward`` computes ``activation(W @ x)`` where ``W`` is the
    sparse ``N x N`` matrix reconstructed from the edge indices and trainable weights.
    Parameter count equals the number of connectome edges — it is genuinely sparse
    (``nnz`` params, not ``N**2``).
    """

    def __init__(self, adjacency: sp.csr_matrix) -> None:
        super().__init__()
        coo = adjacency.tocoo()
        # Edge topology is fixed (a registered buffer); edge weights are trainable.
        edge_index = np.vstack([coo.row, coo.col]).astype(np.int64)
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))
        self.n_neurons = int(adjacency.shape[0])
        self.edge_weight = nn.Parameter(torch.as_tensor(coo.data, dtype=torch.float32))

    @property
    def n_connections(self) -> int:
        """Number of trainable connections (== connectome edge count)."""
        return int(self.edge_weight.shape[0])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Propagate ``state`` one hop through the sparse connectome and squash it."""
        state = state.reshape(-1).to(torch.float32)
        edge_index = cast(torch.Tensor, self.edge_index)
        # check_invariants=False: the edge index is built from a valid CSR->COO
        # conversion, so it is well-formed by construction; opting out silences torch's
        # implicit-check warning without changing behaviour.
        w = torch.sparse_coo_tensor(
            edge_index,
            self.edge_weight,
            size=(self.n_neurons, self.n_neurons),
            check_invariants=False,
        ).coalesce()
        propagated = torch.sparse.mm(w, state.unsqueeze(1)).squeeze(1)
        return torch.tanh(propagated)


class ConnectomePolicy(nn.Module):
    """A ``torch.nn.Module`` policy instantiated from loaded connectome data.

    Attempts the AxonWeave primary path first; on any import/attribute failure it falls
    back to :class:`SparseConnectomeLayer`. The active backend is recorded in
    :pyattr:`backend` (``"axonweave"`` or ``"shim"``) — the substitution is never
    silent.

    Attributes
    ----------
    n_neurons:
        Number of neurons (``N``).
    n_connections:
        Number of connectome connections (sparse; equals the loaded edge count).
    backend:
        Which propagation backend is active.
    """

    def __init__(self, data: ConnectomeData, *, force_shim: bool = False) -> None:
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
            self.layer = SparseConnectomeLayer(data.adjacency)
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
    """Build a real AxonWeave torch layer, or return ``None`` if unavailable.

    Uses the documented AxonWeave facade
    (``core.graph.ConnectomeGraph`` -> ``core.brain.BiologicalBrain`` ->
    ``.torch_layer(trainable_edges=True)``). Any missing module/attribute (AxonWeave
    not installed, or its ``core.*`` internals renamed on the installed version) yields
    ``None`` so the caller falls back to the in-repo shim — we never improvise around a
    private API.
    """
    try:  # pragma: no cover - exercised only where AxonWeave is installed
        import numpy as _np
        from axonweave.core.brain import BiologicalBrain
        from axonweave.core.graph import ConnectomeGraph

        body_ids = _np.arange(adjacency.shape[0])
        graph = ConnectomeGraph(weights=adjacency.tocsr(), body_ids=body_ids)
        brain = BiologicalBrain(graph)
        return brain.torch_layer(trainable_edges=True)
    except Exception:  # noqa: BLE001 - any failure => documented shim fallback
        return None
