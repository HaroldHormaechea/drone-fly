"""Offline loader for a cached MaleCNS connectivity matrix.

This module reads a connectome adjacency matrix that has already been cached to
local disk (in the ``connectome_data_prep`` format: a SciPy sparse ``.npz`` plus a
sidecar ``*_meta.csv``). It performs **no network I/O** and never imports
``neuprint`` — provisioning the artifact (from neuPrint or the
``connectome_data_prep`` repo) is a separate, network-gated concern handled by
``scripts/build_test_fixture.py`` and by manual dataset provisioning. Keeping the
load path offline is what lets the UC-01 smoke test run in CI with networking
disabled (AC2).

Data format (``connectome_data_prep``)
--------------------------------------
* ``<name>.npz`` — a SciPy sparse matrix, ``N x N``, where entry ``[i, j]`` is the
  (input-proportion) connection weight from neuron ``j`` onto neuron ``i``. Loaded
  and normalised to a CSR ``float32`` matrix.
* ``<name>_meta.csv`` — one row per neuron. The matrix row/column index is the
  ``idx`` column; the stable biological identifier is the ``bodyid`` column. Either
  may be absent in reduced fixtures, in which case identifiers fall back to the row
  order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

#: Environment variable naming the directory that holds the cached artifact.
CONNECTOME_DIR_ENV = "DRONE_FLY_CONNECTOME_DIR"

#: Default artifact directory (relative to the current working directory), used
#: when neither an explicit path nor the environment variable is supplied.
DEFAULT_CONNECTOME_DIR = "data/connectome"


@dataclass(frozen=True)
class ExpectedScale:
    """A documented (neuron, edge) count pair used for scale assertions."""

    neuron_count: int
    edge_count: int


# --- Documented scale constants -------------------------------------------------
#
# FIXTURE_EXPECTED_SCALE describes the small, committed, real-MaleCNS subgraph that
# ships under tests/fixtures/. It is produced deterministically by
# scripts/build_test_fixture.py (top-degree slice of the canonical MaleCNS matrix),
# so the counts below are exact and reproducible. Tests assert the loaded fixture
# matches these numbers exactly (AC1).
FIXTURE_EXPECTED_SCALE = ExpectedScale(neuron_count=300, edge_count=10600)

# MALECNS_V1_EXPECTED_SCALE describes the full canonical MaleCNS whole-brain matrix
# (connectome_data_prep: data/maleCNS/mcns_inprop_all_neuron.npz). These counts come
# from that published artifact and are asserted only by a test that is auto-skipped
# unless the multi-GB full dataset is actually present (see AC1 reconciliation in the
# plan). They document the real dataset scale the POC is a stand-in for.
MALECNS_V1_EXPECTED_SCALE = ExpectedScale(neuron_count=161429, edge_count=25083972)


@dataclass
class ConnectomeData:
    """A loaded connectome, exposing the fields downstream stages depend on.

    Attributes
    ----------
    adjacency:
        ``N x N`` CSR ``float32`` matrix. ``adjacency[i, j]`` is the connection
        weight from neuron ``j`` onto neuron ``i``.
    neuron_ids:
        Length-``N`` array of stable neuron identifiers (``bodyid`` where available,
        otherwise the row index), aligned to the matrix row/column order.
    source:
        A human-readable provenance tag (e.g. the artifact directory or fixture
        name) recorded for traceability.
    """

    adjacency: sp.csr_matrix
    neuron_ids: np.ndarray
    source: str

    @property
    def neuron_count(self) -> int:
        """Number of neurons (matrix dimension ``N``)."""
        return int(self.adjacency.shape[0])

    @property
    def edge_count(self) -> int:
        """Number of connections (stored non-zeros in the sparse matrix)."""
        return int(self.adjacency.nnz)

    @property
    def synapse_count(self) -> int:
        """Alias of :pyattr:`edge_count`; connectome edges are synaptic links."""
        return self.edge_count


def _resolve_artifact_dir(path: str | os.PathLike[str] | None) -> Path:
    """Resolve the artifact directory: explicit arg -> env var -> default."""
    if path is not None:
        return Path(path)
    env = os.environ.get(CONNECTOME_DIR_ENV)
    if env:
        return Path(env)
    return Path(DEFAULT_CONNECTOME_DIR)


def _find_matrix_and_meta(target: Path) -> tuple[Path, Path | None]:
    """Locate the ``.npz`` matrix (and optional ``*_meta.csv``) at ``target``.

    ``target`` may be a directory containing exactly one ``.npz`` (with an optional
    sidecar ``<stem>_meta.csv``), or a direct path to a ``.npz`` file (meta inferred
    as the sibling ``<stem>_meta.csv``).
    """
    if target.is_file() and target.suffix == ".npz":
        npz_path = target
    elif target.is_dir():
        candidates = sorted(target.glob("*.npz"))
        if not candidates:
            raise FileNotFoundError(
                f"No cached connectome '.npz' matrix found in '{target}'. "
                f"Provision one (e.g. run scripts/build_test_fixture.py to build the "
                f"committed test fixture, or download a connectome_data_prep matrix "
                f"into this directory). This loader never fetches over the network."
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Expected exactly one '.npz' in '{target}', found {len(candidates)}: "
                f"{[p.name for p in candidates]}. Point DRONE_FLY_CONNECTOME_DIR (or the "
                f"path argument) at a directory or file with a single matrix."
            )
        npz_path = candidates[0]
    else:
        raise FileNotFoundError(
            f"Connectome artifact path '{target}' does not exist. Set "
            f"{CONNECTOME_DIR_ENV} or pass an explicit path to a directory (or .npz) "
            f"holding a cached connectome matrix. No network fetch is performed."
        )

    meta_path = npz_path.with_name(f"{npz_path.stem}_meta.csv")
    return npz_path, (meta_path if meta_path.is_file() else None)


def _load_neuron_ids(meta_path: Path | None, n: int) -> np.ndarray:
    """Read stable neuron identifiers from meta, aligned to matrix row order.

    Prefers the ``bodyid`` column ordered by ``idx``; falls back to ``idx``; finally
    falls back to a plain ``range(n)`` when no usable metadata is present.
    """
    if meta_path is None:
        return np.arange(n, dtype=np.int64)

    meta = pd.read_csv(meta_path)
    if "idx" in meta.columns:
        meta = meta.sort_values("idx")
    if "bodyid" in meta.columns:
        ids = meta["bodyid"].to_numpy()
    elif "idx" in meta.columns:
        ids = meta["idx"].to_numpy()
    else:
        return np.arange(n, dtype=np.int64)

    if len(ids) != n:
        raise ValueError(
            f"Metadata row count ({len(ids)}) does not match matrix dimension ({n}); "
            f"the '.npz' and '*_meta.csv' are out of sync."
        )
    return ids


def load_connectome(path: str | os.PathLike[str] | None = None) -> ConnectomeData:
    """Load a cached connectome from local disk (offline; no network, no neuPrint).

    Parameters
    ----------
    path:
        Optional directory (or direct ``.npz`` path) holding the cached artifact.
        Resolution order when ``None``: the ``DRONE_FLY_CONNECTOME_DIR`` environment
        variable, then the ``data/connectome`` default.

    Returns
    -------
    ConnectomeData
        The loaded adjacency matrix, neuron identifiers, and a provenance tag.

    Raises
    ------
    FileNotFoundError
        If the artifact directory or matrix file is absent. The error message is
        actionable and this function never silently falls back to a network fetch.
    """
    target = _resolve_artifact_dir(path)
    npz_path, meta_path = _find_matrix_and_meta(target)

    matrix = sp.load_npz(npz_path)
    adjacency = matrix.tocsr().astype(np.float32)
    n = adjacency.shape[0]
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"Connectome matrix must be square (N x N); got shape {adjacency.shape} "
            f"from '{npz_path.name}'."
        )

    neuron_ids = _load_neuron_ids(meta_path, n)
    return ConnectomeData(adjacency=adjacency, neuron_ids=neuron_ids, source=str(npz_path))
