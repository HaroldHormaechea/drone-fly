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
# scripts/build_test_fixture.py — the UC-13 modality-aware slice: a soma-populated
# top-degree central core UNION a top-degree mechanosensory_proprioceptive afferent
# quota (see that script + FIXTURE_PROVENANCE.md) — so the counts below are exact and
# reproducible. Tests assert the loaded fixture matches these numbers exactly (AC1).
FIXTURE_EXPECTED_SCALE = ExpectedScale(neuron_count=300, edge_count=8288)

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
    superclass:
        Optional length-``N`` array of per-neuron MaleCNS superclass labels (e.g.
        ``"descending_neuron"``, ``"visual_projection"``), aligned to the matrix row
        order. ``None`` when the metadata lacks a ``superclass`` column (older/reduced
        fixtures) — downstream population selection then degrades to a documented
        placeholder rather than fabricating biology. Consumed by UC-02's
        :mod:`drone_fly.controller.populations`.
    sign:
        Optional length-``N`` ``int8`` array of per-neuron excitatory/inhibitory sign
        (``+1`` excitatory, ``-1`` inhibitory), derived from MaleCNS neurotransmitter
        predictions and aligned to the matrix row order. ``None`` when the metadata
        lacks a ``sign`` column, in which case the E/I mask degrades to a documented
        default. The stored ``.npz`` weights are magnitudes only (all positive); this
        column is the sole source of E/I biology.
    top_nt:
        Optional length-``N`` array of each neuron's predicted majority
        neurotransmitter (``"gaba"``, ``"acetylcholine"``, ...), aligned to the matrix
        row order, or ``None`` when absent. Retained for provenance/traceability; the
        E/I contract is carried by :pyattr:`sign`.
    neuron_class:
        Optional length-``N`` array of per-neuron MaleCNS ``class`` labels (e.g.
        ``"mechanosensory_proprioceptive"``, ``"olfactory"``, ``"gustatory"``), aligned to
        the matrix row order. Finer-grained than :pyattr:`superclass`, this is what lets
        population selection resolve *sensory modalities* rather than only the coarse
        superclass buckets (UC-13, consumed by :mod:`drone_fly.controller.modality`). The
        attribute is named ``neuron_class`` (not ``class``) because ``class`` is a Python
        reserved word; it is read from / written back to the ``class`` CSV column. ``None``
        when the metadata lacks a ``class`` column (older/reduced fixtures) — documented
        degrade, never fabricated.
    subclass:
        Optional length-``N`` array of per-neuron MaleCNS ``subclass`` labels (e.g.
        ``"campaniform sensilla"``, ``"haltere"``), aligned to the matrix row order, or
        ``None`` when the metadata lacks a ``subclass`` column. Finer still than
        :pyattr:`neuron_class`; the approximate (curated-name-list) modalities read it.
    """

    adjacency: sp.csr_matrix
    neuron_ids: np.ndarray
    source: str
    superclass: np.ndarray | None = None
    sign: np.ndarray | None = None
    top_nt: np.ndarray | None = None
    neuron_class: np.ndarray | None = None
    subclass: np.ndarray | None = None

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


#: Optional per-neuron metadata columns surfaced onto :class:`ConnectomeData`, mapping each
#: ``*_meta.csv`` **column name** to the ``ConnectomeData`` **attribute name** it populates.
#: The two differ only for ``class`` -> ``neuron_class`` (``class`` is a Python reserved
#: word). Absent columns leave the attribute ``None`` (documented degrade — never
#: fabricated). :func:`save_connectome` inverts this map to write attributes back under
#: their original CSV column names, so a load/save round-trips column-for-column.
_OPTIONAL_META_COLUMNS: dict[str, str] = {
    "superclass": "superclass",
    "class": "neuron_class",
    "subclass": "subclass",
    "sign": "sign",
    "top_nt": "top_nt",
}


def _load_neuron_attrs(meta_path: Path | None, n: int) -> dict[str, np.ndarray | None]:
    """Read optional per-neuron attribute columns, aligned to matrix row order.

    Returns a mapping of ``ConnectomeData`` attribute name -> aligned array (or ``None``
    when the column is absent). Alignment matches :func:`_load_neuron_ids`: rows are
    sorted by ``idx`` when that column exists. ``sign`` is normalised to ``int8`` so the
    downstream E/I mask is exactly ``+1`` / ``-1``. A column whose length disagrees with
    the matrix dimension is treated as absent (``None``) rather than silently misaligned.
    """
    attrs: dict[str, np.ndarray | None] = dict.fromkeys(_OPTIONAL_META_COLUMNS.values(), None)
    if meta_path is None:
        return attrs

    meta = pd.read_csv(meta_path)
    if "idx" in meta.columns:
        meta = meta.sort_values("idx")
    for column, attr in _OPTIONAL_META_COLUMNS.items():
        if column not in meta.columns:
            continue
        values = meta[column].to_numpy()
        if len(values) != n:
            continue
        attrs[attr] = values.astype(np.int8) if column == "sign" else values
    return attrs


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
    attrs = _load_neuron_attrs(meta_path, n)
    return ConnectomeData(
        adjacency=adjacency,
        neuron_ids=neuron_ids,
        source=str(npz_path),
        superclass=attrs["superclass"],
        sign=attrs["sign"],
        top_nt=attrs["top_nt"],
        neuron_class=attrs["neuron_class"],
        subclass=attrs["subclass"],
    )


#: Default matrix stem used by :func:`save_connectome`. The meta sidecar is written as
#: ``<stem>_meta.csv`` so :func:`load_connectome` (which infers the sidecar from the npz
#: stem) round-trips the directory.
DEFAULT_SAVE_STEM = "connectome_pruned"


def save_connectome(
    data: ConnectomeData,
    out_dir: str | os.PathLike[str],
    *,
    stem: str = DEFAULT_SAVE_STEM,
) -> tuple[Path, Path]:
    """Write ``data`` to ``out_dir`` in the on-disk format :func:`load_connectome` reads.

    Produces ``<out_dir>/<stem>.npz`` (the CSR ``float32`` adjacency) plus the sidecar
    ``<out_dir>/<stem>_meta.csv`` carrying ``idx`` (contiguous ``0..N-1``), ``bodyid``
    (from :pyattr:`ConnectomeData.neuron_ids`), and whichever of ``superclass`` / ``class``
    / ``subclass`` / ``top_nt`` / ``sign`` are present — the same columns the loader consumes
    (``neuron_class`` is written back under its original ``class`` column name). The pair
    round-trips:
    ``load_connectome(out_dir)`` reproduces an equivalent :class:`ConnectomeData` (same
    neuron/edge counts, meta aligned). Deterministic — identical input yields byte-identical
    files. Returns the ``(npz_path, meta_path)`` written.

    Raises
    ------
    NotADirectoryError
        If ``out_dir`` exists but is a regular file (not a directory).
    """
    out = Path(out_dir)
    if out.exists() and not out.is_dir():
        raise NotADirectoryError(
            f"Cannot write connectome: output path '{out}' exists and is not a directory."
        )
    out.mkdir(parents=True, exist_ok=True)

    npz_path = out / f"{stem}.npz"
    meta_path = out / f"{stem}_meta.csv"

    sp.save_npz(npz_path, data.adjacency.tocsr().astype(np.float32))

    n = data.neuron_count
    # Column order mirrors scripts/build_test_fixture.py (minus type/cell_type, which
    # ConnectomeData does not carry). 'idx' is contiguous so the loader's idx-sort is a
    # no-op and row order is preserved exactly.
    columns: dict[str, np.ndarray] = {
        "idx": np.arange(n, dtype=np.int64),
        "bodyid": np.asarray(data.neuron_ids),
    }
    if data.superclass is not None:
        columns["superclass"] = np.asarray(data.superclass)
    if data.neuron_class is not None:
        columns["class"] = np.asarray(data.neuron_class)
    if data.subclass is not None:
        columns["subclass"] = np.asarray(data.subclass)
    if data.top_nt is not None:
        columns["top_nt"] = np.asarray(data.top_nt)
    if data.sign is not None:
        columns["sign"] = np.asarray(data.sign)
    pd.DataFrame(columns).to_csv(meta_path, index=False)

    return npz_path, meta_path
