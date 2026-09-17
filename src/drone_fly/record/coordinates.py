"""Neuron soma-coordinate provisioning + role labelling for the activation viewer (UC-05).

The spatial "brain map" panel draws each neuron as a dot at its **soma position**, so a
recording needs one 3-D position per neuron (positions depend only on the neuron set, so
they are provisioned **once** and stored in the recording — never per frame). This module
resolves those positions through a documented source order and, when no anatomy is
reachable, falls back to a deterministic computed layout that is *clearly labelled as not
anatomical* so the map is never mistaken for biology.

Source order (highest-fidelity first)
-------------------------------------
1. **Committed / local anatomical CSV** — tokenless, the CI/offline default:
   * ``DRONE_FLY_SOMA_CSV`` (if set) pointing at either a plain ``bodyid,x,y,z`` CSV or a
     ``connectome_data_prep`` meta CSV carrying a ``somaLocation`` column
     (``"[x y z]"`` in neuPrint MaleCNS 8 nm voxel space); or
   * the **fixture sidecar** ``<connectome-stem>_soma.csv`` sitting beside the connectome
     ``.npz`` (produced by ``scripts/fetch_soma_positions.py``). This is what makes the
     committed fixture demo anatomical offline.
2. **neuPrint** via ``neuprint-python`` + ``NEUPRINT_TOKEN`` — a dev-time, networked step
   for arbitrary slices whose bodyids are not in a committed sidecar. Never required by
   CI (guarded import; any failure degrades to the fallback).
3. **Deterministic spectral layout** — a seed-free spectral embedding of the (undirected,
   magnitude-weighted) connectome graph, sign-canonicalised so it is bit-reproducible.
   Labelled ``"computed (NOT anatomical …)"``.

Neurons that lack a real soma position are **flagged** (``has_position=False``) and
fallback-placed — never dropped and never fabricated with a made-up anatomical value.

Roles
-----
:func:`neuron_roles` groups neurons as ``sensory`` (``visual_projection``), ``motor``
(``descending_neuron``) or ``interneuron`` (everything else) — the same superclass
vocabulary UC-02/UC-04 use — so the viewer can colour dots and order heatmap rows by the
sensory→interneuron→motor structure rather than an arbitrary index order.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.populations import MOTOR_SUPERCLASS, SENSORY_SUPERCLASS

logger = logging.getLogger(__name__)

#: Role labels used by the viewer (dot colour + heatmap row ordering).
ROLE_SENSORY = "sensory"
ROLE_MOTOR = "motor"
ROLE_INTERNEURON = "interneuron"

#: Canonical role ordering (sensory → interneuron → motor) for heatmap grouping.
ROLE_ORDER = (ROLE_SENSORY, ROLE_INTERNEURON, ROLE_MOTOR)

#: Environment variable naming a local anatomical soma CSV (tokenless override).
SOMA_CSV_ENV = "DRONE_FLY_SOMA_CSV"

#: Supported top-down projection planes → the pair of 3-D axis indices they select.
#: ``xz`` (default) is the dorsal view (looking down the y / dorsal-ventral axis).
PROJECTIONS: dict[str, tuple[int, int]] = {"xz": (0, 2), "xy": (0, 1), "yz": (1, 2)}

#: Default projection plane (dorsal top-down).
DEFAULT_PROJECTION = "xz"

#: Label prefixes recorded in ``positions.source`` so the viewer/tests can tell the
#: provenance apart at a glance (anatomical vs. computed).
SOURCE_COMPUTED = "computed (NOT anatomical; spectral layout)"


def neuron_roles(data: ConnectomeData) -> list[str]:
    """Return one role label per neuron, aligned to the matrix row order.

    ``visual_projection`` → :data:`ROLE_SENSORY`, ``descending_neuron`` →
    :data:`ROLE_MOTOR`, everything else → :data:`ROLE_INTERNEURON`. When ``superclass``
    metadata is absent (older/reduced fixtures) every neuron is labelled an interneuron
    (documented degrade — never fabricated biology).
    """
    if data.superclass is None:
        return [ROLE_INTERNEURON] * data.neuron_count
    roles: list[str] = []
    for label in np.asarray(data.superclass).tolist():
        if label == SENSORY_SUPERCLASS:
            roles.append(ROLE_SENSORY)
        elif label == MOTOR_SUPERCLASS:
            roles.append(ROLE_MOTOR)
        else:
            roles.append(ROLE_INTERNEURON)
    return roles


def _parse_soma_location(value: object) -> tuple[float, float, float] | None:
    """Parse a ``connectome_data_prep`` ``somaLocation`` cell → ``(x, y, z)`` or ``None``.

    The cell is ``"[x y z]"`` (whitespace-separated ints in neuPrint voxel space). Empty
    cells, ``NaN``, or malformed triples yield ``None`` (the neuron then has no anatomical
    position — flagged, not fabricated).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().strip("[]")
    if not text:
        return None
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        return None
    try:
        x, y, z = (float(p) for p in parts)
    except ValueError:
        return None
    if not all(np.isfinite(v) for v in (x, y, z)):
        return None
    return (x, y, z)


def _mapping_from_frame(frame: pd.DataFrame) -> dict[int, tuple[float, float, float]]:
    """Build a ``bodyid -> (x, y, z)`` mapping from a soma CSV/meta dataframe.

    Accepts either explicit ``x``/``y``/``z`` columns or a ``somaLocation`` column; keyed
    on ``bodyid``. Rows without a usable position are skipped (not fabricated).
    """
    if "bodyid" not in frame.columns:
        return {}
    mapping: dict[int, tuple[float, float, float]] = {}
    has_xyz = {"x", "y", "z"}.issubset(frame.columns)
    for _, row in frame.iterrows():
        try:
            bid = int(row["bodyid"])
        except (TypeError, ValueError):
            continue
        if has_xyz:
            xyz = _parse_soma_location(f"{row['x']} {row['y']} {row['z']}")
        elif "somaLocation" in frame.columns:
            xyz = _parse_soma_location(row["somaLocation"])
        else:
            return {}
        if xyz is not None:
            mapping[bid] = xyz
    return mapping


def _load_soma_csv(path: Path) -> dict[int, tuple[float, float, float]]:
    """Read a local soma CSV (``bodyid,x,y,z`` or a meta with ``somaLocation``)."""
    frame = pd.read_csv(path, low_memory=False)
    return _mapping_from_frame(frame)


def _sidecar_path(data: ConnectomeData) -> Path | None:
    """Derive the ``<connectome-stem>_soma.csv`` sidecar path from ``data.source``."""
    base = str(getattr(data, "source", "") or "").split(" [", 1)[0]
    if not base:
        return None
    p = Path(base)
    if p.suffix != ".npz":
        return None
    return p.with_name(f"{p.stem}_soma.csv")


def _load_anatomical(
    data: ConnectomeData,
) -> tuple[dict[int, tuple[float, float, float]], str] | None:
    """Resolve anatomical soma positions via the documented source order.

    Returns ``(mapping, source_label)`` for the first source that yields at least one
    position, else ``None`` (caller then uses the computed fallback).
    """
    env_csv = os.environ.get(SOMA_CSV_ENV)
    if env_csv:
        path = Path(env_csv)
        if path.is_file():
            mapping = _load_soma_csv(path)
            if mapping:
                return mapping, f"anatomical (local CSV: {path.name})"
            logger.warning("%s=%s held no usable soma positions.", SOMA_CSV_ENV, env_csv)
        else:
            logger.warning("%s=%s does not exist; skipping.", SOMA_CSV_ENV, env_csv)

    sidecar = _sidecar_path(data)
    if sidecar is not None and sidecar.is_file():
        mapping = _load_soma_csv(sidecar)
        if mapping:
            return mapping, f"anatomical (fixture sidecar: {sidecar.name})"

    neuprint = _load_from_neuprint(np.asarray(data.neuron_ids))
    if neuprint:
        return neuprint, "anatomical (neuPrint)"

    return None


def _load_from_neuprint(ids: np.ndarray) -> dict[int, tuple[float, float, float]] | None:
    """Best-effort dev-time neuPrint soma fetch (guarded; ``None`` on any failure).

    Requires ``NEUPRINT_TOKEN`` (and a reachable server); never invoked by CI. Any import,
    auth, or network failure degrades silently to ``None`` so the caller falls back to the
    computed layout rather than erroring.
    """
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        return None
    try:  # pragma: no cover - dev-time networked path, not exercised in CI
        from neuprint import Client, NeuronCriteria, fetch_neurons

        server = os.environ.get("NEUPRINT_SERVER", "https://neuprint.janelia.org")
        dataset = os.environ.get("NEUPRINT_DATASET", "male-cns")
        client = Client(server, dataset=dataset, token=token)
        body_ids = [int(x) for x in ids.tolist()]
        neurons, _ = fetch_neurons(NeuronCriteria(bodyId=body_ids, client=client), client=client)
        mapping: dict[int, tuple[float, float, float]] = {}
        for _, row in neurons.iterrows():
            loc = row.get("somaLocation")
            xyz = None
            if isinstance(loc, dict):
                coords = loc.get("coordinates") or loc.get("value")
                if coords is not None and len(coords) == 3:
                    xyz = _parse_soma_location(" ".join(str(c) for c in coords))
            elif isinstance(loc, (list, tuple)) and len(loc) == 3:
                xyz = _parse_soma_location(" ".join(str(c) for c in loc))
            else:
                xyz = _parse_soma_location(loc)
            if xyz is not None:
                mapping[int(row["bodyId"])] = xyz
        return mapping or None
    except Exception as exc:  # pragma: no cover - dev-time only
        logger.warning("neuPrint soma fetch failed (%s); using computed fallback.", exc)
        return None


def _canonical_sign(vector: np.ndarray) -> np.ndarray:
    """Flip an eigenvector to a canonical sign so the layout is reproducible.

    Eigenvectors are defined only up to sign; LAPACK may return either polarity. We fix
    the sign so the entry of largest magnitude is non-negative (ties break on the lowest
    index via ``argmax``), making the spectral layout deterministic across runs/machines.
    """
    if vector.size == 0:
        return vector
    pivot = int(np.argmax(np.abs(vector)))
    return -vector if vector[pivot] < 0 else vector


def _spectral_embedding(data: ConnectomeData) -> np.ndarray:
    """Return a deterministic ``(N, 3)`` spectral layout of the connectome graph.

    Builds the undirected, magnitude-weighted affinity ``|A| + |A|ᵀ``, forms its graph
    Laplacian, and takes eigenvectors 1–3 (skipping the trivial constant eigenvector 0).
    Each vector is sign-canonicalised (:func:`_canonical_sign`) so the embedding is
    bit-reproducible. Dense ``eigh`` is used (deterministic LAPACK); for a few-thousand-node
    pruned slice this is a one-time seconds-scale step.
    """
    n = data.neuron_count
    affinity = abs(data.adjacency.tocsr().astype(np.float64))
    affinity = affinity + affinity.T
    degree = np.asarray(affinity.sum(axis=1)).ravel()
    laplacian = (sp.diags(degree) - affinity).toarray()
    if n > 6000:
        logger.warning(
            "Spectral fallback layout on %d neurons uses a dense eigendecomposition; this "
            "may be slow. Provision anatomical soma coordinates to avoid it.",
            n,
        )
    _, vectors = np.linalg.eigh(laplacian)  # ascending eigenvalues
    embedding = np.zeros((n, 3), dtype=np.float64)
    for out_col, src_col in enumerate((1, 2, 3)):
        if src_col < n:
            embedding[:, out_col] = _canonical_sign(vectors[:, src_col])
    return embedding


def _project(coords3d: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    """Select the two projection axes from a ``(N, 3)`` array → ``(N, 2)``."""
    return coords3d[:, list(axes)]


def _scale_into_box(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Min-max map ``source`` points into the per-axis bounding box of ``reference``.

    Used to place fallback (non-anatomical) neurons *within* the anatomical map's extent so
    they are visible near the real dots, without inventing an anatomical coordinate — they
    stay flagged ``has_position=False``.
    """
    out = np.array(source, dtype=np.float64, copy=True)
    for axis in range(source.shape[1]):
        s = source[:, axis]
        s_lo, s_hi = float(s.min()), float(s.max())
        r = reference[:, axis]
        r_lo, r_hi = float(r.min()), float(r.max())
        if s_hi - s_lo < 1e-12:
            out[:, axis] = (r_lo + r_hi) / 2.0
        else:
            out[:, axis] = r_lo + (s - s_lo) / (s_hi - s_lo) * (r_hi - r_lo)
    return out


def provision_positions(data: ConnectomeData, *, projection: str = DEFAULT_PROJECTION) -> dict:
    """Provision soma positions + a top-down projection for every neuron (once).

    Returns a JSON-ready dict matching the recording schema's ``positions`` block::

        {
          "source": <provenance label>,          # "anatomical (…)" | computed fallback
          "projection": <plane, e.g. "xz">,
          "coords3d": [[x, y, z] | null, ...],    # null only for fallback-placed neurons
          "coords2d": [[u, v], ...],              # ALWAYS finite (real or fallback)
          "has_position": [bool, ...],            # True iff a real anatomical soma
        }

    Anatomical positions (source 1/2) are used where available; neurons missing a soma are
    flagged ``has_position=False`` and fallback-placed inside the anatomical extent (their
    ``coords3d`` is ``null`` — no fabricated 3-D anatomy). When no anatomy is reachable at
    all, every neuron gets the deterministic spectral layout, labelled as *computed*.
    """
    if projection not in PROJECTIONS:
        raise ValueError(
            f"Unknown projection {projection!r}; supported: {sorted(PROJECTIONS)} "
            f"(default {DEFAULT_PROJECTION!r})."
        )
    axes = PROJECTIONS[projection]
    n = data.neuron_count
    ids = np.asarray(data.neuron_ids).tolist()

    anatomical = _load_anatomical(data)
    if anatomical is not None:
        mapping, source = anatomical
        coords3d: list[list[float] | None] = []
        has_position: list[bool] = []
        for bid in ids:
            xyz = mapping.get(int(bid)) if _is_int_like(bid) else None
            if xyz is not None:
                coords3d.append([float(v) for v in xyz])
                has_position.append(True)
            else:
                coords3d.append(None)
                has_position.append(False)

        real_rows = np.array([c for c in coords3d if c is not None], dtype=np.float64)
        real_2d = _project(real_rows, axes)
        coords2d = np.zeros((n, 2), dtype=np.float64)
        real_iter = iter(real_2d)
        missing_idx = [i for i, ok in enumerate(has_position) if not ok]
        for i, ok in enumerate(has_position):
            if ok:
                coords2d[i] = next(real_iter)
        if missing_idx:
            spec2 = _project(_spectral_embedding(data), axes)
            placed = _scale_into_box(spec2[missing_idx], real_2d)
            for slot, i in enumerate(missing_idx):
                coords2d[i] = placed[slot]
            logger.info(
                "Provisioned anatomical soma positions for %d/%d neurons (%d fallback-placed).",
                n - len(missing_idx),
                n,
                len(missing_idx),
            )
        else:
            logger.info("Provisioned anatomical soma positions for %d/%d neurons.", n, n)
        return {
            "source": source,
            "projection": projection,
            "coords3d": coords3d,
            "coords2d": coords2d.tolist(),
            "has_position": has_position,
        }

    embedding = _spectral_embedding(data)
    coords2d = _project(embedding, axes)
    logger.info(
        "No anatomical soma positions available; using the deterministic computed "
        "spectral layout for all %d neurons (labelled non-anatomical).",
        n,
    )
    return {
        "source": SOURCE_COMPUTED,
        "projection": projection,
        "coords3d": embedding.tolist(),
        "coords2d": coords2d.tolist(),
        "has_position": [False] * n,
    }


def _is_int_like(value: object) -> bool:
    """True if ``value`` can be interpreted as an integer bodyid."""
    try:
        int(value)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False
