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

Provisioning at slice time (UC-27)
----------------------------------
Positions are provisioned **once, when a connectome artifact is created** — the ``prune``
slice, the ``fetch-connectome`` base download, and the ``prune-trained`` subcircuit — and
cached as a ``<connectome-stem>_positions.csv`` sidecar beside the ``.npz`` (and, when
neuPrint anatomy is fetched, a ``<stem>_soma.csv`` sidecar). Recording-enabled training then
only *loads* the sidecar: no per-run neuPrint fetch, no per-run spectral eigendecomposition.
:func:`resolve_positions` is the single entry point for both sides.

* **Sidecar format.** One row per neuron, columns
  ``bodyid,has_position,x,y,z,u,v,source,projection`` (a verbatim serialisation of the
  :func:`provision_positions` dict). ``x/y/z`` are blank iff that neuron has no 3-D anatomy
  (``coords3d[i] is None``); ``has_position`` is an *independent* column. Floats use ``%.17g``
  so a load reproduces the compute bit-exactly.
* **Node-set binding.** A sidecar is only reused when its ``bodyid`` set equals the loaded
  connectome's neuron-id set exactly *and* its ``projection`` matches; otherwise it is a MISS
  and positions are recomputed. A pruned subcircuit can never pick up the full connectome's
  positions, and a stale/corrupt sidecar is never applied to the wrong neurons.
* **Self-heal.** A recording run against an older artifact with no sidecar computes the layout
  once, WARNs, persists the sidecar (best-effort — a read-only dir does not crash the run),
  and every subsequent run takes the fast load path.
* **Large-connectome guard.** A dense ``eigh`` at full-MaleCNS scale (~161k neurons) is
  infeasible, so when a compute would need the spectral fallback for more than
  :func:`_spectral_cap` neurons (:data:`SPECTRAL_MAX_NEURONS`, overridable via
  :data:`SPECTRAL_MAX_ENV` = ``DRONE_FLY_SPECTRAL_MAX``) *and* anatomy is absent/partial,
  provisioning is deferred (WARN, no ``_positions.csv``): provide anatomy (``NEUPRINT_TOKEN``
  or ``DRONE_FLY_SOMA_CSV``) or prune the connectome before recording. Complete real anatomy
  at any scale still provisions (no spectral needed).

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
from dataclasses import replace
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

#: Default upper bound (neuron count) above which the dense-``eigh`` spectral fallback is
#: refused: an ``N x N`` dense Laplacian at full-MaleCNS scale (~161k) is ~200 GB and must
#: never be built. Overridable via :data:`SPECTRAL_MAX_ENV`. See :func:`_spectral_cap`.
SPECTRAL_MAX_NEURONS = 50000

#: Environment variable overriding :data:`SPECTRAL_MAX_NEURONS` (positive int).
SPECTRAL_MAX_ENV = "DRONE_FLY_SPECTRAL_MAX"

#: Float format used when serialising positions to the sidecar. ``%.17g`` round-trips an
#: IEEE-754 ``float64`` bit-exactly, so a load reproduces the compute exactly (UC-27 AC-8).
_FLOAT_FMT = "%.17g"

#: Column order of the ``<stem>_positions.csv`` sidecar (a verbatim serialisation of the
#: :func:`provision_positions` dict). ``x/y/z`` are blank iff ``coords3d[i] is None``;
#: ``has_position`` is an INDEPENDENT column (the no-anatomy spectral branch stores
#: ``has_position=False`` with non-null ``coords3d``), so null-ness is serialised directly.
POSITIONS_SIDECAR_COLUMNS = (
    "bodyid",
    "has_position",
    "x",
    "y",
    "z",
    "u",
    "v",
    "source",
    "projection",
)


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


def _base_npz_path(
    data: ConnectomeData, artifact_npz: str | os.PathLike[str] | None = None
) -> Path | None:
    """Resolve the connectome ``.npz`` a sidecar keys off.

    On the **write** side (slice/fetch/prune hooks) ``artifact_npz`` is the just-saved
    artifact — used verbatim, because a pruned ``ConnectomeData.source`` is the slice-source
    string (``"… [pruned:…]"``), not the new npz. On the **read** side (recorder) it is
    ``None`` and the base is derived from ``data.source`` (the loaded artifact) — the
    historical behaviour, unchanged. Returns ``None`` when no ``.npz`` base can be resolved.
    """
    if artifact_npz is not None:
        p = Path(artifact_npz)
        return p if p.suffix == ".npz" else None
    base = str(getattr(data, "source", "") or "").split(" [", 1)[0]
    if not base:
        return None
    p = Path(base)
    return p if p.suffix == ".npz" else None


def _sidecar_path(
    data: ConnectomeData, artifact_npz: str | os.PathLike[str] | None = None
) -> Path | None:
    """Derive the ``<connectome-stem>_soma.csv`` sidecar path (see :func:`_base_npz_path`)."""
    p = _base_npz_path(data, artifact_npz)
    return None if p is None else p.with_name(f"{p.stem}_soma.csv")


def _positions_sidecar_path(
    data: ConnectomeData, artifact_npz: str | os.PathLike[str] | None = None
) -> Path | None:
    """Derive the ``<connectome-stem>_positions.csv`` sidecar path (see :func:`_base_npz_path`)."""
    p = _base_npz_path(data, artifact_npz)
    return None if p is None else p.with_name(f"{p.stem}_positions.csv")


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


def _canonical_id(value: object) -> int | str:
    """Canonicalise a bodyid for set-equality + lookup: ``int`` when int-coercible.

    Both the sidecar's ``bodyid`` column and the connectome's ``neuron_ids`` are run through
    this before the node-set comparison, so ``"720575"`` (read back from CSV) and ``720575``
    (an ``int64`` neuron id) compare equal — avoiding a false MISS. Non-int-coercible ids
    (some fixtures carry string/range ids) fall back to a stripped-string form on both sides.
    """
    return int(value) if _is_int_like(value) else str(value).strip()  # type: ignore[arg-type]


def _spectral_cap() -> int:
    """Return the effective spectral-fallback neuron cap (env override → default).

    Reads :data:`SPECTRAL_MAX_ENV`; a positive integer overrides :data:`SPECTRAL_MAX_NEURONS`.
    An unset/blank/invalid value keeps the default (a warning is logged for a malformed one).
    """
    raw = os.environ.get(SPECTRAL_MAX_ENV)
    if not raw:
        return SPECTRAL_MAX_NEURONS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r (not an integer); using default cap %d.",
            SPECTRAL_MAX_ENV,
            raw,
            SPECTRAL_MAX_NEURONS,
        )
        return SPECTRAL_MAX_NEURONS
    if value <= 0:
        logger.warning(
            "Ignoring non-positive %s=%r; using default cap %d.",
            SPECTRAL_MAX_ENV,
            raw,
            SPECTRAL_MAX_NEURONS,
        )
        return SPECTRAL_MAX_NEURONS
    return value


def _write_positions_sidecar(
    pos_path: Path,
    data: ConnectomeData,
    positions: dict,
    *,
    strict: bool = True,
) -> None:
    """Serialise a :func:`provision_positions` dict to ``<stem>_positions.csv``.

    One row per neuron, columns :data:`POSITIONS_SIDECAR_COLUMNS`. ``coords3d`` null-ness is
    serialised **directly**: ``x/y/z`` cells are left blank iff ``coords3d[i] is None`` (they
    read back as ``NaN``); otherwise the three values are written at :data:`_FLOAT_FMT`
    (bit-exact ``float64`` — AC-8). ``u/v`` (``coords2d``) are always written. ``has_position``
    is its own column, independent of ``coords3d`` null-ness.

    When ``strict`` is false any :class:`OSError` (e.g. a read-only dir on the recorder
    self-heal path) is logged and swallowed rather than raised.
    """
    ids = np.asarray(data.neuron_ids).tolist()
    coords3d = positions["coords3d"]
    coords2d = positions["coords2d"]
    has_position = positions["has_position"]
    source = positions["source"]
    projection = positions["projection"]

    rows: list[dict[str, object]] = []
    for i, bid in enumerate(ids):
        c3 = coords3d[i]
        if c3 is None:
            x = y = z = ""
        else:
            x = _FLOAT_FMT % float(c3[0])
            y = _FLOAT_FMT % float(c3[1])
            z = _FLOAT_FMT % float(c3[2])
        u, v = coords2d[i]
        rows.append(
            {
                "bodyid": bid,
                "has_position": bool(has_position[i]),
                "x": x,
                "y": y,
                "z": z,
                "u": _FLOAT_FMT % float(u),
                "v": _FLOAT_FMT % float(v),
                "source": source,
                "projection": projection,
            }
        )

    frame = pd.DataFrame(rows, columns=list(POSITIONS_SIDECAR_COLUMNS))
    try:
        frame.to_csv(pos_path, index=False)
    except OSError as exc:
        if strict:
            raise
        logger.warning(
            "Could not persist positions sidecar %s (%s); positions were provisioned in "
            "memory but not cached — a later run will recompute.",
            pos_path,
            exc,
        )
        return
    logger.info("Persisted %d neuron positions to %s.", len(rows), pos_path)


def _load_positions_sidecar(
    pos_path: Path | None, data: ConnectomeData, projection: str
) -> dict | None:
    """Load + validate a positions sidecar; return the positions dict, or ``None`` on a MISS.

    A HIT requires ALL of: the file is readable, carries every column in
    :data:`POSITIONS_SIDECAR_COLUMNS`, has exactly ``data.neuron_count`` rows, its
    ``projection`` equals ``projection``, and its ``bodyid`` set equals the connectome's
    ``neuron_ids`` set exactly (both canonicalised via :func:`_canonical_id`). On a HIT the
    rows are realigned to the connectome's row order by bodyid and the dict is reconstructed
    (blank ``x/y/z`` → ``coords3d[i] = None``). Anything else (unreadable, missing columns,
    wrong length, id-set mismatch — a stale/wrong-subcircuit artifact —, projection mismatch)
    is a MISS → caller recomputes. Never applies partial or wrong-neuron positions (AC-7).
    """
    if pos_path is None or not pos_path.is_file():
        return None
    try:
        frame = pd.read_csv(pos_path, low_memory=False, float_precision="round_trip")
    except Exception as exc:  # noqa: BLE001 - any parse error is a safe MISS → recompute
        logger.warning("Positions sidecar %s is unreadable (%s); recomputing.", pos_path, exc)
        return None

    if not set(POSITIONS_SIDECAR_COLUMNS).issubset(frame.columns):
        logger.warning("Positions sidecar %s is missing required columns; recomputing.", pos_path)
        return None
    if len(frame) != data.neuron_count:
        logger.warning(
            "Positions sidecar %s has %d rows but the connectome has %d neurons; recomputing.",
            pos_path,
            len(frame),
            data.neuron_count,
        )
        return None

    proj_values = {str(p) for p in frame["projection"].tolist()}
    if proj_values != {projection}:
        logger.warning(
            "Positions sidecar %s was written for projection(s) %s, not %r; recomputing.",
            pos_path,
            sorted(proj_values),
            projection,
        )
        return None

    side_ids = [_canonical_id(b) for b in frame["bodyid"].tolist()]
    conn_ids = [_canonical_id(b) for b in np.asarray(data.neuron_ids).tolist()]
    if set(side_ids) != set(conn_ids):
        logger.warning(
            "Positions sidecar %s does not match this connectome's neuron-id set "
            "(stale artifact or wrong subcircuit); recomputing.",
            pos_path,
        )
        return None

    row_by_id = {cid: ridx for ridx, cid in enumerate(side_ids)}
    order = [row_by_id[cid] for cid in conn_ids]
    frame = frame.iloc[order].reset_index(drop=True)

    coords3d: list[list[float] | None] = []
    coords2d: list[list[float]] = []
    has_position: list[bool] = []
    for i in range(len(frame)):
        x = frame["x"].iloc[i]
        y = frame["y"].iloc[i]
        z = frame["z"].iloc[i]
        if pd.isna(x) and pd.isna(y) and pd.isna(z):
            coords3d.append(None)
        else:
            coords3d.append([float(x), float(y), float(z)])
        coords2d.append([float(frame["u"].iloc[i]), float(frame["v"].iloc[i])])
        has_position.append(bool(frame["has_position"].iloc[i]))

    return {
        "source": str(frame["source"].iloc[0]),
        "projection": projection,
        "coords3d": coords3d,
        "coords2d": coords2d,
        "has_position": has_position,
    }


def _maybe_persist_neuprint_soma(
    data: ConnectomeData, soma_path: Path | None, *, strict: bool
) -> None:
    """Fetch real anatomy from neuPrint once and persist it as ``<stem>_soma.csv`` (AC-6).

    A no-op unless ``NEUPRINT_TOKEN`` is set, ``DRONE_FLY_SOMA_CSV`` is NOT set (that tier-1
    override wins on its own), and the soma sidecar does not already exist. On success the
    ``bodyid,x,y,z`` sidecar is written beside the artifact so later loads reuse it via the
    existing tier-2 path with no further network calls — and so the subsequent compute finds
    it (neuPrint preferred over the spectral fallback). Dev-time only; never hit by CI.
    """
    if soma_path is None or soma_path.is_file():
        return
    if os.environ.get(SOMA_CSV_ENV):
        return
    if not os.environ.get("NEUPRINT_TOKEN"):
        return
    mapping = _load_from_neuprint(np.asarray(data.neuron_ids))
    if not mapping:  # pragma: no cover - dev-time networked path, not exercised in CI
        return
    frame = pd.DataFrame(  # pragma: no cover - dev-time networked path, not exercised in CI
        [{"bodyid": bid, "x": xyz[0], "y": xyz[1], "z": xyz[2]} for bid, xyz in mapping.items()]
    )
    try:  # pragma: no cover - dev-time networked path, not exercised in CI
        frame.to_csv(soma_path, index=False)
    except OSError as exc:
        if strict:
            raise
        logger.warning("Could not persist soma sidecar %s (%s); continuing.", soma_path, exc)
        return
    logger.info(  # pragma: no cover - dev-time networked path, not exercised in CI
        "Persisted %d neuPrint soma positions to %s.", len(mapping), soma_path
    )


def _anatomy_coverage(data: ConnectomeData) -> int:
    """Count neurons for which :func:`_load_anatomical` yields a real soma position.

    Used only by the large-connectome guard to decide whether a compute would need the
    spectral fallback. Returns 0 when no anatomy is reachable at all.
    """
    anatomical = _load_anatomical(data)
    if anatomical is None:
        return 0
    mapping, _ = anatomical
    ids = np.asarray(data.neuron_ids).tolist()
    return sum(1 for bid in ids if _is_int_like(bid) and int(bid) in mapping)


def resolve_positions(
    data: ConnectomeData,
    *,
    projection: str = DEFAULT_PROJECTION,
    artifact_npz: str | os.PathLike[str] | None = None,
    persist: bool = True,
    persist_strict: bool = True,
) -> dict | None:
    """Resolve neuron positions via the cached sidecar, else compute-once-and-persist.

    This is the single orchestration point for UC-27: provisioning is done at slice/artifact
    time (``artifact_npz`` set, write hooks) and training only ever *loads* the sidecar. The
    recorder calls it on the read side (``artifact_npz=None``, keyed off ``data.source``) and
    self-heals older artifacts that predate the sidecar.

    Flow:

    1. **Fast path** — a valid ``<stem>_positions.csv`` (see :func:`_load_positions_sidecar`)
       returns immediately with **no** spectral eigendecomposition and **no** neuPrint fetch
       (AC-4).
    2. **neuPrint persist** — real anatomy is fetched once and cached as ``<stem>_soma.csv``
       (AC-6), so the compute below finds it via tier-2 (neuPrint preferred over spectral).
    3. **Large-connectome guard** — when a compute would need the dense spectral fallback for
       ``n > _spectral_cap()`` neurons AND anatomy is absent/partial, do NOT build the giant
       Laplacian: WARN, leave ``_positions.csv`` unwritten, and return ``None`` (defer). A
       full connectome with COMPLETE real anatomy still provisions (no spectral needed).
    4. **Compute** via :func:`provision_positions` (output-identical to today — AC-8).
    5. **Persist** the result to ``<stem>_positions.csv`` when ``persist`` (best-effort when
       ``persist_strict`` is false — the recorder's read-only-dir case).

    Parameters
    ----------
    artifact_npz:
        The just-saved connectome ``.npz`` (write hooks). Sidecars are keyed off it, never off
        the pruned ``data.source``. ``None`` on the read side (recorder).
    persist / persist_strict:
        Whether to write the positions sidecar, and whether a write ``OSError`` is fatal.

    Returns
    -------
    dict | None
        The positions dict (schema identical to :func:`provision_positions`), or ``None`` when
        the large-connectome guard deferred provisioning.
    """
    pos_path = _positions_sidecar_path(data, artifact_npz)
    soma_path = _sidecar_path(data, artifact_npz)

    # 1. Fast path — a valid cached sidecar wins with zero fetch/compute.
    cached = _load_positions_sidecar(pos_path, data, projection)
    if cached is not None:
        logger.info("Loaded cached neuron positions from %s (no compute).", pos_path)
        return cached

    # For the compute below, make provision_positions resolve the soma sidecar beside the
    # ARTIFACT (write side) rather than beside the pruned data.source. On the read side
    # (artifact_npz is None) compute_data is data unchanged.
    compute_data = data if artifact_npz is None else replace(data, source=str(Path(artifact_npz)))

    # 2. Persist real anatomy once (dev-time neuPrint), preferred over spectral.
    _maybe_persist_neuprint_soma(compute_data, soma_path, strict=persist_strict)

    # 3. Large-connectome refuse guard — never build a full-MaleCNS-scale dense eigh.
    n = data.neuron_count
    if n > _spectral_cap() and _anatomy_coverage(compute_data) < n:
        logger.warning(
            "No/partial anatomy for a %d-neuron connectome (> spectral cap %d): positions "
            "were NOT provisioned (a dense spectral layout at this scale is infeasible). Set "
            "%s or %s, or prune the connectome before recording.",
            n,
            _spectral_cap(),
            "NEUPRINT_TOKEN",
            SOMA_CSV_ENV,
        )
        return None

    # 4. Compute (output-identical to the historical path). On the read side a miss means the
    # artifact predates the sidecar — WARN that we are provisioning on the fly (AC-5 self-heal).
    if artifact_npz is None:
        logger.warning(
            "No positions sidecar for this connectome (%s); provisioning positions on the fly "
            "and caching them beside the artifact so later runs load instead of recompute.",
            pos_path,
        )
    result = provision_positions(compute_data, projection=projection)

    # 5. Persist for next time.
    if persist and pos_path is not None:
        _write_positions_sidecar(pos_path, data, result, strict=persist_strict)
    return result
