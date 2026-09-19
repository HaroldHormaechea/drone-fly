"""Neuron soma-coordinate provisioning + role labelling for the activation viewer (UC-05).

The spatial "brain map" panel draws each neuron as a dot at its **soma position**, so a
recording needs one 3-D position per neuron (positions depend only on the neuron set, so
they are provisioned **once** and stored in the recording — never per frame). This module
resolves those positions through a documented source order and, when no anatomy is
reachable, falls back to a deterministic computed layout that is *clearly labelled as not
anatomical* so the map is never mistaken for biology.

Source order (highest-fidelity first)
-------------------------------------
0. **Connectome meta ``somaLocation`` (tier-0, primary, tokenless)** — the MaleCNS download
   already ships ``mcns_all_neuron_meta.csv`` (saved as ``<npz-stem>_meta.csv``) whose
   ``somaLocation`` column holds real neuPrint MaleCNS 8 nm-voxel soma coordinates (CC-BY).
   Provisioning reads it for the artifact's bodyids **before any network call** — real anatomy,
   offline, by default (UC-27 AC-10). See :func:`_resolve_meta_soma`.
1. **Committed / local anatomical CSV** — tokenless:
   * ``DRONE_FLY_SOMA_CSV`` (if set) pointing at either a plain ``bodyid,x,y,z`` CSV or a
     ``connectome_data_prep`` meta CSV carrying a ``somaLocation`` column
     (``"[x y z]"`` in neuPrint MaleCNS 8 nm voxel space); or
   * the **fixture sidecar** ``<connectome-stem>_soma.csv`` sitting beside the connectome
     ``.npz`` (produced by ``scripts/fetch_soma_positions.py`` or persisted from tier-0). This
     is what makes the committed fixture demo anatomical offline.
2. **neuPrint** via ``neuprint-python`` + ``NEUPRINT_TOKEN`` — a dev-time, networked step
   for arbitrary slices whose bodyids are not in the meta or a committed sidecar. Never required
   by CI (guarded import; any failure degrades to the fallback).
3. **Deterministic spectral layout** — a seed-free spectral embedding of the (undirected,
   magnitude-weighted) connectome graph, sign-canonicalised so it is bit-reproducible. A
   **last-resort** fallback only for neurons/connectomes with no ``somaLocation`` anywhere.
   Labelled ``"computed (NOT anatomical …)"``.

Full precedence: **meta ``somaLocation`` → ``DRONE_FLY_SOMA_CSV`` → ``<stem>_soma.csv`` →
neuPrint → spectral** (enforced in :func:`_load_anatomical`).

Neurons that lack a real soma position anywhere — e.g. peripheral sensory afferents whose cell
bodies sit outside the brain volume — are **flagged** (``has_position=False``) and
fallback-placed — never dropped and never fabricated with a made-up anatomical value.

Full coverage + body-schematic placement (UC-28)
------------------------------------------------
Every neuron additionally carries a finite ``display3d`` render coordinate so the viewer can
draw all N neurons on any plane (AC-2). Real-soma neurons render at their true anatomy
(``placement="anatomical"``). Soma-less afferents keep ``coords3d=None`` but are placed in a
deterministic **schematic fly body around the brain** (``placement="schematic"``), grouped by
a real categorical body-region label (:func:`resolve_body_region`: ``leg`` / ``haltere`` /
``campaniform`` / ``vnc`` / ``ascending``, generic → ``torso``) at bounded offsets from the
real-soma brain bbox (:func:`_schematic_body_coords`) — honest (no fabricated precise xyz) and
brain-scale-guarded (:data:`BRAIN_DOMINANCE_MIN_FRACTION`). When no anatomy anchors the layout
at all, every neuron is ``placement="computed"`` (spectral). Arrays are never reordered.

Provisioning at slice time (UC-27)
----------------------------------
Positions are provisioned **once, when a connectome artifact is created** — the ``prune``
slice, the ``fetch-connectome`` base download, and the ``prune-trained`` subcircuit — and
cached as a ``<connectome-stem>_positions.csv`` sidecar beside the ``.npz`` (plus a
``<stem>_soma.csv`` sidecar carrying the real anatomy resolved from tier-0 meta
``somaLocation``, tokenless — or from neuPrint when the meta lacks it). Recording-enabled
training then only *loads* the sidecar: no per-run neuPrint fetch, no per-run spectral
eigendecomposition. :func:`resolve_positions` is the single entry point for both sides.

Because ``save_connectome`` does not copy ``somaLocation`` into a pruned artifact's own meta,
the ``prune`` / ``prune-trained`` hooks pass the **source** connectome (``source_data``) so
tier-0 anatomy is read from the source meta and subset to the artifact's bodyids;
``fetch-connectome`` reads the fetched connectome's own sibling meta directly.

* **Sidecar format.** One row per neuron, columns
  ``bodyid,has_position,x,y,z,u,v,source,projection,placement,region,x3d,y3d,z3d`` (a verbatim
  serialisation of the :func:`provision_positions` dict). ``x/y/z`` are blank iff that neuron
  has no 3-D anatomy (``coords3d[i] is None``); ``x3d/y3d/z3d`` (``display3d``) are always
  finite; ``has_position`` is an *independent* column. Floats use ``%.17g`` so a load
  reproduces the compute bit-exactly. A pre-UC-28 sidecar lacks the last five columns → MISS
  → recompute (self-heal).
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
  :data:`SPECTRAL_MAX_ENV` = ``DRONE_FLY_SPECTRAL_MAX``) *and* there is zero real anatomy to
  anchor a schematic body, provisioning is deferred (WARN, no ``_positions.csv``): provide
  anatomy (``NEUPRINT_TOKEN`` or ``DRONE_FLY_SOMA_CSV``) or prune the connectome before
  recording. Partial or complete real anatomy at any scale still provisions (real soma coords
  plus cheap schematic placement for soma-less afferents — no spectral needed).

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
import zlib
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
#: UC-28 appends three additive,
#: always-present columns — ``placement`` / ``region`` (see below) and ``x3d/y3d/z3d`` (the
#: full-coverage :data:`display3d` render coords, always finite). Old sidecars written before
#: UC-28 lack these columns, so :func:`_load_positions_sidecar`'s column-subset check turns
#: them into a MISS → the recorder self-heals by recomputing (there is no committed sidecar,
#: so nothing breaks).
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
    "placement",
    "region",
    "x3d",
    "y3d",
    "z3d",
)

# --- UC-28: full-coverage body-schematic placement ------------------------------------------
#
# UC-27 renders real-soma neurons at true anatomy and flags soma-less peripheral afferents
# (``has_position=False``); those afferents were previously fallback-placed *inside* the brain
# extent (``_scale_into_box``), overlapping the brain. UC-28 replaces that with a **schematic
# fly body around the brain**: each soma-less afferent is placed in a deterministic body-region
# cluster derived from its real categorical label (``subclass`` for limbs, ``superclass`` for
# vnc/ascending). This is honest — no fabricated precise xyz; ``coords3d`` stays real-or-``None``
# and the new :data:`display3d` carries the schematic render coords, distinctly flagged via
# :data:`placement`.

#: Per-neuron placement category recorded in ``positions.placement`` (drives the viewer's
#: AC-6 visual distinction and per-neuron honesty).
PLACEMENT_ANATOMICAL = "anatomical"  # real soma (``has_position=True``)
PLACEMENT_SCHEMATIC = "schematic"  # soma-less afferent placed in a body-region cluster
PLACEMENT_COMPUTED = "computed"  # whole-slice spectral fallback (no brain to anchor to)

#: Body-region labels for schematic-placed afferents (recorded in ``positions.region``;
#: ``""`` for anatomical & computed neurons).
REGION_LEG = "leg"
REGION_HALTERE = "haltere"
REGION_CAMPANIFORM = "campaniform"
REGION_VNC = "vnc"
REGION_ASCENDING = "ascending"
REGION_TORSO = "torso"  # generic / unrecognised label → neutral torso group (AC-4)

#: Ordered region-label rules with documented precedence. Each rule is
#: ``(attr, match, values, region)`` where ``attr`` is a :class:`ConnectomeData` label column,
#: ``match`` is ``"exact"`` or ``"prefix"``, and ``values`` the label(s) to test. Rules are
#: tried in order and the **first** match wins, so the fine-grained limb ``subclass`` (leg /
#: haltere / campaniform) takes precedence over the coarse ``superclass`` (vnc_sensory,
#: sensory_ascending). Anything matching no rule falls to :data:`REGION_TORSO`. The region
#: label lives in ``subclass`` + ``superclass`` (verified against the MaleCNS fixture), NOT in
#: ``cell_type`` (opaque codes).
REGION_LABEL_RULES: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("subclass", "exact", ("leg",), REGION_LEG),
    ("subclass", "exact", ("haltere",), REGION_HALTERE),
    ("subclass", "prefix", ("campaniform",), REGION_CAMPANIFORM),
    ("superclass", "exact", ("vnc_sensory",), REGION_VNC),
    ("superclass", "exact", ("sensory_ascending",), REGION_ASCENDING),
)

#: Schematic body-cluster centre offsets, expressed as **multiples of the real brain's per-axis
#: bounding-box extent** and added to the brain centre (see :func:`_schematic_body_coords`).
#: They are a documented schematic choice (no real peripheral xyz exists), chosen so the six
#: clusters sit *outside* and around the brain in distinct directions (spatial separation, AC-6)
#: while every component stays within :data:`MAX_BODY_OFFSET_FACTOR` (the brain-dominance
#: guardrail, AC-5). Axis order is neuPrint MaleCNS voxel space ``(x, y, z)``; the default
#: ``xz`` projection is the dorsal top-down view (``y`` is dorsal-ventral).
REGION_CLUSTER_OFFSETS: dict[str, tuple[float, float, float]] = {
    REGION_LEG: (0.0, 0.75, -0.55),  # legs: ventral & below the brain
    REGION_HALTERE: (0.70, 0.55, 0.0),  # haltere: postero-lateral thorax
    REGION_CAMPANIFORM: (0.70, -0.55, 0.0),  # wing campaniform: opposite lateral thorax
    REGION_VNC: (-0.75, 0.55, 0.0),  # ventral nerve cord: posterior
    REGION_ASCENDING: (-0.75, -0.55, 0.0),  # ascending sensory: posterior, opposite side
    REGION_TORSO: (0.0, 0.80, 0.0),  # generic torso group: ventral, distinct from limbs
}

#: Upper bound on any single-axis component of a :data:`REGION_CLUSTER_OFFSETS` entry, in
#: brain-extent units. The guardrail math below depends on this being a true cap.
MAX_BODY_OFFSET_FACTOR = 0.9

#: Radius of the deterministic within-cluster spread, in brain-extent units. A neuron's exact
#: point inside its cluster is a deterministic function of its bodyid (crc32), never random.
REGION_CLUSTER_RADIUS_FRAC = 0.10

#: Documented lower bound on the brain's share of the total rendered extent (AC-5). The
#: conservative worst-case relationship is::
#:
#:     brain_fraction >= 1 / (1 + 2 * (MAX_BODY_OFFSET_FACTOR + REGION_CLUSTER_RADIUS_FRAC))
#:
#: i.e. the farthest a body point can sit beyond the brain, on each side, is
#: ``(MAX + RADIUS) * brain_extent``. With MAX=0.9 and RADIUS=0.10 that bound is 1/3 ≈ 0.333,
#: so any :data:`BRAIN_DOMINANCE_MIN_FRACTION` at or below it is guaranteed. The real geometry
#: (offsets are not all maximal and rarely symmetric on one axis) leaves the brain well above
#: this floor. Keep this relationship true if the constants are ever retuned — the guardrail
#: test asserts the *actual* brain bbox stays ≥ this fraction of the total.
BRAIN_DOMINANCE_MIN_FRACTION = 0.30

# Guardrail invariants (kept honest at import time so the constants and the documented
# relationship above can never silently drift apart).
assert all(
    max(abs(c) for c in offset) <= MAX_BODY_OFFSET_FACTOR
    for offset in REGION_CLUSTER_OFFSETS.values()
), "a REGION_CLUSTER_OFFSETS component exceeds MAX_BODY_OFFSET_FACTOR"
assert (
    1.0 / (1.0 + 2.0 * (MAX_BODY_OFFSET_FACTOR + REGION_CLUSTER_RADIUS_FRAC))
) >= BRAIN_DOMINANCE_MIN_FRACTION, "BRAIN_DOMINANCE_MIN_FRACTION violates the guardrail bound"
assert set(REGION_CLUSTER_OFFSETS) == {
    REGION_LEG,
    REGION_HALTERE,
    REGION_CAMPANIFORM,
    REGION_VNC,
    REGION_ASCENDING,
    REGION_TORSO,
}, "REGION_CLUSTER_OFFSETS must cover every body region exactly"


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


def _resolve_meta_soma(
    data: ConnectomeData,
    source_data: ConnectomeData | None = None,
) -> dict[int, tuple[float, float, float]]:
    """Tier-0 real anatomy: read ``somaLocation`` from the connectome's OWN meta CSV.

    The MaleCNS download ships ``mcns_all_neuron_meta.csv`` (saved as ``<npz-stem>_meta.csv``)
    whose ``somaLocation`` column holds real neuPrint 8 nm-voxel soma coordinates (CC-BY) —
    tokenless, offline, the PRIMARY anatomy source (UC-27 AC-10). This returns a (possibly
    partial/empty) ``bodyid -> (x, y, z)`` mapping subset to ``data.neuron_ids``.

    Anchor selection:

    * **Write side** (``source_data`` given — prune/prune-trained): a pruned artifact's own
      ``save_connectome`` meta carries NO ``somaLocation`` (and ``ConnectomeData`` has no soma
      field), so anatomy must come from the **SOURCE** connectome's meta. ``_base_npz_path``
      strips the ``" [pruned:…]"`` / ``" [activation-pruned …]"`` provenance tag to recover the
      real on-disk source ``.npz``, then reads its sibling ``_meta.csv``. As a WRITE-side-only
      fallback (when the source meta lacks ``somaLocation``), the source's ``<stem>_soma.csv``
      is used.
    * **Read / fetch side** (``source_data=None``): the anchor is ``data`` itself — a freshly
      fetched full connectome reads its own sibling meta (which has ``somaLocation``); a pruned
      artifact's meta has none, so tier-0 is empty here and anatomy resolves at its normal
      tier-2 ``<stem>_soma.csv`` (written at slice time) inside :func:`_load_anatomical`.

    Empty/partial results are expected: soma-less peripheral afferents flow to the
    partial-anatomy fill (never faked). Returns an int-keyed dict (``{}`` when nothing found).
    """
    anchor = source_data if source_data is not None else data
    base_npz = _base_npz_path(anchor)
    if base_npz is None:
        return {}

    mapping: dict[int, tuple[float, float, float]] = {}
    meta_path = base_npz.with_name(f"{base_npz.stem}_meta.csv")
    if meta_path.is_file():
        try:
            frame = pd.read_csv(meta_path, usecols=["bodyid", "somaLocation"], low_memory=False)
        except ValueError:
            frame = None  # no somaLocation column in this meta → tier-0 unavailable here
        except OSError as exc:
            logger.warning(
                "Could not read meta CSV %s (%s); skipping tier-0 anatomy.", meta_path, exc
            )
            frame = None
        if frame is not None:
            mapping = _mapping_from_frame(frame)

    # WRITE-side-only fallback: the source's own soma sidecar when its meta has no somaLocation.
    if not mapping and source_data is not None:
        soma_path = base_npz.with_name(f"{base_npz.stem}_soma.csv")
        if soma_path.is_file():
            mapping = _load_soma_csv(soma_path)

    if not mapping:
        return {}

    # Subset to this artifact's neuron set (the meta may cover a superset, e.g. the full graph),
    # using the same int canonicalization as the node-set HIT rule.
    wanted = {int(b) for b in np.asarray(data.neuron_ids).tolist() if _is_int_like(b)}
    return {bid: xyz for bid, xyz in mapping.items() if bid in wanted}


def _load_anatomical(
    data: ConnectomeData,
    *,
    override: dict[int, tuple[float, float, float]] | None = None,
) -> tuple[dict[int, tuple[float, float, float]], str] | None:
    """Resolve anatomical soma positions via the documented source order.

    Precedence (highest first): meta ``somaLocation`` (tier-0, passed in as ``override``) →
    ``DRONE_FLY_SOMA_CSV`` → ``<stem>_soma.csv`` sidecar → neuPrint → (caller's) spectral. A
    non-empty ``override`` wins outright as tier-0; any neurons it does not cover flow to the
    partial-anatomy fill in :func:`provision_positions` (they are not looked up in lower tiers —
    once real meta anatomy is present it is authoritative). ``override=None`` (the default)
    reproduces the historical order exactly, so numeric output is unchanged (AC-8).

    Returns ``(mapping, source_label)`` for the first source that yields at least one position,
    else ``None`` (caller then uses the computed fallback).
    """
    if override:
        return dict(override), "anatomical (connectome meta somaLocation)"

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


def _label_column(data: ConnectomeData, attr: str) -> list[str]:
    """Return one label string per neuron for ``attr`` (``""`` for missing / NaN / absent).

    A ``None`` attribute (older/reduced fixture lacking the column) yields all-``""`` — those
    neurons then fall through :data:`REGION_LABEL_RULES` to :data:`REGION_TORSO` rather than
    erroring (documented degrade, never fabricated).
    """
    values = getattr(data, attr, None)
    if values is None:
        return [""] * data.neuron_count
    return ["" if pd.isna(v) else str(v) for v in np.asarray(values).tolist()]


def resolve_body_region(data: ConnectomeData) -> list[str]:
    """Return one body-region label per neuron (index-aligned), for schematic placement.

    Applies :data:`REGION_LABEL_RULES` in order (first match wins), so the fine-grained limb
    ``subclass`` (leg / haltere / campaniform) takes precedence over the coarse ``superclass``
    (vnc_sensory → :data:`REGION_VNC`, sensory_ascending → :data:`REGION_ASCENDING`). Any
    neuron matching no rule — a blank or unrecognised label — maps to :data:`REGION_TORSO`
    (AC-4). The label is computed for *every* neuron; the caller uses it only for soma-less
    (schematic) neurons and records ``""`` for anatomical / computed ones.
    """
    columns: dict[str, list[str]] = {}
    regions: list[str] = []
    for i in range(data.neuron_count):
        label = REGION_TORSO
        for attr, match, values, region_label in REGION_LABEL_RULES:
            if attr not in columns:
                columns[attr] = _label_column(data, attr)
            text = columns[attr][i]
            if match == "exact":
                hit = text in values
            else:  # "prefix"
                hit = any(text.startswith(v) for v in values)
            if hit:
                label = region_label
                break
        regions.append(label)
    return regions


def _crc_unit_ball(bodyid: object) -> np.ndarray:
    """Deterministic point in the unit ball keyed on ``bodyid`` (bit-reproducible cross-process).

    Derives three components in ``[-1, 1]`` from ``zlib.crc32`` over ``"{bodyid}:{axis}"`` and
    clamps the vector to the unit ball (project onto the surface when it lands outside). The
    builtin :func:`hash` is **deliberately not used**: it is salted per process
    (``PYTHONHASHSEED``) for ``str`` inputs, so it would give different layouts on different
    runs/machines. ``crc32`` is a fixed polynomial → identical output everywhere (AC-3).
    """

    def component(axis: str) -> float:
        digest = zlib.crc32(f"{bodyid}:{axis}".encode())
        return (digest / 0xFFFFFFFF) * 2.0 - 1.0

    vector = np.array([component("x"), component("y"), component("z")], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > 1.0:
        vector /= norm
    return vector


def _schematic_body_coords(
    data: ConnectomeData,
    regions: list[str],
    coords3d: list[list[float] | None],
    missing_idx: list[int],
) -> np.ndarray | None:
    """Compute schematic body-cluster 3-D coords for the soma-less (missing) neurons.

    The real-soma coordinates in ``coords3d`` define the brain bounding box (centre + per-axis
    extent). Each missing neuron is placed at ``brain_center + REGION_CLUSTER_OFFSETS[region] *
    brain_extent`` plus a deterministic within-cluster spread of :data:`REGION_CLUSTER_RADIUS_FRAC`
    · ``brain_extent`` (via :func:`_crc_unit_ball`). Returns an ``(len(missing_idx), 3)`` array
    aligned to ``missing_idx``, or ``None`` when there is no real soma to anchor to (the caller
    then degrades to the computed spectral layout).
    """
    real = np.array([c for c in coords3d if c is not None], dtype=np.float64)
    if real.shape[0] == 0:
        return None
    brain_min = real.min(axis=0)
    brain_max = real.max(axis=0)
    brain_center = (brain_min + brain_max) / 2.0
    brain_extent = brain_max - brain_min
    # A degenerate (zero-extent) axis would collapse the whole body onto the brain; guard it so
    # offsets/spread stay meaningful and finite.
    brain_extent = np.where(brain_extent < 1e-9, 1.0, brain_extent)

    ids = np.asarray(data.neuron_ids).tolist()
    out = np.zeros((len(missing_idx), 3), dtype=np.float64)
    for slot, i in enumerate(missing_idx):
        offset = np.asarray(REGION_CLUSTER_OFFSETS[regions[i]], dtype=np.float64)
        center = brain_center + offset * brain_extent
        spread = _crc_unit_ball(ids[i]) * (REGION_CLUSTER_RADIUS_FRAC * brain_extent)
        out[slot] = center + spread
    return out


def _computed_positions(
    data: ConnectomeData, projection: str, axes: tuple[int, int], n: int
) -> dict:
    """Whole-slice computed (spectral) positions when no anatomy anchors the layout.

    Every neuron gets the deterministic spectral embedding, labelled non-anatomical
    (:data:`SOURCE_COMPUTED`) with ``placement="computed"`` and ``region=""``. ``display3d``
    mirrors the finite spectral coords so the viewer still renders all neurons (AC-2/AC-8).
    """
    embedding = _spectral_embedding(data)
    coords2d = _project(embedding, axes)
    logger.info(
        "No anatomical soma positions available; using the deterministic computed "
        "spectral layout for all %d neurons (labelled non-anatomical).",
        n,
    )
    emb_list = embedding.tolist()
    return {
        "source": SOURCE_COMPUTED,
        "projection": projection,
        "coords3d": emb_list,
        "coords2d": coords2d.tolist(),
        "has_position": [False] * n,
        "placement": [PLACEMENT_COMPUTED] * n,
        "region": [""] * n,
        "display3d": [list(row) for row in emb_list],
    }


def provision_positions(
    data: ConnectomeData,
    *,
    projection: str = DEFAULT_PROJECTION,
    anatomy_override: dict[int, tuple[float, float, float]] | None = None,
) -> dict:
    """Provision soma positions + a top-down projection for every neuron (once).

    Returns a JSON-ready dict matching the recording schema's ``positions`` block::

        {
          "source": <provenance label>,          # "anatomical (…)" | computed fallback
          "projection": <plane, e.g. "xz">,
          "coords3d": [[x, y, z] | null, ...],    # null only for soma-less neurons (honest)
          "coords2d": [[u, v], ...],              # ALWAYS finite (projection of display3d)
          "has_position": [bool, ...],            # True iff a real anatomical soma
          "placement": [str, ...],                # "anatomical" | "schematic" | "computed"
          "region": [str, ...],                   # body region for schematic; "" otherwise
          "display3d": [[x, y, z], ...],          # UC-28: full-coverage finite render coords
        }

    UC-28 completeness + honesty: real-soma neurons keep their true ``coords3d`` and are
    ``placement="anatomical"``. Soma-less afferents keep ``coords3d=null`` (no fabricated
    anatomy) but get a deterministic **schematic body-cluster** position in ``display3d`` —
    ``placement="schematic"`` with a ``region`` label (:func:`resolve_body_region`) — placed
    around the real brain (:func:`_schematic_body_coords`). ``display3d`` is finite for every
    neuron and ``coords2d`` is its projection, so the viewer renders all N neurons on any plane
    (AC-2). When no anatomy is reachable, every neuron gets the computed spectral layout
    (``placement="computed"``). Arrays are never reordered — the positional activation↔neuron
    join stays exact (AC-8).

    ``anatomy_override`` (UC-27 AC-10) is the tier-0 meta-``somaLocation`` mapping; when non-empty
    it is the authoritative anatomy source (see :func:`_load_anatomical`).
    """
    if projection not in PROJECTIONS:
        raise ValueError(
            f"Unknown projection {projection!r}; supported: {sorted(PROJECTIONS)} "
            f"(default {DEFAULT_PROJECTION!r})."
        )
    axes = PROJECTIONS[projection]
    n = data.neuron_count
    ids = np.asarray(data.neuron_ids).tolist()

    anatomical = _load_anatomical(data, override=anatomy_override)
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

        missing_idx = [i for i, ok in enumerate(has_position) if not ok]
        regions = resolve_body_region(data)
        schematic = _schematic_body_coords(data, regions, coords3d, missing_idx)
        if schematic is None:
            # Anatomy source matched none of this connectome's ids (a subset/id mismatch): with
            # no real soma there is no brain to anchor a schematic body to, so degrade to the
            # whole-slice computed spectral layout rather than crash or place a body around
            # nothing (AC-8).
            logger.info(
                "Anatomy source %r covered no neuron in this connectome; using the computed "
                "spectral layout for all %d neurons.",
                source,
                n,
            )
            return _computed_positions(data, projection, axes, n)

        # display3d: full-coverage finite render coords for EVERY neuron — real soma where we
        # have one, schematic body cluster otherwise. coords3d stays real-or-None (honesty);
        # coords2d is display3d projected, so every neuron renders on the default plane too.
        display3d: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(n)]
        placement: list[str] = [""] * n
        region: list[str] = [""] * n
        for i, ok in enumerate(has_position):
            if ok:
                display3d[i] = list(coords3d[i])  # type: ignore[arg-type]
                placement[i] = PLACEMENT_ANATOMICAL
        for slot, i in enumerate(missing_idx):
            display3d[i] = [float(v) for v in schematic[slot]]
            placement[i] = PLACEMENT_SCHEMATIC
            region[i] = regions[i]

        coords2d = _project(np.asarray(display3d, dtype=np.float64), axes)
        if missing_idx:
            logger.info(
                "Provisioned anatomical soma positions for %d/%d neurons "
                "(%d schematic body-placed).",
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
            "placement": placement,
            "region": region,
            "display3d": display3d,
        }

    return _computed_positions(data, projection, axes, n)


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
    is its own column, independent of ``coords3d`` null-ness. UC-28 also writes ``placement`` /
    ``region`` and the always-finite ``x3d/y3d/z3d`` (``display3d``) full-coverage render coords.

    When ``strict`` is false any :class:`OSError` (e.g. a read-only dir on the recorder
    self-heal path) is logged and swallowed rather than raised.
    """
    ids = np.asarray(data.neuron_ids).tolist()
    coords3d = positions["coords3d"]
    coords2d = positions["coords2d"]
    has_position = positions["has_position"]
    source = positions["source"]
    projection = positions["projection"]
    # UC-28 additive columns (always present in a freshly-provisioned dict).
    placement = positions["placement"]
    region = positions["region"]
    display3d = positions["display3d"]

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
        d3 = display3d[i]  # display3d is always finite (full coverage)
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
                "placement": placement[i],
                "region": region[i],
                "x3d": _FLOAT_FMT % float(d3[0]),
                "y3d": _FLOAT_FMT % float(d3[1]),
                "z3d": _FLOAT_FMT % float(d3[2]),
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
    (blank ``x/y/z`` → ``coords3d[i] = None``; ``placement`` / ``region`` / ``display3d`` read
    back verbatim). Anything else (unreadable, missing columns, wrong length, id-set mismatch —
    a stale/wrong-subcircuit artifact —, projection mismatch) is a MISS → caller recomputes.
    A pre-UC-28 sidecar lacks the ``placement`` / ``region`` / ``x3d/y3d/z3d`` columns, so it
    MISSes here and self-heals into a recompute. Never applies partial/wrong positions (AC-7).
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
    placement: list[str] = []
    region: list[str] = []
    display3d: list[list[float]] = []
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
        p = frame["placement"].iloc[i]
        r = frame["region"].iloc[i]
        placement.append("" if pd.isna(p) else str(p))
        region.append("" if pd.isna(r) else str(r))
        display3d.append(
            [
                float(frame["x3d"].iloc[i]),
                float(frame["y3d"].iloc[i]),
                float(frame["z3d"].iloc[i]),
            ]
        )

    return {
        "source": str(frame["source"].iloc[0]),
        "projection": projection,
        "coords3d": coords3d,
        "coords2d": coords2d,
        "has_position": has_position,
        "placement": placement,
        "region": region,
        "display3d": display3d,
    }


def _write_soma_sidecar(
    soma_path: Path,
    mapping: dict[int, tuple[float, float, float]],
    *,
    strict: bool,
) -> None:
    """Write a ``bodyid,x,y,z`` soma sidecar from an in-memory mapping.

    This is the persisted, tokenless-reusable real-anatomy artifact (the existing tier-2
    ``<stem>_soma.csv``). Coordinates are written at :data:`_FLOAT_FMT`. A write ``OSError`` is
    raised when ``strict`` (write hooks), else logged and swallowed (recorder best-effort).
    """
    frame = pd.DataFrame(
        [
            {
                "bodyid": bid,
                "x": _FLOAT_FMT % float(xyz[0]),
                "y": _FLOAT_FMT % float(xyz[1]),
                "z": _FLOAT_FMT % float(xyz[2]),
            }
            for bid, xyz in mapping.items()
        ]
    )
    try:
        frame.to_csv(soma_path, index=False)
    except OSError as exc:
        if strict:
            raise
        logger.warning("Could not persist soma sidecar %s (%s); continuing.", soma_path, exc)
        return
    logger.info("Persisted %d real soma positions to %s.", len(mapping), soma_path)


def _maybe_persist_neuprint_soma(
    data: ConnectomeData, soma_path: Path | None, *, strict: bool
) -> None:
    """Fetch real anatomy from neuPrint once and persist it as ``<stem>_soma.csv`` (AC-6).

    Only reached when tier-0 meta ``somaLocation`` was empty (see :func:`resolve_positions`).
    A no-op unless ``NEUPRINT_TOKEN`` is set, ``DRONE_FLY_SOMA_CSV`` is NOT set (that tier-1
    override wins on its own), and the soma sidecar does not already exist. On success the
    sidecar is written beside the artifact so later loads reuse it via tier-2 with no further
    network calls (neuPrint preferred over the spectral fallback). Dev-time only; never hit by CI.
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
    _write_soma_sidecar(  # pragma: no cover - dev-time networked path, not exercised in CI
        soma_path, mapping, strict=strict
    )


def _anatomy_coverage(
    data: ConnectomeData,
    *,
    override: dict[int, tuple[float, float, float]] | None = None,
) -> int:
    """Count neurons for which :func:`_load_anatomical` yields a real soma position.

    Used only by the large-connectome guard to decide whether a compute would need the
    spectral fallback. ``override`` is the tier-0 meta mapping (so the full connectome's real
    meta anatomy counts against the cap — AC-11). Returns 0 when no anatomy is reachable.
    """
    anatomical = _load_anatomical(data, override=override)
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
    source_data: ConnectomeData | None = None,
    persist: bool = True,
    persist_strict: bool = True,
) -> dict | None:
    """Resolve neuron positions via the cached sidecar, else compute-once-and-persist.

    This is the single orchestration point for UC-27: provisioning is done at slice/artifact
    time (``artifact_npz`` set, write hooks) and training only ever *loads* the sidecar. The
    recorder calls it on the read side (``artifact_npz=None``, keyed off ``data.source``) and
    self-heals older artifacts that predate the sidecar.

    Anatomy precedence (AC-10): **meta ``somaLocation`` (tier-0) → ``DRONE_FLY_SOMA_CSV`` →
    ``<stem>_soma.csv`` → neuPrint → spectral**. Real anatomy is tokenless by default: the
    connectome's own meta CSV supplies real soma coordinates offline.

    Flow:

    1. **Fast path** — a valid ``<stem>_positions.csv`` returns immediately with **no** spectral
       eigendecomposition and **no** neuPrint fetch (AC-4).
    2a. **Tier-0 meta anatomy** — read real ``somaLocation`` for the artifact's bodyids from the
        (source) connectome meta (:func:`_resolve_meta_soma`), tokenless (AC-10).
    2b. **Persist real soma** — when tier-0 is non-empty, (over)write ``<stem>_soma.csv`` from it
        so it is authoritative for later self-heal loads (never a stale sidecar).
    2c. **neuPrint** — only when tier-0 is empty and a token is set (AC-6).
    3. **Large-connectome guard** — when a compute would still need the dense spectral fallback
       for ``n > _spectral_cap()`` neurons AND there is zero real anatomy (incl. tier-0) to anchor
       a schematic body, do NOT build the giant Laplacian: WARN, leave ``_positions.csv``
       unwritten, return ``None``. Any real anatomy (partial or complete) means the connectome
       provisions without an ``eigh`` — real soma coords plus cheap schematic placement (AC-11).
    4. **Compute** via :func:`provision_positions` (``anatomy_override`` = tier-0 map; identical
       output when the map is empty — AC-8).
    5. **Persist** the result to ``<stem>_positions.csv`` when ``persist`` (best-effort when
       ``persist_strict`` is false — the recorder's read-only-dir case).

    Parameters
    ----------
    artifact_npz:
        The just-saved connectome ``.npz`` (write hooks). Sidecars are keyed off it, never off
        the pruned ``data.source``. ``None`` on the read side (recorder).
    source_data:
        The SOURCE connectome the artifact was derived from (prune → the pre-prune ``data``;
        prune-trained → ``base``). Its meta carries the real ``somaLocation``; a pruned
        artifact's own meta does not. ``None`` on the read/fetch side (anatomy comes from the
        artifact's own sibling meta / sidecar).
    persist / persist_strict:
        Whether to write the sidecars, and whether a write ``OSError`` is fatal.

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

    # 2a. Tier-0 real anatomy from the connectome meta CSV (tokenless, primary) — AC-10.
    meta_map = _resolve_meta_soma(compute_data, source_data)

    # 2b. Persist tier-0 anatomy as the authoritative real soma sidecar (overwrite any stale one).
    if persist and meta_map and soma_path is not None:
        _write_soma_sidecar(soma_path, meta_map, strict=persist_strict)
    # 2c. neuPrint only when the meta yielded no real anatomy (dev-time; preferred over spectral).
    elif not meta_map:
        _maybe_persist_neuprint_soma(compute_data, soma_path, strict=persist_strict)

    # 3. Large-connectome refuse guard — never build a full-MaleCNS-scale dense eigh. Provisioning
    # only needs the expensive spectral fallback when there is ZERO real anatomy to anchor: with
    # ≥1 real soma, soma-less afferents are placed via cheap schematic body clusters (UC-28, no
    # eigh). So defer only when coverage is exactly zero (and n > cap); partial or full anatomy
    # provisions at any scale.
    n = data.neuron_count
    if n > _spectral_cap() and _anatomy_coverage(compute_data, override=meta_map) == 0:
        logger.warning(
            "Zero real anatomy for a %d-neuron connectome (> spectral cap %d): with no real soma "
            "to anchor a schematic body, the full-graph position layout would require a dense "
            "spectral layout at this scale, which is infeasible, so it was NOT computed. "
            "Provide anatomy (%s / %s) or prune the connectome before recording.",
            n,
            _spectral_cap(),
            "NEUPRINT_TOKEN",
            SOMA_CSV_ENV,
        )
        return None

    # 4. Compute (output-identical to the historical path when meta_map is empty — AC-8). On the
    # read side a miss means the artifact predates the sidecar — WARN (AC-5 self-heal).
    if artifact_npz is None:
        logger.warning(
            "No positions sidecar for this connectome (%s); provisioning positions on the fly "
            "and caching them beside the artifact so later runs load instead of recompute.",
            pos_path,
        )
    result = provision_positions(compute_data, projection=projection, anatomy_override=meta_map)

    # 5. Persist for next time.
    if persist and pos_path is not None:
        _write_positions_sidecar(pos_path, data, result, strict=persist_strict)
    return result
