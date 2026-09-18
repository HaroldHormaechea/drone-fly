"""Function-targeted connectome subgraph pruning (UC-04).

Running the full MaleCNS connectome (~161k neurons / ~25M edges) as the live policy
network makes PPO intractable: every forward/backward pass propagates the whole graph
even though only the directed ``visual_projection`` (sensory) → ``descending_neuron``
(motor) subcircuit drives the 4 control outputs. This module reduces a loaded
:class:`~drone_fly.connectome.loader.ConnectomeData` to that subcircuit with a
deterministic, **direction-aware** rule, returning a new (smaller) ``ConnectomeData``
whose meta (``neuron_ids`` / ``superclass`` / ``neuron_class`` / ``subclass`` / ``sign`` /
``top_nt``) is re-aligned to the pruned matrix rows. The input is never mutated.

Direction convention
--------------------
The adjacency matrix follows the loader contract: ``adjacency[i, j]`` is the weight from
neuron ``j`` onto neuron ``i`` — i.e. a directed edge ``j -> i`` has ``row = i``,
``col = j``. Therefore:

* **successors** of ``j`` (nodes ``j`` projects onto) are the nonzero *rows* of column
  ``j`` → read them from the CSC column slice;
* **predecessors** of ``i`` (nodes projecting onto ``i``) are the nonzero *columns* of
  row ``i`` → read them from the CSR row slice.

Forward reachability from the sensory set follows successors; backward reachability from
the motor set follows predecessors. Getting this right is what makes the retained graph
the *directed* sensory→…→motor circuit rather than an undirected blob (AC3).

Retained-set construction (path-slack rule, Option A — path inclusion)
----------------------------------------------------------------------
Given per-neuron ``superclass`` metadata, the sensory endpoints are every
``visual_projection`` neuron and the motor endpoints every ``descending_neuron`` neuron
(the full superclass sets — not the capped UC-02 populations — so downstream population
selection still finds them). Then:

1. Forward BFS from **all** sensory endpoints over successors, tracking parent pointers
   → ``d_fwd`` (hops sensory→node) and ``parent``.
2. Backward BFS from **all** motor endpoints over predecessors → ``d_bwd`` (hops
   node→motor). ``L = min d_bwd`` over the sensory endpoints is the shortest sensory→motor
   path length.
3. Retained =
   all sensory endpoints
   ∪ (for each sensory-reachable motor ``m``: the nodes of one reconstructed shortest
   sensory→``m`` path, via ``parent``)
   ∪ the **corridor** ``{u : d_fwd(u) + d_bwd(u) <= L + k}``.

The path-inclusion term guarantees AC2 by construction: every ``parent[v] -> v`` edge
survives ``adjacency[kept][:, kept]``, so each retained motor stays reachable from a
sensory neuron *within the pruned subgraph* for any ``k`` — a pure corridor could strand
a motor whose only sensory path runs outside the ``L + k`` band. ``k`` (default
:data:`DEFAULT_PRUNE_K`) trades tightness for richness: ``k = 0`` keeps only the tightest
shortest-path corridor; larger ``k`` yields a monotone superset (AC5).

A motor not reachable from any sensory neuron is dropped with a warning (never an error).
Absent ``superclass`` meta, a missing sensory/motor population, ``k < 0``, an unknown
rule, or an empty/degenerate retained set all raise a clear, actionable error (AC9).
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.populations import MOTOR_SUPERCLASS, SENSORY_SUPERCLASS

logger = logging.getLogger(__name__)

#: The one supported pruning rule: a direction-aware shortest-path corridor widened by a
#: slack parameter ``k`` (``k = 0`` = tight shortest-path corridor, larger ``k`` = richer
#: k-hop neighbourhood). Exposed as a documented constant (AC5).
PRUNE_RULE_PATH_SLACK = "path_slack"

#: Default slack parameter. ``2`` favours richness — descending (motor) neurons integrate
#: broadly, so a slightly wider corridor retains more of the real control circuitry while
#: still discarding the ~160k neurons off the sensory→motor pathway.
DEFAULT_PRUNE_K = 2

#: Default pruning rule.
DEFAULT_PRUNE_RULE = PRUNE_RULE_PATH_SLACK


def _bfs(
    indptr: np.ndarray, indices: np.ndarray, sources: np.ndarray, n: int, *, track_parent: bool
):
    """Unit-cost BFS over a CSR/CSC neighbour list; O(V + E).

    ``indptr`` / ``indices`` are a sparse matrix's neighbour arrays (CSC columns for a
    forward/successor sweep, CSR rows for a backward/predecessor sweep). Returns
    ``(dist, parent)`` where unreached nodes have ``dist == -1`` and ``parent == -1``;
    ``parent`` is ``None`` when ``track_parent`` is ``False``.
    """
    dist = np.full(n, -1, dtype=np.int64)
    parent = np.full(n, -1, dtype=np.int64) if track_parent else None
    queue: deque[int] = deque()
    for s in sources:
        s = int(s)
        if dist[s] == -1:
            dist[s] = 0
            queue.append(s)
    while queue:
        u = queue.popleft()
        du = dist[u]
        for idx in range(indptr[u], indptr[u + 1]):
            v = int(indices[idx])
            if dist[v] == -1:
                dist[v] = du + 1
                if parent is not None:
                    parent[v] = u
                queue.append(v)
    return dist, parent


def _reconstruct_path(node: int, parent: np.ndarray) -> list[int]:
    """Walk ``parent`` pointers from ``node`` up to its BFS seed; return all nodes on it."""
    nodes: list[int] = []
    v = int(node)
    while v != -1:
        nodes.append(v)
        v = int(parent[v])
    return nodes


def _endpoints(superclass: np.ndarray, label: str) -> np.ndarray:
    """Indices of every neuron whose ``superclass`` equals ``label`` (the full set)."""
    return np.nonzero(np.asarray(superclass) == label)[0]


def slice_connectome(
    data: ConnectomeData,
    kept: np.ndarray,
    *,
    source: str,
) -> ConnectomeData:
    """Return the induced subgraph of ``data`` on the neuron indices ``kept`` (a new object).

    This is the neuron-removal + meta-realignment primitive shared by UC-04's structural
    pruning (:func:`prune_to_subcircuit`) and UC-07's post-training activation pruning
    (:mod:`drone_fly.prune_trained`). Given a 1-D array of retained neuron indices, it keeps
    only the intra-retained edges (``adjacency[kept][:, kept]``) and re-aligns every
    per-neuron meta array (``neuron_ids`` / ``superclass`` / ``neuron_class`` / ``subclass`` /
    ``sign`` / ``top_nt``) to the pruned matrix rows, remapping indices to ``0..M-1``. Per-edge
    signs are preserved because
    :class:`~drone_fly.controller.policy.SparseConnectomeLayer` rebuilds the sign mask from
    the presynaptic (column) neuron's sign, and ``sign[kept]`` carries every retained
    neuron's sign. **The input ``data`` is never mutated** — a fresh :class:`ConnectomeData`
    is returned.

    Parameters
    ----------
    data:
        The connectome to slice.
    kept:
        1-D integer array of retained neuron indices, in the row order the pruned graph
        should adopt (callers pass a sorted array for a deterministic layout).
    source:
        Provenance tag stamped onto the returned :class:`ConnectomeData`.
    """
    kept = np.asarray(kept, dtype=np.int64)
    sliced_adj = data.adjacency.tocsr()[kept][:, kept].tocsr().astype(np.float32)
    sliced_ids = np.asarray(data.neuron_ids)[kept]
    sliced_superclass = None if data.superclass is None else np.asarray(data.superclass)[kept]
    sliced_sign = None if data.sign is None else np.asarray(data.sign)[kept]
    sliced_top_nt = None if data.top_nt is None else np.asarray(data.top_nt)[kept]
    sliced_neuron_class = None if data.neuron_class is None else np.asarray(data.neuron_class)[kept]
    sliced_subclass = None if data.subclass is None else np.asarray(data.subclass)[kept]
    return ConnectomeData(
        adjacency=sliced_adj,
        neuron_ids=sliced_ids,
        source=source,
        superclass=sliced_superclass,
        sign=sliced_sign,
        top_nt=sliced_top_nt,
        neuron_class=sliced_neuron_class,
        subclass=sliced_subclass,
    )


def prune_to_subcircuit(
    data: ConnectomeData,
    *,
    k: int = DEFAULT_PRUNE_K,
    rule: str = DEFAULT_PRUNE_RULE,
) -> ConnectomeData:
    """Reduce ``data`` to its directed sensory→motor subcircuit (a new ``ConnectomeData``).

    Retains the neurons and edges on the directed ``visual_projection`` → ``descending_neuron``
    pathway under the path-slack rule (see the module docstring), re-aligning
    ``neuron_ids`` / ``superclass`` / ``neuron_class`` / ``subclass`` / ``sign`` / ``top_nt``
    to the pruned matrix rows and remapping indices to ``0..M-1``. **The input ``data`` is
    never mutated** — a fresh
    ``ConnectomeData`` is returned (AC1).

    Parameters
    ----------
    data:
        The loaded connectome to prune. Must carry ``superclass`` metadata.
    k:
        Corridor slack (``>= 0``). ``0`` = tight shortest-path corridor; larger values
        yield a monotone superset (AC5). Defaults to :data:`DEFAULT_PRUNE_K`.
    rule:
        Pruning rule; only :data:`PRUNE_RULE_PATH_SLACK` is supported.

    Raises
    ------
    ValueError
        If ``rule`` is unknown, ``k < 0``, ``superclass`` metadata is absent, the sensory
        or motor population is empty, or the rule yields an empty/degenerate subgraph.
    """
    if rule != PRUNE_RULE_PATH_SLACK:
        raise ValueError(
            f"Unknown pruning rule {rule!r}; the only supported rule is "
            f"{PRUNE_RULE_PATH_SLACK!r} (see drone_fly.connectome.prune.PRUNE_RULE_PATH_SLACK)."
        )
    if k < 0:
        raise ValueError(f"Pruning slack k must be >= 0, got {k}.")
    if data.superclass is None:
        raise ValueError(
            "Cannot prune: the connectome lacks per-neuron 'superclass' metadata, so the "
            "sensory (visual_projection) and motor (descending_neuron) populations cannot be "
            "identified. Provision a connectome whose *_meta.csv carries a 'superclass' column."
        )

    n = data.neuron_count
    superclass = np.asarray(data.superclass)
    sensory = _endpoints(superclass, SENSORY_SUPERCLASS)
    motor = _endpoints(superclass, MOTOR_SUPERCLASS)
    if sensory.size == 0:
        raise ValueError(
            f"Cannot prune: no sensory neurons found (superclass == {SENSORY_SUPERCLASS!r}). "
            f"The connectome has no identifiable sensory population to anchor the subcircuit."
        )
    if motor.size == 0:
        raise ValueError(
            f"Cannot prune: no motor neurons found (superclass == {MOTOR_SUPERCLASS!r}). "
            f"The connectome has no identifiable motor population to anchor the subcircuit."
        )

    # Direction-aware neighbour lists: forward = successors (CSC columns), backward =
    # predecessors (CSR rows). See the module docstring for the A[i, j] = j->i convention.
    csr = data.adjacency.tocsr()
    csc = data.adjacency.tocsc()

    d_fwd, parent = _bfs(csc.indptr, csc.indices, sensory, n, track_parent=True)
    d_bwd, _ = _bfs(csr.indptr, csr.indices, motor, n, track_parent=False)

    # Shortest sensory->motor path length L = min forward distance from a sensory neuron
    # to any motor (d_bwd measures hops-to-motor). If no sensory neuron can reach a motor,
    # the subcircuit is degenerate.
    reachable_sensory = d_bwd[sensory]
    reachable_sensory = reachable_sensory[reachable_sensory != -1]
    if reachable_sensory.size == 0:
        raise ValueError(
            "Cannot prune: no directed path exists from any sensory (visual_projection) neuron "
            "to any motor (descending_neuron) neuron in this connectome. The requested "
            "subcircuit is empty/degenerate."
        )
    L = int(reachable_sensory.min())

    retained: set[int] = set(int(s) for s in sensory)

    # Path inclusion: every sensory-reachable motor contributes one reconstructed shortest
    # sensory->motor path, so it stays reachable within the pruned subgraph for any k (AC2).
    dropped_motor: list[int] = []
    for m in motor:
        if d_fwd[m] == -1:
            dropped_motor.append(int(m))
            continue
        retained.update(_reconstruct_path(int(m), parent))
    if dropped_motor:
        logger.warning(
            "Pruning: %d motor neuron(s) are not reachable from any sensory neuron and were "
            "dropped from the subcircuit (e.g. indices %s).",
            len(dropped_motor),
            dropped_motor[:8],
        )

    # Corridor: nodes whose shortest sensory->node->motor path is within slack k of L.
    inf = n + 1
    df = np.where(d_fwd == -1, inf, d_fwd)
    db = np.where(d_bwd == -1, inf, d_bwd)
    corridor = np.nonzero(df + db <= L + k)[0]
    retained.update(int(u) for u in corridor)

    if not retained:
        raise ValueError(
            "Cannot prune: the path-slack rule produced an empty subgraph. Increase k or "
            "check that the connectome actually wires sensory to motor."
        )

    kept = np.array(sorted(retained), dtype=np.int64)
    if kept.size >= n:
        logger.warning(
            "Pruning retained all %d neurons (k=%d on a densely-connected graph); the subgraph "
            "was not reduced. This is expected on the small hub-slice fixture at larger k.",
            n,
            k,
        )

    # Submatrix + meta re-slice via the shared primitive: adjacency[kept][:, kept] keeps only
    # intra-retained edges and per-neuron meta is re-aligned to rows 0..M-1. Per-edge signs are
    # preserved because policy.py derives them from the presynaptic (column) neuron's sign, and
    # sign[kept] carries every retained neuron's sign (AC4).
    pruned_source = f"{data.source} [pruned:{rule} k={k}]"
    pruned = slice_connectome(data, kept, source=pruned_source)

    logger.info(
        "Pruned connectome '%s' (rule=%s, k=%d): %d -> %d neurons, %d -> %d edges "
        "(sensory=%d, motor kept=%d/%d).",
        getattr(data, "source", "?"),
        rule,
        k,
        n,
        int(kept.size),
        data.edge_count,
        int(pruned.edge_count),
        int(sensory.size),
        int(motor.size - len(dropped_motor)),
        int(motor.size),
    )

    return pruned
