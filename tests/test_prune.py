"""UC-04 — function-targeted connectome subgraph pruning.

Verifies :func:`drone_fly.connectome.prune.prune_to_subcircuit` and the reusable
export path (:func:`drone_fly.connectome.save_connectome` round-trip) against the 11
acceptance criteria. All tests are hermetic: they run on the committed real-MaleCNS
fixture (``tests/fixtures/``) or on tiny hand-built toy graphs — no network, no full
matrix (AC10).

Direction convention (loader contract): ``adjacency[i, j]`` is the weight from neuron
``j`` onto neuron ``i`` — a directed edge ``u -> v`` has ``row = v``, ``col = u``.
Successors are read from CSC columns, predecessors from CSR rows.

Verified fixture reduction (path-inclusion rule), asserted by AC7:

======  =========  =======
k       neurons    edges
======  =========  =======
full    300        10600
0       59         751
1       175        5988
2       274        10070
======  =========  =======
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import pytest
import scipy.sparse as sp

from drone_fly.connectome import (
    DEFAULT_PRUNE_K,
    DEFAULT_PRUNE_RULE,
    PRUNE_RULE_PATH_SLACK,
    load_connectome,
    prune_to_subcircuit,
    save_connectome,
)
from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.populations import (
    MOTOR_SUPERCLASS,
    SENSORY_SUPERCLASS,
    select_populations,
)

# Expected fixture reduction under the path-inclusion rule (AC7). Independently
# reproduced against the committed fixture (see module docstring).
FIXTURE_PRUNE_SCALE = {
    0: (59, 751),
    1: (175, 5988),
    2: (274, 10070),
}

# A neutral (neither sensory nor motor) superclass label used to pad toy graphs.
NEUTRAL_SUPERCLASS = "cb_intrinsic"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _build_toy(
    n: int,
    edges: list[tuple[int, int]],
    superclass: list[str],
    *,
    ids: np.ndarray | None = None,
    with_sign: bool = True,
) -> ConnectomeData:
    """Build a tiny :class:`ConnectomeData` from a directed edge list.

    Each ``(u, v)`` in ``edges`` is a directed edge ``u -> v`` and is placed at
    ``adjacency[v, u]`` per the loader's ``A[i, j] = j -> i`` convention.
    """
    rows = [v for (_u, v) in edges]
    cols = [u for (u, _v) in edges]
    weights = [1.0] * len(edges)
    adjacency = sp.csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float32)
    neuron_ids = np.arange(1000, 1000 + n, dtype=np.int64) if ids is None else np.asarray(ids)
    return ConnectomeData(
        adjacency=adjacency,
        neuron_ids=neuron_ids,
        source="toy-synthetic",
        superclass=np.array(superclass, dtype=object),
        sign=(np.ones(n, dtype=np.int8) if with_sign else None),
        top_nt=np.array(["acetylcholine"] * n, dtype=object),
    )


def _bfs_reachable_from_sensory(data: ConnectomeData) -> np.ndarray:
    """BFS over successors (CSC columns) from every sensory neuron in ``data``.

    Returns the per-neuron hop distance (``-1`` == unreachable), computed *within the
    given graph* — used to prove reachability holds in the pruned subgraph itself.
    """
    sc = np.asarray(data.superclass)
    sensory = np.nonzero(sc == SENSORY_SUPERCLASS)[0]
    csc = data.adjacency.tocsc()
    dist = np.full(data.neuron_count, -1, dtype=np.int64)
    queue: deque[int] = deque()
    for s in sensory:
        s = int(s)
        if dist[s] == -1:
            dist[s] = 0
            queue.append(s)
    while queue:
        u = queue.popleft()
        for i in range(csc.indptr[u], csc.indptr[u + 1]):
            v = int(csc.indices[i])
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def _ids(data: ConnectomeData) -> set[int]:
    return set(int(x) for x in np.asarray(data.neuron_ids).tolist())


# --------------------------------------------------------------------------- #
# AC1 — pure function, input not mutated, new object returned
# --------------------------------------------------------------------------- #
def test_ac1_input_not_mutated(connectome) -> None:
    """prune returns a *new* ConnectomeData; the input is byte-identical afterwards."""
    before_n = connectome.neuron_count
    before_e = connectome.edge_count
    before_nnz = connectome.adjacency.nnz
    before_adj = connectome.adjacency.copy()
    before_ids = np.asarray(connectome.neuron_ids).copy()

    pruned = prune_to_subcircuit(connectome, k=0)

    assert pruned is not connectome
    assert pruned.adjacency is not connectome.adjacency
    # Input untouched.
    assert connectome.neuron_count == before_n
    assert connectome.edge_count == before_e
    assert connectome.adjacency.nnz == before_nnz
    assert (connectome.adjacency != before_adj).nnz == 0
    assert np.array_equal(np.asarray(connectome.neuron_ids), before_ids)
    # Result is genuinely smaller.
    assert pruned.neuron_count < before_n


def test_ac1_signature_defaults() -> None:
    """The documented defaults are the path-slack rule and DEFAULT_PRUNE_K."""
    assert DEFAULT_PRUNE_RULE == PRUNE_RULE_PATH_SLACK
    assert DEFAULT_PRUNE_K == 2


# --------------------------------------------------------------------------- #
# AC2 — population retention + reachability within the pruned subgraph
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [0, 1, 2])
def test_ac2_all_retained_motor_reachable_in_pruned_graph(connectome, k) -> None:
    """Every retained motor is reachable from a sensory neuron *within* the pruned graph."""
    pruned = prune_to_subcircuit(connectome, k=k)
    sc = np.asarray(pruned.superclass)
    sensory = np.nonzero(sc == SENSORY_SUPERCLASS)[0]
    motor = np.nonzero(sc == MOTOR_SUPERCLASS)[0]

    # Populations are present in the pruned graph.
    assert sensory.size > 0
    assert motor.size > 0
    # Every visual_projection / descending_neuron of the full fixture is retained.
    full_sc = np.asarray(connectome.superclass)
    assert sensory.size == int(np.sum(full_sc == SENSORY_SUPERCLASS))
    assert motor.size == int(np.sum(full_sc == MOTOR_SUPERCLASS))

    dist = _bfs_reachable_from_sensory(pruned)
    unreachable = int(np.sum(dist[motor] == -1))
    assert unreachable == 0, f"{unreachable} retained motor neurons unreachable at k={k}"


# --------------------------------------------------------------------------- #
# AC3 — direction-correctness (reverse-only nodes pruned)
# --------------------------------------------------------------------------- #
def test_ac3_direction_correct_reverse_only_pruned() -> None:
    """A forward sensory->motor path is kept; upstream/downstream-only nodes are pruned.

    Toy graph (u -> v):
      * 0 (sensory) -> 1 -> 2 (motor)     : the forward corridor
      * 3 -> 2                            : node feeding INTO motor only (upstream-only)
      * 2 -> 4                            : node fed FROM motor only (downstream-only)
    If edges were treated as undirected, 3 and 4 would attach to the pathway and be kept.
    Direction-correct pruning drops both (neither lies on a directed sensory->motor path).
    """
    edges = [(0, 1), (1, 2), (3, 2), (2, 4)]
    sc = [
        SENSORY_SUPERCLASS,
        NEUTRAL_SUPERCLASS,
        MOTOR_SUPERCLASS,
        NEUTRAL_SUPERCLASS,
        NEUTRAL_SUPERCLASS,
    ]
    toy = _build_toy(5, edges, sc)  # ids 1000..1004

    pruned = prune_to_subcircuit(toy, k=0)
    kept = _ids(pruned)

    assert {1000, 1001, 1002} <= kept  # forward corridor retained
    assert 1003 not in kept  # upstream-only node (3 -> motor) pruned
    assert 1004 not in kept  # downstream-only node (motor -> 4) pruned

    # Reachability holds directionally in the pruned graph.
    dist = _bfs_reachable_from_sensory(pruned)
    pid = np.asarray(pruned.neuron_ids)
    motor_row = int(np.nonzero(pid == 1002)[0][0])
    assert dist[motor_row] != -1


# --------------------------------------------------------------------------- #
# AC4 — meta / index integrity, per-edge sign preservation, population compat
# --------------------------------------------------------------------------- #
def test_ac4_meta_lengths_and_alignment(connectome) -> None:
    """All per-neuron meta arrays are length M and re-aligned to the pruned rows."""
    pruned = prune_to_subcircuit(connectome, k=1)
    m = pruned.neuron_count
    assert len(np.asarray(pruned.neuron_ids)) == m
    assert len(np.asarray(pruned.superclass)) == m
    assert len(np.asarray(pruned.sign)) == m
    assert len(np.asarray(pruned.top_nt)) == m

    # Each pruned neuron's meta matches the original neuron with the same bodyid.
    ids = np.asarray(connectome.neuron_ids).tolist()
    orig_id_to_sc = dict(zip(ids, np.asarray(connectome.superclass).tolist(), strict=True))
    orig_id_to_sign = dict(zip(ids, np.asarray(connectome.sign).tolist(), strict=True))
    orig_id_to_nt = dict(zip(ids, np.asarray(connectome.top_nt).tolist(), strict=True))
    for i, bodyid in enumerate(np.asarray(pruned.neuron_ids).tolist()):
        assert pruned.superclass[i] == orig_id_to_sc[bodyid]
        assert int(pruned.sign[i]) == int(orig_id_to_sign[bodyid])
        assert pruned.top_nt[i] == orig_id_to_nt[bodyid]


def test_ac4_per_edge_sign_preserved(connectome) -> None:
    """Per-edge sign (presynaptic/column-derived) survives the index remap.

    policy.py derives each edge's E/I sign from its presynaptic (column) neuron:
    ``edge_sign = sign[coo.col]``. After pruning, the sign read for each retained edge
    must equal the original sign of that same presynaptic neuron (identified by bodyid).
    """
    pruned = prune_to_subcircuit(connectome, k=2)
    orig_id_to_sign = dict(
        zip(
            np.asarray(connectome.neuron_ids).tolist(),
            np.asarray(connectome.sign).tolist(),
            strict=True,
        )
    )

    coo = pruned.adjacency.tocoo()
    presyn_ids = np.asarray(pruned.neuron_ids)[coo.col]
    expected = np.array([orig_id_to_sign[int(b)] for b in presyn_ids], dtype=np.int8)
    actual = np.asarray(pruned.sign)[coo.col].astype(np.int8)
    assert np.array_equal(actual, expected)
    assert np.asarray(pruned.sign).dtype == np.int8


def test_ac4_population_selection_still_works_on_pruned(connectome) -> None:
    """UC-02 population selection finds biological descending/visual pops on the pruned graph."""
    pruned = prune_to_subcircuit(connectome, k=0)
    (motor_idx, motor_mode), (sensory_idx, sensory_mode) = select_populations(pruned)
    assert motor_mode == "biological"
    assert sensory_mode == "biological"
    # Capped population sizes (MOTOR_POP_SIZE=16, SENSORY_POP_SIZE=32) on the fixture.
    assert motor_idx.size == 16
    assert sensory_idx.size == 32
    # Sets are disjoint.
    assert set(motor_idx.tolist()).isdisjoint(sensory_idx.tolist())


# --------------------------------------------------------------------------- #
# AC5 — configurable rule, monotone superset in k
# --------------------------------------------------------------------------- #
def test_ac5_monotone_superset_in_k(connectome) -> None:
    """kept(k) is a subset of kept(k+1): a larger corridor yields a superset."""
    ids0 = _ids(prune_to_subcircuit(connectome, k=0))
    ids1 = _ids(prune_to_subcircuit(connectome, k=1))
    ids2 = _ids(prune_to_subcircuit(connectome, k=2))
    assert ids0 <= ids1 <= ids2
    # Strictly growing on this fixture (documents the reduction dynamics).
    assert len(ids0) < len(ids1) < len(ids2)


# --------------------------------------------------------------------------- #
# AC6 — determinism
# --------------------------------------------------------------------------- #
def test_ac6_deterministic(connectome) -> None:
    """Same input + params -> byte-identical pruned graph (indices/indptr/data + ids)."""
    a = prune_to_subcircuit(connectome, k=2)
    b = prune_to_subcircuit(connectome, k=2)
    aa = a.adjacency.tocsr()
    bb = b.adjacency.tocsr()
    assert np.array_equal(aa.indices, bb.indices)
    assert np.array_equal(aa.indptr, bb.indptr)
    assert np.array_equal(aa.data, bb.data)
    assert np.array_equal(np.asarray(a.neuron_ids), np.asarray(b.neuron_ids))


# --------------------------------------------------------------------------- #
# AC7 — reduction reporting (returned counts + logged line)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [0, 1, 2])
def test_ac7_returned_counts_match_expected(connectome, k) -> None:
    pruned = prune_to_subcircuit(connectome, k=k)
    exp_neurons, exp_edges = FIXTURE_PRUNE_SCALE[k]
    assert pruned.neuron_count == exp_neurons
    assert pruned.edge_count == exp_edges


def test_ac7_logs_input_to_pruned_counts(connectome, caplog) -> None:
    """The prune step logs the input->pruned neuron/edge counts (AC7)."""
    with caplog.at_level(logging.INFO, logger="drone_fly.connectome.prune"):
        prune_to_subcircuit(connectome, k=0)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "300" in msgs and "59" in msgs  # input -> pruned neuron counts
    assert "10600" in msgs and "751" in msgs  # input -> pruned edge counts


# --------------------------------------------------------------------------- #
# AC8 — pipeline wiring / back-compat
# --------------------------------------------------------------------------- #
def test_ac8_train_with_prune_smoke(connectome, tmp_path) -> None:
    """train(..., prune=True) trains on the pruned subcircuit end-to-end (hermetic)."""
    from drone_fly.train.config import TrainConfig
    from drone_fly.train.loop import smoke_train

    cfg = TrainConfig(
        models_dir=str(tmp_path / "models"),
        logs_dir=str(tmp_path / "logs"),
        checkpoint_freq=64,
        n_envs=1,
        n_steps=64,
        batch_size=32,
        seed=0,
    )
    model = smoke_train(connectome=connectome, cfg=cfg, timesteps=128, prune=True, prune_k=1)
    assert model.num_timesteps == 128


def test_ac8_default_is_backcompatible(connectome) -> None:
    """Default (no prune) leaves the connectome full — UC-01/02/03 behaviour unchanged."""
    # The pruning entry point is strictly opt-in; the fixture itself is untouched at 300/10600.
    assert connectome.neuron_count == 300
    assert connectome.edge_count == 10600


def test_ac8_prune_is_noop_under_resume(connectome, tmp_path, caplog) -> None:
    """--prune is skipped (with a warning) on a resume run; the checkpoint's graph wins."""
    from drone_fly.train.config import TrainConfig
    from drone_fly.train.loop import find_latest_checkpoint, smoke_train, train

    cfg = TrainConfig(
        models_dir=str(tmp_path / "models"),
        logs_dir=str(tmp_path / "logs"),
        checkpoint_freq=64,
        n_envs=1,
        n_steps=64,
        batch_size=32,
        seed=0,
    )
    smoke_train(connectome=connectome, cfg=cfg, timesteps=128)
    ckpt = find_latest_checkpoint(cfg.models_dir)
    assert ckpt is not None

    with caplog.at_level(logging.WARNING, logger="drone_fly.train.loop"):
        train(
            cfg,
            connectome=connectome,
            adapter="simple",
            device="cpu",
            resume=ckpt,
            total_timesteps=64,
            prune=True,
            prune_k=1,
        )
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "Ignoring --prune on a resume run" in msgs


# --------------------------------------------------------------------------- #
# AC9 — degrade / error handling
# --------------------------------------------------------------------------- #
def test_ac9_missing_superclass_errors(synthetic_connectome) -> None:
    with pytest.raises(ValueError, match="superclass"):
        prune_to_subcircuit(synthetic_connectome, k=0)


def test_ac9_no_sensory_population_errors() -> None:
    edges = [(0, 1)]
    sc = [MOTOR_SUPERCLASS, NEUTRAL_SUPERCLASS]  # no visual_projection
    toy = _build_toy(2, edges, sc)
    with pytest.raises(ValueError, match="sensory"):
        prune_to_subcircuit(toy, k=0)


def test_ac9_no_motor_population_errors() -> None:
    edges = [(0, 1)]
    sc = [SENSORY_SUPERCLASS, NEUTRAL_SUPERCLASS]  # no descending_neuron
    toy = _build_toy(2, edges, sc)
    with pytest.raises(ValueError, match="motor"):
        prune_to_subcircuit(toy, k=0)


def test_ac9_degenerate_no_path_errors() -> None:
    """Sensory and motor exist but no directed sensory->motor path -> clear error."""
    # motor -> sensory only (wrong direction); no forward sensory->motor path.
    edges = [(1, 0)]
    sc = [SENSORY_SUPERCLASS, MOTOR_SUPERCLASS]
    toy = _build_toy(2, edges, sc)
    with pytest.raises(ValueError, match="degenerate|no directed path"):
        prune_to_subcircuit(toy, k=0)


def test_ac9_negative_k_errors(connectome) -> None:
    with pytest.raises(ValueError, match="k"):
        prune_to_subcircuit(connectome, k=-1)


def test_ac9_unknown_rule_errors(connectome) -> None:
    with pytest.raises(ValueError, match="rule|Unknown"):
        prune_to_subcircuit(connectome, k=0, rule="bogus_rule")


# --------------------------------------------------------------------------- #
# AC10 — hermetic (offline) pruning
# --------------------------------------------------------------------------- #
def test_ac10_prune_is_offline(connectome, no_network) -> None:
    """Pruning performs no network I/O (runs with sockets disabled)."""
    pruned = prune_to_subcircuit(connectome, k=0)
    assert pruned.neuron_count == 59
    assert pruned.neuron_count < connectome.neuron_count


# --------------------------------------------------------------------------- #
# AC11 — export the pruned slice to disk + round-trip
# --------------------------------------------------------------------------- #
def test_ac11_export_roundtrip(connectome, tmp_path) -> None:
    """prune -> save_connectome -> load_connectome reproduces the pruned graph exactly."""
    pruned = prune_to_subcircuit(connectome, k=0)
    npz_path, meta_path = save_connectome(pruned, tmp_path)

    assert npz_path.is_file()
    assert meta_path.is_file()
    assert npz_path.name == "connectome_pruned.npz"
    assert meta_path.name == "connectome_pruned_meta.csv"

    reloaded = load_connectome(tmp_path)
    assert reloaded.neuron_count == 59
    assert reloaded.edge_count == 751
    assert np.array_equal(np.asarray(reloaded.neuron_ids), np.asarray(pruned.neuron_ids))
    assert np.array_equal(np.asarray(reloaded.superclass), np.asarray(pruned.superclass))
    assert np.array_equal(np.asarray(reloaded.sign), np.asarray(pruned.sign))
    assert np.array_equal(np.asarray(reloaded.top_nt), np.asarray(pruned.top_nt))
    assert (pruned.adjacency != reloaded.adjacency).nnz == 0


def test_ac11_save_to_file_path_errors(connectome, tmp_path) -> None:
    """save_connectome to a path that is a regular file raises NotADirectoryError."""
    pruned = prune_to_subcircuit(connectome, k=0)
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("i am a file")
    with pytest.raises(NotADirectoryError):
        save_connectome(pruned, file_path)


def test_ac11_exported_slice_is_reusable_meta_present(connectome, tmp_path) -> None:
    """The exported slice reloads with populations intact (reuse via train --connectome)."""
    pruned = prune_to_subcircuit(connectome, k=0)
    save_connectome(pruned, tmp_path)
    reloaded = load_connectome(tmp_path)
    (motor_idx, motor_mode), (sensory_idx, sensory_mode) = select_populations(reloaded)
    assert motor_mode == "biological" and sensory_mode == "biological"
    assert motor_idx.size == 16 and sensory_idx.size == 32


# --------------------------------------------------------------------------- #
# REQUIRED synthetic long-chain (fixture has L=1; cannot cover a far motor)
# --------------------------------------------------------------------------- #
def test_synthetic_long_chain_far_motor_retained_and_reachable() -> None:
    """A far motor (d_fwd=5) with L=1 is retained + reachable; junk nodes are pruned.

    Toy graph (u -> v), ids 1000..1010:
      * long chain   0 -> 1 -> 2 -> 3 -> 4 -> 5   (0 sensory, 5 motor at d_fwd=5)
      * short direct 6 -> 7                       (6 sensory, 7 motor => L=1)
      * isolated     8                            (no edges)
      * pure-upstream 9 -> 0                      (feeds sensory only; not sensory-reachable)
      * reverse-only 5 -> 10                      (downstream of far motor; reaches no motor)
    At k=2 the corridor is {d_fwd+d_bwd <= 3}, which excludes the far motor (5+0=5); it is
    retained only because path-inclusion keeps one shortest sensory->motor path per motor.
    """
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (9, 0), (5, 10)]
    sc = [
        SENSORY_SUPERCLASS,  # 0
        NEUTRAL_SUPERCLASS,  # 1
        NEUTRAL_SUPERCLASS,  # 2
        NEUTRAL_SUPERCLASS,  # 3
        NEUTRAL_SUPERCLASS,  # 4
        MOTOR_SUPERCLASS,  # 5  far motor
        SENSORY_SUPERCLASS,  # 6
        MOTOR_SUPERCLASS,  # 7  near motor (L=1)
        NEUTRAL_SUPERCLASS,  # 8  isolated
        NEUTRAL_SUPERCLASS,  # 9  pure-upstream
        NEUTRAL_SUPERCLASS,  # 10 reverse-only
    ]
    toy = _build_toy(11, edges, sc)
    pruned = prune_to_subcircuit(toy, k=2)
    kept = _ids(pruned)

    # Far motor retained despite falling outside the k=2 corridor band.
    assert 1005 in kept
    # Whole shortest sensory->far-motor chain retained.
    assert {1000, 1001, 1002, 1003, 1004, 1005} <= kept
    # Junk nodes pruned.
    assert 1008 not in kept  # isolated
    assert 1009 not in kept  # pure-upstream
    assert 1010 not in kept  # reverse-only (downstream of motor)

    # Far motor is reachable from a sensory neuron *within* the pruned subgraph.
    dist = _bfs_reachable_from_sensory(pruned)
    pid = np.asarray(pruned.neuron_ids)
    far_motor_row = int(np.nonzero(pid == 1005)[0][0])
    assert dist[far_motor_row] == 5
