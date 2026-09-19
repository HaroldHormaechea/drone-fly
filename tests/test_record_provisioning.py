"""UC-27 — provision neuron positions at slice time (not on every training run).

Covers the position-provisioning orchestrator :func:`drone_fly.record.coordinates.resolve_positions`
and its sidecars — the single mechanism behind UC-27:

* **AC-4** — a valid ``<stem>_positions.csv`` is a fast-path HIT: **zero** spectral
  eigendecomposition, **zero** neuPrint fetch, **zero** anatomy resolution.
* **AC-5** — self-heal: a read-side run with no sidecar computes once, WARNs, persists the
  sidecar, and a second run fast-loads; a write failure is best-effort (WARN, never fatal).
* **AC-6 / AC-10** — anatomy precedence (meta ``somaLocation`` tier-0 → ``DRONE_FLY_SOMA_CSV`` →
  ``<stem>_soma.csv`` → neuPrint → spectral); real tokenless anatomy from the connectome meta CSV
  is the primary source and is persisted as ``<stem>_soma.csv``; soma-less neurons are flagged.
* **AC-7** — node-set binding: a sidecar is reused only when its ``bodyid`` set + ``projection``
  match exactly; otherwise MISS + recompute. Partial-anatomy fallback rows reload as ``None``.
* **AC-8** — a loaded sidecar equals a fresh compute **exactly** (bit-exact ``%.17g`` round-trip)
  across all three branches (anatomical, partial, spectral).
* **AC-9** — scope containment: a ``record: false`` run provisions nothing; the whole path is
  offline (no network).
* **AC-11** — the large-connectome guard: complete real anatomy provisions without a dense
  ``eigh`` at any scale; no/partial anatomy above the cap defers (WARN, real soma sidecar written,
  no ``_positions.csv``).

All hermetic: synthetic connectomes + a committed tier-0 meta fixture (``uc27_soma_meta.csv``),
no network, no token.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import drone_fly.record.coordinates as C
from drone_fly.connectome.loader import ConnectomeData
from drone_fly.record.coordinates import (
    DEFAULT_PROJECTION,
    SOMA_CSV_ENV,
    SOURCE_COMPUTED,
    SPECTRAL_MAX_ENV,
    _anatomy_coverage,
    _load_anatomical,
    _load_positions_sidecar,
    _positions_sidecar_path,
    _resolve_meta_soma,
    _sidecar_path,
    provision_positions,
    resolve_positions,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
UC27_META = FIXTURE_DIR / "uc27_soma_meta.csv"


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch):
    """No local-CSV override, no neuPrint token, default spectral cap for every test."""
    monkeypatch.delenv(SOMA_CSV_ENV, raising=False)
    monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
    monkeypatch.delenv(SPECTRAL_MAX_ENV, raising=False)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _make_data(
    neuron_ids: list[int],
    npz_path: Path,
    *,
    superclass: list[str] | None = None,
    seed: int = 0,
) -> ConnectomeData:
    """A tiny deterministic connectome whose ``source`` is ``npz_path`` (need not exist).

    ``source`` is an ``.npz`` path so the sidecar paths resolve beside it; the file itself is
    not read on the provisioning read path (the in-memory adjacency drives any spectral compute).
    """
    n = len(neuron_ids)
    rng = np.random.default_rng(seed)
    dense = rng.random((n, n)).astype(np.float32)
    dense[dense < 0.6] = 0.0
    return ConnectomeData(
        adjacency=sp.csr_matrix(dense),
        neuron_ids=np.asarray(neuron_ids, dtype=np.int64),
        source=str(npz_path),
        superclass=(None if superclass is None else np.asarray(superclass, dtype=object)),
    )


def _write_soma_sidecar(npz_path: Path, soma_map: dict[int, tuple[float, float, float]]) -> Path:
    """Write a ``bodyid,x,y,z`` soma sidecar beside ``npz_path`` (tier-2 anatomy)."""
    p = Path(npz_path)
    soma = p.with_name(f"{p.stem}_soma.csv")
    pd.DataFrame(
        [{"bodyid": b, "x": xyz[0], "y": xyz[1], "z": xyz[2]} for b, xyz in soma_map.items()]
    ).to_csv(soma, index=False)
    return soma


def _spy(monkeypatch: pytest.MonkeyPatch, name: str) -> list:
    """Wrap ``coordinates.<name>`` to count calls while preserving behaviour."""
    real = getattr(C, name)
    calls: list = []

    def wrapper(*a, **k):  # noqa: ANN002, ANN003
        calls.append((a, k))
        return real(*a, **k)

    monkeypatch.setattr(C, name, wrapper)
    return calls


def _assert_positions_equal(a: dict, b: dict) -> None:
    """Exact (bit-for-bit) equality of two positions dicts across all three branches."""
    assert a["source"] == b["source"]
    assert a["projection"] == b["projection"]
    assert a["has_position"] == b["has_position"]
    assert all(isinstance(h, bool) for h in a["has_position"])
    assert len(a["coords3d"]) == len(b["coords3d"])
    for ca, cb in zip(a["coords3d"], b["coords3d"], strict=True):
        if ca is None or cb is None:
            assert ca is None and cb is None
        else:
            assert list(ca) == list(cb)  # exact float equality (no allclose)
    assert a["coords2d"] == b["coords2d"]  # exact


# --------------------------------------------------------------------------- #
# AC-8 — a loaded sidecar reproduces a fresh compute EXACTLY (all three branches)
# --------------------------------------------------------------------------- #
def test_ac8_round_trip_anatomical_branch(tmp_path) -> None:
    """Full anatomy → loaded sidecar == fresh compute, exactly (real 3-D coords everywhere)."""
    npz = tmp_path / "conn.npz"
    ids = [10, 20, 30, 40]
    _write_soma_sidecar(npz, {i: (float(i), float(i * 2), float(i * 3)) for i in ids})
    data = _make_data(ids, npz)

    fresh = provision_positions(data)
    assert fresh["source"].lower().startswith("anatomical")
    assert all(c is not None for c in fresh["coords3d"])

    first = resolve_positions(data, persist=True)  # computes + writes
    cached = resolve_positions(data, persist=True)  # loads
    _assert_positions_equal(first, fresh)
    _assert_positions_equal(cached, fresh)


def test_ac8_round_trip_partial_branch(tmp_path) -> None:
    """Partial anatomy → soma-less rows are ``coords3d=None``, reloaded as ``None`` (AC-7)."""
    npz = tmp_path / "conn.npz"
    ids = [10, 20, 30, 40]
    _write_soma_sidecar(npz, {10: (1.0, 2.0, 3.0), 20: (4.0, 5.0, 6.0)})  # 30/40 soma-less
    data = _make_data(ids, npz)

    fresh = provision_positions(data)
    assert fresh["has_position"] == [True, True, False, False]
    assert fresh["coords3d"][2] is None and fresh["coords3d"][3] is None

    resolve_positions(data, persist=True)
    cached = resolve_positions(data, persist=True)
    _assert_positions_equal(cached, fresh)
    # The reloaded fallback rows are None (not NaN, not a fabricated coordinate).
    assert cached["coords3d"][2] is None and cached["coords3d"][3] is None
    assert cached["has_position"][2] is False and cached["has_position"][3] is False


def test_ac8_round_trip_spectral_branch(tmp_path) -> None:
    """No anatomy → spectral layout; coords3d NON-null but has_position all False; exact reload."""
    npz = tmp_path / "conn.npz"
    ids = [10, 20, 30, 40, 50]
    data = _make_data(ids, npz)  # no soma sidecar → spectral fallback

    fresh = provision_positions(data)
    assert fresh["source"] == SOURCE_COMPUTED
    assert fresh["has_position"] == [False] * 5
    # null-ness of coords3d is NOT derivable from has_position (spectral branch: all non-null).
    assert all(c is not None for c in fresh["coords3d"])

    resolve_positions(data, persist=True)
    cached = resolve_positions(data, persist=True)
    _assert_positions_equal(cached, fresh)
    # Reloaded coords3d stay non-null for every neuron despite has_position=False.
    assert all(c is not None for c in cached["coords3d"])


# --------------------------------------------------------------------------- #
# AC-4 — a valid sidecar is a fast-path HIT: zero compute / fetch / anatomy work
# --------------------------------------------------------------------------- #
def test_ac4_fast_path_does_zero_compute(tmp_path, monkeypatch) -> None:
    npz = tmp_path / "conn.npz"
    ids = [1, 2, 3, 4]
    _write_soma_sidecar(npz, {i: (float(i), float(i), float(i)) for i in ids})
    data = _make_data(ids, npz)
    resolve_positions(data, persist=True)  # prime the sidecar
    assert _positions_sidecar_path(data).is_file()

    spectral = _spy(monkeypatch, "_spectral_embedding")
    neuprint = _spy(monkeypatch, "_load_from_neuprint")
    anatomical = _spy(monkeypatch, "_load_anatomical")
    meta = _spy(monkeypatch, "_resolve_meta_soma")

    cached = resolve_positions(data, persist=True)

    assert cached is not None
    assert spectral == [] and neuprint == [] and anatomical == [] and meta == []


# --------------------------------------------------------------------------- #
# AC-5 — self-heal (compute once, WARN, persist; second run fast-loads; write best-effort)
# --------------------------------------------------------------------------- #
def test_ac5_self_heal_computes_warns_and_persists(tmp_path, monkeypatch, caplog) -> None:
    npz = tmp_path / "conn.npz"
    ids = [1, 2, 3, 4]
    _write_soma_sidecar(npz, {i: (float(i), float(i), float(i)) for i in ids})
    data = _make_data(ids, npz)
    pos_path = _positions_sidecar_path(data)
    assert not pos_path.is_file()

    with caplog.at_level(logging.WARNING, logger="drone_fly.record.coordinates"):
        first = resolve_positions(data, persist=True, persist_strict=False)
    assert first is not None
    assert pos_path.is_file()  # the sidecar self-healed
    assert any("on the fly" in r.getMessage() for r in caplog.records)

    # Second run: fast load, no compute, no self-heal warning.
    spectral = _spy(monkeypatch, "_spectral_embedding")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="drone_fly.record.coordinates"):
        second = resolve_positions(data, persist=True, persist_strict=False)
    assert second is not None
    assert spectral == []
    assert not any("on the fly" in r.getMessage() for r in caplog.records)


def test_ac5_write_failure_is_best_effort_not_fatal(tmp_path, caplog) -> None:
    """A sidecar write into a missing dir warns and returns the computed dict (never raises)."""
    npz = tmp_path / "does_not_exist" / "conn.npz"  # parent dir absent → to_csv OSError
    data = _make_data([1, 2, 3, 4], npz)  # no anatomy → spectral compute, then failed write

    with caplog.at_level(logging.WARNING, logger="drone_fly.record.coordinates"):
        result = resolve_positions(data, persist=True, persist_strict=False)

    assert result is not None  # computed in memory despite the write failing
    assert not npz.with_name("conn_positions.csv").exists()
    assert any(
        "not cached" in r.getMessage() or "Could not persist" in r.getMessage()
        for r in caplog.records
    )


def test_ac5_strict_write_failure_raises(tmp_path) -> None:
    """On a write hook (``persist_strict=True``) a write failure is fatal, not swallowed."""
    npz = tmp_path / "missing" / "conn.npz"
    data = _make_data([1, 2, 3, 4], npz)
    with pytest.raises(OSError):
        resolve_positions(data, persist=True, persist_strict=True)


# --------------------------------------------------------------------------- #
# AC-7 — node-set binding: HIT only on exact id-set + projection; else MISS + recompute
# --------------------------------------------------------------------------- #
def _prime_sidecar(tmp_path, ids, soma_map=None) -> tuple[ConnectomeData, Path]:
    npz = tmp_path / "conn.npz"
    if soma_map:
        _write_soma_sidecar(npz, soma_map)
    data = _make_data(ids, npz)
    resolve_positions(data, persist=True)
    return data, _positions_sidecar_path(data)


def test_ac7_hit_on_exact_match_and_realigns_row_order(tmp_path) -> None:
    ids = [11, 22, 33]
    data, pos_path = _prime_sidecar(tmp_path, ids, {i: (float(i), float(i), float(i)) for i in ids})
    # A permutation of the SAME id set is a HIT, realigned to the requested row order.
    permuted = _make_data([33, 11, 22], tmp_path / "conn.npz")
    loaded = _load_positions_sidecar(pos_path, permuted, DEFAULT_PROJECTION)
    assert loaded is not None
    # Row 0 now corresponds to bodyid 33 → its real coord is (33,33,33).
    assert loaded["coords3d"][0] == [33.0, 33.0, 33.0]
    assert loaded["coords3d"][1] == [11.0, 11.0, 11.0]


def test_ac7_miss_on_id_set_mismatch(tmp_path) -> None:
    ids = [11, 22, 33]
    _, pos_path = _prime_sidecar(tmp_path, ids, {i: (float(i),) * 3 for i in ids})
    # Same length, different member (44 instead of 33) → stale/wrong-subcircuit → MISS.
    other = _make_data([11, 22, 44], tmp_path / "conn.npz")
    assert _load_positions_sidecar(pos_path, other, DEFAULT_PROJECTION) is None


def test_ac7_miss_on_wrong_length(tmp_path) -> None:
    ids = [11, 22, 33]
    _, pos_path = _prime_sidecar(tmp_path, ids, {i: (float(i),) * 3 for i in ids})
    shorter = _make_data([11, 22], tmp_path / "conn.npz")
    assert _load_positions_sidecar(pos_path, shorter, DEFAULT_PROJECTION) is None


def test_ac7_miss_on_projection_mismatch(tmp_path) -> None:
    ids = [11, 22, 33]
    data, pos_path = _prime_sidecar(tmp_path, ids, {i: (float(i),) * 3 for i in ids})
    # Sidecar was written for the default plane; a different plane must MISS.
    other_plane = "xy" if DEFAULT_PROJECTION != "xy" else "yz"
    assert _load_positions_sidecar(pos_path, data, other_plane) is None


def test_ac7_miss_on_corrupt_sidecar(tmp_path) -> None:
    ids = [11, 22, 33]
    data, pos_path = _prime_sidecar(tmp_path, ids, {i: (float(i),) * 3 for i in ids})
    pos_path.write_text("not,a,valid\npositions,side,car\n", encoding="utf-8")  # missing columns
    assert _load_positions_sidecar(pos_path, data, DEFAULT_PROJECTION) is None


def test_ac7_mismatch_triggers_recompute_via_resolve(tmp_path) -> None:
    """End-to-end: a sidecar whose id-set differs is ignored and the run recomputes fresh."""
    npz = tmp_path / "conn.npz"
    ids = [11, 22, 33]
    _write_soma_sidecar(npz, {i: (float(i),) * 3 for i in ids})
    resolve_positions(_make_data(ids, npz), persist=True)  # writes a 3-neuron sidecar

    # A 4-neuron connectome at the SAME source must NOT reuse the 3-neuron sidecar.
    data4 = _make_data([11, 22, 33, 44], npz)
    _write_soma_sidecar(npz, {i: (float(i),) * 3 for i in [11, 22, 33, 44]})
    result = resolve_positions(data4, persist=True)
    assert result is not None and len(result["has_position"]) == 4


# --------------------------------------------------------------------------- #
# AC-6 / AC-10 — tier-0 real anatomy from the connectome meta ``somaLocation`` (tokenless)
# --------------------------------------------------------------------------- #
def _build_meta_connectome(tmp_path, stem: str = "conn") -> tuple[ConnectomeData, Path]:
    """A connectome whose sibling ``<stem>_meta.csv`` is the committed tier-0 fixture."""
    import shutil

    meta_dst = tmp_path / f"{stem}_meta.csv"
    shutil.copy(UC27_META, meta_dst)
    ids = [90001, 90002, 90003, 90004, 90005, 90006]  # 90005/90006 are soma-less
    npz = tmp_path / f"{stem}.npz"
    return _make_data(ids, npz), npz


def test_ac10_meta_soma_is_primary_tokenless_anatomy(tmp_path, monkeypatch, no_network) -> None:
    data, npz = _build_meta_connectome(tmp_path)

    neuprint = _spy(monkeypatch, "_load_from_neuprint")
    result = resolve_positions(data, persist=True)

    assert result is not None
    assert result["source"] == "anatomical (connectome meta somaLocation)"
    # Soma-bearing neurons get the real committed coordinates; soma-less are flagged, not faked.
    assert result["has_position"] == [True, True, True, True, False, False]
    assert result["coords3d"][0] == [1000.0, 2000.0, 3000.0]
    assert result["coords3d"][3] == [1300.0, 2300.0, 3300.0]
    assert result["coords3d"][4] is None and result["coords3d"][5] is None
    # Tokenless: neuPrint was never consulted.
    assert neuprint == []
    # The real anatomy was persisted as <stem>_soma.csv for offline reuse (AC-6).
    soma_path = _sidecar_path(data)
    assert soma_path.is_file()
    soma = pd.read_csv(soma_path)
    assert set(soma["bodyid"].tolist()) == {90001, 90002, 90003, 90004}


def test_ac10_resolve_meta_soma_subsets_to_node_set(tmp_path) -> None:
    """Tier-0 mapping is subset to the artifact's bodyids (the meta may be a superset)."""
    import shutil

    shutil.copy(UC27_META, tmp_path / "conn_meta.csv")
    npz = tmp_path / "conn.npz"
    # Ask for only 2 of the 6 meta bodyids → mapping restricted to those.
    data = _make_data([90002, 90005], npz)
    mapping = _resolve_meta_soma(data, None)
    assert set(mapping) == {90002}  # 90005 has no somaLocation
    assert mapping[90002] == (1100.0, 2100.0, 3100.0)


def test_ac10_meta_soma_write_side_uses_source_data(tmp_path) -> None:
    """Prune/prune-trained side: tier-0 reads the SOURCE meta and keys writes off the artifact."""
    import shutil

    # SOURCE connectome (carries the somaLocation meta).
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    shutil.copy(UC27_META, src_dir / "mcns_meta.csv")
    source = _make_data([90001, 90002, 90003, 90004, 90005, 90006], src_dir / "mcns.npz")

    # The pruned artifact: a subset, saved elsewhere, whose OWN meta has no somaLocation.
    art_npz = tmp_path / "out" / "connectome_pruned.npz"
    art_npz.parent.mkdir()
    pruned = _make_data([90001, 90003, 90005], art_npz)
    pruned.source = "mcns.npz [pruned: reverse-bfs k=0]"  # slice-source tag, NOT the artifact npz

    result = resolve_positions(pruned, artifact_npz=art_npz, source_data=source, persist=True)
    assert result is not None
    assert result["source"] == "anatomical (connectome meta somaLocation)"
    # Real coords for the two soma-bearing pruned bodyids; the soma-less one is flagged.
    assert result["has_position"] == [True, True, False]
    # Sidecars land beside the ARTIFACT, keyed off artifact_npz.
    assert (art_npz.parent / "connectome_pruned_soma.csv").is_file()
    assert (art_npz.parent / "connectome_pruned_positions.csv").is_file()
    soma = pd.read_csv(art_npz.parent / "connectome_pruned_soma.csv")
    assert set(soma["bodyid"].tolist()) == {90001, 90003}


# --------------------------------------------------------------------------- #
# AC-6 / AC-10 — precedence: meta override wins over env CSV; env CSV wins over sidecar
# --------------------------------------------------------------------------- #
def test_precedence_meta_override_beats_env_csv(tmp_path, monkeypatch) -> None:
    npz = tmp_path / "conn.npz"
    ids = [1, 2]
    # A valid env CSV that would otherwise be tier-1.
    env_csv = tmp_path / "env.csv"
    pd.DataFrame([{"bodyid": 1, "x": 9, "y": 9, "z": 9}]).to_csv(env_csv, index=False)
    monkeypatch.setenv(SOMA_CSV_ENV, str(env_csv))
    data = _make_data(ids, npz)

    override = {1: (1.0, 1.0, 1.0)}
    result = _load_anatomical(data, override=override)
    assert result is not None
    mapping, label = result
    assert label == "anatomical (connectome meta somaLocation)"
    assert mapping == override  # the meta override, not the env CSV


def test_precedence_env_csv_beats_sidecar(tmp_path, monkeypatch) -> None:
    npz = tmp_path / "conn.npz"
    ids = [1, 2]
    _write_soma_sidecar(npz, {1: (5.0, 5.0, 5.0), 2: (6.0, 6.0, 6.0)})  # tier-2 sidecar
    env_csv = tmp_path / "env.csv"
    pd.DataFrame([{"bodyid": 1, "x": 1, "y": 2, "z": 3}]).to_csv(env_csv, index=False)
    monkeypatch.setenv(SOMA_CSV_ENV, str(env_csv))
    data = _make_data(ids, npz)

    result = _load_anatomical(data, override=None)
    assert result is not None
    mapping, label = result
    assert "local CSV" in label  # env CSV won over the sidecar
    assert mapping == {1: (1.0, 2.0, 3.0)}


# --------------------------------------------------------------------------- #
# AC-11 / (d) — large-connectome guard (defer without eigh; complete anatomy always provisions)
# --------------------------------------------------------------------------- #
def test_ac11_no_anatomy_above_cap_defers_without_eigh(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(SPECTRAL_MAX_ENV, "3")  # cap below the connectome size
    npz = tmp_path / "conn.npz"
    data = _make_data([1, 2, 3, 4, 5, 6], npz)  # 6 > cap 3, no anatomy anywhere

    spectral = _spy(monkeypatch, "_spectral_embedding")
    result = resolve_positions(data, artifact_npz=npz, persist=True)

    assert result is None  # deferred
    assert spectral == []  # the giant eigendecomposition was never attempted
    assert not _positions_sidecar_path(data).is_file()  # no positions.csv written


def test_ac11_partial_anatomy_above_cap_writes_soma_but_defers_positions(
    tmp_path, monkeypatch, caplog
) -> None:
    monkeypatch.setenv(SPECTRAL_MAX_ENV, "3")
    data, npz = _build_meta_connectome(tmp_path)  # 6 neurons, only 4 soma-bearing → partial

    spectral = _spy(monkeypatch, "_spectral_embedding")
    with caplog.at_level(logging.WARNING, logger="drone_fly.record.coordinates"):
        result = resolve_positions(data, artifact_npz=npz, persist=True)

    assert result is None  # partial anatomy above the cap → defer
    assert spectral == []
    # Real soma sidecar for the covered subset WAS written; positions.csv was NOT.
    assert _sidecar_path(data).is_file()
    assert set(pd.read_csv(_sidecar_path(data))["bodyid"].tolist()) == {90001, 90002, 90003, 90004}
    assert not _positions_sidecar_path(data).is_file()
    assert any(
        "not computed" in r.getMessage().lower() and "soma sidecar" in r.getMessage().lower()
        for r in caplog.records
    )


def test_ac11_complete_anatomy_above_cap_provisions_without_eigh(
    tmp_path, monkeypatch, no_network
) -> None:
    """Complete real anatomy at any scale provisions with no dense ``eigh`` (the AC-11 point)."""
    import shutil

    monkeypatch.setenv(SPECTRAL_MAX_ENV, "2")
    shutil.copy(UC27_META, tmp_path / "conn_meta.csv")
    npz = tmp_path / "conn.npz"
    # Only the 4 soma-bearing bodyids → anatomy is COMPLETE for this node set, n(4) > cap(2).
    data = _make_data([90001, 90002, 90003, 90004], npz)
    assert _anatomy_coverage(data, override=_resolve_meta_soma(data, None)) == data.neuron_count

    spectral = _spy(monkeypatch, "_spectral_embedding")
    result = resolve_positions(data, artifact_npz=npz, persist=True)

    assert result is not None  # provisioned despite exceeding the cap
    assert spectral == []  # complete anatomy → no spectral layout needed
    assert result["has_position"] == [True, True, True, True]
    assert _positions_sidecar_path(data).is_file()


# --------------------------------------------------------------------------- #
# AC-9 — scope containment: a record-disabled run provisions nothing
# --------------------------------------------------------------------------- #
def test_ac9_record_disabled_run_provisions_nothing(fixture_dir_copy, tmp_path) -> None:
    """A default (``record: false``) smoke-train writes no positions sidecar by the connectome."""
    from drone_fly.connectome import load_connectome
    from drone_fly.train.config import TrainConfig
    from drone_fly.train.loop import smoke_train

    connectome = load_connectome(fixture_dir_copy)
    cfg = TrainConfig(
        models_dir=str(tmp_path / "models"),
        logs_dir=str(tmp_path / "logs"),
        checkpoint_freq=64,
        n_envs=1,
        n_steps=32,
        batch_size=16,
        seed=0,
    )
    smoke_train(connectome=connectome, cfg=cfg, timesteps=64)  # record defaults off
    # No ActivationRecorder was constructed → no positions sidecar provisioned.
    assert not (fixture_dir_copy / "mcns_fixture_positions.csv").is_file()
