"""AC1 + AC2 — offline loader reads the cached matrix at the expected scale.

* **AC1** — the loader returns a structure exposing neuron and synapse/edge counts;
  both are > 0 and match the documented expected scale of the committed fixture.
  The full-dataset scale constant is defined and positive (asserted here; the actual
  full-matrix load is an auto-skipped test that runs only when the multi-GB artifact
  is present).
* **AC2** — the load path performs no network I/O and never imports ``neuprint``; it
  succeeds with networking disabled.

UC-02 additions
---------------
The loader also surfaces optional per-neuron metadata (``superclass`` / ``sign`` /
``top_nt``) from the sidecar ``*_meta.csv`` when present, aligned to the matrix row
order, and leaves each attribute ``None`` when its column is absent (documented degrade,
never fabricated). These tests cover both the present and absent cases.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from drone_fly.connectome import (
    CONNECTOME_DIR_ENV,
    FIXTURE_EXPECTED_SCALE,
    MALECNS_V1_EXPECTED_SCALE,
    ConnectomeData,
    load_connectome,
)

# --- AC1: loader returns counts matching the documented fixture scale --------------


def test_load_returns_connectome_data(connectome: ConnectomeData) -> None:
    assert isinstance(connectome, ConnectomeData)
    assert connectome.adjacency.shape[0] == connectome.adjacency.shape[1]
    assert connectome.neuron_ids.shape[0] == connectome.neuron_count


def test_counts_are_positive(connectome: ConnectomeData) -> None:
    assert connectome.neuron_count > 0
    assert connectome.edge_count > 0
    # synapse_count is a documented alias of edge_count.
    assert connectome.synapse_count == connectome.edge_count


def test_counts_match_fixture_expected_scale(connectome: ConnectomeData) -> None:
    assert connectome.neuron_count == FIXTURE_EXPECTED_SCALE.neuron_count
    assert connectome.edge_count == FIXTURE_EXPECTED_SCALE.edge_count


# --- UC-02: optional per-neuron metadata surfaced from the sidecar meta CSV --------


def test_fixture_exposes_aligned_metadata(connectome: ConnectomeData) -> None:
    """The committed fixture carries real superclass/sign/top_nt, aligned to N."""
    n = connectome.neuron_count
    for attr in ("superclass", "sign", "top_nt"):
        values = getattr(connectome, attr)
        assert values is not None, f"{attr} should be present on the committed fixture"
        assert len(values) == n, f"{attr} not aligned to neuron count"


def test_fixture_sign_is_plus_minus_one(connectome: ConnectomeData) -> None:
    """``sign`` is a normalised int8 ±1 array (the sole source of E/I biology)."""
    assert connectome.sign.dtype == np.int8
    assert set(np.unique(connectome.sign).tolist()) == {-1, 1}


def test_fixture_superclass_has_expected_biological_labels(
    connectome: ConnectomeData,
) -> None:
    labels = set(np.asarray(connectome.superclass).tolist())
    # The populations the actor selects must be present in the fixture metadata.
    assert "descending_neuron" in labels
    assert "visual_projection" in labels


def _write_fixture(dir_path: Path, columns: dict[str, np.ndarray], n: int) -> None:
    """Write a tiny square .npz matrix + sidecar meta CSV with the given columns."""
    matrix = sp.random(n, n, density=0.2, format="csr", dtype=np.float32, random_state=0)
    sp.save_npz(dir_path / "mini.npz", matrix)
    frame = pd.DataFrame({"idx": np.arange(n), "bodyid": np.arange(1000, 1000 + n)})
    for name, values in columns.items():
        frame[name] = values
    frame.to_csv(dir_path / "mini_meta.csv", index=False)


def test_metadata_absent_leaves_fields_none(tmp_path: Path) -> None:
    """A meta CSV without the optional columns -> attributes are None, not fabricated."""
    _write_fixture(tmp_path, columns={}, n=20)
    data = load_connectome(tmp_path)
    assert data.superclass is None
    assert data.sign is None
    assert data.top_nt is None


def test_metadata_present_is_loaded_and_aligned(tmp_path: Path) -> None:
    n = 20
    superclass = np.array(["descending_neuron"] * n, dtype=object)
    sign = np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int64)
    top_nt = np.array(["gaba"] * n, dtype=object)
    _write_fixture(tmp_path, {"superclass": superclass, "sign": sign, "top_nt": top_nt}, n)

    data = load_connectome(tmp_path)
    assert data.superclass is not None and len(data.superclass) == n
    assert data.top_nt is not None and len(data.top_nt) == n
    # sign is normalised to int8 ±1.
    assert data.sign.dtype == np.int8
    np.testing.assert_array_equal(data.sign, sign.astype(np.int8))


# --- UC-13 (AC1): loader surfaces class/subclass without breaking superclass -------


def test_fixture_exposes_class_and_subclass(connectome: ConnectomeData) -> None:
    """The regenerated fixture carries real ``class``/``subclass`` aligned to N (AC1)."""
    n = connectome.neuron_count
    assert connectome.neuron_class is not None, "class must be surfaced (read from 'class' col)"
    assert connectome.subclass is not None, "subclass must be surfaced"
    assert len(connectome.neuron_class) == n
    assert len(connectome.subclass) == n
    # The proprioceptive afferent population the schema binds to must be present.
    classes = {str(x) for x in np.asarray(connectome.neuron_class).tolist()}
    assert "mechanosensory_proprioceptive" in classes


def test_surfacing_class_does_not_break_superclass_selection(connectome: ConnectomeData) -> None:
    """AC1 — adding class/subclass is purely additive: superclass selection is untouched."""
    from drone_fly.controller.populations import select_populations

    # superclass is still surfaced and still drives the existing population selection.
    assert connectome.superclass is not None
    (motor_idx, _), (sensory_idx, _) = select_populations(connectome)
    assert motor_idx.size > 0 and sensory_idx.size > 0


def test_class_subclass_absent_leaves_fields_none(tmp_path: Path) -> None:
    """A meta CSV without class/subclass -> attributes None (documented degrade, AC1)."""
    _write_fixture(tmp_path, columns={"superclass": np.array(["x"] * 20, dtype=object)}, n=20)
    data = load_connectome(tmp_path)
    assert data.neuron_class is None
    assert data.subclass is None
    # superclass still loads — the degrade is per-column.
    assert data.superclass is not None


def test_class_subclass_present_loaded_under_reserved_word_alias(tmp_path: Path) -> None:
    """The CSV 'class' column loads onto the ``neuron_class`` attribute (reserved-word alias)."""
    n = 12
    neuron_class = np.array(["mechanosensory_proprioceptive"] * n, dtype=object)
    subclass = np.array(["haltere"] * n, dtype=object)
    _write_fixture(tmp_path, {"class": neuron_class, "subclass": subclass}, n)

    data = load_connectome(tmp_path)
    assert data.neuron_class is not None and len(data.neuron_class) == n
    assert data.subclass is not None and len(data.subclass) == n
    np.testing.assert_array_equal(np.asarray(data.neuron_class), neuron_class)
    np.testing.assert_array_equal(np.asarray(data.subclass), subclass)


def test_class_subclass_save_load_round_trips_column_for_column(tmp_path: Path) -> None:
    """save_connectome writes neuron_class back as 'class'; load reproduces both (AC1)."""
    from drone_fly.connectome import save_connectome
    from drone_fly.connectome.loader import ConnectomeData

    n = 8
    matrix = sp.random(n, n, density=0.3, format="csr", dtype=np.float32, random_state=1)
    original = ConnectomeData(
        adjacency=matrix,
        neuron_ids=np.arange(2000, 2000 + n, dtype=np.int64),
        source="crafted",
        superclass=np.array(["visual_projection"] * n, dtype=object),
        neuron_class=np.array(["olfactory"] * n, dtype=object),
        subclass=np.array(["ORN"] * n, dtype=object),
    )
    save_connectome(original, tmp_path)
    reloaded = load_connectome(tmp_path)

    np.testing.assert_array_equal(np.asarray(reloaded.neuron_class), original.neuron_class)
    np.testing.assert_array_equal(np.asarray(reloaded.subclass), original.subclass)
    np.testing.assert_array_equal(np.asarray(reloaded.superclass), original.superclass)
    np.testing.assert_array_equal(np.asarray(reloaded.neuron_ids), original.neuron_ids)


def test_cell_type_exposed_on_regenerated_fixture(connectome) -> None:
    """UC-17: the loader exposes the ``cell_type`` column (where hunger/feeding neurons are
    labelled). Independently checked against the committed fixture."""
    assert connectome.cell_type is not None
    ct = np.asarray(connectome.cell_type)
    assert ct.shape[0] == connectome.neuron_count
    # The hunger quota's labels are present.
    labels = {str(x) for x in ct.tolist()}
    assert {"IPC", "Hugin-RG", "NPFL1-I"} <= labels


def test_cell_type_save_load_round_trips_column_for_column(tmp_path: Path) -> None:
    """UC-17: save_connectome writes ``cell_type`` back explicitly; load reproduces it (else the
    ``hunger`` binding would stop resolving on a pruned graft graph)."""
    from drone_fly.connectome import save_connectome
    from drone_fly.connectome.loader import ConnectomeData

    n = 8
    matrix = sp.random(n, n, density=0.3, format="csr", dtype=np.float32, random_state=1)
    cell_type = np.array(["IPC", "Hugin-RG", "NPFL1-I", "x", "y", "z", "IPC", "w"], dtype=object)
    original = ConnectomeData(
        adjacency=matrix,
        neuron_ids=np.arange(2000, 2000 + n, dtype=np.int64),
        source="crafted",
        superclass=np.array(["visual_projection"] * n, dtype=object),
        neuron_class=np.array(["olfactory"] * n, dtype=object),
        subclass=np.array(["ORN"] * n, dtype=object),
        cell_type=cell_type,
    )
    save_connectome(original, tmp_path)
    reloaded = load_connectome(tmp_path)

    assert reloaded.cell_type is not None
    np.testing.assert_array_equal(np.asarray(reloaded.cell_type), cell_type)


def test_prune_save_reload_preserves_cell_type_and_hunger(connectome, tmp_path: Path) -> None:
    """UC-17 (challenger fold): prune→save→reload must keep ``cell_type`` so ``hunger`` still
    resolves on the pruned graft graph. This is the load-bearing round-trip for the graft path.

    ``slice_connectome`` re-aligns ``cell_type`` to the kept rows (like the other meta columns);
    ``save_connectome`` persists it and the loader reloads it — so hunger keeps resolving after a
    prune→save→reload cycle instead of raising ``ModalityMetadataError`` on the pruned graph."""
    from drone_fly.connectome import prune_to_subcircuit, save_connectome
    from drone_fly.controller.modality import select_modality

    pruned = prune_to_subcircuit(connectome, k=2)
    assert pruned.cell_type is not None
    save_connectome(pruned, tmp_path)
    reloaded = load_connectome(tmp_path)

    assert reloaded.cell_type is not None
    np.testing.assert_array_equal(np.asarray(reloaded.cell_type), np.asarray(pruned.cell_type))
    # hunger resolves identically before and after the round-trip (does not vanish).
    before = select_modality(pruned, "hunger").indices
    after = select_modality(reloaded, "hunger").indices
    np.testing.assert_array_equal(before, after)
    assert after.size > 0


def test_malecns_full_scale_constant_defined_and_positive() -> None:
    assert MALECNS_V1_EXPECTED_SCALE.neuron_count > 0
    assert MALECNS_V1_EXPECTED_SCALE.edge_count > 0
    # The fixture is a small stand-in; the full dataset must be strictly larger.
    assert MALECNS_V1_EXPECTED_SCALE.neuron_count > FIXTURE_EXPECTED_SCALE.neuron_count
    assert MALECNS_V1_EXPECTED_SCALE.edge_count > FIXTURE_EXPECTED_SCALE.edge_count


def test_load_via_env_var(fixture_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader resolves the artifact dir from DRONE_FLY_CONNECTOME_DIR."""
    monkeypatch.setenv(CONNECTOME_DIR_ENV, str(fixture_dir))
    data = load_connectome()
    assert data.neuron_count == FIXTURE_EXPECTED_SCALE.neuron_count


# --- AC1 (full scale): auto-skipped unless the multi-GB artifact is present ---------


def _full_dataset_dir() -> Path | None:
    """Return a dir holding the full MaleCNS matrix if one is configured, else None."""
    env = os.environ.get("DRONE_FLY_FULL_CONNECTOME_DIR")
    if env and Path(env).exists():
        return Path(env)
    return None


@pytest.mark.skipif(
    _full_dataset_dir() is None,
    reason="Full multi-GB MaleCNS artifact not present (set DRONE_FLY_FULL_CONNECTOME_DIR).",
)
def test_full_dataset_matches_malecns_scale() -> None:
    data = load_connectome(_full_dataset_dir())
    assert data.neuron_count == MALECNS_V1_EXPECTED_SCALE.neuron_count
    assert data.edge_count == MALECNS_V1_EXPECTED_SCALE.edge_count


# --- AC2: no network, no neuprint --------------------------------------------------


def test_load_succeeds_with_networking_disabled(fixture_dir: Path, no_network: None) -> None:
    """Loading works even when all socket access raises (AC2)."""
    data = load_connectome(fixture_dir)
    assert data.neuron_count == FIXTURE_EXPECTED_SCALE.neuron_count


def test_load_does_not_import_neuprint(fixture_dir: Path, no_network: None) -> None:
    """The offline load path must never pull in the neuprint client."""
    # Drop any pre-imported copy so we detect an import triggered *by* the load.
    for mod in list(sys.modules):
        if mod == "neuprint" or mod.startswith("neuprint."):
            del sys.modules[mod]
    load_connectome(fixture_dir)
    assert "neuprint" not in sys.modules


def test_missing_artifact_raises_filenotfound(tmp_path: Path) -> None:
    """An absent artifact raises a clear error and never silently hits the network."""
    with pytest.raises(FileNotFoundError):
        load_connectome(tmp_path)
