"""UC-13 (AC2/AC6) — the modality → neuron-population selector.

:func:`drone_fly.controller.modality.select_modality` is the DISTINCT, fail-loud path
(contrast :mod:`drone_fly.controller.populations`, which degrades to a placeholder). It
maps a modality name to the neuron indices of that biological population in a given
connectome / slice, exposes cleanly-labelled modalities as authoritative and the
curated-name-list ones (``motion`` / ``hunger``) as ``approximate=True``, and RAISES when
the population is absent from the active slice (AC6) — never binding to nothing.

All fixture-based cases run on the committed real-MaleCNS slice; the approximate-flag and
metadata-degrade cases use tiny crafted connectomes so the assertions are exact.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.connectome.prune import prune_to_subcircuit
from drone_fly.controller.modality import (
    MODALITY_RULES,
    ModalityAbsentError,
    ModalityMetadataError,
    ModalitySelection,
    UnknownModalityError,
    available_modalities,
    select_modality,
)


def _craft(
    n: int,
    *,
    superclass=None,
    neuron_class=None,
    subclass=None,
    cell_type=None,
) -> ConnectomeData:
    """A tiny deterministic connectome carrying exactly the label columns given."""
    matrix = sp.random(n, n, density=0.3, format="csr", dtype=np.float32, random_state=0)
    return ConnectomeData(
        adjacency=matrix,
        neuron_ids=np.arange(n, dtype=np.int64),
        source="crafted",
        superclass=None if superclass is None else np.asarray(superclass, dtype=object),
        neuron_class=None if neuron_class is None else np.asarray(neuron_class, dtype=object),
        subclass=None if subclass is None else np.asarray(subclass, dtype=object),
        cell_type=None if cell_type is None else np.asarray(cell_type, dtype=object),
    )


# --- registry surface ------------------------------------------------------------------
def test_available_modalities_covers_clean_and_approximate() -> None:
    names = set(available_modalities())
    # Cleanly-labelled (per the use case): selectable, authoritative.
    assert {
        "vision",
        "proprioceptive",
        "mechanosensory",
        "gustatory",
        "olfactory",
        "thermosensory",
        "hygrosensory",
    } <= names
    # Approximate ones are exposed too, but flagged in the registry.
    assert {"motion", "hunger"} <= names
    assert MODALITY_RULES["motion"].approximate is True
    assert MODALITY_RULES["hunger"].approximate is True
    assert MODALITY_RULES["vision"].approximate is False


# --- clean selection on the real committed fixture (AC2) --------------------------------
def test_vision_selects_visual_projection(connectome: ConnectomeData) -> None:
    sel = select_modality(connectome, "vision")
    assert isinstance(sel, ModalitySelection)
    assert sel.approximate is False
    assert sel.indices.dtype == np.int64
    # Indices index visual_projection rows in the connectome's own row order.
    superclass = np.asarray(connectome.superclass)
    assert sel.indices.size > 0
    assert all(superclass[i] == "visual_projection" for i in sel.indices.tolist())
    # Sorted, unique, in range.
    assert np.array_equal(sel.indices, np.unique(sel.indices))
    assert sel.indices.min() >= 0 and sel.indices.max() < connectome.neuron_count


def test_proprioceptive_selects_mechanosensory_proprioceptive_class(
    connectome: ConnectomeData,
) -> None:
    sel = select_modality(connectome, "proprioceptive")
    assert sel.approximate is False
    neuron_class = np.asarray(connectome.neuron_class)
    assert sel.indices.size > 0
    assert all(neuron_class[i] == "mechanosensory_proprioceptive" for i in sel.indices.tolist())


def test_mechanosensory_prefix_superset_of_proprioceptive(connectome: ConnectomeData) -> None:
    mech = select_modality(connectome, "mechanosensory")
    prop = select_modality(connectome, "proprioceptive")
    # The mechanosensory prefix is a superset of the proprioceptive class.
    assert set(prop.indices.tolist()) <= set(mech.indices.tolist())


def test_indices_are_into_the_given_slice_row_order(connectome: ConnectomeData) -> None:
    """Selecting on a slice returns indices into THAT slice, not the parent graph (AC2)."""
    # k=2 keeps a genuinely-wired proprioceptive subset; select within the pruned graph.
    pruned = prune_to_subcircuit(connectome, k=2)
    sel = select_modality(pruned, "proprioceptive")
    neuron_class = np.asarray(pruned.neuron_class)
    assert sel.indices.size > 0
    assert sel.indices.max() < pruned.neuron_count
    assert all(neuron_class[i] == "mechanosensory_proprioceptive" for i in sel.indices.tolist())


# --- approximate (curated-name-list) modalities, flagged (AC2) --------------------------
def test_motion_is_flagged_approximate_and_matches_substring() -> None:
    data = _craft(4, subclass=["T4a", "unrelated", "T5b", "other"])
    sel = select_modality(data, "motion")
    assert sel.approximate is True
    assert sel.indices.tolist() == [0, 2]
    assert "approximate" in sel.rule.lower()


def test_hunger_is_flagged_approximate() -> None:
    # UC-17: hunger is bound over ``cell_type`` (where IPC/Hugin/NPF/insulin/DILP are labelled),
    # NOT ``subclass``. Craft the feeding population via cell_type tokens.
    data = _craft(4, cell_type=["IPC", "unrelated", "Hugin-RG", "NPFL1-I"])
    sel = select_modality(data, "hunger")
    assert sel.approximate is True
    assert sel.indices.tolist() == [0, 2, 3]


def test_hunger_ignores_subclass_now() -> None:
    """UC-17: the old ``subclass`` tokens no longer bind hunger — only ``cell_type`` does.

    A connectome whose feeding names live only in ``subclass`` (the pre-UC-17 layout) now has an
    absent hunger population, because the rule reads ``cell_type``.
    """
    data = _craft(3, cell_type=["x", "y", "z"], subclass=["feeding-related", "x", "npf-neuron"])
    with pytest.raises(ModalityAbsentError):
        select_modality(data, "hunger")


def test_hunger_resolves_on_regenerated_fixture(connectome: ConnectomeData) -> None:
    """UC-17 AC4/AC7: hunger resolves to the 22-neuron feeding population on the regenerated
    fixture (IPC×16 + Hugin-RG×4 + NPFL1-I×2), flagged approximate. Independently re-derived
    against the committed fixture's ``cell_type`` column."""
    sel = select_modality(connectome, "hunger")
    assert sel.approximate is True
    assert sel.indices.size == 22
    cell_type = np.asarray(connectome.cell_type)
    tokens = ("ipc", "hugin", "npf", "insulin", "dilp")
    for i in sel.indices.tolist():
        assert any(t in str(cell_type[i]).lower() for t in tokens)
    # Sorted, unique, in range.
    assert np.array_equal(sel.indices, np.unique(sel.indices))
    assert sel.indices.max() < connectome.neuron_count


# --- fail-loud cases on REAL data (AC6) ------------------------------------------------
def test_absent_class_raises_on_fixture(connectome: ConnectomeData) -> None:
    """Gustatory/olfactory/thermo/hygro are absent from the fixture -> raise (AC6)."""
    for modality in ("gustatory", "olfactory", "thermosensory", "hygrosensory"):
        with pytest.raises(ModalityAbsentError, match="absent from the active slice"):
            select_modality(connectome, modality)


def test_pruned_away_population_raises(connectome: ConnectomeData) -> None:
    """Proprioceptive neurons are dropped by the k=0 vision->motor prune -> raise (AC6).

    This is the load-bearing case: binding a proprioceptive block to a vision->motor slice
    that contains none of them must fail loudly, not bind to nothing.
    """
    pruned = prune_to_subcircuit(connectome, k=0)
    # Sanity: the prune really did remove the whole proprioceptive population.
    neuron_class = np.asarray(pruned.neuron_class)
    assert not (neuron_class == "mechanosensory_proprioceptive").any()
    with pytest.raises(ModalityAbsentError):
        select_modality(pruned, "proprioceptive")


def test_unknown_modality_raises() -> None:
    data = _craft(3, neuron_class=["a", "b", "c"])
    with pytest.raises(UnknownModalityError, match="Unknown modality"):
        select_modality(data, "sonar")


def test_missing_metadata_column_raises_distinct_error() -> None:
    """A connectome lacking the needed label column raises ModalityMetadataError, not Absent."""
    # No class column at all -> can't even ask whether proprioceptive is present.
    no_class = _craft(3, superclass=["visual_projection", "x", "y"])
    with pytest.raises(ModalityMetadataError, match="lacks the 'class'"):
        select_modality(no_class, "proprioceptive")
    # No superclass column -> vision (which reads superclass) also fails as metadata error.
    no_superclass = _craft(3, neuron_class=["a", "b", "c"])
    with pytest.raises(ModalityMetadataError, match="lacks the 'superclass'"):
        select_modality(no_superclass, "vision")


def test_never_returns_empty_selection() -> None:
    """When a population matches, the selection is non-empty; when not, it raises (never [])."""
    data = _craft(5, neuron_class=["mechanosensory_proprioceptive"] * 5)
    sel = select_modality(data, "proprioceptive")
    assert sel.indices.size == 5
