"""AC6/AC7/AC9 — soma-coordinate provisioning + role labelling for the viewer.

Covers :func:`drone_fly.record.coordinates.provision_positions` and
:func:`~drone_fly.record.coordinates.neuron_roles`:

* **AC6/AC7** — the committed fixture ships **real** anatomical soma positions
  (tokenless ``somaLocation`` sidecar): source labelled ``anatomical …``, coords finite +
  deterministic. Per UC-13 (Option B) the fixture now has **partial** soma coverage by
  construction — the soma-populated central core carries real coordinates while the real
  ``mechanosensory_proprioceptive`` afferent quota has no brain-volume soma. Coverage is
  asserted as an **exact partition** (``has_position[i]`` iff a soma exists for neuron ``i``),
  the stronger successor to UC-05's old 100%-coverage assertion (recorded decision #3).
* **AC6** — a neuron absent from the sidecar is **flagged** (``has_position=False``,
  ``coords3d=None``) and fallback-placed (``coords2d`` still finite) — never dropped
  and never fabricated with an anatomical value.
* **AC6** — with no anatomy reachable at all the layout is the deterministic computed
  spectral fallback, clearly labelled *not anatomical*, finite and reproducible.
* **AC9** — roles group neurons sensory (``visual_projection``) / motor
  (``descending_neuron``) / interneuron, aligned to the matrix row order.

All hermetic: the fixture sidecar is committed, no network / token used (the neuPrint
path degrades to the fallback without ``NEUPRINT_TOKEN``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.populations import MOTOR_SUPERCLASS, SENSORY_SUPERCLASS
from drone_fly.record.coordinates import (
    DEFAULT_PROJECTION,
    PROJECTIONS,
    SOMA_CSV_ENV,
    SOURCE_COMPUTED,
    neuron_roles,
    provision_positions,
)


@pytest.fixture(autouse=True)
def _no_token_no_override(monkeypatch: pytest.MonkeyPatch):
    """Keep provisioning hermetic: no local-CSV override, no neuPrint token."""
    monkeypatch.delenv(SOMA_CSV_ENV, raising=False)
    monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)


def _make_connectome(
    neuron_ids: list[int], source: str, *, superclass: list[str] | None = None
) -> ConnectomeData:
    """A tiny deterministic connectome with a chosen ``source`` and ``neuron_ids``."""
    n = len(neuron_ids)
    rng = np.random.default_rng(0)
    dense = rng.random((n, n)).astype(np.float32)
    dense[dense < 0.6] = 0.0
    return ConnectomeData(
        adjacency=sp.csr_matrix(dense),
        neuron_ids=np.asarray(neuron_ids, dtype=np.int64),
        source=source,
        superclass=(None if superclass is None else np.asarray(superclass, dtype=object)),
    )


# --- AC6/AC7 anatomical coverage on the committed fixture (UC-13 exact partition) ------
def test_fixture_positions_are_exact_soma_partition(
    connectome: ConnectomeData, fixture_dir: Path
) -> None:
    """The regenerated fixture has an EXACT soma partition, not 100% coverage (UC-13, B).

    Real sensory afferents (the ``mechanosensory_proprioceptive`` quota) have no
    brain-volume soma, so coverage is partial by construction. This asserts the exact
    partition (as strong as, not a relaxation of, the old 322/322 assertion; recorded
    decision #3):

    * the number of soma-populated neurons equals the committed sidecar's row count (the
      central core);
    * ``has_position[i]`` is ``True`` **iff** a soma exists for neuron ``i`` — for EVERY
      neuron (the load-bearing identity);
    * every proprioceptive-quota afferent is flagged soma-less (``coords3d`` ``None``),
      never dropped and never fabricated;
    * every soma-populated neuron carries a real, finite, deterministic 3-D coordinate.
    """
    import pandas as pd

    pos = provision_positions(connectome)
    n = connectome.neuron_count

    # Ground truth: a neuron has a real position iff its bodyid is in the committed soma CSV.
    soma_ids = {
        int(b) for b in pd.read_csv(fixture_dir / "mcns_fixture_soma.csv")["bodyid"].tolist()
    }
    ids = np.asarray(connectome.neuron_ids)
    expected_has = [int(i) in soma_ids for i in ids.tolist()]
    core_size = sum(expected_has)

    assert pos["source"].lower().startswith("anatomical")
    assert pos["projection"] == DEFAULT_PROJECTION

    # (1) exact count: soma-populated core == sidecar rows (partial, < n by construction).
    assert len(pos["has_position"]) == n
    assert sum(pos["has_position"]) == core_size == len(soma_ids)
    assert core_size < n, "UC-13 fixture must have partial (not full) soma coverage"

    # (2) identity: has_position[i] iff a soma exists for neuron i — for every neuron.
    assert pos["has_position"] == expected_has

    # (3) every mechanosensory_proprioceptive afferent is flagged soma-less.
    neuron_class = [str(x) for x in np.asarray(connectome.neuron_class).tolist()]
    proprio = [i for i, c in enumerate(neuron_class) if c == "mechanosensory_proprioceptive"]
    assert proprio, "fixture must contain the proprioceptive afferent quota"
    assert all(pos["has_position"][i] is False for i in proprio)
    assert all(pos["coords3d"][i] is None for i in proprio)

    # (4) soma-populated neurons: real, finite 3-D coords; soma-less: coords3d None (no fake).
    coords3d = pos["coords3d"]
    assert len(coords3d) == n
    for i in range(n):
        if expected_has[i]:
            assert coords3d[i] is not None
            arr = np.asarray(coords3d[i], dtype=float)
            assert arr.shape == (3,)
            assert np.isfinite(arr).all()
        else:
            assert coords3d[i] is None

    # 2-D fallback keeps EVERY neuron drawable (finite), soma-less included.
    coords2d = np.asarray(pos["coords2d"], dtype=float)
    assert coords2d.shape == (n, 2)
    assert np.isfinite(coords2d).all()


def test_fixture_positions_are_deterministic(connectome: ConnectomeData) -> None:
    """Provisioning the same connectome twice gives identical coordinates (AC6)."""
    a = provision_positions(connectome)
    b = provision_positions(connectome)
    assert np.array_equal(np.asarray(a["coords2d"]), np.asarray(b["coords2d"]))
    assert a["has_position"] == b["has_position"]
    # coords3d carries None for soma-less afferents; compare element-wise (a plain
    # np.asarray(..., dtype=float) would choke on the ragged None/[x,y,z] mix).
    assert len(a["coords3d"]) == len(b["coords3d"])
    for ca, cb in zip(a["coords3d"], b["coords3d"], strict=True):
        if ca is None or cb is None:
            assert ca is None and cb is None
        else:
            assert np.array_equal(np.asarray(ca, dtype=float), np.asarray(cb, dtype=float))
    assert a["source"] == b["source"]


# --- AC6 missing-neuron handling (flag + fallback-place, never drop/fabricate) ---------
def test_missing_neuron_is_flagged_not_dropped(
    connectome: ConnectomeData, fixture_dir: Path
) -> None:
    """A neuron absent from the sidecar is flagged + fallback-placed, never dropped."""
    real_ids = [int(x) for x in np.asarray(connectome.neuron_ids)[:5]]
    fake_id = int(max(np.asarray(connectome.neuron_ids))) + 10_000  # guaranteed off-sidecar
    source = str(fixture_dir / "mcns_fixture.npz")  # so the fixture sidecar resolves
    data = _make_connectome([*real_ids, fake_id], source)

    pos = provision_positions(data)

    assert pos["source"].lower().startswith("anatomical")  # 5/6 real -> still anatomical
    assert len(pos["has_position"]) == 6  # nothing dropped
    assert pos["has_position"][:5] == [True] * 5
    assert pos["has_position"][5] is False

    # The missing neuron carries no fabricated 3-D anatomy...
    assert pos["coords3d"][5] is None
    assert all(c is not None for c in pos["coords3d"][:5])
    # ...but is still placed with a finite 2-D fallback so the viewer can draw it.
    coords2d = np.asarray(pos["coords2d"], dtype=float)
    assert coords2d.shape == (6, 2)
    assert np.isfinite(coords2d).all()


# --- AC6 computed spectral fallback (no anatomy reachable) -----------------------------
def test_spectral_fallback_labelled_and_finite(synthetic_connectome: ConnectomeData) -> None:
    """With no sidecar/token, all neurons get the labelled computed layout (AC6)."""
    pos = provision_positions(synthetic_connectome)

    assert pos["source"] == SOURCE_COMPUTED
    assert "not anatomical" in pos["source"].lower()
    n = synthetic_connectome.neuron_count
    assert pos["has_position"] == [False] * n  # nothing masquerades as anatomy

    coords2d = np.asarray(pos["coords2d"], dtype=float)
    assert coords2d.shape == (n, 2)
    assert np.isfinite(coords2d).all()
    coords3d = np.asarray(pos["coords3d"], dtype=float)
    assert coords3d.shape == (n, 3)
    assert np.isfinite(coords3d).all()


def test_spectral_fallback_is_deterministic(synthetic_connectome: ConnectomeData) -> None:
    """The spectral fallback is sign-canonicalised → bit-reproducible within a process."""
    a = provision_positions(synthetic_connectome)
    b = provision_positions(synthetic_connectome)
    assert np.array_equal(np.asarray(a["coords2d"]), np.asarray(b["coords2d"]))
    assert np.array_equal(np.asarray(a["coords3d"]), np.asarray(b["coords3d"]))


# --- AC9 role labelling ----------------------------------------------------------------
def test_roles_map_superclass_to_sensory_motor_interneuron(connectome: ConnectomeData) -> None:
    """visual_projection→sensory, descending_neuron→motor, else interneuron (AC9)."""
    roles = neuron_roles(connectome)
    assert len(roles) == connectome.neuron_count
    assert set(roles) <= {"sensory", "interneuron", "motor"}

    superclass = np.asarray(connectome.superclass).tolist()
    for role, sc in zip(roles, superclass, strict=True):
        if sc == SENSORY_SUPERCLASS:
            assert role == "sensory"
        elif sc == MOTOR_SUPERCLASS:
            assert role == "motor"
        else:
            assert role == "interneuron"
    # The fixture actually carries all three role classes (structure is visible).
    assert {"sensory", "interneuron", "motor"} <= set(roles)


def test_roles_degrade_to_interneuron_without_superclass(
    synthetic_connectome: ConnectomeData,
) -> None:
    """Metadata-less connectome → every neuron labelled interneuron (no fabricated biology)."""
    roles = neuron_roles(synthetic_connectome)
    assert roles == ["interneuron"] * synthetic_connectome.neuron_count


# --- projection plane handling ---------------------------------------------------------
def test_unknown_projection_raises(connectome: ConnectomeData) -> None:
    with pytest.raises(ValueError, match="projection"):
        provision_positions(connectome, projection="zzz")


def test_projection_selects_documented_axes(connectome: ConnectomeData) -> None:
    """coords2d for a plane equals the corresponding pair of 3-D axes (AC6 projection doc)."""
    for plane, (a0, a1) in PROJECTIONS.items():
        pos = provision_positions(connectome, projection=plane)
        assert pos["projection"] == plane
        # Only soma-populated (anatomical) neurons project straight from their 3-D coords;
        # the soma-less afferents get a finite 2-D fallback (asserted in the partition test).
        anat = [i for i, h in enumerate(pos["has_position"]) if h]
        assert anat, "fixture must contain anatomical neurons"
        arr3 = np.asarray([pos["coords3d"][i] for i in anat], dtype=float)
        arr2 = np.asarray(pos["coords2d"], dtype=float)[anat]
        assert np.allclose(arr2[:, 0], arr3[:, a0])
        assert np.allclose(arr2[:, 1], arr3[:, a1])
