"""AC6/AC7/AC9 — soma-coordinate provisioning + role labelling for the viewer.

Covers :func:`drone_fly.record.coordinates.provision_positions` and
:func:`~drone_fly.record.coordinates.neuron_roles`:

* **AC6/AC7** — the committed fixture ships **real** anatomical soma positions
  (tokenless ``somaLocation`` sidecar): source labelled ``anatomical …``,
  ``has_position`` all ``True``, 300/300 coverage, coords finite + deterministic.
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


# --- AC6/AC7 anatomical coverage on the committed fixture ------------------------------
def test_fixture_positions_are_real_anatomical_full_coverage(connectome: ConnectomeData) -> None:
    """The committed fixture sidecar yields 300/300 real soma positions (AC6/AC7)."""
    pos = provision_positions(connectome)

    assert pos["source"].lower().startswith("anatomical")
    assert pos["projection"] == DEFAULT_PROJECTION
    n = connectome.neuron_count

    assert len(pos["has_position"]) == n
    assert all(pos["has_position"]), "every fixture neuron must have a real soma position"
    assert sum(pos["has_position"]) == n == 300

    coords3d = pos["coords3d"]
    assert len(coords3d) == n
    assert all(c is not None for c in coords3d), "no anatomical neuron may have null coords3d"
    arr3 = np.asarray(coords3d, dtype=float)
    assert arr3.shape == (n, 3)
    assert np.isfinite(arr3).all()

    coords2d = np.asarray(pos["coords2d"], dtype=float)
    assert coords2d.shape == (n, 2)
    assert np.isfinite(coords2d).all()


def test_fixture_positions_are_deterministic(connectome: ConnectomeData) -> None:
    """Provisioning the same connectome twice gives identical coordinates (AC6)."""
    a = provision_positions(connectome)
    b = provision_positions(connectome)
    assert np.array_equal(np.asarray(a["coords2d"]), np.asarray(b["coords2d"]))
    assert np.array_equal(
        np.asarray(a["coords3d"], dtype=float), np.asarray(b["coords3d"], dtype=float)
    )
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
        arr3 = np.asarray(pos["coords3d"], dtype=float)
        arr2 = np.asarray(pos["coords2d"], dtype=float)
        # All fixture neurons are anatomical, so 2-D is exactly the selected axis pair.
        assert np.allclose(arr2[:, 0], arr3[:, a0])
        assert np.allclose(arr2[:, 1], arr3[:, a1])
