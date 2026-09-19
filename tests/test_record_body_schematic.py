"""UC-28 AC-3/AC-4/AC-5/AC-8/AC-7 — body-schematic placement of soma-less afferents.

UC-27 renders real-soma neurons at true anatomy; UC-28 completes coverage by placing the
genuinely soma-less peripheral afferents in a deterministic **schematic fly body around the
brain**, grouped by their real categorical body-region label. This module exercises that
placement in isolation with a small **synthetic** connectome that — unlike the committed
fixture (which only carries ``leg`` / ``haltere`` / ``campaniform`` soma-less afferents) —
covers **all six** :data:`REGION_LABEL_RULES` branches, including the ``vnc`` / ``ascending``
(``superclass``) and blank→``torso`` (AC-4) cases the fixture cannot reach.

Coverage:

* **AC-3** — deterministic body-region placement: each soma-less afferent lands inside its
  region's cluster (offset from the brain bbox); the layout is **bit-reproducible across
  processes** with a different ``PYTHONHASHSEED`` (proving the placement uses ``zlib.crc32``,
  never the salted builtin :func:`hash`).
* **AC-3 (precedence)** — a fine-grained limb ``subclass`` wins over the coarse ``superclass``.
* **AC-4** — a blank / unrecognised label maps to the neutral ``torso`` group, never a limb.
* **AC-5** — the brain-scale guardrail keeps the real brain bbox ≥
  :data:`BRAIN_DOMINANCE_MIN_FRACTION` of the total rendered extent, in **3-D** and on the
  **default projection plane**.
* **AC-8** — a coordinate set mixing real (anatomical) and schematic entries renders with no
  NaN/None display coords and keeps every array index-aligned (the positional
  activation↔neuron join stays exact).
* **AC-7** — the recorder's per-neuron ``meta.modality`` tag set is exactly
  ``{vision, proprioceptive, hunger} ∪ {""}`` (the fixed RECORDED_MODALITIES set + untagged).

All hermetic: no network, no token, no committed-fixture writes (the synthetic connectomes are
built in-memory; the modality recorder writes only into ``tmp_path``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.record.coordinates import (
    BRAIN_DOMINANCE_MIN_FRACTION,
    DEFAULT_PROJECTION,
    PLACEMENT_ANATOMICAL,
    PLACEMENT_SCHEMATIC,
    REGION_ASCENDING,
    REGION_CAMPANIFORM,
    REGION_CLUSTER_OFFSETS,
    REGION_CLUSTER_RADIUS_FRAC,
    REGION_HALTERE,
    REGION_LEG,
    REGION_TORSO,
    REGION_VNC,
    provision_positions,
    resolve_body_region,
)
from drone_fly.record.recorder import RECORDED_MODALITIES, ActivationRecorder

_REPO_ROOT = Path(__file__).resolve().parents[1]

# --- synthetic connectome exercising every region branch --------------------------------------
# Brain neurons: real soma coordinates supplied via ``anatomy_override`` (tier-0). Their spread
# defines a non-degenerate brain bounding box (min (0,0,0), max (10,10,10)) that anchors the
# schematic body clusters.
_BRAIN: tuple[tuple[int, tuple[float, float, float]], ...] = (
    (1, (0.0, 0.0, 0.0)),
    (2, (10.0, 0.0, 0.0)),
    (3, (0.0, 10.0, 0.0)),
    (4, (0.0, 0.0, 10.0)),
    (5, (10.0, 10.0, 10.0)),
)

#: Soma-less afferents: ``(bodyid, subclass, superclass, expected_region)``. Covers all six
#: REGION_LABEL_RULES outcomes — leg/haltere/campaniform via ``subclass``, vnc/ascending via
#: ``superclass``, blank→torso (AC-4) — plus a precedence case (limb subclass over superclass).
_REGION: tuple[tuple[int, str, str, str], ...] = (
    (101, "leg", "", REGION_LEG),
    (102, "haltere", "", REGION_HALTERE),
    (103, "campaniform_sensilla", "", REGION_CAMPANIFORM),  # prefix match
    (104, "", "vnc_sensory", REGION_VNC),
    (105, "", "sensory_ascending", REGION_ASCENDING),
    (106, "", "", REGION_TORSO),  # blank/unrecognised → torso (AC-4)
    (107, "leg", "vnc_sensory", REGION_LEG),  # limb subclass wins over superclass (precedence)
)

#: Tier-0 anatomy override: real soma xyz for the brain neurons only.
ANATOMY_OVERRIDE: dict[int, tuple[float, float, float]] = {bid: xyz for bid, xyz in _BRAIN}


def build_region_connectome() -> ConnectomeData:
    """A small synthetic connectome covering every region branch (importable by subprocesses).

    Module-level + deterministic so a child process (see the cross-process determinism test)
    reconstructs a byte-identical connectome and must produce identical schematic coordinates.
    """
    ids = [bid for bid, _ in _BRAIN] + [row[0] for row in _REGION]
    subclass = [""] * len(_BRAIN) + [row[1] for row in _REGION]
    superclass = [""] * len(_BRAIN) + [row[2] for row in _REGION]
    n = len(ids)
    rng = np.random.default_rng(0)
    dense = rng.random((n, n)).astype(np.float32)
    dense[dense < 0.6] = 0.0
    return ConnectomeData(
        adjacency=sp.csr_matrix(dense),
        neuron_ids=np.asarray(ids, dtype=np.int64),
        source="synthetic-uc28-region",
        superclass=np.asarray(superclass, dtype=object),
        subclass=np.asarray(subclass, dtype=object),
    )


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch):
    """No local-CSV override, no neuPrint token — keep provisioning offline/deterministic."""
    monkeypatch.delenv("DRONE_FLY_SOMA_CSV", raising=False)
    monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)


def _brain_bbox() -> tuple[np.ndarray, np.ndarray]:
    """(center, extent) of the real-soma brain bounding box from the override."""
    real = np.array(list(ANATOMY_OVERRIDE.values()), dtype=float)
    lo, hi = real.min(axis=0), real.max(axis=0)
    return (lo + hi) / 2.0, (hi - lo)


def _cluster_center(region: str) -> np.ndarray:
    center, extent = _brain_bbox()
    return center + np.asarray(REGION_CLUSTER_OFFSETS[region], dtype=float) * extent


# --- AC-3 region labelling covers all six branches --------------------------------------------
def test_region_labels_cover_all_six_branches() -> None:
    """resolve_body_region assigns each afferent its expected region; all six labels appear."""
    data = build_region_connectome()
    regions = resolve_body_region(data)
    assert len(regions) == data.neuron_count

    offset = len(_BRAIN)
    for slot, (bodyid, _, _, expected) in enumerate(_REGION):
        got = regions[offset + slot]
        assert got == expected, f"neuron {bodyid} expected region {expected!r}, got {got!r}"

    # Every one of the six documented region labels is exercised by this fixture.
    assert {
        REGION_LEG,
        REGION_HALTERE,
        REGION_CAMPANIFORM,
        REGION_VNC,
        REGION_ASCENDING,
        REGION_TORSO,
    } <= set(regions)


def test_limb_subclass_precedence_over_superclass() -> None:
    """AC-3 precedence: a limb ``subclass`` (leg) wins over a coarse ``superclass`` (vnc)."""
    data = build_region_connectome()
    regions = resolve_body_region(data)
    # Neuron 107 carries subclass="leg" AND superclass="vnc_sensory" → leg (first rule wins).
    idx = [i for i, b in enumerate(np.asarray(data.neuron_ids).tolist()) if int(b) == 107][0]
    assert regions[idx] == REGION_LEG


# --- AC-3 deterministic cluster placement -----------------------------------------------------
def test_schematic_afferents_land_in_their_region_cluster() -> None:
    """AC-3: each soma-less afferent's display3d lands inside its region cluster (bounded)."""
    data = build_region_connectome()
    pos = provision_positions(data, anatomy_override=ANATOMY_OVERRIDE)
    display3d = np.asarray(pos["display3d"], dtype=float)
    _, extent = _brain_bbox()
    # Max per-axis spread from the cluster centre is REGION_CLUSTER_RADIUS_FRAC * brain_extent.
    tol = REGION_CLUSTER_RADIUS_FRAC * extent + 1e-9

    offset = len(_BRAIN)
    for slot, (bodyid, _, _, region) in enumerate(_REGION):
        i = offset + slot
        assert pos["placement"][i] == PLACEMENT_SCHEMATIC
        assert pos["region"][i] == region
        assert pos["coords3d"][i] is None  # no fabricated anatomy (honesty)
        center = _cluster_center(region)
        assert np.all(np.abs(display3d[i] - center) <= tol), (
            f"neuron {bodyid} ({region}) at {display3d[i]} not within its cluster {center} ± {tol}"
        )

    # Brain (real-soma) neurons keep anatomical placement + their true coords as display coords.
    for slot, (_bodyid, xyz) in enumerate(_BRAIN):
        assert pos["placement"][slot] == PLACEMENT_ANATOMICAL
        assert pos["region"][slot] == ""
        assert np.allclose(display3d[slot], np.asarray(xyz, dtype=float))


def test_schematic_placement_is_deterministic_in_process() -> None:
    """AC-3: provisioning the same connectome twice yields identical schematic display coords."""
    data = build_region_connectome()
    a = provision_positions(data, anatomy_override=ANATOMY_OVERRIDE)
    b = provision_positions(data, anatomy_override=ANATOMY_OVERRIDE)
    assert np.array_equal(
        np.asarray(a["display3d"], dtype=float), np.asarray(b["display3d"], dtype=float)
    )


def test_schematic_placement_is_reproducible_across_processes() -> None:
    """AC-3: schematic coords are bit-identical across processes with different PYTHONHASHSEED.

    This is the load-bearing guarantee that the within-cluster spread uses ``zlib.crc32``
    (fixed polynomial) and NOT the builtin :func:`hash` (salted per process by ``PYTHONHASHSEED``
    for ``str`` inputs, which would scatter the layout run-to-run).
    """
    child = textwrap.dedent(
        """
        import json, sys
        from tests.test_record_body_schematic import build_region_connectome, ANATOMY_OVERRIDE
        from drone_fly.record.coordinates import provision_positions
        pos = provision_positions(build_region_connectome(), anatomy_override=ANATOMY_OVERRIDE)
        json.dump(pos["display3d"], sys.stdout)
        """
    )

    def run(seed: str) -> list[list[float]]:
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", child],
            env=env,
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"child (seed={seed}) failed:\n{result.stderr}"
        return json.loads(result.stdout)

    local = provision_positions(build_region_connectome(), anatomy_override=ANATOMY_OVERRIDE)[
        "display3d"
    ]
    seed_a = run("0")
    seed_b = run("123456789")

    assert np.array_equal(np.asarray(seed_a, dtype=float), np.asarray(seed_b, dtype=float)), (
        "schematic layout changed across PYTHONHASHSEED — placement must use crc32, not hash()"
    )
    assert np.array_equal(np.asarray(seed_a, dtype=float), np.asarray(local, dtype=float))


# --- AC-4 generic/unknown label → torso, never a limb -----------------------------------------
def test_blank_label_lands_in_torso_not_a_limb() -> None:
    """AC-4: the blank-label afferent is closest to the torso cluster, not any limb cluster."""
    data = build_region_connectome()
    pos = provision_positions(data, anatomy_override=ANATOMY_OVERRIDE)
    display3d = np.asarray(pos["display3d"], dtype=float)

    idx = [i for i, b in enumerate(np.asarray(data.neuron_ids).tolist()) if int(b) == 106][0]
    assert pos["region"][idx] == REGION_TORSO
    point = display3d[idx]

    centers = {region: _cluster_center(region) for region in REGION_CLUSTER_OFFSETS}
    nearest = min(centers, key=lambda r: float(np.linalg.norm(point - centers[r])))
    assert nearest == REGION_TORSO, f"blank-label afferent nearest {nearest!r}, expected torso"


# --- AC-5 brain-scale guardrail (3-D + default projection plane) ------------------------------
def test_brain_dominance_guardrail_3d_and_plane() -> None:
    """AC-5: the real brain bbox stays ≥ BRAIN_DOMINANCE_MIN_FRACTION of the total extent."""
    data = build_region_connectome()
    pos = provision_positions(data, anatomy_override=ANATOMY_OVERRIDE)
    display3d = np.asarray(pos["display3d"], dtype=float)
    has_position = pos["has_position"]

    brain = display3d[[i for i, ok in enumerate(has_position) if ok]]
    assert brain.shape[0] == len(_BRAIN)

    # (a) 3-D: per-axis brain extent vs total rendered extent.
    total_extent = display3d.max(axis=0) - display3d.min(axis=0)
    brain_extent = brain.max(axis=0) - brain.min(axis=0)
    fractions_3d = brain_extent / total_extent
    assert np.all(fractions_3d >= BRAIN_DOMINANCE_MIN_FRACTION), (
        f"brain bbox drops below the guardrail in 3-D: fractions {fractions_3d.tolist()}"
    )

    # (b) default projection plane: same guarantee on the two projected axes.
    assert pos["projection"] == DEFAULT_PROJECTION
    coords2d = np.asarray(pos["coords2d"], dtype=float)
    brain2d = coords2d[[i for i, ok in enumerate(has_position) if ok]]
    total2d = coords2d.max(axis=0) - coords2d.min(axis=0)
    brain2d_extent = brain2d.max(axis=0) - brain2d.min(axis=0)
    fractions_2d = brain2d_extent / total2d
    assert np.all(fractions_2d >= BRAIN_DOMINANCE_MIN_FRACTION), (
        f"brain bbox drops below the guardrail on the {DEFAULT_PROJECTION} plane: "
        f"{fractions_2d.tolist()}"
    )


# --- AC-8 mixed real + schematic renders finite + index-aligned -------------------------------
def test_mixed_real_and_schematic_is_finite_and_aligned() -> None:
    """AC-8: real + schematic coexist — no NaN/None display coords, arrays stay index-aligned."""
    data = build_region_connectome()
    pos = provision_positions(data, anatomy_override=ANATOMY_OVERRIDE)
    n = data.neuron_count

    # Every additive array is full-length (no neuron dropped) and aligned to neuron_ids order.
    for key in ("coords3d", "coords2d", "has_position", "placement", "region", "display3d"):
        assert len(pos[key]) == n, f"{key} length {len(pos[key])} != {n}"

    # display3d / coords2d are ALWAYS finite (full coverage — AC-2/AC-8).
    assert np.isfinite(np.asarray(pos["display3d"], dtype=float)).all()
    assert np.isfinite(np.asarray(pos["coords2d"], dtype=float)).all()

    # Both placement kinds are present (genuinely mixed), and every neuron is classified.
    assert set(pos["placement"]) == {PLACEMENT_ANATOMICAL, PLACEMENT_SCHEMATIC}

    # Index alignment / honesty: coords3d is real iff anatomical (has_position); None iff schematic.
    ids = np.asarray(data.neuron_ids).tolist()
    for i in range(n):
        if pos["placement"][i] == PLACEMENT_ANATOMICAL:
            assert pos["has_position"][i] is True
            assert pos["coords3d"][i] is not None
            # anatomical display coords == the real soma coords (no schematic drift).
            assert np.allclose(
                np.asarray(pos["display3d"][i], dtype=float),
                np.asarray(pos["coords3d"][i], dtype=float),
            )
            assert pos["region"][i] == ""
        else:
            assert pos["has_position"][i] is False
            assert pos["coords3d"][i] is None
            assert pos["region"][i] != ""

    # The bodyid order is preserved verbatim (the positional activation↔neuron join stays exact).
    assert ids == [bid for bid, _ in _BRAIN] + [row[0] for row in _REGION]


# --- AC-7 modality tag set is exactly the recorded set ∪ {""} ---------------------------------
def _make_modality_connectome() -> ConnectomeData:
    """A synthetic connectome carrying exactly the three recorded modality populations + blanks."""
    ids = [10, 11, 12, 13, 14]
    superclass = ["visual_projection", "", "", "", ""]  # → vision
    neuron_class = ["", "mechanosensory_proprioceptive", "", "", ""]  # → proprioceptive
    cell_type = ["", "", "IPC-1", "", ""]  # substring "ipc" → hunger
    n = len(ids)
    rng = np.random.default_rng(1)
    dense = rng.random((n, n)).astype(np.float32)
    dense[dense < 0.6] = 0.0
    return ConnectomeData(
        adjacency=sp.csr_matrix(dense),
        neuron_ids=np.asarray(ids, dtype=np.int64),
        source="synthetic-uc28-modality",
        superclass=np.asarray(superclass, dtype=object),
        neuron_class=np.asarray(neuron_class, dtype=object),
        cell_type=np.asarray(cell_type, dtype=object),
    )


def test_meta_modality_tag_set_is_exactly_recorded_plus_blank(tmp_path: Path) -> None:
    """AC-7: recorder tags neurons with exactly {vision, proprioceptive, hunger} ∪ {""}."""
    data = _make_modality_connectome()
    rec = ActivationRecorder(data, tmp_path / "rec", backend="simple", dt=0.05)

    assert len(rec.modality) == data.neuron_count
    assert set(rec.modality) == set(RECORDED_MODALITIES) | {""}
    # No modality outside the fixed recorded set leaked in.
    assert set(rec.modality) - {""} <= set(RECORDED_MODALITIES)
    # First-match / index alignment: the three tagged neurons are the ones we labelled.
    tags = list(rec.modality)
    assert tags[0] == "vision"
    assert tags[1] == "proprioceptive"
    assert tags[2] == "hunger"
    assert tags[3] == "" and tags[4] == ""
