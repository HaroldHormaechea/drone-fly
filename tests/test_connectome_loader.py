"""AC1 + AC2 — offline loader reads the cached matrix at the expected scale.

* **AC1** — the loader returns a structure exposing neuron and synapse/edge counts;
  both are > 0 and match the documented expected scale of the committed fixture.
  The full-dataset scale constant is defined and positive (asserted here; the actual
  full-matrix load is an auto-skipped test that runs only when the multi-GB artifact
  is present).
* **AC2** — the load path performs no network I/O and never imports ``neuprint``; it
  succeeds with networking disabled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

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
