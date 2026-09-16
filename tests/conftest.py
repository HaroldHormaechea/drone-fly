"""Shared pytest fixtures for the connectome test suite (UC-01 plumbing + UC-02 hardening).

Provides:
* ``fixture_dir`` — path to the committed real-MaleCNS test fixture
  (``tests/fixtures/``), built by ``scripts/build_test_fixture.py``.
* ``connectome`` — the loaded :class:`~drone_fly.connectome.loader.ConnectomeData`
  for that fixture (carries real ``superclass`` / ``sign`` / ``top_nt`` metadata).
* ``synthetic_connectome`` — a small metadata-less :class:`ConnectomeData` (no
  ``superclass`` / ``sign``) used to exercise UC-02's documented degrade paths.
* ``no_network`` — an offline-enforcement fixture that monkeypatches ``socket``
  so any attempt to open a network connection raises, proving the load/roundtrip
  path is genuinely offline (AC2).
"""

from __future__ import annotations

import socket
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from drone_fly.connectome import load_connectome
from drone_fly.connectome.loader import ConnectomeData

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    """Absolute path to the committed test-fixture directory."""
    npz = FIXTURE_DIR / "mcns_fixture.npz"
    if not npz.is_file():
        pytest.fail(
            f"Committed fixture missing: {npz}. Regenerate it with "
            f"`uv run python scripts/build_test_fixture.py`."
        )
    return FIXTURE_DIR


@pytest.fixture
def connectome(fixture_dir: Path):
    """The loaded connectome for the committed fixture."""
    return load_connectome(fixture_dir)


@pytest.fixture
def synthetic_connectome() -> ConnectomeData:
    """A small metadata-less connectome for exercising UC-02 degrade paths.

    Has neither ``superclass`` nor ``sign`` (both ``None``), so population selection
    must fall back to the UC-01 placeholder encoding and the sign mask must degrade to
    the documented default. Deterministically constructed (fixed seed, no network).
    ``N=50`` is comfortably above the ``2 * ACTION_DIM`` minimum the encoding requires.
    """
    rng = np.random.default_rng(0)
    n = 50
    dense = rng.random((n, n)).astype(np.float32)
    dense[dense < 0.8] = 0.0  # ~20% density, all-positive magnitudes (like the real .npz)
    adjacency = sp.csr_matrix(dense)
    return ConnectomeData(
        adjacency=adjacency,
        neuron_ids=np.arange(n, dtype=np.int64),
        source="synthetic-metadata-less",
    )


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch):
    """Make any network access raise, to prove the code path is offline (AC2).

    Patches the socket constructor and the common connection helpers so that any
    attempt to reach the network — directly or via a library like ``requests`` /
    ``neuprint`` — fails loudly instead of silently succeeding.
    """

    def _blocked(*args, **kwargs):  # noqa: ANN002, ANN003
        raise OSError("Network access is disabled for this test (AC2: offline load path).")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    return None
