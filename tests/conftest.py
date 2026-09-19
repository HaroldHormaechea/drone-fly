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

import shutil
import socket
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from drone_fly.connectome import load_connectome
from drone_fly.connectome.loader import ConnectomeData

FIXTURE_DIR = Path(__file__).parent / "fixtures"

#: The committed fixture files copied into a throwaway dir before any provisioning write.
#: The soma sidecar is included so anatomical resolution (tier-2) still works against the copy.
_FIXTURE_FILES = ("mcns_fixture.npz", "mcns_fixture_meta.csv", "mcns_fixture_soma.csv")


def _copy_fixture_into(dst: Path) -> Path:
    """Copy the committed fixture artifacts into ``dst`` and return it.

    UC-27 provisioning writes ``<stem>_positions.csv`` / ``<stem>_soma.csv`` sidecars *beside*
    the connectome ``.npz``. Loading the connectome from a throwaway copy (rather than the
    committed ``tests/fixtures/``) means those slice-time / self-heal writes land in ``tmp``
    and the committed fixture is never dirtied.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for name in _FIXTURE_FILES:
        src = FIXTURE_DIR / name
        if src.is_file():
            shutil.copy2(src, dst / name)
    return dst


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    """Absolute path to the committed test-fixture directory (read-only ground truth).

    Use :func:`fixture_dir_copy` (not this) whenever a test constructs an
    ``ActivationRecorder`` or runs a record-enabled train/eval, so UC-27 sidecar writes never
    dirty the committed fixture.
    """
    npz = FIXTURE_DIR / "mcns_fixture.npz"
    if not npz.is_file():
        pytest.fail(
            f"Committed fixture missing: {npz}. Regenerate it with "
            f"`uv run python scripts/build_test_fixture.py`."
        )
    return FIXTURE_DIR


@pytest.fixture
def fixture_dir_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A throwaway per-test copy of the committed fixture dir (UC-27 sidecar isolation).

    Pass this (as a directory path) into any real record-enabled ``train`` / ``evaluate`` run —
    provisioning writes its ``<stem>_positions.csv`` / ``<stem>_soma.csv`` sidecars here in
    ``tmp`` instead of into ``tests/fixtures/``.
    """
    return _copy_fixture_into(tmp_path_factory.mktemp("connectome_copy"))


@pytest.fixture
def connectome(fixture_dir_copy: Path):
    """The loaded connectome for the committed fixture, from a throwaway copy.

    UC-27: ``ActivationRecorder`` construction self-heals a missing positions sidecar by
    writing it *beside* the connectome ``.npz``. Loading from a per-test copy (not the
    committed ``tests/fixtures/``) keeps that write in ``tmp`` — the committed fixture is never
    dirtied. Content is byte-identical to the committed fixture, so every existing assertion
    (scale, metadata, anatomy) holds unchanged.
    """
    return load_connectome(fixture_dir_copy)


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
