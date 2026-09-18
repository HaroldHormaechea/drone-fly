"""AC7 — fast end-to-end smoke test: load + full roundtrip, well under ~60s.

Exercises the whole UC-01 plumbing on the small committed fixture: load the cached
matrix, instantiate the connectome-seeded policy, run the encode -> network -> decode
roundtrip, and assert a finite ``(4,)`` action. Kept deliberately quick so it runs on
every CI push. The existing ``tests/test_smoke.py`` (import checks) is left unchanged.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from drone_fly.connectome import load_connectome
from drone_fly.controller import ACTION_DIM, ConnectomePolicy, run_roundtrip

# Generous ceiling — the real run is sub-second on the 322-neuron fixture. This guards
# against an accidental regression that loads the full multi-GB matrix or does N**2 work.
MAX_SECONDS = 60.0


def test_uc01_end_to_end_smoke(fixture_dir: Path) -> None:
    start = time.perf_counter()

    data = load_connectome(fixture_dir)
    assert data.neuron_count > 0 and data.edge_count > 0

    policy = ConnectomePolicy(data, force_shim=True)
    assert policy.n_connections == data.edge_count

    action = run_roundtrip(data, seed=0, force_shim=True)
    assert action.shape == (ACTION_DIM,)
    assert torch.isfinite(action).all()

    elapsed = time.perf_counter() - start
    assert elapsed < MAX_SECONDS, f"smoke test took {elapsed:.1f}s (budget {MAX_SECONDS}s)"
