"""AC5 — the full encode -> network -> decode roundtrip.

A single forward pass returns a finite ``(4,)`` action for a dummy observation, in the
documented ranges, and is deterministic for a fixed seed. ``force_shim=True`` keeps the
test stable in CI regardless of whether AxonWeave is installed.
"""

from __future__ import annotations

import torch

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller import (
    ACTION_DIM,
    ATTITUDE_RANGE,
    THROTTLE_INDEX,
    THROTTLE_RANGE,
    dummy_observation,
    run_roundtrip,
)
from drone_fly.controller.encoding import OBS_DIM


def _assert_valid_action(action: torch.Tensor) -> None:
    assert action.shape == (ACTION_DIM,)
    assert torch.isfinite(action).all()
    assert THROTTLE_RANGE[0] <= action[THROTTLE_INDEX].item() <= THROTTLE_RANGE[1]
    for i in range(ACTION_DIM):
        if i != THROTTLE_INDEX:
            assert ATTITUDE_RANGE[0] <= action[i].item() <= ATTITUDE_RANGE[1]


def test_roundtrip_returns_finite_action_in_range(connectome: ConnectomeData) -> None:
    action = run_roundtrip(connectome, seed=0, force_shim=True)
    _assert_valid_action(action)


def test_roundtrip_is_deterministic_for_fixed_seed(connectome: ConnectomeData) -> None:
    a1 = run_roundtrip(connectome, seed=42, force_shim=True)
    a2 = run_roundtrip(connectome, seed=42, force_shim=True)
    assert torch.equal(a1, a2)  # byte-identical


def test_roundtrip_accepts_explicit_observation(connectome: ConnectomeData) -> None:
    obs = dummy_observation(seed=3)
    assert obs.shape == (OBS_DIM,)
    action = run_roundtrip(connectome, obs=obs, seed=3, force_shim=True)
    _assert_valid_action(action)


def test_roundtrip_offline(connectome: ConnectomeData, no_network: None) -> None:
    """The roundtrip runs with networking disabled (reinforces AC2 end to end)."""
    action = run_roundtrip(connectome, seed=1, force_shim=True)
    _assert_valid_action(action)


def test_dummy_observation_is_deterministic() -> None:
    assert torch.equal(dummy_observation(seed=5), dummy_observation(seed=5))
