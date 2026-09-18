"""AC4 + AC6 — the sensory-encode / motor-decode I/O contract.

* **AC4** — a stub sensory-encoder maps a dummy observation into the network inputs;
  a stub motor-decoder reads designated motor neurons out to a finite ``(4,)`` action
  in the documented THROTTLE/ROLL/PITCH/YAW layout and ranges.
* **AC6** — the contract (obs shape in, ``(4,)`` action out, neuron-index mapping) is
  fixed and documented so UC-02 can depend on it; these tests pin that contract.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drone_fly.controller import (
    ACTION_DIM,
    ACTION_LAYOUT,
    ATTITUDE_RANGE,
    OBS_DIM,
    THROTTLE_INDEX,
    THROTTLE_RANGE,
    decode_action,
    encode_observation,
    motor_neuron_indices,
    sensory_neuron_indices,
)

N = 322  # fixture neuron count


def test_contract_constants() -> None:
    assert ACTION_DIM == 4
    assert ACTION_LAYOUT == ("THROTTLE", "ROLL", "PITCH", "YAW")
    assert ACTION_LAYOUT[THROTTLE_INDEX] == "THROTTLE"
    assert THROTTLE_RANGE == (0.0, 1.0)
    assert ATTITUDE_RANGE == (-1.0, 1.0)


def test_sensory_and_motor_indices_disjoint_and_deterministic() -> None:
    s1 = sensory_neuron_indices(N)
    s2 = sensory_neuron_indices(N)
    m1 = motor_neuron_indices(N)
    m2 = motor_neuron_indices(N)
    # deterministic
    assert np.array_equal(s1, s2)
    assert np.array_equal(m1, m2)
    # disjoint (AC4 stub populations must not overlap)
    assert set(s1.tolist()).isdisjoint(set(m1.tolist()))
    # motor reads exactly ACTION_DIM channels
    assert m1.shape[0] == ACTION_DIM
    # all indices in range
    assert s1.min() >= 0 and s1.max() < N
    assert m1.min() >= 0 and m1.max() < N


def test_encode_observation_shape_and_scatter() -> None:
    obs = torch.arange(OBS_DIM, dtype=torch.float32) + 1.0  # nonzero values
    neuron_input = encode_observation(obs, N)
    assert neuron_input.shape == (N,)
    assert torch.isfinite(neuron_input).all()
    # exactly the sensory indices carry the (nonzero) observation values
    idx = torch.as_tensor(sensory_neuron_indices(N), dtype=torch.long)
    assert (neuron_input[idx] != 0).all()
    # everything outside the sensory indices is zero
    mask = torch.ones(N, dtype=torch.bool)
    mask[idx] = False
    assert torch.count_nonzero(neuron_input[mask]) == 0


def test_encode_observation_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        encode_observation(torch.zeros(OBS_DIM + 1), N)


def test_decode_action_shape_layout_and_ranges() -> None:
    # Use large-magnitude state to prove the squashing keeps outputs in range.
    state = torch.full((N,), 50.0, dtype=torch.float32)
    action = decode_action(state)
    assert action.shape == (ACTION_DIM,)
    assert torch.isfinite(action).all()
    throttle = action[THROTTLE_INDEX].item()
    assert THROTTLE_RANGE[0] <= throttle <= THROTTLE_RANGE[1]
    for i in range(ACTION_DIM):
        if i != THROTTLE_INDEX:
            assert ATTITUDE_RANGE[0] <= action[i].item() <= ATTITUDE_RANGE[1]


def test_decode_action_negative_state_in_range() -> None:
    state = torch.full((N,), -50.0, dtype=torch.float32)
    action = decode_action(state)
    assert THROTTLE_RANGE[0] <= action[THROTTLE_INDEX].item() <= THROTTLE_RANGE[1]
    for i in range(ACTION_DIM):
        if i != THROTTLE_INDEX:
            assert ATTITUDE_RANGE[0] <= action[i].item() <= ATTITUDE_RANGE[1]


def test_indices_raise_when_connectome_too_small() -> None:
    with pytest.raises(ValueError):
        sensory_neuron_indices(2 * ACTION_DIM - 1)
    with pytest.raises(ValueError):
        motor_neuron_indices(2 * ACTION_DIM - 1)
