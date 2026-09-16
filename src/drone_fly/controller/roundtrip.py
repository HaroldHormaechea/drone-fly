"""Full encode -> connectome-network -> decode roundtrip (UC-01 success bar).

This ties the pieces together into the single call UC-02 will build on: a dummy
observation goes in, is scattered into the network's sensory neurons, propagates one
hop through the connectome-seeded policy, and the motor neurons are read out to a
finite ``(4,)`` throttle/roll/pitch/yaw action.

Determinism (AC5): the seed is set with :func:`torch.manual_seed` **before** the policy
is instantiated, because the trainable edge weights of the AxonWeave layer are randomly
initialised at construction. Seeding first makes the whole roundtrip byte-reproducible.
"""

from __future__ import annotations

import torch

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.encoding import OBS_DIM, decode_action, encode_observation
from drone_fly.controller.policy import ConnectomePolicy


def dummy_observation(seed: int = 0) -> torch.Tensor:
    """A fixed, deterministic length-``OBS_DIM`` dummy observation for the POC."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(OBS_DIM, generator=generator, dtype=torch.float32)


def run_roundtrip(
    data: ConnectomeData,
    obs: torch.Tensor | None = None,
    seed: int = 0,
    *,
    force_shim: bool = False,
) -> torch.Tensor:
    """Run one encode -> propagate -> decode pass and return a finite ``(4,)`` action.

    Parameters
    ----------
    data:
        The loaded connectome.
    obs:
        Optional observation of shape ``(OBS_DIM,)``. When ``None``, a deterministic
        dummy observation derived from ``seed`` is used.
    seed:
        Seed applied via :func:`torch.manual_seed` **before** policy instantiation, so
        randomly-initialised trainable weights (and hence the whole roundtrip) are
        reproducible.
    force_shim:
        Force the in-repo shim backend (used by tests to exercise it deterministically
        regardless of whether AxonWeave is installed).

    Returns
    -------
    torch.Tensor
        Shape ``(4,)``, finite, laid out as ``ACTION_LAYOUT`` (THROTTLE, ROLL, PITCH,
        YAW); THROTTLE in ``[0, 1]``, the rest in ``[-1, 1]``.
    """
    torch.manual_seed(seed)
    if obs is None:
        obs = dummy_observation(seed)

    policy = ConnectomePolicy(data, force_shim=force_shim)
    policy.eval()

    with torch.no_grad():
        neuron_input = encode_observation(obs, data.neuron_count)
        state = policy(neuron_input)
        action = decode_action(state)
    return action
