"""AC5 — the hardened policy is a valid Stable-Baselines3 feature extractor.

:class:`ConnectomeFeaturesExtractor` subclasses SB3's ``BaseFeaturesExtractor`` and wraps
the connectome actor. Verified with a smoke test: instantiate against a ``Box`` obs
space, run one forward + one backward pass, confirm gradients reach the sparse connectome
weights, and confirm CPU determinism for a fixed seed.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.encoding import ACTION_DIM, OBS_DIM
from drone_fly.controller.sb3 import ConnectomeFeaturesExtractor


def _obs_space() -> gym.spaces.Box:
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)


def test_is_sb3_features_extractor(connectome: ConnectomeData) -> None:
    torch.manual_seed(0)
    ext = ConnectomeFeaturesExtractor(_obs_space(), connectome)
    assert isinstance(ext, BaseFeaturesExtractor)
    assert ext.features_dim == ACTION_DIM


def test_forward_backward_reaches_sparse_weights(connectome: ConnectomeData) -> None:
    torch.manual_seed(0)
    ext = ConnectomeFeaturesExtractor(_obs_space(), connectome)
    obs = torch.randn(4, OBS_DIM)
    features = ext(obs)
    assert features.shape == (4, ACTION_DIM)
    assert torch.isfinite(features).all()

    features.sum().backward()
    grad = ext.actor.layer.edge_weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert (grad != 0).any(), "gradient did not reach the sparse connectome weights"


def test_cpu_deterministic_for_fixed_seed(connectome: ConnectomeData) -> None:
    obs = torch.randn(3, OBS_DIM)

    torch.manual_seed(123)
    ext_a = ConnectomeFeaturesExtractor(_obs_space(), connectome)
    out_a = ext_a(obs)

    torch.manual_seed(123)
    ext_b = ConnectomeFeaturesExtractor(_obs_space(), connectome)
    out_b = ext_b(obs)

    assert torch.equal(out_a, out_b)
