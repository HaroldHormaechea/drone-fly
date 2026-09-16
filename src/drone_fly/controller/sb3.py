"""Stable-Baselines3 seam for the connectome actor (AC5).

Wraps :class:`~drone_fly.controller.actor.ConnectomeActorNetwork` as an SB3
``BaseFeaturesExtractor`` so UC-03 can drop the connectome substrate into a PPO policy
via ``policy_kwargs={"features_extractor_class": ConnectomeFeaturesExtractor,
"features_extractor_kwargs": {"data": connectome, ...}}``.

The extractor emits the 4-channel ``(THROTTLE, ROLL, PITCH, YAW)`` action vector as its
feature output (``features_dim == ACTION_DIM``); PPO's downstream policy/value heads
consume it. This keeps the connectome as the trainable feature backbone while staying
inside SB3's contract. No sim, training loop, reward, or flight is introduced here —
this is purely the wiring seam (UC-02 scope guard).
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.actor import ConnectomeActorNetwork
from drone_fly.controller.encoding import ACTION_DIM
from drone_fly.controller.policy import DEFAULT_N_STEPS
from drone_fly.controller.populations import MOTOR_POP_SIZE, SENSORY_POP_SIZE


class ConnectomeFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor backed by the connectome actor network.

    Parameters
    ----------
    observation_space:
        The environment observation space (a ``Box`` of shape ``(OBS_DIM,)``).
    data:
        Loaded connectome the actor is built from.
    n_steps, propagation_mode, motor_size, sensory_size:
        Forwarded to :class:`ConnectomeActorNetwork`.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        data: ConnectomeData,
        *,
        n_steps: int = DEFAULT_N_STEPS,
        propagation_mode: str = "scatter",
        motor_size: int = MOTOR_POP_SIZE,
        sensory_size: int = SENSORY_POP_SIZE,
    ) -> None:
        super().__init__(observation_space, features_dim=ACTION_DIM)
        self.actor = ConnectomeActorNetwork(
            data,
            n_steps=n_steps,
            propagation_mode=propagation_mode,
            motor_size=motor_size,
            sensory_size=sensory_size,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the connectome actor's ``(B, ACTION_DIM)`` feature output."""
        return self.actor(observations)
