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
    sensory_index, motor_index:
        Optional pinned sub-populations (UC-07), forwarded verbatim to
        :class:`ConnectomeActorNetwork`. When supplied they bypass ``select_populations``;
        crucially, because they live in ``features_extractor_kwargs`` they are pickled into
        the checkpoint, so ``PPO.load`` (which passes no connectome) rebuilds the actor with
        the *same* pinned neurons rather than re-selecting a possibly-misaligned sub-pop.
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
        sensory_index=None,
        motor_index=None,
    ) -> None:
        super().__init__(observation_space, features_dim=ACTION_DIM)
        self.actor = ConnectomeActorNetwork(
            data,
            n_steps=n_steps,
            propagation_mode=propagation_mode,
            motor_size=motor_size,
            sensory_size=sensory_size,
            sensory_index=sensory_index,
            motor_index=motor_index,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the connectome actor's ``(B, ACTION_DIM)`` feature output."""
        return self.actor(observations)

    # -- UC-05 activation-recording passthrough -----------------------------------------
    # The recorder toggles capture via the actor's ``sink`` (see
    # :class:`~drone_fly.controller.actor.ConnectomeActorNetwork`). These pin the hook to
    # the single feature-extraction (pi/actor) path, so capture happens exactly once per
    # env step — SB3 calls the shared features extractor once to build the policy latent,
    # not separately for the value head — avoiding any double-capture of the same frame.
    @property
    def sink(self):
        """The actor's activation sink (``None`` when recording is off)."""
        return self.actor.sink

    @sink.setter
    def sink(self, fn) -> None:
        self.actor.sink = fn


def actor_from_model(model) -> ConnectomeActorNetwork:
    """Return the :class:`ConnectomeActorNetwork` inside a loaded SB3 model.

    The connectome substrate lives at ``model.policy.features_extractor.actor``. Raises a
    clear error if the model was not built with :class:`ConnectomeFeaturesExtractor` (so a
    recording request against an incompatible checkpoint fails loudly, not cryptically).
    """
    extractor = getattr(getattr(model, "policy", None), "features_extractor", None)
    actor = getattr(extractor, "actor", None)
    if not isinstance(actor, ConnectomeActorNetwork):
        raise TypeError(
            "The loaded model's features extractor is not a ConnectomeFeaturesExtractor; "
            "activation recording requires a connectome-seeded policy (UC-02/03)."
        )
    return actor
