"""The privileged teacher (UC68 D1): a plain MLP actor-critic trained from scratch.

* **Actor**: 256-256-128 ELU on the ``teacher-obs`` v1 observation (148 values), scaled inside
  the module by the schema's fixed per-value ``input_scale`` (a buffer, saved with the weights);
  a Gaussian head on the raw v2 action with a learned state-independent ``log_std``.
* **Critic** (asymmetric): 512-512 ELU on the teacher observation plus critic-only features that
  are never part of the observation schema (the dynamics and timing samples, ``t / time_limit``
  and the route progress fraction).

It implements the PPO actor protocol of :mod:`drone_fly.training.ppo`: ``features(obs)`` (an
empty ``(n, 0)`` tensor: nothing is frozen), ``act(features, extra) -> (mean, std)``,
``value(critic_input)`` and ``readout_state_dict()`` (the full state dict).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

ACTOR_HIDDEN = (256, 256, 128)
CRITIC_HIDDEN = (512, 512)
ACT_DIM = 4


def _mlp(i: int, hidden: Sequence[int], o: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(i, h), nn.ELU()]
        i = h
    layers.append(nn.Linear(i, o))
    return nn.Sequential(*layers)


class MLPTeacher(nn.Module):
    """See the module docstring."""

    kind = "mlp-teacher"

    def __init__(
        self,
        obs_dim: int,
        critic_extra_dim: int,
        input_scale: Sequence[float] | None = None,
        *,
        logstd_init: float = -0.5,
        actor_hidden: Sequence[int] = ACTOR_HIDDEN,
        critic_hidden: Sequence[int] = CRITIC_HIDDEN,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.critic_extra_dim = int(critic_extra_dim)
        scale = torch.ones(obs_dim) if input_scale is None else torch.as_tensor(input_scale)
        if scale.shape != (obs_dim,):
            raise ValueError(f"input_scale must have {obs_dim} values")
        self.register_buffer("input_scale", scale.float())
        self.pi = _mlp(obs_dim, actor_hidden, ACT_DIM)
        self.vf = _mlp(obs_dim + critic_extra_dim, critic_hidden, 1)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), float(logstd_init)))
        nn.init.zeros_(self.pi[-1].bias)
        with torch.no_grad():
            self.pi[-1].weight.mul_(0.01)  # start near hover (a0 = 0 is 1 g in contract v2)
        self.feat_dim = 0

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.new_zeros(obs.shape[0], 0)

    def act(self, features: torch.Tensor, extra: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.pi(extra * self.input_scale)
        return mean, self.log_std.exp().expand_as(mean)

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.pi(obs * self.input_scale)

    def value(self, critic_input: torch.Tensor) -> torch.Tensor:
        obs = critic_input[:, : self.obs_dim] * self.input_scale
        x = torch.cat([obs, critic_input[:, self.obs_dim :]], dim=-1)
        return self.vf(x).squeeze(-1)

    def readout_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> MLPTeacher:
        obs_dim = int(state["input_scale"].shape[0])
        crit_in = int(state["vf.0.weight"].shape[1])
        m = cls(obs_dim, crit_in - obs_dim, state["input_scale"].tolist())
        m.load_state_dict(state)
        return m
