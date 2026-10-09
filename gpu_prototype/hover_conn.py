"""Hover on the GPU-parallel sim, but with the CONNECTOME actor (the project's whole point).

Actor = ConnectomeActorNetwork (obs 12 -> 4 connectome features) + a small nonlinear pi head (4->64->4)
matching the project's winning pi=[64]. Separate MLP critic on obs. If this learns a stable hover at
scale, the connectome CAN learn control given enough compute — the pybullet failure was sample-starvation.
"""
import torch
import torch.nn as nn

from hover_gpu import AC, HOVER_THR, dev, train  # noqa: F401  (reuse sim + PPO)
from drone_fly.connectome import load_connectome
from drone_fly.connectome.prune import prune_to_subcircuit
from drone_fly.controller.actor import ConnectomeActorNetwork


class ConnectomeAC(nn.Module):
    def __init__(self, obs_dim=12, act_dim=4, hid=128):
        super().__init__()
        data = prune_to_subcircuit(load_connectome("/workspace/drone-fly/tests/fixtures"), k=2)
        self.feat = ConnectomeActorNetwork(data)          # (B,12) -> (B,4) connectome features
        self.pi = nn.Sequential(nn.Linear(act_dim, 64), nn.Tanh(), nn.Linear(64, act_dim))
        self.vf = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
        with torch.no_grad():
            self.pi[-1].bias[0] = 2 * HOVER_THR - 1       # bias collective toward hover

    def forward(self, obs):
        mean = self.pi(self.feat(obs))
        return mean, self.log_std.exp(), self.vf(obs).squeeze(-1)


if __name__ == "__main__":
    train(ConnectomeAC(), tag="conn", N=2048, budget=120e6, save_path="/workspace/drone-fly/gpu_prototype/hover_conn.pt")
