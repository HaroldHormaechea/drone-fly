import torch, torch.nn as nn
from hover_gpu import HOVER_THR, train
from gates_gpu import BatchedGateCourse
from drone_fly.connectome import load_connectome
from drone_fly.controller.actor import ConnectomeActorNetwork

class K1AC(nn.Module):
    def __init__(self, obs_dim=12, act_dim=4, hid=128):
        super().__init__()
        data = load_connectome("/workspace/drone-fly/artifacts/pruned/k1")  # 25.6k, no re-prune
        self.feat = ConnectomeActorNetwork(data)
        self.pi = nn.Sequential(nn.Linear(act_dim,64), nn.Tanh(), nn.Linear(64,act_dim))
        self.vf = nn.Sequential(nn.Linear(obs_dim,hid), nn.Tanh(), nn.Linear(hid,hid), nn.Tanh(), nn.Linear(hid,1))
        self.log_std = nn.Parameter(torch.zeros(act_dim)-0.5)
        with torch.no_grad(): self.pi[-1].bias[0] = 2*HOVER_THR-1
    def forward(self, obs):
        return self.pi(self.feat(obs)), self.log_std.exp(), self.vf(obs).squeeze(-1)

if __name__ == "__main__":
    import sys
    # sparse connectome kernel (committed) makes K1 fit at real batch — standard PPO settings now.
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    train(K1AC(), tag=f"gates_k1_N{N}", N=N, budget=25e6, env_fn=BatchedGateCourse, ent_coef=0.01,
          roll=32, mb=8, epochs=4, save_path="/workspace/drone-fly/gpu_prototype/gates_k1.pt")
