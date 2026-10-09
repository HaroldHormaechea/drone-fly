"""Micro-benchmark: time one full PPO training iteration for the K1 actor on gates.

Breaks the per-iter cost into (a) rollout = ROLL forward passes @ batch N, and (b) PPO update =
EPOCHS*MB fwd+bwd passes @ batch N*ROLL/MB. Prints true training fps so we can size N/budget.
"""
import sys, time, math
import torch
from hover_gpu import dev
from gates_gpu import BatchedGateCourse
from gates_k1 import K1AC

N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ROLL, EPOCHS, MB = 32, 4, 8

ac = K1AC().to(dev)
opt = torch.optim.Adam(ac.parameters(), lr=3e-4)
env = BatchedGateCourse(N)
obs = env.obs()
torch.cuda.synchronize()
print(f"K1 params: {sum(p.numel() for p in ac.parameters())/1e6:.2f}M  N={N} ROLL={ROLL} "
      f"ppo_batch={N*ROLL} minibatch={N*ROLL//MB}", flush=True)

def timed(fn, label, iters=3):
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    print(f"  {label}: {dt*1000:7.1f} ms/iter", flush=True)
    return dt

# (a) rollout: ROLL forward passes, no grad, batch N
def rollout():
    o = env.obs()
    with torch.no_grad():
        for _ in range(ROLL):
            mean, std, val = ac(o)
            act = torch.distributions.Normal(mean, std).sample()
            o, rew, done, tilt = env.step(act)
t_roll = timed(rollout, "rollout (32 fwd @ N)")

# (b) PPO update: EPOCHS*MB fwd+bwd @ batch N*ROLL/MB
bo = torch.randn(N*ROLL, 12, device=dev)
ba = torch.randn(N*ROLL, 4, device=dev)
badv = torch.randn(N*ROLL, device=dev)
bret = torch.randn(N*ROLL, device=dev)
bl = torch.randn(N*ROLL, device=dev)
def ppo_update():
    idx_all = torch.randperm(bo.shape[0], device=dev)
    for _ in range(EPOCHS):
        for mb in idx_all.chunk(MB):
            mean, std, val = ac(bo[mb])
            dist = torch.distributions.Normal(mean, std)
            logp = dist.log_prob(ba[mb]).sum(-1)
            ratio = (logp - bl[mb]).exp()
            pg = -torch.min(ratio*badv[mb], torch.clamp(ratio,0.8,1.2)*badv[mb]).mean()
            vl = 0.5*(val-bret[mb]).pow(2).mean()
            loss = pg + 0.5*vl
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ac.parameters(),0.5); opt.step()
t_ppo = timed(ppo_update, "ppo update (32 fwd+bwd @ minibatch)")

t_iter = t_roll + t_ppo
fps = N*ROLL / t_iter
print(f"\n  per-iter total: {t_iter*1000:.0f} ms   -> {fps:.0f} training fps", flush=True)
for budget in (5e6, 10e6, 25e6):
    print(f"  {budget/1e6:.0f}M steps -> {budget/fps/60:.0f} min", flush=True)
