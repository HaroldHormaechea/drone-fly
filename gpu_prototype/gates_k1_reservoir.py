"""Real K1 (25.6k-neuron / 3.86M-edge) connectome as a FROZEN RESERVOIR flying the gate course.

Why frozen: training the sparse edges backprops through 3.86M magnitudes 32x/iter -> 368 fps -> ~19h
for 25M steps. Biologically you do not gradient-edit a connectome; its wiring + synaptic magnitudes
(the real MaleCNS values, loaded into edge_weight) are a FIXED substrate. Learning lives in the READOUT.

So: body = ConnectomeActorNetwork pieces (input_projection -> real-weight sparse propagation), ALL frozen.
Every observation is processed by the full 25.6k-neuron brain. We tap a wide fixed sub-population of the
propagated state (R neurons) as features, and train a small policy/value head on them. Because the body
is frozen, its features are a fixed function of obs -> we compute them ONCE per rollout and REUSE them
across PPO epochs, so the expensive sparse op runs only ROLL times/iter (rollout), never in the update.
Result: PPO update touches only the tiny head -> ~10x faster, fits in ~2h.
"""
import sys, math, time
import torch, torch.nn as nn

from hover_gpu import HOVER_THR, dev
from gates_gpu import BatchedGateCourse
from drone_fly.connectome import load_connectome
from drone_fly.controller.actor import ConnectomeActorNetwork

R = 1024         # reservoir readout width (neurons tapped from the propagated 25.6k state)
HID = 128


class K1Reservoir(nn.Module):
    """Frozen real-K1 connectome body + trainable policy/value readout."""

    def __init__(self, obs_dim=12, act_dim=4):
        super().__init__()
        data = load_connectome("/workspace/drone-fly/artifacts/pruned/k1")  # 25.6k, real edge weights
        body = ConnectomeActorNetwork(data)                                  # input_proj + sparse layer
        for p in body.parameters():
            p.requires_grad_(False)                                          # FREEZE the whole brain
        body.eval()
        self.body = body
        self.N = body.n_neurons
        # Wide, deterministic reservoir tap: evenly spread across neuron indices + include the motor pop.
        spread = torch.linspace(0, self.N - 1, R).long()
        tap = torch.unique(torch.cat([body.motor_index.cpu(), spread]))[:R]
        self.register_buffer("tap_index", tap)
        self.feat_dim = int(tap.shape[0])
        # Trainable readout on the CACHED raw tap. A LayerNorm first: the real-connectome propagation
        # produces features on wildly different scales (biological edge magnitudes) that a bare Linear
        # readout struggles to use; normalizing them is the standard reservoir-readout fix and is cheap
        # + fully cache-compatible (it trains, but operates on the stored raw tap each epoch).
        self.pi = nn.Sequential(nn.LayerNorm(self.feat_dim), nn.Linear(self.feat_dim, HID), nn.Tanh(),
                                nn.Linear(HID, act_dim))
        self.vf = nn.Sequential(nn.Linear(obs_dim, HID), nn.Tanh(), nn.Linear(HID, HID), nn.Tanh(), nn.Linear(HID, 1))
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
        with torch.no_grad():
            self.pi[-1].bias[0] = 2 * HOVER_THR - 1                          # bias throttle toward hover

    @torch.no_grad()
    def features(self, obs):
        """Full-brain forward (FROZEN): obs -> sensory injection -> real-weight propagation -> tap."""
        x = obs.to(torch.float32)
        projected = self.body.input_projection(x)                           # (B, |sensory|)
        state = torch.zeros(x.shape[0], self.N, dtype=torch.float32, device=x.device)
        state = state.index_add(1, self.body.sensory_index, projected)
        propagated = self.body.layer(state)                                 # (B, N) real K1 dynamics
        return propagated.index_select(1, self.tap_index)                   # (B, R)

    def act_from_feats(self, feats):
        return self.pi(feats), self.log_std.exp()

    def forward(self, obs):
        """Convenience (eval): obs -> (mean, std, value)."""
        feats = self.features(obs)
        return self.pi(feats), self.log_std.exp(), self.vf(obs).squeeze(-1)


def train(N=512, budget=25e6, lr=3e-4, roll=32, epochs=4, mb=8, ent_coef=0.01,
          save_path="/workspace/drone-fly/gpu_prototype/gates_k1_reservoir.pt"):
    ac = K1Reservoir().to(dev)
    # Only the readout trains (body is frozen) -> optimizer sees the small head params.
    opt = torch.optim.Adam([p for p in ac.parameters() if p.requires_grad], lr=lr)
    env = BatchedGateCourse(N)
    GAMMA, LAM, CLIP = 0.99, 0.95, 0.2
    print(f"K1 reservoir: feat_dim={ac.feat_dim} trainable={sum(p.numel() for p in ac.parameters() if p.requires_grad)/1e3:.1f}k "
          f"frozen={sum(p.numel() for p in ac.parameters() if not p.requires_grad)/1e6:.2f}M  N={N} ppo_batch={N*roll}", flush=True)
    obs = env.obs()
    total_steps = 0
    t0 = time.time()
    for it in range(1, 2_000_001):
        feat_b = torch.zeros(roll, N, ac.feat_dim, device=dev)
        obs_b = torch.zeros(roll, N, 12, device=dev)
        act_b = torch.zeros(roll, N, 4, device=dev)
        logp_b = torch.zeros(roll, N, device=dev)
        rew_b = torch.zeros(roll, N, device=dev)
        val_b = torch.zeros(roll, N, device=dev)
        done_b = torch.zeros(roll, N, device=dev)
        tilt_acc = []
        with torch.no_grad():
            for s in range(roll):
                feats = ac.features(obs)                 # frozen sparse forward (the only K1 cost)
                mean = ac.pi(feats); std = ac.log_std.exp()
                val = ac.vf(obs).squeeze(-1)
                dist = torch.distributions.Normal(mean, std)
                act = dist.sample(); logp = dist.log_prob(act).sum(-1)
                nobs, rew, done, tilt = env.step(act)
                feat_b[s], obs_b[s], act_b[s], logp_b[s], rew_b[s], val_b[s], done_b[s] = \
                    feats, obs, act, logp, rew, val, done.float()
                obs = nobs
                tilt_acc.append(tilt.mean())
            lastf = ac.features(obs); last_val = ac.vf(obs).squeeze(-1)
        # GAE
        adv = torch.zeros(roll, N, device=dev); lastgae = torch.zeros(N, device=dev)
        for s in reversed(range(roll)):
            nextval = last_val if s == roll - 1 else val_b[s + 1]
            nonterm = 1.0 - done_b[s]
            delta = rew_b[s] + GAMMA * nextval * nonterm - val_b[s]
            lastgae = delta + GAMMA * LAM * nonterm * lastgae
            adv[s] = lastgae
        ret = adv + val_b
        bf = feat_b.reshape(-1, ac.feat_dim); bo = obs_b.reshape(-1, 12); ba = act_b.reshape(-1, 4)
        bl = logp_b.reshape(-1); badv = adv.reshape(-1); bret = ret.reshape(-1)
        badv = (badv - badv.mean()) / (badv.std() + 1e-8)
        idx_all = torch.randperm(bf.shape[0], device=dev)
        for _ in range(epochs):
            for mbi in idx_all.chunk(mb):
                mean = ac.pi(bf[mbi]); std = ac.log_std.exp()   # tiny head on CACHED features
                val = ac.vf(bo[mbi]).squeeze(-1)
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(ba[mbi]).sum(-1)
                ratio = (logp - bl[mbi]).exp()
                pg = -torch.min(ratio * badv[mbi], torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * badv[mbi]).mean()
                vl = 0.5 * (val - bret[mbi]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + 0.5 * vl - ent_coef * ent
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()
        total_steps += roll * N
        if it % 20 == 0:
            fps = total_steps / (time.time() - t0)
            extra = env.metric() if hasattr(env, "metric") else ""
            print(f"[k1_res] it {it:4d} steps {total_steps/1e6:6.1f}M fps {fps:6.0f} "
                  f"ep_rew/step {rew_b.mean():6.3f} tilt {math.degrees(float(torch.stack(tilt_acc).mean())):5.1f}deg {extra}", flush=True)
        if it % 200 == 0:
            torch.save(ac.state_dict(), save_path)
        if total_steps > budget:
            break
    torch.save(ac.state_dict(), save_path)
    print(f"[k1_res] saved {save_path}", flush=True)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 25e6
    train(N=N, budget=budget)
