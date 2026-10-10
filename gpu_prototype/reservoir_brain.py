"""Brain-integrated reservoir: feed the raycasts INTO the connectome (opt 1) + unfreeze the input
projection (opt 3), so the 25.6k-neuron brain -- not just the ~168k readout -- does the obstacle
reasoning. The sparse EDGES stay frozen (the real MaleCNS weights); only the brain's INPUT (obs
projection + a new ray projection) and the readout are trainable.

Cost: training the brain input requires backprop through the sparse layer -> no feature caching, ~370
fps (vs ~4200 for the frozen-reservoir). Tests whether brain capacity fixes the deterministic
random-obstacle avoidance that the readout alone could not (collapsed to ~5% under annealing).
"""
import os, sys, math, time
import numpy as np
import torch, torch.nn as nn

from hover_gpu import HOVER_THR, dev
from gates_k1_reservoir import R, HID
from drone_fly.connectome import load_connectome
from drone_fly.controller.actor import ConnectomeActorNetwork
from drone_fly.controller.populations import select_sensory_population, select_motor_population

INIT_LOGSTD = -0.5
K1_PATH = "/workspace/drone-fly/artifacts/pruned/k1"
RAY_POP = int(os.environ.get("RAY_POP", "256"))  # dedicated sensory neurons for the ray cone (separate from the 32 obs neurons)


class K1ReservoirBrainRay(nn.Module):
    def __init__(self, extra_dim, n_rays, obs_dim=12, act_dim=4):
        super().__init__()
        self.extra_dim = extra_dim; self.n_rays = n_rays
        data = load_connectome(K1_PATH)
        body = ConnectomeActorNetwork(data)
        for p in body.parameters():            # freeze the WHOLE brain (edges + obs input projection):
            p.requires_grad_(False)                # opt 3 (unfreezing input_proj) disrupted the warm-start,
                                                   # so test opt 1 alone -- preserve the warm flier.
        self.body = body
        self.N = body.n_neurons
        # Separate, WIDER ray population: give the 40-ray cone its own dedicated sensory neurons
        # instead of summing it onto the 32 obs neurons (that collision conflated flight-state with
        # obstacle-sensing). Carve by set-difference so the ray pop is disjoint from the obs 32.
        m_idx, _ = select_motor_population(data)
        wide, _ = select_sensory_population(data, size=body.sensory_size + RAY_POP, exclude=m_idx)
        s_obs = body.sensory_index.cpu().numpy()
        ray_idx = wide[~np.isin(wide, s_obs)][:RAY_POP]
        self.register_buffer("ray_index", torch.as_tensor(ray_idx, dtype=torch.long))
        self.ray_size = int(self.ray_index.shape[0])
        self.ray_proj = nn.Linear(n_rays, self.ray_size)       # opt 1: rays -> dedicated ray neurons (width RAY_POP)
        nn.init.zeros_(self.ray_proj.weight); nn.init.zeros_(self.ray_proj.bias)  # start with NO effect -> preserve warm-started brain features; learn to use rays
        spread = torch.linspace(0, self.N - 1, R).long()
        tap = torch.unique(torch.cat([body.motor_index.cpu(), spread]))[:R]
        self.register_buffer("tap_index", tap)
        self.feat_dim = int(tap.shape[0])
        self.tap_norm = nn.LayerNorm(self.feat_dim)
        self.pi = nn.Sequential(nn.Linear(self.feat_dim + extra_dim, HID), nn.Tanh(), nn.Linear(HID, act_dim))
        self.vf = nn.Sequential(nn.Linear(extra_dim, HID), nn.Tanh(), nn.Linear(HID, HID), nn.Tanh(), nn.Linear(HID, 1))
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
        with torch.no_grad():
            self.pi[-1].bias[0] = 2 * HOVER_THR - 1

    def propagated_full(self, obs, rays):
        """Full (B,N) propagated neuron state: inject obs + rays into the sensory neurons, propagate
        (frozen edges). Used for recording the whole-brain activation; tap_feats() taps it."""
        x = obs.to(torch.float32)
        proj = self.body.input_projection(x)                      # (B,|sensory|) trainable
        state = torch.zeros(x.shape[0], self.N, dtype=torch.float32, device=x.device)
        state = state.index_add(1, self.body.sensory_index, proj)                  # obs -> 32 obs neurons
        state = state.index_add(1, self.ray_index, self.ray_proj(rays))            # rays -> separate wide ray population
        return self.body.layer(state)                             # frozen real-weight dynamics

    def tap(self, propagated):
        return propagated.index_select(1, self.tap_index)

    def tap_feats(self, obs, rays):
        """DIFFERENTIABLE brain forward: propagate then read the tap. Trainable via input_projection + ray_proj."""
        return self.tap(self.propagated_full(obs, rays))

    def act_from_tap(self, tap, extra):
        return self.pi(torch.cat([self.tap_norm(tap), extra], dim=-1)), self.log_std.exp()

    def act(self, obs, rays, extra):
        feats = self.tap_feats(obs, rays)
        return self.pi(torch.cat([self.tap_norm(feats), extra], dim=-1)), self.log_std.exp()

    def forward(self, obs, rays, extra):
        mean, std = self.act(obs, rays, extra)
        return mean, std, self.vf(extra).squeeze(-1)


def warm_start_brain(ac, path, src_extra_dim):
    """Copy compatible weights from a frozen-reservoir readout (incl body.input_projection + indices).
    pi.0 copies first R+src_extra cols; ray_proj + new extra cols start fresh/zero."""
    src = torch.load(path, map_location="cpu"); tgt = ac.state_dict(); R_ = ac.feat_dim
    for k, v in src.items():
        if k not in tgt:
            continue
        tv = tgt[k]
        if v.shape == tv.shape:
            tv.copy_(v)
        elif k == "pi.0.weight":
            tv.zero_(); tv[:, : R_ + src_extra_dim].copy_(v)
        elif k == "vf.0.weight":
            tv.zero_(); tv[:, :src_extra_dim].copy_(v)
    ac.load_state_dict(tgt)
    print(f"brain warm-start from {path} (src_extra={src_extra_dim}); input_projection trainable, ray_proj fresh", flush=True)


def train_brain(env_cls, extra_dim, n_rays, N=384, budget=6e6, lr=3e-4, roll=32, epochs=3, mb=8,
                ent_coef=0.01, logstd_final=-1.5, anneal_start=0.55, logstd_init=-1.2,
                pos_curr_frac=0.4, warm_from=None, warm_src=30, save_path=None):
    ac = K1ReservoirBrainRay(extra_dim, n_rays).to(dev)
    if warm_from:
        warm_start_brain(ac, warm_from, warm_src)
    trainable = [p for n, p in ac.named_parameters() if p.requires_grad and n != "log_std"]
    opt = torch.optim.Adam(trainable, lr=lr)
    env = env_cls(N)
    GAMMA, LAM, CLIP = 0.99, 0.95, 0.2
    nr = n_rays
    print(f"K1ReservoirBrainRay: trainable {sum(p.numel() for p in trainable)/1e3:.0f}k "
          f"(incl input_proj {ac.body.input_projection.weight.numel()} + ray_proj {ac.ray_proj.weight.numel()}) "
          f"ray_pop={ac.ray_size} (obs_pop={ac.body.sensory_size}, disjoint) N={N}", flush=True)
    obs = env.obs(); ext = env.extra()
    total = 0; t0 = time.time()
    for it in range(1, 2_000_001):
        frac = total / budget
        # POSITION curriculum: slide obstacles parked-off-path -> on-path over the first pos_curr_frac.
        if hasattr(env, "obs_pos_curr"):
            env.obs_pos_curr = min(1.0, frac / max(1e-6, pos_curr_frac))
        if logstd_final is not None:
            a = max(0.0, (frac - anneal_start) / max(1e-6, 1 - anneal_start))
            with torch.no_grad():
                ac.log_std.fill_(logstd_init + (logstd_final - logstd_init) * min(1.0, a))
        ob_b = torch.zeros(roll, N, 12, device=dev); ex_b = torch.zeros(roll, N, extra_dim, device=dev)
        ac_b = torch.zeros(roll, N, 4, device=dev); lp_b = torch.zeros(roll, N, device=dev)
        rw_b = torch.zeros(roll, N, device=dev); vl_b = torch.zeros(roll, N, device=dev); dn_b = torch.zeros(roll, N, device=dev)
        tilt_acc = []
        with torch.no_grad():
            for s in range(roll):
                rays = ext[:, -nr:]
                mean, std = ac.act(obs, rays, ext); val = ac.vf(ext).squeeze(-1)
                dist = torch.distributions.Normal(mean, std); act = dist.sample(); logp = dist.log_prob(act).sum(-1)
                nobs, rew, done, tilt = env.step(act); next_ext = env.extra()
                ob_b[s], ex_b[s], ac_b[s], lp_b[s], rw_b[s], vl_b[s], dn_b[s] = obs, ext, act, logp, rew, val, done.float()
                obs, ext = nobs, next_ext; tilt_acc.append(tilt.mean())
            last_val = ac.vf(ext).squeeze(-1)
        adv = torch.zeros(roll, N, device=dev); lg = torch.zeros(N, device=dev)
        for s in reversed(range(roll)):
            nv = last_val if s == roll - 1 else vl_b[s + 1]; nt = 1 - dn_b[s]
            delta = rw_b[s] + GAMMA * nv * nt - vl_b[s]; lg = delta + GAMMA * LAM * nt * lg; adv[s] = lg
        ret = adv + vl_b
        bo = ob_b.reshape(-1, 12); be = ex_b.reshape(-1, extra_dim); ba = ac_b.reshape(-1, 4)
        bl = lp_b.reshape(-1); badv = adv.reshape(-1); bret = ret.reshape(-1)
        badv = (badv - badv.mean()) / (badv.std() + 1e-8)
        idx = torch.randperm(bo.shape[0], device=dev)
        for _ in range(epochs):
            for mbi in idx.chunk(mb):
                rays = be[mbi][:, -nr:]
                mean, std = ac.act(bo[mbi], rays, be[mbi]); val = ac.vf(be[mbi]).squeeze(-1)
                d = torch.distributions.Normal(mean, std); logp = d.log_prob(ba[mbi]).sum(-1)
                ratio = (logp - bl[mbi]).exp()
                pg = -torch.min(ratio * badv[mbi], torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * badv[mbi]).mean()
                vlo = 0.5 * (val - bret[mbi]).pow(2).mean(); ent = d.entropy().sum(-1).mean()
                loss = pg + 0.5 * vlo - ent_coef * ent
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(trainable, 0.5); opt.step()
        total += roll * N
        if it % 5 == 0:
            fps = total / (time.time() - t0); extra_m = env.metric() if hasattr(env, "metric") else ""
            posc = getattr(env, "obs_pos_curr", 1.0)
            print(f"[brain] it {it:4d} steps {total/1e6:5.1f}M fps {fps:5.0f} ep_rew {rw_b.mean():6.3f} "
                  f"tilt {math.degrees(float(torch.stack(tilt_acc).mean())):5.1f}deg posc {posc:.2f} {extra_m}", flush=True)
        if it % 50 == 0 and save_path:
            torch.save(ac.state_dict(), save_path)
        if total > budget:
            break
    if save_path:
        torch.save(ac.state_dict(), save_path); print(f"[brain] saved {save_path}", flush=True)


if __name__ == "__main__":
    from gates_proc_env import BatchedProcCourse, EXTRA_DIM, N_RAYS
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 384
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 6e6
    train_brain(BatchedProcCourse, EXTRA_DIM, N_RAYS, N=N, budget=budget,
                warm_from="/workspace/drone-fly/training/07-proc-course-flight/model_readout.pt", warm_src=30,
                save_path="/workspace/drone-fly/gpu_prototype/proc_brain.pt")
