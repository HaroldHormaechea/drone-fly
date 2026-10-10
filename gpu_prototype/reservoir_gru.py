"""Recurrent (GRU) readout on the frozen K1 reservoir — gives the controller WORKING MEMORY so it can
hold a multi-step avoidance arc through occlusion, instead of re-deriving it from scratch every 20ms
frame (the structural limit of the memoryless readout that capped random on-path obstacle avoidance at
~5% deterministic). See docs/FLIGHT_MODELS.md and the GRU design note.

Design (informed by the failed memoryless attempts):
- Brain FULLY frozen (edges + input_projection) -> the 1024-tap is cacheable -> fast recurrent PPO.
- Obstacle perception reaches the controller via `extra` (raycasts + obstacle vision), NOT through the
  brain (injecting rays into the brain was proven not to be the bottleneck).
- GRU head over [LayerNorm(tap) || extra] -> hidden state carries intent across steps.
- Truncated BPTT over short chunks; hidden state resets at episode boundaries.
- Gentler, mildly-stochastic annealing (logstd_final=-1.0) to avoid the global-timidity collapse the
  hard anneal (-1.5) produced.
"""
import sys, math, time
import torch, torch.nn as nn

from hover_gpu import HOVER_THR, dev
from gates_k1_reservoir import R, HID
from drone_fly.connectome import load_connectome
from drone_fly.controller.actor import ConnectomeActorNetwork

K1_PATH = "/workspace/drone-fly/artifacts/pruned/k1"
GRU_HID = 128


class K1ReservoirGRU(nn.Module):
    """Frozen K1 body + GRU readout over [normalized brain tap || extra senses]."""

    def __init__(self, extra_dim, obs_dim=12, act_dim=4, gru_hid=GRU_HID):
        super().__init__()
        self.extra_dim = extra_dim; self.gru_hid = gru_hid
        data = load_connectome(K1_PATH)
        body = ConnectomeActorNetwork(data)
        for p in body.parameters():
            p.requires_grad_(False)
        body.eval()
        self.body = body
        self.N = body.n_neurons
        spread = torch.linspace(0, self.N - 1, R).long()
        tap = torch.unique(torch.cat([body.motor_index.cpu(), spread]))[:R]
        self.register_buffer("tap_index", tap)
        self.feat_dim = int(tap.shape[0])
        self.tap_norm = nn.LayerNorm(self.feat_dim)
        self.gru = nn.GRUCell(self.feat_dim + extra_dim, gru_hid)   # working memory over [tap || extra]
        self.pi = nn.Linear(gru_hid, act_dim)                       # action from the hidden state
        self.vf = nn.Sequential(nn.Linear(extra_dim, HID), nn.Tanh(),
                                nn.Linear(HID, HID), nn.Tanh(), nn.Linear(HID, 1))  # memoryless critic on extra
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
        with torch.no_grad():
            self.pi.bias[0] = 2 * HOVER_THR - 1

    @torch.no_grad()
    def features(self, obs):
        """Cacheable frozen-brain tap (B, feat_dim)."""
        x = obs.to(torch.float32)
        projected = self.body.input_projection(x)
        state = torch.zeros(x.shape[0], self.N, dtype=torch.float32, device=x.device)
        state = state.index_add(1, self.body.sensory_index, projected)
        return self.body.layer(state).index_select(1, self.tap_index)

    def step(self, feats_tap, extra, h):
        """One recurrent step: (tap, extra, h_{t-1}) -> (mean, h_t)."""
        x = torch.cat([self.tap_norm(feats_tap), extra], dim=-1)
        h = self.gru(x, h)
        return self.pi(h), h

    def init_h(self, n):
        return torch.zeros(n, self.gru_hid, device=dev)


def warm_start_gru(ac, path):
    """Copy the frozen brain (incl body.input_projection) + tap_norm from an augmented-reservoir readout,
    so the tap features match a known-good flier. The GRU/pi/vf start fresh (shapes differ from an MLP)."""
    src = torch.load(path, map_location="cpu"); tgt = ac.state_dict(); n = 0
    for k, v in src.items():
        if k in tgt and tgt[k].shape == v.shape and (k.startswith("body.") or k.startswith("tap_norm.")):
            tgt[k].copy_(v); n += 1
    ac.load_state_dict(tgt)
    print(f"GRU warm-start: copied {n} frozen-brain/tap_norm tensors from {path}; GRU+pi+vf fresh", flush=True)


def train_gru(env_cls, extra_dim, N=384, budget=20e6, lr=3e-4, roll=32, chunk=16, epochs=3, mb_env=4,
              ent_coef=0.01, logstd_init=-0.5, logstd_final=-1.0, anneal_start=0.5,
              pos_curr_frac=0.4, warm_from=None, save_path=None):
    assert roll % chunk == 0, "roll must be a multiple of chunk"
    ac = K1ReservoirGRU(extra_dim).to(dev)
    if warm_from:
        warm_start_gru(ac, warm_from)
    trainable = [p for n, p in ac.named_parameters() if p.requires_grad and n != "log_std"]
    opt = torch.optim.Adam(trainable, lr=lr)
    env = env_cls(N)
    GAMMA, LAM, CLIP = 0.99, 0.95, 0.2
    n_chunks = roll // chunk
    print(f"K1ReservoirGRU: feat={ac.feat_dim} extra={extra_dim} gru_hid={ac.gru_hid} "
          f"trainable={sum(p.numel() for p in trainable)/1e3:.0f}k roll={roll} chunk={chunk} N={N}", flush=True)
    obs = env.obs(); ext = env.extra(); h = ac.init_h(N)
    total = 0; t0 = time.time()
    for it in range(1, 2_000_001):
        frac = total / budget
        with torch.no_grad():
            a = max(0.0, (frac - anneal_start) / max(1e-6, 1 - anneal_start))
            ac.log_std.fill_(logstd_init + (logstd_final - logstd_init) * min(1.0, a))
        if hasattr(env, "obs_pos_curr"):
            env.obs_pos_curr = min(1.0, frac / max(1e-6, pos_curr_frac))
        tap_b = torch.zeros(roll, N, ac.feat_dim, device=dev)
        ext_b = torch.zeros(roll, N, extra_dim, device=dev)
        hin_b = torch.zeros(roll, N, ac.gru_hid, device=dev)   # hidden state fed INTO each step (for replay)
        act_b = torch.zeros(roll, N, 4, device=dev); logp_b = torch.zeros(roll, N, device=dev)
        rew_b = torch.zeros(roll, N, device=dev); val_b = torch.zeros(roll, N, device=dev)
        done_b = torch.zeros(roll, N, device=dev)
        tilt_acc = []
        with torch.no_grad():
            for s in range(roll):
                feats = ac.features(obs)
                hin_b[s] = h
                mean, h = ac.step(feats, ext, h); std = ac.log_std.exp()
                val = ac.vf(ext).squeeze(-1)
                dist = torch.distributions.Normal(mean, std); act = dist.sample(); logp = dist.log_prob(act).sum(-1)
                nobs, rew, done, tilt = env.step(act); next_ext = env.extra()
                tap_b[s], ext_b[s], act_b[s], logp_b[s], rew_b[s], val_b[s], done_b[s] = \
                    feats, ext, act, logp, rew, val, done.float()
                h = h * (1.0 - done.float()).unsqueeze(-1)   # reset memory at episode boundary
                obs, ext = nobs, next_ext; tilt_acc.append(tilt.mean())
            last_val = ac.vf(ext).squeeze(-1)
        # GAE (memoryless critic)
        adv = torch.zeros(roll, N, device=dev); lg = torch.zeros(N, device=dev)
        for s in reversed(range(roll)):
            nv = last_val if s == roll - 1 else val_b[s + 1]; nt = 1 - done_b[s]
            delta = rew_b[s] + GAMMA * nv * nt - val_b[s]; lg = delta + GAMMA * LAM * nt * lg; adv[s] = lg
        ret = adv + val_b
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        # Recurrent PPO: replay each time-chunk with truncated BPTT, minibatching over envs.
        for _ in range(epochs):
            for c in range(n_chunks):
                s0 = c * chunk
                perm = torch.randperm(N, device=dev)
                for mbi in perm.chunk(mb_env):
                    h_r = hin_b[s0, mbi].detach()
                    pg_terms = []; v_terms = []; ent_terms = []
                    for s in range(s0, s0 + chunk):
                        mean, h_r = ac.step(tap_b[s, mbi], ext_b[s, mbi], h_r)
                        std = ac.log_std.exp()
                        val = ac.vf(ext_b[s, mbi]).squeeze(-1)
                        dist = torch.distributions.Normal(mean, std)
                        logp = dist.log_prob(act_b[s, mbi]).sum(-1)
                        ratio = (logp - logp_b[s, mbi]).exp()
                        a_s = advn[s, mbi]
                        pg_terms.append(-torch.min(ratio * a_s, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a_s))
                        v_terms.append((val - ret[s, mbi]).pow(2))
                        ent_terms.append(dist.entropy().sum(-1))
                        h_r = h_r * (1.0 - done_b[s, mbi]).unsqueeze(-1)   # same resets as rollout
                    pg = torch.stack(pg_terms).mean(); vl = 0.5 * torch.stack(v_terms).mean()
                    ent = torch.stack(ent_terms).mean()
                    loss = pg + 0.5 * vl - ent_coef * ent
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(trainable, 0.5); opt.step()
        total += roll * N
        if it % 10 == 0:
            fps = total / (time.time() - t0); extra_m = env.metric() if hasattr(env, "metric") else ""
            posc = getattr(env, "obs_pos_curr", 1.0)
            print(f"[gru] it {it:4d} steps {total/1e6:5.1f}M fps {fps:5.0f} ep_rew {rew_b.mean():6.3f} "
                  f"tilt {math.degrees(float(torch.stack(tilt_acc).mean())):5.1f}deg posc {posc:.2f} {extra_m}", flush=True)
        if it % 50 == 0 and save_path:
            torch.save(ac.state_dict(), save_path)
        if total > budget:
            break
    if save_path:
        torch.save(ac.state_dict(), save_path); print(f"[gru] saved {save_path}", flush=True)


if __name__ == "__main__":
    from gates_proc_env import BatchedProcCourse, EXTRA_DIM
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 384
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 20e6
    train_gru(BatchedProcCourse, EXTRA_DIM, N=N, budget=budget,
              warm_from="/workspace/drone-fly/training/proc-course-flight/model_readout.pt",
              save_path="/workspace/drone-fly/gpu_prototype/proc_gru.pt")
