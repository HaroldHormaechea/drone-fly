"""Augmented frozen-K1 reservoir: the brain processes the 12-dim FLIGHT obs; the READOUT additionally
sees EXTRA features (obstacle vision, pad/battery state, ...).

The frozen K1 body has a fixed 12-dim input (input_projection = Linear(12, sensory)), so new SENSES
cannot be added to the brain without unfreezing it. Instead they are concatenated onto the readout
input: readout sees [LayerNorm(tap(1024)) || extra(E)]. The brain stays frozen and fully load-bearing
for flight; the small readout learns to combine the brain's flight features with the extra senses.

train_aug() mirrors gates_k1_reservoir.train() (feature caching, PPO, exploration annealing) but the
env must expose extra(): a (n, E) tensor of extra features each step, alongside the 12-dim obs().
"""
import sys, math, time
import torch, torch.nn as nn

from hover_gpu import HOVER_THR, dev
from gates_k1_reservoir import R, HID
from drone_fly.connectome import load_connectome
from drone_fly.controller.actor import ConnectomeActorNetwork

INIT_LOGSTD = -0.5
K1_PATH = "/workspace/drone-fly/artifacts/pruned/k1"


class K1ReservoirAug(nn.Module):
    """Frozen K1 body + readout over [normalized brain tap || extra senses]."""

    def __init__(self, extra_dim, obs_dim=12, act_dim=4):
        super().__init__()
        self.extra_dim = extra_dim
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
        # readout: [norm(tap) || extra] -> action; value: extra -> scalar. `extra` carries the raw
        # 12-dim flight obs plus any extra senses (obstacle vision, pad/battery), so the readout sees
        # both the brain's flight features AND the clean geometry/senses the harder courses need.
        self.pi = nn.Sequential(nn.Linear(self.feat_dim + extra_dim, HID), nn.Tanh(), nn.Linear(HID, act_dim))
        self.vf = nn.Sequential(nn.Linear(extra_dim, HID), nn.Tanh(),
                                nn.Linear(HID, HID), nn.Tanh(), nn.Linear(HID, 1))
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
        with torch.no_grad():
            self.pi[-1].bias[0] = 2 * HOVER_THR - 1

    @torch.no_grad()
    def propagated_full(self, obs):
        x = obs.to(torch.float32)
        projected = self.body.input_projection(x)
        state = torch.zeros(x.shape[0], self.N, dtype=torch.float32, device=x.device)
        state = state.index_add(1, self.body.sensory_index, projected)
        return self.body.layer(state)

    def tap(self, propagated):
        return propagated.index_select(1, self.tap_index)

    @torch.no_grad()
    def features(self, obs):
        return self.tap(self.propagated_full(obs))

    def act(self, feats_tap, extra):
        return self.pi(torch.cat([self.tap_norm(feats_tap), extra], dim=-1)), self.log_std.exp()

    def forward(self, obs, extra):
        feats = self.features(obs)
        mean, std = self.act(feats, extra)
        val = self.vf(extra).squeeze(-1)
        return mean, std, val


def warm_start(ac, path, src_extra_dim):
    """Initialize an augmented reservoir from a readout trained with FEWER extra senses.

    The readout input is [tap(R) || extra]; a source policy with `src_extra_dim` senses has its first
    R+src_extra_dim input columns copied in, and the NEW sense columns (e.g. obstacle vision) start at
    ZERO influence, so the policy begins with the source's exact flight behavior and only has to LEARN
    the new senses. Value head (input = extra) is warm-started the same way. Same-shape heads copy directly.
    """
    src = torch.load(path, map_location="cpu")
    tgt = ac.state_dict()
    R = ac.feat_dim
    for k, v in src.items():
        if k not in tgt:
            continue
        tv = tgt[k]
        if v.shape == tv.shape:
            tv.copy_(v)
        elif k == "pi.0.weight":                       # (HID, R+extra): copy [tap||src-extra], zero new
            tv.zero_(); tv[:, : R + src_extra_dim].copy_(v)
        elif k == "vf.0.weight":                       # (HID, extra): copy src-extra cols, zero new
            tv.zero_(); tv[:, :src_extra_dim].copy_(v)
    ac.load_state_dict(tgt)
    print(f"warm-started from {path} (src_extra_dim={src_extra_dim}, new senses zero-init)", flush=True)


def train_aug(env_cls, extra_dim, N=512, budget=20e6, lr=3e-4, roll=32, epochs=4, mb=8,
              ent_coef=0.01, logstd_final=-2.0, anneal_start=0.4, save_path=None,
              warm_from=None, warm_src_extra_dim=12, logstd_init=INIT_LOGSTD, curriculum_frac=None,
              pos_curr_frac=None, count_curr_frac=None):
    ac = K1ReservoirAug(extra_dim).to(dev)
    if warm_from is not None:
        warm_start(ac, warm_from, warm_src_extra_dim)
    trainable = [p for n, p in ac.named_parameters()
                 if p.requires_grad and not (logstd_final is not None and n == "log_std")]
    opt = torch.optim.Adam(trainable, lr=lr)
    env = env_cls(N)
    GAMMA, LAM, CLIP = 0.99, 0.95, 0.2
    print(f"K1ReservoirAug: feat_dim={ac.feat_dim} extra={extra_dim} "
          f"trainable={sum(p.numel() for p in trainable)/1e3:.1f}k frozen={sum(p.numel() for p in ac.parameters() if not p.requires_grad)/1e6:.2f}M N={N}", flush=True)
    obs = env.obs(); ext = env.extra()
    total_steps = 0; t0 = time.time()
    for it in range(1, 2_000_001):
        if logstd_final is not None:
            frac = total_steps / budget
            a = max(0.0, (frac - anneal_start) / max(1e-6, 1.0 - anneal_start))
            with torch.no_grad():
                ac.log_std.fill_(logstd_init + (logstd_final - logstd_init) * min(1.0, a))
        # Curriculum: ramp env.difficulty 0->1 over the first curriculum_frac of the budget (hazards
        # start easy/off-path and move in). Affects envs as they respawn, so difficulty rises smoothly.
        if curriculum_frac is not None and hasattr(env, "difficulty"):
            env.difficulty = min(1.0, (total_steps / budget) / curriculum_frac)
        # Position curriculum: slide obstacles off-path -> on-path over the first pos_curr_frac.
        if pos_curr_frac is not None and hasattr(env, "obs_pos_curr"):
            env.obs_pos_curr = min(1.0, (total_steps / budget) / pos_curr_frac)
        # Count curriculum: grow active obstacles 1 -> all over the first count_curr_frac (master one
        # in-path obstacle before facing multi-obstacle courses, the dominant failure mode).
        if count_curr_frac is not None and hasattr(env, "obs_count_curr"):
            env.obs_count_curr = min(1.0, (total_steps / budget) / count_curr_frac)
        tap_b = torch.zeros(roll, N, ac.feat_dim, device=dev)
        ext_b = torch.zeros(roll, N, extra_dim, device=dev)
        obs_b = torch.zeros(roll, N, 12, device=dev)
        act_b = torch.zeros(roll, N, 4, device=dev)
        logp_b = torch.zeros(roll, N, device=dev)
        rew_b = torch.zeros(roll, N, device=dev)
        val_b = torch.zeros(roll, N, device=dev)
        done_b = torch.zeros(roll, N, device=dev)
        tilt_acc = []
        with torch.no_grad():
            for s in range(roll):
                feats = ac.features(obs)
                mean, std = ac.act(feats, ext)
                val = ac.vf(ext).squeeze(-1)
                dist = torch.distributions.Normal(mean, std)
                act = dist.sample(); logp = dist.log_prob(act).sum(-1)
                nobs, rew, done, tilt = env.step(act)
                next_ext = env.extra()
                tap_b[s], ext_b[s], obs_b[s], act_b[s], logp_b[s], rew_b[s], val_b[s], done_b[s] = \
                    feats, ext, obs, act, logp, rew, val, done.float()
                obs = nobs; ext = next_ext
                tilt_acc.append(tilt.mean())
            last_val = ac.vf(ext).squeeze(-1)
        adv = torch.zeros(roll, N, device=dev); lastgae = torch.zeros(N, device=dev)
        for s in reversed(range(roll)):
            nextval = last_val if s == roll - 1 else val_b[s + 1]
            nonterm = 1.0 - done_b[s]
            delta = rew_b[s] + GAMMA * nextval * nonterm - val_b[s]
            lastgae = delta + GAMMA * LAM * nonterm * lastgae
            adv[s] = lastgae
        ret = adv + val_b
        bt = tap_b.reshape(-1, ac.feat_dim); be = ext_b.reshape(-1, extra_dim)
        bo = obs_b.reshape(-1, 12); ba = act_b.reshape(-1, 4)
        bl = logp_b.reshape(-1); badv = adv.reshape(-1); bret = ret.reshape(-1)
        badv = (badv - badv.mean()) / (badv.std() + 1e-8)
        idx_all = torch.randperm(bt.shape[0], device=dev)
        for _ in range(epochs):
            for mbi in idx_all.chunk(mb):
                mean, std = ac.act(bt[mbi], be[mbi])
                val = ac.vf(be[mbi]).squeeze(-1)
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
            print(f"[aug] it {it:4d} steps {total_steps/1e6:6.1f}M fps {fps:6.0f} "
                  f"ep_rew/step {rew_b.mean():6.3f} tilt {math.degrees(float(torch.stack(tilt_acc).mean())):5.1f}deg {extra}", flush=True)
        if it % 200 == 0 and save_path:
            torch.save(ac.state_dict(), save_path)
        if total_steps > budget:
            break
    if save_path:
        torch.save(ac.state_dict(), save_path)
        print(f"[aug] saved {save_path}", flush=True)
