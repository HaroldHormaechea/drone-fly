"""Standalone inference + reference plant for a trained K1 flight model.

This is the deploy-side counterpart to the training loop: load a `model_readout.pt`, call `act(obs)` to
get a raw 4-vector action, and (optionally) integrate flight with `plant_step()` — a pure-NumPy copy of
the simulator's CTBR dynamics, so a liftoff/deployment harness needs NO training-env import.

See docs/INFERENCE_CONTRACT.md for the obs/action/plant/file-format contract this implements.

    from inference import load_policy, init_state, plant_step, make_obs
    policy = load_policy("../training/oval-lap-flight/model_readout.pt", extra_dim=30)
    s = init_state(pos=(0, 0, 0.12))
    for _ in range(500):
        obs = make_obs(s, target=(2.0, 0.0, 1.0))
        a = policy.act(obs, extra=None)        # extra=None -> pure-reservoir; pass a vector for augmented
        s = plant_step(s, a)
"""
from __future__ import annotations
import numpy as np
import torch

# --- plant constants (mirrors hover_gpu.py; keep in sync via docs/INFERENCE_CONTRACT.md) ---
M = 0.032
G = 9.81
TW = 2.5
MAX_THRUST = TW * M * G
HOVER_THR = 1.0 / TW
MAX_BODY_RATE = 4.0
RATE_KP = 18.0
DT = 0.02
K1_PATH = "/workspace/drone-fly/artifacts/pruned/k1"


# ----------------------------- model -----------------------------
class Policy:
    """Thin wrapper: deterministic act(obs[, extra]) -> np.float32[4] raw action."""

    def __init__(self, actor, extra_dim, device):
        self.actor = actor
        self.extra_dim = extra_dim
        self.device = device

    @torch.no_grad()
    def act(self, obs, extra=None):
        o = torch.as_tensor(np.asarray(obs, np.float32), device=self.device).reshape(1, -1)
        feats = self.actor.tap(self.actor.propagated_full(o))      # brain tap (both actor families)
        if self.extra_dim is None:                                 # pure reservoir
            mean, _ = self.actor.act_from_feats(feats)
        else:                                                      # augmented: readout sees [tap || extra]
            e = torch.as_tensor(np.asarray(extra, np.float32), device=self.device).reshape(1, -1)
            mean, _ = self.actor.act(feats, e)
        return mean[0].cpu().numpy()


def _load_strict_false(actor, path, device):
    missing, unexpected = actor.load_state_dict(torch.load(path, map_location=device), strict=False)
    assert all(k.startswith("body.layer.") for k in missing), f"unexpected missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    actor.eval()


def load_policy(path, extra_dim=30, device="cpu"):
    """Load a stripped model_readout.pt. extra_dim=None -> pure reservoir (K1Reservoir);
    an int -> augmented reservoir (K1ReservoirAug) expecting that extra width."""
    dev = torch.device(device)
    if extra_dim is None:
        from gates_k1_reservoir import K1Reservoir
        actor = K1Reservoir().to(dev)
    else:
        from reservoir_aug import K1ReservoirAug
        actor = K1ReservoirAug(extra_dim).to(dev)
    _load_strict_false(actor, path, dev)
    return Policy(actor, extra_dim, dev)


# ----------------------------- reference plant (pure NumPy) -----------------------------
def quat_mul(q, r):
    w0, x0, y0, z0 = q; w1, x1, y1, z1 = r
    return np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ], np.float32)


def quat_rotate(q, v):
    """Rotate vector v by quaternion q=[w,x,y,z] (body->world)."""
    qv = np.array([0.0, *v], np.float32)
    qc = np.array([q[0], -q[1], -q[2], -q[3]], np.float32)
    return quat_mul(quat_mul(q, qv), qc)[1:]


def quat_rotate_inv(q, v):
    qc = np.array([q[0], -q[1], -q[2], -q[3]], np.float32)
    return quat_rotate(qc, v)


def init_state(pos=(0.0, 0.0, 0.12), vel=(0, 0, 0), quat=(1, 0, 0, 0), omega=(0, 0, 0)):
    return {"pos": np.array(pos, np.float32), "vel": np.array(vel, np.float32),
            "quat": np.array(quat, np.float32), "omega": np.array(omega, np.float32)}


def plant_step(s, action, dt=DT):
    """One 50 Hz CTBR step. action: raw [collective, wx, wy, wz] in ~[-1,1]. Returns a new state dict."""
    a = np.asarray(action, np.float32)
    thr = np.clip((a[0] + 1) * 0.5, 0, 1) * MAX_THRUST
    rate_cmd = np.tanh(a[1:4]) * MAX_BODY_RATE
    omega = s["omega"] + RATE_KP * (rate_cmd - s["omega"]) * dt
    quat = s["quat"] + 0.5 * quat_mul(s["quat"], np.array([0.0, *omega], np.float32)) * dt
    quat = quat / (np.linalg.norm(quat) + 1e-8)
    thrust_world = quat_rotate(quat, np.array([0.0, 0.0, thr], np.float32))
    acc = thrust_world / M + np.array([0.0, 0.0, -G], np.float32)
    vel = s["vel"] + acc * dt
    pos = s["pos"] + vel * dt
    return {"pos": pos, "vel": vel, "quat": quat, "omega": omega}


def make_obs(s, target):
    """Build the 12-dim obs (docs/INFERENCE_CONTRACT.md §1) from a state dict and a target position."""
    rel = np.asarray(target, np.float32) - s["pos"]
    g_body = quat_rotate_inv(s["quat"], np.array([0.0, 0.0, -1.0], np.float32))
    return np.concatenate([rel, g_body, s["vel"], s["omega"]]).astype(np.float32)


def tilt_deg(s):
    """Angle from upright, degrees (0 = level, 180 = inverted)."""
    g_body = quat_rotate_inv(s["quat"], np.array([0.0, 0.0, -1.0], np.float32))
    return float(np.degrees(np.arccos(np.clip(-g_body[2], -1.0, 1.0))))


if __name__ == "__main__":
    # smoke: load the oval model, fly toward a waypoint, report it climbs off the ground and stays upright.
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/drone-fly/training/oval-lap-flight/model_readout.pt"
    extra_dim = int(sys.argv[2]) if len(sys.argv) > 2 else 12   # oval/track=12, proc=30, proc+raycast=70
    pol = load_policy(path, extra_dim=extra_dim, device="cpu")
    s = init_state()
    for t in range(250):
        obs = make_obs(s, target=(2.0, 0.0, 1.2))
        extra = np.concatenate([obs, np.zeros(extra_dim - 12, np.float32)])  # obstacle-free basic flight
        s = plant_step(s, pol.act(obs, extra))
    print(f"after 5s: pos={s['pos'].round(2)} vel={s['vel'].round(2)} tilt={tilt_deg(s):.1f}deg")
