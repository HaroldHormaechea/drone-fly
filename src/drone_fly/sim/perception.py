"""Simulated gate perception under a measured detector-noise model (UC70 D2).

Batched torch, two slots per env (the *target* and the *next* gate), with the live tracker's
semantics (``harness.liftoff.gate_cv.TrackingGateEstimator``) driven by the TRUE route progress.
Each control tick (:meth:`GatePerception.step`), in this order:

1. **Hand-over** on a truth ``k`` advance: the target becomes the next slot's estimate when one is
   known (``last_seen = now``, the slot's detection / dropout / AR state carried over), otherwise
   the target goes back to acquiring (``last_event = now``); the next slot restarts unseen.
2. **Frame arrival**: a per-env frame clock (period ``T_f`` + Gaussian jitter, a per-frame skip
   probability) from the model's ``timing``; the detection latency ``L`` is drawn per episode.
   An arriving frame is evaluated at the **delayed pose** (``t - L``, from a ring of the last
   :data:`RING` ticks' poses).
3. **Frustum** (:func:`visibility_torch`, the torch twin of
   :func:`drone_fly.sim.noise_model.visibility`): ``p_det = 0`` outside it.
4. **Markov dropout** (two states, stationary ``p_det``, mean miss-burst length per range bin).
5. **Detected**: the error (bias + AR(1) noise, line-of-sight frame) is applied to the gate at the
   delayed viewpoint and stored in the delayed BODY frame; while the frame's detection persists
   (until the next frame) the world estimate is recomputed each tick with the CURRENT pose, i.e.
   ``pos_now + R_now R_{t-L}^T (g + e - pos_{t-L})`` — exactly the harness's "current pose for a
   late image" transform. Yaw likewise; the aperture is the class nominal; ``last_seen = now``.
6. **Hold / lost / acquire**: held estimates; ``lost`` after ``lost_timeout_s`` without a match;
   before acquisition the target is the drone's own position, ``curr_gate = [1, 0, nominal]``,
   lost after ``acquire_timeout_s`` (``lost_timeout_s`` after a hand-over with no next known).
7. **False positives** (Poisson rate per frame): while the target is unacquired a false positive
   can be acquired as the target (then held: it never re-detects, so it ends lost unless the truth
   route advances first); at the measured next-gate mis-association rate the next slot takes a
   false positive. Samples come from the fitted (range, azimuth) histogram.

**Lost policy** (``PerceptionConfig.lost_policy``; the live tracker has the same switch):

* ``terminate`` (legacy, the default): a held target unmatched for ``lost_timeout_s`` is ``lost``
  and the episode ends (``gate_lost``); before the first acquisition, after ``acquire_timeout_s``.
* ``reacquire`` (UC70 amendment, ADR 0009): (a) a held target unmatched for more than
  ``drop_after_s`` may be replaced by the acquisition rule on any frame that has a candidate (a
  true detection, or a false positive with the fitted pick probability); (b) after
  ``lost_timeout_s`` unmatched it is RELEASED: its last estimate stays the target
  (``target_valid`` 0) and the slot acquires again. ``lost`` only when nothing good (a match, an
  acquisition or a pass) happened for ``lost_hard_stop_s``.

Modes: ``noise`` (the model) and ``exact`` (same tracker + frustum + frame clock; every in-frustum
frame detected, no error, no latency, no false positive).

Randomness comes from a fixed per-tick draw layout (:data:`TICK_DRAWS`, :data:`EPISODE_DRAWS`) so
the NumPy reference (``drone_fly.student.perception_ref``) can replay identical draws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from drone_fly.sim.noise_model import (
    DEFAULT_MAX_RANGE_M,
    DEFAULT_MIN_IN_FRAME_FRAC,
    MIN_Z_CAM_M,
    VIS_BEHIND,
    VIS_CENTRE_OUT,
    VIS_MASKED,
    VIS_OK,
    VIS_OUTLINE,
    VIS_RANGE,
    NoiseModel,
    body_to_camera,
)

#: Ticks of pose history (covers the measured detection lag at the slowest control rate).
RING = 16
MODES = ("noise", "exact")
LOST_POLICIES = ("terminate", "reacquire")
#: Status codes (as the harness's ``tracking`` / ``holding`` / ``lost``).
TRACKING, HOLDING, LOST = 0, 1, 2
STATUS_NAMES = ("tracking", "holding", "lost")
#: ``next_mode``: visible this frame, unseen (bearing to its last estimate), unseen toward the
#: target, unseen with nothing known (+x).
NEXT_VISIBLE, NEXT_UNSEEN_KNOWN, NEXT_UNSEEN_TARGET, NEXT_UNSEEN_NONE = range(4)
#: Per-tick draws: (name, width, kind) with kind "u" (uniform [0, 1)) or "n" (standard normal).
TICK_DRAWS: tuple[tuple[str, int, str], ...] = (
    ("jitter", 1, "n"),
    ("skip", 1, "u"),
    ("mk_t", 1, "u"),
    ("mk_n", 1, "u"),
    ("eps_t", 4, "n"),
    ("eps_n", 4, "n"),
    ("fp_any", 1, "u"),
    ("fp_pick", 1, "u"),
    ("fp_bin", 1, "u"),
    ("fp_r", 1, "u"),
    ("fp_a", 1, "u"),
    ("next_mis", 1, "u"),
)
EPISODE_DRAWS: tuple[tuple[str, int, str], ...] = (("lat", 1, "u"), ("phase", 1, "u"))


def draw_width(spec: tuple[tuple[str, int, str], ...]) -> int:
    return sum(w for _, w, _ in spec)


def split_draws(x: torch.Tensor, spec: tuple[tuple[str, int, str], ...]) -> dict[str, torch.Tensor]:
    """``(n, W)`` packed draws -> ``{name: (n,) or (n, w)}``."""
    out, i = {}, 0
    for name, w, _ in spec:
        out[name] = x[:, i] if w == 1 else x[:, i : i + w]
        i += w
    return out


def sample_draws(
    n: int,
    spec: tuple[tuple[str, int, str], ...],
    gen: torch.Generator | None,
    device: Any,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``(n, W)`` packed draws in ``spec`` order."""
    cols = []
    for _, w, kind in spec:
        if kind == "u":
            cols.append(torch.rand(n, w, generator=gen, device=device, dtype=dtype))
        else:
            cols.append(torch.randn(n, w, generator=gen, device=device, dtype=dtype))
    return torch.cat(cols, dim=1)


# ----------------------------------------------------------------------------- geometry


def quat_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        dim=-2,
    )


def _mv(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (m * v.unsqueeze(-2)).sum(-1)


def _mtv(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (m * v.unsqueeze(-1)).sum(-2)


@dataclass(frozen=True)
class TorchCamera:
    width: float
    height: float
    fx: float
    fy: float
    cx: float
    cy: float
    r_cb: tuple[tuple[float, ...], ...]
    masks: tuple[tuple[float, float, float, float], ...]
    min_in_frame_frac: float
    max_range_m: float

    @classmethod
    def from_block(cls, cam: dict[str, Any]) -> TorchCamera:
        w, h = (float(v) for v in cam["resolution"])
        fx = (w / 2.0) / math.tan(math.radians(float(cam["hfov_deg"])) / 2.0)
        r = body_to_camera(float(cam.get("uptilt_deg", 0.0)))
        return cls(
            w, h, fx, fx, w / 2.0, h / 2.0,
            tuple(tuple(float(x) for x in row) for row in r),
            tuple(tuple(float(x) for x in m) for m in cam.get("masks_norm") or []),  # type: ignore[misc]
            float(cam.get("min_in_frame_frac", DEFAULT_MIN_IN_FRAME_FRAC)),
            float(cam.get("max_range_m", DEFAULT_MAX_RANGE_M)),
        )  # fmt: skip


def to_camera_torch(p_w: torch.Tensor, pos: torch.Tensor, R: torch.Tensor, cam: TorchCamera
                    ) -> torch.Tensor:  # fmt: skip
    """World points ``(n, ..., 3)`` -> camera frame for poses ``pos (n, 3)``, ``R (n, 3, 3)``."""
    rcb = torch.tensor(cam.r_cb, dtype=p_w.dtype, device=p_w.device)
    extra = p_w.dim() - 2
    d = p_w - pos.view(pos.shape[0], *([1] * extra), 3)
    Rv = R.view(R.shape[0], *([1] * extra), 3, 3)
    body = _mtv(Rv, d)
    return (rcb * body.unsqueeze(-2)).sum(-1)


def project_torch(p_cam: torch.Tensor, cam: TorchCamera) -> tuple[torch.Tensor, torch.Tensor]:
    z = p_cam[..., 2]
    zs = torch.where(z > 1e-6, z, torch.ones_like(z))
    return cam.fx * p_cam[..., 0] / zs + cam.cx, cam.fy * p_cam[..., 1] / zs + cam.cy


def _in_image(u: torch.Tensor, v: torch.Tensor, front: torch.Tensor, cam: TorchCamera
              ) -> torch.Tensor:  # fmt: skip
    return front & (u >= 0) & (u <= cam.width - 1) & (v >= 0) & (v <= cam.height - 1)


def gate_corners_torch(center: torch.Tensor, rot: torch.Tensor, wh: torch.Tensor) -> torch.Tensor:
    """``(n, 4, 3)`` aperture corners (``rot`` columns: across, up, normal)."""
    signs = torch.tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=center.dtype,
                         device=center.device)  # fmt: skip
    hw = 0.5 * wh[:, 0:1]
    hh = 0.5 * wh[:, 1:2]
    a = signs[:, 0].unsqueeze(0) * hw  # (n, 4)
    b = signs[:, 1].unsqueeze(0) * hh
    return (center.unsqueeze(1) + a.unsqueeze(-1) * rot[:, :, 0].unsqueeze(1)
            + b.unsqueeze(-1) * rot[:, :, 1].unsqueeze(1))  # fmt: skip


def visibility_torch(
    center: torch.Tensor,
    corners: torch.Tensor,
    pos: torch.Tensor,
    R: torch.Tensor,
    cam: TorchCamera,
    range_scale: torch.Tensor | float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(code (n,), range_m (n,), edge (n,))`` — the torch twin of ``noise_model.visibility``.

    Positions may be in course units: ``range_scale`` (course unit -> m) applies to the range
    test only (the projection is scale-free).
    """
    c_cam = to_camera_torch(center, pos, R, cam)
    k_cam = to_camera_torch(corners, pos, R, cam)
    rng = (center - pos).norm(dim=-1) * range_scale
    front = c_cam[:, 2] * range_scale > MIN_Z_CAM_M
    u, v = project_torch(c_cam, cam)
    centre_in = _in_image(u, v, front, cam)
    ku, kv = project_torch(k_cam, cam)
    frac = _in_image(ku, kv, k_cam[..., 2] > 1e-6, cam).to(center.dtype).mean(-1)
    masked = torch.zeros_like(front)
    un, vn = u / cam.width, v / cam.height
    for x0, y0, x1, y1 in cam.masks:
        masked = masked | ((un >= x0) & (un < x1) & (vn >= y0) & (vn < y1))
    code = torch.full_like(rng, VIS_OK, dtype=torch.long)
    code = torch.where(masked, VIS_MASKED, code)
    code = torch.where(frac < cam.min_in_frame_frac, VIS_OUTLINE, code)
    code = torch.where(~centre_in, VIS_CENTRE_OUT, code)
    code = torch.where(rng > cam.max_range_m, VIS_RANGE, code)
    code = torch.where(~front, VIS_BEHIND, code)
    un_c, vn_c = u / cam.width, v / cam.height
    edge = (2.0 * torch.minimum(torch.minimum(un_c, 1.0 - un_c), torch.minimum(vn_c, 1.0 - vn_c))
            ).clamp(0.0, 1.0)  # fmt: skip
    return code, rng, edge


def los_frame_torch(rel: torch.Tensor) -> torch.Tensor:
    """``(n, 3, 3)`` rows radial, tangential-h, tangential-v (twin of ``noise_model.los_frame``)."""
    radial = rel / rel.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    up = torch.zeros_like(radial)
    up[:, 2] = 1.0
    th = torch.cross(up, radial, dim=-1)
    nh = th.norm(dim=-1, keepdim=True)
    fb = torch.zeros_like(radial)
    fb[:, 1] = 1.0
    th = torch.where(nh > 1e-6, th / nh.clamp(min=1e-9), fb)
    tv = torch.cross(radial, th, dim=-1)
    return torch.stack([radial, th, tv], dim=-2)


# ----------------------------------------------------------------------------- perception


@dataclass(frozen=True)
class PerceptionConfig:
    """Tracker + timing parameters (what the student checkpoint pins)."""

    mode: str = "noise"
    lost_timeout_s: float = 1.0
    acquire_timeout_s: float = 3.0
    frame_period_s: float = 0.033
    frame_jitter_s: float = 0.005
    frame_skip_p: float = 0.0
    detection_lag_s: tuple[float, float] = (0.03, 0.06)
    #: Course-frame nominal aperture of the detector's class (what ``curr_gate`` reports).
    nominal_aperture: float = 3.69
    #: The tracker's association gate (course units): a detection re-associates with a held
    #: target only within ``assoc_radius * max(1, aperture)`` of it, both in the harness's
    #: training scale ``k`` -> ``k * assoc_radius * max(1, nominal / k)`` here. ``inf`` = off.
    assoc_limit: float = math.inf
    #: ``terminate`` | ``reacquire`` (module docstring).
    lost_policy: str = "terminate"
    lost_hard_stop_s: float = 6.0
    drop_after_s: float = 0.5

    def __post_init__(self) -> None:
        if self.lost_policy not in LOST_POLICIES:
            raise ValueError(
                f"lost_policy must be one of {LOST_POLICIES}, got {self.lost_policy!r}"
            )
        if self.lost_hard_stop_s <= 0 or self.drop_after_s < 0:
            raise ValueError("lost_hard_stop_s > 0 and drop_after_s >= 0")
        if self.mode not in MODES:
            raise ValueError(f"perception.mode must be one of {MODES}, got {self.mode!r}")
        if self.lost_timeout_s <= 0 or self.acquire_timeout_s < 0 or self.frame_period_s <= 0:
            raise ValueError("lost_timeout_s > 0, acquire_timeout_s >= 0, frame_period_s > 0")

    @classmethod
    def from_model(cls, model: NoiseModel | None, mode: str, **over: Any) -> PerceptionConfig:
        kw: dict[str, Any] = {"mode": mode}
        k = float(over.pop("world_scale_k", 1.0))
        if model is not None:
            tm = model.timing()
            tr = model.tracker
            rec = (model.data.get("recommended") or {}).get("lost_timeout_s")
            ap = model.class_noise().aperture_m
            kw.update(
                lost_timeout_s=float(rec if rec is not None else tr.get("lost_timeout_s", 1.0)),
                acquire_timeout_s=float(tr.get("acquire_timeout_s", 3.0)),
                frame_period_s=float(tm["frame_period_s"]),
                frame_jitter_s=float(tm.get("frame_jitter_s") or 0.0),
                frame_skip_p=float(tm.get("frame_skip_p") or 0.0),
                detection_lag_s=tuple(float(x) for x in tm["detection_lag_s"]),
                nominal_aperture=float(min(ap)),
                assoc_limit=assoc_limit(float(tr.get("assoc_radius", 1.0)), float(min(ap)), k),
            )
        kw.update(over)
        return cls(**kw)


def assoc_limit(assoc_radius: float, nominal_aperture: float, k: float) -> float:
    """The tracker's association distance in course units for a harness world scale ``k``."""
    return float(k) * assoc_radius * max(1.0, nominal_aperture / float(k))


class GatePerception:
    """See the module docstring. Positions are course units; ``scale`` (n,) = metres per unit."""

    def __init__(
        self,
        model: NoiseModel | None,
        n: int,
        device: Any,
        *,
        config: PerceptionConfig,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if config.mode == "noise" and model is None:
            raise ValueError("perception mode 'noise' needs a noise model")
        self.cfg = config
        self.model = model
        self.n = n
        self.device = torch.device(device)
        self.dtype = dtype
        self.gen = generator
        cam_block = model.camera if model is not None else None
        if cam_block is None:
            raise ValueError("GatePerception needs a noise model for the camera block")
        self.cam = TorchCamera.from_block(cam_block)
        self.tab = model.to_tables(self.device, dtype) if model is not None else {}
        self.exact = config.mode == "exact"
        self.reacq = config.lost_policy == "reacquire"
        z = lambda *s: torch.zeros(*s, dtype=dtype, device=self.device)  # noqa: E731
        b = lambda: torch.zeros(n, dtype=torch.bool, device=self.device)  # noqa: E731
        self.ring_t = torch.zeros(n, RING, dtype=torch.float64, device=self.device)
        self.ring_pos = z(n, RING, 3)
        self.ring_quat = z(n, RING, 4)
        self.ring_i = 0
        self.lat = z(n)
        self.next_frame_t = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.t0 = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.last_event = torch.full((n,), -math.inf, dtype=torch.float64, device=self.device)
        # target slot
        self.t_acq, self.t_fp, self.t_active, self.t_miss = b(), b(), b(), b()
        self.t_relb, self.t_nb, self.t_world = z(n, 3), z(n, 3), z(n, 3)
        self.t_yaw = z(n)
        self.t_seen = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.t_src = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.t_z = z(n, 4)
        # next slot
        self.n_known, self.n_fp, self.n_active, self.n_miss = b(), b(), b(), b()
        self.n_relb, self.n_nb, self.n_world = z(n, 3), z(n, 3), z(n, 3)
        self.n_yaw = z(n)
        self.n_seen = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.n_src = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.n_z = z(n, 4)
        self.ticks_since = torch.zeros(n, dtype=torch.long, device=self.device)
        # reacquire policy: a released target's last estimate is kept; the last good event time
        self.t_last = b()
        self.t_good = torch.zeros(n, dtype=torch.float64, device=self.device)
        self.hand_over = b()
        self.passed = b()
        self.status = torch.full((n,), HOLDING, dtype=torch.long, device=self.device)

    # ---- helpers ----
    def _draw(self, spec: tuple, draws: torch.Tensor | None) -> dict[str, torch.Tensor]:
        x = draws if draws is not None else sample_draws(self.n, spec, self.gen, self.device,
                                                         self.dtype)  # fmt: skip
        return split_draws(x.to(self.dtype), spec)

    def reset(self, mask: torch.Tensor, t: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor,
              draws: torch.Tensor | None = None) -> None:  # fmt: skip
        """Clear both slots, refill the pose ring, redraw latency / frame phase / dropout / AR."""
        d = self._draw(EPISODE_DRAWS, draws)
        m = mask
        m1 = m.unsqueeze(-1)
        t = t.to(torch.float64)
        c = self.cfg
        lat = (
            0.0
            if self.exact
            else c.detection_lag_s[0] + (c.detection_lag_s[1] - c.detection_lag_s[0]) * d["lat"]
        )
        self.lat = torch.where(m, torch.as_tensor(lat, dtype=self.dtype, device=self.device)
                               .expand(self.n), self.lat)  # fmt: skip
        phase = t + c.frame_period_s * d["phase"].to(torch.float64)
        self.next_frame_t = torch.where(m, phase, self.next_frame_t)
        self.ring_t = torch.where(m1, t.unsqueeze(-1).expand(-1, RING), self.ring_t)
        self.ring_pos = torch.where(m1.unsqueeze(-1), pos.to(self.dtype).unsqueeze(1)
                                    .expand(-1, RING, -1), self.ring_pos)  # fmt: skip
        self.ring_quat = torch.where(m1.unsqueeze(-1), quat.to(self.dtype).unsqueeze(1)
                                     .expand(-1, RING, -1), self.ring_quat)  # fmt: skip
        self.t0 = torch.where(m, t, self.t0)
        self.last_event = torch.where(m, torch.full_like(t, -math.inf), self.last_event)
        for name in ("t_acq", "t_fp", "t_active", "t_miss", "n_known", "n_fp", "n_active",
                     "n_miss", "hand_over", "passed", "t_last"):  # fmt: skip
            setattr(self, name, getattr(self, name) & ~m)
        for name in ("t_relb", "t_nb", "t_world", "t_z", "n_relb", "n_nb", "n_world", "n_z"):
            v = getattr(self, name)
            setattr(self, name, torch.where(m1, torch.zeros_like(v), v))
        for name in ("t_yaw", "n_yaw"):
            v = getattr(self, name)
            setattr(self, name, torch.where(m, torch.zeros_like(v), v))
        for name in ("t_seen", "t_src", "n_seen", "n_src", "t_good"):
            setattr(self, name, torch.where(m, t, getattr(self, name)))
        self.ticks_since = torch.where(m, torch.zeros_like(self.ticks_since), self.ticks_since)
        self.status = torch.where(m, torch.full_like(self.status, HOLDING), self.status)
        p = pos.to(self.dtype)
        self._pos = torch.where(m1, p, self._pos) if hasattr(self, "_pos") else p.clone()

    def push_pose(self, t: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor) -> None:
        i = self.ring_i % RING
        self.ring_t[:, i] = t.to(torch.float64)
        self.ring_pos[:, i] = pos.to(self.dtype)
        self.ring_quat[:, i] = quat.to(self.dtype)
        self.ring_i += 1

    def delayed_pose(self, tc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The newest ring pose at or before ``tc`` (else the oldest)."""
        ok = self.ring_t <= tc.unsqueeze(-1) + 1e-9
        tt = torch.where(ok, self.ring_t, torch.full_like(self.ring_t, -math.inf))
        idx = tt.argmax(dim=1)
        none = ~ok.any(dim=1)
        idx = torch.where(none, self.ring_t.argmin(dim=1), idx)
        ar = torch.arange(self.n, device=self.device)
        return self.ring_pos[ar, idx], self.ring_quat[ar, idx]

    # ---- noise ----
    def _p_det(self, rng: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        tab = self.tab
        if tab["binned"]:
            i = torch.searchsorted(tab["bin_edges"], rng.contiguous(), right=True)
            return tab["bin_p_det"][i]
        a, b, c = tab["p_det_coef"].unbind(0)
        return torch.sigmoid(a + b * rng + c * edge)

    def _burst_frames(self, rng: torch.Tensor) -> torch.Tensor:
        tab = self.tab
        i = torch.searchsorted(tab["burst_edges"], rng.contiguous(), right=True) - 1
        i = i.clamp(0, tab["burst_s"].shape[0] - 1)
        return (tab["burst_s"][i] / self.cfg.frame_period_s).clamp(min=1.0)

    def _error(self, rng: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``(n, 4)`` [radial, th, tv (m), yaw (rad)] errors for AR states ``z``."""
        tab = self.tab
        if tab["binned"]:
            i = torch.searchsorted(tab["bin_edges"], rng.contiguous(), right=True)
            return tab["bin_bias"][i] + (tab["bin_chol"][i] * z.unsqueeze(-2)).sum(-1)
        s = tab["sigma_per_m"]
        bias = torch.stack(
            [
                tab["radial_bias_per_m"] * rng,
                torch.zeros_like(rng),
                torch.zeros_like(rng),
                torch.full_like(rng, tab["yaw_bias"]),
            ],
            -1,
        )
        sig = torch.cat([rng.unsqueeze(-1) * s, torch.full_like(rng, tab["yaw_sigma"])
                         .unsqueeze(-1)], -1)  # fmt: skip
        return bias + sig * z

    def _detect(
        self,
        center: torch.Tensor,
        rot: torch.Tensor,
        wh: torch.Tensor,
        valid: torch.Tensor,
        pd: torch.Tensor,
        Rd: torch.Tensor,
        scale: torch.Tensor,
        miss_prev: torch.Tensor,
        z_prev: torch.Tensor,
        u_mk: torch.Tensor,
        eps: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """One slot on a frame: ``(detected, miss_state, z, rel_body, normal_body)``."""
        corners = gate_corners_torch(center, rot, wh)
        code, rng, edge = visibility_torch(center, corners, pd, Rd, self.cam, scale)
        vis = (code == VIS_OK) & valid
        n_true = rot[:, :, 2]
        if self.exact:
            det = vis
            miss = ~det
            z = z_prev
            e = torch.zeros(self.n, 4, dtype=self.dtype, device=self.device)
        else:
            p = self._p_det(rng, edge).clamp(1e-4, 1.0)
            # lengthen bursts that are too short for p (noise_model.effective_burst)
            L = torch.maximum(self._burst_frames(rng), (1.0 - p) / p)
            stay = 1.0 - 1.0 / L
            d2m = ((1.0 - p) / p * (1.0 - stay)).clamp(0.0, 1.0)
            p_miss = torch.where(miss_prev, stay, d2m)
            det = vis & (u_mk >= p_miss)
            miss = ~det
            tau = self.tab["ar1_tau_s"]
            rho = math.exp(-self.cfg.frame_period_s / tau) if tau > 0 else 0.0
            z = rho * z_prev + math.sqrt(max(1.0 - rho * rho, 0.0)) * eps
            e = self._error(rng, z)
        rel = center - pd
        e_world = _mtv(los_frame_torch(rel), e[:, :3]) / scale.unsqueeze(-1)
        est = center + e_world
        yaw = torch.atan2(n_true[:, 1], n_true[:, 0]) + e[:, 3]
        n_est = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], -1)
        return det, miss, z, _mtv(Rd, est - pd), _mtv(Rd, n_est)

    def _fp_sample(self, d: dict[str, torch.Tensor], pd: torch.Tensor, Rd: torch.Tensor,
                   scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
        """A false-positive detection (body-frame rel + normal at the delayed pose)."""
        tab = self.tab
        cdf = torch.cumsum(tab["fp_probs"], 0)
        k = torch.searchsorted(cdf, d["fp_bin"].contiguous().clamp(max=cdf[-1] - 1e-6))
        nr, na = tab["fp_shape"]
        ri, ai = k // na, k % na
        re, ae = tab["fp_range_edges"], tab["fp_az_edges_deg"]
        r = re[ri] + (re[ri + 1] - re[ri]) * d["fp_r"]
        a = torch.deg2rad(ae[ai] + (ae[ai + 1] - ae[ai]) * d["fp_a"])
        p_cam = torch.stack([r * torch.sin(a), torch.zeros_like(r), r * torch.cos(a)], -1)
        rcb = torch.tensor(self.cam.r_cb, dtype=self.dtype, device=self.device)
        body = (rcb.T * p_cam.unsqueeze(-2)).sum(-1) / scale.unsqueeze(-1)
        nb = body.clone()
        nb[:, 2] = 0.0  # facing along the line of sight, horizontal in the body frame
        nb = nb / nb.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return body, nb

    # ---- the tick ----
    def step(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        quat: torch.Tensor,
        target: dict[str, torch.Tensor],
        nxt: dict[str, torch.Tensor],
        next_valid: torch.Tensor,
        crossed: torch.Tensor,
        scale: torch.Tensor,
        draws: torch.Tensor | None = None,
    ) -> None:
        """Advance one control tick (see the module docstring); ``pos``/``quat`` = this tick."""
        d = self._draw(TICK_DRAWS, draws)
        t = t.to(torch.float64)
        pos = pos.to(self.dtype)
        R = quat_matrix(quat.to(self.dtype))
        c = self.cfg
        scale = scale.to(self.dtype)
        # 1. hand-over on a truth advance
        ho = crossed
        take = ho & self.n_known
        self.hand_over = ho
        self.passed = ho
        tw = take.unsqueeze(-1)
        self.t_acq = torch.where(ho, take, self.t_acq)
        self.t_fp = torch.where(ho, take & self.n_fp, self.t_fp)
        self.t_active = torch.where(ho, take & self.n_active, self.t_active)
        self.t_miss = torch.where(ho, torch.where(take, self.n_miss, self.t_miss), self.t_miss)
        self.t_relb = torch.where(tw, self.n_relb, self.t_relb)
        self.t_nb = torch.where(tw, self.n_nb, self.t_nb)
        self.t_world = torch.where(tw, self.n_world, self.t_world)
        self.t_yaw = torch.where(take, self.n_yaw, self.t_yaw)
        self.t_z = torch.where(tw, self.n_z, self.t_z)
        self.t_seen = torch.where(take, t, self.t_seen)
        self.t_src = torch.where(take, self.n_src, self.t_src)
        self.last_event = torch.where(ho, t, self.last_event)
        if self.reacq:  # a pass is a good event; the passed gate's estimate is no target
            self.t_good = torch.where(ho, t, self.t_good)
            self.t_last = self.t_last & ~ho
        for name in ("n_known", "n_fp", "n_active", "n_miss"):
            setattr(self, name, getattr(self, name) & ~ho)
        self.n_z = torch.where(ho.unsqueeze(-1), torch.zeros_like(self.n_z), self.n_z)
        # 2. frame arrival
        arrive = t >= self.next_frame_t - 1e-9
        step_t = (c.frame_period_s + c.frame_jitter_s * d["jitter"].to(torch.float64)).clamp(
            min=0.25 * c.frame_period_s)  # fmt: skip
        self.next_frame_t = torch.where(arrive, torch.maximum(self.next_frame_t + step_t, t),
                                        self.next_frame_t)  # fmt: skip
        frame = arrive & (d["skip"] >= c.frame_skip_p)
        tc = t - self.lat.to(torch.float64)
        pd, qd = self.delayed_pose(tc)
        Rd = quat_matrix(qd)
        det_t, miss_t, z_t, relb_t, nb_t = self._detect(
            target["center"], target["rot"], target["wh"], torch.ones_like(frame), pd, Rd, scale,
            self.t_miss, self.t_z, d["mk_t"], d["eps_t"],
        )  # fmt: skip
        det_n, miss_n, z_n, relb_n, nb_n = self._detect(
            nxt["center"], nxt["rot"], nxt["wh"], next_valid, pd, Rd, scale, self.n_miss,
            self.n_z, d["mk_n"], d["eps_n"],
        )  # fmt: skip
        if self.exact:
            any_fp = wrong = torch.zeros_like(frame)
        else:
            any_fp = d["fp_any"] < 1.0 - math.exp(-self.tab["fp_rate"])
            wrong = d["fp_any"] < self.tab["acq_wrong_present"]
        fp_relb, fp_nb = (self._fp_sample(d, pd, Rd, scale) if not self.exact
                          else (relb_t, nb_t))  # fmt: skip
        # Markov / AR state advances on every processed frame
        self.t_miss = torch.where(frame & ~self.t_fp, miss_t, self.t_miss)
        self.t_z = torch.where(frame.unsqueeze(-1), z_t, self.t_z)
        self.n_miss = torch.where(frame, miss_n, self.n_miss)
        self.n_z = torch.where(frame.unsqueeze(-1), z_n, self.n_z)
        # target slot: acquisition (true or false positive) / association
        unacq = frame & ~self.t_acq
        if self.reacq:  # (a) a stale held target re-acquires from this frame's candidates
            stale = self.t_acq & ~self.t_active & (t - self.t_seen > c.drop_after_s)
            unacq = unacq | (frame & stale & (det_t | wrong))
        pick_fp = unacq & wrong & (~det_t | (d["fp_pick"] < self.tab.get("acq_pick_wrong", 0.5)))
        acq_true = unacq & det_t & ~pick_fp
        new_t = pos + _mv(R, relb_t)
        assoc = (new_t - self.t_world).norm(dim=-1) <= c.assoc_limit
        upd_t = (frame & self.t_acq & ~self.t_fp & det_t & assoc) | acq_true
        self.t_active = torch.where((frame & ~self.t_fp) | acq_true, upd_t, self.t_active)
        self.t_active = torch.where(pick_fp, torch.zeros_like(pick_fp), self.t_active)
        self.t_relb = torch.where(upd_t.unsqueeze(-1), relb_t, self.t_relb)
        self.t_nb = torch.where(upd_t.unsqueeze(-1), nb_t, self.t_nb)
        self.t_src = torch.where(upd_t | pick_fp, tc, self.t_src)
        self.t_acq = self.t_acq | acq_true | pick_fp
        self.t_fp = torch.where(acq_true, torch.zeros_like(acq_true), self.t_fp | pick_fp)
        if self.reacq:
            self.t_good = torch.where(acq_true | pick_fp, t, self.t_good)
            self.t_last = self.t_last & ~(acq_true | pick_fp)
        # a false-positive target is fixed in the world at its sighting
        fp_world = pd + _mv(Rd, fp_relb)
        fp_n = _mv(Rd, fp_nb)
        self.t_world = torch.where(pick_fp.unsqueeze(-1), fp_world, self.t_world)
        self.t_yaw = torch.where(pick_fp, torch.atan2(fp_n[:, 1], fp_n[:, 0]), self.t_yaw)
        self.t_seen = torch.where(pick_fp, t, self.t_seen)
        # next slot (only while a target is held)
        nframe = frame & self.t_acq & ~pick_fp
        mis = nframe & any_fp & (d["next_mis"] < (self.tab.get("fp_next_misassoc", 0.0)
                                                  if not self.exact else 0.0))  # fmt: skip
        upd_n = nframe & det_n & ~mis
        self.n_active = torch.where(nframe, upd_n | mis, self.n_active)
        self.n_relb = torch.where(
            upd_n.unsqueeze(-1), relb_n, torch.where(mis.unsqueeze(-1), fp_relb, self.n_relb)
        )
        self.n_nb = torch.where(
            upd_n.unsqueeze(-1), nb_n, torch.where(mis.unsqueeze(-1), fp_nb, self.n_nb)
        )
        self.n_src = torch.where(upd_n | mis, tc, self.n_src)
        self.n_fp = torch.where(upd_n | mis, mis, self.n_fp)
        self.n_known = self.n_known | upd_n | mis
        # per tick: an active detection re-places the estimate with the CURRENT pose
        a_t = self.t_active & self.t_acq
        w_t = pos + _mv(R, self.t_relb)
        n_t = _mv(R, self.t_nb)
        self.t_world = torch.where(a_t.unsqueeze(-1), w_t, self.t_world)
        self.t_yaw = torch.where(a_t, _travel_yaw(n_t, w_t - pos), self.t_yaw)
        self.t_seen = torch.where(a_t, t, self.t_seen)
        a_n = self.n_active & self.n_known
        w_n = pos + _mv(R, self.n_relb)
        n_n = _mv(R, self.n_nb)
        self.n_world = torch.where(a_n.unsqueeze(-1), w_n, self.n_world)
        self.n_yaw = torch.where(a_n, _travel_yaw(n_n, w_n - pos), self.n_yaw)
        self.n_seen = torch.where(a_n, t, self.n_seen)
        self.ticks_since = torch.where(a_t, torch.zeros_like(self.ticks_since),
                                       self.ticks_since + 1)  # fmt: skip
        if self.reacq:
            self.t_good = torch.where(a_t, t, self.t_good)
            # (b) release a target unmatched for lost_timeout: keep its estimate, acquire again
            rel = self.t_acq & (t - self.t_seen > c.lost_timeout_s)
            self.t_acq = self.t_acq & ~rel
            self.t_fp = self.t_fp & ~rel
            self.t_active = self.t_active & ~rel
            self.t_last = self.t_last | rel
            st = torch.where(a_t & ~ho, TRACKING, HOLDING)
            self.status = torch.where(t - self.t_good > c.lost_hard_stop_s, LOST, st)
            self._pos = pos
            return
        # status
        age = t - self.t_seen
        acq_status = torch.where(a_t & ~ho, TRACKING, HOLDING)
        acq_status = torch.where(age > c.lost_timeout_s, LOST, acq_status)
        since = t - torch.where(torch.isfinite(self.last_event), self.last_event, self.t0)
        limit = torch.where(torch.isfinite(self.last_event),
                            torch.full_like(since, c.lost_timeout_s),
                            torch.full_like(since, c.acquire_timeout_s))  # fmt: skip
        un_status = torch.where(since <= limit, HOLDING, LOST)
        self.status = torch.where(self.t_acq, acq_status, un_status)
        self._pos = pos

    # ---- outputs ----
    @property
    def lost(self) -> torch.Tensor:
        return self.status == LOST

    def outputs(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """Raw perception for the observation builder (course units)."""
        pos = self._pos
        t = t.to(torch.float64)
        c = self.cfg
        # a target estimate exists: held, or (reacquire) the last released one
        has = self.t_acq | self.t_last
        target = torch.where(has.unsqueeze(-1), self.t_world, pos)
        mode = torch.where(
            self.n_active & self.n_known, NEXT_VISIBLE,
            torch.where(self.n_known, NEXT_UNSEEN_KNOWN,
                        torch.where(has, NEXT_UNSEEN_TARGET, NEXT_UNSEEN_NONE)),
        )  # fmt: skip
        next_center = torch.where((mode <= NEXT_UNSEEN_KNOWN).unsqueeze(-1), self.n_world,
                                  torch.where((mode == NEXT_UNSEEN_TARGET).unsqueeze(-1),
                                              self.t_world, pos + torch.tensor(
                                                  [1.0, 0.0, 0.0], dtype=self.dtype,
                                                  device=self.device)))  # fmt: skip
        t_age = torch.where(self.t_acq, (t - self.t_src).to(self.dtype),
                            torch.full_like(self.lat, c.lost_timeout_s))  # fmt: skip
        n_age = torch.where(self.n_known, (t - self.n_src).to(self.dtype),
                            torch.full_like(self.lat, c.lost_timeout_s))  # fmt: skip
        return {
            "status": self.status,
            "target": target,
            "target_yaw": torch.where(has, self.t_yaw, torch.zeros_like(self.t_yaw)),
            "acquired": has,
            "next_mode": mode,
            "next_center": next_center,
            "next_yaw": self.n_yaw,
            "target_age_s": t_age,
            "next_age_s": n_age,
            "target_valid": self.t_acq,
            "hand_over": self.hand_over,
            "passed": self.passed,
            "matched": self.t_active & self.t_acq,
            "ticks_since_detection": self.ticks_since,
            "fp_target": self.t_fp,
        }


def _travel_yaw(n: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Gate yaw of a world normal, oriented away from the drone (``detection_from_pose``)."""
    nh = n[:, :2]
    small = nh.norm(dim=-1) < 1e-6
    nh = torch.where(small.unsqueeze(-1), d[:, :2], nh)
    flip = (nh * d[:, :2]).sum(-1) < 0
    nh = torch.where(flip.unsqueeze(-1), -nh, nh)
    return torch.atan2(nh[:, 1], nh[:, 0])
