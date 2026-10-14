"""NumPy single-drone reference of :class:`drone_fly.sim.perception.GatePerception` (UC70 D2).

Driven by EXPLICIT random draws (the same :data:`~drone_fly.sim.perception.TICK_DRAWS` /
:data:`~drone_fly.sim.perception.EPISODE_DRAWS` layout), so a test can feed identical draws to the
batched torch twin and compare tick by tick; it is also what the detector-noise validation replay
flies against the real ``TrackingGateEstimator``. :meth:`ReferencePerception.estimate` returns a
real harness :class:`~drone_fly.harness.types.GateEstimate` at a world scale ``k`` (what the
tracker would hand the assembler), with the UC70 ``info`` ages.

Every step mirrors the torch module line for line; see its docstring for the semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from drone_fly.sim.noise_model import (
    VIS_OK,
    NoiseModel,
    VisibilitySpec,
    bearing_cam,
    edge_distance_norm,
    gate_corners,
    los_frame,
    markov_from,
    project,
    quat_matrix_np,
    sigmoid,
    to_camera,
    visibility,
)
from drone_fly.sim.perception import (
    EPISODE_DRAWS,
    HOLDING,
    LOST,
    NEXT_UNSEEN_KNOWN,
    NEXT_UNSEEN_NONE,
    NEXT_UNSEEN_TARGET,
    NEXT_VISIBLE,
    RING,
    STATUS_NAMES,
    TICK_DRAWS,
    TRACKING,
    PerceptionConfig,
    draw_width,
)


def unpack(x: Any, spec: tuple[tuple[str, int, str], ...]) -> dict[str, Any]:
    """A packed draw row (W,) -> ``{name: float or (w,) array}``."""
    x = np.asarray(x, np.float64).reshape(-1)
    out, i = {}, 0
    for name, w, _ in spec:
        out[name] = float(x[i]) if w == 1 else x[i : i + w].copy()
        i += w
    return out


def sample(rng: np.random.Generator, spec: tuple[tuple[str, int, str], ...]) -> np.ndarray:
    cols = []
    for _, w, kind in spec:
        cols.append(rng.random(w) if kind == "u" else rng.standard_normal(w))
    return np.concatenate(cols)


@dataclass
class _Slot:
    acq: bool = False  # target: acquired; next: known
    fp: bool = False
    active: bool = False
    miss: bool = False
    relb: np.ndarray = field(default_factory=lambda: np.zeros(3))
    nb: np.ndarray = field(default_factory=lambda: np.zeros(3))
    world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    seen: float = 0.0
    src: float = 0.0
    z: np.ndarray = field(default_factory=lambda: np.zeros(4))


def _travel_yaw(n: np.ndarray, d: np.ndarray) -> float:
    nh = n[:2].copy()
    if np.linalg.norm(nh) < 1e-6:
        nh = d[:2].copy()
    if float(np.dot(nh, d[:2])) < 0:
        nh = -nh
    return float(math.atan2(nh[1], nh[0]))


class ReferencePerception:
    """One drone; positions in course units, ``scale`` = metres per course unit."""

    def __init__(self, model: NoiseModel | None, config: PerceptionConfig) -> None:
        if model is None:
            raise ValueError("ReferencePerception needs a noise model (camera block)")
        self.model = model
        self.cfg = config
        self.exact = config.mode == "exact"
        self.reacq = config.lost_policy == "reacquire"
        self.spec: VisibilitySpec = model.visibility_spec()
        self.cn = model.class_noise()
        self.fp = model.data["false_positives"]
        self.reset(0.0, np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(draw_width(EPISODE_DRAWS)))

    # ---- state ----
    def reset(self, t: float, pos: Any, quat: Any, draws: Any) -> None:
        d = unpack(draws, EPISODE_DRAWS)
        c = self.cfg
        self.lat = 0.0 if self.exact else c.detection_lag_s[0] + (
            c.detection_lag_s[1] - c.detection_lag_s[0]) * d["lat"]  # fmt: skip
        self.next_frame_t = float(t) + c.frame_period_s * d["phase"]
        self.ring = [(float(t), np.asarray(pos, np.float64).copy(),
                      np.asarray(quat, np.float64).copy()) for _ in range(RING)]  # fmt: skip
        self.ring_i = 0
        self.t0 = float(t)
        self.last_event = -math.inf
        self.tg = _Slot(seen=float(t), src=float(t))
        self.nx = _Slot(seen=float(t), src=float(t))
        self.ticks_since = 0
        self.t_last = False  # reacquire: a released target's estimate is kept
        self.t_good = float(t)  # reacquire: the last good event (match, acquisition, pass)
        self.hand_over = False
        self.passed = False
        self.status = HOLDING
        self.pos = np.asarray(pos, np.float64).copy()
        self.t = float(t)
        # UC70 run-2: per-episode pointer bias / dropout; the noisy pointer + availability bit; the
        # per-slot gate memory and the FP temporal streaks (mirrors GatePerception).
        self.ptr_bias = np.asarray(d["ptr_bias"], np.float64) * c.pointer_bias_sigma_rad
        self.ptr_dropped = bool(d["ptr_drop"] < c.pointer_dropout_p)
        self.ptr_body = np.zeros(3)
        self.ptr_avail = False
        self.tm_world, self.tm_yaw, self.tm_stamp, self.tm_valid = np.zeros(3), 0.0, float(t), False
        self.nm_world, self.nm_yaw, self.nm_stamp, self.nm_valid = np.zeros(3), 0.0, float(t), False
        self.t_streak = 0
        self.n_streak = 0

    def push_pose(self, t: float, pos: Any, quat: Any) -> None:
        self.ring[self.ring_i % RING] = (float(t), np.asarray(pos, np.float64).copy(),
                                         np.asarray(quat, np.float64).copy())  # fmt: skip
        self.ring_i += 1

    def delayed_pose(self, tc: float) -> tuple[np.ndarray, np.ndarray]:
        ok = [i for i, (ti, _, _) in enumerate(self.ring) if ti <= tc + 1e-9]
        if ok:
            i = max(ok, key=lambda j: (self.ring[j][0], -j))
        else:
            i = min(range(RING), key=lambda j: (self.ring[j][0], j))
        return self.ring[i][1], self.ring[i][2]

    # ---- noise ----
    def _error(self, rng: float, z: np.ndarray) -> np.ndarray:
        cn = self.cn
        if cn.model == "binned" and cn.bins:
            tab = cn.bins["range_table"]
            edges = np.asarray(cn.bins["range_edges"], np.float64)
            i = int(np.clip(np.searchsorted(edges, rng, "right") - 1, 0, len(tab) - 1))
            b = np.asarray(tab[i]["bias"], np.float64).copy()
            b[3] = math.radians(b[3])
            cov = np.asarray(tab[i]["cov"], np.float64).copy()
            cov[3, :] *= math.pi / 180.0
            cov[:, 3] *= math.pi / 180.0
            return b + np.linalg.cholesky(cov + 1e-9 * np.eye(4)) @ z
        bias = np.array([cn.radial_bias_per_m * rng, 0.0, 0.0, cn.yaw_bias])
        sig = np.array([*(rng * np.asarray(cn.sigma_per_m)), cn.yaw_sigma])
        return bias + sig * z

    def _detect(
        self,
        p: dict[str, Any] | None,
        pd: np.ndarray,
        qd: np.ndarray,
        scale: float,
        miss_prev: bool,
        z_prev: np.ndarray,
        u_mk: float,
        eps: np.ndarray,
    ) -> tuple:
        """``(detected, miss, z, rel_body, normal_body)`` (as the torch ``_detect``)."""
        if p is None:
            return False, True, z_prev, np.zeros(3), np.zeros(3)
        center = np.asarray(p["center"], np.float64)
        rot = np.asarray(p["rot"], np.float64)
        corners = gate_corners(center, rot, p["wh"])
        spec = self.spec
        # positions in course units: scale the range test (and the z cut-off) only
        code = visibility(center[None] * scale, corners[None] * scale, pd * scale, qd, spec)[0]
        c_cam = to_camera(center[None], pd, qd, spec.camera)
        rng = float(bearing_cam(c_cam)[0][0]) * scale
        u, v = project(c_cam, spec.camera)
        edge = float(edge_distance_norm(u, v, spec.camera.width, spec.camera.height)[0])
        vis = code == VIS_OK
        n_true = rot[:, 2]
        if self.exact:
            det, z, e = vis, z_prev, np.zeros(4)
        else:
            cn = self.cn
            if cn.model == "binned" and cn.bins:
                pdet = float(cn.p_det(np.array([rng]), np.array([edge]))[0])
            else:
                a, b, c = cn.p_det_coef
                pdet = float(sigmoid(a + b * rng + c * edge))
            pdet = min(max(pdet, 1e-4), 1.0)
            L = float(cn.burst_frames(np.array([rng]), self.cfg.frame_period_s)[0])
            stay, d2m = (float(x) for x in markov_from(pdet, L))
            det = vis and u_mk >= (stay if miss_prev else d2m)
            tau = cn.ar1_tau_s
            rho = math.exp(-self.cfg.frame_period_s / tau) if tau > 0 else 0.0
            z = rho * z_prev + math.sqrt(max(1.0 - rho * rho, 0.0)) * np.asarray(eps)
            e = self._error(rng, z)
        rel = center - pd
        e_world = los_frame(rel).T @ e[:3] / scale
        est = center + e_world
        yaw = math.atan2(n_true[1], n_true[0]) + e[3]
        n_est = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        Rd = quat_matrix_np(qd)
        return bool(det), not bool(det), np.asarray(z, np.float64), Rd.T @ (est - pd), Rd.T @ n_est

    def _fp_sample(self, d: dict[str, Any], scale: float) -> tuple[np.ndarray, np.ndarray]:
        h = self.fp.get("hist") or {}
        counts = np.asarray(h.get("counts") or [[1.0]], np.float64)
        re = np.asarray(h.get("range_edges") or [0.0, 10.0], np.float64)
        ae = np.asarray(h.get("azimuth_edges") or [-30.0, 30.0], np.float64)
        probs = counts.reshape(-1)
        probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(probs), 1 / len(probs))
        cdf = np.cumsum(probs)
        k = int(np.searchsorted(cdf, min(d["fp_bin"], cdf[-1] - 1e-6)))
        na = counts.shape[1]
        ri, ai = k // na, k % na
        r = re[ri] + (re[ri + 1] - re[ri]) * d["fp_r"]
        a = math.radians(ae[ai] + (ae[ai + 1] - ae[ai]) * d["fp_a"])
        p_cam = np.array([r * math.sin(a), 0.0, r * math.cos(a)])
        body = np.asarray(self.spec.camera.body_to_camera(), np.float64).T @ p_cam / scale
        nb = body.copy()
        nb[2] = 0.0
        return body, nb / max(np.linalg.norm(nb), 1e-9)

    def _pointer(self, R: np.ndarray, pos: np.ndarray, center: np.ndarray, jit: np.ndarray,
                 target_valid: bool) -> None:  # fmt: skip
        """Store the noisy body-frame pointer (3) + availability bit (mirrors
        :meth:`GatePerception._pointer`)."""
        rel_b = R.T @ (np.asarray(center, np.float64) - pos)
        unit = rel_b / max(float(np.linalg.norm(rel_b)), 1e-9)
        j = np.asarray(jit, np.float64) * self.cfg.pointer_jitter_sigma_rad
        az = math.atan2(unit[1], unit[0]) + self.ptr_bias[0] + j[0]
        el = math.asin(float(np.clip(unit[2], -1.0, 1.0))) + self.ptr_bias[1] + j[1]
        ce = math.cos(el)
        ptr = np.array([ce * math.cos(az), ce * math.sin(az), math.sin(el)])
        avail = bool(target_valid) and not self.ptr_dropped
        self.ptr_body = ptr if avail else np.zeros(3)
        self.ptr_avail = avail

    # ---- the tick ----
    def step(
        self,
        t: float,
        pos: Any,
        quat: Any,
        target: dict[str, Any],
        nxt: dict[str, Any] | None,
        crossed: bool,
        scale: float,
        draws: Any,
        *,
        target_valid: bool = True,
    ) -> None:
        """One control tick (``nxt`` None = no valid next passage)."""
        d = unpack(draws, TICK_DRAWS)
        c = self.cfg
        pos = np.asarray(pos, np.float64)
        R = quat_matrix_np(quat)
        tg, nx = self.tg, self.nx
        # UC70 run-2: the noisy next-gate pointer (truth-driven, single-sourced here).
        self._pointer(R, pos, target["center"], d["ptr_jit"], target_valid)
        # 1. hand-over
        self.hand_over = self.passed = bool(crossed)
        if crossed:
            if nx.acq:
                self.tg = tg = _Slot(True, nx.fp, nx.active, nx.miss, nx.relb.copy(),
                                     nx.nb.copy(), nx.world.copy(), nx.yaw, float(t), nx.src,
                                     nx.z.copy())  # fmt: skip
            else:
                tg.acq = tg.fp = tg.active = False
            self.last_event = float(t)
            if self.reacq:
                self.t_good = float(t)
                self.t_last = False
            nx.acq = nx.fp = nx.active = nx.miss = False
            nx.z = np.zeros(4)
        # 2. frame arrival
        arrive = t >= self.next_frame_t - 1e-9
        step_t = max(c.frame_period_s + c.frame_jitter_s * d["jitter"], 0.25 * c.frame_period_s)
        if arrive:
            self.next_frame_t = max(self.next_frame_t + step_t, float(t))
        frame = arrive and d["skip"] >= c.frame_skip_p
        tc = float(t) - self.lat
        pd, qd = self.delayed_pose(tc)
        Rd = quat_matrix_np(qd)
        det_t, miss_t, z_t, relb_t, nb_t = self._detect(target, pd, qd, scale, tg.miss, tg.z,
                                                        d["mk_t"], d["eps_t"])  # fmt: skip
        det_n, miss_n, z_n, relb_n, nb_n = self._detect(nxt, pd, qd, scale, nx.miss, nx.z,
                                                        d["mk_n"], d["eps_n"])  # fmt: skip
        any_fp = (not self.exact) and d["fp_any"] < 1.0 - math.exp(
            -float(self.fp.get("rate_per_frame", 0.0)))  # fmt: skip
        p_present, p_pick = self.model.acquisition()
        wrong = (not self.exact) and d["fp_any"] < p_present
        fp_relb, fp_nb = self._fp_sample(d, scale) if not self.exact else (relb_t, nb_t)
        if frame:
            if not tg.fp:
                tg.miss = miss_t
            tg.z = z_t
            nx.miss = miss_n
            nx.z = z_n
        unacq = frame and not tg.acq
        if self.reacq:  # (a) a stale held target re-acquires from this frame's candidates
            stale = tg.acq and not tg.active and (t - tg.seen > c.drop_after_s)
            unacq = unacq or (frame and stale and (det_t or wrong))
        pick_fp = unacq and wrong and ((not det_t) or d["fp_pick"] < p_pick)
        acq_true = unacq and det_t and not pick_fp
        assoc = float(np.linalg.norm(pos + R @ relb_t - tg.world)) <= c.assoc_limit
        upd_t = (frame and tg.acq and not tg.fp and det_t and assoc) or acq_true
        if (frame and not tg.fp) or acq_true:
            tg.active = upd_t
        if pick_fp:
            tg.active = False
        if upd_t:
            tg.relb, tg.nb = relb_t, nb_t
        if upd_t or pick_fp:
            tg.src = tc
        tg.acq = tg.acq or acq_true or pick_fp
        tg.fp = False if acq_true else (tg.fp or pick_fp)
        if self.reacq and (acq_true or pick_fp):
            self.t_good = float(t)
            self.t_last = False
        if pick_fp:
            tg.world = pd + Rd @ fp_relb
            fn = Rd @ fp_nb
            tg.yaw = float(math.atan2(fn[1], fn[0]))
            tg.seen = float(t)
        nframe = frame and tg.acq and not pick_fp
        mis_rate = 0.0 if self.exact else float(self.fp.get("next_misassoc_rate", 0.0))
        mis = nframe and any_fp and d["next_mis"] < mis_rate
        upd_n = nframe and det_n and not mis
        if nframe:
            nx.active = upd_n or mis
        if upd_n:
            nx.relb, nx.nb = relb_n, nb_n
        elif mis:
            nx.relb, nx.nb = fp_relb, fp_nb
        if upd_n or mis:
            nx.src = tc
            nx.fp = mis
        nx.acq = nx.acq or upd_n or mis
        # per tick re-placement with the current pose
        a_t = tg.active and tg.acq
        if a_t:
            tg.world = pos + R @ tg.relb
            tg.yaw = _travel_yaw(R @ tg.nb, tg.world - pos)
            tg.seen = float(t)
        if nx.active and nx.acq:
            nx.world = pos + R @ nx.relb
            nx.yaw = _travel_yaw(R @ nx.nb, nx.world - pos)
            nx.seen = float(t)
        self.ticks_since = 0 if a_t else self.ticks_since + 1
        if self.reacq:
            if a_t:
                self.t_good = float(t)
            if tg.acq and t - tg.seen > c.lost_timeout_s:  # (b) release, keep the estimate
                tg.acq = tg.fp = tg.active = False
                self.t_last = True
            st = TRACKING if (a_t and not crossed) else HOLDING
            self.status = LOST if t - self.t_good > c.lost_hard_stop_s else st
            self.pos = pos
            self.t = float(t)
            return
        if tg.acq:
            st = TRACKING if (a_t and not crossed) else HOLDING
            if t - tg.seen > c.lost_timeout_s:
                st = LOST
        else:
            fin = math.isfinite(self.last_event)
            since = t - (self.last_event if fin else self.t0)
            limit = c.lost_timeout_s if fin else c.acquire_timeout_s
            st = HOLDING if since <= limit else LOST
        self.status = st
        self.pos = pos
        self.t = float(t)

    # ---- outputs ----
    def outputs(self) -> dict[str, Any]:
        tg, nx, c = self.tg, self.nx, self.cfg
        has = tg.acq or self.t_last
        if nx.active and nx.acq:
            mode = NEXT_VISIBLE
        elif nx.acq:
            mode = NEXT_UNSEEN_KNOWN
        elif has:
            mode = NEXT_UNSEEN_TARGET
        else:
            mode = NEXT_UNSEEN_NONE
        nc = (
            nx.world
            if mode <= NEXT_UNSEEN_KNOWN
            else tg.world
            if mode == NEXT_UNSEEN_TARGET
            else self.pos + np.array([1.0, 0, 0])
        )
        return {
            "status": self.status,
            "target": tg.world.copy() if has else self.pos.copy(),
            "target_yaw": tg.yaw if has else 0.0,
            "acquired": has,
            "next_mode": mode,
            "next_center": np.asarray(nc, np.float64).copy(),
            "next_yaw": nx.yaw,
            "target_age_s": self.t - tg.src if tg.acq else c.lost_timeout_s,
            "next_age_s": self.t - nx.src if nx.acq else c.lost_timeout_s,
            "target_valid": tg.acq,
            "hand_over": self.hand_over,
            "passed": self.passed,
            "matched": tg.active and tg.acq,
            "ticks_since_detection": self.ticks_since,
            "fp_target": tg.fp,
            "pointer": np.concatenate([np.asarray(self.ptr_body, np.float64),
                                       [1.0 if self.ptr_avail else 0.0]]),  # fmt: skip
        }

    def estimate(self, k: float = 1.0, gate_index: int = 0) -> Any:
        """The harness :class:`GateEstimate` at world scale ``k`` (course units / ``k``)."""
        return estimate_from_raw(self.outputs(), self.pos, self.cfg, k, self.t, gate_index)


def estimate_from_raw(raw: dict[str, Any], pos: Any, cfg: PerceptionConfig, k: float,
                      now: float, gate_index: int = 0) -> Any:  # fmt: skip
    """Raw perception outputs -> a harness ``GateEstimate`` at world scale ``k``."""
    from drone_fly.harness.course_senses import lookahead_unseen, lookahead_visible
    from drone_fly.harness.types import GateEstimate

    p = np.asarray(pos, np.float64) / k
    f32 = np.float32
    target = (np.asarray(raw["target"], np.float64) / k).astype(f32)
    if raw["acquired"]:
        y = f32(raw["target_yaw"])
        curr = np.array([np.cos(y), np.sin(y), f32(cfg.nominal_aperture / k)], f32)
    else:
        curr = np.array([1.0, 0.0, cfg.nominal_aperture / k], f32)
    nc = np.asarray(raw["next_center"], np.float64) / k
    if raw["next_mode"] == NEXT_VISIBLE:
        nxt = lookahead_visible(p, nc, float(raw["next_yaw"]))
    else:
        nxt = lookahead_unseen(p, nc)
    return GateEstimate(
        target=target,
        curr_gate=curr,
        next_lookahead=nxt,
        gate_index=gate_index,
        status=STATUS_NAMES[int(raw["status"])],  # type: ignore[arg-type]
        stamp=float(now),
        passed=bool(raw["passed"]),
        info={
            "matched": bool(raw["matched"]),
            "target_age_s": float(raw["target_age_s"]),
            "next_age_s": float(raw["next_age_s"]),
            "target_valid": bool(raw["target_valid"]),
            "hand_over": bool(raw["hand_over"]),
            # UC70 run-2: the noisy next-gate pointer (body bearing 3 + bit) flows raw -> info ->
            # the numpy obs builder; the torch builder reads perc["pointer"] (one source).
            "pointer": None if raw.get("pointer") is None
            else np.asarray(raw["pointer"], np.float32),
        },
    )
