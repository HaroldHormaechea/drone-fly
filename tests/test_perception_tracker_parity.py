"""UC70 AC2 / AC3: the simulated perception behaves like the live tracker.

* The REAL harness tracker (``harness.liftoff.gate_cv.TrackingGateEstimator``, fed scripted
  detections of the truth gate whenever the shared visibility predicate says it is in view) and the
  NumPy reference (``student.perception_ref.ReferencePerception`` in ``exact`` mode) agree tick by
  tick on status, target, ``curr_gate``, ``next_lookahead`` and the UC70 ``info`` ages through
  acquisition, tracking, out-of-view holding, ``lost`` and re-acquisition.
* The batched torch twin (``sim.perception.GatePerception``) reproduces the reference tick by tick
  in ``noise`` mode when both are fed identical injected draws (dropout, AR(1) errors, latency,
  frame clock, false positives, hand-overs).
* The association gate matches the harness's at any world scale.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from drone_fly.harness.geometry import CameraModel
from drone_fly.harness.liftoff.gate_cv import GateDetection, TrackingGateEstimator
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import FlightState, Frame
from drone_fly.sim.noise_model import camera_visible, gate_axes, gate_corners
from drone_fly.sim.perception import (
    EPISODE_DRAWS,
    TICK_DRAWS,
    GatePerception,
    PerceptionConfig,
    assoc_limit,
)
from drone_fly.student.perception_ref import ReferencePerception, sample
from tests.student_support import gate_rot, load_model, yaw_quat

GATE = np.array([10.0, 0.0, 0.0])


# ----------------------------------------------------------------------------- real tracker


class ScriptedTracker(TrackingGateEstimator):
    """Detects the truth gate exactly when the shared visibility predicate says it is visible."""

    def __init__(self, camera, spec, **kw):
        super().__init__(camera, WorldScale(1.0), **kw)
        self.spec = spec

    def detect(self, image, state):
        q = np.asarray(state.quat, np.float64)
        pos = np.asarray(state.pos, np.float64)
        k = gate_corners(GATE, gate_axes(0.0), (2.0, 2.0))
        if not camera_visible(GATE[None], k[None], pos, q, self.spec)[0]:
            return []
        return [GateDetection(np.zeros((4, 2)), 0.05, np.zeros(3), np.zeros(3),
                              GATE.astype(np.float32), 0.0, 2.0,
                              float(np.linalg.norm(GATE - pos)), 0.0,
                              "inflatable_arch")]  # fmt: skip


def _yaw_schedule(t: float) -> float:
    """Away (90 deg) until 0.5 s, facing 0.5-1.0 s, away 1.0-2.2 s, facing again after 2.2 s."""
    if t < 0.5 or 1.0 <= t < 2.2:
        return math.pi / 2
    return 0.0


@pytest.mark.parametrize("acquire_timeout", [3.0, 0.3])
def test_real_tracker_and_reference_agree_tick_by_tick(tmp_path, acquire_timeout):
    lost = 0.6
    timing = {"frame_period_s": 0.02, "frame_jitter_s": 0.0, "frame_skip_p": 0.0,
              "detection_lag_s": [0.0, 0.0]}  # fmt: skip
    tracker = {"lost_timeout_s": lost, "acquire_timeout_s": acquire_timeout}
    model = load_model(tmp_path, timing=timing, tracker=tracker)
    cfg = PerceptionConfig.from_model(model, "exact")
    ref = ReferencePerception(model, cfg)
    cam = CameraModel(640, 480, 90.0)
    trk = ScriptedTracker(cam, model.visibility_spec(), default_aperture=2.0, assoc_radius=1.0,
                          lost_timeout_s=lost, acquire_timeout_s=acquire_timeout)  # fmt: skip
    dt, pos = 0.02, np.array([0.0, 0.0, 0.0])
    ref.reset(0.0, pos, yaw_quat(math.pi / 2), np.zeros(len(EPISODE_DRAWS)))
    trk.update(None, FlightState(pos=pos.astype(np.float32), vel=np.zeros(3, np.float32),
                                 quat=yaw_quat(math.pi / 2).astype(np.float32),
                                 omega=np.zeros(3, np.float32), t=0.0), 0.0)  # fmt: skip
    target = {"center": GATE, "rot": gate_rot(0.0), "wh": (2.0, 2.0)}
    no_draws = np.zeros(sum(w for _, w, _ in TICK_DRAWS))
    seen = set()
    for i in range(1, 160):
        t = i * dt
        q = yaw_quat(_yaw_schedule(t))
        ref.push_pose(t, pos, q)
        ref.step(t, pos, q, target, None, False, 1.0, no_draws)
        r = ref.estimate(1.0)
        st = FlightState(pos=pos.astype(np.float32), vel=np.zeros(3, np.float32),
                         quat=q.astype(np.float32), omega=np.zeros(3, np.float32), t=t)  # fmt: skip
        h = trk.update(Frame(np.zeros((480, 640, 3), np.uint8), t, i), st, t)
        ctx = (round(t, 3), r.status, h.status)
        assert r.status == h.status, ctx
        np.testing.assert_allclose(r.target, h.target, atol=1e-5, err_msg=str(ctx))
        np.testing.assert_allclose(r.curr_gate, h.curr_gate, atol=1e-5, err_msg=str(ctx))
        np.testing.assert_allclose(r.next_lookahead, h.next_lookahead, atol=1e-5,
                                   err_msg=str(ctx))  # fmt: skip
        for key in ("matched", "target_valid", "hand_over"):
            assert r.info[key] == h.info[key], (key, ctx)
        for key in ("target_age_s", "next_age_s"):
            assert r.info[key] == pytest.approx(h.info[key], abs=1e-6), (key, ctx)
        seen.add(h.status)
    assert seen == {"holding", "tracking", "lost"}


# ----------------------------------------------------------------------------- torch twin


def _route(n_ticks: int, dt: float):
    """A drone flying +x at 6 m/s, weaving in yaw and height, through gates at x = 8, 16, 24."""
    gates = [np.array([8.0, 0.5, 1.0]), np.array([16.0, -0.8, 1.2]), np.array([24.0, 0.3, 0.9])]
    poses = []
    for i in range(n_ticks + 1):
        t = i * dt
        p = np.array([6.0 * t, 0.6 * math.sin(2.0 * t), 1.0 + 0.2 * math.sin(3.0 * t)])
        q = yaw_quat(0.25 * math.sin(4.0 * t))
        poses.append((t, p, q))
    return gates, poses


@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_torch_twin_reproduces_the_reference_with_injected_draws(tmp_path, scale):
    model = load_model(
        tmp_path, a=1.0, b=-0.05, c=0.5, radial_bias_per_m=0.02, sigma_per_m=(0.02, 0.02, 0.02),
        yaw_bias_deg=1.0, yaw_sigma_deg=4.0, burst_s=0.08, ar1_tau_s=0.06, fp_rate=0.4,
        next_misassoc_rate=0.3, acquisition={"p_wrong_present": 0.3, "p_pick_wrong": 0.2},
        timing={"frame_period_s": 0.033, "frame_jitter_s": 0.004, "frame_skip_p": 0.15,
                "detection_lag_s": [0.02, 0.07]},
        tracker={"lost_timeout_s": 0.12, "acquire_timeout_s": 0.4, "assoc_radius": 2.0},
    )  # fmt: skip
    cfg = PerceptionConfig.from_model(model, "noise")
    n, dt, ticks = 6, 0.01, 450
    rng = np.random.default_rng(int(scale * 10))
    ep = np.stack([sample(rng, EPISODE_DRAWS) for _ in range(n)])
    tick = [np.stack([sample(rng, TICK_DRAWS) for _ in range(n)]) for _ in range(ticks)]
    gates, poses = _route(ticks, dt)
    # course units: positions / scale (metres per unit = scale)
    G = [g / scale for g in gates]
    tw = GatePerception(model, n, "cpu", config=cfg, dtype=torch.float64)
    refs = [ReferencePerception(model, cfg) for _ in range(n)]
    t0, p0, q0 = poses[0]
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)  # noqa: E731
    tw.reset(torch.ones(n, dtype=torch.bool), T([t0] * n), T([p0 / scale] * n), T([q0] * n),
             draws=T(ep))  # fmt: skip
    for j, r in enumerate(refs):
        r.reset(t0, p0 / scale, q0, ep[j])
    k = 0
    stats = {"fp": 0, "handover": 0, "matched": 0, "lost": 0}
    for i in range(1, ticks + 1):
        t, p, q = poses[i]
        pc = p / scale
        crossed = bool(k < len(G) and pc[0] >= G[k][0])
        if crossed:
            k += 1
        tgt = G[min(k, len(G) - 1)]
        nxt = G[k + 1] if k + 1 < len(G) else None

        def pas(c):
            return {"center": c, "rot": gate_rot(0.0), "wh": (2.0 / scale, 2.0 / scale)}

        def tpas(c):
            return {k_: T([v] * n) for k_, v in pas(c).items()}

        tw.push_pose(T([t] * n), T([pc] * n), T([q] * n))
        nx_pas = tpas(nxt if nxt is not None else tgt)
        tw.step(T([t] * n), T([pc] * n), T([q] * n), tpas(tgt), nx_pas,
                torch.full((n,), nxt is not None), torch.full((n,), crossed),
                torch.full((n,), scale, dtype=torch.float64), draws=T(tick[i - 1]))  # fmt: skip
        out = tw.outputs(T([t] * n))
        for j, r in enumerate(refs):
            r.push_pose(t, pc, q)
            r.step(t, pc, q, pas(tgt), None if nxt is None else pas(nxt), crossed, scale,
                   tick[i - 1][j])  # fmt: skip
            o = r.outputs()
            ctx = (i, j, crossed)
            for key in ("status", "acquired", "next_mode", "target_valid", "hand_over",
                        "matched", "ticks_since_detection", "fp_target"):  # fmt: skip
                assert int(out[key][j]) == int(o[key]), (key, ctx, int(out[key][j]), o[key])
            np.testing.assert_allclose(out["target"][j].numpy(), o["target"], atol=1e-6,
                                       err_msg=str(ctx))  # fmt: skip
            np.testing.assert_allclose(out["next_center"][j].numpy(), o["next_center"], atol=1e-6,
                                       err_msg=str(ctx))  # fmt: skip
            for key in ("target_yaw", "next_yaw", "target_age_s", "next_age_s"):
                assert float(out[key][j]) == pytest.approx(float(o[key]), abs=1e-6), (key, ctx)
            stats["fp"] += int(o["fp_target"])
            stats["handover"] += int(o["hand_over"])
            stats["matched"] += int(o["matched"])
            stats["lost"] += int(o["status"] == 2)
    # the scenario exercised every branch
    assert all(v > 0 for v in stats.values()), stats


# ----------------------------------------------------------------------------- association gate


@pytest.mark.parametrize("k", [1.0, 4.0, 6.0, 10.0])
def test_association_gate_matches_the_harness_at_any_world_scale(k):
    """Harness: ``assoc_radius * max(1, aperture)`` in student units (aperture = nominal / k);
    the simulator works in Liftoff metres: ``k * assoc_radius * max(1, nominal / k)``."""
    nominal, radius = 3.69, 1.0
    harness_student_units = radius * max(1.0, WorldScale(k).length(nominal))
    assert assoc_limit(radius, nominal, k) == pytest.approx(harness_student_units * k)


def test_reference_requires_a_model():
    with pytest.raises(ValueError, match="noise model"):
        ReferencePerception(None, PerceptionConfig(mode="exact"))
