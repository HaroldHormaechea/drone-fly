"""UC70 AC2 / AC9: the batched simulated perception (``sim.perception.GatePerception``).

Statistics are checked against the noise model over >= 1e5 samples (about 4 sigma, with the
autocorrelation of the two-state dropout accounted for): detection probability, the mean
miss-burst length, the position / yaw bias and covariance in the line-of-sight frame, the AR(1)
correlation, false-positive acquisition, the latency error against the analytic
``v * L`` / ``omega * L`` formula, the frame clock (phase, period, skips). The tracker semantics
(hold / lost / acquire / hand-over) are driven tick by tick with injected draws, and ``exact``
mode is checked to be error- and latency-free. CPU only, synthetic model and gates.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from drone_fly.sim.perception import (
    HOLDING,
    LOST,
    TICK_DRAWS,
    TRACKING,
    draw_width,
)
from tests.student_support import StaticDrone, gate_batch, load_model, perception, yaw_quat

GATE = (10.0, 0.0, 0.0)
FAST = {"frame_period_s": 0.02, "frame_jitter_s": 0.0, "frame_skip_p": 0.0,
        "detection_lag_s": [0.03, 0.06]}  # fmt: skip
#: An association gate wide enough that large synthetic errors never break the track.
WIDE = {"assoc_radius": 100.0}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def draws(n: int, **values: float) -> torch.Tensor:
    """Packed tick draws: every uniform 0.5 / normal 0 unless given."""
    cols = []
    for name, w, kind in TICK_DRAWS:
        v = values.get(name, 0.5 if kind == "u" else 0.0)
        cols.append(torch.as_tensor(v, dtype=torch.float32).reshape(-1).expand(w)
                    .unsqueeze(0).expand(n, w))  # fmt: skip
    out = torch.cat(cols, 1)
    assert out.shape == (n, draw_width(TICK_DRAWS))
    return out


def run(perc_model, n: int, ticks: int, dt: float, *, mode: str = "noise", seed: int = 0,
        drone_kw: dict | None = None, **cfg):  # fmt: skip
    p = perception(perc_model, n, mode=mode, seed=seed, **cfg)
    d = StaticDrone(p, n, dt, **(drone_kw or {}))
    tgt = gate_batch(n, GATE)
    return p, d, tgt


# ----------------------------------------------------------------------------- statistics


def test_detection_probability_and_miss_bursts_follow_the_model(tmp_path):
    """Stationary p_det = sigmoid(a); mean miss burst = burst_s / frame period (5 frames)."""
    model = load_model(tmp_path, a=0.4, burst_s=0.1, timing=FAST)
    n, warm, ticks = 2048, 40, 100
    p, d, tgt = run(model, n, warm + ticks, 0.02)
    seq = []
    for i in range(warm + ticks):
        out = d.tick(tgt)
        if i >= warm:
            assert bool(out["acquired"].all())  # long since acquired
            seq.append(out["matched"].clone())
    det = torch.stack(seq).numpy()  # (ticks, n)
    assert det.size >= 1e5
    p_true = sigmoid(0.4)
    L = 0.1 / 0.02
    # two-state chain: variance inflation (1 + lam) / (1 - lam), lam = 1 - P(d->m) - P(m->d)
    lam = 1.0 - (1 - p_true) / p_true / L - 1.0 / L
    sigma = math.sqrt(p_true * (1 - p_true) / det.size * (1 + lam) / (1 - lam))
    assert abs(det.mean() - p_true) < 4.5 * sigma, (det.mean(), p_true, sigma)
    # transitions -> the mean burst (unbiased by window censoring)
    a, b = det[:-1], det[1:]
    m_to_d = np.sum(~a & b) / np.sum(~a)
    d_to_m = np.sum(a & ~b) / np.sum(a)
    assert 1.0 / m_to_d == pytest.approx(L, rel=0.05)
    assert d_to_m == pytest.approx((1 - p_true) / p_true / L, rel=0.06)


def test_detection_probability_is_range_and_edge_dependent(tmp_path):
    """p_det = sigmoid(a + b * range + c * edge): a farther gate is detected less often."""
    model = load_model(tmp_path, a=3.0, b=-0.2, c=0.0, burst_s=0.1, timing=FAST)
    n = 4096
    rates = {}
    for rng in (5.0, 20.0):
        p = perception(model, n, seed=3)
        d = StaticDrone(p, n, 0.02)
        tgt = gate_batch(n, (rng, 0.0, 0.0))
        hits = []
        for i in range(60):
            out = d.tick(tgt)
            if i >= 20:
                hits.append(out["matched"].float().mean().item())
        rates[rng] = float(np.mean(hits))
    assert rates[5.0] == pytest.approx(sigmoid(3.0 - 1.0), abs=0.02)
    assert rates[20.0] == pytest.approx(sigmoid(3.0 - 4.0), abs=0.02)


def test_bias_and_covariance_in_the_line_of_sight_frame(tmp_path):
    """Radial bias c_r * range, sigmas s_i * range, yaw bias / sigma, uncorrelated axes."""
    model = load_model(tmp_path, a=10.0, radial_bias_per_m=0.05, sigma_per_m=(0.02, 0.03, 0.04),
                       yaw_bias_deg=2.0, yaw_sigma_deg=5.0, ar1_tau_s=0.0, timing=FAST,
                       tracker=WIDE)  # fmt: skip
    n, ticks = 2048, 60
    p, d, tgt = run(model, n, ticks, 0.02)
    errs, yaws = [], []
    for _ in range(ticks):
        out = d.tick(tgt)
        m = out["matched"]
        errs.append((out["target"][m] - torch.tensor(GATE)).double())
        yaws.append(out["target_yaw"][m].double())
    e = torch.cat(errs).numpy()
    y = np.degrees(torch.cat(yaws).numpy())
    assert len(e) >= 1e5
    # the gate is on +x: LOS rows (radial, tangential-h, tangential-v) = (x, y, z)
    sd = np.array([0.2, 0.3, 0.4])
    np.testing.assert_allclose(e.mean(0), [0.5, 0.0, 0.0], atol=4.5 * sd.max() / math.sqrt(len(e)))
    np.testing.assert_allclose(e.std(0), sd, rtol=0.02)
    corr = np.corrcoef(e.T)
    assert np.all(np.abs(corr[np.triu_indices(3, 1)]) < 0.02)
    assert y.mean() == pytest.approx(2.0, abs=4.5 * 5.0 / math.sqrt(len(y)))
    assert y.std() == pytest.approx(5.0, rel=0.02)


def test_errors_are_ar1_correlated_with_the_fitted_time_constant(tmp_path):
    tau, period = 0.05, 0.02
    model = load_model(tmp_path, a=30.0, sigma_per_m=(0.05, 0.05, 0.05), ar1_tau_s=tau,
                       timing=FAST, tracker=WIDE)  # fmt: skip
    n, warm, ticks = 2048, 20, 80
    p, d, tgt = run(model, n, warm + ticks, period)
    xs = []
    for i in range(warm + ticks):
        out = d.tick(tgt)
        if i >= warm:
            assert bool(out["matched"].all())
            xs.append((out["target"][:, 0] - GATE[0]).double())
    x = torch.stack(xs).numpy()  # (ticks, n)
    rho = np.sum(x[:-1] * x[1:]) / math.sqrt(np.sum(x[:-1] ** 2) * np.sum(x[1:] ** 2))
    assert x.size >= 1e5
    assert rho == pytest.approx(math.exp(-period / tau), abs=0.02)
    assert x.std() == pytest.approx(0.5, rel=0.03)  # stationary sigma unchanged by the AR(1)


def test_false_positive_acquisition_rates(tmp_path):
    """While unacquired: P(acquire a false positive) = p_wrong_present * p_pick_wrong when the
    target is detected too, p_wrong_present when it is not."""
    acq = {"p_wrong_present": 0.73, "p_pick_wrong": 0.15}
    n = 40000
    for a, expect in ((12.0, 0.73 * 0.15), (-30.0, 0.73)):
        model = load_model(tmp_path, name=f"m{a}.yaml", a=a, fp_rate=1.2, acquisition=acq,
                           timing=FAST)  # fmt: skip
        p, d, tgt = run(model, n, 1, 0.02)
        out = d.tick(tgt)
        frac = out["fp_target"].float().mean().item()
        sigma = math.sqrt(expect * (1 - expect) / n)
        assert abs(frac - expect) < 4.5 * sigma, (a, frac, expect)
        assert bool((out["acquired"] == (out["fp_target"] | out["matched"])).all())


def test_a_false_positive_target_is_held_then_lost(tmp_path):
    acq = {"p_wrong_present": 1.0, "p_pick_wrong": 1.0}
    model = load_model(tmp_path, a=10.0, fp_rate=1.0, acquisition=acq, timing=FAST,
                       tracker={"lost_timeout_s": 0.2})  # fmt: skip
    n = 64
    p, d, tgt = run(model, n, 30, 0.02)
    first = d.tick(tgt)
    assert bool(first["fp_target"].all()) and not bool(first["matched"].any())
    held = first["target"].clone()
    # FP sampled from the histogram: 5-15 m ahead within +-10 deg of the optical axis
    r = held.norm(dim=-1)
    assert bool(((r >= 5.0 - 1e-4) & (r <= 15.0 + 1e-4)).all())
    statuses = []
    for _ in range(15):
        out = d.tick(tgt)
        torch.testing.assert_close(out["target"], held)  # fixed in the world, never re-detected
        assert not bool(out["matched"].any())
        statuses.append(int(out["status"][0]))
    assert statuses[0] == HOLDING and statuses[-1] == LOST
    assert statuses.index(LOST) == pytest.approx(10, abs=1)  # 0.2 s / 0.02 s


# ----------------------------------------------------------------------------- latency + clock


def test_latency_error_matches_v_times_l(tmp_path):
    """A translating drone: the late image placed with the CURRENT pose is off by v * L."""
    L, v, dt = 0.05, 2.0, 0.01
    model = load_model(tmp_path, a=12.0, timing={**FAST, "frame_period_s": dt,
                                                 "detection_lag_s": [L, L]})  # fmt: skip
    n = 4
    p, d, tgt = run(model, n, 0, dt)
    errs = []
    for i in range(1, 30):
        out = d.tick(tgt, pos=(0.0, v * (i * dt), 0.0))
        if i > 10:
            errs.append((out["target"][0] - torch.tensor(GATE)).double().numpy())
    np.testing.assert_allclose(np.array(errs), np.tile([0.0, v * L, 0.0], (len(errs), 1)),
                               atol=2e-4)  # fmt: skip


def test_latency_error_matches_omega_times_l_under_rotation(tmp_path):
    L, w, dt = 0.05, 1.0, 0.01
    model = load_model(tmp_path, a=12.0, timing={**FAST, "frame_period_s": dt,
                                                 "detection_lag_s": [L, L]})  # fmt: skip
    n = 2
    p, d, tgt = run(model, n, 0, dt)
    for i in range(1, 25):
        out = d.tick(tgt, quat=yaw_quat(w * i * dt))
    est = out["target"][0].double().numpy()
    assert math.atan2(est[1], est[0]) == pytest.approx(w * L, abs=2e-4)
    assert np.linalg.norm(est) == pytest.approx(10.0, abs=1e-3)
    assert float(out["target_yaw"][0]) == pytest.approx(w * L, abs=2e-4)
    # the age reported to the student includes the detection lag
    assert float(out["target_age_s"][0]) == pytest.approx(L, abs=1e-6)


def _frame_updates(model, n, dt, seconds, seed=0):
    p = perception(model, n, seed=seed)
    d = StaticDrone(p, n, dt)
    tgt = gate_batch(n, GATE)
    last = torch.full((n,), math.nan, dtype=torch.float64)
    count = torch.zeros(n)
    first = torch.full((n,), math.nan, dtype=torch.float64)
    for _ in range(int(round(seconds / dt))):
        d.tick(tgt)
        src = p.t_src.clone()
        new = p.t_acq & (src != last)
        count += new.float()
        first = torch.where(torch.isnan(first) & new, torch.full_like(first, d.t), first)
        last = torch.where(p.t_acq, src, last)
    return count.numpy(), first.numpy()


def test_frame_clock_period_phase_and_skips(tmp_path):
    period, T, dt = 0.033, 3.0, 0.005
    base = {"frame_period_s": period, "frame_jitter_s": 0.0, "detection_lag_s": [0.0, 0.0]}
    m0 = load_model(tmp_path, name="m0.yaml", a=12.0, timing={**base, "frame_skip_p": 0.0})
    count, first = _frame_updates(m0, 512, dt, T)
    assert count.mean() == pytest.approx(T / period, abs=1.0)
    # the first frame's phase is uniform over one period (quantised up to the tick)
    assert first.mean() == pytest.approx(period / 2 + dt / 2, abs=0.004)
    assert first.min() <= dt + 1e-9 and first.max() >= period - dt
    m1 = load_model(tmp_path, name="m1.yaml", a=12.0, timing={**base, "frame_skip_p": 0.3})
    count, _ = _frame_updates(m1, 512, dt, T)
    assert count.mean() == pytest.approx(T / period * 0.7, abs=1.5)


# ----------------------------------------------------------------------------- tracker semantics


def test_acquire_timeout_before_the_first_detection(tmp_path):
    model = load_model(tmp_path, a=0.0, timing=FAST, tracker={"acquire_timeout_s": 0.5})
    p, d, tgt = run(model, 1, 0, 0.02, drone_kw={"pos": (1.0, 2.0, 3.0)})
    statuses = []
    for _ in range(30):
        out = d.tick(tgt, draws=draws(1, mk_t=0.0, mk_n=0.0))
        assert not bool(out["acquired"][0])
        torch.testing.assert_close(out["target"][0], torch.tensor([1.0, 2.0, 3.0]))  # own pos
        statuses.append(int(out["status"][0]))
    assert statuses[:24].count(HOLDING) == 24 and statuses[-1] == LOST
    assert statuses.index(LOST) in (24, 25)  # 0.5 s / 0.02 s


def test_hold_then_lost_after_lost_timeout(tmp_path):
    model = load_model(tmp_path, a=0.0, timing=FAST, tracker={"lost_timeout_s": 0.31})
    p, d, tgt = run(model, 1, 0, 0.02)
    out = d.tick(tgt, draws=draws(1, mk_t=1.0))
    assert int(out["status"][0]) == TRACKING and bool(out["acquired"][0])
    held = out["target"][0].clone()
    t_seen = d.t
    statuses = []
    for _ in range(25):
        out = d.tick(tgt, draws=draws(1, mk_t=0.0))
        torch.testing.assert_close(out["target"][0], held)
        statuses.append((round(d.t - t_seen, 6), int(out["status"][0])))
    for age, st in statuses:
        assert st == (LOST if age > 0.31 else HOLDING), (age, st)
    assert float(out["target_age_s"][0]) > 0.31
    assert int(out["ticks_since_detection"][0]) == 25
    # a new detection re-tracks
    out = d.tick(tgt, draws=draws(1, mk_t=1.0))
    assert int(out["status"][0]) == TRACKING and int(out["ticks_since_detection"][0]) == 0


def test_hand_over_takes_the_next_estimate(tmp_path):
    model = load_model(tmp_path, a=0.0, timing=FAST)
    p, d, g1 = run(model, 1, 0, 0.02)
    g2 = gate_batch(1, (14.0, 2.0, 0.0))
    g3 = gate_batch(1, (18.0, -1.0, 0.0))
    out = d.tick(g1, g2, draws=draws(1, mk_t=1.0, mk_n=1.0))
    assert out["next_mode"][0].item() == 0  # NEXT_VISIBLE
    torch.testing.assert_close(out["next_center"][0], g2["center"][0])
    # the drone passes gate 1 (truth k advances): gate 2's estimate becomes the target
    out = d.tick(g2, g3, crossed=torch.tensor([True]), draws=draws(1, mk_t=0.0, mk_n=0.0))
    assert bool(out["hand_over"][0]) and bool(out["passed"][0]) and bool(out["acquired"][0])
    torch.testing.assert_close(out["target"][0], g2["center"][0])
    assert int(out["status"][0]) == HOLDING  # no match on the hand-over tick
    assert out["next_mode"][0].item() == 2  # NEXT_UNSEEN_TARGET: the next slot restarted
    out = d.tick(g2, g3, draws=draws(1, mk_t=0.0, mk_n=0.0))
    assert not bool(out["hand_over"][0])


def test_hand_over_without_a_known_next_reacquires_with_the_lost_timeout(tmp_path):
    model = load_model(tmp_path, a=0.0, timing=FAST,
                       tracker={"lost_timeout_s": 0.21, "acquire_timeout_s": 5.0})  # fmt: skip
    p, d, g1 = run(model, 1, 0, 0.02)
    g2 = gate_batch(1, (14.0, 2.0, 0.0))
    d.tick(g1, g2, draws=draws(1, mk_t=1.0, mk_n=0.0))
    out = d.tick(g2, None, crossed=torch.tensor([True]), draws=draws(1, mk_t=0.0, mk_n=0.0))
    assert not bool(out["acquired"][0]) and bool(out["hand_over"][0])
    torch.testing.assert_close(out["target"][0], d.pos[0])
    statuses = [int(d.tick(g2, draws=draws(1, mk_t=0.0))["status"][0]) for _ in range(15)]
    assert statuses.index(LOST) == pytest.approx(10, abs=1)  # lost_timeout, not acquire


def test_reset_clears_only_the_masked_envs(tmp_path):
    model = load_model(tmp_path, a=12.0, timing=FAST)
    p, d, tgt = run(model, 2, 0, 0.02)
    d.tick(tgt)
    assert bool(p.t_acq.all())
    p.reset(torch.tensor([True, False]), d.tt(), d.pos, d.quat)
    assert p.t_acq.tolist() == [False, True]
    assert p.status[0].item() == HOLDING


# ----------------------------------------------------------------------------- exact mode


def test_exact_mode_is_error_and_latency_free(tmp_path):
    model = load_model(tmp_path, a=-30.0, radial_bias_per_m=0.3, sigma_per_m=(0.5, 0.5, 0.5),
                       yaw_sigma_deg=30.0, fp_rate=3.0, timing={**FAST, "frame_period_s": 0.01,
                       "detection_lag_s": [0.08, 0.08]})  # fmt: skip
    n, dt, v = 64, 0.01, 2.0
    p = perception(model, n, mode="exact")
    d = StaticDrone(p, n, dt)
    tgt = gate_batch(n, GATE)
    assert float(p.lat.abs().max()) == 0.0
    for i in range(1, 20):
        out = d.tick(tgt, pos=(0.0, v * i * dt, 0.0))
        assert bool(out["matched"].all())  # every in-frustum frame, despite a = -30
        assert not bool(out["fp_target"].any())
        torch.testing.assert_close(out["target"], tgt["center"], atol=1e-5, rtol=0)
        assert float(out["target_yaw"].abs().max()) < 1e-5
    # out of the frustum: not detected even in exact mode
    behind = gate_batch(n, (-10.0, 0.0, 0.0), yaw=math.pi)
    p2 = perception(model, n, mode="exact")
    d2 = StaticDrone(p2, n, dt)
    for _ in range(5):
        out = d2.tick(behind)
    assert not bool(out["acquired"].any())


def test_noise_mode_without_a_model_is_refused():
    from drone_fly.sim.perception import GatePerception, PerceptionConfig

    with pytest.raises(ValueError, match="needs a noise model"):
        GatePerception(None, 1, "cpu", config=PerceptionConfig(mode="noise"))
    with pytest.raises(ValueError, match="perception.mode"):
        PerceptionConfig(mode="optimistic")
