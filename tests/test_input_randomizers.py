"""UC-87 AC4: the input domain randomisers (``drone_fly.sim.course.CourseEnv``), statistics-tested.

Each randomiser is self-contained in ``CourseEnv`` (NOT via ``sim/perception.py``) and default-off,
so the model-11 path stays byte-identical. CPU-hermetic: large env batches, the dedicated UC-87
input-DR generator, checked against each randomiser's intended distribution:

* gyro/omega observation noise (white + AR(1)) — injected into the OBSERVED omega only (training-
  only; the plant's true omega and the harness are untouched);
* floor-visible raycast; spawn yaw 0..360°; spawn distance 1.5..10 m; ``obstacle_vision`` zeroed;
* gate-estimate noise / 10 % outliers / staleness / 30 % Markov dropout.
"""

from __future__ import annotations

import math

import pytest
import torch

from drone_fly.sim.action import ActionSpec
from drone_fly.sim.course import CourseConfig, CourseEnv, StartConfig
from drone_fly.sim.plant import PlantConfig
from drone_fly.sim.timing import TimingConfig
from tests.sim_support import meteor75

V2 = ActionSpec.v2(1.0, 4.0)
REF_DT = 0.02


def _env(*, n=1024, seed=0, no_obs=True, raycast=False, **course_kw):
    cfg = CourseConfig(
        reward_reference_dt_s=REF_DT,
        no_obs=no_obs,
        raycast=raycast,
        pool_size=64,
        episode_s=28.0,
        ceiling_z=50.0,
        start=StartConfig(mode="mixed", ground_fraction=0.0, air_z_m=(3.0, 3.0)),
        **course_kw,
    )
    return CourseEnv(
        meteor75(), n, "cpu", seed, cfg, PlantConfig(), TimingConfig.fixed(50.0), 0.0,
        action_spec=V2,
    )  # fmt: skip


def _hover(n):
    return torch.zeros(n, 4, dtype=torch.float64)


# ----------------------------------------------------------------------------- gyro / omega noise

def test_white_gyro_noise_has_the_configured_sigma_and_is_observed_only():
    sigma = 0.4
    env = _env(gyro_noise_sigma=sigma)
    env.step(_hover(env.n))
    true_omega = env.omega
    observed = env.obs()[:, 9:12]  # the omega slice of the core obs
    injected = observed - true_omega
    assert injected.std().item() == pytest.approx(sigma, rel=0.1)
    # the injection is the env's gyro-obs state; the PLANT's true omega is not perturbed
    torch.testing.assert_close(injected, env._gyro_obs)
    assert env.omega.abs().max().item() < 1.0  # hover: the true rate stays tiny


def test_ar1_vibration_is_coloured_with_the_configured_rho_and_sigma():
    vib, rho = 0.3, 0.6
    env = _env(n=2048, gyro_vibration_sigma=vib, gyro_vibration_rho=rho)
    series = []
    for _ in range(60):  # burn in the AR(1) state, then sample
        env.step(_hover(env.n))
        series.append(env._gyro_obs.clone())
    tail = torch.stack(series[-30:])  # (T, n, 3)
    assert tail.std().item() == pytest.approx(vib, rel=0.15)  # stationary std == vib
    a = tail[:-1].reshape(-1)
    b = tail[1:].reshape(-1)
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert corr == pytest.approx(rho, abs=0.15)  # lag-1 autocorrelation ≈ rho


def test_gyro_noise_off_is_byte_identical_observed_omega():
    env = _env()  # no gyro DR
    env.step(_hover(env.n))
    torch.testing.assert_close(env.obs()[:, 9:12], env.omega)


# ----------------------------------------------------------------------------- floor-visible rays

def test_floor_visible_rays_see_the_ground_when_pointing_down():
    down = torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float64)
    low = torch.tensor([[0.0, 0.0, 0.3]], dtype=torch.float64)
    seen = _env(n=8, raycast=True, raycast_floor=True)
    hidden = _env(n=8, raycast=True, raycast_floor=False)
    for env in (seen, hidden):
        env.vel = down.expand(env.n, 3).clone()
        env.pos = low.expand(env.n, 3).clone()
    # with the floor visible some downward rays return < 1 (a hit); hidden ⇒ all clear (no obs)
    assert (seen._ray_obs() < 0.999).any()
    torch.testing.assert_close(
        hidden._ray_obs(), torch.ones_like(hidden._ray_obs()), atol=1e-6, rtol=0
    )


# ----------------------------------------------------------------------------- spawn yaw / distance

def test_spawn_yaw_is_uniform_over_the_full_circle():
    env = _env(n=4096, spawn_yaw=True)
    q = env.quat
    yaw = 2.0 * torch.atan2(q[:, 3], q[:, 0])  # quat_arg = [cos(y/2),0,0,sin(y/2)]
    yaw = torch.remainder(yaw, 2 * math.pi)
    # covers the full 0..2π and is roughly uniform (mean near π, all four quadrants populated)
    assert yaw.min() < 0.2 and yaw.max() > 2 * math.pi - 0.2
    for q0, q1 in [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]:
        frac = ((yaw >= q0 * math.pi) & (yaw < q1 * math.pi)).float().mean().item()
        assert frac == pytest.approx(0.25, abs=0.05)


def test_spawn_yaw_off_is_level():
    env = _env(n=16, spawn_yaw=False)
    level = torch.zeros_like(env.quat)
    level[:, 0] = 1.0
    torch.testing.assert_close(env.quat, level)


def test_spawn_distance_is_within_the_configured_range():
    lo, hi = 1.5, 10.0
    env = _env(n=4096, spawn_distance_m=(lo, hi))
    gate0 = env.P["cen"][env.cid, 0]
    dist = (env.pos - gate0)[:, :2].norm(dim=-1)
    assert dist.min().item() >= lo - 1e-4 and dist.max().item() <= hi + 1e-4
    assert dist.min().item() < lo + 0.3 and dist.max().item() > hi - 0.3  # range is exercised


# ----------------------------------------------------------------------------- obstacle_vision

def test_obstacle_vision_zero_toggle():
    on_obs = _env(n=256, no_obs=False, obstacle_vision_zero=False)
    zeroed = _env(n=256, no_obs=False, obstacle_vision_zero=True)
    assert zeroed._obstacle_vision().abs().sum().item() == 0.0
    # without zeroing, courses with obstacles produce a non-zero vision block
    assert on_obs._obstacle_vision().abs().sum().item() > 0.0


# ----------------------------------------------------------------------------- gate randomisers

def test_gate_position_noise_has_the_configured_sigma():
    sigma = 0.06
    env = _env(n=4096, gate_pos_noise_m=sigma)
    env.step(_hover(env.n))
    resid = env._gate_rel - (env._target_pos() - env.pos)
    assert resid.std().item() == pytest.approx(sigma, rel=0.15)


def test_gate_outliers_fire_at_the_configured_rate_and_magnitude():
    p, mag = 0.10, 0.4
    env = _env(n=8192, gate_pos_noise_m=0.0, gate_outlier_p=p, gate_outlier_m=mag)
    env.step(_hover(env.n))
    resid = env._gate_rel - (env._target_pos() - env.pos)
    norm = resid.norm(dim=-1)
    outlier = norm > 0.3
    assert outlier.float().mean().item() == pytest.approx(p, abs=0.02)
    torch.testing.assert_close(
        norm[outlier], torch.full_like(norm[outlier], mag), rtol=0, atol=1e-5
    )


def test_gate_staleness_hold_is_drawn_within_the_configured_window():
    stale_s = 0.16
    env = _env(n=4096, gate_staleness_s=stale_s)
    hi = stale_s / REF_DT  # 8 steps
    hold = env._gate_hold
    assert hold.min().item() >= 0 and hold.max().item() <= round(hi)
    assert hold.float().mean().item() == pytest.approx(hi / 2.0, abs=0.6)


def test_markov_dropout_holds_the_last_estimate():
    env = _env(n=512, gate_pos_noise_m=0.06, gate_dropout_p=1.0)  # always lost ⇒ frozen estimate
    a = torch.zeros(env.n, 4, dtype=torch.float64)
    a[:, 0] = 0.3  # climb a little so the TRUE rel genuinely changes step to step
    env.step(a.clone())
    frozen = env._gate_rel.clone()
    for _ in range(5):
        env.step(a.clone())
    torch.testing.assert_close(env._gate_rel, frozen)  # never refreshed while "lost"


def test_all_gate_randomisers_off_use_the_true_rel():
    env = _env(n=64)  # no gate DR ⇒ _gate_on is False
    assert env._gate_on is False
    env.step(_hover(env.n))
    rel = env.obs()[:, :3]
    torch.testing.assert_close(rel, env._target_pos() - env.pos)


# ----------------------------------------------------------------------------- input_dr_scale ramp
# UC-87 redesign: CourseEnv.input_dr_scale ramps the input-noise magnitude in over the budget. At
# scale 0 the noise has ZERO EFFECT on the observation (not RNG-identity); it opens to full at 1.

def test_input_dr_scale_zero_zeroes_the_gyro_noise_effect():
    env = _env(gyro_noise_sigma=0.4, gyro_vibration_sigma=0.3, gyro_vibration_rho=0.6)
    env.input_dr_scale = 0.0
    env.reset_all()
    for _ in range(5):  # even after several steps the coloured-noise memory stays zero
        env.step(_hover(env.n))
    torch.testing.assert_close(env.obs()[:, 9:12], env.omega)  # observed omega == true omega
    assert torch.count_nonzero(env._gyro_obs) == 0


def test_input_dr_scale_zero_makes_the_gate_estimate_the_true_rel():
    env = _env(gate_pos_noise_m=0.06, gate_outlier_p=0.1, gate_staleness_s=0.16, gate_dropout_p=0.3)
    assert env._gate_on is True  # the path still runs; it just contributes nothing at scale 0
    env.input_dr_scale = 0.0
    env.reset_all()
    env.step(_hover(env.n))
    torch.testing.assert_close(env._gate_rel, env._target_pos() - env.pos)


def test_input_dr_scale_half_scales_the_gyro_sigma():
    env = _env(n=4096, gyro_noise_sigma=0.4)  # white only, so std is exactly σ·scale
    env.input_dr_scale = 0.5
    env.reset_all()
    env.step(_hover(env.n))
    injected = env.obs()[:, 9:12] - env.omega
    assert injected.std().item() == pytest.approx(0.4 * 0.5, rel=0.1)


def test_input_dr_scale_default_is_full():
    env = _env(gyro_noise_sigma=0.4)
    assert env.input_dr_scale == 1.0  # default ⇒ full magnitude (pre-redesign behaviour)
