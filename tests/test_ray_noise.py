"""UC-82 AC1/AC2/AC4: the obstacle-ray noise model and its sim injection.

CPU-only and hermetic (no Liftoff, no captures, no GPU). Covers:

* ``RayNoiseModel`` load / schema validation and the batched ``apply`` error injection
  (miss hit->clear, false-alarm clear->obstacle, bias/sigma on detected rays, output in ``[0, 1]``,
  per-generator-seed determinism).
* The ``eyes noise-model`` builder (:mod:`drone_fly.eyes.ray_noise_model`) — the documented
  fallback and the ``--from-eval`` derivation — writes a loadable model.
* ``CameraTwin.in_view`` matches the NumPy reference :func:`drone_fly.eyes.camera_rays.project_rays`
  (0 mismatches over the ray set).
* The **no-injection byte-identity guard (AC2)**: with ``ray_noise`` / ``ray_blank`` unset the ray
  block routes to the exact cast and draws no randomness, and even with noise on the global RNG
  stream is untouched (a dedicated generator is used).
* The blanked-ray control (AC4): ``ray_blank`` -> all 40 rays are a constant clear 1.0 (not zeroed),
  plus the config validation guards.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from drone_fly.eyes.camera_rays import project_rays
from drone_fly.eyes.ray_noise_model import (
    DOC_FA_IN_VIEW,
    DOC_MISS_IN_VIEW,
    build_ray_noise_model,
    write_ray_noise_model,
)
from drone_fly.harness.geometry import CameraModel
from drone_fly.sim.action import ActionSpec
from drone_fly.sim.course import (
    N_RAYS,
    RAY_RANGE,
    CourseConfig,
    CourseEnv,
    StartConfig,
)
from drone_fly.sim.plant import PlantConfig
from drone_fly.sim.ray_noise import (
    N_DIST_BINS,
    SCHEMA,
    VERSION,
    CameraTwin,
    RayNoiseModel,
    RayNoiseModelError,
)
from drone_fly.sim.timing import TimingConfig
from tests.sim_support import meteor75

V2 = ActionSpec.v2(1.0, 4.0)


# ----------------------------------------------------------------------------- fixtures / helpers


def _model_dict(
    *,
    miss_in=(0.0, 0.0, 0.0, 0.0),
    miss_out=(0.0, 0.0, 0.0, 0.0),
    fa_in=0.0,
    fa_out=0.0,
    bias_m=0.0,
    sigma_m=0.0,
    ray_range_m=4.0,
    source="test",
):
    """A minimal, valid ``ray-noise`` model mapping for the apply tests."""
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "source": source,
        "n_rays": N_RAYS,
        "ray_range_m": ray_range_m,
        "camera": {
            "resolution": [64, 32],
            "hfov_deg": 120.0,
            "uptilt_deg": 0.0,
            "masks_norm": [],
        },
        "distance_edges_m": [0.0, 1.0, 2.0, 3.0, 4.0],
        "miss_prob": {"in_view": list(miss_in), "out_of_view": list(miss_out)},
        "false_alarm_clear": {"in_view": fa_in, "out_of_view": fa_out},
        "radial_bias_m": bias_m,
        "sigma_m": sigma_m,
    }


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _env(*, seed=0, n=4, **course_kw):
    cfg = CourseConfig(
        reward_reference_dt_s=0.02,
        no_obs=False,
        pool_size=16,
        start=StartConfig(mode="mixed", ground_fraction=0.0, air_z_m=(1.2, 1.2)),
        **course_kw,
    )
    return CourseEnv(
        meteor75(),
        n,
        "cpu",
        seed,
        cfg,
        PlantConfig(),
        TimingConfig.fixed(50.0),
        0.0,
        action_spec=V2,
    )


# ----------------------------------------------------------------------------- model load + schema


def test_model_loads_from_yaml_and_validates(tmp_path):
    p = tmp_path / "ray-noise.yaml"
    p.write_text(yaml.safe_dump(_model_dict(source="unit")), encoding="utf-8")
    m = RayNoiseModel.load(p)
    assert m.source == "unit"
    assert m.ray_range_m == 4.0
    assert m.sha256 and len(m.sha256) == 64
    cam = m.camera_twin()
    assert isinstance(cam, CameraTwin) and (cam.width, cam.height) == (64, 32)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda d: d.update(schema="nope"), "ray-noise"),
        (lambda d: d.update(version=99), "version"),
        (lambda d: d.update(source=""), "source"),
        (lambda d: d.pop("camera"), "camera"),
        (lambda d: d["miss_prob"].update(in_view=[0.1, 0.2]), "miss_prob.in_view"),
        (lambda d: d.update(distance_edges_m=[0.0, 1.0]), "distance_edges_m"),
    ],
)
def test_malformed_models_are_refused(mutate, match):
    d = _model_dict()
    mutate(d)
    with pytest.raises(RayNoiseModelError, match=match):
        RayNoiseModel.from_dict(d)


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(RayNoiseModelError, match="not found"):
        RayNoiseModel.load(tmp_path / "nope.yaml")


# ----------------------------------------------------------------------------- apply() injection


def test_apply_output_in_unit_range_and_deterministic():
    m = RayNoiseModel.from_dict(
        _model_dict(miss_in=(0.5, 0.5, 0.5, 0.5), fa_in=0.2, sigma_m=0.5, bias_m=0.2)
    )
    free = torch.rand(6, N_RAYS, generator=_gen(1))  # a mix of hits and near-clear
    in_view = torch.ones(6, N_RAYS, dtype=torch.bool)
    out1 = m.apply(free, in_view, _gen(7))
    out2 = m.apply(free, in_view, _gen(7))
    assert out1.shape == (6, N_RAYS)
    assert torch.equal(out1, out2)  # same generator seed -> identical
    assert float(out1.min()) >= 0.0 and float(out1.max()) <= 1.0
    # a different seed perturbs the stochastic rays
    assert not torch.equal(out1, m.apply(free, in_view, _gen(8)))


def test_miss_turns_a_hit_into_clear():
    # miss_prob = 1 everywhere: every truth-hit ray is dropped and reads clear (1.0).
    m = RayNoiseModel.from_dict(_model_dict(miss_in=(1.0, 1.0, 1.0, 1.0)))
    free = torch.full((3, N_RAYS), 0.4)  # all hits (< 1)
    in_view = torch.ones(3, N_RAYS, dtype=torch.bool)
    out = m.apply(free, in_view, _gen(0))
    assert torch.allclose(out, torch.ones_like(out))


def test_false_alarm_invents_an_obstacle_on_a_clear_ray():
    # fa = 1 on clear rays: every truth-clear ray becomes a (sub-1.0) spurious obstacle.
    m = RayNoiseModel.from_dict(_model_dict(fa_in=1.0))
    free = torch.ones(3, N_RAYS)  # all clear
    in_view = torch.ones(3, N_RAYS, dtype=torch.bool)
    out = m.apply(free, in_view, _gen(0))
    assert bool((out < 1.0).all())  # torch.rand is [0, 1) -> strictly an obstacle
    assert float(out.min()) >= 0.0


def test_detected_ray_is_unchanged_without_bias_or_spread():
    # no miss, no FA, no bias, no sigma: a detected hit passes through exactly.
    m = RayNoiseModel.from_dict(_model_dict())
    free = torch.full((2, N_RAYS), 0.5)
    in_view = torch.ones(2, N_RAYS, dtype=torch.bool)
    out = m.apply(free, in_view, _gen(3))
    assert torch.allclose(out, free)


def test_radial_bias_shifts_detected_rays():
    bias_m = 0.4
    m = RayNoiseModel.from_dict(_model_dict(bias_m=bias_m))  # sigma 0, no miss/FA
    free = torch.full((2, N_RAYS), 0.25)
    in_view = torch.ones(2, N_RAYS, dtype=torch.bool)
    out = m.apply(free, in_view, _gen(0))
    expected = (free + bias_m / RAY_RANGE).clamp(0.0, 1.0)
    assert torch.allclose(out, expected)


def test_out_of_view_rays_use_the_out_of_view_tables():
    # in-view never misses, out-of-view always misses: splits purely on the in_view flag.
    m = RayNoiseModel.from_dict(
        _model_dict(miss_in=(0.0, 0.0, 0.0, 0.0), miss_out=(1.0, 1.0, 1.0, 1.0))
    )
    free = torch.full((1, N_RAYS), 0.4)
    in_view = torch.zeros(1, N_RAYS, dtype=torch.bool)
    in_view[0, : N_RAYS // 2] = True
    out = m.apply(free, in_view, _gen(0))
    half = N_RAYS // 2
    assert torch.allclose(out[0, :half], torch.full((half,), 0.4))  # in-view detected
    assert torch.allclose(out[0, half:], torch.ones(N_RAYS - half))  # out-of-view missed -> clear


# ----------------------------------------------------------------------------- the AC1 builder


def test_build_documented_fallback_model_is_loadable():
    model = build_ray_noise_model()  # documented UC-78 eyes v1 fallback
    m = RayNoiseModel.from_dict(model)  # must validate
    assert m.source == "uc78-eval-documented-fallback"
    assert model["miss_prob"]["in_view"] == list(DOC_MISS_IN_VIEW)
    assert model["false_alarm_clear"]["in_view"] == DOC_FA_IN_VIEW
    assert len(model["distance_edges_m"]) == N_DIST_BINS + 1
    assert len(model["miss_prob"]["out_of_view"]) == N_DIST_BINS


def test_build_from_eval_report_derives_rates(tmp_path):
    report = {
        "dataset": "uc82-unit",
        "split": "held-out",
        "tol_m": 0.5,
        "distance_bins": [
            {"bin": "0-25%", "in_view": {"miss_rate": 0.8}, "out_of_view": {"miss_rate": 0.95}},
            {"bin": "25-50%", "in_view": {"miss_rate": 0.6}, "out_of_view": {"miss_rate": None}},
        ],
        "aggregate": {
            "in_view": {"false_alarm_rate_clear": 0.05},
            "out_of_view": {"false_alarm_rate_clear": 0.0},
        },
    }
    rep = tmp_path / "eval.json"
    rep.write_text(__import__("json").dumps(report), encoding="utf-8")
    model = build_ray_noise_model(from_eval=rep)
    m = RayNoiseModel.from_dict(model)
    assert m.source.startswith("uc78-eval:uc82-unit:held-out")
    assert model["miss_prob"]["in_view"][0] == 0.8  # straight from the report
    # a bin with no hit (None) falls back to the documented constant
    assert model["miss_prob"]["in_view"][1] == 0.6
    assert model["miss_prob"]["in_view"][2] == DOC_MISS_IN_VIEW[2]
    assert model["false_alarm_clear"]["in_view"] == 0.05


def test_write_round_trips(tmp_path):
    out = write_ray_noise_model(build_ray_noise_model(), tmp_path / "sub" / "ray-noise.yaml")
    assert out.is_file()
    assert RayNoiseModel.load(out).source == "uc78-eval-documented-fallback"


# ----------------------------------------------------------------------------- in-view twin


def _random_unit_rays(n_rays, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n_rays, 3))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float64)


def _random_quats(n, seed=1):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4))
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float64)


@pytest.mark.parametrize(
    "masks",
    [(), ((0.4, 0.0, 0.6, 0.2), (0.0, 0.85, 0.1, 1.0))],
    ids=["no-masks", "hud-masks"],
)
def test_camera_twin_in_view_matches_numpy_reference(masks):
    w, h, hfov, uptilt = 2560, 1080, 120.0, 20.0
    twin = CameraTwin(w, h, hfov, uptilt, masks)
    camera = CameraModel(w, h, hfov, uptilt)
    R, n = 64, 6
    rays = _random_unit_rays(R)
    quats = _random_quats(n)
    # twin: batched over all poses at once
    twin_view = twin.in_view(
        torch.as_tensor(np.broadcast_to(rays, (n, R, 3)).copy()),
        torch.as_tensor(quats),
    ).numpy()
    # reference: the NumPy projection, pose by pose
    ref = np.stack([project_rays(rays, quats[i], camera, masks)[2] for i in range(n)])
    assert twin_view.shape == (n, R)
    assert int((twin_view != ref).sum()) == 0


# ----------------------------------------------------------------------------- AC2 byte-identity


def test_default_ray_path_has_no_noise_state_and_equals_the_exact_cast():
    env = _env(raycast=True)
    assert env._ray_noise is None and env._ray_gen is None and env._ray_camera is None
    # the ray block routes straight to the exact cast
    assert torch.equal(env._ray_obs(), env._raycast())
    # ... and the exact cast is the LAST N_RAYS columns of extra()
    assert torch.equal(env.extra()[:, -N_RAYS:], env._raycast())


def test_default_ray_path_draws_no_randomness():
    env = _env(raycast=True)
    before = torch.random.get_rng_state()
    _ = env._ray_obs()  # default path must not touch the global RNG stream
    assert torch.equal(torch.random.get_rng_state(), before)


def test_noise_path_uses_a_dedicated_generator_not_the_global_stream(tmp_path):
    p = tmp_path / "ray-noise.yaml"
    p.write_text(yaml.safe_dump(_model_dict(miss_in=(0.5, 0.5, 0.5, 0.5), fa_in=0.1)), "utf-8")
    env = _env(raycast=True, ray_noise=str(p))
    assert env._ray_noise is not None and env._ray_gen is not None
    before = torch.random.get_rng_state()
    out = env._ray_obs()  # draws from the dedicated self._ray_gen, never the global stream
    assert torch.equal(torch.random.get_rng_state(), before)
    assert out.shape == (env.n, N_RAYS)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


# ----------------------------------------------------------------------------- AC4 blanked control


def test_blank_rays_are_constant_clear_one():
    env = _env(raycast=True, ray_blank=True)
    rays = env._ray_obs()
    assert rays.shape == (env.n, N_RAYS)
    assert torch.equal(rays, torch.ones_like(rays))  # constant clear 1.0, NOT zeroed
    assert torch.equal(env.extra()[:, -N_RAYS:], torch.ones(env.n, N_RAYS))


def test_ray_options_require_raycast():
    with pytest.raises(ValueError, match="require course.raycast"):
        CourseConfig(reward_reference_dt_s=0.02, ray_blank=True)
    with pytest.raises(ValueError, match="require course.raycast"):
        CourseConfig(reward_reference_dt_s=0.02, ray_noise="x.yaml")


def test_ray_noise_and_blank_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        CourseConfig(reward_reference_dt_s=0.02, raycast=True, ray_noise="x.yaml", ray_blank=True)
