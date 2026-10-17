"""UC-82 AC8: a tiny CPU smoke retrain exercising the ray-noise injection end-to-end.

One PPO iteration of the UC-82 ``uc82-obstacle-rays`` config on the fixture connectome — raycast on,
``no_obs`` off, the documented ray-noise model injected into the 40 obstacle rays — asserting the
train loop runs and checkpoints. The blanked-ray control config (AC4) is smoke-run the same way.

Hermetic: CPU-only, the fixture connectome, ``warm_start`` disabled (the real 70-input checkpoint is
not needed to prove the plumbing), tiny env / budget. This is the only slow UC-82 test; keep it in
its own file so it runs alone (memory safety).
"""

from __future__ import annotations

import pytest

from drone_fly import drones
from drone_fly.eyes.ray_noise_model import build_ray_noise_model, write_ray_noise_model
from drone_fly.training.config import load_train_config
from drone_fly.training.run_dir import CHECKPOINT_NAME


@pytest.fixture(scope="module")
def fixture_k1(harness_contract, harness_k1_dir):
    """The fixture connectome the session's contract was loaded with (as in test_training_cli)."""
    from drone_fly.connectome import load_connectome

    return load_connectome(harness_k1_dir)


@pytest.fixture(scope="module")
def ray_noise_yaml(tmp_path_factory):
    """A real documented-fallback ray-noise model on disk for the injection path."""
    out = tmp_path_factory.mktemp("rn") / "ray-noise.yaml"
    return write_ray_noise_model(build_ray_noise_model(), out)


def _tiny(course_extra):
    base = {
        "n_envs": 4,
        "budget": 32,  # roll(8) * n_envs(4) = 32 -> exactly one iteration
        "device": "cpu",
        "course": {"pool_size": 16, "episode_s": 0.1, **course_extra},
        "ppo": {"roll": 8, "epochs": 1, "minibatches": 2},
        "warm_start": {"checkpoint": None},  # random init: we only prove the plumbing
        "log_every_iters": 1,
    }
    return base


def _run(cfg, fixture_k1, root):
    from drone_fly.training.ppo import train

    return train(
        cfg,
        drones.load(cfg.drone),
        run_root=root,
        connectome=fixture_k1,
        device="cpu",
        max_iterations=1,
    )


def test_smoke_retrain_with_ray_noise_injection(fixture_k1, ray_noise_yaml, tmp_path):
    cfg = load_train_config("uc82-obstacle-rays", _tiny({"ray_noise": str(ray_noise_yaml)}))
    assert cfg.actor.extra_dim == 70 and cfg.course.extra_dim == 70
    assert cfg.course.raycast and not cfg.course.no_obs
    assert cfg.course.ray_noise == str(ray_noise_yaml)

    res = _run(cfg, fixture_k1, tmp_path / "rays")
    assert res.iterations == 1 and res.steps == 32
    assert res.checkpoint.is_file() and res.checkpoint.name == CHECKPOINT_NAME
    assert res.run.path in res.checkpoint.parents
    assert (res.run.path / "config.yaml").is_file()


def test_smoke_retrain_blanked_control(fixture_k1, tmp_path):
    cfg = load_train_config("uc82-obstacle-blank", _tiny({}))
    assert cfg.course.ray_blank and cfg.course.ray_noise is None
    assert cfg.course.raycast and cfg.actor.extra_dim == 70

    res = _run(cfg, fixture_k1, tmp_path / "blank")
    assert res.iterations == 1 and res.steps == 32
    assert res.checkpoint.is_file()
