"""UC67 AC3a: with ``drones/meteor75`` and randomization off, the new plant matches
``gpu_prototype``'s ``BatchedDrone`` within 1e-4 per state component over 200 steps.

The reference is the real prototype code, loaded on CPU through the session ``harness_contract``
fixture (no copy). ``BatchedDrone``'s auto-reset (``_perturb``) is disabled on the instance and its
state is copied from the plant; identical bounded raw actions go through the v1 decoder into
``Plant.step`` (``euler-legacy``, one substep: the claimed-equivalent configuration) and straight
into ``BatchedDrone.step``. The float64 run sets torch's default dtype so the prototype's own
tensors are float64 too, and restores it afterwards [C11]; a float32 run checks the training
dtype against the same tolerance.
"""

from __future__ import annotations

import sys

import pytest
import torch

from drone_fly.sim.action import ActionSpec, decode_torch
from drone_fly.sim.plant import Plant, PlantConfig
from tests.sim_support import all_envs, meteor75

N_ENVS = 64  # reduced env count (AC9)
STEPS = 200
TOL = 1e-4


@pytest.fixture
def default_dtype():
    """Set torch's default dtype for one test and always restore it [C11]."""
    prev = torch.get_default_dtype()

    def _set(dtype: torch.dtype) -> None:
        torch.set_default_dtype(dtype)

    try:
        yield _set
    finally:
        torch.set_default_dtype(prev)


def _initial_state(n: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(11)
    pos = torch.randn(n, 3, generator=g, dtype=dtype) * 0.3
    pos[:, 2] += 1.5
    ax = torch.randn(n, 3, generator=g, dtype=dtype)
    ax = ax / ax.norm(dim=-1, keepdim=True)
    ang = 0.3 * torch.randn(n, 1, generator=g, dtype=dtype)
    quat = torch.cat([torch.cos(ang / 2), torch.sin(ang / 2) * ax], -1)
    return {
        "pos": pos,
        "vel": 0.3 * torch.randn(n, 3, generator=g, dtype=dtype),
        "quat": quat,
        "omega": 0.3 * torch.randn(n, 3, generator=g, dtype=dtype),
    }


def _run(dtype: torch.dtype) -> float:
    hover_gpu = sys.modules["hover_gpu"]
    assert str(hover_gpu.dev) == "cpu"
    prof = meteor75()
    cfg = PlantConfig(
        integrator="euler-legacy",
        substeps=1,
        dtype="float64" if dtype == torch.float64 else "float32",
    )
    plant = Plant(prof, N_ENVS, "cpu", 0, cfg, randomization_scale=0.0)
    s0 = _initial_state(N_ENVS, dtype)
    plant.reset(all_envs(N_ENVS), **s0, motor_g=1.0)
    ref = hover_gpu.BatchedDrone(N_ENVS)
    ref._perturb = lambda mask: None  # no auto-reset: compare the raw dynamics
    for key, val in s0.items():
        setattr(ref, key, val.clone())
    assert ref.pos.dtype == dtype  # the prototype's own tensors follow the default dtype

    g = torch.Generator().manual_seed(5)
    spec = ActionSpec()  # v1: the prototype's action semantics
    worst = 0.0
    for _ in range(STEPS):
        raw = (torch.rand(N_ENVS, 4, generator=g, dtype=dtype) * 2.0 - 1.0) * 1.2
        cg, rates = decode_torch(raw, spec)
        plant.step(cg, rates, hover_gpu.DT)
        ref.step(raw)
        for key in ("pos", "vel", "quat", "omega"):
            diff = (getattr(plant, key) - getattr(ref, key)).abs().max().item()
            worst = max(worst, diff)
            assert diff <= TOL, (key, diff)
    return worst


def test_plant_matches_batched_drone_float64(harness_contract, default_dtype):
    default_dtype(torch.float64)
    worst = _run(torch.float64)
    assert worst < 1e-9  # identical arithmetic up to rounding


def test_plant_matches_batched_drone_float32(harness_contract):
    assert torch.get_default_dtype() == torch.float32  # [C11]: the float64 test restored it
    _run(torch.float32)


def test_prototype_constants_are_the_profile_values(harness_contract):
    """The airframe numbers live in drones/meteor75, not in drone_fly.sim (AC2 / [C4])."""
    hover_gpu = sys.modules["hover_gpu"]
    prof = meteor75()
    assert prof.max_g == pytest.approx(hover_gpu.TW)
    assert prof.hover_throttle == pytest.approx(hover_gpu.HOVER_THR)
    for a in ("roll", "pitch", "yaw"):
        assert prof.rates[a].tau_s == pytest.approx(1.0 / hover_gpu.RATE_KP)
    plant = Plant(prof, 1, "cpu", 0, PlantConfig(dtype="float64"), 0.0)
    assert plant.full_stick_rate.tolist() == pytest.approx([hover_gpu.MAX_BODY_RATE] * 3)
    assert plant.gravity == pytest.approx(hover_gpu.G)
