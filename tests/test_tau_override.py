"""UC-87 D4: the per-key ``range_overrides`` hook on ``DynamicsRandomizer`` (sim/randomization.py),
threaded through ``PlantConfig`` → ``Plant`` (sim/plant.py).

The profile's ``bounds()`` samples only along the nominal→profile-range axis, so it cannot reach a
fast-Betaflight rate-loop τ (~5 ms). An absolute per-key ``(min, max)`` override takes precedence
over ``profile.randomization[key]`` WITHOUT editing ``drones/meteor75``. All CPU, no connectome.
"""

from __future__ import annotations

import pytest
import torch

from drone_fly.sim.plant import Plant, PlantConfig
from drone_fly.sim.randomization import DynamicsRandomizer
from tests.sim_support import meteor75

FAST_TAU = (0.005, 0.083)
PROFILE_MIN = 0.038461538461538464  # meteor75 rates.*.tau_s profile range minimum


def _rand(scale, overrides=None, seed=0):
    return DynamicsRandomizer(
        meteor75(), scale, torch.Generator().manual_seed(seed),
        range_overrides=overrides or {},
    )  # fmt: skip


def test_bounds_without_override_is_the_profile_range():
    lo, hi = _rand(1.0).bounds("rates.roll.tau_s")
    assert lo[0] == pytest.approx(PROFILE_MIN)
    assert hi[0] == pytest.approx(0.08333333333333333)


def test_bounds_with_override_replaces_the_profile_range():
    lo, hi = _rand(1.0, {"rates.roll.tau_s": FAST_TAU}).bounds("rates.roll.tau_s")
    assert lo[0] == pytest.approx(0.005)  # below the profile's 0.0385 floor
    assert hi[0] == pytest.approx(0.083)


def test_override_still_respects_the_scale_axis():
    """At scale=0 the sample is nominal regardless of the override (so scale=0 stays the nominal
    plant); the override only widens the scale=1 envelope."""
    nominal = meteor75().rates["roll"].tau_s
    lo0, hi0 = _rand(0.0, {"rates.roll.tau_s": FAST_TAU}).bounds("rates.roll.tau_s")
    assert lo0[0] == pytest.approx(nominal) and hi0[0] == pytest.approx(nominal)
    # a half-scale draw interpolates nominal→override, reaching below the profile floor
    loH, hiH = _rand(0.5, {"rates.roll.tau_s": FAST_TAU}).bounds("rates.roll.tau_s")
    assert loH[0] == pytest.approx(nominal + 0.5 * (0.005 - nominal))
    assert loH[0] < PROFILE_MIN


def test_sampled_tau_widens_below_the_profile_floor_with_the_override():
    overrides = {f"rates.{a}.tau_s": FAST_TAU for a in ("roll", "pitch", "yaw")}
    with_ov = _rand(1.0, overrides).sample(8192).tau_s
    without = _rand(1.0).sample(8192).tau_s
    # the plain profile sampler never dips below its own floor; the override reaches ~5 ms
    assert without.min().item() >= PROFILE_MIN - 1e-6
    assert with_ov.min().item() < PROFILE_MIN
    assert with_ov.min().item() == pytest.approx(0.005, abs=2e-3)
    assert with_ov.max().item() == pytest.approx(0.083, abs=2e-3)


def test_override_is_per_key_leaving_other_keys_on_the_profile_range():
    ov = {"rates.yaw.tau_s": FAST_TAU}
    s = _rand(1.0, ov).sample(8192)
    # yaw (axis 2) widened; roll/pitch (axes 0,1) stay on the profile floor
    assert s.tau_s[:, 2].min().item() < PROFILE_MIN
    assert s.tau_s[:, 0].min().item() >= PROFILE_MIN - 1e-6
    assert s.tau_s[:, 1].min().item() >= PROFILE_MIN - 1e-6


def test_unknown_override_key_is_rejected():
    with pytest.raises(ValueError, match="range_overrides"):
        _rand(1.0, {"rates.roll.not_a_param": FAST_TAU})


def test_plant_config_threads_the_override_into_the_plant():
    ov = {f"rates.{a}.tau_s": FAST_TAU for a in ("roll", "pitch", "yaw")}
    cfg = PlantConfig(dtype="float64", range_overrides=ov)
    p = Plant(meteor75(), 4096, "cpu", 0, cfg, randomization_scale=1.0)
    assert p.randomizer.range_overrides["rates.roll.tau_s"] == (0.005, 0.083)
    assert p.params.tau_s.min().item() < PROFILE_MIN  # the plant's sampled τ reaches the fast loop


def test_plant_config_from_dict_parses_overrides():
    cfg = PlantConfig.from_dict({"range_overrides": {"rates.roll.tau_s": [0.005, 0.083]}})
    assert cfg.range_overrides["rates.roll.tau_s"] == (0.005, 0.083)
