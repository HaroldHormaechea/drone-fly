"""UC68 run 2: the battery + thrust-at-speed plant model (:mod:`drone_fly.sim.power`).

Covers the opt-in contract (power off is the UC67 plant, bit for bit), the available-thrust cap
(axial inflow with its floor, pack voltage), sag / drain behaviour, the latched low-voltage
cutoff and the env's ``power_loss`` outcome, the per-env randomization and its own generator,
the ``PowerSpec`` schema, the critic's battery features and the ``power_fit`` math.

Most cases use a synthetic :class:`~drone_fly.drones.schema.PowerSpec` (:func:`spec`) attached
to the committed micro profile, so the expected numbers are closed form; the sag cases use the
micro's committed (fitted) block.
"""

from __future__ import annotations

import copy
import dataclasses
import math

import numpy as np
import pytest
import torch

from drone_fly import drones
from drone_fly.drones import DroneProfileError
from drone_fly.drones.schema import PowerSpec, Range, parse_profile
from drone_fly.sim.plant import Plant, PlantConfig
from drone_fly.sim.power import POWER_RANDOMIZABLE, PowerState
from tests.sim_support import all_envs, meteor75, micro

D = torch.float64
CPU = torch.device("cpu")


def spec(**kw) -> PowerSpec:
    """A synthetic power block: inflow only (no voltage term, no drain / sag / IR) unless set."""
    base = PowerSpec(
        cells=3,
        ocv_charge=(0.0, 1.0),
        ocv_volts=(10.5, 12.6),
        sag_rate_v_per_s=0.0,
        sag_exponent=1.0,
        ir_v=0.0,
        ir_exponent=1.0,
        drain_idle_per_s=0.0,
        drain_load_per_s=0.0,
        drain_exponent=1.0,
        thrust_voltage_exponent=0.0,
        inflow_coeff=0.4,
        inflow_v_ref=20.0,
        inflow_min_factor=0.2,
        effort_inflow_exponent=0.8,
        cutoff_v_per_cell=2.7,
        randomization={},
    )
    return dataclasses.replace(base, **kw)


_RANGES = {
    "drain_scale": Range(0.9, 1.1),
    "sag_scale": Range(0.8, 1.2),
    "ir_scale": Range(0.8, 1.2),
    "inflow_scale": Range(0.85, 1.15),
    "initial_charge": Range(0.95, 1.0),
}
#: A range for every randomizable power key (``ir_scale`` joined the set in UC68 run 2).
RND = {k: _RANGES[k] for k in POWER_RANDOMIZABLE}


def with_power(profile, power):
    return dataclasses.replace(profile, power=power)


def state(sp, n=1, *, seed=0, scale=0.0, dtype=D) -> PowerState:
    return PowerState(sp, n, device=CPU, dtype=dtype, seed=seed, randomization_scale=scale)


def _plant(profile, n=1, *, power=True, seed=0, scale=0.0, dtype="float64", vel=None, motor=1.0):
    p = Plant(profile, n, "cpu", seed, PlantConfig(dtype=dtype, power=power), scale)
    pos = torch.tensor([0.0, 0.0, 5.0]).expand(n, 3)
    p.reset(all_envs(n), pos=pos, vel=vel, motor_g=motor, resample=False)
    return p


# ----------------------------------------------------------------------------- power off = UC67


def _rollout(profile, power, *, steps=50, seed=3, cg_frac=None):
    p = Plant(profile, 16, "cpu", seed, PlantConfig(power=power), 1.0)
    p.reset(all_envs(16), pos=torch.tensor([0.0, 0.0, 2.0]).expand(16, 3))
    g = torch.Generator().manual_seed(123)  # identical actions for every plant
    out = []
    for _ in range(steps):
        if cg_frac is None:
            cg = 1.0 + 0.3 * torch.randn(16, generator=g)
        else:
            cg = cg_frac * p.max_g * torch.rand(16, generator=g)
        rates = 2.0 * torch.randn(16, 3, generator=g)
        p.step(cg, rates, 0.02, 0.01)
        out.append(torch.cat([p.pos, p.vel, p.quat, p.omega, p.motor_g[:, None]], 1).clone())
    return p, torch.stack(out)


def test_power_off_is_bit_identical_to_a_profile_without_a_power_block():
    prof = micro()
    assert prof.power is not None
    a, ta = _rollout(prof, False)
    b, tb = _rollout(with_power(prof, None), False)
    assert a.power is None and b.power is None
    assert torch.equal(ta, tb)
    assert torch.equal(a.params.columns(), b.params.columns())


def test_plant_config_power_defaults_off():
    assert PlantConfig().power is False
    assert PlantConfig.from_dict({"power": True}).power is True
    assert Plant(micro(), 2, "cpu", 0, PlantConfig(), 0.0).power is None


def test_a_neutral_power_model_does_not_change_the_dynamics():
    # no inflow loss, no voltage term, no drain / sag / IR: the cap is max_g * thrust_scale, which
    # every (clamped) command already respects, so the trajectory is the power-off one bit for bit
    prof = with_power(micro(), spec(inflow_coeff=0.0))
    a, ta = _rollout(prof, False, cg_frac=0.9)
    b, tb = _rollout(prof, True, cg_frac=0.9)
    assert b.power is not None
    assert torch.equal(ta, tb)


def test_power_has_its_own_generator_so_dynamics_samples_are_unchanged():
    prof = with_power(micro(), spec(randomization=dict(RND)))
    off = Plant(prof, 64, "cpu", 7, PlantConfig(), 1.0)
    on = Plant(prof, 64, "cpu", 7, PlantConfig(power=True), 1.0)
    assert torch.equal(off.params.columns(), on.params.columns())
    mask = torch.arange(64) % 2 == 0
    off.reset(mask)
    on.reset(mask)
    assert torch.equal(off.params.columns(), on.params.columns())


def test_power_on_needs_a_power_block():
    assert meteor75().power is None
    with pytest.raises(ValueError, match="power"):
        Plant(meteor75(), 1, "cpu", 0, PlantConfig(power=True), 0.0)


# ----------------------------------------------------------------------------- available thrust


@pytest.mark.parametrize(
    ("v_axial", "inflow"),
    [(-5.0, 1.0), (0.0, 1.0), (10.0, 0.8), (20.0, 0.6), (40.0, 0.2), (1000.0, 0.2)],
)
def test_available_thrust_drops_with_axial_inflow_down_to_the_floor(v_axial, inflow):
    ps = state(spec())
    avail = ps.available_g(torch.tensor([3.0], dtype=D), torch.tensor([v_axial], dtype=D))
    assert float(ps.inflow[0]) == pytest.approx(inflow)
    assert float(avail[0]) == pytest.approx(3.0 * inflow)


def test_available_thrust_follows_the_pack_voltage():
    ps = state(spec(inflow_coeff=0.0, thrust_voltage_exponent=0.5), 2)
    ps.volts = torch.tensor([12.6, 12.6 * 0.81], dtype=D)
    avail = ps.available_g(torch.full((2,), 3.0, dtype=D), torch.zeros(2, dtype=D))
    np.testing.assert_allclose(avail.numpy(), [3.0, 3.0 * 0.9], rtol=1e-12)


def test_plant_caps_full_stick_at_speed_and_honours_a_partial_command():
    prof = with_power(micro(), spec())
    vz = torch.tensor([-5.0, 0.0, 10.0, 20.0, 1000.0], dtype=D)
    vel = torch.zeros(5, 3, dtype=D)
    vel[:, 2] = vz  # level: body z = world z
    p = _plant(prof, 5, vel=vel, motor=prof.max_g)
    p.step(torch.full((5,), p.max_g, dtype=D), torch.zeros(5, 3, dtype=D), 0.002)
    expected = p.max_g * np.array([1.0, 1.0, 0.8, 0.6, 0.2])
    np.testing.assert_allclose(p.motor_g.numpy(), expected, rtol=1e-9)
    # below the cap the command is honoured (Liftoff's RPM follows the throttle)
    q = _plant(prof, 1, vel=vel[2:3], motor=1.0)
    q.step(torch.ones(1, dtype=D), torch.zeros(1, 3, dtype=D), 0.002)
    assert float(q.motor_g[0]) == pytest.approx(1.0, abs=1e-12)


def test_capped_thrust_slows_the_climb():
    vel = torch.tensor([[0.0, 0.0, 15.0]], dtype=D)
    prof = with_power(micro(), spec())
    on = _plant(prof, 1, vel=vel, motor=prof.max_g)
    off = _plant(prof, 1, power=False, vel=vel, motor=prof.max_g)
    for p in (on, off):
        for _ in range(10):
            p.step(torch.full((1,), p.max_g, dtype=D), torch.zeros(1, 3, dtype=D), 0.01)
    assert float(on.vel[0, 2]) < float(off.vel[0, 2]) - 0.5


# ----------------------------------------------------------------------------- drain and sag


def test_sag_is_monotone_under_full_effort_and_negligible_at_hover():
    prof = micro()  # the committed (fitted) block
    ps = state(prof.power, 2)
    thrust = torch.tensor([prof.max_g, 1.0], dtype=D)
    sag, volts, charge = [], [], []
    for _ in range(1000):  # 10 s
        ps.update(thrust, torch.full((2,), 0.01, dtype=D))
        sag.append(ps.sag.clone())
        volts.append(ps.volts.clone())
        charge.append(ps.charge.clone())
    sag, volts, charge = torch.stack(sag), torch.stack(volts), torch.stack(charge)
    assert bool((torch.diff(sag[:, 0]) > 0).all())
    assert bool((torch.diff(volts[:, 0]) < 0).all())
    assert bool((torch.diff(charge[:, 0]) < 0).all())
    assert float(sag[-1, 0]) > 0.3  # a hard push costs a visible, persistent sag
    assert float(sag[-1, 1]) < 5e-3  # hover-level effort: negligible
    assert float(sag[-1, 0]) > 100 * float(sag[-1, 1])
    assert not bool(ps.cutoff.any())
    # persistent: idling afterwards never recovers it
    s_end = ps.sag.clone()
    for _ in range(100):
        ps.update(torch.zeros(2, dtype=D), torch.full((2,), 0.01, dtype=D))
    assert torch.equal(ps.sag, s_end)


def test_drain_and_voltage_follow_the_closed_form():
    sp = spec(drain_idle_per_s=0.001, drain_load_per_s=0.002, drain_exponent=2.0,
              sag_rate_v_per_s=0.01, sag_exponent=1.0, ir_v=0.1, ir_exponent=1.0,
              inflow_coeff=0.0)  # fmt: skip
    ps = state(sp)
    ps.update(torch.tensor([2.0], dtype=D), torch.tensor([0.5], dtype=D))
    q = 1.0 - (0.001 + 0.002 * 4.0) * 0.5
    s = 0.01 * 2.0 * 0.5
    assert float(ps.charge[0]) == pytest.approx(q)
    assert float(ps.sag[0]) == pytest.approx(s)
    assert float(ps.volts[0]) == pytest.approx(10.5 + 2.1 * q - s - 0.1 * 2.0)
    assert float(ps.effort[0]) == pytest.approx(2.0)


@pytest.mark.skipif("ir_scale" not in POWER_RANDOMIZABLE, reason="no ir_scale randomization")
def test_ir_scale_multiplies_the_load_drop():
    ps = state(spec(ir_v=0.2, ir_exponent=1.0, randomization={"ir_scale": Range(0.5, 1.5)}), 64,
               seed=3, scale=1.0)  # fmt: skip
    ps.update(torch.full((64,), 2.0, dtype=D), torch.full((64,), 0.01, dtype=D))
    np.testing.assert_allclose(ps.volts.numpy(), (12.6 - ps.ir_scale * 0.2 * 2.0).numpy())
    assert float(ps.ir_scale.std()) > 0.05


def test_effort_rises_with_inflow_by_the_effort_exponent():
    ps = state(spec())
    ps.available_g(torch.tensor([3.0], dtype=D), torch.tensor([20.0], dtype=D))  # inflow 0.6
    ps.update(torch.tensor([1.5], dtype=D), torch.tensor([0.01], dtype=D))
    assert float(ps.effort[0]) == pytest.approx(1.5 * 0.6**-0.8)


# ----------------------------------------------------------------------------- cutoff


def test_cutoff_latches_below_the_cell_threshold_and_idles_the_motors():
    ps = state(spec(), 2)
    assert ps.v_cut == pytest.approx(8.1)
    h = torch.full((2,), 0.01, dtype=D)
    ps.sag = torch.tensor([10.0, 0.0], dtype=D)  # env 0 below 2.7 V / cell
    ps.update(torch.zeros(2, dtype=D), h)
    assert ps.cutoff.tolist() == [True, False]
    ps.sag = torch.zeros(2, dtype=D)  # even if the voltage came back...
    ps.update(torch.zeros(2, dtype=D), h)
    assert ps.cutoff.tolist() == [True, False]  # ...it stays latched
    avail = ps.available_g(torch.full((2,), 3.0, dtype=D), torch.zeros(2, dtype=D))
    assert avail.tolist() == [0.0, 3.0]
    ps.reset(torch.tensor([True, False]))
    assert not bool(ps.cutoff.any())


def test_plant_after_cutoff_has_idle_motors_and_falls():
    p = _plant(with_power(micro(), spec()), 1, motor=1.0)
    p.power.sag = torch.tensor([10.0], dtype=D)
    full = torch.full((1,), p.max_g, dtype=D)
    zero = torch.zeros(1, 3, dtype=D)
    p.step(full, zero, 0.01)  # latches during this step
    assert bool(p.power.cutoff[0])
    for i in range(20):
        p.step(full, zero, 0.01)
        assert float(p.motor_g[0]) == 0.0, i
    assert bool(p.power.cutoff[0])
    assert float(p.vel[0, 2]) < -9.81 * 0.15  # ~free fall over 0.2 s


# ----------------------------------------------------------------------------- randomization


def test_randomization_ranges_are_respected_and_span_them():
    ps = state(spec(randomization=dict(RND)), 4000, seed=1, scale=1.0)
    for key, r in RND.items():
        v = getattr(ps, key)
        assert float(v.min()) >= r.min - 1e-12 and float(v.max()) <= r.max + 1e-12, key
        w = r.max - r.min
        assert float(v.min()) < r.min + 0.01 * w and float(v.max()) > r.max - 0.01 * w, key
    assert torch.equal(ps.charge, ps.initial_charge)
    assert torch.allclose(ps.volts, ps.ocv(ps.charge))


def test_randomization_scale_shrinks_ranges_toward_nominal():
    half = state(spec(randomization=dict(RND)), 4000, seed=1, scale=0.5)
    for key, r in RND.items():
        v = getattr(half, key)
        lo, hi = 1.0 + (r.min - 1.0) * 0.5, 1.0 + (r.max - 1.0) * 0.5
        assert float(v.min()) >= lo - 1e-12 and float(v.max()) <= hi + 1e-12, key
    zero = state(spec(randomization=dict(RND)), 64, seed=1, scale=0.0)
    for key in POWER_RANDOMIZABLE:
        assert bool((getattr(zero, key) == 1.0).all()), key
    assert bool((zero.charge == 1.0).all())


def test_randomized_inflow_scale_changes_the_cap_per_env():
    ps = state(spec(randomization={"inflow_scale": Range(0.5, 1.5)}), 256, seed=2, scale=1.0)
    ps.available_g(torch.full((256,), 3.0, dtype=D), torch.full((256,), 10.0, dtype=D))
    np.testing.assert_allclose(ps.inflow.numpy(), (1 - 0.4 * ps.inflow_scale * 0.5).numpy())
    assert float(ps.inflow.std()) > 0.01


def test_power_randomization_is_deterministic_per_seed():
    sp = spec(randomization=dict(RND))
    a, b, c = state(sp, 128, seed=5, scale=1.0), state(sp, 128, seed=5, scale=1.0), state(
        sp, 128, seed=6, scale=1.0)  # fmt: skip
    for key in RND:
        assert torch.equal(getattr(a, key), getattr(b, key)), key
        assert not torch.equal(getattr(a, key), getattr(c, key)), key
    mask = torch.arange(128) < 64
    a.reset(mask)
    b.reset(mask)
    for key in RND:
        assert torch.equal(getattr(a, key), getattr(b, key)), key
    # a masked reset resamples only the masked envs
    keep = state(sp, 128, seed=5, scale=1.0)
    for key in RND:
        assert torch.equal(getattr(a, key)[~mask], getattr(keep, key)[~mask]), key
        assert not torch.equal(getattr(a, key)[mask], getattr(keep, key)[mask]), key


def test_plant_power_runs_are_deterministic_per_seed():
    prof = with_power(micro(), spec(randomization=dict(RND), drain_load_per_s=0.004,
                                    sag_rate_v_per_s=0.001, sag_exponent=3.0, ir_v=0.3,
                                    thrust_voltage_exponent=0.1))  # fmt: skip
    _, ta = _rollout(prof, True, seed=9)
    _, tb = _rollout(prof, True, seed=9)
    _, tc = _rollout(prof, True, seed=10)
    assert torch.equal(ta, tb) and not torch.equal(ta, tc)


def test_plant_reset_restores_a_fresh_pack():
    p = _plant(with_power(micro(), spec(drain_load_per_s=0.05, sag_rate_v_per_s=0.01)), 2)
    for _ in range(50):
        p.step(torch.full((2,), 2.0, dtype=D), torch.zeros(2, 3, dtype=D), 0.02)
    assert bool((p.power.charge < 1.0).all()) and bool((p.power.sag > 0).all())
    p.reset(torch.tensor([True, False]), resample=False)
    assert float(p.power.charge[0]) == 1.0 and float(p.power.sag[0]) == 0.0
    assert float(p.power.volts[0]) == pytest.approx(12.6)
    assert float(p.power.charge[1]) < 1.0


# ----------------------------------------------------------------------------- schema


def test_micro_profile_carries_the_power_block():
    pw = micro().power
    assert isinstance(pw, PowerSpec)
    assert pw.cells == 3 and pw.ocv_charge == (0.0, 1.0)
    assert pw.inflow_min_factor == pytest.approx(0.2)
    assert pw.inflow_v_ref == pytest.approx(20.0)
    assert pw.cutoff_v_per_cell == pytest.approx(2.70)
    assert set(pw.randomization) == set(POWER_RANDOMIZABLE)
    for r in pw.randomization.values():
        assert r.min <= 1.0 <= r.max
    assert pw.fit is not None


def test_profiles_without_a_power_block_load():
    assert meteor75().power is None
    from tests.test_drones_schema import sysid_profile

    assert parse_profile(sysid_profile(), expected_id="liftoff-test-micro").power is None


POWER_BLOCK = {
    "cells": 3,
    "ocv": {"charge": [0.0, 1.0], "volts": [10.4, 12.6]},
    "sag": {"rate_v_per_s": 0.0001, "exponent": 5.0},
    "ir": {"v": 0.5, "exponent": 0.7},
    "drain": {"idle_per_s": 0.0, "load_per_s": 0.004, "exponent": 0.95},
    "thrust_voltage_exponent": 0.05,
    "inflow": {"coeff": 0.38, "v_ref_mps": 20.0, "min_factor": 0.2, "effort_exponent": 0.8},
    "cutoff_v_per_cell": 2.7,
    "randomization": {"drain_scale": {"min": 0.9, "max": 1.1},
                      "initial_charge": {"min": 0.95, "max": 1.0}},
}  # fmt: skip


def _with_block(block):
    from tests.test_drones_schema import sysid_profile

    data = sysid_profile()
    data["power"] = block
    return parse_profile(data, expected_id="liftoff-test-micro")


def test_a_valid_power_block_parses():
    pw = _with_block(copy.deepcopy(POWER_BLOCK)).power
    assert pw.cells == 3 and pw.ocv_volts == (10.4, 12.6)
    assert pw.inflow_coeff == pytest.approx(0.38) and pw.effort_inflow_exponent == pytest.approx(0.8)
    assert pw.randomization["drain_scale"] == Range(0.9, 1.1)
    assert set(pw.randomization) == {"drain_scale", "initial_charge"}


def _bad(path, value):
    block = copy.deepcopy(POWER_BLOCK)
    node = block
    keys = path.split(".")
    for k in keys[:-1]:
        node = node[k]
    if value is _DELETE:
        del node[keys[-1]]
    else:
        node[keys[-1]] = value
    return block


_DELETE = object()


@pytest.mark.parametrize(
    ("path", "value", "field"),
    [
        ("inflow.min_factor", 1.5, "power.inflow.min_factor"),
        ("inflow.min_factor", -0.1, "power.inflow.min_factor"),
        ("inflow.effort_exponent", 1.2, "power.inflow.effort_exponent"),
        ("inflow.v_ref_mps", 0.0, "power.inflow.v_ref_mps"),
        ("inflow.coeff", -0.1, "power.inflow.coeff"),
        ("cells", 0, "power.cells"),
        ("cutoff_v_per_cell", "low", "power.cutoff_v_per_cell"),
        ("drain.load_per_s", -1.0, "power.drain.load_per_s"),
        ("sag.exponent", float("nan"), "power.sag.exponent"),
        ("ocv.volts", [12.6], "power.ocv"),
        ("ocv.charge", [1.0, 0.0], "power.ocv"),
        ("sag", _DELETE, "power.sag"),
        ("ir.v", _DELETE, "power.ir.v"),
        ("inflow.extra", 1.0, "power.inflow.extra"),
        ("battery_mah", 300, "power.battery_mah"),
        ("randomization.thrust_scale", {"min": 0.9, "max": 1.1}, "power.randomization.thrust_scale"),
        ("randomization.drain_scale", {"min": 1.05, "max": 1.1}, "power.randomization.drain_scale"),
        ("randomization.drain_scale", {"min": 0.9, "max": 0.95}, "power.randomization.drain_scale"),
        ("randomization.drain_scale", {"min": -0.1, "max": 1.1}, "power.randomization.drain_scale.min"),
    ],
)  # fmt: skip
def test_bad_power_values_are_refused_naming_the_field(path, value, field):
    with pytest.raises(DroneProfileError) as exc:
        _with_block(_bad(path, value))
    assert exc.value.field == field


def test_power_block_must_be_a_mapping():
    with pytest.raises(DroneProfileError) as exc:
        _with_block([1, 2, 3])
    assert exc.value.field == "power"


# ----------------------------------------------------------------------------- the env


@pytest.fixture(scope="module")
def field_course(tmp_path_factory):
    from tests.tracks_support import build_library, load

    lib = build_library(tmp_path_factory.mktemp("lib"))
    return load(lib, "fake-field-01")


def _env(course, *, power, n=1):
    from drone_fly.sim.multi_course import MultiCourseConfig, MultiCourseEnv
    from drone_fly.sim.pool import PoolGroup, pack_courses
    from drone_fly.sim.timing import TimingConfig
    from tests.tracks_support import action_spec

    cfg = MultiCourseConfig.from_dict({"reward_reference_dt_s": 0.02,
                                       "start": {"ground_fraction": 0.0}})  # fmt: skip
    env = MultiCourseEnv(
        drones.load("liftoff-gremlin-micro"), n, pack_courses([course]),
        [PoolGroup("all", [0], 1.0)], device="cpu", seed=0, config=cfg,
        plant_config=PlantConfig(power=power), timing=TimingConfig.fixed(50.0),
        randomization_scale=0.0, action_spec=action_spec(), log_capacity=16,
    )  # fmt: skip
    env.reset_all(torch.zeros(n, dtype=torch.long))
    return env


def _place_before_first_gate(env, course):
    from drone_fly.sim.multi_course import START_AIR

    n = env.n
    p0 = course.passages[0]
    pos = torch.as_tensor(np.asarray(p0.center - 2 * p0.normal, np.float64), dtype=env.dtype)
    env.plant.reset(all_envs(n), pos=pos.expand(n, 3).clone(), motor_g=1.0, grounded=False,
                    resample=False)  # fmt: skip
    env.start_mode = torch.full((n,), START_AIR, dtype=torch.long)
    env.arm_until = torch.zeros(n, dtype=torch.float64)
    env.lift_t = torch.zeros(n, dtype=torch.float64)
    env.k = torch.zeros(n, dtype=torch.long)


def _hover(env):
    return torch.zeros(env.n, 4, dtype=env.dtype)


def test_power_loss_is_the_last_outcome_and_counts_as_a_crash():
    from drone_fly.sim.multi_course import OUTCOMES

    assert OUTCOMES[-1] == "power_loss"
    assert OUTCOMES[:6] == ("none", "floor", "collision", "out-of-bounds", "timeout", "completed")


def test_cutoff_ends_the_episode_with_power_loss(field_course):
    from drone_fly.sim.multi_course import EPISODE_COLUMNS, OUTCOMES

    env = _env(field_course, power=True)
    _place_before_first_gate(env, field_course)
    _, done = env.step(_hover(env))
    assert not bool(done[0])  # control: a healthy pack flies on
    env.plant.power.sag = torch.full_like(env.plant.power.sag, 100.0)
    r, done = env.step(_hover(env))
    assert bool(done[0])
    assert OUTCOMES[int(env.last_outcome[0])] == "power_loss"
    assert bool(env.last_crashed[0]) and not bool(env.last_completed[0])
    assert float(r[0]) < -9.0  # the crash penalty
    rows = env.episode_rows()
    assert int(rows[-1, EPISODE_COLUMNS.index("outcome")]) == OUTCOMES.index("power_loss")
    # the respawn gets a fresh pack
    assert not bool(env.plant.power.cutoff[0])
    assert float(env.plant.power.sag[0]) == 0.0


def test_no_power_model_never_reports_power_loss(field_course):
    from drone_fly.sim.multi_course import OUTCOMES

    env = _env(field_course, power=False)
    assert env.plant.power is None
    _place_before_first_gate(env, field_course)
    for _ in range(5):
        env.step(_hover(env))
        assert OUTCOMES[int(env.last_outcome[0])] != "power_loss"


def test_critic_extra_carries_the_battery(field_course):
    from drone_fly.sim.multi_course import CRITIC_EXTRA_DIM, FULL_CELL_V
    from drone_fly.sim.randomization import LOG_COLUMNS
    from drone_fly.sim.timing import TIMING_LOG_COLUMNS

    assert CRITIC_EXTRA_DIM == len(LOG_COLUMNS) + len(TIMING_LOG_COLUMNS) + 4
    off = _env(field_course, power=False, n=2)
    ce = off.critic_extra()
    assert ce.shape == (2, CRITIC_EXTRA_DIM)
    np.testing.assert_allclose(ce[:, -2:].numpy(), [[1.0, FULL_CELL_V]] * 2)

    on = _env(field_course, power=True, n=2)
    ce = on.critic_extra()
    assert ce.shape == (2, CRITIC_EXTRA_DIM)
    pw = on.plant.power
    full = pw.v_full / pw.spec.cells
    np.testing.assert_allclose(ce[:, -2:].numpy(), [[1.0, full]] * 2, rtol=1e-6)
    pw.charge = torch.tensor([0.5, 0.25], dtype=pw.dtype)
    pw.volts = torch.tensor([11.1, 10.2], dtype=pw.dtype)
    np.testing.assert_allclose(on.critic_extra()[:, -2:].numpy(),
                               [[0.5, 3.7], [0.25, 3.4]], rtol=1e-6)  # fmt: skip
    # the leading columns are the power-off ones (power appends, it does not shift)
    np.testing.assert_allclose(on.critic_extra()[:, :-2].numpy(), off.critic_extra()[:, :-2].numpy())


# ----------------------------------------------------------------------------- power_fit math


def test_fit_inflow_recovers_known_parameters():
    import pandas as pd

    from drone_fly.sysid.power_fit import V_REF_INFLOW, fit_inflow

    max_g, v_full, n, c, beta, rpm_max = 3.6, 12.6, 0.5, 0.38, 0.8, 33000.0
    rng = np.random.default_rng(0)
    m = 400
    v_ax = np.concatenate([[0.0], rng.uniform(-3.0, 28.0, m - 1)])
    volts = np.concatenate([[v_full], rng.uniform(10.0, 12.6, m - 1)])
    inflow = np.clip(1 - c * np.clip(v_ax, 0, None) / V_REF_INFLOW, 0.2, None)
    T = max_g * (volts / v_full) ** n * inflow
    rpm = rpm_max * inflow ** (1 - beta)  # E = T_zero_volt * inflow^-beta, max at inflow 1
    win = pd.DataFrame({"T_meas": T, "v_ax": v_ax, "speed": np.abs(v_ax), "rpm": rpm,
                        "volts": volts, "thr": np.full(m, 1.0)})  # fmt: skip
    # part-throttle windows are ignored
    win = pd.concat([win, pd.DataFrame({"T_meas": [9.0], "v_ax": [5.0], "speed": [5.0],
                                        "rpm": [20000.0], "volts": [12.0], "thr": [0.5]})])  # fmt: skip
    out = fit_inflow(win, max_g, v_full)
    assert out["thrust_voltage_exponent"] == pytest.approx(n, abs=1e-6)
    assert out["inflow_coeff"] == pytest.approx(c, abs=1e-6)
    assert out["effort_inflow_exponent"] == pytest.approx(beta, abs=1e-6)
    assert out["rmse_g"] < 1e-9 and out["windows"] == m


def test_fit_battery_recovers_known_parameters():
    from drone_fly.sysid.power_fit import fit_battery, simulate_battery

    true = [0.0005, 0.004, 1.0, 0.0002, 4.0, 2.2, 0.5, 0.7]
    v_full, v_cut = 12.6, 8.1
    rng = np.random.default_rng(0)
    sessions = []
    for _ in range(3):
        t = np.arange(0.0, 40.0, 0.05)
        levels = rng.uniform(0.5, 3.5, size=len(t) // 40 + 1)
        s = {"t": t, "E": np.repeat(levels, 40)[: len(t)], "q": np.ones_like(t),
             "V": np.zeros_like(t)}  # fmt: skip
        s["q"], s["V"] = simulate_battery(true, s, v_full)
        sessions.append(s)
    fit = fit_battery(sessions, v_full, v_cut)
    names = ["drain_idle_per_s", "drain_load_per_s", "drain_exponent", "sag_rate_v_per_s",
             "sag_exponent", "ocv_slope_v", "ir_v", "ir_exponent"]  # fmt: skip
    for name, v in zip(names, true, strict=True):
        assert fit[name] == pytest.approx(v, rel=1e-3), name
    for s in fit["sessions"]:
        assert s["v_rmse"] < 1e-6 and s["q_rmse"] < 1e-6


def test_simulate_battery_matches_the_plant_power_state():
    # the fit's replay and the plant's PowerState are the same ODE (no inflow, linear OCV)
    from drone_fly.sysid.power_fit import simulate_battery

    d0, d1, k, a, p, b, r, pr = 0.0005, 0.004, 1.0, 0.0002, 4.0, 2.1, 0.5, 0.7
    sp = spec(drain_idle_per_s=d0, drain_load_per_s=d1, drain_exponent=k, sag_rate_v_per_s=a,
              sag_exponent=p, ir_v=r, ir_exponent=pr, inflow_coeff=0.0)  # fmt: skip
    t = np.arange(0.0, 5.0, 0.01)
    E = np.where(t < 2.5, 3.0, 1.0)
    q, V = simulate_battery([d0, d1, k, a, p, b, r, pr],
                            {"t": t, "E": E, "q": np.ones_like(t)}, 12.6)  # fmt: skip
    ps = state(sp)
    for i in range(1, len(t)):
        ps.update(torch.tensor([E[i - 1]], dtype=D), torch.tensor([0.01], dtype=D))
        assert float(ps.charge[0]) == pytest.approx(q[i], abs=1e-9)
    assert math.isclose(float(ps.ocv(ps.charge)[0]), 12.6 - b * (1 - q[-1]), abs_tol=1e-9)
