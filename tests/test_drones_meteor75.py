"""UC-66 AC3: ``drones/meteor75/drone.yaml`` reproduces the plant the policies train on today.

The sources are ``gpu_prototype/hover_gpu.py`` (BatchedDrone nominal), ``gpu_prototype/gates_dr.py``
(the domain-randomization ranges) and ``src/drone_fly/adapter/meteor75.py`` (the analog block).
The two ``gpu_prototype`` modules bind CUDA at import, so their constants are read by AST literal
extraction and NEVER imported here.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from drone_fly import drones
from drone_fly.adapter import meteor75 as m75

REPO = Path(__file__).resolve().parent.parent
GPU = REPO / "gpu_prototype"


def _constants(path: Path, names: set[str]) -> dict[str, object]:
    """Module-level ``NAME = <expr>`` constants, evaluated over literals and earlier constants."""
    env: dict[str, object] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    value = eval(  # noqa: S307 - literal arithmetic over our own constants
                        compile(ast.Expression(node.value), str(path), "eval"),
                        {"__builtins__": {}},
                        dict(env),
                    )
                except Exception:  # noqa: BLE001 - not a literal constant
                    continue
                env[target.id] = value
    missing = names - set(env)
    assert not missing, f"{path.name}: constants not found {missing}"
    return {k: env[k] for k in names}


@pytest.fixture(scope="module")
def hover():
    return _constants(
        GPU / "hover_gpu.py", {"M", "G", "TW", "MAX_BODY_RATE", "RATE_KP", "DT", "HOVER_THR"}
    )


@pytest.fixture(scope="module")
def dr():
    return _constants(GPU / "gates_dr.py", {"MASS", "TW", "LAG_TAU", "DRAG", "RATE_KP", "LAT"})


@pytest.fixture(scope="module")
def prof():
    return drones.load("meteor75", REPO / "drones")


def test_profile_is_an_analog(prof):
    assert prof.id == "meteor75" and prof.source == "analog"
    assert prof.camera is None


def test_hover_and_thrust_curve_match_batched_drone(prof, hover):
    assert prof.hover_throttle == pytest.approx(1.0 / hover["TW"])
    assert prof.hover_throttle == pytest.approx(hover["HOVER_THR"])
    # linear in collective: 0 g at 0, TW g at full
    assert prof.thrust_curve.throttle == (0.0, 1.0)
    assert prof.thrust_curve.accel_g[0] == 0.0
    assert prof.max_g == pytest.approx(hover["TW"])
    assert prof.thrust_g(prof.hover_throttle) == pytest.approx(1.0)


@pytest.mark.parametrize("axis", ["roll", "pitch", "yaw"])
def test_rates_match_batched_drone(prof, hover, axis):
    r = prof.rates[axis]
    rp = prof.rate_profile(axis)
    full = math.radians(rp.rate_deg_s(1.0))
    assert full == pytest.approx(hover["MAX_BODY_RATE"])
    # linear stick -> rate (no super rate, no expo, no rc_rate boost branch)
    assert r.super_rate == 0.0 and r.expo == 0.0 and r.rc_rate < 2.0
    for s in (0.1, 0.25, 0.5, 0.8):
        assert math.radians(rp.rate_deg_s(s)) == pytest.approx(s * hover["MAX_BODY_RATE"])
    # tau is the continuous equivalent of the Euler step's gain (no step-response claim)
    assert 1.0 / r.tau_s == pytest.approx(hover["RATE_KP"])
    assert r.delay_s == 0.0


def test_no_drag_lag_or_latency_in_the_nominal(prof):
    assert prof.drag.linear_per_s == 0.0 and prof.drag.quadratic_per_m == 0.0
    assert prof.motor_lag_s == 0.0
    assert prof.latency.input_s == 0.0 and prof.latency.control_to_gyro_s == 0.0


def test_analog_block_matches_meteor75_adapter(prof, hover):
    a = prof.analog
    assert a["mass_kg"] == pytest.approx(m75.METEOR75_MASS) == pytest.approx(hover["M"])
    assert a["thrust_to_weight"] == pytest.approx(m75.METEOR75_TW) == pytest.approx(hover["TW"])
    assert a["wheelbase_m"] == pytest.approx(m75.METEOR75_WHEELBASE)
    assert a["arm_m"] == pytest.approx(m75.METEOR75_ARM)
    assert a["prop_diameter_m"] == pytest.approx(m75.METEOR75_PROP_DIAMETER)
    assert a["motor_fraction"] == pytest.approx(m75.METEOR75_MOTOR_FRACTION)
    # gravity / dt are the batched trainer's (9.8 in meteor75.py is the pybullet aviary's)
    assert a["gravity_m_s2"] == pytest.approx(hover["G"])
    assert a["sim_dt_s"] == pytest.approx(hover["DT"])


def test_randomization_matches_gates_dr(prof, hover, dr):
    rnd = prof.randomization
    tw_lo, tw_hi = dr["TW"]
    assert rnd["thrust_scale"].min == pytest.approx(tw_lo / hover["TW"])
    assert rnd["thrust_scale"].max == pytest.approx(tw_hi / hover["TW"])
    kp_lo, kp_hi = dr["RATE_KP"]
    for axis in ("roll", "pitch", "yaw"):
        r = rnd[f"rates.{axis}.tau_s"]
        assert r.min == pytest.approx(1.0 / kp_hi) and r.max == pytest.approx(1.0 / kp_lo)
    # motor lag: upper bound from LAG_TAU; lower widened to 0 to contain the nominal 0
    assert rnd["motor_lag_s"].max == pytest.approx(dr["LAG_TAU"][1])
    assert rnd["motor_lag_s"].min == 0.0
    # latency: 0..LAT action steps of DT (gates_dr delays every channel, throttle too)
    assert rnd["latency.control_to_gyro_s"].min == 0.0
    assert rnd["latency.control_to_gyro_s"].max == pytest.approx(dr["LAT"] * hover["DT"])
    # drag: a quadratic FORCE coefficient divided by the randomized mass -> accel 0..DRAG/MASS_min
    d = rnd["drag.quadratic_per_m"]
    assert d.min == pytest.approx(dr["DRAG"][0] / dr["MASS"][0]) == 0.0
    assert d.max == pytest.approx(dr["DRAG"][1] / dr["MASS"][0])
    # mass cancels out of thrust (maxthrust = TW * mass * G): no mass randomization
    assert not any("mass" in k for k in rnd)


def test_nominal_sits_inside_every_range(prof):
    nominal = {
        "thrust_scale": 1.0,
        "motor_lag_s": prof.motor_lag_s,
        "latency.control_to_gyro_s": prof.latency.control_to_gyro_s,
        "drag.quadratic_per_m": prof.drag.quadratic_per_m,
        **{f"rates.{a}.tau_s": prof.rates[a].tau_s for a in ("roll", "pitch", "yaw")},
    }
    for key, r in prof.randomization.items():
        assert r.min <= nominal[key] <= r.max, key


def test_meteor75_folder_has_a_report():
    text = (REPO / "drones" / "meteor75" / "sysid.md").read_text(encoding="utf-8")
    assert "analog" in text.lower()
