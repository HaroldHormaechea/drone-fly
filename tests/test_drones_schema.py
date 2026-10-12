"""UC-66 AC2: the drone profile schema (``drones/<id>/drone.yaml``) and its loader.

Every validation error must be a :class:`DroneProfileError` naming the dotted field path; the
loader resolves the root from the argument, then ``DRONE_FLY_DRONES_DIR``, then ``<repo>/drones``.
Hermetic: profiles are written into ``tmp_path``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from drone_fly import drones
from drone_fly.drones import DroneProfileError
from drone_fly.drones.schema import RANDOMIZABLE, parse_profile
from drone_fly.drones.store import ENV_ROOT, drones_root

SESSION = "20261012T100000Z"


def _m(value: Any) -> dict[str, Any]:
    """A measured leaf with session + residual provenance."""
    return {"value": value, "session": SESSION, "residual": 0.01, "sigma": 0.001}


def sysid_profile(pid: str = "liftoff-test-micro") -> dict[str, Any]:
    """A minimal valid ``source: liftoff-sysid`` profile (every leaf with provenance)."""
    axis = {
        "rc_rate": _m(1.0),
        "super_rate": _m(0.7),
        "expo": _m(0.0),
        "tau_s": _m(0.03),
        "delay_s": _m(0.01),
    }
    return {
        "schema_version": 1,
        "id": pid,
        "name": "Test micro",
        "source": "liftoff-sysid",
        "fit_quality": "good",
        "liftoff": {"drone_name": "Test small drone", "map": "Flat"},
        "nominal": {
            "hover_throttle": _m(0.47),
            "thrust_curve": {
                "throttle": [0.0, 0.47, 1.0],
                "accel_g": [0.0, 1.0, 2.4],
                "session": SESSION,
                "residual": 0.02,
            },
            "rates": {a: copy.deepcopy(axis) for a in ("roll", "pitch", "yaw")},
            "drag": {"linear_per_s": _m(0.3), "quadratic_per_m": _m(0.05)},
            "motor_lag_s": _m(0.03),
            "latency": {"input_s": _m(0.008), "control_to_gyro_s": _m(0.018)},
            "camera": {"hfov_deg": 120.0, "uptilt_deg": 30.0, "ref": "preset liftoff camera"},
        },
        "randomization": {
            "thrust_scale": {"min": 0.9, "max": 1.1},
            "rates.roll.tau_s": {"min": 0.02, "max": 0.04},
            "drag.linear_per_s": {"min": 0.2, "max": 0.4},
            "motor_lag_s": {"min": 0.0, "max": 0.06},
            "latency.control_to_gyro_s": {"min": 0.0, "max": 0.04},
        },
    }


def analog_profile(pid: str = "analog-test") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": pid,
        "name": "Analog",
        "source": "analog",
        "nominal": {
            "hover_throttle": 0.4,
            "thrust_curve": {"throttle": [0.0, 1.0], "accel_g": [0.0, 2.5]},
            "rates": {
                a: {"rc_rate": 1.0, "super_rate": 0.0, "expo": 0.0, "tau_s": 0.05, "delay_s": 0.0}
                for a in ("roll", "pitch", "yaw")
            },
            "drag": {"linear_per_s": 0.0, "quadratic_per_m": 0.0},
            "motor_lag_s": 0.0,
            "latency": {"input_s": 0.0, "control_to_gyro_s": 0.0},
            "camera": None,
        },
    }


def write(root: Path, data: dict[str, Any], folder: str | None = None) -> Path:
    d = root / (folder or data["id"])
    d.mkdir(parents=True, exist_ok=True)
    path = d / "drone.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _set(data: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    node = data
    for k in keys[:-1]:
        node = node[k]
    if value is _DELETE:
        del node[keys[-1]]
    else:
        node[keys[-1]] = value


_DELETE = object()


# ----------------------------------------------------------------------------- happy path


def test_valid_sysid_profile_loads_with_provenance(tmp_path):
    write(tmp_path, sysid_profile())
    prof = drones.load("liftoff-test-micro", tmp_path)
    assert prof.source == "liftoff-sysid" and prof.fit_quality == "good"
    assert prof.hover_throttle == pytest.approx(0.47)
    assert prof.rates["roll"].tau_s == pytest.approx(0.03)
    assert prof.drag.linear_per_s == pytest.approx(0.3)
    assert prof.latency.control_to_gyro_s == pytest.approx(0.018)
    assert prof.camera is not None and prof.camera.hfov_deg == 120.0
    # AC2: every measured value carries session + residual
    p = prof.provenance("rates.roll.tau_s")
    assert p["session"] == SESSION and p["residual"] == 0.01
    assert prof.provenance("nominal.hover_throttle")["session"] == SESSION
    # block-level provenance covers the leaves inside the block
    assert prof.provenance("thrust_curve.accel_g")["residual"] == 0.02
    assert prof.provenance("camera.hfov_deg")["ref"] == "preset liftoff camera"
    assert prof.randomization["thrust_scale"].max == pytest.approx(1.1)


def test_analog_profile_needs_no_provenance_or_fit_quality(tmp_path):
    write(tmp_path, analog_profile())
    prof = drones.load("analog-test", tmp_path)
    assert prof.fit_quality is None and prof.camera is None and prof.randomization == {}
    assert prof.provenance("hover_throttle") is None


def test_isotropic_or_body_axis_drag(tmp_path):
    data = sysid_profile()
    data["nominal"]["drag"]["quadratic_per_m"] = _m([0.05, 0.06, 0.1])
    data["randomization"]["drag.quadratic_per_m"] = {"min": 0.0, "max": 0.2}
    prof = parse_profile(data, expected_id="liftoff-test-micro")
    assert prof.drag.quadratic_per_m == (0.05, 0.06, 0.1)
    data["randomization"]["drag.quadratic_per_m"] = {"min": 0.055, "max": 0.2}
    with pytest.raises(DroneProfileError) as exc:
        parse_profile(data)
    assert exc.value.field == "randomization.drag.quadratic_per_m"


# ----------------------------------------------------------------------------- field-path errors

#: (dotted path to mutate, new value, expected DroneProfileError.field)
FIELD_ERRORS = [
    ("schema_version", 2, "schema_version"),
    ("schema_version", _DELETE, "schema_version"),
    ("id", "Bad_Id", "id"),
    ("name", _DELETE, "name"),
    ("source", "guess", "source"),
    ("fit_quality", _DELETE, "fit_quality"),
    ("fit_quality", "great", "fit_quality"),
    ("liftoff", _DELETE, "liftoff"),
    ("liftoff.drone_name", "", "liftoff.drone_name"),
    ("nominal", _DELETE, "nominal"),
    ("nominal.hover_throttle", _DELETE, "nominal.hover_throttle"),
    ("nominal.hover_throttle", _m(1.0), "nominal.hover_throttle"),
    ("nominal.hover_throttle", _m(0.0), "nominal.hover_throttle"),
    ("nominal.hover_throttle", _m("half"), "nominal.hover_throttle"),
    ("nominal.hover_throttle", _m(float("nan")), "nominal.hover_throttle"),
    # hover must sit on the curve (1.0 g +- 0.05)
    ("nominal.hover_throttle", _m(0.6), "nominal.hover_throttle"),
    ("nominal.thrust_curve", _DELETE, "nominal.thrust_curve"),
    ("nominal.thrust_curve.throttle", [0.0], "nominal.thrust_curve.throttle"),
    ("nominal.thrust_curve.throttle", [0.1, 0.47, 1.0], "nominal.thrust_curve.throttle"),
    ("nominal.thrust_curve.throttle", [0.0, 0.47, 0.9], "nominal.thrust_curve.throttle"),
    ("nominal.thrust_curve.throttle", [0.0, 1.0, 1.0], "nominal.thrust_curve.throttle[2]"),
    ("nominal.thrust_curve.accel_g", [0.0, 1.0], "nominal.thrust_curve.accel_g"),
    ("nominal.thrust_curve.accel_g", [0.0, 1.0, 0.9], "nominal.thrust_curve.accel_g[2]"),
    ("nominal.thrust_curve.accel_g", [-0.1, 1.0, 2.4], "nominal.thrust_curve.accel_g[0]"),
    ("nominal.rates.yaw", _DELETE, "nominal.rates.yaw"),
    ("nominal.rates.roll.rc_rate", _m(0.0), "nominal.rates.roll.rc_rate"),
    ("nominal.rates.pitch.super_rate", _m(1.0), "nominal.rates.pitch.super_rate"),
    ("nominal.rates.pitch.expo", _m(1.5), "nominal.rates.pitch.expo"),
    ("nominal.rates.yaw.tau_s", _m(0.0), "nominal.rates.yaw.tau_s"),
    ("nominal.rates.yaw.tau_s", _m(1.5), "nominal.rates.yaw.tau_s"),
    ("nominal.rates.roll.delay_s", _m(-0.01), "nominal.rates.roll.delay_s"),
    ("nominal.rates.roll.delay_s", _m(0.6), "nominal.rates.roll.delay_s"),
    ("nominal.rates.roll.delay_s", _DELETE, "nominal.rates.roll.delay_s"),
    ("nominal.rates.roll.wobble", 1.0, "nominal.rates.roll.wobble"),
    ("nominal.drag.linear_per_s", _m(-0.1), "nominal.drag.linear_per_s"),
    ("nominal.drag.quadratic_per_m", _m([0.1, 0.2]), "nominal.drag.quadratic_per_m"),
    ("nominal.drag.quadratic_per_m", _m([0.1, -0.2, 0.1]), "nominal.drag.quadratic_per_m[1]"),
    ("nominal.motor_lag_s", _m(0.7), "nominal.motor_lag_s"),
    ("nominal.latency.input_s", _DELETE, "nominal.latency.input_s"),
    ("nominal.latency.control_to_gyro_s", _m(-1.0), "nominal.latency.control_to_gyro_s"),
    ("nominal.camera.hfov_deg", 180.0, "nominal.camera.hfov_deg"),
    ("nominal.camera.uptilt_deg", 95.0, "nominal.camera.uptilt_deg"),
    ("nominal.mass_kg", 0.03, "nominal.mass_kg"),
    ("nominal.hover_throttle", {"session": SESSION}, "nominal.hover_throttle"),
    ("nominal.hover_throttle", {"value": 0.47, "who": "me"}, "nominal.hover_throttle.who"),
    ("colour", "red", "colour"),
    ("randomization.thrust_scale", {"min": 1.1, "max": 1.3}, "randomization.thrust_scale"),
    ("randomization.thrust_scale", {"min": 1.1, "max": 0.9}, "randomization.thrust_scale.max"),
    ("randomization.thrust_scale", {"min": 0.9}, "randomization.thrust_scale.max"),
    ("randomization.thrust_scale", {"min": 0.9, "max": 1.1, "p": 1}, "randomization.thrust_scale.p"),  # noqa: E501
    ("randomization.motor_lag_s", {"min": 0.04, "max": 0.06}, "randomization.motor_lag_s"),
    ("randomization.rates.roll.tau_s", {"min": -0.1, "max": 0.04}, "randomization.rates.roll.tau_s.min"),  # noqa: E501
    ("randomization.mass_kg", {"min": 0.02, "max": 0.04}, "randomization.mass_kg"),
]  # fmt: skip


def _mutate(data: dict[str, Any], path: str, value: Any) -> None:
    if path.startswith("randomization."):
        key = path[len("randomization.") :]
        if value is _DELETE:
            del data["randomization"][key]
        else:
            data["randomization"][key] = value
    else:
        _set(data, path, value)


@pytest.mark.parametrize(("path", "value", "field"), FIELD_ERRORS)
def test_every_error_names_its_field(tmp_path, path, value, field):
    data = sysid_profile()
    _mutate(data, path, copy.deepcopy(value) if value is not _DELETE else value)
    file = write(tmp_path, data, folder="liftoff-test-micro")
    with pytest.raises(DroneProfileError) as exc:
        drones.load("liftoff-test-micro", tmp_path)
    assert exc.value.field == field
    # the message names the file and the field (AC2: "a clear error naming the field")
    assert field in str(exc.value) and str(file) in str(exc.value)


@pytest.mark.parametrize(
    "leaf",
    [
        "hover_throttle",
        "rates.roll.tau_s",
        "rates.yaw.delay_s",
        "drag.quadratic_per_m",
        "motor_lag_s",
        "latency.input_s",
    ],
)
def test_measured_leaf_without_session_and_residual_is_rejected(leaf):
    data = sysid_profile()
    node = data["nominal"]
    keys = leaf.split(".")
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = {"value": node[keys[-1]]["value"], "session": SESSION}  # no residual
    with pytest.raises(DroneProfileError) as exc:
        parse_profile(data)
    assert exc.value.field == f"nominal.{leaf}" and "provenance" in str(exc.value)
    # a bare value is rejected too; a ref satisfies the rule
    node[keys[-1]] = node[keys[-1]]["value"]
    with pytest.raises(DroneProfileError):
        parse_profile(data)
    node[keys[-1]] = {"value": node[keys[-1]], "ref": "UC64 hand-flown"}
    assert parse_profile(data).provenance(leaf)["ref"] == "UC64 hand-flown"


def test_block_provenance_covers_its_leaves():
    data = sysid_profile()
    data["nominal"]["rates"]["roll"] = {
        "rc_rate": 1.0,
        "super_rate": 0.7,
        "expo": 0.0,
        "tau_s": 0.03,
        "delay_s": 0.01,
        "session": SESSION,
        "residual": 0.2,
    }
    prof = parse_profile(data)
    assert prof.provenance("rates.roll.expo")["residual"] == 0.2


def test_folder_must_match_id(tmp_path):
    write(tmp_path, sysid_profile("liftoff-other-micro"), folder="liftoff-test-micro")
    with pytest.raises(DroneProfileError) as exc:
        drones.load("liftoff-test-micro", tmp_path)
    assert exc.value.field == "id" and "folder" in str(exc.value)


def test_invalid_yaml_is_a_profile_error(tmp_path):
    d = tmp_path / "broken"
    d.mkdir()
    (d / "drone.yaml").write_text("nominal: [unclosed\n", encoding="utf-8")
    with pytest.raises(DroneProfileError, match="invalid YAML"):
        drones.load("broken", tmp_path)


def test_randomizable_paths_are_the_documented_set():
    expected = {"thrust_scale", "drag.linear_per_s", "drag.quadratic_per_m", "motor_lag_s"}
    expected |= {"latency.control_to_gyro_s"}
    for a in ("roll", "pitch", "yaw"):
        expected |= {f"rates.{a}.rate_scale", f"rates.{a}.tau_s", f"rates.{a}.delay_s"}
    assert set(RANDOMIZABLE) == expected


# ----------------------------------------------------------------------------- store


def test_unknown_id_lists_the_available_ids(tmp_path):
    write(tmp_path, sysid_profile())
    write(tmp_path, analog_profile())
    with pytest.raises(DroneProfileError) as exc:
        drones.load("nope", tmp_path)
    msg = str(exc.value)
    assert exc.value.field == "id" and "nope" in msg
    assert "liftoff-test-micro" in msg and "analog-test" in msg
    assert drones.list_ids(tmp_path) == ["analog-test", "liftoff-test-micro"]


def test_list_ids_of_missing_root_is_empty(tmp_path):
    assert drones.list_ids(tmp_path / "absent") == []


def test_root_resolution_arg_then_env_then_repo(tmp_path, monkeypatch):
    env_root = tmp_path / "env"
    write(env_root, analog_profile("env-only"))
    monkeypatch.setenv(ENV_ROOT, str(env_root))
    assert drones_root() == env_root
    assert drones.load("env-only").id == "env-only"
    arg_root = tmp_path / "arg"
    assert drones_root(arg_root) == arg_root
    with pytest.raises(DroneProfileError):
        drones.load("env-only", arg_root)
    monkeypatch.delenv(ENV_ROOT)
    root = drones_root()
    assert root.name == "drones" and (root / "README.md").is_file()
    assert "meteor75" in drones.list_ids()


def test_dump_round_trips_and_validates(tmp_path):
    data = sysid_profile()
    path = drones.dump(
        data, tmp_path / "liftoff-test-micro" / "drone.yaml", header="line 1\nline 2"
    )
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# line 1\n# line 2\n")
    # stable key order: schema_version first, randomization last
    keys = [ln.split(":")[0] for ln in text.splitlines() if ln and ln[0].isalpha()]
    assert keys[0] == "schema_version" and keys[-1] == "randomization"
    assert drones.load("liftoff-test-micro", tmp_path) == parse_profile(
        data, file=str(path), expected_id="liftoff-test-micro"
    )
    bad = sysid_profile()
    bad["nominal"]["motor_lag_s"] = _m(-1.0)
    with pytest.raises(DroneProfileError):
        drones.dump(bad, tmp_path / "liftoff-test-micro" / "drone.yaml")
    assert drones.load("liftoff-test-micro", tmp_path).motor_lag_s == pytest.approx(0.03)


# ----------------------------------------------------------------------------- helpers


def test_thrust_curve_and_its_inverse():
    prof = parse_profile(sysid_profile())
    assert prof.thrust_g(0.47) == pytest.approx(1.0)
    assert prof.thrust_g(1.0) == pytest.approx(2.4) == prof.max_g
    assert prof.thrust_g(0.235) == pytest.approx(0.5)
    for g in (0.0, 0.3, 1.0, 1.7, 2.4):
        assert prof.thrust_g(prof.throttle_for_g(g)) == pytest.approx(g)
    assert prof.throttle_for_g(1.0) == pytest.approx(0.47)
    # the inverse clamps to [0, 1]
    assert prof.throttle_for_g(5.0) == 1.0 and prof.throttle_for_g(-1.0) == 0.0


def test_rate_profile_is_the_gamepad_rate_curve():
    prof = parse_profile(sysid_profile())
    rp = prof.rate_profile("roll")
    assert (rp.rc_rate, rp.super_rate, rp.expo) == (1.0, 0.7, 0.0)
    # Betaflight: rc 1.0, super 0.7 at full stick = 200 / (1 - 0.7) deg/s
    assert rp.rate_deg_s(1.0) == pytest.approx(200.0 / 0.3)


def test_package_is_a_leaf():
    """``drone_fly.drones`` imports only YAML / NumPy at module level (no harness, no torch)."""
    import ast

    pkg = Path(drones.__file__).parent
    for f in pkg.glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] in {
                    "__future__",
                    "drone_fly",
                    "dataclasses",
                    "typing",
                    "re",
                    "os",
                    "pathlib",
                    "yaml",
                }
                if node.module.startswith("drone_fly"):
                    assert node.module.startswith("drone_fly.drones"), (f.name, node.module)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name.split(".")[0] in {"re", "os", "yaml", "numpy"}, (f.name, a.name)
