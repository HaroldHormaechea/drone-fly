"""UC-64 AC1: the packaged micro/Arena preset, the preset mechanism and the 5-inch fixes.

Preset chain: packaged default <- packaged preset <- user ``--config`` <- dotted overrides.
Every value in the preset must carry a provenance comment (inline, or a comment block heading
its group / parent key).
"""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path

import pytest

from drone_fly.harness.config import list_presets, load_config, load_preset
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.gate_types import GateTypeTable
from drone_fly.harness.liftoff.gate_learned import configured_weights
from drone_fly.harness.liftoff.telemetry import GyroMap

PRESET = "liftoff"
REPO = Path(__file__).resolve().parent.parent


#: One measured itemID per shipped gate class (UC71 P3, gate_types/liftoff.yaml comments).
SHIPPED_ITEMS = {
    "InflatableArchBrandless01": "inflatable_arch",
    "AirgateBigLiftoffWhite01": "airgate_big",
    "AirgateLiftoffDark01": "airgate_small",
    "GenericGate200x250cm01": "generic_gate",
    "GenericDoubleGate200x250cm01": "generic_double_gate",
    "MultiGPStandard7x6Frame01": "multigp_frame_7x6",
    "MultiGPChampTrack8x10Frame01": "multigp_frame_8x10",
    "LightGate300x220cmVarBlue01": "light_gate_300x220",
    "ASLLightHoopGate03": "light_hoop",
    "InflatableGateLiftoff01": "inflatable_gate",
    "TrussGateFinishGeneric01": "truss_gate_finish",
}


@pytest.fixture(scope="module")
def cfg():
    return load_config("liftoff", preset=PRESET)


def _preset_text() -> str:
    return (
        resources.files("drone_fly.harness")
        .joinpath("configs", f"{PRESET}.yaml")
        .read_text("utf-8")
    )


#: UC71: the preset names its gate table by file reference; the table's provenance lives there.
GATE_TYPES_FILE = "gate_types/liftoff.yaml"


def _gate_types_text() -> str:
    return (
        resources.files("drone_fly.harness")
        .joinpath("configs", *GATE_TYPES_FILE.split("/"))
        .read_text("utf-8")
    )


# ----------------------------------------------------------------------------- mechanism


def test_preset_is_listed_for_liftoff_only():
    assert list_presets("liftoff") == {PRESET: "liftoff"}
    assert PRESET in list_presets()
    assert list_presets("sim-loopback") == {}
    # harness defaults are not presets
    assert "liftoff_5inch" not in list_presets() and "loopback" not in list_presets()


def test_unknown_preset_lists_the_available_ones():
    with pytest.raises(HarnessConfigError, match=f"unknown preset 'nope'.*{PRESET}"):
        load_config("liftoff", preset="nope")


def test_preset_for_another_harness_is_refused():
    with pytest.raises(HarnessConfigError, match="is for harness 'liftoff'.*'sim-loopback'"):
        load_config("sim-loopback", preset=PRESET)


def test_preset_name_may_carry_the_yaml_suffix():
    assert load_preset("liftoff", f"{PRESET}.yaml") == load_preset("liftoff", PRESET)


def test_chain_order_user_config_then_overrides_win(tmp_path, cfg):
    user = tmp_path / "user.yaml"
    user.write_text("scaling:\n  world_scale: 3.0\ncontrols:\n  rate_scale: 1.1\n")
    merged = load_config("liftoff", user, overrides={"controls.rate_scale": 1.2}, preset=PRESET)
    assert merged.world_scale == 3.0  # user beats preset (4.0)
    assert merged.section("controls")["rate_scale"] == 1.2  # override beats user
    # untouched preset values survive the user overlay
    assert merged.section("camera")["hfov_deg"] == 120.0
    assert merged.section("telemetry")["rpm_count_prefix"] is True


@pytest.mark.parametrize(
    "overlay",
    ["controls:\n  hover_stick: -0.1\n", "controls:\n  rates:\n    roll: {rc_rate: 1.0}\n"],
)
def test_drone_and_hand_set_controls_conflict(tmp_path, overlay):
    """UC67 AC6: the preset names its drone; duplicating its rates / hover stick is an error."""
    user = tmp_path / "user.yaml"
    user.write_text(overlay)
    with pytest.raises(HarnessConfigError, match="conflict"):
        load_config("liftoff", user, preset=PRESET)


def test_drone_null_restores_the_hand_set_controls(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text("drone: null\ncontrols:\n  hover_stick: -0.1\n")
    merged = load_config("liftoff", user, preset=PRESET)
    assert merged.data["drone"] is None
    assert merged.section("controls")["hover_stick"] == -0.1


def test_user_config_still_rejects_unknown_keys_on_top_of_a_preset(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text("camera:\n  hfvo_deg: 90\n")
    with pytest.raises(HarnessConfigError, match="camera.hfvo_deg"):
        load_config("liftoff", user, preset=PRESET)


def test_preset_only_uses_keys_of_the_default():
    # load_config overlays the preset with the strict overlay: success means no unknown keys.
    raw = load_preset("liftoff", PRESET)
    default = load_config("liftoff").data
    assert set(raw) <= set(default)


# ----------------------------------------------------------------------------- AC1 values


def test_preset_values(cfg):
    assert cfg.harness == "liftoff"
    t = cfg.section("telemetry")
    assert t["rpm_count_prefix"] is True and t["gyro_units"] == "deg_s"
    assert t["gyro_map"] == {
        "wx": {"source": "roll", "sign": -1},
        "wy": {"source": "pitch", "sign": 1},
        "wz": {"source": "yaw", "sign": -1},
    }
    cam = cfg.section("camera")
    assert cam["resolution"] == [2560, 1080]
    assert cam["hfov_deg"] == 120.0 and cam["uptilt_deg"] == 20.0
    # UC67: the airframe comes from its drone profile, not from hand-set controls
    assert cfg.data["drone"] == "liftoff-gremlin-micro"
    c = cfg.section("controls")
    assert c.get("hover_stick") is None and c.get("rates") is None
    assert cfg.world_scale == 4.0
    assert cfg.parts.gate_estimator == "learned"
    learned = cfg.section("gates")["learned"]
    assert learned.get("weights") is None and learned["release"] == "v1"  # UC71: by release
    from drone_fly import paths

    assert configured_weights(learned) == str(paths.models_root("detector-v1") / "liftoff-gates.pt")
    d = cfg.section("depth")
    assert d["inverse"] is True
    assert d["calibration"]["scale"] == pytest.approx(-0.00535155)
    assert d["calibration"]["shift"] == pytest.approx(0.247091)
    assert cfg.section("offline")["laps"] is None


def test_preset_masks_cover_the_arena_hud(cfg):
    masks = cfg.section("camera")["masks"]
    names = {m["name"] for m in masks}
    assert names == {"mode_battery", "timer_lap", "alt_speed", "ghost_list", "stick_widgets", "rec"}
    for m in masks:
        x, y, w, h = m["rect"]
        assert w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= 2560 and y + h <= 1080
        # the centre crosshair stays unmasked (it would drop every gate dead ahead)
        assert not (x <= 1280 < x + w and y <= 540 < y + h)


def test_preset_gate_type_table(cfg):
    table = GateTypeTable.from_config(cfg.data["gate_types"])
    assert table.classes[0] == "inflatable_arch" and len(table) == len(SHIPPED_ITEMS)
    for item_id, cls in SHIPPED_ITEMS.items():  # UC71 P3: the captured tracks' gate types
        assert table.classify(item_id).cls == cls, item_id
    arch = table.classify("InflatableArchBrandless01")
    assert arch is not None and arch.cls == "inflatable_arch"
    assert arch.aperture == (6.18, 3.69) and arch.outer == (8.0, 4.46)
    assert arch.center_offset == (0.0, 1.74, 0.0)  # open at the bottom: centre above the ground
    assert arch.n_keypoints == 10 and len(arch.flip_pairs) == 4
    assert table.classify("ConradCrowdControlBarrier01") is None


def test_preset_item_table_covers_infinity_loop_obstacles(cfg):
    kinds = {e["match"]: e["kind"] for e in cfg.section("offline")["item_table"]}
    assert kinds == {
        "^AirflagLiftoffSlalom": "cylinder",
        "^ConradCrowdControlBarrier": "box",
        "^ConcreteBarrier": "box",
    }


def test_five_inch_default_has_the_liftoff_fixes_but_keeps_its_airframe():
    five = load_config("liftoff")
    t = five.section("telemetry")
    assert t["rpm_count_prefix"] is True and t["gyro_units"] == "deg_s"
    assert GyroMap.from_config(t) == GyroMap(("roll", "pitch", "yaw"), (-1.0, 1.0, -1.0))
    # UC67: the airframe is the identified 5-inch profile; the CALIBRATE placeholders are gone
    assert five.data["drone"] == "liftoff-5inch-vortex-pro-250"
    assert five.section("controls")["hover_stick"] is None
    assert five.section("controls")["rates"] is None
    assert five.section("camera")["uptilt_deg"] == 25.0
    assert five.world_scale == 2.0
    assert five.parts.gate_estimator == "opencv"
    assert five.parts.focus_probe == "win32-foreground"
    assert five.section("camera")["masks"] == []
    # the generic gate type reproduces today's 1.5 x 1.5 m gate
    table = GateTypeTable.from_config(five.data["gate_types"])
    sq = table["square_gate"]
    assert sq.aperture == (1.5, 1.5) and sq.n_keypoints == 4


# ----------------------------------------------------------------------------- provenance


_LEAF = re.compile(r"^(\s*)(- |[A-Za-z_][\w]*:\s*\S)")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parent(lines: list[str], i: int) -> int | None:
    d = _indent(lines[i])
    for j in range(i - 1, -1, -1):
        s = lines[j]
        if s.strip() and not s.lstrip().startswith("#") and _indent(s) < d:
            return j
    return None


def _covered(lines: list[str], i: int) -> bool:
    """Inline comment, a comment heading its run of sibling lines, or a covered parent.

    Scanning upward, contiguous same-indent siblings share the comment above them; a deeper
    line (the end of an earlier sibling's block) or the parent ends the run.
    """
    if "#" in lines[i]:
        return True
    d = _indent(lines[i])
    for j in range(i - 1, -1, -1):
        s = lines[j]
        if not s.strip():
            continue
        if s.lstrip().startswith("#"):
            if _indent(s) <= d + 2:
                return True
            continue
        if _indent(s) != d:
            break
    p = _parent(lines, i)
    return p is not None and _covered(lines, p)


@pytest.mark.parametrize("text", [_preset_text, _gate_types_text], ids=["preset", "gate-types"])
def test_every_preset_value_names_its_source(text):
    lines = text().splitlines()
    uncovered = [
        ln
        for i, ln in enumerate(lines)
        if ln.strip()
        and not ln.lstrip().startswith("#")
        and _LEAF.match(ln)
        and not _covered(lines, i)
    ]
    assert uncovered == []


@pytest.mark.parametrize(
    "pattern",
    [
        r"rpm_count_prefix",
        r"gyro_units",
        r"hfov_deg",
        r"uptilt_deg",
        r"resolution",
        r"hover_stick",
        r"world_scale",
        r"scale: -0\.00535155",
    ],
)
def test_key_values_have_a_measurement_source_nearby(pattern):
    _assert_source_near(_preset_text(), pattern)


def test_preset_references_the_gate_types_file(cfg):
    assert re.search(rf"^gate_types: {re.escape(GATE_TYPES_FILE)}\s*(#.*)?$", _preset_text(), re.M)
    assert cfg.gate_types_file == GATE_TYPES_FILE
    assert [t["class"] for t in cfg.data["gate_types"]][0] == "inflatable_arch"


@pytest.mark.parametrize(
    "pattern", [r"aperture:", r"outer:", r"center_offset:", r"keypoints:", r"outline:"]
)
def test_gate_table_values_have_a_measurement_source_nearby(pattern):
    """UC71: the gate-table keys moved (with their comments) into the gate-types file."""
    assert not re.search(rf"^\s*-?\s*{pattern}", _preset_text(), re.M)  # not back in the preset
    _assert_source_near(_gate_types_text(), pattern)


def _assert_source_near(text: str, pattern: str) -> None:
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if re.search(pattern, ln))
    context = "\n".join(lines[max(0, i - 12) : i + 1])
    assert re.search(
        r"MEASURED|VERIFIED|FITTED|PROVISIONAL|props\.json|drone file|Drone file|PlayerPrefs|"
        r"trigger box|capture",
        context,
    ), context


# ----------------------------------------------------------------------------- real outputs


def test_preset_depth_calibration_equals_the_depth_fit_report(cfg):
    report = REPO / "artifacts" / "harness" / "liftoff" / "reports" / "depth-fit"
    path = report / "depth_fit_report.json"
    if not path.is_file():
        pytest.skip("depth-fit report absent (local output of `drone-fly harness depth-fit`)")
    fit = json.loads(path.read_text(encoding="utf-8"))["fit"]
    d = cfg.section("depth")
    assert d["inverse"] is bool(fit["inverse"])
    assert d["calibration"]["scale"] == pytest.approx(fit["scale"], rel=1e-5)
    assert d["calibration"]["shift"] == pytest.approx(fit["shift"], rel=1e-5)
