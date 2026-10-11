"""UC-65 AC2 / AC4 / AC15: the ``overlay`` config section and its validation.

Both packaged harness defaults carry the section (the config loader rejects keys absent from the
default, so a user ``overlay:`` block only works because of it); ``OverlayConfig.from_config``
validates every key, including the port collision with the telemetry listener, and the
``--overlay`` / ``--overlay-launch`` overrides map onto ``overlay.enabled``.
"""

from __future__ import annotations

import pytest

from drone_fly.harness import registry
from drone_fly.harness.config import (
    OVERLAY_BRAIN_SCALES,
    OVERLAY_LAYOUTS,
    OVERLAY_PANELS,
    OverlayConfig,
    load_config,
    load_packaged_default,
)
from drone_fly.harness.errors import HarnessConfigError

EXPECTED_KEYS = {
    "enabled",
    "port",
    "brain_every_n",
    "brain_scale",
    "layout",
    "fps",
    "stale_s",
    "gain_window_s",
    "opacity",
    "tap_outline",
    "recordable",
    "panels",
}


def _default_names() -> list[str]:
    return sorted(
        {
            registry.entry("harness", n).default_config
            for n in registry.names("harness")
            if registry.entry("harness", n).default_config
        }
    )


@pytest.mark.parametrize("name", _default_names())
def test_every_packaged_default_has_the_overlay_section(name):
    data = load_packaged_default(name)
    sec = data["overlay"]
    assert set(sec) == EXPECTED_KEYS
    assert sec["enabled"] is False  # opt-in
    assert set(sec["panels"]) == set(OVERLAY_PANELS)
    assert sec["port"] != (data.get("telemetry") or {}).get("port")
    assert sec["brain_scale"] == "recorder"  # AC2's recorder quantisation is the default


@pytest.mark.parametrize("harness", ["liftoff", "sim-loopback"])
def test_defaults_parse_to_the_documented_values(harness):
    cfg = OverlayConfig.from_config(load_config(harness))
    assert cfg == OverlayConfig()
    assert (cfg.enabled, cfg.port, cfg.brain_every_n, cfg.fps) == (False, 9100, 3, 20.0)
    assert (cfg.stale_s, cfg.gain_window_s, cfg.layout) == (1.0, 5.0, "default")
    assert all(cfg.panel_enabled(p) for p in OVERLAY_PANELS)
    assert not cfg.panel_enabled("no-such-panel")


def test_absent_section_gives_defaults():
    assert OverlayConfig.from_config({"harness": "x"}) == OverlayConfig()


def test_user_values_and_partial_panels():
    cfg = OverlayConfig.from_config(
        load_config(
            "liftoff",
            {
                "overlay": {
                    "enabled": True,
                    "port": 9200,
                    "brain_every_n": 5,
                    "brain_scale": "frame_peak",
                    "layout": "mask_confined",
                    "fps": 10,
                    "opacity": 0.5,
                    "tap_outline": True,
                    "panels": {"raycast": False, "brain": False},
                }
            },
        )
    )
    assert cfg.enabled and cfg.port == 9200 and cfg.brain_every_n == 5
    assert cfg.brain_scale == "frame_peak" and cfg.layout == "mask_confined"
    assert cfg.fps == 10.0 and cfg.opacity == 0.5 and cfg.tap_outline
    assert not cfg.panel_enabled("raycast") and not cfg.panel_enabled("brain")
    assert cfg.panel_enabled("health")


def test_overlay_port_equal_to_telemetry_port_is_rejected():
    with pytest.raises(HarnessConfigError, match="telemetry.port"):
        OverlayConfig.from_config(load_config("liftoff", {"overlay": {"port": 9001}}))
    with pytest.raises(HarnessConfigError, match="telemetry.port"):
        OverlayConfig.from_config(
            load_config("liftoff", {"telemetry": {"port": 9555}, "overlay": {"port": 9555}})
        )


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("port", 80, "overlay.port"),
        ("port", 70000, "65535"),
        ("port", "9100", "overlay.port"),
        ("port", True, "overlay.port"),
        ("brain_every_n", 0, "brain_every_n"),
        ("brain_every_n", 1.5, "brain_every_n"),
        ("brain_scale", "per_neuron", "brain_scale"),
        ("layout", "minimal", "overlay.layout"),
        ("fps", 21, "overlay.fps"),
        ("fps", 0.5, "overlay.fps"),
        ("stale_s", 0, "stale_s"),
        ("gain_window_s", -1.0, "gain_window_s"),
        ("opacity", 1.5, "opacity"),
        ("opacity", -0.1, "opacity"),
        ("enabled", "yes", "overlay.enabled"),
        ("tap_outline", 1, "tap_outline"),
        ("panels", ["brain"], "panels"),
    ],
)
def test_invalid_values_are_rejected(key, value, match):
    data = load_config("liftoff").data
    data["overlay"][key] = value
    with pytest.raises(HarnessConfigError, match=match):
        OverlayConfig.from_config(data)


def test_unknown_and_non_bool_panels_are_rejected():
    data = load_config("liftoff").data
    data["overlay"]["panels"] = {"minimap": True}
    with pytest.raises(HarnessConfigError, match="unknown overlay panel"):
        OverlayConfig.from_config(data)
    data["overlay"]["panels"] = {"brain": "on"}
    with pytest.raises(HarnessConfigError, match="overlay.panels.brain"):
        OverlayConfig.from_config(data)
    data["overlay"] = "on"
    with pytest.raises(HarnessConfigError, match="mapping"):
        OverlayConfig.from_config(data)


def test_unknown_overlay_key_is_rejected_by_the_loader():
    with pytest.raises(HarnessConfigError, match="overlay.launch"):
        load_config("liftoff", {"overlay": {"launch": True}})
    with pytest.raises(HarnessConfigError, match="overlay.no_such"):
        load_config("sim-loopback", None, {"overlay.no_such": 1})


def test_overlay_enabled_override():
    cfg = load_config("sim-loopback", None, {"overlay.enabled": True})
    assert OverlayConfig.from_config(cfg).enabled is True


def test_choice_tuples_match_the_publisher_and_layouts():
    from drone_fly.harness.overlay import BRAIN_SCALES
    from drone_fly.overlay.layout import CAPTURE_MASK_CONFINED

    assert OVERLAY_BRAIN_SCALES == BRAIN_SCALES
    assert OVERLAY_LAYOUTS == ("default", CAPTURE_MASK_CONFINED)


def test_overlay_launch_with_overlay_explicitly_disabled_is_a_config_error():
    """Raised before the contract is touched, so no model / GPU is needed."""
    from drone_fly.harness.session import run_harness

    with pytest.raises(HarnessConfigError, match="overlay-launch"):
        run_harness("sim-loopback", "x.pt", 70, overlay=False, overlay_launch=True)


def test_bad_overlay_section_refuses_the_run_before_the_model_loads():
    from drone_fly.harness.session import run_harness

    with pytest.raises(HarnessConfigError, match="overlay.fps"):
        run_harness("sim-loopback", "x.pt", 70, config={"overlay": {"fps": 60}})
