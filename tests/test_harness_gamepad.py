"""UC-62 AC9: ViGEm action sink — inverse Betaflight rate curve, throttle map, stick signs.

``vgamepad`` is never imported: every sink gets a mock pad through ``pad_factory`` (installing
the real package triggers the ViGEmBus driver installer on Windows).
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.harness.errors import HarnessConfigError, HarnessDependencyError
from drone_fly.harness.liftoff.gamepad import (
    MODE2,
    RateProfile,
    ThrottleMap,
    VgamepadActionSink,
    default_pad_factory,
)
from drone_fly.harness.types import ContractCommand
from tests.harness_support import MockPad

HOVER = 0.4


def _cmd(collective=HOVER, rates=(0.0, 0.0, 0.0)) -> ContractCommand:
    return ContractCommand(
        raw=np.zeros(4, np.float32),
        collective=collective,
        thrust=0.0,
        rates=np.array(rates, np.float32),
    )


def _sink(**kw) -> tuple[VgamepadActionSink, MockPad]:
    pad = MockPad()
    profiles = kw.pop("profiles", {ax: RateProfile() for ax in ("roll", "pitch", "yaw")})
    sink = VgamepadActionSink(profiles, ThrottleMap(HOVER, -0.4), pad_factory=lambda: pad, **kw)
    return sink, pad


# ----------------------------------------------------------------------------- rate curve


def test_betaflight_rate_curve_reference_values():
    p = RateProfile(rc_rate=1.0, super_rate=0.7, expo=0.0)
    assert p.rate_deg_s(0.0) == 0.0
    assert p.max_rate_deg_s == pytest.approx(200.0 / 0.3)
    assert p.rate_deg_s(0.5) == pytest.approx(100.0 / (1 - 0.35))
    assert p.rate_deg_s(-0.5) == pytest.approx(-p.rate_deg_s(0.5))
    assert RateProfile(1.0, 0.0, 0.0).rate_deg_s(1.0) == pytest.approx(200.0)
    # rc_rate above 2 uses the Betaflight incremental slope (14.54 per unit).
    hi = RateProfile(rc_rate=2.5, super_rate=0.0, expo=0.0)
    assert hi.rate_deg_s(0.5) == pytest.approx(200.0 * (2.5 + 14.54 * 0.5) * 0.5)
    # Expo softens the centre.
    assert RateProfile(1.0, 0.0, 0.5).rate_deg_s(0.2) < RateProfile(1.0, 0.0, 0.0).rate_deg_s(0.2)


@pytest.mark.parametrize(
    "profile",
    [
        RateProfile(),
        RateProfile(1.2, 0.6, 0.3),
        RateProfile(0.8, 0.0, 0.0),
        RateProfile(2.4, 0.5, 0.9),
    ],
)
def test_inverse_rate_curve_round_trips(profile):
    for stick in np.linspace(-0.99, 0.99, 41):
        rate = profile.rate_deg_s(stick)
        assert profile.stick_for_rate(rate) == pytest.approx(stick, abs=1e-6)
    assert profile.stick_for_rate(0.0) == 0.0


def test_inverse_saturates_at_full_stick():
    p = RateProfile()
    assert p.stick_for_rate(10 * p.max_rate_deg_s) == 1.0
    assert p.stick_for_rate(-10 * p.max_rate_deg_s) == -1.0
    assert p.rate_deg_s(5.0) == p.max_rate_deg_s  # stick clamped to [-1, 1]


@pytest.mark.parametrize(
    "bad", [(0.0, 0.7, 0.0), (1.0, 1.0, 0.0), (1.0, -0.1, 0.0), (1.0, 0.5, 1.5)]
)
def test_invalid_rate_profiles_rejected(bad):
    with pytest.raises(HarnessConfigError):
        RateProfile(*bad)


# ----------------------------------------------------------------------------- throttle


def test_throttle_map_anchors():
    t = ThrottleMap(HOVER, -0.4)
    assert t.stick(0.0) == -1.0
    assert t.stick(HOVER) == pytest.approx(-0.4)
    assert t.stick(1.0) == 1.0
    assert t.stick(-0.5) == -1.0 and t.stick(2.0) == 1.0
    assert t.stick(HOVER / 2) == pytest.approx(-0.7)
    assert t.stick(0.7) == pytest.approx(0.3)
    sticks = [t.stick(c) for c in np.linspace(0, 1, 21)]
    assert all(b > a for a, b in zip(sticks, sticks[1:], strict=False))  # monotonic
    for bad in ((0.0, -0.4), (1.0, -0.4), (0.4, -1.0), (0.4, 1.0)):
        with pytest.raises(HarnessConfigError):
            ThrottleMap(*bad)


# ----------------------------------------------------------------------------- stick mapping


def test_hover_command_is_hover_stick_with_centred_rates():
    sink, _ = _sink()
    assert sink.sticks_for(_cmd()) == {
        "left_y": pytest.approx(-0.4),
        "left_x": 0.0,
        "right_x": 0.0,
        "right_y": 0.0,
    }


def test_flu_sign_conventions_pinned():
    sink, _ = _sink()
    roll = sink.sticks_for(_cmd(rates=(1.0, 0, 0)))  # +wx: right wing down
    assert roll["right_x"] > 0 and roll["right_y"] == 0 and roll["left_x"] == 0
    pitch = sink.sticks_for(_cmd(rates=(0, 1.0, 0)))  # +wy: nose down
    assert pitch["right_y"] > 0 and pitch["right_x"] == 0
    yaw = sink.sticks_for(_cmd(rates=(0, 0, 1.0)))  # +wz: counter-clockwise from above
    assert yaw["left_x"] < 0 and yaw["right_x"] == 0
    up = sink.sticks_for(_cmd(collective=0.9))
    down = sink.sticks_for(_cmd(collective=0.1))
    assert up["left_y"] > sink.sticks_for(_cmd())["left_y"] > down["left_y"]


def test_commanded_body_rate_yields_intended_deg_s():
    p = RateProfile(1.2, 0.6, 0.2)
    sink, _ = _sink(profiles={"roll": p, "pitch": p, "yaw": p})
    for w in (0.3, -1.5, 4.0):
        s = sink.sticks_for(_cmd(rates=(w, 0, 0)))["right_x"]
        assert p.rate_deg_s(s) == pytest.approx(math.degrees(w), rel=1e-6)
    sink2, _ = _sink(profiles={"roll": p, "pitch": p, "yaw": p}, rate_scale=2.0)
    s2 = sink2.sticks_for(_cmd(rates=(0, 0.5, 0)))["right_y"]
    assert p.rate_deg_s(s2) == pytest.approx(2.0 * math.degrees(0.5), rel=1e-6)


def test_saturating_rate_clamps_stick():
    sink, _ = _sink()
    s = sink.sticks_for(_cmd(rates=(100.0, -100.0, 100.0)))
    assert s["right_x"] == 1.0 and s["right_y"] == -1.0 and s["left_x"] == -1.0


def test_axis_map_and_stick_signs_are_configurable():
    mode1 = {"throttle": "right_y", "yaw": "left_x", "pitch": "left_y", "roll": "right_x"}
    sink, _ = _sink(axis_map=mode1, signs={"yaw": 1.0})
    s = sink.sticks_for(_cmd(collective=1.0, rates=(0, 1.0, 1.0)))
    assert s["right_y"] == 1.0  # throttle moved to the right stick
    assert s["left_y"] > 0  # pitch on the left stick
    assert s["left_x"] > 0  # yaw sign flipped by config
    assert sink.neutral_sticks() == {"left_x": 0.0, "left_y": 0.0, "right_x": 0.0, "right_y": -1.0}
    with pytest.raises(HarnessConfigError, match="axis_map"):
        _sink(axis_map={**MODE2, "roll": "left_x"})


# ----------------------------------------------------------------------------- pad lifecycle


def test_arm_send_failsafe_close_with_mock_pad():
    sink, pad = _sink()
    with pytest.raises(RuntimeError, match="not armed"):
        sink.send(_cmd())
    sink.failsafe()  # no pad yet: harmless no-op
    assert pad.updates == 0
    sink.arm()
    neutral = {"left_x": 0.0, "left_y": -1.0, "right_x": 0.0, "right_y": 0.0}
    assert pad.sticks == neutral and pad.updates == 1  # armed in neutral (throttle cut)
    sink.arm()  # idempotent: the factory is not called twice
    out = sink.send(_cmd(collective=0.8, rates=(0.5, -0.5, 0.2)))
    assert out["sticks"] == pad.sticks and pad.sticks["left_y"] > -0.4
    sink.failsafe()
    assert pad.sticks == neutral
    sink.close()
    assert pad.sticks == neutral and sink.pad is None
    sink.close()  # second close is a no-op


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def _arm_sink(disarm="B"):
    clock = _Clock()
    sink, pad = _sink(
        arm_button="A",
        disarm_button=disarm,
        arm_press_s=0.15,
        button_resolver=lambda name: f"btn-{name}",
        clock=clock,
        sleep=clock.sleep,
    )
    return sink, pad, clock


def _log(pad):
    return [(e[0], e[1]) for e in pad.button_log]


def test_no_arm_button_never_touches_buttons():
    sink, pad = _sink()
    sink.arm()
    sink.send(_cmd())
    sink.failsafe()
    sink.close()
    assert pad.button_log == [] and not sink.armed


def test_arm_pulses_on_first_command_and_disarm_pulses_on_failsafe():
    sink, pad, clock = _arm_sink()
    sink.arm()
    assert pad.buttons == set() and not sink.armed  # nothing before the game has focus
    assert pad.sticks["left_y"] == -1.0
    sink.send(_cmd())  # first command: press Arm
    assert pad.buttons == {"btn-A"} and sink.armed
    clock.t += 0.1
    sink.send_sticks({"throttle": 0.0})
    assert pad.buttons == {"btn-A"}  # still inside the pulse
    clock.t += 0.1
    sink.send(_cmd())
    assert pad.buttons == set()  # released after arm_press_s
    clock.t += 1.0
    sink.send(_cmd())
    assert _log(pad) == [("press", "btn-A"), ("release", "btn-A")]  # one arm pulse only
    sink.failsafe()
    assert _log(pad)[2:] == [("press", "btn-B"), ("release", "btn-B")]
    assert clock.sleeps == [0.3] and pad.buttons == set() and not sink.armed  # disarm_press_s
    assert pad.sticks["left_y"] == -1.0
    sink.close()  # disarm happens once per run
    assert len(pad.button_log) == 4


def test_failsafe_inside_the_arm_pulse_releases_arm_then_disarms():
    sink, pad, _ = _arm_sink()
    sink.arm()
    sink.send(_cmd())  # Arm pressed, not yet released
    sink.failsafe()
    assert _log(pad) == [
        ("press", "btn-A"),
        ("release", "btn-A"),
        ("press", "btn-B"),
        ("release", "btn-B"),
    ]


def test_failsafe_before_any_command_still_pulses_disarm():
    sink, pad, _ = _arm_sink()
    sink.arm()
    sink.failsafe()
    assert _log(pad) == [("press", "btn-B"), ("release", "btn-B")]


def test_close_without_failsafe_disarms():
    sink, pad, _ = _arm_sink()
    sink.arm()
    sink.send(_cmd())
    sink.close()
    assert ("press", "btn-B") in _log(pad) and sink.pad is None


def test_without_disarm_button_arm_toggles_only_if_this_run_armed():
    sink, pad, _ = _arm_sink(disarm=None)
    sink.arm()
    sink.failsafe()
    assert pad.button_log == []  # never armed: no toggle pulse
    sink2, pad2, clock2 = _arm_sink(disarm=None)
    sink2.arm()
    sink2.send(_cmd())
    clock2.t += 0.2
    sink2.send(_cmd())
    sink2.failsafe()
    assert [e[0] for e in pad2.button_log] == ["press", "release", "press", "release"]
    assert {e[1] for e in pad2.button_log} == {"btn-A"}


def test_arm_config_is_validated():
    with pytest.raises(HarnessConfigError, match="arm_button"):
        _sink(arm_button="Z")
    with pytest.raises(HarnessConfigError, match="disarm_button"):
        _sink(arm_button="A", disarm_button="Q")
    with pytest.raises(HarnessConfigError, match="needs controls.arm_button"):
        _sink(disarm_button="B")
    from drone_fly.harness.config import load_config

    ctx = SimpleNamespace(contract=SimpleNamespace(HOVER_THR=0.4))
    cfg = load_config("liftoff").section("controls")
    assert cfg["arm_button"] is None and cfg["disarm_button"] is None
    preset = load_config("liftoff", preset="liftoff").section("controls")
    assert (preset["arm_button"], preset["disarm_button"]) == ("A", "B")
    sink = VgamepadActionSink.from_config(preset, ctx)
    assert (sink.arm_button, sink.disarm_button) == ("A", "B")
    with pytest.raises(HarnessConfigError, match="arm_press_s"):
        VgamepadActionSink.from_config({**cfg, "arm_press_s": 0}, ctx)


def test_from_config_uses_contract_hover_and_validates():
    from drone_fly.harness.config import load_config

    cfg = load_config("liftoff").section("controls")
    ctx = SimpleNamespace(contract=SimpleNamespace(HOVER_THR=0.4))
    sink = VgamepadActionSink.from_config(cfg, ctx)
    assert sink.throttle == ThrottleMap(0.4, -0.4)
    assert sink.signs == {"roll": 1.0, "pitch": 1.0, "yaw": -1.0}
    assert sink.axis_map == MODE2 and sink.pad is None  # constructing never opens the device
    assert sink.profiles["yaw"] == RateProfile(1.0, 0.7, 0.0)
    with pytest.raises(HarnessConfigError, match="stick_signs"):
        VgamepadActionSink.from_config({**cfg, "stick_signs": {"roll": 2}}, ctx)
    with pytest.raises(HarnessConfigError):
        VgamepadActionSink.from_config({**cfg, "stick_signs": {"throttle": 1}}, ctx)
    with pytest.raises(HarnessConfigError, match="rate_scale"):
        VgamepadActionSink.from_config({**cfg, "rate_scale": 0}, ctx)


def test_missing_vgamepad_raises_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "vgamepad", None)  # behave as "not installed"
    with pytest.raises(HarnessDependencyError, match="uv sync --extra harness") as ei:
        default_pad_factory()
    assert ei.value.package == "vgamepad"
    sink = VgamepadActionSink(
        {ax: RateProfile() for ax in ("roll", "pitch", "yaw")}, ThrottleMap(0.4)
    )
    with pytest.raises(HarnessDependencyError):
        sink.arm()


def test_throttle_is_held_at_minimum_while_arming():
    sink, pad, clock = _arm_sink()
    sink.arm()
    full = _cmd(collective=1.0, rates=(1.0, -1.0, 0.5))
    out = sink.send(full)  # Arm pressed now: settle window starts
    assert out["sticks"] == sink.neutral_sticks() and pad.sticks["left_y"] == -1.0
    clock.t += 0.3
    assert sink.send(full)["sticks"]["left_y"] == -1.0  # still settling (arm_settle_s 0.5)
    assert sink.send_sticks({"throttle": 0.5})["left_y"] == -1.0
    clock.t += 0.3
    assert sink.send(full)["sticks"]["left_y"] > -1.0  # settled: the policy flies


def test_arm_settle_is_validated_and_defaults():
    from drone_fly.harness.config import load_config

    ctx = SimpleNamespace(contract=SimpleNamespace(HOVER_THR=0.4))
    cfg = load_config("liftoff").section("controls")
    assert cfg["arm_settle_s"] == 0.5
    with pytest.raises(HarnessConfigError, match="arm_settle_s"):
        VgamepadActionSink.from_config({**cfg, "arm_settle_s": -1}, ctx)


def test_disarm_re_presses_disarm_with_throttle_cut():
    sink, pad, clock = _arm_sink()
    assert sink.disarm() is False  # no pad yet
    sink.arm()
    assert sink.disarm() is True
    assert _log(pad) == [("press", "btn-B"), ("release", "btn-B")]
    assert clock.sleeps == [0.3] and pad.sticks["left_y"] == -1.0
    toggle, _, _ = _arm_sink(disarm=None)
    toggle.arm()
    assert toggle.disarm() is False  # never pulse a toggle arm button blindly


def test_keep_pad_connected_holds_the_pad_neutral_and_disarmed_after_close():
    from drone_fly.harness.liftoff import gamepad

    gamepad.release_kept_pads()
    sink, pad, _ = _arm_sink()
    sink.keep_pad_connected = True
    sink.arm()
    sink.send(_cmd(collective=1.0))
    sink.failsafe()
    sink.close()
    assert sink.pad is None and gamepad.kept_pads() == [pad]
    assert pad.sticks["left_y"] == -1.0 and pad.buttons == set()
    gamepad.release_kept_pads()
    assert gamepad.kept_pads() == []
    plain, plain_pad = _sink()
    plain.arm()
    plain.close()
    assert gamepad.kept_pads() == []  # default: the pad disconnects on close


def test_keep_pad_connected_applies_to_local_pads_only():
    from drone_fly.harness.config import load_config

    ctx = SimpleNamespace(contract=SimpleNamespace(HOVER_THR=0.4))
    cfg = load_config("liftoff").section("controls")
    assert VgamepadActionSink.from_config(cfg, ctx).keep_pad_connected is False
    local = {**cfg, "keep_pad_connected": True}
    assert VgamepadActionSink.from_config(local, ctx).keep_pad_connected is True
    server = {**local, "pad": "server"}  # the long-lived pad server stays connected anyway
    assert VgamepadActionSink.from_config(server, ctx).keep_pad_connected is False


# ----------------------------------------------------------------------------- UC67 drone profile


def _micro():
    from drone_fly import drones

    return drones.load("liftoff-gremlin-micro")


def _v2(collective_g, rates=(0.0, 0.0, 0.0)) -> ContractCommand:
    return ContractCommand(
        raw=np.zeros(4, np.float32),
        collective=0.0,  # deliberately wrong: a drone sink must read collective_g
        thrust=0.0,
        rates=np.array(rates, np.float32),
        collective_g=collective_g,
        contract_version=2,
    )


def test_drone_throttle_map_inverts_the_thrust_curve():
    from drone_fly.harness.liftoff.gamepad import DroneThrottleMap

    prof = _micro()
    t = DroneThrottleMap.from_profile(prof)
    assert t.drone_id == "liftoff-gremlin-micro"
    # 1.0 g is the measured hover: throttle 0.4708 -> stick -0.058 (UC64 hand-flown: -0.06)
    assert t.stick(1.0) == pytest.approx(-0.058, abs=0.001)
    assert t.hover_stick == t.stick(1.0)
    assert t.stick(1.0) == pytest.approx(2 * prof.hover_throttle - 1, abs=0.01)
    gs = np.linspace(0.0, prof.max_g, 41)
    sticks = [t.stick(g) for g in gs]
    assert all(b >= a for a, b in zip(sticks, sticks[1:], strict=False))  # monotone
    for g in gs[gs > 0.01]:  # the inverse round-trips through the forward curve
        frac = (t.stick(g) + 1.0) / 2.0
        assert np.interp(frac, prof.thrust_curve.throttle, prof.thrust_curve.accel_g) == (
            pytest.approx(g, abs=1e-6)
        )
    assert t.stick(0.0) == -1.0 and t.stick(-1.0) == -1.0  # clamps
    assert t.stick(prof.max_g) == 1.0 and t.stick(10.0) == 1.0


def test_drone_sink_maps_collective_g_and_profile_rates():
    from drone_fly.harness.liftoff.gamepad import DroneThrottleMap

    prof = _micro()
    profiles = {a: prof.rate_profile(a) for a in ("roll", "pitch", "yaw")}
    sink = VgamepadActionSink(
        profiles, DroneThrottleMap.from_profile(prof), pad_factory=MockPad, rate_scale=1.0
    )
    hover = sink.sticks_for(_v2(1.0))
    assert hover["left_y"] == pytest.approx(-0.058, abs=0.001)
    assert sink.hover_stick == pytest.approx(hover["left_y"])
    assert sink.sticks_for(_v2(2.0))["left_y"] > hover["left_y"]
    for w in (0.3, -1.5, 4.0):
        s = sink.sticks_for(_v2(1.0, rates=(w, 0, 0)))["right_x"]
        assert profiles["roll"].rate_deg_s(s) == pytest.approx(math.degrees(w), rel=1e-6)
        # the measured micro curve, not the old preset's 1.0 / 0.7
        assert s != pytest.approx(RateProfile(1.0, 0.7, 0.0).stick_for_rate(math.degrees(w)))


def test_drone_sink_without_collective_g_uses_the_reference_fraction():
    from drone_fly.harness.liftoff.gamepad import DroneThrottleMap

    t = DroneThrottleMap.from_profile(_micro())
    legacy = _cmd(collective=0.4)  # a command built without collective_g
    assert legacy.collective_g is None
    assert t.command_stick(legacy) == pytest.approx(t.stick(1.0))  # 0.4 of meteor75 = 1 g


def test_from_config_with_the_liftoff_preset_uses_its_drone():
    from drone_fly.harness.config import load_config
    from drone_fly.harness.interfaces import BuildContext
    from drone_fly.harness.liftoff.gamepad import DroneThrottleMap

    cfg = load_config("liftoff", preset="liftoff")
    ctx = BuildContext(config=cfg, contract=SimpleNamespace(HOVER_THR=0.4))
    sink = VgamepadActionSink.from_config(cfg.section("controls"), ctx)
    prof = _micro()
    assert isinstance(sink.throttle, DroneThrottleMap) and sink.throttle.drone_id == prof.id
    for a in ("roll", "pitch", "yaw"):
        assert sink.profiles[a] == prof.rate_profile(a)
    assert sink.hover_stick == pytest.approx(-0.058, abs=0.001)
    assert sink.signs == {"roll": 1.0, "pitch": 1.0, "yaw": -1.0}  # still preset settings


def test_from_config_without_a_drone_is_the_pre_uc67_mapping():
    from drone_fly.harness.config import load_config
    from drone_fly.harness.interfaces import BuildContext

    cfg = load_config(
        "liftoff",
        overrides={
            "drone": None,
            "controls.hover_stick": -0.4,
            "controls.rates": {a: {"rc_rate": 1.0, "super_rate": 0.7, "expo": 0.0}
                               for a in ("roll", "pitch", "yaw")},
        },
    )  # fmt: skip
    ctx = BuildContext(config=cfg, contract=SimpleNamespace(HOVER_THR=0.4))
    sink = VgamepadActionSink.from_config(cfg.section("controls"), ctx)
    assert sink.throttle == ThrottleMap(0.4, -0.4)
    assert sink.profiles["roll"] == RateProfile(1.0, 0.7, 0.0)
    # a v2 hover command carries the reference fraction 0.4 -> the configured hover stick
    hover = ContractCommand(np.zeros(4, np.float32), 0.4, 0.0, np.zeros(3, np.float32), 1.0, 2)
    assert sink.sticks_for(hover)["left_y"] == pytest.approx(-0.4)


def test_drone_key_is_validated():
    from drone_fly.harness.config import check_drone, load_drone_profile

    assert check_drone({}) is None and check_drone({"drone": None}) is None
    assert check_drone({"drone": "meteor75", "controls": {"rates": None}}) == "meteor75"
    with pytest.raises(HarnessConfigError, match="conflict"):
        check_drone({"drone": "meteor75", "controls": {"hover_stick": -0.1}})
    with pytest.raises(HarnessConfigError, match="conflict"):
        check_drone({"drone": "meteor75", "controls": {"rates": {"roll": {}}}})
    with pytest.raises(HarnessConfigError, match="profile id or null"):
        check_drone({"drone": 7})
    with pytest.raises(HarnessConfigError, match="config 'drone'"):
        load_drone_profile({"drone": "no-such-drone"})
    assert load_drone_profile({"drone": "liftoff-gremlin-micro"}).id == "liftoff-gremlin-micro"
