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
