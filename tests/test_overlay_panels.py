"""UC-65 AC8 / AC10: the debug-panel data models (toolkit-free) and the default layout.

Every number the window draws is computed by ``drone_fly.overlay.panels`` from a state snapshot,
so the action / inputs / raycast / perception / health / telemetry panels are pinned here without
a display: grouping by contract block, out-of-FOV rays, frame-pixel -> screen mapping of the gate
keypoints, amber/red age colouring, and the default layout keeping the game centre clear.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from drone_fly.overlay.layout import DEFAULT_RECTS, default_layout
from drone_fly.overlay.panels import (
    AMBER_FRACTION,
    DEFAULT_HOVER_THR,
    DEFAULT_MAX_BODY_RATE,
    ScreenMap,
    age_level,
    build_action,
    build_health,
    build_inputs,
    build_panels,
    build_perception,
    build_raycast,
    build_telemetry,
    contract_cone_xy,
    frame_to_screen,
    quat_to_euler_deg,
)
from drone_fly.overlay.state import MODE_LIVE, MODE_STOPPED, MODE_WAITING, StateSnapshot

LAYOUT_70 = [
    ["obs", 0, 12],
    ["curr_gate", 12, 3],
    ["next_lookahead", 15, 7],
    ["obstacle_vision", 22, 8],
    ["raycast", 30, 40],
]
META = {
    "extra_layout": LAYOUT_70,
    "max_body_rate": 4.0,
    "hover_thr": 0.4,
    "world_scale": 2.0,
    "rate_hz": 50,
    "age_limits": {"telemetry": 0.25, "free_space": 0.3, "gate": 1.0},
}


def _tick(**over) -> dict:
    rays = [1.0 - 0.02 * i for i in range(40)]
    t = {
        "tick": 7,
        "state": {
            "pos": [0.0, 0.0, 1.5],
            "vel": [3.0, 4.0, 0.0],
            "quat": [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)],  # yaw +90 deg
            "omega": [0.0, 0.0, 0.0],
        },
        "obs": [1.0, -2.0, 0.5, 0.0, 0.0, -1.0, 6.0, 0.0, 0.0, 0.0, 0.0, -8.0],
        "extra": list(range(70)),
        "command": {"raw": [-0.2, 0.5, -3.0, 0.0], "collective": 0.4, "rates": [2.0, -8.0, 0.0]},
        "sink_output": {"sticks": {"throttle": 0.3, "yaw": -0.1, "flag": True}},
        "gate": {
            "status": "tracking",
            "gate_index": 2,
            "lap": 1,
            "target": [3.0, 4.0, 1.5],
            "info": {
                "gate_class": "square",
                "detections": 1,
                "matched": True,
                "detections_px": [
                    {
                        "cls": "square",
                        "points": [[100, 50], [300, 50], [300, 250], [100, 250]],
                        "center": [200, 150],
                        "matched": True,
                    }
                ],
                "frame_size": [1280, 540],
            },
        },
        "free_space": {"raycast": rays, "in_fov_mask": [1.0] * 30 + [0.0] * 10, "coverage": 0.75},
        "latency_s": {
            "state": 0.001,
            "inference": 0.004,
            "actuation": 0.0005,
            "overlay": 0.0007,
            "total": 0.01,
        },
        "perception_s": {"gate": 0.003, "free_space": 0.002},
        "ages_s": {"telemetry": 0.1, "frame": 0.04, "gate": 0.85, "free_space": 0.31},
        "missed_deadlines": 2,
        "drops": 5,
        "focus": True,
        "focus_window": "liftoff.exe",
        "run_info": {
            "harness": "liftoff",
            "checkpoint": "m.pt",
            "sha8": "deadbeef",
            "extra_dim": 70,
        },
        "sample": {"battery": [0.8, 16.2], "inputs": [0.5, 0.0, -0.1, 0.2]},
    }
    t.update(over)
    return t


def _snap(tick=None, meta=None, *, mode=MODE_LIVE, stop=None, status=None) -> StateSnapshot:
    return StateSnapshot(
        mode=mode,
        message="live",
        run_id="a" * 16,
        tick=tick,
        brain=None,
        meta=meta,
        status=status,
        last_stop=stop,
        tick_rate_hz=49.8,
        since_last_s=0.01,
        brain_seq=0,
        packets=10,
        garbled=0,
    )


# ----------------------------------------------------------------------------- action


def test_action_panel():
    a = build_action(_tick(), META)
    assert a.available
    assert [b.label for b in a.raw] == ["a0 col", "a1 wx", "a2 wy", "a3 wz"]
    assert a.raw[1].fraction == pytest.approx(0.5) and a.raw[2].fraction == -1.0  # clipped
    assert a.collective.fraction == pytest.approx(0.4) and a.collective.marker == 0.4
    assert not a.collective.signed
    assert [b.fraction for b in a.rates] == [pytest.approx(0.5), -1.0, 0.0]  # vs MAX_BODY_RATE
    assert a.rates[0].text == "+2.00 rad/s"
    assert [b.label for b in a.sticks] == ["throttle", "yaw"]  # non-numeric values skipped


def test_action_panel_defaults_and_absent_command():
    a = build_action(_tick(), {})
    assert a.collective.marker == DEFAULT_HOVER_THR == 0.4
    assert a.rates[0].fraction == pytest.approx(2.0 / DEFAULT_MAX_BODY_RATE)
    assert build_action(None, META).available is False
    assert build_action(_tick(command=None), META).available is False
    assert build_action(_tick(sink_output={"stepped": True}), META).sticks == ()


# ----------------------------------------------------------------------------- inputs


def test_inputs_grouped_by_obs_and_contract_block():
    m = build_inputs(_tick(), META)
    titles = [g.title for g in m.groups]
    assert titles == [
        "rel",
        "g_body",
        "vel",
        "omega",
        "curr_gate",
        "next_lookahead",
        "obstacle_vision",
        "raycast",
    ]
    widths = {g.title: len(g.bars) for g in m.groups}
    assert widths == {
        "rel": 3,
        "g_body": 3,
        "vel": 3,
        "omega": 3,
        "curr_gate": 3,
        "next_lookahead": 7,
        "obstacle_vision": 8,
        "raycast": 40,
    }
    rel = m.groups[0].bars
    assert rel[1].fraction == pytest.approx(-2.0 / 5.0) and rel[1].text == "-2.00"
    vel = m.groups[2].bars
    assert vel[0].fraction == 1.0  # 6 m/s clipped at the 5 m/s full scale
    curr = m.groups[4].bars
    assert curr[0].text == "+12.00"  # the block starts at extra[12]


@pytest.mark.parametrize(("extra_dim", "last"), [(12, "omega"), (30, "obstacle_vision")])
def test_inputs_without_meta_use_the_contract_layout(extra_dim, last):
    m = build_inputs(_tick(extra=list(range(extra_dim))), {})
    assert m.groups[-1].title == last
    assert "obs" not in [g.title for g in m.groups]


def test_inputs_absent():
    assert build_inputs(_tick(obs=None, extra=None), META).available is False
    assert build_inputs(_tick(obs=[1, 2], extra=None), META).available is False  # short obs


# ----------------------------------------------------------------------------- raycast


def test_contract_cone_mirror_matches_the_real_contract(harness_contract):
    np.testing.assert_allclose(
        contract_cone_xy(), np.asarray(harness_contract.CONE)[:, :2], atol=1e-6
    )


def test_raycast_fan_greys_out_of_fov_rays():
    m = build_raycast(_tick(), META)
    assert m.available and len(m.rays) == 40 and m.coverage == 0.75
    assert [r.in_fov for r in m.rays] == [True] * 30 + [False] * 10
    radii = [math.hypot(r.x, r.y) for r in m.rays]
    assert max(radii) == pytest.approx(1.0, abs=1e-3) and radii[0] == pytest.approx(0.0)
    assert m.rays[0].value == 1.0 and m.rays[39].value == pytest.approx(0.22)


def test_raycast_uses_the_meta_cone_when_present():
    cone = [[0.5 * math.sin(math.radians(45)), 0.0]] * 40
    m = build_raycast(_tick(), {"cone_xy": cone})
    assert all(r.x == pytest.approx(0.5) and r.y == 0.0 for r in m.rays)
    wrong_size = build_raycast(_tick(), {"cone_xy": [[0.0, 0.0]] * 3})
    np.testing.assert_allclose(
        [(r.x, r.y) for r in wrong_size.rays],
        contract_cone_xy() / math.sin(math.radians(45)),
        atol=1e-6,
    )


def test_raycast_absent_and_no_mask():
    m = build_raycast(_tick(free_space={"obstacle_vision": [0] * 8}), META)
    assert not m.available and "no raycast" in m.message
    m = build_raycast(_tick(free_space={"raycast": [0.5] * 40}), META)
    assert all(r.in_fov for r in m.rays) and m.coverage is None


# ----------------------------------------------------------------------------- perception


def test_frame_to_screen_full_monitor_region_and_dpr():
    pts = [[0, 0], [1280, 540], [640, 270]]
    full = frame_to_screen(pts, (1280, 540), None, (0, 0, 2560, 1080))
    np.testing.assert_allclose(full, [[0, 0], [2560, 1080], [1280, 540]])
    region = frame_to_screen(pts, (1280, 540), (100, 50, 1280, 540), (0, 0, 2560, 1080))
    np.testing.assert_allclose(region, [[100, 50], [1380, 590], [740, 320]])
    dpr = frame_to_screen(pts, (1280, 540), (100, 50, 1280, 540), (0, 0, 2560, 1080), dpr=1.25)
    np.testing.assert_allclose(dpr, np.asarray(region) / 1.25)
    with pytest.raises(ValueError, match="frame_size"):
        frame_to_screen(pts, (0, 540), None, (0, 0, 10, 10))


def test_perception_lines_and_keypoints_in_screen_space():
    screen = ScreenMap((0, 0, 2560, 1080), region=None)
    m = build_perception(_tick(), META, screen)
    assert m.available and m.frame_age_s == pytest.approx(0.04)
    text = "\n".join(m.lines)
    assert "gate tracking  idx 2  lap 1" in text and "class square" in text
    assert "range 5.00 m train / 10.00 m game" in text  # world_scale 2
    assert "rel +1.00 -2.00 +0.50" in text and "frame age 40 ms" in text
    (kp,) = m.keypoints
    np.testing.assert_allclose(kp.points, [[200, 100], [600, 100], [600, 500], [200, 500]])
    assert kp.center == (400.0, 300.0) and kp.matched and kp.cls == "square"


def test_perception_keypoints_disabled_or_unmappable():
    screen = ScreenMap((0, 0, 2560, 1080))
    assert build_perception(_tick(), META, screen, draw_keypoints=False).keypoints == ()
    assert build_perception(_tick(), META, None).keypoints == ()
    no_size = _tick()
    del no_size["gate"]["info"]["frame_size"]
    assert build_perception(no_size, META, screen).keypoints == ()
    assert build_perception(_tick(gate=None), META, screen).available is False


# ----------------------------------------------------------------------------- health


@pytest.mark.parametrize(
    ("age", "limit", "level"),
    [
        (0.1, 1.0, "ok"),
        (AMBER_FRACTION * 1.0, 1.0, "amber"),
        (0.99, 1.0, "amber"),
        (1.0, 1.0, "amber"),
        (1.01, 1.0, "red"),
        (5.0, None, "ok"),
        (5.0, 0.0, "ok"),
    ],
)
def test_age_level(age, limit, level):
    assert age_level(age, limit) == level


def test_health_panel():
    h = build_health(_snap(_tick(), META))
    assert h.mode == MODE_LIVE and h.tick_rate_hz == 49.8
    assert h.budget_s == pytest.approx(0.02)
    assert [n for n, _ in h.latency] == [
        "state",
        "gate",
        "free_space",
        "inference",
        "actuation",
        "overlay",
    ]
    assert dict(h.latency)["overlay"] == pytest.approx(0.0007)
    levels = {row.name: row.level for row in h.ages}
    assert levels == {"telemetry": "ok", "frame": "ok", "gate": "amber", "free_space": "red"}
    assert h.missed_deadlines == 2 and h.drops == 5
    assert h.focus is True and h.focus_window == "liftoff.exe"
    assert h.run_line == "liftoff m.pt deadbeef extra 70"
    assert h.stop_line is None


def test_health_stop_line_and_failsafe_failure():
    stop = {"reason": "focus_lost", "error": "target window lost focus", "failsafe_ok": True}
    h = build_health(_snap(None, META, mode=MODE_STOPPED, stop=stop))
    assert h.stop_line == "stop: focus_lost (target window lost focus)  fail-safe ok"
    bad = build_health(_snap(None, {}, mode=MODE_STOPPED, stop={"reason": "sink_error"}))
    assert "FAIL-SAFE FAILED" in bad.stop_line


def test_health_without_overlay_row_and_while_waiting():
    t = _tick()
    del t["latency_s"]["overlay"]
    assert "overlay" not in dict(build_health(_snap(t, META)).latency)
    h = build_health(_snap(None, None, mode=MODE_WAITING, status={"drops": 1, "run_info": {}}))
    assert h.latency == () and h.ages == () and h.drops == 1 and h.budget_s is None
    assert h.focus is None and h.missed_deadlines is None


# ----------------------------------------------------------------------------- telemetry


def test_quat_to_euler():
    assert quat_to_euler_deg([1, 0, 0, 0]) == pytest.approx((0.0, 0.0, 0.0))
    r, p, y = quat_to_euler_deg([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
    assert (r, p, y) == pytest.approx((0.0, 0.0, 90.0))
    r, p, y = quat_to_euler_deg([math.cos(math.pi / 8), math.sin(math.pi / 8), 0, 0])
    assert r == pytest.approx(45.0)


def test_telemetry_panel():
    m = build_telemetry(_tick(), META)
    assert m.lines == (
        "speed 10.0 m/s",  # |(3, 4, 0)| = 5 training m/s x world_scale 2
        "alt 3.0 m",
        "roll +0  pitch +0  yaw +90 deg",
        "battery 80%  16.20 V",
        "inputs T +0.50 Y +0.00 P -0.10 R +0.20",
    )
    bare = build_telemetry(_tick(sample=None), {})
    assert bare.lines[-2:] == ("battery n/a", "inputs n/a") and bare.lines[0] == "speed 5.0 m/s"
    assert build_telemetry(_tick(state={}), META).available is False


# ----------------------------------------------------------------------------- all panels / layout


def test_build_panels_tolerates_empty_and_garbage_snapshots():
    empty = build_panels(_snap(None, None, mode=MODE_WAITING))
    assert not empty.action.available and not empty.inputs.available
    assert not empty.raycast.available and not empty.perception.available
    junk = _tick(
        obs="nope",
        extra={"a": 1},
        command={"raw": "x", "collective": "y", "rates": None},
        free_space={"raycast": "zz"},
        latency_s={"state": "slow"},
        ages_s={"gate": None},
    )
    m = build_panels(_snap(junk, {"extra_layout": "bad"}), ScreenMap((0, 0, 100, 100)))
    assert m.action.raw == () and m.action.collective is None
    assert "state" not in dict(m.health.latency) and m.health.ages == ()


def test_default_layout_keeps_the_game_centre_clear_and_stays_on_screen():
    w, h = 2560, 1080
    lay = default_layout(w, h, {p: True for p in DEFAULT_RECTS})
    assert set(lay.rects) == set(DEFAULT_RECTS)
    for name, r in lay.rects.items():
        assert 0 <= r.x and r.x + r.w <= w and 0 <= r.y and r.y + r.h <= h, name
        assert r.x + r.w <= 0.2 * w or r.x >= 0.8 * w, name  # edge columns only
    rects = list(lay.rects.values())
    for i, a in enumerate(rects):  # no two panels overlap
        for b in rects[i + 1 :]:
            assert a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y
