"""Debug-panel data models (toolkit-free): one :class:`PanelModels` per redraw.

Pure functions of a :class:`~drone_fly.overlay.state.StateSnapshot` (the latest tick / meta /
status / stop), so every number the window draws is unit-testable without a display. The painter
only lays these out; it computes nothing.

Panels (UC-65 AC8): **action** (raw action, collective vs hover, body rates vs
``MAX_BODY_RATE``, the sink's stick values), **inputs** (obs 12 as rel / g_body / vel / omega and
the extra vector by contract block, as signed bars), **raycast** (the 40-ray fan on a disk, out-of-
FOV rays greyed), **perception** (gate status / index / lap / class / range / rel / frame age and
the detected keypoints mapped to screen pixels), **health** (tick rate, stacked per-stage latency,
missed deadlines, estimate ages vs their fail-safe limits, focus, stop reason / fail-safe,
publisher drops) and **telemetry** (speed, altitude, attitude, battery, stick inputs).

Contract facts come from the ``meta`` datagram; without one (e.g. before the first meta) the
documented contract values below are used — :func:`contract_cone_xy` is a NumPy mirror of
``gpu_prototype/inference.py::_build_cone`` (the overlay never imports the contract).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from drone_fly.overlay.state import StateSnapshot

#: Contract defaults (INFERENCE_CONTRACT.md) used when no meta datagram has arrived.
DEFAULT_MAX_BODY_RATE = 4.0
DEFAULT_HOVER_THR = 0.4
N_RAYS = 40
CONE_HALF_DEG = 45.0
#: Age colouring: amber at >= this fraction of the fail-safe limit, red past the limit.
AMBER_FRACTION = 0.8
LATENCY_STAGES = (
    ("state", "latency_s", "state"),
    ("gate", "perception_s", "gate"),
    ("free_space", "perception_s", "free_space"),
    ("inference", "latency_s", "inference"),
    ("actuation", "latency_s", "actuation"),
    ("overlay", "latency_s", "overlay"),
)
AGE_KEYS = ("telemetry", "frame", "gate", "free_space")
#: obs(12) groups (INFERENCE_CONTRACT.md §1) and the bar full-scale of each.
OBS_GROUPS = (("rel", 0, 3, 5.0), ("g_body", 3, 6, 1.0), ("vel", 6, 9, 5.0), ("omega", 9, 12, 4.0))
EXTRA_BLOCK_SCALE = {
    "curr_gate": 2.0,
    "next_lookahead": 2.0,
    "obstacle_vision": 1.0,
    "raycast": 1.0,
}


def contract_cone_xy(n_rays: int = N_RAYS, half_deg: float = CONE_HALF_DEG) -> np.ndarray:
    """``(n_rays, 2)``: the x/y of the contract's ray cone (mirror of ``inference._build_cone``).

    The cone is a Fibonacci spiral around +z, so x/y are the rays' offsets from the heading axis;
    their magnitude is ``sin(angle off axis)`` (at most ``sin(half_deg)``).
    """
    ga = math.pi * (3 - math.sqrt(5))
    cz_min = math.cos(math.radians(half_deg))
    out = []
    for i in range(n_rays):
        cz = 1.0 - (i / max(1, n_rays - 1)) * (1 - cz_min)
        st = math.sqrt(max(0.0, 1 - cz * cz))
        phi = i * ga
        out.append((st * math.cos(phi), st * math.sin(phi)))
    return np.asarray(out, np.float32)


# ----------------------------------------------------------------------------- models


@dataclass(frozen=True)
class Bar:
    """One bar: ``fraction`` in [-1, 1] (signed) or [0, 1]; ``text`` the value as shown."""

    label: str
    fraction: float
    text: str
    signed: bool = True
    marker: float | None = None  # e.g. the hover collective, same units as ``fraction``


@dataclass(frozen=True)
class BarGroup:
    title: str
    bars: tuple[Bar, ...]


@dataclass(frozen=True)
class ActionModel:
    available: bool
    raw: tuple[Bar, ...] = ()
    collective: Bar | None = None
    rates: tuple[Bar, ...] = ()
    sticks: tuple[Bar, ...] = ()


@dataclass(frozen=True)
class InputsModel:
    available: bool
    groups: tuple[BarGroup, ...] = ()


@dataclass(frozen=True)
class Ray:
    x: float  # disk coordinates in [-1, 1] (cone edge = radius 1); +x right, +y up
    y: float
    value: float  # normalised free distance (0 = touching, 1 = clear at RAY_RANGE)
    in_fov: bool


@dataclass(frozen=True)
class RaycastModel:
    available: bool
    rays: tuple[Ray, ...] = ()
    coverage: float | None = None
    message: str = ""


@dataclass(frozen=True)
class Keypoints:
    points: np.ndarray  # (K, 2) window-local logical pixels
    center: tuple[float, float]
    matched: bool
    cls: str | None


@dataclass(frozen=True)
class PerceptionModel:
    available: bool
    lines: tuple[str, ...] = ()
    keypoints: tuple[Keypoints, ...] = ()
    frame_age_s: float | None = None


@dataclass(frozen=True)
class AgeRow:
    name: str
    age_s: float
    limit_s: float | None
    level: str  # ok | amber | red


@dataclass(frozen=True)
class HealthModel:
    mode: str
    message: str
    tick_rate_hz: float | None
    budget_s: float | None
    latency: tuple[tuple[str, float], ...]
    missed_deadlines: int | None
    ages: tuple[AgeRow, ...]
    focus: bool | None
    focus_window: str | None
    stop_line: str | None
    drops: int | None
    run_line: str


@dataclass(frozen=True)
class TelemetryModel:
    available: bool
    lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class PanelModels:
    action: ActionModel
    inputs: InputsModel
    raycast: RaycastModel
    perception: PerceptionModel
    health: HealthModel
    telemetry: TelemetryModel
    meta: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------- helpers


def _vec(v: Any, n: int | None = None) -> np.ndarray | None:
    if v is None:
        return None
    try:
        a = np.asarray(v, np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if n is not None and a.shape[0] < n:
        return None
    return a


def _bar(label: str, value: float, scale: float, *, fmt: str = "{:+.2f}") -> Bar:
    frac = 0.0 if not math.isfinite(value) else float(np.clip(value / scale, -1, 1))
    return Bar(label, frac, fmt.format(value))


def age_level(age: float, limit: float | None) -> str:
    """``ok`` / ``amber`` (>= 80 % of the limit) / ``red`` (past the limit)."""
    if limit is None or not limit > 0:
        return "ok"
    if age > limit:
        return "red"
    if age >= AMBER_FRACTION * limit:
        return "amber"
    return "ok"


def quat_to_euler_deg(q: Sequence[float]) -> tuple[float, float, float]:
    """``[w, x, y, z]`` (body -> world, Z-up) -> roll, pitch, yaw in degrees."""
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def frame_to_screen(
    points: Any,
    frame_size: Sequence[float],
    region: Sequence[float] | None,
    monitor_rect: Sequence[float],
    dpr: float = 1.0,
) -> np.ndarray:
    """Frame pixels -> **window-local logical** pixels of an overlay covering ``monitor_rect``.

    ``region`` is the captured ``[left, top, width, height]`` **relative to the monitor**
    (``camera.region``; ``None`` = the whole monitor) and ``frame_size`` the captured frame's
    ``[w, h]``; a frame pixel maps to ``region_origin + p * region_size / frame_size`` in monitor
    pixels, divided by ``dpr`` for the toolkit's logical coordinates (the overlay disables Qt's
    high-DPI scaling, so ``dpr`` is 1 unless a toolkit forces scaling).
    """
    p = np.asarray(points, np.float64).reshape(-1, 2)
    fw, fh = float(frame_size[0]), float(frame_size[1])
    if region is None:
        rx, ry, rw, rh = 0.0, 0.0, float(monitor_rect[2]), float(monitor_rect[3])
    else:
        rx, ry, rw, rh = (float(v) for v in region)
    if fw <= 0 or fh <= 0:
        raise ValueError(f"frame_size must be positive, got {frame_size!r}")
    out = np.empty_like(p)
    out[:, 0] = (rx + p[:, 0] * rw / fw) / dpr
    out[:, 1] = (ry + p[:, 1] * rh / fh) / dpr
    return out


@dataclass(frozen=True)
class ScreenMap:
    """What :func:`frame_to_screen` needs, fixed for the overlay's lifetime."""

    monitor_rect: tuple[int, int, int, int]
    region: tuple[float, float, float, float] | None = None
    dpr: float = 1.0

    def map(self, points: Any, frame_size: Sequence[float]) -> np.ndarray:
        return frame_to_screen(points, frame_size, self.region, self.monitor_rect, self.dpr)


# ----------------------------------------------------------------------------- builders


def build_action(tick: Mapping[str, Any] | None, meta: Mapping[str, Any]) -> ActionModel:
    cmd = (tick or {}).get("command")
    if not cmd:
        return ActionModel(False)
    raw = _vec(cmd.get("raw"), 4)
    rates = _vec(cmd.get("rates"), 3)
    max_rate = float(meta.get("max_body_rate") or DEFAULT_MAX_BODY_RATE)
    hover = float(meta.get("hover_thr") or DEFAULT_HOVER_THR)
    raw_bars = (
        ()
        if raw is None
        else tuple(
            _bar(n, v, 1.0)
            for n, v in zip(("a0 col", "a1 wx", "a2 wy", "a3 wz"), raw[:4], strict=True)
        )
    )
    col = cmd.get("collective")
    collective = None
    if isinstance(col, (int, float)):
        collective = Bar(
            "collective", float(np.clip(col, 0, 1)), f"{col:.2f}", signed=False, marker=hover
        )
    rate_bars = (
        ()
        if rates is None
        else tuple(
            _bar(n, v, max_rate, fmt="{:+.2f} rad/s")
            for n, v in zip(("wx", "wy", "wz"), rates[:3], strict=True)
        )
    )
    sticks_src = (tick or {}).get("sink_output") or {}
    sticks = sticks_src.get("sticks") if isinstance(sticks_src.get("sticks"), dict) else None
    stick_bars = (
        ()
        if not sticks
        else tuple(_bar(str(k), float(v), 1.0) for k, v in sticks.items() if _num(v))
    )
    return ActionModel(True, raw_bars, collective, rate_bars, stick_bars)


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def build_inputs(tick: Mapping[str, Any] | None, meta: Mapping[str, Any]) -> InputsModel:
    t = tick or {}
    obs = _vec(t.get("obs"), 12)
    extra = _vec(t.get("extra"))
    if obs is None and extra is None:
        return InputsModel(False)
    groups: list[BarGroup] = []
    if obs is not None:
        for name, a, b, scale in OBS_GROUPS:
            groups.append(
                BarGroup(name, tuple(_bar(f"{name}[{i}]", obs[a + i], scale) for i in range(b - a)))
            )
    if extra is not None:
        layout = meta.get("extra_layout") or _default_layout(extra.shape[0])
        for name, start, width in layout:
            if name == "obs":
                continue  # the extra vector repeats obs(12) first; shown above
            seg = extra[int(start) : int(start) + int(width)]
            scale = EXTRA_BLOCK_SCALE.get(name, 1.0)
            groups.append(
                BarGroup(name, tuple(_bar(f"{name}[{i}]", v, scale) for i, v in enumerate(seg)))
            )
    return InputsModel(True, tuple(groups))


def _default_layout(extra_dim: int) -> list[list[Any]]:
    out, start = [], 0
    for name, width in (
        ("obs", 12),
        ("curr_gate", 3),
        ("next_lookahead", 7),
        ("obstacle_vision", 8),
        ("raycast", 40),
    ):
        if start >= extra_dim:
            break
        out.append([name, start, min(width, extra_dim - start)])
        start += width
    return out


def build_raycast(tick: Mapping[str, Any] | None, meta: Mapping[str, Any]) -> RaycastModel:
    fs = (tick or {}).get("free_space") or {}
    rays = _vec(fs.get("raycast"))
    if rays is None:
        return RaycastModel(False, message="no raycast (free-space estimator without rays)")
    cone = _vec(meta.get("cone_xy"))
    cone_xy = (
        cone.reshape(-1, 2)
        if cone is not None and cone.size == 2 * rays.shape[0]
        else contract_cone_xy(rays.shape[0])
    )
    radius = math.sin(math.radians(CONE_HALF_DEG))
    mask = _vec(fs.get("in_fov_mask"))
    out = []
    for i, v in enumerate(rays):
        in_fov = True if mask is None or i >= mask.shape[0] else bool(mask[i] > 0.5)
        x, y = cone_xy[i] / radius
        out.append(Ray(float(x), float(y), float(np.clip(v, 0, 1)), in_fov))
    cov = fs.get("coverage")
    return RaycastModel(True, tuple(out), float(cov) if _num(cov) else None)


def build_perception(
    tick: Mapping[str, Any] | None,
    meta: Mapping[str, Any],
    screen: ScreenMap | None = None,
    *,
    draw_keypoints: bool = True,
) -> PerceptionModel:
    t = tick or {}
    gate = t.get("gate")
    if not gate:
        return PerceptionModel(False, ("no gate estimate",))
    info = gate.get("info") or {}
    scale = float(meta.get("world_scale") or 1.0)
    lines = [
        f"gate {gate.get('status', '?')}  idx {gate.get('gate_index', '?')}  "
        f"lap {gate.get('lap', '?')}",
    ]
    if info.get("gate_class"):
        lines.append(f"class {info['gate_class']}")
    target = _vec(gate.get("target"), 3)
    pos = _vec((t.get("state") or {}).get("pos"), 3)
    if target is not None and pos is not None:
        rng = float(np.linalg.norm(target - pos))
        lines.append(f"range {rng:.2f} m train / {rng * scale:.2f} m game")
    obs = _vec(t.get("obs"), 3)
    if obs is not None:
        lines.append(f"rel {obs[0]:+.2f} {obs[1]:+.2f} {obs[2]:+.2f}")
    if "detections" in info:
        lines.append(f"detections {info['detections']}  matched {bool(info.get('matched'))}")
    frame_age = (t.get("ages_s") or {}).get("frame")
    frame_age = float(frame_age) if _num(frame_age) else None
    if frame_age is not None:
        lines.append(f"frame age {frame_age * 1000:.0f} ms")
    kps: list[Keypoints] = []
    size = info.get("frame_size")
    if draw_keypoints and screen is not None and size and info.get("detections_px"):
        for d in info["detections_px"]:
            pts = _vec(d.get("points"))
            ctr = _vec(d.get("center"), 2)
            if pts is None or pts.size < 2 or ctr is None:
                continue
            mapped = screen.map(pts.reshape(-1, 2), size)
            c = screen.map(ctr[:2].reshape(1, 2), size)[0]
            kps.append(
                Keypoints(mapped, (float(c[0]), float(c[1])), bool(d.get("matched")), d.get("cls"))
            )
    return PerceptionModel(True, tuple(lines), tuple(kps), frame_age)


def build_health(snap: StateSnapshot) -> HealthModel:
    t = snap.tick or {}
    meta = snap.meta or {}
    lat_src = {"latency_s": t.get("latency_s") or {}, "perception_s": t.get("perception_s") or {}}
    latency = tuple(
        (name, float(lat_src[where][key]))
        for name, where, key in LATENCY_STAGES
        if _num(lat_src[where].get(key))
    )
    limits = meta.get("age_limits") or {}
    ages = []
    for k in AGE_KEYS:
        v = (t.get("ages_s") or {}).get(k)
        if _num(v):
            lim = limits.get(k)
            lim = float(lim) if _num(lim) else None
            ages.append(AgeRow(k, float(v), lim, age_level(float(v), lim)))
    rate = meta.get("rate_hz")
    stop = snap.last_stop
    stop_line = None
    if stop is not None:
        fs = "fail-safe ok" if stop.get("failsafe_ok") else "FAIL-SAFE FAILED"
        stop_line = f"stop: {stop.get('reason')}" + (
            f" ({stop.get('error')})" if stop.get("error") else ""
        )
        stop_line += f"  {fs}"
    src = t or (snap.status or {}) or (stop or {})
    info = src.get("run_info") or meta.get("run_info") or {}
    run_line = " ".join(
        str(x)
        for x in (
            info.get("harness"),
            info.get("checkpoint"),
            info.get("sha8"),
            None if info.get("extra_dim") is None else f"extra {info.get('extra_dim')}",
            None if not info.get("replay") else f"replay {info['replay']}",
        )
        if x
    )
    drops = src.get("drops")
    return HealthModel(
        mode=snap.mode,
        message=snap.message,
        tick_rate_hz=snap.tick_rate_hz,
        budget_s=1.0 / float(rate) if _num(rate) and rate > 0 else None,
        latency=latency,
        missed_deadlines=t.get("missed_deadlines") if _num(t.get("missed_deadlines")) else None,
        ages=tuple(ages),
        focus=t.get("focus") if isinstance(t.get("focus"), bool) else None,
        focus_window=t.get("focus_window"),
        stop_line=stop_line,
        drops=int(drops) if _num(drops) else None,
        run_line=run_line,
    )


def build_telemetry(tick: Mapping[str, Any] | None, meta: Mapping[str, Any]) -> TelemetryModel:
    t = tick or {}
    state = t.get("state") or {}
    vel, pos, quat = (
        _vec(state.get("vel"), 3),
        _vec(state.get("pos"), 3),
        _vec(state.get("quat"), 4),
    )
    if vel is None and pos is None:
        return TelemetryModel(False, ("no state",))
    scale = float(meta.get("world_scale") or 1.0)
    lines = []
    if vel is not None:
        lines.append(f"speed {np.linalg.norm(vel) * scale:.1f} m/s")
    if pos is not None:
        lines.append(f"alt {pos[2] * scale:.1f} m")
    if quat is not None:
        r, p, y = quat_to_euler_deg(quat)
        lines.append(f"roll {r:+.0f}  pitch {p:+.0f}  yaw {y:+.0f} deg")
    sample = t.get("sample") or {}
    batt = _vec(sample.get("battery"))
    if batt is not None and batt.size >= 2:
        lines.append(f"battery {batt[0]:.0%}  {batt[1]:.2f} V")
    elif batt is not None and batt.size == 1:
        lines.append(f"battery {batt[0]:.2f}")
    else:
        lines.append("battery n/a")
    inputs = _vec(sample.get("inputs"))
    if inputs is not None and inputs.size >= 4:
        lines.append(
            f"inputs T {inputs[0]:+.2f} Y {inputs[1]:+.2f} P {inputs[2]:+.2f} R {inputs[3]:+.2f}"
        )
    else:
        lines.append("inputs n/a")
    return TelemetryModel(True, tuple(lines))


def build_panels(
    snap: StateSnapshot, screen: ScreenMap | None = None, *, draw_keypoints: bool = True
) -> PanelModels:
    """Every panel model for one redraw."""
    meta = dict(snap.meta or {})
    return PanelModels(
        action=build_action(snap.tick, meta),
        inputs=build_inputs(snap.tick, meta),
        raycast=build_raycast(snap.tick, meta),
        perception=build_perception(snap.tick, meta, screen, draw_keypoints=draw_keypoints),
        health=build_health(snap),
        telemetry=build_telemetry(snap.tick, meta),
        meta=meta,
    )
