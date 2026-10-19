"""``harness flight-report``: what a live Liftoff run physically did, in real metres.

Reads the run folder's raw ``telemetry.csv`` (Liftoff's own Unity frame: metres, ``y`` up), so the
figures never depend on ``scaling.world_scale``. Do NOT read heights or distances off
``ticks.jsonl`` ``state.pos``: that is the policy's training-scale state (Liftoff metres divided by
``world_scale``, e.g. 12) and understates every distance by that factor.

Figures: duration, motors-on window (rpm > 0), airborne window (> :data:`AIRBORNE_M` above the
spawn), max height above spawn, path length (per-sample jumps above
``telemetry.max_position_jump_m`` are respawns and not counted), max horizontal distance from the
spawn, top speed, peak body rate, and the state at the stop (height / rpm: was it still flying?).

Gates are optional and always ground truth: ``--recording`` replays the saved Liftoff recording's
pose against the track (:func:`~drone_fly.harness.offline.recordings.completeness_from_pose`).
The harness's live ``gate_passed`` events are a heuristic and are never reported here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

AIRBORNE_M = 0.3  # height above the spawn that counts as airborne (Liftoff metres)
DEFAULT_JUMP_M = 5.0  # telemetry.max_position_jump_m default (liftoff_5inch.yaml)


def _round(v: float | None, nd: int = 2) -> float | None:
    return None if v is None else round(float(v), nd)


def flight_figures(samples: list[Any], *, max_jump_m: float = DEFAULT_JUMP_M) -> dict[str, Any]:
    """Physical figures of raw Liftoff samples (``TelemetrySample``, Unity frame, metres)."""
    if not samples:
        return {"samples": 0}
    t = np.array([s.t for s in samples], np.float64)
    t = t - t[0]
    pos = np.array([s.position for s in samples], np.float64)
    vel = np.array([s.velocity for s in samples], np.float64)
    gyro = np.array([s.gyro for s in samples], np.float64)
    spawn = pos[0]
    height = pos[:, 1] - spawn[1]
    horiz = np.hypot(pos[:, 0] - spawn[0], pos[:, 2] - spawn[2])
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    jumps = steps > max_jump_m
    up = np.flatnonzero(height > AIRBORNE_M)
    rpm = [s.rpm for s in samples]
    spinning = np.flatnonzero([r is not None and float(np.max(r)) > 0.0 for r in rpm])
    last_rpm = rpm[-1]
    return {
        "samples": len(samples),
        "duration_s": _round(t[-1]),
        "motors_on_s": None
        if not spinning.size
        else [_round(t[spinning[0]]), _round(t[spinning[-1]])],
        "airborne_s": None if not up.size else [_round(t[up[0]]), _round(t[up[-1]])],
        "airborne_duration_s": None if not up.size else _round(t[up[-1]] - t[up[0]]),
        "max_height_m": _round(height.max()),
        "path_m": _round(steps[~jumps].sum(), 1),
        "max_horizontal_m": _round(horiz.max(), 1),
        "top_speed_m_s": _round(np.linalg.norm(vel, axis=1).max()),
        "peak_rate": _round(np.abs(gyro).max(), 0),
        "respawns": int(jumps.sum()),
        "final_height_m": _round(height[-1]),
        "final_rpm": None if last_rpm is None else _round(float(np.max(last_rpm)), 0),
        "still_flying_at_stop": bool(
            height[-1] > AIRBORNE_M and last_rpm is not None and float(np.max(last_rpm)) > 0.0
        ),
    }


def resolve_recording(name_or_path: str) -> Path:
    """A recording path, or a bare name saved under Liftoff's ``Recordings/Race`` folder."""
    p = Path(name_or_path)
    if p.suffix.lower() == ".xml" or p.is_file():
        return p
    from drone_fly.harness.offline.recordings import default_recordings_dir

    return default_recordings_dir() / "Race" / f"{name_or_path}.xml"


def recording_gates(path: str | Path, trackdata_root: str | Path | None = None) -> dict[str, Any]:
    """Ground-truth gate crossings of a saved Liftoff recording (pose replayed on the track)."""
    from drone_fly.harness.offline.recordings import (
        completeness_from_pose,
        match_entry,
        parse_recording,
    )
    from drone_fly.tracks import catalogue as cat
    from drone_fly.tracks.loader import load_course

    catalogue = cat.load_named("liftoff")
    rec = parse_recording(path)
    entry, _how = match_entry(rec, catalogue)
    if entry is None:
        return {"recording": str(path), "track": None, "passed": None, "required": None}
    root = cat.resolve_root(trackdata_root, catalogue.header)
    course = load_course(entry, catalogue=catalogue, root=root, scenes_root=None)[0]
    comp = completeness_from_pose(rec, course)
    return {"recording": str(path), "track": entry["id"], **comp}


def flight_report(
    run_dir: str | Path,
    *,
    recording: str | None = None,
    trackdata_root: str | Path | None = None,
) -> dict[str, Any]:
    """Every figure of ``harness flight-report`` for one run folder."""
    import yaml

    from drone_fly.harness.liftoff.telemetry import read_telemetry_csv

    d = Path(run_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    cfg_path = d / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    tel = (cfg or {}).get("telemetry") or {}
    samples = read_telemetry_csv(d / "telemetry.csv")
    out: dict[str, Any] = {
        "run": d.name,
        "checkpoint": meta.get("checkpoint"),
        "stop_reason": meta.get("stop_reason"),
        "world_scale": (meta.get("harness_info") or {}).get("world_scale"),
        "rate_units": tel.get("gyro_units", "deg_s"),
        **flight_figures(samples, max_jump_m=float(tel.get("max_position_jump_m", DEFAULT_JUMP_M))),
    }
    if recording:
        out["gates"] = recording_gates(resolve_recording(recording), trackdata_root)
    return out


def _span(v: list[float] | None) -> str:
    return "-" if v is None else f"{v[0]:.1f} - {v[1]:.1f} s"


def flight_markdown(r: dict[str, Any]) -> str:
    """The report as a Markdown table (all distances in Liftoff metres)."""
    if not r.get("samples"):
        return f"{r['run']}: no telemetry samples"
    rows = [
        ("stop reason", r.get("stop_reason")),
        ("duration", f"{r['duration_s']:.1f} s"),
        ("motors on", _span(r["motors_on_s"])),
        (f"airborne (> {AIRBORNE_M:.1f} m)", _span(r["airborne_s"])),
        ("max height above spawn", f"{r['max_height_m']:.1f} m"),
        ("path flown", f"{r['path_m']:.0f} m"),
        ("max horizontal distance from spawn", f"{r['max_horizontal_m']:.0f} m"),
        ("top speed", f"{r['top_speed_m_s']:.1f} m/s"),
        ("peak body rate", f"{r['peak_rate']:.0f} {r['rate_units']}"),
        ("respawns", r["respawns"]),
        (
            "at stop",
            f"{r['final_height_m']:.1f} m up, rpm {r['final_rpm']}"
            + (" (still flying)" if r["still_flying_at_stop"] else ""),
        ),
    ]
    g = r.get("gates")
    if g is not None:
        gates = "no track match" if g["passed"] is None else f"{g['passed']} / {g['required']}"
        rows.append(("gates (recording pose vs track)", f"{gates} [{g.get('track')}]"))
    lines = [f"### {r['run']}", "", "| figure | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += [
        "",
        "Units: Liftoff metres from telemetry.csv (independent of world_scale "
        f"{r.get('world_scale')}; ticks.jsonl state.pos is scaled down by it).",
    ]
    return "\n".join(lines)


__all__ = [
    "AIRBORNE_M",
    "flight_figures",
    "flight_markdown",
    "flight_report",
    "recording_gates",
    "resolve_recording",
]
