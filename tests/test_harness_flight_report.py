"""``harness flight-report``: physical figures of a live run in Liftoff metres (telemetry.csv)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from drone_fly.cli.main import EXIT_OK, main
from drone_fly.harness.flight_report import flight_figures, flight_markdown, flight_report
from drone_fly.harness.liftoff.telemetry import CSV_COLUMNS, read_telemetry_csv

SPAWN = (-37.7, 0.87, 0.0)  # Unity: y is up


def _row(t: float, x: float, y: float, z: float, v: tuple, rpm: float, gyro: float = 0.0) -> dict:
    vals = [t, x, y, z, 0.0, 0.0, 0.0, 1.0, *v, gyro, 0.0, 0.0]
    vals += [0.0, 0.0, 0.0, 0.0, 1.0, 12.6, rpm, rpm, rpm, rpm]
    return dict(zip(CSV_COLUMNS, vals, strict=True))


def _write_run(tmp_path: Path, rows: list[dict], world_scale: float = 12.0) -> Path:
    d = tmp_path / "liftoff-run"
    d.mkdir(parents=True)
    with (d / "telemetry.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    meta = {"stop_reason": "stale_telemetry", "harness_info": {"world_scale": world_scale}}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "config.yaml").write_text(
        "telemetry:\n  gyro_units: deg_s\n  max_position_jump_m: 5.0\n", encoding="utf-8"
    )
    return d


def _flight() -> list[dict]:
    """Waypoints (t, dx, dy, dz, vel, rpm, gyro) sampled at 10 Hz (real telemetry: ~100 Hz)."""
    x0, y0, z0 = SPAWN
    way = [
        (10.0, 0.0, 0.0, 0.0, (0, 0, 0), 0.0, 0.0),  # armed on the pad
        (11.0, 0.0, 0.0, 0.0, (0, 0, 0), 15000.0, 0.0),  # spinning up
        (12.0, 0.0, 6.0, 0.0, (0, 6.0, 0), 18000.0, 0.0),  # climbed 6 m
        (13.0, 3.0, 10.0, 4.0, (3.0, 4.0, 4.0), 18000.0, -900.0),
        (14.0, 3.0, 2.0, 4.0, (0, -8.0, 0), 0.0, 0.0),  # motors cut, falling
    ]
    rows = [_row(way[0][0], x0, y0, z0, way[0][4], way[0][5])]
    for a, b in zip(way, way[1:], strict=False):
        for i in range(1, 11):
            f = i / 10
            dx, dy, dz = (a[k] + f * (b[k] - a[k]) for k in (1, 2, 3))
            last = i == 10
            rows.append(
                _row(
                    a[0] + f,
                    x0 + dx,
                    y0 + dy,
                    z0 + dz,
                    b[4] if last else a[4],
                    b[5] if last else a[5],
                    b[6] if last else a[6],
                )
            )
    return rows


def test_figures_are_liftoff_metres(tmp_path):
    d = _write_run(tmp_path, _flight())
    r = flight_figures(read_telemetry_csv(d / "telemetry.csv"))
    assert r["duration_s"] == pytest.approx(4.0)
    assert r["max_height_m"] == pytest.approx(10.0)  # NOT 10 / world_scale
    assert r["max_horizontal_m"] == pytest.approx(5.0)
    assert r["path_m"] == pytest.approx(6.0 + 41**0.5 + 8.0, abs=0.1)
    assert r["top_speed_m_s"] == pytest.approx(8.0)
    assert r["peak_rate"] == pytest.approx(900.0)
    assert r["motors_on_s"] == [1.0, 3.9]  # rpm drops to 0 on the last sample
    assert r["airborne_s"] == [1.1, 4.0]  # first sample > 0.3 m (0.6 m at 1.1 s)
    assert r["final_rpm"] == 0.0
    assert r["still_flying_at_stop"] is False


def test_respawn_jump_not_counted_in_path(tmp_path):
    rows = _flight()
    x0, y0, z0 = SPAWN
    rows.append(_row(15.0, x0, y0, z0, (0, 0, 0), 0.0))  # 5.4 m teleport back to the spawn
    d = _write_run(tmp_path, rows)
    r = flight_figures(read_telemetry_csv(d / "telemetry.csv"))
    assert r["respawns"] == 1
    assert r["path_m"] == pytest.approx(6.0 + 41**0.5 + 8.0, abs=0.1)


def test_report_independent_of_world_scale(tmp_path):
    a = flight_report(_write_run(tmp_path / "a", _flight(), world_scale=12.0))
    b = flight_report(_write_run(tmp_path / "b", _flight(), world_scale=1.0))
    keys = ("max_height_m", "path_m", "max_horizontal_m", "top_speed_m_s")
    assert {k: a[k] for k in keys} == {k: b[k] for k in keys}
    assert a["stop_reason"] == "stale_telemetry"
    md = flight_markdown(a)
    assert "| max height above spawn | 10.0 m |" in md
    assert "world_scale 12.0" in md


def test_cli_flight_report(tmp_path, capsys):
    d = _write_run(tmp_path, _flight())
    assert main(["harness", "flight-report", str(d)]) == EXIT_OK
    assert "| path flown | 20 m |" in capsys.readouterr().out
    assert main(["harness", "flight-report", str(d), "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["max_height_m"] == pytest.approx(10.0)
