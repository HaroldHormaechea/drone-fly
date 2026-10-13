"""Fit and validate the plant's battery + thrust-at-speed model (UC68; :mod:`drone_fly.sim.power`).

``python -m drone_fly.sysid.power_fit fit <telemetry.csv | glob> ... [--json out.json]``
``python -m drone_fly.sysid.power_fit validate --drone <id> <live telemetry glob>``

Inputs are Liftoff telemetry CSVs (the sysid recorder's / the live harness's columns: ``t``,
velocity, quaternion, ``in_*`` sticks, ``batt_charge``, ``batt_volts``, ``rpm_0..3``).

**Fit.**

* *Thrust at speed*: windows of 10 samples give the measured body-z specific thrust (velocity
  derivative + gravity projected on body up, plus the profile drag along body z). Full-stick
  windows fit ``T = max_g * (V / V_full) ** n_v * (1 - c * max(v_axial, 0) / v_ref)`` by grid
  search (``v_ref`` = 20 m/s); the effort exponent ``beta`` (``E = T * inflow ** -beta``) is the
  least-squares slope of ``log(E_rpm / T)`` on ``-log(inflow)``.
* *Battery*: per sample, motor effort ``E = max_g * rpm / rpm_max`` (Liftoff's RPM is linear in
  the zero-inflow thrust: hover RPM gives 1 g); the ODE ``q' = -(d0 + d1 E^k)``,
  ``s' = a E^p``, ``V = V_full - b (1 - q) - s - r E^pr`` is replayed per session from spin-up and
  fitted by least squares to the logged charge and volts (before the cutoff latches).

**Validate** (the profile's fitted block, nominal, float64, CPU):

* *open loop*: each live attempt's sticks drive the full plant (``power: true``) from a level
  ground start; reported: the cutoff time vs the logged one, both from spin-up (the first sample
  with a motor above 20000 rpm, UC69's convention);
* *thrust at speed*: the power model alone, driven by the logged effort and axial speed, against
  the measured full-stick specific thrust per speed band.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from typing import Any

import numpy as np

G = 9.81
V_REF_INFLOW = 20.0


def _read(path: str) -> Any:
    import pandas as pd

    return pd.read_csv(path).drop_duplicates("t").sort_values("t").reset_index(drop=True)


def thrust_windows(path: str, drag: tuple[float, float], win: int = 10) -> Any:
    """Measured body-z specific thrust (g) per window, with v_axial, speed, rpm, volts."""
    import pandas as pd

    d = _read(path)
    t = d.t.to_numpy()
    v = d[["vel_x", "vel_y", "vel_z"]].to_numpy()
    x, y, z, w = d[["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy().T
    up = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], 1)
    rpm = d[["rpm_0", "rpm_1", "rpm_2", "rpm_3"]].to_numpy().mean(1)
    thr = (d.in_throttle.to_numpy() + 1) / 2
    k1, k2 = drag
    out = []
    for i in range(0, len(d) - win, win // 2):
        j = i + win
        dt = t[j] - t[i]
        if not 0 < dt <= 0.5:
            continue
        u = up[i : j + 1].mean(0)
        u /= np.linalg.norm(u)
        vm = v[i : j + 1].mean(0)
        vbz, spd = float(vm @ u), float(np.linalg.norm(vm))
        f = (v[j] - v[i]) / dt + np.array([0.0, G, 0.0])  # Unity: y up
        out.append({
            "T_meas": (f @ u + k1 * vbz + k2 * spd * vbz) / G, "v_ax": vbz, "speed": spd,
            "rpm": rpm[i : j + 1].mean(), "volts": d.batt_volts.to_numpy()[i : j + 1].mean(),
            "thr": thr[i : j + 1].mean(),
        })  # fmt: skip
    return pd.DataFrame(out)


def fit_inflow(windows: Any, max_g: float, v_full: float) -> dict[str, float]:
    fs = windows[(windows.thr > 0.93) & (windows.rpm > 3000)]
    best = None
    for n in np.linspace(0, 1.5, 31):
        for c in np.linspace(0, 0.8, 81):
            pred = (
                max_g
                * (fs.volts / v_full) ** n
                * np.clip(1 - c * fs.v_ax.clip(lower=0) / V_REF_INFLOW, 0.2, None)
            )
            e = float(np.sqrt(np.mean((pred - fs.T_meas) ** 2)))
            if best is None or e < best[0]:
                best = (e, n, c)
    e, n, c = best
    # effort: RPM-proportional E vs the model thrust at full stick, E = T inflow^-beta
    inflow = np.clip(1 - c * fs.v_ax.clip(lower=0) / V_REF_INFLOW, 0.2, None)
    rpm_max = float(windows.rpm.max())
    x = -np.log(inflow)
    y = np.log((max_g * fs.rpm / rpm_max) / (max_g * inflow))
    beta = float((x * y).sum() / max(float((x * x).sum()), 1e-12))
    return {"effort_inflow_exponent": beta, "thrust_voltage_exponent": float(n),
            "inflow_coeff": float(c), "inflow_v_ref_mps": V_REF_INFLOW, "rmse_g": e,
            "windows": int(len(fs))}  # fmt: skip


def battery_series(path: str, max_g: float, rpm_max: float) -> dict[str, np.ndarray] | None:
    d = _read(path)
    rpm = d[["rpm_0", "rpm_1", "rpm_2", "rpm_3"]].mean(axis=1).to_numpy()
    on = np.flatnonzero(rpm > 100)
    if not len(on):
        return None
    d = d.iloc[on[0] :].reset_index(drop=True)
    return {"t": d.t.to_numpy() - d.t.iloc[0], "E": max_g * rpm[on[0] :] / rpm_max,
            "q": d.batt_charge.to_numpy(), "V": d.batt_volts.to_numpy()}  # fmt: skip


def simulate_battery(x: Any, s: dict[str, np.ndarray], v_full: float) -> tuple[Any, Any]:
    d0, d1, k, a, p, b, r, pr = x
    t, E = s["t"], s["E"]
    q, sag = np.empty_like(t), np.empty_like(t)
    qq, ss = s["q"][0], 0.0
    for i in range(len(t)):
        if i:
            h, e = t[i] - t[i - 1], E[i - 1]
            qq -= (d0 + d1 * e**k) * h
            ss += a * e**p * h
        q[i], sag[i] = qq, ss
    return q, v_full - b * (1 - q) - sag - r * E**pr


def fit_battery(sessions: list[dict[str, np.ndarray]], v_full: float, v_cut: float) -> dict:
    from scipy.optimize import least_squares

    def resid(x: Any) -> np.ndarray:
        out = []
        for s in sessions:
            q, V = simulate_battery(x, s, v_full)
            m = s["V"] > v_cut + 0.02
            out += [10.0 * (q - s["q"])[m][::5], (V - s["V"])[m][::5]]
        return np.concatenate(out)

    x0 = [0.0005, 0.003, 1.0, 0.005, 2.0, 2.0, 0.2, 1.0]
    lo = [0, 0, 0.3, 0, 0.5, 0, 0, 0.3]
    hi = [0.02, 0.05, 3.0, 1.0, 10.0, 6.0, 3.0, 3.0]
    x = least_squares(resid, x0, bounds=(lo, hi)).x
    names = ["drain_idle_per_s", "drain_load_per_s", "drain_exponent", "sag_rate_v_per_s",
             "sag_exponent", "ocv_slope_v", "ir_v", "ir_exponent"]  # fmt: skip
    fit: dict[str, Any] = dict(zip(names, map(float, x), strict=True))
    per = []
    for s in sessions:
        q, V = simulate_battery(x, s, v_full)
        m = s["V"] > v_cut + 0.02
        cm = (
            float(s["t"][np.argmax(s["V"] <= v_cut + 0.02)])
            if (s["V"] <= v_cut + 0.02).any()
            else None
        )
        cp = float(s["t"][np.argmax(V <= v_cut)]) if (V <= v_cut).any() else None
        per.append({"v_rmse": float(np.sqrt(np.mean((V - s["V"])[m] ** 2))),
                    "q_rmse": float(np.sqrt(np.mean((q - s["q"])[m] ** 2))),
                    "cutoff_meas_s": cm, "cutoff_pred_s": cp})  # fmt: skip
    fit["sessions"] = per
    return fit


# ----------------------------------------------------------------------------- validate


def validate(profile: Any, paths: list[str]) -> dict[str, Any]:
    """Open-loop stick replay (cutoff time) + power-model thrust at speed (see the docstring)."""
    import torch

    from drone_fly.sim.plant import Plant, PlantConfig
    from drone_fly.sim.power import PowerState

    spec = profile.power
    v_cut = spec.cutoff_v_per_cell * spec.cells
    rows = []
    for path in paths:
        d = _read(path)
        rpm = d[["rpm_0", "rpm_1", "rpm_2", "rpm_3"]].mean(axis=1).to_numpy()
        on = np.flatnonzero(rpm > 100)
        if not len(on):
            continue
        d = d.iloc[on[0] :].reset_index(drop=True)
        t = d.t.to_numpy() - d.t.iloc[0]
        # report on UC69's clock: spin-up = the first sample with a motor above 20000 rpm
        hot = d[["rpm_0", "rpm_1", "rpm_2", "rpm_3"]].max(axis=1).to_numpy() > 20000
        t_spin = float(t[np.argmax(hot)]) if hot.any() else 0.0
        low = d.batt_volts.to_numpy() <= v_cut + 0.02
        cut_meas = float(t[np.argmax(low)]) - t_spin if low.any() else None
        plant = Plant(profile, 1, "cpu", seed=0,
                      config=PlantConfig(dtype="float64", power=True, substeps=1))  # fmt: skip
        plant.reset(torch.ones(1, dtype=torch.bool), grounded=True, motor_g=0.0)
        cut_t = None
        thr = torch.as_tensor((d.in_throttle.to_numpy() + 1) / 2, dtype=torch.float64)
        sticks = torch.as_tensor(
            d[["in_roll", "in_pitch", "in_yaw"]].to_numpy(), dtype=torch.float64
        )
        for i in range(1, len(t)):
            plant.step_sticks(thr[i - 1 : i], sticks[i - 1 : i], float(t[i] - t[i - 1]))
            if cut_t is None and bool(plant.power.cutoff[0]):
                cut_t = float(t[i]) - t_spin
        rows.append({"attempt": path, "cutoff_meas_s": cut_meas, "cutoff_replay_s": cut_t})
    # thrust at speed: power model alone, driven by the logged effort and axial speed
    k1, k2 = profile.drag.linear_per_s, profile.drag.quadratic_per_m
    k1 = float(np.mean(k1)) if isinstance(k1, tuple) else float(k1)
    k2 = float(np.mean(k2)) if isinstance(k2, tuple) else float(k2)
    import pandas as pd

    win = pd.concat([thrust_windows(p, (k1, k2)) for p in paths], ignore_index=True)
    fs = win[(win.thr > 0.93) & (win.rpm > 3000)]
    ps = PowerState(spec, len(fs), device=torch.device("cpu"), dtype=torch.float64, seed=0,
                    randomization_scale=0.0)  # fmt: skip
    ps.volts = torch.as_tensor(fs.volts.to_numpy())
    pred = ps.available_g(torch.full((len(fs),), float(profile.max_g), dtype=torch.float64),
                          torch.as_tensor(fs.v_ax.to_numpy())).numpy()  # fmt: skip
    bands = []
    for lo, hi in ((0, 8), (8, 16), (16, 22), (22, 30)):
        m = ((fs.speed >= lo) & (fs.speed < hi)).to_numpy()
        if m.sum():
            meas, pr = float(np.median(fs.T_meas[m])), float(np.median(pred[m]))
            bands.append({"speed_m_s": [lo, hi], "n": int(m.sum()), "measured_g": meas,
                          "model_g": pr, "rel_err": (pr - meas) / meas})  # fmt: skip
    return {"attempts": rows, "thrust_at_speed": bands}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m drone_fly.sysid.power_fit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--drone", default="liftoff-gremlin-micro")
    f.add_argument("paths", nargs="+")
    f.add_argument("--json")
    v = sub.add_parser("validate")
    v.add_argument("--drone", default="liftoff-gremlin-micro")
    v.add_argument("paths", nargs="+")
    v.add_argument("--json")
    a = ap.parse_args(argv)
    from drone_fly import drones

    prof = drones.load(a.drone)
    paths = sorted({p for g in a.paths for p in glob.glob(g)})
    if a.cmd == "fit":
        k1, k2 = prof.drag.linear_per_s, prof.drag.quadratic_per_m
        k1 = float(np.mean(k1)) if isinstance(k1, tuple) else float(k1)
        k2 = float(np.mean(k2)) if isinstance(k2, tuple) else float(k2)
        import pandas as pd

        win = pd.concat([thrust_windows(p, (k1, k2)) for p in paths], ignore_index=True)
        rpm_max = float(win.rpm.max())
        v_full = float(_read(paths[0]).batt_volts.max())
        out = {"rpm_max": rpm_max, "v_full": v_full, "sessions": paths,
               "inflow": fit_inflow(win, prof.max_g, v_full)}  # fmt: skip
        series = [s for s in (battery_series(p, prof.max_g, rpm_max) for p in paths) if s]
        out["battery"] = fit_battery(series, v_full, 2.70 * 3)
    else:
        out = validate(prof, paths)
    text = json.dumps(out, indent=1)
    print(text)
    if a.json:
        open(a.json, "w", encoding="utf-8").write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
