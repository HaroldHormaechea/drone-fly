"""Write a fit as ``drones/<id>/drone.yaml`` + ``sysid.md`` + residual PNGs.

Every measured leaf carries ``{value, session, residual, sigma}``. Default randomization ranges are
``nominal +- max(3 sigma, 10 %)`` clamped at 0; the multipliers (``thrust_scale``,
``rates.<axis>.rate_scale``) are ``1 +- max(3 sigma_rel, 10 %)``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.drones import store
from drone_fly.sysid.fit import ACCEL_LIMIT_G, RATE_LIMIT_PCT, Estimate, FitResult
from drone_fly.sysid.model import AXES, stick_to_throttle

#: UC64: the hand-flown hover of "Test small drone" was at throttle stick -0.06.
UC64_HOVER_STICK = -0.06
PLOT_FILES = {
    "rates": "rates_{axis}.png",
    "thrust": "thrust_curve.png",
    "drag": "drag.png",
    "delay": "delay.png",
}


def _r(x: float | None, n: int = 6) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return float(f"{x:.{n}g}")


def _leaf(e: Estimate, session: str) -> dict[str, Any]:
    out: dict[str, Any] = {"value": _r(e.value), "session": session}
    out["residual"] = _r(e.residual) if e.residual is not None else 0.0
    if e.sigma is not None and np.isfinite(e.sigma):
        out["sigma"] = _r(e.sigma)
    return out


def _range(nominal: float, sigma: float | None) -> dict[str, float]:
    half = max(3.0 * (sigma or 0.0), 0.1 * abs(nominal))
    return {"min": _r(max(0.0, nominal - half)), "max": _r(nominal + half)}


def _scale(rel_sigma: float | None) -> dict[str, float]:
    half = max(3.0 * (rel_sigma or 0.0), 0.1)
    return {"min": _r(max(0.0, 1.0 - half)), "max": _r(1.0 + half)}


def profile_data(
    fit: FitResult,
    *,
    drone_id: str,
    name: str | None = None,
    liftoff: dict[str, Any] | None = None,
    camera: dict[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """The ``drone.yaml`` mapping for a fit (``source: liftoff-sysid``)."""
    s = fit.session
    e = fit.estimates
    p = fit.params
    rates: dict[str, Any] = {}
    for a in AXES:
        rates[a] = {
            k: _leaf(e[f"rates.{a}.{k}"], s)
            for k in ("rc_rate", "super_rate", "expo", "tau_s", "delay_s")
        }
    nominal: dict[str, Any] = {
        "hover_throttle": _leaf(e["hover_throttle"], s),
        "thrust_curve": {
            "throttle": [_r(t) for t in p.knots_throttle],
            "accel_g": [_r(g) for g in p.knots_g],
            "session": s,
            "residual": _r(fit.residuals["thrust_curve"]),
            "sigma": [_r(x) if np.isfinite(x) else None for x in fit.knot_sigma],
        },
        "rates": rates,
        "drag": {
            "linear_per_s": _leaf(e["drag.linear_per_s"], s),
            "quadratic_per_m": _leaf(e["drag.quadratic_per_m"], s),
        },
        "motor_lag_s": _leaf(e["motor_lag_s"], s),
        "latency": {
            "input_s": _leaf(e["latency.input_s"], s),
            "control_to_gyro_s": _leaf(e["latency.control_to_gyro_s"], s),
        },
        "camera": camera,
    }
    hov = e["hover_throttle"]
    rnd: dict[str, Any] = {"thrust_scale": _scale((hov.sigma or 0.0) / max(hov.value, 1e-6))}
    for a in AXES:
        rc = e[f"rates.{a}.rc_rate"]
        rnd[f"rates.{a}.rate_scale"] = _scale((rc.sigma or 0.0) / max(rc.value, 1e-6))
        for k in ("tau_s", "delay_s"):
            x = e[f"rates.{a}.{k}"]
            rnd[f"rates.{a}.{k}"] = _range(x.value, x.sigma)
    for key in ("drag.linear_per_s", "drag.quadratic_per_m", "motor_lag_s"):
        rnd[key] = _range(e[key].value, e[key].sigma)
    c2g = e["latency.control_to_gyro_s"]
    rnd["latency.control_to_gyro_s"] = _range(c2g.value, c2g.sigma)
    data: dict[str, Any] = {
        "schema_version": 1,
        "id": drone_id,
        "name": name or drone_id,
        "source": "liftoff-sysid",
        "fit_quality": fit.fit_quality,
        "notes": notes or f"Identified by drone-fly sysid from session {s}. See sysid.md.",
        "liftoff": liftoff or {"drone_name": str(fit.meta.get("liftoff_name") or drone_id)},
        "nominal": nominal,
        "randomization": rnd,
    }
    return data


# ----------------------------------------------------------------------------- plots


def write_plots(fit: FitResult, out_dir: Path) -> list[str]:
    """Residual / fit plots next to ``sysid.md`` (matplotlib, Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    traces = fit.plots.get("traces", {})
    for axis in AXES:
        windows = traces.get(axis) or []
        if not windows:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, w in enumerate(windows):
            ax.plot(
                w["t"],
                np.degrees(w["meas"]),
                color="C0",
                lw=1,
                label="measured" if i == 0 else None,
            )
            ax.plot(
                w["t"],
                np.degrees(w["pred"]),
                color="C1",
                lw=1,
                ls="--",
                label="model (open loop)" if i == 0 else None,
            )
        ax.set(
            title=f"{axis}: held-out doublets, 1 s open-loop prediction",
            xlabel="time from window start (s)",
            ylabel="body rate (deg/s)",
        )
        ax.legend()
        name = PLOT_FILES["rates"].format(axis=axis)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=90)
        plt.close(fig)
        written.append(name)
    th = fit.plots.get("thrust")
    if th:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            th["throttle"], th["accel_g"], s=2, alpha=0.3, label="measured (body-z, drag removed)"
        )
        ax.plot(
            fit.params.knots_throttle, fit.params.knots_g, "o-", color="C1", label="fitted curve"
        )
        ax.axhline(1.0, color="grey", lw=0.8)
        ax.set(xlabel="throttle fraction", ylabel="specific force (g)", title="thrust curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / PLOT_FILES["thrust"], dpi=90)
        plt.close(fig)
        written.append(PLOT_FILES["thrust"])
    dr = fit.plots.get("drag")
    if dr:
        v = np.asarray(dr["v"])
        sp = np.asarray(dr["speed"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(v, np.asarray(dr["f"]), s=2, alpha=0.3, label="measured (body x / y)")
        order = np.argsort(v)
        k1, k2 = fit.params.drag_linear, fit.params.drag_quadratic
        ax.plot(v[order], (-k1 * v - k2 * sp * v)[order], color="C1", lw=1, label="fit")
        ax.set(xlabel="body velocity (m/s)", ylabel="specific force (m/s^2)", title="drag")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / PLOT_FILES["drag"], dpi=90)
        plt.close(fig)
        written.append(PLOT_FILES["drag"])
    dl = fit.plots.get("input_delay")
    ml = fit.plots.get("motor_lag")
    if dl or ml:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        if dl:
            axes[0].plot(np.asarray(dl["lag_s"]) * 1000, dl["sse"])
            axes[0].axvline(dl["chosen"] * 1000, color="C1", ls="--")
            axes[0].set(xlabel="pad -> echoed input delay (ms)", ylabel="SSE", title="input delay")
        if ml:
            axes[1].plot(np.asarray(ml["lag_s"]) * 1000, ml["sse"])
            axes[1].axvline(fit.params.motor_lag_s * 1000, color="C1", ls="--")
            axes[1].set(xlabel="motor lag (ms)", ylabel="SSE", title="motor lag")
        fig.tight_layout()
        fig.savefig(out_dir / PLOT_FILES["delay"], dpi=90)
        plt.close(fig)
        written.append(PLOT_FILES["delay"])
    return written


# ----------------------------------------------------------------------------- markdown


def _fmt(x: float | None, n: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{n}g}"


def sysid_markdown(fit: FitResult, data: dict[str, Any], plots: list[str], *, micro: bool) -> str:
    m = fit.meta
    e = fit.estimates
    v = fit.validation
    schedule = m.get("schedule") or []
    passes = 1 + max((x["pass_index"] for x in schedule), default=0)
    lines = [
        f"# System identification: {data['name']} (`{data['id']}`)",
        "",
        f"- Session: `{fit.session}` ({m.get('created_utc', '?')}), plant: "
        f"{m.get('plant', 'liftoff')}, seed {m.get('seed', '-')}",
        f"- Liftoff drone: {data['liftoff'].get('drone_name', '-')}; map: "
        f"{data['liftoff'].get('map') or m.get('map') or '-'}",
        f"- Settings: preset `{m.get('preset', '-')}`; schedule {len(schedule)} experiments in "
        f"{passes} passes (last pass held out)",
        f"- Git: `{m.get('git_head') or '-'}`",
        "",
        f"**fit_quality: {fit.fit_quality}.** {fit.verdict}",
        "",
        "## Fitted values",
        "",
        "| parameter | value | sigma | fit residual |",
        "|---|---|---|---|",
    ]
    for key, est in e.items():
        lines.append(f"| {key} | {_fmt(est.value)} | {_fmt(est.sigma)} | {_fmt(est.residual)} |")
    lines += [
        "",
        "Residuals: rates in rad/s (output error), thrust in g, drag in g, delays as RMS of the "
        "echo error (stick units). Units: throttle fraction 0..1 (stick = 2f - 1), rates "
        "Betaflight (rc_rate 1.0 = 100), seconds.",
        "",
        "### Thrust curve (throttle fraction -> body-z specific force, g)",
        "",
        "| throttle | " + " | ".join(_fmt(t, 3) for t in fit.params.knots_throttle) + " |",
        "|---|" + "---|" * len(fit.params.knots_throttle),
        "| accel (g) | " + " | ".join(_fmt(g, 3) for g in fit.params.knots_g) + " |",
        "| sigma | " + " | ".join(_fmt(s, 2) for s in fit.knot_sigma) + " |",
        "",
        "## Held-out validation (last pass)",
        "",
        "| signal | one-step error | 1 s open-loop error | windows |",
        "|---|---|---|---|",
    ]
    for a in AXES:
        lines.append(
            f"| {a} rate (% of commanded) | {_fmt(v['rate_1step_pct'].get(a), 3)} % | "
            f"{_fmt(v['rate_1s_pct'].get(a), 3)} % | {v['windows'].get(a, 0)} |"
        )
    lines.append(
        f"| vertical accel (g) | {_fmt(v.get('accel_1step_g'), 3)} | "
        f"{_fmt(v.get('accel_1s_g'), 3)} | {v['windows'].get('thrust', 0)} |"
    )
    lines += [
        "",
        f"Limits (AC8): median 1 s open-loop rate error <= {RATE_LIMIT_PCT:.0f} % of the "
        f"commanded rate amplitude, vertical acceleration <= {ACCEL_LIMIT_G} g. {fit.verdict}",
        "",
        "## Hover",
        "",
    ]
    hov = fit.params.hover_throttle
    hc = fit.hover_check
    lines.append(
        f"- From the thrust curve: throttle {hov:.4f} (stick {2 * hov - 1:+.4f})"
        + (
            f"; settled hover throttle {hc['settled_hover_throttle']:.4f} "
            f"({hc['samples']} samples), difference {hc['difference']:+.4f}"
            if hc
            else ""
        )
    )
    if micro:
        uc64 = float(stick_to_throttle(UC64_HOVER_STICK))
        stick = 2 * hov - 1
        lines.append(
            f"- UC64 comparison: measured hover stick {stick:+.3f} vs. UC64's hand-flown "
            f"{UC64_HOVER_STICK:+.2f} (throttle {uc64:.3f}); difference "
            f"{stick - UC64_HOVER_STICK:+.3f} stick."
        )
    ac = m.get("airframe_check") or []
    lines += ["", "## Airframe check (ground-ramp hover vs. the expected band)", ""]
    for entry in ac or [{"status": "not recorded"}]:
        lines.append(
            f"- {entry.get('status')}: h0 {_fmt(entry.get('h0'))}, band {entry.get('band')}, "
            f"source {entry.get('source')}"
        )
    bat = fit.battery
    lines += ["", "## Battery", ""]
    if bat:
        lines.append(
            f"- Voltage {bat['volts_start']:.2f} V -> {bat['volts_end']:.2f} V (min "
            f"{bat['volts_min']:.2f} V); the level is reset (full battery) between passes."
        )
        if "thrust_per_volt" in bat:
            lines.append(
                f"- Thrust sensitivity: {bat['thrust_per_volt'] * 100:+.1f} % thrust per volt "
                "(measured / model ratio vs. voltage). Not voltage-parameterised in the profile."
            )
    else:
        lines.append("- No battery voltage in the telemetry.")
    lines += ["", "## Violations and aborted experiments", ""]
    viol = m.get("violations") or []
    aborted = {k: x for k, x in (m.get("experiments") or {}).items() if x.get("status") != "done"}
    if not viol and not aborted:
        lines.append("- None.")
    for x in viol:
        lines.append(
            f"- {x.get('name')} ({x.get('action')}): {x.get('detail')} in {x.get('experiment')}"
        )
    for k, x in aborted.items():
        lines.append(f"- `{k}`: {x.get('status')}")
    lines.append(f"- Session stop reason: {m.get('stop_reason')}")
    lines += [
        "",
        "## Default randomization (UC67)",
        "",
        "nominal +- max(3 sigma, 10 %), clamped at 0; multipliers 1 +- max(3 sigma_rel, 10 %).",
        "",
        "| parameter | min | max |",
        "|---|---|---|",
    ]
    for k, rng in data["randomization"].items():
        lines.append(f"| {k} | {_fmt(rng['min'])} | {_fmt(rng['max'])} |")
    if fit.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in fit.notes]
    signs = ", ".join(f"{a} {fit.params.signs[a]:+.0f}" for a in AXES)
    lines += ["", f"Stick -> body-rate signs measured: {signs}."]
    if plots:
        lines += ["", "## Plots", ""] + [f"![{p}]({p})" for p in plots]
    return "\n".join(lines) + "\n"


def write_report(
    fit: FitResult,
    out_dir: str | Path,
    *,
    drone_id: str,
    name: str | None = None,
    liftoff: dict[str, Any] | None = None,
    camera: dict[str, Any] | None = None,
    micro: bool = False,
) -> tuple[Path, Path]:
    """Write ``drone.yaml`` (validated), ``sysid.md`` and the PNGs into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = profile_data(fit, drone_id=drone_id, name=name, liftoff=liftoff, camera=camera)
    header = (
        f"Drone profile {drone_id}: identified by `drone-fly sysid fit` from session "
        f"{fit.session}.\nfit_quality: {fit.fit_quality}. Report: sysid.md. Do not hand-edit "
        "measured values; re-run the fit instead."
    )
    yaml_path = store.dump(data, out / store.PROFILE_FILE, header=header)
    plots = write_plots(fit, out)
    md = out / store.REPORT_FILE
    md.write_text(sysid_markdown(fit, data, plots, micro=micro), encoding="utf-8")
    return yaml_path, md
