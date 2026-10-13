"""Drone profile schema (``drones/<id>/drone.yaml``): frozen dataclasses + validation.

A profile describes one airframe **relative to hover**, because mass is not observable in
Liftoff: the thrust curve maps throttle fraction (0..1; Liftoff stick ``s`` -> ``(s + 1) / 2``)
to body-z specific force in g (hover = 1.0 g), drag is an acceleration
(``a = -k1 * v - k2 * |v| * v``, m/s^2), and every time constant is in seconds.

File layout (``schema_version: 1``)::

    schema_version: 1
    id: liftoff-gremlin-micro          # == folder name, kebab-case <source>-<class>-<model>
    name: ...
    source: analog | liftoff-sysid | manual
    fit_quality: good | poor           # required for liftoff-sysid
    notes: ...
    liftoff: {drone_name, frame, flight_controller, map, settings, battery}   # liftoff-sysid
    analog: {mass_kg, thrust_to_weight, gravity_m_s2, ...}                   # informational
    nominal:
      hover_throttle: 0.47
      thrust_curve: {throttle: [0, ..., 1], accel_g: [0, ..., 2.9]}
      rates: {roll|pitch|yaw: {rc_rate, super_rate, expo, tau_s, delay_s}}
      drag: {linear_per_s, quadratic_per_m}       # scalar or body-axis [x, y, z]
      motor_lag_s: 0.03
      latency: {input_s, control_to_gyro_s}
      camera: {hfov_deg, uptilt_deg} | null
    randomization:
      <path>: {min, max}                          # see RANDOMIZABLE
    power:                                        # optional battery + thrust-at-speed model
      cells: 3                                    # (UC68; see sim/power.py and PowerSpec)
      ocv: {charge: [..], volts: [..]}            # pack open-circuit volts vs charge (1 = full)
      sag: {rate_v_per_s, exponent}               # persistent sag s' = rate E^exponent
      ir: {v, exponent}                           # V = OCV(q) - s - v E^exponent
      drain: {idle_per_s, load_per_s, exponent}   # dq/dt = -(idle + load E^exponent), E = effort
      thrust_voltage_exponent: 2.0                # full-stick thrust x (V / V_full)^n
      inflow: {coeff, v_ref_mps, min_factor, effort_exponent}  # thrust x (1 - coeff v_ax / v_ref);
                                                  # effort E = T inflow^-effort_exponent
      cutoff_v_per_cell: 2.7
      randomization: {drain_scale|sag_scale|inflow_scale|initial_charge: {min, max}}
      fit: {...}                                  # provenance of the fit (free-form)

Provenance: any nominal leaf may be a bare value or ``{value, session, residual, sigma}`` /
``{value, ref}``; a block (``thrust_curve``, ``rates.roll``, ``drag``, ...) may carry the same
keys next to its fields, covering every leaf inside it. For ``source: liftoff-sysid`` every
nominal leaf must be covered by (``session`` and ``residual``) or ``ref``.

Every validation error is a :class:`DroneProfileError` naming the dotted field path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1
SOURCES = ("analog", "liftoff-sysid", "manual")
FIT_QUALITIES = ("good", "poor")
AXES = ("roll", "pitch", "yaw")
ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
PROVENANCE_KEYS = ("session", "residual", "sigma", "ref")
TOP_KEYS = (
    "schema_version",
    "id",
    "name",
    "source",
    "fit_quality",
    "notes",
    "liftoff",
    "analog",
    "nominal",
    "randomization",
    "power",
)
NOMINAL_KEYS = (
    "hover_throttle",
    "thrust_curve",
    "rates",
    "drag",
    "motor_lag_s",
    "latency",
    "camera",
)
RATE_KEYS = ("rc_rate", "super_rate", "expo", "tau_s", "delay_s")
#: Randomizable parameters (UC67) -> the nominal path they scale or range over. ``None`` means a
#: multiplier whose nominal is 1.0.
RANDOMIZABLE: dict[str, str | None] = {
    "thrust_scale": None,
    **{f"rates.{a}.rate_scale": None for a in AXES},
    **{f"rates.{a}.tau_s": f"rates.{a}.tau_s" for a in AXES},
    **{f"rates.{a}.delay_s": f"rates.{a}.delay_s" for a in AXES},
    "drag.linear_per_s": "drag.linear_per_s",
    "drag.quadratic_per_m": "drag.quadratic_per_m",
    "motor_lag_s": "motor_lag_s",
    "latency.control_to_gyro_s": "latency.control_to_gyro_s",
}
#: Hover must sit on the curve: ``thrust_g(hover_throttle)`` within this of 1.0 g.
HOVER_TOLERANCE_G = 0.05


class DroneProfileError(ValueError):
    """An invalid drone profile; ``field`` is the dotted path of the offending value."""

    def __init__(self, message: str, field: str | None = None, file: str | None = None) -> None:
        self.field = field
        self.file = file
        where = ": ".join(p for p in (file, field) if p)
        super().__init__(f"{where}: {message}" if where else message)


# ----------------------------------------------------------------------------- records


@dataclass(frozen=True)
class ThrustCurve:
    """Piecewise-linear throttle fraction -> body-z specific force (g)."""

    throttle: tuple[float, ...]
    accel_g: tuple[float, ...]

    def g_at(self, throttle: float) -> float:
        import numpy as np

        return float(np.interp(float(throttle), self.throttle, self.accel_g))

    def throttle_for(self, accel_g: float) -> float:
        """Inverse of :meth:`g_at` (the curve is strictly increasing); clamps to [0, 1]."""
        import numpy as np

        return float(np.interp(float(accel_g), self.accel_g, self.throttle))


@dataclass(frozen=True)
class AxisRates:
    rc_rate: float
    super_rate: float
    expo: float
    tau_s: float
    delay_s: float


@dataclass(frozen=True)
class Drag:
    """``linear_per_s`` / ``quadratic_per_m``: a float (isotropic) or a body-axis ``(x, y, z)``."""

    linear_per_s: float | tuple[float, float, float]
    quadratic_per_m: float | tuple[float, float, float]


@dataclass(frozen=True)
class Latency:
    input_s: float
    control_to_gyro_s: float


@dataclass(frozen=True)
class Camera:
    hfov_deg: float
    uptilt_deg: float


@dataclass(frozen=True)
class Range:
    min: float
    max: float


@dataclass(frozen=True)
class PowerSpec:
    """The optional ``power`` block (battery + thrust at speed; :mod:`drone_fly.sim.power`)."""

    cells: int
    ocv_charge: tuple[float, ...]
    ocv_volts: tuple[float, ...]
    sag_rate_v_per_s: float
    sag_exponent: float
    ir_v: float
    ir_exponent: float
    drain_idle_per_s: float
    drain_load_per_s: float
    drain_exponent: float
    thrust_voltage_exponent: float
    inflow_coeff: float
    inflow_v_ref: float
    inflow_min_factor: float
    effort_inflow_exponent: float
    cutoff_v_per_cell: float
    randomization: dict[str, Range] = field(default_factory=dict)
    fit: dict[str, Any] | None = None


@dataclass(frozen=True)
class DroneProfile:
    schema_version: int
    id: str
    name: str
    source: str
    hover_throttle: float
    thrust_curve: ThrustCurve
    rates: dict[str, AxisRates]
    drag: Drag
    motor_lag_s: float
    latency: Latency
    camera: Camera | None
    randomization: dict[str, Range]
    fit_quality: str | None = None
    notes: str | None = None
    liftoff: dict[str, Any] | None = None
    analog: dict[str, Any] | None = None
    #: Dotted nominal path (``nominal.rates.roll.tau_s``) -> provenance dict.
    provenance_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    path: str | None = None
    power: PowerSpec | None = None

    def thrust_g(self, throttle: float) -> float:
        return self.thrust_curve.g_at(throttle)

    def throttle_for_g(self, accel_g: float) -> float:
        return self.thrust_curve.throttle_for(accel_g)

    @property
    def max_g(self) -> float:
        return float(self.thrust_curve.accel_g[-1])

    def rate_profile(self, axis: str) -> Any:
        """The axis's Betaflight rates as a ``gamepad.RateProfile``."""
        from drone_fly.harness.liftoff.gamepad import RateProfile

        r = self.rates[axis]
        return RateProfile(rc_rate=r.rc_rate, super_rate=r.super_rate, expo=r.expo)

    def provenance(self, path: str) -> dict[str, Any] | None:
        """Provenance of a nominal leaf (``"rates.roll.tau_s"`` or ``"nominal.rates..."``)."""
        key = path if path.startswith("nominal.") else f"nominal.{path}"
        while key:
            if key in self.provenance_map:
                return self.provenance_map[key]
            key = key.rpartition(".")[0]
        return None


# ----------------------------------------------------------------------------- parsing


class _Parser:
    def __init__(self, file: str | None) -> None:
        self.file = file
        self.prov: dict[str, dict[str, Any]] = {}

    def err(self, field: str, message: str) -> DroneProfileError:
        return DroneProfileError(message, field, self.file)

    def mapping(self, value: Any, field: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise self.err(field, f"must be a mapping, got {type(value).__name__}")
        return value

    def keys(self, data: dict[str, Any], field: str, allowed: tuple[str, ...]) -> None:
        unknown = sorted(set(data) - set(allowed) - set(PROVENANCE_KEYS))
        if unknown:
            prefix = f"{field}." if field else ""
            raise self.err(f"{prefix}{unknown[0]}", f"unknown key (allowed: {list(allowed)})")

    def block_provenance(self, data: dict[str, Any], field: str) -> None:
        prov = {k: data[k] for k in PROVENANCE_KEYS if k in data}
        if prov:
            self.prov[field] = prov

    def leaf(self, value: Any, field: str) -> Any:
        """Unwrap ``{value, ...provenance}`` -> value, recording the provenance."""
        if isinstance(value, dict):
            if "value" not in value:
                raise self.err(field, "a wrapped value needs a 'value' key")
            extra = sorted(set(value) - {"value", *PROVENANCE_KEYS})
            if extra:
                raise self.err(f"{field}.{extra[0]}", "unknown provenance key")
            self.block_provenance(value, field)
            return value["value"]
        return value

    def number(
        self,
        value: Any,
        field: str,
        *,
        lo: float | None = None,
        hi: float | None = None,
        lo_open: bool = False,
        hi_open: bool = False,
    ) -> float:
        v = self.leaf(value, field)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise self.err(field, f"must be a number, got {v!r}")
        v = float(v)
        if v != v:  # NaN
            raise self.err(field, "must not be NaN")
        if lo is not None and (v <= lo if lo_open else v < lo):
            raise self.err(field, f"{v} out of range: must be {'>' if lo_open else '>='} {lo}")
        if hi is not None and (v >= hi if hi_open else v > hi):
            raise self.err(field, f"{v} out of range: must be {'<' if hi_open else '<='} {hi}")
        return v

    def required(self, data: dict[str, Any], key: str, field: str) -> Any:
        if key not in data or data[key] is None:
            raise self.err(field, "missing required field")
        return data[key]


def _covered(prov: dict[str, dict[str, Any]], path: str) -> bool:
    key = path
    while key:
        p = prov.get(key)
        if p is not None and (("session" in p and "residual" in p) or "ref" in p):
            return True
        key = key.rpartition(".")[0]
    return False


def parse_profile(
    data: Any, *, file: str | None = None, expected_id: str | None = None
) -> DroneProfile:
    """Validate a loaded ``drone.yaml`` mapping and build a :class:`DroneProfile`."""
    p = _Parser(file)
    top = p.mapping(data, "<root>")
    p.keys(top, "", TOP_KEYS)

    version = p.required(top, "schema_version", "schema_version")
    if version != SCHEMA_VERSION:
        raise p.err("schema_version", f"unsupported {version!r} (supported: {SCHEMA_VERSION})")
    pid = p.required(top, "id", "id")
    if not isinstance(pid, str) or not ID_RE.match(pid):
        raise p.err("id", f"{pid!r} must be kebab-case (<source>-<class>-<model>)")
    if expected_id is not None and pid != expected_id:
        raise p.err("id", f"{pid!r} does not match its folder {expected_id!r}")
    name = p.required(top, "name", "name")
    if not isinstance(name, str):
        raise p.err("name", "must be a string")
    source = p.required(top, "source", "source")
    if source not in SOURCES:
        raise p.err("source", f"{source!r} must be one of {list(SOURCES)}")
    fit_quality = top.get("fit_quality")
    if source == "liftoff-sysid" and fit_quality is None:
        raise p.err("fit_quality", "missing required field (required for liftoff-sysid)")
    if fit_quality is not None and fit_quality not in FIT_QUALITIES:
        raise p.err("fit_quality", f"{fit_quality!r} must be one of {list(FIT_QUALITIES)}")
    liftoff = top.get("liftoff")
    if liftoff is not None:
        p.mapping(liftoff, "liftoff")
    if source == "liftoff-sysid":
        if liftoff is None:
            raise p.err("liftoff", "missing required field (required for liftoff-sysid)")
        dn = liftoff.get("drone_name")
        if not isinstance(dn, str) or not dn:
            raise p.err("liftoff.drone_name", "missing required field (exact Liftoff drone name)")
    analog = top.get("analog")
    if analog is not None:
        p.mapping(analog, "analog")

    nom = p.mapping(p.required(top, "nominal", "nominal"), "nominal")
    p.keys(nom, "nominal", NOMINAL_KEYS)
    p.block_provenance(nom, "nominal")

    hover = p.number(
        p.required(nom, "hover_throttle", "nominal.hover_throttle"),
        "nominal.hover_throttle",
        lo=0.0,
        hi=1.0,
        lo_open=True,
        hi_open=True,
    )

    f = "nominal.thrust_curve"
    tc = p.mapping(p.required(nom, "thrust_curve", f), f)
    p.keys(tc, f, ("throttle", "accel_g"))
    p.block_provenance(tc, f)
    knots_t = p.required(tc, "throttle", f"{f}.throttle")
    knots_g = p.required(tc, "accel_g", f"{f}.accel_g")
    for key, seq in (("throttle", knots_t), ("accel_g", knots_g)):
        if not isinstance(seq, list) or len(seq) < 2:
            raise p.err(f"{f}.{key}", "must be a list of at least 2 numbers")
    if len(knots_t) != len(knots_g):
        raise p.err(f"{f}.accel_g", f"needs {len(knots_t)} values (one per throttle knot)")
    th = [p.number(v, f"{f}.throttle[{i}]", lo=0.0, hi=1.0) for i, v in enumerate(knots_t)]
    ag = [p.number(v, f"{f}.accel_g[{i}]", lo=0.0) for i, v in enumerate(knots_g)]
    if th[0] != 0.0 or th[-1] != 1.0:
        raise p.err(f"{f}.throttle", "must start at 0 and end at 1")
    for i in range(1, len(th)):
        if th[i] <= th[i - 1]:
            raise p.err(f"{f}.throttle[{i}]", "throttle knots must be strictly increasing")
        if ag[i] <= ag[i - 1]:
            raise p.err(f"{f}.accel_g[{i}]", "accel_g must be strictly increasing (invertible)")
    curve = ThrustCurve(tuple(th), tuple(ag))
    at_hover = curve.g_at(hover)
    if abs(at_hover - 1.0) > HOVER_TOLERANCE_G:
        raise p.err(
            "nominal.hover_throttle",
            f"thrust_curve gives {at_hover:.3f} g at hover {hover:.3f}; must be 1.0 g "
            f"+- {HOVER_TOLERANCE_G}",
        )

    f = "nominal.rates"
    rates_in = p.mapping(p.required(nom, "rates", f), f)
    p.keys(rates_in, f, AXES)
    p.block_provenance(rates_in, f)
    rates: dict[str, AxisRates] = {}
    for axis in AXES:
        fa = f"{f}.{axis}"
        r = p.mapping(p.required(rates_in, axis, fa), fa)
        p.keys(r, fa, RATE_KEYS)
        p.block_provenance(r, fa)

        def req(key: str, r: dict[str, Any] = r, fa: str = fa) -> Any:
            return p.required(r, key, f"{fa}.{key}")

        rates[axis] = AxisRates(
            rc_rate=p.number(req("rc_rate"), f"{fa}.rc_rate", lo=0.0, lo_open=True),
            super_rate=p.number(
                req("super_rate"), f"{fa}.super_rate", lo=0.0, hi=1.0, hi_open=True
            ),
            expo=p.number(req("expo"), f"{fa}.expo", lo=0.0, hi=1.0),
            tau_s=p.number(req("tau_s"), f"{fa}.tau_s", lo=0.0, hi=1.0, lo_open=True),
            delay_s=p.number(req("delay_s"), f"{fa}.delay_s", lo=0.0, hi=0.5),
        )

    f = "nominal.drag"
    dr = p.mapping(p.required(nom, "drag", f), f)
    p.keys(dr, f, ("linear_per_s", "quadratic_per_m"))
    p.block_provenance(dr, f)

    def drag_value(key: str) -> float | tuple[float, float, float]:
        fk = f"{f}.{key}"
        v = p.leaf(p.required(dr, key, fk), fk)
        if isinstance(v, list):
            if len(v) != 3:
                raise p.err(fk, "must be a number or a body-axis [x, y, z] list")
            x, y, z = (p.number(c, f"{fk}[{i}]", lo=0.0) for i, c in enumerate(v))
            return (x, y, z)
        return p.number(v, fk, lo=0.0)

    drag = Drag(drag_value("linear_per_s"), drag_value("quadratic_per_m"))
    motor_lag = p.number(
        p.required(nom, "motor_lag_s", "nominal.motor_lag_s"), "nominal.motor_lag_s", lo=0.0, hi=0.5
    )

    f = "nominal.latency"
    la = p.mapping(p.required(nom, "latency", f), f)
    p.keys(la, f, ("input_s", "control_to_gyro_s"))
    p.block_provenance(la, f)
    latency = Latency(
        input_s=p.number(p.required(la, "input_s", f"{f}.input_s"), f"{f}.input_s", lo=0.0, hi=0.5),
        control_to_gyro_s=p.number(
            p.required(la, "control_to_gyro_s", f"{f}.control_to_gyro_s"),
            f"{f}.control_to_gyro_s",
            lo=0.0,
            hi=0.5,
        ),
    )

    camera = None
    cam = nom.get("camera")
    if cam is not None:
        f = "nominal.camera"
        cam = p.mapping(cam, f)
        p.keys(cam, f, ("hfov_deg", "uptilt_deg"))
        p.block_provenance(cam, f)
        camera = Camera(
            hfov_deg=p.number(
                p.required(cam, "hfov_deg", f"{f}.hfov_deg"),
                f"{f}.hfov_deg",
                lo=0.0,
                hi=180.0,
                lo_open=True,
                hi_open=True,
            ),
            uptilt_deg=p.number(
                p.required(cam, "uptilt_deg", f"{f}.uptilt_deg"), f"{f}.uptilt_deg", lo=-90, hi=90
            ),
        )

    nominal_values: dict[str, Any] = {
        "motor_lag_s": motor_lag,
        "latency.control_to_gyro_s": latency.control_to_gyro_s,
        "drag.linear_per_s": drag.linear_per_s,
        "drag.quadratic_per_m": drag.quadratic_per_m,
        **{f"rates.{a}.tau_s": rates[a].tau_s for a in AXES},
        **{f"rates.{a}.delay_s": rates[a].delay_s for a in AXES},
    }
    randomization: dict[str, Range] = {}
    rnd = top.get("randomization") or {}
    p.mapping(rnd, "randomization")
    for key, spec in rnd.items():
        fr = f"randomization.{key}"
        if key not in RANDOMIZABLE:
            raise p.err(fr, f"not a randomizable parameter (allowed: {sorted(RANDOMIZABLE)})")
        spec = p.mapping(spec, fr)
        extra = sorted(set(spec) - {"min", "max"})
        if extra:
            raise p.err(f"{fr}.{extra[0]}", "unknown key (allowed: min, max)")
        lo = p.number(p.required(spec, "min", f"{fr}.min"), f"{fr}.min", lo=0.0)
        hi = p.number(p.required(spec, "max", f"{fr}.max"), f"{fr}.max", lo=0.0)
        if hi < lo:
            raise p.err(f"{fr}.max", f"max {hi} < min {lo}")
        target = RANDOMIZABLE[key]
        nominal = 1.0 if target is None else nominal_values[target]
        values = nominal if isinstance(nominal, tuple) else (nominal,)
        for v in values:
            if not lo - 1e-12 <= v <= hi + 1e-12:
                raise p.err(fr, f"range [{lo}, {hi}] must contain the nominal {v}")
        randomization[key] = Range(lo, hi)

    if source == "liftoff-sysid":
        leaves = ["nominal.hover_throttle", "nominal.thrust_curve", "nominal.motor_lag_s"]
        leaves += [f"nominal.rates.{a}.{k}" for a in AXES for k in RATE_KEYS]
        leaves += ["nominal.drag.linear_per_s", "nominal.drag.quadratic_per_m"]
        leaves += ["nominal.latency.input_s", "nominal.latency.control_to_gyro_s"]
        if camera is not None:
            leaves += ["nominal.camera.hfov_deg", "nominal.camera.uptilt_deg"]
        for leaf in leaves:
            if not _covered(p.prov, leaf):
                raise p.err(leaf, "measured value needs provenance: (session and residual) or ref")

    power = _parse_power(p, top["power"]) if top.get("power") is not None else None
    notes = top.get("notes")
    return DroneProfile(
        schema_version=SCHEMA_VERSION,
        id=pid,
        name=name,
        source=source,
        hover_throttle=hover,
        thrust_curve=curve,
        rates=rates,
        drag=drag,
        motor_lag_s=motor_lag,
        latency=latency,
        camera=camera,
        randomization=randomization,
        fit_quality=fit_quality,
        notes=None if notes is None else str(notes),
        liftoff=liftoff,
        analog=analog,
        provenance_map=dict(p.prov),
        path=file,
        power=power,
    )


def _parse_power(p: _Parser, data: Any) -> PowerSpec:
    from drone_fly.sim.power import POWER_RANDOMIZABLE

    f = "power"
    d = p.mapping(data, f)
    p.keys(d, f, ("cells", "ocv", "sag", "ir", "drain", "thrust_voltage_exponent", "inflow",
                  "cutoff_v_per_cell", "randomization", "fit"))  # fmt: skip

    def block(key: str, keys: tuple[str, ...]) -> dict[str, Any]:
        b = p.mapping(p.required(d, key, f"{f}.{key}"), f"{f}.{key}")
        p.keys(b, f"{f}.{key}", keys)
        return b

    def num(b: dict[str, Any], key: str, field: str, **kw: Any) -> float:
        return p.number(p.required(b, key, field), field, **kw)

    cells = int(num(d, "cells", f"{f}.cells", lo=1.0))
    ocv = block("ocv", ("charge", "volts"))
    q = [float(x) for x in p.required(ocv, "charge", f"{f}.ocv.charge")]
    v = [float(x) for x in p.required(ocv, "volts", f"{f}.ocv.volts")]
    if len(q) != len(v) or len(q) < 2 or any(b <= a for a, b in zip(q, q[1:], strict=False)):
        raise p.err(f"{f}.ocv", "charge must be increasing with one volts value per knot")
    sag = block("sag", ("rate_v_per_s", "exponent"))
    ir = block("ir", ("v", "exponent"))
    drain = block("drain", ("idle_per_s", "load_per_s", "exponent"))
    inflow = block("inflow", ("coeff", "v_ref_mps", "min_factor", "effort_exponent"))
    rnd: dict[str, Range] = {}
    for key, spec in (d.get("randomization") or {}).items():
        fr = f"{f}.randomization.{key}"
        if key not in POWER_RANDOMIZABLE:
            raise p.err(fr, f"not a power randomization key (allowed: {list(POWER_RANDOMIZABLE)})")
        spec = p.mapping(spec, fr)
        lo, hi = num(spec, "min", f"{fr}.min", lo=0.0), num(spec, "max", f"{fr}.max", lo=0.0)
        if hi < lo or not lo <= 1.0 <= hi:
            raise p.err(fr, f"range [{lo}, {hi}] must contain the nominal 1.0")
        rnd[key] = Range(lo, hi)
    return PowerSpec(
        cells=cells,
        ocv_charge=tuple(q),
        ocv_volts=tuple(v),
        sag_rate_v_per_s=num(sag, "rate_v_per_s", f"{f}.sag.rate_v_per_s", lo=0.0),
        sag_exponent=num(sag, "exponent", f"{f}.sag.exponent", lo=0.0),
        ir_v=num(ir, "v", f"{f}.ir.v", lo=0.0),
        ir_exponent=num(ir, "exponent", f"{f}.ir.exponent", lo=0.0),
        drain_idle_per_s=num(drain, "idle_per_s", f"{f}.drain.idle_per_s", lo=0.0),
        drain_load_per_s=num(drain, "load_per_s", f"{f}.drain.load_per_s", lo=0.0),
        drain_exponent=num(drain, "exponent", f"{f}.drain.exponent", lo=0.0),
        thrust_voltage_exponent=num(
            d, "thrust_voltage_exponent", f"{f}.thrust_voltage_exponent", lo=0.0
        ),  # fmt: skip
        inflow_coeff=num(inflow, "coeff", f"{f}.inflow.coeff", lo=0.0),
        inflow_v_ref=num(inflow, "v_ref_mps", f"{f}.inflow.v_ref_mps", lo=0.0, lo_open=True),
        inflow_min_factor=num(inflow, "min_factor", f"{f}.inflow.min_factor", lo=0.0, hi=1.0),
        effort_inflow_exponent=num(
            inflow, "effort_exponent", f"{f}.inflow.effort_exponent", lo=0.0, hi=1.0
        ),  # fmt: skip
        cutoff_v_per_cell=num(d, "cutoff_v_per_cell", f"{f}.cutoff_v_per_cell", lo=0.0),
        randomization=rnd,
        fit=dict(d["fit"]) if isinstance(d.get("fit"), dict) else None,
    )
