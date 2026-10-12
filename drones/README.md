# Drone profiles

One folder per airframe. The folder name is the profile id.

| file | what |
|------|------|
| `drones/<id>/drone.yaml` | **the contract**: the airframe's dynamics, loaded and validated by `drone_fly.drones.load("<id>")` |
| `drones/<id>/sysid.md` | the report: how the values were obtained, their uncertainty, the held-out validation |
| `drones/<id>/*.png` | the report's fit and residual plots |

Raw measurement data is **not** committed. A sysid session writes telemetry, pad commands and a
`meta.json` to `artifacts/sysid/<id>/<session>/`, which is gitignored. Only the fitted `drone.yaml`,
`sysid.md` and PNGs belong here.

## Ids

`<source>-<class>-<model>`, kebab-case, for example `liftoff-gremlin-micro` or
`liftoff-5inch-vortex-pro-250`. The training analog has the short id `meteor75`.

| id | source | what |
|----|--------|------|
| `meteor75` | analog | the plant the policies train on today (`gpu_prototype/hover_gpu.py`), not measured |
| `liftoff-gremlin-micro` | liftoff-sysid | Liftoff "Test small drone" (FT Gremlin, ZetaFlight Acro), the UC-64 micro |
| `liftoff-5inch-vortex-pro-250` | liftoff-sysid | Liftoff's stock 5-inch "Vortex Pro 250" |

## Units and conventions

Everything is relative to hover, because Liftoff does not expose the drone's mass.

- **Throttle** is a fraction from 0 to 1. Liftoff's throttle stick `s` (-1..1) maps to `(s + 1) / 2`.
- **Thrust curve**: throttle fraction -> body-z specific force in **g**. Hover is 1.0 g by
  definition, so `hover_throttle` must sit on the curve at 1.0 g (within 0.05 g).
- **Rates** are Betaflight rates per axis (`rc_rate` 1.0 = 100 in the configurator, `super_rate`,
  `expo`), followed by a first-order response `tau_s` and a pure delay `delay_s`.
- **Drag** is an acceleration: `a = -linear_per_s * v - quadratic_per_m * |v| * v` (m/s^2), either
  isotropic (a number) or per body axis (`[x, y, z]`).
- **Delays** are in seconds. `latency.input_s` runs from the pad command to Liftoff's echoed input.
  `latency.control_to_gyro_s` is the end-to-end delay from the pad to the gyro, equal to
  `input_s + delay_s`.
- **Provenance**: every measured value is `{value, session, residual, sigma}`; a value taken from
  a reference is `{value, ref}`.
- **randomization**: a `{min, max}` range per parameter, for the training randomization (UC67). The
  defaults are `nominal +- max(3 sigma, 10 %)`.

The schema is documented field by field in `src/drone_fly/drones/schema.py`, and the decisions
behind it in [ADR 0005](../docs/adr/0005-drone-profiles-and-sysid.md).

## Measuring a new drone

See "Measure a new drone" in
[docs/harness/liftoff-live-session.md](../docs/harness/liftoff-live-session.md#measure-a-new-drone).
In short:

```powershell
.venv-cuda\Scripts\drone-fly.exe sysid liftoff --drone <id> --liftoff-name "<Liftoff drone name>" --map "<map>"
.venv-cuda\Scripts\drone-fly.exe sysid fit artifacts\sysid\<id>\<session> --drone <id>
```

Then check `fit_quality` in `drones/<id>/drone.yaml` and commit only `drones/<id>/`.

To try the pipeline without the game, run it on a simulated plant with known parameters:
`sysid liftoff --drone <id> --plant synthetic`.
