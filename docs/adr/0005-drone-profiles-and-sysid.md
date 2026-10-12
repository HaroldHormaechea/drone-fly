# ADR 0005: Drone profiles and system identification of Liftoff drones

- **Status:** accepted (UC-66)
- **Date:** 2026-10-12
- **Builds on:** [ADR 0003](0003-real-liftoff-calibration.md) (telemetry convention, pad, rates),
  [ADR 0002](0002-live-harness-layering.md) (injectable parts)
- **Code:** `src/drone_fly/drones/` (schema + loader), `src/drone_fly/sysid/` (`model`, `schedule`,
  `controller`, `safety`, `recorder`, `session`, `fit`, `report`, `synthetic`),
  `src/drone_fly/harness/liftoff/anticheat.py`, `harness/liftoff/ui_driver.py` (drone selection)
- **Data:** `drones/<id>/` (committed), `artifacts/sysid/<id>/<session>/` (gitignored)

## Context

In the live Liftoff runs the policy saturates every stick from the first tick and tumbles. The
policies train on a hard-coded Meteor75 analog in `gpu_prototype/` (T/W 2.5, linear 4 rad/s rates,
hover at 0.4 throttle), while Liftoff flies its own airframes through Betaflight rate curves and
PIDs. Before the simulator can be matched to Liftoff (UC67), we need to know what Liftoff's drones
do, in a form the simulator can consume.

## Decision

### One profile per airframe, relative to hover

Each airframe gets a folder `drones/<id>/` with `drone.yaml` (the contract), `sysid.md` (the report)
and PNGs. The id is `<source>-<class>-<model>` in kebab-case and equals the folder name. Raw data is
gitignored under `artifacts/sysid/`.

Liftoff does not expose mass, and only accelerations are observable. So the profile has no mass and
no forces. Thrust is a curve from throttle fraction to body-z specific force in g, with hover at 1.0 g;
drag is an acceleration (`a = -k1 v - k2 |v| v`). A mass appears only in the informational `analog`
block of the Meteor75. The curve must be strictly increasing so UC67 can invert it.

Rates are stored as Betaflight rates (rc rate, super rate, expo) plus a first-order `tau_s` and a pure
`delay_s` per axis. Two more delays are kept separate because they are separately identifiable:
`input_s` (pad to Liftoff's echoed input) and `control_to_gyro_s` (pad to gyro, `input_s + delay_s`).

Every measured value carries its provenance (`session`, `residual`, `sigma`), or a `ref` when it is
taken from somewhere else. `drone_fly.drones.load` rejects a `liftoff-sysid` profile in which any
measured value lacks it. Every validation error names the dotted field path.

### The Meteor75 port

`drones/meteor75/drone.yaml` reproduces `hover_gpu.py` and `adapter/meteor75.py` without changing a
value; a test reads both sources with `ast` (they bind CUDA at import) and checks the agreement. The
port records four findings:

- Hover is **0.4** (`1/TW`) in the batched GPU plant. The "hover about 0.5" in the use case is the
  pybullet adapter's mixer and does not apply to the trained plant.
- Gravity is **9.81** in `hover_gpu.py`. The 9.8 in `meteor75.py` is the pybullet aviary's.
- `gates_dr.py` randomizes mass, but mass cancels out of thrust (`TW * mass * G`). It only scales
  drag, so its effect goes into the quadratic-drag range (`0 .. DRAG_max / MASS_min = 3.2`) and the
  profile has no mass range.
- The rate loop is a discrete Euler step (`RATE_KP * DT = 0.36`). `tau_s = 1/18` is its
  continuous-time equivalent, and UC67 must keep the Euler form to reproduce `BatchedDrone`. The
  randomized action delay applies to **all four** channels, throttle included.

### Measuring: a stabilized, scripted session

`drone-fly sysid liftoff` flies the drone itself through the long-lived virtual pad. It uses a simple
controller on telemetry: attitude to body-rate commands through the configured Betaflight rates, a
position loop, and an altitude loop in g around a hover estimate `h0`. `h0` comes from a slow
throttle ramp on the ground (the throttle at lift-off). Scaling by `h0` lets the same gains fly a micro
and a 5-inch.

The schedule is seeded and fixed. It has P passes (default 3), each containing every experiment type
in its own seeded order:

- throttle steps up and down, and a chirp;
- short full-throttle bursts;
- roll, pitch and yaw doublets at amplitudes 0.2 to 1.0 (full stick);
- coasts at three tilts, alternating heading.

Steps and bursts are cut short when they would change the vertical speed by more than 4 m/s, so a
T/W 7 drone does not leave the flight volume. Each experiment starts from a settled hover and its
open-loop window is followed by a recovery. The session lands, disarms and resets the level
between passes for a full battery.

Safety follows `harness run`: F12, stale telemetry, focus loss and Ctrl+C fail safe with an
RPM-confirmed disarm, and the pad watchdog cuts the throttle if the process dies. On top of that:

- An altitude floor and ceiling and a tilt bound apply outside the experiment window.
- A hard floor, a horizontal geofence and crash detection apply always.
- A violation aborts the experiment and re-stabilizes, or lands if that fails. Violations are
  recorded and never crash the run.
- A crash ends the session cleanly, and `--resume` continues from the next experiment in a new
  segment.

Two guards were added for unattended runs:

- **Anti-cheat refusal.** No input is sent while League of Legends, the Riot client or Vanguard is
  running: before a live session, before a drone selection and at every level reset.
- **Airframe guard.** The ground-ramp `h0` must fall inside the expected hover band, taken from
  `--expect-hover`, the existing profile (±0.15) or `ui_driver.DRONE_SLOTS`. Otherwise the session
  lands, disarms and stops with `drone_mismatch` before the first experiment. The guard is coarse:
  it catches a micro flown instead of a 5-inch, not two similar 5-inch builds.

### Fitting with a time split

`sysid fit` uses the fit passes only and validates on the last pass (a time split):

1. **Clocks.** Telemetry time is mapped to the session clock with a lower-envelope line fit, so
   network jitter does not bias the delays.
2. **`input_s`.** From the echoed sticks, by error minimisation on a 1 ms grid. When commands and
   telemetry are phase-locked the minimum is flat; the middle of the flat part is used.
3. **Rates.** For each pure delay on a 1 ms grid, bounded least squares on the output error gives
   `rc_rate`, `super_rate`, `expo` and `tau_s`. If `expo` is ill-conditioned (sigma > 0.15), it is
   fixed at 0 and the report says so.
4. **Drag.** Non-negative least squares on body x/y specific force. Thrust acts only along body z,
   so those axes see only drag.
5. **Thrust knots and motor lag.** For each lag on a grid, NNLS on knot increments, which keeps the
   curve monotone with 0 g at 0. The model is passed through the same Savitzky-Golay derivative as
   the measurement. Without that, the filter's smoothing would show up as extra motor lag.

The report gives one-step and 1 s open-loop errors on the held-out pass. `fit_quality: good` requires
the median 1 s rate error to be at most 15 % of the commanded rate, and the vertical acceleration
error at most 0.15 g (AC8). Otherwise the profile is still written, marked `poor`, and the report
says "NOT MET".

Default randomization ranges are nominal ± max(3σ, 10 %), clamped at 0.

### Hermetic tests on a synthetic plant

`sysid/synthetic.py` is a plant with known parameters that implements the same signal chain: full
quaternion, 500 Hz, ground contact, arm logic, noisy float32 telemetry in Liftoff's Unity frame, and
receive jitter. The real sequencer drives it through the real `VgamepadActionSink`, so the recorder,
the CSV round trip and the fitter are exercised end to end. `--plant synthetic` runs the same thing
as a dry run.

## Consequences

- UC67 can build the simulator from `drones/<id>/drone.yaml`. The training code is not switched over
  here.
- The model is a deliberate simplification of Betaflight in Liftoff: first-order rate response, a
  scalar motor lag and isotropic drag. The held-out metrics say how well it holds for each drone.
- Fitting on synthetic data is partly circular, because the model that generates the data is the
  model being fitted. Noise, jitter, the real sequencer and the CSV round trip reduce this. The real
  model mismatch shows only in the held-out metrics of live sessions.
- Battery sag: voltage is recorded, the level is reset between passes, and the report gives the
  thrust sensitivity per volt. Thrust is not parameterised by voltage.
- Selecting a drone through the menus is automated, but it depends on screen positions measured on
  Liftoff's current UI (`ui_driver.DRONE_LIST`, `DRONE_MENU_PATH`). The selection is verified by
  pixels and, in flight, by the airframe guard.
