# Meteor75 (analog): not identified

`drones/meteor75/drone.yaml` is **not** a measurement. It is the BetaFPV Meteor75 Pro *analog* that
the policies are trained on today, written in the profile format so that later work (UC67) can
build the simulator from profiles. Every value has a `ref` to its source in the code.

| profile field | value | source |
|---|---|---|
| `hover_throttle` | 0.4 | `gpu_prototype/hover_gpu.py`: `HOVER_THR = WEIGHT / MAX_THRUST = 1 / TW` |
| `thrust_curve` | 0 -> 0 g, 1 -> 2.5 g (linear) | `hover_gpu.py`: `MAX_THRUST = TW * WEIGHT`, `TW = 2.5` |
| `rates.*.rc_rate` | 1.1459 (229.18 deg/s), super 0, expo 0 | `hover_gpu.py`: `MAX_BODY_RATE = 4.0` rad/s, linear after `tanh` |
| `rates.*.tau_s` | 1/18 s | `hover_gpu.py`: `RATE_KP = 18` |
| drag, motor lag, delays | 0 | `hover_gpu.py` has none |
| `analog.*` | mass 0.032 kg, wheelbase 0.075 m, ... | `src/drone_fly/adapter/meteor75.py` |
| `analog.gravity_m_s2` | 9.81 | `hover_gpu.py` `G` (the pybullet adapter uses 9.8) |

The randomization ranges come from `gpu_prototype/gates_dr.py`:

| parameter | range | from |
|---|---|---|
| `thrust_scale` | 0.72 - 1.28 | `TW (1.8, 3.2) / 2.5` |
| `rates.*.tau_s` | 1/26 - 1/12 s | `RATE_KP (12, 26)`, inverted |
| `motor_lag_s` | 0 - 0.08 s | `LAG_TAU (0.02, 0.08)`, lower bound widened to contain the nominal 0 |
| `latency.control_to_gyro_s` | 0 - 0.04 s | `LAT` = 0..2 steps of `DT` = 0.02 s, on all four channels |
| `drag.quadratic_per_m` | 0 - 3.2 /m | `DRAG (0, 0.08)` force coefficient / `MASS` min 0.025 kg |

Caveats:

- **Hover 0.4, not 0.5.** The pybullet adapter hovers at about 0.5 because of its own mixer. The
  batched GPU plant, which is what the policies train on, hovers at 0.4.
- **The rate loop is a discrete Euler step.** `omega += RATE_KP * (cmd - omega) * DT` with
  `RATE_KP * DT = 0.36` per 50 Hz step. `tau_s = 1/RATE_KP` is its continuous-time equivalent. To
  reproduce `BatchedDrone` exactly, UC67 must keep the Euler form.
- **Mass randomization folds into drag.** Thrust is `TW * mass * G`, so mass cancels from the
  acceleration. It only scales drag (force / mass), which is why the profile has no mass range and
  the quadratic-drag range uses the smallest mass.
- **Latency applies to every channel**, throttle included, as `gates_dr.py` delays the whole action.

`tests/test_drones_meteor75.py` checks that this file agrees with both source files.
