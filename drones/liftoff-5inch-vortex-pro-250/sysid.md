# System identification: Liftoff Vortex Pro 250 (stock 5-inch) (`liftoff-5inch-vortex-pro-250`)

- Session: `vortex-full` (2026-10-12T16:23:50Z), plant: liftoff, seed 0
- Liftoff drone: Vortex Pro 250; map: Liftoff Arena (free flight, no track)
- Settings: preset `liftoff`; schedule 93 experiments in 3 passes (last pass held out)
- Git: `337ffcac103069cbd0b50765d06788307a9df58c`

**fit_quality: poor.** AC8 NOT MET: held-out 1 s open-loop error 4.2 % of the commanded rate (limit 15 %) and 0.153 g vertical (limit 0.15 g); the profile is written with fit_quality: poor.


## Fitted values

| parameter | value | sigma | fit residual |
|---|---|---|---|
| latency.input_s | 0.009 | 1.181e-05 | 0.0236 |
| rates.roll.rc_rate | 0.9713 | 0.04051 | 1.655 |
| rates.roll.super_rate | 0.8233 | 0.009019 | 1.655 |
| rates.roll.expo | 0 | - | 1.655 |
| rates.roll.tau_s | 0.01168 | 0.0007694 | 1.655 |
| rates.roll.delay_s | 0.003439 | 0.0004225 | 1.655 |
| rates.pitch.rc_rate | 0.929 | 0.03387 | 1.369 |
| rates.pitch.super_rate | 0.8241 | 0.007747 | 1.369 |
| rates.pitch.expo | 0 | - | 1.369 |
| rates.pitch.tau_s | 0.01379 | 0.0008582 | 1.369 |
| rates.pitch.delay_s | 4.711e-16 | 0.0005827 | 1.369 |
| rates.yaw.rc_rate | 1.484 | 0.02838 | 0.8734 |
| rates.yaw.super_rate | 0.5557 | 0.01005 | 0.8734 |
| rates.yaw.expo | 0 | - | 0.8734 |
| rates.yaw.tau_s | 0.004 | 0.0003143 | 0.8734 |
| rates.yaw.delay_s | 0.001085 | 0.0001539 | 0.8734 |
| drag.linear_per_s | 0.05814 | 0.002595 | 0.02421 |
| drag.quadratic_per_m | 0.01186 | 0.0005078 | 0.02421 |
| motor_lag_s | 0 | - | 0.1464 |
| hover_throttle | 0.3311 | 0.0009207 | 0.1464 |
| latency.control_to_gyro_s | 0.01051 | 0.0003866 | 0.001436 |

Residuals: rates in rad/s (output error), thrust in g, drag in g, delays as RMS of the echo error (stick units). Units: throttle fraction 0..1 (stick = 2f - 1), rates Betaflight (rc_rate 1.0 = 100), seconds.

### Thrust curve (throttle fraction -> body-z specific force, g)

| throttle | 0 | 0.15 | 0.3 | 0.45 | 0.6 | 0.8 | 1 |
|---|---|---|---|---|---|---|---|
| accel (g) | 0 | 0.141 | 0.795 | 1.78 | 2.88 | 4.74 | 7.72 |
| sigma | 0 | 0.0033 | 0.0023 | 0.0056 | 0.015 | 0.022 | 0.023 |

## Held-out validation (last pass)

| signal | one-step error | 1 s open-loop error | windows |
|---|---|---|---|
| roll rate (% of commanded) | 0.363 % | 0.794 % | 5 |
| pitch rate (% of commanded) | 0.663 % | 1.06 % | 5 |
| yaw rate (% of commanded) | 3.86 % | 4.21 % | 5 |
| vertical accel (g) | 0.147 | 0.153 | 13 |

Limits (AC8): median 1 s open-loop rate error <= 15 % of the commanded rate amplitude, vertical acceleration <= 0.15 g. AC8 NOT MET: held-out 1 s open-loop error 4.2 % of the commanded rate (limit 15 %) and 0.153 g vertical (limit 0.15 g); the profile is written with fit_quality: poor.

## Hover

- From the thrust curve: throttle 0.3311 (stick -0.3379); settled hover throttle 0.3337 (5960 samples), difference -0.0026

## Airframe check (ground-ramp hover vs. the expected band)

- ok: h0 0.2988, band [0.1, 0.4], source ui_driver.DRONE_SLOTS['Vortex Pro 250']
- ok: h0 0.298, band [0.1, 0.4], source ui_driver.DRONE_SLOTS['Vortex Pro 250']
- ok: h0 0.298, band [0.1, 0.4], source ui_driver.DRONE_SLOTS['Vortex Pro 250']

## Battery

- Voltage 21.01 V -> 19.30 V (min 17.53 V); the level is reset (full battery) between passes.
- Thrust sensitivity: -0.1 % thrust per volt (measured / model ratio vs. voltage). Not voltage-parameterised in the profile.

## Violations and aborted experiments

- None.
- Session stop reason: completed

## Default randomization (UC67)

nominal +- max(3 sigma, 10 %), clamped at 0; multipliers 1 +- max(3 sigma_rel, 10 %).

| parameter | min | max |
|---|---|---|
| thrust_scale | 0.9 | 1.1 |
| rates.roll.rate_scale | 0.8749 | 1.125 |
| rates.roll.tau_s | 0.009368 | 0.01398 |
| rates.roll.delay_s | 0.002172 | 0.004707 |
| rates.pitch.rate_scale | 0.8906 | 1.109 |
| rates.pitch.tau_s | 0.01121 | 0.01636 |
| rates.pitch.delay_s | 0 | 0.001748 |
| rates.yaw.rate_scale | 0.9 | 1.1 |
| rates.yaw.tau_s | 0.003057 | 0.004943 |
| rates.yaw.delay_s | 0.0006228 | 0.001546 |
| drag.linear_per_s | 0.05035 | 0.06593 |
| drag.quadratic_per_m | 0.01034 | 0.01339 |
| motor_lag_s | 0 | 0 |
| latency.control_to_gyro_s | 0.009348 | 0.01167 |

## Notes

- roll: expo ill-conditioned (sigma 0.47); fixed at 0, which fits as well
- pitch: expo ill-conditioned (sigma 0.41); fixed at 0, which fits as well
- yaw: expo ill-conditioned (sigma 0.27); fixed at 0, which fits as well

Stick -> body-rate signs measured: roll +1, pitch +1, yaw -1.

## Plots

![rates_roll.png](rates_roll.png)
![rates_pitch.png](rates_pitch.png)
![rates_yaw.png](rates_yaw.png)
![thrust_curve.png](thrust_curve.png)
![drag.png](drag.png)
![delay.png](delay.png)
