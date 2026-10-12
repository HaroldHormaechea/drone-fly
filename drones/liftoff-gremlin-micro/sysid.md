# System identification: Liftoff Test small drone (FT Gremlin micro) (`liftoff-gremlin-micro`)

- Session: `micro-full` (2026-10-12T15:37:55Z), plant: liftoff, seed 0
- Liftoff drone: Test small drone; map: Liftoff Arena (free flight, no track)
- Settings: preset `liftoff`; schedule 93 experiments in 3 passes (last pass held out)
- Git: `67f99c5614c8a9e3bbd93430c75d56fdf1c3cd25`

**fit_quality: good.** AC8 met: held-out 1 s open-loop error 0.6 % of the commanded rate (<= 15 %) and 0.093 g vertical (<= 0.15 g).


## Fitted values

| parameter | value | sigma | fit residual |
|---|---|---|---|
| latency.input_s | 0.003 | 1.212e-05 | 0.02441 |
| rates.roll.rc_rate | 0.8547 | 0.02424 | 0.6978 |
| rates.roll.super_rate | 0.7074 | 0.01056 | 0.6978 |
| rates.roll.expo | 0 | - | 0.6978 |
| rates.roll.tau_s | 0.007101 | 0.0003778 | 0.6978 |
| rates.roll.delay_s | 0.008383 | 0.0001666 | 0.6978 |
| rates.pitch.rc_rate | 0.8191 | 0.0261 | 0.7293 |
| rates.pitch.super_rate | 0.7106 | 0.01203 | 0.7293 |
| rates.pitch.expo | 0 | - | 0.7293 |
| rates.pitch.tau_s | 0.01209 | 0.0007237 | 0.7293 |
| rates.pitch.delay_s | 0.005462 | 0.0004703 | 0.7293 |
| rates.yaw.rc_rate | 0.9084 | 0.01242 | 0.5371 |
| rates.yaw.super_rate | 0.6958 | 0.004797 | 0.5371 |
| rates.yaw.expo | 0 | - | 0.5371 |
| rates.yaw.tau_s | 0.004 | 0.0002424 | 0.5371 |
| rates.yaw.delay_s | 0.006738 | 0.000109 | 0.5371 |
| drag.linear_per_s | 0.0578 | 0.004536 | 0.03503 |
| drag.quadratic_per_m | 0.02509 | 0.0009731 | 0.03503 |
| motor_lag_s | 0.003217 | 0.000142 | 0.06743 |
| hover_throttle | 0.4708 | 0.0007188 | 0.06743 |
| latency.control_to_gyro_s | 0.009861 | 0.0002489 | 0.001196 |

Residuals: rates in rad/s (output error), thrust in g, drag in g, delays as RMS of the echo error (stick units). Units: throttle fraction 0..1 (stick = 2f - 1), rates Betaflight (rc_rate 1.0 = 100), seconds.

### Thrust curve (throttle fraction -> body-z specific force, g)

| throttle | 0 | 0.15 | 0.3 | 0.45 | 0.6 | 0.8 | 1 |
|---|---|---|---|---|---|---|---|
| accel (g) | 0 | 0.001 | 0.303 | 0.898 | 1.63 | 2.85 | 4.23 |
| sigma | 0 | 0.0055 | 0.0043 | 0.00092 | 0.0034 | 0.0054 | 0.0073 |

## Held-out validation (last pass)

| signal | one-step error | 1 s open-loop error | windows |
|---|---|---|---|
| roll rate (% of commanded) | 0.261 % | 0.408 % | 5 |
| pitch rate (% of commanded) | 0.277 % | 0.589 % | 5 |
| yaw rate (% of commanded) | 0.55 % | 0.599 % | 4 |
| vertical accel (g) | 0.0898 | 0.0932 | 13 |

Limits (AC8): median 1 s open-loop rate error <= 15 % of the commanded rate amplitude, vertical acceleration <= 0.15 g. AC8 met: held-out 1 s open-loop error 0.6 % of the commanded rate (<= 15 %) and 0.093 g vertical (<= 0.15 g).

## Hover

- From the thrust curve: throttle 0.4708 (stick -0.0583); settled hover throttle 0.4706 (5697 samples), difference +0.0002
- UC64 comparison: measured hover stick -0.058 vs. UC64's hand-flown -0.06 (throttle 0.470); difference +0.002 stick.

## Airframe check (ground-ramp hover vs. the expected band)

- ok: h0 0.4628, band [0.4, 0.56], source ui_driver.DRONE_SLOTS['Test small drone']
- ok: h0 0.4628, band [0.4, 0.56], source ui_driver.DRONE_SLOTS['Test small drone']
- ok: h0 0.4644, band [0.4, 0.56], source ui_driver.DRONE_SLOTS['Test small drone']

## Battery

- Voltage 12.61 V -> 11.04 V (min 10.08 V); the level is reset (full battery) between passes.
- Thrust sensitivity: +0.3 % thrust per volt (measured / model ratio vs. voltage). Not voltage-parameterised in the profile.

## Violations and aborted experiments

- altitude_ceiling (abort_experiment): altitude 12.98 m in p0-coast-35+
- altitude_ceiling (abort_experiment): altitude 12.29 m in p1-step-0.20+
- altitude_ceiling (abort_experiment): altitude 12.66 m in p2-doublet-yaw-1.00-
- `p0-coast-35+`: aborted:altitude_ceiling
- `p1-step-0.20+`: aborted:altitude_ceiling
- `p2-doublet-yaw-1.00-`: aborted:altitude_ceiling
- Session stop reason: completed

## Default randomization (UC67)

nominal +- max(3 sigma, 10 %), clamped at 0; multipliers 1 +- max(3 sigma_rel, 10 %).

| parameter | min | max |
|---|---|---|
| thrust_scale | 0.9 | 1.1 |
| rates.roll.rate_scale | 0.9 | 1.1 |
| rates.roll.tau_s | 0.005968 | 0.008235 |
| rates.roll.delay_s | 0.007544 | 0.009221 |
| rates.pitch.rate_scale | 0.9 | 1.1 |
| rates.pitch.tau_s | 0.009922 | 0.01426 |
| rates.pitch.delay_s | 0.004051 | 0.006873 |
| rates.yaw.rate_scale | 0.9 | 1.1 |
| rates.yaw.tau_s | 0.003273 | 0.004727 |
| rates.yaw.delay_s | 0.006064 | 0.007412 |
| drag.linear_per_s | 0.04419 | 0.07141 |
| drag.quadratic_per_m | 0.02217 | 0.02801 |
| motor_lag_s | 0.002791 | 0.003643 |
| latency.control_to_gyro_s | 0.008875 | 0.01085 |

## Notes

- roll: expo ill-conditioned (sigma 0.82); fixed at 0, which fits as well
- pitch: expo ill-conditioned (sigma 0.89); fixed at 0, which fits as well
- yaw: expo ill-conditioned (sigma 0.38); fixed at 0, which fits as well

Stick -> body-rate signs measured: roll +1, pitch +1, yaw -1.

## Plots

![rates_roll.png](rates_roll.png)
![rates_pitch.png](rates_pitch.png)
![rates_yaw.png](rates_yaw.png)
![thrust_curve.png](thrust_curve.png)
![drag.png](drag.png)
![delay.png](delay.png)
