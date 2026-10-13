# UC69 oracle teacher session: Liftoff Arena "04 - Infinity Loop" (liftoff-arena-04)

> **ORACLE perception: diagnostic only, never used for training or held-out tuning.**
> TRAIN track, stale-build scene. Summary of the gitignored session report; no telemetry or frames
> are committed.

## Run 2: teacher_it1374 (teacher-obs v2, battery-aware), 2026-10-13: AC9 result

| | |
|---|---|
| Teacher | `20261013T075647Z-teacher-liftoff-aug`, `teacher_it1374.pt`, val-selected, sha `a140e970` |
| Observation | `teacher-obs` v2: v1 plus the pack block (charge, volts per cell, 3 cells) |
| Training | trained with the power model fitted on run 1's telemetry |
| Course | conversion v4 `7d4c8e44dcbb995e`; world scale 1.0 |
| Session | `artifacts/eval/liftoff-gremlin-micro/20261013T075647Z-teacher-liftoff-aug/oracle/liftoff-arena-04/sessions/20261013T134302Z-a140e970/` |
| Attempts | 13 recorded: **10 flown (N_eff 10)** and 3 refused before arming (see below) |

### Result: 10 of 10 finished the race

**All 10 flown attempts finished the full race: 33 of 33 crossings, 4 laps, no contact under the
sim's rules, no power loss.**

| | |
|---|---|
| Verdict | **control transfers** |
| Live completion, sim rules | 100 % (Wilson 95 % interval 72.2-100 %, N_eff 10) |
| Sim completion, same checkpoint and track | 87.5 % randomized, 100 % nominal |

| attempt | race time (1st to last crossing) | lap times | first gate after arming | min pack V | at the finish |
|---|---|---|---|---|---|
| 1 | 73.78 s | 14.12 / 18.19 / 20.30 / 21.16 s | 3.59 s | 8.83 V | 9.98 V, charge 0.48 |
| 2 | 73.76 s | 14.09 / 18.15 / 20.35 / 21.17 s | 3.59 s | 8.84 V | 9.95 V, charge 0.48 |
| 3 | 73.68 s | 14.07 / 18.15 / 20.31 / 21.15 s | 3.59 s | 8.84 V | 9.95 V, charge 0.48 |
| 4 | 73.92 s | 14.12 / 18.23 / 20.36 / 21.20 s | 3.59 s | 8.84 V | 9.96 V, charge 0.48 |
| 5 | 73.80 s | 14.09 / 18.23 / 20.32 / 21.16 s | 3.59 s | 8.83 V | 9.94 V, charge 0.48 |
| 6 | 73.84 s | 14.08 / 18.23 / 20.32 / 21.21 s | 3.61 s | 8.85 V | 9.96 V, charge 0.48 |
| 7 | 73.80 s | 14.09 / 18.23 / 20.33 / 21.15 s | 3.57 s | 8.83 V | 9.96 V, charge 0.48 |
| 8 | 73.71 s | 14.04 / 18.20 / 20.36 / 21.11 s | 3.59 s | 8.82 V | 9.94 V, charge 0.48 |
| 9 | 73.87 s | 14.14 / 18.19 / 20.35 / 21.19 s | 3.59 s | 8.83 V | 9.94 V, charge 0.48 |
| 10 | 73.82 s | 14.07 / 18.24 / 20.31 / 21.20 s | 3.59 s | 8.84 V | 9.96 V, charge 0.48 |

- **Race time:** 73.80 s p50 (73.68 to 73.92). Liftoff's own result screen read 1:13.7, 1st or 2nd
  against the AI ghosts.
- **Sim race, same checkpoint:** 79.3 s randomized, 77.2 s nominal. The live race is about 4-7 %
  faster.
- **Consistency:** the flights are near-deterministic. The spread is 0.24 s over the race and
  under 0.15 s per lap.
- **Battery:** the lowest pack voltage was 8.82 V, above Liftoff's 8.10 V cutoff. Each race ends
  about 76 s after spin-up with 48 % charge left.
- **Thrust at full stick:** body-z specific force 1.8 g p50 / 3.5 g p95 at 23 m/s. That is the
  same shortfall against the profile's 4.23 g as run 1, and the teacher now flies within it.
- **Loop:** 50.0 Hz (p50 and min), 11 missed deadlines in 38,949 ticks. Live vs offline crossings:
  0 mismatches.

### Refused attempts and the retry fix

Three attempts were refused at the start check (exit 6) and never armed. They are excluded from
N_eff. Each followed a finished race: the telemetry was 37.7 m off, rotated 90 deg and moving at
2.7-10 m/s.

- **Cause:** after a finished race, Liftoff shows its post-race screen with the replay running
  behind it, and Esc -> Reset does not respawn from there.
- **The interim `ui retry`, a click on that screen's Retry, failed in two ways:**
  - Inside the loop it clicked within about 1-5 s of the finish, before the screen accepted
    input.
  - Its at-rest check passed falsely, because the post-race state streams position (0, 0, 0).
- **The verified retry** (`ui_driver.retry`):
  1. waits for the screen by a pixel probe of its five buttons;
  2. hovers and clicks Retry;
  3. accepts only a **non-zero** position at the level's spawn (`spawn_matches`) at rest over
     2 s;
  4. re-clicks up to 3 times, then stops with `RetryError`.
- **Result:** it restarted the race on the first try after all 7 later finishes (1 by hand, 6 in
  the loop).
- **Side effect:** the last attempt's in-game recording was not saved. The save hook runs at the
  stop, but the pause-menu save path is not available from the post-race screen.

### Comparison with run 1

| | run 1: it1920 (v1, no power model) | run 2: it1374 (v2, battery-aware) |
|---|---|---|
| attempts | 10 flown | 10 flown |
| crossings | 11 / 33 in all 10 | **33 / 33 in all 10** |
| end | power_loss: 8.10 V cutoff 18.3 s after spin-up | race finished at about 76 s, min 8.82 V |
| lap 1 | 13.0 s | 14.1 s |
| race | not finished | 73.8 s (sim: 77-79 s) |
| speed on the long legs | about 24 m/s | about 23 m/s |

### Conclusion

With the battery modelled in the simulator, and the teacher seeing its pack and trained under it,
**control transfers completely with oracle gates**. The privileged teacher races all 4 laps of
Infinity Loop in Liftoff in 10 of 10 attempts, slightly faster than in its own sim. Run 1's
failure was a power-modelling gap, not a control gap. **The remaining gap to a live student is
perception (UC70/UC71).**

---

# Run 1: teacher_it1920 (teacher-obs v1, no power model)


## Session

| | |
|---|---|
| Date | 2026-10-13 |
| Drone | `liftoff-gremlin-micro` ("Test small drone") |
| Teacher | `20261013T014931Z-teacher-liftoff`, `teacher_it1920.pt`, val-selected, sha `14179a9d` |
| Course | conversion v3 `2a257cf0fa6b2fdb` |
| World scale | 1.0 (the preset's 4.0 is ignored) |
| Route | 33 crossings: 8 passages x 4 laps, a circuit |
| Session | `artifacts/eval/liftoff-gremlin-micro/20261013T014931Z-teacher-liftoff/oracle/liftoff-arena-04/sessions/20261013T051746Z-14179a9d/` |
| Attempts | 10 (`Invoke-LiftoffAttempts.ps1 -Teacher ... -Attempts 10`) |
| Excluded attempts | none (all 10 ended by the episode monitor) |
| Training | finished; this is not a concurrent-training session |

## Result

| figure | live (Liftoff, oracle gates) | sim randomized | sim nominal |
|---|---|---|---|
| completion, sim rules (headline) | **0 %** [0, 27.8] % (N_eff 10) | 100 % | 100 % |
| completion, raw | 0 % [0, 27.8] % | | |
| crossings passed | **11 / 33 in every attempt** | 33 / 33 | 33 / 33 |
| per-passage success | 91.7 % | 100 % | 100 % |
| lap time (lap 1) | 13.0 s (12.99 to 13.04) | 10.6 s | 10.7 s |
| time to the first gate after arming | 3.57 s | | |
| speed on the long legs | about 24 m/s | about 30 m/s (p50) | |

- **Mechanical verdict:** does not transfer. The dominant failure mode is **power_loss** in 10 of
  10 attempts.
- **Live vs offline crossings:** 0 mismatches.
- **Tick rate:** 50.0 Hz (p50 and min), 1 missed deadline in 11,400 ticks.
- **Latency (p95 / p99):** total 7.6 / 8.6 ms; oracle 1.4 ms; teacher observation assembly 6.1 ms;
  policy 0.5 ms.
- **Arming:** confirmed by motor RPM 0.06 s after the Arm press.

## Diagnosis

Every attempt takes off cleanly, flies all of lap 1 and passages 0-2 of lap 2, then falls. All 10
fall at the same place: the leg from passage 2 to passage 3 on lap 2, right after crossing 11. The
flights are deterministic, so they are near-identical.

**Not the cause:**
- **Lap logic or geometry.** The route progress advances correctly across the finish-line
  re-crossing (crossings 8 -> 9 -> 10 -> 11). The live and offline crossings match exactly. Lap 1
  flies the same leg at the same speed (24 m/s, 65 deg bank) and holds its altitude. The surface
  hit is the floor; there were no gate-frame or prop impacts.
- **Take-off.** The 85 deg tilt in the first 2 s is an aggressive racing take-off: the first gate
  is at 3.6 s. The classifier now counts a take-off tumble only for an attempt that never reaches
  the first gate.

**The cause: Liftoff's battery low-voltage cutoff.**
- In all 10 attempts, 18.2-18.4 s after the motors spin up, the 3S pack reaches exactly
  **8.10 V (2.70 V/cell)**. The charge still reads 74 %.
- At that moment all four motors drop from about 31-33k rpm to about 1.1-1.6k rpm, even though the
  throttle stick is still at 0.95-1.0.
- The body-z specific force goes from about +2.4 g to about -0.5 g within one tick, and the drone
  falls to the floor.
- The sag across the attempt: 12.25 V at spin-up, about 10.7 V within 5 s, then 8.1 V at the
  cut.
- Why the pack runs down: the teacher holds the collective at the drone's `max_g` (4.23 g) almost
  all the time. In the sim replay of the same checkpoint the collective is saturated 100 % of the
  time. The simulator has no battery, voltage-sag or cutoff model, so it never pays for this.

**Thrust shortfall at speed.**
- The measured body-z specific force at full stick, with the motors spun up, is:
  - **1.8 g p50 / 2.9 g p95** in the first 5 s, at 24 m/s and 10.7 V;
  - **2.0 g p50 / 3.0 g p95** in the last 3 s before the cut, at 24 m/s and 8.3 V.
- The drone profile's static thrust curve gives **4.23 g**.
- The shortfall does not grow visibly with the voltage sag: the force is lower from the start and
  stays lower.
- This consistent loss of thrust at speed explains why Liftoff flies slower than the sim: 24 vs
  30 m/s, and a 13.0 s vs 10.6 s lap.

## Conclusion

With exact gate poses, **control transfers**:
- take-off is clean;
- the teacher flies a full lap plus three gates, with 91.7 % per-passage success;
- the 50 Hz loop holds.

What breaks the attempts is **power and aerodynamics modelling**, not perception and not the
policy's geometry:
1. The sim has no battery. The teacher's always-full-throttle style drains Liftoff's pack to its
   8.1 V cutoff after about 18 s, which is roughly 1.4 laps of a 4-lap race.
2. The thrust available at racing speed is about half of the profile's static `max_g`.

Before UC70/UC71 are judged in Liftoff, the simulator needs:
- a pack model: sag, cutoff, and the energy used per unit of collective;
- thrust that falls off with airspeed and voltage;
- a teacher retrained under them, or a penalty on sustained saturation.

The per-attempt power time series for that fit are exported (gitignored) to the session's
`power/` folder.
