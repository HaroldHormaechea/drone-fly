# UC69 oracle teacher session: Liftoff Arena "04 - Infinity Loop" (liftoff-arena-04)

> **ORACLE perception: diagnostic only, never used for training or held-out tuning.**
> TRAIN track, stale-build scene. Summary of the gitignored session report; no telemetry or frames
> are committed.

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
