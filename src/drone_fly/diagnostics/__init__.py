"""Read-mostly diagnostic harnesses for drone-fly (UC-47+).

Modules here characterize the simulator / control pathway to CONFIRM or REFUTE a
mechanism hypothesis with reproducible numbers, rather than change flight behaviour. They
are import-safe in hermetic CI: any real-simulator (pybullet) dependency is imported
**lazily**, inside the code path that needs it, so importing a diagnostics module never
requires the sim toolchain.

* :mod:`drone_fly.diagnostics.thrust_pathway` (UC-47) — maps achieved vertical thrust /
  T-W across collective-throttle and body-rate commands through the open-loop CTBR→RPM
  mixer, reproduces the recorded free-fall through ``RaceEnv``, and delivers a verdict on
  the collective-desaturation hypothesis.
"""

from __future__ import annotations
