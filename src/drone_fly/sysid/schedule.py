"""The fixed, seeded experiment schedule of a sysid session.

A schedule is ``passes`` passes (default 3, at least 2); every pass holds every experiment type in
its own seeded order, and the **last pass is held out** for validation (the fitter never sees it).

Types (amplitudes default to 0.2, 0.4, 0.6, 0.8, 1.0):

* ``throttle_step``  open-loop throttle step up and down per amplitude (stabilizer on, altitude loop
  frozen): up to ``hover + a * step_headroom * (1 - hover)``, down to ``hover * (1 - a * 0.8)``
* ``throttle_chirp`` one throttle sine sweep around hover (0.5 -> 5 Hz)
* ``throttle_burst`` short high-throttle bursts (~0.25 s) to reach the top of the thrust curve
* ``rate_doublet``   roll / pitch / yaw stick doublets (+a then -a, or the reverse) per amplitude,
  open-loop with the throttle frozen at hover; amplitude 1.0 is the full-stick rate
* ``coast``          tilt to accelerate, level out and coast (drag), alternating heading
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

AXES = ("roll", "pitch", "yaw")
KINDS = ("throttle_step", "throttle_chirp", "throttle_burst", "rate_doublet", "coast")
THROTTLE_KINDS = ("throttle_step", "throttle_chirp", "throttle_burst")
DEFAULT_AMPLITUDES = (0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class ScheduleSettings:
    amplitudes: tuple[float, ...] = DEFAULT_AMPLITUDES
    passes: int = 3
    #: Multiplies every hold / window duration.
    duration_scale: float = 1.0
    coast_tilts_deg: tuple[float, ...] = (15.0, 25.0, 35.0)
    burst_amplitudes: tuple[float, ...] = (0.8, 1.0)
    chirp: bool = True

    def __post_init__(self) -> None:
        if self.passes < 2:
            raise ValueError("a sysid schedule needs at least 2 passes (the last is held out)")
        if not self.amplitudes or not all(0.0 < a <= 1.0 for a in self.amplitudes):
            raise ValueError(f"amplitudes must be in (0, 1]: {self.amplitudes}")
        if self.duration_scale <= 0:
            raise ValueError("duration_scale must be positive")


@dataclass(frozen=True)
class Experiment:
    id: str
    kind: str
    pass_index: int
    held_out: bool
    amplitude: float = 0.0
    sign: int = 1
    axis: str | None = None
    #: Open-loop hold (step / burst level, doublet half-period cap, coast acceleration).
    hold_s: float = 0.0
    #: Open-loop tail after the hold (still inside the fit window).
    tail_s: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Experiment:
        return cls(**d)


def _pass_experiments(p: int, held_out: bool, s: ScheduleSettings, rng: Any) -> list[Experiment]:
    k = s.duration_scale
    out: list[Experiment] = []
    sgn = "+-"
    for a in s.amplitudes:
        for sign in (1, -1):
            out.append(
                Experiment(
                    f"p{p}-step-{a:.2f}{sgn[sign < 0]}",
                    "throttle_step",
                    p,
                    held_out,
                    a,
                    sign,
                    hold_s=0.4 * k,
                    tail_s=0.4 * k,
                )
            )
    if s.chirp:
        out.append(
            Experiment(
                f"p{p}-chirp",
                "throttle_chirp",
                p,
                held_out,
                0.15,
                1,
                hold_s=3.0 * k,
                tail_s=0.3 * k,
                params={"f0_hz": 0.5, "f1_hz": 5.0},
            )
        )
    for a in s.burst_amplitudes:
        out.append(
            Experiment(
                f"p{p}-burst-{a:.2f}+", "throttle_burst", p, held_out, a, 1, 0.25 * k, 0.4 * k
            )
        )
    for axis in AXES:
        for a in s.amplitudes:
            sign = int(rng.choice([1, -1]))
            out.append(
                Experiment(
                    f"p{p}-doublet-{axis}-{a:.2f}{sgn[sign < 0]}",
                    "rate_doublet",
                    p,
                    held_out,
                    a,
                    sign,
                    axis,
                    hold_s=(0.3 if axis == "yaw" else 0.15) * k,
                    tail_s=0.25 * k,
                    params={"max_angle_deg": 180.0 if axis == "yaw" else 45.0},
                )
            )
    for i, tilt in enumerate(s.coast_tilts_deg):
        sign = 1 if i % 2 == 0 else -1
        out.append(
            Experiment(
                f"p{p}-coast-{tilt:.0f}{sgn[sign < 0]}",
                "coast",
                p,
                held_out,
                tilt,
                sign,
                hold_s=1.2 * k,
                tail_s=1.5 * k,
            )
        )
    return [out[i] for i in rng.permutation(len(out))]


def build_schedule(seed: int, settings: ScheduleSettings | None = None) -> list[Experiment]:
    """The full schedule (pass by pass); deterministic for a seed."""
    s = settings or ScheduleSettings()
    experiments: list[Experiment] = []
    for p in range(s.passes):
        rng = np.random.default_rng([int(seed), p])
        experiments += _pass_experiments(p, p == s.passes - 1, s, rng)
    return experiments
