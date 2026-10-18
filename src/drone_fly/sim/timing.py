"""Control-rate and observation-to-action latency randomization (UC67 AC5).

Per environment and episode: a control period (from a Hz range or a set of rates), a base action
delay and a skip probability. Per tick: the delay plus Gaussian jitter (clamped at 0) and a
Bernoulli skipped tick.

**Skip semantics.** A skipped tick mirrors the live failure (a missed deadline): no new action is
computed in time, so the action in force is held for about two periods. In the batched env that
is ONE transition whose plant time is ``2 * period`` (a per-env dt multiplier of 2), not a second
policy step; the dt-scaled reward absorbs the longer transition.

**Default ranges** cover the live harness's measured latency. Source: the UC66-era live run
``artifacts/harness/liftoff/runs/liftoff-20261011T233116352187Z/latency_summary.json`` (07
checkpoint, preset ``liftoff``): rate 50 Hz target / 36.9 Hz effective, 41 of 116 deadlines
missed (35 %), tick period p50 20.3 ms / p95 40.4 ms, compute p50 17.3 ms / p95 25.5 ms / max
31 ms, telemetry age p50 7.7 ms / p95 14.4 ms, i.e. observation -> action about 28 ms p50 and
43 ms p95. Hence 20-50 Hz, 0-60 ms of action delay, 5 ms jitter and up to 35 % skipped ticks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

#: The live run the defaults cite (relative to the main checkout's artifacts/).
LATENCY_SOURCE = (
    "artifacts/harness/liftoff/runs/liftoff-20261011T233116352187Z/latency_summary.json"
)

TIMING_LOG_COLUMNS = ("control_hz", "action_delay_s", "skip_prob")


@dataclass(frozen=True)
class TimingConfig:
    """``mode: random`` (the ranges below) or ``fixed`` (one rate, zero delay, zero skip)."""

    mode: str = "random"
    #: ``[lo, hi]`` Hz drawn uniformly, unless ``control_hz_set`` lists discrete rates.
    control_hz: tuple[float, float] = (20.0, 50.0)  # live: 50 Hz target, 36.9 Hz effective
    control_hz_set: tuple[float, ...] | None = None
    #: Base observation -> action delay per episode, uniform (s). Live p50 ~28 ms, p95 ~43 ms.
    action_delay_s: tuple[float, float] = (0.0, 0.060)
    #: Per-episode delay distribution: ``uniform`` (default) or ``triangular`` (peaked at
    #: ``action_delay_mode_s``, defaulting to the range midpoint) — UC-87's deploy-timing option to
    #: concentrate the base delay near the measured live ~75 ms rather than spreading it uniformly.
    action_delay_dist: str = "uniform"
    #: Triangular peak (s); None ⇒ the midpoint of ``action_delay_s``. Ignored for ``uniform``.
    action_delay_mode_s: float | None = None
    #: Per-tick Gaussian jitter on the delay (s, std); the delay is clamped at 0.
    delay_jitter_s: float = 0.005  # live compute p50 17.3 ms vs p95 25.5 ms
    #: Per-episode skip probability, uniform. Live: 41/116 = 35 % missed deadlines.
    skip_prob: tuple[float, float] = (0.0, 0.35)
    #: ``fixed`` mode rate (Hz).
    fixed_hz: float = 50.0

    def __post_init__(self) -> None:
        if self.mode not in ("random", "fixed"):
            raise ValueError(f"timing.mode must be random or fixed, got {self.mode!r}")
        lo, hi = self.control_hz
        if not 0 < lo <= hi:
            raise ValueError(f"timing.control_hz needs 0 < lo <= hi, got {self.control_hz}")
        if self.control_hz_set is not None and (
            not self.control_hz_set or min(self.control_hz_set) <= 0
        ):
            raise ValueError("timing.control_hz_set must list positive rates")
        dlo, dhi = self.action_delay_s
        if not 0 <= dlo <= dhi:
            raise ValueError(f"timing.action_delay_s needs 0 <= lo <= hi, got {(dlo, dhi)}")
        if self.action_delay_dist not in ("uniform", "triangular"):
            raise ValueError("timing.action_delay_dist must be uniform or triangular")
        if self.action_delay_mode_s is not None and not dlo <= self.action_delay_mode_s <= dhi:
            raise ValueError("timing.action_delay_mode_s must lie within action_delay_s")
        if self.delay_jitter_s < 0:
            raise ValueError("timing.delay_jitter_s must be >= 0")
        plo, phi = self.skip_prob
        if not 0 <= plo <= phi < 1:
            raise ValueError(f"timing.skip_prob needs 0 <= lo <= hi < 1, got {(plo, phi)}")
        if self.fixed_hz <= 0:
            raise ValueError("timing.fixed_hz must be positive")

    @property
    def delay_mode_s(self) -> float:
        """The triangular peak (midpoint of ``action_delay_s`` when unset)."""
        if self.action_delay_mode_s is not None:
            return float(self.action_delay_mode_s)
        return 0.5 * (self.action_delay_s[0] + self.action_delay_s[1])

    @classmethod
    def fixed(cls, hz: float) -> TimingConfig:
        return cls(mode="fixed", fixed_hz=float(hz))

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> TimingConfig:
        d = dict(d or {})
        kw: dict[str, Any] = {}
        for key in ("control_hz", "action_delay_s", "skip_prob"):
            if key in d:
                kw[key] = tuple(float(x) for x in d.pop(key))
        if d.get("control_hz_set") is not None:
            kw["control_hz_set"] = tuple(float(x) for x in d.pop("control_hz_set"))
        else:
            d.pop("control_hz_set", None)
        for key in ("delay_jitter_s", "fixed_hz"):
            if key in d:
                kw[key] = float(d.pop(key))
        if "action_delay_mode_s" in d:
            v = d.pop("action_delay_mode_s")
            kw["action_delay_mode_s"] = None if v is None else float(v)
        if "action_delay_dist" in d:
            kw["action_delay_dist"] = str(d.pop("action_delay_dist"))
        if "mode" in d:
            kw["mode"] = str(d.pop("mode"))
        if d:
            raise ValueError(f"unknown timing keys {sorted(d)}")
        return cls(**kw)

    @property
    def max_period_s(self) -> float:
        if self.mode == "fixed":
            return 1.0 / self.fixed_hz
        lo = min(self.control_hz_set) if self.control_hz_set else self.control_hz[0]
        return 1.0 / lo

    @property
    def max_delay_s(self) -> float:
        """Upper bound of a tick's delay (base + 5 sigma of jitter)."""
        if self.mode == "fixed":
            return 0.0
        return self.action_delay_s[1] + 5.0 * self.delay_jitter_s


@dataclass
class ControlTiming:
    """Per-env timing state: sampled per episode, drawn per tick (see the module docstring)."""

    config: TimingConfig
    n: int
    device: torch.device
    generator: torch.Generator | None = None
    dtype: torch.dtype = torch.float64
    #: UC-87 deployment ramp (0..1). 1.0 ⇒ the full configured ranges (byte-identical default);
    #: 0.0 ⇒ the clean baseline (fixed ``config.fixed_hz`` period, zero delay, zero skip). The env
    #: sets this to ``deploy_dr_scale`` before each resample.
    scale: float = 1.0
    period_s: torch.Tensor = field(init=False)
    delay_s: torch.Tensor = field(init=False)
    skip_prob: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        z = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        self.period_s = z.clone()
        self.delay_s = z.clone()
        self.skip_prob = z.clone()
        self.resample(torch.ones(self.n, dtype=torch.bool, device=self.device))

    def _u(self) -> torch.Tensor:
        return torch.rand(self.n, generator=self.generator, device=self.device, dtype=self.dtype)

    def resample(self, mask: torch.Tensor) -> None:
        """Draw a new period / base delay / skip probability where ``mask`` is set."""
        c = self.config
        if c.mode == "fixed":
            period = torch.full_like(self.period_s, 1.0 / c.fixed_hz)
            delay = torch.zeros_like(self.delay_s)
            skip = torch.zeros_like(self.skip_prob)
        else:
            if c.control_hz_set:
                rates = torch.tensor(c.control_hz_set, dtype=self.dtype, device=self.device)
                idx = torch.randint(
                    0,
                    len(c.control_hz_set),
                    (self.n,),
                    generator=self.generator,
                    device=self.device,
                )
                hz = rates[idx]
            else:
                hz = c.control_hz[0] + (c.control_hz[1] - c.control_hz[0]) * self._u()
            period = 1.0 / hz
            dlo, dhi = c.action_delay_s
            if c.action_delay_dist == "triangular" and dhi > dlo:
                # inverse-CDF triangular sampling, peaked at ``delay_mode_s``
                mode = c.delay_mode_s
                u = self._u()
                split = (mode - dlo) / (dhi - dlo)
                lower = dlo + torch.sqrt(u * (dhi - dlo) * (mode - dlo))
                upper = dhi - torch.sqrt((1.0 - u) * (dhi - dlo) * (dhi - mode))
                delay = torch.where(u < split, lower, upper)
            else:
                delay = dlo + (dhi - dlo) * self._u()
            skip = c.skip_prob[0] + (c.skip_prob[1] - c.skip_prob[0]) * self._u()
            # UC-87 deployment ramp: interpolate from the clean baseline toward the sampled values.
            # The clean period anchor is ``config.fixed_hz`` — the SAME rate the `perfect` preset
            # uses (``TimingConfig.fixed(PERFECT_HZ)`` in smooth_eval); keep fixed_hz and PERFECT_HZ
            # consistent if either changes. Delay/skip ramp from 0.
            s = float(self.scale)
            if s < 1.0:
                clean_period = torch.full_like(period, 1.0 / c.fixed_hz)
                period = clean_period + s * (period - clean_period)
                delay = s * delay
                skip = s * skip
        self.period_s = torch.where(mask, period, self.period_s)
        self.delay_s = torch.where(mask, delay, self.delay_s)
        self.skip_prob = torch.where(mask, skip, self.skip_prob)

    def tick(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(dt_s, action_delay_s, skipped)`` for this tick, per env.

        ``dt_s`` is the period, doubled where the tick is skipped; the delay is the episode's
        base delay plus jitter, clamped at 0.
        """
        if self.config.mode == "fixed":
            return self.period_s, self.delay_s, torch.zeros_like(self.period_s, dtype=torch.bool)
        jit = self.config.delay_jitter_s * torch.randn(
            self.n, generator=self.generator, device=self.device, dtype=self.dtype
        )
        delay = (self.delay_s + jit).clamp(min=0.0)
        skipped = self._u() < self.skip_prob
        dt = torch.where(skipped, 2.0 * self.period_s, self.period_s)
        return dt, delay, skipped

    def columns(self) -> torch.Tensor:
        """``(n, 3)`` log columns in :data:`TIMING_LOG_COLUMNS` order."""
        return torch.stack([1.0 / self.period_s, self.delay_s, self.skip_prob], dim=1)
