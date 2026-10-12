"""Per-episode dynamics randomization from a drone profile's ``randomization`` block (UC67 AC4).

At every (masked) reset each environment draws EVERY key listed in ``profile.randomization``
uniformly from its range scaled about the nominal::

    lo' = nominal + scale * (lo - nominal),   hi' = nominal + scale * (hi - nominal)

``scale = 0`` is the nominal plant, ``scale = 1`` the profile's range. Multipliers
(``thrust_scale``, ``rates.<axis>.rate_scale``) have nominal 1. Keys the profile does not list
stay nominal. A per-axis (``[x, y, z]``) drag nominal samples each axis independently; a scalar
drag nominal draws one value shared by the three body axes.

``latency.control_to_gyro_s`` is derived in the profile (``input_s + mean(delay_s)``), so its
sample enters the plant only as the common-mode offset ``sample - nominal`` (see
:mod:`drone_fly.sim.plant`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

AXES = ("roll", "pitch", "yaw")

#: Log column order (``episodes.csv``): one column per sampled scalar.
LOG_COLUMNS = (
    "thrust_scale",
    *(f"rate_scale_{a}" for a in AXES),
    *(f"tau_{a}_s" for a in AXES),
    *(f"delay_{a}_s" for a in AXES),
    *(f"drag_linear_{c}" for c in "xyz"),
    *(f"drag_quadratic_{c}" for c in "xyz"),
    "motor_lag_s",
    "c2g_offset_s",
)


def _vec3(v: float | tuple[float, float, float]) -> tuple[float, float, float]:
    return (float(v),) * 3 if not isinstance(v, tuple) else tuple(float(x) for x in v)


@dataclass
class DynamicsSamples:
    """Per-environment parameter tensors (``n`` rows)."""

    thrust_scale: torch.Tensor  # (n,)
    rate_scale: torch.Tensor  # (n, 3)
    tau_s: torch.Tensor  # (n, 3) rate response
    delay_s: torch.Tensor  # (n, 3) per-axis delay
    drag_linear: torch.Tensor  # (n, 3) body axes, 1/s
    drag_quadratic: torch.Tensor  # (n, 3) body axes, 1/m
    motor_lag_s: torch.Tensor  # (n,)
    c2g_offset_s: torch.Tensor  # (n,) common-mode delay offset

    def columns(self) -> torch.Tensor:
        """``(n, len(LOG_COLUMNS))`` in :data:`LOG_COLUMNS` order."""
        return torch.cat(
            [
                self.thrust_scale[:, None],
                self.rate_scale,
                self.tau_s,
                self.delay_s,
                self.drag_linear,
                self.drag_quadratic,
                self.motor_lag_s[:, None],
                self.c2g_offset_s[:, None],
            ],
            dim=1,
        )


class DynamicsRandomizer:
    """Samples :class:`DynamicsSamples` for a profile; see the module docstring."""

    def __init__(
        self,
        profile: Any,
        scale: float = 0.0,
        generator: torch.Generator | None = None,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not 0.0 <= float(scale):
            raise ValueError(f"randomization scale must be >= 0, got {scale}")
        self.profile = profile
        self.scale = float(scale)
        self.generator = generator
        self.device = torch.device(device)
        self.dtype = dtype
        rates = profile.rates
        # nominal values, keyed like profile.randomization; each is a 1- or 3-tuple
        self.nominal: dict[str, tuple[float, ...]] = {
            "thrust_scale": (1.0,),
            **{f"rates.{a}.rate_scale": (1.0,) for a in AXES},
            **{f"rates.{a}.tau_s": (rates[a].tau_s,) for a in AXES},
            **{f"rates.{a}.delay_s": (rates[a].delay_s,) for a in AXES},
            "drag.linear_per_s": _vec3(profile.drag.linear_per_s),
            "drag.quadratic_per_m": _vec3(profile.drag.quadratic_per_m),
            "motor_lag_s": (profile.motor_lag_s,),
            "latency.control_to_gyro_s": (profile.latency.control_to_gyro_s,),
        }
        self._iso = {
            "drag.linear_per_s": not isinstance(profile.drag.linear_per_s, tuple),
            "drag.quadratic_per_m": not isinstance(profile.drag.quadratic_per_m, tuple),
        }
        self.randomized = sorted(profile.randomization)

    def bounds(self, key: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Scaled ``(lo', hi')`` per component of ``key`` (equal to the nominal if unlisted)."""
        nom = self.nominal[key]
        rng = self.profile.randomization.get(key)
        if rng is None:
            return nom, nom
        s = self.scale
        lo = tuple(n + s * (rng.min - n) for n in nom)
        hi = tuple(n + s * (rng.max - n) for n in nom)
        return lo, hi

    def _draw(self, key: str, n: int) -> torch.Tensor:
        """``(n, k)`` samples of ``key`` (k = number of components)."""
        lo, hi = self.bounds(key)
        lo_t = torch.tensor(lo, dtype=self.dtype, device=self.device)
        hi_t = torch.tensor(hi, dtype=self.dtype, device=self.device)
        k = 1 if self._iso.get(key, False) else len(lo)
        u = torch.rand(n, k, generator=self.generator, device=self.device, dtype=self.dtype)
        return lo_t + (hi_t - lo_t) * u  # isotropic drag: (n, 1) broadcasts over the axes

    def sample(self, n: int) -> DynamicsSamples:
        """Fresh samples for ``n`` environments (every listed key is drawn)."""

        def axes(name: str) -> torch.Tensor:
            return torch.cat([self._draw(f"rates.{a}.{name}", n) for a in AXES], dim=1)

        def drag(key: str) -> torch.Tensor:
            return self._draw(key, n).expand(n, 3).clone()

        c2g = self._draw("latency.control_to_gyro_s", n)[:, 0]
        return DynamicsSamples(
            thrust_scale=self._draw("thrust_scale", n)[:, 0],
            rate_scale=axes("rate_scale"),
            tau_s=axes("tau_s"),
            delay_s=axes("delay_s"),
            drag_linear=drag("drag.linear_per_s"),
            drag_quadratic=drag("drag.quadratic_per_m"),
            motor_lag_s=self._draw("motor_lag_s", n)[:, 0],
            c2g_offset_s=c2g - float(self.nominal["latency.control_to_gyro_s"][0]),
        )

    def resample(self, current: DynamicsSamples, mask: torch.Tensor) -> None:
        """Redraw ``current`` in place where ``mask`` is set (no host synchronisation)."""
        fresh = self.sample(mask.shape[0])
        for name in DynamicsSamples.__dataclass_fields__:
            cur = getattr(current, name)
            new = getattr(fresh, name)
            m = mask.view(-1, *([1] * (cur.dim() - 1)))
            cur.copy_(torch.where(m, new, cur))
