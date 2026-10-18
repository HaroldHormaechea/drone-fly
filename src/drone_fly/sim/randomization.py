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


#: (randomization key, component) of every LOG_COLUMNS column, in order.
_LAYOUT = (
    ("thrust_scale", 0),
    *((f"rates.{a}.rate_scale", 0) for a in AXES),
    *((f"rates.{a}.tau_s", 0) for a in AXES),
    *((f"rates.{a}.delay_s", 0) for a in AXES),
    *(("drag.linear_per_s", i) for i in range(3)),
    *(("drag.quadratic_per_m", i) for i in range(3)),
    ("motor_lag_s", 0),
    ("latency.control_to_gyro_s", 0),
)


class DynamicsSamples:
    """Per-environment parameter tensors: views into one ``(n, len(LOG_COLUMNS))`` table."""

    def __init__(self, table: torch.Tensor) -> None:
        self.table = table

    @property
    def thrust_scale(self) -> torch.Tensor:  # (n,)
        return self.table[:, 0]

    @property
    def rate_scale(self) -> torch.Tensor:  # (n, 3)
        return self.table[:, 1:4]

    @property
    def tau_s(self) -> torch.Tensor:  # (n, 3) rate response
        return self.table[:, 4:7]

    @property
    def delay_s(self) -> torch.Tensor:  # (n, 3) per-axis delay
        return self.table[:, 7:10]

    @property
    def drag_linear(self) -> torch.Tensor:  # (n, 3) body axes, 1/s
        return self.table[:, 10:13]

    @property
    def drag_quadratic(self) -> torch.Tensor:  # (n, 3) body axes, 1/m
        return self.table[:, 13:16]

    @property
    def motor_lag_s(self) -> torch.Tensor:  # (n,)
        return self.table[:, 16]

    @property
    def c2g_offset_s(self) -> torch.Tensor:  # (n,) common-mode delay offset
        return self.table[:, 17]

    def columns(self) -> torch.Tensor:
        """``(n, len(LOG_COLUMNS))`` in :data:`LOG_COLUMNS` order."""
        return self.table


class DynamicsRandomizer:
    """Samples :class:`DynamicsSamples` for a profile; see the module docstring.

    One uniform draw per sampled scalar (an isotropic drag shares one draw over its three axes),
    in a single ``rand`` call: no host synchronisation and few kernels per reset.
    """

    def __init__(
        self,
        profile: Any,
        scale: float = 0.0,
        generator: torch.Generator | None = None,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        range_overrides: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if not 0.0 <= float(scale):
            raise ValueError(f"randomization scale must be >= 0, got {scale}")
        self.profile = profile
        self.scale = float(scale)
        self.generator = generator
        self.device = torch.device(device)
        self.dtype = dtype
        # UC-87 D4: per-key absolute ``(min, max)`` ranges that take precedence over
        # ``profile.randomization[key]`` in :meth:`bounds`, so training can target a range the
        # profile's nominal->range axis cannot reach (e.g. a fast-Betaflight rate-loop τ) WITHOUT
        # editing the drone profile. Keys use the ``_LAYOUT`` form, e.g. ``rates.roll.tau_s``.
        self.range_overrides: dict[str, tuple[float, float]] = {
            k: (float(v[0]), float(v[1])) for k, v in (range_overrides or {}).items()
        }
        unknown = set(self.range_overrides) - {k for k, _ in _LAYOUT}
        if unknown:
            raise ValueError(f"unknown randomization range_overrides keys {sorted(unknown)}")
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
        iso = {
            "drag.linear_per_s": not isinstance(profile.drag.linear_per_s, tuple),
            "drag.quadratic_per_m": not isinstance(profile.drag.quadratic_per_m, tuple),
        }
        self.randomized = sorted(profile.randomization)
        lo, hi, src = [], [], []
        draws: dict[tuple[str, int], int] = {}
        for key, comp in _LAYOUT:
            klo, khi = self.bounds(key)
            # the c2g column holds the common-mode OFFSET (sample - nominal)
            shift = self.nominal[key][0] if key == "latency.control_to_gyro_s" else 0.0
            lo.append(klo[comp] - shift)
            hi.append(khi[comp] - shift)
            draw_key = (key, 0 if iso.get(key, False) else comp)
            src.append(draws.setdefault(draw_key, len(draws)))
        self._lo = torch.tensor(lo, dtype=dtype, device=self.device)
        self._span = torch.tensor(hi, dtype=dtype, device=self.device) - self._lo
        self._src = torch.tensor(src, dtype=torch.long, device=self.device)
        self._n_draws = len(draws)

    def bounds(self, key: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Scaled ``(lo', hi')`` per component of ``key`` (equal to the nominal if unlisted).

        A ``range_overrides[key]`` (UC-87 D4) replaces ``profile.randomization[key]``'s range; the
        same nominal->range scaling by ``self.scale`` still applies (so ``scale=0`` is nominal).
        """
        nom = self.nominal[key]
        ov = self.range_overrides.get(key)
        if ov is not None:
            rng_min, rng_max = ov
        else:
            rng = self.profile.randomization.get(key)
            if rng is None:
                return nom, nom
            rng_min, rng_max = rng.min, rng.max
        s = self.scale
        lo = tuple(n + s * (rng_min - n) for n in nom)
        hi = tuple(n + s * (rng_max - n) for n in nom)
        return lo, hi

    def _table(self, n: int) -> torch.Tensor:
        u = torch.rand(
            n, self._n_draws, generator=self.generator, device=self.device, dtype=self.dtype
        )
        return self._lo + self._span * u.index_select(1, self._src)

    def sample(self, n: int) -> DynamicsSamples:
        """Fresh samples for ``n`` environments (every listed key is drawn)."""
        return DynamicsSamples(self._table(n))

    def resample(self, current: DynamicsSamples, mask: torch.Tensor) -> None:
        """Redraw ``current`` in place where ``mask`` is set (no host synchronisation)."""
        fresh = self._table(mask.shape[0])
        current.table.copy_(torch.where(mask[:, None], fresh, current.table))
