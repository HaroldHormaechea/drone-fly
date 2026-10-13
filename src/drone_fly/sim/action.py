"""The policy action contract, version 2: hover-relative collective (UC67 AC6).

**v2 (``CONTRACT_VERSION``).** The canonical action is ``[collective_g, roll_rate, pitch_rate,
yaw_rate]``: body-z specific force in g (1.0 = hover on ANY airframe) and body rates in rad/s.
The network's raw output ``a`` is normalised as::

    collective_g = clamp(1 + collective_scale_g * a0, 0, max_g)   # a0 = 0 is hover
    rates        = tanh(a1:4) * max_rate_rad_s

so a readout whose collective bias is 0 starts at hover. ``collective_scale_g`` and
``max_rate_rad_s`` have NO default here: they come from the training config and are written to
the checkpoint's sidecar (``proc-course-dr.yaml`` uses 1.0 g and 4.0 rad/s, i.e. the legacy
semantics on meteor75, so a v1 -> v2 warm start stays affine).

**v1 (legacy).** The pre-UC67 models output ``clip((a0 + 1) / 2, 0, 1)`` as the collective
FRACTION of the reference plant (``drones/meteor75``) and ``tanh(a1:4)`` times its full-stick
rate. The v1 decoders map that to the same canonical action through the reference profile
(``collective_g = thrust_g(fraction)``), loaded from ``drones/meteor75`` at runtime. On meteor75
a v1 action relates to v2 by ``collective_g = 1.25 * a0 + 1.25`` (affine in ``a0``), which is how
a v1 readout can warm-start a v2 one (UC68).

**Sidecar.** ``<checkpoint stem>.contract.json`` beside the ``.pt``::

    {"contract_version": 2, "action": {"collective_scale_g", "max_rate_rad_s", "max_g"},
     "extra_dim": 30, "actor": "k1-reservoir-aug", "drone": "<id>", "run_id": "..."}

A checkpoint without a sidecar is v1. Move the two files together. A UC68 teacher sidecar also
carries ``obs_schema`` (``teacher-obs`` v1, :mod:`drone_fly.sim.teacher_obs`), ``world_scale``
and ``course_conversion`` (the conversion version, rules, parameters and per-map scene hashes);
evaluation refuses a checkpoint whose schema or conversion it does not know.

The torch decoders (sim) and the NumPy decoders (harness) are parity-tested.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_VERSION = 2
LEGACY_VERSION = 1
SUPPORTED_VERSIONS = (LEGACY_VERSION, CONTRACT_VERSION)
#: The reference plant of the v1 contract.
REFERENCE_DRONE = "meteor75"
SIDECAR_SUFFIX = ".contract.json"


@dataclass(frozen=True)
class ActionSpec:
    """How a checkpoint's raw action decodes (see the module docstring)."""

    version: int = LEGACY_VERSION
    collective_scale_g: float | None = None
    max_rate_rad_s: float | None = None
    #: v2: upper clamp of ``collective_g`` (the training drone's ``max_g``); None = no clamp.
    max_g: float | None = None

    def __post_init__(self) -> None:
        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"unsupported action contract version {self.version!r} "
                f"(supported: {list(SUPPORTED_VERSIONS)})"
            )
        if self.version == CONTRACT_VERSION:
            for key in ("collective_scale_g", "max_rate_rad_s"):
                v = getattr(self, key)
                if v is None or not math.isfinite(float(v)) or float(v) <= 0:
                    raise ValueError(f"action contract v2 needs a positive {key}, got {v!r}")

    @classmethod
    def v2(
        cls, collective_scale_g: float, max_rate_rad_s: float, max_g: float | None = None
    ) -> ActionSpec:
        return cls(CONTRACT_VERSION, float(collective_scale_g), float(max_rate_rad_s), max_g)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ActionSpec:
        d = dict(d or {})
        version = int(d.pop("version", d.pop("contract_version", LEGACY_VERSION)))
        if version == LEGACY_VERSION:
            return cls()
        unknown = set(d) - {"collective_scale_g", "max_rate_rad_s", "max_g"}
        if unknown:
            raise ValueError(f"unknown action keys {sorted(unknown)}")
        mg = d.get("max_g")
        return cls(
            version,
            d.get("collective_scale_g"),
            d.get("max_rate_rad_s"),
            None if mg is None else float(mg),
        )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"version": self.version}
        if self.version == CONTRACT_VERSION:
            out.update(
                collective_scale_g=self.collective_scale_g,
                max_rate_rad_s=self.max_rate_rad_s,
                max_g=self.max_g,
            )
        return out

    def collective_bias(self, reference: LegacyReference | None = None) -> float:
        """Readout bias of the collective output that commands hover (1.0 g)."""
        if self.version == CONTRACT_VERSION:
            return 0.0
        ref = reference or legacy_reference()
        return 2.0 * ref.hover_fraction - 1.0


@dataclass(frozen=True)
class LegacyReference:
    """The v1 reference plant: thrust curve and full-stick rates of ``drones/meteor75``."""

    knots_throttle: tuple[float, ...]
    knots_g: tuple[float, ...]
    full_stick_rate: tuple[float, float, float]
    hover_fraction: float

    @property
    def max_g(self) -> float:
        return float(self.knots_g[-1])

    def thrust_g(self, fraction: Any) -> Any:
        return np.interp(fraction, self.knots_throttle, self.knots_g)

    def fraction_for_g(self, accel_g: Any) -> Any:
        return np.interp(accel_g, self.knots_g, self.knots_throttle)

    @classmethod
    def from_profile(cls, profile: Any) -> LegacyReference:
        from drone_fly.sysid.model import betaflight_rate_rad

        full = tuple(
            float(betaflight_rate_rad(1.0, r.rc_rate, r.super_rate, r.expo))
            for r in (profile.rates[a] for a in ("roll", "pitch", "yaw"))
        )
        return cls(
            tuple(profile.thrust_curve.throttle),
            tuple(profile.thrust_curve.accel_g),
            full,  # type: ignore[arg-type]
            float(profile.hover_throttle),
        )


@lru_cache(maxsize=4)
def legacy_reference(root: str | None = None) -> LegacyReference:
    """The v1 reference (``drones/<REFERENCE_DRONE>``), loaded once per drones root."""
    from drone_fly.drones import load

    return LegacyReference.from_profile(load(REFERENCE_DRONE, root))


# ----------------------------------------------------------------------------- torch (sim)


def decode_torch(
    raw: Any,
    spec: ActionSpec,
    *,
    max_g: float | None = None,
    reference: LegacyReference | None = None,
) -> tuple[Any, Any]:
    """Batched ``(collective_g (n,), rates (n, 3))`` from raw actions ``(n, 4)``."""
    import torch

    a0 = raw[:, 0]
    if spec.version == LEGACY_VERSION:
        ref = reference or legacy_reference()
        frac = ((a0 + 1) * 0.5).clamp(0.0, 1.0)
        xp = torch.as_tensor(ref.knots_throttle, dtype=raw.dtype, device=raw.device)
        fp = torch.as_tensor(ref.knots_g, dtype=raw.dtype, device=raw.device)
        from drone_fly.sim.plant import interp

        cg = interp(frac, xp, fp)
        full = torch.as_tensor(ref.full_stick_rate, dtype=raw.dtype, device=raw.device)
        return cg, torch.tanh(raw[:, 1:4]) * full
    hi = spec.max_g if max_g is None else max_g
    cg = 1.0 + float(spec.collective_scale_g) * a0
    cg = cg.clamp(min=0.0) if hi is None else cg.clamp(0.0, float(hi))
    return cg, torch.tanh(raw[:, 1:4]) * float(spec.max_rate_rad_s)


# ----------------------------------------------------------------------------- numpy (harness)


@dataclass(frozen=True)
class Decoded:
    """One decoded action (float32 rates)."""

    collective_g: float
    rates: np.ndarray
    #: The reference-plant collective fraction (v1: the raw clip; v2: derived on meteor75).
    fraction: float


def decode_numpy(
    raw: Any,
    spec: ActionSpec,
    *,
    max_g: float | None = None,
    reference: LegacyReference | None = None,
) -> Decoded:
    """Single raw action ``(4,)`` -> :class:`Decoded` (the harness path)."""
    a = np.asarray(raw, dtype=np.float32).reshape(-1)
    if a.shape[0] != 4:
        raise ValueError(f"raw action must have 4 entries, got {a.shape[0]}")
    ref = reference or legacy_reference()
    if spec.version == LEGACY_VERSION:
        frac = float(np.clip((a[0] + 1) * 0.5, 0, 1))
        rates = (np.tanh(a[1:4]) * np.asarray(ref.full_stick_rate)).astype(np.float32)
        return Decoded(float(ref.thrust_g(frac)), rates, frac)
    hi = spec.max_g if max_g is None else max_g
    cg = 1.0 + float(spec.collective_scale_g) * float(a[0])
    cg = max(cg, 0.0) if hi is None else min(max(cg, 0.0), float(hi))
    rates = (np.tanh(a[1:4]) * float(spec.max_rate_rad_s)).astype(np.float32)
    frac = float(np.clip(ref.fraction_for_g(cg), 0.0, 1.0))
    return Decoded(float(cg), rates, frac)


# ----------------------------------------------------------------------------- sidecar


def sidecar_path(checkpoint: str | Path) -> Path:
    p = Path(checkpoint)
    return p.with_name(p.stem + SIDECAR_SUFFIX)


def write_sidecar(
    checkpoint: str | Path,
    spec: ActionSpec,
    *,
    extra_dim: int | None,
    actor: str,
    drone: str | None,
    run_id: str | None,
    obs_schema: dict[str, Any] | None = None,
    world_scale: float | None = None,
    course_conversion: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``<stem>.contract.json``; ``extra`` adds further keys (UC70 student sidecars:
    ``teacher``, ``noise_model``, ``perception``, ``camera``, ``timing``, ``actor_arch`` ...)."""
    data: dict[str, Any] = {
        "contract_version": spec.version,
        "action": {k: v for k, v in spec.as_dict().items() if k != "version"},
        "extra_dim": extra_dim,
        "actor": actor,
        "drone": drone,
        "run_id": run_id,
    }
    # UC68 teacher keys (absent from legacy sidecars)
    for key, value in (
        ("obs_schema", obs_schema),
        ("world_scale", world_scale),
        ("course_conversion", course_conversion),
    ):
        if value is not None:
            data[key] = value
    for key, value in (extra or {}).items():
        data[key] = value
    path = sidecar_path(checkpoint)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def read_sidecar(checkpoint: str | Path) -> dict[str, Any] | None:
    """The sidecar mapping, or None when the checkpoint has none (v1)."""
    path = sidecar_path(checkpoint)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a contract sidecar must be a JSON object")
    return data


def spec_for_checkpoint(checkpoint: str | Path) -> ActionSpec:
    """The checkpoint's :class:`ActionSpec` (no sidecar = v1)."""
    data = read_sidecar(checkpoint)
    if data is None:
        return ActionSpec()
    return ActionSpec.from_dict(
        {"version": data.get("contract_version", LEGACY_VERSION), **(data.get("action") or {})}
    )
