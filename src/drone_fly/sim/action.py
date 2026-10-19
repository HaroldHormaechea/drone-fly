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

#: Observation-layout contract version (DISTINCT from the action ``CONTRACT_VERSION`` above). v1 is
#: the model-11 layout whose ``extra`` is a clean prefix of ``obs||curr||next||obstacle_vision||
#: raycast`` (extra_dim 12 / 30 / 70). v2 (UC-87) appends the policy-memory block
#: ``prev_action(4)||obs_history(H*12)`` strictly AFTER raycast (the D2 end-append invariant), e.g.
#: extra_dim 98 at H=2. Old checkpoints carry no ``obs_contract_version`` key and are read as v1, so
#: they keep working unchanged (AC2).
OBS_CONTRACT_VERSION = 2
#: The model-11 (obs-contract v1) extra widths — a clean prefix of the fixed layout.
LEGACY_EXTRA_DIMS = (12, 30, 70)


def obs_contract_version(extra_dim: int | None) -> int:
    """The observation-layout contract version an ``extra_dim`` implies: v1 for the model-11 prefix
    widths (12 / 30 / 70), v2 once the UC-87 memory block is appended (any other width)."""
    return 1 if extra_dim in LEGACY_EXTRA_DIMS else OBS_CONTRACT_VERSION


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


# ----------------------------------------------------------------------------- slew limiter (UC-88)


def _is_torch(x: Any) -> bool:
    """True when ``x`` is a torch tensor, without importing torch at module import time."""
    return type(x).__module__.split(".", 1)[0] == "torch"


@dataclass(frozen=True)
class SlewSpec:
    """A controller-realistic command slew-rate limit (UC-88 action filter).

    ``traversal_time_s`` is the time the command would take to cross its FULL per-axis range at the
    cap — rates span ``2*max_rate_rad_s`` and the collective spans ``max_g`` — so the per-axis
    per-step cap is ``full_range_axis * dt / traversal_time_s``. ``None`` DISABLES the filter (the
    default), which is byte-identical to UC-87: the command is passed through unchanged and the
    policy-memory ``prev_applied_raw`` stays equal to the raw intent.
    """

    traversal_time_s: float | None = None

    def __post_init__(self) -> None:
        if self.traversal_time_s is not None:
            v = float(self.traversal_time_s)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(
                    "SlewSpec.traversal_time_s must be a positive, finite number when set, got "
                    f"{self.traversal_time_s!r}"
                )

    @property
    def enabled(self) -> bool:
        return self.traversal_time_s is not None

    @classmethod
    def disabled(cls) -> SlewSpec:
        return cls(None)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> SlewSpec:
        if not d:
            return cls(None)
        unknown = set(d) - {"traversal_time_s"}
        if unknown:
            raise ValueError(f"unknown action_filter keys {sorted(unknown)}")
        t = d.get("traversal_time_s")
        return cls(None if t is None else float(t))

    def as_dict(self) -> dict[str, Any]:
        return {"traversal_time_s": self.traversal_time_s}


def command_full_range(
    spec: ActionSpec, max_g: float | None = None
) -> tuple[float, float, float, float]:
    """Per-axis full command range ``(collective_g, rate_x, rate_y, rate_z)`` the slew cap spans:
    the collective over ``max_g`` and each body rate over ``2*max_rate_rad_s`` (contract v2).
    ``max_g`` overrides the spec's (the sim passes the plant envelope)."""
    if spec.version == CONTRACT_VERSION:
        hi = spec.max_g if max_g is None else max_g
        if hi is None:
            raise ValueError("slew limiting a v2 action needs max_g (the training envelope)")
        mr = 2.0 * float(spec.max_rate_rad_s)
        return (float(hi), mr, mr, mr)
    ref = legacy_reference()
    f = ref.full_stick_rate
    return (ref.max_g, 2.0 * f[0], 2.0 * f[1], 2.0 * f[2])


def raw_of_command(cmd: Any, spec: ActionSpec, *, eps: float = 1e-6) -> Any:
    """Effective-raw inverse of the decoders: the raw action ``a'`` that decodes to ``cmd`` (the
    APPLIED command ``[collective_g, rate_x, rate_y, rate_z]``). v2 only (the UC-87 policy memory
    exists only on v2 checkpoints).

    Collective is the exact affine inverse (``a0' = (cg'-1)/collective_scale_g``); the rates invert
    ``tanh`` with a symmetric ``eps`` guard (``a_i' = atanh(clip(rate_i'/max_rate_rad_s, -1+eps,
    1-eps))``). Supports a single ``(4,)`` command or a batch ``(..., 4)``; torch + numpy parity.
    """
    if spec.version != CONTRACT_VERSION:
        raise ValueError("raw_of_command requires a v2 action contract")
    scale = float(spec.collective_scale_g)
    mr = float(spec.max_rate_rad_s)
    if _is_torch(cmd):
        import torch

        a0 = (cmd[..., 0] - 1.0) / scale
        r = (cmd[..., 1:4] / mr).clamp(-1.0 + eps, 1.0 - eps)
        ai = torch.atanh(r)
        return torch.cat([a0.unsqueeze(-1), ai], dim=-1)
    c = np.asarray(cmd, dtype=np.float32)
    a0 = (c[..., 0] - 1.0) / scale
    r = np.clip(c[..., 1:4] / mr, -1.0 + eps, 1.0 - eps)
    ai = np.arctanh(r)
    return np.concatenate([a0[..., None], ai], axis=-1).astype(np.float32)


def slew_clamp(
    prev_cmd: Any,
    new_cmd: Any,
    dt: Any,
    spec: SlewSpec,
    action_spec: ActionSpec,
    *,
    max_g: float | None = None,
) -> Any:
    """Clamp the per-axis change ``new_cmd - prev_cmd`` to the slew cap ``full_range * dt /
    traversal_time_s``, returning the applied command (same type/shape as ``new_cmd``). torch (sim)
    and numpy (harness) parity. ``dt`` is a scalar, or — for torch — a per-env ``(n,)`` tensor.
    Pass-through when ``spec`` is disabled (byte-identical to no filter)."""
    if not spec.enabled:
        return new_cmd
    full = command_full_range(action_spec, max_g)
    T = float(spec.traversal_time_s)
    if _is_torch(new_cmd):
        import torch

        full_t = torch.as_tensor(full, dtype=new_cmd.dtype, device=new_cmd.device)
        if torch.is_tensor(dt):
            cap = full_t * (dt.to(new_cmd.dtype).reshape(*dt.shape, 1) / T)
        else:
            cap = full_t * (float(dt) / T)
        delta = torch.minimum(torch.maximum(new_cmd - prev_cmd, -cap), cap)
        return prev_cmd + delta
    cap = np.asarray(full, dtype=np.float32) * (float(dt) / T)
    prev = np.asarray(prev_cmd, dtype=np.float32)
    delta = np.clip(np.asarray(new_cmd, dtype=np.float32) - prev, -cap, cap)
    return (prev + delta).astype(np.float32)


class SlewLimiter:
    """Stateful per-axis slew-rate limiter over the APPLIED command ``[collective_g, rate_x,
    rate_y, rate_z]`` for the harness single-env command path (numpy).

    Zero-seeded; ``reset()`` on the same boundaries as the policy-memory ring (run start /
    telemetry discontinuity). ``step`` clamps one command against the carried previous applied
    command and updates the carry. Disabled (``spec`` None) ⇒ pass-through + carry == command.
    """

    def __init__(
        self, spec: SlewSpec, action_spec: ActionSpec, *, max_g: float | None = None
    ) -> None:
        self.spec = spec
        self.action_spec = action_spec
        self.max_g = max_g
        self._prev = np.zeros(4, dtype=np.float32)
        if spec.enabled:  # validate ranges up front (raises on a v2 spec with no max_g)
            command_full_range(action_spec, max_g)

    @property
    def enabled(self) -> bool:
        return self.spec.enabled

    def reset(self) -> None:
        self._prev = np.zeros(4, dtype=np.float32)

    def step(self, command4: Any, dt: float) -> np.ndarray:
        new = np.asarray(command4, dtype=np.float32).reshape(4)
        applied = np.asarray(
            slew_clamp(self._prev, new, dt, self.spec, self.action_spec, max_g=self.max_g),
            dtype=np.float32,
        ).reshape(4)
        self._prev = applied.copy()
        return applied


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
    lineage: dict[str, Any] | None = None,
    obs_contract_version: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``<stem>.contract.json``; ``extra`` adds further keys (UC70 student sidecars:
    ``teacher``, ``noise_model``, ``perception``, ``camera``, ``timing``, ``actor_arch`` ...).

    ``obs_contract_version`` (UC-87) records the observation-layout version; it is only written when
    provided and > 1, so legacy / teacher / student sidecars stay byte-identical.
    """
    data: dict[str, Any] = {
        "contract_version": spec.version,
        "action": {k: v for k, v in spec.as_dict().items() if k != "version"},
        "extra_dim": extra_dim,
        "actor": actor,
        "drone": drone,
        "run_id": run_id,
    }
    if obs_contract_version is not None and obs_contract_version > 1:
        data["obs_contract_version"] = int(obs_contract_version)
    # UC68 teacher keys (absent from legacy sidecars)
    for key, value in (
        ("obs_schema", obs_schema),
        ("world_scale", world_scale),
        ("course_conversion", course_conversion),
        ("lineage", lineage),
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


def slew_spec_for_checkpoint(checkpoint: str | Path) -> SlewSpec:
    """The checkpoint's UC-88 :class:`SlewSpec` from the sidecar ``action_filter`` key (disabled
    when the key or the whole sidecar is absent, so legacy / UC-87 checkpoints are unaffected)."""
    data = read_sidecar(checkpoint)
    if data is None:
        return SlewSpec(None)
    return SlewSpec.from_dict(data.get("action_filter"))
