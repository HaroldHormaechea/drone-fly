"""The perception-provider contract (UC-86): what any "eyes" plugin promises the harness.

A *perception provider* is a harness part that turns the screen frame and the telemetry state
into the contract blocks the brain consumes. There are two runner slots
(:class:`~drone_fly.harness.interfaces.GateEstimator` and
:class:`~drone_fly.harness.interfaces.FreeSpaceEstimator`); both roles derive from
:class:`PerceptionProvider`, which declares the contract below. The brain's input layout
(``assembler.EXTRA_LAYOUT``) never changes: providers convert into it.

Contract (version :data:`CONTRACT_VERSION`)
-------------------------------------------
* **Blocks** -- ``provides`` names blocks from :data:`BLOCKS` (shape, dtype, range, frame and
  units per block) and only blocks of the provider's role (:data:`ROLE_BLOCKS`).
* **stamp** -- the capture time (``perf_counter``) of the newest input the estimate was built
  from. A frame-based free-space provider stamps ``-inf`` until it has a valid estimate, so the
  runner's ``max_estimate_age_s`` check fails safe; an invalid output (startup, a history gap)
  never replaces the held estimate, and a gap (no new frame) holds the stamp. Exceptions: a gate
  estimator stamps ``now`` on a tick without a frame (``frame.stamp`` otherwise), and constant
  providers (``none``, the loopback / oracle ground truth) stamp ``now``.
* **provenance** -- where the environment-aware blocks come from: ``screen`` (pixels on screen),
  ``telemetry`` (the game's telemetry stream, or a constant that reads no environment data) or
  ``metadata`` (track files, labels, ground truth). An undeclared provenance counts as
  ``metadata`` for the live screen-only guard (fail closed).
* **Lifecycle** -- ``start()`` (may start a worker thread) / ``update()`` (synchronous, never
  blocks on inference) / ``reset()`` (forget held state on a telemetry discontinuity) /
  ``stop()`` (joins its threads). Async providers own one thread each; a ``reset()`` while the
  worker runs is applied on the worker thread.
* **CUDA** -- an async provider on CUDA runs its worker (and warm-up) on a dedicated CUDA stream so
  it never serialises behind the policy on the default stream; only host NumPy results cross
  threads. Its GPU duty (forward time x rate) should stay <= 0.5; declare it in ``gpu_duty``.
* **info** -- optional ``frame_index`` and ``inference_s`` (per-frame wall time) feed the
  per-provider latency rows.

:func:`check_spec` validates the declaration, :func:`validate_estimate` an output,
:func:`check_session` the whole perception set-up at session build (unused providers, the
screen-only guard) and :class:`PerceptionRuntime` validates the first estimates live and records
per-provider latency and age. numpy + stdlib only.
"""

from __future__ import annotations

import logging
import math
import numbers
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from drone_fly.harness.errors import PerceptionContractError

log = logging.getLogger(__name__)

#: Bumped whenever a block, its meaning or the stamp semantics change.
CONTRACT_VERSION = 1
PROVENANCES = ("screen", "telemetry", "metadata")
REQUIRE_PROVENANCE = ("auto", "screen", "any")


@dataclass(frozen=True)
class BlockSpec:
    """One contract block a provider can supply."""

    name: str
    width: int
    dtype: str = "float32"
    lo: float | None = None
    hi: float | None = None
    frame: str = ""
    units: str = ""
    description: str = ""
    #: False for blocks that are not read from the estimate record (``oracle``: shared state).
    in_estimate: bool = True


_BLOCK_LIST = (
    BlockSpec(
        "target",
        3,
        frame="contract world (right-handed, Z-up)",
        units="training-scale m",
        description="current gate centre; the assembler turns it into obs 'rel'",
    ),
    BlockSpec(
        "curr_gate",
        3,
        frame="world yaw of the current gate",
        units="[cos yaw, sin yaw, aperture m]",
        description="INFERENCE_CONTRACT.md / course_senses.curr_gate_block",
    ),
    BlockSpec(
        "next_lookahead",
        7,
        frame="body/world per course_senses.next_lookahead_block",
        units="bearing(3), orient(2), dist/10, vis",
        description="next gate lookahead (course_senses.next_lookahead_block)",
    ),
    BlockSpec(
        "obstacle_vision",
        8,
        frame="body-relative, training scale",
        units="training-scale m",
        description="the 2 nearest obstacles' offset + size (course_senses.obstacle_vision_block)",
    ),
    BlockSpec(
        "raycast",
        40,
        lo=0.0,
        hi=1.0,
        frame="contract CONE ray order",
        units="fraction of RAY_RANGE",
        description="40-ray free-space scan, 1 = clear to RAY_RANGE",
    ),
    BlockSpec(
        "oracle",
        0,
        frame="",
        units="",
        description="teacher-only exact track state (shared with the teacher assembler)",
        in_estimate=False,
    ),
)
BLOCKS: dict[str, BlockSpec] = {b.name: b for b in _BLOCK_LIST}

#: Perception block names per runner slot (``target`` feeds ``rel`` in the obs).
GATE_BLOCKS = frozenset({"target", "curr_gate", "next_lookahead"})
FREE_SPACE_BLOCKS = frozenset({"obstacle_vision", "raycast"})
ROLE_BLOCKS: dict[str, frozenset[str]] = {
    "gate_estimator": GATE_BLOCKS | {"oracle"},
    "free_space_estimator": FREE_SPACE_BLOCKS,
}
IN_FOV_WIDTH = BLOCKS["raycast"].width
_TOL = 1e-6


class PerceptionProvider:
    """Mixin declaring the perception contract (see the module docstring).

    Class attributes may be overridden per instance (e.g. ``provides`` depends on config).
    """

    #: Blocks this provider fills (names from :data:`BLOCKS`).
    provides: frozenset[str] = frozenset()
    contract_version: int = CONTRACT_VERSION
    #: ``screen`` | ``telemetry`` | ``metadata``; ``None`` (undeclared) is treated as metadata.
    provenance: str | None = None
    #: True: providing blocks the model does not consume is not an error (e.g. ``none``).
    optional: bool = False
    #: Declared worst-case GPU duty of an async CUDA worker (forward time x rate), if any.
    gpu_duty: float | None = None

    def start(self) -> None:  # noqa: B027 - optional hook
        """Start a worker thread, if the implementation uses one."""

    def stop(self) -> None:  # noqa: B027 - optional hook
        """Stop (and join) the worker thread."""

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Forget held / tracking state (called on a telemetry discontinuity)."""

    def describe(self) -> dict[str, Any]:
        """The declared contract as a small JSON-able dict (recorded in the run meta)."""
        provides = sorted(getattr(self, "provides", ()) or ())
        return {
            "contract_version": getattr(self, "contract_version", None),
            "provenance": getattr(self, "provenance", None),
            "optional": bool(getattr(self, "optional", False)),
            "gpu_duty": getattr(self, "gpu_duty", None),
            "blocks": {b: asdict(BLOCKS[b]) for b in provides if b in BLOCKS},
        }


def provider_name(provider: Any) -> str:
    """The configured name of a provider (``provider_name``), else its class name."""
    return getattr(provider, "provider_name", None) or type(provider).__name__


def describe(provider: Any) -> dict[str, Any]:
    fn = getattr(provider, "describe", None)
    if callable(fn):
        return fn()
    return PerceptionProvider.describe(provider)


def effective_provenance(provider: Any) -> str:
    """The provenance the guard uses: undeclared (or invalid) counts as ``metadata``."""
    p = getattr(provider, "provenance", None)
    return p if p in PROVENANCES else "metadata"


def check_spec(provider: Any, role: str, name: str | None = None) -> None:
    """Validate a provider's declaration for ``role``; raise :class:`PerceptionContractError`."""
    name = name or provider_name(provider)
    if role not in ROLE_BLOCKS:
        raise PerceptionContractError(name, None, f"unknown perception role {role!r}")
    version = getattr(provider, "contract_version", CONTRACT_VERSION)
    if version != CONTRACT_VERSION:
        raise PerceptionContractError(
            name, None,
            f"declares contract version {version!r}; this harness speaks {CONTRACT_VERSION}",
        )  # fmt: skip
    prov = getattr(provider, "provenance", None)
    if prov is not None and prov not in PROVENANCES:
        raise PerceptionContractError(
            name, None, f"provenance {prov!r} is not one of {', '.join(PROVENANCES)}"
        )
    for block in sorted(getattr(provider, "provides", ()) or ()):
        if block not in BLOCKS:
            raise PerceptionContractError(
                name, block, f"is not a contract block (known: {', '.join(BLOCKS)})"
            )
        if block not in ROLE_BLOCKS[role]:
            raise PerceptionContractError(
                name, block,
                f"is not a {role} block (allowed: {', '.join(sorted(ROLE_BLOCKS[role]))})",
            )  # fmt: skip


def _check_stamp(name: str, stamp: Any) -> None:
    if isinstance(stamp, bool) or not isinstance(stamp, numbers.Real):
        raise PerceptionContractError(name, "stamp", f"must be a real number, got {stamp!r}")
    s = float(stamp)
    if math.isnan(s) or s == math.inf:
        raise PerceptionContractError(name, "stamp", f"must not be NaN or +inf, got {s}")


def validate_estimate(name: str, provider: Any, estimate: Any, role: str) -> None:
    """Check ``estimate`` against every block ``provider`` declares; raise on the first mismatch.

    Checks per block: present, length == width, a floating dtype, all finite, inside the
    declared range. Also the ``stamp`` (a real number, ``-inf`` allowed) and, when present, the
    ``in_fov_mask`` (bool-like, 40 wide).
    """
    for block in sorted(getattr(provider, "provides", ()) or ()):
        spec = BLOCKS.get(block)
        if spec is None or not spec.in_estimate:
            continue
        value = getattr(estimate, block, None)
        if value is None:
            raise PerceptionContractError(name, block, "is declared but missing from the estimate")
        arr = np.asarray(value)
        if arr.dtype.kind != "f":
            raise PerceptionContractError(
                name, block, f"must be a floating array, got dtype {arr.dtype}"
            )
        flat = arr.reshape(-1)
        if flat.shape[0] != spec.width:
            raise PerceptionContractError(
                name, block, f"must have {spec.width} values, got shape {arr.shape}"
            )
        if not np.all(np.isfinite(flat)):
            raise PerceptionContractError(name, block, "contains non-finite values")
        if spec.lo is not None and float(flat.min()) < spec.lo - _TOL:
            raise PerceptionContractError(
                name, block, f"value {float(flat.min()):g} below the range [{spec.lo}, {spec.hi}]"
            )
        if spec.hi is not None and float(flat.max()) > spec.hi + _TOL:
            raise PerceptionContractError(
                name, block, f"value {float(flat.max()):g} above the range [{spec.lo}, {spec.hi}]"
            )
    mask = getattr(estimate, "in_fov_mask", None)
    if mask is not None:
        m = np.asarray(mask)
        if m.dtype.kind not in "biuf" or m.reshape(-1).shape[0] != IN_FOV_WIDTH:
            raise PerceptionContractError(
                name, "in_fov_mask", f"must be bool-like with {IN_FOV_WIDTH} values, got "
                f"dtype {m.dtype} shape {m.shape}",
            )  # fmt: skip
        if m.dtype.kind != "b" and not np.all((m == 0) | (m == 1)):
            raise PerceptionContractError(name, "in_fov_mask", "must hold only 0/1 values")
    _check_stamp(name, getattr(estimate, "stamp", None))


# ----------------------------------------------------------------------------- session checks


@dataclass(frozen=True)
class InputsConfig:
    """The ``inputs:`` config section."""

    require_provenance: str = "auto"
    free_space: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_config(cls, cfg: Any) -> InputsConfig:
        from drone_fly.harness.config import require_choice
        from drone_fly.harness.errors import HarnessConfigError

        data = cfg.get("inputs") if cfg is not None else None
        data = data or {}
        if not isinstance(data, dict):
            raise HarnessConfigError("config 'inputs' must be a mapping")
        req = require_choice(
            data.get("require_provenance", "auto"), "inputs.require_provenance", REQUIRE_PROVENANCE
        )
        fs = data.get("free_space") or []
        if not isinstance(fs, (list, tuple)):
            raise HarnessConfigError("config 'inputs.free_space' must be a list")
        return cls(req, tuple(fs))

    def screen_only(self, harness: str | None) -> bool:
        if self.require_provenance == "auto":
            return harness == "liftoff"
        return self.require_provenance == "screen"


def session_providers(harness: Any) -> list[tuple[str, str, Any]]:
    """``(role, name, provider)`` for the gate, free-space and composite-child providers."""
    out: list[tuple[str, str, Any]] = [
        ("gate_estimator", provider_name(harness.gate_estimator), harness.gate_estimator)
    ]
    fs = harness.free_space_estimator
    if fs is not None:
        out.append(("free_space_estimator", provider_name(fs), fs))
        children = getattr(fs, "children", None)
        if callable(children):
            for child in children():
                out.append(("free_space_estimator", provider_name(child), child))
    return out


def check_session(harness: Any, assembler: Any, extra_dim: int | None, cfg: Any) -> dict[str, Any]:
    """Refuse a perception set-up that cannot be right (AC2 / AC5); return the meta record.

    Per provider: :func:`check_spec`; a provider none of whose blocks the model consumes is
    refused unless ``optional`` (partially unconsumed blocks are logged); with the screen-only
    guard on, ``metadata`` / undeclared provenance is refused.
    """
    inputs = InputsConfig.from_config(cfg)
    harness_name = getattr(cfg, "harness", None)
    screen = inputs.screen_only(harness_name)
    required = set(assembler.required_blocks(extra_dim))
    meta: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "require_provenance": inputs.require_provenance,
        "screen_only": screen,
        "providers": {},
    }
    fs = harness.free_space_estimator
    owned = getattr(fs, "owned_blocks", None)
    for role, name, provider in session_providers(harness):
        check_spec(provider, role, name)
        provides = set(getattr(provider, "provides", ()) or ())
        if callable(owned) and provider is not fs and provider is not harness.gate_estimator:
            provides = set(owned(provider))  # a composite child: the blocks it won
        used = provides & required
        unused = sorted(provides - required)
        if provides and not used and not getattr(provider, "optional", False):
            raise PerceptionContractError(
                name, ", ".join(sorted(provides)),
                f"none of these blocks are consumed by the loaded model (extra_dim={extra_dim} "
                f"reads {', '.join(sorted(required))}). Select a provider for the blocks the model "
                f"needs, or a model whose extra_dim consumes them",
            )  # fmt: skip
        if unused and used:
            log.info("perception provider %s: blocks %s are not consumed", name, unused)
        prov = effective_provenance(provider)
        if screen and prov == "metadata":
            declared = getattr(provider, "provenance", None)
            what = "undeclared provenance" if declared is None else f"provenance {declared!r}"
            raise PerceptionContractError(
                name, ", ".join(sorted(provides)) or "-",
                f"has {what}, but this live session only accepts screen- or telemetry-derived "
                f"perception (inputs.require_provenance: {inputs.require_provenance}). Use an "
                f"on-screen provider, or set inputs.require_provenance: any for an offline / "
                f"diagnostic run",
            )  # fmt: skip
        meta["providers"][name] = {"role": role, **describe(provider), "unused_blocks": unused}
    return meta


# ----------------------------------------------------------------------------- runtime


class PerceptionRuntime:
    """Per-tick validation of the first estimates + per-provider latency / age rows (AC2, AC10).

    Every estimate of a provider is validated up to and including its first one with a finite
    stamp (so the ``-inf`` placeholders before the first valid estimate never end validation).
    Called by the runner after the perception stage; read-only on the estimates.
    """

    def __init__(self, harness: Any) -> None:
        self.harness = harness
        self.gate = harness.gate_estimator
        self.free_space = harness.free_space_estimator
        self.gate_name = provider_name(self.gate)
        self.fs_name = None if self.free_space is None else provider_name(self.free_space)
        self._validated: set[str] = set()
        self._frames: dict[str, Any] = {}

    def _validate(self, name: str, provider: Any, estimate: Any, role: str) -> None:
        if name in self._validated:
            return
        validate_estimate(name, provider, estimate, role)
        if math.isfinite(float(estimate.stamp)):
            self._validated.add(name)

    def _rows(self, record: Any, name: str, wall_s: float | None, stamp: Any, info: dict,
              now: float) -> None:  # fmt: skip
        key = f"provider:{name}"
        if wall_s is not None:
            record.perception_s[key] = float(wall_s)
        if stamp is not None:
            record.ages_s[key] = float(now - float(stamp))
        inference_s = (info or {}).get("inference_s")
        frame_index = (info or {}).get("frame_index")
        if inference_s is not None and frame_index is not None:
            if self._frames.get(name) != frame_index:
                self._frames[name] = frame_index
                record.perception_s[f"{key}:frame"] = float(inference_s)

    def after_update(
        self, record: Any, gate: Any, fs: Any, gate_wall_s: float | None,
        fs_wall_s: float | None, now: float,
    ) -> None:  # fmt: skip
        self._validate(self.gate_name, self.gate, gate, "gate_estimator")
        self._rows(record, self.gate_name, gate_wall_s, gate.stamp, gate.info, now)
        if fs is None or self.free_space is None:
            return
        last = getattr(self.free_space, "last_estimates", None)
        children = getattr(self.free_space, "children", None)
        if callable(children):
            estimates = last() if callable(last) else {}
            providers = (fs.info or {}).get("providers") or {}
            for child in children():
                cname = provider_name(child)
                if cname in estimates:
                    self._validate(cname, child, estimates[cname], "free_space_estimator")
                entry = providers.get(cname) or {}
                self._rows(record, cname, entry.get("wall_s"), entry.get("stamp"),
                           entry, now)  # fmt: skip
        self._validate(self.fs_name, self.free_space, fs, "free_space_estimator")
        self._rows(record, self.fs_name, fs_wall_s, fs.stamp, fs.info, now)
