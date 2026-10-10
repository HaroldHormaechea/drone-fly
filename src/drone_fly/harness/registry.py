"""Name-based registry of harness parts.

Each *kind* (``harness``, ``state_source``, ``frame_source``, ``gate_estimator``,
``free_space_estimator``, ``action_sink``, ``abort_signal``, ``depth_model``) maps names to a
target. A target is normally a lazy ``"module:attr"`` string, imported only when
:func:`resolve` is called, so registering a part never imports its backend. Programmatic
registration may also pass the object itself (handy for tests and notebooks).

Adding a new harness or estimator = implement the interface + :func:`register` it. Built-ins
are registered in :mod:`drone_fly.harness.builtins`. There is deliberately no entry-point
plugin group: registration is explicit Python.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any

from drone_fly.harness.errors import HarnessConfigError, HarnessDependencyError

KINDS: tuple[str, ...] = (
    "harness",
    "state_source",
    "frame_source",
    "gate_estimator",
    "free_space_estimator",
    "action_sink",
    "abort_signal",
    "depth_model",
)


@dataclass(frozen=True)
class Entry:
    """One registered part.

    ``requires`` lists import names probed by ``harness list`` (any one of ``a|b`` suffices);
    ``packages`` maps an import name to its pip distribution name for error messages.
    ``default_config`` (harness kind only) names the packaged YAML the harness overlays.
    """

    kind: str
    name: str
    target: Any
    requires: tuple[str, ...] = ()
    packages: tuple[tuple[str, str], ...] = ()
    description: str = ""
    default_config: str | None = None


_REGISTRY: dict[str, dict[str, Entry]] = {k: {} for k in KINDS}


def _check_kind(kind: str) -> None:
    if kind not in _REGISTRY:
        raise HarnessConfigError(f"unknown part kind {kind!r}; expected one of {', '.join(KINDS)}")


def register(
    kind: str,
    name: str,
    target: Any,
    *,
    requires: tuple[str, ...] | list[str] = (),
    packages: dict[str, str] | None = None,
    description: str = "",
    default_config: str | None = None,
    replace: bool = False,
) -> Entry:
    """Register ``target`` under ``(kind, name)``. Re-registering needs ``replace=True``."""
    _check_kind(kind)
    if isinstance(target, str) and ":" not in target:
        raise HarnessConfigError(f"lazy target {target!r} must be 'module:attr'")
    if name in _REGISTRY[kind] and not replace:
        raise HarnessConfigError(f"{kind} {name!r} is already registered")
    entry = Entry(
        kind=kind,
        name=name,
        target=target,
        requires=tuple(requires),
        packages=tuple(sorted((packages or {}).items())),
        description=description,
        default_config=default_config,
    )
    _REGISTRY[kind][name] = entry
    return entry


def unregister(kind: str, name: str) -> None:
    _check_kind(kind)
    _REGISTRY[kind].pop(name, None)


def names(kind: str) -> list[str]:
    _check_kind(kind)
    return sorted(_REGISTRY[kind])


def entry(kind: str, name: str) -> Entry:
    _check_kind(kind)
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        known = ", ".join(names(kind)) or "none"
        raise HarnessConfigError(f"unknown {kind} {name!r}; registered: {known}") from None


def _package_for(e: Entry, module: str) -> str:
    top = module.split(".")[0]
    return dict(e.packages).get(top, top)


def resolve(kind: str, name: str) -> Any:
    """Import (if lazy) and return the registered object.

    A missing backend surfaces as :class:`HarnessDependencyError` naming the package and the
    ``uv sync --extra harness`` hint.
    """
    e = entry(kind, name)
    if not isinstance(e.target, str):
        return e.target
    module_name, _, attr = e.target.partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or module_name
        if isinstance(exc, HarnessDependencyError):
            raise
        raise HarnessDependencyError(_package_for(e, missing), f"{kind} {name!r}") from exc
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def probe(kind: str, name: str) -> dict[str, Any]:
    """Check (without importing) whether the part's backend packages are installed."""
    e = entry(kind, name)
    missing: list[str] = []
    for req in e.requires:
        alternatives = req.split("|")
        if not any(importlib.util.find_spec(mod) is not None for mod in alternatives):
            missing.append(" or ".join(_package_for(e, m) for m in alternatives))
    return {"available": not missing, "missing": missing}


def build_part(kind: str, name: str, cfg: dict[str, Any], ctx: Any) -> Any:
    """Resolve a part and construct it from its config section.

    The resolved object is called as ``obj.from_config(cfg, ctx)`` when it has that
    classmethod, else ``obj(cfg, ctx)``.
    """
    obj = resolve(kind, name)
    factory = getattr(obj, "from_config", obj)
    return factory(cfg, ctx)
