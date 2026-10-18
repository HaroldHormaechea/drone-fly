"""Name-based registry of harness parts.

Each *kind* (``harness``, ``state_source``, ``frame_source``, ``gate_estimator``,
``free_space_estimator``, ``action_sink``, ``abort_signal``, ``depth_model``, ``focus_probe``)
maps names to a
target. A target is normally a lazy ``"module:attr"`` string, imported only when
:func:`resolve` is called, so registering a part never imports its backend. Programmatic
registration may also pass the object itself (handy for tests and notebooks).

Adding a new harness or estimator = implement the interface + :func:`register` it. Built-ins
are registered in :mod:`drone_fly.harness.builtins`.

:func:`build_part` also accepts names that are not registered (UC-86 pluggable perception):

1. a registered name builds as always;
2. otherwise a name containing ``:`` is an import reference ``module.path:Attr``;
3. otherwise, for the perception kinds (``gate_estimator`` / ``free_space_estimator``), the name
   is looked up in the :data:`ENTRY_POINT_GROUP` entry points (lookup only: nothing is registered,
   and ``harness list`` shows registered parts only);
4. otherwise the usual "unknown ... registered: ..." error.

An import-reference or entry-point perception part must be an instance of its role's ABC.
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
    "focus_probe",
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

#: Python entry-point group searched for unregistered perception provider names.
ENTRY_POINT_GROUP = "drone_fly.perception"
#: Part kinds that are perception providers (``drone_fly.harness.perception``).
PERCEPTION_KINDS = ("gate_estimator", "free_space_estimator")


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


def _import_reference(kind: str, ref: str) -> Any:
    """Import ``module.path:Attr``. A missing package -> :class:`HarnessDependencyError`."""
    module_name, _, attr = ref.partition(":")
    if not module_name or not attr:
        raise HarnessConfigError(f"{kind} {ref!r}: an import reference must be 'module.path:Attr'")
    try:
        obj: Any = importlib.import_module(module_name)
    except HarnessDependencyError:
        raise
    except ModuleNotFoundError as exc:
        missing = exc.name or module_name
        own = missing == module_name or module_name.startswith(f"{missing}.")
        if own and "." in missing:
            raise HarnessConfigError(f"{kind} {ref!r}: module {missing!r} not found") from exc
        raise HarnessDependencyError(missing.split(".")[0], f"{kind} {ref!r}") from exc
    except Exception as exc:  # noqa: BLE001 - name the reference whatever the module raised
        raise HarnessConfigError(
            f"{kind} {ref!r}: importing {module_name!r} failed: {exc!r}"
        ) from exc
    try:
        for part in attr.split("."):
            obj = getattr(obj, part)
    except AttributeError as exc:
        raise HarnessConfigError(f"{kind} {ref!r}: {module_name!r} has no {attr!r}") from exc
    return obj


def _entry_point(kind: str, name: str) -> Any | None:
    from importlib.metadata import entry_points

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            try:
                return ep.load()
            except ImportError as exc:
                if isinstance(exc, HarnessDependencyError):
                    raise
                missing = getattr(exc, "name", None) or ep.value.split(":")[0]
                raise HarnessDependencyError(
                    missing.split(".")[0], f"{kind} {name!r} (entry point {ep.value})"
                ) from exc
    return None


def _check_role(kind: str, name: str, part: Any) -> None:
    from drone_fly.harness import interfaces

    role = {
        "gate_estimator": interfaces.GateEstimator,
        "free_space_estimator": interfaces.FreeSpaceEstimator,
    }[kind]
    if not isinstance(part, role):
        raise HarnessConfigError(
            f"{kind} {name!r} built a {type(part).__name__}, which is not a {role.__name__} "
            "(subclass drone_fly.harness.interfaces." + role.__name__ + ")"
        )


def build_part(kind: str, name: str, cfg: dict[str, Any], ctx: Any) -> Any:
    """Resolve a part and construct it from its config section.

    The resolved object is called as ``obj.from_config(cfg, ctx)`` when it has that
    classmethod, else ``obj(cfg, ctx)``. ``name`` may also be an import reference or (perception
    kinds) an entry-point name; see the module docstring. A built perception part gets
    ``provider_name = name``.
    """
    _check_kind(kind)
    plugin = False
    if name in _REGISTRY[kind]:
        obj = resolve(kind, name)
    elif ":" in name:
        obj, plugin = _import_reference(kind, name), True
    else:
        obj = _entry_point(kind, name) if kind in PERCEPTION_KINDS else None
        if obj is None:
            entry(kind, name)  # raises the usual "unknown ... registered: ..." error
        plugin = True
    factory = getattr(obj, "from_config", obj)
    part = factory(cfg, ctx)
    if kind in PERCEPTION_KINDS:
        if plugin:
            _check_role(kind, name, part)
        try:
            part.provider_name = name
        except AttributeError:
            pass
    return part
