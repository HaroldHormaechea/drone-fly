"""The ONLY bridge to ``gpu_prototype/`` — read-only, by reference, never reimplemented.

``gpu_prototype`` modules are top-level siblings (not a package) with import-time traps:

* ``hover_gpu.dev = "cuda"`` is a module global, and ``gates_gpu``, ``gates_k1_reservoir`` and
  ``reservoir_aug`` bind ``dev`` *by value* when imported; ``gates_gpu`` / ``gates_proc_env``
  build tensors on it at import. So the device must be overridden on ``hover_gpu`` **before**
  anything else imports it — otherwise a CPU-only machine fails at import.
* ``reservoir_aug.K1_PATH`` is an overridable module global (the augmented actor reads it at
  construction); ``gates_k1_reservoir.K1Reservoir`` hardcodes its path, so the pure reservoir
  (``extra_dim=None``) is refused by the harness.

:func:`load_contract` handles both, then returns a :class:`ContractModule` facade that re-exports
the contract (``load_policy``, ``make_obs``, ``plant_step``, ``raycast``, ``init_state``,
``quat_rotate``/``quat_rotate_inv``, ``CONE``, ``N_RAYS``, ``RAY_RANGE``, ``MAX_BODY_RATE``,
``MAX_THRUST``, ``HOVER_THR``, ``DT``) by reference. ``import_reference`` is the only way to
reach ``gates_proc_env`` / ``course_gen`` (reference maths for drift tests and course sampling).

The device is process-global once these modules are imported, so a second call with the same
paths and device is an idempotent no-op; a conflicting call raises.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from drone_fly.harness.config import find_repo_root
from drone_fly.harness.errors import HarnessConfigError, lazy_import

log = logging.getLogger(__name__)

GPU_PROTOTYPE_ENV = "DRONE_FLY_GPU_PROTOTYPE"
K1_PATH_ENV = "DRONE_FLY_K1_PATH"

#: gpu_prototype modules whose ``dev`` binding / location the loader guards.
GUARDED_MODULES = (
    "hover_gpu",
    "gates_gpu",
    "gates_k1_reservoir",
    "reservoir_aug",
    "gates_proc_env",
    "course_gen",
    "inference",
)
REFERENCE_MODULES = frozenset({"gates_proc_env", "course_gen"})
_REQUIRED_FILES = ("inference.py", "hover_gpu.py", "reservoir_aug.py", "gates_k1_reservoir.py")


@dataclass
class ContractModule:
    """Facade over ``gpu_prototype/inference.py`` (re-exports by reference)."""

    inference: ModuleType
    device: str
    gpu_prototype_dir: Path
    k1_path: Path | None
    _refs: dict[str, ModuleType] = field(default_factory=dict, repr=False)

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes not found normally: forward to inference.py.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return getattr(self.inference, name)
        except AttributeError:
            raise AttributeError(f"inference contract has no attribute {name!r}") from None

    def import_reference(self, name: str) -> ModuleType:
        """Import a reference module (``gates_proc_env`` or ``course_gen``) read-only."""
        if name not in REFERENCE_MODULES:
            raise HarnessConfigError(
                f"{name!r} is not an allowed reference module ({sorted(REFERENCE_MODULES)})"
            )
        if name not in self._refs:
            mod = importlib.import_module(name)
            _check_location(mod, self.gpu_prototype_dir)
            self._refs[name] = mod
        return self._refs[name]


_LOADED: ContractModule | None = None


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _check_location(mod: ModuleType, root: Path) -> None:
    f = getattr(mod, "__file__", None)
    if f is None or not _within(Path(f), root):
        raise HarnessConfigError(
            f"module {mod.__name__!r} resolved to {f}, outside the configured gpu_prototype "
            f"dir {root}. Another module with the same name shadows the contract; fix sys.path."
        )


def resolve_gpu_prototype_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """CLI/config value -> ``$DRONE_FLY_GPU_PROTOTYPE`` -> ``<repo-root>/gpu_prototype``."""
    raw = explicit or os.environ.get(GPU_PROTOTYPE_ENV)
    if raw:
        p = Path(raw).expanduser()
    else:
        root = find_repo_root()
        if root is None:
            raise HarnessConfigError(
                "cannot locate gpu_prototype/: pass --gpu-prototype-dir, set "
                f"contract.gpu_prototype_dir, or set {GPU_PROTOTYPE_ENV}"
            )
        p = root / "gpu_prototype"
    missing = [f for f in _REQUIRED_FILES if not (p / f).is_file()]
    if missing:
        raise HarnessConfigError(
            f"{p} is not a usable gpu_prototype dir (missing {', '.join(missing)}); pass "
            f"--gpu-prototype-dir or set {GPU_PROTOTYPE_ENV}"
        )
    return p.resolve()


def resolve_k1_path(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """CLI/config value -> ``$DRONE_FLY_K1_PATH`` -> the data root's ``pruned/k1`` ->
    ``<repo-root>/artifacts/pruned/k1`` (legacy).

    An explicit (CLI/config/env) path must exist. The repo default may be absent (the K1
    connectome is not committed to every checkout): then ``None`` is returned and loading a
    policy fails later with an actionable error.
    """
    raw = explicit or os.environ.get(K1_PATH_ENV)
    if raw:
        p = Path(raw).expanduser()
        if not p.exists():
            raise HarnessConfigError(
                f"K1 connectome path {p} does not exist; pass --k1-path or set {K1_PATH_ENV}"
            )
        return p.resolve()
    from drone_fly import paths

    if (paths.pruned_root() / "k1").exists():
        return (paths.pruned_root() / "k1").resolve()
    root = find_repo_root()
    if root is not None and (root / "artifacts" / "pruned" / "k1").exists():
        return (root / "artifacts" / "pruned" / "k1").resolve()
    return None


def _resolve_device(requested: str) -> str:
    torch = lazy_import("torch", purpose="loading the K1 flight model")
    has_cuda = bool(torch.cuda.is_available())
    if requested in (None, "", "auto"):
        return "cuda" if has_cuda else "cpu"
    if requested == "cuda" and not has_cuda:
        log.warning("CUDA requested but not available; falling back to CPU")
        return "cpu"
    if requested not in ("cpu", "cuda"):
        raise HarnessConfigError(f"device must be auto, cpu or cuda, got {requested!r}")
    return requested


def load_contract(
    gpu_prototype_dir: str | os.PathLike[str] | None = None,
    k1_path: str | os.PathLike[str] | None = None,
    device: str = "auto",
) -> ContractModule:
    """Import the inference contract with a safe device override; see the module docstring."""
    global _LOADED
    gdir = resolve_gpu_prototype_dir(gpu_prototype_dir)
    k1 = resolve_k1_path(k1_path)
    dev = _resolve_device(device)

    if _LOADED is not None:
        if _LOADED.gpu_prototype_dir == gdir and _LOADED.device == dev and _LOADED.k1_path == k1:
            return _LOADED
        raise HarnessConfigError(
            "the inference contract is already loaded in this process with "
            f"dir={_LOADED.gpu_prototype_dir}, device={_LOADED.device}, k1={_LOADED.k1_path}; "
            f"cannot reload with dir={gdir}, device={dev}, k1={k1} (device is bound at import)"
        )

    for name in GUARDED_MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        _check_location(mod, gdir)
        bound = getattr(mod, "dev", None)
        if bound is not None and str(bound) != dev:
            raise HarnessConfigError(
                f"gpu_prototype module {name!r} is already imported with dev={bound!r}; the "
                f"harness needs dev={dev!r}. Load the contract before importing gpu_prototype "
                "modules, or request the same device."
            )

    if str(gdir) not in sys.path:
        sys.path.insert(0, str(gdir))

    hover_gpu = importlib.import_module("hover_gpu")
    _check_location(hover_gpu, gdir)
    hover_gpu.dev = dev  # must precede every import that binds `dev` by value

    reservoir_aug = importlib.import_module("reservoir_aug")
    _check_location(reservoir_aug, gdir)
    if k1 is not None:
        reservoir_aug.K1_PATH = str(k1)

    inference = importlib.import_module("inference")
    _check_location(inference, gdir)
    for name in ("gates_gpu", "gates_k1_reservoir"):
        if name in sys.modules:
            _check_location(sys.modules[name], gdir)

    _LOADED = ContractModule(inference=inference, device=dev, gpu_prototype_dir=gdir, k1_path=k1)
    return _LOADED


def loaded_contract() -> ContractModule | None:
    """The contract loaded in this process, if any."""
    return _LOADED
