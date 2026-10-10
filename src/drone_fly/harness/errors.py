"""Harness error types and the lazy-import helper for optional backend dependencies.

Heavy backends (torch inside the contract bridge, OpenCV, screen capture, the depth model,
``vgamepad``) are never imported at module import time. Code that needs one calls
:func:`lazy_import`, which turns a missing package into a :class:`HarnessDependencyError`
naming the package and the install command, so the hermetic CI path (which has none of them)
still imports the whole harness package.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from types import ModuleType

#: The command that installs every optional harness backend.
INSTALL_HINT = "uv sync --extra harness"


class HarnessError(Exception):
    """Base class for every harness error."""


class HarnessDependencyError(HarnessError, ImportError):
    """An optional backend package is not installed."""

    def __init__(self, package: str, purpose: str = "", hint: str = INSTALL_HINT) -> None:
        self.package = package
        self.purpose = purpose
        what = f" (needed for {purpose})" if purpose else ""
        super().__init__(
            f"Missing optional dependency '{package}'{what}. Install the harness extra: {hint}"
        )


class HarnessConfigError(HarnessError, ValueError):
    """The harness configuration (YAML, CLI flags or paths) is invalid."""


class PerceptionCoverageError(HarnessError):
    """The configured perception cannot fill the observation blocks the model needs."""

    def __init__(self, missing: Iterable[str], extra_dim: int | None = None) -> None:
        self.missing = tuple(missing)
        self.extra_dim = extra_dim
        dim = f" (extra_dim={extra_dim})" if extra_dim is not None else ""
        super().__init__(
            f"Configured perception cannot provide the observation blocks the model needs{dim}: "
            f"missing {', '.join(self.missing)}. Configure an estimator that provides them, or "
            f"use a model with a smaller extra_dim."
        )


def lazy_import(module: str, *, package: str | None = None, purpose: str = "") -> ModuleType:
    """Import ``module`` on demand, raising :class:`HarnessDependencyError` when it is missing.

    ``package`` is the pip distribution name shown to the user when it differs from the
    import name (e.g. ``opencv-python`` for ``cv2``).
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise HarnessDependencyError(package or module.split(".")[0], purpose) from exc
