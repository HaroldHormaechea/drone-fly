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
#: The command that installs the live overlay's GUI toolkit (PySide6: ``harness overlay``).
OVERLAY_INSTALL_HINT = "uv sync --extra overlay"


class HarnessError(Exception):
    """Base class for every harness error."""


class HarnessDependencyError(HarnessError, ImportError):
    """An optional backend package is not installed."""

    def __init__(self, package: str, purpose: str = "", hint: str = INSTALL_HINT) -> None:
        self.package = package
        self.purpose = purpose
        what = f" (needed for {purpose})" if purpose else ""
        super().__init__(
            f"Missing optional dependency '{package}'{what}. Install the optional extra: {hint}"
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


class PerceptionContractError(HarnessConfigError):
    """A perception provider's declaration or output breaks the perception contract (UC-86)."""

    def __init__(self, provider: str, block: str | None, problem: str) -> None:
        self.provider = provider
        self.block = block
        where = f" block {block!r}" if block else ""
        super().__init__(f"perception provider {provider!r}{where}: {problem}")


class EstimatorStop(HarnessError):
    """A gate source ends the attempt on purpose (not a fault): the runner stops with ``reason``.

    Raised from :meth:`~drone_fly.harness.interfaces.GateEstimator.update` (the oracle's
    episode monitor, UC69). The runner records the partial tick and stops before anything is
    sent on that tick; the fail-safe still runs.
    """

    def __init__(self, reason: str, detail: str | None = None) -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


class StartMisalignedError(EstimatorStop):
    """The telemetry spawn does not match the track's start (UC69 AC3): exit 6, never armed."""

    REASON = "start_misaligned"

    def __init__(self, detail: str) -> None:
        super().__init__(self.REASON, detail)


class OracleLeakageError(HarnessError):
    """The oracle gate source was requested outside an evaluation / diagnostic purpose."""


class TeacherSchemaError(HarnessConfigError):
    """A teacher checkpoint's observation schema / provenance does not match this build."""


def lazy_import(
    module: str, *, package: str | None = None, purpose: str = "", hint: str = INSTALL_HINT
) -> ModuleType:
    """Import ``module`` on demand, raising :class:`HarnessDependencyError` when it is missing.

    ``package`` is the pip distribution name shown to the user when it differs from the
    import name (e.g. ``opencv-python`` for ``cv2``); ``hint`` is the install command shown
    (default: the ``harness`` extra).
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise HarnessDependencyError(package or module.split(".")[0], purpose, hint) from exc


class NoiseLeakageError(HarnessConfigError):
    """A detector-noise fit would use frames the detector trained on (UC70: held-out only)."""


class StudentSchemaError(HarnessConfigError):
    """A student checkpoint's observation schema / action contract / camera does not match."""


class StudentDeviceError(HarnessConfigError):
    """A K1 student would run on the CPU (too slow live) without ``--allow-slow-cpu``."""


class ExportError(HarnessConfigError):
    """``train export`` refused (e.g. a baseline actor without ``--baseline-export``)."""
