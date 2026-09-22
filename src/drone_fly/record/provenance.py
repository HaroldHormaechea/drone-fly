"""Record-time provenance helpers (UC-45 AC8).

A recording is only reproducible after the fact if it pins the exact code that produced
it. :func:`resolve_git_sha` resolves the drone-fly working tree's current commit as a
``git describe`` string so every recording's ``meta.git_sha`` can travel with it, and it is
deliberately **total** — it never raises, returning ``"unknown"`` on any failure (git
missing, not a repository, timeout) so that stamping provenance can never crash a training
or evaluation run.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

#: Fallback stamped into ``meta.git_sha`` when the commit cannot be resolved.
UNKNOWN_GIT_SHA = "unknown"

#: Hard cap on the git subprocess so a hung/slow git can never stall recording setup.
_GIT_TIMEOUT_SECONDS = 2.0


def resolve_git_sha(cwd: str | Path | None = None) -> str:
    """Return the source tree's current commit as a ``git describe`` string, or ``"unknown"``.

    Runs ``git describe --always --dirty --abbrev=7`` in ``cwd`` and returns its trimmed
    stdout on success (e.g. ``"a1b2c3d"``, ``"a1b2c3d-dirty"``, or a tag form like
    ``"v1.2.3-4-ga1b2c3d-dirty"`` when the tree is tagged). The ``-dirty`` suffix marks
    uncommitted edits, so the stamp reflects the *actual* working-tree state, not just HEAD.

    ``cwd`` defaults to this module's directory (``Path(__file__).resolve().parent``) — the
    drone-fly **source tree** — deliberately NOT the process working directory: the UC-45
    reproduction harness (and any downstream tool) runs from arbitrary CWDs, and resolving
    against the source tree makes the stamp CWD-immune.

    This function is **total**: any failure — git not installed (``FileNotFoundError`` /
    ``OSError``), the timeout expiring (``TimeoutExpired``), a non-zero return code (not a
    git work tree), or any other error — is caught and logged at debug level, and
    :data:`UNKNOWN_GIT_SHA` is returned. It never raises.
    """
    if cwd is None:
        cwd = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=7"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # FileNotFoundError (git absent), TimeoutExpired, and any other subprocess error land
        # here — provenance is best-effort and must never crash a training/eval run.
        logger.debug("git_sha resolution failed (%s); using %r.", exc, UNKNOWN_GIT_SHA)
        return UNKNOWN_GIT_SHA
    if result.returncode != 0:
        # Non-zero return code means cwd is not a git work tree (or git errored) — fall back.
        logger.debug(
            "git describe returned %d (%s); using %r.",
            result.returncode,
            result.stderr.strip(),
            UNKNOWN_GIT_SHA,
        )
        return UNKNOWN_GIT_SHA
    sha = result.stdout.strip()
    return sha or UNKNOWN_GIT_SHA
