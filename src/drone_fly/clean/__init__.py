"""Clean stage: wipe training *outputs* so a run can start from scratch (UC-10).

``drone-fly clean`` removes everything a training/evaluation cycle produces — checkpoints +
VecNormalize stats, learning-curve logs, activation recordings, and any per-run
``training/<name>/`` trees — while leaving source code, the connectome, prepared prune
slices (unless ``--include-prunes``), use cases, and the brief untouched.

Source of truth, never hardcoded
---------------------------------
The output locations are imported from the modules that own them, so ``clean`` can never
drift from where the other stages actually write:

* :attr:`MODELS_ROOT` / :attr:`LOGS_ROOT` come from
  :class:`drone_fly.train.config.TrainConfig` (``models_dir`` / ``logs_dir``).
* :attr:`ACTIVATIONS_ROOT` is :data:`drone_fly.record.recorder.DEFAULT_RECORD_DIR`.
* :attr:`TRAINING_DIR` is :data:`drone_fly.config.TRAINING_ROOT` (the UC-11 per-run layout).

These roots are exported so tests can assert they still equal those constants — a guard
against a future edit re-hardcoding a path here.

Safety model
------------
* **Child-level granularity.** For ``artifacts/models``, ``artifacts/logs``,
  ``artifacts/activations`` and ``training/`` the *children* are removed and the root dir
  itself (plus any ``.gitkeep`` marker) is preserved, so the directory layout survives a
  wipe. Per-run ``training/<name>/`` folders are removed whole.
* **Confinement.** Every candidate is resolved and asserted to live under ``root`` before it
  is touched; anything that escapes (e.g. via a symlink pointing out of the tree) is skipped
  with a warning. In normal use ``root`` is the CWD and all targets are literal relative
  joins, so escapes cannot arise — the guard is defence in depth (AC6).
* **Dry-run first.** Discovery is side-effect-free; only :func:`execute_clean` with
  ``delete=True`` removes anything. Missing roots contribute nothing and are not an error
  (AC7).

Prune-slice reconciliation
--------------------------
AC4 names ``data/pruned*``, but the real ``drone-fly prune`` writes to a user-chosen ``out``
and every committed example (``configs/prune/k*.yaml``) uses ``artifacts/pruned/k*``.
``--include-prunes`` therefore clears ``pruned*`` under **both** ``artifacts/`` and
``data/``. The raw connectome (``data/connectome``) never matches ``pruned*``, so it always
survives (AC5).
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from drone_fly.config import TRAINING_ROOT
from drone_fly.record.recorder import DEFAULT_RECORD_DIR
from drone_fly.train.config import TrainConfig

logger = logging.getLogger(__name__)

# --- Target roots (imported from their owners, never hardcoded) ---------------------------

#: ``artifacts/models`` — checkpoints + VecNormalize stats.
MODELS_ROOT = TrainConfig().models_dir
#: ``artifacts/logs`` — TensorBoard + CSV learning curves.
LOGS_ROOT = TrainConfig().logs_dir
#: ``artifacts/activations`` — recorded activation playback files.
ACTIVATIONS_ROOT = DEFAULT_RECORD_DIR
#: ``training`` — the UC-11 per-run ``training/<name>/`` output root.
TRAINING_DIR = TRAINING_ROOT

#: Roots whose *children* are removed (the root dir + any ``.gitkeep`` are preserved), in a
#: stable display order.
CHILD_ROOTS: tuple[str, ...] = (MODELS_ROOT, LOGS_ROOT, ACTIVATIONS_ROOT, TRAINING_DIR)

#: Glob patterns (relative to ``root``) removed only under ``--include-prunes``. Scoped to
#: ``pruned*`` so the raw connectome (``data/connectome``) can never match (AC4/AC5).
PRUNE_GLOBS: tuple[str, ...] = ("artifacts/pruned*", "data/pruned*")


@dataclass(frozen=True)
class CleanReport:
    """Result of a clean pass — what was (or, in dry-run, would be) removed.

    ``paths`` are the removed / would-be-removed targets in discovery order, relative to
    ``root`` when possible for readable output. ``dry_run`` records whether anything was
    actually deleted. :meth:`summary` renders the one-line human summary.
    """

    root: Path
    dry_run: bool
    include_prunes: bool
    paths: tuple[Path, ...]

    def relative_paths(self) -> list[str]:
        """The target paths as strings, relative to ``root`` where possible."""
        out: list[str] = []
        for p in self.paths:
            try:
                out.append(str(p.relative_to(self.root)))
            except ValueError:  # pragma: no cover - confinement guard keeps these under root
                out.append(str(p))
        return out

    def summary(self) -> str:
        """One-line human summary; ``"nothing to clean"`` when there is nothing to remove."""
        n = len(self.paths)
        if n == 0:
            return "nothing to clean"
        verb = "Would remove" if self.dry_run else "Removed"
        noun = "target" if n == 1 else "targets"
        extra = " (including prune slices)" if self.include_prunes else ""
        return f"{verb} {n} {noun}{extra}."


def _is_confined(candidate: Path, root: Path) -> bool:
    """True when ``candidate`` resolves to a path at or under ``root`` (AC6)."""
    resolved = candidate.resolve()
    return resolved == root or root in resolved.parents


def _children(directory: Path, root: Path) -> list[Path]:
    """Removable children of ``directory``: sorted, ``.gitkeep`` skipped, confined to ``root``.

    A missing (or non-directory) root contributes nothing (AC7).
    """
    if not directory.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(directory.iterdir(), key=lambda p: p.name):
        if child.name == ".gitkeep":
            continue
        if not _is_confined(child, root):
            logger.warning("Skipping %s: resolves outside the clean root %s.", child, root)
            continue
        out.append(child)
    return out


def discover_targets(root: Path | str, *, include_prunes: bool = False) -> list[Path]:
    """Return the ordered list of paths ``clean`` would remove under ``root`` (no side effects).

    The default set is the children of :data:`CHILD_ROOTS` (each root dir + ``.gitkeep``
    preserved). With ``include_prunes`` the :data:`PRUNE_GLOBS` matches under ``root`` are
    appended. Non-existent roots and empty globs contribute nothing (AC7); every candidate is
    confinement-checked against ``root`` (AC6).
    """
    root = Path(root)
    targets: list[Path] = []
    for rel in CHILD_ROOTS:
        targets.extend(_children(root / rel, root))
    if include_prunes:
        for pattern in PRUNE_GLOBS:
            for match in sorted(root.glob(pattern), key=lambda p: str(p)):
                if not _is_confined(match, root):
                    logger.warning("Skipping %s: resolves outside the clean root %s.", match, root)
                    continue
                targets.append(match)
    return targets


def _remove(path: Path) -> None:
    """Remove one target: symlink/file → ``unlink``, real dir → ``rmtree``; missing tolerated.

    Removing a symlinked child with :func:`shutil.rmtree` raises ``OSError`` (rmtree refuses
    a symlink), which would break the exit-0 / partial-delete guarantee — so links and files
    are unlinked and only real directories go through ``rmtree``.
    """
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return
    # Already gone between discovery and removal — nothing to do (AC7).


def execute_clean(targets: list[Path], *, delete: bool) -> list[Path]:
    """Process ``targets``: dry-run removes nothing; ``delete`` removes each. Returns them.

    Always succeeds (the CLI maps this to exit 0). In dry-run the list is returned unchanged;
    in delete mode each target is removed (already-missing tolerated) and then returned so the
    report reflects what was cleared.
    """
    if delete:
        for target in targets:
            _remove(target)
    return list(targets)


def clean(root: Path | str, *, delete: bool = False, include_prunes: bool = False) -> CleanReport:
    """Discover training-output targets under ``root`` and (when ``delete``) remove them.

    ``delete=False`` is the safe dry-run default (AC1): it reports what *would* be removed and
    touches nothing. ``delete=True`` performs the wipe (AC2). Either way the result is a
    :class:`CleanReport`; the operation always succeeds (AC7).
    """
    root = Path(root)
    targets = discover_targets(root, include_prunes=include_prunes)
    removed = execute_clean(targets, delete=delete)
    return CleanReport(
        root=root,
        dry_run=not delete,
        include_prunes=include_prunes,
        paths=tuple(removed),
    )


__all__ = [
    "MODELS_ROOT",
    "LOGS_ROOT",
    "ACTIVATIONS_ROOT",
    "TRAINING_DIR",
    "CHILD_ROOTS",
    "PRUNE_GLOBS",
    "CleanReport",
    "discover_targets",
    "execute_clean",
    "clean",
]
