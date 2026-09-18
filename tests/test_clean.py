"""Unit tests for the ``drone-fly clean`` stage (UC-10).

``clean`` wipes training *outputs* — checkpoints + VecNormalize stats, learning-curve logs,
activation recordings, and any per-run ``training/<name>/`` trees — while leaving source
code, the connectome, prepared prune slices (unless ``--include-prunes``), use cases, and the
brief untouched. It is a **dry-run by default**; an explicit delete is required to remove
anything, and the operation always succeeds (exit 0) even with nothing to clean.

Every test builds a self-contained fake project tree under ``tmp_path`` (never the real repo)
whose output locations mirror the source-of-truth roots the ``clean`` module imports, so the
assertions here track the actual constants rather than hardcoded strings.

Acceptance-criteria coverage (see ``use-cases/10-clean-training-data.md``):

* AC1 — no-flag dry-run lists targets, deletes nothing.
* AC2 — delete removes the outputs and reports counts / "nothing to clean".
* AC3 — default target set = children of models/logs/activations + ``training/<name>/`` folders.
* AC4 — ``include_prunes`` removes ``pruned*`` under ``artifacts/`` and ``data/``; default
  leaves them (and every other ``data/`` input) intact.
* AC5 — protected paths survive a delete.
* AC6 — targets that resolve outside the root are refused (symlink escape skipped).
* AC7 — missing output roots are graceful ("nothing to clean"); ``.gitkeep`` preserved.
* AC8 — covered at the CLI layer (``--yes --dry-run`` → dry-run wins) in ``test_cli.py``.

Plus the challenger-mandated edge cases: an in-tree symlink child is unlinked (never
``rmtree``-d) so no ``OSError`` breaks the exit-0 guarantee, and a source-of-truth guard that
the module's target roots still equal the constants they are imported from.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from drone_fly.clean import (
    ACTIVATIONS_ROOT,
    CHILD_ROOTS,
    LOGS_ROOT,
    MODELS_ROOT,
    PRUNE_GLOBS,
    TRAINING_DIR,
    CleanReport,
    clean,
    discover_targets,
    execute_clean,
)
from drone_fly.config import TRAINING_ROOT
from drone_fly.record.recorder import DEFAULT_RECORD_DIR
from drone_fly.train.config import TrainConfig

# --------------------------------------------------------------------------- #
# Fixtures: a self-contained fake project tree
# --------------------------------------------------------------------------- #

#: Paths (relative to the project root) that must NEVER be touched by any clean (AC5). These
#: mirror the use case's protected list plus the raw connectome under ``data/``.
PROTECTED = (
    "src/drone_fly/__init__.py",
    "viz/plot.py",
    "tests/test_x.py",
    "use-cases/01-foo.md",
    "scripts/train.sh",
    "configs/train/base.yaml",
    "PROJECT_BRIEF.md",
    "USE_CASES.md",
    "data/connectome/mcns.npz",
)


def _touch(path: Path, content: bytes = b"x") -> Path:
    """Create ``path`` (and parents) as a small file; return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _build_tree(root: Path) -> None:
    """Populate ``root`` with training outputs, prune slices, inputs, and protected files."""
    # artifacts/models: a .gitkeep marker (preserved) + two removable children.
    _touch(root / MODELS_ROOT / ".gitkeep", b"")
    _touch(root / MODELS_ROOT / "ppo_100_steps.zip")
    _touch(root / MODELS_ROOT / "vecnormalize.pkl")

    # artifacts/logs: a nested run dir + a top-level file (both removable children).
    _touch(root / LOGS_ROOT / ".gitkeep", b"")
    _touch(root / LOGS_ROOT / "run1" / "events.out.tfevents")
    _touch(root / LOGS_ROOT / "progress.csv")

    # artifacts/activations: one recording child.
    _touch(root / ACTIVATIONS_ROOT / ".gitkeep", b"")
    _touch(root / ACTIVATIONS_ROOT / "episode_0.npz")

    # training/<name>/: two per-run trees (removed whole), plus a preserved .gitkeep.
    _touch(root / TRAINING_DIR / ".gitkeep", b"")
    _touch(root / TRAINING_DIR / "myrun" / "checkpoints" / "ppo_final.zip")
    _touch(root / TRAINING_DIR / "myrun" / "logs" / "progress.csv")
    _touch(root / TRAINING_DIR / "other" / "recordings" / "episode_0.npz")

    # Prune slices under BOTH artifacts/ and data/ (only removed with include_prunes).
    _touch(root / "artifacts" / "pruned" / "k0" / "connectome_pruned.npz")
    _touch(root / "data" / "pruned_slice" / "connectome_pruned.npz")

    # Non-pruned data + protected files that must always survive (AC5).
    _touch(root / "data" / "other_input.txt")
    for rel in PROTECTED:
        _touch(root / rel)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A fully-populated fake project root under a dedicated subdir of ``tmp_path``.

    Using a subdir (not ``tmp_path`` itself) leaves room for an out-of-tree sibling in the
    AC6 escape test.
    """
    root = tmp_path / "proj"
    _build_tree(root)
    return root


def _assert_protected_survive(root: Path) -> None:
    """Every protected path (AC5) plus the non-pruned data input still exists."""
    for rel in PROTECTED:
        assert (root / rel).is_file(), f"protected path was removed: {rel}"
    assert (root / "data" / "other_input.txt").is_file()


def _assert_roots_and_gitkeep_preserved(root: Path) -> None:
    """The child-root dirs and their ``.gitkeep`` markers survive any clean (AC3/AC7)."""
    for rel in CHILD_ROOTS:
        assert (root / rel).is_dir(), f"root dir was removed: {rel}"
    for rel in (MODELS_ROOT, LOGS_ROOT, ACTIVATIONS_ROOT, TRAINING_DIR):
        assert (root / rel / ".gitkeep").is_file(), f".gitkeep removed under {rel}"


# --------------------------------------------------------------------------- #
# Source-of-truth guard — module roots equal the constants they import
# --------------------------------------------------------------------------- #


def test_target_roots_are_the_imported_constants() -> None:
    """The clean roots must stay wired to their owners, never re-hardcoded (plan guard)."""
    assert MODELS_ROOT == TrainConfig().models_dir == "artifacts/models"
    assert LOGS_ROOT == TrainConfig().logs_dir == "artifacts/logs"
    assert ACTIVATIONS_ROOT == DEFAULT_RECORD_DIR == "artifacts/activations"
    assert TRAINING_DIR == TRAINING_ROOT == "training"
    assert CHILD_ROOTS == (MODELS_ROOT, LOGS_ROOT, ACTIVATIONS_ROOT, TRAINING_DIR)
    assert PRUNE_GLOBS == ("artifacts/pruned*", "data/pruned*")


# --------------------------------------------------------------------------- #
# AC1 — no-flag dry-run lists targets but deletes nothing
# --------------------------------------------------------------------------- #


def test_dry_run_default_lists_but_deletes_nothing(project: Path) -> None:
    report = clean(project)  # delete defaults to False

    assert isinstance(report, CleanReport)
    assert report.dry_run is True
    assert report.paths, "dry-run should still list what would be removed"
    # Nothing was actually removed: every discovered target is still on disk.
    for target in report.paths:
        assert target.exists(), f"dry-run deleted {target}"
    _assert_protected_survive(project)
    _assert_roots_and_gitkeep_preserved(project)


def test_dry_run_summary_uses_would_remove(project: Path) -> None:
    report = clean(project)
    summary = report.summary()
    assert "Would remove" in summary
    assert "nothing to clean" not in summary


# --------------------------------------------------------------------------- #
# AC2 — delete removes the outputs and reports counts
# --------------------------------------------------------------------------- #


def test_delete_removes_training_outputs_and_reports(project: Path) -> None:
    report = clean(project, delete=True)

    assert report.dry_run is False
    assert report.paths, "delete should report the removed targets"
    # Every reported target is gone from disk.
    for target in report.paths:
        assert not target.exists(), f"target not removed: {target}"
    # The removable children specifically are gone…
    assert not (project / MODELS_ROOT / "ppo_100_steps.zip").exists()
    assert not (project / LOGS_ROOT / "run1").exists()
    assert not (project / ACTIVATIONS_ROOT / "episode_0.npz").exists()
    assert not (project / TRAINING_DIR / "myrun").exists()
    assert not (project / TRAINING_DIR / "other").exists()
    # …but roots, .gitkeep and every protected path survive.
    _assert_roots_and_gitkeep_preserved(project)
    _assert_protected_survive(project)
    # Summary reports a non-zero count with the "Removed" verb.
    assert report.summary().startswith("Removed ")


# --------------------------------------------------------------------------- #
# AC3 — the default target set is exactly the training-output children
# --------------------------------------------------------------------------- #


def test_default_target_set_is_output_children(project: Path) -> None:
    targets = discover_targets(project)
    got = {p.relative_to(project).as_posix() for p in targets}

    expected = {
        f"{MODELS_ROOT}/ppo_100_steps.zip",
        f"{MODELS_ROOT}/vecnormalize.pkl",
        f"{LOGS_ROOT}/run1",
        f"{LOGS_ROOT}/progress.csv",
        f"{ACTIVATIONS_ROOT}/episode_0.npz",
        f"{TRAINING_DIR}/myrun",
        f"{TRAINING_DIR}/other",
    }
    assert got == expected
    # No .gitkeep, no root dir itself, and no prune slices in the default set.
    assert not any(p.name == ".gitkeep" for p in targets)
    assert not any(p.as_posix().endswith(rel) for p in targets for rel in CHILD_ROOTS)
    assert not any("pruned" in p.as_posix() for p in targets)


# --------------------------------------------------------------------------- #
# AC4 — --include-prunes removes pruned* under artifacts/ and data/; default leaves them
# --------------------------------------------------------------------------- #


def test_default_leaves_prune_slices_and_data_intact(project: Path) -> None:
    clean(project, delete=True)  # default: include_prunes False
    assert (project / "artifacts" / "pruned" / "k0" / "connectome_pruned.npz").is_file()
    assert (project / "data" / "pruned_slice" / "connectome_pruned.npz").is_file()
    assert (project / "data" / "other_input.txt").is_file()
    assert (project / "data" / "connectome" / "mcns.npz").is_file()


def test_include_prunes_discovers_both_locations(project: Path) -> None:
    targets = discover_targets(project, include_prunes=True)
    got = {p.relative_to(project).as_posix() for p in targets}
    assert "artifacts/pruned" in got
    assert "data/pruned_slice" in got


def test_include_prunes_removes_slices_but_keeps_other_data(project: Path) -> None:
    report = clean(project, delete=True, include_prunes=True)

    assert report.include_prunes is True
    assert not (project / "artifacts" / "pruned").exists()
    assert not (project / "data" / "pruned_slice").exists()
    # Non-pruned data + the raw connectome (never matches pruned*) survive (AC4/AC5).
    assert (project / "data" / "other_input.txt").is_file()
    assert (project / "data" / "connectome" / "mcns.npz").is_file()
    assert "(including prune slices)" in report.summary()


# --------------------------------------------------------------------------- #
# AC5 — protected paths survive a delete (with prunes included, the most aggressive run)
# --------------------------------------------------------------------------- #


def test_protected_paths_survive_aggressive_delete(project: Path) -> None:
    clean(project, delete=True, include_prunes=True)
    _assert_protected_survive(project)


# --------------------------------------------------------------------------- #
# AC6 — targets that resolve outside the root are refused (symlink escape skipped)
# --------------------------------------------------------------------------- #


def test_symlink_escape_is_skipped(tmp_path: Path) -> None:
    """A symlinked child pointing outside the root is skipped, and its target survives (AC6)."""
    root = tmp_path / "proj"
    outside = tmp_path / "outside"
    secret = _touch(outside / "secret.txt", b"keep me")

    _touch(root / MODELS_ROOT / ".gitkeep", b"")
    _touch(root / MODELS_ROOT / "real_child.zip")
    escape = root / MODELS_ROOT / "escape"
    escape.symlink_to(outside, target_is_directory=True)

    targets = discover_targets(root)
    rels = {p.relative_to(root).as_posix() for p in targets}
    # The in-tree child is discovered; the escaping symlink is refused.
    assert f"{MODELS_ROOT}/real_child.zip" in rels
    assert f"{MODELS_ROOT}/escape" not in rels

    # A full delete never follows the escape: the outside file is untouched.
    clean(root, delete=True)
    assert secret.is_file() and secret.read_bytes() == b"keep me"
    assert escape.exists() is True or escape.is_symlink()  # link itself left in place (skipped)


# --------------------------------------------------------------------------- #
# In-tree symlink child is unlinked, not rmtree-d (challenger #2 — no OSError)
# --------------------------------------------------------------------------- #


def test_in_tree_symlink_child_is_unlinked_without_error(tmp_path: Path) -> None:
    """A symlink child that resolves under root is removed via unlink; its target survives.

    ``shutil.rmtree`` on a symlink-to-dir raises ``OSError`` — that would break the exit-0 /
    partial-delete guarantee. The clean module must ``unlink`` links instead.
    """
    root = tmp_path / "proj"
    # A real directory inside data/ (survives) that the symlink will point at.
    target_dir = _touch(root / "data" / "connectome" / "mcns.npz").parent
    _touch(root / ACTIVATIONS_ROOT / ".gitkeep", b"")
    link = root / ACTIVATIONS_ROOT / "linkdir"
    link.symlink_to(target_dir, target_is_directory=True)

    # Must not raise, must remove the link, must leave the link's target intact.
    report = clean(root, delete=True)
    assert (root / ACTIVATIONS_ROOT / "linkdir").is_symlink() is False
    assert not link.exists()
    assert target_dir.is_dir()
    assert (target_dir / "mcns.npz").is_file()
    assert any(p.name == "linkdir" for p in report.paths)


# --------------------------------------------------------------------------- #
# AC7 — missing output roots are graceful; .gitkeep preserved
# --------------------------------------------------------------------------- #


def test_missing_roots_report_nothing_to_clean(tmp_path: Path) -> None:
    """An empty project (no artifacts/ or training/) reports 'nothing to clean', no error."""
    root = tmp_path / "empty"
    root.mkdir()

    report = clean(root)
    assert report.paths == ()
    assert report.summary() == "nothing to clean"

    # Delete mode on an empty tree is equally graceful.
    report_del = clean(root, delete=True, include_prunes=True)
    assert report_del.paths == ()
    assert report_del.summary() == "nothing to clean"


def test_gitkeep_children_are_preserved(project: Path) -> None:
    clean(project, delete=True)
    for rel in (MODELS_ROOT, LOGS_ROOT, ACTIVATIONS_ROOT, TRAINING_DIR):
        assert (project / rel / ".gitkeep").is_file()


# --------------------------------------------------------------------------- #
# execute_clean / CleanReport unit behaviour
# --------------------------------------------------------------------------- #


def test_execute_clean_dry_run_returns_targets_unchanged(project: Path) -> None:
    targets = discover_targets(project)
    out = execute_clean(targets, delete=False)
    assert out == targets
    for t in targets:
        assert t.exists()  # dry-run touched nothing


def test_execute_clean_tolerates_already_missing(tmp_path: Path) -> None:
    """A target that vanishes between discovery and removal is tolerated (AC7)."""
    ghost = tmp_path / "already_gone.zip"
    out = execute_clean([ghost], delete=True)
    assert out == [ghost]  # returned for the report; no error


def test_clean_report_relative_paths(project: Path) -> None:
    report = clean(project)
    rels = report.relative_paths()
    assert all(not Path(r).is_absolute() for r in rels)
    assert f"{MODELS_ROOT}/ppo_100_steps.zip" in rels


def test_clean_report_is_frozen(project: Path) -> None:
    report = clean(project)
    with pytest.raises(FrozenInstanceError):
        report.dry_run = True  # type: ignore[misc]
