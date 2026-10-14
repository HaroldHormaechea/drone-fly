"""Where the (gitignored) training data lives: one data root and its fixed layout.

The data root holds everything needed to keep training for the Liftoff simulator; its layout is
described in ``<data root>/MANIFEST.md``::

    captures/        Liftoff gate captures (frames + telemetry), one folder per map
    trackdata/       track XMLs and item data read from Liftoff
    scenes/          scene geometry caches (labelling, occlusion)
    track-library/   UC68 track library (scene candidates)
    labels/          detector labels
    models/          detector-v1/, detector-v2/, teacher/, student/
    reference/       detector reports and diagnoses, live run telemetry
    sysid/           system-identification data
    pruned/          pruned connectomes

Resolution of the root, first match wins: the ``DRONE_FLY_DATA_ROOT`` environment variable, then
``<repo>/artifacts/harness/liftoff/training-data``. There is no config-file key: the harness YAMLs
are per-harness and strict, so no file is shared by every entry point. Explicit command-line paths
always win over these defaults (callers pass them through instead of calling these helpers).
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "DRONE_FLY_DATA_ROOT"
DEFAULT_DATA_ROOT = Path("artifacts") / "harness" / "liftoff" / "training-data"
MODEL_KINDS = ("detector-v1", "detector-v2", "teacher", "student")


def data_root() -> Path:
    """The data root: ``$DRONE_FLY_DATA_ROOT``, else the repo's ``training-data`` folder.

    Without a repo checkout (an installed package), the default is relative to the current
    directory.
    """
    env = os.environ.get(DATA_ROOT_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    from drone_fly.harness.config import find_repo_root

    return (find_repo_root() or Path.cwd()) / DEFAULT_DATA_ROOT


def captures_root() -> Path:
    return data_root() / "captures"


def trackdata_root() -> Path:
    return data_root() / "trackdata"


def scenes_root() -> Path:
    return data_root() / "scenes"


def track_library_root() -> Path:
    return data_root() / "track-library"


def labels_root() -> Path:
    return data_root() / "labels"


def reference_root() -> Path:
    return data_root() / "reference"


def sysid_root() -> Path:
    return data_root() / "sysid"


def pruned_root() -> Path:
    return data_root() / "pruned"


def models_root(kind: str | None = None) -> Path:
    """``models/``, or one of its :data:`MODEL_KINDS` subfolders."""
    root = data_root() / "models"
    if kind is None:
        return root
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown model kind {kind!r}; expected one of {MODEL_KINDS}")
    return root / kind
