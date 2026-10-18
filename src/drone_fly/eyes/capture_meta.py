"""Capture discovery and a minimal UC71 ``meta.json`` reader (offline).

UC-71's capture tooling is not on this branch, so this is a schema-compatible reader of the
fields the eyes pipeline needs (``schema_version`` 1): ``catalogue_id``, ``map``, ``camera``
(``resolution``, ``hfov_deg``, ``uptilt_deg``), ``alignment.verified`` / ``alignment.lag_ms``,
``pose_source`` and ``source``. Unify with UC-71's reader once both are on ``main``.

* **Discovery** (:func:`discover_captures`): a capture is a folder holding ``frames.csv``, at depth
  1 or 2 under a captures root (``<root>/<capture>`` or ``<root>/<Map>/<capture>``); the root
  itself counts when it is one.
* **Legacy captures** (no ``meta.json``) are labelled only with an explicit ``catalogue_id`` and
  ``lag_ms`` (CLI ``--catalogue-id`` / ``--lag-ms``); otherwise :func:`read_capture_meta` returns
  ``None`` and the caller skips the capture as ``no-meta``.
* **Capture sha** (:func:`capture_sha`): sha256 over ``meta.json``, ``frames.csv`` and the
  telemetry file, so a re-captured folder of the same name gets a new identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drone_fly.harness.errors import HarnessConfigError

META_FILE = "meta.json"
META_SCHEMA_VERSION = 1
TELEMETRY_FILES = ("telemetry_raw.csv", "telemetry.csv")


@dataclass(frozen=True)
class CaptureMeta:
    root: Path
    catalogue_id: str
    map: str | None
    camera: dict[str, Any] | None
    alignment_verified: bool
    lag_ms: float
    pose_source: str | None
    source: str | None
    legacy: bool  # no meta.json: identity from the command line

    @property
    def name(self) -> str:
        return self.root.name

    def as_dict(self) -> dict[str, Any]:
        return {
            "capture": str(self.root),
            "name": self.name,
            "catalogue_id": self.catalogue_id,
            "map": self.map,
            "camera": self.camera,
            "alignment_verified": self.alignment_verified,
            "lag_ms": self.lag_ms,
            "pose_source": self.pose_source,
            "source": self.source,
            "legacy": self.legacy,
        }


def is_capture(path: Path) -> bool:
    return (path / "frames.csv").is_file()


def discover_captures(root: str | Path) -> list[Path]:
    """Capture folders at depth 0-2 under ``root`` (sorted)."""
    root = Path(root)
    if is_capture(root):
        return [root]
    out = []
    for a in sorted(p for p in root.iterdir() if p.is_dir()):
        if is_capture(a):
            out.append(a)
            continue
        out += [b for b in sorted(p for p in a.iterdir() if p.is_dir()) if is_capture(b)]
    return out


def read_capture_meta(
    root: str | Path, *, catalogue_id: str | None = None, lag_ms: float | None = None
) -> CaptureMeta | None:
    """The capture's identity (see the module docstring); ``None`` = no meta and no override.

    Command-line ``catalogue_id`` / ``lag_ms`` override the meta's values when given.
    """
    root = Path(root)
    path = root / META_FILE
    if not path.is_file():
        if catalogue_id is None or lag_ms is None:
            return None
        return CaptureMeta(root, str(catalogue_id), None, None, True, float(lag_ms), None, None,
                           True)  # fmt: skip
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HarnessConfigError(f"cannot read {path}: {exc}") from exc
    if m.get("schema_version") != META_SCHEMA_VERSION:
        raise HarnessConfigError(
            f"{path}: unsupported capture meta schema_version {m.get('schema_version')!r}"
        )
    align = m.get("alignment") or {}
    cid = catalogue_id or m.get("catalogue_id")
    if not cid:
        raise HarnessConfigError(f"{path} has no catalogue_id (pass --catalogue-id)")
    lag = lag_ms if lag_ms is not None else align.get("lag_ms")
    return CaptureMeta(
        root=root,
        catalogue_id=str(cid),
        map=m.get("map"),
        camera=m.get("camera"),
        alignment_verified=bool(align.get("verified", False)),
        lag_ms=float(lag) if lag is not None else 0.0,
        pose_source=m.get("pose_source"),
        source=m.get("source"),
        legacy=False,
    )


def capture_sha(root: str | Path) -> str:
    """sha256 over ``meta.json`` + ``frames.csv`` + the telemetry file (those present)."""
    root = Path(root)
    h = hashlib.sha256()
    for name in (META_FILE, "frames.csv", *TELEMETRY_FILES):
        p = root / name
        if not p.is_file():
            continue
        h.update(name.encode())
        with p.open("rb") as fh:
            for chunk in iter(lambda fh=fh: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()
