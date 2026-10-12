"""Find, load and write drone profiles under ``drones/<id>/drone.yaml``.

The profiles root is, in order: the ``root`` argument, the ``DRONE_FLY_DRONES_DIR`` environment
variable, ``<repo root>/drones``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from drone_fly.drones.schema import DroneProfile, DroneProfileError, parse_profile

PROFILE_FILE = "drone.yaml"
REPORT_FILE = "sysid.md"
ENV_ROOT = "DRONE_FLY_DRONES_DIR"
#: Key order of a written profile (``dump``); anything else follows in insertion order.
KEY_ORDER = (
    "schema_version",
    "id",
    "name",
    "source",
    "fit_quality",
    "notes",
    "liftoff",
    "analog",
    "nominal",
    "randomization",
)


def drones_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get(ENV_ROOT)
    if env:
        return Path(env)
    from drone_fly.harness.config import find_repo_root

    repo = find_repo_root()
    if repo is None:
        raise DroneProfileError(
            f"cannot find the drones/ folder: not inside the drone-fly repo (set {ENV_ROOT})"
        )
    return repo / "drones"


def profile_path(drone_id: str, root: str | Path | None = None) -> Path:
    return drones_root(root) / drone_id / PROFILE_FILE


def list_ids(root: str | Path | None = None) -> list[str]:
    """Ids of every folder under the root that holds a ``drone.yaml``."""
    base = drones_root(root)
    if not base.is_dir():
        return []
    return sorted(p.parent.name for p in base.glob(f"*/{PROFILE_FILE}"))


def load(drone_id: str, root: str | Path | None = None) -> DroneProfile:
    """Load and validate ``drones/<drone_id>/drone.yaml``."""
    path = profile_path(drone_id, root)
    if not path.is_file():
        known = list_ids(root)
        raise DroneProfileError(
            f"unknown drone id {drone_id!r} (no {path}); available: {known or 'none'}", "id"
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DroneProfileError(f"invalid YAML: {exc}", None, str(path)) from exc
    return parse_profile(data, file=str(path), expected_id=drone_id)


def _ordered(data: dict[str, Any]) -> dict[str, Any]:
    out = {k: data[k] for k in KEY_ORDER if k in data}
    out.update({k: v for k, v in data.items() if k not in out})
    return out


def dump(data: dict[str, Any], path: str | Path, header: str | None = None) -> Path:
    """Validate ``data`` and write it as YAML (stable key order, optional ``#`` header)."""
    out = Path(path)
    parse_profile(data, file=str(out), expected_id=out.parent.name)
    text = yaml.safe_dump(_ordered(data), sort_keys=False, allow_unicode=True, width=100)
    if header:
        text = "".join(f"# {line}\n" if line else "#\n" for line in header.splitlines()) + text
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out
