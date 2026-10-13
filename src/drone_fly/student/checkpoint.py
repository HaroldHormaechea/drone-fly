"""The deployable student export (UC70 D7): ONE ``.pt`` file, format ``drone-fly-student`` v1.

``torch.save`` mapping::

    {"format": "drone-fly-student", "format_version": 1,
     "state_dict": <the readout only: k1.body.layer.* stripped>,
     "meta": {<the run's sidecar: obs_schema, world_scale (k), action, actor, actor_arch, teacher,
              noise_model, perception, camera, course_conversion, timing>,
              "k1": {prune provenance sha256s}, "latency": {p50, p99, budget_ms, pass} | null,
              "deployable": bool, "role": "deployable" | "baseline",
              "requires": {gate_estimator, detector_sha256, free_space_estimator,
                           assume_no_obstacles, camera}}}

plus a ``.contract.json`` sidecar (the same ``meta``). :func:`validate_meta` is the version guard
the harness runs before arming: the embedded ``obs_schema`` must deep-equal this build's
``sim.student_obs.schema(...)`` for the same parameters, the action contract must be v2 and the
format version known; otherwise :class:`~drone_fly.harness.errors.StudentSchemaError` names both
versions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from drone_fly.harness.errors import StudentSchemaError

FORMAT = "drone-fly-student"
FORMAT_VERSION = 1
LATENCY_BUDGET_MS = 10.0
REQUIRED_META = ("obs_schema", "world_scale", "action", "actor", "actor_arch", "perception",
                 "camera", "deployable", "role")  # fmt: skip


def is_student_checkpoint(path: str | Path) -> bool:
    """True when ``path`` is an export (its sidecar or the file itself says so)."""
    p = Path(path)
    side = p.with_name(p.stem + ".contract.json")
    if side.is_file():
        try:
            return json.loads(side.read_text(encoding="utf-8")).get("format") == FORMAT
        except (OSError, ValueError):
            return False
    return False


def save_export(path: str | Path, state: dict[str, Any], meta: dict[str, Any]) -> Path:
    import os

    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {"format": FORMAT, "format_version": FORMAT_VERSION, "state_dict": state, "meta": meta}
    tmp = p.with_suffix(".tmp")
    torch.save(blob, tmp)
    os.replace(tmp, p)
    side = {"format": FORMAT, "format_version": FORMAT_VERSION, **meta}
    p.with_name(p.stem + ".contract.json").write_text(
        json.dumps(side, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return p


def load_export(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(state_dict, meta)``; refuses another format / version."""
    import torch

    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - unreadable file
        raise StudentSchemaError(f"cannot read student checkpoint {path}: {exc}") from exc
    if not isinstance(blob, dict) or blob.get("format") != FORMAT:
        raise StudentSchemaError(f"{path} is not a {FORMAT} export")
    if blob.get("format_version") != FORMAT_VERSION:
        raise StudentSchemaError(
            f"{path}: {FORMAT} format v{blob.get('format_version')} is not this build's "
            f"v{FORMAT_VERSION}"
        )
    return blob["state_dict"], blob["meta"]


def expected_schema(meta: dict[str, Any]) -> dict[str, Any]:
    """This build's ``student-obs`` schema for the checkpoint's own parameters."""
    from drone_fly.sim.student_obs import schema

    sch = meta.get("obs_schema") or {}
    tr = sch.get("tracker") or {}
    return schema(
        world_scale=float(meta.get("world_scale", sch.get("world_scale", 0.0))),
        lost_timeout_s=float(tr.get("lost_timeout_s", 0.0)),
        acquire_timeout_s=float(tr.get("acquire_timeout_s", 0.0)),
        camera=meta.get("camera") or sch.get("camera") or {"resolution": [0, 0], "hfov_deg": 0},
        assoc_radius=float(tr.get("assoc_radius", 1.0)),
    )


def validate_meta(meta: dict[str, Any], ckpt: str | Path = "checkpoint") -> None:
    """The version guard (module docstring); raises :class:`StudentSchemaError`."""
    from drone_fly.sim.student_obs import DIM, NAME, VERSION

    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise StudentSchemaError(f"{ckpt}: student meta lacks {missing}")
    sch = meta["obs_schema"] or {}
    have = (sch.get("name"), sch.get("version"), sch.get("dim"))
    want = (NAME, VERSION, DIM)
    if have != want:
        raise StudentSchemaError(
            f"{ckpt}: observation schema {have[0]} v{have[1]} ({have[2]} values) is not this "
            f"build's {want[0]} v{want[1]} ({want[2]} values)"
        )
    if float(sch.get("world_scale", -1)) != float(meta["world_scale"]):
        raise StudentSchemaError(
            f"{ckpt}: obs_schema world_scale {sch.get('world_scale')} != meta world_scale "
            f"{meta['world_scale']}"
        )
    exp = expected_schema(meta)
    if json.loads(json.dumps(sch)) != json.loads(json.dumps(exp)):
        diff = sorted(k for k in set(sch) | set(exp) if sch.get(k) != exp.get(k))
        raise StudentSchemaError(
            f"{ckpt}: obs_schema differs from this build's {NAME} v{VERSION} in {diff}"
        )
    version = meta.get("contract_version", (meta.get("action") or {}).get("version"))
    if version != 2:
        raise StudentSchemaError(
            f"{ckpt}: action contract v{version} is not supported (this build deploys v2)"
        )
