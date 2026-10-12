"""The single course loader for both sources, and the training / evaluation pool builder.

* :func:`load_course` turns ONE catalogue entry into simulator courses: a Liftoff entry gives
  one :class:`~drone_fly.sim.pool.SimCourse`, a procedural entry its pool of ``spec.pool_size``
  courses. It checks the entry's file hashes, applies the map's scene status and the scene
  policy, and caches the (expensive) conversion under ``artifacts/tracks/cache/`` keyed by every
  input hash and the conversion fingerprint.
* :func:`build_pool` resolves a mix (``[{select, weight, scene_weight}]``) through
  :func:`drone_fly.tracks.catalogue.select` with a ``purpose`` -- ``train`` refuses val /
  held-out entries (AC9) -- and returns the courses, their groups and a record of every file it
  read (for the run snapshot).
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drone_fly.harness.offline.track_xml import load_track_geometry
from drone_fly.sim.pool import PoolGroup, SimCourse
from drone_fly.tracks import catalogue as cat
from drone_fly.tracks.catalogue import Catalogue, TrackDataError, sha256_file
from drone_fly.tracks.liftoff import (
    CONVERSION_VERSION,
    ConversionConfig,
    PropsTable,
    convert_liftoff,
)
from drone_fly.tracks.scenes import USABLE, find_scene, scene_path

log = logging.getLogger(__name__)

CACHE_DIR = "artifacts/tracks/cache"


@dataclass
class LoadRecord:
    """Every file a load touched, with hashes (the run's ``tracks/files.sha256``)."""

    files: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(
        self, kind: str, path: Path, sha: str | None, extra: dict[str, Any] | None = None
    ) -> None:
        self.files[str(path)] = {"kind": kind, "sha256": sha, **(extra or {})}


@dataclass
class Pool:
    """Courses + groups for one purpose (see the module docstring)."""

    courses: list[SimCourse]
    groups: list[PoolGroup]
    purpose: str
    catalogues: dict[str, Catalogue]
    record: LoadRecord

    def table(self) -> list[dict[str, Any]]:
        """``courses.json``: index -> id, source, split, map, scene status, group."""
        gof = {}
        for g in self.groups:
            for i in g.courses:
                gof[i] = g.name
        return [
            {"index": i, "id": c.id, "source": c.meta.get("source"), "split": c.meta.get("split"),
             "map": c.meta.get("map"), "scene": c.meta.get("scene"), "group": gof.get(i),
             "flags": c.meta.get("flags", [])}
            for i, c in enumerate(self.courses)
        ]  # fmt: skip


def cache_root() -> Path:
    return cat.repo_root() / CACHE_DIR


def _cache_key(parts: dict[str, Any]) -> str:
    blob = repr(sorted(parts.items())).encode()
    return hashlib.sha256(blob).hexdigest()[:20]


def _cached(key: str, entry_id: str, build: Any, cache_dir: Path | None) -> Any:
    if cache_dir is None:
        return build()
    path = Path(cache_dir) / f"{entry_id}-{key}.pkl"
    if path.is_file():
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:  # a stale / truncated cache entry is rebuilt
            log.warning("ignoring unreadable conversion cache %s", path)
    value = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return value


def entry_scene_status(catalogue: Catalogue, entry: dict[str, Any]) -> dict[str, Any]:
    return dict((catalogue.maps().get(entry["map"]) or {}).get("scene") or {"status": "missing"})


def load_course(
    entry: dict[str, Any],
    *,
    catalogue: Catalogue,
    root: Path | None = None,
    scenes_root: Path | None = None,
    conversion: ConversionConfig | None = None,
    props: PropsTable | None = None,
    record: LoadRecord | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> list[SimCourse]:
    """Courses of one entry (see the module docstring)."""
    conv = conversion or ConversionConfig.from_dict(None)
    record = record if record is not None else LoadRecord()
    cdir = (cache_dir or cache_root()) if use_cache else None
    base_meta = {"id": entry["id"], "source": entry["source"], "split": entry["split"],
                 "map": entry["map"], "flags": list(entry.get("flags") or [])}  # fmt: skip
    if entry["status"] != cat.STATUS_OK:
        raise TrackDataError(f"{entry['id']} is {entry['status']}")
    if entry["source"] == "procedural":
        from drone_fly.tracks.procedural import convert_procedural, validate_spec

        spec = validate_spec(entry["spec"])
        key = _cache_key({"spec": repr(spec), "conv": conv.fingerprint(), "v": CONVERSION_VERSION})

        def build_proc() -> list[SimCourse]:
            return [
                convert_procedural(spec, i, conv, {**base_meta, "id": f"{entry['id']}/{i}",
                                                   "entry": entry["id"], "scene": "procedural"})
                for i in range(spec["pool_size"])
            ]  # fmt: skip

        return _cached(key, entry["id"], build_proc, cdir)
    root = root or cat.resolve_root(None, catalogue.header)
    scenes_root = scenes_root or cat.resolve_scenes_root(None, catalogue.header, root)
    track, race = root / entry["track"], root / entry["race"]
    sha = {}
    for k, p in (("track", track), ("race", race)):
        if not p.is_file():
            raise TrackDataError(
                f"{entry['id']}: {k} file {p} is missing (run `drone-fly tracks check`)"
            )
        sha[k] = sha256_file(p)
        want = (entry.get("sha256") or {}).get(k)
        if want and want != sha[k]:
            raise TrackDataError(
                f"{entry['id']}: {p.name} changed since the scan (re-run tracks scan)"
            )
        record.add(k, p, sha[k], {"entry": entry["id"]})
    if props is None:
        props = PropsTable.load(root / "props.json")
    record.add(
        "props", props.path or root / "props.json", props.sha256, {"build_id": props.build_id}
    )
    scene_meta = entry_scene_status(catalogue, entry)
    scene = None
    status = "no-scene"
    if scene_meta.get("status") in USABLE:
        sp = scene_path(scenes_root, entry["map"])
        scene = find_scene(scenes_root, entry["map"])
        if scene is None:
            msg = f"{entry['id']}: the catalogue's {entry['map']} scene is missing at {sp}"
            if conv.scene_policy == "require":
                raise TrackDataError(msg)
            log.warning("%s; using no-scene geometry", msg)
        elif scene_meta.get("sha256") and scene.sha256 != scene_meta["sha256"]:
            raise TrackDataError(
                f"{entry['id']}: scene {sp} changed since the scan (re-run tracks scan)"
            )
        else:
            status = scene_meta["status"]
            record.add("scene", scene.path, scene.sha256,
                       {"map": entry["map"], "bundle": scene.bundle, "build_id": scene.build_id,
                        "status": status, "resolved": str(scene.path.resolve())})  # fmt: skip
    elif conv.scene_policy == "require":
        raise TrackDataError(
            f"{entry['id']}: no usable scene for {entry['map']} (scene_policy require)"
        )
    flags = list(base_meta["flags"])
    if scene is None:
        flags.append("no-scene")
    flags += [f for f in scene_meta.get("flags", []) if scene is not None]
    meta = {**base_meta, "scene": status, "flags": sorted(set(flags))}
    key = _cache_key({"track": sha["track"], "race": sha["race"], "props": props.sha256,
                      "scene": scene.sha256 if scene else None, "conv": conv.fingerprint(),
                      "v": CONVERSION_VERSION})  # fmt: skip

    def build_liftoff() -> list[SimCourse]:
        geom = load_track_geometry(track, race)
        course = convert_liftoff(
            geom, props, conv,
            scene_colliders=scene.colliders if scene else None,
            scene_heightfields=scene.heightfields if scene else None,
            meta=meta,
        )  # fmt: skip
        return [course]

    courses = _cached(key, entry["id"], build_liftoff, cdir)
    for c in courses:
        c.meta.update(meta)
    return courses


def build_pool(
    mix: Sequence[dict[str, Any]],
    purpose: str,
    *,
    conversion: ConversionConfig | None = None,
    root: str | Path | None = None,
    scenes_root: str | Path | None = None,
    catalogues: dict[str, Catalogue] | None = None,
    use_cache: bool = True,
) -> Pool:
    """Resolve a mix into a :class:`Pool` (``purpose`` ``train`` = leakage guarded)."""
    conv = conversion or ConversionConfig.from_dict(None)
    cats = dict(catalogues or {})
    record = LoadRecord()
    courses: list[SimCourse] = []
    groups: list[PoolGroup] = []
    props_cache: dict[Path, PropsTable] = {}
    for g in mix:
        sel = cat.parse_selector(str(g["select"]))
        if sel.source not in cats:
            cats[sel.source] = cat.load_named(sel.source)
        c = cats[sel.source]
        entries = cat.select(c, sel, purpose)
        idx = []
        for e in entries:
            r = cat.resolve_root(root, c.header) if e["source"] == "liftoff-xml" else None
            sr = cat.resolve_scenes_root(scenes_root, c.header, r) if r is not None else None
            props = None
            if r is not None:
                if r not in props_cache:
                    props_cache[r] = PropsTable.load(r / "props.json")
                props = props_cache[r]
            for course in load_course(e, catalogue=c, root=r, scenes_root=sr, conversion=conv,
                                      props=props, record=record, use_cache=use_cache):  # fmt: skip
                course.meta["group"] = str(g["select"])
                idx.append(len(courses))
                courses.append(course)
        groups.append(PoolGroup(str(g["select"]), idx, float(g.get("weight", 1.0)),
                                float(g.get("scene_weight", 1.0))))  # fmt: skip
    return Pool(courses, groups, purpose, cats, record)
