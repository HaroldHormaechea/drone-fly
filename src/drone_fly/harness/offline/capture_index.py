"""Capture index (UC71): ``meta.json`` per capture, discovery, capture hash, catalogue linking.

Every capture used for multi-map gate training carries a ``meta.json`` that ties it to a track
catalogue entry (``tracks/liftoff.yaml``) and records how it was made:

* ``schema_version`` (1), ``name`` (= the folder name), ``catalogue_id``, ``map``, ``split``
  (informational only: the split is always re-read from the catalogue when the capture is used);
* ``track`` / ``race``: ``{file, sha256}`` of the catalogue entry's XMLs (from the catalogue);
* ``source``: ``manual`` | ``replay``; ``recording``: ``{name, sha256, origin}`` for a replay
  (``origin`` ``user`` | ``ai-attempt``), else ``null``;
* ``completeness``: ``{status: complete|incomplete|unknown, source, passed, required}``;
* ``pose_source``: ``udp`` | ``recording-xml``; ``time``: ``{offset_s, rate}`` of the pose clock
  against the frame clock (``0`` / ``1`` for UDP captures);
* ``camera``: ``{resolution, hfov_deg, uptilt_deg, source}`` — overrides the preset camera for
  this capture (an older drone / camera setup); ``masks``: extra masks (``FrameMasks`` entries)
  on top of the preset's, e.g. a replay HUD;
* ``drone``: the drone profile (``drones/<id>``) — the hook for more drones;
* ``alignment``: ``{verified, margin, lag_ms, uptilt_deg, frames, method}`` written by
  ``captures verify`` (:mod:`~drone_fly.harness.offline.align`); unverified captures are excluded
  from datasets;
* ``git_head``, ``created_utc``.

**Layout.** New captures live at ``artifacts/harness/captures/<Map>/<session>/`` with globally
unique session names ``<map-kebab>-<YYYY-MM-DD>-<replay|manual>-<NN>``; that name is what weights
metadata and ``detector_noise`` held-out windows key on. Legacy captures directly under
``captures/`` (the reference ``2026-10-11-infinity-manual``) stay where they are: discovery accepts
both depths, and ``captures register`` adds a ``meta.json`` in place.

**Capture hash.** sha256 over ``meta.json``, ``frames.csv``, the telemetry / pose file and a digest
of the sorted ``(frame file name, size)`` list (hashing every JPEG would read gigabytes).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from drone_fly.harness.config import find_repo_root
from drone_fly.harness.errors import HarnessConfigError

SCHEMA_VERSION = 1
META_FILE = "meta.json"
CAPTURES_DIR = Path("artifacts") / "harness" / "captures"  # legacy, before the data root
SOURCES = ("manual", "replay")
ORIGINS = ("user", "ai-attempt")
POSE_SOURCES = ("udp", "recording-xml")
COMPLETENESS = ("complete", "incomplete", "unknown")
TELEMETRY_FILES = ("telemetry_raw.csv", "telemetry.csv")
SESSION_RE = re.compile(r"^(?P<map>[a-z0-9-]+)-(?P<date>\d{4}-\d{2}-\d{2})-(?P<mode>replay|manual)-"
                        r"(?P<nn>\d{2,})$")  # fmt: skip
_REQUIRED = ("schema_version", "name", "catalogue_id", "map", "track", "race", "source",
             "completeness", "pose_source", "time", "camera", "masks", "drone", "alignment",
             "created_utc")  # fmt: skip


class CaptureMetaError(HarnessConfigError):
    """A capture's ``meta.json`` is missing, malformed or inconsistent with the catalogue."""


def captures_root() -> Path:
    """The data root's ``captures/`` (:func:`drone_fly.paths.captures_root`)."""
    from drone_fly import paths

    return paths.captures_root()


def session_name(map_name: str, date: str, mode: str, nn: int) -> str:
    """``<map-kebab>-<YYYY-MM-DD>-<replay|manual>-<NN>`` (see the module docstring)."""
    from drone_fly.tracks.catalogue import kebab

    if mode not in SOURCES:
        raise CaptureMetaError(f"capture mode must be one of {list(SOURCES)}, got {mode!r}")
    return f"{kebab(map_name)}-{date}-{mode}-{nn:02d}"


def next_session_dir(root: Path, map_name: str, date: str, mode: str) -> Path:
    """The first free ``<root>/<Map>/<session>`` for that map, date and mode."""
    names = {p.name for p in iter_capture_dirs(root)}
    nn = 1
    while session_name(map_name, date, mode, nn) in names:
        nn += 1
    return root / map_name / session_name(map_name, date, mode, nn)


# ----------------------------------------------------------------------------- schema


def _need(cond: bool, where: str, msg: str) -> None:
    if not cond:
        raise CaptureMetaError(f"{where}: {msg}")


def validate_meta(d: Any, where: str = META_FILE) -> dict[str, Any]:
    """Check a ``meta.json`` mapping against the schema; returns it unchanged."""
    _need(isinstance(d, dict), where, "must be a JSON object")
    missing = [k for k in _REQUIRED if k not in d]
    _need(not missing, where, f"missing keys {missing}")
    _need(d["schema_version"] == SCHEMA_VERSION, where,
          f"schema_version {d['schema_version']!r} is not {SCHEMA_VERSION}")  # fmt: skip
    for k in ("name", "catalogue_id", "map", "drone"):
        _need(isinstance(d[k], str) and bool(d[k]), where, f"{k} must be a non-empty string")
    for k in ("track", "race"):
        f = d[k]
        _need(isinstance(f, dict) and isinstance(f.get("file"), str) and bool(f.get("sha256")),
              where, f"{k} must be {{file, sha256}}")  # fmt: skip
    _need(d["source"] in SOURCES, where, f"source must be one of {list(SOURCES)}")
    rec = d.get("recording")
    if d["source"] == "replay":
        _need(isinstance(rec, dict) and bool(rec.get("name")) and bool(rec.get("sha256"))
              and rec.get("origin") in ORIGINS, where,
              f"a replay needs recording {{name, sha256, origin in {list(ORIGINS)}}}")  # fmt: skip
    else:
        _need(rec is None, where, "recording must be null unless source is replay")
    c = d["completeness"]
    _need(isinstance(c, dict) and c.get("status") in COMPLETENESS, where,
          f"completeness.status must be one of {list(COMPLETENESS)}")  # fmt: skip
    _need(d["pose_source"] in POSE_SOURCES, where,
          f"pose_source must be one of {list(POSE_SOURCES)}")  # fmt: skip
    t = d["time"]
    _need(isinstance(t, dict) and _num(t.get("offset_s")) and _num(t.get("rate"))
          and t["rate"] > 0, where, "time must be {offset_s, rate > 0}")  # fmt: skip
    cam = d["camera"]
    _need(isinstance(cam, dict), where, "camera must be an object")
    res = cam.get("resolution")
    _need(isinstance(res, list) and len(res) == 2 and all(isinstance(v, int) and v > 0
          for v in res), where, "camera.resolution must be [width, height]")  # fmt: skip
    _need(_num(cam.get("hfov_deg")) and 0 < cam["hfov_deg"] < 180, where,
          "camera.hfov_deg must be in (0, 180)")  # fmt: skip
    _need(_num(cam.get("uptilt_deg")), where, "camera.uptilt_deg must be a number")
    _need(isinstance(cam.get("source"), str), where, "camera.source must be a string")
    _need(isinstance(d["masks"], list), where, "masks must be a list")
    a = d["alignment"]
    _need(isinstance(a, dict) and isinstance(a.get("verified"), bool), where,
          "alignment must be an object with a boolean verified")  # fmt: skip
    return d


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def read_meta(capture_dir: str | Path) -> dict[str, Any]:
    """The validated ``meta.json`` of a capture folder (:class:`CaptureMetaError` if none)."""
    p = Path(capture_dir) / META_FILE
    if not p.is_file():
        raise CaptureMetaError(
            f"{Path(capture_dir)} has no {META_FILE}; run `drone-fly harness captures register "
            f"{capture_dir} --track <catalogue-id>`"
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CaptureMetaError(f"{p} is not valid JSON: {exc}") from exc
    validate_meta(data, str(p))
    if data["name"] != Path(capture_dir).name:
        raise CaptureMetaError(f"{p}: name {data['name']!r} != folder {Path(capture_dir).name!r}")
    return data


def write_meta(capture_dir: str | Path, meta: dict[str, Any]) -> Path:
    validate_meta(meta)
    p = Path(capture_dir) / META_FILE
    p.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


# ----------------------------------------------------------------------------- discovery


def is_capture_dir(p: Path) -> bool:
    return (p / "frames.csv").is_file() and any((p / f).is_file() for f in TELEMETRY_FILES)


def iter_capture_dirs(root: str | Path | None = None) -> list[Path]:
    """Capture folders under ``root``: ``<root>/<session>`` (legacy), ``<root>/<Map>/<session>``."""
    base = Path(root) if root is not None else captures_root()
    if not base.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if is_capture_dir(p):
            out.append(p)
            continue
        out.extend(q for q in sorted(p.iterdir()) if q.is_dir() and is_capture_dir(q))
    return out


@dataclass(frozen=True)
class IndexedCapture:
    path: Path
    meta: dict[str, Any] | None  # None: no / invalid meta.json
    error: str | None = None

    @property
    def name(self) -> str:
        return self.path.name


def discover(root: str | Path | None = None) -> list[IndexedCapture]:
    """Every capture folder with its meta (or the reason it has none); duplicate names refused."""
    out, seen = [], {}
    for p in iter_capture_dirs(root):
        if p.name in seen:
            raise CaptureMetaError(f"capture name {p.name!r} is not unique: {seen[p.name]} and {p}")
        seen[p.name] = p
        try:
            out.append(IndexedCapture(p, read_meta(p)))
        except CaptureMetaError as exc:
            out.append(IndexedCapture(p, None, str(exc)))
    return out


def resolve_captures(specs: list[str], root: str | Path | None = None) -> list[Path]:
    """``--captures`` arguments (folders, globs, or bare capture names) -> capture folders."""
    import glob

    known = {p.name: p for p in iter_capture_dirs(root)}
    out: list[Path] = []
    for s in specs:
        p = Path(s)
        if p.is_dir() and is_capture_dir(p):
            out.append(p)
        elif s in known:
            out.append(known[s])
        else:
            hits = [Path(h) for h in sorted(glob.glob(s)) if is_capture_dir(Path(h))]
            if not hits:
                raise CaptureMetaError(f"no capture folder matches {s!r}")
            out.extend(hits)
    uniq: dict[str, Path] = {}
    for p in out:
        uniq.setdefault(str(p.resolve()), p)
    return list(uniq.values())


# ----------------------------------------------------------------------------- hash


def telemetry_file(capture_dir: Path) -> Path:
    for f in TELEMETRY_FILES:
        if (capture_dir / f).is_file():
            return capture_dir / f
    raise CaptureMetaError(f"{capture_dir} has no telemetry ({' / '.join(TELEMETRY_FILES)})")


def capture_sha256(capture_dir: str | Path) -> str:
    """See the module docstring; stable across copies and machines."""
    d = Path(capture_dir)
    h = hashlib.sha256()
    for f in (d / META_FILE, d / "frames.csv", telemetry_file(d)):
        h.update(f.name.encode())
        h.update(f.read_bytes() if f.is_file() else b"<missing>")
    frames = d / "frames"
    listing = sorted(
        (q.name, q.stat().st_size) for q in (frames.iterdir() if frames.is_dir() else ())
        if q.is_file()
    )  # fmt: skip
    h.update(json.dumps(listing).encode())
    return h.hexdigest()


# ----------------------------------------------------------------------------- catalogue


def link_entry(meta: dict[str, Any], catalogue: Any) -> dict[str, Any]:
    """The catalogue entry of ``meta.catalogue_id``; map and track/race hashes must agree."""
    from drone_fly.tracks.catalogue import TrackDataError

    try:
        e = catalogue.by_id(meta["catalogue_id"])
    except (TrackDataError, KeyError) as exc:
        raise CaptureMetaError(
            f"capture {meta['name']}: catalogue id {meta['catalogue_id']!r} is unknown"
        ) from exc
    if e.get("map") != meta["map"]:
        raise CaptureMetaError(
            f"capture {meta['name']}: map {meta['map']!r} != catalogue {e.get('map')!r}"
        )
    sha = e.get("sha256") or {}
    for k in ("track", "race"):
        if sha.get(k) and sha[k] != meta[k]["sha256"]:
            raise CaptureMetaError(
                f"capture {meta['name']}: {k} sha256 differs from the catalogue entry "
                f"{e['id']} (the track changed since the capture)"
            )
    return e


def split_of(meta: dict[str, Any], catalogue: Any) -> str:
    """The capture's split, always from the catalogue (never from ``meta.split``)."""
    return str(link_entry(meta, catalogue)["split"])


def camera_config(base_camera: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, Any]:
    """The preset ``camera`` section with this capture's camera and extra masks applied."""
    cam = dict(base_camera)
    if not meta:
        return cam
    c = meta["camera"]
    cam["resolution"] = list(c["resolution"])
    cam["hfov_deg"] = float(c["hfov_deg"])
    cam["uptilt_deg"] = float(c["uptilt_deg"])
    if meta.get("masks"):
        cam["masks"] = [*(base_camera.get("masks") or []), *meta["masks"]]
    return cam


# ----------------------------------------------------------------------------- register


def completeness_from_capture(cap: Any, course: Any) -> dict[str, Any]:
    """Completeness source 2: the capture's pose replayed through the env pass rule."""
    from drone_fly.harness.offline.detector_noise import truth_progress
    from drone_fly.sim.pool import total_crossings

    ks = truth_progress(cap, course)
    required = int(total_crossings(course.n_passages, course.laps, course.circuit))
    passed = int(ks.max()) if len(ks) else 0
    return {
        "status": "complete" if passed >= required else "incomplete",
        "source": "pose",
        "passed": passed,
        "required": required,
    }


def build_meta(
    capture_dir: Path,
    entry: dict[str, Any],
    *,
    camera: dict[str, Any],
    drone: str,
    source: str = "manual",
    recording: dict[str, Any] | None = None,
    completeness: dict[str, Any] | None = None,
    pose_source: str = "udp",
    time: dict[str, Any] | None = None,
    masks: list[Any] | None = None,
    camera_source: str = "preset",
) -> dict[str, Any]:
    """A fresh ``meta.json`` (alignment unverified) for ``capture_dir`` and catalogue ``entry``."""
    from drone_fly.harness.recording import git_head

    sha = entry.get("sha256") or {}
    meta = {
        "schema_version": SCHEMA_VERSION,
        "name": capture_dir.name,
        "catalogue_id": entry["id"],
        "map": entry["map"],
        "split": entry.get("split"),
        "track": {"file": entry["track"], "sha256": sha.get("track")},
        "race": {"file": entry["race"], "sha256": sha.get("race")},
        "source": source,
        "recording": recording,
        "completeness": completeness
        or {"status": "unknown", "source": "none", "passed": None, "required": None},
        "pose_source": pose_source,
        "time": time or {"offset_s": 0.0, "rate": 1.0},
        "camera": {
            "resolution": [int(v) for v in camera["resolution"]],
            "hfov_deg": float(camera["hfov_deg"]),
            "uptilt_deg": float(camera["uptilt_deg"]),
            "source": camera_source,
        },
        "masks": list(masks or []),
        "drone": drone if drone.startswith("drones/") else f"drones/{drone}",
        "alignment": {"verified": False, "margin": None},
        "git_head": git_head(find_repo_root()),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return validate_meta(meta)


def register(
    capture_dir: str | Path,
    catalogue_id: str,
    *,
    config: Any = None,
    preset: str | None = "liftoff",
    trackdata_root: str | Path | None = None,
    source: str = "manual",
    force: bool = False,
) -> dict[str, Any]:
    """Back-fill ``meta.json`` for an existing capture (AC4); returns the written meta.

    The camera and drone come from the harness config; completeness is judged from the capture's
    own telemetry when the track files are available (else ``unknown``).
    """
    from drone_fly.harness.config import load_config
    from drone_fly.harness.offline.capture import load_capture
    from drone_fly.tracks import catalogue as cat
    from drone_fly.tracks.catalogue import TrackDataError
    from drone_fly.tracks.loader import load_course

    d = Path(capture_dir)
    if not is_capture_dir(d):
        raise CaptureMetaError(f"{d} is not a capture folder (frames.csv + telemetry)")
    if (d / META_FILE).exists() and not force:
        raise CaptureMetaError(f"{d / META_FILE} exists; pass --force to rewrite it")
    cfg = load_config("liftoff", config, preset=preset)
    c = cat.load_named("liftoff")
    try:
        entry = c.by_id(catalogue_id)
    except (TrackDataError, KeyError) as exc:
        raise CaptureMetaError(f"no liftoff catalogue entry {catalogue_id!r}") from exc
    completeness = None
    try:
        root = cat.resolve_root(trackdata_root, c.header)
        course = load_course(entry, catalogue=c, root=root, scenes_root=None)[0]
        cap = load_capture(d, cfg.section("telemetry"), 1.0)
        completeness = completeness_from_capture(cap, course)
    except (TrackDataError, HarnessConfigError, OSError) as exc:
        completeness = {"status": "unknown", "source": "none", "passed": None, "required": None,
                        "reason": str(exc)}  # fmt: skip
    meta = build_meta(
        d,
        entry,
        camera=cfg.section("camera"),
        drone=str(cfg.data.get("drone") or "unknown"),
        source=source,
        completeness=completeness,
    )
    write_meta(d, meta)
    return meta
