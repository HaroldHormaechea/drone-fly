"""Track catalogues (``tracks/<source>.yaml``): schema, selectors, the split and the leakage guard.

A catalogue is a committed YAML file per source (``tracks/liftoff.yaml``,
``tracks/procedural.yaml``; the schema is documented in ``tracks/README.md``). It names files and
records hashes and derived stats only: never coordinates, caches or absolute local paths.

**Selectors** are ``<source>:<term>``: a split (``train`` / ``val`` / ``held-out`` / ``all``), an
entry id (``liftoff:liftoff-arena-04``, ``procedural:proc-liftoff``) or ``map=<Map>``.

**The split (UC68 D6)** is by map, filled by track share and independent of which scene caches
exist: maps are ordered by ``sha256("<seed>:<map>")``; held-out takes maps until it has at least
``round(ratio * N)`` tracks and 2 maps, then val likewise, and the rest is train. On a re-scan
existing maps keep their split, new tracks inherit their map's split and a new map is bucketed
by its hash against the cumulative ratios.

**Leakage guard (AC9).** :func:`select` is the single choke point: ``purpose="train"`` raises
:class:`TrackLeakageError` for any entry whose split is not ``train``. Training never loads a
course any other way.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from drone_fly.harness.errors import HarnessConfigError

SCHEMA_VERSION = 1
SOURCES = ("liftoff-xml", "procedural")
#: Selector source name -> catalogue file stem and entry ``source``.
SELECTOR_SOURCES = {
    "liftoff": ("liftoff", "liftoff-xml"),
    "procedural": ("procedural", "procedural"),
}
SPLITS = ("train", "val", "held-out")
DEFAULT_SEED = 68
DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "held-out": 0.15}
MIN_MAPS_PER_SPLIT = 2
TRACKDATA_ENV = "DRONE_FLY_TRACKDATA"
SCENES_ENV = "DRONE_FLY_SCENES"
DEFAULT_ROOT = "artifacts/harness/trackdata"  # legacy header value: means the data root
ENTRY_KEYS = ("id", "source", "map", "gate_types", "laps", "split", "status")
STATUS_OK = "ok"


class TrackDataError(HarnessConfigError):
    """Missing / unreadable / inconsistent track data or catalogue (config-error exit code)."""


class TrackLeakageError(TrackDataError):
    """A non-train catalogue entry was requested for training (AC9)."""


@dataclass
class Catalogue:
    """One loaded catalogue file (``header`` = everything but ``entries``)."""

    path: Path | None
    header: dict[str, Any]
    entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def source(self) -> str:
        return str(self.header.get("source"))

    def by_id(self, entry_id: str) -> dict[str, Any]:
        for e in self.entries:
            if e["id"] == entry_id:
                return e
        raise TrackDataError(f"no catalogue entry {entry_id!r} in {self.path}")

    def maps(self) -> dict[str, Any]:
        return dict(self.header.get("maps") or {})

    def as_dict(self) -> dict[str, Any]:
        return {**self.header, "entries": self.entries}


def repo_root() -> Path:
    from drone_fly.harness.config import find_repo_root

    return find_repo_root() or Path.cwd()


def catalogue_dir() -> Path:
    return repo_root() / "tracks"


def catalogue_path(name: str) -> Path:
    """``tracks/<name>.yaml`` for a selector source name (``liftoff`` / ``procedural``)."""
    stem = SELECTOR_SOURCES.get(name, (name, name))[0]
    return catalogue_dir() / f"{stem}.yaml"


def validate_entry(e: Any, where: str = "entry") -> None:
    if not isinstance(e, dict):
        raise TrackDataError(f"{where} must be a mapping")
    missing = [k for k in ENTRY_KEYS if k not in e]
    if missing:
        raise TrackDataError(f"{where} {e.get('id', '?')!r} is missing {missing}")
    if e["source"] not in SOURCES:
        raise TrackDataError(f"{where} {e['id']!r}: source must be one of {SOURCES}")
    if e["split"] not in SPLITS:
        raise TrackDataError(f"{where} {e['id']!r}: split must be one of {SPLITS}")
    st = str(e["status"])
    if st != STATUS_OK and not st.startswith("unconvertible:"):
        raise TrackDataError(f"{where} {e['id']!r}: status must be ok or 'unconvertible: <reason>'")
    if e["source"] == "liftoff-xml":
        for k in ("track", "race"):
            if not e.get(k):
                raise TrackDataError(f"{where} {e['id']!r} needs '{k}' (file relative to the root)")
    else:
        if not isinstance(e.get("spec"), dict):
            raise TrackDataError(f"{where} {e['id']!r} (procedural) needs a 'spec' mapping")
        if e["split"] != "train":
            raise TrackDataError(f"procedural entry {e['id']!r} must be split: train")


def load_catalogue(path: str | Path) -> Catalogue:
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise TrackDataError(f"cannot read track catalogue {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise TrackDataError(f"{p}: a catalogue must be a mapping")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise TrackDataError(f"{p}: unsupported schema_version {data.get('schema_version')!r}")
    if data.get("source") not in SOURCES:
        raise TrackDataError(f"{p}: source must be one of {SOURCES}")
    entries = data.pop("entries", None) or []
    ids = set()
    for i, e in enumerate(entries):
        validate_entry(e, f"{p.name} entries[{i}]")
        if e["id"] in ids:
            raise TrackDataError(f"{p}: duplicate entry id {e['id']!r}")
        ids.add(e["id"])
    return Catalogue(p, data, list(entries))


def dump_catalogue(cat: Catalogue, path: str | Path, comment: str | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(cat.as_dict(), sort_keys=False, width=100, allow_unicode=True)
    if comment:
        text = "".join(f"# {line}\n" if line else "#\n" for line in comment.splitlines()) + text
    p.write_text(text, encoding="utf-8")
    return p


def load_named(name: str) -> Catalogue:
    path = catalogue_path(name)
    if not path.is_file():
        raise TrackDataError(f"no catalogue {path} (run `drone-fly tracks scan --source {name}`)")
    return load_catalogue(path)


# ----------------------------------------------------------------------------- selectors


@dataclass(frozen=True)
class Selector:
    source: str  # selector source name: liftoff | procedural
    term: str

    def __str__(self) -> str:
        return f"{self.source}:{self.term}"


def parse_selector(text: str) -> Selector:
    if ":" not in text:
        raise TrackDataError(f"track selector {text!r} must be <source>:<split|id|map=Map>")
    src, term = text.split(":", 1)
    if src not in SELECTOR_SOURCES:
        raise TrackDataError(f"unknown track source {src!r} ({sorted(SELECTOR_SOURCES)})")
    if not term:
        raise TrackDataError(f"track selector {text!r} has an empty term")
    return Selector(src, term)


def matches(entry: dict[str, Any], term: str) -> bool:
    if term == "all":
        return True
    if term in SPLITS:
        return entry["split"] == term
    if term.startswith("map="):
        return entry["map"] == term[4:]
    return entry["id"] == term


def select(
    catalogue: Catalogue,
    selector: str | Selector,
    purpose: str,
    *,
    include_unconvertible: bool = False,
) -> list[dict[str, Any]]:
    """Entries matching ``selector``. ``purpose="train"`` refuses any non-train entry (AC9).

    Unconvertible entries are skipped unless ``include_unconvertible``. A train request that
    matches a val / held-out entry raises :class:`TrackLeakageError` (it is never filtered out
    silently, so a wrong selector cannot shrink the training set unnoticed).
    """
    sel = parse_selector(selector) if isinstance(selector, str) else selector
    want_source = SELECTOR_SOURCES[sel.source][1]
    out = []
    for e in catalogue.entries:
        if e["source"] != want_source or not matches(e, sel.term):
            continue
        if purpose == "train" and e["split"] != "train":
            raise TrackLeakageError(
                f"selector {sel} matches {e['id']!r} (split {e['split']}): training may only "
                "load split: train entries (val is for selection, held-out for the final report)"
            )
        if e["status"] != STATUS_OK and not include_unconvertible:
            continue
        out.append(e)
    if not out:
        raise TrackDataError(f"selector {sel} matches no convertible entry in {catalogue.path}")
    return out


# ----------------------------------------------------------------------------- the split


def map_hash(seed: int, name: str) -> str:
    return hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()


def assign_split(
    track_counts: dict[str, int],
    *,
    seed: int = DEFAULT_SEED,
    ratios: dict[str, float] | None = None,
    existing: dict[str, str] | None = None,
) -> dict[str, str]:
    """``{map: split}`` (see the module docstring). A pure function of seed + maps + counts.

    With ``existing`` (a re-scan) every known map keeps its split; only new maps are assigned,
    bucketed by ``sha256(seed:map)`` against the cumulative ratios (train, val, held-out).
    """
    ratios = dict(ratios or DEFAULT_RATIOS)
    existing = dict(existing or {})
    if existing:
        out = {m: s for m, s in existing.items() if m in track_counts}
        for m in track_counts:
            if m in out:
                continue
            u = int(map_hash(seed, m)[:12], 16) / float(16**12)
            cum = 0.0
            out[m] = SPLITS[-1]
            for s in SPLITS:
                cum += ratios[s]
                if u < cum:
                    out[m] = s
                    break
        return out
    total = sum(track_counts.values())
    order = sorted(track_counts, key=lambda m: map_hash(seed, m))
    target = {s: int(round(ratios[s] * total)) for s in ("held-out", "val")}
    got_t = {s: 0 for s in SPLITS}
    got_m = {s: 0 for s in SPLITS}
    out = {}
    for m in order:
        for s in ("held-out", "val"):
            if got_t[s] < target[s] or got_m[s] < MIN_MAPS_PER_SPLIT:
                break
        else:
            s = "train"
        out[m] = s
        got_t[s] += track_counts[m]
        got_m[s] += 1
    return out


def split_counts(entries: Iterable[dict[str, Any]], splits: dict[str, str]) -> dict[str, Any]:
    counts: dict[str, Any] = {s: {"maps": 0, "tracks": 0} for s in SPLITS}
    for s in splits.values():
        counts[s]["maps"] += 1
    for e in entries:
        counts[e["split"]]["tracks"] += 1
    return counts


# ----------------------------------------------------------------------------- roots + ids


def resolve_root(explicit: str | os.PathLike[str] | None, header: dict[str, Any]) -> Path:
    """``--root`` -> ``$DRONE_FLY_TRACKDATA`` -> the header's ``root`` -> the data root's
    ``trackdata/`` (:func:`drone_fly.paths.trackdata_root`).

    A header ``root`` equal to the legacy :data:`DEFAULT_ROOT` (or none) means the data root: the
    committed catalogue keeps its bytes (its sha256 is part of every gate dataset manifest).
    """
    from drone_fly import paths

    raw = explicit or os.environ.get(TRACKDATA_ENV)
    if not raw:
        header_root = header.get("root")
        if not header_root or Path(header_root).as_posix() == DEFAULT_ROOT:
            return paths.trackdata_root()
        raw = header_root
    p = Path(raw).expanduser()
    return p if p.is_absolute() else repo_root() / p


def resolve_scenes_root(
    explicit: str | os.PathLike[str] | None, header: dict[str, Any], root: Path
) -> Path:
    """``--scenes-root`` -> ``$DRONE_FLY_SCENES`` -> header ``scenes_root`` -> the data root's
    ``scenes/`` when ``root`` is the data root's ``trackdata/``, else ``<root>/scenes``."""
    from drone_fly import paths

    raw = explicit or os.environ.get(SCENES_ENV) or header.get("scenes_root")
    if not raw:
        return paths.scenes_root() if root == paths.trackdata_root() else root / "scenes"
    p = Path(raw).expanduser()
    return p if p.is_absolute() else repo_root() / p


def kebab(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", name)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "-", s)
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entries_for(cat: Catalogue, ids: Sequence[str]) -> list[dict[str, Any]]:
    return [cat.by_id(i) for i in ids]
