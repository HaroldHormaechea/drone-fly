"""``drone-fly tracks scan --source liftoff``: pair, convert and catalogue the local track data.

1. **Pairing.** ``index.json`` maps each track uuid to its race (``tracks[uuid].race`` ->
   ``races[uuid].file``); a track the index does not pair falls back to the file-name pattern
   ``<Map>Track<NN>_<v>.xml`` <-> ``<Map>Race<NN>_<v>.xml``.
2. **Conversion.** Every pair is converted (unconvertible pairs are kept with their reason);
   per map, the scene cache (if any) is validated geometrically against the current XMLs with
   the synthetic witness and gets its status; the final stats come from the conversion with the
   usable scene (``allow-missing`` fallback otherwise).
3. **Split.** Existing map splits are kept; otherwise
   :func:`~drone_fly.tracks.catalogue.assign_split`.
4. **Write** ``tracks/liftoff.yaml``: names, hashes and derived stats only.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.offline.track_xml import load_track_geometry
from drone_fly.tracks import catalogue as cat
from drone_fly.tracks.liftoff import (
    CONVERSION_VERSION,
    ConversionConfig,
    PropsTable,
    UnconvertibleError,
    convert_liftoff,
    passage_witness,
)
from drone_fly.tracks.scenes import (
    USABLE,
    find_scene,
    scene_status,
    synthetic_witness,
    validate_scene,
)

log = logging.getLogger(__name__)

HARNESS_CAP_MIN = 64
OUTLIER_THRESHOLDS = {
    "spacing_max_m": 100.0,
    "altitude_span_m": 50.0,
    "extent_m": 500.0,
    "passages": 64,
    "aperture_min_m": 1.0,
}
_NAME = re.compile(r"^(?P<map>.+?)(?P<kind>Track|Race)(?P<num>\d+)_(?P<ver>\d+)\.xml$")


def pair_files(root: Path) -> list[dict[str, Any]]:
    """``[{map, track, race, num, name, uuids}]`` (index.json first, then the name pattern)."""
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    idx_path = root / "index.json"
    if idx_path.is_file():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        races = idx.get("races") or {}
        for tid, t in (idx.get("tracks") or {}).items():
            race = races.get(t.get("race") or "")
            if not race or not (root / t["file"]).is_file() or not (root / race["file"]).is_file():
                continue
            m = _NAME.match(t["file"])
            pairs.append({
                "map": t.get("environment") or (m.group("map") if m else "Unknown"),
                "track": t["file"], "race": race["file"],
                "num": m.group("num") if m else None, "ver": m.group("ver") if m else "0001",
                "name": t.get("name"), "uuids": {"track": tid, "race": t.get("race")},
            })  # fmt: skip
            seen.add(t["file"])
    for p in sorted(root.glob("*Track*_*.xml")):
        if p.name in seen:
            continue
        m = _NAME.match(p.name)
        if not m or m.group("kind") != "Track":
            continue
        race = root / f"{m.group('map')}Race{m.group('num')}_{m.group('ver')}.xml"
        if not race.is_file():
            continue
        pairs.append({"map": m.group("map"), "track": p.name, "race": race.name,
                      "num": m.group("num"), "ver": m.group("ver"), "name": None,
                      "uuids": {}})  # fmt: skip
    return pairs


def entry_id(pair: dict[str, Any]) -> str:
    num = pair.get("num") or "00"
    base = f"{cat.kebab(pair['map'])}-{int(num):02d}"
    ver = pair.get("ver") or "0001"
    return base if int(ver) == 1 else f"{base}-v{int(ver)}"


def outlier_flags(stats: dict[str, Any], th: dict[str, Any]) -> list[str]:
    out = []
    if stats.get("spacing_m", {}).get("max", 0) > th["spacing_max_m"]:
        out.append("outlier-spacing")
    if stats.get("altitude_span_m", 0) > th["altitude_span_m"]:
        out.append("outlier-altitude")
    if stats.get("extent_m", 0) > th["extent_m"]:
        out.append("outlier-extent")
    if stats.get("passages", 0) > th["passages"]:
        out.append("outlier-passages")
    if stats.get("aperture_m", {}).get("min", 1e9) < th["aperture_min_m"]:
        out.append("outlier-aperture")
    return out


def scan_liftoff(
    *,
    root: Path,
    scenes_root: Path,
    out_path: Path,
    seed: int = cat.DEFAULT_SEED,
    conversion: ConversionConfig | None = None,
    log_fn: Any = print,
) -> cat.Catalogue:
    """Scan ``root`` and write the catalogue (see the module docstring)."""
    conv = conversion or ConversionConfig.from_dict(None)
    existing: cat.Catalogue | None = cat.load_catalogue(out_path) if out_path.is_file() else None
    props = PropsTable.load(root / "props.json")
    idx_path = root / "index.json"
    idx_build = None
    if idx_path.is_file():
        idx_build = (json.loads(idx_path.read_text(encoding="utf-8")).get("source") or {}).get(
            "build_id"
        )
    current_build = str(idx_build or props.build_id) if (idx_build or props.build_id) else None
    pairs = pair_files(root)
    if not pairs:
        raise cat.TrackDataError(f"no track/race pairs under {root}")
    by_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        by_map[p["map"]].append(p)
    entries: list[dict[str, Any]] = []
    maps: dict[str, Any] = {}
    for mp in sorted(by_map):
        # pass 1: geometry + passages (no scene) for the scene validation witness
        prelim: dict[str, Any] = {}
        openings, spawns = [], []
        for p in by_map[mp]:
            try:
                geom = load_track_geometry(root / p["track"], root / p["race"])
                course = convert_liftoff(geom, props, conv, with_cells=False)
                ops, sp = passage_witness(course)
                openings += ops
                spawns.append(sp)
                prelim[p["track"]] = (geom, None)
            except UnconvertibleError as exc:
                prelim[p["track"]] = (None, exc.reason)
            except HarnessConfigError as exc:
                prelim[p["track"]] = (None, str(exc))
        scene = find_scene(scenes_root, mp)
        valid = None
        validation = None
        if scene is not None:
            gates_u, wit_u = synthetic_witness(openings, spawns)
            validation = validate_scene(scene, gates_u, wit_u)
            valid = validation["valid"]
        status = scene_status(scene, current_build, valid)
        summary = scene.summary(status, current_build) if scene else {"status": "missing"}
        if validation:
            summary["validation"] = validation
        maps[mp] = {"scene": summary}
        usable = scene if status in USABLE else None
        log_fn(f"{mp}: {len(by_map[mp])} tracks, scene {status}"
               + (f" (gates near {validation['near']:.2f}, witness inside "
                  f"{100 * validation['inside']:.2f} %)" if validation else ""))  # fmt: skip
        # pass 2: final conversion + stats
        for p in by_map[mp]:
            geom, reason = prelim[p["track"]]
            e: dict[str, Any] = {
                "id": entry_id(p), "source": "liftoff-xml", "track": p["track"], "race": p["race"],
                "map": mp, "gate_types": [], "laps": None, "split": "train", "status": "ok",
                "name": p.get("name"), "uuids": p.get("uuids") or {},
                "sha256": {"track": cat.sha256_file(root / p["track"]),
                           "race": cat.sha256_file(root / p["race"])},
            }  # fmt: skip
            if geom is not None:
                e["gate_types"] = sorted({it.item_id for it in geom.passages})
                e["laps"] = int(geom.laps or 1)
                try:
                    course = convert_liftoff(
                        geom, props, conv,
                        scene_colliders=usable.colliders if usable else None,
                        scene_heightfields=usable.heightfields if usable else None,
                    )  # fmt: skip
                    flags = list(course.meta.get("flags", []))
                    flags += outlier_flags(course.stats, OUTLIER_THRESHOLDS)
                    flags += ["no-scene"] if usable is None else summary.get("flags", [])
                    e["stats"] = course.stats
                    e["flags"] = sorted(set(flags))
                except UnconvertibleError as exc:
                    reason = exc.reason
            if reason is not None:
                e["status"] = f"unconvertible: {reason}"
                e.setdefault("flags", [])
            entries.append(e)
    # split
    counts = defaultdict(int)
    for e in entries:
        counts[e["map"]] += 1
    prev = {m: v["split"] for m, v in (existing.maps() if existing else {}).items() if "split" in v}
    prev_seed = (
        int(((existing.header.get("split") or {}).get("seed")) or seed) if existing else seed
    )
    splits = cat.assign_split(dict(counts), seed=prev_seed, existing=prev or None)
    for e in entries:
        e["split"] = splits[e["map"]]
    for mp in maps:
        maps[mp] = {"split": splits[mp], **maps[mp]}
    harness_cap = harness_cap_from_train(entries)
    for e in entries:
        st = e.get("stats")
        if st and "harness_candidates" in st:
            st["harness_list"] = min(st["harness_candidates"], harness_cap["value"])
            st["harness_overflow"] = max(0, st["harness_candidates"] - harness_cap["value"])
    entries.sort(key=lambda e: e["id"])
    header = {
        "schema_version": cat.SCHEMA_VERSION,
        "source": "liftoff-xml",
        "root": cat.DEFAULT_ROOT,
        "scenes_root": None,
        "world_scale": conv.world_scale,
        "conversion_version": CONVERSION_VERSION,
        "scanned": datetime.now(UTC).strftime("%Y-%m-%d"),
        "split": {
            "method": "map-hash-greedy: held-out, then val, to round(ratio*N) tracks + 2 maps",
            "seed": prev_seed,
            "ratios": dict(cat.DEFAULT_RATIOS),
            "counts": cat.split_counts(entries, splits),
        },
        "maps": maps,
        "root_files": {
            "index.json": {
                "sha256": cat.sha256_file(idx_path) if idx_path.is_file() else None,
                "build_id": idx_build,
            },
            "props.json": {"sha256": props.sha256, "build_id": props.build_id},
        },  # fmt: skip
        "harness_cap": harness_cap,
        "outlier_thresholds": dict(OUTLIER_THRESHOLDS),
    }
    catalogue = cat.Catalogue(out_path, header, entries)
    cat.dump_catalogue(
        catalogue,
        out_path,
        comment=(
            "Liftoff track catalogue (UC68), generated by\n"
            "`drone-fly tracks scan --source liftoff`; see tracks/README.md. Names, hashes and\n"
            "derived stats only: no game data, no coordinates, no absolute paths. Splits are\n"
            "stable across re-scans."
        ),
    )
    return catalogue


def harness_cap_from_train(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """``O_h = max(64, ceil(p99))`` of the TRAIN entries' uncapped harness-list sizes."""
    sizes = [
        e["stats"]["harness_candidates"]
        for e in entries
        if e["split"] == "train" and e["status"] == cat.STATUS_OK and "stats" in e
    ]
    p99 = float(np.percentile(sizes, 99)) if sizes else 0.0
    return {
        "value": int(max(HARNESS_CAP_MIN, math.ceil(p99))),
        "p99": round(p99, 2),
        "tracks": len(sizes),
        "source": "train stats: max(64, ceil(p99 of the train tracks' harness-list sizes))",
    }


def aggregate_stats(
    entries: list[dict[str, Any]], root: Path, conv: ConversionConfig
) -> dict[str, Any]:
    """Percentiles of leg spacing and apertures over the entries' convertible tracks."""
    props = PropsTable.load(root / "props.json")
    legs: list[float] = []
    aps: list[float] = []
    for e in entries:
        if e["status"] != cat.STATUS_OK or e["source"] != "liftoff-xml":
            continue
        geom = load_track_geometry(root / e["track"], root / e["race"])
        course = convert_liftoff(geom, props, conv, with_cells=False)
        pc = np.array([p.center for p in course.passages]) * conv.world_scale
        d = np.linalg.norm(np.diff(pc, axis=0), axis=1)
        legs += [float(x) for x in d]
        aps += [p.aperture * conv.world_scale for p in course.passages]

    def pct(v: list[float]) -> dict[str, float]:
        a = np.asarray(v)
        return {f"p{q}": round(float(np.percentile(a, q)), 3) for q in (10, 25, 50, 75, 90)}

    sizes = [e["stats"]["harness_candidates"] for e in entries if "harness_candidates" in
             (e.get("stats") or {})]  # fmt: skip
    hl = (
        {"harness_list": pct(sizes) | {"p99": round(float(np.percentile(sizes, 99)), 2)}}
        if sizes
        else {}
    )
    return {**hl, "tracks": len({e["id"] for e in entries}), "legs": len(legs),
            "spacing_m": pct(legs), "aperture_m": pct(aps)}  # fmt: skip
