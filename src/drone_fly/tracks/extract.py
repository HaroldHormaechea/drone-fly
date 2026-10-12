"""``tracks extract-scenes`` / ``extract-props``: optional scene and props extraction (UC68 D4).

Reads the installed Liftoff bundles only (never launches the game). For every map without a
valid cache it builds the map's fingerprint (gate points + free-space witness: a real witness
file when given, else the synthetic one), reads each candidate scene bundle ONCE, scores it
against every map, assigns bundles uniquely (:func:`.liftoff_extract.scene.assign`) and writes the
``ok`` maps' caches into the scenes root. Ambiguous or unmatched maps keep no cache (``no-scene``);
nothing here blocks training.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.offline.track_xml import load_track_geometry
from drone_fly.tracks import catalogue as cat
from drone_fly.tracks.liftoff import (
    ConversionConfig,
    PropsTable,
    UnconvertibleError,
    convert_liftoff,
    passage_witness,
)
from drone_fly.tracks.scenes import USABLE, synthetic_witness

log = logging.getLogger(__name__)


def map_clouds(root: Path, maps: list[str] | None = None) -> dict[str, Any]:
    """``{map: MapCloud}`` (Unity frame) from the current XMLs (synthetic witness)."""
    from drone_fly.tracks.liftoff_extract.scene import MapCloud
    from drone_fly.tracks.scan import pair_files

    props = PropsTable.load(root / "props.json")
    conv = ConversionConfig.from_dict(None)
    ops: dict[str, list] = defaultdict(list)
    spawns: dict[str, list] = defaultdict(list)
    for p in pair_files(root):
        if maps and p["map"] not in maps:
            continue
        try:
            geom = load_track_geometry(root / p["track"], root / p["race"])
            course = convert_liftoff(geom, props, conv, with_cells=False)
        except (UnconvertibleError, HarnessConfigError):
            continue
        o, sp = passage_witness(course)
        ops[p["map"]] += o
        spawns[p["map"]].append(sp)
    out = {}
    for m in ops:
        g, w = synthetic_witness(ops[m], spawns[m])
        out[m] = MapCloud(g, w)
    return out


def extract_scenes(
    *,
    maps: list[str] | None = None,
    bundles: dict[str, str] | None = None,
    witnesses: dict[str, str] | None = None,
    force: bool = False,
    root: str | None = None,
    scenes_root: str | None = None,
    game_dir: str | None = None,
    max_mb: int | None = None,
) -> dict[str, Any]:
    from drone_fly.tracks.liftoff_extract import game
    from drone_fly.tracks.liftoff_extract import scene as ex

    header = {}
    path = cat.catalogue_path("liftoff")
    known: dict[str, Any] = {}
    if path.is_file():
        c = cat.load_catalogue(path)
        header = c.header
        known = c.maps()
    r = cat.resolve_root(root, header)
    sroot = cat.resolve_scenes_root(scenes_root, header, r)
    clouds = map_clouds(r, maps)
    todo = sorted(
        m for m in clouds
        if force or ((known.get(m) or {}).get("scene") or {}).get("status") not in USABLE
    )  # fmt: skip
    if maps:
        todo = [m for m in todo if m in maps]
    for m, wpath in (witnesses or {}).items():
        if m in clouds:
            pts = np.asarray(json.loads(Path(wpath).read_text(encoding="utf-8")), np.float64)
            clouds[m] = ex.MapCloud(clouds[m].gates, pts.reshape(-1, 3))
    inst = game.find_game(game_dir)
    if inst is None:
        raise cat.TrackDataError("Liftoff install not found (pass --game-dir or set LIFTOFF_DIR)")
    uv = game.unity_version()
    bid = game.build_id(inst)
    bdir = game.bundle_dir(inst)
    ex.need_unitypy(uv)
    report: dict[str, Any] = {"maps": todo, "build_id": bid, "results": {}}
    if not todo:
        return report
    fixed = {m: b for m, b in (bundles or {}).items() if m in todo}
    candidates = [p for p in ex.scene_bundles(bdir)]
    if max_mb:
        candidates = [p for p in candidates if p.stat().st_size <= max_mb * 1024 * 1024]
    scores: dict[str, dict[str, tuple[float, float]]] = {m: {} for m in todo if m not in fixed}
    keep: dict[str, tuple[list, dict, list]] = {}
    started = time.time()
    for cand in candidates:
        t0 = time.time()
        try:
            cols, counts, hfs = ex.read_scene(cand, uv)
        except MemoryError:
            log.warning("%s: out of memory; skipped", cand.name)
            continue
        except Exception as exc:  # an unreadable bundle is skipped, as in the analyzer
            log.warning("%s: unreadable (%s); skipped", cand.name, str(exc)[:60])
            continue
        if len(cols) < ex.MIN_COLLIDERS and cand.name not in fixed.values():
            continue
        arr = ex.ColliderArrays.from_colliders(cols)
        passing = False
        for m in scores:
            near, inside = ex.score(arr, clouds[m])
            scores[m][cand.name] = (near, inside)
            passing |= near >= ex.NEED_FRACTION and inside <= ex.MAX_PATH_INSIDE
        if passing or cand.name in fixed.values():
            keep[cand.name] = (cols, counts, hfs)
        log.info("%s: %d colliders (%.1f s)", cand.name, len(cols), time.time() - t0)
    assigned = ex.assign(scores)
    for m, b in fixed.items():
        assigned[m] = {"status": "ok", "bundle": b, "near": None, "inside": None, "given": True}
    for m in todo:
        res = assigned.get(m, {"status": "missing"})
        if res["status"] == "ok" and res["bundle"] in keep:
            cols, counts, hfs = keep[res["bundle"]]
            how = ("bundle given" if res.get("given") else
                   f"{100 * res['near']:.0f}% gates near, {100 * res['inside']:.2f}% of the "
                   f"{'real' if m in (witnesses or {}) else 'synthetic'} witness in")  # fmt: skip
            out = ex.write_cache(
                sroot, m, bdir / res["bundle"], cols, counts, hfs, build_id=bid,
                unity_version=uv, identified_by=how, witness=(witnesses or {}).get(m),
            )  # fmt: skip
            res = {**res, "cache": str(out), "colliders": len(cols), "counts": counts,
                   "heightfields": len(hfs)}  # fmt: skip
        elif res["status"] == "ok":
            res = {**res, "status": "missing", "note": "assigned bundle was not kept"}
        report["results"][m] = res
    report["seconds"] = round(time.time() - started, 1)
    report["candidates"] = len(candidates)
    return report


def extract_props(
    *, force: bool = False, root: str | None = None, game_dir: str | None = None
) -> dict:
    """Rebuild ``<root>/props.json`` when missing or built from another game build."""
    from drone_fly.tracks.liftoff_extract import game
    from drone_fly.tracks.liftoff_extract.props import build_props
    from drone_fly.tracks.scan import pair_files

    header = {}
    path = cat.catalogue_path("liftoff")
    if path.is_file():
        header = cat.load_catalogue(path).header
    r = cat.resolve_root(root, header)
    inst = game.find_game(game_dir)
    if inst is None:
        raise cat.TrackDataError("Liftoff install not found (pass --game-dir or set LIFTOFF_DIR)")
    bid = game.build_id(inst)
    out = r / "props.json"
    if out.is_file() and not force:
        cur = PropsTable.load(out)
        if cur.build_id == bid:
            return {"status": "current", "build_id": bid, "path": str(out)}
    wanted: set[str] = set()
    for p in pair_files(r):
        for it in load_track_geometry(r / p["track"], None).items:
            wanted.add(it.item_id)
    table = build_props(wanted, game.bundle_dir(inst), out, build_id=bid,
                        unity_version=game.unity_version())  # fmt: skip
    return {"status": "rebuilt", "build_id": bid, "counts": table["counts"], "path": str(out)}
