"""``drone-fly tracks list | check | scan | stats | extract-scenes | extract-props``.

* ``list [--split S] [--source liftoff|procedural]``: the catalogues as a table.
* ``check [--root] [--scenes-root] [--require-scenes] [--entry ID ...] [--gate-coverage]
  [--require-gate-coverage] [--preset P]``: every entry's files exist, match their hashes and
  convert; with ``--require-scenes`` a map without a usable scene fails. ``--gate-coverage``
  (UC71) also lists, per liftoff entry, the itemIDs the preset's gate-type table does not cover
  and the route passages with no visible gate (``drone_fly.tracks.gate_coverage``);
  ``--require-gate-coverage`` makes either a failure.
* ``scan --source liftoff [--root] [--scenes-root] [--seed]``: (re)write ``tracks/liftoff.yaml``.
* ``stats --split S``: leg spacing / aperture percentiles over ONE split (the procedural ranges
  are taken from ``--split train`` only).
* ``extract-scenes [--map M ...] [--bundle M=<name>] [--witness M=<points.json>] [--force]``:
  build scene caches for maps without a valid one (needs UnityPy; reads the installed game files,
  never launches the game). ``extract-props [--force]``: rebuild ``props.json`` if missing/stale.

Light at import (argparse only); the work modules load inside the commands.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def _cat_names(source: str | None) -> list[str]:
    return [source] if source else ["liftoff", "procedural"]


def cmd_list(args: argparse.Namespace) -> int:
    from drone_fly.tracks import catalogue as cat

    for name in _cat_names(args.source):
        path = cat.catalogue_path(name)
        if not path.is_file():
            print(f"{name}: no catalogue at {path}")
            continue
        c = cat.load_catalogue(path)
        maps = c.maps()
        rows = [e for e in c.entries if not args.split or e["split"] == args.split]
        print(f"{path.name}: {len(rows)} of {len(c.entries)} entries")
        print(
            f"  {'id':30s} {'map':22s} {'split':9s} {'laps':>4s} {'pass':>4s}  scene        status"
        )
        for e in rows:
            scene = ((maps.get(e["map"]) or {}).get("scene") or {}).get("status", "-")
            if e["source"] == "procedural":
                scene = "procedural"
            st = e.get("stats") or {}
            print(
                f"  {e['id']:30s} {e['map']:22s} {e['split']:9s} {e.get('laps') or '-':>4} "
                f"{st.get('passages', '-'):>4}  {scene:12s} {e['status']}"
            )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from drone_fly.tracks import catalogue as cat
    from drone_fly.tracks.liftoff import ConversionConfig
    from drone_fly.tracks.loader import load_course

    bad = 0
    conv = ConversionConfig.from_dict({"scene_policy": "require"} if args.require_scenes else None)
    coverage = args.gate_coverage or args.require_gate_coverage
    table = not_gates = None
    if coverage:
        from drone_fly.harness.config import load_config
        from drone_fly.harness.gate_types import GateTypeTable

        hc = load_config("liftoff", None, preset=args.preset)
        table, not_gates = GateTypeTable.from_config(hc.get("gate_types")), hc.not_gates
    wanted = set(args.entry or ())
    for name in _cat_names(args.source):
        c = cat.load_named(name)
        root = cat.resolve_root(args.root, c.header) if c.source == "liftoff-xml" else None
        sroot = cat.resolve_scenes_root(args.scenes_root, c.header, root) if root else None
        for e in c.entries:
            if wanted and e["id"] not in wanted:
                continue
            if e["status"] != cat.STATUS_OK:
                print(f"  skip {e['id']}: {e['status']}")
                continue
            try:
                courses = load_course(e, catalogue=c, root=root, scenes_root=sroot, conversion=conv)
                scene = courses[0].meta.get("scene")
                print(f"  ok   {e['id']} ({len(courses)} course(s), scene {scene})")
            except cat.TrackDataError as exc:
                bad += 1
                print(f"  FAIL {e['id']}: {exc}")
            if coverage and root is not None:
                bad += _check_coverage(e, root, table, not_gates, args.require_gate_coverage)
    missing = sorted(wanted - {e["id"] for n in _cat_names(args.source)
                               for e in cat.load_named(n).entries})  # fmt: skip
    if missing:
        bad += len(missing)
        print(f"  FAIL unknown entr(y/ies) {missing}")
    print("check: " + ("all entries convert" if not bad else f"{bad} entr(y/ies) failed"))
    return 0 if not bad else 1


def _check_coverage(e: dict[str, Any], root: Any, table: Any, not_gates: Any, strict: bool) -> int:
    """Print one entry's gate coverage; 1 if ``strict`` and it is not covered, else 0."""
    from drone_fly.harness.errors import HarnessConfigError
    from drone_fly.tracks.gate_coverage import entry_coverage

    try:
        cov = entry_coverage(e, root, table, not_gates)
    except (HarnessConfigError, OSError) as exc:
        print(f"       gates FAIL {e['id']}: {exc}")
        return 1
    tag = "gates ok  " if cov.ok else ("gates FAIL" if strict else "gates warn")
    print(f"       {tag} {e['id']}: {cov.summary()}")
    return int(strict and not cov.ok)


def cmd_scan(args: argparse.Namespace) -> int:
    from drone_fly.tracks import catalogue as cat
    from drone_fly.tracks.scan import scan_liftoff

    if args.source != "liftoff":
        raise cat.TrackDataError(
            "only --source liftoff is scanned (procedural.yaml is hand-written)"
        )
    out = cat.catalogue_path("liftoff")
    header = cat.load_catalogue(out).header if out.is_file() else {}
    root = cat.resolve_root(args.root, header)
    sroot = cat.resolve_scenes_root(args.scenes_root, header, root)
    c = scan_liftoff(root=root, scenes_root=sroot, out_path=out, seed=args.seed)
    counts = c.header["split"]["counts"]
    bad = sum(1 for e in c.entries if e["status"] != cat.STATUS_OK)
    print(
        f"wrote {out}: {len(c.entries)} entries ({bad} unconvertible); split "
        + ", ".join(f"{s} {v['tracks']} tracks / {v['maps']} maps" for s, v in counts.items())
    )
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from drone_fly.tracks import catalogue as cat
    from drone_fly.tracks.liftoff import ConversionConfig
    from drone_fly.tracks.scan import aggregate_stats

    c = cat.load_named("liftoff")
    entries = cat.select(c, f"liftoff:{args.split}", "stats")
    root = cat.resolve_root(args.root, c.header)
    res = aggregate_stats(entries, root, ConversionConfig.from_dict(None))
    res["split"] = args.split
    print(json.dumps(res, indent=2))
    return 0


def _pairs(values: list[str] | None, flag: str) -> dict[str, str]:
    from drone_fly.tracks.catalogue import TrackDataError

    out = {}
    for v in values or []:
        if "=" not in v:
            raise TrackDataError(f"{flag} must be MAP=VALUE, got {v!r}")
        k, val = v.split("=", 1)
        out[k] = val
    return out


def cmd_extract_scenes(args: argparse.Namespace) -> int:
    from drone_fly.tracks.extract import extract_scenes

    res = extract_scenes(
        maps=args.map,
        bundles=_pairs(args.bundle, "--bundle"),
        witnesses=_pairs(args.witness, "--witness"),
        force=args.force,
        root=args.root,
        scenes_root=args.scenes_root,
        game_dir=args.game_dir,
        max_mb=args.max_bundle_mb,
    )
    print(json.dumps(res, indent=2))
    print("re-run `drone-fly tracks scan --source liftoff` to record the new scene statuses")
    return 0


def cmd_extract_props(args: argparse.Namespace) -> int:
    from drone_fly.tracks.extract import extract_props

    res = extract_props(force=args.force, root=args.root, game_dir=args.game_dir)
    print(json.dumps(res, indent=2))
    return 0


def add_parser(sub: Any) -> None:
    tracks = sub.add_parser("tracks", help="track catalogues (tracks/*.yaml) and course data")
    tsub = tracks.add_subparsers(dest="tracks_command", required=True)
    p = tsub.add_parser("list", help="print the catalogue(s)")
    p.add_argument("--split", choices=("train", "val", "held-out"))
    p.add_argument("--source", choices=("liftoff", "procedural"))
    p.set_defaults(func=cmd_list)
    p = tsub.add_parser("check", help="validate that local files exist, match and convert")
    p.add_argument("--source", choices=("liftoff", "procedural"))
    p.add_argument("--root", help="trackdata root (default $DRONE_FLY_TRACKDATA / catalogue)")
    p.add_argument("--scenes-root", help="scene caches (default $DRONE_FLY_SCENES / <root>/scenes)")
    p.add_argument("--require-scenes", action="store_true", help="fail maps without a scene")
    p.add_argument("--entry", action="append", help="only these entry ids (repeatable)")
    p.add_argument("--gate-coverage", action="store_true",
                   help="report unknown gate itemIDs / volume-only passages (UC71)")  # fmt: skip
    p.add_argument("--require-gate-coverage", action="store_true",
                   help="fail entries with unknown itemIDs or volume-only passages")  # fmt: skip
    p.add_argument("--preset", default="liftoff", help="harness preset whose gate_types to use")
    p.set_defaults(func=cmd_check)
    p = tsub.add_parser("scan", help="scan the local trackdata and write tracks/<source>.yaml")
    p.add_argument("--source", required=True, choices=("liftoff",))
    p.add_argument("--root")
    p.add_argument("--scenes-root")
    p.add_argument("--seed", type=int, default=68, help="split seed (new catalogues only)")
    p.set_defaults(func=cmd_scan)
    p = tsub.add_parser("stats", help="spacing / aperture percentiles over one split")
    p.add_argument("--split", required=True, choices=("train", "val", "held-out"))
    p.add_argument("--root")
    p.set_defaults(func=cmd_stats)
    p = tsub.add_parser(
        "extract-scenes", help="build scene caches (UnityPy; never launches Liftoff)"
    )
    p.add_argument("--map", action="append", help="only these maps (repeatable)")
    p.add_argument("--bundle", action="append", metavar="MAP=NAME", help="known scene bundle")
    p.add_argument("--witness", action="append", metavar="MAP=POINTS",
                   help="JSON list of Unity-world [x, y, z] points flown in that map")  # fmt: skip
    p.add_argument("--force", action="store_true", help="also re-extract maps with a valid cache")
    p.add_argument("--root")
    p.add_argument("--scenes-root")
    p.add_argument("--game-dir", help="Liftoff install (default $LIFTOFF_DIR / Steam)")
    p.add_argument("--max-bundle-mb", type=int, default=None, help="skip larger scene bundles")
    p.set_defaults(func=cmd_extract_scenes)
    p = tsub.add_parser("extract-props", help="rebuild props.json if missing or stale (UnityPy)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--root")
    p.add_argument("--game-dir")
    p.set_defaults(func=cmd_extract_props)
