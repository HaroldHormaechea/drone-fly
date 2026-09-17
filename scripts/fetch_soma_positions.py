#!/usr/bin/env python3
"""Fetch real anatomical soma positions for the committed fixture (UC-05, dev-time).

This is a **dev-time** utility (like ``scripts/build_test_fixture.py``). It downloads the
canonical ``connectome_data_prep`` MaleCNS neuron metadata and extracts the ``somaLocation``
column — ``[x y z]`` in neuPrint MaleCNS 8 nm-isotropic voxel space — for exactly the
bodyids in the committed test fixture, writing a small ``bodyid,x,y,z`` sidecar CSV that the
viewer's anatomical brain map reads offline.

Why this is tokenless (the tested-vs-untested boundary — AC7)
-------------------------------------------------------------
``somaLocation`` in ``mcns_all_neuron_meta.csv`` is the **same value** neuPrint's
``fetch_neurons().somaLocation`` returns, mirrored on the public ``connectome_data_prep``
GitHub repo under CC-BY. So the committed fixture ships **real anatomy with no token**, and
the anatomical map is the offline/CI default — not a fallback. A ``NEUPRINT_TOKEN`` is only
needed for arbitrary user slices whose bodyids are not in this committed sidecar
(handled at runtime by :mod:`drone_fly.record.coordinates`); neurons still lacking a soma
fall back to the deterministic computed layout, clearly labelled non-anatomical.

Tested boundary: this script is **run** by the dev-team (QA commits its output CSV, per the
UC-01 fixture precedent) and its 300/300 coverage is asserted. Not run in CI (needs network).

Usage
-----
    python scripts/fetch_soma_positions.py            # writes tests/fixtures/mcns_fixture_soma.csv
    python scripts/fetch_soma_positions.py --out <p>  # custom output path
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RAW_BASE = "https://raw.githubusercontent.com/YijieYin/connectome_data_prep/main/data"
DEFAULT_META_URL = f"{RAW_BASE}/maleCNS/mcns_all_neuron_meta.csv"
DEFAULT_FIXTURE_META = Path("tests/fixtures/mcns_fixture_meta.csv")
DEFAULT_OUT = Path("tests/fixtures/mcns_fixture_soma.csv")


def _parse_soma_location(value: object) -> tuple[float, float, float] | None:
    """Parse a ``"[x y z]"`` somaLocation cell → ``(x, y, z)`` floats, or ``None``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().strip("[]")
    if not text:
        return None
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        return None
    try:
        x, y, z = (float(p) for p in parts)
    except ValueError:
        return None
    if not all(np.isfinite(v) for v in (x, y, z)):
        return None
    return (x, y, z)


def fetch_soma_positions(
    fixture_meta: Path, out_path: Path, meta_url: str, cache_dir: Path
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    src_meta = cache_dir / Path(meta_url).name
    if not src_meta.exists():
        print(f"  downloading {meta_url}", file=sys.stderr)
        urllib.request.urlretrieve(meta_url, src_meta)  # noqa: S310 - fixed trusted https URL

    fixture = pd.read_csv(fixture_meta)
    if "bodyid" not in fixture.columns:
        raise SystemExit(f"Fixture meta {fixture_meta} lacks a 'bodyid' column.")
    body_ids = [int(b) for b in fixture["bodyid"].tolist()]

    meta = pd.read_csv(
        src_meta, usecols=lambda c: c in {"bodyid", "somaLocation"}, low_memory=False
    )
    meta = meta[meta["bodyid"].isin(body_ids)]
    positions: dict[int, tuple[float, float, float]] = {}
    for _, row in meta.iterrows():
        xyz = _parse_soma_location(row.get("somaLocation"))
        if xyz is not None:
            positions[int(row["bodyid"])] = xyz

    rows = []
    missing = []
    for bid in body_ids:
        if bid in positions:
            x, y, z = positions[bid]
            rows.append({"bodyid": bid, "x": x, "y": y, "z": z})
        else:
            missing.append(bid)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["bodyid", "x", "y", "z"]).to_csv(out_path, index=False)

    total = len(body_ids)
    print(f"Wrote {len(rows)}/{total} soma positions to {out_path}")
    if missing:
        print(
            f"  {len(missing)} bodyids had no somaLocation (e.g. {missing[:8]}).", file=sys.stderr
        )
        print(
            "  Runtime provisioning fallback-places these (flagged non-anatomical).",
            file=sys.stderr,
        )
    else:
        print(f"  Coverage: {len(rows)}/{total} = 100% real anatomical soma positions.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-meta", type=Path, default=DEFAULT_FIXTURE_META)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--meta-url", default=DEFAULT_META_URL)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "drone-fly" / "connectome_src",
    )
    args = parser.parse_args(argv)
    fetch_soma_positions(args.fixture_meta, args.out, args.meta_url, args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
