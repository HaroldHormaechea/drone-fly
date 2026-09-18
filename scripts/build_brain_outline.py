#!/usr/bin/env python3
"""Build the static MaleCNS brain-outline asset for the viewer (UC-12, dev-time).

This is a **dev-time** utility (like ``scripts/fetch_soma_positions.py``). It downloads the
canonical ``connectome_data_prep`` MaleCNS neuron metadata, parses **every** ``somaLocation``
value — ``[x y z]`` in neuPrint MaleCNS 8 nm-isotropic voxel space — and emits a compact,
committed JavaScript asset (``viz/brain_outline.js``) holding a per-plane concave silhouette
of the soma cloud plus its full 3-D bounding box. The viewer's anatomical panel overlays this
outline and registers the activation heatmap to it by construction: the outline is projected
from the *same* soma coordinate frame the recordings use, so no separate registration step is
needed.

Why this is tokenless (the tested-vs-untested boundary — mirrors ``fetch_soma_positions.py``)
---------------------------------------------------------------------------------------------
``somaLocation`` in ``mcns_all_neuron_meta.csv`` is the same value neuPrint's
``fetch_neurons().somaLocation`` returns, mirrored on the public ``connectome_data_prep``
GitHub repo under CC-BY. So the outline is derived from **real anatomy with no token**. This
script is **not** run in CI (it needs network to download the meta CSV on a cold cache); its
output ``viz/brain_outline.js`` is committed, exactly like ``tests/fixtures/mcns_fixture_soma.csv``.

Region selection (UC-12 condition C4)
-------------------------------------
The full MaleCNS meta covers the whole central nervous system (brain + ventral nerve cord).
If the CSV carries a usable region field (``superclass``), this script filters to **brain**
somas (optic lobe + central brain + visual + descending neurons) so a typical brain-circuit
recording lights up against a brain-shaped outline rather than a tiny speck of a whole-CNS
hull. The chosen region is recorded in the asset (``region``) and in the README. Registration
is exact either way — only the *readability* of the outline changes.

Usage
-----
    python scripts/build_brain_outline.py                 # writes viz/brain_outline.js
    python scripts/build_brain_outline.py --out <path>    # custom output path
    python scripts/build_brain_outline.py --region whole-cns   # skip the brain filter
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from contourpy import LineType, contour_generator
from scipy import ndimage

RAW_BASE = "https://raw.githubusercontent.com/YijieYin/connectome_data_prep/main/data"
DEFAULT_META_URL = f"{RAW_BASE}/maleCNS/mcns_all_neuron_meta.csv"
DEFAULT_OUT = Path("viz/brain_outline.js")

# Bare attribution string — MUST NOT contain an ``http(s)://`` literal (UC-12 asset is loaded
# from file://; a URL literal in the committed asset is what the viz-contract test forbids).
ATTRIBUTION = "MaleCNS connectome soma positions — connectome_data_prep (Yin et al.), CC-BY 4.0"
VOXEL_SPACE = "neuPrint MaleCNS 8nm"

# View-preset planes, mirroring viewer.js: MAP_VIEW_PRESETS/PROJECTIONS. Each plane picks the
# two coordinate axes (0=x, 1=y, 2=z) the panel projects onto.
PLANES = {
    "top": (0, 2),  # xz — dorsal / top-down (anatomically pinned)
    "front": (0, 1),  # xy
    "side": (1, 2),  # yz
}

# Brain superclasses (soma physically in the brain). Everything else — vnc_*, ascending_neuron,
# efferent_* — has its soma in the ventral nerve cord and is dropped for the brain outline.
_BRAIN_PREFIXES = ("ol_", "cb_", "visual")
_BRAIN_EXTRA = frozenset({"descending_neuron"})


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


def _is_brain(superclass: object) -> bool:
    if not isinstance(superclass, str):
        return False
    s = superclass.strip().lower()
    return s in _BRAIN_EXTRA or s.startswith(_BRAIN_PREFIXES)


def _load_points(src_meta: Path, region: str) -> tuple[np.ndarray, str]:
    """Load all soma xyz points, optionally filtered to the brain region.

    Returns ``(points Nx3, region_label)``. ``region_label`` is the region actually applied:
    ``"brain"`` when the filter ran, else ``"whole-CNS"``.
    """
    usecols = lambda c: c in {"somaLocation", "superclass"}  # noqa: E731
    meta = pd.read_csv(src_meta, usecols=usecols, low_memory=False)

    want_brain = region == "brain"
    have_superclass = "superclass" in meta.columns
    if want_brain and not have_superclass:
        print(
            "  no 'superclass' column found — shipping the whole-CNS hull.",
            file=sys.stderr,
        )
        want_brain = False

    if want_brain:
        mask = meta["superclass"].map(_is_brain)
        meta = meta[mask]
        region_label = "brain"
    else:
        region_label = "whole-CNS"

    pts = []
    for value in meta["somaLocation"]:
        xyz = _parse_soma_location(value)
        if xyz is not None:
            pts.append(xyz)
    if not pts:
        raise SystemExit("No usable somaLocation values found in the meta CSV.")
    return np.asarray(pts, dtype=float), region_label


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker polyline simplification (keeps shape, drops redundant verts)."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    seg = end - start
    seg_len = float(np.hypot(seg[0], seg[1]))
    if seg_len < 1e-9:
        dists = np.hypot(points[:, 0] - start[0], points[:, 1] - start[1])
    else:
        # perpendicular distance of each point to the start-end segment
        dists = (
            np.abs(seg[0] * (start[1] - points[:, 1]) - (start[0] - points[:, 0]) * seg[1])
            / seg_len
        )
    idx = int(np.argmax(dists))
    if dists[idx] > epsilon:
        left = _rdp(points[: idx + 1], epsilon)
        right = _rdp(points[idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def _silhouette(u: np.ndarray, v: np.ndarray, nbins: int, max_verts: int) -> list[list[int]]:
    """Concave silhouette polygon of a 2-D point cloud, in the cloud's own (u, v) units.

    Density-grid + largest-connected-component + hole-fill + marching-squares contour, then
    RDP-simplified to ``<= max_verts`` integer vertices. Returns ``[[u, v], ...]`` (closed).
    """
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    hist, u_edges, v_edges = np.histogram2d(
        u, v, bins=nbins, range=[[u_min, u_max], [v_min, v_max]]
    )
    # Binary occupancy → clean it up: close small gaps, fill interior holes, keep the largest
    # connected blob (drops stray far-flung somas that would otherwise spawn islands).
    mask = hist > 0
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
        mask = labels == (int(np.argmax(sizes)) + 1)
    # Smooth the binary field so the 0.5 contour is a soft silhouette rather than a staircase.
    field = ndimage.gaussian_filter(mask.astype(float), sigma=1.6)

    u_cen = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_cen = 0.5 * (v_edges[:-1] + v_edges[1:])
    # contourpy expects z[y, x]; our hist is [u, v] so transpose to [v, u].
    grid_u, grid_v = np.meshgrid(u_cen, v_cen)
    cg = contour_generator(x=grid_u, y=grid_v, z=field.T, line_type=LineType.Separate)
    lines = cg.lines(0.5)
    if not lines:
        # Degenerate (flat) cloud — fall back to the bbox rectangle.
        return [
            [int(round(u_min)), int(round(v_min))],
            [int(round(u_max)), int(round(v_min))],
            [int(round(u_max)), int(round(v_max))],
            [int(round(u_min)), int(round(v_max))],
            [int(round(u_min)), int(round(v_min))],
        ]
    poly = max(lines, key=len)
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])

    # Simplify: grow the RDP tolerance until the vertex count fits the budget.
    span = max(u_max - u_min, v_max - v_min)
    eps = span * 0.004
    simplified = _rdp(poly, eps)
    for _ in range(12):
        if len(simplified) <= max_verts:
            break
        eps *= 1.5
        simplified = _rdp(poly, eps)
    return [[int(round(p[0])), int(round(p[1]))] for p in simplified]


def build_outline(
    out_path: Path, meta_url: str, cache_dir: Path, region: str, nbins: int, max_verts: int
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    src_meta = cache_dir / Path(meta_url).name
    if not src_meta.exists():
        print(f"  downloading {meta_url}", file=sys.stderr)
        urllib.request.urlretrieve(meta_url, src_meta)  # noqa: S310 - fixed trusted https URL

    points, region_label = _load_points(src_meta, region)
    print(f"  {len(points)} soma points (region: {region_label})", file=sys.stderr)

    bbox_min = [int(round(points[:, k].min())) for k in range(3)]
    bbox_max = [int(round(points[:, k].max())) for k in range(3)]

    planes: dict[str, dict] = {}
    for name, (a0, a1) in PLANES.items():
        polygon = _silhouette(points[:, a0], points[:, a1], nbins, max_verts)
        planes[name] = {"axes": [a0, a1], "polygon": polygon}
        print(f"  plane {name} (axes {a0},{a1}): {len(polygon)} verts", file=sys.stderr)

    outline = {
        "voxel_space": VOXEL_SPACE,
        "attribution": ATTRIBUTION,
        "region": region_label,
        "bbox3d": {"min": bbox_min, "max": bbox_max},
        "planes": planes,
    }

    header = (
        "/* MaleCNS brain-outline asset — GENERATED by scripts/build_brain_outline.py "
        "(UC-12).\n"
        " * Do not hand-edit. Regenerate with: python scripts/build_brain_outline.py\n"
        " * Static silhouette of the soma cloud in neuPrint MaleCNS 8nm voxel space, one\n"
        " * concave polygon per view plane (top=xz, front=xy, side=yz) plus the full 3D\n"
        " * bbox. Loaded via a classic <script> before viewer.js (file://-safe, no fetch,\n"
        " * no build step); the viewer registers the activation heatmap to it directly. */\n"
    )
    body = "const BRAIN_OUTLINE = " + json.dumps(outline, separators=(",", ":")) + ";\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + body, encoding="utf-8")
    print(f"Wrote {out_path} (region: {region_label}, bbox {bbox_min}..{bbox_max})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--meta-url", default=DEFAULT_META_URL)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "drone-fly" / "connectome_src",
    )
    parser.add_argument(
        "--region",
        choices=["brain", "whole-cns"],
        default="brain",
        help="Filter somas to the brain (default) or use the whole CNS.",
    )
    parser.add_argument("--bins", type=int, default=200, help="Density-grid resolution.")
    parser.add_argument(
        "--max-verts", type=int, default=150, help="Max vertices per silhouette polygon."
    )
    args = parser.parse_args(argv)
    # _load_points treats "brain" as the filter and anything else as whole-CNS.
    region = "brain" if args.region == "brain" else "whole-cns"
    build_outline(args.out, args.meta_url, args.cache_dir, region, args.bins, args.max_verts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
