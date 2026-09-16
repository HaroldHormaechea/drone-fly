#!/usr/bin/env python3
"""Build the small, committed, real-MaleCNS test fixture for UC-01.

This is a **dev-time** utility. It requires network access (to download the
canonical MaleCNS matrix from the ``connectome_data_prep`` repository) and is
therefore **never run in CI** — CI consumes the committed fixture files it
produces. Run it once (on a networked dev box) to (re)generate the fixture, then
commit the outputs.

Provenance strategy (ordered — see use-cases/plans/01-connectome-plumbing-poc.md)
--------------------------------------------------------------------------------
1. **Preferred (this script's default):** slice a genuinely MaleCNS-derived matrix.
   We use ``data/maleCNS/mcns_inprop_all_neuron.npz`` + ``mcns_all_neuron_meta.csv``
   from ``connectome_data_prep`` — the canonical whole-brain MaleCNS connectivity
   (real neuPrint ``bodyid`` identifiers), CC-BY.
2. **Acceptable fallback:** a real central-brain / FAFB-adult matrix from the same
   repo (e.g. ``adult_type_inprop.npz``), documented honestly as a strict-MaleCNS
   deviation for UC-02 to tighten. Pass ``--source-name`` / ``--source-url`` to use one.
3. **HARD RULE:** if no *real* fly-connectome matrix is obtainable, STOP and escalate.
   Synthetic / ``numpy.random`` fabrication is forbidden. This script never fabricates
   data — it only slices a real downloaded matrix.

Reduction rule (deterministic, sparsity-preserving, logged — never random)
--------------------------------------------------------------------------
Select the ``--neurons`` (default 300) neurons with the highest total degree
(in-degree + out-degree, counted as stored non-zeros), tie-broken by ascending
matrix index. Take the induced submatrix over exactly those neurons. This yields a
dense-enough real subgraph to exercise the encode->network->decode roundtrip while
staying tiny. The selection is fully deterministic: the same source matrix always
yields the same fixture.

Outputs (default: ``tests/fixtures/``)
--------------------------------------
* ``mcns_fixture.npz``        — CSR ``float32`` submatrix, ``K x K``.
* ``mcns_fixture_meta.csv``   — per-neuron metadata (``idx``, ``bodyid``, ...).
* ``FIXTURE_PROVENANCE.md``   — records source URL, slice rule, resulting counts,
                                and CC-BY attribution so the reduction is logged.

After running, update ``FIXTURE_EXPECTED_SCALE`` in
``src/drone_fly/connectome/loader.py`` to the printed (neurons, edges) counts.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

RAW_BASE = "https://raw.githubusercontent.com/YijieYin/connectome_data_prep/main/data"

# Tier-1 default: genuine canonical MaleCNS whole-brain connectivity.
DEFAULT_SOURCE_NAME = "MaleCNS (mcns_inprop_all_neuron)"
DEFAULT_NPZ_URL = f"{RAW_BASE}/maleCNS/mcns_inprop_all_neuron.npz"
DEFAULT_META_URL = f"{RAW_BASE}/maleCNS/mcns_all_neuron_meta.csv"

# Columns kept in the fixture meta (a subset of the full MaleCNS meta), when present.
META_KEEP_COLUMNS = ["idx", "bodyid", "type", "cell_type", "superclass", "top_nt", "sign"]

ATTRIBUTION = (
    "Derived from the connectome_data_prep dataset "
    "(https://github.com/YijieYin/connectome_data_prep), which packages the MaleCNS "
    "connectome (Janelia FlyEM / neuPrint). MaleCNS is released under CC-BY; this "
    "fixture is a small deterministic slice redistributed with attribution."
)


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed trusted https URL


def build_fixture(
    out_dir: Path,
    n_neurons: int,
    npz_url: str,
    meta_url: str,
    source_name: str,
    cache_dir: Path,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_npz = cache_dir / Path(npz_url).name
    src_meta = cache_dir / Path(meta_url).name
    if not src_npz.exists():
        _download(npz_url, src_npz)
    if not src_meta.exists():
        _download(meta_url, src_meta)

    matrix = sp.load_npz(src_npz).tocsr().astype(np.float32)
    n_total = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise SystemExit(f"Source matrix is not square: {matrix.shape}")
    if n_neurons > n_total:
        raise SystemExit(f"Requested {n_neurons} neurons but source has only {n_total}.")

    # Deterministic slice: top-N by total degree (in + out nnz), tie-break by index.
    in_deg = np.asarray((matrix != 0).sum(axis=0)).ravel()
    out_deg = np.asarray((matrix != 0).sum(axis=1)).ravel()
    total_deg = in_deg + out_deg
    # lexsort: primary key last -> sort by index asc, then by -degree desc (stable).
    order = np.lexsort((np.arange(n_total), -total_deg))
    selected = np.sort(order[:n_neurons])

    sub = matrix[selected][:, selected].tocsr().astype(np.float32)
    sub.eliminate_zeros()

    meta = pd.read_csv(src_meta, low_memory=False)
    if "idx" in meta.columns:
        meta = meta.sort_values("idx").reset_index(drop=True)
    sub_meta = meta.iloc[selected].copy()
    keep = [c for c in META_KEEP_COLUMNS if c in sub_meta.columns]
    if keep:
        sub_meta = sub_meta[keep]
    # Re-index the fixture rows to a contiguous 0..K-1 so the meta 'idx' matches the
    # submatrix row order (the loader sorts by 'idx').
    sub_meta = sub_meta.reset_index(drop=True)
    if "idx" in sub_meta.columns:
        sub_meta["idx"] = np.arange(len(sub_meta), dtype=np.int64)

    npz_out = out_dir / "mcns_fixture.npz"
    meta_out = out_dir / "mcns_fixture_meta.csv"
    prov_out = out_dir / "FIXTURE_PROVENANCE.md"

    sp.save_npz(npz_out, sub)
    sub_meta.to_csv(meta_out, index=False)

    prov = f"""# UC-01 test fixture provenance

- **Source dataset:** {source_name}
- **Source matrix URL:** {npz_url}
- **Source meta URL:** {meta_url}
- **Source scale:** {n_total} neurons, {matrix.nnz} edges (stored non-zeros)
- **Slice rule:** top-{n_neurons} neurons by total degree (in-degree + out-degree,
  counted as stored non-zeros), tie-broken by ascending matrix index; induced
  submatrix over exactly those neurons. Fully deterministic — no randomness.
- **Resulting fixture scale:** {sub.shape[0]} neurons, {sub.nnz} edges.

## Attribution

{ATTRIBUTION}
"""
    prov_out.write_text(prov)

    print("Fixture written:")
    print(f"  {npz_out}  ({sub.shape[0]}x{sub.shape[1]}, nnz={sub.nnz})")
    print(f"  {meta_out}")
    print(f"  {prov_out}")
    print()
    print("=> Set FIXTURE_EXPECTED_SCALE in src/drone_fly/connectome/loader.py to:")
    print(f"     ExpectedScale(neuron_count={sub.shape[0]}, edge_count={sub.nnz})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/fixtures"),
        help="Directory to write the fixture into (default: tests/fixtures).",
    )
    parser.add_argument(
        "--neurons",
        type=int,
        default=300,
        help="Number of neurons in the sliced fixture (default: 300).",
    )
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    parser.add_argument("--source-url", dest="npz_url", default=DEFAULT_NPZ_URL)
    parser.add_argument("--meta-url", default=DEFAULT_META_URL)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "drone-fly" / "connectome_src",
        help="Where to cache downloaded source matrices.",
    )
    args = parser.parse_args(argv)

    build_fixture(
        out_dir=args.out_dir,
        n_neurons=args.neurons,
        npz_url=args.npz_url,
        meta_url=args.meta_url,
        source_name=args.source_name,
        cache_dir=args.cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
