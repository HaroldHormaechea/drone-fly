#!/usr/bin/env python3
"""Build the small, committed, real-MaleCNS test fixture.

This is a **dev-time** utility. It requires the canonical MaleCNS matrix + meta from the
``connectome_data_prep`` repository, cached locally under ``--cache-dir`` (downloaded once on
a networked box); it is **never run in CI** — CI consumes the committed fixture files it
produces. Run it to (re)generate the fixture, then commit the outputs.

Provenance strategy (ordered — see use-cases/plans/01-connectome-plumbing-poc.md)
--------------------------------------------------------------------------------
1. **Preferred (this script's default):** slice a genuinely MaleCNS-derived matrix.
   We use ``data/maleCNS/mcns_inprop_all_neuron.npz`` + ``mcns_all_neuron_meta.csv``
   from ``connectome_data_prep`` — the canonical whole-brain MaleCNS connectivity
   (real neuPrint ``bodyid`` identifiers), CC-BY.
2. **Acceptable fallback:** a real central-brain / FAFB-adult matrix from the same
   repo, documented honestly as a strict-MaleCNS deviation. Pass ``--source-name`` /
   ``--source-url`` to use one.
3. **HARD RULE:** if no *real* fly-connectome matrix is obtainable, STOP and escalate.
   Synthetic / ``numpy.random`` fabrication is forbidden. This script never fabricates
   data — it only slices a real downloaded matrix, and it never invents soma coordinates.

Reduction rule (UC-13/UC-17: modality-aware, deterministic, offline — never random)
-----------------------------------------------------------------------------------
The fixture must carry (a) a soma-complete central "core" so UC-05's anatomical map holds,
(b) a genuinely-wired *proprioceptive/mechanosensory* afferent population so UC-13's
modality selector and re-bind run against real biology, AND (c) the approximate internal-state
/ feeding ("hunger") population so UC-17's battery observation block binds to real neurons.
Real sensory afferents have no brain-volume somata, so (a) and (b) pull apart — the rule builds
the union of:

* **CORE** — the ``--core`` (default 250) neurons of highest total degree **among neurons that
  have a real, finite, parseable ``somaLocation``** (in + out nnz; tie-break ascending index).
  These are the central hubs — ``visual_projection`` sensory, ``descending_neuron`` motor, and
  hub intrinsics — every one soma-populated.
* **PROPRIO QUOTA** — the ``--proprio-quota`` (default 50) highest-degree
  ``class == "mechanosensory_proprioceptive"`` neurons, selected **without** the soma filter
  (they are real afferents that are intentionally soma-less).
* **HUNGER QUOTA (UC-17)** — the ``--hunger-quota`` (default 22) highest-degree neurons whose
  ``cell_type`` contains one of the hunger tokens (IPC / Hugin / NPF / insulin / DILP), selected
  **without** the soma filter. In the canonical MaleCNS matrix this matches exactly 22 neurons
  ({IPC, Hugin-RG, NPFL1-I}), which ARE soma-populated (endocrine/intrinsic). The ``hunger``
  modality is labelled in ``cell_type``, not ``subclass`` — hence the cell_type match.

The induced submatrix is taken over the union. Fully deterministic: the same source matrix
always yields the same fixture. By construction the fixture has **partial** soma coverage
(the core + hunger quota are soma-populated; the proprioceptive quota is 0%), which UC-05's
coverage test asserts as an exact partition rather than 100%.

Outputs (default: ``tests/fixtures/``)
--------------------------------------
* ``mcns_fixture.npz``        — CSR ``float32`` submatrix, ``K x K``.
* ``mcns_fixture_meta.csv``   — per-neuron metadata (``idx``, ``bodyid``, ``class``,
                                ``subclass``, ``superclass``, ...).
* ``mcns_fixture_soma.csv``   — ``bodyid,x,y,z`` for exactly the soma-populated neurons
                                (the core), parsed offline from the source ``somaLocation``.
                                Soma-less afferents get NO row (flagged missing, never faked).
* ``FIXTURE_PROVENANCE.md``   — records source URL, slice rule, resulting counts, soma
                                partition, and CC-BY attribution so the reduction is logged.

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

# Columns kept in the fixture meta (a subset of the full MaleCNS meta), when present. UC-13
# adds 'class'/'subclass' so the loader can surface neuron_class/subclass for modality selection.
META_KEEP_COLUMNS = [
    "idx",
    "bodyid",
    "type",
    "cell_type",
    "superclass",
    "class",
    "subclass",
    "top_nt",
    "sign",
]

#: The MaleCNS ``class`` label of the proprioceptive afferents the quota draws from.
PROPRIO_CLASS = "mechanosensory_proprioceptive"

#: Curated, case-insensitive ``cell_type`` substrings that identify the approximate
#: internal-state / feeding ("hunger") population UC-17 binds the battery observation block to.
#: Mirrors :data:`drone_fly.controller.modality.MODALITY_RULES` ["hunger"] verbatim. In the
#: canonical MaleCNS matrix these match exactly 22 neurons: {IPC x16, Hugin-RG x4, NPFL1-I x2}.
HUNGER_CELL_TYPE_TOKENS = ("ipc", "hugin", "npf", "insulin", "dilp")

#: Default core / proprioceptive-quota / hunger-quota sizes (see the module docstring). The
#: hunger population is small and biologically fixed, so the quota is the full matched set (22).
DEFAULT_CORE = 250
DEFAULT_PROPRIO_QUOTA = 50
DEFAULT_HUNGER_QUOTA = 22

ATTRIBUTION = (
    "Derived from the connectome_data_prep dataset "
    "(https://github.com/YijieYin/connectome_data_prep), which packages the MaleCNS "
    "connectome (Janelia FlyEM / neuPrint). MaleCNS is released under CC-BY; this "
    "fixture is a small deterministic slice redistributed with attribution."
)


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed trusted https URL


def _parse_soma_location(value: object) -> tuple[float, float, float] | None:
    """Parse a ``"[x y z]"`` somaLocation cell -> ``(x, y, z)`` floats, or ``None``.

    Mirrors :func:`drone_fly.record.coordinates` parsing so the fixture's soma partition
    matches what the runtime reads. A missing / malformed / non-finite location -> ``None``.
    """
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


def _rank_by_degree(candidates: np.ndarray, total_deg: np.ndarray, size: int) -> np.ndarray:
    """Top-``size`` of ``candidates`` by descending total degree, tie-broken by ascending index.

    Deterministic and stable: ``lexsort`` with the index as the secondary key.
    """
    cand = np.asarray(candidates, dtype=np.int64)
    order = cand[np.lexsort((cand, -total_deg[cand]))]
    return order[: max(0, size)]


def _soma_mask(meta: pd.DataFrame) -> np.ndarray:
    """Boolean mask (row-aligned to ``meta``) of neurons with a parseable ``somaLocation``."""
    if "somaLocation" not in meta.columns:
        raise SystemExit(
            "Source meta lacks a 'somaLocation' column; cannot build the soma-complete core "
            "or the soma sidecar. This script never fabricates coordinates."
        )
    return meta["somaLocation"].map(lambda v: _parse_soma_location(v) is not None).to_numpy()


def build_fixture(
    out_dir: Path,
    core_size: int,
    proprio_quota: int,
    hunger_quota: int,
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

    meta = pd.read_csv(src_meta, low_memory=False)
    if "idx" in meta.columns:
        meta = meta.sort_values("idx").reset_index(drop=True)
    if len(meta) != n_total:
        raise SystemExit(
            f"Source meta rows ({len(meta)}) != matrix dimension ({n_total}); out of sync."
        )
    if "class" not in meta.columns:
        raise SystemExit("Source meta lacks a 'class' column; cannot select the proprio quota.")
    if "cell_type" not in meta.columns:
        raise SystemExit(
            "Source meta lacks a 'cell_type' column; cannot select the hunger quota (UC-17). "
            "The 'hunger' population is labelled in cell_type, not subclass."
        )

    # Total degree (in + out nnz) over the whole source matrix — the ranking key.
    in_deg = np.asarray((matrix != 0).sum(axis=0)).ravel()
    out_deg = np.asarray((matrix != 0).sum(axis=1)).ravel()
    total_deg = in_deg + out_deg

    soma_mask = _soma_mask(meta)
    # astype(str) (not "string") so missing labels become the literal "nan" rather than pd.NA,
    # which would make the ``== PROPRIO_CLASS`` comparison raise on ambiguous truth values.
    class_labels = meta["class"].astype(str).to_numpy()
    # Lowercased cell_type for the hunger substring match (UC-17); "nan" for missing labels.
    cell_type_lower = meta["cell_type"].astype(str).str.lower().to_numpy()

    # CORE: top soma-bearing neurons by degree. PROPRIO: top proprioceptive by degree, no soma
    # filter. HUNGER: top hunger-cell_type neurons by degree, no soma filter (UC-17; a small,
    # biologically-fixed population — the quota takes the full matched set, ~22). Union (dedup)
    # sorted for a deterministic contiguous layout.
    core = _rank_by_degree(np.nonzero(soma_mask)[0], total_deg, core_size)
    proprio_candidates = np.nonzero(class_labels == PROPRIO_CLASS)[0]
    if proprio_candidates.size < proprio_quota:
        raise SystemExit(
            f"Only {proprio_candidates.size} '{PROPRIO_CLASS}' neurons available, "
            f"need {proprio_quota}."
        )
    proprio = _rank_by_degree(proprio_candidates, total_deg, proprio_quota)
    hunger_mask = np.array(
        [any(tok in ct for tok in HUNGER_CELL_TYPE_TOKENS) for ct in cell_type_lower],
        dtype=bool,
    )
    hunger_candidates = np.nonzero(hunger_mask)[0]
    if hunger_candidates.size == 0:
        raise SystemExit(
            "No neurons match the hunger cell_type tokens "
            f"{list(HUNGER_CELL_TYPE_TOKENS)} in the source meta; cannot build the UC-17 hunger "
            "quota. This script never fabricates data (STOP + flag)."
        )
    # Take the top-degree matched neurons up to the quota; the matched set is small and fixed by
    # biology, so a quota >= the number available simply selects them all (no raise — unlike the
    # proprio quota, which draws from a large class).
    hunger = _rank_by_degree(hunger_candidates, total_deg, hunger_quota)
    selected = np.sort(np.union1d(np.union1d(core, proprio), hunger))

    sub = matrix[selected][:, selected].tocsr().astype(np.float32)
    sub.eliminate_zeros()

    sub_meta = meta.iloc[selected].copy()
    keep = [c for c in META_KEEP_COLUMNS if c in sub_meta.columns]
    if keep:
        sub_meta = sub_meta[keep]
    # Re-index the fixture rows to a contiguous 0..K-1 so meta 'idx' matches the submatrix
    # row order (the loader sorts by 'idx').
    sub_meta = sub_meta.reset_index(drop=True)
    if "idx" in sub_meta.columns:
        sub_meta["idx"] = np.arange(len(sub_meta), dtype=np.int64)

    # --- Soma sidecar: real coords for the soma-populated neurons only (offline). -----------
    # Backstop (never fabricate): every CORE neuron was soma-filtered, so it MUST parse here.
    selected_meta = meta.iloc[selected].reset_index(drop=True)
    soma_rows: list[dict] = []
    core_set = set(int(i) for i in core.tolist())
    missing_core: list[int] = []
    for row_i, src_idx in enumerate(selected.tolist()):
        bodyid = int(selected_meta["bodyid"].iloc[row_i])
        xyz = _parse_soma_location(selected_meta["somaLocation"].iloc[row_i])
        if xyz is None:
            if int(src_idx) in core_set:
                missing_core.append(bodyid)
            continue
        x, y, z = xyz
        soma_rows.append({"bodyid": bodyid, "x": x, "y": y, "z": z})
    if missing_core:
        raise SystemExit(
            f"{len(missing_core)} core bodyids lack a usable somaLocation "
            f"(e.g. {missing_core[:8]}); refusing to build. Core selection is soma-filtered, so "
            f"this indicates a parse/data mismatch — never fabricate coordinates (STOP + flag)."
        )

    npz_out = out_dir / "mcns_fixture.npz"
    meta_out = out_dir / "mcns_fixture_meta.csv"
    soma_out = out_dir / "mcns_fixture_soma.csv"
    prov_out = out_dir / "FIXTURE_PROVENANCE.md"

    sp.save_npz(npz_out, sub)
    sub_meta.to_csv(meta_out, index=False)
    pd.DataFrame(soma_rows, columns=["bodyid", "x", "y", "z"]).to_csv(soma_out, index=False)

    n_sel = int(selected.size)
    n_core = int(core.size)
    n_proprio_in_sel = int((class_labels[selected] == PROPRIO_CLASS).sum())
    n_hunger_in_sel = int(hunger_mask[selected].sum())
    n_soma = len(soma_rows)
    # Matched hunger cell_type breakdown (for provenance): the exact {label: count} set selected.
    hunger_types = pd.Series(meta["cell_type"].to_numpy()[selected][hunger_mask[selected]])
    hunger_breakdown = hunger_types.value_counts().to_dict()
    hunger_set_str = ", ".join(f"{k} x{v}" for k, v in sorted(hunger_breakdown.items()))

    prov = f"""# Test fixture provenance (UC-13 modality-aware slice; UC-17 hunger quota)

- **Source dataset:** {source_name}
- **Source matrix URL:** {npz_url}
- **Source meta URL:** {meta_url}
- **Source scale:** {n_total} neurons, {matrix.nnz} edges (stored non-zeros)
- **Slice rule (deterministic — no randomness):** the union of
  - **CORE:** top-{core_size} neurons by total degree (in + out nnz, tie-break ascending
    index) **among neurons with a parseable ``somaLocation``** (soma-complete central hubs:
    visual_projection, descending_neuron, hub intrinsics);
  - **PROPRIO QUOTA:** top-{proprio_quota} ``class == "{PROPRIO_CLASS}"`` neurons by total
    degree (tie-break ascending index), selected **without** the soma filter (real afferents,
    intentionally soma-less);
  - **HUNGER QUOTA (UC-17):** top-{hunger_quota} neurons whose ``cell_type`` contains one of
    {list(HUNGER_CELL_TYPE_TOKENS)} (case-insensitive) by total degree (tie-break ascending
    index), selected **without** the soma filter. This is the approximate internal-state /
    feeding ("hunger") population the battery observation block binds to. Matched set in this
    build: {{{hunger_set_str}}} ({n_hunger_in_sel} neurons).

  Induced submatrix over exactly the union.
- **Resulting fixture scale:** {sub.shape[0]} neurons, {sub.nnz} edges.
- **Soma coverage partition (by construction — partial, not 100%):**
  - {n_soma} / {n_sel} neurons have a real, finite soma coordinate (the {n_core}-neuron core
    plus the {n_hunger_in_sel} soma-bearing hunger-quota neurons — the hunger population is
    endocrine/intrinsic and soma-populated);
  - {n_proprio_in_sel} proprioceptive-quota neurons have NO soma (flagged missing, never faked).
  - ``mcns_fixture_soma.csv`` carries a ``bodyid,x,y,z`` row for exactly the {n_soma}
    soma-populated neurons; soma-less neurons are absent from it.

## Attribution

{ATTRIBUTION}
"""
    prov_out.write_text(prov)

    print("Fixture written:")
    print(f"  {npz_out}  ({sub.shape[0]}x{sub.shape[1]}, nnz={sub.nnz})")
    print(f"  {meta_out}")
    print(f"  {soma_out}  ({n_soma} soma rows)")
    print(f"  {prov_out}")
    print()
    print(
        f"Selection: core={n_core} (soma-filtered) + proprio_quota={n_proprio_in_sel} "
        f"+ hunger_quota={n_hunger_in_sel} => {n_sel} unique neurons; "
        f"soma coverage {n_soma}/{n_sel}."
    )
    print(f"Matched hunger cell_type set: {{{hunger_set_str}}}")
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
        "--core",
        type=int,
        default=DEFAULT_CORE,
        help=f"Soma-populated central core size (default: {DEFAULT_CORE}).",
    )
    parser.add_argument(
        "--proprio-quota",
        type=int,
        default=DEFAULT_PROPRIO_QUOTA,
        help=f"Proprioceptive ({PROPRIO_CLASS}) afferent quota (default: {DEFAULT_PROPRIO_QUOTA}).",
    )
    parser.add_argument(
        "--hunger-quota",
        type=int,
        default=DEFAULT_HUNGER_QUOTA,
        help=(
            "Hunger (internal-state/feeding cell_type) quota for the UC-17 battery obs binding "
            f"(default: {DEFAULT_HUNGER_QUOTA}; the biological population is small, so a quota "
            ">= the matched count selects them all)."
        ),
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
        core_size=args.core,
        proprio_quota=args.proprio_quota,
        hunger_quota=args.hunger_quota,
        npz_url=args.npz_url,
        meta_url=args.meta_url,
        source_name=args.source_name,
        cache_dir=args.cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
