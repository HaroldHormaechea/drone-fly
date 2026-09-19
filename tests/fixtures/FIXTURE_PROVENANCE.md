# Test fixture provenance (UC-13 modality-aware slice; UC-17 hunger quota)

- **Source dataset:** MaleCNS (mcns_inprop_all_neuron)
- **Source matrix URL:** https://raw.githubusercontent.com/YijieYin/connectome_data_prep/main/data/maleCNS/mcns_inprop_all_neuron.npz
- **Source meta URL:** https://raw.githubusercontent.com/YijieYin/connectome_data_prep/main/data/maleCNS/mcns_all_neuron_meta.csv
- **Source scale:** 161429 neurons, 25083972 edges (stored non-zeros)
- **Slice rule (deterministic — no randomness):** the union of
  - **CORE:** top-250 neurons by total degree (in + out nnz, tie-break ascending
    index) **among neurons with a parseable ``somaLocation``** (soma-complete central hubs:
    visual_projection, descending_neuron, hub intrinsics);
  - **PROPRIO QUOTA:** top-50 ``class == "mechanosensory_proprioceptive"`` neurons by total
    degree (tie-break ascending index), selected **without** the soma filter (real afferents,
    intentionally soma-less);
  - **HUNGER QUOTA (UC-17):** top-22 neurons whose ``cell_type`` contains one of
    ['ipc', 'hugin', 'npf', 'insulin', 'dilp'] (case-insensitive) by total degree (tie-break ascending
    index), selected **without** the soma filter. This is the approximate internal-state /
    feeding ("hunger") population the battery observation block binds to. Matched set in this
    build: {Hugin-RG x4, IPC x16, NPFL1-I x2} (22 neurons).

  Induced submatrix over exactly the union.
- **Resulting fixture scale:** 322 neurons, 8413 edges.
- **Soma coverage partition (by construction — partial, not 100%):**
  - 272 / 322 neurons have a real, finite soma coordinate (the 250-neuron core
    plus the 22 soma-bearing hunger-quota neurons — the hunger population is
    endocrine/intrinsic and soma-populated);
  - 50 proprioceptive-quota neurons have NO soma (flagged missing, never faked).
  - ``mcns_fixture_soma.csv`` carries a ``bodyid,x,y,z`` row for exactly the 272
    soma-populated neurons; soma-less neurons are absent from it.

## `uc27_soma_meta.csv` — synthetic tier-0 anatomy fixture (UC-27 AC-10)

- **Purpose:** exercise the UC-27 tier-0 "real anatomy from the connectome meta ``somaLocation``
  column" path. The main ``mcns_fixture_meta.csv`` (above) has **no** ``somaLocation`` column, so a
  distinct fixture is needed that *does* carry one.
- **Provenance:** **synthetic / hand-authored** for UC-27 — it is NOT derived from MaleCNS. The six
  ``bodyid``s (``90001``–``90006``) and their ``somaLocation`` coordinates are arbitrary finite
  integers chosen to be recognisable in assertions; they are not real neuron identities or
  positions. Format mirrors ``connectome_data_prep`` meta: ``somaLocation`` is ``"[x y z]"`` in an
  (arbitrary) integer voxel space, blank for the two intentionally soma-less rows (``90005``,
  ``90006``) that model peripheral afferents flagged missing (never faked).
- **Shape:** 6 rows; 4 soma-bearing (``90001``–``90004``), 2 soma-less. Tests pair it with a
  matching synthetic 6×6 adjacency ``.npz`` built at runtime (never committed).

## Attribution

Derived from the connectome_data_prep dataset (https://github.com/YijieYin/connectome_data_prep), which packages the MaleCNS connectome (Janelia FlyEM / neuPrint). MaleCNS is released under CC-BY; this fixture is a small deterministic slice redistributed with attribution.
