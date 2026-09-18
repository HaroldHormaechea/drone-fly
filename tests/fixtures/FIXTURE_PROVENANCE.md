# Test fixture provenance (UC-13 modality-aware slice)

- **Source dataset:** MaleCNS (mcns_inprop_all_neuron)
- **Source matrix URL:** https://raw.githubusercontent.com/YijieYin/connectome_data_prep/main/data/maleCNS/mcns_inprop_all_neuron.npz
- **Source meta URL:** https://raw.githubusercontent.com/YijieYin/connectome_data_prep/main/data/maleCNS/mcns_all_neuron_meta.csv
- **Source scale:** 161429 neurons, 25083972 edges (stored non-zeros)
- **Slice rule (UC-13, deterministic — no randomness):** the union of
  - **CORE:** top-250 neurons by total degree (in + out nnz, tie-break ascending
    index) **among neurons with a parseable ``somaLocation``** (soma-complete central hubs:
    visual_projection, descending_neuron, hub intrinsics);
  - **PROPRIO QUOTA:** top-50 ``class == "mechanosensory_proprioceptive"`` neurons by total
    degree (tie-break ascending index), selected **without** the soma filter (real afferents,
    intentionally soma-less).

  Induced submatrix over exactly the union.
- **Resulting fixture scale:** 300 neurons, 8288 edges.
- **Soma coverage partition (by construction — partial, not 100%):**
  - 250 / 300 neurons have a real, finite soma coordinate (the core);
  - 50 proprioceptive-quota neurons have NO soma (flagged missing, never faked).
  - ``mcns_fixture_soma.csv`` carries a ``bodyid,x,y,z`` row for exactly the 250
    soma-populated neurons; soma-less neurons are absent from it.

## Attribution

Derived from the connectome_data_prep dataset (https://github.com/YijieYin/connectome_data_prep), which packages the MaleCNS connectome (Janelia FlyEM / neuPrint). MaleCNS is released under CC-BY; this fixture is a small deterministic slice redistributed with attribution.
